"""
Tiny-Math-Solver — GRPO on GSM8K with TRL.

Trains Qwen2.5-1.5B-Instruct on a curated subset of GSM8K problems
requiring 3+ entity tracking, using <think> tags for structured reasoning.

Reward: binary correctness only (1.0 if final answer matches gold, 0.0
otherwise). No auxiliary rewards -- GRPO learns entirely from the contrast
between correct and incorrect completions within each group.

Ghost-batching mitigation:
  The entity-filtered dataset contains many problems that are too easy (all 8
  completions correct → zero gradient) or too hard (all wrong → zero gradient).
  Only ~33% of problems produce useful signal. To compensate:
  - Large effective batch (128 prompts via gradient_accumulation_steps=32)
    ensures each update sees ~40+ "sweet spot" problems
  - KL penalty (beta) prevents policy drift from noisy updates
  - DAPO loss normalizes across active tokens in the global batch

Generation: vanilla model.generate() -- no vLLM, no paged attention.
vLLM colocate has PEFT convergence bugs (trl#2856, vllm#14483).
Paged attention runs out of cache blocks at our batch size.
For 1.5B on L40S, vanilla generation is fast enough (~5GB KV cache).

Usage:
    bash setup.sh                        # install deps + build dataset (once)
    python src/train_grpo.py             # train
    accelerate launch src/train_grpo.py  # multi-GPU
"""

import os
import logging
from datetime import datetime, timezone

# ── Logging (set up BEFORE importing libraries that configure logging) ─────
RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{RUN_TIMESTAMP}.log")

# Explicitly add handlers to root logger (basicConfig is a no-op if
# any library has already configured logging, which transformers/trl do)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.INFO)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_file_handler)

logger = logging.getLogger("tiny_math_solver")
logger.setLevel(logging.INFO)

from datasets import load_from_disk
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from reward_func import correctness_reward, format_reward

logger.info(f"=== Run started: {RUN_TIMESTAMP} ===")
logger.info(f"Log file: {LOG_FILE}")


# ── Dataset ─────────────────────────────────────────────────────────────────
# Curated dataset (built by src/build_entity_dataset.py).
# Contains GSM8K problems with 3+ named entities (the model's weak spot).
# TRL expects a 'prompt' column.

ENTITY_TRACKING_PROMPT = (
    "Think step by step inside <think> tags before answering. "
    "For each person or item in the problem, explicitly state "
    "what you know about them and calculate their value."
)


def build_prompt(example):
    """Convert GSM8K question to chat format with <think> tag instruction.

    The system prompt forces structured entity-by-entity reasoning,
    creating diverse completions that GRPO can learn from.
    """
    example["prompt"] = [
        {"role": "system", "content": ENTITY_TRACKING_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    return example


logger.info("Loading entity-tracking dataset ...")
dataset = load_from_disk("data/entity_tracking_dataset")
dataset = dataset.map(build_prompt)
logger.info(f"Dataset loaded: {len(dataset)} entity-tracking problems")


# ── LoRA config ─────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# ── Training config ─────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir=f"./checkpoint/run_{RUN_TIMESTAMP}",
    run_name=f"grpo_{RUN_TIMESTAMP}",

    # GRPO sampling
    # - 8 completions per prompt at temp=1.0 for diverse reasoning paths
    # - 1024 completion length for <think> section + answer
    num_generations=8,
    max_completion_length=1024,
    temperature=1.0,

    # ── Generation ──────────────────────────────────────────────────────
    # Plain model.generate() -- no vLLM, no paged attention.
    # vLLM colocate: PEFT + vLLM has known convergence bugs (trl#2856).
    # Paged attention: runs out of cache blocks at our batch size (128
    # sequences × 1024 tokens exceeds the default block pool).
    # For 1.5B on L40S (48GB), vanilla generation is fast enough and
    # the KV cache (~5GB for 128 sequences) fits easily.
    use_vllm=False,

    # ── Ghost-batching mitigation ──────────────────────────────────────
    # With entity-only filtering (~33% of problems in sweet spot), most
    # mini-batches contain zero-gradient prompts (all-correct or all-wrong).
    # A small effective batch means some updates see NO useful signal at all,
    # causing massive W&B noise.
    #
    # Fix: accumulate over 16 mini-batches so each optimizer step sees
    # 64 prompts. Even if only 33% produce gradient, that's ~20 useful
    # prompts per update -- enough for a reasonable direction.
    # (32 was designed for vLLM; 16 balances signal vs wall-clock time
    # with vanilla model.generate on L40S: ~7 hrs for 500 steps.)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # effective batch = 4*16 = 64 prompts
    #                                  64 / 8 generations = 8 unique prompts
    #                                  per accumulation step

    # Training schedule
    num_train_epochs=1,
    max_steps=500,                   # ~5 epochs over ~1500 entity problems
    learning_rate=5e-5,              # halved from 1e-4 for stability
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,

    # ── Stability: KL penalty ──────────────────────────────────────────
    # Prevents policy drift from noisy updates. Without this, a few
    # outlier batches (all sweet-spot problems with extreme advantages)
    # can push the model too far from the base policy.
    # DeepSeek-R1 uses 0.001; we use 0.04 because our effective signal
    # is noisier (ghost batching) so we need a stronger anchor.
    beta=0.04,

    # ── DAPO loss + truncation masking ─────────────────────────────────
    # loss_type="dapo" (default): normalizes by active tokens in global
    # batch, eliminating length bias.
    # mask_truncated_completions: excludes cut-off completions from loss
    # so they don't get incorrectly penalized (DAPO paper recommendation).
    mask_truncated_completions=True,

    # Logging & saving
    logging_steps=1,
    save_steps=50,
    log_completions=True,            # log (prompt, completion) pairs to W&B
    num_completions_to_print=2,      # only print 2 examples to terminal
    report_to="wandb" if os.environ.get("WANDB_TOKEN") else "none",

    # Misc
    seed=42,
)


# ── Log config ──────────────────────────────────────────────────────────────
eff_batch = (training_args.per_device_train_batch_size
             * training_args.gradient_accumulation_steps)
unique_prompts_per_step = eff_batch // training_args.num_generations

logger.info("Config:")
logger.info(f"  Model:            Qwen/Qwen2.5-1.5B-Instruct")
logger.info(f"  Mode:             GRPO (correctness + format reward)")
logger.info(f"  System prompt:    {ENTITY_TRACKING_PROMPT[:60]}...")
logger.info(f"  LoRA rank:        {peft_config.r}")
logger.info(f"  Num generations:  {training_args.num_generations}")
logger.info(f"  Batch size:       {training_args.per_device_train_batch_size}")
logger.info(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
logger.info(f"  Effective batch:  {eff_batch} sequences = {unique_prompts_per_step} unique prompts")
logger.info(f"  Learning rate:    {training_args.learning_rate}")
logger.info(f"  Beta (KL pen.):   {training_args.beta}")
logger.info(f"  Temperature:      {training_args.temperature}")
logger.info(f"  Max compl len:    {training_args.max_completion_length}")
logger.info(f"  Max steps:        {training_args.max_steps}")
logger.info(f"  Mask truncated:   {training_args.mask_truncated_completions}")
logger.info(f"  Generation:       model.generate() (no vLLM, no paged attn)")
logger.info(f"  Output dir:       {training_args.output_dir}")
logger.info(f"  W&B:              {training_args.report_to}")


# ── Trainer ─────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, format_reward],
    peft_config=peft_config,
)


# ── Train ───────────────────────────────────────────────────────────────────
if os.environ.get("WANDB_TOKEN"):
    import wandb
    wandb.login(key=os.environ["WANDB_TOKEN"])
    # If your W&B account requires a team entity, set WANDB_ENTITY env var
    if os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = os.environ["WANDB_ENTITY"]
        wandb.init(project="tiny-math-solver", entity=os.environ["WANDB_ENTITY"])

logger.info("Starting training ...")
trainer.train()
trainer.save_model()

logger.info(f"Training complete! Model saved to: {training_args.output_dir}")
logger.info(f"=== Run finished: {datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')} ===")
