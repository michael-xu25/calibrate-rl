"""
SFT Warm-Up: Fine-tune Qwen on self-generated correct reasoning paths.

This is the "teach the model to be its best self" step. Before running GRPO,
we fine-tune on the shortest correct solutions the model itself generated
during Pass@K evaluation. This gives GRPO a stronger starting point.

Pipeline:
  1. eval_pass_at_k.py  →  logs/pass_at_k.jsonl  (generate K solutions)
  2. extract_sft_data.py →  data/sft_paths.jsonl  (extract best correct paths)
  3. train_sft.py        →  checkpoint/sft_warmup/ (fine-tune on correct paths)
  4. SFT_CHECKPOINT=./checkpoint/sft_warmup python src/train_grpo.py  (RL)

Uses the same LoRA config as GRPO for consistency. Short training (~1 epoch)
to avoid overfitting on the small self-generated dataset.

Usage:
    python src/train_sft.py [--data data/sft_paths.jsonl]
"""

import argparse
import json
import os
import logging
from datetime import datetime, timezone

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ── Logging ─────────────────────────────────────────────────────────────────
RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"sft_{RUN_TIMESTAMP}.log")

_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.INFO)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_file_handler)

logger = logging.getLogger("tiny_math_solver_sft")
logger.setLevel(logging.INFO)


# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WANDB_TOKEN = os.environ.get("WANDB_TOKEN")


def load_sft_dataset(path: str) -> Dataset:
    """Load SFT data from JSONL. Each line has a 'messages' field with
    [{"role": "user", ...}, {"role": "assistant", ...}] chat format.
    """
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                records.append({"messages": entry["messages"]})

    dataset = Dataset.from_list(records)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="SFT warm-up training")
    parser.add_argument(
        "--data", default="data/sft_paths.jsonl",
        help="Path to SFT training data (from extract_sft_data.py)",
    )
    parser.add_argument(
        "--output", default=f"./checkpoint/sft_warmup",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1, keep short to avoid overfit)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (lower than GRPO since this is supervised)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device training batch size",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        print("Run extract_sft_data.py first to generate SFT training data.")
        return

    logger.info(f"=== SFT Run started: {RUN_TIMESTAMP} ===")
    logger.info(f"Log file: {LOG_FILE}")

    # ── Load dataset ────────────────────────────────────────────────────
    print(f"Loading SFT data from {args.data} ...")
    dataset = load_sft_dataset(args.data)
    print(f"Loaded {len(dataset)} training examples\n")
    logger.info(f"Dataset: {len(dataset)} examples from {args.data}")

    # ── LoRA config (same as GRPO for consistency) ──────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # ── SFT Training config ─────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output,
        run_name=f"sft_{RUN_TIMESTAMP}",

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,

        # Sequence length
        max_seq_length=1280,  # 256 prompt + 1024 completion

        # Logging & saving
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb" if WANDB_TOKEN else "none",

        # Misc
        seed=42,
    )

    # ── Log config ──────────────────────────────────────────────────────
    logger.info("SFT Config:")
    logger.info(f"  Model:          {MODEL}")
    logger.info(f"  LoRA rank:      {peft_config.r}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  Batch size:     {args.batch_size}")
    logger.info(f"  Learning rate:  {args.lr}")
    logger.info(f"  Output dir:     {args.output}")

    print(f"{'='*70}")
    print(f"  SFT WARM-UP TRAINING")
    print(f"{'='*70}")
    print(f"  Model:          {MODEL}")
    print(f"  Dataset:        {len(dataset)} examples")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} (x2 grad accum = {args.batch_size * 2} effective)")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Output:         {args.output}")
    print(f"{'='*70}\n")

    # ── W&B setup ───────────────────────────────────────────────────────
    if WANDB_TOKEN:
        import wandb
        wandb.login(key=WANDB_TOKEN)
        if os.environ.get("WANDB_ENTITY"):
            wandb.init(project="tiny-math-solver",
                      entity=os.environ["WANDB_ENTITY"])

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=MODEL,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # ── Train ───────────────────────────────────────────────────────────
    logger.info("Starting SFT training ...")
    trainer.train()
    trainer.save_model()

    logger.info(f"SFT complete! Model saved to: {args.output}")
    print(f"\n{'='*70}")
    print(f"  SFT TRAINING COMPLETE")
    print(f"  Model saved to: {args.output}")
    print(f"")
    print(f"  Next step — run GRPO with the SFT checkpoint:")
    print(f"    SFT_CHECKPOINT={args.output} python src/train_grpo.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
