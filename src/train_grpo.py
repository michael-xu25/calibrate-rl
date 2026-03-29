"""
GRPO Training with Dynamic Curriculum — CalibrateRL

Trains a model using GRPO with a curriculum that updates every phase.
Each phase = STEPS_PER_PHASE gradient steps on the current Goldilocks set.
Between phases: re-evaluate active problems, evict saturated/unreachable,
replenish from reserve pool using the current model.

Optimized for H100 (Nebius) on Lightning AI:
  - Full bf16, Flash Attention 2 (auto-detected)
  - LoRA r=32 across all projection layers
  - Batch size 8 × grad_accum 2 = effective batch 16
  - Scoring uses batched generation (16 rollouts in one forward pass)
  - Resume from checkpoint if job is killed

No vLLM — standard HuggingFace generation (Lightning AI compatible).

Usage:
    python src/train_grpo.py --model qwen-2.5-7b
    python src/train_grpo.py --model qwen-2.5-7b --resume   # resume from saved state

Environment:
    HF_TOKEN — required for Llama; not needed for Qwen
"""

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Model registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "llama-3-8b":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

GOLDILOCKS_FILES = {
    "llama-3-8b":  Path("data/goldilocks_llama-3-8b.json"),
    "qwen-2.5-7b": Path("data/goldilocks_qwen-2.5-7b.json"),
}

EVAL_RESULTS_PATH = Path("data/evaluation_results_L1L2.json")
RESERVE_POOL_PATH = Path("data/profile_dataset_L1L2L3.json")

# ── Curriculum parameters ──────────────────────────────────────────────────────

STEPS_PER_PHASE      = 40     # gradient steps per phase before curriculum refresh
N_ROLLOUTS_TRAIN     = 8      # rollouts GRPOTrainer generates per problem (for loss)
N_ROLLOUTS_SCORE     = 4      # rollouts for curriculum scoring — enough to classify
                               # saturated/trainable/unreachable; 5 distinct values:
                               # 0, 0.25, 0.50, 0.75, 1.0
SATURATE_THRESHOLD   = 0.75   # 3/4 correct → learned, evict (equiv. to 6/8)
UNREACHABLE_PATIENCE = 2      # consecutive 0-pass evals before evicting
TARGET_CURRICULUM    = 128    # active problems to maintain
MIN_CURRICULUM       = 16     # stop training if pool falls below this
MAX_PROBE            = 40     # reserve candidates to probe per refresh

# ── Generation parameters ──────────────────────────────────────────────────────

GEN_MAX_NEW_TOKENS = 512
GEN_TEMPERATURE    = 0.8
GEN_TOP_P          = 0.95

SYSTEM_PROMPT = (
    "You are a concise mathematical reasoning assistant. "
    "Solve problems step by step, then give your final answer in <answer> tags."
)
ANSWER_REMINDER = (
    "\n\nFinal answer must be last, inside <answer> tags — e.g. <answer>42</answer>"
)

# ── Reward function ────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()
    candidates = re.findall(r"(?<![/\w])(-?\d+(?:[./]\d+)?)(?![/\w])", text)
    return candidates[-1] if candidates else None


def normalize(raw: str) -> str:
    s = raw.strip().strip("$")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right)[(\[{)\]}|.]", "", s)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"^(-?\d+)\.0+$", r"\1", s)
    return s.lower().strip()


def score(completion: str, ground_truth: str) -> float:
    extracted = extract_answer(completion)
    if extracted is None:
        return 0.0
    return 1.0 if normalize(extracted) == normalize(ground_truth) else 0.0


def reward_fn(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    return [score(c, a) for c, a in zip(completions, answer)]


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(problem_text: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{problem_text}{ANSWER_REMINDER}\n"
        f"<|assistant|>\n"
    )


# ── Batched scoring ────────────────────────────────────────────────────────────

def score_problems_batched(
    items: List[Tuple[str, dict]],   # [(problem_id, problem_dict), ...]
    model,
    tokenizer,
    label: str = "scoring",
) -> Dict[str, float]:
    """
    Score each problem with N_ROLLOUTS_SCORE rollouts using a single batched
    generate call per problem (batch_size = N_ROLLOUTS_SCORE).

    This is ~16× faster than sequential calls because the H100 processes
    all rollouts in parallel rather than waiting for each to complete.

    Falls back to sequential if the batch OOMs (very long problems).
    """
    results = {}
    model.eval()
    t_start = time.monotonic()

    with torch.inference_mode():
        for i, (pid, prob) in enumerate(items):
            prompt   = build_prompt(prob["problem"])
            enc      = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            input_len = enc["input_ids"].shape[1]

            # Repeat prompt N times → single batched generate call
            input_ids      = enc["input_ids"].repeat(N_ROLLOUTS_SCORE, 1).to(model.device)
            attention_mask = enc["attention_mask"].repeat(N_ROLLOUTS_SCORE, 1).to(model.device)

            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                texts = [
                    tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
                    for j in range(N_ROLLOUTS_SCORE)
                ]
            except torch.cuda.OutOfMemoryError:
                # Fallback: score sequentially if this problem's prompt is unusually long
                print(f"\n  [OOM] {pid} — falling back to sequential scoring")
                torch.cuda.empty_cache()
                texts = []
                single_input = enc.to(model.device)
                for _ in range(N_ROLLOUTS_SCORE):
                    out  = model.generate(
                        **single_input,
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        temperature=GEN_TEMPERATURE,
                        top_p=GEN_TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    texts.append(
                        tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
                    )

            rewards = [score(t, prob["answer"]) for t in texts]
            pr = sum(rewards) / len(rewards)
            results[pid] = pr

            # Per-problem progress line
            elapsed = time.monotonic() - t_start
            rate    = (i + 1) / elapsed
            eta     = (len(items) - i - 1) / rate if rate > 0 else 0
            correct = sum(rewards)
            print(
                f"  [{label}] {i+1:>3}/{len(items)}  "
                f"{pid:<35}  pass={pr:.3f} ({correct}/{N_ROLLOUTS_SCORE})  "
                f"eta {eta:.0f}s",
                flush=True,
            )

    model.train()
    return results


# ── Curriculum ─────────────────────────────────────────────────────────────────

@dataclass
class ProblemState:
    problem: dict
    consecutive_zero_evals: int = 0


@dataclass
class Curriculum:
    active:     Dict[str, ProblemState] = field(default_factory=dict)
    reserve:    List[dict]              = field(default_factory=list)
    evicted:    set                     = field(default_factory=set)
    phase_logs: List[dict]              = field(default_factory=list)
    phase:      int                     = 0
    total_steps: int                    = 0

    def size(self) -> int:
        return len(self.active)

    def to_dataset(self) -> Dataset:
        return Dataset.from_list([
            {
                "prompt":     build_prompt(s.problem["problem"]),
                "answer":     s.problem["answer"],
                "problem_id": pid,
            }
            for pid, s in self.active.items()
        ])

    def refresh(self, model, tokenizer, out_dir: Path) -> None:
        self.phase += 1
        t0 = time.monotonic()
        print(f"\n[Phase {self.phase} refresh] Scoring {self.size()} active problems…")

        # Score active set
        pass_rates = score_problems_batched(
            [(pid, s.problem) for pid, s in self.active.items()],
            model, tokenizer,
            label="active",
        )

        # Evict
        evict_sat, evict_unreach = [], []
        for pid, state in list(self.active.items()):
            pr = pass_rates.get(pid, 0.0)
            if pr >= SATURATE_THRESHOLD:
                evict_sat.append(pid)
            elif pr == 0.0:
                state.consecutive_zero_evals += 1
                if state.consecutive_zero_evals >= UNREACHABLE_PATIENCE:
                    evict_unreach.append(pid)
            else:
                state.consecutive_zero_evals = 0

        evicted_ids = set(evict_sat + evict_unreach)
        for pid in evicted_ids:
            self.evicted.add(pid)
            del self.active[pid]

        # Replenish from reserve — random sample so full pool explored over time.
        # Includes 0/8 base-model problems; they get re-probed with the current
        # (improving) model and enter if now in Goldilocks range.
        needed = TARGET_CURRICULUM - self.size()
        added  = []
        if needed > 0 and self.reserve:
            n_probe   = min(MAX_PROBE, len(self.reserve))
            probe     = random.sample(self.reserve, n_probe)
            print(f"[Phase {self.phase} refresh] Probing {n_probe} reserve candidates…")
            scored    = score_problems_batched(
                [(p["id"], p) for p in probe],
                model, tokenizer,
                label="probe",
            )
            # Remove added IDs from reserve in one pass (O(n) not O(n²))
            to_add = {}
            for prob in probe:
                if len(to_add) >= needed:
                    break
                if 0.0 < scored.get(prob["id"], 0.0) < 1.0:
                    to_add[prob["id"]] = prob

            added_ids = set(to_add.keys())
            self.reserve = [r for r in self.reserve if r["id"] not in added_ids]
            for pid, prob in to_add.items():
                self.active[pid] = ProblemState(problem=prob)
                added.append(pid)

        elapsed = time.monotonic() - t0
        log = {
            "phase":             self.phase,
            "evict_saturated":   len(evict_sat),
            "evict_unreachable": len(evict_unreach),
            "added":             len(added),
            "active_size":       self.size(),
            "reserve_size":      len(self.reserve),
            "pass_rates":        pass_rates,
            "refresh_time_sec":  round(elapsed, 1),
        }
        self.phase_logs.append(log)

        print(
            f"[Phase {self.phase} refresh] "
            f"evicted(sat={len(evict_sat)} unreach={len(evict_unreach)})  "
            f"added={len(added)}  active={self.size()}  reserve={len(self.reserve)}  "
            f"({elapsed:.0f}s)"
        )

        # Save curriculum state after every refresh so we can resume if job dies
        self._save_state(out_dir)

    def _save_state(self, out_dir: Path) -> None:
        state = {
            "phase":       self.phase,
            "total_steps": self.total_steps,
            "evicted":     list(self.evicted),
            "active": [
                {
                    "id":                    pid,
                    "consecutive_zero_evals": s.consecutive_zero_evals,
                    "problem":               s.problem,
                }
                for pid, s in self.active.items()
            ],
            "reserve": self.reserve,
            "phase_logs": self.phase_logs,
        }
        path = out_dir / "curriculum_state.json"
        tmp  = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(path)   # atomic write

    @classmethod
    def load_state(cls, out_dir: Path) -> "Curriculum":
        path = out_dir / "curriculum_state.json"
        state = json.loads(path.read_text())
        active = {
            e["id"]: ProblemState(
                problem=e["problem"],
                consecutive_zero_evals=e["consecutive_zero_evals"],
            )
            for e in state["active"]
        }
        c = cls(
            active=active,
            reserve=state["reserve"],
            evicted=set(state["evicted"]),
            phase_logs=state["phase_logs"],
            phase=state["phase"],
            total_steps=state["total_steps"],
        )
        print(
            f"  Resumed from phase {c.phase}  "
            f"({c.size()} active, {len(c.reserve)} reserve, "
            f"{c.total_steps} steps done)"
        )
        return c


# ── Model loading — H100 optimized ────────────────────────────────────────────

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        return False


def load_model_and_tokenizer(model_id: str):
    use_fa2   = _flash_attn_available()
    attn_impl = "flash_attention_2" if use_fa2 else "eager"
    print(f"\nLoading {model_id} (bf16, attn={attn_impl})…")
    if not use_fa2:
        print("  flash-attn not available — using eager attention")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # Required for gradient_checkpointing + PEFT: frozen base layers don't
    # require grad by default, which breaks gradient checkpointing's
    # save/restore mechanism. This enables gradients on the input embeddings
    # so the chain is unbroken.
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


# ── Curriculum initialization ──────────────────────────────────────────────────

def build_curriculum(model_key: str) -> Curriculum:
    gold_path = GOLDILOCKS_FILES[model_key]
    if not gold_path.exists():
        raise FileNotFoundError(
            f"{gold_path} not found.\n"
            f"Run: python src/export_goldilocks.py --results data/evaluation_results_L1L2.json"
        )

    with open(gold_path) as f:
        goldilocks = json.load(f)

    seed   = goldilocks[:TARGET_CURRICULUM]
    active = {p["id"]: ProblemState(problem=p) for p in seed}
    active_ids = set(active.keys())

    # Build reserve:
    # - Remaining Goldilocks problems (guaranteed trainable on base model)
    # - L1+L2 problems with pass_rate = 0 on base model (may become trainable)
    # - Exclude pass_rate = 1 on base model — they'll never be Goldilocks
    reserve = [p for p in goldilocks[TARGET_CURRICULUM:]]

    if RESERVE_POOL_PATH.exists() and EVAL_RESULTS_PATH.exists():
        with open(RESERVE_POOL_PATH) as f:
            full_pool = json.load(f)
        with open(EVAL_RESULTS_PATH) as f:
            eval_results = json.load(f)

        # Index base-model pass rates
        base_pass = {}
        for rec in eval_results.values():
            if model_key in rec.get("models", {}):
                base_pass[rec["id"]] = rec["models"][model_key]["pass_rate"]

        saturated_excluded = 0
        for p in full_pool:
            if p["id"] in active_ids:
                continue
            pr = base_pass.get(p["id"], -1)
            if pr >= 1.0:
                saturated_excluded += 1
                continue   # already perfect — waste to probe
            reserve.append(p)

        print(f"  Excluded {saturated_excluded} base-model-saturated problems from reserve")

    print(f"  Active:  {len(active)} problems")
    print(f"  Reserve: {len(reserve)} problems")
    return Curriculum(active=active, reserve=reserve)


# ── Training loop ──────────────────────────────────────────────────────────────

def run_training(model, tokenizer, curriculum: Curriculum, args) -> None:
    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if curriculum.size() < MIN_CURRICULUM:
            print(
                f"\n[Phase {curriculum.phase + 1}] Pool below minimum "
                f"({curriculum.size()} < {MIN_CURRICULUM}). "
                f"Natural ceiling reached — stopping."
            )
            break

        if curriculum.total_steps >= args.max_steps:
            print(f"\nReached max_steps ({args.max_steps}). Stopping.")
            break

        steps_this_phase = min(STEPS_PER_PHASE, args.max_steps - curriculum.total_steps)
        print(f"\n{'='*60}")
        print(
            f"  Phase {curriculum.phase + 1}  |  "
            f"{curriculum.size()} problems  |  "
            f"{steps_this_phase} steps  |  "
            f"{curriculum.total_steps}/{args.max_steps} total steps done"
        )
        print(f"{'='*60}")

        grpo_config = GRPOConfig(
            output_dir=str(out_dir / f"phase_{curriculum.phase + 1:03d}"),
            max_steps=steps_this_phase,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=min(5, steps_this_phase // 4),
            logging_steps=5,
            save_strategy="no",
            num_generations=N_ROLLOUTS_TRAIN,
            max_completion_length=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            use_vllm=False,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=curriculum.to_dataset(),
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )
        trainer.train()
        curriculum.total_steps += steps_this_phase

        ckpt = out_dir / f"checkpoint_phase_{curriculum.phase + 1:03d}"
        model.save_pretrained(ckpt)
        print(f"  Checkpoint → {ckpt}")

        curriculum.refresh(model, tokenizer, out_dir)

    # Final adapter
    final_path = out_dir / "final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal adapter → {final_path}")
    print(f"Curriculum log → {out_dir}/curriculum_state.json")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="qwen-2.5-7b")
    parser.add_argument("--max-steps",  type=int,  default=400)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from saved curriculum_state.json")
    args = parser.parse_args()

    model_id = MODEL_REGISTRY[args.model]
    out_dir  = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_id)

    state_path = out_dir / "curriculum_state.json"
    if args.resume and state_path.exists():
        print("\nResuming curriculum from saved state…")
        curriculum = Curriculum.load_state(out_dir)
        # Load the most recent checkpoint adapter weights
        checkpoints = sorted(out_dir.glob("checkpoint_phase_*"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"  Loading adapter weights from {latest}…")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, latest)
    else:
        print("\nBuilding curriculum…")
        curriculum = build_curriculum(args.model)

    run_training(model, tokenizer, curriculum, args)


if __name__ == "__main__":
    main()
