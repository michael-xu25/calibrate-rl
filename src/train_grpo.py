"""
GRPO Training with Dynamic Curriculum — CalibrateRL

Trains a model using GRPO with a curriculum that updates every phase.
Each phase = STEPS_PER_PHASE gradient steps on the current Goldilocks set.
Between phases: re-evaluate active problems, evict saturated/unreachable,
replenish from reserve pool using the current model.

Optimized for H100 (Nebius) on Lightning AI:
  - Full bf16 (no quantization needed — H100 has 80GB VRAM)
  - Flash Attention 2
  - LoRA r=32 (larger rank affordable with full precision)
  - Batch size 8, gradient accumulation 2 → effective batch 16
  - 16 rollouts per problem (better gradient estimates, H100 is fast enough)
  - 128 active curriculum problems

No vLLM — standard HuggingFace generation (Lightning AI compatible).

Usage:
    python src/train_grpo.py --model llama-3-8b
    python src/train_grpo.py --model qwen-2.5-7b

Environment:
    HF_TOKEN — required for Llama (gated); not needed for Qwen
"""

import argparse
import json
import os
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

RESERVE_POOL_PATH = Path("data/profile_dataset_L1L2.json")

# ── H100 curriculum parameters ─────────────────────────────────────────────────

STEPS_PER_PHASE      = 20     # gradient steps per phase before re-evaluating
N_ROLLOUTS_EVAL      = 16     # rollouts for curriculum scoring (H100: 2× for richer signal)
SATURATE_THRESHOLD   = 0.875  # pass_rate >= this → learned, evict
UNREACHABLE_PATIENCE = 2      # consecutive 0-pass evals before evicting
TARGET_CURRICULUM    = 128    # active problems per phase (H100: 2× vs A10G)
MIN_CURRICULUM       = 16     # stop if pool drops below this

# ── H100 generation parameters ─────────────────────────────────────────────────

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

    def refresh(self, model, tokenizer, phase: int) -> dict:
        """
        Score active problems, evict saturated/unreachable, replenish from reserve.
        Reserve holds ALL unevaluated problems — 0/8 candidates get re-probed
        with the improving model and may enter the curriculum on later phases.
        """
        print(f"\n[Phase {phase}] Scoring {self.size()} active problems…")
        pass_rates = self._score_problems(list(self.active.items()), model, tokenizer)

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

        for pid in evict_sat + evict_unreach:
            self.evicted.add(pid)
            del self.active[pid]

        # Replenish: random sample so full reserve pool gets explored over time
        needed = TARGET_CURRICULUM - self.size()
        added = []
        if needed > 0 and self.reserve:
            probe_pool = random.sample(self.reserve, min(needed * 4, len(self.reserve)))
            scored = self._score_problems(
                [(p["id"], ProblemState(problem=p)) for p in probe_pool],
                model, tokenizer,
            )
            for prob in probe_pool:
                if len(added) >= needed:
                    break
                pr = scored.get(prob["id"], 0.0)
                if 0.0 < pr < 1.0:
                    self.active[prob["id"]] = ProblemState(problem=prob)
                    self.reserve = [r for r in self.reserve if r["id"] != prob["id"]]
                    added.append(prob["id"])

        log = {
            "phase":             phase,
            "evict_saturated":   len(evict_sat),
            "evict_unreachable": len(evict_unreach),
            "added":             len(added),
            "active_size":       self.size(),
            "reserve_size":      len(self.reserve),
            "pass_rates":        pass_rates,
        }
        self.phase_logs.append(log)
        print(
            f"[Phase {phase}] "
            f"evicted(sat={len(evict_sat)} unreach={len(evict_unreach)})  "
            f"added={len(added)}  active={self.size()}  reserve={len(self.reserve)}"
        )
        return log

    def _score_problems(
        self,
        items: List[Tuple[str, ProblemState]],
        model,
        tokenizer,
    ) -> Dict[str, float]:
        results = {}
        model.eval()
        with torch.no_grad():
            for pid, state in items:
                prob   = state.problem
                prompt = build_prompt(prob["problem"])
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024
                ).to(model.device)
                rewards = []
                for _ in range(N_ROLLOUTS_EVAL):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        temperature=GEN_TEMPERATURE,
                        top_p=GEN_TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = out[0][inputs["input_ids"].shape[1]:]
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    rewards.append(score(text, prob["answer"]))
                results[pid] = sum(rewards) / len(rewards)
        model.train()
        return results


# ── Model loading — H100 optimized ────────────────────────────────────────────

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        # Package present — but also needs CUDA + a compatible GPU (Ampere+)
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        return False


def load_model_and_tokenizer(model_id: str):
    use_fa2 = _flash_attn_available()
    attn_impl = "flash_attention_2" if use_fa2 else "eager"
    print(f"\nLoading {model_id} (bf16, attn={attn_impl})…")
    if not use_fa2:
        print("  flash-attn not found or GPU < Ampere — falling back to eager attention")

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

    # r=32 affordable in full bf16 on H100 (vs r=16 with QLoRA on A10G)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
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

    seed = goldilocks[:TARGET_CURRICULUM]
    active = {p["id"]: ProblemState(problem=p) for p in seed}
    active_ids = set(active.keys())

    reserve = [p for p in goldilocks[TARGET_CURRICULUM:]]
    if RESERVE_POOL_PATH.exists():
        with open(RESERVE_POOL_PATH) as f:
            full_pool = json.load(f)
        reserve += [p for p in full_pool if p["id"] not in active_ids]

    print(f"  Active:  {len(active)} problems")
    print(f"  Reserve: {len(reserve)} problems")
    return Curriculum(active=active, reserve=reserve)


# ── Training loop ──────────────────────────────────────────────────────────────

def run_training(model, tokenizer, curriculum: Curriculum, args) -> None:
    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    phase       = 0
    total_steps = 0

    while True:
        phase += 1
        dataset = curriculum.to_dataset()

        if curriculum.size() < MIN_CURRICULUM:
            print(
                f"\n[Phase {phase}] Pool below minimum "
                f"({curriculum.size()} < {MIN_CURRICULUM}). "
                f"Natural ceiling reached — stopping."
            )
            break

        if total_steps >= args.max_steps:
            print(f"\nReached max_steps ({args.max_steps}). Stopping.")
            break

        steps_this_phase = min(STEPS_PER_PHASE, args.max_steps - total_steps)
        print(f"\n{'='*60}")
        print(f"  Phase {phase}  |  {curriculum.size()} problems  |  {steps_this_phase} steps")
        print(f"{'='*60}")

        grpo_config = GRPOConfig(
            output_dir=str(out_dir / f"phase_{phase:03d}"),
            max_steps=steps_this_phase,
            # H100: batch 8 × grad_accum 2 = effective batch 16
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=min(5, steps_this_phase // 4),
            logging_steps=5,
            save_strategy="no",
            num_generations=N_ROLLOUTS_EVAL,
            max_completion_length=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            use_vllm=False,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,   # trade compute for memory at large batch
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )
        trainer.train()
        total_steps += steps_this_phase

        ckpt = out_dir / f"checkpoint_phase_{phase:03d}"
        model.save_pretrained(ckpt)
        print(f"  Checkpoint → {ckpt}")

        curriculum.refresh(model, tokenizer, phase)

    log_path = out_dir / "curriculum_log.json"
    with open(log_path, "w") as f:
        json.dump(curriculum.phase_logs, f, indent=2)
    print(f"\nCurriculum log → {log_path}")

    final_path = out_dir / "final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final adapter  → {final_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="llama-3-8b")
    parser.add_argument("--max-steps",  type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    model_id = MODEL_REGISTRY[args.model]
    model, tokenizer = load_model_and_tokenizer(model_id)

    print("\nBuilding curriculum…")
    curriculum = build_curriculum(args.model)

    run_training(model, tokenizer, curriculum, args)


if __name__ == "__main__":
    main()
