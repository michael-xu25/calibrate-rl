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

EVAL_RESULTS_PATH = Path("data/evaluation_results_full.json")   # full 630-problem pass@8 eval
RESERVE_POOL_PATH = Path("data/profile_dataset_L1L2L3.json")

# ── Curriculum parameters ──────────────────────────────────────────────────────

STEPS_PER_PHASE      = 80     # gradient steps per phase before curriculum refresh.
                               # At 150 active problems and batch_size=8: each problem
                               # gets ~5 training presentations per phase before rescoring.
N_ROLLOUTS_TRAIN     = 8      # rollouts GRPOTrainer generates per problem (for loss)
N_ROLLOUTS_SCORE     = 8      # rollouts for curriculum scoring — 9 distinct pass_rate values
                               # (0, 0.125, …, 1.0) giving 6 Goldilocks levels; pass@4
                               # gives only 3, making eviction thresholds too coarse
SATURATE_THRESHOLD   = 0.875  # 7/8 correct → learned, send to cooldown. Advantage at this
                               # point is only 0.22 per rollout (vs 0.50 at p=0.5).
UNREACHABLE_PATIENCE = 3      # consecutive 0/8 evals before sending to cooldown (3 reduces
                               # noise-driven evictions — a 10% pass_rate problem has real
                               # chance of 0/8 twice by chance alone)
TARGET_CURRICULUM    = 150    # active problems to maintain. Chosen so each problem gets
                               # ~5 presentations per 80-step phase (80×8/150 ≈ 4.3).
                               # Below 128 → gradient correlation increases (same problems
                               # repeat too often); above 200 → presentations drop below 3×
                               # and eviction decisions become noise-driven.
MIN_CURRICULUM       = 8      # warn (but keep training) below this; hard stop only if
                               # pool drops to 0. A pool of 8 still fills one full batch
                               # and provides useful gradient — stopping prematurely because
                               # of reserve exhaustion is worse than training on a small pool.
MAX_PROBE            = 40     # reserve candidates to probe per refresh
RESERVE_PATIENCE     = 4      # consecutive 0/N probe fails → send to cooldown
PROBE_DECAY          = 0.5    # each probe failure halves sampling probability
MIN_PROMOTE_PASS_RATE = 0.25  # minimum pass_rate (2/8) to promote from reserve into active.
                               # Filters noise: a true-5%-pass-rate problem has ~28% chance
                               # of scoring 1/8 by luck on a single 8-rollout probe.
SATURATE_COOLDOWN    = 2      # phases a saturated problem spends frozen before re-entering
                               # reserve. 2 phases = 160 steps — enough model movement for
                               # some previously-saturated problems to drop back to Goldilocks.
UNREACHABLE_COOLDOWN = 1      # phases an unreachable problem spends frozen. Shorter because
                               # the model may unlock easy problems quickly in early phases.
MAX_COOLDOWN_CYCLES  = 2      # after this many cooldown cycles, permanently retire the problem.

# ── Generation parameters ──────────────────────────────────────────────────────

GEN_MAX_NEW_TOKENS = 768   # 512 risks cutting off L3 chain-of-thought before <answer> tag
GEN_TEMPERATURE    = 0.8
GEN_TOP_P          = 0.95
HELDOUT_BATCH             = 8   # problems per greedy generate call — batching amortizes H100
                                 # overhead; 8 fits comfortably in VRAM with 7-8B models
STATIC_GOLDILOCKS_SAMPLE  = 64  # problems to sample from static active set each phase
                                 # to measure Goldilocks fraction (key diagnostic)

SYSTEM_PROMPT = (
    "You are a concise mathematical reasoning assistant. "
    "Solve problems step by step, then give your final answer in <answer> tags."
)
ANSWER_REMINDER = (
    "\n\nFinal answer must be last, inside <answer> tags — e.g. <answer>42</answer>"
)

# ── Reward function ────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    # Tier 1: explicit <answer> tag (model is prompted to use this)
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Tier 2: \boxed{} — handles one level of nesting (covers most MATH answers)
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()
    # Tier 3: last bare number in the completion (rare; fires when model skips both formats)
    candidates = re.findall(r"(?<![/\w])(-?\d+(?:[./]\d+)?)(?![/\w])", text)
    return candidates[-1] if candidates else None


def normalize(raw: str) -> str:
    s = raw.strip()
    # Strip all LaTeX math delimiters (strip("$") only removes one from each end)
    s = s.replace("$", "")
    # Normalize fraction variants so \dfrac{6}{11} == \frac{6}{11}
    s = re.sub(r"\\[dt]frac\{", r"\\frac{", s)
    # Convert simple numeric fractions to a/b so \frac{6}{11} == 6/11
    # Handles integers and decimals; complex fracs (nested) stay as \frac and must match verbatim
    s = re.sub(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", r"\1/\2", s)
    # Remove LaTeX thin/medium/thick spacing commands that appear in formatted numbers
    # e.g. 10,\!080 (thin space) or 10,\,080
    s = re.sub(r"\\[,;!: ]", "", s)
    s = s.replace("\\!", "")   # explicit \! not covered by the class above
    # Strip \text{...} → inner content (e.g. \text{ ways} → ways)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # Strip \left / \right bracket decorators
    s = re.sub(r"\\(?:left|right)[(\[{)\]}|.]", "", s)
    # Remove thousands-separator commas: 10,080 → 10080
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    # Collapse trailing .0 on integers: 42.0 → 42
    s = re.sub(r"^(-?\d+)\.0+$", r"\1", s)
    # Remove all whitespace — spaces in LaTeX math are decorative, so "9 \pi" == "9\pi"
    s = re.sub(r"\s+", "", s)
    return s.lower()


def score(completion: str, ground_truth: str) -> float:
    extracted = extract_answer(completion)
    if extracted is None:
        return 0.0
    return 1.0 if normalize(extracted) == normalize(ground_truth) else 0.0


def reward_fn(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    return [score(c, a) for c, a in zip(completions, answer)]


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(problem_text: str, tokenizer) -> str:
    """Format a problem using the model's native chat template.

    Uses tokenizer.apply_chat_template() so Qwen 2.5 gets <|im_start|> tokens
    and Llama 3 gets <|start_header_id|> tokens — each model sees the exact
    format it was instruction-tuned on. The hardcoded <|system|> / <|user|>
    placeholder format is NOT valid special tokens for either model.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem_text + ANSWER_REMINDER},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Batched scoring ────────────────────────────────────────────────────────────

def score_problems_batched(
    items:      List[Tuple[str, dict]],   # [(problem_id, problem_dict), ...]
    model,
    tokenizer,
    label:      str = "scoring",
    n_rollouts: int = N_ROLLOUTS_SCORE,   # caller can override (e.g. N_ROLLOUTS_HELDOUT)
) -> Dict[str, float]:
    """
    Score each problem with n_rollouts rollouts using a single batched
    generate call per problem (batch_size = n_rollouts).

    This is ~8-16× faster than sequential calls because the H100 processes
    all rollouts in parallel rather than waiting for each to complete.

    Falls back to sequential if the batch OOMs (very long problems).
    """
    results = {}
    model.eval()
    t_start = time.monotonic()

    with torch.inference_mode():
        for i, (pid, prob) in enumerate(items):
            prompt   = build_prompt(prob["problem"], tokenizer)
            enc      = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            input_len = enc["input_ids"].shape[1]

            # Repeat prompt n_rollouts times → single batched generate call
            input_ids      = enc["input_ids"].repeat(n_rollouts, 1).to(model.device)
            attention_mask = enc["attention_mask"].repeat(n_rollouts, 1).to(model.device)

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
                    for j in range(n_rollouts)
                ]
            except torch.cuda.OutOfMemoryError:
                # Fallback: score sequentially if this problem's prompt is unusually long
                print(f"\n  [OOM] {pid} — falling back to sequential scoring")
                torch.cuda.empty_cache()
                texts = []
                single_input = enc.to(model.device)
                for _ in range(n_rollouts):
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
                f"{pid:<35}  pass={pr:.3f} ({correct}/{n_rollouts})  "
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
    active:          Dict[str, ProblemState] = field(default_factory=dict)
    reserve:         List[dict]              = field(default_factory=list)
    reserve_strikes: Dict[str, int]          = field(default_factory=dict)
    evicted:         set                     = field(default_factory=set)
    # cooldown: problems temporarily frozen, keyed by pid.
    # Each entry: {"problem": dict, "thaw_phase": int, "initial_strikes": int}
    # Problems sit here for N phases then re-enter reserve at reduced probe priority.
    # After MAX_COOLDOWN_CYCLES total cycles they are permanently retired to evicted.
    cooldown:        Dict[str, dict]         = field(default_factory=dict)
    cooldown_count:  Dict[str, int]          = field(default_factory=dict)  # total cycles per pid
    phase_logs:      List[dict]              = field(default_factory=list)
    phase:           int                     = 0
    total_steps:     int                     = 0
    target_size:     int                     = TARGET_CURRICULUM  # replenishment target for refresh()

    def size(self) -> int:
        return len(self.active)

    def to_dataset(self, tokenizer) -> Dataset:
        return Dataset.from_list([
            {
                "prompt":     build_prompt(s.problem["problem"], tokenizer),
                "answer":     s.problem["answer"],
                "problem_id": pid,
            }
            for pid, s in self.active.items()
        ])

    def _freeze(self, pid: str, problem: dict, reason: str, cooldown_phases: int,
                initial_strikes: int) -> None:
        """Send a problem to the cooldown queue instead of permanently retiring it.

        After MAX_COOLDOWN_CYCLES total cycles the problem is permanently retired.
        'reason' is for logging only.
        """
        cycles_done = self.cooldown_count.get(pid, 0)
        if cycles_done >= MAX_COOLDOWN_CYCLES:
            self.evicted.add(pid)
        else:
            self.cooldown[pid] = {
                "problem":         problem,
                "thaw_phase":      self.phase + cooldown_phases,
                "initial_strikes": initial_strikes,
                "reason":          reason,
            }

    def refresh(self, model, tokenizer, out_dir: Path) -> None:
        self.phase += 1
        t0 = time.monotonic()
        print(f"\n[Phase {self.phase} refresh] Scoring {self.size()} active problems…")

        # ── Step 0: thaw problems whose cooldown has elapsed ─────────────────────
        thawed_pids = [pid for pid, info in self.cooldown.items()
                       if info["thaw_phase"] <= self.phase]
        n_thaw_reserve = n_thaw_retired = 0
        for pid in thawed_pids:
            info  = self.cooldown.pop(pid)
            count = self.cooldown_count.get(pid, 0) + 1
            self.cooldown_count[pid] = count
            if count >= MAX_COOLDOWN_CYCLES:
                self.evicted.add(pid)
                n_thaw_retired += 1
            else:
                self.reserve.append(info["problem"])
                self.reserve_strikes[pid] = info["initial_strikes"]
                n_thaw_reserve += 1
        if thawed_pids:
            print(
                f"[Phase {self.phase} refresh] Thawed {len(thawed_pids)} from cooldown "
                f"(→ reserve: {n_thaw_reserve}, → retired: {n_thaw_retired})"
            )

        # ── Score active set ──────────────────────────────────────────────────────
        pass_rates = score_problems_batched(
            [(pid, s.problem) for pid, s in self.active.items()],
            model, tokenizer,
            label="active",
        )

        # ── Evict → cooldown (not permanent) ─────────────────────────────────────
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

        for pid in evict_sat:
            self._freeze(pid, self.active[pid].problem, "saturated",
                         SATURATE_COOLDOWN, initial_strikes=2)
            del self.active[pid]
        for pid in evict_unreach:
            self._freeze(pid, self.active[pid].problem, "unreachable",
                         UNREACHABLE_COOLDOWN, initial_strikes=1)
            del self.active[pid]

        # ── Replenish from reserve ────────────────────────────────────────────────
        #
        # Probe weighting: base_signal × PROBE_DECAY^strikes
        #   Tier-1 (Goldilocks overflow): base_signal = 1 − |baseline_pr − 0.5| / 0.5
        #     → problems close to p=0.5 at baseline get higher probe priority.
        #     → near-saturated overflow (baseline pr ≈ 0.875) gets low priority since
        #       they've likely already saturated after a phase or two of training.
        #   Tier-2 (unreachable at baseline, no pass_rate field): base_signal = 0.4
        #     → conservative but nonzero — the model may have improved enough.
        #
        # Promotion: only Goldilocks hits (MIN_PROMOTE_PASS_RATE ≤ pr < SATURATE_THRESHOLD)
        # are eligible. Hits sorted by |pr − 0.5| so highest-signal problems are promoted
        # first when supply exceeds demand.
        def probe_weight(p: dict) -> float:
            strikes     = self.reserve_strikes.get(p["id"], 0)
            baseline_pr = p.get("pass_rate")  # present for tier-1, absent for tier-2
            if baseline_pr is not None:
                signal = max(1.0 - abs(baseline_pr - 0.5) / 0.5, 0.1)
            else:
                signal = 0.4
            return signal * (PROBE_DECAY ** strikes)

        needed = self.target_size - self.size()
        added  = []
        if needed > 0 and self.reserve:
            # Send reserve-exhausted problems to cooldown instead of hard-deleting
            exhausted_probs = [p for p in self.reserve
                               if self.reserve_strikes.get(p["id"], 0) >= RESERVE_PATIENCE]
            if exhausted_probs:
                exhausted_ids = {p["id"] for p in exhausted_probs}
                self.reserve = [p for p in self.reserve if p["id"] not in exhausted_ids]
                for prob in exhausted_probs:
                    pid = prob["id"]
                    self._freeze(pid, prob, "probe_exhausted",
                                 SATURATE_COOLDOWN, initial_strikes=2)
                    self.reserve_strikes.pop(pid, None)
                print(
                    f"[Phase {self.phase} refresh] "
                    f"Froze {len(exhausted_probs)} probe-exhausted reserve problems"
                )

            if self.reserve:
                # Probe up to two rounds if the first round undershoots.
                # Round 1: probe max(MAX_PROBE, needed*2) candidates — scales budget
                #   to the actual deficit rather than a fixed ceiling.
                # Round 2: only if round 1 filled < 50% of needed. Probes a fresh
                #   weighted sample from the remaining reserve, capped at needed*2.
                #   Avoids a half-empty active set after unusually large eviction phases.
                goldilocks_hits: List[Tuple[float, str, dict]] = []
                already_probed: set = set()

                for probe_round in range(2):
                    still_needed = needed - len(goldilocks_hits)
                    if still_needed <= 0:
                        break
                    # Only do round 2 if round 1 filled < 50% of deficit
                    if probe_round == 1 and len(goldilocks_hits) >= needed * 0.5:
                        break
                    if not self.reserve:
                        break

                    # Exclude already-probed problems from this round's candidates
                    candidates = [p for p in self.reserve if p["id"] not in already_probed]
                    if not candidates:
                        break

                    weights  = [probe_weight(p) for p in candidates]
                    n_probe  = min(max(MAX_PROBE, still_needed * 2), len(candidates))
                    probe    = random.choices(candidates, weights=weights, k=n_probe)
                    probe    = list({p["id"]: p for p in probe}.values())  # deduplicate
                    already_probed.update(p["id"] for p in probe)

                    round_label = "probe" if probe_round == 0 else "probe-r2"
                    print(
                        f"[Phase {self.phase} refresh] "
                        f"Probing {len(probe)} reserve candidates "
                        f"(round {probe_round + 1}, need {still_needed})…"
                    )
                    scored = score_problems_batched(
                        [(p["id"], p) for p in probe],
                        model, tokenizer,
                        label=round_label,
                    )

                    refreeze: List[Tuple[str, dict]] = []
                    for prob in probe:
                        pid = prob["id"]
                        pr  = scored.get(pid, 0.0)

                        if pr >= SATURATE_THRESHOLD:
                            # Still saturated — re-freeze rather than clogging reserve
                            refreeze.append((pid, prob))
                        elif pr == 0.0:
                            self.reserve_strikes[pid] = self.reserve_strikes.get(pid, 0) + 1
                        else:
                            self.reserve_strikes[pid] = 0  # reset on any nonzero score
                            if pr >= MIN_PROMOTE_PASS_RATE:
                                goldilocks_hits.append((pr, pid, prob))

                    # Remove re-saturated probes from reserve, freeze them
                    if refreeze:
                        refreeze_ids = {pid for pid, _ in refreeze}
                        self.reserve = [r for r in self.reserve if r["id"] not in refreeze_ids]
                        for pid, prob in refreeze:
                            self._freeze(pid, prob, "saturated",
                                         SATURATE_COOLDOWN, initial_strikes=2)
                            self.reserve_strikes.pop(pid, None)

                # Promote highest-signal Goldilocks first (ascending |pr − 0.5|)
                goldilocks_hits.sort(key=lambda x: abs(x[0] - 0.5))
                to_add: Dict[str, dict] = {}
                for pr, pid, prob in goldilocks_hits:
                    if len(to_add) >= needed:
                        break
                    to_add[pid] = prob

                if len(to_add) < needed:
                    shortfall = needed - len(to_add)
                    print(
                        f"[Phase {self.phase} refresh] "
                        f"Reserve shortfall: filled {len(to_add)}/{needed} slots "
                        f"({shortfall} slots temporarily empty — "
                        f"will recover as cooldown thaws or model improves)"
                    )

                added_ids = set(to_add.keys())
                self.reserve = [r for r in self.reserve if r["id"] not in added_ids]
                for pid, prob in to_add.items():
                    self.active[pid] = ProblemState(problem=prob)
                    added.append(pid)

        elapsed = time.monotonic() - t0

        # Goldilocks fraction: fraction of active problems with 0 < pass_rate < 1.
        # Mean gradient signal: mean(2p(1-p)) over ALL active problems (including 0
        # and 1). This is the expected advantage magnitude per rollout, averaged across
        # the batch. For static it declines as saturated/unreachable problems accumulate;
        # for dynamic it stays high because those problems are cycled out.
        n_active = self.size()
        all_pr   = [pass_rates.get(pid, 0.0) for pid in self.active]
        n_goldilocks    = sum(1 for p in all_pr if 0.0 < p < 1.0)
        goldilocks_frac = n_goldilocks / n_active if n_active > 0 else 0.0
        mean_grad_signal = (
            sum(2 * p * (1 - p) for p in all_pr) / n_active if n_active > 0 else 0.0
        )

        log = {
            "phase":              self.phase,
            "evict_saturated":    len(evict_sat),
            "evict_unreachable":  len(evict_unreach),
            "added":              len(added),
            "active_size":        n_active,
            "reserve_size":       len(self.reserve),
            "cooldown_size":      len(self.cooldown),
            "goldilocks_frac":    round(goldilocks_frac, 3),
            "mean_grad_signal":   round(mean_grad_signal, 4),
            "pass_rates":         pass_rates,
            "refresh_time_sec":   round(elapsed, 1),
        }
        self.phase_logs.append(log)

        print(
            f"[Phase {self.phase} refresh] "
            f"frozen(sat={len(evict_sat)} unreach={len(evict_unreach)})  "
            f"added={len(added)}  active={n_active}  "
            f"reserve={len(self.reserve)}  cooldown={len(self.cooldown)}  "
            f"goldilocks={n_goldilocks}/{n_active} ({goldilocks_frac:.1%})  "
            f"grad_signal={mean_grad_signal:.3f}  "
            f"({elapsed:.0f}s)"
        )

        # Save curriculum state after every refresh so we can resume if job dies
        self._save_state(out_dir)

    def _save_state(self, out_dir: Path) -> None:
        state = {
            "phase":           self.phase,
            "total_steps":     self.total_steps,
            "target_size":     self.target_size,
            "evicted":         list(self.evicted),
            "reserve_strikes": self.reserve_strikes,
            "cooldown":        self.cooldown,
            "cooldown_count":  self.cooldown_count,
            "active": [
                {
                    "id":                    pid,
                    "consecutive_zero_evals": s.consecutive_zero_evals,
                    "problem":               s.problem,
                }
                for pid, s in self.active.items()
            ],
            "reserve":     self.reserve,
            "phase_logs":  self.phase_logs,
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
            reserve_strikes=state.get("reserve_strikes", {}),
            evicted=set(state["evicted"]),
            cooldown=state.get("cooldown", {}),
            cooldown_count=state.get("cooldown_count", {}),
            phase_logs=state["phase_logs"],
            phase=state["phase"],
            total_steps=state["total_steps"],
            target_size=state.get("target_size", TARGET_CURRICULUM),
        )
        print(
            f"  Resumed from phase {c.phase}  "
            f"({c.size()} active, {len(c.reserve)} reserve, "
            f"{len(c.cooldown)} in cooldown, {c.total_steps} steps done)"
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

def build_curriculum(model_key: str, static: bool = False) -> Curriculum:
    gold_path = GOLDILOCKS_FILES[model_key]
    if not gold_path.exists():
        raise FileNotFoundError(
            f"{gold_path} not found.\n"
            f"Run: python src/export_goldilocks.py --results data/evaluation_results_full.json"
        )

    with open(gold_path) as f:
        goldilocks = json.load(f)

    if static:
        # Static baseline: ALL 630 training problems, never updated.
        # This is the realistic practitioner baseline — no pre-filtering, just
        # train on your full dataset. Only Goldilocks problems produce nonzero
        # gradient; saturated and unreachable problems silently waste compute.
        # Comparing dynamic (smart curriculum) vs this shows total value of
        # curriculum management: filtering + adaptation combined.
        if not RESERVE_POOL_PATH.exists():
            raise FileNotFoundError(f"Full pool not found: {RESERVE_POOL_PATH}")
        with open(RESERVE_POOL_PATH) as f:
            full_pool = json.load(f)
        active = {p["id"]: ProblemState(problem=p) for p in full_pool}
        print(f"  [static] Active: {len(active)} problems (full 630-problem pool, no filtering)")
        print(f"  [static] No reserve — curriculum frozen for entire run")
        return Curriculum(active=active, reserve=[])

    # ── Dynamic curriculum ───────────────────────────────────────────────────────
    #
    # Seed selection: sort Goldilocks by proximity to pass_rate=0.5.
    # The per-rollout advantage magnitude is 2p(1-p), maximized at p=0.5:
    #   p=0.5  → advantage=0.50 (maximum signal)
    #   p=0.25 → advantage=0.375
    #   p=0.75 → advantage=0.375
    #   p=0.125→ advantage=0.219 (near-unreachable, low signal)
    #   p=0.875→ advantage=0.219 (near-saturated, low signal, will be evicted soon)
    #
    # Starting with the highest-signal problems maximizes useful gradient in phase 1.
    # Low-signal overflow Goldilocks go to the reserve as first-priority fill.
    goldilocks_sorted = sorted(goldilocks, key=lambda p: abs(p["pass_rate"] - 0.5))
    seed     = goldilocks_sorted[:TARGET_CURRICULUM]
    overflow = goldilocks_sorted[TARGET_CURRICULUM:]   # known Goldilocks, lower signal

    active     = {p["id"]: ProblemState(problem=p) for p in seed}
    active_ids = set(active.keys())

    # Reserve: two tiers, differentiated by initial strike count.
    # Tier 1 — overflow Goldilocks (strikes=0): known to produce gradient, promoted first.
    # Tier 2 — unreachable problems (strikes=1): half the probe probability of tier 1.
    #   This biases early probing toward known-useful problems without hard-excluding
    #   unreachable ones (which may unlock as the model improves).
    reserve        = [p for p in overflow]
    reserve_strikes = {}   # tier 1: strikes=0 (default, not set)

    if not RESERVE_POOL_PATH.exists():
        print(f"  WARNING: Reserve pool not found at {RESERVE_POOL_PATH} — reserve is Goldilocks overflow only")
    elif not EVAL_RESULTS_PATH.exists():
        # eval_results_full.json is gitignored (9MB) and won't exist on fresh Lightning AI runs.
        # Fallback: add ALL pool problems not already in active/reserve as tier-2 candidates.
        # Saturated ones will be re-frozen immediately on first probe (pr >= SATURATE_THRESHOLD);
        # unreachable ones get strike=1. This is noisier than the eval-guided version but
        # prevents Qwen's reserve (only 56 overflow) from starving the curriculum.
        print(
            f"  WARNING: {EVAL_RESULTS_PATH} not found — adding ALL pool problems as tier-2 reserve.\n"
            f"  Saturated problems will be re-frozen on first probe. Upload eval_results_full.json\n"
            f"  to {EVAL_RESULTS_PATH} for cleaner tier-2 filtering."
        )
        with open(RESERVE_POOL_PATH) as f:
            full_pool = json.load(f)
        reserve_ids = {r["id"] for r in reserve}
        for p in full_pool:
            if p["id"] not in active_ids and p["id"] not in reserve_ids:
                reserve.append(p)
    else:
        with open(RESERVE_POOL_PATH) as f:
            full_pool = json.load(f)
        with open(EVAL_RESULTS_PATH) as f:
            eval_results = json.load(f)

        base_pass = {}
        for rec in eval_results.values():
            if model_key in rec.get("models", {}):
                base_pass[rec["id"]] = rec["models"][model_key]["pass_rate"]

        saturated_excluded = 0
        for p in full_pool:
            if p["id"] in active_ids or p["id"] in {r["id"] for r in reserve}:
                continue
            pr = base_pass.get(p["id"], -1)
            if pr >= 1.0:
                saturated_excluded += 1
                continue
            if pr == 0.0:
                reserve.append(p)
                # tier-2 priority handled by probe_weight(): no pass_rate field → signal=0.4

        print(f"  Excluded {saturated_excluded} base-model-saturated problems from reserve")

    n_overflow = len(overflow)
    n_unreach  = len(reserve) - n_overflow
    print(f"  Active:  {len(active)} problems (highest-signal Goldilocks, sorted by |pr−0.5|)")
    print(f"  Reserve: {len(reserve)} problems  "
          f"({n_overflow} Goldilocks overflow [tier 1]  +  {n_unreach} unreachable [tier 2])")
    return Curriculum(active=active, reserve=reserve,
                      reserve_strikes=reserve_strikes, target_size=TARGET_CURRICULUM)


# ── Held-out evaluation ────────────────────────────────────────────────────────

def eval_heldout(model, tokenizer, heldout_path: Path, phase: int, out_dir: Path) -> None:
    """
    Evaluate current model on held-out problems using greedy decoding (pass@1).

    Greedy is the standard MATH benchmark protocol and is the right choice here:
    - Deterministic: running the same checkpoint twice gives the same number.
    - Measures the model's best answer, not a noisy sample — cleaner learning curve.
    - 1 forward pass per problem (vs 4 for pass@4): overhead is 210/5120 ≈ 4%.
    - SE of mean over 210 problems ≈ 3.5%, sufficient to track trends across 6 phases.
    Curriculum scoring still uses pass@8 (N_ROLLOUTS_SCORE) because per-problem
    eviction decisions need precision; the held-out curve only needs trend signal.
    """
    with open(heldout_path) as f:
        problems = json.load(f)

    print(f"\n[Phase {phase}] Held-out eval ({len(problems)} problems, greedy, batch={HELDOUT_BATCH})…")

    model.eval()
    pass_rates: Dict[str, float] = {}
    t_start = time.monotonic()

    # Batch greedy decoding: process HELDOUT_BATCH problems simultaneously.
    # Greedy is naturally batchable (no per-rollout branching) — this uses the
    # H100's throughput rather than running 210 sequential single-sequence calls.
    with torch.inference_mode():
        for batch_start in range(0, len(problems), HELDOUT_BATCH):
            batch = problems[batch_start : batch_start + HELDOUT_BATCH]
            prompts = [build_prompt(p["problem"], tokenizer) for p in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Track each sequence's original prompt length for trimming the output
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

            out = model.generate(
                **enc,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j, p in enumerate(batch):
                text   = tokenizer.decode(
                    out[j][int(prompt_lens[j]):], skip_special_tokens=True
                )
                reward = score(text, p["answer"])
                pass_rates[p["id"]] = float(reward)

            done    = min(batch_start + HELDOUT_BATCH, len(problems))
            elapsed = time.monotonic() - t_start
            rate    = done / elapsed
            eta     = (len(problems) - done) / rate if rate > 0 else 0
            correct_in_batch = sum(pass_rates.get(p["id"], 0) for p in batch)
            print(
                f"  [heldout] {done:>3}/{len(problems)}  "
                f"batch correct: {correct_in_batch}/{len(batch)}  eta {eta:.0f}s",
                flush=True,
            )

    model.train()

    by_level   = {1: [], 2: [], 3: []}
    by_subject = {}
    for p in problems:
        pr = pass_rates.get(p["id"], 0.0)
        by_level[p["level"]].append(pr)
        by_subject.setdefault(p["subject"], []).append(pr)

    overall = sum(pass_rates.values()) / len(pass_rates) if pass_rates else 0.0
    result  = {
        "phase":    phase,
        "overall":  round(overall, 4),
        "by_level": {f"L{l}": round(sum(v)/len(v), 4) if v else 0.0
                     for l, v in by_level.items()},
        "by_subject": {s: round(sum(v)/len(v), 4) for s, v in by_subject.items()},
        "pass_rates": pass_rates,
    }

    # Append to JSONL so each phase is a row
    log_path = out_dir / "heldout_results.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(
        f"[Phase {phase}] Held-out — overall={overall:.3f}  "
        + "  ".join(f"L{l}={result['by_level'][f'L{l}']:.3f}" for l in (1,2,3))
    )


# ── Static Goldilocks fraction logging ─────────────────────────────────────────

def log_static_goldilocks(
    curriculum: "Curriculum",
    model,
    tokenizer,
    phase: int,
    out_dir: Path,
) -> None:
    """
    Sample STATIC_GOLDILOCKS_SAMPLE problems from the static active set and score
    them with pass@8 to measure the current Goldilocks fraction.

    This is the key diagnostic for the static baseline: as training progresses,
    saturated problems accumulate and the Goldilocks fraction collapses — showing
    that the static run is wasting an increasing fraction of compute on zero-gradient
    problems. Without this measurement, the compute waste is invisible.

    Logs one JSON line per phase to static_goldilocks_log.jsonl.
    """
    active_list = list(curriculum.active.items())
    n_sample    = min(STATIC_GOLDILOCKS_SAMPLE, len(active_list))
    sample      = random.sample(active_list, n_sample)

    print(f"\n[Phase {phase}] Static Goldilocks probe ({n_sample} problems)…")
    pass_rates = score_problems_batched(
        [(pid, state.problem) for pid, state in sample],
        model, tokenizer,
        label="static-probe",
    )

    all_pr = list(pass_rates.values())
    n_gold = sum(1 for p in all_pr if 0.0 < p < 1.0)
    n_sat  = sum(1 for p in all_pr if p >= SATURATE_THRESHOLD)
    n_un   = sum(1 for p in all_pr if p == 0.0)
    frac   = n_gold / n_sample if n_sample > 0 else 0.0
    mean_grad_signal = (
        sum(2 * p * (1 - p) for p in all_pr) / n_sample if n_sample > 0 else 0.0
    )

    log = {
        "phase":             phase,
        "sample_size":       n_sample,
        "n_goldilocks":      n_gold,
        "n_saturated":       n_sat,
        "n_unreachable":     n_un,
        "goldilocks_frac":   round(frac, 3),
        "mean_grad_signal":  round(mean_grad_signal, 4),
        "pass_rates":        pass_rates,
    }

    log_path = out_dir / "static_goldilocks_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log) + "\n")

    print(
        f"[Phase {phase}] Static Goldilocks fraction: "
        f"{n_gold}/{n_sample} ({frac:.1%})  "
        f"[sat={n_sat} unreach={n_un}]"
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def run_training(model, tokenizer, curriculum: Curriculum, args) -> None:
    run_label = "static" if args.static else "dynamic"
    out_dir = args.output_dir / f"{args.model}_{run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if curriculum.size() == 0:
            print(f"\n[Phase {curriculum.phase + 1}] Active pool empty — stopping.")
            break

        if curriculum.size() < MIN_CURRICULUM:
            print(
                f"\n[Phase {curriculum.phase + 1}] WARNING: pool low "
                f"({curriculum.size()} < {MIN_CURRICULUM}). "
                f"Continuing — cooldown may thaw problems next phase."
            )

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
            learning_rate=2e-5,   # 2e-5 > conservative 1e-5: need enough model movement
                                  # per phase to trigger saturation events within budget
            warmup_steps=min(5, steps_this_phase // 4) if curriculum.phase == 0 else 0,
            logging_steps=5,
            save_strategy="no",
            num_generations=N_ROLLOUTS_TRAIN,
            max_completion_length=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            beta=0.02,            # KL penalty: 0.02 (halved from 0.04 default) to let
                                  # the policy drift faster, making saturation events
                                  # visible within 640 steps. Safe with exact-match
                                  # binary rewards — reward hacking requires actually
                                  # solving the problem, not fooling a soft judge.
            use_vllm=False,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        # Callback: write loss + reward metrics to a persistent JSONL after every
        # logging_steps. This survives session disconnects (unlike stdout) and gives
        # a step-level training signal to compare against held-out accuracy curves.
        loss_log_path = out_dir / "train_loss.jsonl"

        from transformers import TrainerCallback

        class LossLogger(TrainerCallback):
            def on_log(self, _args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                entry = {
                    "step":          state.global_step + curriculum.total_steps,
                    "phase":         curriculum.phase + 1,
                    **{k: v for k, v in logs.items()
                       if isinstance(v, (int, float))},
                }
                with open(loss_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=curriculum.to_dataset(tokenizer),
            reward_funcs=reward_fn,
            processing_class=tokenizer,
            callbacks=[LossLogger()],
        )
        trainer.train()
        curriculum.total_steps += steps_this_phase

        ckpt = out_dir / f"checkpoint_phase_{curriculum.phase + 1:03d}"
        model.save_pretrained(ckpt)
        print(f"  Checkpoint → {ckpt}")

        # Held-out eval after every phase (same problems, never trained on)
        if args.heldout.exists():
            eval_heldout(model, tokenizer, args.heldout, curriculum.phase + 1, out_dir)

        if args.static:
            # Static baseline: skip curriculum refresh, train on same problems forever.
            # But sample-score a subset to track Goldilocks fraction over time — this
            # is the key diagnostic showing gradient quality collapsing as the fixed
            # set saturates. Without it, the compute waste is invisible in the logs.
            log_static_goldilocks(curriculum, model, tokenizer, curriculum.phase + 1, out_dir)
            curriculum.phase += 1
            curriculum._save_state(out_dir)
        else:
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
    parser.add_argument("--max-steps",  type=int,  default=640)  # 8 phases × 80 steps
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from saved curriculum_state.json")
    parser.add_argument("--static", action="store_true",
                        help="Static baseline: no curriculum refresh (control experiment)")
    parser.add_argument("--heldout", type=Path, default=Path("data/heldout_eval.json"),
                        help="Held-out eval set evaluated after each phase")
    args = parser.parse_args()

    model_id  = MODEL_REGISTRY[args.model]
    run_label = "static" if args.static else "dynamic"
    out_dir   = args.output_dir / f"{args.model}_{run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write run_config.json so every checkpoint directory is self-documenting.
    # Critical for comparing runs later — hyperparameters are pinned at launch time,
    # not reconstructed from memory.
    import datetime
    run_config = {
        "model":               args.model,
        "model_id":            model_id,
        "condition":           run_label,
        "max_steps":           args.max_steps,
        "steps_per_phase":     STEPS_PER_PHASE,
        "n_rollouts_train":    N_ROLLOUTS_TRAIN,
        "n_rollouts_score":    N_ROLLOUTS_SCORE,
        "saturate_threshold":  SATURATE_THRESHOLD,
        "unreachable_patience":UNREACHABLE_PATIENCE,
        "target_curriculum":   TARGET_CURRICULUM,
        "min_curriculum":      MIN_CURRICULUM,
        "max_probe":           MAX_PROBE,
        "reserve_patience":    RESERVE_PATIENCE,
        "probe_decay":         PROBE_DECAY,
        "min_promote_pass_rate": MIN_PROMOTE_PASS_RATE,
        "saturate_cooldown":   SATURATE_COOLDOWN,
        "unreachable_cooldown":UNREACHABLE_COOLDOWN,
        "max_cooldown_cycles": MAX_COOLDOWN_CYCLES,
        "lora_r":              32,
        "lora_alpha":          64,
        "learning_rate":       2e-5,
        "beta_kl":             0.02,
        "gen_max_new_tokens":  GEN_MAX_NEW_TOKENS,
        "gen_temperature":     GEN_TEMPERATURE,
        "started_at":          datetime.datetime.utcnow().isoformat() + "Z",
        "resumed":             args.resume,
    }
    config_path = out_dir / "run_config.json"
    if not config_path.exists() or not args.resume:
        config_path.write_text(json.dumps(run_config, indent=2))
        print(f"  Run config → {config_path}")

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
        curriculum = build_curriculum(args.model, static=args.static)

        # Phase 0: eval base model on held-out set before any training.
        # This is the anchor point for the learning curve — without it you
        # can't see where the model started or how much RL actually helped.
        # Skipped on resume because the base model no longer exists at that point.
        if args.heldout.exists():
            print("\nPhase 0 — base model eval (before any training)…")
            eval_heldout(model, tokenizer, args.heldout, phase=0, out_dir=out_dir)

    run_training(model, tokenizer, curriculum, args)


if __name__ == "__main__":
    main()
