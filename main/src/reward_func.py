"""
Rule-Based Reward Functions for GRPO on GSM8K.

Designed from Pass@16 analysis of Qwen2.5-1.5B-Instruct:
- pass@1: 67.7%, pass@16: 95%
- Key failure mode: entity tracking (forgetting people/items in multi-entity problems)
- Training dataset: GSM8K filtered to 3+ entity problems

Active rewards:
  1. correctness_reward:  1.0 if final answer matches gold, 0.0 otherwise
  2. format_reward:       0.1 if answer is wrapped in \\boxed{}, 0.0 otherwise

The format reward prevents RL from degrading the model's answer formatting.
Without it, the model may stop using \\boxed{} since correctness_reward
extracts answers via multiple fallback methods -- the model gets rewarded
the same whether it formats nicely or dumps a bare number.

Legacy (kept but unused):
  intermediate_step_reward:  partial credit for intermediate computation values.
  Dropped because it added no useful within-group variance -- all completions
  tended to produce the same intermediate numbers.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger("tiny_math_solver")

_step_counter = 0
LOG_EVERY = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_text(completion) -> str:
    """Extract plain text from a completion (handles both str and chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        return ""
    return str(completion)


def _normalize_number(s: str) -> float | None:
    """Convert a string number to float for numeric comparison."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _numbers_match(a: str, b: str) -> bool:
    """Numeric comparison: '38.00' == '38', '1,000' == '1000'."""
    a_f = _normalize_number(a)
    b_f = _normalize_number(b)
    if a_f is None or b_f is None:
        return False
    return abs(a_f - b_f) < 1e-6


def _extract_all_numbers(text: str) -> set[float]:
    """Extract all numeric values from text as floats.

    Finds patterns like: 750, 1,200, 36.5, $4,400
    Returns a set of float values for numeric comparison.
    """
    raw = re.findall(r"[\d,]+(?:\.\d+)?", text)
    result = set()
    for r in raw:
        val = _normalize_number(r)
        if val is not None:
            result.add(val)
    return result


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """Extract the model's final answer. Returns (answer, method)."""
    # 1. \boxed{<number>} -- model's preferred format
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"

    # 2. #### <number>
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"

    # 3. "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"

    # 4. **<number>**
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"

    # 5. Last number in text (fallback)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"

    return None, "no_answer"


# ── Intermediate step helpers ────────────────────────────────────────────────

def _extract_intermediate_values(gold_answer: str) -> list[float]:
    """Extract intermediate computation results from GSM8K gold solution.

    GSM8K gold answers contain annotations like:
        "16 + 20 = <<16+20=36>>36"
        "$430 + $320 = <<430+320=750>>$750"

    This extracts the result values after '=' inside <<...>>: [36.0, 750.0, ...]
    These are the key intermediate steps the model should reproduce.

    Returns sorted list (not set) for deterministic behavior.
    """
    results = []
    seen = set()
    for match in re.finditer(r"<<[^>]+=([^>]+)>>", gold_answer):
        val = _normalize_number(match.group(1).strip())
        if val is not None and val not in seen:
            seen.add(val)
            results.append(val)
    return sorted(results)


# ── Reward function 1: Correctness ──────────────────────────────────────────

def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """
    Core reward: 1.0 if final answer is numerically correct, 0.0 otherwise.
    """
    global _step_counter
    _step_counter += 1

    rewards = []
    samples = []
    for completion, gold_label in zip(completions, answer):
        text = _get_text(completion)
        gold = extract_gold_answer(gold_label)
        pred, method = extract_predicted_answer(text)
        correct = gold is not None and pred is not None and _numbers_match(pred, gold)
        rewards.append(1.0 if correct else 0.0)
        samples.append((text, gold, pred, method, correct))

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        n_correct = int(sum(rewards))
        n_parsed = sum(1 for _, _, pred, _, _ in samples if pred is not None)
        print(flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  CORRECTNESS (step {_step_counter}) — "
              f"{n_correct}/{len(rewards)} correct, "
              f"{n_parsed}/{len(rewards)} parsed", flush=True)
        print(f"{'='*70}", flush=True)
        for i, (text, gold, pred, method, correct) in enumerate(samples[:4]):
            snippet = text[:500].replace('\n', ' | ')
            if len(text) > 500:
                snippet += "..."
            status = "CORRECT" if correct else "WRONG  "
            print(f"  [{status}] gold={gold}  pred={pred}  method={method}", flush=True)
            print(f"  Output: {snippet}", flush=True)
            print(f"  ---", flush=True)
        print(f"{'='*70}\n", flush=True)

    return rewards


# ── Reward function 2: Format ─────────────────────────────────────────────────

def format_reward(completions, **kwargs) -> list[float]:
    """
    Small reward for presenting the final answer in \\boxed{}.

    Returns 0.1 if the completion contains \\boxed{<number>}, 0.0 otherwise.
    This prevents RL from degrading the model's formatting -- without it,
    the model learns that bare numbers get the same correctness reward as
    nicely boxed answers, and may stop formatting.

    The 0.1 magnitude is intentionally small relative to correctness (1.0)
    so it never overrides the math signal, but large enough that GRPO
    slightly prefers boxed completions when correctness is tied.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        has_boxed = bool(re.search(r"\\boxed\{[\d,]+(?:\.\d+)?\}", text))
        rewards.append(0.1 if has_boxed else 0.0)
    return rewards


# ── Legacy: Intermediate Step Accuracy (unused) ──────────────────────────────

def intermediate_step_reward(completions, answer, **kwargs) -> list[float]:
    """
    Partial credit for correctly computing intermediate values.

    Extracts intermediate computation results from the gold answer's
    <<expr=result>> annotations and checks how many appear in each completion.

    WHY THIS WORKS FOR GRPO:
    The old entity_tracking_reward gave identical scores to all completions
    (all mention same names → std=0 → zero gradient). This reward VARIES
    because different completions compute different intermediate steps
    correctly. E.g. for intermediates {36, 750, 1450}:
      - Completion A might produce {36, 750, 1450} → score=0.5
      - Completion B might produce {36, 680, 1360} → score=0.167
      - Completion C might produce {40, 800, 1600} → score=0.0
    This creates the within-group variance GRPO needs for learning.

    Score = (matching intermediates) / (total intermediates) * 0.5
    Max reward: 0.5 (correctness_reward at 1.0 always dominates)
    """
    rewards = []
    debug_info = []

    for completion, gold_label in zip(completions, answer):
        text = _get_text(completion)

        # Get intermediate values from gold solution
        gold_intermediates = _extract_intermediate_values(gold_label)

        if not gold_intermediates:
            rewards.append(0.0)
            debug_info.append(("no_intermediates", [], set(), set()))
            continue

        # Extract all numbers the model produced
        completion_numbers = _extract_all_numbers(text)

        # Check which gold intermediates appear in the completion
        # Use numeric matching (handles formatting differences: 1,000 vs 1000)
        matched = set()
        for gold_val in gold_intermediates:
            for comp_val in completion_numbers:
                if abs(gold_val - comp_val) < 1e-6:
                    matched.add(gold_val)
                    break

        # Exclude the final answer from intermediate credit --
        # we don't want to double-count with correctness_reward.
        # The final answer is the last intermediate value.
        final_answer_match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", gold_label)
        if final_answer_match:
            final_val = _normalize_number(final_answer_match.group(1))
            if final_val is not None:
                gold_intermediates_no_final = [v for v in gold_intermediates if abs(v - final_val) > 1e-6]
                matched_no_final = {v for v in matched if abs(v - final_val) > 1e-6}
            else:
                gold_intermediates_no_final = gold_intermediates
                matched_no_final = matched
        else:
            gold_intermediates_no_final = gold_intermediates
            matched_no_final = matched

        n_total = len(gold_intermediates_no_final)
        if n_total == 0:
            rewards.append(0.0)
            debug_info.append(("no_intermediates", [], set(), set()))
            continue

        n_matched = len(matched_no_final)

        # Score: fraction of intermediate steps matched, scaled to [0, 0.5]
        score = (n_matched / n_total) * 0.5
        rewards.append(score)
        debug_info.append(("scored", gold_intermediates_no_final, matched_no_final, completion_numbers))

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"\n  INTERMEDIATE STEPS (step {_step_counter}) — avg={avg_reward:.3f}", flush=True)
        for i, (status, gold_ints, matched, comp_nums) in enumerate(debug_info[:4]):
            if status == "scored":
                n_total = len(gold_ints)
                n_matched = len(matched)
                gold_str = [int(v) if v == int(v) else v for v in gold_ints]
                match_str = [int(v) if v == int(v) else v for v in sorted(matched)]
                print(f"  [{i}] {n_matched}/{n_total} intermediates matched → "
                      f"reward={rewards[i]:.3f}  gold={gold_str} matched={match_str}", flush=True)
            else:
                print(f"  [{i}] {status} → reward=0.0", flush=True)
        print(flush=True)

    return rewards
