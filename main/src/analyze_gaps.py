"""
Gap Analysis: Categorize *why* first attempts fail on problems the model CAN solve.

Reads logs/pass_at_k.jsonl (output of eval_pass_at_k.py) and for every problem
where pass@1 failed but pass@K >= 1, heuristically classifies the failure into
one of five thematic categories:

  1. Calculation Error  — correct reasoning path but wrong arithmetic
  2. Premise Mismatch   — hallucinated or misread quantities
  3. Repetition / Loop  — model gets stuck repeating reasoning
  4. Early Termination  — abnormally short response or mid-sentence cutoff
  5. Format Failure      — reasoning looks fine but no parseable final answer

Usage:
    python src/analyze_gaps.py [--input logs/pass_at_k.jsonl]

Outputs:
    - Terminal: summary statistics per failure category
    - logs/gap_analysis.jsonl: per-problem annotated results
"""

import argparse
import json
import os
import re
from collections import Counter

# ── Reuse helpers from reward_func ──────────────────────────────────────────
# We import from reward_func to stay DRY. These are the same functions used
# during training, so our analysis categories align with the reward signal.
import sys
sys.path.insert(0, os.path.dirname(__file__))

from reward_func import (
    extract_gold_answer,
    extract_gold_steps,
    extract_predicted_answer,
    extract_question_numbers,
    _extract_all_numbers,
    _normalize_number,
    _numbers_match,
)


# ── Failure classifiers ────────────────────────────────────────────────────

def _trigram_repetition_ratio(text: str) -> float:
    """Fraction of trigrams that are repeated 3+ times.

    High ratio → model is stuck in a reasoning loop.
    """
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    if not trigrams:
        return 0.0
    repeated = sum(1 for t in trigrams if counts[t] >= 3)
    return repeated / len(trigrams)


def _has_repeated_lines(text: str, threshold: int = 3) -> bool:
    """Check if any non-trivial line appears >= threshold times."""
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 15]
    if not lines:
        return False
    counts = Counter(lines)
    return any(c >= threshold for c in counts.values())


def classify_failure(
    question: str,
    gold_label: str,
    wrong_response: str,
    correct_response: str | None,
    gold_answer: str | None,
) -> tuple[str, str]:
    """Classify why a wrong response failed.

    Returns (category, explanation) where category is one of:
        calculation_error, premise_mismatch, repetition_loop,
        early_termination, format_failure
    """
    pred, method = extract_predicted_answer(wrong_response)
    response_len = len(wrong_response.split())
    correct_len = len(correct_response.split()) if correct_response else response_len

    # ── 1. Format Failure: no parseable answer at all ───────────────────
    if pred is None:
        return "format_failure", "No parseable numerical answer found in response."

    # ── 2. Early Termination ────────────────────────────────────────────
    # Response is <40% the length of a correct response, or ends mid-word
    if correct_response and response_len < correct_len * 0.4 and response_len < 50:
        return "early_termination", (
            f"Response length ({response_len} words) is "
            f"<40% of correct solution ({correct_len} words)."
        )
    if wrong_response.rstrip()[-1:] not in (".", "}", ")", "!", "?", "*", "0", "1",
                                              "2", "3", "4", "5", "6", "7", "8", "9"):
        # Ends mid-sentence (no terminal punctuation or digit)
        if response_len > 10:  # not just a short answer
            pass  # could be mid-sentence, but let other checks run first

    # ── 3. Repetition / Logical Loop ────────────────────────────────────
    rep_ratio = _trigram_repetition_ratio(wrong_response)
    has_rep_lines = _has_repeated_lines(wrong_response)
    if rep_ratio > 0.25 or has_rep_lines:
        return "repetition_loop", (
            f"Trigram repetition ratio={rep_ratio:.2f}, "
            f"repeated lines={has_rep_lines}."
        )

    # ── 4. Calculation Error ────────────────────────────────────────────
    # Model found many of the correct intermediate steps but got the final
    # answer wrong → it had the right *logic* but messed up arithmetic.
    if gold_label:
        gold_steps = extract_gold_steps(gold_label)
        question_nums = extract_question_numbers(question)
        intermediate = gold_steps[:-1] if len(gold_steps) >= 2 else []
        unique_steps = [
            s for s in intermediate
            if not any(abs(s - qn) < 1e-6 for qn in question_nums)
        ]
        if unique_steps:
            response_nums = _extract_all_numbers(wrong_response)
            n_found = sum(
                1 for s in unique_steps
                if any(abs(s - rn) < 1e-6 for rn in response_nums)
            )
            if n_found >= len(unique_steps) * 0.5 and len(unique_steps) >= 1:
                return "calculation_error", (
                    f"Found {n_found}/{len(unique_steps)} correct intermediate "
                    f"steps but final answer '{pred}' != gold '{gold_answer}'."
                )

    # ── 5. Premise Mismatch (hallucinated quantities) ───────────────────
    # Numbers in the response that are NOT in the question and NOT in any
    # gold intermediate step → the model invented quantities.
    if gold_label:
        gold_steps_all = extract_gold_steps(gold_label)
        gold_nums = set(gold_steps_all) | extract_question_numbers(question)
        response_nums = _extract_all_numbers(wrong_response)
        # Filter out very small numbers (1, 2, 3...) which are common counters
        ungrounded = {
            n for n in response_nums
            if n > 10 and not any(abs(n - gn) < 1e-6 for gn in gold_nums)
        }
        if len(ungrounded) >= 2:
            examples = sorted(ungrounded)[:5]
            return "premise_mismatch", (
                f"Response contains {len(ungrounded)} numbers not in "
                f"question or gold steps: {examples}."
            )

    # ── Fallback: calculation error (wrong answer, no strong signal) ────
    return "calculation_error", (
        f"Final answer '{pred}' != gold '{gold_answer}', "
        f"no strong signal for other categories."
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Pass@1 vs Pass@K gap")
    parser.add_argument(
        "--input", default="logs/pass_at_k.jsonl",
        help="Path to pass_at_k.jsonl output file",
    )
    parser.add_argument(
        "--output", default="logs/gap_analysis.jsonl",
        help="Path to write annotated gap analysis results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        print("Run eval_pass_at_k.py first to generate Pass@K results.")
        sys.exit(1)

    # Load results
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} problems from {args.input}\n")

    # Filter to "gap" problems: pass@1 failed but pass@K succeeded
    # We identify gap problems as those with 0 < n_correct < K
    # (at least one correct, but not all correct = model is inconsistent)
    # More specifically, the "gap" = problems where wrong_sample exists AND
    # correct_sample exists (model got it right sometimes, wrong other times).
    gap_problems = [
        r for r in records
        if r["n_correct"] > 0 and r["n_correct"] < r["n_total"]
    ]

    print(f"{'='*70}")
    print(f"  GAP ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total problems:        {len(records)}")
    print(f"  Always correct (K/K):  {sum(1 for r in records if r['n_correct'] == r['n_total'])}")
    print(f"  Always wrong (0/K):    {sum(1 for r in records if r['n_correct'] == 0)}")
    print(f"  Gap problems:          {len(gap_problems)} (have both correct & wrong attempts)")
    print(f"{'='*70}\n")

    # Classify each gap problem
    categories = Counter()
    annotated = []

    for r in gap_problems:
        category, explanation = classify_failure(
            question=r["question"],
            gold_label="",  # We don't have gold_label in pass_at_k output
            wrong_response=r.get("wrong_sample") or "",
            correct_response=r.get("correct_sample"),
            gold_answer=r.get("gold_answer"),
        )
        categories[category] += 1
        annotated.append({
            **r,
            "failure_category": category,
            "failure_explanation": explanation,
        })

    # Print summary
    print(f"  --- Failure Category Breakdown ({len(gap_problems)} gap problems) ---")
    for cat, count in categories.most_common():
        pct = 100 * count / len(gap_problems) if gap_problems else 0
        bar = "█" * int(pct / 2)
        print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)  {bar}")
    print()

    # Show examples from each category
    for cat in ["calculation_error", "premise_mismatch", "repetition_loop",
                "early_termination", "format_failure"]:
        examples = [a for a in annotated if a["failure_category"] == cat]
        if examples:
            print(f"  --- Example: {cat} ---")
            ex = examples[0]
            q_short = ex["question"][:150] + ("..." if len(ex["question"]) > 150 else "")
            wrong_short = (ex.get("wrong_sample") or "")[:300]
            wrong_short = wrong_short.replace("\n", " | ")
            print(f"  Q: {q_short}")
            print(f"  Gold: {ex['gold_answer']}")
            print(f"  Why:  {ex['failure_explanation']}")
            print(f"  Wrong: {wrong_short}...")
            print()

    # Save annotated results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for record in annotated:
            f.write(json.dumps(record) + "\n")
    print(f"  Annotated results saved to: {args.output}")

    # Print actionable summary
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATIONS FOR REWARD DESIGN")
    print(f"{'='*70}")
    if categories.get("calculation_error", 0) > len(gap_problems) * 0.2:
        print(f"  >> High calculation errors ({categories['calculation_error']}):")
        print(f"     reasoning_reward already helps; consider increasing its weight.")
    if categories.get("premise_mismatch", 0) > len(gap_problems) * 0.1:
        print(f"  >> Premise mismatches ({categories['premise_mismatch']}):")
        print(f"     Add hallucination_penalty to penalize ungrounded numbers.")
    if categories.get("repetition_loop", 0) > len(gap_problems) * 0.1:
        print(f"  >> Repetition loops ({categories['repetition_loop']}):")
        print(f"     Add repetition_penalty to discourage reasoning loops.")
    if categories.get("early_termination", 0) > len(gap_problems) * 0.1:
        print(f"  >> Early terminations ({categories['early_termination']}):")
        print(f"     format_reward can encourage complete step-by-step solutions.")
    if categories.get("format_failure", 0) > len(gap_problems) * 0.1:
        print(f"  >> Format failures ({categories['format_failure']}):")
        print(f"     format_reward will incentivize clear answer formatting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
