"""
Analyze Sweet Spot vs Entity Filter Approach.

Uses the existing pass@16 test data to demonstrate why difficulty-based
filtering is superior to entity-count filtering.

Usage:
    python src/analyze_sweet_spot.py
"""

import json
from pathlib import Path
from collections import Counter

# Config
PASS_AT_16_FILE = "rl-intro_logs/baseline_16_pass.jsonl"
MIN_CORRECT = 2
MAX_CORRECT = 12
K = 16


def estimate_ghost_batch_rate(n_correct, k=16, batch_size=8):
    """Estimate probability of a ghost batch (all correct or all wrong).

    For a problem with n_correct/k pass rate, what's the probability that
    a random sample of batch_size completions are all correct or all wrong?

    This is a simplified estimate assuming independence.
    """
    p = n_correct / k  # pass rate

    # P(all correct) + P(all wrong)
    all_correct = p ** batch_size
    all_wrong = (1 - p) ** batch_size

    return all_correct + all_wrong


def main():
    print("=" * 70)
    print("  Sweet Spot Analysis: Why Difficulty Beats Entity Count")
    print("=" * 70)

    # Load pass@16 data
    if not Path(PASS_AT_16_FILE).exists():
        print(f"ERROR: {PASS_AT_16_FILE} not found")
        return

    print(f"\n>>> Loading pass@16 test data ...")
    results = []
    with open(PASS_AT_16_FILE, "r") as f:
        for line in f:
            results.append(json.loads(line))
    print(f"    Loaded {len(results)} problems\n")

    # Categorize problems
    too_hard = [r for r in results if r["n_correct"] < MIN_CORRECT]
    sweet_spot = [r for r in results if MIN_CORRECT <= r["n_correct"] <= MAX_CORRECT]
    too_easy = [r for r in results if r["n_correct"] > MAX_CORRECT]

    print(f">>> Difficulty breakdown:")
    print(f"    Too hard (0-{MIN_CORRECT-1}/{K}):   {len(too_hard):3d} problems ({100*len(too_hard)/len(results):.0f}%)")
    print(f"    Sweet spot ({MIN_CORRECT}-{MAX_CORRECT}/{K}): {len(sweet_spot):3d} problems ({100*len(sweet_spot)/len(results):.0f}%)")
    print(f"    Too easy ({MAX_CORRECT+1}-{K}/{K}):  {len(too_easy):3d} problems ({100*len(too_easy)/len(results):.0f}%)")

    # Estimate ghost batch rates
    print(f"\n>>> Ghost batch rate estimates (batch_size=8):")

    ghost_rates = {
        "too_hard": [estimate_ghost_batch_rate(r["n_correct"]) for r in too_hard],
        "sweet_spot": [estimate_ghost_batch_rate(r["n_correct"]) for r in sweet_spot],
        "too_easy": [estimate_ghost_batch_rate(r["n_correct"]) for r in too_easy],
    }

    for category, rates in ghost_rates.items():
        if rates:
            avg_ghost = sum(rates) / len(rates)
            print(f"    {category:15s}: {avg_ghost*100:5.1f}% avg ghost batches")
        else:
            print(f"    {category:15s}: N/A (no problems)")

    # Overall ghost rate for different filtering strategies
    print(f"\n>>> Expected ghost batch rate by filtering strategy:")

    # Strategy 1: No filter (use all problems)
    all_ghost_rates = [estimate_ghost_batch_rate(r["n_correct"]) for r in results]
    avg_all = sum(all_ghost_rates) / len(all_ghost_rates)
    print(f"    No filter (all problems):    {avg_all*100:5.1f}%")

    # Strategy 2: Sweet spot filter
    if ghost_rates["sweet_spot"]:
        avg_sweet = sum(ghost_rates["sweet_spot"]) / len(ghost_rates["sweet_spot"])
        improvement = ((avg_all - avg_sweet) / avg_all) * 100
        print(f"    Sweet spot filter:           {avg_sweet*100:5.1f}% ({improvement:+.0f}% reduction)")

    # Key insight: Show examples
    print(f"\n>>> Why sweet spot is optimal:")
    print(f"\n    Example: Problem with 16/16 correct (too easy)")
    print(f"    - Ghost batch probability: {estimate_ghost_batch_rate(16)*100:.1f}%")
    print(f"    - All 8 completions succeed → no gradient signal")
    print(f"    - Wastes compute, no learning")

    print(f"\n    Example: Problem with 0/16 correct (too hard)")
    print(f"    - Ghost batch probability: {estimate_ghost_batch_rate(0)*100:.1f}%")
    print(f"    - All 8 completions fail → no gradient signal")
    print(f"    - Wastes compute, no learning")

    print(f"\n    Example: Problem with 7/16 correct (sweet spot)")
    print(f"    - Ghost batch probability: {estimate_ghost_batch_rate(7)*100:.1f}%")
    print(f"    - Mix of success/failure → strong learning signal")
    print(f"    - GRPO learns from within-group contrast")

    # Show actual training log comparison
    print(f"\n{'=' * 70}")
    print(f"  Comparison to Entity Filter Results")
    print(f"{'=' * 70}")

    print(f"\n  Entity filter (from tiny-math-solver-breakdown.md):")
    print(f"    - Ghost batch rate: 40-42% (measured during training)")
    print(f"    - 42.9% of steps had ≥50% ghost batches")
    print(f"    - Training plateaued at step ~15")
    print(f"    - Only targeted 10/55 identified failure modes")

    print(f"\n  Sweet spot filter (predicted):")
    if ghost_rates["sweet_spot"]:
        print(f"    - Ghost batch rate: ~{avg_sweet*100:.0f}% (estimated)")
        print(f"    - Direct measurement of difficulty")
        print(f"    - Targets all problems with learning signal")
        print(f"    - Should reduce plateau, improve sample efficiency")

    print(f"\n{'=' * 70}")
    print(f"  Conclusion: Sweet spot filter is more principled")
    print(f"{'=' * 70}")
    print(f"\n  The entity filter was a proxy for difficulty.")
    print(f"  The sweet spot filter MEASURES difficulty directly.")
    print(f"\n  This is the core CalibrateRL insight:")
    print(f"  Calibrate training data to model capability, not heuristics.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
