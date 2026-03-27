"""
Compare Entity Filter vs Sweet Spot training runs.

Extracts key metrics from training logs and presents side-by-side comparison.

Usage:
    python3 compare_runs.py
"""

import re
from pathlib import Path
from collections import defaultdict


def parse_training_log(log_path):
    """Extract metrics from GRPO training log."""
    if not Path(log_path).exists():
        return None

    metrics_by_step = defaultdict(dict)

    with open(log_path, 'r') as f:
        content = f.read()

    # Look for step metrics in the log
    # TRL logs usually have format like: {'train/correctness': 0.58, ...}

    # Try to extract from structured logging
    step_pattern = r"Step (\d+)"
    correctness_pattern = r"correctness['\"]?\s*[:=]\s*([\d.]+)"
    ghost_pattern = r"ghost['\"]?\s*[:=]\s*([\d.]+)"
    kl_pattern = r"kl['\"]?\s*[:=]\s*([\d.]+)"

    # Split by steps
    step_sections = re.split(r'Step \d+', content)
    step_nums = [int(m.group(1)) for m in re.finditer(step_pattern, content)]

    for step_num, section in zip(step_nums, step_sections[1:]):
        metrics = {}

        # Extract metrics
        corr_match = re.search(correctness_pattern, section, re.IGNORECASE)
        if corr_match:
            metrics['correctness'] = float(corr_match.group(1))

        ghost_match = re.search(ghost_pattern, section, re.IGNORECASE)
        if ghost_match:
            metrics['ghost'] = float(ghost_match.group(1))

        kl_match = re.search(kl_pattern, section, re.IGNORECASE)
        if kl_match:
            metrics['kl'] = float(kl_match.group(1))

        if metrics:
            metrics_by_step[step_num] = metrics

    return metrics_by_step


def aggregate_metrics(metrics_by_step, step_range):
    """Compute average metrics over a step range."""
    start, end = step_range
    values = defaultdict(list)

    for step in range(start, end + 1):
        if step in metrics_by_step:
            for key, val in metrics_by_step[step].items():
                values[key].append(val)

    if not values:
        return None

    return {
        key: sum(vals) / len(vals)
        for key, vals in values.items()
    }


def format_comparison_table(entity_metrics, sweet_metrics):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("TRAINING COMPARISON: Entity Filter vs Sweet Spot")
    print("=" * 80)
    print()

    # Define step ranges to compare
    ranges = [
        ("Steps 1-10", (1, 10)),
        ("Steps 11-50", (11, 50)),
    ]

    for range_name, step_range in ranges:
        print(f"\n{range_name}:")
        print("-" * 80)

        if entity_metrics:
            entity_agg = aggregate_metrics(entity_metrics, step_range)
        else:
            entity_agg = None

        if sweet_metrics:
            sweet_agg = aggregate_metrics(sweet_metrics, step_range)
        else:
            sweet_agg = None

        # Print table
        print(f"{'Metric':<20} {'Entity Filter':<20} {'Sweet Spot':<20} {'Delta':<15}")
        print("-" * 80)

        metrics_to_show = ['correctness', 'ghost', 'kl']

        for metric in metrics_to_show:
            entity_val = entity_agg.get(metric) if entity_agg else None
            sweet_val = sweet_agg.get(metric) if sweet_agg else None

            entity_str = f"{entity_val:.3f}" if entity_val is not None else "N/A"
            sweet_str = f"{sweet_val:.3f}" if sweet_val is not None else "N/A"

            if entity_val is not None and sweet_val is not None:
                delta = sweet_val - entity_val

                # For ghost rate, lower is better
                if metric == 'ghost':
                    pct_change = ((entity_val - sweet_val) / entity_val) * 100
                    delta_str = f"{delta:+.3f} ({pct_change:+.0f}%)"
                else:
                    delta_str = f"{delta:+.3f}"
            else:
                delta_str = "N/A"

            print(f"{metric:<20} {entity_str:<20} {sweet_str:<20} {delta_str:<15}")

    print()
    print("=" * 80)


def print_verdict(entity_metrics, sweet_metrics):
    """Print final verdict."""
    print("\nVERDICT")
    print("=" * 80)

    if not sweet_metrics:
        print("⚠️  Sweet spot training not complete yet.")
        print("   Run: python3 src/train_grpo.py")
        return

    # Get final metrics
    entity_final = aggregate_metrics(entity_metrics, (11, 50)) if entity_metrics else {}
    sweet_final = aggregate_metrics(sweet_metrics, (11, 50)) if sweet_metrics else {}

    entity_ghost = entity_final.get('ghost', 0)
    sweet_ghost = sweet_final.get('ghost', 0)

    ghost_reduction = ((entity_ghost - sweet_ghost) / entity_ghost) * 100 if entity_ghost > 0 else 0

    print(f"\nGhost Batch Rate Reduction: {ghost_reduction:.0f}%")
    print(f"  Entity filter: {entity_ghost*100:.1f}%")
    print(f"  Sweet spot:    {sweet_ghost*100:.1f}%")

    if ghost_reduction > 50:
        print("\n✅ THESIS VALIDATED: Difficulty calibration >> entity heuristic")
        print(f"   Ghost batching reduced by {ghost_reduction:.0f}%")
    elif ghost_reduction > 20:
        print("\n✓ Improvement confirmed, but less dramatic than predicted")
    else:
        print("\n⚠️  Ghost rate similar to entity filter")
        print("   May need larger sweet spot sample or different hyperparameters")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("COMPARING TRAINING RUNS")
    print("=" * 80)

    # Known entity filter results from breakdown doc
    print("\nEntity Filter Results (from tiny-math-solver-breakdown.md):")
    print("  Steps 1-10:   Correctness 57.8%, Ghost 18.8%, KL 0.0002")
    print("  Steps 11-50:  Correctness 81.4%, Ghost 41.6%, KL 0.019")
    print("  Plateaued at step ~15")

    # Try to parse sweet spot log
    sweet_log_path = "logs/grpo_sweet_spot_50step.log"

    if Path(sweet_log_path).exists():
        print(f"\nParsing sweet spot log: {sweet_log_path}")
        sweet_metrics = parse_training_log(sweet_log_path)

        if sweet_metrics:
            print(f"  Found metrics for {len(sweet_metrics)} steps")
        else:
            print("  ⚠️  Could not parse metrics from log")
            sweet_metrics = None
    else:
        print(f"\n⚠️  Sweet spot log not found: {sweet_log_path}")
        print("   Train first with: python3 src/train_grpo.py")
        sweet_metrics = None

    # Use hardcoded entity metrics from breakdown doc
    entity_metrics = {
        1: {'correctness': 0.578, 'ghost': 0.188, 'kl': 0.0002},
        10: {'correctness': 0.578, 'ghost': 0.188, 'kl': 0.0002},
        11: {'correctness': 0.814, 'ghost': 0.416, 'kl': 0.019},
        50: {'correctness': 0.814, 'ghost': 0.416, 'kl': 0.019},
    }

    # Fill in intermediate steps with interpolation
    for step in range(2, 10):
        entity_metrics[step] = entity_metrics[1].copy()
    for step in range(12, 50):
        entity_metrics[step] = entity_metrics[11].copy()

    # Print comparison
    format_comparison_table(entity_metrics, sweet_metrics)

    if sweet_metrics:
        print_verdict(entity_metrics, sweet_metrics)
    else:
        print("\n" + "=" * 80)
        print("Next: Run training to get sweet spot metrics")
        print("  python3 src/train_grpo.py")
        print("=" * 80)


if __name__ == "__main__":
    main()
