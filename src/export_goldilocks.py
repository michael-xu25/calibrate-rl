"""
Goldilocks Problem Inventory — CalibrateRL

Exports per-model curated problem sets where the model is in the
"Goldilocks zone": 0 < pass_rate < 1.0  AND  max_score == 1.

These are the only problems that provide useful GRPO gradient signal:
  - max_score == 1  → model CAN solve it (not impossible noise)
  - 0 < pass_rate < 1 → model DOESN'T always solve it (not saturated)
  - advantage_estimates are non-trivially distributed (not all zeros)

Output files (one per model):
    data/goldilocks_{model_key}.json

Each file is a list of problems sorted by pass_rate ascending within
each subject (hardest solvable first — most gradient signal).

Fields per problem:
    id, subject, level, problem, answer,
    pass_rate, mean_advantage, advantage_std, rollout_rewards

Usage:
    python src/export_goldilocks.py
    python src/export_goldilocks.py --results data/evaluation_results_L1L2.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ── Core logic ─────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Results not found: {path}\nRun src/run_evaluation.py first.")
    with open(path) as f:
        return json.load(f)


def discover_models(results: dict) -> List[str]:
    models = set()
    for rec in results.values():
        models.update(rec.get("models", {}).keys())
    if not models:
        sys.exit("No models found in results file.")
    return sorted(models)


def is_goldilocks(m_data: dict) -> bool:
    return (
        m_data["max_score"] == 1
        and 0.0 < m_data["pass_rate"] < 1.0
    )


def build_inventory(results: dict, model: str) -> List[dict]:
    """
    Collect all Goldilocks problems for a model, sorted by subject
    then pass_rate ascending (hardest solvable first within each subject).
    """
    by_subject = {s: [] for s in SUBJECTS}

    for rec in results.values():
        m_data = rec.get("models", {}).get(model)
        if m_data is None:
            continue
        if not is_goldilocks(m_data):
            continue

        ae = m_data["advantage_estimates"]
        entry = {
            "id":            rec["id"],
            "subject":       rec["subject"],
            "level":         rec.get("level", 1),
            "problem":       rec["problem"],
            "answer":        rec["answer"],
            "pass_rate":     round(m_data["pass_rate"], 4),
            "mean_advantage": round(float(np.mean([abs(x) for x in ae])), 4),
            "advantage_std": round(float(np.std(ae)), 4),
            "rollout_rewards": m_data["rollout_rewards"],
        }
        subj = rec["subject"]
        if subj in by_subject:
            by_subject[subj].append(entry)

    # Sort each subject: ascending pass_rate (hardest solvable first)
    ordered = []
    for subj in SUBJECTS:
        ordered.extend(sorted(by_subject[subj], key=lambda x: x["pass_rate"]))

    return ordered


def print_inventory_summary(inventory: List[dict], model: str, total_evaluated: int) -> None:
    print(f"\n  Model: {model.upper()}")
    print(f"  Goldilocks problems: {len(inventory)} / {total_evaluated} evaluated "
          f"({100*len(inventory)/total_evaluated:.0f}%)" if total_evaluated else "")

    # Per-subject breakdown
    print(f"\n  {'Subject':<28}  {'Count':>5}  {'Pass rate range':>20}")
    print("  " + "-" * 58)
    for subj in SUBJECTS:
        probs = [p for p in inventory if p["subject"] == subj]
        if not probs:
            print(f"  {subj:<28}  {'0':>5}")
            continue
        lo = min(p["pass_rate"] for p in probs)
        hi = max(p["pass_rate"] for p in probs)
        mean_pr = sum(p["pass_rate"] for p in probs) / len(probs)
        mean_adv = sum(p["mean_advantage"] for p in probs) / len(probs)
        print(f"  {subj:<28}  {len(probs):>5}  "
              f"[{lo:.3f} – {hi:.3f}]  mean_pr={mean_pr:.3f}  |adv|={mean_adv:.3f}")

    # Pass rate distribution
    buckets = {
        "(0.000, 0.250]": 0,
        "(0.250, 0.500]": 0,
        "(0.500, 0.750]": 0,
        "(0.750, 1.000)": 0,
    }
    for p in inventory:
        pr = p["pass_rate"]
        if pr <= 0.25:   buckets["(0.000, 0.250]"] += 1
        elif pr <= 0.50: buckets["(0.250, 0.500]"] += 1
        elif pr <= 0.75: buckets["(0.500, 0.750]"] += 1
        else:            buckets["(0.750, 1.000)"] += 1

    print(f"\n  Pass-rate distribution:")
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"    {bucket}  {count:>3}  {bar}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path("data/evaluation_results.json"),
        help="Path to evaluation_results.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data"),
        help="Directory to write goldilocks_{model}.json files",
    )
    args = parser.parse_args()

    print(f"Loading {args.results}…")
    results = load_results(args.results)
    print(f"  {len(results)} problems loaded")

    models = discover_models(results)
    print(f"  Models: {models}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print("  Goldilocks Inventory Export")
    print(f"{'='*62}")

    for model in models:
        total_evaluated = sum(
            1 for r in results.values() if model in r.get("models", {})
        )
        inventory = build_inventory(results, model)
        print_inventory_summary(inventory, model, total_evaluated)

        out_path = args.output_dir / f"goldilocks_{model}.json"
        with open(out_path, "w") as f:
            json.dump(inventory, f, indent=2)
        print(f"\n  Saved → {out_path}  ({len(inventory)} problems)\n")

    print("=" * 62)


if __name__ == "__main__":
    main()
