"""
Step 1: Dataset Engineering for CalibrateRL Model Profiling

Usage:
    python src/build_profile_dataset.py               # Level 1 only, 20/subject → 140 problems
    python src/build_profile_dataset.py --levels 1 2  # Level 1+2, 15/subject/level → 210 problems
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_ID  = "EleutherAI/hendrycks_math"
RANDOM_SEED = 42

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_level(level_str: str) -> Optional[int]:
    try:
        return int(level_str.split()[-1])
    except (ValueError, AttributeError, IndexError):
        return None


def extract_boxed_answer(solution: str) -> str:
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    return match.group(1).strip() if match else ""


def is_valid_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    if answer.strip().startswith("\\begin{"):
        return False
    if re.fullmatch(r"\\text\{[^}]{1,6}\}", answer.strip()):
        return False
    if len(answer) > 40:
        return False
    return True


# ── Core pipeline ──────────────────────────────────────────────────────────────

def load_and_filter(levels: List[int]) -> Dict[Tuple[str, int], List[dict]]:
    """
    Load all subject configs, keep only problems at the requested levels with valid answers.
    Returns {(subject, level): [problem_dict, ...]}
    """
    print(f"Loading MATH dataset (Levels {levels})…")
    by_subj_level: Dict[Tuple[str, int], List[dict]] = defaultdict(list)

    total_raw = dropped_level = dropped_answer = 0

    for subj in SUBJECTS:
        rows = []
        for split in ("train", "test"):
            try:
                ds = load_dataset(DATASET_ID, subj, split=split)
                rows.extend(list(ds))
            except Exception as e:
                print(f"  Warning: {subj}/{split}: {e}")

        total_raw += len(rows)

        for local_idx, row in enumerate(rows):
            lvl = parse_level(row.get("level", ""))
            if lvl not in levels:
                dropped_level += 1
                continue

            answer = extract_boxed_answer(row.get("solution", ""))
            if not is_valid_answer(answer):
                dropped_answer += 1
                continue

            by_subj_level[(subj, lvl)].append({
                "id":       f"{subj}_L{lvl}_{local_idx}",
                "subject":  subj,
                "level":    lvl,
                "problem":  row["problem"].strip(),
                "solution": row["solution"].strip(),
                "answer":   answer,
            })

    print(f"  Raw loaded:            {total_raw:,}")
    print(f"  Dropped (wrong level): {dropped_level:,}")
    print(f"  Dropped (bad answer):  {dropped_answer:,}")
    print(f"\n  Available problems per (subject, level):")
    for subj in SUBJECTS:
        for lvl in levels:
            n = len(by_subj_level.get((subj, lvl), []))
            print(f"    {subj:<32} L{lvl}  {n:>4}")

    return dict(by_subj_level)


def balanced_sample(
    by_subj_level: Dict[Tuple[str, int], List[dict]],
    levels: List[int],
    n_per_cell: int,
) -> List[dict]:
    """
    Sample exactly n_per_cell problems from each (subject, level) cell.
    """
    rng = random.Random(RANDOM_SEED)
    sampled = []

    for subj in SUBJECTS:
        for lvl in levels:
            pool = by_subj_level.get((subj, lvl), [])
            if len(pool) < n_per_cell:
                raise ValueError(
                    f"({subj}, L{lvl}) only has {len(pool)} valid problems "
                    f"(need {n_per_cell}). Lower n_per_cell or relax filters."
                )
            sampled.extend(rng.sample(pool, n_per_cell))

    rng.shuffle(sampled)
    return sampled


def print_summary(problems: List[dict], levels: List[int], n_per_cell: int) -> None:
    total = len(problems)
    print(f"\n{'='*54}")
    print(f"  Final dataset: {total} problems")
    print(f"  {n_per_cell} per subject per level × {len(SUBJECTS)} subjects × {len(levels)} level(s)")
    print(f"{'='*54}")
    counts = Counter((p["subject"], p["level"]) for p in problems)
    print(f"  {'Subject':<32} " + "  ".join(f"L{l}" for l in levels))
    print(f"  {'-'*48}")
    for subj in SUBJECTS:
        row = "  ".join(f"{counts.get((subj, l), 0):>4}" for l in levels)
        print(f"  {subj:<32} {row}")

    ex = problems[0]
    print(f"\n  Example record:")
    print(f"    id:      {ex['id']}")
    print(f"    subject: {ex['subject']}")
    print(f"    level:   {ex['level']}")
    print(f"    answer:  {ex['answer']!r}")
    print(f"    problem: {ex['problem'][:100]}…")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MATH profiling dataset")
    parser.add_argument(
        "--levels", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1],
        help="Which MATH levels to include (default: 1)",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=None,
        help="Problems per (subject, level) cell (default: 20 for single level, 15 for multi)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: data/profile_dataset[_tag].json)",
    )
    args = parser.parse_args()
    levels = sorted(set(args.levels))

    # Problems per (subject, level) cell
    if args.n_per_cell is not None:
        n_per_cell = args.n_per_cell
    else:
        n_per_cell = 15 if len(levels) > 1 else 20

    # Output path
    if args.output is not None:
        output_path = args.output
    elif levels == [1]:
        output_path = Path("data/profile_dataset.json")
    else:
        tag = "L" + "L".join(str(l) for l in levels)
        output_path = Path(f"data/profile_dataset_{tag}.json")

    by_subj_level = load_and_filter(levels)
    problems      = balanced_sample(by_subj_level, levels, n_per_cell)
    print_summary(problems, levels, n_per_cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"  Saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
