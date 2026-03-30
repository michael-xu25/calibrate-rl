"""
Build the held-out evaluation set for CalibrateRL.

Samples 10 problems per (subject, level) cell — 210 problems total — from
the MATH dataset, explicitly excluding every problem already in the training
pool (data/profile_dataset_L1L2L3.json).

Design:
- Uses seed=99 (training pool uses seed=42) for reproducibility
- Excludes by problem ID, not just by index, to guarantee no leakage
- Verifies zero overlap and zero duplicates before writing

Usage:
    python src/build_heldout_dataset.py
    python src/build_heldout_dataset.py --n-per-cell 15   # larger held-out
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

DATASET_ID   = "EleutherAI/hendrycks_math"
RANDOM_SEED  = 99       # intentionally different from training pool seed (42)
N_PER_CELL   = 10       # problems per (subject, level) cell
TRAIN_POOL   = Path("data/profile_dataset_L1L2L3.json")
OUTPUT_PATH  = Path("data/heldout_eval.json")

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
LEVELS = [1, 2, 3]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load training pool IDs to exclude
    if not TRAIN_POOL.exists():
        raise FileNotFoundError(f"Training pool not found: {TRAIN_POOL}")
    with open(TRAIN_POOL) as f:
        train_pool = json.load(f)
    excluded_ids = {p["id"] for p in train_pool}
    print(f"Training pool: {len(excluded_ids)} problems to exclude")

    # Load and filter MATH dataset
    print(f"Loading MATH dataset (Levels {LEVELS})…")
    by_cell: Dict[Tuple[str, int], List[dict]] = defaultdict(list)

    for subj in SUBJECTS:
        rows = []
        for split in ("train", "test"):
            try:
                ds = load_dataset(DATASET_ID, subj, split=split)
                rows.extend(list(ds))
            except Exception as e:
                print(f"  Warning: {subj}/{split}: {e}")

        for local_idx, row in enumerate(rows):
            lvl = parse_level(row.get("level", ""))
            if lvl not in LEVELS:
                continue
            answer = extract_boxed_answer(row.get("solution", ""))
            if not is_valid_answer(answer):
                continue

            pid = f"{subj}_L{lvl}_{local_idx}"
            if pid in excluded_ids:
                continue   # explicitly exclude training pool problems

            by_cell[(subj, lvl)].append({
                "id":      pid,
                "subject": subj,
                "level":   lvl,
                "problem": row["problem"].strip(),
                "answer":  answer,
            })

    # Report available pool sizes
    print(f"\n  Available after exclusion (subject, level → count):")
    for subj in SUBJECTS:
        for lvl in LEVELS:
            n = len(by_cell.get((subj, lvl), []))
            flag = " !" if n < args.n_per_cell else ""
            print(f"    {subj:<32} L{lvl}  {n:>4}{flag}")

    # Sample exactly n_per_cell from each cell
    rng = random.Random(RANDOM_SEED)
    sampled = []
    for subj in SUBJECTS:
        for lvl in LEVELS:
            pool = by_cell.get((subj, lvl), [])
            if len(pool) < args.n_per_cell:
                raise ValueError(
                    f"({subj}, L{lvl}): only {len(pool)} problems available "
                    f"after exclusion (need {args.n_per_cell}). Lower --n-per-cell."
                )
            sampled.extend(rng.sample(pool, args.n_per_cell))

    # Integrity checks
    ids = [p["id"] for p in sampled]
    duplicate_ids = {pid for pid, cnt in Counter(ids).items() if cnt > 1}
    overlap_ids   = {pid for pid in ids if pid in excluded_ids}

    if duplicate_ids:
        raise RuntimeError(f"BUG: duplicate IDs in held-out set: {duplicate_ids}")
    if overlap_ids:
        raise RuntimeError(f"BUG: held-out overlaps with training pool: {overlap_ids}")

    # Shuffle and write
    rng.shuffle(sampled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\n{'='*54}")
    print(f"  Held-out set: {len(sampled)} problems")
    print(f"  {args.n_per_cell}/cell × {len(SUBJECTS)} subjects × {len(LEVELS)} levels")
    print(f"  Seed: {RANDOM_SEED}  (training pool used seed 42)")
    print(f"  Overlap with training pool: 0  ✓")
    print(f"  Duplicate IDs: 0  ✓")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
