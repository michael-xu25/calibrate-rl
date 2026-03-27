"""
Extract SFT Training Data from Pass@K Results.

Reads logs/pass_at_k.jsonl (with full samples from the updated eval_pass_at_k.py)
and extracts the **shortest** correct reasoning chain for each problem that has
at least one correct solution. This "teaches the model to be its best self"
by fine-tuning on its own successful, concise reasoning paths before RL.

Selection strategy:
  - For each problem with >= 1 correct solution, pick the SHORTEST correct one
    (prefer concise reasoning — avoids teaching the model to be verbose)
  - Skip problems that are always correct (16/16) — the model already handles these
  - Skip problems that are always wrong (0/16) — no correct path to learn from
  - Focus on the "gap" problems (1-15/16) where RL will benefit most

Usage:
    python src/extract_sft_data.py [--input logs/pass_at_k.jsonl] [--output data/sft_paths.jsonl]

Outputs:
    - data/sft_paths.jsonl: chat-format SFT training data
    - Terminal: summary statistics
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Extract SFT data from Pass@K results")
    parser.add_argument(
        "--input", default="logs/pass_at_k.jsonl",
        help="Path to pass_at_k.jsonl (must have full correct_samples)",
    )
    parser.add_argument(
        "--output", default="data/sft_paths.jsonl",
        help="Path to write SFT training data",
    )
    parser.add_argument(
        "--include-easy", action="store_true",
        help="Also include problems with 100%% pass rate (default: skip them)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=900,
        help="Skip correct responses longer than this many words (avoid verbose paths)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        print("Run eval_pass_at_k.py first (updated version that saves full samples).")
        sys.exit(1)

    # Load pass@K results
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} problems from {args.input}")

    # Check if the data has full samples (new format)
    sample_record = records[0] if records else {}
    if "correct_samples" not in sample_record:
        # Try old format with single truncated sample
        if "correct_sample" in sample_record:
            print("\nWarning: pass_at_k.jsonl uses OLD format (truncated samples).")
            print("Re-run eval_pass_at_k.py with the updated version to get full samples.")
            print("Falling back to truncated samples (may produce lower-quality SFT data).\n")
            # Convert old format to new format for compatibility
            for r in records:
                cs = r.pop("correct_sample", None)
                ws = r.pop("wrong_sample", None)
                r["correct_samples"] = [cs] if cs else []
                r["wrong_samples"] = [ws] if ws else []
        else:
            print("Error: Unrecognized pass_at_k.jsonl format.")
            sys.exit(1)

    # Filter and extract
    sft_data = []
    stats = {
        "total": len(records),
        "always_correct": 0,
        "always_wrong": 0,
        "gap_problems": 0,
        "extracted": 0,
        "skipped_too_long": 0,
        "skipped_no_samples": 0,
    }

    for r in records:
        n_correct = r["n_correct"]
        n_total = r["n_total"]

        if n_correct == 0:
            stats["always_wrong"] += 1
            continue

        if n_correct == n_total and not args.include_easy:
            stats["always_correct"] += 1
            continue

        if n_correct == n_total:
            stats["always_correct"] += 1

        stats["gap_problems"] += 1

        correct_samples = r.get("correct_samples", [])
        if not correct_samples:
            stats["skipped_no_samples"] += 1
            continue

        # Pick the SHORTEST correct response (prefer concise reasoning)
        best = min(correct_samples, key=lambda s: len(s.split()))
        word_count = len(best.split())

        if word_count > args.max_tokens:
            stats["skipped_too_long"] += 1
            continue

        # Format as chat-style SFT data
        sft_entry = {
            "messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": best},
            ],
            # Metadata for debugging (not used in training)
            "_meta": {
                "gold_answer": r.get("gold_answer"),
                "pass_rate": r.get("pass_rate"),
                "n_correct": n_correct,
                "n_total": n_total,
                "selected_word_count": word_count,
                "n_correct_candidates": len(correct_samples),
            },
        }
        sft_data.append(sft_entry)
        stats["extracted"] += 1

    # Save SFT data
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + "\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SFT DATA EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total problems in input:    {stats['total']}")
    print(f"  Always correct (skipped):   {stats['always_correct']}")
    print(f"  Always wrong (skipped):     {stats['always_wrong']}")
    print(f"  Gap problems considered:    {stats['gap_problems']}")
    print(f"  Extracted for SFT:          {stats['extracted']}")
    print(f"  Skipped (too long):         {stats['skipped_too_long']}")
    print(f"  Skipped (no full samples):  {stats['skipped_no_samples']}")
    print()

    if sft_data:
        word_counts = [e["_meta"]["selected_word_count"] for e in sft_data]
        avg_words = sum(word_counts) / len(word_counts)
        print(f"  --- Selected Response Stats ---")
        print(f"  Mean words:    {avg_words:.0f}")
        print(f"  Min words:     {min(word_counts)}")
        print(f"  Max words:     {max(word_counts)}")
        print()

    print(f"  SFT data saved to: {args.output}")
    print(f"  Use with: python src/train_sft.py --data {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
