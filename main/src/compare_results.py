"""
Compare baseline vs checkpoint results, broken down by entity count.

Cross-references per-problem pass rates from the baseline pass@16 eval
and the checkpoint eval, then breaks down by number of entities in each
question to see if entity-tracking training helped where it should.

Usage:
    python src/compare_results.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from build_entity_dataset import extract_entity_names


def load_results(path: str) -> dict[int, dict]:
    """Load JSONL results, keyed by problem index."""
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["idx"]] = r
    return results


def main():
    baseline_path = "logs/pass_at_k.jsonl"
    checkpoint_path = "logs/eval_checkpoint_no_system_prompt.jsonl"

    if not os.path.exists(baseline_path):
        print(f"Missing {baseline_path} — run eval_pass_at_k.py first")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Missing {checkpoint_path} — run eval_checkpoint.py --no-system-prompt first")
        return

    baseline = load_results(baseline_path)
    checkpoint = load_results(checkpoint_path)

    common_idxs = sorted(set(baseline) & set(checkpoint))
    print(f"Comparing {len(common_idxs)} problems present in both evals\n")

    # Analyze each problem
    rows = []
    for idx in common_idxs:
        b = baseline[idx]
        c = checkpoint[idx]
        question = b.get("question", "")
        entities = extract_entity_names(question)
        n_entities = len(entities)

        b_rate = b["n_correct"] / b["n_total"]
        c_rate = c["n_correct"] / c["n_total"]
        delta = c_rate - b_rate

        rows.append({
            "idx": idx,
            "question": question[:80],
            "entities": entities,
            "n_entities": n_entities,
            "baseline": b["n_correct"],
            "checkpoint": c["n_correct"],
            "b_rate": b_rate,
            "c_rate": c_rate,
            "delta": delta,
        })

    # ── Overall comparison ───────────────────────────────────────────────
    avg_b = sum(r["b_rate"] for r in rows) / len(rows)
    avg_c = sum(r["c_rate"] for r in rows) / len(rows)

    print(f"{'='*70}")
    print(f"  OVERALL (all {len(rows)} problems)")
    print(f"{'='*70}")
    print(f"  Baseline pass@1:    {avg_b*100:.1f}%")
    print(f"  Checkpoint pass@1:  {avg_c*100:.1f}%")
    print(f"  Delta:              {(avg_c-avg_b)*100:+.1f}%")

    # ── Breakdown by entity count ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BREAKDOWN BY ENTITY COUNT")
    print(f"{'='*70}")
    print(f"  {'Entities':<10s} {'Count':>6s} {'Baseline':>10s} {'Checkpoint':>12s} {'Delta':>8s}")
    print(f"  {'-'*46}")

    entity_groups = {}
    for r in rows:
        key = min(r["n_entities"], 4)  # group 4+ together
        entity_groups.setdefault(key, []).append(r)

    for n in sorted(entity_groups):
        group = entity_groups[n]
        b_avg = sum(r["b_rate"] for r in group) / len(group)
        c_avg = sum(r["c_rate"] for r in group) / len(group)
        label = f"{n}" if n < 4 else "4+"
        print(f"  {label:<10s} {len(group):>6d} {b_avg*100:>9.1f}% {c_avg*100:>11.1f}% {(c_avg-b_avg)*100:>+7.1f}%")

    # ── Biggest improvements ─────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: r["delta"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  TOP 10 IMPROVEMENTS")
    print(f"{'='*70}")
    for r in rows_sorted[:10]:
        ents = ", ".join(r["entities"][:4])
        if len(r["entities"]) > 4:
            ents += "..."
        print(f"  {r['baseline']:>2d}/16 → {r['checkpoint']:>2d}/16  "
              f"({r['delta']*100:>+5.1f}%)  "
              f"[{r['n_entities']} ent: {ents}]")
        print(f"    {r['question']}...")

    # ── Biggest regressions ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TOP 10 REGRESSIONS")
    print(f"{'='*70}")
    for r in rows_sorted[-10:]:
        ents = ", ".join(r["entities"][:4])
        if len(r["entities"]) > 4:
            ents += "..."
        print(f"  {r['baseline']:>2d}/16 → {r['checkpoint']:>2d}/16  "
              f"({r['delta']*100:>+5.1f}%)  "
              f"[{r['n_entities']} ent: {ents}]")
        print(f"    {r['question']}...")

    # ── Statistical significance ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  NOISE CHECK")
    print(f"{'='*70}")
    improved = sum(1 for r in rows if r["delta"] > 0)
    regressed = sum(1 for r in rows if r["delta"] < 0)
    unchanged = sum(1 for r in rows if r["delta"] == 0)
    print(f"  Improved:   {improved} problems")
    print(f"  Regressed:  {regressed} problems")
    print(f"  Unchanged:  {unchanged} problems")

    # Paired sign test
    if improved + regressed > 0:
        ratio = improved / (improved + regressed)
        print(f"  Win ratio:  {ratio:.1%} ({improved}/{improved+regressed})")
        if ratio > 0.6:
            print(f"  → Likely real improvement (wins > 60% of changed problems)")
        elif ratio < 0.4:
            print(f"  → Likely real regression")
        else:
            print(f"  → Inconclusive (too close to 50/50)")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
