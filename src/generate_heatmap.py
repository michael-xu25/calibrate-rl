"""
Step 3: Statistical Testing & Divergent Capability Heatmap

Loads data/evaluation_results.json (produced by run_evaluation.py),
aggregates pass rates by subject for each model, runs 2-proportion
Z-tests to identify statistically significant divergences, and renders
a two-panel figure:

  Panel A — Pass-rate heatmap (models × subjects)
             Asterisks mark cells where the two models differ significantly.
  Panel B — Δ pass-rate bar chart (Llama minus Qwen per subject)
             Shows direction and magnitude of divergence.

Output:
  data/heatmap.png          — the figure
  data/stats_report.json    — full per-subject stats table
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

# ── Paths & constants ──────────────────────────────────────────────────────────

RESULTS_PATH = Path("data/evaluation_results.json")
HEATMAP_PATH = Path("data/heatmap.png")
REPORT_PATH  = Path("data/stats_report.json")

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SUBJECT_LABELS = {
    "algebra":                  "Algebra",
    "counting_and_probability": "Counting &\nProbability",
    "geometry":                 "Geometry",
    "intermediate_algebra":     "Intermediate\nAlgebra",
    "number_theory":            "Number\nTheory",
    "prealgebra":               "Pre-Algebra",
    "precalculus":              "Pre-Calculus",
}

# Significance thresholds (two-tailed Z-test)
# *  p < 0.05  →  |z| > 1.96
# ** p < 0.01  →  |z| > 2.576
ALPHA_1 = 0.05
ALPHA_2 = 0.01


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results() -> dict:
    if not RESULTS_PATH.exists():
        sys.exit(f"Results file not found: {RESULTS_PATH}\nRun src/run_evaluation.py first.")
    with open(RESULTS_PATH) as f:
        return json.load(f)


def discover_models(results: dict) -> list[str]:
    """Infer model keys from the first fully-evaluated problem."""
    for rec in results.values():
        keys = list(rec.get("models", {}).keys())
        if len(keys) >= 2:
            return sorted(keys)
    sys.exit("No problems with both models evaluated yet.")


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_by_subject(results: dict, models: list[str]) -> dict:
    """
    For each (model, subject) pair, collect every individual rollout reward
    (not just the per-problem pass_rate) so the Z-test uses the full n.

    Returns:
        {
          subject: {
            model_key: {
              "successes": int,   # total correct rollouts
              "trials":    int,   # total rollouts (n_problems × 16)
              "pass_rate": float, # successes / trials
              "n_problems": int,
            }
          }
        }
    """
    agg: dict = {s: {m: {"successes": 0, "trials": 0, "n_problems": 0}
                     for m in models}
                 for s in SUBJECTS}

    skipped = 0
    for rec in results.values():
        subj = rec.get("subject")
        if subj not in SUBJECTS:
            continue
        for model_key in models:
            m_data = rec.get("models", {}).get(model_key)
            if m_data is None:
                skipped += 1
                continue
            rewards = m_data.get("rollout_rewards", [])
            agg[subj][model_key]["successes"]  += sum(rewards)
            agg[subj][model_key]["trials"]     += len(rewards)
            agg[subj][model_key]["n_problems"] += 1

    # Compute pass_rate
    for subj in SUBJECTS:
        for model_key in models:
            d = agg[subj][model_key]
            d["pass_rate"] = d["successes"] / d["trials"] if d["trials"] > 0 else 0.0

    if skipped:
        print(f"  Note: {skipped} (problem, model) pairs missing — run may still be in progress.")

    return agg


# ── Statistical testing ────────────────────────────────────────────────────────

def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """
    Two-proportion Z-test (two-tailed).
    H0: p1 == p2.

    Uses the pooled proportion under H0.
    Returns (z_statistic, p_value).
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    p1 = s1 / n1
    p2 = s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)

    denom = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 0.0, 1.0

    z = (p1 - p2) / denom
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))   # two-tailed
    return float(z), float(p_value)


def significance_stars(p_value: float) -> str:
    if p_value < ALPHA_2:
        return "**"
    if p_value < ALPHA_1:
        return "*"
    return ""


def compute_stats(agg: dict, models: list[str]) -> dict:
    """
    For each subject, run the Z-test comparing model[0] vs model[1].
    Returns a dict keyed by subject with full stats.
    """
    m0, m1 = models[0], models[1]
    report = {}

    for subj in SUBJECTS:
        d0 = agg[subj][m0]
        d1 = agg[subj][m1]
        z, p = two_proportion_ztest(d0["successes"], d0["trials"],
                                    d1["successes"], d1["trials"])
        stars = significance_stars(p)
        delta = d1["pass_rate"] - d0["pass_rate"]   # positive → m1 better

        report[subj] = {
            m0: {k: d0[k] for k in ("pass_rate", "successes", "trials", "n_problems")},
            m1: {k: d1[k] for k in ("pass_rate", "successes", "trials", "n_problems")},
            "z_statistic": round(z, 4),
            "p_value":     round(p, 6),
            "significant": stars != "",
            "stars":       stars,
            "delta":       round(delta, 4),  # m1 - m0
            "stronger_model": m1 if delta > 0 else (m0 if delta < 0 else "tie"),
        }

    return report


# ── Heatmap rendering ──────────────────────────────────────────────────────────

def render(agg: dict, stats_report: dict, models: list[str]) -> None:
    n_subj  = len(SUBJECTS)
    m0, m1  = models[0], models[1]

    # ── Build arrays ──────────────────────────────────────────────────────
    # rows = models, cols = subjects
    pass_matrix = np.array([
        [agg[subj][m]["pass_rate"] for subj in SUBJECTS]
        for m in models
    ])  # shape (2, 7)

    delta_arr  = np.array([stats_report[s]["delta"]  for s in SUBJECTS])
    pval_arr   = np.array([stats_report[s]["p_value"] for s in SUBJECTS])
    stars_arr  = [stats_report[s]["stars"]            for s in SUBJECTS]

    # Annotation: pass_rate % + stars (stars go on the row that's BETTER)
    annot = []
    for row_idx, m in enumerate(models):
        row = []
        for col_idx, subj in enumerate(SUBJECTS):
            pct   = f"{pass_matrix[row_idx, col_idx]:.0%}"
            stars = stats_report[subj]["stars"]
            stronger = stats_report[subj]["stronger_model"]
            # Place stars only on the stronger model's cell
            label = f"{pct}{stars}" if (stars and stronger == m) else pct
            row.append(label)
        annot.append(row)
    annot = np.array(annot)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0f1117")

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 1.6],
        hspace=0.55,
        left=0.13, right=0.97,
        top=0.88,  bottom=0.10,
    )
    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    label_color   = "#e0e0e0"
    grid_color    = "#0f1117"

    # ── Panel A: heatmap ──────────────────────────────────────────────────
    cmap = sns.color_palette("RdYlGn", as_cmap=True)   # red=low, green=high

    sns.heatmap(
        pass_matrix,
        ax=ax_heat,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        linewidths=2,
        linecolor=grid_color,
        cbar_kws={"shrink": 0.7, "label": "Pass Rate"},
        annot_kws={"size": 13, "weight": "bold", "color": "#1a1a1a"},
    )

    model_labels = [m.upper() for m in models]
    subj_labels  = [SUBJECT_LABELS[s] for s in SUBJECTS]

    ax_heat.set_xticklabels(subj_labels, color=label_color, fontsize=10.5, rotation=0)
    ax_heat.set_yticklabels(model_labels, color=label_color, fontsize=11,
                             fontweight="bold", rotation=0, va="center")
    ax_heat.tick_params(left=False, bottom=False)

    cbar = ax_heat.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(label_color)
    cbar.ax.tick_params(colors=label_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=label_color)

    ax_heat.set_title(
        "Divergent Capability Heatmap  —  Pass Rate by Subject",
        color=label_color, fontsize=14, fontweight="bold", pad=14,
    )
    ax_heat.set_facecolor("#1a1d27")

    # ── Panel B: delta bar chart ──────────────────────────────────────────
    bar_colors = ["#ef5350" if d < 0 else "#66bb6a" for d in delta_arr]
    bars = ax_bar.bar(range(n_subj), delta_arr, color=bar_colors,
                      width=0.55, zorder=3)

    # Significance markers above/below bars
    for i, (d, stars) in enumerate(zip(delta_arr, stars_arr)):
        if stars:
            offset = 0.025 if d >= 0 else -0.025
            va     = "bottom" if d >= 0 else "top"
            ax_bar.text(i, d + offset, stars, ha="center", va=va,
                        color="white", fontsize=13, fontweight="bold")

    ax_bar.axhline(0, color="#888", linewidth=1, zorder=2)
    ax_bar.set_xticks(range(n_subj))
    ax_bar.set_xticklabels(subj_labels, color=label_color, fontsize=10, rotation=0)
    ax_bar.set_ylabel(f"Δ Pass Rate\n({m1} − {m0})", color=label_color, fontsize=9.5)
    ax_bar.tick_params(colors=label_color, left=True)
    ax_bar.yaxis.label.set_color(label_color)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444")
    ax_bar.set_facecolor("#1a1d27")
    ax_bar.tick_params(axis="y", colors=label_color)
    ax_bar.set_xlim(-0.5, n_subj - 0.5)

    # Y-axis: symmetric around 0 with a little headroom
    max_delta = max(abs(delta_arr).max(), 0.05)
    ax_bar.set_ylim(-max_delta * 1.4, max_delta * 1.4)

    # Format y-axis as percentage
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:+.0%}")
    )

    ax_bar.set_title(
        f"Δ Pass Rate per Subject  ({m1.upper()} − {m0.upper()})"
        "    green = Llama stronger  |  red = Qwen stronger",
        color=label_color, fontsize=10.5, pad=8,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="none", label="* p < 0.05"),
        mpatches.Patch(color="none", label="** p < 0.01"),
    ]
    fig.legend(
        handles=legend_items, loc="lower right",
        framealpha=0.15, labelcolor=label_color,
        fontsize=9, ncol=2,
        bbox_to_anchor=(0.97, 0.01),
    )

    # ── Suptitle ──────────────────────────────────────────────────────────
    n_evaluated = sum(
        1 for rec in {}  # placeholder — patched below
        if all(m in rec.get("models", {}) for m in models)
    )
    fig.suptitle(
        "CalibrateRL  —  Model Capability Profiling  (MATH Level 1-2)",
        color=label_color, fontsize=15, fontweight="bold", y=0.97,
    )

    HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved heatmap → {HEATMAP_PATH}")


# ── Console summary ────────────────────────────────────────────────────────────

def print_stats_table(stats_report: dict, models: list[str]) -> None:
    m0, m1 = models[0], models[1]
    w = 24

    header = (f"  {'Subject':<{w}}  {m0.upper():>8}  {m1.upper():>8}"
              f"  {'Δ':>7}  {'z':>7}  {'p':>8}  sig")
    print("\n" + "="*75)
    print("  Per-Subject Statistical Summary")
    print("="*75)
    print(header)
    print("  " + "-"*71)

    for subj in SUBJECTS:
        r  = stats_report[subj]
        p0 = r[m0]["pass_rate"]
        p1 = r[m1]["pass_rate"]
        print(
            f"  {SUBJECT_LABELS[subj].replace(chr(10),' '):<{w}}"
            f"  {p0:>7.1%}"
            f"  {p1:>7.1%}"
            f"  {r['delta']:>+7.1%}"
            f"  {r['z_statistic']:>7.3f}"
            f"  {r['p_value']:>8.4f}"
            f"  {r['stars'] or '—'}"
        )

    sig = [s for s in SUBJECTS if stats_report[s]["significant"]]
    print("="*75)
    print(f"  Significant divergences ({len(sig)}/{len(SUBJECTS)} subjects): "
          + (", ".join(SUBJECT_LABELS[s].replace("\n"," ") for s in sig) or "none"))

    # Which model is stronger overall
    avg0 = np.mean([stats_report[s][m0]["pass_rate"] for s in SUBJECTS])
    avg1 = np.mean([stats_report[s][m1]["pass_rate"] for s in SUBJECTS])
    print(f"\n  Overall avg pass rate —  {m0}: {avg0:.1%}   {m1}: {avg1:.1%}")
    print("="*75 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading results from {RESULTS_PATH}…")
    results = load_results()
    print(f"  {len(results)} problems loaded")

    models = discover_models(results)
    print(f"  Models detected: {models}")

    # Warn if either model is incomplete
    for m in models:
        evaluated = sum(1 for r in results.values() if m in r.get("models", {}))
        print(f"  {m}: {evaluated}/{len(results)} problems evaluated")

    agg          = aggregate_by_subject(results, models)
    stats_report = compute_stats(agg, models)

    print_stats_table(stats_report, models)

    print("Rendering heatmap…")
    render(agg, stats_report, models)

    # Save JSON report
    with open(REPORT_PATH, "w") as f:
        json.dump(stats_report, f, indent=2)
    print(f"  Saved stats report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
