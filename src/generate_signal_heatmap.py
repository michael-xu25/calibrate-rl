"""
Training Signal Heatmap — CalibrateRL

Replaces the pass-rate comparison heatmap with a per-model view of
where gradient signal actually lives.

Panel A — Goldilocks Yield: % of problems per subject where the model
           is in the trainable zone (0 < pass_rate < 1, max_score = 1).
Panel B — Mean |Advantage|: average absolute advantage estimate per
           subject. Higher = richer gradient signal per training step.

Usage:
    python src/generate_signal_heatmap.py
    python src/generate_signal_heatmap.py --results data/evaluation_results_L1L2.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

SUBJECT_LABELS = {
    "algebra":                  "Algebra",
    "counting_and_probability": "Counting &\nProbability",
    "geometry":                 "Geometry",
    "intermediate_algebra":     "Intermediate\nAlgebra",
    "number_theory":            "Number\nTheory",
    "prealgebra":               "Pre-Algebra",
    "precalculus":              "Pre-Calculus",
}

BG_DARK     = "#0f1117"
AX_BG       = "#1a1d27"
LABEL_COLOR = "#e0e0e0"
SPINE_COLOR = "#444"


# ── Data ───────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Results not found: {path}\nRun src/run_evaluation.py first.")
    with open(path) as f:
        return json.load(f)


def discover_models(results: dict) -> List[str]:
    for rec in results.values():
        keys = list(rec.get("models", {}).keys())
        if len(keys) >= 2:
            return sorted(keys)
    sys.exit("Need at least 2 models fully evaluated.")


def compute_signal_metrics(
    results: dict, models: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        yield_matrix  — shape (n_models, n_subjects), fraction of Goldilocks problems
        signal_matrix — shape (n_models, n_subjects), mean |advantage estimate|
    """
    n_m, n_s = len(models), len(SUBJECTS)
    yield_matrix  = np.zeros((n_m, n_s))
    signal_matrix = np.zeros((n_m, n_s))

    for mi, model in enumerate(models):
        for si, subj in enumerate(SUBJECTS):
            probs = [
                r for r in results.values()
                if r["subject"] == subj and model in r.get("models", {})
            ]
            if not probs:
                continue

            goldilocks = [
                p for p in probs
                if 0 < p["models"][model]["pass_rate"] < 1.0
                and p["models"][model]["max_score"] == 1
            ]
            yield_matrix[mi, si] = len(goldilocks) / len(probs)

            all_ae = [
                abs(ae)
                for p in probs
                for ae in p["models"][model]["advantage_estimates"]
            ]
            signal_matrix[mi, si] = float(np.mean(all_ae)) if all_ae else 0.0

    return yield_matrix, signal_matrix


# ── Rendering ─────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)


def _style_cbar(cbar):
    cbar.ax.yaxis.label.set_color(LABEL_COLOR)
    cbar.ax.tick_params(colors=LABEL_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)


def render(
    yield_matrix: np.ndarray,
    signal_matrix: np.ndarray,
    models: List[str],
    output_path: Path,
    subtitle: str = "",
) -> None:
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(BG_DARK)

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=0.65,
        left=0.13, right=0.97,
        top=0.88,  bottom=0.09,
    )
    ax_yield  = fig.add_subplot(gs[0])
    ax_signal = fig.add_subplot(gs[1])

    model_labels = [m.upper() for m in models]
    subj_labels  = [SUBJECT_LABELS[s] for s in SUBJECTS]

    # ── Panel A: Goldilocks yield ─────────────────────────────────────────
    annot_yield = np.array([
        [f"{v*100:.0f}%" for v in row] for row in yield_matrix
    ])

    sns.heatmap(
        yield_matrix,
        ax=ax_yield,
        annot=annot_yield,
        fmt="",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        vmin=0.0, vmax=1.0,
        linewidths=2, linecolor=BG_DARK,
        cbar_kws={"shrink": 0.7, "label": "Goldilocks Yield"},
        annot_kws={"size": 14, "weight": "bold", "color": "#1a1a1a"},
    )
    ax_yield.set_xticklabels(subj_labels, color=LABEL_COLOR, fontsize=10.5, rotation=0)
    ax_yield.set_yticklabels(model_labels, color=LABEL_COLOR, fontsize=11,
                              fontweight="bold", rotation=0, va="center")
    ax_yield.tick_params(left=False, bottom=False)
    ax_yield.set_title(
        "Panel A — Goldilocks Yield  (% of problems in trainable zone: 0 < pass_rate < 1)",
        color=LABEL_COLOR, fontsize=11, fontweight="bold", pad=10,
    )
    _style_ax(ax_yield)
    _style_cbar(ax_yield.collections[0].colorbar)

    # ── Panel B: Mean |advantage| ─────────────────────────────────────────
    sig_max = max(float(signal_matrix.max()), 0.35)
    annot_signal = np.array([
        [f"{v:.3f}" for v in row] for row in signal_matrix
    ])

    sns.heatmap(
        signal_matrix,
        ax=ax_signal,
        annot=annot_signal,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        vmin=0.0, vmax=sig_max,
        linewidths=2, linecolor=BG_DARK,
        cbar_kws={"shrink": 0.7, "label": "Mean |Advantage|"},
        annot_kws={"size": 14, "weight": "bold", "color": "#1a1a1a"},
    )
    ax_signal.set_xticklabels(subj_labels, color=LABEL_COLOR, fontsize=10.5, rotation=0)
    ax_signal.set_yticklabels(model_labels, color=LABEL_COLOR, fontsize=11,
                               fontweight="bold", rotation=0, va="center")
    ax_signal.tick_params(left=False, bottom=False)
    ax_signal.set_title(
        "Panel B — Mean |Advantage Estimate|  (gradient signal strength per rollout)",
        color=LABEL_COLOR, fontsize=11, fontweight="bold", pad=10,
    )
    _style_ax(ax_signal)
    _style_cbar(ax_signal.collections[0].colorbar)

    fig.suptitle(
        f"CalibrateRL — Per-Model Training Signal Profiling{subtitle}",
        color=LABEL_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(
    yield_matrix: np.ndarray,
    signal_matrix: np.ndarray,
    models: List[str],
    results: dict,
) -> None:
    print(f"\n{'='*70}")
    print("  Training Signal Summary")
    print(f"{'='*70}")
    print(f"  {'Subject':<28} " +
          "  ".join(f"{m.upper():<22}" for m in models))
    print(f"  {'':28} " +
          "  ".join(f"{'Yield':>8} {'|Adv|':>8}      " for _ in models))
    print("  " + "-"*66)

    for si, subj in enumerate(SUBJECTS):
        label = SUBJECT_LABELS[subj].replace("\n", " ")
        row = ""
        for mi in range(len(models)):
            row += f"  {yield_matrix[mi,si]*100:>7.0f}%  {signal_matrix[mi,si]:>7.3f}      "
        print(f"  {label:<28}{row}")

    print(f"\n  Goldilocks totals (trainable problems per model):")
    for mi, model in enumerate(models):
        total = sum(
            1 for r in results.values()
            if model in r.get("models", {})
            and 0 < r["models"][model]["pass_rate"] < 1.0
            and r["models"][model]["max_score"] == 1
        )
        n_total = sum(1 for r in results.values() if model in r.get("models", {}))
        print(f"    {model}: {total}/{n_total} ({100*total/n_total:.0f}%)")
    print(f"{'='*70}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path("data/evaluation_results.json"),
        help="Path to evaluation_results.json (default: Level 1 results)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path (default: data/signal_heatmap[_suffix].png)",
    )
    args = parser.parse_args()

    # Auto-name output based on input
    if args.output is None:
        stem = args.results.stem.replace("evaluation_results", "signal_heatmap")
        args.output = Path("data") / f"{stem}.png"

    # Subtitle reflects which dataset
    subtitle = ""
    if "L1L2" in args.results.stem:
        subtitle = "  (MATH Level 1+2, 210 problems × 8 rollouts)"
    else:
        subtitle = "  (MATH Level 1, 140 problems × 8 rollouts)"

    print(f"Loading {args.results}…")
    results = load_results(args.results)
    print(f"  {len(results)} problems loaded")

    models = discover_models(results)
    print(f"  Models: {models}")

    yield_matrix, signal_matrix = compute_signal_metrics(results, models)
    print_summary(yield_matrix, signal_matrix, models, results)

    print("Rendering…")
    render(yield_matrix, signal_matrix, models, args.output, subtitle)


if __name__ == "__main__":
    main()
