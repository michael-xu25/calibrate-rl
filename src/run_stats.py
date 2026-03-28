"""
Rigorous statistical analysis of model capability divergence.

The Z-test in generate_heatmap.py treats all rollouts as independent,
inflating effective N by 8x. The correct unit of analysis is the problem
(n=20 per subject), using paired tests since both models see the same problems.

Tests applied:
  1. Wilcoxon signed-rank  — paired, non-parametric, appropriate for n=20
                             with discrete pass-rate data. Primary test.
  2. Paired t-test         — parametric reference; assumes normality (suspect at n=20)
  3. Bonferroni correction — 7 simultaneous subject comparisons, α=0.05/7=0.00714
  4. Cohen's d             — effect size on problem-level pass rates
  5. Bootstrap 95% CI      — on the mean paired difference, 10k resamples
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_PATH = Path("data/evaluation_results.json")
N_BOOTSTRAP  = 10_000
ALPHA        = 0.05
N_SUBJECTS   = 7
ALPHA_BONF   = ALPHA / N_SUBJECTS   # 0.00714

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

LABELS = {
    "algebra":                  "Algebra",
    "counting_and_probability": "Counting & Prob.",
    "geometry":                 "Geometry",
    "intermediate_algebra":     "Intermediate Alg.",
    "number_theory":            "Number Theory",
    "prealgebra":               "Pre-Algebra",
    "precalculus":              "Pre-Calculus",
}


def bootstrap_mean_diff_ci(diffs: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 42) -> tuple:
    """95% bootstrap CI on mean paired difference (Qwen - Llama)."""
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    diff = a - b
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def stars(p: float, bonf: bool = False) -> str:
    threshold = ALPHA_BONF if bonf else ALPHA
    if p < threshold / 5:   return "**"
    if p < threshold:        return "*"
    return "—"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    models = ["qwen-2.5-7b", "llama-3-8b"]
    m0, m1 = models

    print("=" * 78)
    print("  RIGOROUS STATISTICAL ANALYSIS — CalibrateRL Capability Divergence")
    print(f"  Unit of analysis: problem (n=20/subject)  |  Paired tests (same problems)")
    print(f"  Bonferroni-corrected α = {ALPHA}/{N_SUBJECTS} = {ALPHA_BONF:.5f}")
    print("=" * 78)

    results = {}

    for subj in SUBJECTS:
        probs = [r for r in data.values() if r["subject"] == subj]
        probs.sort(key=lambda x: x["id"])   # consistent ordering for pairing

        q_rates = np.array([p["models"][m0]["pass_rate"] for p in probs])
        l_rates = np.array([p["models"][m1]["pass_rate"] for p in probs])
        diffs   = q_rates - l_rates   # positive = Qwen better

        n = len(probs)
        mean_q  = q_rates.mean()
        mean_l  = l_rates.mean()
        mean_d  = diffs.mean()

        # 1. Wilcoxon signed-rank (primary)
        if np.all(diffs == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")

        # 2. Paired t-test (reference)
        t_stat, t_p = stats.ttest_rel(q_rates, l_rates)

        # 3. Effect size
        d = cohens_d_paired(q_rates, l_rates)

        # 4. Bootstrap CI on mean diff
        ci_lo, ci_hi = bootstrap_mean_diff_ci(diffs)

        # 5. Bonferroni significance
        sig_raw  = stars(w_p, bonf=False)
        sig_bonf = stars(w_p, bonf=True)
        survives = w_p < ALPHA_BONF

        results[subj] = {
            "n": n,
            "mean_qwen": mean_q, "mean_llama": mean_l,
            "mean_diff": mean_d,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "ttest_stat": t_stat, "ttest_p": t_p,
            "cohens_d": d,
            "bootstrap_ci": (ci_lo, ci_hi),
            "sig_uncorrected": sig_raw,
            "sig_bonferroni": sig_bonf,
            "survives_bonferroni": survives,
        }

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n  {'Subject':<20} {'Qwen':>6} {'Llama':>6} {'Δ':>7}  "
          f"{'W-p':>9}  {'t-p':>9}  {'d':>6}  {'95% CI (diff)':>18}  {'raw':>4}  {'Bonf':>4}")
    print("  " + "-" * 96)

    for subj in SUBJECTS:
        r = results[subj]
        ci = r["bootstrap_ci"]
        print(
            f"  {LABELS[subj]:<20}"
            f"  {r['mean_qwen']:>5.1%}"
            f"  {r['mean_llama']:>5.1%}"
            f"  {r['mean_diff']:>+6.1%}"
            f"  {r['wilcoxon_p']:>9.4f}"
            f"  {r['ttest_p']:>9.4f}"
            f"  {r['cohens_d']:>+5.2f}"
            f"  [{ci[0]:>+.3f}, {ci[1]:>+.3f}]"
            f"  {r['sig_uncorrected']:>4}"
            f"  {r['sig_bonferroni']:>4}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    surviving = [s for s in SUBJECTS if results[s]["survives_bonferroni"]]
    uncorrected_sig = [s for s in SUBJECTS if results[s]["wilcoxon_p"] < ALPHA]

    print("\n" + "=" * 78)
    print(f"  Uncorrected  (p < 0.05):           {len(uncorrected_sig)}/7 subjects significant")
    print(f"  Bonferroni   (p < {ALPHA_BONF:.5f}):  {len(surviving)}/7 subjects significant")
    print(f"\n  Subjects surviving Bonferroni correction:")
    for s in surviving:
        r = results[s]
        print(f"    {LABELS[s]:<22}  Δ={r['mean_diff']:>+.1%}  d={r['cohens_d']:>+.2f}  "
              f"W-p={r['wilcoxon_p']:.5f}  CI=[{r['bootstrap_ci'][0]:+.3f}, {r['bootstrap_ci'][1]:+.3f}]")

    print(f"\n  Effect size guide: |d| > 0.2 small, > 0.5 medium, > 0.8 large")
    print(f"\n  Key caveat: n=20 problems/subject limits power.")
    print(f"  Wilcoxon is conservative with discrete data (pass rates in steps of 0.125).")
    print(f"  Bootstrap CIs that exclude 0 are the most reliable divergence evidence.")
    print("=" * 78)

    # Save
    import json as _json
    def to_py(v):
        if isinstance(v, tuple): return list(v)
        if isinstance(v, (np.bool_, np.integer)): return int(v)
        if isinstance(v, np.floating): return float(v)
        return v
    report = {s: {k: to_py(v) for k, v in r.items()} for s, r in results.items()}
    Path("data/stats_rigorous.json").write_text(_json.dumps(report, indent=2))
    print(f"\n  Full report saved → data/stats_rigorous.json")


if __name__ == "__main__":
    main()
