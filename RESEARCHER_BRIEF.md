# CalibrateRL — Researcher Review Brief

Full implementation details in `EXPERIMENT_DESIGN.md` and `src/train_grpo.py`.
This doc summarizes the planned runs and the open questions I'd like your input on before launch.

---

## Planned Experiments

Four runs total: dynamic curriculum vs. static baseline on two models.
GCP H100 on Lightning AI at **$5.37/hour**.

| # | Condition | Model | Command | Est. Duration | Est. Cost |
|---|---|---|---|---|---|
| 1 | Dynamic curriculum | Qwen 2.5-7B | `python3 src/train_grpo.py --model qwen-2.5-7b` | ~14 hrs | ~$75 |
| 2 | Static baseline | Qwen 2.5-7B | `python3 src/train_grpo.py --model qwen-2.5-7b --static` | ~11 hrs | ~$59 |
| 3 | Dynamic curriculum | Llama 3-8B | `python3 src/train_grpo.py --model llama-3-8b` | ~14 hrs | ~$75 |
| 4 | Static baseline | Llama 3-8B | `python3 src/train_grpo.py --model llama-3-8b --static` | ~11 hrs | ~$59 |

**Total: ~50 hours, ~$268**

Time estimates assume ~60 GRPO steps/hour on H100 without vLLM (training dominates; scoring
adds ~3.5 hrs/run for dynamic, ~1.2 hrs/run for static). These are rough — the first run will
calibrate actual throughput.

Runs 1 and 2 can be launched in parallel on separate H100 instances. Same for 3 and 4.

---

## Design Summary (TL;DR)

- **Task**: MATH benchmark, Levels 1–3. Binary exact-match reward.
- **Algorithm**: GRPO, 640 total gradient steps (8 phases × 80 steps), LoRA r=32, LR=2e-5, KL β=0.02.
- **Dynamic**: 150 active problems, refreshed every 80 steps. Saturated (≥7/8 correct) and
  unreachable (0/8 for 3 consecutive phases) problems are frozen into cooldown, then re-evaluated
  and either returned to reserve or permanently retired after 2 cycles.
- **Static**: All 630 training problems, never updated. Realistic practitioner baseline.
- **Metric**: Greedy pass@1 on 210 held-out problems, evaluated after each phase.

---

## Things I'd Like Your Input On

### 1. TRL advantage normalization scope — does this match what we want?

TRL's GRPOTrainer normalizes advantages. I need to confirm whether it normalizes
within each prompt's group (per-prompt mean/std) or across the full batch.

- **Within-group** (per-prompt): each problem's advantages are centered to zero with unit variance
  independently. This is standard GRPO and is what we want.
- **Cross-batch**: a problem at pass_rate=0.1 and one at pass_rate=0.9 would have their
  advantages mixed together, changing the effective update magnitude per problem.

We're using `trl>=0.17.0,<0.18.0`. Worth checking `GRPOTrainer._compute_advantages`
or the loss implementation to confirm.

### 2. Optimizer state reset between phases — is this a problem?

Each phase creates a new `GRPOTrainer` instance (`src/train_grpo.py` line ~932).
This resets Adam's momentum/second-moment to zero at the start of every phase.
With only 80 steps per phase and 5 warmup steps in phase 0, the optimizer may spend
a non-trivial fraction of each phase warming up.

Options if this is a concern:
- Extract and re-inject optimizer state between phases
- Or accept it — with LoRA the parameter space is small and Adam warms up fast

### 3. Is 640 steps enough to see meaningful curriculum dynamics?

The hypothesis requires at least one or two saturation cycles to manifest:
- Phase 0→1: some Goldilocks problems saturate → evicted, reserve probed
- Phase 2→3: second-generation active set starts saturating, cooldown thaws

With 8 phases and 80 steps each, we probably get 2–3 eviction rounds before the
run ends. Is that enough signal to distinguish dynamic vs. static on the
held-out curve (210 problems, SE ≈ 3.4% at p=0.5)?

If this is a proof-of-concept demo rather than a publishable result, one run per
condition is probably fine. If you think the effect size needs to be statistically
robust, we'd need either more steps or multiple seeds.

### 4. MIN_PROMOTE_PASS_RATE = 0.25 — the promotion floor

When probing reserve problems, we only promote a candidate into the active set if it
scores ≥0.25 (≥2/8) on the probe. The intent is to filter noise: a true-5%-pass-rate
problem has a ~28% chance of scoring 1/8 by luck on a single 8-rollout probe.

The tradeoff: a legitimate 20%-pass-rate problem has a `0.8^8 ≈ 17%` chance of scoring
0/8, incrementing its strike count. With `UNREACHABLE_PATIENCE=3` it has `0.17^3 ≈ 0.5%`
chance of being falsely retired. Acceptable?

A related question: should the promotion floor scale with training progress? Early on,
a 20% problem is genuinely Goldilocks. Later in training, after the model has improved,
the same problem might be at 70% — even better. The floor doesn't need to be adaptive,
but wanted your take.

### 5. Stale pass_rate on thawed cooldown problems

When a problem is evicted from active and frozen into cooldown, it carries its most
recent pass_rate (from the eviction scoring round). When it thaws back into reserve,
`probe_weight()` uses this stored pass_rate to assign probe priority.

Problem: the stored pass_rate reflects the model state *at eviction time*, not now.
A near-saturated problem evicted at phase 2 returns to reserve at phase 4 with its
old pass_rate still labeled near 0.875 — getting low probe priority — even though the
model may have drifted and the problem could now be genuinely Goldilocks again.

Simple fix: clear the pass_rate field when a problem enters cooldown so it re-enters
reserve as a tier-2 unknown. Does this seem worth doing, or is the noise negligible
over a short 640-step run?

### 6. Static baseline framing — is this the right comparison?

The static baseline trains on all 630 problems with no filtering. At initialization,
only ~33% (Qwen) to ~59% (Llama) of those 630 problems are in the Goldilocks zone.
The rest produce zero gradient (fully saturated or unreachable), so ~40–67% of every
static training batch is dead compute.

The comparison is: **curriculum-managed 150 active problems (≈90% Goldilocks at start)
vs. unmanaged 630 problems (33–59% Goldilocks, declining over time)**.

This is intentionally the practitioner baseline — the point is to show that the
overhead of profiling + dynamic management pays off vs. naively running RL on your
full dataset. If you think there's a fairer or more interesting control condition,
I'm open to adding a third condition (e.g., static-150 with goldilocks pre-filter
but no dynamic refresh).

### 7. Single seed — is effect size large enough?

One run per condition. RL training variance comes from: problem ordering, probe
sampling randomness, generation temperature, LoRA initialization. The held-out
test set is 210 problems (SE ≈ ±3.4%), evaluated at 8 points on the learning curve.

For this to be a convincing POC without multiple seeds, the dynamic advantage needs
to be visually obvious on the learning curve — probably ≥5–7pp gap sustained across
multiple phases. Does that seem realistic given the design, or would you recommend
changing scope (fewer total steps, more seeds) to get a cleaner result?

---

## Known Limitations (already aware)

- No vLLM — standard HF generation. Chosen for Lightning AI compatibility; about 3–5×
  slower than vLLM for the scoring/probing passes.
- Qwen reserve is thin without `evaluation_results_full.json` (56 overflow problems only).
  The file is tracked in git so this shouldn't be an issue on Lightning AI.
- 640 steps is a short run by RL standards — designed to keep GPU cost under $300 total
  for all 4 conditions while still demonstrating the core curriculum mechanic.
