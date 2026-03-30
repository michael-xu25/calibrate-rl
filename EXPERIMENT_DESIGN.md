# CalibrateRL — Experiment Design

## What We're Trying to Prove

The central claim: **dynamically updating the training curriculum during RL is more efficient than training on a fixed set of problems.**

"More efficient" means: equal or better improvement on held-out math problems, using fewer gradient steps or less wasted compute.

This matters because the motivating startup thesis is that static curricula waste compute grinding on problems the model has already learned or can never learn, while a dynamic curriculum continuously keeps the model in its productive learning zone.

---

## The Core Insight: The Goldilocks Zone

In RL with a binary reward (correct = 1, wrong = 0), the GRPO loss is:

```
advantage_i = r_i - mean(r)
```

This means:
- A problem the model **always gets right** (pass_rate = 1.0): all rewards = 1, advantage = 0 for every rollout → **zero gradient**.
- A problem the model **never gets right** (pass_rate = 0.0): all rewards = 0, advantage = 0 → **zero gradient**.
- Only problems where the model **sometimes gets it right** produce nonzero advantage and actually update the weights.

We call this the **Goldilocks zone**: 0 < pass_rate < 1, AND at least one rollout must be correct (max_score = 1). These are problems the model can solve but doesn't always — the exact zone where learning happens.

### Why pass@8 instead of pass@4

With 8 rollouts, pass_rate has 9 possible values: {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}. The Goldilocks zone covers 6 of those. With only 4 rollouts you get values {0, 0.25, 0.5, 0.75, 1.0} — only 3 are Goldilocks, and they're too coarse to distinguish a problem at 0.3 pass rate from one at 0.5. More resolution makes the curriculum filter more accurate and the saturation threshold (0.875 = 7/8) more meaningful.

### Why SATURATE_THRESHOLD = 0.875 (not 1.0)

A problem at 7/8 pass rate has mean advantage ≈ 0.125 — almost nothing. Waiting for 8/8 wastes steps. Evicting at ≥7/8 reclaims those compute slots for harder problems where the gradient signal is larger.

---

## The Experiment: Dynamic vs. Static RL

### Why not compare to the base model?

Obvious: any RL training improves on base. The base model comparison doesn't tell you anything about curriculum strategy.

### The baseline: naive RL on all 630 problems

The static baseline trains on the full 630-problem pool with no filtering, never updated. This is the realistic practitioner baseline — what you'd do if you just handed RL your entire dataset. The key fact, invisible to the practitioner:

- **Qwen**: only 206/630 = 32.7% of problems are Goldilocks. Every batch wastes ~67% of compute on zero-gradient problems.
- **Llama**: only 372/630 = 59% are Goldilocks. Still 41% waste.

Dynamic curriculum automatically focuses on the productive subset and adapts as it changes. The comparison shows the **total value of curriculum management**: both the initial filtering and the ongoing adaptation.

| | Dynamic | Static |
|---|---|---|
| Starting active set | 150 highest-signal Goldilocks (sorted by \|pr−0.5\|) | All 630 problems |
| Reserve | Goldilocks overflow + unreachable problems | None |
| During training | Evicts saturated/unreachable, promotes reserve | Fixed forever |
| Useful gradient fraction | ~100% of each batch | 33–59% of each batch |
| Can discover harder problems | Yes — reserve problems enter as model improves | No |

The hypothesis is that dynamic wins because:
1. Every training step is concentrated on problems that actually produce gradient
2. As Goldilocks problems saturate, static grinds near-zero gradient on them; dynamic evicts and replaces with harder ones
3. The active Goldilocks fraction stays near 100% in dynamic mode and collapses in static

### Evaluation: held-out set

210 problems (10 per subject per level, L1-L3, seed=99) that are **never in the training pool**. After each 40-step phase, we run pass@8 on these 210 problems for both the dynamic and static models. This gives a clean learning curve that neither model has seen.

---

## Data Pipeline

### Step 1: Build the training pool
`data/profile_dataset_L1L2L3.json` — 630 problems
- 30 problems × 7 subjects × 3 levels (L1, L2, L3)
- Sampled from `EleutherAI/hendrycks_math` with seed=42
- Answers validated: no `\begin{...}`, no pure `\text{...}`, max 40 chars
- The 210 held-out problems are drawn from the same distribution but with a **different seed (99) and no overlap** with this pool

### Step 2: Pre-evaluation (currently running)
`data/evaluation_results_full.json`
Pass@8 on all 630 problems for both Qwen-2.5-7B and Llama-3-8B via OpenRouter.
- 630 × 2 models × 8 rollouts = **10,080 API calls**
- Purpose: accurately identify each problem's difficulty relative to each model before training starts
- This means no cold-start waste: we go into training already knowing which problems are Goldilocks, which are saturated, which are unreachable

### Step 3: Export Goldilocks inventories
```bash
python3 src/export_goldilocks.py --results data/evaluation_results_full.json
```
Produces `data/goldilocks_qwen-2.5-7b.json` and `data/goldilocks_llama-3-8b.json`.
These seed both the dynamic and static curricula.

### Step 4: Training (Lightning AI H100)
Four runs total (2 models × 2 conditions):
```bash
python src/train_grpo.py --model qwen-2.5-7b              # dynamic
python src/train_grpo.py --model qwen-2.5-7b --static     # static baseline
python src/train_grpo.py --model llama-3-8b               # dynamic
python src/train_grpo.py --model llama-3-8b --static      # static baseline
```

---

## Training Architecture

### GRPO with phase-based curriculum refresh

Every 40 gradient steps = one phase. End of each phase:
1. Score all active problems (pass@8 using the current checkpoint)
2. Evict saturated (≥7/8) and confirmed unreachable (0/8 three times consecutively)
3. Probe up to 40 reserve candidates; add those that score Goldilocks
4. Log Goldilocks fraction of the active set
5. Save curriculum state atomically to `curriculum_state.json`

```
Phase length:             80 steps
Training rollouts:        8 per problem (GRPOTrainer loss)
Scoring rollouts:         8 per problem (pass@8) — 6 Goldilocks-zone values
Held-out rollouts:        4 per problem (pass@4) — sufficient for trend signal
Saturation threshold:     7/8 (0.875)
Unreachable patience:     3 consecutive 0/8 evals
Target active pool:       150 problems (sorted by |pass_rate − 0.5|, highest signal first)
Min pool to continue:     16 problems
Max reserve probed:       40 per refresh
Reserve patience:         4 consecutive 0/8 probes → hard-remove
Probe decay:              0.5× per consecutive failure
Reserve tier 1:           Goldilocks overflow (known gradient, strikes=0)
Reserve tier 2:           Unreachable at baseline (strikes=1, half probe priority)
Total training steps:     480 (6 phases)
```

### Seed selection: sort by proximity to pass_rate=0.5

The per-rollout advantage magnitude for a problem with pass_rate p is `2p(1-p)`, maximized at p=0.5:

| pass_rate | advantage magnitude |
|---|---|
| 0.125 | 0.22 — near-unreachable, low signal |
| 0.25 | 0.375 |
| **0.5** | **0.50 — maximum gradient signal** |
| 0.75 | 0.375 |
| 0.875 | 0.22 — near-saturated, will be evicted soon anyway |

Seeding with the 150 problems closest to p=0.5 maximizes gradient signal from phase 1. The remaining Goldilocks (further from 0.5, lower immediate signal) go to the reserve as **tier-1** fill — they're known to produce gradient and are preferred over unreachable problems when replenishing. Unreachable problems are **tier-2** reserve, given one initial "strike" to halve their probe probability relative to tier-1.

### Why TARGET_CURRICULUM = 150

At 80 steps per phase with batch_size=8: each active problem gets ~5 training presentations per phase (80×8/150 ≈ 4.3). This is the key ratio:
- Too few active (< 128): consecutive batches repeat the same problems → high gradient correlation, less stable updates, poor subject diversity
- Too many active (> 200): presentations per problem drop below 3× → eviction decisions are based on underpowered pass_rate estimates, curriculum is noise-driven
- 150 sits in the sweet spot, with gradient diversity across 7 subjects × 3 levels

### Why UNREACHABLE_PATIENCE = 3 (not 2)

At G=8, the variance on a problem with true pass_rate=0.1 is non-trivial — there's a real chance of scoring 0/8 twice by chance alone. Evicting after just 2 zeroes risks discarding problems the model is just barely learning. Three consecutive zeroes is a much stronger signal of genuine unreachability, at the cost of ~1 extra phase of wasted rollouts per false positive.

### Weighted reserve sampling

Problems that repeatedly fail probing aren't instantly removed — their selection probability decays by 0.5× per consecutive 0/8 result (`PROBE_DECAY=0.5`). After 4 consecutive failures (`RESERVE_PATIENCE=4`) they're hard-removed. This prevents a handful of impossible problems from monopolizing probe slots every phase.

### Compute overhead analysis (per phase, 80 steps)

| Component | Forward passes | % of training |
|---|---|---|
| Training (80 steps × batch 8 × 8 rollouts) | 5,120 | — |
| Active scoring (150 problems × pass@8) | 1,200 | 23% |
| Reserve probing (40 candidates × pass@8) | 320 | 6% |
| Held-out eval (210 problems × **pass@4**) | 840 | 16% |
| **Total overhead** | **2,360** | **46%** |

Held-out eval uses pass@4 (not pass@8) because it only needs trend signal across phases, not per-problem precision. The standard error of the mean pass rate over 210 problems at pass@4 is ~1.7% — more than sufficient to track a learning curve. This halves held-out cost vs pass@8 (which would add 32% overhead, making total overhead ~62%).

---

## Model Configuration

### LoRA
- Rank r=32, alpha=64 (alpha/r=2, standard scaling)
- Applied to all 7 modules: q/k/v/o projections + MLP gate/up/down
- r=32 is larger than the instruction-tuning default (r=16) — justified because RL needs to develop new reasoning patterns, not just style-transfer
- Including MLP layers matters for math reasoning; attention-only LoRA is more common but less expressive

### Training setup
- **bf16**: H100 supports it natively; no quantization needed
- **Flash Attention 2**: auto-detected (requires Ampere+ GPU and `flash-attn` package); falls back to eager
- **Gradient checkpointing**: enabled; requires `model.enable_input_require_grads()` with PEFT (frozen base layers break checkpointing without it)
- **KL penalty**: `beta=0.04` (GRPO paper default) — prevents reward hacking / mode collapse by penalizing the policy for drifting far from the reference model
- **LR**: 2e-5, warmup only on phase 1. Higher than the usual conservative 1e-5 because at 480 total steps, we need enough model movement per phase to trigger saturation events — the curriculum dynamics only become visible if problems actually saturate. Subsequent phases start at full LR to avoid 12.5% wasted warmup steps per phase.

### Why LoRA not full fine-tune

For a POC on an H100 with 8B parameter models, LoRA keeps memory usage manageable, speeds up each forward/backward pass, and reduces the risk of catastrophic forgetting on general capabilities. Full fine-tune would also require much more careful LR tuning. The tradeoff: LoRA may limit how much the model can shift its reasoning distribution, which could understate the benefit of dynamic curriculum for a more capable training regime.

### Why no partial-credit reward

We considered adding 0.1 reward for `\boxed{}` formatting. Rejected: it trains format, not math ability. Our three-tier answer extraction (`<answer>` tag → `\boxed{}` → last number) already handles format variation on the scoring side. Adding a formatting reward would pollute the gradient signal and potentially reward models that learn to write `\boxed{wrong_answer}` rather than solve the problem.

---

## Key Diagnostics

The most important thing to watch during training — more important than loss curves — is **Goldilocks fraction per phase** for both dynamic and static runs:

- **Dynamic**: should stay relatively high (>50%) throughout training as saturated problems are evicted and fresh ones promoted
- **Static**: will start high but collapse as the fixed problem set saturates; by late training most problems may score ≥7/8 and contribute near-zero gradient

If the Goldilocks fraction curves diverge, that's the mechanism working as designed — even if held-out accuracy is similar, you've demonstrated that static training degrades into wasted compute.

This is logged explicitly in `curriculum_state.json` as `goldilocks_frac` after every phase.

---

## What Success Looks Like

On the held-out 210-problem eval, after the same number of total gradient steps:

- **Strong result**: Dynamic reaches the same held-out accuracy as static in fewer steps (compute efficiency win)
- **Acceptable result**: Dynamic reaches higher held-out accuracy at the same step count (quality win)
- **Weak but still meaningful**: Goldilocks fraction stays high in dynamic and collapses in static, demonstrating the mechanism — even if held-out accuracy gap is small at 400 steps

The clearest proof would be a learning curve where static plateaus (because most of its Goldilocks problems saturate and the gradient vanishes) while dynamic keeps improving by discovering harder problems from the reserve. 400 total training steps may not be enough to see a strong divergence — 800-1200 steps would give a cleaner result if budget allows.

---

## Known Limitations of This POC

1. **No online difficulty estimation**: Pass rates are only updated at phase boundaries. Within a phase, the curriculum is stale by up to 40 steps. A Bayesian difficulty model would be more accurate but adds complexity.

2. **Binary reward only**: No partial credit. Problems the model nearly solves contribute the same gradient as problems it's totally wrong about. Richer reward signals (e.g., step-level process reward) could improve learning efficiency for both conditions.

3. **No multi-GPU**: Single H100. GRPOTrainer with `device_map="auto"` handles this, but multi-GPU would allow larger effective batch sizes and more rollout diversity.

4. **LoRA ceiling**: LoRA limits how much the model's reasoning distribution can shift. This may understate the dynamic curriculum benefit — with full fine-tune, saturation events might be more dramatic and the adaptive mechanism more necessary.

---

## Current Status (as of 2026-03-29)

| Step | Status |
|---|---|
| Build 630-problem L1L2L3 training pool | Done |
| Build 210-problem held-out set | Done |
| Pass@8 pre-eval (Qwen-2.5-7B) | Done — 206 Goldilocks, 161 unreachable |
| Pass@8 pre-eval (Llama-3-8B) | Done — 372 Goldilocks, 154 unreachable |
| Export Goldilocks inventories | Done |
| Dynamic training run × 2 models | Pending |
| Static training run × 2 models | Pending |
