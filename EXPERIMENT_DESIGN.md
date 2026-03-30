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

### Why not compare dynamic to a random-subset static baseline?

This is the key design decision we had to think through carefully.

Suppose static uses 63 problems and dynamic can access 630. If dynamic wins, you've just proven "more training data = better results" — not "adaptive curriculum selection = better results." That's a trivially true and uninteresting claim.

In RL with binary rewards, the comparison is even more subtle: **problems outside the Goldilocks zone contribute exactly zero gradient regardless of how many you include.** Including 500 saturated or unreachable problems in a static training set doesn't help at all — they just don't fire. So "more problems" isn't trivially better even in principle.

### The fair comparison

Both conditions have access to **exactly the same set of pre-identified Goldilocks problems** (from the full pass@8 pre-eval). The only difference is:

| | Dynamic | Static |
|---|---|---|
| Starting active set | **All pre-identified Goldilocks** (same as static) | All pre-identified Goldilocks |
| Reserve | Unreachable problems (pass=0 at baseline) | None |
| During training | Evicts saturated/unreachable, promotes reserve problems | Fixed — never changes |
| Can discover harder problems | Yes — previously unreachable problems enter as model improves | No |
| Wastes compute on near-saturated | No — evicts at ≥7/8 | Yes — keeps grinding them |

The hypothesis is that dynamic wins because:
1. As problems saturate, static keeps computing near-zero gradient on them; dynamic evicts and replaces
2. Dynamic can discover L3 problems that become Goldilocks as the model improves; static is fundamentally capped at the initial inventory
3. The effective **Goldilocks fraction** of the active set stays high in dynamic mode and collapses in static — this is the mechanism we measure directly

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
Phase length:             40 steps
Scoring rollouts:         8 per problem (pass@8)
Training rollouts:        8 per problem (GRPOTrainer loss)
Saturation threshold:     7/8 (0.875)
Unreachable patience:     3 consecutive 0/8 evals
Target active pool:       64 problems
Min pool to continue:     16 problems
Max reserve probed:       40 per refresh
Reserve patience:         4 consecutive 0/8 probes → hard-remove
Probe decay:              0.5× per consecutive failure
```

### Why dynamic starts with all Goldilocks (same as static)

The starting active set is set to the full pre-identified Goldilocks inventory — the same set static uses. This removes the most important confound: if dynamic started with a subset (e.g., 64 problems) while static started with all 206, static would have 3× more gradient signal per phase early in training. Static could win simply because of that head start, not because fixed curricula are better.

By starting both conditions identically, the only variable is whether the curriculum updates. The reserve for dynamic contains the unreachable problems (pass=0 at baseline) — the harder problems that may unlock as the model improves through training.

`target_size` (used for replenishment in `refresh()`) is set dynamically to `len(goldilocks)` at build time, so it scales correctly per model rather than being a hardcoded constant.

### Why UNREACHABLE_PATIENCE = 3 (not 2)

At G=8, the variance on a problem with true pass_rate=0.1 is non-trivial — there's a real chance of scoring 0/8 twice by chance alone. Evicting after just 2 zeroes risks discarding problems the model is just barely learning. Three consecutive zeroes is a much stronger signal of genuine unreachability, at the cost of ~1 extra phase of wasted rollouts per false positive.

### Weighted reserve sampling

Problems that repeatedly fail probing aren't instantly removed — their selection probability decays by 0.5× per consecutive 0/8 result (`PROBE_DECAY=0.5`). After 4 consecutive failures (`RESERVE_PATIENCE=4`) they're hard-removed. This prevents a handful of impossible problems from monopolizing probe slots every phase.

### Compute overhead analysis

For Qwen (206 Goldilocks, 161 unreachable reserve): scoring 206 active problems at pass@8 = 1,648 forward passes per phase. Probing up to 40 reserve candidates = 320 more. Training 40 steps at batch size 8 × 8 rollouts = 2,560 forward passes.

- Active scoring overhead: 1648 / (1648 + 2560) ≈ **39%**
- Total scoring (active + probe): 1968 / (1968 + 2560) ≈ **44%**

This is the tax for the adaptive mechanism with a fair starting set. It's higher than a truncated-curriculum design, but it's the honest cost — and static pays the same active-scoring cost without the replenishment benefit. The experiment tests whether that cost is worth it.

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
- **LR**: 1e-5, warmup only on phase 1; subsequent phases start at full LR to avoid 12.5% wasted steps per phase from re-warming

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
| Pass@8 pre-eval (Qwen-2.5-7B) | ~370/630 in progress |
| Pass@8 pre-eval (Llama-3-8B) | 0/630, starts after Qwen |
| Export Goldilocks inventories | Pending pre-eval |
| Dynamic training run × 2 models | Pending |
| Static training run × 2 models | Pending |
