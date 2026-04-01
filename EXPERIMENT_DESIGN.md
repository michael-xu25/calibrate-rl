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
- A problem the model **always gets right** (pass_rate = 1.0): all rewards = 1, advantage = 0 → **zero reward gradient**.
- A problem the model **never gets right** (pass_rate = 0.0): all rewards = 0, advantage = 0 → **zero reward gradient**.
- Only problems where the model **sometimes gets it right** produce nonzero advantage and actually update the weights.

We call this the **Goldilocks zone**: 0 < pass_rate < 1. These are problems the model can solve but doesn't always — the exact zone where learning happens.

The gradient signal magnitude for a problem at pass_rate p is `2p(1−p)`, maximized at p=0.5:

| pass_rate | gradient signal `2p(1−p)` |
|---|---|
| 0.125 | 0.22 — near-unreachable, low signal |
| 0.25 | 0.375 |
| **0.5** | **0.50 — maximum** |
| 0.75 | 0.375 |
| 0.875 | 0.22 — near-saturated, will be evicted |

We log `mean_grad_signal = mean(2p(1−p))` across all active problems each phase. For dynamic this stays high as low-signal problems are cycled out; for static it declines as saturated and unreachable problems accumulate.

### A subtlety: saturated problems still contribute KL regularization

Even at 8/8 correct (zero reward gradient), a problem in the training batch still contributes to the KL penalty term that penalizes the policy for drifting from the reference model. This means evicting saturated problems removes a small "don't forget easy problems" anchor. In practice this is mitigated by: (1) LoRA freezes base model weights so catastrophic forgetting is limited, (2) the KL penalty on harder problems still overlaps with easier reasoning patterns, (3) 640 steps is short. The held-out per-level breakdown will catch any forgetting empirically.

### Why pass@8 for scoring

With 8 rollouts, pass_rate has 9 possible values: {0, 0.125, 0.25, …, 1.0}. The Goldilocks zone covers 6 of those. With 4 rollouts you get only 3 Goldilocks-zone values — too coarse to distinguish a 0.3 pass_rate problem from a 0.5 one. More resolution makes eviction thresholds more accurate and the saturation cutoff (7/8 = 0.875) more meaningful.

### Why SATURATE_THRESHOLD = 0.875 (not 1.0)

A problem at 7/8 has `mean_grad_signal = 0.22` — almost nothing. Waiting for 8/8 wastes steps. Evicting at ≥7/8 reclaims those compute slots for harder problems.

---

## The Experiment: Dynamic vs. Static RL

### The baseline: naive RL on all 630 problems

The static baseline trains on the full 630-problem pool with no filtering, never updated. This is the realistic practitioner baseline — what you'd do if you just handed RL your entire dataset.

- **Qwen**: only 206/630 = 32.7% of problems are Goldilocks. Every batch wastes ~67% of compute.
- **Llama**: only 372/630 = 59% are Goldilocks. Still 41% waste.

| | Dynamic | Static |
|---|---|---|
| Starting active set | 150 highest-signal Goldilocks (sorted by \|pr−0.5\|) | All 630 problems |
| Reserve | Goldilocks overflow + unreachable problems | None |
| During training | Cycles saturated/unreachable out, promotes reserve | Fixed forever |
| Useful gradient fraction | ~100% of each batch initially | 33–59% of each batch, declining |
| Can discover harder problems | Yes — reserve problems enter as model improves | No |

### Evaluation: held-out set

210 problems (10 per subject per level, L1–L3, seed=99) never in the training pool. After each 80-step phase, greedy decoding on all 210 problems for both conditions. Provides a clean learning curve with 9 data points (phase 0 = base model, phases 1–8 = after each training phase).

---

## Data Pipeline

### Step 1: Build the training pool
`data/profile_dataset_L1L2L3.json` — 630 problems
- 30 problems × 7 subjects × 3 levels (L1, L2, L3), seed=42
- Answers validated: no `\begin{...}`, no pure `\text{...}`, max 40 chars

### Step 2: Pre-evaluation
`data/evaluation_results_full.json` — pass@8 on all 630 problems for both models via OpenRouter (10,080 API calls). Identifies each problem's difficulty before training starts — no cold-start waste.

### Step 3: Export Goldilocks inventories
```bash
python3 src/export_goldilocks.py --results data/evaluation_results_full.json
```
Produces `data/goldilocks_qwen-2.5-7b.json` (206 problems) and `data/goldilocks_llama-3-8b.json` (372 problems).

### Step 4: Training (Lightning AI H100)
```bash
# Run order: Qwen dynamic first (canary), then static, then Llama pair
nohup python3 src/train_grpo.py --model qwen-2.5-7b         > checkpoints/qwen_dynamic.log 2>&1 &
nohup python3 src/train_grpo.py --model qwen-2.5-7b --static > checkpoints/qwen_static.log 2>&1 &
nohup python3 src/train_grpo.py --model llama-3-8b           > checkpoints/llama_dynamic.log 2>&1 &
nohup python3 src/train_grpo.py --model llama-3-8b --static  > checkpoints/llama_static.log 2>&1 &
```

Qwen is the right first model: it has only 32.7% Goldilocks at baseline (vs 59% for Llama), so the dynamic curriculum's advantage should be most visible there. If Qwen dynamic looks healthy after phase 1–2, launch the rest.

---

## Training Architecture

### Parameters

```
Phase length:              80 steps
Training rollouts:         8 per problem (GRPOTrainer loss)
Scoring rollouts:          8 per problem (pass@8)
Held-out eval:             greedy (1 deterministic pass per problem), batch=8
Saturation threshold:      7/8 (0.875)
Unreachable patience:      3 consecutive 0/8 evals before cooldown
Target active pool:        150 problems
Min pool (warning):        8 problems — warns but keeps training; stops only at 0
Max probe (round 1):       max(40, needed×2) candidates
Reserve patience:          4 consecutive 0/8 probes → cooldown
Probe decay:               0.5× per consecutive probe failure
Min promotion pass_rate:   2/8 = 0.25 (filters noise from single-probe luck)
Saturate cooldown:         2 phases frozen before re-entering reserve
Unreachable cooldown:      1 phase frozen before re-entering reserve
Max cooldown cycles:       2 (permanently retired after 2nd cycle)
Total training steps:      640 (8 phases)
Static probe sample:       64 problems/phase for Goldilocks fraction logging
```

### Seed selection: sort by proximity to pass_rate=0.5

Goldilocks sorted by `|pass_rate − 0.5|` ascending. Top 150 (closest to 0.5, highest signal) → active set. The rest become tier-1 reserve. Unreachable problems become tier-2 reserve. Seeding with the highest-signal problems maximizes useful gradient from phase 1.

### Why TARGET_CURRICULUM = 150

At 80 steps per phase with batch_size=8: each problem gets ~5 training presentations per phase (80×8/150 ≈ 4.3).
- Too few active (< 128): batches repeat the same problems → high gradient correlation, poor diversity
- Too many active (> 200): presentations per problem drop below 3× → eviction decisions are noise-driven
- 150 sits in the sweet spot across 7 subjects × 3 levels

### Why UNREACHABLE_PATIENCE = 3

A problem with true pass_rate=0.1 has P(score 0/8) = 0.9^8 ≈ 0.43. Three consecutive 0/8 events: P ≈ 0.08. Patience=2 would have 18% false-eviction rate for a barely-learnable problem. Three is the right balance between responsiveness and noise.

---

## Reserve Management

This is the core of what makes dynamic curriculum non-trivial. The reserve is divided into two tiers based on what we know about each problem at baseline:

**Tier 1 — Goldilocks overflow** (has `pass_rate` field in JSON, 0 < pr < 0.875 at baseline): Known to produce gradient. These are the 56 Qwen / 222 Llama problems that were Goldilocks at baseline but didn't make the initial top-150 active cut due to lower signal.

**Tier 2 — unreachable at baseline** (no `pass_rate` field): Problems the model scored 0/8 on before training. May unlock as the model improves.

### Probe weighting: base_signal × PROBE_DECAY^strikes

Rather than treating all reserve problems equally, probe sampling weights each problem by:

```
weight = base_signal × (0.5 ^ strikes)
```

Where:
- **Tier-1**: `base_signal = max(1 − |baseline_pr − 0.5| / 0.5, 0.1)` — problems near pr=0.5 at baseline get highest probe priority (signal=1.0); near-saturated overflow (pr=0.875) get lowest (signal=0.25). This avoids wasting probes on tier-1 problems that have likely already fully saturated after a few training phases.
- **Tier-2**: `base_signal = 0.4` fixed — conservative but nonzero.

The `strikes` counter tracks consecutive 0/8 probe failures. Each failure halves the problem's probe probability. After 4 consecutive failures it enters cooldown.

### Scaled probe budget

Probe budget scales with the actual deficit: `n_probe = min(max(40, needed × 2), len(reserve))`. If 30 problems saturate in one phase, we probe 60 candidates, not a fixed 40. A second probe round runs automatically if round 1 fills < 50% of the needed slots.

### Promotion: quality-filtered, signal-sorted

A reserve problem is only promoted to active if:
1. Its probe pass_rate ≥ **MIN_PROMOTE_PASS_RATE = 0.25** (2/8). This filters noise — a true-5% problem has ~28% chance of scoring 1/8 by luck on a single probe, but only ~6% chance of scoring ≥2/8.
2. Its probe pass_rate < SATURATE_THRESHOLD (0.875). Already-saturated probes are re-frozen immediately.

When multiple candidates qualify, they're sorted by `|pr − 0.5|` ascending — highest-signal problems promoted first.

### Cooldown instead of permanent eviction

Problems are never permanently removed on first failure. Instead they enter a time-limited cooldown queue and re-enter the reserve when the cooldown expires:

| Path | Cooldown | Re-entry strikes | Why |
|---|---|---|---|
| Active → saturated (≥7/8) | 2 phases | 2 (low probe priority) | Model needs 160 steps to meaningfully shift before a re-check is useful |
| Active → unreachable (0/8 × 3) | 1 phase | 1 (moderate priority) | Model may improve rapidly in early phases; check soon |
| Reserve → probe-exhausted (4 failures) | 2 phases | 2 (low priority) | Multiple chances failed; longer break before retry |
| Probe → still saturated (≥7/8) | 2 phases | 2 | Re-freeze immediately rather than clogging reserve |

After **2 total cooldown cycles** (`MAX_COOLDOWN_CYCLES = 2`), a problem is permanently retired. This gives each problem: original active/reserve period + 2 additional chances through the reserve before the door closes.

### Pool floor: warn, don't stop

If the active pool drops below MIN_CURRICULUM=8, training logs a warning but continues. A pool of 8 still fills one full batch and provides useful gradient. Only stops if the pool reaches 0. Premature halting due to temporary reserve exhaustion is worse than training on a small pool — cooldown thaws may recover it within 1–2 phases.

---

## Compute Overhead (per phase, 80 steps)

| Component | Forward passes | % of training | Notes |
|---|---|---|---|
| Training (80 steps × batch 8 × 8 rollouts) | 5,120 | — | |
| Active scoring (150 × pass@8) | 1,200 | 23% | dynamic only |
| Reserve probing (≤80 × pass@8) | ≤640 | ≤13% | dynamic only; scales with deficit |
| Held-out eval (210 × greedy, batch=8) | 210 | 4% | both conditions |
| Static Goldilocks probe (64 × pass@8) | 512 | 10% | static only |
| **Dynamic total overhead** | **~2,050** | **~40%** | |
| **Static total overhead** | **722** | **14%** | |

Reserve probing overhead is variable — 40 candidates in normal phases (~6%), up to 80 in high-eviction phases (~13%). The 40% dynamic overhead is the cost of always knowing which problems are Goldilocks.

### Time estimate (H100)

| | Per phase | 8 phases total |
|---|---|---|
| Dynamic | ~35–45 min | ~4.5–6 hours |
| Static | ~23–28 min | ~3–4 hours |
| **All 4 runs sequential** | | **~15–20 hours** |

Running dynamic + static in parallel on 2 H100s cuts wall-clock time in half per model pair.

---

## Model Configuration

### LoRA
- Rank r=32, alpha=64 (standard 2× scaling)
- Applied to all 7 modules: q/k/v/o projections + MLP gate/up/down
- r=32 > typical instruction-tuning default (r=16): RL needs to develop new reasoning patterns, not just style-transfer
- Including MLP layers matters for math reasoning; attention-only LoRA is less expressive

### Training setup
- **bf16**: H100 native; no quantization needed
- **Flash Attention 2**: auto-detected, falls back to eager if unavailable
- **Gradient checkpointing**: enabled; requires `model.enable_input_require_grads()` with PEFT
- **KL penalty**: `beta=0.04` (GRPO paper default) — prevents reward hacking and partially anchors easy-problem reasoning even after those problems are evicted from the curriculum
- **LR**: 2e-5, warmup only on phase 1. Higher than conservative 1e-5 because at 640 total steps we need enough model movement per phase to trigger saturation events. Subsequent phases start at full LR to avoid wasting 12.5% of each phase on warmup.
- **Batch size**: 8 per device, grad_accum=2 → effective batch 16

### Why no partial-credit reward

We considered adding 0.1 reward for `\boxed{}` formatting. Rejected: trains format, not math. Our three-tier answer extraction (`<answer>` tag → `\boxed{}` → last number) handles format variation on the scoring side. A formatting reward would reward `\boxed{wrong_answer}`.

---

## Logging and Output Artifacts

Every run produces a self-contained directory at `checkpoints/<model>_<condition>/`:

| File | Written | Contents |
|---|---|---|
| `run_config.json` | job start | all hyperparameters, model, condition, timestamp |
| `heldout_results.jsonl` | after each phase | held-out accuracy overall + by level (L1/L2/L3) + by subject |
| `train_loss.jsonl` | every 5 steps | loss, reward, KL — step-level training signal, survives disconnects |
| `curriculum_state.json` | after each refresh | active/reserve/cooldown state, goldilocks_frac, mean_grad_signal, per-problem pass_rates |
| `static_goldilocks_log.jsonl` | static only | sampled goldilocks_frac + mean_grad_signal per phase |
| `checkpoint_phase_NNN/` | after each phase | LoRA adapter weights |
| `final_adapter/` | end of training | final weights + tokenizer |

All files are written to the persistent studio volume on Lightning AI — survive machine crashes and timeouts. Check progress remotely via the Lightning AI file browser or `tail -f`.

---

## Key Diagnostics

Watch these in order of importance:

**1. mean_grad_signal per phase (both conditions)**
`mean(2p(1−p))` over all active problems. The direct measure of gradient quality.
- Dynamic: should stay ≥ 0.30 throughout as low-signal problems are cycled out
- Static: will start at ~0.13 (Qwen) and decline further as saturation accumulates

**2. goldilocks_frac per phase (both conditions)**
Fraction of active problems with 0 < pass_rate < 1.
- Dynamic: should stay >70% throughout
- Static: starts at 32.7% (Qwen) / 59% (Llama) and will decline

**3. Held-out accuracy per phase (primary outcome)**
Overall + L1/L2/L3 breakdown. L1 specifically: if dynamic shows L1 decline while L2/L3 improve, that's evidence of catastrophic forgetting and would be an important finding.

**4. train_loss.jsonl**
Step-level loss and KL. Useful for detecting reward hacking (reward rises while held-out accuracy doesn't) or training instability.

---

## What Success Looks Like

**Strong result**: Dynamic reaches the same held-out accuracy as static in fewer steps (compute efficiency win).

**Acceptable result**: Dynamic reaches higher held-out accuracy at the same step count (quality win).

**Weak but meaningful**: Goldilocks fraction and mean_grad_signal stay high in dynamic and collapse in static — demonstrating the mechanism even if accuracy gaps are small at 640 steps. The static run proving that 67% of its compute is wasted gradient is itself a meaningful finding.

640 steps (8 phases) should be enough to see the gradient quality curves diverge. A clear plateau in static held-out accuracy while dynamic keeps improving is the ideal result. If budget allows, 1200 steps would give a stronger signal.

---

## Known Limitations

1. **Phase-boundary scoring only**: Pass rates update at phase boundaries, not continuously. The curriculum is stale by up to 80 steps within a phase.

2. **Binary reward only**: No partial credit. Problems the model nearly solves contribute the same zero gradient as ones it's totally wrong about.

3. **Single H100**: Multi-GPU would allow larger effective batch sizes and more rollout diversity.

4. **LoRA ceiling**: Limited expressivity may understate the dynamic curriculum benefit — with full fine-tune, saturation events would be more dramatic and the adaptive mechanism more necessary.

5. **Qwen reserve thinness**: Qwen has only 56 tier-1 reserve problems (vs 222 for Llama). If tier-2 unlocking is slow, Qwen's active pool may shrink toward the floor in late phases. The warning log will flag this; the two-round probe and cooldown thaws partially mitigate it.

---

## Current Status (as of 2026-03-31)

| Step | Status |
|---|---|
| Build 630-problem L1L2L3 training pool | Done |
| Build 210-problem held-out set (seed=99, 0 overlap, 0 duplicates) | Done |
| Pass@8 pre-eval (Qwen-2.5-7B) | Done — 206 Goldilocks, 161 unreachable |
| Pass@8 pre-eval (Llama-3-8B) | Done — 372 Goldilocks, 154 unreachable |
| Export Goldilocks inventories | Done |
| Training script + curriculum design | Done — ready to launch |
| Dynamic training run × 2 models | **Pending — launch Qwen first** |
| Static training run × 2 models | **Pending** |
