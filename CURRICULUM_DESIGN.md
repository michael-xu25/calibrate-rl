# Dynamic Curriculum Architecture — CalibrateRL

## Core Insight

GRPO needs gradient signal. Gradient signal only exists in the **Goldilocks zone**:
- `max_score == 1` → model has latent ability to solve it (not impossible)
- `0 < pass_rate < 1` → model doesn't solve it reliably (not saturated)

A static dataset goes stale. As training progresses, easy problems saturate
(pass_rate → 1) and stop contributing signal. The curriculum must track the
model's moving frontier.

## The Three Zones

```
pass_rate = 0              pass_rate ∈ (0, 1)         pass_rate = 1
─────────────────────────────────────────────────────────────────────
   UNREACHABLE                  GOLDILOCKS                SATURATED
   (no latent ability           (train here)              (already learned,
    or too hard right now)                                 gradient = 0)
```

Problems move **left to right** as training improves the model. The curriculum
window slides with them.

## The Re-evaluation Loop

Every `EVAL_INTERVAL` gradient steps:

1. **Score** current curriculum: run N_ROLLOUTS generations per problem,
   compute pass_rate.
2. **Evict** saturated problems (pass_rate ≥ SATURATE_THRESHOLD).
3. **Evict** unreachable problems (pass_rate == 0 for 2+ consecutive evals).
4. **Replenish** from the reserve pool (full L1+L2 dataset minus already-used)
   by sampling problems that fall in the Goldilocks window on the current model.

## Plateau Mechanics

The ceiling is set by `max_score`. At L4–L5, an 8B model rarely gets even one
correct rollout → reserve pool of valid Goldilocks candidates shrinks to zero →
curriculum can't be replenished → training stalls naturally.

Expected trajectory for an 8B model on MATH:
- **L1 → L2**: Strong improvement (latent ability unlocked)
- **L2 → L3**: Diminishing returns, pool thins out
- **L3+**: Plateau — Goldilocks pool approaches empty

## Implementation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `N_ROLLOUTS` | 8 | Pass rates in [0, 0.125, …, 1.0] steps |
| `EVAL_INTERVAL` | 20 steps | Balance signal freshness vs. eval cost |
| `SATURATE_THRESHOLD` | 0.875 | 7/8 rollouts correct = effectively learned |
| `UNREACHABLE_PATIENCE` | 2 evals | Avoid premature eviction on noisy evals |
| `MIN_CURRICULUM_SIZE` | 16 | Stop training if pool can't fill a batch |
| `TARGET_CURRICULUM_SIZE` | 64 | Problems kept active at any time |

## Why Not vLLM

Lightning AI training environment is incompatible with vLLM. All generation
uses standard HuggingFace `model.generate()` calls, which is slower but fully
compatible with LoRA/QLoRA and the trl GRPOTrainer.
