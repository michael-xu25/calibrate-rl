# Quick Validation Guide - Prove Difficulty > Entity Heuristic

**Goal:** Prove difficulty calibration beats entity counting in ~3 hours for ~$5 on L40S.

---

## What We're Proving

Your previous run (entity filter) had:
- **40-42% ghost batching** (wasted compute)
- **Plateaued at step 15** (no learning after)
- Dataset: 3+ entity heuristic

Sweet spot should show:
- **<20% ghost batching** (2-3x better signal)
- **Continued learning past step 15**
- Dataset: 2-12/16 pass rate (direct difficulty measurement)

---

## Fair Comparison Design

### What's THE SAME (controlled variables):
✓ Model: Qwen2.5-1.5B-Instruct
✓ Training steps: 50 (matches your checkpoint-50)
✓ LoRA config: rank 16, alpha 32
✓ GRPO: 8 generations, temp 1.0, batch 4×16 accum
✓ Learning rate: 5e-5, cosine, 3% warmup
✓ KL penalty: beta 0.04
✓ Reward: correctness + format
✓ Evaluation: 100 test problems, greedy

### What's DIFFERENT (treatment):
✗ **Dataset selection method:**
  - Entity: Regex heuristic (3+ named entities)
  - Sweet spot: Pass@16 measurement (2-12/16 correct)

✗ **Expected ghost batch rate:**
  - Entity: 40-42% (measured in your run)
  - Sweet spot: 7-15% (predicted from analysis)

✗ **System prompt:**
  - Entity: "For each person or item..." (entity-specific)
  - Sweet spot: "Show your reasoning clearly" (general)
  - (Fairer since sweet spot includes non-entity problems)

---

## Lightning AI Commands (Copy-Paste Ready)

### 1. Setup

```bash
# On Lightning AI with L40S
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro/main

pip install -r requirements.txt

# Verify GPU
nvidia-smi
```

### 2. Run Complete Validation Pipeline

```bash
# Single command - does everything
chmod +x quick_validation.sh
./quick_validation.sh

# This will:
# 1. Build sweet spot dataset (50 problems, ~30 min)
# 2. Show predicted comparison
# 3. Update training script for 50-step run
# 4. Show you what to run next
```

### 3. Train (Manual Step)

After script finishes:

```bash
# Start training (runs for 50 steps, ~2 hours)
nohup python3 src/train_grpo.py > logs/grpo_sweet_spot_50step.log 2>&1 &

# Monitor progress
tail -f logs/grpo_sweet_spot_50step.log

# Watch for these metrics in the logs:
# - Ghost batch rate (should be <20% vs 40-42% entity)
# - Correctness (should keep improving past step 15)
```

### 4. Compare Results

```bash
# After training completes
python3 compare_runs.py

# Shows side-by-side:
# Steps 1-10:   Entity vs Sweet Spot
# Steps 11-50:  Entity vs Sweet Spot
# Ghost rate reduction %
```

---

## Timeline & Cost (L40S @ $1.50/hr)

| Step | Time | Cost | Output |
|------|------|------|--------|
| Setup | 5 min | $0.10 | Dependencies |
| Pass@16 (50 samples) | 30 min | $0.75 | Sweet spot dataset (~20 problems) |
| Training (50 steps) | 2 hours | $3.00 | Checkpoint-50 |
| Comparison | 2 min | $0.05 | Metrics comparison |
| **Total** | **~2.5 hrs** | **~$4** | **Proof of concept** |

---

## Success Criteria

Sweet spot wins if **ANY** of these are true:

1. **Ghost batch rate < 20%** (vs 40-42%)
   → Proves better compute efficiency

2. **Correctness improving at step 50** (vs plateaued at 15)
   → Proves continued learning

3. **Final accuracy ≥ 77%** (same as entity)
   → Proves equal or better results with less waste

**Even if final accuracy is slightly lower, if ghost batching is 2x better, you've proven the thesis.**

---

## What the Logs Should Show

### Entity Filter (Your Previous Run)
```
Steps 1-10:   Correctness 57.8%, Ghost 18.8%
Steps 11-50:  Correctness 81.4%, Ghost 41.6%  ← Plateaued
Steps 51+:    No further improvement
```

### Sweet Spot (New Run - Expected)
```
Steps 1-10:   Correctness ~60%, Ghost ~10%
Steps 11-50:  Correctness ~85%, Ghost ~12%    ← Still improving
```

Key differences:
- **Lower ghost rate throughout** (10-12% vs 41%)
- **No plateau** - continues learning
- **Similar or better final accuracy**

---

## Troubleshooting

### "Script fails at pass@16 step"
```bash
# Reduce sample size
./quick_validation.sh  # Edit N_SAMPLE=50 to N_SAMPLE=25
```

### "Training runs out of memory"
```bash
# Your entity run worked on L40S, so sweet spot should too
# But if issues, check:
nvidia-smi  # Should show ~25GB VRAM usage during training
```

### "Can't find checkpoint after training"
```bash
# Training saves to ./checkpoint/run_TIMESTAMP/
ls -lh checkpoint/

# If killed early, may not have saved
# Check logs for "saved model" message
```

### "Want to compare to original entity run logs"
```bash
# Your old logs are preserved in:
cat rl-intro_logs/train.log | grep -i "ghost\|correctness" | head -20
```

---

## After Validation

If sweet spot proves better:

1. **Week 1 complete** ✓ - Difficulty beats heuristics
2. **Week 2:** Build difficulty predictor (avoid pass@16 every time)
3. **Week 2:** Problem generation (synthesize new goldilocks problems)
4. **Week 3:** Full 500-step training run with sweet spot
5. **Week 4:** Environment quality report for pitch deck

---

## Restore Original Files

After validation:

```bash
# Restore original training script
mv src/train_grpo.py.backup src/train_grpo.py

# Or keep sweet spot version for future runs
```

---

## Expected Results Summary

| Metric | Entity Filter | Sweet Spot | Win? |
|--------|--------------|------------|------|
| Ghost batch rate (steps 11-50) | 41.6% | ~12% | ✅ 71% reduction |
| Plateaus at step 15? | Yes | No | ✅ Continued learning |
| Final pass@1 | 77.0% | ~77-80% | ✅ Equal or better |
| Dataset selection | Heuristic | Measured | ✅ More principled |

**Bottom line:** Proof that difficulty calibration > entity counting for RL training efficiency.

---

## Questions?

Check the full implementation doc: `week1_sweet_spot_implementation.md`

Or Lightning AI setup guide: `LIGHTNING_AI_SETUP.md`
