# Lightning AI Setup Guide - CalibrateRL Week 1

## GPU Recommendations

### For Pass@16 Evaluation (Building Sweet Spot Dataset)
**Task:** Run 500 problems × 16 samples each = 8,000 generations

| GPU | Time Estimate | Cost/hr | Total Cost | Recommendation |
|-----|--------------|---------|------------|----------------|
| **L40S** | ~4 hours | $1.50 | ~$6 | ⭐ **RECOMMENDED** - Best balance |
| **H100** | ~2.5 hours | $3.50 | ~$9 | Fast but expensive |
| **L4** | ~8-10 hours | $0.60 | ~$5 | Budget option (slower) |
| T4 | ~20+ hours | $0.40 | ~$8 | ❌ Too slow, not worth it |

### For GRPO Training
**Task:** RL training with batch size 8 × gradient accumulation 16

| GPU | VRAM | Status | Recommendation |
|-----|------|--------|----------------|
| **L40S** | 48GB | ✅ Tested | ⭐ **RECOMMENDED** |
| **H100** | 80GB | ✅ Will work | Faster but pricier |
| A100 | 40GB | ⚠️ Tight | May need smaller batch |
| L4 | 24GB | ❌ Too small | Won't fit |

**TLDR: Use L40S for both evaluation and training**

---

## Lightning AI Commands

### 1. Create Studio Session

```bash
# In Lightning AI web interface:
# 1. Create New Studio
# 2. Select: L40S GPU (1x)
# 3. Click "Start"
```

### 2. Clone Repository and Setup

```bash
# Clone your repo
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro/main

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Build Sweet Spot Dataset (Pass@16 Evaluation)

```bash
# IMPORTANT: Run this in a persistent way in case connection drops
# Lightning AI Studios automatically reconnect, but use nohup for safety

nohup python3 src/build_sweet_spot_dataset.py \
    --n_sample 500 \
    > logs/sweet_spot_build.log 2>&1 &

# Check progress
tail -f logs/sweet_spot_build.log

# Monitor GPU usage in another terminal
watch -n 5 nvidia-smi
```

**Expected output:**
- Runtime: ~4 hours on L40S
- Output: `data/sweet_spot_dataset/` (200-250 problems)
- Pass@16 results: `logs/train_pass_at_16.jsonl`

**If you want a quick test first:**
```bash
# Test with 50 problems (~30 min)
python3 src/build_sweet_spot_dataset.py --n_sample 50
```

### 4. Analyze Results

```bash
# Run the comparison analysis
python3 src/analyze_sweet_spot.py

# Should show:
# - Ghost batch rate: ~7% (vs 40-42% entity filter)
# - Difficulty distribution
# - Sweet spot: 41% of problems
```

### 5. Update Training Script

```bash
# Edit src/train_grpo.py line 90
# Change: dataset = load_from_disk("data/entity_tracking_dataset")
# To:     dataset = load_from_disk("data/sweet_spot_dataset")

sed -i 's/data\/entity_tracking_dataset/data\/sweet_spot_dataset/g' src/train_grpo.py

# Verify the change
grep "sweet_spot_dataset" src/train_grpo.py
```

### 6. Run GRPO Training

```bash
# Run training with logging
nohup python3 src/train_grpo.py > logs/grpo_sweet_spot.log 2>&1 &

# Monitor progress
tail -f logs/grpo_sweet_spot.log

# Or use the latest timestamped log
tail -f logs/run_*.log | tail -100
```

### 7. Monitor Training

```bash
# Check training progress
tail -100 logs/run_*.log

# Look for these metrics:
# - Ghost batch rate (should be ~7-15%, down from 40%)
# - Correctness reward (accuracy on training problems)
# - KL divergence (policy drift)
# - Loss values
```

### 8. Download Results

```bash
# After training, download checkpoint and logs
# In Lightning AI terminal:

# Compress results
tar -czf sweet_spot_results.tar.gz \
    checkpoint/ \
    logs/ \
    data/sweet_spot_dataset/

# Download via Lightning AI file browser or:
# Use Lightning CLI to download to local machine
```

---

## Full Pipeline Script

Save this as `run_week1_pipeline.sh`:

```bash
#!/bin/bash
set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════"
echo "  CalibrateRL Week 1 Pipeline - Sweet Spot Implementation"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Configuration
N_SAMPLE=500  # Number of train problems to evaluate
GPU_CHECK=true

# 1. GPU Check
if [ "$GPU_CHECK" = true ]; then
    echo ">>> Checking GPU availability..."
    nvidia-smi
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
    echo "✓ GPU available"
    echo ""
fi

# 2. Create directories
echo ">>> Setting up directories..."
mkdir -p logs data
echo "✓ Directories ready"
echo ""

# 3. Build sweet spot dataset
echo ">>> Building sweet spot dataset (this takes ~4 hours on L40S)..."
echo "    Evaluating $N_SAMPLE training problems with pass@16"
echo ""

python3 src/build_sweet_spot_dataset.py \
    --n_sample $N_SAMPLE \
    2>&1 | tee logs/sweet_spot_build_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✓ Sweet spot dataset created"
echo ""

# 4. Analyze results
echo ">>> Running analysis..."
python3 src/analyze_sweet_spot.py | tee logs/analysis_$(date +%Y%m%d_%H%M%S).log
echo "✓ Analysis complete"
echo ""

# 5. Update training script
echo ">>> Updating training script to use sweet spot dataset..."
sed -i.bak 's/data\/entity_tracking_dataset/data\/sweet_spot_dataset/g' src/train_grpo.py
echo "✓ Training script updated"
echo ""

# 6. Summary
echo "═══════════════════════════════════════════════════════════"
echo "  Week 1 Pipeline Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next step: Run GRPO training with:"
echo "  nohup python3 src/train_grpo.py > logs/grpo_sweet_spot.log 2>&1 &"
echo ""
echo "Files created:"
echo "  - data/sweet_spot_dataset/         (filtered training data)"
echo "  - logs/train_pass_at_16.jsonl      (pass@16 results)"
echo "  - logs/sweet_spot_build_*.log      (build log)"
echo "  - logs/analysis_*.log              (comparison analysis)"
echo ""
echo "═══════════════════════════════════════════════════════════"
```

Then run:
```bash
chmod +x run_week1_pipeline.sh
./run_week1_pipeline.sh
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce sample size for pass@16
python3 src/build_sweet_spot_dataset.py --n_sample 100

# Or reduce batch size in evaluation (edit build_sweet_spot_dataset.py)
# Change K=16 to K=8 or K=4
```

### Connection Drops
Lightning AI Studios automatically reconnect. Your processes keep running.

To check if your process is still running:
```bash
ps aux | grep python3
tail logs/sweet_spot_build.log
```

### Slow Progress
```bash
# Check GPU utilization
nvidia-smi

# Should see:
# - GPU memory usage: 15-25GB (during generation)
# - GPU utilization: 80-100%
# - If low: model might be on CPU, check device_map="auto"
```

### Dataset Not Found
```bash
# Verify dataset was created
ls -lh data/sweet_spot_dataset/

# Check logs for errors
tail -100 logs/sweet_spot_build.log
```

---

## Expected Timeline (on L40S)

| Step | Time | Output |
|------|------|--------|
| Setup & Install | 10 min | Dependencies installed |
| Pass@16 Evaluation (500) | 4 hours | `data/sweet_spot_dataset/` |
| Analysis | 2 min | Ghost batch comparison |
| GRPO Training | 5-8 hours | Checkpoint + logs |
| **Total** | **~10-12 hours** | Complete Week 1 |

---

## Cost Estimate (L40S @ $1.50/hr)

| Task | Hours | Cost |
|------|-------|------|
| Pass@16 Evaluation | 4 | $6 |
| GRPO Training | 6 | $9 |
| Buffer (testing, analysis) | 2 | $3 |
| **Total** | **12** | **~$18** |

---

## What to Expect in Logs

### Pass@16 Build Log
```
[  1/500] ✓ 12/16 |████████████░░░░| gold=240
[  2/500] ✗  0/16 |░░░░░░░░░░░░░░░░| gold=15
[  3/500] ✓  7/16 |███████░░░░░░░░░| gold=120
...

>>> Difficulty distribution (500 problems):
     0/16 correct:  40 ( 8.0%)
     2/16 correct:  15 ( 3.0%)  ← SWEET SPOT
     ...
    12/16 correct:  35 ( 7.0%)  ← SWEET SPOT
    13/16 correct:  50 (10.0%)
    ...

>>> Training set breakdown:
    Too hard (0-1/16):    40 ( 8%)
    Sweet spot (2-12/16): 205 (41%)  ← Your training data
    Too easy (13-16/16):  255 (51%)
```

### Training Log (Good Signs)
```
Step 1:  Correctness: 0.58  Ghost: 15%  KL: 0.002
Step 10: Correctness: 0.72  Ghost: 12%  KL: 0.015
Step 50: Correctness: 0.85  Ghost:  8%  KL: 0.018

✓ Ghost rate staying low (8-15% vs old 40%)
✓ Correctness improving steadily (not plateauing at step 15)
✓ KL controlled (not drifting too far from base policy)
```

---

## Quick Start (TLDR)

```bash
# On Lightning AI (L40S):
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro/main
pip install -r requirements.txt

# Build dataset (~4 hours)
nohup python3 src/build_sweet_spot_dataset.py --n_sample 500 > logs/build.log 2>&1 &

# Update training script
sed -i 's/entity_tracking_dataset/sweet_spot_dataset/g' src/train_grpo.py

# Train (~6 hours)
nohup python3 src/train_grpo.py > logs/train.log 2>&1 &

# Monitor
tail -f logs/build.log  # or logs/train.log
```

**Cost: ~$18 on L40S for complete Week 1 pipeline**

---

## Next Steps After Week 1

See `week1_sweet_spot_implementation.md` for:
- Expected improvements (83% ghost batch reduction)
- Week 2 roadmap (difficulty predictor, problem generation)
- Comparison analysis vs entity filter
