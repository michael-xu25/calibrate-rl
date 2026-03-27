#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Quick Validation: Sweet Spot vs Entity Filter                ║"
echo "║   Fair comparison using same config, same 50 steps, same eval        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pass@16 sampling size (50 = ~30 min, 100 = ~1 hr)
N_SAMPLE=50

# Training steps (50 matches your previous checkpoint-50 evaluation)
MAX_STEPS=50

# Evaluation size (100 matches your baseline/checkpoint eval)
EVAL_SIZE=100

echo "Configuration:"
echo "  Pass@16 sample size:  $N_SAMPLE problems"
echo "  Training steps:       $MAX_STEPS (matches checkpoint-50)"
echo "  Evaluation size:      $EVAL_SIZE test problems"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1: Build Sweet Spot Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: Building Sweet Spot Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "logs/train_pass_at_16.jsonl" ]; then
    echo "⚠️  Found existing pass@16 results. Delete logs/train_pass_at_16.jsonl to rebuild."
    echo "   Using --skip_eval to reuse existing data..."
    python3 src/build_sweet_spot_dataset.py --n_sample $N_SAMPLE --skip_eval
else
    echo "Running pass@16 evaluation on $N_SAMPLE training problems..."
    echo "Expected time: ~30 min for 50 problems, ~1 hr for 100 problems"
    echo ""
    python3 src/build_sweet_spot_dataset.py --n_sample $N_SAMPLE
fi

echo ""
echo "✓ Sweet spot dataset created: data/sweet_spot_dataset/"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: Show Comparison Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: Comparison Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 src/analyze_sweet_spot.py

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3: Update Training Config for 50-Step Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 3: Preparing Training Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup original
cp src/train_grpo.py src/train_grpo.py.backup

# Update dataset path
sed -i.tmp 's/data\/entity_tracking_dataset/data\/sweet_spot_dataset/g' src/train_grpo.py

# Update max_steps to 50 for quick validation
sed -i.tmp 's/max_steps=500/max_steps='"$MAX_STEPS"'/g' src/train_grpo.py

# Update system prompt to be more general (not entity-specific)
# Keep the <think> tag structure but remove entity-specific instructions
cat > /tmp/new_prompt.txt << 'EOF'
SWEET_SPOT_PROMPT = (
    "Think step by step inside <think> tags before answering. "
    "Show your reasoning clearly and state your final answer."
)
EOF

# Replace the prompt definition
sed -i.tmp '/^ENTITY_TRACKING_PROMPT = (/,/^)/c\
SWEET_SPOT_PROMPT = (\
    "Think step by step inside <think> tags before answering. "\
    "Show your reasoning clearly and state your final answer."\
)' src/train_grpo.py

# Update references to the prompt variable
sed -i.tmp 's/ENTITY_TRACKING_PROMPT/SWEET_SPOT_PROMPT/g' src/train_grpo.py

echo "✓ Updated training script:"
echo "  - Dataset: sweet_spot_dataset"
echo "  - Max steps: $MAX_STEPS"
echo "  - System prompt: General reasoning (not entity-specific)"
echo "  - All other hyperparameters: UNCHANGED (same as entity run)"
echo ""

# Show what changed
echo "Key config (unchanged from entity run):"
grep -A 15 "# GRPO sampling" src/train_grpo.py | head -16
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4: Print Comparison Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FAIR COMPARISON DESIGN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat << 'EOF'
┌─────────────────────────────────────────────────────────────────────┐
│                   WHAT WE'RE COMPARING                              │
└─────────────────────────────────────────────────────────────────────┘

Your Previous Run (Entity Filter):
  Dataset:          entity_tracking_dataset (~2500 problems, 3+ entities)
  Training steps:   324/500 (killed early, but plateaued at step ~15)
  Checkpoint eval:  checkpoint-50 only
  Results:
    Steps 1-10:   57.8% correctness, 18.8% ghost batches
    Steps 11-50:  81.4% correctness, 41.6% ghost batches
    Final (step 50): +9.3% greedy pass@1 (67.7% → 77.0%)
  Key issue:      Plateaued at step 15, 40-42% ghost batching

New Run (Sweet Spot Filter):
  Dataset:          sweet_spot_dataset (~20 problems from 50 sample)
  Training steps:   50 (same checkpoint-50 comparison point)
  Checkpoint eval:  checkpoint-50
  Prediction:
    Steps 1-10:   Should show similar initial learning
    Steps 11-50:  7-15% ghost batches (vs 41.6%)
    Final:        Should NOT plateau at step 15
    Expected:     Better or equal final accuracy, less wasted compute

┌─────────────────────────────────────────────────────────────────────┐
│                   WHAT'S THE SAME (FAIR)                            │
└─────────────────────────────────────────────────────────────────────┘

✓ Model:                  Qwen2.5-1.5B-Instruct
✓ LoRA config:            rank=16, alpha=32, dropout=0.05
✓ GRPO config:            8 generations, temp=1.0, batch=4, grad_accum=16
✓ Training steps:         50 steps
✓ Learning rate:          5e-5, cosine schedule, 3% warmup
✓ KL penalty:             beta=0.04
✓ Reward function:        correctness + format (same as before)
✓ Evaluation:             100 test problems, greedy generation
✓ System prompt style:    <think> tags for reasoning

┌─────────────────────────────────────────────────────────────────────┐
│                   WHAT'S DIFFERENT (TREATMENT)                      │
└─────────────────────────────────────────────────────────────────────┘

✗ Dataset selection:      Entity count (heuristic) → Pass@16 (measured)
✗ Dataset size:           ~2500 problems → ~20 problems (but higher quality)
✗ Expected ghost rate:    40-42% → 7-15%
✗ System prompt content:  Entity-specific → General reasoning
                         (fairer since sweet spot isn't all entity problems)

┌─────────────────────────────────────────────────────────────────────┐
│                   SUCCESS CRITERIA                                  │
└─────────────────────────────────────────────────────────────────────┘

Sweet spot wins if:
  1. Ghost batch rate < 20% (vs 40-42% entity)
  2. Correctness still improving at step 50 (vs plateaued at 15)
  3. Final accuracy ≥ entity run (77.0% greedy pass@1)

Even if final accuracy is slightly lower, if ghost batching is 2-3x
better, we've proven the thesis: difficulty calibration > heuristics.

EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "READY TO TRAIN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run training with:"
echo ""
echo "  nohup python3 src/train_grpo.py > logs/grpo_sweet_spot_50step.log 2>&1 &"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/grpo_sweet_spot_50step.log"
echo ""
echo "Expected runtime: ~2 hours on L40S"
echo ""
echo "After training, evaluate with:"
echo "  python3 src/eval_checkpoint.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To restore original training script:"
echo "  mv src/train_grpo.py.backup src/train_grpo.py"
echo ""
