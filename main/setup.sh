#!/usr/bin/env bash
# ============================================================================
# Environment Setup for Tiny-Math-Solver
# ============================================================================
# Run this ONCE on a fresh Lightning Studios instance (or any machine).
# Handles the dependency conflicts between TRL, vLLM, numpy, flash-attn, etc.
#
# Usage:
#     bash setup.sh
#
# What this does:
#   1. Removes openrlhf (legacy, conflicts with flash-attn/numpy)
#   2. Pins numpy to <2.3 (numba compatibility for vLLM)
#   3. Installs/upgrades TRL, PEFT, transformers, etc. to compatible versions
#   4. Builds the entity-tracking dataset from GSM8K (if not already built)
#
# NOTE: We intentionally do NOT use vLLM colocate mode for generation.
# As of Feb 2026, PEFT + vLLM colocate has known convergence bugs
# (github.com/huggingface/trl/issues/2856, vllm-project/vllm/issues/14483).
# We use transformers paged attention instead -- slower but reliable.
# ============================================================================
set -euo pipefail

echo "============================================"
echo "  Tiny-Math-Solver — Environment Setup"
echo "============================================"
echo ""

# ── Step 1: Remove conflicting packages ─────────────────────────────────────
echo ">>> Step 1: Removing conflicting packages ..."
pip uninstall openrlhf -y 2>/dev/null && echo "    Removed openrlhf" || echo "    openrlhf not installed (OK)"
echo ""

# ── Step 2: Install pinned dependencies ─────────────────────────────────────
echo ">>> Step 2: Installing dependencies ..."
pip install -r requirements.txt
echo ""

# ── Step 3: Verify critical imports ─────────────────────────────────────────
echo ">>> Step 3: Verifying imports ..."
python -c "
import sys
errors = []

try:
    import trl; print(f'    trl:          {trl.__version__}')
except Exception as e: errors.append(f'trl: {e}')

try:
    import transformers; print(f'    transformers: {transformers.__version__}')
except Exception as e: errors.append(f'transformers: {e}')

try:
    import peft; print(f'    peft:         {peft.__version__}')
except Exception as e: errors.append(f'peft: {e}')

try:
    import datasets; print(f'    datasets:     {datasets.__version__}')
except Exception as e: errors.append(f'datasets: {e}')

try:
    import numpy; print(f'    numpy:        {numpy.__version__}')
except Exception as e: errors.append(f'numpy: {e}')

try:
    import torch; print(f'    torch:        {torch.__version__}')
    if torch.cuda.is_available():
        print(f'    CUDA:         {torch.version.cuda}')
        print(f'    GPU:          {torch.cuda.get_device_name(0)}')
        print(f'    VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        errors.append('CUDA not available!')
except Exception as e: errors.append(f'torch: {e}')

if errors:
    print()
    print('  ERRORS:')
    for e in errors:
        print(f'    ✗ {e}')
    sys.exit(1)
else:
    print('    All imports OK')
"
echo ""

# ── Step 4: Build dataset if needed ─────────────────────────────────────────
if [ ! -d "data/entity_tracking_dataset" ]; then
    echo ">>> Step 4: Building entity-tracking dataset from GSM8K ..."
    python src/build_entity_dataset.py
else
    echo ">>> Step 4: Dataset already exists at data/entity_tracking_dataset/"
fi

echo ""
echo "============================================"
echo "  Setup complete! Run training with:"
echo "    bash src/train_math.sh"
echo "  or:"
echo "    WANDB_TOKEN=... bash src/train_math.sh"
echo "============================================"
