#!/bin/bash
# CalibrateRL — H100 (Nebius) Lightning AI Setup & Launch
#
# Usage:
#   bash setup_and_train.sh llama-3-8b    # default
#   bash setup_and_train.sh qwen-2.5-7b
#
# Before running:
#   export HF_TOKEN=hf_...   (required for Llama; skip for Qwen)

set -e

MODEL=${1:-llama-3-8b}
echo "==> Model: $MODEL"

# ── Install dependencies ───────────────────────────────────────────────────────
echo "==> Installing dependencies…"
pip install -q \
    "torch>=2.2.0" \
    "transformers>=4.40.0" \
    "trl>=0.8.6" \
    "peft>=0.10.0" \
    "accelerate>=0.28.0" \
    "datasets>=2.18.0"

# flash-attn: ~2-3× attention speedup on H100. Compile takes ~10-15 min but
# subsequent runs skip it (wheel is cached). Required for best H100 utilization.
echo "==> Installing flash-attn (may take ~10-15 min to compile; cached on reruns)…"
pip install -q flash-attn --no-build-isolation

# ── Verify GPU ─────────────────────────────────────────────────────────────────
echo "==> GPU:"
python3 -c "
import torch
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f'  {d.name}  {d.total_memory // 1024**3}GB VRAM')
    print(f'  BF16 support: {torch.cuda.is_bf16_supported()}')
else:
    print('  WARNING: No GPU found')
"

# ── Flash Attention 2 status ───────────────────────────────────────────────────
python3 -c "
import torch
try:
    import flash_attn
    capable = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    status = 'ENABLED' if capable else 'installed but GPU < Ampere, will use eager'
    print(f'  flash-attn {flash_attn.__version__} — {status}')
except ImportError:
    print('  flash-attn not installed — using eager attention (still works fine)')
    print('  To enable: pip install flash-attn --no-build-isolation')
"

# ── Verify data files ──────────────────────────────────────────────────────────
echo "==> Checking data files…"
python3 -c "
from pathlib import Path
files = [
    'data/goldilocks_llama-3-8b.json',
    'data/goldilocks_qwen-2.5-7b.json',
    'data/profile_dataset_L1L2L3.json',
    'data/heldout_eval.json',
]
ok = True
for p in files:
    exists = Path(p).exists()
    print(f'  [{\"OK\" if exists else \"MISSING\"}] {p}')
    if not exists: ok = False
if not ok:
    raise SystemExit('Missing data files — cannot start training')
"

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "==> Starting GRPO training on H100…"
echo ""
echo "Run this for dynamic curriculum (default):"
echo "  python3 src/train_grpo.py --model $MODEL"
echo ""
echo "Run this for static baseline:"
echo "  python3 src/train_grpo.py --model $MODEL --static"
echo ""
echo "Both use --max-steps 640 (8 phases × 80 steps) by default."
echo "Add --resume to recover from a killed job."
echo ""

# Uncomment one of these to launch automatically:
# python3 src/train_grpo.py --model "$MODEL"
# python3 src/train_grpo.py --model "$MODEL" --static

echo "==> Setup complete. Launch training with the commands above."
