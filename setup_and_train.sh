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
# Versions pinned to avoid silent API breakage:
#   trl 0.17.x: uses reward_funcs= and processing_class= (our API)
#   trl <0.9:   uses reward_function= and tokenizer= (breaks silently)
#   transformers 4.47+: required for Qwen2.5 and Llama3 tokenizers
pip install -q \
    "torch>=2.2.0" \
    "transformers>=4.47.0" \
    "trl>=0.17.0,<0.18.0" \
    "peft>=0.14.0" \
    "accelerate>=0.28.0" \
    "datasets>=2.18.0"

# flash-attn: optional ~2-3× attention speedup on H100.
# Requires a C++ compiler and CUDA headers in the image. If your environment
# supports it, run once manually: pip install flash-attn --no-build-isolation
# The training script auto-detects it and falls back to eager attention if absent.

# ── Verify TRL API ─────────────────────────────────────────────────────────────
echo "==> Checking TRL API compatibility…"
python3 -c "
import trl, inspect
sig = inspect.signature(trl.GRPOTrainer.__init__)
params = set(sig.parameters)
ok = True
for needed in ('reward_funcs', 'processing_class'):
    if needed not in params:
        print(f'  MISSING param: {needed} — wrong TRL version ({trl.__version__})')
        ok = False
    else:
        print(f'  [OK] {needed}')
if not ok:
    raise SystemExit('TRL version incompatible. Run: pip install \"trl>=0.17.0,<0.18.0\"')
print(f'  TRL version: {trl.__version__}')
"

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
