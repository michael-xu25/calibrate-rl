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
    "datasets>=2.18.0" \
    "flash-attn>=2.5.0" --no-build-isolation

# bitsandbytes not needed on H100 (full bf16, no quantization)

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

# ── Verify Flash Attention 2 ───────────────────────────────────────────────────
python3 -c "
try:
    import flash_attn; print(f'  flash-attn {flash_attn.__version__} OK')
except ImportError:
    print('  WARNING: flash-attn not available — training will still work but slower')
"

# ── Verify data files ──────────────────────────────────────────────────────────
echo "==> Checking data files…"
python3 -c "
from pathlib import Path
files = [
    'data/goldilocks_llama-3-8b.json',
    'data/goldilocks_qwen-2.5-7b.json',
    'data/profile_dataset_L1L2.json',
]
ok = True
for p in files:
    exists = Path(p).exists()
    print(f'  [{'OK' if exists else 'MISSING'}] {p}')
    if not exists: ok = False
if not ok:
    raise SystemExit('Missing data files — cannot start training')
"

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "==> Starting GRPO training on H100…"
python3 src/train_grpo.py \
    --model "$MODEL" \
    --max-steps 400

echo "==> Done. Checkpoints in checkpoints/$MODEL/"
