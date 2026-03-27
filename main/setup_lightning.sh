#!/usr/bin/env bash
# ============================================================================
# One-time setup script for Lightning AI Studio
# ============================================================================
# Run this ONCE after cloning the repo:
#     bash setup_lightning.sh
# ============================================================================
set -euo pipefail

echo "============================================"
echo "  Tiny-Math-Solver — Lightning AI Setup"
echo "============================================"

echo ""
echo ">>> [1/3] Installing Python dependencies ..."
pip install trl peft datasets wandb accelerate bitsandbytes

echo ""
echo ">>> [2/3] Pre-downloading model ..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen/Qwen2.5-1.5B-Instruct ...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('Done.')
"

echo ""
echo ">>> [3/3] Pre-downloading GSM8K dataset ..."
python3 -c "
from datasets import load_dataset
load_dataset('openai/gsm8k', 'main', split='train')
load_dataset('openai/gsm8k', 'main', split='test')
print('GSM8K (train + test) downloaded.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next step:"
echo "    WANDB_TOKEN=your_token bash src/train_math.sh"
echo "============================================"
