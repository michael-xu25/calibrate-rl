# Tiny-Math-Solver

Train **Qwen/Qwen2.5-1.5B-Instruct** to solve grade-school math problems (GSM8K) using **GRPO** with a **rule-based reward function** -- powered by [TRL](https://github.com/huggingface/trl).

No trained reward model needed. The reward function checks whether the model's numerical answer matches the ground truth.

## Quick Start

### 1. Clone on Lightning AI Studio

Open a **Lightning AI Studio** with an H200 GPU, then:

```bash
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro
```

### 2. Run setup (once)

```bash
bash setup_lightning.sh
```

### 3. Run Pass@16 capability test (before training)

```bash
python src/eval_pass_at_k.py
```

This generates 16 solutions per problem to determine if the model has latent reasoning capability that RL can reinforce.

### 4. Run training

```bash
bash src/train_math.sh
```

With Weights & Biases logging:

```bash
WANDB_TOKEN=your_token bash src/train_math.sh
```

## Project Structure

```
rl-intro/
├── README.md
├── setup_lightning.sh          # One-time setup (pip install trl peft ...)
├── .gitignore
└── src/
    ├── train_math.sh           # Shell wrapper
    ├── train_grpo.py           # Training script (TRL GRPOTrainer)
    ├── reward_func.py          # Rule-based rewards (answer checker)
    ├── eval_baseline.py        # Greedy baseline evaluation
    └── eval_pass_at_k.py       # Pass@K capability test
```

## How It Works

```
GSM8K question
    │
    ▼
Actor (Qwen 1.5B + LoRA)
    │  generates 8 solutions per question
    ▼
reward_func.py
    ├── correctness_reward: 1.0 if answer matches gold, else 0.0
    └── reasoning_reward:   partial credit for correct intermediate steps
    │
    ▼
GRPO advantage = normalize rewards across the 8 samples
    │
    ▼
Update LoRA adapter
```

## Configuration

Edit `src/train_grpo.py` to tune:

| Parameter | Default | Notes |
|---|---|---|
| `r` (LoRA rank) | 16 | Higher = more capacity, more memory |
| `num_generations` | 8 | Samples per prompt for GRPO |
| `per_device_train_batch_size` | 4 | Prompts per micro-batch |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 prompts |
| `learning_rate` | 1e-4 | LoRA learning rate |
| `temperature` | 0.9 | Generation temperature |
| `max_completion_length` | 1024 | Max tokens per completion |

## References

- [TRL GRPO Trainer Docs](https://huggingface.co/docs/trl/main/grpo_trainer)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [GRPO Paper (DeepSeekMath)](https://huggingface.co/papers/2402.03300)
