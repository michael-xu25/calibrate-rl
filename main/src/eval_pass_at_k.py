"""
Pass@K Capability Test for Qwen2.5-3B-Instruct on GSM8K.

For each problem, generate K solutions at temperature 0.7.
This tells us whether the model *can* solve the problem (knowledge exists
in the weights) vs *reliably* solves it (policy is correct).

If pass@16 is high but pass@1 is low → RL can fix the policy.
If pass@16 is also low → need better SFT first.

Usage:
    python src/eval_pass_at_k.py

Outputs:
    - Terminal: per-problem results + summary + histogram + verdict
    - logs/pass_at_k.jsonl: structured results for analysis
"""

import json
import os
import re
import random
import time
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROBLEMS = 100
K = 16                  # samples per problem
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 1024
SEED = 42               # same seed as baseline for same test questions

random.seed(SEED)


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """Extract the model's final answer. Returns (answer, method)."""
    # 1. \boxed{<number>}
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"

    # 2. #### <number>
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"

    # 3. "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"

    # 4. **<number>**
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"

    # 5. Last number on its own line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        match = re.fullmatch(r"\$?\s*([\d,]+(?:\.\d+)?)\s*\$?\s*\.?", line)
        if match:
            return match.group(1).replace(",", ""), "last_line_number"

    # 6. Last number in text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"

    return None, "no_answer"


def nums_match(a: str | None, b: str | None) -> bool:
    """Numeric comparison: '38.00' == '38'."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return False


# ── Load model and data ────────────────────────────────────────────────────
print(f"Loading {MODEL} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="auto"
)
print(f"Model loaded in {time.time() - t0:.1f}s\n")

print("Loading GSM8K (test split) ...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
indices = random.sample(range(len(dataset)), min(N_PROBLEMS, len(dataset)))
print(f"Selected {len(indices)} questions (same seed={SEED} as baseline)\n")


# ── Run Pass@K ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
jsonl_path = "logs/pass_at_k.jsonl"

print(f"{'='*70}")
print(f"  PASS@{K} TEST: {MODEL}")
print(f"  {N_PROBLEMS} problems × {K} samples @ temp={TEMPERATURE}")
print(f"{'='*70}\n")

all_results = []
total_correct_at_1 = 0  # sum of (n_correct/K) for pass@1 estimate
total_pass_at_k = 0     # problems with at least 1 correct

with open(jsonl_path, "w") as f:
    for prob_i, idx in enumerate(indices):
        question = dataset[idx]["question"]
        gold_label = dataset[idx]["answer"]
        gold = extract_gold_answer(gold_label)

        # Build prompt (no system prompt)
        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate K solutions
        t_gen = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=K,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_time = time.time() - t_gen

        # Decode and check each solution
        prompt_len = inputs["input_ids"].shape[1]
        n_correct = 0
        methods = Counter()
        token_counts = []
        correct_samples = []   # ALL correct responses (full text)
        wrong_samples = []     # ALL wrong responses (full text)
        predictions = []

        for seq_i in range(K):
            gen_tokens = outputs[seq_i][prompt_len:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            n_tokens = len(gen_tokens)
            token_counts.append(n_tokens)

            pred, method = extract_predicted_answer(response)
            methods[method] += 1
            is_correct = nums_match(pred, gold)
            predictions.append({
                "pred": pred,
                "method": method,
                "correct": is_correct,
                "n_tokens": n_tokens,
            })

            if is_correct:
                n_correct += 1
                correct_samples.append(response)
            else:
                wrong_samples.append(response)

        pass_rate = n_correct / K
        total_correct_at_1 += pass_rate
        if n_correct > 0:
            total_pass_at_k += 1

        # Save structured result (full samples for downstream SFT extraction)
        record = {
            "idx": idx,
            "question": question,
            "gold_label": gold_label,
            "gold_answer": gold,
            "n_correct": n_correct,
            "n_total": K,
            "pass_rate": round(pass_rate, 4),
            "methods_used": dict(methods),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
            "correct_samples": correct_samples,   # full text, all correct
            "wrong_samples": wrong_samples,        # full text, all wrong
        }
        all_results.append(record)
        f.write(json.dumps(record) + "\n")

        # Progress output
        bar = "█" * n_correct + "░" * (K - n_correct)
        emoji = "✓" if n_correct > 0 else "✗"
        print(f"[{prob_i+1:3d}/{N_PROBLEMS}] {emoji} {n_correct:2d}/{K} correct  "
              f"|{bar}|  gold={gold}  ({gen_time:.1f}s)")

        # Show details for first 10 problems
        if prob_i < 10:
            q_short = question[:120] + ("..." if len(question) > 120 else "")
            print(f"         Q: {q_short}")
            if correct_samples:
                c_short = correct_samples[0][:200].replace('\n', ' | ')
                print(f"         Correct: {c_short}...")
            if wrong_samples and n_correct < K:
                w_short = wrong_samples[0][:200].replace('\n', ' | ')
                print(f"         Wrong:   {w_short}...")
            print()


# ── Summary statistics ──────────────────────────────────────────────────────
est_pass_at_1 = total_correct_at_1 / N_PROBLEMS
pass_at_k = total_pass_at_k / N_PROBLEMS

# Difficulty histogram
buckets = {"0/16 (hopeless)": 0, "1-2/16 (hard)": 0, "3-5/16 (medium)": 0,
           "6-10/16 (learnable)": 0, "11-16/16 (easy)": 0}
for r in all_results:
    nc = r["n_correct"]
    if nc == 0:
        buckets["0/16 (hopeless)"] += 1
    elif nc <= 2:
        buckets["1-2/16 (hard)"] += 1
    elif nc <= 5:
        buckets["3-5/16 (medium)"] += 1
    elif nc <= 10:
        buckets["6-10/16 (learnable)"] += 1
    else:
        buckets["11-16/16 (easy)"] += 1

# Detailed distribution
dist = Counter(r["n_correct"] for r in all_results)

print(f"\n{'='*70}")
print(f"  PASS@{K} RESULTS: {MODEL}")
print(f"{'='*70}")
print(f"  Problems evaluated:     {N_PROBLEMS}")
print(f"  Samples per problem:    {K}")
print(f"  Temperature:            {TEMPERATURE}")
print()
print(f"  ┌─────────────────────────────────────┐")
print(f"  │  Estimated pass@1:   {est_pass_at_1*100:5.1f}%           │")
print(f"  │  Pass@{K}:            {pass_at_k*100:5.1f}%           │")
print(f"  └─────────────────────────────────────┘")
print()
print(f"  --- Difficulty Histogram ---")
for bucket, count in buckets.items():
    bar = "█" * count
    print(f"  {bucket:25s}: {count:3d}  {bar}")
print()
print(f"  --- Detailed Distribution (correct out of {K}) ---")
for k_val in range(K + 1):
    count = dist.get(k_val, 0)
    bar = "█" * count
    print(f"  {k_val:2d}/{K}: {count:3d}  {bar}")
print()

# RL Readiness Verdict
hopeless = buckets["0/16 (hopeless)"]
rl_ready = (buckets["1-2/16 (hard)"] + buckets["3-5/16 (medium)"] +
            buckets["6-10/16 (learnable)"])
easy = buckets["11-16/16 (easy)"]

print(f"  --- RL READINESS VERDICT ---")
print(f"  Hopeless (0/{K}, RL can't help):  {hopeless}/{N_PROBLEMS} ({100*hopeless/N_PROBLEMS:.0f}%)")
print(f"  RL sweet spot (1-10/{K}):         {rl_ready}/{N_PROBLEMS} ({100*rl_ready/N_PROBLEMS:.0f}%)")
print(f"  Already easy (11+/{K}):           {easy}/{N_PROBLEMS} ({100*easy/N_PROBLEMS:.0f}%)")
print()

if pass_at_k < 0.10:
    print(f"  ⚠ VERDICT: pass@{K} < 10%. Model lacks capability.")
    print(f"  → Do NOT proceed with RL. Improve SFT with shorter CoT first.")
elif est_pass_at_1 > 0.80:
    print(f"  ⚠ VERDICT: pass@1 > 80%. Model is already strong.")
    print(f"  → RL will have marginal gains. Consider harder dataset.")
elif hopeless > N_PROBLEMS * 0.5:
    print(f"  ⚠ VERDICT: >50% of problems are hopeless (0/{K}).")
    print(f"  → RL signal will be sparse. Consider SFT first.")
else:
    print(f"  ✓ VERDICT: Model has latent capability. RL can improve policy.")
    print(f"  → Proceed with GRPO training.")
    print(f"  → {rl_ready} problems in RL sweet spot = good training signal.")

print()
print(f"{'='*70}")
print(f"  Full results saved to: {jsonl_path}")
print(f"{'='*70}")
