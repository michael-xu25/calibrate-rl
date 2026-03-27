"""
Honest baseline evaluation of Qwen 0.5B Instruct on GSM8K.

NO system prompt. NO format tricks. Just the raw model answering math
questions naturally. We log everything so we can analyze failure modes
and design RL rewards around reality.

Usage:
    python src/eval_baseline.py

Outputs:
    - Terminal: detailed samples + summary statistics
    - logs/baseline_eval.jsonl: full structured results for analysis
"""

import json
import os
import re
import random
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES = 100
MAX_NEW_TOKENS = 1024
SEED = 42

random.seed(SEED)


# ── Answer extraction (broad: tries many formats) ──────────────────────────

def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """
    Try to extract the model's final numerical answer.
    Returns (answer, method) where method describes how it was found.
    Tries formats in order of specificity.
    """
    # 1. #### <number> (GSM8K format)
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"

    # 2. \boxed{<number>}
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"

    # 3. "the answer is <number>" / "the final answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"

    # 4. **<number>** (bold, anywhere)
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"

    # 5. Last number on its own line (common for models that just state the answer)
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        match = re.fullmatch(r"\$?\s*([\d,]+(?:\.\d+)?)\s*\$?\s*\.?", line)
        if match:
            return match.group(1).replace(",", ""), "last_line_number"

    # 6. Last number in text (most aggressive fallback)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"

    return None, "no_answer"


# ── Load model and data ─────────────────────────────────────────────────────
print(f"Loading {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

print("Loading GSM8K (test split) ...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
indices = random.sample(range(len(dataset)), min(N_SAMPLES, len(dataset)))
print(f"Selected {len(indices)} questions from test set ({len(dataset)} total)\n")


# ── Evaluate ────────────────────────────────────────────────────────────────
results = []
correct = 0
parsed = 0
total = 0
format_counts = Counter()     # how the model formats answers
error_types = Counter()       # what goes wrong
token_lengths = []            # how long responses are

os.makedirs("logs", exist_ok=True)
jsonl_path = "logs/baseline_eval.jsonl"

print(f"{'='*70}")
print(f"  BASELINE EVALUATION: {MODEL}")
print(f"  {N_SAMPLES} GSM8K test questions, {MAX_NEW_TOKENS} max tokens, no system prompt")
print(f"{'='*70}\n")

with open(jsonl_path, "w") as f:
    for i, idx in enumerate(indices):
        question = dataset[idx]["question"]
        gold_label = dataset[idx]["answer"]
        gold = extract_gold_answer(gold_label)

        # NO system prompt -- just the user question
        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode only the generated part
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        n_tokens = len(gen_tokens)
        token_lengths.append(n_tokens)

        # Check if it hit the token limit (truncated)
        truncated = n_tokens >= MAX_NEW_TOKENS

        # Extract answer
        pred, method = extract_predicted_answer(response)
        format_counts[method] += 1

        # Numeric comparison (not string) -- "38.00" should match "38"
        def _nums_match(a, b):
            try:
                return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
            except (ValueError, TypeError, AttributeError):
                return False
        is_correct = gold is not None and pred is not None and _nums_match(pred, gold)
        if pred is not None:
            parsed += 1
        if is_correct:
            correct += 1

        # Categorize error
        if is_correct:
            error_type = "correct"
        elif pred is None:
            error_type = "no_parseable_answer"
        elif truncated:
            error_type = "truncated"
        else:
            error_type = "wrong_answer"
        error_types[error_type] += 1

        total += 1

        # Save structured result
        record = {
            "idx": idx,
            "question": question,
            "gold_label": gold_label,
            "gold_answer": gold,
            "response": response,
            "predicted_answer": pred,
            "extraction_method": method,
            "is_correct": is_correct,
            "error_type": error_type,
            "n_tokens": n_tokens,
            "truncated": truncated,
        }
        results.append(record)
        f.write(json.dumps(record) + "\n")

        # Print detailed output for first 25, then progress every 10
        if i < 25:
            status = "CORRECT" if is_correct else "WRONG  "
            formatted_response = response.replace('\n', '\n    ')
            print(f"[{i+1:3d}/{N_SAMPLES}] [{status}] gold={gold}  pred={pred}  "
                  f"method={method}  tokens={n_tokens}  truncated={truncated}")
            print(f"  Q: {question}")
            print(f"  A: {formatted_response}")
            print()
        elif (i + 1) % 10 == 0:
            print(f"[{i+1:3d}/{N_SAMPLES}] {correct}/{total} correct so far "
                  f"({100*correct/total:.1f}%)")


# ── Summary ─────────────────────────────────────────────────────────────────
avg_tokens = sum(token_lengths) / len(token_lengths)
median_tokens = sorted(token_lengths)[len(token_lengths) // 2]
max_tokens_seen = max(token_lengths)
min_tokens_seen = min(token_lengths)

print(f"\n{'='*70}")
print(f"  BASELINE RESULTS: {MODEL}")
print(f"{'='*70}")
print(f"  Questions evaluated:  {total}")
print(f"  Correct:              {correct}/{total} ({100*correct/total:.1f}%)")
print(f"  Answers parsed:       {parsed}/{total} ({100*parsed/total:.1f}%)")
print()
print(f"  --- Response Length (tokens) ---")
print(f"  Mean:    {avg_tokens:.0f}")
print(f"  Median:  {median_tokens}")
print(f"  Min:     {min_tokens_seen}")
print(f"  Max:     {max_tokens_seen}")
print(f"  Truncated (hit {MAX_NEW_TOKENS} limit): "
      f"{sum(1 for t in token_lengths if t >= MAX_NEW_TOKENS)}/{total}")
print()
print(f"  --- Answer Format Distribution ---")
for fmt, count in format_counts.most_common():
    print(f"  {fmt:25s}: {count:3d} ({100*count/total:.1f}%)")
print()
print(f"  --- Error Breakdown ---")
for err, count in error_types.most_common():
    print(f"  {err:25s}: {count:3d} ({100*count/total:.1f}%)")
print()
print(f"{'='*70}")
print(f"  Full results saved to: {jsonl_path}")
print(f"{'='*70}")
