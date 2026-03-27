"""
Build Sweet-Spot Training Dataset by Running Pass@16 on Train Set Sample.

Week 1 implementation: Identify goldilocks-zone problems (2-12/16 correct)
from GSM8K train set for calibrated RL training.

This directly measures which training problems are in the productive learning
zone for the model, replacing the entity-count heuristic.

Usage:
    python src/build_sweet_spot_dataset.py [--n_sample 500] [--skip_eval]

Outputs:
    - logs/train_pass_at_16.jsonl: pass@16 results for train set sample
    - data/sweet_spot_dataset/: filtered dataset in goldilocks zone
    - Terminal: difficulty distribution and comparison to entity filter
"""

import json
import os
import re
import random
import time
import argparse
from collections import Counter
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
K = 16                      # samples per problem
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 1024
SEED = 42

# Sweet spot bounds (from analysis in tiny-math-solver-breakdown.md)
MIN_CORRECT = 2   # problems with <2/16 are too hard (model rarely succeeds)
MAX_CORRECT = 12  # problems with >12/16 are too easy (ghost batching)

random.seed(SEED)


# ── Answer extraction (same as eval_pass_at_k.py) ──────────────────────────

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


def run_pass_at_k_on_problems(model, tokenizer, problems, k=16):
    """Run pass@k evaluation on a list of problems.

    Args:
        model: Loaded HF model
        tokenizer: Loaded HF tokenizer
        problems: List of dicts with 'question' and 'answer' keys
        k: Number of samples per problem

    Returns:
        List of dicts with pass@k results
    """
    results = []

    for prob_i, prob in enumerate(problems):
        question = prob["question"]
        gold_label = prob["answer"]
        gold = extract_gold_answer(gold_label)

        # Build prompt (no system prompt, matching baseline eval)
        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate K solutions
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Check each solution
        prompt_len = inputs["input_ids"].shape[1]
        n_correct = 0
        methods = Counter()

        for seq_i in range(k):
            gen_tokens = outputs[seq_i][prompt_len:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            pred, method = extract_predicted_answer(response)
            methods[method] += 1
            if nums_match(pred, gold):
                n_correct += 1

        pass_rate = n_correct / k

        # Store result
        record = {
            "idx": prob.get("idx", prob_i),  # use original idx if available
            "question": question,
            "gold_answer": gold,
            "n_correct": n_correct,
            "n_total": k,
            "pass_rate": round(pass_rate, 4),
            "methods_used": dict(methods),
        }
        results.append(record)

        # Progress
        bar = "█" * n_correct + "░" * (k - n_correct)
        emoji = "✓" if n_correct > 0 else "✗"
        print(f"[{prob_i+1:3d}/{len(problems)}] {emoji} {n_correct:2d}/{k} |{bar}| "
              f"gold={gold}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=500,
                        help="Number of train problems to evaluate")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, use existing logs/train_pass_at_16.jsonl")
    args = parser.parse_args()

    print("=" * 70)
    print("  Building Sweet-Spot Dataset from GSM8K Train Set")
    print(f"  Pass@{K} evaluation → filter to goldilocks zone ({MIN_CORRECT}-{MAX_CORRECT}/{K})")
    print("=" * 70)

    # Load GSM8K train split
    print("\n>>> Loading GSM8K train split ...")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"    Total training problems: {len(train_dataset)}")

    # Sample problems for evaluation
    sample_size = min(args.n_sample, len(train_dataset))
    sample_indices = random.sample(range(len(train_dataset)), sample_size)
    sample_problems = [
        {
            "idx": idx,
            "question": train_dataset[idx]["question"],
            "answer": train_dataset[idx]["answer"],
        }
        for idx in sample_indices
    ]
    print(f"    Sampled {sample_size} problems for pass@{K} evaluation")

    # Run pass@K evaluation (or load existing)
    os.makedirs("logs", exist_ok=True)
    results_path = "logs/train_pass_at_16.jsonl"

    if args.skip_eval and Path(results_path).exists():
        print(f"\n>>> Loading existing pass@{K} results from {results_path} ...")
        results = []
        with open(results_path, "r") as f:
            for line in f:
                results.append(json.loads(line))
        print(f"    Loaded {len(results)} results")
    else:
        print(f"\n>>> Running pass@{K} evaluation on {sample_size} train problems ...")
        print(f"    Model: {MODEL}")
        print(f"    This will take ~{sample_size * K * 0.5 / 60:.0f} minutes\n")

        # Load model
        print(f"Loading {MODEL} ...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", torch_dtype="auto"
        )
        print(f"Model loaded in {time.time() - t0:.1f}s\n")

        # Evaluate
        results = run_pass_at_k_on_problems(model, tokenizer, sample_problems, k=K)

        # Save results
        with open(results_path, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")
        print(f"\n>>> Saved pass@{K} results to {results_path}")

    # Analyze difficulty distribution
    print(f"\n>>> Difficulty distribution ({len(results)} problems):")
    difficulty_dist = Counter(r["n_correct"] for r in results)

    for n in range(K + 1):
        count = difficulty_dist.get(n, 0)
        pct = 100 * count / len(results)
        bar = "█" * (count // 5 if len(results) > 50 else count)
        in_sweet_spot = "  ← SWEET SPOT" if MIN_CORRECT <= n <= MAX_CORRECT else ""
        print(f"    {n:2d}/{K} correct: {count:3d} ({pct:4.1f}%)  {bar}{in_sweet_spot}")

    # Count problems in each zone
    too_hard = sum(1 for r in results if r["n_correct"] < MIN_CORRECT)
    sweet_spot_count = sum(1 for r in results
                           if MIN_CORRECT <= r["n_correct"] <= MAX_CORRECT)
    too_easy = sum(1 for r in results if r["n_correct"] > MAX_CORRECT)

    print(f"\n>>> Training set breakdown:")
    print(f"    Too hard (0-{MIN_CORRECT-1}/{K}):   {too_hard:3d} ({100*too_hard/len(results):.0f}%)")
    print(f"    Sweet spot ({MIN_CORRECT}-{MAX_CORRECT}/{K}): {sweet_spot_count:3d} ({100*sweet_spot_count/len(results):.0f}%)")
    print(f"    Too easy ({MAX_CORRECT+1}-{K}/{K}):  {too_easy:3d} ({100*too_easy/len(results):.0f}%)")

    # Filter to sweet spot
    sweet_spot_indices = [
        r["idx"] for r in results
        if MIN_CORRECT <= r["n_correct"] <= MAX_CORRECT
    ]

    print(f"\n>>> Creating sweet-spot dataset ...")
    print(f"    Filtering full train set ({len(train_dataset)} problems)")
    print(f"    Keeping {len(sweet_spot_indices)} measured sweet-spot problems")

    # Create filtered dataset
    sweet_spot_data = train_dataset.select(sweet_spot_indices)

    # Add pass@K metadata
    pass_at_k_map = {r["idx"]: r for r in results}

    def add_pass_at_k_metadata(example, idx):
        original_idx = sweet_spot_indices[idx]
        pass_data = pass_at_k_map.get(original_idx, {})
        example["pass_at_16_correct"] = pass_data.get("n_correct", -1)
        example["pass_at_16_rate"] = pass_data.get("pass_rate", -1.0)
        return example

    sweet_spot_data = sweet_spot_data.map(add_pass_at_k_metadata, with_indices=True)

    # Save dataset
    output_path = "data/sweet_spot_dataset"
    print(f"\n>>> Saving to {output_path} ...")
    sweet_spot_data.save_to_disk(output_path)
    print(f"    Saved {len(sweet_spot_data)} problems")

    # Compare to entity filter
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Sweet Spot vs Entity Filter")
    print(f"{'=' * 70}")

    # Load entity dataset if it exists
    entity_path = "data/entity_tracking_dataset"
    if Path(entity_path).exists():
        from datasets import load_from_disk
        entity_dataset = load_from_disk(entity_path)
        print(f"\n  Entity filter (3+ entities):")
        print(f"    Dataset size: {len(entity_dataset)} problems")
        print(f"    Approach: Heuristic (regex entity counting)")
        print(f"    Target: Entity tracking errors (~10/55 failures)")
    else:
        print(f"\n  Entity filter: Not found at {entity_path}")

    print(f"\n  Sweet spot filter ({MIN_CORRECT}-{MAX_CORRECT}/{K} correct):")
    print(f"    Dataset size: {len(sweet_spot_data)} problems")
    print(f"    Approach: Direct measurement (pass@{K})")
    print(f"    Target: Goldilocks zone (within-group variance for GRPO)")
    print(f"    Expected ghost batch reduction: 40% → <20% (estimated)")

    print(f"\n{'=' * 70}")
    print(f"  DONE - Sweet-Spot Dataset Created")
    print(f"{'=' * 70}")
    print(f"  Output: {output_path}/")
    print(f"  Pass@{K} results: {results_path}")
    print(f"\n  NEXT STEP: Update train.py to use sweet_spot_dataset instead of")
    print(f"             entity_tracking_dataset for GRPO training")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
