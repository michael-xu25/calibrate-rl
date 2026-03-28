"""
Step 2: Evaluation Harness for CalibrateRL Model Profiling

Design decisions:
- 8 rollouts per problem: gives pass rates [0, 0.125, 0.25 ... 1.0] — enough
  resolution for Z-tests while halving API calls vs. 16.
- Per-problem concurrency: all 8 rollouts for a problem fire simultaneously,
  then we sleep INTER_PROBLEM_DELAY seconds before the next problem. This gives
  OpenRouter time to recover between bursts and eliminates throttling.
- Three-tier answer extraction: <answer> tag → \boxed{} → last number fallback.

Environment variables:
    API_KEY   — OpenRouter / Together AI / OpenAI key
    BASE_URL  — e.g. https://openrouter.ai/api/v1

Usage:
    python src/run_evaluation.py            # full run
    python src/run_evaluation.py --dry-run  # validate without API calls
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = Path(os.environ.get("EVAL_DATASET", "data/profile_dataset.json"))
OUTPUT_PATH  = Path(os.environ.get("EVAL_OUTPUT",  "data/evaluation_results.json"))

MODELS = {
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "llama-3-8b":  "meta-llama/llama-3-8b-instruct",
}

N_ROLLOUTS           = 8      # per problem; 9 distinct pass-rate values
TEMPERATURE          = 0.8
MAX_TOKENS           = 512
INTER_PROBLEM_DELAY  = 3.0    # seconds to sleep between problems (rate-limit pacing)
MAX_RETRIES          = 4
BASE_BACKOFF_SEC     = 2.0
CALL_TIMEOUT_SEC     = 30.0   # httpx-level timeout — properly closes the TCP socket

# For ETA: rolling average over this many recent problems
ETA_WINDOW = 15

SYSTEM_PROMPT = "You are a concise mathematical reasoning assistant. Solve problems step by step, then give your final answer in <answer> tags."

ANSWER_REMINDER = "\n\nFinal answer must be last, inside <answer> tags — e.g. <answer>42</answer>"


# ── Reward function ────────────────────────────────────────────────────────────

def extract_model_answer(text: str) -> Optional[str]:
    """
    Three-tier extraction:
      1. <answer>...</answer> tag
      2. \\boxed{...} (natural LaTeX)
      3. Last standalone number/fraction (simple numeric answers only)
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()

    candidates = re.findall(r"(?<![/\w])(-?\d+(?:[./]\d+)?)(?![/\w])", text)
    if candidates:
        return candidates[-1]

    return None


def normalize_answer(raw: str) -> str:
    s = raw.strip().strip("$")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right)[(\[{)\]}|.]", "", s)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 → 1000
    s = re.sub(r"^(-?\d+)\.0+$", r"\1", s)          # 5.0 → 5
    return s.lower().strip()


def compute_reward(model_output: str, ground_truth: str) -> int:
    extracted = extract_model_answer(model_output)
    if extracted is None:
        return 0
    return int(normalize_answer(extracted) == normalize_answer(ground_truth))


# ── Async inference ────────────────────────────────────────────────────────────

async def single_call(
    client: AsyncOpenAI,
    model_id: str,
    problem_text: str,
) -> str:
    """One API call with exponential-backoff retry. Returns completion text or ''."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem_text + ANSWER_REMINDER},
    ]
    backoff = BASE_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e).lower()
            retriable = "rate" in err or "429" in err or any(
                c in err for c in ("500", "502", "503", "timeout", "timed out", "connect")
            )
            if attempt == MAX_RETRIES or not retriable:
                print(f"    [FAIL] {type(e).__name__}: {e}", flush=True)
                return ""
            wait = backoff * (2 ** (attempt - 1))
            print(f"    [RETRY {attempt}] waiting {wait:.0f}s…", flush=True)
            await asyncio.sleep(wait)
    return ""


async def evaluate_problem(
    client: AsyncOpenAI,
    model_id: str,
    problem: dict,
) -> dict:
    """
    Fire all N_ROLLOUTS calls concurrently for one problem.
    No global semaphore — concurrency is bounded naturally by N_ROLLOUTS (8).
    """
    # return_exceptions=True means one failed/timed-out call doesn't
    # cancel the other 7 — we just score it as an empty string (reward=0).
    raw = await asyncio.gather(*[
        single_call(client, model_id, problem["problem"])
        for _ in range(N_ROLLOUTS)
    ], return_exceptions=True)
    outputs = [r if isinstance(r, str) else "" for r in raw]
    rewards   = [compute_reward(out, problem["answer"]) for out in outputs]
    pass_rate = sum(rewards) / N_ROLLOUTS
    return {
        "pass_rate":           pass_rate,
        "max_score":           max(rewards),
        "advantage_estimates": [r - pass_rate for r in rewards],
        "rollout_rewards":     rewards,
        "rollout_outputs":     list(outputs),
    }


# ── Progress ───────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def checkpoint_save(results: dict) -> None:
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(OUTPUT_PATH)


# ── Evaluation loop ────────────────────────────────────────────────────────────

async def run_model(
    client: AsyncOpenAI,
    model_key: str,
    model_id: str,
    problems: list[dict],
    results: dict,
) -> None:
    total = len(problems)
    print(f"\n{'='*58}")
    print(f"  Model : {model_key}  ({model_id})")
    print(f"  Problems: {total}  |  Rollouts: {N_ROLLOUTS}  |  "
          f"API calls: {total * N_ROLLOUTS:,}")
    print(f"  Pacing: {N_ROLLOUTS} concurrent per problem, "
          f"{INTER_PROBLEM_DELAY}s between problems")
    print(f"{'='*58}")

    done = skipped = 0
    recent_times: deque = deque(maxlen=ETA_WINDOW)

    for problem in problems:
        pid = problem["id"]

        if pid in results and model_key in results[pid].get("models", {}):
            skipped += 1
            done += 1
            continue

        t0 = time.monotonic()
        metrics = await evaluate_problem(client, model_id, problem)
        elapsed = time.monotonic() - t0

        done += 1
        recent_times.append(elapsed)

        if pid not in results:
            results[pid] = {
                "id":      pid,
                "subject": problem["subject"],
                "level":   problem["level"],
                "problem": problem["problem"],
                "answer":  problem["answer"],
                "models":  {},
            }
        results[pid]["models"][model_key] = metrics
        checkpoint_save(results)

        avg_t  = sum(recent_times) / len(recent_times)
        remaining = total - done
        eta_min = (remaining * (avg_t + INTER_PROBLEM_DELAY)) / 60

        print(
            f"  {_bar(done, total)}  "
            f"pass={metrics['pass_rate']:.2f}  "
            f"ETA {eta_min:.1f}m",
            flush=True,
        )

        if remaining > 0:
            await asyncio.sleep(INTER_PROBLEM_DELAY)

    if skipped:
        print(f"  (skipped {skipped} already-evaluated problems)")

    evaluated = [
        r["models"][model_key]["pass_rate"]
        for r in results.values()
        if model_key in r.get("models", {})
    ]
    print(f"\n  Avg pass rate: {sum(evaluated)/len(evaluated):.3f}")


# ── Dry run ────────────────────────────────────────────────────────────────────

def dry_run(problems: list[dict]) -> None:
    print("\n" + "="*58)
    print("  DRY-RUN — no API calls")
    print("="*58)

    for p in problems[:2]:
        print(f"\n  [{p['id']}]  subject={p['subject']}  answer={p['answer']!r}")
        user_msg = p["problem"] + ANSWER_REMINDER
        print(f"  user msg ({len(user_msg)} chars): {user_msg[:160]}…")

    print("\n  Reward function tests:")
    cases = [
        ("The answer is \\boxed{42}.",       "42",  1, "boxed fallback"),
        ("<answer>3/4</answer>",              "3/4", 1, "tag match"),
        ("<answer>42.0</answer>",             "42",  1, "trailing .0"),
        ("<answer>1,000</answer>",            "1000",1, "comma stripping"),
        ("So the result is 7.",               "7",   1, "last-number fallback"),
        ("<answer>wrong</answer>",            "42",  0, "wrong answer"),
        ("no answer here",                    "42",  0, "no extraction → 0"),
        ("<answer>$\\frac{1}{2}$</answer>",   "1/2", 0, "LaTeX fraction — known limitation"),
    ]
    ok = True
    for output, truth, expected, label in cases:
        got = compute_reward(output, truth)
        status = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{status}] {label}")
    print(f"\n  Result: {'ALL PASS ✓' if ok else 'SOME FAILED ✗'}")

    n_calls = len(problems) * N_ROLLOUTS * len(MODELS)
    est_min = len(problems) * (5 + INTER_PROBLEM_DELAY) * len(MODELS) / 60
    print(f"\n  Full run estimate: {n_calls} API calls, ~{est_min:.0f} min")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    if not DATASET_PATH.exists():
        sys.exit(f"Dataset not found: {DATASET_PATH}. Run src/build_profile_dataset.py first.")

    with open(DATASET_PATH) as f:
        problems: list[dict] = json.load(f)
    print(f"Loaded {len(problems)} problems from {DATASET_PATH}")

    if args.dry_run:
        dry_run(problems)
        return

    api_key  = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    if not api_key:
        sys.exit("Set API_KEY env var. E.g.: export API_KEY=sk-or-v1-...")
    if not base_url:
        sys.exit("Set BASE_URL env var. E.g.: export BASE_URL=https://openrouter.ai/api/v1")

    # httpx.Timeout sets socket-level timeouts so hung connections are
    # properly closed rather than abandoned (unlike asyncio.wait_for).
    # connect=5s: fail fast if the server is unreachable.
    # read=CALL_TIMEOUT_SEC: kill slow responses before they hang forever.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(CALL_TIMEOUT_SEC, connect=5.0),
    )

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            results: dict = json.load(f)
        print(f"Resuming from checkpoint ({len(results)} problems saved)")
    else:
        results = {}
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected = {k: MODELS[k] for k in args.models}
    for model_key, model_id in selected.items():
        await run_model(client, model_key, model_id, problems, results)

    print(f"\nDone. Results → {OUTPUT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                   default=list(MODELS.keys()))
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
