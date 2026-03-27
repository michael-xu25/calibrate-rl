# Tiny-Math-Solver: What I Did and What I Learned

A complete breakdown of my first RL training project — training Qwen2.5-1.5B-Instruct on GSM8K using GRPO with rule-based rewards. Every line of code was written by Cursor; this document records what the code actually does, what worked, what didn't, and what I'd do differently.

---

## The Pipeline (4 Phases)

### Phase 1: Baseline Evaluation

**What I did:** Ran greedy (temp=0) evaluation on 100 random GSM8K test problems to establish a baseline. No system prompt, no special formatting — just the raw model answering math problems.

**Result:** 67.7% pass@1. The model is decent at grade school math out of the box but makes mistakes on roughly 1 in 3 problems.

**Code:** `src/eval_baseline.py` — straightforward: load model, generate one completion per problem, extract answer, compare to gold.

### Phase 2: Capability Assessment (Pass@16)

**What I did:** Sampled 16 completions per problem at temp=0.7 to measure how often the model *can* get each problem right if given multiple tries. This uses the unbiased pass@k estimator from the Codex paper.

**Result:** pass@1 = 67.7% (consistent with greedy), pass@16 = 95%. The model has the latent capability to solve 95% of problems — it just doesn't do so reliably on the first try.

**Code:** `src/eval_pass_at_k.py` — generates 16 samples, extracts answers with a 6-step fallback chain (boxed → #### → "the answer is" → bold → last line number → last number fallback), computes pass@k.

**Answer extraction verified:** After auditing all extraction methods against real log data, the numbers are trustworthy. The `last_number_fallback` correctly identified answers ~98.8% of the time. The `\boxed{}` regex has a minor bug (can't handle `\boxed{-5}` or `\boxed{$50}`) but affected very few problems in practice (2 confirmed cases).

### Phase 3: Failure Analysis and Dataset Curation

**What I did:** Analyzed *why* the model fails when it does, using both automated heuristics (`src/analyze_gaps.py`) and manual/LLM categorization (`analysis/pass_at_16_breakdown.md`).

**Failure categories identified:**
- Misread referent/modifier (~25 cases) — confusing "each" vs "total," "$6000 each" scope
- Entity tracking (~10 cases) — forgetting a person or constraint mid-problem
- Directional errors (~8 cases) — subtracting when should add, "faster" = less time
- Nested structure (~5 cases) — "40% of the remaining" after taking 1/3
- Optimization/multi-part (~4 cases) — comparing options, conditional pricing
- Factual knowledge (~3 cases) — days in May, alphabet length

**Targeting decision:** Chose to focus on entity tracking (3+ named entities) using a regex-based entity extractor (`src/build_entity_dataset.py`). This filtered GSM8K's ~7,400 training problems down to a curated subset.

**In hindsight:** Entity tracking was only ~10/55 of the identified failures. Misread referent/modifier was 2.5x more common. The analysis even suggested 4 different targeted datasets, but training only used one. A difficulty-based filter (problems with 2-12/16 pass rate — the "sweet spot") would have been more principled and directly targeted what GRPO needs: within-group variance.

### Phase 4: GRPO Training

**What I did:** Trained with TRL's GRPOTrainer using the following setup:

**Model:** Qwen2.5-1.5B-Instruct with LoRA (rank 16, alpha 32, dropout 0.05, all-linear targets). LoRA means only ~0.5% of parameters are trainable — the rest of the model is frozen. This is parameter-efficient fine-tuning: cheaper, faster, and less risk of catastrophic forgetting than full fine-tuning.

**GRPO algorithm:** For each training prompt, generate 8 completions at temperature 1.0. Score each completion. Compute advantages relative to the group mean. Update the policy to increase probability of above-average completions and decrease probability of below-average ones. This is the core of GRPO — it learns from the *contrast* between good and bad completions for the same problem, without needing a separate reward model.

**System prompt:** Forces structured reasoning — `"Think step by step inside <think> tags before answering. For each person or item in the problem, explicitly state what you know about them and calculate their value."` This creates more diverse completions for GRPO to learn from.

**Reward design:**
- Correctness reward: 1.0 if final extracted answer matches gold, 0.0 otherwise (binary)
- Format reward: 0.1 if `\boxed{}` is present, 0.0 otherwise (small nudge for answer formatting)
- No intermediate step reward — tried it but dropped it because all completions produced the same intermediate numbers, adding no within-group variance

**Ghost-batch mitigation:** With entity-filtered data, ~60% of prompt groups produce zero gradient (all 8 completions correct or all wrong). To compensate:
- Large effective batch: 4 per device × 16 gradient accumulation = 64 prompts per optimizer step, so ~25+ prompts produce useful gradient even at 40% ghost rate
- KL penalty (beta=0.04): prevents policy drift from noisy updates. This is 40x higher than DeepSeek-R1's 0.001, justified by noisier signal
- DAPO loss normalization: normalizes by active tokens in global batch, eliminating length bias
- Truncation masking: excludes cut-off completions from loss

**No vLLM:** Tried it but hit known convergence bugs with PEFT adapters (trl#2856, vllm#14483). Paged attention exceeded cache block limits at batch size 128×1024 tokens. Vanilla `model.generate()` on L40S (48GB) was fast enough (~5GB KV cache).

**Training schedule:** Planned 500 steps with cosine LR schedule (5e-5 peak, 3% warmup). Checkpoints every 50 steps.

---

## Training Dynamics (from train.log)

The run reached step 324/500 before being killed at ~5h10m. But the important story is in the first 50 steps:

| Phase | Correctness | Format | Ghost % | KL | Entropy |
|-------|------------|--------|---------|-----|---------|
| Steps 1-10 | 57.8% | 0.010 | 18.8% | 0.0002 | 0.413 |
| Steps 11-50 | 81.4% | 0.091 | 41.6% | 0.019 | 0.223 |
| Steps 51-100 | 77.6% | 0.096 | 39.5% | 0.018 | 0.292 |
| Steps 101-200 | 79.5% | 0.095 | 41.5% | 0.017 | 0.271 |
| Steps 201-324 | 80.9% | 0.095 | 41.4% | 0.019 | 0.284 |

**Key observations:**
- Correctness jumped from 58% to 81% in the first ~15 steps, then flatlined for 300+ more steps
- Format reward was learned almost immediately (0.01 → 0.09 by step 15, indicating `\boxed{}` adoption)
- Ghost batching averaged 40-42% throughout — nearly half of all compute produced zero gradient
- KL stabilized at ~0.019 — the model moved away from the base policy early and then stopped drifting (beta=0.04 anchoring hard)
- Entropy dropped (0.41 → 0.22) as model converged on a reasoning style, then partially recovered

**42.9% of all steps had ≥50% ghost batches.** 7.4% had ≥75% ghost. This is the biggest inefficiency in the training.

---

## Results

Only checkpoint-50 was evaluated (and rightfully so — the metrics show no improvement after step ~15).

| Metric | Baseline | Checkpoint-50 | Delta |
|--------|----------|---------------|-------|
| Greedy pass@1 | 67.7% | 77.0% | +9.3% |
| Pass@16 pass@1 | 67.7% | 72.0% | +4.3% |
| Pass@16 | 95% | 98% | +3.0% |
| `\boxed{}` usage | 74% | 95% | +21% |

**Entity breakdown (100 test problems, pass@16 comparison):**

| Entities | Count | Baseline | Checkpoint | Delta |
|----------|-------|----------|------------|-------|
| 0 | 13 | 72.1% | 63.5% | **-8.7%** |
| 1 | 60 | 63.9% | 70.3% | +6.5% |
| 2 | 17 | 72.4% | 79.8% | +7.4% |
| 3 | 4 | 76.6% | 81.2% | +4.7% |
| 4+ | 6 | 77.1% | 79.2% | +2.1% |

Win ratio: 64.9% (50 improved, 27 regressed, 23 unchanged).

---

## What Worked

1. **GRPO + binary reward works.** +9.3% greedy pass@1 from 50 steps of RL on a curated subset is a real, meaningful improvement.
2. **Format reward succeeded completely.** `\boxed{}` usage jumped from 74% → 95%. The model learned the formatting incentive almost instantly.
3. **The ghost-batch mitigations were well-reasoned.** Large effective batch + KL anchor + DAPO loss is the right toolkit for noisy RL signal.
4. **LoRA was the right call.** Fast training, small checkpoint, and the base model's general capability was mostly preserved.

## What Didn't Work / What to Improve

1. **Entity count was the wrong filter.** The biggest failure mode was misread referents/modifiers, not entity tracking. A difficulty-based filter (pass@16 sweet spot: 2-12/16 correct) would have directly targeted what GRPO needs: problems with high within-group variance.
2. **Binary reward + ghost batching is a compounding problem.** Binary 0/1 reward maximizes ghost batching because there's no partial credit. If every completion either fully succeeds or fully fails, groups are more likely to be unanimous. Process reward (partial credit for intermediate steps) or soft correctness scoring could reduce ghost rate.
3. **0-entity regression.** The system prompt forcing entity-by-entity reasoning hurts on simple arithmetic problems. The training made the model *always* reason in this style, even when it's unnecessary overhead.
4. **Training plateu at step ~15.** 300+ steps of additional training produced no measurable improvement. Earlier stopping or more aggressive exploration (lower beta) could have saved compute.
5. **KL beta may be too conservative.** 0.04 vs DeepSeek-R1's 0.001. The KL plateau at 0.019 suggests the model can't explore far enough from the base policy.
6. **3 and 4+ entity test groups are too small** (4 and 6 problems) to draw conclusions about targeted improvement. Need larger eval set.
7. **Eval/reward extraction mismatch.** `eval_pass_at_k.py` has a 6-step extraction chain but `reward_func.py` has a 5-step chain (missing "last line number" step). This means training and evaluation may disagree on whether an answer was extracted correctly.

---

## Key Lesson for CalibrateRL

This project is actually a proof-of-concept for the CalibrateRL thesis: **the most impactful thing you can do in RL training is get the difficulty calibration right.** The entity filter was a rough proxy for difficulty. A direct measurement of per-problem difficulty (via pass@k) and training on the sweet spot would have been more principled, reduced ghost batching, and probably produced better results with less compute. That's the core CalibrateRL insight — adaptive environments calibrated to model weaknesses.

## Known Bugs to Fix in Next Iteration

1. **`\boxed{}` regex too strict:** Can't parse `\boxed{-5}`, `\boxed{$50}`, `\boxed{20%}`. Needs a more permissive inner pattern.
2. **Eval vs reward extraction mismatch:** `eval_pass_at_k.py` uses 6-step extraction, `reward_func.py` uses 5-step. Should be identical so training and evaluation agree.
3. **Multiple `\boxed{}` takes first not last:** `re.search()` grabs the first match. If the model self-corrects with a second `\boxed{}`, the wrong answer is used. Should use `re.findall()[-1]`.

## Connection to CalibrateRL Roadmap

See `cowork_brief.md` in this folder for the full 4-week startup roadmap. The three core ideas:
1. **Model-calibrated environments** — this project proved the need: entity filtering was a blunt proxy, difficulty-based filtering via pass@k is the principled version
2. **Dynamic updates** — this project showed the model plateaus at step ~15, meaning the training data should be re-calibrated as the model improves
3. **Predictive quality metrics** — ghost batch fraction (40%+ here) is a direct signal that training data is miscalibrated; tracking it in real-time could trigger automatic difficulty adjustment

The source code and logs live in `rl-intro copy/` in this folder. The coding puzzle project (`RL_Coding copy/`) is a separate evaluation pipeline, not the GSM8K training.
