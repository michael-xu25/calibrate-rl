# Startup Brief: CalibrateRL — Adaptive RL Environment Creation

## Context for Opus (Cowork)

This brief captures a full strategic planning session. The founders are two 18-year-olds who build fast, have some ML/RL experience (including training Qwen 1.5B on GSM8K with RL), and have a mentor on Anthropic's RL research team. They need a pitch deck (.pptx) and a 4-week roadmap to get funding.

---

## The Company

**One-liner:** We build adaptive training environments that match a model's specific weaknesses, so every dollar of RL compute produces maximum improvement.

**Analogy:** "Khan Academy for LLM training" — today, RL training gives every model the same textbook. We diagnose what a specific model is bad at, generate environments calibrated to its weaknesses, and make every training dollar 2-5x more efficient.

**Domain focus (Phase 1):** Math/reasoning — chosen because rewards are verifiable (correct answer = binary check), difficulty is naturally quantifiable, this is where RL is producing the biggest breakthroughs right now (DeepSeek-R1, Qwen, etc.), and rich benchmarks already exist (MATH, GSM8K, AMC, AIME).

---

## The Three Core Ideas (from mentor)

### Idea 1: Model-Calibrated Environment Creation (LEAD PRODUCT)
- The same RL environment that's great for one model is terrible for another
- Probe a model via API, identify its capability/weakness profile, generate a training environment calibrated to its specific needs
- Problems that are too easy (model always solves) = no learning signal
- Problems that are too hard (model never solves) = no learning signal  
- The sweet spot: problems the model CAN solve but doesn't RELIABLY solve
- **This is the product we ship first**

### Idea 2: Dynamic Environment Updates During Training (PHASE 2 FEATURE)
- As RL training progresses, the model gets better and the static dataset becomes too easy
- Dynamically reweight or swap out problems during training to keep the model in its productive learning zone
- Mentor says this is "currently very underutilized" — strong whitespace signal
- Harder to demo (requires running actual RL training), so it's phase 2
- Natural extension of Idea 1: Idea 1 = calibrate before training, Idea 2 = recalibrate during training

### Idea 3: Predictive Quality Metrics (INTERNAL TOOLKIT / VALIDATION LAYER)
- Before spending $50K+ on GPU compute, predict whether an RL environment will be effective
- Key metrics:
  - **Advantage distribution:** How much better is a good action vs average? Near-zero = no signal
  - **Mean score vs max score (16 rollouts):** Run the model 16 times per problem
    - Mean high + max high = too easy
    - Mean low + max low = too hard  
    - Mean low + max high = GOLDILOCKS ZONE (model can solve it but not reliably)
- **This is the measurement layer that validates Idea 1's output and enables Idea 2**
- Built as internal toolkit, surfaces in product as "environment quality report"

---

## 4-Week Roadmap

### Week 1: Foundation & Model Profiling (CURRENT WEEK — be ambitious)
**Goal:** Build the model probing pipeline that diagnoses a model's capability profile on math/reasoning

- Set up evaluation harness for math problems (GSM8K, MATH benchmark subsets)
- Implement 16-rollout sampling per problem for any model accessible via API
- Compute core metrics per problem: pass rate (mean score), max score, advantage estimates
- Build capability profiling: categorize problems by type (algebra, geometry, number theory, combinatorics, word problems) and difficulty tier
- Generate a "model capability report" — heatmap of strengths/weaknesses by category × difficulty
- **Testbed:** Run on Qwen 1.5B (already have experience), plus one other model (e.g., Llama 3 8B) to demonstrate calibration differences between models
- **Deliverable:** Side-by-side capability profiles showing two different models have different weakness patterns

### Week 2: Calibrated Environment Generation
**Goal:** Use the profiling from Week 1 to actually generate calibrated environments

- Build problem difficulty estimator (given a problem, predict its difficulty for a specific model — use Week 1 data)
- Implement environment generation: given a model's profile, select/generate problems that fall in the goldilocks zone (mean low, max high)
- Create problem generation pipeline using an LLM to synthesize new math problems at target difficulty levels and categories
- Build filtering pipeline: generate candidates → run 16 rollouts → keep only goldilocks-zone problems
- **Deliverable:** For a given model, produce a calibrated environment and show (via metrics) that it has better signal properties than a random/uncalibrated environment

### Week 3: Validation & Proof of Concept
**Goal:** Prove that calibrated environments actually produce better training outcomes

- Run small-scale RL training on Qwen 1.5B:
  - Control: train on uncalibrated (random) math environment
  - Treatment: train on CalibrateRL-generated environment
- Measure improvement rate per training step, final benchmark scores, compute efficiency
- Build the "environment quality report" product artifact — the thing you'd actually show a customer
- Start building a simple web UI or CLI tool that a customer could use
- **Deliverable:** Head-to-head comparison showing calibrated env → faster improvement per compute dollar

### Week 4: Polish, Pitch, & Demo
**Goal:** Fundable demo and materials

- Polish the end-to-end pipeline: model in → capability report + calibrated environment out
- Create compelling visualizations of the results (before/after training curves, capability heatmaps, environment quality metrics)
- Record demo video
- Finalize pitch deck with real data and results
- Prepare for investor/advisor conversations
- **Deliverable:** Working demo, pitch deck with real results, clear ask

---

## Pitch Deck Structure (12 slides)

### Slide 1: Title
- "CalibrateRL" (or chosen name)
- Tagline: "Adaptive Training Environments for LLMs"
- Subtitle: "Making every dollar of RL compute count"

### Slide 2: The Problem
- RL training for LLMs is exploding — every lab and fine-tuning company needs training environments
- Today's approach is one-size-fits-all: same dataset for every model
- This wastes massive compute — problems that are too easy or too hard produce zero learning signal
- Visual: spectrum showing "too easy → goldilocks → too hard" with the wasted zones highlighted

### Slide 3: The Insight
- Different models have different capability profiles — what works for GPT-4 fails for Llama 8B
- The same environment can be gold for one model and garbage for another
- "The Khan Academy problem": personalized learning works for humans. Nobody has built it for AI training yet.
- Visual: two models with different heatmaps on the same environment

### Slide 4: Our Solution
- CalibrateRL: Give us your model, we diagnose its weaknesses and generate a training environment calibrated specifically for it
- Three-step process: Probe → Profile → Generate
- Built on predictive quality metrics that can assess environment quality before you spend a dollar on training
- Visual: the 3-step pipeline diagram

### Slide 5: How It Works (Technical)
- Model probing: 16 rollouts per problem, compute pass rate, advantage, mean vs max analysis
- Capability profiling: categorize by problem type × difficulty, identify weakness zones
- Environment generation: synthesize + filter problems that land in the goldilocks zone for THIS specific model
- Visual: the mean vs max quadrant chart (too easy / too hard / goldilocks / impossible)

### Slide 6: Why Math First
- Verifiable rewards (correct/incorrect — no judge model needed)
- Quantifiable difficulty gradients
- Where RL is producing the biggest breakthroughs right now (DeepSeek-R1, etc.)
- Clear path to code generation (also verifiable), then general tasks
- Visual: domain expansion roadmap

### Slide 7: Demo / Results
- Head-to-head: calibrated vs uncalibrated environment on same model
- Show training curves, compute savings, benchmark improvements
- Capability heatmaps before and after
- (Fill with real data from Week 3)

### Slide 8: Market Opportunity
- RL training market is expanding rapidly — every frontier lab, every fine-tuning startup, every enterprise AI team
- Compute costs are the #1 bottleneck — anything that improves efficiency per dollar has immediate value
- The shift from supervised fine-tuning to RL training is accelerating (DeepSeek, OpenAI o-series, etc.)

### Slide 9: Product Roadmap
- Phase 1 (Now): Model-calibrated environment creation for math/reasoning
- Phase 2: Dynamic environment updates during training (keep environments optimal as model improves)
- Phase 3: Expand to code generation, then general instruction following
- Phase 4: Full RL training optimization platform

### Slide 10: Team & Advisor
- Two technical co-founders: young, fast, committed, hands-on ML/RL experience
- Advisor: Senior RL researcher at Anthropic (one of the world's leading AI safety labs)
- "We have direct access to frontier RL research thinking and move at startup speed"

### Slide 11: The Ask
- Funding amount (fill in)
- Use of funds: compute for validation experiments, team growth, scaling to code domain
- Timeline: 4 weeks to proof of concept, 3 months to paying customers

### Slide 12: Contact / Closing
- Contact info
- "Making every training dollar count"

---

## Design Direction for Pitch Deck
- Dark, technical, premium feel — not corporate blue
- Suggested palette: deep navy/midnight primary, electric teal accent, off-white text
- Include data visualizations: heatmaps, training curves, quadrant charts
- Keep text minimal per slide — this deck should be visual and punchy
- The "mean vs max quadrant" (Slide 5) is the signature visual — make it memorable

---

## Key Talking Points for Mentor Discussion
1. The product is Idea 1 (calibrated environments), validated by Idea 3 (quality metrics), with Idea 2 (dynamic updates) as phase 2
2. Starting with math/reasoning for verifiable rewards and fast iteration
3. Week 1 focus: build model probing pipeline, demonstrate that two different models have different capability profiles on the same problem set
4. The core demo: side-by-side comparison showing calibrated env produces better training signal per compute dollar
5. Business model: API service — customer sends model access, receives calibrated environment + quality report
