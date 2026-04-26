# CRUCIBLE: When the Environment Learns From Your Failures

*A self-improving RL environment for procurement compliance — with adversarial red-teaming, mid-episode regulatory shocks, and cross-jurisdiction generalization*

*Team Digital Yudh — Sangisetti Akarsh & Sarika Jivrajika | OpenEnv Hackathon × Scaler × Meta PyTorch | April 2026*

---

## The Problem With Fixed Environments

Every RL environment has a ceiling. Once your agent masters the task set, learning stops — not because the agent hit its limit, but because the environment did. You add more tasks manually. The agent masters those. You add more. It never ends.

What if the environment itself learned from every failure and generated the next hard problem automatically?

That's CRUCIBLE.

---

## What We Built

CRUCIBLE is a procurement compliance RL environment for AXIOM Corporation — a fictional aerospace and defense contractor operating under US Federal Acquisition Regulations (FAR/DFARS) and EU Directive 2014/24/EU.

The agent must detect violations in contracts: expired SAM.gov registrations, undisclosed conflicts of interest, defective cost pricing, ITAR violations, missing mandatory clauses, and more.

What makes it different from every prior RL environment: **four interlocking mechanics that have never been combined before.**

---

## The Four Innovations

### 1. Self-Improving Curriculum (Architect)

The **Architect** reads the Executor's last N failures, computes which regulatory axis was weakest (correctness, completeness, reasoning transparency, efficiency, or generalization), and generates a new task specifically designed to land the Executor in the productive learning band (0.45–0.70).

```
Score < 0.20  → Too hard, no signal. Architect backs off.
Score 0.45–0.70 → Productive learning zone. Architect stays here.
Score > 0.90  → Too easy, no signal. Architect escalates.
```

The Architect uses LLM-powered task generation with specific escalation techniques:
- Hide violations inside legitimate-looking contract language
- Chain two regulatory frameworks that conflict
- Add a vendor with undisclosed prior violations
- Use correct FAR clause numbers but wrong applicability thresholds

The curriculum is an emergent property of failure. There is no ceiling.

### 2. Adversarial Vendor (Red-Team RL)

The **Vendor** is a new agent with one goal: make the Executor fail.

The Vendor receives the same scenario and crafts a contract document that *looks* professionally compliant but hides real FAR/DFARS/EU violations using ten concealment techniques:

- Citing the correct FAR clause number but the wrong threshold value
- Listing many clauses prominently to make a missing one hard to notice
- Using "verbal approval obtained" instead of "written consent" (required by FAR 52.244-2)
- Referencing SAM.gov registration without specifying it expired 45 days ago

**Vendor reward = 1.0 − executor.correctness**

This creates a genuine zero-sum adversarial dynamic inside a single environment. Over training, the Executor learns to detect concealed violations → the Vendor's reward falls → the Vendor has to work harder to conceal.

![Adversarial Arms Race](plots/adversarial_arms_race.png)

### 3. Regulation Shock Events

Every 30% of episodes, a **RegulationShock** fires mid-episode:

- "BREAKING: FAR 52.232-16 progress payment standard rate reduced from 80% to 75% effective today"
- "URGENT: SAM.gov exclusion list update — TechForce LLC added to debarment list effective 09:00 EST"
- "DFARS 252.204-7012 now mandatory for ALL DoD contracts over $500K regardless of classification"
- "OFAC update: Apex Avionics parent company added to SDN list"

The Executor must incorporate the shock into its analysis. Arbiter checks whether the Executor acknowledged the change. Successful adaptation earns a **+0.05 reward bonus**.

This tests distribution-shift adaptation — the ability to update reasoning based on new information without retraining. One of the four hackathon themes directly.

![Shock Adaptation](plots/shock_adaptation.png)

### 4. Multi-Jurisdiction Generalization

The same Executor — without fine-tuning — handles both:

**US (FAR/DFARS):** Small business subcontracting goals, SAM.gov registration, progress payment rates, OCI disclosures, TINA certified cost pricing, DFARS cybersecurity clauses

**EU (Directive 2014/24/EU):** OJEU notice requirements, MEAT award criteria (Art 67), mandatory exclusion grounds (Art 57), framework agreement rules (Art 33), urgency exceptions (Art 32), remedies directives

Cross-jurisdiction generalization without retraining proves the environment tests genuine understanding, not regulatory memorization.

![Jurisdiction Comparison](plots/jurisdiction_comparison.png)

---

## The Full Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CRUCIBLE Environment                    │
│                                                          │
│  VENDOR ──crafts adversarial contract──► EXECUTOR        │
│  (adversary)                                │            │
│                                    attempt_1│            │
│                                             ▼            │
│                                       ARBITER scores     │
│                                       + consequence      │
│                                       + shock check      │
│                                             │ feedback   │
│                                    attempt_2▼            │
│                                       ARBITER final      │
│                                             │            │
│                              failure_record ▼            │
│                                        ARCHITECT         │
│                                 (generates next task)    │
│                                                          │
│  ⚡ REGULATION SHOCK (30% probability per episode)      │
└─────────────────────────────────────────────────────────┘
```

| Agent | Role | Reward |
|---|---|---|
| Executor | Compliance analyst — detect FAR/DFARS/EU violations | Weighted Arbiter score (5 axes) + shock bonus |
| Arbiter | Frozen 5-axis scorer + counterfactual consequences | None (frozen reference judge) |
| Architect | Generates next task targeting weakest axis | +0.15 in learning band, −0.10 too hard/easy |
| Vendor | Crafts adversarial contracts with hidden violations | 1.0 − executor.correctness |

---

## Reward Design

The reward function is composable and hard to game:

```python
base       = score_2.weighted_total          # 5-axis Arbiter score
delta      = min((score_2 - score_1) * 0.20, 0.15)   # learning from feedback
coherence  = +0.10 if world_coherent else -0.10        # world state validity
malformed  = -0.15 if complete parse failure           # format compliance
shock_bonus = +0.05 if executor adapted to RegulationShock

final_reward = clamp(base + delta + coherence + malformed + shock_bonus, 0, 1)
```

An agent that ignores reasoning quality, skips regulatory citations, or fails to adapt to shocks cannot game the reward by pattern-matching decisions.

---

## Training Results

### Phase 1: Baseline (No Training)

Baseline rewards before any training. Many episodes fall below the productive learning band.

![Baseline Rewards](plots/baseline_reward.png)

### Phase 2: Training Progress

Reward trend across all episodes. Architect activates at episode 51 (orange line). Gold stars mark breakthrough events — episodes where the Executor solved a task type it previously failed.

![Training Curve](plots/training_reward_curve.png)

### Phase 3: Before vs After

Direct comparison of first 10 episodes vs final 10 episodes.

![Before vs After](plots/before_after_comparison.png)

### Architect Calibration

The Architect lands the Executor in the productive learning band consistently above the random baseline.

![Architect Calibration](plots/architect_calibration.png)

### Where AXIOM Keeps Failing

Axis × difficulty heatmap — exactly what the Architect uses to target its next generated task.

![Axis Heatmap](plots/axis_heatmap.png)

---

## Counterfactual Consequences

Every Arbiter score includes a `consequence_if_approved` field — a simulation of what happens to AXIOM if the Executor's decision stands:

> "If approved: $3.4M contract awarded to debarred vendor. Upon discovery, contract void. Contracting officer subject to FAR 9.405 liability. DoD Inspector General referral triggered."

> "If approved: AXIOM incurs $480K in potential overbilling exposure under TINA (False Claims Act liability). Criminal referral to DoJ if intent established."

> "Correct decision — AXIOM avoids $2.1M compliance exposure and maintains clean audit status."

This makes the environment's stakes tangible. Judges can follow the story.

---

## Technical Stack

- **LLM Backend:** HuggingFace Inference API (Llama-3.1-8B-Instruct) with Anthropic fallback
- **Training:** HF TRL GRPOTrainer + Unsloth (4-bit quantization, 2× faster, 70% less memory)
- **Tracking:** Weights & Biases — real reward curves and training metrics
- **Environment:** OpenEnv-compliant FastAPI server (reset/step/state endpoints)
- **Deployment:** HuggingFace Spaces via `openenv push`
- **Dashboard:** Gradio — mission-control view including reward curves, run logs, and backend budget status

---

## How to Run

```bash
# Install
pip install -r requirements.txt

# Configure (HuggingFace Inference API — uses your $30 HF credit)
export HF_TOKEN=hf_your_token_here

# Run a single episode
python main.py episode

# Full training pipeline
python main.py baseline
python main.py train
python main.py architect

# Adversarial red-team
python main.py adversarial

# Regulation shock test
python main.py shock

# EU jurisdiction
python main.py eu

# Dashboard
python main.py dashboard
```

---

## Why It Matters

Procurement fraud costs the US Department of Defense an estimated **$36 billion annually** — roughly 10% of the entire DoD procurement budget. Contracting officers review thousands of documents per year with no AI assistance. A single missed FAR violation can void a multi-million dollar contract, trigger Inspector General referrals, or worse.

CRUCIBLE is the first RL environment that:
1. **Never saturates** — the curriculum grows with the agent
2. **Teaches adversarial robustness** — the agent trains against an active attacker, not just static cases
3. **Tests regulatory distribution shift** — the agent must adapt when the rules change mid-task
4. **Generalizes across jurisdictions** — one model, US FAR/DFARS *and* EU procurement law

The same architecture applies to medical device compliance, financial regulation, tax law, and any domain where rules are dense, adversaries are motivated, and mistakes are expensive.

The environment has no ceiling. Neither does the agent.

---

## Links

- **HuggingFace Space (live env):** [`spaces/Flake56/crucible-env`](https://huggingface.co/spaces/Flake56/crucible-env)
- **Training Notebook:** [`crucible_env/training/crucible_grpo.ipynb`](crucible_env/training/crucible_grpo.ipynb)
- **GitHub Repo:** See HuggingFace Space submission

---

*Built at the OpenEnv Hackathon, Bangalore, April 25–26, 2026.*
*Team Digital Yudh — [Sangisetti Akarsh](https://huggingface.co/Flake56) & [Sarika Jivrajika](https://huggingface.co/False45)*
