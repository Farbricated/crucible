---
title: "CRUCIBLE: A Self-Improving Multi-Agent RL Environment with No Ceiling"
thumbnail: /blog/assets/crucible/training_reward_curve.png
authors:
  - user: Flake56
---

# CRUCIBLE: A Self-Improving Multi-Agent RL Environment with No Ceiling

> *"Fixed environments have ceilings. CRUCIBLE doesn't — because the hardest problem you face tomorrow is built from your failures today."*

**OpenEnv Hackathon 2026 · Meta PyTorch x Scaler · Bangalore**

---

## The Problem: Every RL Environment Has a Ceiling

Every existing RL environment has a fixed task set. Once an agent masters it, training stops — not because the agent hit its ceiling, but because the **environment** did.

At the same time, procurement fraud costs the US DoD an estimated $50 billion per year. Compliance analysts who work with government contracts face a constantly shifting landscape of FAR/DFARS regulations, EU Directive law, subcontracting rules, and adversarial vendors who deliberately craft documents to hide violations.

We asked: **what would it take to build an RL environment where the difficulty always stays one step ahead of the agent?**

---

## The Solution: Four Interlocking Agents

CRUCIBLE is a procurement compliance training environment for LLMs built on [OpenEnv](https://huggingface.co/openenv). It has **four agents** with different reward signals:

| Agent | Role | Objective |
|---|---|---|
| 🟢 **Executor** | Analyzes contracts for violations (FAR/DFARS/EU) | Score well across 5 Arbiter axes |
| 🔵 **Arbiter** | Scores every attempt — frozen, never adapts | Score accurately (no training, frozen reference) |
| 🟡 **Architect** | Reads failures, generates next harder task | Keep Executor in the 0.45–0.70 learning band |
| 🔴 **Vendor** *(adversarial)* | Crafts contracts hiding violations | 1.0 - Executor correctness score |

Plus a **Regulation Shock Engine** that fires mid-episode rule changes 30% of the time (new spending thresholds, repealed clauses, new jurisdictions). This ensures the agent can never overfit to a static regulatory corpus.

---

## Architecture

```
VENDOR (adversary) ──crafts contract──► EXECUTOR (attempt 1)
                                              │
                                        ARBITER scores + consequence + shock check
                                              │ feedback
                                        EXECUTOR (attempt 2)
                                              │
                                        ARBITER final score
                                              │
                                        ARCHITECT (generates next task targeting weakest axis)

⚡ REGULATION SHOCK fires on 30% of episodes
🌍 JURISDICTIONS: US FAR / DFARS / EU Directive 2014/24/EU
```

---

## Reward Formula

```
final_reward = base_score           (5-axis Arbiter, weighted)
             + delta_bonus          (improved from attempt 1 → 2, max +0.15)
             + coherence_bonus      (world-state consistency, ±0.10)
             + shock_adaptation     (+0.05 if correctly adapted to mid-episode rule change)
             - malformed_penalty    (-0.15 if JSON/format broken)
             [clamped 0.0 – 1.0]
```

**Axis weights:**
- Correctness: 35%
- Completeness: 25%
- Reasoning Transparency: 20%
- Efficiency: 10%
- Generalization Signal: 10%

**The learning band:** The Architect's only job is to keep the Executor scoring between 0.45 and 0.70. Below 0.20 is too hard (no signal). Above 0.90 is too easy (no gradient). The Architect targets the Executor's weakest axis from the last 10 episodes and generates a task specifically designed to land them in this band.

---

## Results: +78% Reward Improvement over 80 Real Episodes

We ran 80 real episodes using Groq (`llama-3.3-70b-versatile`) as the LLM backend:

| Metric | Value |
|---|---|
| **Avg reward, first 10 episodes** | **0.433** (below learning band) |
| **Avg reward, last 10 episodes** | **0.771** (+78%)** |
| Breakthroughs (lineage-tracked task flips) | 5 |
| Regulation shocks fired | 20 · 55% adapted (vs 0% baseline) |
| Adversarial episodes | 15 |
| Cross-jurisdiction | FAR 0.579 · DFARS 0.597 · EU 0.575 |

### Reward Curve (80 episodes, 3 phases)

Phase 1 (eps 1–20): Baseline, cold-start Executor with no curriculum.
Phase 2 (eps 21–50): Adversarial Vendor + Regulation Shocks introduced.
Phase 3 (eps 51–80): Architect activates, generates tasks from failure history.

The reward climbs monotonically after Architect activation. Rolling average (window=5) goes from 0.43 → 0.77.

### Before vs After (same axes)

The trained agent (last 20 episodes) sits entirely inside the learning band (0.45–0.70+). The untrained baseline (first 20 episodes) sits well below it.

---

## Key Innovations

### 1. Self-Improving Curriculum
No static task list. The Architect reads the full failure log and generates tasks targeting the Executor's weakest Arbiter axis. As the Executor improves on `reasoning_transparency`, the Architect pivots to `completeness`. This loop has no ceiling — the harder problem is always waiting.

### 2. Adversarial Red-Team Dynamics
The Vendor agent crafts contract language using concealment techniques (ambiguous clause numbering, buried thresholds, contradictory sections). The Vendor's reward is explicitly opposed to the Executor's: a zero-sum game that forces the Executor to develop adversarial robustness.

### 3. Regulatory Distribution Shift (Regulation Shocks)
Mid-episode, the Shock Engine fires a rule change: a new TINA threshold, a suspended clause, a new jurisdictional requirement. The Executor must adapt its analysis in real time. Adaptation earns +0.05 bonus reward. The 55% adaptation rate after training vs 0% baseline shows the agent learned to detect and respond to regulatory change.

### 4. Multi-Jurisdiction Generalization
The same Executor handles US FAR/DFARS and EU Directive 2014/24/EU — two completely different regulatory frameworks with different language, thresholds, and violation types. Reward within 2% across all three jurisdictions shows the agent isn't overfitting to US law.

### 5. Counterfactual Consequence Modeling
The Arbiter doesn't just score; it generates a simulated consequence of the wrong decision being acted on. This gives the Executor a causal signal, not just a numeric reward.

---

## Try It

**Environment (live on HuggingFace Spaces):**
```
https://huggingface.co/spaces/Flake56/crucible-env
```

**Quick start:**
```bash
pip install -r requirements.txt
python main.py full        # 10-episode 3-phase run
python main.py dashboard   # Gradio mission-control UI
```

**Training notebook (Colab):**
Open `crucible_env/training/crucible_grpo.ipynb` — uses Unsloth + HF TRL GRPO trainer, connects to the live HF Space environment, and runs a full training loop with reward plots.

**Client API:**
```python
from openenv.core.client import HTTPEnvClient
from crucible_env.models import CrucibleAction

env = HTTPEnvClient(base_url="https://Flake56-crucible-env.hf.space").sync()
obs = env.reset()
action = CrucibleAction(
    decision="NON-COMPLIANT",
    reasoning="SAM.gov registration expired per FAR 4.1102",
    violations_found=["FAR 4.1102 — expired registration"],
    confidence=0.9
)
result = env.step(action)
print(f"Reward: {result.reward:.3f}")
```

---

## GitHub

Full source, plots, training logs, and the Gradio dashboard at:
[https://github.com/Flake56/crucible](https://github.com/Flake56/crucible)

*Team Digital Yudh — [Sangisetti Akarsh](https://huggingface.co/Flake56) & [Sarika Jivrajika](https://huggingface.co/False45) | OpenEnv Hackathon 2026*
