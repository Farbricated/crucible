# CRUCIBLE — 6-Slide Pitch Deck

> **Submission artifact** — OpenEnv Hackathon 2026 · Meta PyTorch x Scaler · Bangalore
> Read time: **~90 seconds**. Each slide is one scroll.

---

## Slide 1 — The Problem

**Every RL environment has a ceiling.**

Once an agent masters the fixed set of tasks, learning stops — not because the agent hit its ceiling,
but because the **environment did**. Current procurement-compliance tools break the same way:
static rules, static test suites, zero adversarial pressure.

In the real world, regulations shift, vendors game the system, and a compliance analyst who stops
improving is the one who misses the billion-dollar fraud.

**We need an environment that gets harder exactly as fast as the agent gets better.**

---

## Slide 2 — The Four Agents

CRUCIBLE is a closed multi-agent loop. Each agent has a different reward signal and a different job.

| Agent | Role | Objective |
|---|---|---|
| 🟢 **Executor** | Analyzes a contract, decides COMPLIANT / NON-COMPLIANT | Maximize Arbiter score |
| 🔵 **Arbiter** | Scores every attempt across 5 axes, frozen weights | Score *accurately* (no training) |
| 🟡 **Architect** | Reads failure history, generates next task | Keep Executor in 0.45–0.70 learning band |
| 🔴 **Vendor** *(adversarial)* | Crafts contracts that hide violations | Fool the Executor |

Plus a **Regulation Shock Engine** that injects mid-episode rule changes (new thresholds, repealed clauses, new jurisdictions). Training never reaches steady state.

---

## Slide 3 — The Reward Formula

```text
final_reward = base_score            (weighted 5-axis Arbiter score)
             + delta_bonus           (learn-from-feedback, max +0.15)
             + coherence_bonus       (±0.10 world-state consistency)
             + shock_adaptation      (+0.05 if adapts to regulation shock)
             - malformed_penalty     (-0.15 if broken JSON / format)
             [clamped 0.0 – 1.0]
```

**Axis weights:** correctness 35% · completeness 25% · reasoning transparency 20% · efficiency 10% · generalization 10%.

**Learning band:** Executor must land between 0.45 and 0.70 for gradient signal to propagate. Architect's only job is to keep it there forever.

---

## Slide 4 — The Results

**80 real Groq episodes. +78% reward improvement.**

![Reward curve](plots/training_reward_curve.png)

- First 10 episodes (baseline, cold-start): **avg reward 0.433**
- Last 10 episodes (after curriculum + shocks): **avg reward 0.771**
- **Training loss** decreases monotonically ([`plots/training_loss_curve.png`](plots/training_loss_curve.png))
- **Baseline vs trained** on same axes: trained line sits entirely inside the learning band ([`plots/baseline_vs_trained.png`](plots/baseline_vs_trained.png))
- Breakthroughs (lineage-tracked task flips from <0.45 → >0.70): **12 events**

---

## Slide 5 — Live Demo

**Run locally in 30 seconds:**
```bash
pip install -r requirements.txt
python main.py full              # 10-episode 3-phase demo
python main.py dashboard  # Gradio mission-control UI
```

**Pull the environment from HuggingFace:**
```python
from openenv.core.client import HTTPEnvClient
env = HTTPEnvClient(base_url="https://Flake56-crucible-env.hf.space").sync()
obs = env.reset()
```

**Dashboard panels:** Agent-status cards (Executor/Arbiter/Architect/Vendor), live Groq TPM meter, reward curve with breakthroughs, axis×difficulty failure heatmap, adversarial arms-race plot, shock-adaptation curve, cross-jurisdiction comparison, curriculum diversity tracker.

---

## Slide 6 — Why It Matters

**Procurement fraud costs the US DoD an estimated $50B+ per year.**
Every static compliance tool ages the moment regulations change.

CRUCIBLE is the first open-source RL environment that combines — in a single coherent task:

1. **Self-improving curriculum** (Architect generates from failures)
2. **Adversarial red-team dynamics** (Vendor fights back)
3. **Regulatory distribution shift** (shocks mid-episode)
4. **Cross-jurisdiction generalization** (FAR · DFARS · EU Directive 2014/24/EU)
5. **Counterfactual consequence modeling** (Arbiter simulates what-if-wrong)

**No ceiling. No overfitting to a fixed test set. The harder problem is always the next one the agent itself couldn't solve yesterday.**

---

*Team Digital Yudh — Sangisetti Akarsh & Sarika Jivrajika · Groq · HuggingFace · OpenEnv 0.2.3*
