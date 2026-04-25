Fixed environments have ceilings. CRUCIBLE doesn't — because the hardest problem you face tomorrow is built from your failures today.

## What Problem Does This Solve
Most RL environments have a fixed task set. Once an agent masters it, learning stops — not because the agent hit its ceiling, but because the environment did. CRUCIBLE solves this by making the training curriculum an emergent property of the agent's own failure history.

## How It Works
| Agent | Role | Reward Signal |
|---|---|---|
| Executor | Solves procurement compliance tasks at AXIOM Corporation (FAR/DFARS/ITAR violations) | Weighted score across 5 axes from Arbiter |
| Arbiter | Frozen scorer — never adapts, never goes easy | N/A (frozen) |
| Architect | Reads last N failures, generates next task targeting weakest axis | +0.15 if Executor lands in 0.45–0.70 band, -0.10 if too hard or too easy |

## The Learning Band
```text
Score < 0.20 → Too hard, no signal
Score 0.45–0.70 → Productive learning zone
Score > 0.90 → Too easy, no signal
Architect's only job: keep Executor in this band forever.
```

## Results
![Phase 1 Baseline](plots/baseline_reward.png)  
Baseline rewards before training; many episodes sit below the productive band.

![Training Reward Curve](plots/training_reward_curve.png)  
Reward trend across all episodes, with Architect activation and breakthrough markers.

![Architect Calibration](plots/architect_calibration.png)  
Rolling calibration accuracy for Architect during adaptive curriculum phase.

![Before vs After](plots/before_after_comparison.png)  
Direct comparison of early episodes versus final episodes after training.

## Quick Start
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python main.py episode
python main.py baseline
python main.py train
python main.py architect
streamlit run demo/dashboard.py
```

## Reward Formula
```text
base = score_2.weighted_total
delta = (score_2.weighted_total - attempt_1_score) * 0.2
delta = min(delta, 0.15)
coherence = 0.10 if score_2.world_coherent else -0.10
malformed = -0.15 if (
    score_2.weighted_total == 0 and
    score_2.reasoning_transparency == 0 and
    score_2.correctness == 0
) else 0.0
final_reward = round(max(0.0, min(1.0, base + delta + coherence + malformed)), 4)
```

## Project Structure
```text
crucible/
├── agents/
│   ├── arbiter.py
│   ├── architect.py
│   └── executor.py
├── core/
│   ├── episode_runner.py
│   ├── schemas.py
│   └── world_state.py
├── crucible_env/
│   ├── agents/
│   ├── core/
│   ├── data/episode_logs/
│   ├── domains/procurement/
│   ├── server/
│   ├── training/crucible_grpo.ipynb
│   ├── client.py
│   ├── models.py
│   ├── openenv.yaml
│   └── pyproject.toml
├── data/episode_logs/
├── demo/
│   ├── dashboard.py
│   └── failure_injector.py
├── domains/procurement/tasks.py
├── eval/regression_checker.py
├── scripts/generate_demo_data.py
├── training/grpo_loop.py
├── main.py
└── requirements.txt
```

Team: Akarsh Sangisetti & Sarika Jivrajika | Meta PyTorch OpenEnv Hackathon × Scaler | April 25–26, 2026 | Bangalore
