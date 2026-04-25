"""
Run a full 3-phase training pipeline and save all results.
Phase 1: 20 baseline episodes
Phase 2: 30 episodes with shocks
Phase 3: 15 episodes with Architect + adversarial
"""
import os, json
from core.episode_runner import EpisodeRunner
from utils.llm_client import get_call_stats, active_backend

print(f"Backend: {active_backend()}")

all_records = []

# ── Phase 1: Baseline (no extras) ────────────────────────────
print("\n" + "="*60)
print("PHASE 1 — Baseline (20 episodes, no special modes)")
print("="*60)
runner1 = EpisodeRunner(seed=42, use_architect=False, use_adversarial=False, use_shocks=False, jurisdiction="FAR")
for i in range(20):
    s = runner1.run_episode()
    all_records.append(runner1.reward_history[-1])

# ── Phase 2: Shocks + Adversarial ────────────────────────────
print("\n" + "="*60)
print("PHASE 2 — Adversarial + Shocks (20 episodes)")
print("="*60)
runner2 = EpisodeRunner(seed=43, use_architect=False, use_adversarial=True, use_shocks=True, shock_probability=0.4, jurisdiction="DFARS")
for i in range(20):
    s = runner2.run_episode()
    all_records.append(runner2.reward_history[-1])

# ── Phase 3: Architect curriculum ─────────────────────────────
print("\n" + "="*60)
print("PHASE 3 — Architect Curriculum + EU (15 episodes)")
print("="*60)
runner3 = EpisodeRunner(seed=44, use_architect=True, use_adversarial=True, use_shocks=True, shock_probability=0.3, jurisdiction="EU")
from domains.eu_procurement.tasks import get_eu_task
for i in range(15):
    task = get_eu_task("medium", i)
    s = runner3.run_episode(task)
    all_records.append(runner3.reward_history[-1])

# ── Save combined log ─────────────────────────────────────────
import os
os.makedirs("data/episode_logs", exist_ok=True)
with open("data/episode_logs/full_run.json", "w") as f:
    json.dump(all_records, f, indent=2)
print(f"\nSaved {len(all_records)} episode records to data/episode_logs/full_run.json")

stats = get_call_stats()
print(f"\nGroq calls: {stats['groq']['calls']}")
print(f"Groq tokens: {stats['groq']['total_tokens']:,}")
