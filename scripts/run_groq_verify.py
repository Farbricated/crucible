"""
Quick script: run 5 live Groq episodes and verify llm_backend in full_run.json.
Usage: python scripts/run_groq_verify.py
"""
import os, json, sys

key = os.environ.get("GROQ_API_KEY", "")
print(f"GROQ_API_KEY loaded: {bool(key)} (len={len(key)})")

from core.episode_runner import EpisodeRunner
from utils.llm_client import get_call_stats, active_backend

print(f"Active backend: {active_backend()}")

runner = EpisodeRunner(seed=99, use_architect=False, use_adversarial=False, use_shocks=False)

for i in range(5):
    print(f"--- Running episode {i+1}/5 ---")
    summary = runner.run_episode()
    rwd = summary["final_reward"]
    print(f"  reward={rwd:.3f}")

runner.save_full_log()

stats = get_call_stats()
print()
print("=== GROQ CALL STATS ===")
print(f"  Calls:             {stats['groq']['calls']}")
print(f"  Prompt tokens:     {stats['groq']['prompt_tokens']:,}")
print(f"  Completion tokens: {stats['groq']['completion_tokens']:,}")
print(f"  Total tokens:      {stats['groq']['total_tokens']:,}")

with open("data/episode_logs/full_run.json") as f:
    data = json.load(f)

backends = set(r.get("llm_backend", "MISSING") for r in data)
models   = set(r.get("llm_model",   "MISSING") for r in data)
print()
print(f"Backends in full_run.json: {backends}")
print(f"Models in full_run.json:   {models}")
print(f"Total records: {len(data)}")
