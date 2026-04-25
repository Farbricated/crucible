"""
Regression Checker — runs every 10 episodes to verify
the Executor hasn't forgotten previous domains while learning new ones.
"""

import json
import os
from core.episode_runner import EpisodeRunner
from domains.procurement.tasks import get_all_static_tasks

LOG_DIR = "data/episode_logs"
BASELINE_PATH = os.path.join(LOG_DIR, "baseline.json")


def load_baseline() -> dict:
    if not os.path.exists(BASELINE_PATH):
        print("No baseline found. Run Phase 1 first.")
        return {}
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return {item["task_id"]: item["final_reward"] for item in data}


def check_regression(threshold: float = -0.10) -> dict:
    """
    Re-run all static tasks and compare against baseline.
    Returns dict of domain -> delta.
    Fires alert if any domain dropped below threshold.
    """
    baseline = load_baseline()
    if not baseline:
        return {}

    print("\n[Regression Check] Running all static tasks...")
    runner = EpisodeRunner(seed=42, use_architect=False)
    static_tasks = get_all_static_tasks()

    current_scores = {}
    domain_deltas = {}

    for task in static_tasks:
        summary = runner.run_episode(task=task)
        current_scores[task.task_id] = summary["final_reward"]

    # Compute deltas vs baseline
    regressions = []
    for task_id, current in current_scores.items():
        baseline_score = baseline.get(task_id, 0.0)
        delta = current - baseline_score
        diff_entry = {
            "task_id": task_id,
            "baseline": baseline_score,
            "current": current,
            "delta": round(delta, 4),
            "regression": delta < threshold
        }
        domain_deltas[task_id] = diff_entry
        if delta < threshold:
            regressions.append(task_id)

    if regressions:
        print(f"\n⚠️  REGRESSION DETECTED in {len(regressions)} tasks:")
        for t in regressions:
            d = domain_deltas[t]
            print(f"   {t}: {d['baseline']:.3f} → {d['current']:.3f} (Δ {d['delta']:+.3f})")
    else:
        print("\n✓ No regressions detected.")

    # Save regression report
    report_path = os.path.join(LOG_DIR, "regression_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "regressions_found": len(regressions),
            "threshold": threshold,
            "details": domain_deltas
        }, f, indent=2)

    return domain_deltas


if __name__ == "__main__":
    check_regression()
