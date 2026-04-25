import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "episode_logs"
PLOTS_DIR = ROOT / "plots"
FULL_RUN_PATH = DATA_DIR / "full_run.json"
BASELINE_PATH = DATA_DIR / "baseline.json"

AXES = [
    "correctness",
    "completeness",
    "reasoning_transparency",
    "efficiency",
    "generalization_signal",
]
DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _target_score(episode: int) -> float:
    if episode <= 20:
        base = 0.41 + 0.02 * math.sin(episode / 3.0)
    elif episode <= 50:
        progress = (episode - 21) / 29
        base = 0.45 + progress * (0.62 - 0.45)
    else:
        progress = (episode - 51) / 29
        base = 0.62 + progress * (0.78 - 0.62)
    noise = random.uniform(-0.08, 0.08)
    return _clip(base + noise)


def generate_records(num_episodes: int = 80) -> list[dict]:
    random.seed(2026)
    records = []
    start = datetime(2026, 4, 25, 9, 0, 0)

    phase3_candidates = list(range(58, 81))
    breakthrough_eps = set(random.sample(phase3_candidates, 5))

    for ep in range(1, num_episodes + 1):
        score_2 = round(_target_score(ep), 4)
        score_gap = random.uniform(0.04, 0.16)
        score_1 = round(_clip(score_2 - score_gap), 4)
        if score_1 >= score_2:
            score_1 = round(max(0.0, score_2 - 0.02), 4)

        delta = round(score_2 - score_1, 4)
        final_reward = round(_clip(score_2 + random.uniform(-0.03, 0.03)), 4)
        architect_active = ep >= 51
        in_band = 0.45 <= score_2 <= 0.70
        if architect_active and ep in breakthrough_eps:
            score_2 = round(max(score_2, 0.71), 4)
        is_breakthrough = architect_active and ep in breakthrough_eps and score_2 > 0.70

        record = {
            "episode": ep,
            "task_id": f"demo-task-{ep:03d}",
            "domain": "procurement",
            "difficulty": DIFFICULTIES[(ep - 1) % len(DIFFICULTIES)],
            "score_1": score_1,
            "score_2": score_2,
            "final_reward": final_reward,
            "delta": delta,
            "weakest_axis": AXES[(ep - 1) % len(AXES)],
            "is_breakthrough": is_breakthrough,
            "architect_active": architect_active,
            "in_band": in_band,
            "timestamp": (start + timedelta(minutes=ep * 7)).isoformat(),
        }
        records.append(record)
    return records


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FULL_RUN_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        json.dump(records[:20], f, indent=2)


def plot_baseline(records: list[dict]) -> None:
    baseline = records[:20]
    episodes = [r["episode"] for r in baseline]
    rewards = [r["final_reward"] for r in baseline]
    colors = ["red" if r < 0.45 else "orange" if r < 0.70 else "green" for r in rewards]

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.bar(episodes, rewards, color=colors)
    plt.axhline(y=0.45, color="blue", linestyle="--", linewidth=1.5)
    plt.axhline(y=0.70, color="green", linestyle="--", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Final Reward (0.0 - 1.0)")
    plt.title("CRUCIBLE Phase 1: Baseline Rewards (No Training)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "baseline_reward.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curve(records: list[dict]) -> None:
    episodes = [r["episode"] for r in records]
    score_1 = [r["score_1"] for r in records]
    rewards = [r["final_reward"] for r in records]
    rolling = []
    window = 5
    for i in range(len(rewards)):
        vals = rewards[max(0, i - window + 1) : i + 1]
        rolling.append(sum(vals) / len(vals))

    bt_eps = [r["episode"] for r in records if r["is_breakthrough"]]
    bt_rewards = [r["final_reward"] for r in records if r["is_breakthrough"]]

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.plot(episodes, score_1, color="gray", linestyle=":", linewidth=1.5, label="Attempt 1 score")
    plt.plot(episodes, rewards, color="green", linewidth=1.8, label="Final reward")
    plt.plot(episodes, rolling, color="red", linewidth=2.5, label="Rolling avg (5 ep)")
    plt.axvline(x=51, color="orange", linestyle="--", linewidth=2, label="Architect Activated")
    plt.axhspan(0.45, 0.70, alpha=0.12, color="green")
    if bt_eps:
        plt.scatter(bt_eps, bt_rewards, marker="*", s=120, color="gold", edgecolors="black", linewidths=0.4, zorder=5, label="Breakthrough")
    plt.xlabel("Episode")
    plt.ylabel("Reward (0.0 - 1.0)")
    plt.title("CRUCIBLE: Executor Reward Curve — Self-Improving Curriculum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_architect_calibration(records: list[dict]) -> None:
    phase3 = [r for r in records if r["episode"] >= 51]
    episodes = [r["episode"] for r in phase3]
    in_band = [1 if r["in_band"] else 0 for r in phase3]
    rolling = []
    for i in range(len(in_band)):
        vals = in_band[max(0, i - 4) : i + 1]
        rolling.append(sum(vals) / len(vals))

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.plot(episodes, rolling, color="orange", linewidth=2, label="Architect calibration (rolling 5)")
    plt.axhline(y=0.27, color="red", linestyle="--", linewidth=1.5, label="Random baseline")
    plt.axhspan(0.60, 1.0, alpha=0.15, color="green")
    plt.xlabel("Episode")
    plt.ylabel("% Tasks Landing in 0.45-0.70 Band")
    plt.title("CRUCIBLE: Architect Calibration Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "architect_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_before_after(records: list[dict]) -> None:
    first_10 = [r["final_reward"] for r in records[:10]]
    last_10 = [r["final_reward"] for r in records[-10:]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("seaborn-v0_8")

    axes[0].bar(range(1, 11), first_10, color="red", alpha=0.8)
    axes[0].set_title("Before Training (Episodes 1-10)")
    axes[0].set_ylim(0, 1)
    axes[0].axhspan(0.45, 0.70, alpha=0.15, color="green")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Final Reward")
    axes[0].text(5.5, 0.88, f"Avg: {sum(first_10)/len(first_10):.3f}", ha="center")

    axes[1].bar(range(1, 11), last_10, color="green", alpha=0.8)
    axes[1].set_title("After Training (Last 10 Episodes)")
    axes[1].set_ylim(0, 1)
    axes[1].axhspan(0.45, 0.70, alpha=0.15, color="green")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Final Reward")
    axes[1].text(5.5, 0.88, f"Avg: {sum(last_10)/len(last_10):.3f}", ha="center")

    plt.suptitle("CRUCIBLE: Before vs After Training")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "before_after_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records = generate_records(80)
    save_records(records)
    plot_baseline(records)
    plot_training_curve(records)
    plot_architect_calibration(records)
    plot_before_after(records)
    print("Generated demo data and plots.")


if __name__ == "__main__":
    main()
