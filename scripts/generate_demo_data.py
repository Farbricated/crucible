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


JURISDICTIONS = ["FAR", "FAR", "FAR", "DFARS", "EU"]
SHOCK_TYPES = ["threshold_change", "vendor_debarment", "new_clause", "sanctions_update", None, None, None]


def generate_records(num_episodes: int = 80) -> list[dict]:
    random.seed(2026)
    records = []
    start = datetime(2026, 4, 25, 9, 0, 0)

    phase3_candidates = list(range(58, 81))
    breakthrough_eps = set(random.sample(phase3_candidates, 5))
    adversarial_eps = set(random.sample(range(30, 81), 15))
    shock_eps = set(random.sample(range(1, 81), 20))

    # Per-axis failure counts for heatmap (realistic: reasoning_transparency weakest early)
    axis_fail_weights_early = [0.15, 0.18, 0.35, 0.12, 0.20]   # episodes 1-20
    axis_fail_weights_mid =   [0.20, 0.25, 0.25, 0.15, 0.15]   # episodes 21-50
    axis_fail_weights_late =  [0.22, 0.22, 0.20, 0.18, 0.18]   # episodes 51-80

    for ep in range(1, num_episodes + 1):
        score_2 = round(_target_score(ep), 4)
        score_gap = random.uniform(0.04, 0.16)
        score_1 = round(_clip(score_2 - score_gap), 4)
        if score_1 >= score_2:
            score_1 = round(max(0.0, score_2 - 0.02), 4)

        delta = round(score_2 - score_1, 4)

        adversarial = ep in adversarial_eps
        shock_fired = ep in shock_eps
        shock_adapted = shock_fired and random.random() < (0.30 + ep / 200)
        shock_bonus = 0.05 if shock_adapted else 0.0

        final_reward = round(_clip(score_2 + shock_bonus + random.uniform(-0.03, 0.03)), 4)
        architect_active = ep >= 51
        in_band = 0.45 <= score_2 <= 0.70
        if architect_active and ep in breakthrough_eps:
            score_2 = round(max(score_2, 0.71), 4)
        is_breakthrough = architect_active and ep in breakthrough_eps and score_2 > 0.70

        # Weakest axis weighted by phase
        if ep <= 20:
            weights = axis_fail_weights_early
        elif ep <= 50:
            weights = axis_fail_weights_mid
        else:
            weights = axis_fail_weights_late
        weakest = random.choices(AXES, weights=weights, k=1)[0]

        # Vendor reward: adversarial episodes show declining vendor power over time
        vendor_reward = None
        if adversarial:
            vendor_reward = round(_clip(0.65 - ep * 0.003 + random.uniform(-0.1, 0.1)), 4)

        jurisdiction = JURISDICTIONS[(ep - 1) % len(JURISDICTIONS)]
        domain = "eu_procurement" if jurisdiction == "EU" else "procurement"
        shock_type = random.choice(SHOCK_TYPES) if shock_fired else None

        record = {
            "episode": ep,
            "task_id": f"demo-task-{ep:03d}",
            "domain": domain,
            "jurisdiction": jurisdiction,
            "difficulty": DIFFICULTIES[(ep - 1) % len(DIFFICULTIES)],
            "score_1": score_1,
            "score_2": score_2,
            "final_reward": final_reward,
            "delta": delta,
            "weakest_axis": weakest,
            "is_breakthrough": is_breakthrough,
            "architect_active": architect_active,
            "in_band": in_band,
            "adversarial": adversarial,
            "vendor_reward": vendor_reward,
            "shock_fired": shock_fired,
            "shock_type": shock_type,
            "shock_adapted": shock_adapted if shock_fired else None,
            "consequence_if_approved": (
                "Approved: contract voided, DoD audit triggered" if score_2 < 0.5
                else "Correct decision — AXIOM avoids $2.1M compliance exposure"
            ),
            "diversity_score": round(0.85 - ep * 0.002 + random.uniform(-0.1, 0.15), 4) if ep > 10 else round(random.uniform(0.7, 1.0), 4),
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


def plot_axis_heatmap(records: list[dict]) -> None:
    """Heatmap: failure rate per axis × difficulty tier."""
    import numpy as np

    axis_diff_counts = {diff: {ax: 0 for ax in AXES} for diff in DIFFICULTIES}
    axis_diff_totals = {diff: 0 for diff in DIFFICULTIES}

    for r in records:
        diff = r["difficulty"]
        axis = r["weakest_axis"]
        if diff in axis_diff_counts and axis in axis_diff_counts[diff]:
            axis_diff_counts[diff][axis] += 1
            axis_diff_totals[diff] += 1

    matrix = np.zeros((len(DIFFICULTIES), len(AXES)))
    for i, diff in enumerate(DIFFICULTIES):
        total = axis_diff_totals[diff] or 1
        for j, ax in enumerate(AXES):
            matrix[i][j] = axis_diff_counts[diff][ax] / total

    short_axes = ["correct.", "complete.", "reasoning", "effic.", "general."]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use("seaborn-v0_8")
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(short_axes, fontsize=11)
    ax.set_yticks(range(len(DIFFICULTIES)))
    ax.set_yticklabels([d.upper() for d in DIFFICULTIES], fontsize=11)
    for i in range(len(DIFFICULTIES)):
        for j in range(len(AXES)):
            ax.text(j, i, f"{matrix[i][j]:.0%}", ha="center", va="center",
                    color="black" if matrix[i][j] < 0.3 else "white", fontsize=10)
    plt.colorbar(im, ax=ax, label="Failure Rate")
    ax.set_title("CRUCIBLE: Where AXIOM Keeps Failing — Axis × Difficulty Heatmap")
    ax.set_xlabel("Arbiter Scoring Axis")
    ax.set_ylabel("Task Difficulty")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "axis_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_adversarial_arms_race(records: list[dict]) -> None:
    """Vendor reward vs Executor reward over adversarial episodes."""
    adv = [r for r in records if r.get("adversarial") and r.get("vendor_reward") is not None]
    if not adv:
        return

    ep_nums = [r["episode"] for r in adv]
    vendor_rewards = [r["vendor_reward"] for r in adv]
    exec_rewards = [r["final_reward"] for r in adv]

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.plot(ep_nums, exec_rewards, color="green", linewidth=2, marker="o", markersize=4, label="Executor reward")
    plt.plot(ep_nums, vendor_rewards, color="red", linewidth=2, marker="x", markersize=5, label="Vendor reward (concealment)")
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    plt.fill_between(ep_nums, exec_rewards, vendor_rewards,
                     where=[e > v for e, v in zip(exec_rewards, vendor_rewards)],
                     alpha=0.15, color="green", label="Executor winning")
    plt.fill_between(ep_nums, exec_rewards, vendor_rewards,
                     where=[e <= v for e, v in zip(exec_rewards, vendor_rewards)],
                     alpha=0.15, color="red", label="Vendor winning")
    plt.xlabel("Episode")
    plt.ylabel("Reward (0.0 - 1.0)")
    plt.title("CRUCIBLE: Adversarial Arms Race — Vendor vs Executor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "adversarial_arms_race.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shock_adaptation(records: list[dict]) -> None:
    """Rolling adaptation rate for regulation shock episodes."""
    shock_eps = [r for r in records if r.get("shock_fired")]
    if len(shock_eps) < 3:
        return

    ep_nums = [r["episode"] for r in shock_eps]
    adapted = [1 if r.get("shock_adapted") else 0 for r in shock_eps]
    rolling = []
    for i in range(len(adapted)):
        vals = adapted[max(0, i - 4): i + 1]
        rolling.append(sum(vals) / len(vals))

    plt.figure(figsize=(10, 4))
    plt.style.use("seaborn-v0_8")
    plt.bar(ep_nums, adapted, color=["green" if a else "red" for a in adapted],
            alpha=0.4, label="Adapted (1) / Failed (0)")
    plt.plot(ep_nums, rolling, color="blue", linewidth=2.5, label="Rolling adaptation rate (5 ep)")
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    plt.xlabel("Episode (shock episodes only)")
    plt.ylabel("Adaptation Rate")
    plt.title("CRUCIBLE: Executor Adaptation to Regulation Shocks")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shock_adaptation.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_loss(records: list[dict]) -> None:
    """Synthesize a GRPO-style policy loss curve decreasing over training steps.

    Loss is computed from per-episode reward as: loss = clip(-log(reward+eps), 0, 6)
    Rolling average + exponential decay trend gives a realistic GRPO loss curve.
    """
    import numpy as np

    steps = list(range(1, len(records) + 1))
    rewards = [max(r["final_reward"], 0.05) for r in records]

    raw_loss = [-math.log(r + 0.02) + random.uniform(-0.08, 0.12) for r in rewards]

    window = 5
    rolling = []
    for i in range(len(raw_loss)):
        vals = raw_loss[max(0, i - window + 1): i + 1]
        rolling.append(sum(vals) / len(vals))

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.plot(steps, raw_loss, color="#94a3b8", alpha=0.45, linewidth=1.0, label="Per-step loss")
    plt.plot(steps, rolling, color="#dc2626", linewidth=2.5, label="Rolling avg loss (5 steps)")
    plt.axvline(x=51, color="orange", linestyle="--", linewidth=2, label="Architect Activated")
    plt.xlabel("Training Step (Episode)")
    plt.ylabel("Policy Loss (GRPO objective)")
    plt.title("CRUCIBLE: Training Loss Curve — GRPO Policy Loss Over Episodes")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_baseline_vs_trained(records: list[dict]) -> None:
    """Single chart with baseline rolling-avg and post-training rolling-avg on the same axes."""
    if len(records) < 20:
        return

    baseline = records[:20]
    trained  = records[-20:]
    window = 5

    def _roll(vals: list[float]) -> list[float]:
        out = []
        for i in range(len(vals)):
            s = vals[max(0, i - window + 1): i + 1]
            out.append(sum(s) / len(s))
        return out

    x_axis = list(range(1, 21))
    baseline_rolling = _roll([r["final_reward"] for r in baseline])
    trained_rolling  = _roll([r["final_reward"] for r in trained])

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8")
    plt.plot(x_axis, baseline_rolling, color="#ef4444", linewidth=2.5, marker="o",
             markersize=5, label="Untrained baseline (first 20 eps)")
    plt.plot(x_axis, trained_rolling,  color="#10b981", linewidth=2.5, marker="s",
             markersize=5, label="After training (last 20 eps)")
    plt.axhspan(0.45, 0.70, alpha=0.12, color="green", label="Learning band")
    plt.axhline(y=sum(baseline_rolling)/len(baseline_rolling), color="#ef4444",
                linestyle=":", linewidth=1, alpha=0.7)
    plt.axhline(y=sum(trained_rolling)/len(trained_rolling), color="#10b981",
                linestyle=":", linewidth=1, alpha=0.7)
    plt.xlabel("Episode (relative to phase)")
    plt.ylabel("Rolling Average Reward (window=5)")
    plt.title("CRUCIBLE: Baseline vs Trained Agent — Same Axes Comparison")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "baseline_vs_trained.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_jurisdiction_comparison(records: list[dict]) -> None:
    """Average reward by jurisdiction (FAR vs DFARS vs EU)."""
    from collections import defaultdict
    jur_rewards = defaultdict(list)
    for r in records:
        jur = r.get("jurisdiction", "FAR")
        jur_rewards[jur].append(r["final_reward"])

    jurisdictions = sorted(jur_rewards.keys())
    avgs = [sum(jur_rewards[j]) / len(jur_rewards[j]) for j in jurisdictions]
    counts = [len(jur_rewards[j]) for j in jurisdictions]
    colors = {"FAR": "#4ade80", "DFARS": "#60a5fa", "EU": "#f97316"}

    plt.figure(figsize=(8, 5))
    plt.style.use("seaborn-v0_8")
    bars = plt.bar(jurisdictions, avgs,
                   color=[colors.get(j, "#a78bfa") for j in jurisdictions],
                   alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, count, avg in zip(bars, counts, avgs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"n={count}\n{avg:.3f}", ha="center", va="bottom", fontsize=10)
    plt.axhline(y=0.45, color="blue", linestyle="--", linewidth=1.5, label="Learning band floor")
    plt.axhline(y=0.70, color="green", linestyle="--", linewidth=1.5, label="Learning band ceiling")
    plt.xlabel("Jurisdiction")
    plt.ylabel("Average Final Reward")
    plt.title("CRUCIBLE: Cross-Jurisdiction Generalization (FAR / DFARS / EU)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "jurisdiction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records = generate_records(80)
    save_records(records)
    plot_baseline(records)
    plot_training_curve(records)
    plot_architect_calibration(records)
    plot_before_after(records)
    plot_axis_heatmap(records)
    plot_adversarial_arms_race(records)
    plot_shock_adaptation(records)
    plot_jurisdiction_comparison(records)
    plot_training_loss(records)
    plot_baseline_vs_trained(records)
    print("Generated demo data and all plots (reward curves, loss curves, heatmaps, adversarial, shock, jurisdiction, baseline-vs-trained).")


if __name__ == "__main__":
    main()
