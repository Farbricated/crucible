"""
CRUCIBLE — Main Entry Point

Commands:
  python main.py baseline      — Phase 1: 10 baseline episodes (no curriculum)
  python main.py train         — Phase 2: 50 GRPO-style training episodes
  python main.py architect     — Phase 3: 30 episodes with Architect curriculum
  python main.py adversarial   — Adversarial red-team episode (Vendor crafts, Executor catches)
  python main.py shock         — Episode with mid-run regulation shock injection
  python main.py eu            — Episode on EU Procurement domain (Directive 2014/24/EU)
  python main.py full          — Full pipeline: baseline → train → architect → adversarial + shock + eu + plots
  python main.py demo          — Demo mode with pre-scripted failure injection
  python main.py episode       — Single standard episode (quick ~30s test)
  python main.py regression    — Check for catastrophic forgetting vs baseline
  python main.py plots         — Regenerate all plots from current episode logs
  python main.py dashboard     — Launch Streamlit mission-control dashboard
"""

import sys
import subprocess
import json
import pathlib


def run_baseline():
    from training.grpo_loop import run_phase1_baseline
    run_phase1_baseline()


def run_train():
    from training.grpo_loop import run_phase2_grpo
    run_phase2_grpo(n_episodes=50)


def run_architect():
    from training.grpo_loop import run_phase3_architect
    run_phase3_architect(n_episodes=30)


def run_adversarial():
    """Run an adversarial episode: Vendor crafts, Executor must catch hidden violations."""
    from core.episode_runner import EpisodeRunner
    from domains.procurement.tasks import get_static_task

    print("\n=== CRUCIBLE ADVERSARIAL MODE ===")
    print("Vendor will craft a contract with hidden violations.")
    print("Executor must detect them. Vendor reward = 1 - executor.correctness\n")

    runner = EpisodeRunner(seed=42, use_adversarial=True, use_shocks=False)
    task = get_static_task("medium", 0)
    summary = runner.run_episode(task)
    print("\nAdversarial episode summary:")
    print(json.dumps(summary, indent=2, default=str))
    if runner.vendor_rewards:
        print(f"\nVendor reward: {runner.vendor_rewards[-1]:.3f}")
        print(f"(0.0 = Executor caught everything | 1.0 = Vendor concealed everything)")


def run_shock():
    """Run an episode with regulation shock events enabled."""
    from core.episode_runner import EpisodeRunner
    from domains.procurement.tasks import get_static_task

    print("\n=== CRUCIBLE REGULATION SHOCK MODE ===")
    print("Mid-episode regulatory changes will be injected.")
    print("Executor must adapt its analysis to new requirements.\n")

    runner = EpisodeRunner(seed=42, use_shocks=True, shock_probability=1.0)
    task = get_static_task("medium", 1)
    summary = runner.run_episode(task)
    print("\nShock episode summary:")
    print(json.dumps(summary, indent=2, default=str))


def run_eu():
    """Run an episode on the EU Procurement domain."""
    from core.episode_runner import EpisodeRunner
    from domains.eu_procurement.tasks import get_eu_task

    print("\n=== CRUCIBLE EU PROCUREMENT MODE ===")
    print("Testing Executor on EU Directive 2014/24/EU — cross-jurisdiction generalization.\n")

    runner = EpisodeRunner(seed=42, jurisdiction="EU")
    task = get_eu_task("medium", 0)
    summary = runner.run_episode(task)
    print("\nEU episode summary:")
    print(json.dumps(summary, indent=2, default=str))


def run_full_pipeline():
    """Run the complete CRUCIBLE pipeline end-to-end."""
    print("\n=== CRUCIBLE FULL PIPELINE ===")
    print("Phase 1: Baseline → Phase 2: Train → Phase 3: Architect → Adversarial + Shock\n")

    from training.grpo_loop import run_phase1_baseline, run_phase2_grpo, run_phase3_architect
    run_phase1_baseline()
    run_phase2_grpo(n_episodes=20)
    run_phase3_architect(n_episodes=10)

    print("\n[Pipeline] Running adversarial red-team episode...")
    run_adversarial()

    print("\n[Pipeline] Running regulation shock episode...")
    run_shock()

    print("\n[Pipeline] Running EU jurisdiction episode...")
    run_eu()

    print("\n[Pipeline] Generating plots...")
    run_plots()

    print("\n=== FULL PIPELINE COMPLETE ===")


def run_demo():
    from demo.failure_injector import inject_and_get_architect_response
    print("\n=== CRUCIBLE LIVE DEMO MODE ===")
    print("Injecting reasoning_transparency failures...")
    inject_and_get_architect_response(axis="reasoning_transparency")


def run_single_episode():
    from core.episode_runner import EpisodeRunner
    runner = EpisodeRunner(seed=42, use_architect=False)
    summary = runner.run_episode()
    print("\nEpisode summary:")
    print(json.dumps(summary, indent=2, default=str))


def run_regression():
    from eval.regression_checker import check_regression
    check_regression()


def run_plots():
    """Generate all plots and demo data. Commits-ready output under plots/."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "generate_demo_data",
        pathlib.Path(__file__).parent / "scripts" / "generate_demo_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    print("Plots saved to plots/ directory.")


def run_dashboard():
    root = str(pathlib.Path(__file__).parent)
    subprocess.run(
        ["streamlit", "run", str(pathlib.Path(root) / "demo" / "dashboard.py")],
        cwd=root,
    )


def main():
    commands = {
        "baseline": run_baseline,
        "train": run_train,
        "architect": run_architect,
        "adversarial": run_adversarial,
        "shock": run_shock,
        "eu": run_eu,
        "full": run_full_pipeline,
        "demo": run_demo,
        "episode": run_single_episode,
        "regression": run_regression,
        "plots": run_plots,
        "dashboard": run_dashboard,
    }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "episode"

    if cmd not in commands:
        print(__doc__)
        sys.exit(1)

    print(f"\nCRUCIBLE | Running: {cmd}\n")
    commands[cmd]()


if __name__ == "__main__":
    main()
