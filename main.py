"""
CRUCIBLE — Main Entry Point

Commands:
  python main.py baseline     — Phase 1: measure base model performance
  python main.py train        — Phase 2: run 50 episodes, no Architect
  python main.py architect    — Phase 3: activate Architect, 30 episodes
  python main.py demo         — Run demo mode with failure injection
  python main.py episode      — Run a single episode (quick test)
  python main.py regression   — Check for catastrophic forgetting
  python main.py dashboard    — Launch Streamlit dashboard
"""

import sys
import subprocess


def run_baseline():
    from training.grpo_loop import run_phase1_baseline
    run_phase1_baseline()


def run_train():
    from training.grpo_loop import _simulate_phase2
    _simulate_phase2(n_episodes=50)


def run_architect():
    from training.grpo_loop import run_phase3_architect
    run_phase3_architect(n_episodes=30)


def run_demo():
    from demo.failure_injector import inject_and_get_architect_response
    print("\n=== CRUCIBLE LIVE DEMO MODE ===")
    print("Injecting reasoning_transparency failures...")
    inject_and_get_architect_response(axis="reasoning_transparency")


def run_single_episode():
    from core.episode_runner import EpisodeRunner
    runner = EpisodeRunner(seed=42, use_architect=False)
    summary = runner.run_episode()
    import json
    print("\nEpisode summary:")
    print(json.dumps(summary, indent=2, default=str))


def run_regression():
    from eval.regression_checker import check_regression
    check_regression()


def run_dashboard():
    subprocess.run(["streamlit", "run", "demo/dashboard.py"])


def main():
    commands = {
        "baseline": run_baseline,
        "train": run_train,
        "architect": run_architect,
        "demo": run_demo,
        "episode": run_single_episode,
        "regression": run_regression,
        "dashboard": run_dashboard,
    }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "episode"

    if cmd not in commands:
        print(__doc__)
        sys.exit(1)

    print(f"\n🔥 CRUCIBLE | Running: {cmd}\n")
    commands[cmd]()


if __name__ == "__main__":
    main()
