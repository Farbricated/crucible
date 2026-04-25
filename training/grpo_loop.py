"""
GRPO Training Loop for CRUCIBLE Executor
Uses: Unsloth + HF TRL GRPOTrainer
Model: Qwen2.5-7B-Instruct (or smaller for testing)
Tracking: Weights & Biases (wandb) — real training curves

Run phases:
  Phase 1 — baseline measurement (no training)
  Phase 2 — GRPO on static tasks
  Phase 3 — GRPO with Architect active
"""

import json
import os
import sys
from core.episode_runner import EpisodeRunner
from domains.procurement.tasks import get_all_static_tasks
from utils.llm_client import backend_info, active_backend

# ── W&B setup ─────────────────────────────────────────────────
_wandb_available = False
_wandb_run = None

def _init_wandb(project: str = "crucible-openenv", name: str = None):
    global _wandb_available, _wandb_run
    try:
        import wandb
        _wandb_run = wandb.init(
            project=project,
            name=name,
            config={
                **backend_info(),
                "environment": "CRUCIBLE",
                "domain": "procurement + eu_procurement",
                "agents": ["executor", "arbiter", "architect", "vendor"],
                "features": ["adversarial", "regulation_shocks", "multi_jurisdiction", "counterfactual"],
            },
            reinit=True,
        )
        _wandb_available = True
        print(f"W&B run: {_wandb_run.url}")
    except Exception as e:
        print(f"W&B not available ({e}). Continuing without tracking.")
        _wandb_available = False


def _log_wandb(metrics: dict, step: int = None):
    if _wandb_available and _wandb_run:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass


def _finish_wandb():
    if _wandb_available and _wandb_run:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

LOG_DIR = "data/episode_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Baseline measurement (no training)
# ─────────────────────────────────────────────────────────────

def run_phase1_baseline():
    """Run all static tasks. Store baseline scores. No training."""
    print("\n" + "="*60)
    print("PHASE 1 — BASELINE MEASUREMENT")
    print(f"LLM Backend: {active_backend()}")
    print("="*60)

    _init_wandb(name="phase1-baseline")

    runner = EpisodeRunner(seed=42, use_architect=False)
    static_tasks = get_all_static_tasks()

    # Also include EU tasks for cross-jurisdiction baseline
    try:
        from domains.eu_procurement.tasks import get_all_eu_tasks
        eu_tasks = get_all_eu_tasks()[:3]  # 3 EU tasks in baseline
    except Exception:
        eu_tasks = []

    all_tasks = static_tasks + eu_tasks
    baselines = []

    for i, task in enumerate(all_tasks):
        summary = runner.run_episode(task=task)
        record = {
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "score_1": summary["score_1"],
            "score_2": summary["score_2"],
            "final_reward": summary["final_reward"],
            "weakest_axis": summary["weakest_axis"],
            "llm_backend": active_backend(),
        }
        baselines.append(record)
        print(f"  Baseline | {task.task_id} | {task.difficulty} | reward: {summary['final_reward']:.3f}")

        _log_wandb({
            "baseline/final_reward": summary["final_reward"],
            "baseline/score_1": summary["score_1"],
            "baseline/score_2": summary["score_2"],
            "baseline/delta": summary["delta"],
            "baseline/domain": task.domain,
        }, step=i + 1)

    path = os.path.join(LOG_DIR, "baseline.json")
    with open(path, "w") as f:
        json.dump(baselines, f, indent=2)

    avg = sum(b["final_reward"] for b in baselines) / len(baselines)
    print(f"\nBaseline saved to {path}")
    print(f"Average baseline reward: {avg:.3f}")
    _log_wandb({"baseline/avg_reward": avg, "baseline/total_tasks": len(baselines)})
    _finish_wandb()
    return baselines


# ─────────────────────────────────────────────────────────────
# PHASE 2 — GRPO fine-tuning on static tasks
# ─────────────────────────────────────────────────────────────

def run_phase2_grpo(n_episodes: int = 50, model_name: str = "unsloth/Qwen2.5-7B-Instruct"):
    """
    GRPO training using TRL + Unsloth.
    Each episode generates reward via Arbiter → feeds GRPO loop.
    """
    print("\n" + "="*60)
    print("PHASE 2 — GRPO FINE-TUNING")
    print("="*60)

    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        import torch
    except ImportError:
        print("WARNING: Unsloth/TRL not installed. Running in simulation mode.")
        return _simulate_phase2(n_episodes)

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Build prompt dataset from static tasks
    from domains.procurement.tasks import get_all_static_tasks
    from core.world_state import WorldStateManager

    world = WorldStateManager()
    static_tasks = get_all_static_tasks()

    def build_prompt(task):
        world.reset()
        world_text = world.render()
        return (
            f"WORLD STATE:\n{world_text}\n\n"
            f"TASK: {task.scenario_context}\n\n"
            f"DOCUMENT:\n{task.contract_text or ''}\n\n"
            f"Respond with JSON only."
        )

    prompts = [build_prompt(t) for t in static_tasks * (n_episodes // len(static_tasks) + 1)]
    prompts = prompts[:n_episodes]

    # Reward function for GRPO — calls Arbiter with stateful task rotation.
    reward_state = {"idx": 0}

    def reward_fn(completions, prompts=None, **kwargs):
        """GRPO reward function — scores each completion via Arbiter."""
        from core.schemas import ExecutorAction
        from agents.arbiter import score as arbiter_score
        import json, re

        rewards = []
        for completion in completions:
            try:
                raw = re.sub(r"```json|```", "", completion).strip()
                data = json.loads(raw)
                action = ExecutorAction(
                    task_id="grpo-eval",
                    attempt_number=1,
                    decision=data.get("decision", ""),
                    reasoning=data.get("reasoning", ""),
                    violations_found=data.get("violations_found", []),
                    supporting_evidence=data.get("supporting_evidence", []),
                    confidence=float(data.get("confidence", 0.0))
                )
                # Rotate reference tasks so reward signal is not tied to one sample.
                ref_task = static_tasks[reward_state["idx"] % len(static_tasks)]
                reward_state["idx"] += 1
                s = arbiter_score(action, ref_task)
                rewards.append(s.weighted_total)
            except Exception:
                rewards.append(0.0)
        return rewards

    # GRPO config
    config = GRPOConfig(
        output_dir="data/grpo_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=20,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=prompts,
    )

    trainer.train()
    model.save_pretrained("data/executor_phase2")
    tokenizer.save_pretrained("data/executor_phase2")
    print("\nPhase 2 complete. Model saved to data/executor_phase2")


def _simulate_phase2(n_episodes: int = 50):
    """
    Simulation mode — runs episodes via HF Inference API / Anthropic (no local GPU needed).
    Includes adversarial vendor episodes and regulation shock events every 5th episode.
    """
    print(f"Running Phase 2 in SIMULATION MODE (LLM: {active_backend()})")
    print("Adversarial vendor and regulation shocks enabled for diversity.")
    _init_wandb(name="phase2-simulation")

    runner = EpisodeRunner(
        seed=42,
        use_architect=False,
        use_adversarial=False,  # Enable after phase 1 baseline
        use_shocks=True,
        shock_probability=0.25,
    )

    for i in range(n_episodes):
        # Every 5th episode, run adversarial mode to build red-team signal
        if (i + 1) % 5 == 0 and i > 0:
            adv_runner = EpisodeRunner(seed=i, use_adversarial=True, use_shocks=False)
            adv_runner.run_episode()
            runner.vendor_rewards.extend(adv_runner.vendor_rewards)

        summary = runner.run_episode()
        # Log every episode to W&B
        _log_wandb({
            "train/final_reward": summary["final_reward"],
            "train/score_1": summary["score_1"],
            "train/score_2": summary["score_2"],
            "train/delta": summary["delta"],
            "train/is_breakthrough": int(summary.get("is_breakthrough", False)),
        }, step=i + 1)

        if (i + 1) % 10 == 0:
            metrics = runner.get_metrics_summary()
            shock_eps = [r for r in runner.reward_history if r.get("shock_fired")]
            shock_adapt = sum(1 for r in shock_eps if r.get("shock_adapted")) / max(1, len(shock_eps))
            print(f"\n[Episode {i+1}] Avg reward: {metrics['avg_final_reward']:.3f} | Shock adapt: {shock_adapt:.0%}")
            _log_wandb({
                "train/avg_reward": metrics["avg_final_reward"],
                "train/shock_adaptation_rate": shock_adapt,
                "train/breakthroughs": metrics.get("breakthroughs", 0),
            }, step=i + 1)

    runner.save_full_log()
    metrics = runner.get_metrics_summary()
    print(f"\nPhase 2 complete. Final metrics: {json.dumps(metrics, indent=2)}")
    _log_wandb({"train/phase2_avg_reward": metrics["avg_final_reward"]})
    _finish_wandb()
    return metrics


# ─────────────────────────────────────────────────────────────
# PHASE 3 — Architect activation
# ─────────────────────────────────────────────────────────────

def run_phase3_architect(n_episodes: int = 30):
    """Run episodes with Architect generating tasks from failure history."""
    print("\n" + "="*60)
    print("PHASE 3 — ARCHITECT ACTIVATION")
    print(f"LLM Backend: {active_backend()}")
    print("="*60)
    _init_wandb(name="phase3-architect")

    # Load failure history from Phase 2 if available
    runner = EpisodeRunner(seed=42, use_architect=True)

    phase2_log = os.path.join(LOG_DIR, "full_run.json")
    if os.path.exists(phase2_log):
        print(f"Loading Phase 2 failure history from {phase2_log}")
        # Pre-populate last_scores from Phase 2 log
        with open(phase2_log) as f:
            history = json.load(f)
        runner.last_scores = [h["final_reward"] for h in history[-10:]]
        runner.episode_count = len(history)

    for i in range(n_episodes):
        summary = runner.run_episode()
        if summary.get("architect_output"):
            print(f"  [Architect] {summary['architect_output']['architect_reasoning'][:100]}...")

        _log_wandb({
            "architect/final_reward": summary["final_reward"],
            "architect/score_2": summary["score_2"],
            "architect/in_band": int(0.45 <= summary["score_2"] <= 0.70),
            "architect/is_breakthrough": int(summary.get("is_breakthrough", False)),
        }, step=runner.episode_count)

        if (i + 1) % 10 == 0:
            metrics = runner.get_metrics_summary()
            print(f"\n[Episode {runner.episode_count}] Avg reward: {metrics['avg_final_reward']:.3f} | Calibration: {metrics['architect_calibration_accuracy']:.2%}")
            _log_wandb({
                "architect/avg_reward": metrics["avg_final_reward"],
                "architect/calibration_accuracy": metrics["architect_calibration_accuracy"],
                "architect/in_band_rate": metrics.get("in_band_rate", 0),
            }, step=runner.episode_count)

    runner.save_full_log()
    metrics = runner.get_metrics_summary()
    print(f"\nPhase 3 complete.")
    print(json.dumps(metrics, indent=2))
    _log_wandb({"architect/phase3_calibration": metrics["architect_calibration_accuracy"]})
    _finish_wandb()
    return metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "2"

    if phase == "1":
        run_phase1_baseline()
    elif phase == "2":
        _simulate_phase2(n_episodes=50)
    elif phase == "3":
        run_phase3_architect(n_episodes=30)
    else:
        print("Usage: python training/grpo_loop.py [1|2|3]")
