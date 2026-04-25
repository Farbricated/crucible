"""
GRPO Training Loop for CRUCIBLE Executor
Uses: Unsloth + HF TRL GRPOTrainer
Model: Qwen2.5-7B-Instruct (or smaller for testing)

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

LOG_DIR = "data/episode_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Baseline measurement (no training)
# ─────────────────────────────────────────────────────────────

def run_phase1_baseline():
    """Run all 20 static tasks. Store baseline scores. No training."""
    print("\n" + "="*60)
    print("PHASE 1 — BASELINE MEASUREMENT")
    print("="*60)

    runner = EpisodeRunner(seed=42, use_architect=False)
    static_tasks = get_all_static_tasks()

    baselines = []
    for task in static_tasks:
        summary = runner.run_episode(task=task)
        baselines.append({
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "score_1": summary["score_1"],
            "score_2": summary["score_2"],
            "final_reward": summary["final_reward"],
            "weakest_axis": summary["weakest_axis"],
        })
        print(f"  Baseline | {task.task_id} | {task.difficulty} | reward: {summary['final_reward']:.3f}")

    path = os.path.join(LOG_DIR, "baseline.json")
    with open(path, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nBaseline saved to {path}")
    avg = sum(b["final_reward"] for b in baselines) / len(baselines)
    print(f"Average baseline reward: {avg:.3f}")
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

    # Reward function for GRPO — calls Arbiter
    runner = EpisodeRunner(seed=42, use_architect=False)

    def reward_fn(completions, prompts=None, **kwargs):
        """GRPO reward function — scores each completion via Arbiter."""
        from core.schemas import ExecutorAction, TaskSpec
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
                # Use a medium difficulty task as scoring reference
                ref_task = static_tasks[0]
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
    """Simulation mode — runs episodes via Anthropic API without local model training."""
    print("Running Phase 2 in SIMULATION MODE (Anthropic API as Executor)")
    runner = EpisodeRunner(seed=42, use_architect=False)

    for i in range(n_episodes):
        summary = runner.run_episode()
        if (i + 1) % 10 == 0:
            metrics = runner.get_metrics_summary()
            print(f"\n[Episode {i+1}] Avg reward: {metrics['avg_final_reward']:.3f}")

    runner.save_full_log()
    metrics = runner.get_metrics_summary()
    print(f"\nPhase 2 complete. Final metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# ─────────────────────────────────────────────────────────────
# PHASE 3 — Architect activation
# ─────────────────────────────────────────────────────────────

def run_phase3_architect(n_episodes: int = 30):
    """Run episodes with Architect generating tasks from failure history."""
    print("\n" + "="*60)
    print("PHASE 3 — ARCHITECT ACTIVATION")
    print("="*60)

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
        if (i + 1) % 10 == 0:
            metrics = runner.get_metrics_summary()
            print(f"\n[Episode {runner.episode_count}] Avg reward: {metrics['avg_final_reward']:.3f} | Calibration: {metrics['architect_calibration_accuracy']:.2%}")

    runner.save_full_log()
    metrics = runner.get_metrics_summary()
    print(f"\nPhase 3 complete.")
    print(json.dumps(metrics, indent=2))
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
