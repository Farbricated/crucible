import json
import uuid
import os
from datetime import datetime
from core.schemas import (
    TaskSpec, ExecutorAction, ArbiterScore,
    FailureRecord, EpisodeLogEntry, ArchitectOutput,
    VendorAction, VendorScore, RegulationShock,
)
from core.world_state import WorldStateManager
from core.regulation_shock import RegulationShockEngine
from agents import arbiter, executor, architect
from agents import vendor as vendor_agent
from domains.procurement.tasks import get_static_task, get_all_static_tasks
from utils.diversity import DiversityTracker
from utils.llm_client import active_backend, GROQ_MODEL, HF_MODEL, _backend

LOG_DIR = "data/episode_logs"
os.makedirs(LOG_DIR, exist_ok=True)


class EpisodeRunner:
    def __init__(
        self,
        seed: int = 42,
        use_architect: bool = False,
        use_adversarial: bool = False,
        use_shocks: bool = False,
        shock_probability: float = 0.30,
        jurisdiction: str = "FAR",
    ):
        self.seed = seed
        self.world = WorldStateManager(seed)
        self.use_architect = use_architect
        self.use_adversarial = use_adversarial
        self.use_shocks = use_shocks
        self.jurisdiction = jurisdiction
        self.episode_count = 0
        self.all_failures: list[FailureRecord] = []
        self.episode_log: list[EpisodeLogEntry] = []
        self.last_scores: list[float] = []
        self.current_difficulty = "easy"
        self.next_task: TaskSpec = None
        self.last_architect_output: ArchitectOutput = None
        self.architect_rewards: list[float] = []
        self.shock_engine = RegulationShockEngine(shock_probability, jurisdiction)
        self.vendor_rewards: list[float] = []
        self.diversity_tracker = DiversityTracker()
        self.diversity_scores: list[float] = []

        # Metrics for dashboard
        self.reward_history: list[dict] = []
        self.architect_calibration: list[bool] = []

    def reset(self, seed: int = None):
        self.world.reset(seed or self.seed)

    def run_episode(self, task: TaskSpec = None) -> dict:
        """Run one full episode (up to 8 steps). Returns episode summary."""
        self.episode_count += 1
        episode_id = f"ep-{self.episode_count:04d}-{uuid.uuid4().hex[:6]}"
        self.reset()

        if task is None:
            task = self._get_next_task()

        print(f"\n{'='*60}")
        print(f"EPISODE {self.episode_count} | {episode_id}")
        print(f"Task: {task.task_id} | Domain: {task.domain} | Difficulty: {task.difficulty}")
        print(f"Target axis: {task.target_axis} | Static: {task.is_static}")
        print(f"{'='*60}")

        world_snapshot_before = self.world.snapshot()
        world_text = self.world.render()

        # ── ADVERSARIAL VENDOR (optional) ───────────────────────
        vendor_action: VendorAction | None = None
        if self.use_adversarial:
            print(f"\n[Vendor] Crafting adversarial contract...")
            vendor_action = vendor_agent.craft(task, self.jurisdiction, task.difficulty)
            # Override task contract with vendor's adversarial version
            task = task.model_copy(update={"contract_text": vendor_action.crafted_contract})
            print(f"  Embedded {len(vendor_action.hidden_violations)} hidden violations")
            print(f"  Techniques: {', '.join(vendor_action.concealment_techniques[:2])}")

        # ── REGULATION SHOCK (optional) ─────────────────────────
        active_shock: RegulationShock | None = None
        shock_text = ""
        if self.use_shocks:
            active_shock = self.shock_engine.maybe_fire(self.episode_count)
            if active_shock:
                shock_text = self.shock_engine.format_for_executor(active_shock)
                # Apply world delta from shock
                if active_shock.world_delta:
                    self.world.apply_delta(active_shock.world_delta)
                print(f"\n[Shock] {active_shock.shock_type}: {active_shock.description[:80]}...")

        # ── ATTEMPT 1 ──────────────────────────────────────────
        print(f"\n[Attempt 1] Executor solving...")
        action_1 = executor.act(task, world_text + shock_text, attempt_number=1)
        world_coherent_1 = self.world.check_coherence(action_1.decision)
        score_1 = arbiter.score(action_1, task, world_coherent_1, shock=active_shock)
        feedback = arbiter.generate_feedback_prompt(score_1)

        print(f"  Decision: {action_1.decision[:80]}...")
        print(f"  Score 1: {score_1.weighted_total:.3f} | Coherent: {world_coherent_1}")
        print(f"  Weakest: {arbiter.get_weakest_axis(score_1)}")
        if active_shock:
            print(f"  Shock adapted: {score_1.shock_adapted}")

        # ── ATTEMPT 2 (with feedback) ───────────────────────────
        print(f"\n[Attempt 2] Executor retrying with feedback...")
        action_2 = executor.act(task, world_text + shock_text, feedback=feedback, attempt_number=2)
        world_coherent_2 = self.world.check_coherence(action_2.decision)
        score_2 = arbiter.score(action_2, task, world_coherent_2, shock=active_shock)

        print(f"  Decision: {action_2.decision[:80]}...")
        print(f"  Score 2: {score_2.weighted_total:.3f} | Coherent: {world_coherent_2}")

        # ── WORLD STATE UPDATE ──────────────────────────────────
        if score_2.world_state_delta:
            self.world.apply_delta(score_2.world_state_delta)

        # ── VENDOR SCORING (adversarial) ────────────────────────
        vendor_score: VendorScore | None = None
        if vendor_action:
            vendor_score = vendor_agent.score_concealment(vendor_action, score_2)
            self.vendor_rewards.append(vendor_score.vendor_reward)
            print(f"\n[Vendor Score] Concealment rate: {vendor_score.concealment_rate:.2f} | Vendor reward: {vendor_score.vendor_reward:.3f}")

        # ── FINAL REWARD ────────────────────────────────────────
        final_reward = arbiter.compute_final_reward(score_2, score_1.weighted_total)
        # Shock adaptation bonus (+0.05 if executor correctly adapted)
        if active_shock and score_2.shock_adapted:
            final_reward = round(min(1.0, final_reward + 0.05), 4)
        score_2.final_reward = final_reward

        # ── BREAKTHROUGH CHECK ──────────────────────────────────
        is_breakthrough = self._check_breakthrough(task, score_2.weighted_total)

        # ── FAILURE RECORD ──────────────────────────────────────
        weakest = arbiter.get_weakest_axis(score_2)
        failure = FailureRecord(
            task_id=task.task_id,
            domain=task.domain,
            difficulty=task.difficulty,
            target_axis=task.target_axis,
            attempt_1_score=score_1.weighted_total,
            attempt_2_score=score_2.weighted_total,
            delta=round(score_2.weighted_total - score_1.weighted_total, 4),
            weakest_axis=weakest,
            feedback_summary=score_2.what_failed[:200],
            lineage_id=task.lineage_id,
            breakthrough=is_breakthrough
        )
        self.all_failures.append(failure)
        self.last_scores.append(score_2.weighted_total)
        if len(self.last_scores) > 10:
            self.last_scores.pop(0)

        # ── ARCHITECT ACTIVATION ────────────────────────────────
        arch_output = None
        arch_reward = None
        if self.use_architect and len(self.all_failures) >= 3:
            print(f"\n[Architect] Reading failure history...")
            last_3 = self.last_scores[-3:]
            next_task_spec, arch_output = architect.generate(
                failures=self.all_failures,
                last_3_scores=last_3,
                current_difficulty=task.difficulty,
                episode_count=self.episode_count
            )
            self.next_task = next_task_spec
            self.last_architect_output = arch_output

            arch_reward = architect.compute_architect_reward(
                executor_score=score_2.weighted_total,
                is_breakthrough=is_breakthrough
            )
            self.architect_rewards.append(arch_reward)
            in_band = 0.45 <= score_2.weighted_total <= 0.70
            self.architect_calibration.append(in_band)

            print(f"  Architect reasoning: {arch_output.architect_reasoning[:150]}...")
            print(f"  Next task: {next_task_spec.task_id} | Difficulty: {next_task_spec.difficulty}")
            print(f"  Architect reward: {arch_reward}")

        # ── EPISODE LOG ENTRY ───────────────────────────────────
        log_entry = EpisodeLogEntry(
            episode_id=episode_id,
            step=1,
            task=task,
            attempt_1=action_1,
            score_1=score_1,
            attempt_2=action_2,
            score_2=score_2,
            failure_record=failure,
            architect_output=arch_output,
            vendor_action=vendor_action,
            vendor_score=vendor_score,
            regulation_shock=active_shock,
            jurisdiction=self.jurisdiction,
        )
        self.episode_log.append(log_entry)
        self._save_log(log_entry)

        # ── DIVERSITY TRACKING ──────────────────────────────────
        diversity_score = self.diversity_tracker.record(task, self.jurisdiction)
        self.diversity_scores.append(diversity_score)

        # ── METRICS ─────────────────────────────────────────────
        record = {
            "episode": self.episode_count,
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "jurisdiction": self.jurisdiction,
            "score_1": score_1.weighted_total,
            "score_2": score_2.weighted_total,
            "final_reward": final_reward,
            "delta": failure.delta,
            "weakest_axis": weakest,
            "is_breakthrough": is_breakthrough,
            "architect_active": self.use_architect,
            "in_band": 0.45 <= score_2.weighted_total <= 0.70,
            "adversarial": self.use_adversarial,
            "shock_fired": active_shock is not None,
            "shock_adapted": score_2.shock_adapted if active_shock else None,
            "vendor_reward": vendor_score.vendor_reward if vendor_score else None,
            "consequence_if_approved": score_2.consequence_if_approved,
            "diversity_score": diversity_score,
            "llm_backend": _backend(),
            "llm_model": active_backend(),
            "timestamp": datetime.now().isoformat(),
        }
        self.reward_history.append(record)
        self._append_full_log(record)

        summary = {
            "episode_id": episode_id,
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "score_1": score_1.weighted_total,
            "score_2": score_2.weighted_total,
            "final_reward": final_reward,
            "delta": failure.delta,
            "weakest_axis": weakest,
            "is_breakthrough": is_breakthrough,
            "architect_output": arch_output.model_dump() if arch_output else None,
        }

        print(f"\n[Summary] Final reward: {final_reward:.3f} | Delta: {failure.delta:+.3f} | Breakthrough: {is_breakthrough}")
        return summary

    def _get_next_task(self) -> TaskSpec:
        """Return Architect-generated task if available, else next static task."""
        if self.next_task and self.use_architect:
            t = self.next_task
            self.next_task = None
            return t

        # Cycle through static tasks by difficulty
        difficulties = ["easy", "easy", "easy", "medium", "medium", "hard", "expert"]
        difficulty = difficulties[min(self.episode_count - 1, len(difficulties) - 1)]
        self.current_difficulty = difficulty
        idx = max(0, self.episode_count - 1)
        return get_static_task(difficulty, idx)

    def _check_breakthrough(self, task: TaskSpec, score: float) -> bool:
        """Breakthrough: Executor solved a task type it previously failed."""
        if task.lineage_id is None:
            return False
        if score < 0.70:
            return False
        # Check if lineage task was previously failed
        for f in self.all_failures[-5:]:
            if f.task_id == task.lineage_id and f.attempt_2_score < 0.45:
                return True
        return False

    def _save_log(self, entry: EpisodeLogEntry):
        path = os.path.join(LOG_DIR, f"{entry.episode_id}.json")
        with open(path, "w") as f:
            f.write(entry.model_dump_json(indent=2))

    def _append_full_log(self, record: dict):
        """Incrementally append one record to full_run.json after every episode."""
        path = os.path.join(LOG_DIR, "full_run.json")
        try:
            existing: list = []
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(record)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            print(f"[warn] Could not append to full_run.json: {exc}")

    def get_metrics_summary(self) -> dict:
        if not self.reward_history:
            return {}

        scores = [r["final_reward"] for r in self.reward_history]
        breakthroughs = sum(1 for r in self.reward_history if r["is_breakthrough"])
        in_band = sum(1 for r in self.reward_history if r["in_band"])
        arch_calibration = (
            sum(self.architect_calibration) / len(self.architect_calibration)
            if self.architect_calibration else 0.0
        )

        diversity = self.diversity_tracker.summary()

        return {
            "total_episodes": self.episode_count,
            "avg_final_reward": round(sum(scores) / len(scores), 4),
            "max_reward": round(max(scores), 4),
            "min_reward": round(min(scores), 4),
            "breakthroughs": breakthroughs,
            "in_band_rate": round(in_band / len(self.reward_history), 4),
            "architect_calibration_accuracy": round(arch_calibration, 4),
            "reward_trend": scores[-10:],
            "diversity": diversity,
        }

    def save_full_log(self):
        path = os.path.join(LOG_DIR, "full_run.json")
        with open(path, "w") as f:
            json.dump(self.reward_history, f, indent=2)
        print(f"Full log saved to {path}")
