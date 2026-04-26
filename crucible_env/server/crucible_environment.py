import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

try:
    from openenv.core.environment import Environment
except ImportError:
    from openenv.core.env_server.interfaces import Environment

sys.path.append("..")

try:
    from agents.arbiter import (  # noqa: E402
        compute_final_reward,
        generate_feedback_prompt,
        get_weakest_axis,
        score as arbiter_score,
    )
    from agents.architect import generate as architect_generate  # noqa: E402
    from agents.executor import act as executor_act  # noqa: E402
    from core.schemas import ExecutorAction, FailureRecord, TaskSpec  # noqa: E402
    from core.world_state import WorldStateManager  # noqa: E402
    from domains.procurement.tasks import get_static_task  # noqa: E402
    from models import CrucibleAction, CrucibleObservation, CrucibleState
except ImportError:
    from crucible_env.agents.arbiter import (  # noqa: E402
        compute_final_reward,
        generate_feedback_prompt,
        get_weakest_axis,
        score as arbiter_score,
    )
    from crucible_env.agents.architect import generate as architect_generate  # noqa: E402
    from crucible_env.agents.executor import act as executor_act  # noqa: E402
    from crucible_env.core.schemas import ExecutorAction, FailureRecord, TaskSpec  # noqa: E402
    from crucible_env.core.world_state import WorldStateManager  # noqa: E402
    from crucible_env.domains.procurement.tasks import get_static_task  # noqa: E402
    from crucible_env.models import CrucibleAction, CrucibleObservation, CrucibleState


class CrucibleEnvironment(Environment):
    def __init__(
        self,
        use_architect: bool = False,
        seed: int = 42,
        use_adversarial: bool = False,
        use_shocks: bool = False,
        shock_probability: float = 0.25,
        jurisdiction: str = "FAR",
    ):
        self.world = WorldStateManager(seed)
        self.use_architect = use_architect
        self.use_adversarial = use_adversarial
        self.use_shocks = use_shocks
        self.jurisdiction = jurisdiction
        self.episode_count = 0
        self.all_failures = []
        self.last_scores = []
        self.current_difficulty = "easy"
        self.next_task = None
        self.last_architect_output = None
        self.reward_history = []
        self.architect_calibration = []
        self.breakthrough_count = 0
        self._current_task = None
        self._current_observation = None
        self._active_shock = None
        self._shock_text = ""

        # Lazy imports for optional features
        self._shock_engine = None
        if use_shocks:
            try:
                from core.regulation_shock import RegulationShockEngine
            except ImportError:
                from crucible_env.core.regulation_shock import RegulationShockEngine
            self._shock_engine = RegulationShockEngine(shock_probability, jurisdiction)

    def reset(self) -> CrucibleObservation:
        """Start a new episode. Called at beginning of each training step."""
        self.episode_count += 1
        self.world.reset()
        self._active_shock = None
        self._shock_text = ""

        task = self._get_next_task()

        # Adversarial: Vendor crafts the contract
        if self.use_adversarial:
            try:
                from agents.vendor import craft as vendor_craft
            except ImportError:
                from crucible_env.agents.vendor import craft as vendor_craft
            vendor_action = vendor_craft(task, self.jurisdiction, task.difficulty)
            task = task.model_copy(update={"contract_text": vendor_action.crafted_contract})

        # Regulation shock
        if self._shock_engine:
            shock = self._shock_engine.maybe_fire(self.episode_count)
            if shock:
                self._active_shock = shock
                self._shock_text = self._shock_engine.format_for_executor(shock)
                if shock.world_delta:
                    self.world.apply_delta(shock.world_delta)

        self._current_task = task

        obs = CrucibleObservation(
            world_state_text=self.world.render() + self._shock_text,
            task_description=task.scenario_context,
            contract_text=task.contract_text or "",
            domain=task.domain,
            difficulty=task.difficulty,
            target_axis=task.target_axis,
            attempt_number=1,
            arbiter_feedback="",
            done=False,
            reward=0.0,
        )
        self._current_observation = obs
        return obs

    def step(self, action: CrucibleAction) -> CrucibleObservation:
        # The HTTP server creates a fresh env per request; auto-reset if needed.
        if self._current_task is None:
            self.reset()
        task = self._current_task

        exec_action_1 = ExecutorAction(
            task_id=task.task_id,
            attempt_number=1,
            decision=action.decision,
            reasoning=action.reasoning,
            violations_found=action.violations_found or [],
            supporting_evidence=action.supporting_evidence or [],
            recommendation=action.recommendation,
            confidence=action.confidence,
        )

        world_coherent_1 = self.world.check_coherence(action.decision)
        score_1 = arbiter_score(exec_action_1, task, world_coherent_1, shock=self._active_shock)
        feedback = generate_feedback_prompt(score_1)

        exec_action_2 = executor_act(
            task=task,
            world_state_text=self.world.render() + self._shock_text,
            feedback=feedback,
            attempt_number=2,
        )

        world_coherent_2 = self.world.check_coherence(exec_action_2.decision)
        score_2 = arbiter_score(exec_action_2, task, world_coherent_2, shock=self._active_shock)

        if score_2.world_state_delta:
            self.world.apply_delta(score_2.world_state_delta)

        shock_adapted = bool(self._active_shock and score_2.shock_adapted)
        final_reward = compute_final_reward(score_2, score_1.weighted_total, shock_adapted=shock_adapted)
        score_2.final_reward = final_reward

        is_breakthrough = self._check_breakthrough(task, score_2.weighted_total)
        if is_breakthrough:
            self.breakthrough_count += 1

        weakest = get_weakest_axis(score_2)
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
            breakthrough=is_breakthrough,
        )
        self.all_failures.append(failure)
        self.last_scores.append(final_reward)
        if len(self.last_scores) > 10:
            self.last_scores.pop(0)

        self.reward_history.append(
            {
                "episode": self.episode_count,
                "score_1": score_1.weighted_total,
                "score_2": score_2.weighted_total,
                "final_reward": final_reward,
                "difficulty": task.difficulty,
                "weakest_axis": weakest,
                "is_breakthrough": is_breakthrough,
                "in_band": 0.45 <= score_2.weighted_total <= 0.70,
                "architect_active": self.use_architect,
            }
        )
        self._save_episode_log(failure, final_reward)

        arch_reasoning = ""
        next_difficulty = self.current_difficulty
        lineage_id = ""

        if self.use_architect and len(self.all_failures) >= 3:
            last_3 = self.last_scores[-3:]
            next_task, arch_output = architect_generate(
                failures=self.all_failures,
                last_3_scores=last_3,
                current_difficulty=task.difficulty,
                episode_count=self.episode_count,
            )
            self.next_task = next_task
            self.last_architect_output = arch_output
            arch_reasoning = arch_output.architect_reasoning
            next_difficulty = arch_output.difficulty
            lineage_id = arch_output.lineage_id

            in_band = 0.45 <= score_2.weighted_total <= 0.70
            self.architect_calibration.append(in_band)

        obs = CrucibleObservation(
            world_state_text=self.world.render(),
            task_description=task.scenario_context,
            contract_text=task.contract_text or "",
            domain=task.domain,
            difficulty=task.difficulty,
            target_axis=task.target_axis,
            attempt_number=2,
            arbiter_feedback=feedback,
            score_attempt_1=score_1.weighted_total,
            score_attempt_2=score_2.weighted_total,
            final_reward=final_reward,
            done=True,
            reward=final_reward,
            architect_reasoning=arch_reasoning,
            next_task_difficulty=next_difficulty,
            lineage_id=lineage_id,
            is_breakthrough=is_breakthrough,
            consequence_if_approved=score_2.consequence_if_approved,
            shock_fired=self._active_shock is not None,
            shock_adapted=score_2.shock_adapted,
        )
        self._current_observation = obs
        return obs

    @property
    def state(self) -> CrucibleState:
        arch_cal = (
            sum(self.architect_calibration) / len(self.architect_calibration)
            if self.architect_calibration
            else 0.0
        )
        avg_reward = sum(self.last_scores) / len(self.last_scores) if self.last_scores else 0.0
        weakest = ""
        if self.all_failures:
            axes = [f.weakest_axis for f in self.all_failures[-10:]]
            weakest = Counter(axes).most_common(1)[0][0]

        return CrucibleState(
            episode_id=f"ep-{self.episode_count:04d}",
            step_count=self.episode_count,
            total_episodes=self.episode_count,
            architect_active=self.use_architect,
            current_difficulty=self.current_difficulty,
            avg_reward_last_10=round(avg_reward, 4),
            architect_calibration_accuracy=round(arch_cal, 4),
            breakthrough_count=self.breakthrough_count,
            weakest_axis=weakest,
            axis_scores=self._get_axis_averages(),
        )

    def _get_next_task(self) -> TaskSpec:
        if self.next_task and self.use_architect:
            t = self.next_task
            self.next_task = None
            return t
        difficulties = ["easy", "easy", "easy", "medium", "medium", "hard", "expert"]
        difficulty = difficulties[min(self.episode_count - 1, len(difficulties) - 1)]
        self.current_difficulty = difficulty
        return get_static_task(difficulty, self.episode_count - 1)

    def _check_breakthrough(self, task: TaskSpec, score: float) -> bool:
        if task.lineage_id is None or score < 0.70:
            return False
        for f in self.all_failures[-5:]:
            if f.task_id == task.lineage_id and f.attempt_2_score < 0.45:
                return True
        return False

    def _get_axis_averages(self) -> dict:
        if not self.all_failures:
            return {}
        counts = defaultdict(int)
        for f in self.all_failures[-10:]:
            counts[f.weakest_axis] += 1
        total = len(self.all_failures[-10:])
        return {axis: round(1 - count / total, 3) for axis, count in counts.items()}

    def _save_episode_log(self, failure: FailureRecord, reward: float):
        os.makedirs("data/episode_logs", exist_ok=True)
        log = {
            "episode": self.episode_count,
            "task_id": failure.task_id,
            "difficulty": failure.difficulty,
            "score_1": failure.attempt_1_score,
            "score_2": failure.attempt_2_score,
            "final_reward": reward,
            "weakest_axis": failure.weakest_axis,
            "breakthrough": failure.breakthrough,
            "is_breakthrough": failure.breakthrough,
            "architect_active": self.use_architect,
            "in_band": 0.45 <= failure.attempt_2_score <= 0.70,
            "timestamp": datetime.now().isoformat(),
        }
        path = f"data/episode_logs/ep_{self.episode_count:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        full_log_path = "data/episode_logs/full_run.json"
        full_log = []
        if os.path.exists(full_log_path):
            with open(full_log_path, encoding="utf-8") as f:
                full_log = json.load(f)
        full_log.append(log)
        with open(full_log_path, "w", encoding="utf-8") as f:
            json.dump(full_log, f, indent=2)
