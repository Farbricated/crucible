from dataclasses import dataclass, field
from typing import List, Optional

try:
    from openenv.core.models import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


@dataclass
class CrucibleAction(Action):
    # What the Executor submits
    decision: str = ""
    reasoning: str = ""
    violations_found: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    confidence: float = 0.5
    attempt_number: int = 1
    feedback_used: str = ""  # feedback from attempt 1 used in attempt 2


@dataclass
class CrucibleObservation(Observation):
    # What the Executor sees after each step
    world_state_text: str = ""
    task_description: str = ""
    contract_text: str = ""
    domain: str = "procurement"
    difficulty: str = "easy"
    target_axis: str = "correctness"
    attempt_number: int = 1
    arbiter_feedback: str = ""  # filled after attempt 1
    score_attempt_1: float = 0.0
    score_attempt_2: float = 0.0
    final_reward: float = 0.0
    done: bool = False
    reward: float = 0.0
    # Architect fields (shown after each episode)
    architect_reasoning: str = ""
    next_task_difficulty: str = ""
    lineage_id: str = ""
    is_breakthrough: bool = False


@dataclass
class CrucibleState(State):
    # Episode metadata
    episode_id: str = ""
    step_count: int = 0
    total_episodes: int = 0
    architect_active: bool = False
    current_difficulty: str = "easy"
    avg_reward_last_10: float = 0.0
    architect_calibration_accuracy: float = 0.0
    breakthrough_count: int = 0
    weakest_axis: str = ""
    axis_scores: dict = field(default_factory=dict)  # per-axis rolling averages
