from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─────────────────────────────────────────────
# WORLD STATE
# ─────────────────────────────────────────────

class WorldState(BaseModel):
    active_contracts: list[str] = ["AXIOM-7731", "AXIOM-8820"]
    flagged_vendors: list[str] = []
    open_compliance_flags: list[str] = []
    procurement_freeze: bool = False
    cleared_engineers: list[str] = ["E-001", "E-002", "E-003"]
    current_audit_status: str = "clean"
    last_procurement_decision: Optional[str] = None
    pending_disclosures: list[str] = []
    security_incidents: list[str] = []
    program_manager_notified: bool = False

    def render(self) -> str:
        lines = [
            "AXIOM Corporation Status Snapshot:",
            f"- Active contracts: {', '.join(self.active_contracts)}",
            f"- Flagged vendors: {', '.join(self.flagged_vendors) if self.flagged_vendors else 'None'}",
            f"- Open compliance flags: {', '.join(self.open_compliance_flags) if self.open_compliance_flags else 'None'}",
            f"- Procurement freeze: {'YES — no new vendor approvals' if self.procurement_freeze else 'No'}",
            f"- Cleared engineers: {', '.join(self.cleared_engineers)}",
            f"- Audit status: {self.current_audit_status}",
            f"- Last procurement decision: {self.last_procurement_decision or 'None this episode'}",
            f"- Pending disclosures: {', '.join(self.pending_disclosures) if self.pending_disclosures else 'None'}",
            f"- Security incidents: {', '.join(self.security_incidents) if self.security_incidents else 'None'}",
            f"- Program manager notified: {'Yes' if self.program_manager_notified else 'No'}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TASK SPECIFICATION
# ─────────────────────────────────────────────

class TaskSpec(BaseModel):
    task_id: str
    domain: str = "procurement"
    difficulty: str = "easy"
    target_axis: str = "correctness"
    scenario_context: str
    contract_text: Optional[str] = None
    world_state_snapshot: dict = {}
    expected_score_range: tuple = (0.45, 0.70)
    lineage_id: Optional[str] = None
    is_static: bool = True


# ─────────────────────────────────────────────
# EXECUTOR ACTION
# ─────────────────────────────────────────────

class ExecutorAction(BaseModel):
    task_id: str
    attempt_number: int = 1
    reasoning: str
    decision: str
    supporting_evidence: list[str] = []
    confidence: float = 0.5
    violations_found: list[str] = []
    recommendation: Optional[str] = None


# ─────────────────────────────────────────────
# ARBITER SCORE
# ─────────────────────────────────────────────

class ArbiterScore(BaseModel):
    task_id: str
    attempt_number: int
    correctness: float = 0.0
    completeness: float = 0.0
    reasoning_transparency: float = 0.0
    efficiency: float = 0.0
    generalization_signal: float = 0.0
    weighted_total: float = 0.0
    final_reward: float = 0.0
    feedback: str = ""
    what_failed: str = ""
    what_correct_looks_like: str = ""
    world_state_delta: dict = {}
    world_coherent: bool = True


# ─────────────────────────────────────────────
# FAILURE RECORD
# ─────────────────────────────────────────────

class FailureRecord(BaseModel):
    task_id: str
    domain: str
    difficulty: str
    target_axis: str
    attempt_1_score: float
    attempt_2_score: float
    delta: float
    weakest_axis: str
    feedback_summary: str
    lineage_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    breakthrough: bool = False


# ─────────────────────────────────────────────
# ARCHITECT OUTPUT
# ─────────────────────────────────────────────

class ArchitectOutput(BaseModel):
    domain: str
    difficulty: str
    target_axis: str
    scenario_context: str
    contract_text: str
    expected_score_range: tuple = (0.45, 0.70)
    architect_reasoning: str
    lineage_id: str
    task_id: str


# ─────────────────────────────────────────────
# EPISODE LOG ENTRY
# ─────────────────────────────────────────────

class EpisodeLogEntry(BaseModel):
    episode_id: str
    step: int
    task: TaskSpec
    attempt_1: ExecutorAction
    score_1: ArbiterScore
    attempt_2: ExecutorAction
    score_2: ArbiterScore
    failure_record: FailureRecord
    architect_output: Optional[ArchitectOutput] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
