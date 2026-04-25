import json
import re
import uuid
from anthropic import Anthropic
from core.schemas import FailureRecord, ArchitectOutput, TaskSpec

client = Anthropic()

ARCHITECT_SYSTEM_PROMPT = """You are the Architect for CRUCIBLE — a curriculum designer whose only job is to generate the next training task for the Executor.

You read the Executor's failure history and generate a task that:
1. Targets the Executor's WEAKEST scoring axis
2. Is calibrated to land the Executor in the 0.45–0.70 scoring band (productive learning zone)
3. Is harder than what the Executor has been getting right easily

THE BAND RULES:
- Below 0.20 = too hard, no learning signal — DO NOT generate tasks this hard
- 0.45–0.70 = productive learning zone — THIS IS YOUR TARGET
- Above 0.90 = too easy, no learning signal — DO NOT generate tasks this easy

ESCALATION TECHNIQUES for procurement domain:
- Hide violations inside legitimate-looking contract language
- Chain two regulatory frameworks that conflict
- Add a vendor with undisclosed prior violations
- Introduce an OCI (conflict of interest) that is subtle
- Use correct FAR clause numbers but wrong applicability thresholds

OUTPUT FORMAT — return ONLY valid JSON, no markdown:
{
  "domain": "procurement",
  "difficulty": "easy|medium|hard|expert",
  "target_axis": "the weakest axis you are targeting",
  "scenario_context": "plain English task instruction for the Executor",
  "contract_text": "the full contract/document text the Executor must analyze",
  "expected_score_range": [0.45, 0.70],
  "architect_reasoning": "plain English explanation of WHY you generated this task and HOW it targets the weakness",
  "lineage_id": "the failure_id that triggered this task"
}"""


def _compute_axis_averages(failures: list[FailureRecord]) -> dict:
    """Compute average score per axis across recent failures."""
    if not failures:
        return {
            "correctness": 0.5,
            "completeness": 0.5,
            "reasoning_transparency": 0.5,
            "efficiency": 0.5,
            "generalization_signal": 0.5
        }

    # We track weakest_axis per failure — count frequency
    axis_counts = {}
    for f in failures:
        axis = f.weakest_axis
        axis_counts[axis] = axis_counts.get(axis, 0) + 1

    # Normalize
    total = len(failures)
    return {
        "correctness": 1.0 - axis_counts.get("correctness", 0) / total,
        "completeness": 1.0 - axis_counts.get("completeness", 0) / total,
        "reasoning_transparency": 1.0 - axis_counts.get("reasoning_transparency", 0) / total,
        "efficiency": 1.0 - axis_counts.get("efficiency", 0) / total,
        "generalization_signal": 1.0 - axis_counts.get("generalization_signal", 0) / total,
    }


def _adjust_difficulty(last_3_scores: list[float], current_difficulty: str) -> str:
    """Rule-based difficulty adjustment before LLM call."""
    order = ["easy", "medium", "hard", "expert"]
    idx = order.index(current_difficulty) if current_difficulty in order else 1

    if not last_3_scores:
        return current_difficulty

    avg = sum(last_3_scores) / len(last_3_scores)

    if avg > 0.85 and idx < 3:
        return order[idx + 1]   # bump up
    if avg < 0.20 and idx > 0:
        return order[idx - 1]   # bump down
    return current_difficulty


def generate(
    failures: list[FailureRecord],
    last_3_scores: list[float],
    current_difficulty: str = "easy",
    episode_count: int = 0,
) -> tuple[TaskSpec, ArchitectOutput]:
    """Generate next task from failure history. Returns (TaskSpec, ArchitectOutput)."""

    axis_averages = _compute_axis_averages(failures)
    weakest_axis = min(axis_averages, key=axis_averages.get)
    adjusted_difficulty = _adjust_difficulty(last_3_scores, current_difficulty)

    # Most recent failure as lineage anchor
    lineage_id = failures[-1].task_id if failures else "bootstrap"

    # Serialize recent failures for prompt
    failure_summary = []
    for f in failures[-10:]:
        failure_summary.append({
            "task_id": f.task_id,
            "difficulty": f.difficulty,
            "weakest_axis": f.weakest_axis,
            "attempt_1_score": f.attempt_1_score,
            "attempt_2_score": f.attempt_2_score,
            "delta": f.delta,
            "feedback_summary": f.feedback_summary[:200],
        })

    prompt = f"""EXECUTOR FAILURE HISTORY:
Episode count: {episode_count}
Recent failures (last 10): {json.dumps(failure_summary, indent=2)}

Axis averages (lower = weaker):
{json.dumps(axis_averages, indent=2)}

Weakest axis identified: {weakest_axis}
Last 3 scores: {last_3_scores}
Recommended difficulty after rule-based adjustment: {adjusted_difficulty}

Generate the next procurement task targeting the weakest axis.
Target score band: 0.45–0.70.
Lineage ID to reference: {lineage_id}

Return JSON only."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=ARCHITECT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback — generate a medium hard task
        data = {
            "domain": "procurement",
            "difficulty": adjusted_difficulty,
            "target_axis": weakest_axis,
            "scenario_context": "Review this contract modification for FAR compliance violations. Identify all issues.",
            "contract_text": "CONTRACT MOD — AXIOM-FALLBACK\nValue: $2.1M\nSAM.gov: Current\nSmall business goal: 20% committed, 9% actual\nFAR 52.203-13: Missing\nProgress payments: 85% (standard is 80%, no justification)",
            "expected_score_range": [0.45, 0.70],
            "architect_reasoning": f"Fallback task generated. Targeting {weakest_axis}.",
            "lineage_id": lineage_id
        }

    task_id = f"arch-{uuid.uuid4().hex[:8]}"

    architect_output = ArchitectOutput(
        domain=data.get("domain", "procurement"),
        difficulty=data.get("difficulty", adjusted_difficulty),
        target_axis=data.get("target_axis", weakest_axis),
        scenario_context=data.get("scenario_context", ""),
        contract_text=data.get("contract_text", ""),
        expected_score_range=tuple(data.get("expected_score_range", [0.45, 0.70])),
        architect_reasoning=data.get("architect_reasoning", ""),
        lineage_id=data.get("lineage_id", lineage_id),
        task_id=task_id
    )

    task_spec = TaskSpec(
        task_id=task_id,
        domain=architect_output.domain,
        difficulty=architect_output.difficulty,
        target_axis=architect_output.target_axis,
        scenario_context=architect_output.scenario_context,
        contract_text=architect_output.contract_text,
        expected_score_range=architect_output.expected_score_range,
        lineage_id=architect_output.lineage_id,
        is_static=False
    )

    return task_spec, architect_output


def compute_architect_reward(executor_score: float, is_breakthrough: bool = False, domain_seen_recently: bool = True) -> float:
    """Separate reward signal for Architect — tracks calibration quality."""
    reward = 0.0

    if 0.45 <= executor_score <= 0.70:
        reward += 0.15   # calibration hit
    elif executor_score < 0.20:
        reward -= 0.10   # too hard
    elif executor_score > 0.90:
        reward -= 0.10   # too easy

    if not domain_seen_recently:
        reward += 0.05   # diversity bonus

    if is_breakthrough:
        reward += 0.10   # lineage bonus

    return round(reward, 4)
