import json
import re
from core.schemas import ExecutorAction, ArbiterScore, TaskSpec, RegulationShock
from utils.llm_client import complete as llm_complete
from core.llm_client import call_llm

WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "reasoning_transparency": 0.20,
    "efficiency": 0.10,
    "generalization_signal": 0.10,
}

ARBITER_SYSTEM = """Score a procurement compliance analysis on 5 axes.
Weights: correctness=0.35, completeness=0.25, reasoning_transparency=0.20, efficiency=0.10, generalization_signal=0.10
Each axis: float 0.0-1.0. Produce weighted_total and 1-sentence consequence_if_approved.
Return JSON only:
{"correctness":0.0,"completeness":0.0,"reasoning_transparency":0.0,"efficiency":0.0,"generalization_signal":0.0,"weighted_total":0.0,"consequence_if_approved":"..."}"""

# Legacy alias kept for any code that still references the old name
ARBITER_SYSTEM_PROMPT = ARBITER_SYSTEM


def score(
    action: ExecutorAction,
    task: TaskSpec,
    world_coherent: bool = True,
    shock: RegulationShock | None = None,
) -> ArbiterScore:
    """Score an Executor action. Returns ArbiterScore."""

    shock_section = ""
    if shock:
        shock_section = (
            f"\nACTIVE REGULATION SHOCK (Executor was notified):\n"
            f"Type: {shock.shock_type} | Clause: {shock.affected_clause}\n"
            f"New requirement: {shock.new_requirement}\n"
            f"IMPORTANT: Check whether the Executor correctly incorporated this change. "
            f"Failure to address the shock should reduce correctness and completeness scores.\n"
        )

    prompt = f"""TASK:
Domain: {task.domain} | Difficulty: {task.difficulty} | Target axis: {task.target_axis}
Scenario: {task.scenario_context}

CONTRACT/DOCUMENT:
{task.contract_text or 'No document provided'}
{shock_section}
EXECUTOR RESPONSE (Attempt {action.attempt_number}):
Decision: {action.decision}
Reasoning: {action.reasoning}
Violations found: {json.dumps(action.violations_found)}
Supporting evidence: {json.dumps(action.supporting_evidence)}
Recommendation: {action.recommendation or 'None provided'}
Confidence: {action.confidence}

Score this response across all 5 axes. Return JSON only."""

    try:
        raw = call_llm([{"role": "user", "content": prompt}], agent="arbiter", system=ARBITER_SYSTEM)
    except Exception:
        raw = llm_complete(ARBITER_SYSTEM_PROMPT, prompt, max_tokens=450)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback scores if JSON parse fails
        data = {
            "correctness": 0.1,
            "completeness": 0.1,
            "reasoning_transparency": 0.1,
            "efficiency": 0.1,
            "generalization_signal": 0.1,
            "feedback": "Arbiter could not parse response. Minimum scores applied.",
            "what_failed": "Response format unreadable",
            "what_correct_looks_like": "Structured JSON with decision, reasoning, and regulatory citations",
            "world_state_delta": {}
        }

    weighted_total = sum(
        data.get(axis, 0.0) * weight
        for axis, weight in WEIGHTS.items()
    )

    shock_adapted = False
    if shock:
        from core.regulation_shock import RegulationShockEngine
        shock_adapted = RegulationShockEngine().check_executor_adapted(
            action.decision + " " + action.reasoning, shock
        )

    return ArbiterScore(
        task_id=action.task_id,
        attempt_number=action.attempt_number,
        correctness=data.get("correctness", 0.0),
        completeness=data.get("completeness", 0.0),
        reasoning_transparency=data.get("reasoning_transparency", 0.0),
        efficiency=data.get("efficiency", 0.0),
        generalization_signal=data.get("generalization_signal", 0.0),
        weighted_total=round(weighted_total, 4),
        feedback=data.get("feedback", ""),
        what_failed=data.get("what_failed", ""),
        what_correct_looks_like=data.get("what_correct_looks_like", ""),
        consequence_if_approved=data.get("consequence_if_approved", ""),
        world_state_delta=data.get("world_state_delta", {}),
        world_coherent=world_coherent,
        shock_adapted=shock_adapted,
    )


def compute_final_reward(
    score_2: ArbiterScore,
    attempt_1_score: float,
    shock_adapted: bool = False,
) -> float:
    base = score_2.weighted_total

    # Delta reward — learning from feedback
    delta = min((score_2.weighted_total - attempt_1_score) * 0.20, 0.15)

    # World coherence bonus/penalty
    coherence = 0.10 if score_2.world_coherent else -0.10

    # Malformed payload penalty
    malformed = -0.15 if (
        score_2.weighted_total == 0 and
        score_2.reasoning_transparency == 0 and
        score_2.correctness == 0
    ) else 0.0

    # Regulation shock adaptation bonus
    shock_bonus = 0.05 if shock_adapted else 0.0

    total = base + delta + coherence + malformed + shock_bonus
    return round(max(0.0, min(1.0, total)), 4)


def get_weakest_axis(score: ArbiterScore) -> str:
    axes = {
        "correctness": score.correctness,
        "completeness": score.completeness,
        "reasoning_transparency": score.reasoning_transparency,
        "efficiency": score.efficiency,
        "generalization_signal": score.generalization_signal,
    }
    return min(axes, key=axes.get)


def generate_feedback_prompt(score: ArbiterScore) -> str:
    """Return feedback string to give Executor for attempt 2."""
    return (
        f"ARBITER FEEDBACK (Attempt {score.attempt_number}):\n"
        f"Overall score: {score.weighted_total:.2f}\n"
        f"What failed: {score.what_failed}\n"
        f"What a correct response looks like: {score.what_correct_looks_like}\n"
        f"Specific feedback: {score.feedback}\n"
        f"Weakest area: {get_weakest_axis(score)}\n"
        f"Use this feedback to improve your attempt 2 response."
    )
