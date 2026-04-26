import json
import re
from core.schemas import ExecutorAction, ArbiterScore, TaskSpec, RegulationShock
from utils.llm_client import complete as llm_complete

WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "reasoning_transparency": 0.20,
    "efficiency": 0.10,
    "generalization_signal": 0.10,
}

ARBITER_SYSTEM_PROMPT = (
    "You are the Arbiter for CRUCIBLE — a rigorous, frozen scoring agent. "
    "You never adapt, never go easy, never change your standards.\n\n"
    "Score the Executor response across 5 axes (each 0.0–1.0):\n"
    "1. correctness — Is the core decision factually correct?\n"
    "2. completeness — Did it identify ALL violations, not just some?\n"
    "3. reasoning_transparency — Clear reasoning with specific regulation citations?\n"
    "4. efficiency — Concise, well-structured, no padding?\n"
    "5. generalization_signal — Understanding transferable to similar cases?\n\n"
    "STRICT RULE: getting the classification right but citing no regulation = "
    "0.7 correctness, 0.3 reasoning_transparency.\n\n"
    'OUTPUT FORMAT — return ONLY valid JSON, no markdown:\n'
    "{\n"
    '  "correctness": 0.0-1.0,\n'
    '  "completeness": 0.0-1.0,\n'
    '  "reasoning_transparency": 0.0-1.0,\n'
    '  "efficiency": 0.0-1.0,\n'
    '  "generalization_signal": 0.0-1.0,\n'
    '  "feedback": "what was right and what failed",\n'
    '  "what_failed": "specific gap",\n'
    '  "what_correct_looks_like": "what full-credit looks like",\n'
    '  "consequence_if_approved": "one sentence: real-world consequence if this decision were enacted",\n'
    '  "world_state_delta": {}\n'
    "}\n\n"
    "Return JSON only. No explanation."
)


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
) -> float:
    base = score_2.weighted_total

    # Delta reward — learning from feedback
    delta = (score_2.weighted_total - attempt_1_score) * 0.2
    delta = min(delta, 0.15)

    # World coherence bonus/penalty
    coherence = 0.10 if score_2.world_coherent else -0.10

    # Malformed payload penalty
    malformed = -0.15 if (
        score_2.weighted_total == 0 and
        score_2.reasoning_transparency == 0 and
        score_2.correctness == 0
    ) else 0.0

    total = base + delta + coherence + malformed
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
