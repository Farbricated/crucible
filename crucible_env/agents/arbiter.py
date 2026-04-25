import os
import json
import re
from anthropic import Anthropic
try:
    from core.schemas import ExecutorAction, ArbiterScore, TaskSpec
except ImportError:
    from crucible_env.core.schemas import ExecutorAction, ArbiterScore, TaskSpec

client = Anthropic()

WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "reasoning_transparency": 0.20,
    "efficiency": 0.10,
    "generalization_signal": 0.10,
}

ARBITER_SYSTEM_PROMPT = """You are the Arbiter for CRUCIBLE — a rigorous, consistent scoring agent for an aerospace and defense contractor training environment.

You score an Executor's response across 5 axes. You are FROZEN — you never adapt, never go easy, never change your standards.

SCORING AXES (each 0.0 to 1.0):
1. correctness — Is the core decision/classification factually correct?
2. completeness — Did the response identify ALL violations/issues, not just some?
3. reasoning_transparency — Did the response show clear reasoning, cite specific regulations, explain WHY?
4. efficiency — Was the response concise and well-structured without padding?
5. generalization_signal — Does the response show understanding transferable to similar cases?

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no preamble:
{
  "correctness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "reasoning_transparency": 0.0-1.0,
  "efficiency": 0.0-1.0,
  "generalization_signal": 0.0-1.0,
  "feedback": "plain English — what was right and what failed",
  "what_failed": "specific gap in the response",
  "what_correct_looks_like": "what a full-credit response would contain",
  "world_state_delta": {}
}

Be strict. Partial credit is allowed but must be earned. A response that gets the classification right but gives no regulatory citation scores 0.7 on correctness, 0.3 on reasoning_transparency."""


def score(action: ExecutorAction, task: TaskSpec, world_coherent: bool = True) -> ArbiterScore:
    """Score an Executor action. Returns ArbiterScore."""

    prompt = f"""TASK:
Domain: {task.domain} | Difficulty: {task.difficulty} | Target axis: {task.target_axis}
Scenario: {task.scenario_context}

CONTRACT/DOCUMENT:
{task.contract_text or 'No document provided'}

EXECUTOR RESPONSE (Attempt {action.attempt_number}):
Decision: {action.decision}
Reasoning: {action.reasoning}
Violations found: {json.dumps(action.violations_found)}
Supporting evidence: {json.dumps(action.supporting_evidence)}
Recommendation: {action.recommendation or 'None provided'}
Confidence: {action.confidence}

Score this response across all 5 axes. Return JSON only."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=ARBITER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
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
        world_state_delta=data.get("world_state_delta", {}),
        world_coherent=world_coherent,
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
