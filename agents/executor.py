import json
import re
from core.schemas import TaskSpec, ExecutorAction
from utils.llm_client import complete as llm_complete, active_backend
from core.llm_client import call_llm

EXECUTOR_SYSTEM = """Procurement compliance analyst at AXIOM Corp (aerospace/defense).
Detect violations. Frameworks: FAR, DFARS, EU Directive 2014/24/EU.
Violations: expired SAM.gov registration, undisclosed OCI, missing mandatory clauses, ITAR breach, defective cost pricing, debarred vendor, TINA threshold breach.
Return JSON only:
{"decision":"COMPLIANT|NON-COMPLIANT","violations_found":["FAR X.XXX - reason"],"reasoning":"...","confidence":0.0}"""

# Legacy alias
EXECUTOR_SYSTEM_PROMPT = EXECUTOR_SYSTEM


def act(
    task: TaskSpec,
    world_state_text: str,
    feedback: str = None,
    attempt_number: int = 1,
) -> ExecutorAction:
    prompt = (
        f"WORLD STATE:\n{world_state_text}\n\n"
        f"TASK CONTEXT:\n{task.scenario_context}\n\n"
        f"CONTRACT/DOCUMENT:\n{task.contract_text or 'No document provided'}\n\n"
    )
    if attempt_number > 1 and feedback:
        prompt += f"ARBITER FEEDBACK:\n{feedback}\n\n"
    prompt += "Provide your analysis in the required JSON format."

    try:
        try:
            raw = call_llm([{"role": "user", "content": prompt}], agent="executor", system=EXECUTOR_SYSTEM)
        except Exception:
            raw = llm_complete(EXECUTOR_SYSTEM_PROMPT, prompt, max_tokens=600)
        raw = re.sub(r"```json|```", "", raw).strip()
    except Exception as exc:
        return ExecutorAction(
            task_id=task.task_id,
            attempt_number=attempt_number,
            decision="PARSE_ERROR",
            reasoning=f"Executor LLM call failed: {type(exc).__name__}: {exc}",
            confidence=0.0,
            violations_found=[],
            supporting_evidence=[],
            recommendation=None,
        )

    try:
        data = json.loads(raw)
        return ExecutorAction(
            task_id=task.task_id,
            attempt_number=attempt_number,
            decision=data.get("decision", "NON-COMPLIANT"),
            reasoning=data.get("reasoning", ""),
            violations_found=data.get("violations_found", []),
            supporting_evidence=data.get("supporting_evidence", []),
            confidence=float(data.get("confidence", 0.5)),
            recommendation=data.get("recommendation"),
        )
    except Exception:
        return ExecutorAction(
            task_id=task.task_id,
            attempt_number=attempt_number,
            decision="PARSE_ERROR",
            reasoning="Failed to parse response",
            confidence=0.0,
            violations_found=[],
            supporting_evidence=[],
            recommendation=None,
        )
