import json
import re
from core.schemas import TaskSpec, ExecutorAction
from utils.llm_client import complete as llm_complete, active_backend
from utils.clauses import build_executor_clause_block

# ── System prompt ──────────────────────────────────────────────
# Static string — never use f-strings here.
# Clause details come from utils/clauses.py (single source of truth).
# Computed once at module load from CLAUSE_REGISTRY; treated as a constant.
_CLAUSE_BLOCK = build_executor_clause_block()

EXECUTOR_SYSTEM_PROMPT = (
    "You are the Executor for CRUCIBLE — a compliance analyst at AXIOM Corporation, "
    "an aerospace and defense contractor with US and European operations.\n\n"
    "Your job: analyze procurement documents for regulatory violations.\n\n"
    "JURISDICTION NOTE: Apply FAR/DFARS rules for US contracts, EU Directive rules for EU contracts. "
    "The contract header will specify the jurisdiction.\n"
    + _CLAUSE_BLOCK
    + "\n\n"
    "OUTPUT FORMAT — return ONLY valid JSON, no markdown:\n"
    "{\n"
    '  "reasoning": "step by step analysis",\n'
    '  "decision": "COMPLIANT or NON-COMPLIANT",\n'
    '  "violations_found": ["specific violations with clause citations"],\n'
    '  "supporting_evidence": ["key text passages supporting your decision"],\n'
    '  "confidence": 0.0-1.0,\n'
    '  "recommendation": "specific corrective action"\n'
    "}\n\n"
    "Return JSON only. No explanation."
)


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
