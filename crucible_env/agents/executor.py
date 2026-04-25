import os
import json
import re
from anthropic import Anthropic
try:
    from core.schemas import TaskSpec, ExecutorAction
except ImportError:
    from crucible_env.core.schemas import TaskSpec, ExecutorAction

client = Anthropic()

EXECUTOR_SYSTEM_PROMPT = """You are the Executor for CRUCIBLE — an AI agent working as a compliance analyst at AXIOM Corporation, an aerospace and defense contractor.

Your job is to analyze procurement documents, contracts, and vendor bids for regulatory violations and compliance issues.

Key regulations you must know:
- FAR (Federal Acquisition Regulation) — primary procurement rules
- DFARS (Defense Federal Acquisition Regulation Supplement) — DoD-specific rules
- FAR 52.219-x — Small Business subcontracting requirements
- FAR 4.1102 — SAM.gov registration requirements
- FAR 6.302-x — Justifications for other than full and open competition
- FAR 52.203-13 — Contractor Code of Business Ethics
- FAR 52.244-2 — Subcontract consent requirements
- FAR 52.232-16 — Progress payment rates (standard: 80%)
- FAR 15.407-1 — Defective pricing
- CAS (Cost Accounting Standards) — 48 CFR 9903
- TINA (Truth in Negotiations Act) — certified cost/pricing data
- FAR 9.5 — Organizational Conflicts of Interest (OCI)

OUTPUT FORMAT — return ONLY valid JSON, no markdown:
{
  "reasoning": "step by step analysis of the document",
  "decision": "COMPLIANT or NON-COMPLIANT",
  "violations_found": ["list of specific violations with FAR clause citations"],
  "supporting_evidence": ["key text passages that support your decision"],
  "confidence": 0.0-1.0,
  "recommendation": "specific action to take"
}
"""


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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=EXECUTOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
        violations = data.get("violations_found", [])
        evidence = data.get("supporting_evidence", [])
        if not isinstance(violations, list):
            violations = []
        if not isinstance(evidence, list):
            evidence = []
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return ExecutorAction(
            task_id=task.task_id,
            attempt_number=attempt_number,
            decision=data.get("decision", "NON-COMPLIANT"),
            reasoning=data.get("reasoning", ""),
            violations_found=violations,
            supporting_evidence=evidence,
            confidence=confidence,
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