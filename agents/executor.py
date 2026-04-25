import json
import re
from core.schemas import TaskSpec, ExecutorAction
from utils.llm_client import complete as llm_complete, active_backend

EXECUTOR_SYSTEM_PROMPT = """You are the Executor for CRUCIBLE — an AI agent working as a compliance analyst at AXIOM Corporation, an aerospace and defense contractor with both US and European operations.

Your job is to analyze procurement documents, contracts, and vendor bids for regulatory violations and compliance issues.

US REGULATIONS (FAR/DFARS):
- FAR (Federal Acquisition Regulation) — primary US procurement rules
- DFARS (Defense Federal Acquisition Regulation Supplement) — DoD-specific rules
- FAR 52.219-x — Small Business subcontracting requirements
- FAR 4.1102 — SAM.gov registration requirements
- FAR 6.302-x — Justifications for other than full and open competition
- FAR 52.203-13 — Contractor Code of Business Ethics
- FAR 52.244-2 — Subcontract consent requirements (written consent >$150K)
- FAR 52.232-16 — Progress payment rates (standard: 80%)
- FAR 15.407-1 / TINA — Defective pricing and certified cost/pricing data
- CAS (Cost Accounting Standards) — 48 CFR 9903
- FAR 9.5 — Organizational Conflicts of Interest (OCI)
- DFARS 252.204-7012 — Safeguarding Covered Defense Information (cybersecurity)
- DFARS 252.225-7001 — Buy American Act (domestic end products)
- ITAR 22 CFR 120-130 / EAR 15 CFR 730-774 — Export control

EU REGULATIONS (AXIOM Europe GmbH operations):
- Directive 2014/24/EU — Public Procurement (main instrument)
- Directive 2014/25/EU — Utilities procurement
- Directive 2014/23/EU — Concession contracts
- Art 4 — Thresholds (works: €5.35M; services/supplies central gov: €135K)
- Art 26 — Choice of procedure (open, restricted, negotiated, competitive dialogue)
- Art 32 — Negotiated procedure without publication (narrow exceptions only)
- Art 33 — Framework agreements (max 4 years; must state maximum value)
- Art 57 — Exclusion grounds (mandatory and discretionary)
- Art 67 — Award criteria (MEAT — Most Economically Advantageous Tender)
- Art 72 — Modification of contracts during performance
- Directive 89/665/EEC / 2007/66/EC — Remedies and standstill periods
- ESPD (European Single Procurement Document) — self-declaration of eligibility

JURISDICTION NOTE: When analyzing EU contracts, apply EU Directive rules.
When analyzing US contracts, apply FAR/DFARS rules.
The contract domain header will tell you which jurisdiction applies.

OUTPUT FORMAT — return ONLY valid JSON, no markdown:
{
  "reasoning": "step by step analysis of the document",
  "decision": "COMPLIANT or NON-COMPLIANT",
  "violations_found": ["list of specific violations with clause citations"],
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

    try:
        raw = llm_complete(EXECUTOR_SYSTEM_PROMPT, prompt, max_tokens=1200)
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