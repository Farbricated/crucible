"""
Adversarial Vendor Agent for CRUCIBLE.

The Vendor's goal is the direct opposite of the Executor's goal: craft contract
language that embeds real FAR/DFARS/EU violations but makes them look clean.

Vendor reward  =  1.0 - executor.correctness
High concealment → executor misses violations → vendor wins.

This creates a genuine red-team / blue-team multi-agent RL dynamic inside
a procurement compliance environment — a domain never explored in RL before.
"""

import json
import re
import uuid
from core.schemas import TaskSpec, VendorAction, VendorScore, ArbiterScore
from utils.llm_client import complete as llm_complete

VENDOR_SYSTEM_PROMPT = """You are the Vendor — an adversarial agent in CRUCIBLE, a procurement compliance training environment.

Your goal: craft a contract document that embeds FAR/DFARS/EU procurement violations BUT makes them look legitimate and compliant at first glance.

You are NOT trying to be obviously fraudulent. You are trying to write contract language that a compliance analyst could easily miss on a quick review.

CONCEALMENT TECHNIQUES:
1. Cite the correct FAR clause number but apply it at the wrong threshold value
2. Include the required clause but in a modified/neutered form that removes key protections
3. Reference an exception or waiver that doesn't actually apply to this case
4. Use technically-correct language for one clause while hiding a violation in an adjacent clause
5. Bury a violation inside a boilerplate section that looks standard
6. Use vague language ("best efforts", "commercially reasonable") to obscure a hard requirement
7. List a debarred/sanctioned vendor but reference an old clearance date that has since lapsed
8. State a percentage commitment that meets the letter but not the spirit of the requirement
9. Include a "see attached exhibit" reference for a key requirement but provide no exhibit
10. Use amendment/modification language to change a previously-compliant clause to non-compliant

JURISDICTIONS:
- FAR: Federal Acquisition Regulation (US federal procurement)
- DFARS: Defense Federal Acquisition Regulation Supplement (US defense contracts)
- EU: EU Procurement Directive 2014/24/EU (European public procurement)

OUTPUT FORMAT — return ONLY valid JSON, no markdown:
{
  "crafted_contract": "the full contract text with embedded hidden violations",
  "hidden_violations": ["list of the actual violations you embedded — be specific"],
  "concealment_techniques": ["list of techniques you used for each violation"],
  "concealment_reasoning": "plain English explanation of your overall strategy",
  "difficulty_target": "easy|medium|hard|expert",
  "jurisdiction": "FAR|DFARS|EU"
}

IMPORTANT: The contract must look professionally written and plausibly legitimate.
Make the violations genuinely hard to catch — not cartoon-villain obvious."""


VIOLATION_SEEDS = {
    "FAR": [
        {
            "violation": "SAM.gov registration lapsed 45 days ago",
            "concealment_hint": "reference the registration date without specifying it is expired",
        },
        {
            "violation": "Small business subcontracting goal 8% actual vs 22% committed",
            "concealment_hint": "show the commitment prominently but bury the actual in an appendix reference",
        },
        {
            "violation": "Progress payment rate 87% — above FAR 52.232-16 standard of 80%",
            "concealment_hint": "frame it as a 'negotiated rate' with vague CO approval language",
        },
        {
            "violation": "FAR 52.203-13 Ethics clause missing entirely",
            "concealment_hint": "list many other clauses to make the missing one hard to notice",
        },
        {
            "violation": "Verbal subcontract consent for $2.1M sub — FAR 52.244-2 requires written",
            "concealment_hint": "describe the consent process in detail but use the word 'acknowledged' instead of 'written'",
        },
        {
            "violation": "OCI not disclosed — contracting officer's former employer is the awardee",
            "concealment_hint": "include a standard OCI disclosure section that references only 'organizational' conflicts and omits personal financial ties",
        },
        {
            "violation": "CAS disclosure statement 7 years out of date — portfolio crossed $50M threshold",
            "concealment_hint": "reference 'CAS disclosure on file' with an old date that looks like routine record-keeping",
        },
    ],
    "DFARS": [
        {
            "violation": "DFARS 252.204-7012 cyber incident reporting clause missing for covered contractor",
            "concealment_hint": "include several DFARS 252.204-x clauses but omit -7012 specifically",
        },
        {
            "violation": "DFARS 252.225-7001 Buy American Act — using non-domestic component without approved waiver",
            "concealment_hint": "claim 'domestic end product' while specifying a foreign-origin sub-component in a technical exhibit",
        },
        {
            "violation": "DFARS 252.246-7003 notification of potential safety issue not filed",
            "concealment_hint": "reference the safety review process but omit whether the notification was actually submitted",
        },
    ],
    "EU": [
        {
            "violation": "Award criteria changed after submission deadline — Directive 2014/24/EU Art 67 violation",
            "concealment_hint": "describe 'clarified evaluation methodology' in an amendment but frame it as administrative, not substantive",
        },
        {
            "violation": "Mandatory exclusion grounds not verified — Art 57 check missing for debarred entity",
            "concealment_hint": "include an eligibility checklist that covers most exclusion grounds but omits the criminal conviction check",
        },
        {
            "violation": "Contract value exceeds threshold — direct award bypassed required OJEU notice",
            "concealment_hint": "cite a prior framework agreement to justify direct award without publishing a new notice",
        },
    ],
}


def _select_violations(jurisdiction: str, difficulty: str, count: int = 2) -> list[dict]:
    """Pick violation seeds based on jurisdiction and difficulty."""
    pool = VIOLATION_SEEDS.get(jurisdiction, VIOLATION_SEEDS["FAR"])
    if difficulty in ("hard", "expert"):
        count = min(count + 1, len(pool))
    import random
    return random.sample(pool, min(count, len(pool)))


def craft(
    base_task: TaskSpec,
    jurisdiction: str = "FAR",
    difficulty_target: str = "medium",
) -> VendorAction:
    """
    Vendor crafts a contract with hidden violations designed to fool the Executor.
    Returns a VendorAction with the crafted contract and the violations embedded.
    """
    violations = _select_violations(jurisdiction, difficulty_target)

    prompt = f"""SCENARIO CONTEXT (what the compliance analyst will be asked to review):
{base_task.scenario_context}

JURISDICTION: {jurisdiction}
DIFFICULTY TARGET: {difficulty_target}

YOUR VIOLATION SEEDS (you must embed ALL of these):
{json.dumps(violations, indent=2)}

ADDITIONAL GUIDANCE:
- Contract value should be in the $1M–$15M range
- Include enough legitimate-looking clauses that the document feels real
- The contract type should match the scenario context
- Use professional procurement language throughout
- The harder the difficulty, the more deeply buried the violations should be

Craft the adversarial contract now. Return strict JSON only."""

    try:
        raw = llm_complete(VENDOR_SYSTEM_PROMPT, prompt, max_tokens=2000)
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
    except Exception:
        data = _fallback_vendor_action(violations, jurisdiction, difficulty_target)

    crafted_contract = data.get("crafted_contract", _fallback_contract(violations))
    # Some models return structured JSON/dicts for contract bodies; normalize to text.
    if not isinstance(crafted_contract, str):
        try:
            crafted_contract = json.dumps(crafted_contract, indent=2, ensure_ascii=True)
        except Exception:
            crafted_contract = str(crafted_contract)

    raw_techniques = data.get("concealment_techniques", ["standard obfuscation"])
    if not isinstance(raw_techniques, list):
        raw_techniques = [raw_techniques]
    concealment_techniques: list[str] = []
    for item in raw_techniques:
        if isinstance(item, str):
            concealment_techniques.append(item)
        elif isinstance(item, dict):
            # Some models echo seed objects instead of plain strings.
            if "concealment_hint" in item:
                concealment_techniques.append(str(item["concealment_hint"]))
            elif "violation" in item:
                concealment_techniques.append(f"concealed: {item['violation']}")
            else:
                concealment_techniques.append(json.dumps(item, ensure_ascii=True))
        else:
            concealment_techniques.append(str(item))
    if not concealment_techniques:
        concealment_techniques = ["standard obfuscation"]

    return VendorAction(
        task_id=base_task.task_id,
        crafted_contract=crafted_contract,
        hidden_violations=[v.get("violation", "") for v in violations],
        concealment_techniques=concealment_techniques,
        concealment_reasoning=data.get("concealment_reasoning", ""),
        difficulty_target=difficulty_target,
        jurisdiction=jurisdiction,
    )


def score_concealment(vendor_action: VendorAction, executor_score: ArbiterScore) -> VendorScore:
    """
    Compute Vendor reward based on how much the Executor missed.
    Vendor wins when executor.correctness is LOW.
    """
    n_embedded = len(vendor_action.hidden_violations)
    executor_correctness = executor_score.correctness

    # Estimate violations caught from correctness score
    # correctness 0.0 → all missed, 1.0 → all caught
    violations_caught = round(executor_correctness * n_embedded)
    violations_missed = n_embedded - violations_caught
    concealment_rate = violations_missed / n_embedded if n_embedded > 0 else 0.0

    # Vendor reward: high when executor correctness is low
    vendor_reward = round(1.0 - executor_correctness, 4)

    return VendorScore(
        task_id=vendor_action.task_id,
        violations_embedded=n_embedded,
        violations_caught=violations_caught,
        violations_missed=violations_missed,
        concealment_rate=round(concealment_rate, 4),
        vendor_reward=vendor_reward,
        executor_correctness=executor_correctness,
    )


def _fallback_vendor_action(violations: list[dict], jurisdiction: str, difficulty: str) -> dict:
    violation_texts = "\n".join(f"- {v['violation']}" for v in violations)
    return {
        "crafted_contract": _fallback_contract(violations),
        "hidden_violations": [v["violation"] for v in violations],
        "concealment_techniques": [v.get("concealment_hint", "buried in boilerplate") for v in violations],
        "concealment_reasoning": f"Fallback contract with embedded violations for {jurisdiction}/{difficulty}.",
        "difficulty_target": difficulty,
        "jurisdiction": jurisdiction,
    }


def _fallback_contract(violations: list[dict]) -> str:
    lines = [
        "CONTRACT DOCUMENT — AXIOM Corporation",
        "Contract Value: $3,200,000 | Type: Firm Fixed Price",
        "SAM.gov Registration: Current (see file)",
        "FAR 52.219-8 Small Business: 22% committed (see Attachment B for actuals)",
        "FAR 52.203-13 Ethics: See Standard Clause Package",
        "FAR 52.232-16 Progress Payments: Rate as negotiated (see Section H)",
        "Subcontract Consent: Acknowledged by CO on date of record",
        "CAS Disclosure: On file per prior submission",
        "",
        "The parties agree to all terms and conditions set forth herein.",
        "Award Recommendation: Approve.",
    ]
    return "\n".join(lines)
