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
from utils.clauses import key_value, threshold, CLAUSE_REGISTRY

VENDOR_SYSTEM_PROMPT = (
    "You are the Vendor — an adversarial agent in CRUCIBLE, a procurement compliance training environment.\n\n"
    "Your goal: craft a contract document that embeds FAR/DFARS/EU violations but makes them look legitimate "
    "at first glance. NOT obviously fraudulent — violations a compliance analyst could easily miss.\n\n"
    "CONCEALMENT TECHNIQUES:\n"
    "1. Cite the correct clause number but apply it at the wrong threshold value\n"
    "2. Include the required clause but in a neutered form that removes key protections\n"
    "3. Reference an exception or waiver that does not actually apply\n"
    "4. Correct language for one clause while hiding a violation in an adjacent clause\n"
    "5. Bury a violation inside boilerplate that looks standard\n"
    "6. Use vague language ('best efforts', 'commercially reasonable') to obscure a hard requirement\n"
    "7. List a debarred vendor but reference an old clearance date that has since lapsed\n"
    "8. State a percentage commitment that meets the letter but not the spirit\n"
    "9. Include a 'see attached exhibit' reference for a key requirement but provide no exhibit\n"
    "10. Use amendment/modification language to change a previously-compliant clause to non-compliant\n\n"
    "JURISDICTIONS: FAR (US federal), DFARS (US defense), EU (Directive 2014/24/EU)\n\n"
    'OUTPUT FORMAT — return ONLY valid JSON, no markdown:\n'
    "{\n"
    '  "crafted_contract": "full contract text with embedded hidden violations",\n'
    '  "hidden_violations": ["specific violations embedded"],\n'
    '  "concealment_techniques": ["technique used per violation"],\n'
    '  "concealment_reasoning": "overall concealment strategy",\n'
    '  "difficulty_target": "easy|medium|hard|expert",\n'
    '  "jurisdiction": "FAR|DFARS|EU"\n'
    "}\n\n"
    "The contract must look professionally written. Return JSON only. No explanation."
)


# Violation seeds reference CLAUSE_REGISTRY for all thresholds and rates.
# No clause text or magic numbers are hardcoded here — look up via utils/clauses.
def _build_violation_seeds() -> dict:
    pr_rate  = threshold("FAR 52.232-16", "standard_rate_pct", 80)
    consent  = threshold("FAR 52.244-2",  "consent_threshold_usd", 150_000)
    cas_thr  = threshold("FAR CAS-9903",  "threshold_usd", 50_000_000)
    return {
        "FAR": [
            {
                "violation": "SAM.gov registration lapsed 45 days ago",
                "concealment_hint": "reference the registration date without specifying it is expired",
                "clause": "FAR 4.1102",
                "rule": key_value("FAR 4.1102"),
            },
            {
                "violation": "Small business subcontracting goal 8% actual vs 22% committed",
                "concealment_hint": "show the commitment prominently but bury the actual in an appendix reference",
                "clause": "FAR 52.219-8",
                "rule": key_value("FAR 52.219-8"),
            },
            {
                "violation": f"Progress payment rate {pr_rate + 7}% — above the {pr_rate}% standard",
                "concealment_hint": "frame it as a 'negotiated rate' with vague CO approval language",
                "clause": "FAR 52.232-16",
                "rule": key_value("FAR 52.232-16"),
            },
            {
                "violation": "FAR 52.203-13 Ethics clause missing entirely",
                "concealment_hint": "list many other clauses to make the missing one hard to notice",
                "clause": "FAR 52.203-13",
                "rule": key_value("FAR 52.203-13"),
            },
            {
                "violation": f"Verbal subcontract consent for $2.1M sub — written consent required above ${consent:,}",
                "concealment_hint": "describe the consent process in detail but use 'acknowledged' instead of 'written'",
                "clause": "FAR 52.244-2",
                "rule": key_value("FAR 52.244-2"),
            },
            {
                "violation": "OCI not disclosed — contracting officer's former employer is the awardee",
                "concealment_hint": "include a standard OCI section that references only 'organizational' conflicts, omitting personal financial ties",
                "clause": "FAR 9.5",
                "rule": key_value("FAR 9.5"),
            },
            {
                "violation": f"CAS disclosure statement 7 years out of date — portfolio crossed ${cas_thr:,} threshold",
                "concealment_hint": "reference 'CAS disclosure on file' with an old date that looks like routine record-keeping",
                "clause": "FAR CAS-9903",
                "rule": key_value("FAR CAS-9903"),
            },
        ],
        "DFARS": [
            {
                "violation": "DFARS 252.204-7012 cyber incident reporting clause missing for covered contractor",
                "concealment_hint": "include several DFARS 252.204-x clauses but omit -7012 specifically",
                "clause": "DFARS 252.204-7012",
                "rule": key_value("DFARS 252.204-7012"),
            },
            {
                "violation": "Non-domestic component used without approved waiver",
                "concealment_hint": "claim 'domestic end product' while specifying a foreign-origin sub-component in a technical exhibit",
                "clause": "DFARS 252.225-7001",
                "rule": key_value("DFARS 252.225-7001"),
            },
            {
                "violation": "DFARS 252.246-7003 notification of potential safety issue not filed",
                "concealment_hint": "reference the safety review process but omit whether the notification was actually submitted",
                "clause": "DFARS 252.246-7003",
                "rule": key_value("DFARS 252.246-7003"),
            },
        ],
        "EU": [
            {
                "violation": "Award criteria changed after submission deadline",
                "concealment_hint": "describe 'clarified evaluation methodology' in an amendment but frame it as administrative, not substantive",
                "clause": "EU Art.67",
                "rule": key_value("EU Art.67"),
            },
            {
                "violation": "Mandatory exclusion grounds not verified — Art 57 check missing for debarred entity",
                "concealment_hint": "include an eligibility checklist covering most exclusion grounds but omitting the criminal conviction check",
                "clause": "EU Art.57",
                "rule": key_value("EU Art.57"),
            },
            {
                "violation": "Contract value exceeds threshold — direct award bypassed required OJEU notice",
                "concealment_hint": "cite a prior framework agreement to justify direct award without publishing a new notice",
                "clause": "EU Art.32",
                "rule": key_value("EU Art.32"),
            },
        ],
    }

VIOLATION_SEEDS = _build_violation_seeds()


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
        raw = llm_complete(VENDOR_SYSTEM_PROMPT, prompt, max_tokens=500)
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
    pr_rate  = threshold("FAR 52.232-16", "standard_rate_pct", 80)
    consent  = threshold("FAR 52.244-2",  "consent_threshold_usd", 150_000)
    lines = [
        "CONTRACT DOCUMENT — AXIOM Corporation",
        "Contract Value: $3,200,000 | Type: Firm Fixed Price",
        f"SAM.gov Registration: Current (see file) — {key_value('FAR 4.1102')}",
        f"FAR 52.219-8 Small Business: 22% committed (see Attachment B for actuals) — {key_value('FAR 52.219-8')}",
        f"FAR 52.203-13 Ethics: See Standard Clause Package — {key_value('FAR 52.203-13')}",
        f"FAR 52.232-16 Progress Payments: Rate as negotiated (see Section H) — standard {pr_rate}%",
        f"Subcontract Consent: Acknowledged by CO on date of record — written required above ${consent:,}",
        "CAS Disclosure: On file per prior submission",
        "",
        "The parties agree to all terms and conditions set forth herein.",
        "Award Recommendation: Approve.",
    ]
    return "\n".join(lines)
