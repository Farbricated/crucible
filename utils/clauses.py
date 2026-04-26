"""
Central clause registry for CRUCIBLE.

Single source of truth for all FAR / DFARS / EU Directive values used
across agent prompts, violation seeds, and fallback contracts.

Rule: never hardcode clause text, thresholds, or rates directly in agent
      files — always look them up here.
"""

# ── FAR ────────────────────────────────────────────────────────
FAR = {
    "4.1102": {
        "title": "SAM.gov Registration",
        "requirement": "Contractor must be registered in SAM.gov before award and remain active throughout performance.",
        "key_value": "active registration required",
    },
    "6.302": {
        "title": "Justifications for Other Than Full and Open Competition",
        "requirement": "Written Justification and Approval (J&A) required for sole-source or limited-competition awards.",
        "key_value": "written J&A required",
    },
    "9.5": {
        "title": "Organizational Conflicts of Interest (OCI)",
        "requirement": "Contracting Officer must identify and resolve actual or potential OCIs before award.",
        "key_value": "OCI mitigation plan required",
    },
    "15.407-1": {
        "title": "TINA / Defective Pricing",
        "requirement": "Certified cost or pricing data required for contracts over $2M (TINA threshold). Defective data = price adjustment.",
        "threshold_usd": 2_000_000,
        "key_value": "$2M TINA threshold",
    },
    "52.203-13": {
        "title": "Contractor Code of Business Ethics and Conduct",
        "requirement": "Required in contracts over $6M with performance period >120 days. Contractor must have written code, compliance program, and hotline.",
        "threshold_usd": 6_000_000,
        "key_value": "ethics clause mandatory >$6M / >120 days",
    },
    "52.219-8": {
        "title": "Utilization of Small Business Concerns",
        "requirement": "Contractor must make maximum practicable use of small business in subcontracting.",
        "key_value": "small business subcontracting required",
    },
    "52.232-16": {
        "title": "Progress Payments",
        "requirement": "Standard progress payment rate is 80% of costs incurred. Unusual rates require written CO approval and documented justification.",
        "standard_rate_pct": 80,
        "key_value": "standard rate 80%; above requires written CO approval",
    },
    "52.244-2": {
        "title": "Subcontracts",
        "requirement": "Prior written consent of the Contracting Officer required for subcontracts at or above the simplified acquisition threshold ($150,000).",
        "consent_threshold_usd": 150_000,
        "key_value": "written CO consent required ≥$150K subcontract",
    },
    "CAS-9903": {
        "title": "Cost Accounting Standards (CAS)",
        "requirement": "CAS disclosure statement required when contractor crosses the $50M CAS threshold. Disclosure must be current and accurate.",
        "threshold_usd": 50_000_000,
        "key_value": "$50M CAS threshold; disclosure must be current",
    },
}

# ── DFARS ──────────────────────────────────────────────────────
DFARS = {
    "252.204-7012": {
        "title": "Safeguarding Covered Defense Information (CDI)",
        "requirement": "Required for all DoD contracts involving CDI or operationally critical support. Contractor must implement NIST SP 800-171, report cyber incidents within 72 hours.",
        "key_value": "mandatory for CDI contracts; 72-hour incident reporting",
    },
    "252.225-7001": {
        "title": "Buy American Act — Domestic End Products",
        "requirement": "End products must be domestic. Foreign-origin components in a 'domestic' end product require an approved waiver.",
        "key_value": "domestic end product required; foreign components need approved waiver",
    },
    "252.246-7003": {
        "title": "Notification of Potential Safety Issues",
        "requirement": "Contractor must notify the Contracting Officer and the cognizant program manager of any potential safety issues that could affect personnel or equipment.",
        "key_value": "immediate notification required for safety issues",
    },
}

# ── EU Procurement Directive 2014/24/EU ────────────────────────
EU = {
    "Art.4": {
        "title": "Thresholds",
        "works_threshold_eur": 5_350_000,
        "services_central_gov_eur": 135_000,
        "services_sub_central_eur": 209_000,
        "key_value": "works €5.35M; services central gov €135K; sub-central €209K",
    },
    "Art.26": {
        "title": "Choice of Procedure",
        "requirement": "Contracting authorities may choose open, restricted, negotiated, or competitive dialogue procedures subject to defined conditions.",
        "key_value": "procedure choice must be justified",
    },
    "Art.32": {
        "title": "Negotiated Procedure Without Publication",
        "requirement": "Direct award only in narrowly defined exceptions (extreme urgency, technical/artistic exclusivity). Cannot apply where competition is possible.",
        "key_value": "narrow exceptions only; must document justification",
    },
    "Art.33": {
        "title": "Framework Agreements",
        "requirement": "Maximum duration 4 years; maximum value must be stated at notice stage; call-off procedures must match the original notice.",
        "max_duration_years": 4,
        "key_value": "max 4 years; max value must be stated",
    },
    "Art.57": {
        "title": "Exclusion Grounds",
        "requirement": "Mandatory exclusion for criminal convictions (corruption, fraud, terrorist offences, money laundering, child labour). Discretionary exclusion for insolvency, grave professional misconduct, etc.",
        "key_value": "mandatory exclusion check required; criminal conviction = automatic exclusion",
    },
    "Art.67": {
        "title": "Award Criteria (MEAT)",
        "requirement": "Award on Most Economically Advantageous Tender (MEAT) basis. Award criteria must be published in the contract notice and cannot be changed after the submission deadline.",
        "key_value": "MEAT criteria; no post-deadline criteria change",
    },
    "Art.72": {
        "title": "Modification of Contracts During Performance",
        "requirement": "Modifications allowed without new procurement procedure only if: value < 10% (services/supplies) or 15% (works) of original contract AND not substantial change. Otherwise new procedure required.",
        "minor_modification_pct_services": 10,
        "minor_modification_pct_works": 15,
        "key_value": "<10% services / <15% works modification without new procedure",
    },
}

# ── Unified registry ───────────────────────────────────────────
CLAUSE_REGISTRY: dict[str, dict] = {
    **{f"FAR {k}": v for k, v in FAR.items()},
    **{f"DFARS {k}": v for k, v in DFARS.items()},
    **{f"EU {k}": v for k, v in EU.items()},
}


def get(clause_id: str) -> dict:
    """Lookup a clause by its full id (e.g. 'FAR 52.232-16')."""
    return CLAUSE_REGISTRY.get(clause_id, {})


def key_value(clause_id: str, fallback: str = "see regulation") -> str:
    """Return the human-readable key value for a clause."""
    return get(clause_id).get("key_value", fallback)


def threshold(clause_id: str, field: str = "threshold_usd", fallback=None):
    """Return a numeric threshold for a clause."""
    return get(clause_id).get(field, fallback)


def build_executor_clause_block() -> str:
    """
    Build the static clause reference block for the Executor system prompt.
    Called once at module load — result is a string constant, not a runtime f-string.
    """
    lines = []
    lines.append("\nUS REGULATIONS (FAR/DFARS):")
    for cid, info in CLAUSE_REGISTRY.items():
        if cid.startswith("FAR") or cid.startswith("DFARS"):
            lines.append(f"- {cid}: {info['title']} — {info['key_value']}")
    lines.append("\nEU REGULATIONS (Directive 2014/24/EU):")
    for cid, info in CLAUSE_REGISTRY.items():
        if cid.startswith("EU"):
            lines.append(f"- {cid}: {info['title']} — {info['key_value']}")
    return "\n".join(lines)
