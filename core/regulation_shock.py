"""
Regulation Shock Engine for CRUCIBLE.

Mid-episode, a RegulationShock can fire — representing real-world procurement
disruptions that change the correct answer:

  - A vendor gets added to the SAM.gov exclusion list mid-review
  - A FAR clause threshold is revised (e.g., progress payment standard changes)
  - A new mandatory clause becomes effective immediately
  - An OFAC sanctions update flags a subcontractor

The Executor must adapt its decision to incorporate the shock.
Arbiter checks whether the Executor acknowledged and correctly applied the change.

This tests long-horizon planning under distribution shift — one of the 4 hackathon themes.
"""

import random
import uuid
from datetime import datetime
from core.schemas import RegulationShock

# ─────────────────────────────────────────────────────────────────────────────
# Shock Library
# ─────────────────────────────────────────────────────────────────────────────

_SHOCK_LIBRARY = [
    # ── FAR threshold changes ──────────────────────────────────────────────
    {
        "shock_type": "threshold_change",
        "description": "BREAKING: FAR Council final rule effective today — progress payment standard rate reduced from 80% to 75% for all new awards (FAR 52.232-16 revision).",
        "affected_clause": "FAR 52.232-16",
        "new_requirement": "Standard progress payment rate is now 75% (down from 80%). Any rate above 75% requires written justification and CO approval.",
        "severity": "high",
        "jurisdiction": "FAR",
        "world_delta": {"open_compliance_flags": ["FAR-52.232-16-threshold-revised"]},
    },
    {
        "shock_type": "threshold_change",
        "description": "FAR 15.403-4 update: certified cost/pricing data (TINA) threshold raised from $2M to $2.5M effective today.",
        "affected_clause": "FAR 15.403-4 / TINA",
        "new_requirement": "TINA certified cost/pricing data required only for contracts over $2.5M. Contracts between $2M-$2.5M now exempt.",
        "severity": "medium",
        "jurisdiction": "FAR",
        "world_delta": {},
    },
    {
        "shock_type": "threshold_change",
        "description": "FAR 52.244-2 subcontract consent threshold doubled: written consent now required for subcontracts exceeding $300K (was $150K).",
        "affected_clause": "FAR 52.244-2",
        "new_requirement": "Written CO consent required for subcontracts over $300K (revised from $150K threshold).",
        "severity": "medium",
        "jurisdiction": "FAR",
        "world_delta": {},
    },

    # ── Vendor debarment / exclusion ───────────────────────────────────────
    {
        "shock_type": "vendor_debarment",
        "description": "URGENT: SAM.gov exclusion list update — TechForce LLC (DUNS 12-345-6789) added to debarment list effective 09:00 EST today. All pending awards to this vendor must be suspended.",
        "affected_clause": "FAR 9.405",
        "new_requirement": "TechForce LLC is now debarred. No contract may be awarded or extended. Existing contracts must be reviewed for termination options.",
        "severity": "critical",
        "jurisdiction": "FAR",
        "world_delta": {"flagged_vendors": ["TechForce LLC"], "procurement_freeze": False},
    },
    {
        "shock_type": "vendor_debarment",
        "description": "OFAC update: Apex Avionics parent company 'GlobalTech Holdings' added to SDN (Specially Designated Nationals) list. ITAR/EAR restrictions apply immediately.",
        "affected_clause": "ITAR / EAR / FAR 52.225-13",
        "new_requirement": "Any contract involving Apex Avionics or any GlobalTech Holdings subsidiary is frozen pending OFAC review. Contracting officer must notify legal immediately.",
        "severity": "critical",
        "jurisdiction": "DFARS",
        "world_delta": {"flagged_vendors": ["Apex Avionics"], "security_incidents": ["OFAC-SDN-GlobalTech"]},
    },

    # ── New mandatory clauses ──────────────────────────────────────────────
    {
        "shock_type": "new_clause",
        "description": "DoD memo effective today: DFARS 252.204-7012 (Safeguarding Covered Defense Information) now mandatory for ALL contracts over $500K regardless of information classification.",
        "affected_clause": "DFARS 252.204-7012",
        "new_requirement": "DFARS 252.204-7012 must be included in all DoD contracts over $500K. Omission renders contract non-compliant and requires modification before award.",
        "severity": "high",
        "jurisdiction": "DFARS",
        "world_delta": {"open_compliance_flags": ["DFARS-252.204-7012-mandatory"]},
    },
    {
        "shock_type": "new_clause",
        "description": "FAR Council interim rule: FAR 52.204-27 (Prohibition on ByteDance) now extended to all sub-tier subcontractors regardless of value.",
        "affected_clause": "FAR 52.204-27",
        "new_requirement": "FAR 52.204-27 must now flow down to ALL sub-tiers. Flow-down waiver authority has been rescinded. Non-compliant flow-down plans must be revised before award.",
        "severity": "medium",
        "jurisdiction": "FAR",
        "world_delta": {},
    },

    # ── Sanctions / geopolitical ───────────────────────────────────────────
    {
        "shock_type": "sanctions_update",
        "description": "State Dept update: Country X added to arms embargo list. Any contract involving components sourced from Country X requires a new export license review under ITAR 22 CFR 126.",
        "affected_clause": "ITAR 22 CFR 126 / FAR 25.701",
        "new_requirement": "Contracts with any Country X-origin components must pause for export license review. Estimated review time: 30-60 days.",
        "severity": "high",
        "jurisdiction": "DFARS",
        "world_delta": {"security_incidents": ["ITAR-Country-X-embargo"]},
    },
    {
        "shock_type": "sanctions_update",
        "description": "GSA Suspension: Orion Systems placed under administrative suspension pending False Claims Act investigation. All AXIOM contracts with Orion must be reviewed immediately.",
        "affected_clause": "FAR 9.407 / False Claims Act",
        "new_requirement": "Orion Systems is suspended. No new task orders. Existing deliverables must be accepted/rejected within 30 days. CO must notify OIG.",
        "severity": "critical",
        "jurisdiction": "FAR",
        "world_delta": {"flagged_vendors": ["Orion Systems"], "program_manager_notified": True},
    },

    # ── EU shocks ──────────────────────────────────────────────────────────
    {
        "shock_type": "threshold_change",
        "description": "EU Commission update: Works contract threshold raised to €5.4M (from €5.35M). Services/supplies threshold for central authorities raised to €140K (from €135K).",
        "affected_clause": "Directive 2014/24/EU Art 4",
        "new_requirement": "Updated EU procurement thresholds apply to all tenders published after today. Contracts near the old threshold must be reassessed.",
        "severity": "medium",
        "jurisdiction": "EU",
        "world_delta": {},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Shock Engine
# ─────────────────────────────────────────────────────────────────────────────

class RegulationShockEngine:
    """Probabilistically fires regulation shocks during episodes."""

    def __init__(self, shock_probability: float = 0.30, jurisdiction: str = "FAR"):
        self.shock_probability = shock_probability
        self.jurisdiction = jurisdiction
        self._fired_shocks: list[str] = []

    def maybe_fire(self, episode_count: int) -> RegulationShock | None:
        """
        Returns a RegulationShock with `shock_probability` chance.
        Avoids repeating the same shock within 3 episodes.
        Returns None if no shock fires.
        """
        if random.random() > self.shock_probability:
            return None

        # Filter by jurisdiction and avoid recent repeats
        candidates = [
            s for s in _SHOCK_LIBRARY
            if s["jurisdiction"] in (self.jurisdiction, "FAR")  # FAR always in scope
            and s["description"] not in self._fired_shocks[-3:]
        ]
        if not candidates:
            candidates = _SHOCK_LIBRARY  # fallback — pick any

        shock_data = random.choice(candidates)
        shock = RegulationShock(
            shock_id=f"shock-{uuid.uuid4().hex[:8]}",
            shock_type=shock_data["shock_type"],
            description=shock_data["description"],
            affected_clause=shock_data["affected_clause"],
            new_requirement=shock_data["new_requirement"],
            severity=shock_data["severity"],
            jurisdiction=shock_data["jurisdiction"],
            world_delta=shock_data.get("world_delta", {}),
        )
        self._fired_shocks.append(shock_data["description"])
        return shock

    def format_for_executor(self, shock: RegulationShock) -> str:
        """Render the shock as a BREAKING NEWS alert for the Executor prompt."""
        severity_prefix = {
            "low": "NOTICE",
            "medium": "ALERT",
            "high": "URGENT ALERT",
            "critical": "CRITICAL REGULATORY ALERT",
        }.get(shock.severity, "ALERT")

        return (
            f"\n{'='*60}\n"
            f"⚡ {severity_prefix} — REGULATORY CHANGE IN EFFECT\n"
            f"{'='*60}\n"
            f"Type: {shock.shock_type.replace('_', ' ').title()}\n"
            f"Clause: {shock.affected_clause}\n"
            f"Event: {shock.description}\n"
            f"New Requirement: {shock.new_requirement}\n"
            f"{'='*60}\n"
            f"You MUST incorporate this regulatory change into your compliance analysis.\n"
            f"Failure to address this change will result in a lower score.\n"
        )

    def check_executor_adapted(self, executor_decision: str, shock: RegulationShock) -> bool:
        """
        Heuristic check: did the Executor mention the shock's clause or type?
        Used by Arbiter to set shock_adapted flag.
        """
        text = executor_decision.lower()
        clause_keywords = shock.affected_clause.lower().split()
        shock_keywords = shock.shock_type.replace("_", " ").lower().split()

        for keyword in clause_keywords + shock_keywords:
            if len(keyword) > 4 and keyword in text:
                return True
        return False
