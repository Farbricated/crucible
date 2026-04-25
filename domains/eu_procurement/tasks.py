"""
EU Procurement Domain for CRUCIBLE.

Tests the Executor's ability to apply EU Procurement Directive 2014/24/EU
and related instruments — demonstrating that the environment and Executor
generalize beyond US FAR/DFARS to a completely different regulatory corpus.

Key instruments:
  - Directive 2014/24/EU (Public Procurement)
  - Directive 2014/25/EU (Utilities)
  - Regulation (EU) 2022/1925 (Digital Markets Act — procurement implications)
  - Directive 2014/23/EU (Concession contracts)
  - OJEU (Official Journal of the European Union) notice requirements
  - Award criteria: MEAT (Most Economically Advantageous Tender)

AXIOM operates a European subsidiary — AXIOM Europe GmbH — based in Germany,
subject to EU procurement rules for contracts above threshold values.
"""

from core.schemas import TaskSpec

EU_TASKS = {
    "easy": [
        {
            "scenario_context": "AXIOM Europe GmbH wants to directly award a €180,000 IT services contract to DataBridge GmbH without publishing a tender notice. Review whether a direct award is permissible.",
            "contract_text": """
PROCUREMENT NOTE — AXIOM Europe GmbH
Contract: IT Support Services — AXIOM-EU-2024-IT-01
Estimated Value: €180,000 (annual, potential 2-year extension = €360,000)
Proposed Award: Direct award to DataBridge GmbH (incumbent supplier)
Justification: Incumbent has performed satisfactorily for 3 years.

Directive Reference: Directive 2014/24/EU
Services Threshold for Central Government: €135,000
""",
            "target_axis": "correctness",
            "answer": "NON-COMPLIANT — total potential contract value €360,000 exceeds the EU services threshold (€135,000 for central government authorities). OJEU notice and open tender required under Art 26 Directive 2014/24/EU. Incumbent preference is not a valid exception."
        },
        {
            "scenario_context": "AXIOM Europe received three tenders for a €2.1M defense logistics contract. The procurement officer wants to award on lowest price alone. Review for compliance.",
            "contract_text": """
TENDER EVALUATION — AXIOM Europe GmbH
Contract: Defense Logistics Support — AXIOM-EU-2024-LOG-03
Value: €2,100,000 | Duration: 3 years
Award Criterion: Lowest price only.
Evaluation Committee Decision: Award to Bidder C (lowest at €1.85M).

Bidder A: €2.1M — Technical score 91/100
Bidder B: €1.95M — Technical score 87/100
Bidder C: €1.85M — Technical score 71/100

OJEU Notice: Published. Deadline observed.
""",
            "target_axis": "reasoning_transparency",
            "answer": "POTENTIALLY NON-COMPLIANT — Directive 2014/24/EU Art 67 requires MEAT (Most Economically Advantageous Tender) criteria for most contracts. Lowest-price-only is permitted only in limited circumstances. For a complex logistics service, ignoring a 20-point technical quality gap creates legal challenge risk. Should document why lowest price alone is justified."
        },
    ],
    "medium": [
        {
            "scenario_context": "AXIOM Europe received a formal challenge from Bidder A claiming the award criteria were changed after submission of tenders. Assess the legal exposure.",
            "contract_text": """
CHALLENGE RESPONSE FILE — AXIOM Europe GmbH
Contract: AXIOM-EU-2024-IT-05 (Cloud Infrastructure, €4.2M)
Challenger: Bidder A (NordCloud Systems)
Allegation: After tender submission deadline, the evaluation panel issued
'Clarification Note C-7' which changed the weighting of Security criterion
from 20% to 30%, and reduced Price from 30% to 20%.
Procurement officer's response: "Clarification Note C-7 merely clarified existing
criteria — it did not change them substantively."

Timeline:
- OJEU Notice published: criteria stated as Price 30%, Quality 40%, Security 20%, Delivery 10%
- Tender deadline: T+60 days
- Clarification Note C-7 issued: T+65 days (after deadline)
- Award decision: T+80 days
""",
            "target_axis": "completeness",
            "answer": "SERIOUS VIOLATION — Directive 2014/24/EU Art 67 prohibits changing award criteria after submission. Changing Security from 20% to 30% is substantive, not a clarification. This creates risk of: (1) Contract annulment by national review body. (2) Damages claim from Bidder A under national remedies directive (2007/66/EC). Procurement must be re-run or formally justified to review body."
        },
        {
            "scenario_context": "AXIOM Europe wants to negotiate directly with a single supplier under the 'extreme urgency' exception. Evaluate whether the exception is validly invoked.",
            "contract_text": """
URGENCY JUSTIFICATION — AXIOM Europe GmbH
Contract: Emergency Radar Maintenance — AXIOM-EU-2024-EMER-01
Value: €890,000
Proposed Exception: Art 32(2)(c) Directive 2014/24/EU — Extreme urgency

Justification:
  Primary contractor (VectorSystems) filed for insolvency 12 days ago.
  Backup contractor pre-qualified but requires 45-day onboarding.
  Safety-critical system has 7-day failure risk window.
  Proposed direct award to AlphaRadar GmbH (only certified EU supplier).

Prior procurement history:
  AXIOM Europe has been aware of VectorSystems' financial difficulties for 6 months.
  No contingency procurement was initiated.
""",
            "target_axis": "reasoning_transparency",
            "answer": "PARTIALLY VALID but legally risky — Art 32(2)(c) requires urgency not attributable to the contracting authority. AXIOM knew of financial difficulties 6 months ago and took no action. This undermines the exception. Recommendation: document the safety risk timeline carefully, use shortest possible duration contract, commence open procedure for replacement contract immediately. Legal review recommended before award."
        },
    ],
    "hard": [
        {
            "scenario_context": "AXIOM Europe's procurement officer excluded a bidder under the mandatory exclusion grounds but failed to conduct formal checks. The excluded bidder is challenging the exclusion. Identify all compliance failures.",
            "contract_text": """
EXCLUSION DECISION — AXIOM Europe GmbH
Contract: AXIOM-EU-2025-SYS-02 (Systems Integration, €6.8M)
Excluded Bidder: TechBridge AG

Exclusion Grounds Cited: Art 57(1)(a) — conviction for participation in criminal organization
Basis: Procurement officer recalled reading a news article 18 months ago about
       TechBridge AG being investigated by German authorities.

Self-Declaration: TechBridge AG submitted ESPD (European Single Procurement Document)
  declaring no criminal convictions.
Formal Criminal Records Check: Not conducted.
Opportunity to Remedy: Not offered (no self-cleaning process under Art 57(6)).
Standstill Period: Not observed before award to next bidder.

Excluded bidder has filed challenge in Vergabekammer (procurement review chamber).
""",
            "target_axis": "completeness",
            "answer": "Multiple violations: (1) Art 57 exclusion requires verified conviction — news article insufficient. ESPD declared clean must be verified via criminal records check before exclusion. (2) No self-cleaning opportunity offered under Art 57(6) — mandatory before exclusion. (3) Standstill period (Alcatel period) not observed — award to next bidder potentially voidable. (4) Vergabekammer will likely suspend contract and order re-evaluation. AXIOM faces damages claim and potential contract annulment."
        },
    ],
    "expert": [
        {
            "scenario_context": "AXIOM Europe is designing a framework agreement for 4 years across 12 EU member state operations. Provide a complete legal compliance assessment covering threshold obligations, call-off procedures, and maximum value calculations.",
            "contract_text": """
FRAMEWORK AGREEMENT DESIGN — AXIOM Europe GmbH
Scope: IT and Systems Integration Services across 12 EU member states
Duration: 4 years (maximum under Directive 2014/24/EU Art 33)
Estimated Total Value: €28M across all call-offs
Maximum Suppliers: 8 operators admitted to framework

Proposed Structure:
- Single framework covering all member states (centralised purchasing)
- Re-competition for individual call-offs above €500K
- Mini-competitions waived for call-offs below €500K (direct award to highest-ranked supplier)
- No maximum call-off value stated in framework documents
- Contracting authorities across all 12 member states can issue call-offs independently

Issues noted by legal team:
1. Some member states have additional national transposition requirements
2. Framework agreement notice published in OJEU but 4 member states' national gazettes not notified
3. One supplier (Rank 3) was acquired by a competitor (Rank 1) after framework award

Issue with supplier acquisition not yet addressed.
""",
            "target_axis": "completeness",
            "answer": "Multiple issues: (1) Art 33 — no maximum call-off value stated violates framework agreement rules; must state maximum quantity and value at OJEU notice stage. (2) Direct award below €500K without mini-competition only valid if terms fully fixed in framework — not stated. (3) National gazette failure in 4 states may invalidate those call-offs — check national transposition. (4) Rank 1 acquiring Rank 3 creates material change to framework — Art 72 substantial modification analysis required; may require re-tendering for Rank 3 slot. (5) Contracting authorities issuing call-offs without framework coordination may breach public procurement law in stricter member states."
        },
    ],
}


def get_eu_task(difficulty: str, index: int = 0) -> TaskSpec:
    tasks = EU_TASKS.get(difficulty, EU_TASKS["easy"])
    task_data = tasks[index % len(tasks)]
    return TaskSpec(
        task_id=f"eu-{difficulty}-{index:03d}",
        domain="eu_procurement",
        difficulty=difficulty,
        target_axis=task_data["target_axis"],
        scenario_context=task_data["scenario_context"],
        contract_text=task_data["contract_text"],
        expected_score_range=(0.45, 0.70),
        is_static=True,
    )


def get_all_eu_tasks() -> list[TaskSpec]:
    all_tasks = []
    for difficulty in ["easy", "medium", "hard", "expert"]:
        for i in range(len(EU_TASKS.get(difficulty, []))):
            all_tasks.append(get_eu_task(difficulty, i))
    return all_tasks
