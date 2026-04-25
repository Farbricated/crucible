from core.schemas import TaskSpec

STATIC_TASKS = {
    "easy": [
        {
            "scenario_context": "Review the vendor bid from TechForce LLC for AXIOM-7731. Classify as COMPLIANT or NON-COMPLIANT and identify the specific violation if any.",
            "contract_text": """
VENDOR BID — TechForce LLC
Contract: AXIOM-7731 (Avionics Component Supply)
Base Price: $2,400,000 | Delivery: 180 days ARO

FAR 52.219-8 Small Business Subcontracting:
  Committed: 23% | Actual: 8%

FAR 52.215-2 Cost Accounting: Not applicable per vendor claim.
SAM.gov registration: Current
Award recommendation: Approve.
""",
            "target_axis": "correctness",
            "answer": "NON-COMPLIANT — small business subcontracting goal not met (8% vs 23% required under FAR 52.219-8)"
        },
        {
            "scenario_context": "Review this vendor registration for ClearPath Defense. Classify as COMPLIANT or NON-COMPLIANT.",
            "contract_text": """
VENDOR REGISTRATION — ClearPath Defense
SAM.gov registration: EXPIRED (3 months ago)
DUNS: 98-761-2341
Past performance: Satisfactory
Technical capability: Meets requirements
Note: Vendor requests expedited processing.
""",
            "target_axis": "correctness",
            "answer": "NON-COMPLIANT — SAM.gov registration expired, vendor ineligible per FAR 4.1102"
        },
        {
            "scenario_context": "Review this sole source justification for Apex Avionics. Is the justification legally sufficient under FAR?",
            "contract_text": """
SOLE SOURCE JUSTIFICATION
Vendor: Apex Avionics
Requirement: Flight control software module
Justification: Apex Avionics is the only vendor we have worked with before.
Estimated value: $890,000
FAR authority cited: FAR 6.302-1 (Only one responsible source)
""",
            "target_axis": "reasoning_transparency",
            "answer": "INSUFFICIENT — prior working relationship does not satisfy FAR 6.302-1. Must demonstrate unique technical capability or patent protection."
        },
        {
            "scenario_context": "Review this bid from Meridian Systems. Identify any FAR violations.",
            "contract_text": """
VENDOR BID — Meridian Systems
Contract: AXIOM-8820 (Radar Integration, $5.1M)
SAM.gov: Current
FAR 52.203-13 Code of Business Ethics: Present
Small business subcontracting: 21% committed, 20% actual — MEETS GOAL
Delivery: 14 months (requirement: 12 months)
Price: $5.1M (government estimate: $4.8M — 6.25% above estimate)
Progress payment requested: 85% of costs incurred
""",
            "target_axis": "completeness",
            "answer": "NON-COMPLIANT on two points: (1) Delivery 14 months exceeds 12-month requirement. (2) Progress payment rate 85% exceeds FAR 52.232-16 standard of 80% with no justification provided."
        },
        {
            "scenario_context": "A contracting officer wants to award to the lowest bidder, DeltaForce Inc. Review for conflicts of interest.",
            "contract_text": """
AWARD MEMO — DeltaForce Inc
Contract: AXIOM-7731-B (Communications Hardware, $920K)
DeltaForce Inc bid: $920,000 (lowest)
Contracting Officer: James Harmon
Note in file: J. Harmon's spouse is listed as a part-owner of DeltaForce Inc.
OCI disclosure: Not filed.
SAM.gov: Current. All FAR clauses present.
""",
            "target_axis": "correctness",
            "answer": "NON-COMPLIANT — undisclosed Organizational Conflict of Interest (OCI). Contracting officer has financial family tie to awardee. FAR 9.5 requires OCI disclosure and recusal."
        },
    ],

    "medium": [
        {
            "scenario_context": "Review these 4 procurement packages and rank from HIGHEST to LOWEST compliance risk. Identify the primary violation in each.",
            "contract_text": """
PACKAGE A — Vertex Systems ($1.2M)
SAM.gov: Current | SB goal: 25% committed, 24% actual
FAR clauses: All present | Price: Within 12% of estimate

PACKAGE B — NovaTech LLC ($3.4M)
SAM.gov: Current | SB goal: 20% committed, 6% actual
FAR 52.203-13 (Ethics Clause): MISSING
Price: Within 8% of estimate

PACKAGE C — Orbital Defense ($780K)
SAM.gov: EXPIRED 2 weeks ago | SB goal: Not addressed
All other clauses present

PACKAGE D — BlueShield Corp ($2.1M)
SAM.gov: Current
Contracting officer is former BlueShield employee (disclosed, 8 months ago)
SB goal: 22% committed, 21% actual | Price: Within 5% of estimate
""",
            "target_axis": "completeness",
            "answer": "Risk: C (expired SAM, ineligible) > B (missing ethics clause + SB gap) > D (OCI needs closer review despite disclosure) > A (minor SB gap, all else clean)"
        },
        {
            "scenario_context": "AXIOM needs to procure drone navigation software urgently. The program manager wants to use FAR 6.302-2 (Unusual and Compelling Urgency) to sole-source to RaptorNav. Evaluate whether the urgency justification is legally supportable.",
            "contract_text": """
URGENCY JUSTIFICATION — RaptorNav Inc
Contract value: $2.8M
FAR authority: 6.302-2 Unusual and Compelling Urgency
Reason stated: Program schedule requires delivery in 60 days.
  Competing bids would take 90 days minimum.
  30-day delay would push delivery past contract milestone.
  Milestone penalty: $150K per week.

History: AXIOM has used RaptorNav on 3 prior contracts.
Market research: Conducted 6 months ago. 2 other vendors identified.
Current market check: Not conducted.
""",
            "target_axis": "reasoning_transparency",
            "answer": "WEAKLY SUPPORTED — schedule pressure alone is insufficient for 6.302-2. AXIOM created the urgency through poor planning. Market research is 6 months stale. Must conduct current market check. Consider accelerated competition instead."
        },
        {
            "scenario_context": "Three bids received for AXIOM-9002. Evaluate and recommend award using best value tradeoff analysis.",
            "contract_text": """
RFP: AXIOM-9002 — Tactical Communications Array ($8.4M ceiling)
Evaluation: Best value tradeoff (Technical 40%, Past Performance 30%, Price 30%)

BIDDER 1 — CommStar Federal: $7.9M
Technical: 85/100 | Past performance: Excellent (5 contracts)
Delivery: On time | SB: 24% (meets 20% goal) | All clauses present

BIDDER 2 — NexGen Defense: $6.8M
Technical: 76/100 | Past performance: Satisfactory (2 contracts)
Delivery: On time | SB: 21% | All clauses present

BIDDER 3 — PrimeWave LLC: $8.1M
Technical: 91/100 | Past performance: Good (3 contracts)
Delivery: 2 weeks late | SB: 19% (below 20% goal — minor gap)
FAR 52.222-26 Equal Opportunity: MISSING
""",
            "target_axis": "completeness",
            "answer": "Recommend Bidder 1 (CommStar). Bidder 3 eliminated: missing mandatory Equal Opportunity clause and SB gap. Between 1 and 2: CommStar's higher technical score and excellent past performance justify $1.1M premium under best value tradeoff."
        },
    ],

    "hard": [
        {
            "scenario_context": "Analyze this contract modification. The document appears compliant on the surface. Identify ALL hidden regulatory violations.",
            "contract_text": """
CONTRACT MODIFICATION — AXIOM-8820-MOD-003
Prime: GlobalAero Systems | Mod value: +$4,200,000 | Total: $18.7M
Purpose: Scope expansion — additional radar component integration

Section 3.1 Subcontracting Plan:
GlobalAero commits to 23% SB per original contract.
CO determination: No updated plan required.
[FAR 52.219-9: Updated plan required when value increases above $1.5M threshold.
Original plan was written for $14.5M scope.]

Section 3.4 Cost Accounting:
CAS disclosure statement on file from 2019. No update required.
[48 CFR 9903.202-3: Modification pushes CAS-covered portfolio above $50M —
disclosure update required.]

Section 4.1 Consent to Subcontract:
DataFlow Analytics added as new subcontractor ($1.8M).
CO verbal approval obtained.
[FAR 52.244-2: Written consent required for subcontracts exceeding $150K.]

Section 5.2 Progress Payments:
Rate: 85% of costs incurred.
[FAR 52.232-16: Standard rate is 80%. No justification for premium rate.]
""",
            "target_axis": "reasoning_transparency",
            "answer": "4 violations: (1) Subcontracting plan not updated after value increase per FAR 52.219-9. (2) CAS disclosure not updated after $50M portfolio threshold breach per 48 CFR 9903.202-3. (3) Verbal not written consent for $1.8M subcontract per FAR 52.244-2. (4) Progress payment rate 85% above FAR 52.232-16 standard with no justification."
        },
        {
            "scenario_context": "AXIOM received a whistleblower complaint that Vendor Orion Systems submitted false certified cost data. Review the file and determine what actions are required.",
            "contract_text": """
WHISTLEBLOWER COMPLAINT — Orion Systems
Contract: AXIOM-7731-C ($6.2M, cost-plus-fixed-fee)
Allegation: Orion submitted certified cost data inflating labor rates by 18%.
Complainant: Former Orion finance manager (left company 3 months ago).

File review:
- Orion's certified cost/pricing data submitted per TINA (Truth in Negotiations Act)
- Certified data showed avg labor rate: $145/hr
- Industry survey data (same period): $123/hr average for same labor categories
- Discrepancy: ~18% — $480K potential overbilling
- Orion's certification signed by CFO: "data is accurate and complete"
- Contract awarded 8 months ago. Deliveries ongoing.

CO response so far: "We should ask Orion to explain the discrepancy."
""",
            "target_axis": "completeness",
            "answer": "CO response insufficient. Required actions: (1) Formal defective pricing investigation under FAR 15.407-1. (2) Preserve all records — legal hold. (3) Notify Inspector General per 10 U.S.C. 2409. (4) Suspend progress payments pending investigation. (5) Consider False Claims Act referral if fraud confirmed. CO cannot simply ask Orion to explain — this is a potential criminal matter."
        },
    ],

    "expert": [
        {
            "scenario_context": "Produce a full source selection recommendation for AXIOM-9001. Include: technical evaluation, price reasonableness analysis, small business compliance, regulatory compliance, and final award recommendation with justification.",
            "contract_text": """
SOURCE SELECTION — AXIOM-9001 (Encrypted Comms System, $12.4M ceiling, 5yr support)
Evaluation: Best value (Technical 40%, Past Perf 30%, Price 20%, SB 10%)

VENDOR 1 — SecureComm Federal ($11.2M)
Technical: 87/100 | FIPS 140-2: Compliant
Delivery: 18 months (requirement: 12 months) — RISK
Past performance: Excellent (3 similar DoD contracts)
SB subcontracting: 18% (requirement: 20%) — GAP
SAM.gov: Current | All FAR clauses: Present

VENDOR 2 — CipherDyne LLC ($13.8M) — Certified Small Business
Technical: 91/100 | FIPS 140-2: Compliant
Delivery: 11 months — MEETS requirement
Past performance: Satisfactory (1 similar contract, smaller scale)
SB: Self-performs (100% counts toward agency SB goals)
SAM.gov: Current | All FAR clauses: Present
Price premium vs Vendor 1: $2.6M

VENDOR 3 — Atlas Defense Systems ($10.9M)
Technical: 79/100 | FIPS 140-2: Partial — waiver requested
Delivery: 10 months
Past performance: POOR — contract terminated for default 2019
SB: 22%
SAM.gov: Current
Prior violation: ITAR disclosure failure 2021 (resolved per vendor claim)
""",
            "target_axis": "completeness",
            "answer": "Eliminate Vendor 3: default termination history is disqualifying; partial FIPS compliance unacceptable for encrypted comms; unverified ITAR violation. Between V1 and V2: V2 wins on technical (91 vs 87), delivery (11 vs 18 months — critical risk for V1), and SB contribution (100% vs gap). $2.6M premium justified under best value by superior technical score, zero delivery risk, and SB credit. Recommend award to CipherDyne LLC."
        },
    ]
}


def get_static_task(difficulty: str, index: int = 0) -> TaskSpec:
    tasks = STATIC_TASKS.get(difficulty, STATIC_TASKS["easy"])
    task_data = tasks[index % len(tasks)]
    return TaskSpec(
        task_id=f"static-{difficulty}-{index:03d}",
        domain="procurement",
        difficulty=difficulty,
        target_axis=task_data["target_axis"],
        scenario_context=task_data["scenario_context"],
        contract_text=task_data["contract_text"],
        expected_score_range=(0.45, 0.70),
        is_static=True
    )


def get_all_static_tasks() -> list[TaskSpec]:
    all_tasks = []
    for difficulty in ["easy", "medium", "hard", "expert"]:
        tasks = STATIC_TASKS.get(difficulty, [])
        for i in range(len(tasks)):
            all_tasks.append(get_static_task(difficulty, i))
    return all_tasks
