"""
Task Diversity Scorer for CRUCIBLE.

Measures how diverse the Architect's generated tasks are across:
  - Regulatory clause coverage  (how many unique FAR/DFARS/EU clauses appear)
  - Violation type variety       (how many distinct violation categories)
  - Difficulty distribution      (is the band well-covered?)
  - Jurisdiction spread          (FAR vs DFARS vs EU)
  - Scenario novelty             (n-gram overlap vs previous tasks)

A diverse curriculum is harder to game and produces better-generalizing agents.
The Architect gets a +0.05 diversity bonus when it generates a task in an
under-explored (domain, axis, difficulty) cell.
"""

import math
import re
from collections import Counter
from core.schemas import TaskSpec, ArchitectOutput

# Known FAR/DFARS/EU clause families
_CLAUSE_PATTERNS = [
    r"FAR\s+\d+\.\d+",
    r"FAR\s+52\.\d+-\d+",
    r"DFARS\s+252\.\d+-\d+",
    r"48\s+CFR\s+\d+",
    r"Art(?:icle)?\s+\d+",     # EU Directive articles
    r"Directive\s+\d{4}/\d+",
    r"TINA",
    r"CAS",
    r"ITAR",
    r"EAR",
    r"OFAC",
    r"FAR\s+9\.\d+",
]

_VIOLATION_KEYWORDS = {
    "sam_registration": ["sam.gov", "sam registration", "expired registration"],
    "small_business": ["small business", "sb subcontracting", "far 52.219"],
    "progress_payment": ["progress payment", "far 52.232-16"],
    "ethics_clause": ["far 52.203-13", "code of business ethics"],
    "oci": ["conflict of interest", "oci", "far 9.5", "recusal"],
    "subcontract_consent": ["subcontract consent", "far 52.244-2"],
    "cas_disclosure": ["cas disclosure", "cost accounting standards", "48 cfr 9903"],
    "tina_defective": ["tina", "defective pricing", "certified cost", "far 15.407"],
    "dfars_cyber": ["252.204-7012", "covered defense information", "cybersecurity"],
    "buy_american": ["buy american", "domestic end product", "252.225-7001"],
    "eu_thresholds": ["threshold", "ojeu", "directive 2014/24"],
    "eu_exclusion": ["art 57", "exclusion grounds", "espd"],
    "eu_award_criteria": ["meat", "art 67", "award criteria"],
    "itar_ear": ["itar", "ear", "export control", "22 cfr"],
    "ofac_sanctions": ["ofac", "sdn", "specially designated"],
    "false_claims": ["false claims act", "whistleblower", "fraud"],
    "competition": ["sole source", "far 6.302", "full and open competition"],
}


def extract_clauses(text: str) -> set[str]:
    """Extract all regulatory clause references from text."""
    found = set()
    text_up = text.upper()
    for pattern in _CLAUSE_PATTERNS:
        matches = re.findall(pattern, text_up, re.IGNORECASE)
        found.update(m.strip() for m in matches)
    return found


def extract_violation_types(text: str) -> set[str]:
    """Identify which violation categories appear in the text."""
    found = set()
    text_lower = text.lower()
    for category, keywords in _VIOLATION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(category)
    return found


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute n-gram overlap between two texts (0 = completely different)."""
    def ngrams(text, n):
        tokens = text.lower().split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    a_grams = ngrams(text_a, n)
    b_grams = ngrams(text_b, n)
    if not a_grams or not b_grams:
        return 0.0
    intersection = a_grams & b_grams
    union = a_grams | b_grams
    return len(intersection) / len(union)


class DiversityTracker:
    """
    Tracks curriculum diversity across the full training run.
    Called by EpisodeRunner after each episode.
    """

    def __init__(self):
        self.seen_clauses: set[str] = set()
        self.seen_violation_types: set[str] = set()
        self.difficulty_counts: Counter = Counter()
        self.jurisdiction_counts: Counter = Counter()
        self.axis_counts: Counter = Counter()
        self.task_texts: list[str] = []
        self.episode_diversity_scores: list[float] = []

    def score_task(self, task: TaskSpec, jurisdiction: str = "FAR") -> float:
        """
        Score how novel/diverse a task is vs what's been seen.
        Returns a diversity score 0.0–1.0.
        High score = task explores new territory.
        """
        contract = (task.contract_text or "") + " " + task.scenario_context

        # New regulatory clauses
        new_clauses = extract_clauses(contract) - self.seen_clauses
        clause_novelty = min(1.0, len(new_clauses) / 3.0)

        # New violation types
        new_violations = extract_violation_types(contract) - self.seen_violation_types
        violation_novelty = min(1.0, len(new_violations) / 2.0)

        # Difficulty rarity (inverse frequency)
        total_tasks = sum(self.difficulty_counts.values()) + 1
        diff_freq = self.difficulty_counts.get(task.difficulty, 0) / total_tasks
        difficulty_novelty = 1.0 - diff_freq

        # Jurisdiction rarity
        jur_freq = self.jurisdiction_counts.get(jurisdiction, 0) / total_tasks
        jurisdiction_novelty = 1.0 - jur_freq

        # Axis rarity
        axis_freq = self.axis_counts.get(task.target_axis, 0) / total_tasks
        axis_novelty = 1.0 - axis_freq

        # Scenario novelty (low overlap with recent tasks)
        if self.task_texts:
            overlaps = [ngram_overlap(contract, t) for t in self.task_texts[-5:]]
            avg_overlap = sum(overlaps) / len(overlaps)
            scenario_novelty = 1.0 - avg_overlap
        else:
            scenario_novelty = 1.0

        diversity = (
            0.25 * clause_novelty
            + 0.20 * violation_novelty
            + 0.15 * difficulty_novelty
            + 0.15 * jurisdiction_novelty
            + 0.15 * axis_novelty
            + 0.10 * scenario_novelty
        )
        return round(diversity, 4)

    def record(self, task: TaskSpec, jurisdiction: str = "FAR") -> float:
        """Score and record a task. Returns diversity score."""
        score = self.score_task(task, jurisdiction)
        contract = (task.contract_text or "") + " " + task.scenario_context

        self.seen_clauses.update(extract_clauses(contract))
        self.seen_violation_types.update(extract_violation_types(contract))
        self.difficulty_counts[task.difficulty] += 1
        self.jurisdiction_counts[jurisdiction] += 1
        self.axis_counts[task.target_axis] += 1
        self.task_texts.append(contract)
        self.episode_diversity_scores.append(score)
        return score

    def summary(self) -> dict:
        """Return aggregate diversity statistics."""
        scores = self.episode_diversity_scores
        return {
            "unique_clauses_seen": len(self.seen_clauses),
            "unique_violation_types": len(self.seen_violation_types),
            "difficulty_distribution": dict(self.difficulty_counts),
            "jurisdiction_distribution": dict(self.jurisdiction_counts),
            "axis_distribution": dict(self.axis_counts),
            "avg_diversity_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "diversity_trend": scores[-10:],
        }
