"""
Failure Injector — Demo Mode
Manually trigger a specific failure type so the Architect responds on cue.
Use during live demo to guarantee the loop closes visibly.

Usage:
  from demo.failure_injector import inject
  inject(axis="reasoning_transparency", domain="procurement")
"""

import json
import uuid
import os
from datetime import datetime
from core.schemas import FailureRecord

LOG_DIR = "data/episode_logs"
os.makedirs(LOG_DIR, exist_ok=True)


DEMO_FAILURES = {
    "reasoning_transparency": {
        "description": "Executor correctly classified but gave zero regulatory citations",
        "feedback": "Decision was correct (NON-COMPLIANT) but no FAR clauses cited. Score: 0.31 on reasoning_transparency.",
        "score_1": 0.38,
        "score_2": 0.42,
    },
    "completeness": {
        "description": "Executor found 1 of 4 violations in a complex contract modification",
        "feedback": "Only one violation identified. Three additional FAR violations missed. Score: 0.28 on completeness.",
        "score_1": 0.29,
        "score_2": 0.35,
    },
    "correctness": {
        "description": "Executor classified compliant vendor as non-compliant (false positive)",
        "feedback": "Incorrect classification. SAM.gov expiry was a renewal delay, not a lapse. Score: 0.22 on correctness.",
        "score_1": 0.25,
        "score_2": 0.30,
    },
    "efficiency": {
        "description": "Executor produced 800-word response for a simple binary classification",
        "feedback": "Response was verbose and poorly structured. 6x longer than necessary. Score: 0.20 on efficiency.",
        "score_1": 0.45,
        "score_2": 0.48,
    },
}


def inject(axis: str = "reasoning_transparency", domain: str = "procurement", n: int = 3) -> list[FailureRecord]:
    """
    Inject n synthetic failure records targeting a specific axis.
    Returns the injected failure records (also saves to episode log).
    """
    if axis not in DEMO_FAILURES:
        raise ValueError(f"Unknown axis: {axis}. Choose from: {list(DEMO_FAILURES.keys())}")

    template = DEMO_FAILURES[axis]
    injected = []

    for i in range(n):
        task_id = f"demo-inject-{uuid.uuid4().hex[:6]}"
        failure = FailureRecord(
            task_id=task_id,
            domain=domain,
            difficulty="medium",
            target_axis=axis,
            attempt_1_score=template["score_1"],
            attempt_2_score=template["score_2"],
            delta=round(template["score_2"] - template["score_1"], 4),
            weakest_axis=axis,
            feedback_summary=template["feedback"],
            lineage_id=None,
            timestamp=datetime.now().isoformat(),
            breakthrough=False
        )
        injected.append(failure)

        # Save to log dir so episode runner picks it up
        path = os.path.join(LOG_DIR, f"{task_id}.json")
        with open(path, "w") as f:
            f.write(json.dumps(failure.model_dump(), indent=2))

    print(f"\n[Demo Injector] Injected {n} failures targeting '{axis}'")
    print(f"  Description: {template['description']}")
    print(f"  Scores: {template['score_1']} → {template['score_2']}")
    print(f"  Architect will now generate a task targeting this weakness.\n")

    return injected


def inject_and_get_architect_response(axis: str = "reasoning_transparency"):
    """
    Full demo flow:
    1. Inject failures
    2. Trigger Architect to generate response task
    3. Print Architect reasoning for judges
    """
    from agents.architect import generate

    failures = inject(axis=axis, n=3)

    print("[Architect] Reading injected failure history...")
    task_spec, arch_output = generate(
        failures=failures,
        last_3_scores=[f.attempt_2_score for f in failures],
        current_difficulty="medium",
        episode_count=10
    )

    print("\n" + "="*60)
    print("ARCHITECT RESPONSE")
    print("="*60)
    print(f"Target axis:   {arch_output.target_axis}")
    print(f"Difficulty:    {arch_output.difficulty}")
    print(f"Lineage:       {arch_output.lineage_id}")
    print(f"\nReasoning:\n{arch_output.architect_reasoning}")
    print(f"\nGenerated task:\n{arch_output.scenario_context}")
    print("="*60)

    return task_spec, arch_output


if __name__ == "__main__":
    import sys
    axis = sys.argv[1] if len(sys.argv) > 1 else "reasoning_transparency"
    inject_and_get_architect_response(axis)


def run_live_demo(env_url: str, axis: str = "reasoning_transparency"):
    """
    Live demo for judges:
    1. Show Executor failing on screen
    2. Show Arbiter scoring it
    3. Show Architect generating response task
    4. Judges watch loop close in real time
    """
    from crucible_env.client import CrucibleEnv
    from crucible_env.models import CrucibleAction

    print("=" * 60)
    print("CRUCIBLE LIVE DEMO")
    print("=" * 60)

    with CrucibleEnv(base_url=env_url).sync() as env:
        obs = env.reset()
        print(f"\n[TASK] {obs.task_description}")
        print(f"[DIFFICULTY] {obs.difficulty}")
        print(f"[TARGET AXIS] {obs.target_axis}")

        weak_action = CrucibleAction(
            decision="NON-COMPLIANT",
            reasoning="The vendor appears to have issues.",
            violations_found=[],
            confidence=0.3,
        )

        result = env.step(weak_action)

        print(f"\n[ARBITER] Score: {result.score_attempt_1:.3f}")
        print(f"[ARBITER] Feedback: {result.arbiter_feedback[:200]}")
        print(f"\n[ATTEMPT 2] Score: {result.score_attempt_2:.3f}")
        print(f"[FINAL REWARD] {result.final_reward:.3f}")

        if result.architect_reasoning:
            print(f"\n{'=' * 60}")
            print("ARCHITECT RESPONDS:")
            print(f"{'=' * 60}")
            print(f"{result.architect_reasoning}")
            print(f"\nNext task difficulty: {result.next_task_difficulty}")
            print(f"Lineage: {result.lineage_id}")

        state = env.state()
        print(f"\n[STATE] Episodes: {state.step_count}")
        print(f"[STATE] Avg reward: {state.avg_reward_last_10:.3f}")
        print(f"[STATE] Breakthroughs: {state.breakthrough_count}")
        print(
            "[STATE] Architect calibration: "
            f"{state.architect_calibration_accuracy:.1%}"
        )
