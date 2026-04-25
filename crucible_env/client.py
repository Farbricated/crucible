from openenv.core.client import HTTPEnvClient

from models import CrucibleAction, CrucibleObservation, CrucibleState


class CrucibleEnv(HTTPEnvClient):
    """
    Client for CRUCIBLE environment.
    Usage:
        with CrucibleEnv(base_url="https://your-space.hf.space").sync() as env:
            obs = env.reset()
            action = CrucibleAction(decision="NON-COMPLIANT", reasoning="...")
            result = env.step(action)
            print(f"Reward: {result.reward}")
    """

    action_class = CrucibleAction
    observation_class = CrucibleObservation
    state_class = CrucibleState
