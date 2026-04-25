import copy
try:
    from core.schemas import WorldState
except ImportError:
    from crucible_env.core.schemas import WorldState


class WorldStateManager:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.state = WorldState()
        self._history = []

    def reset(self, seed: int = None):
        if seed:
            self.seed = seed
        self.state = WorldState()
        self._history = []
        return self.state

    def render(self) -> str:
        return self.state.render()

    def apply_delta(self, delta: dict):
        self._history.append(copy.deepcopy(self.state.model_dump()))
        for key, value in delta.items():
            if hasattr(self.state, key):
                current = getattr(self.state, key)
                if isinstance(current, list) and isinstance(value, list):
                    setattr(self.state, key, list(set(current + value)))
                else:
                    setattr(self.state, key, value)

    def snapshot(self) -> dict:
        return self.state.model_dump()

    def check_coherence(self, action_decision: str) -> bool:
        state = self.state
        if state.procurement_freeze and "approve" in action_decision.lower():
            return False
        if state.security_incidents and "clear" in action_decision.lower():
            return False
        return True

    def compute_diff(self, before: dict, after: dict) -> dict:
        diff = {}
        for key in after:
            if after[key] != before.get(key):
                diff[key] = {"before": before.get(key), "after": after[key]}
        return diff
