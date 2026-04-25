import os
import json
from pathlib import Path

from openenv.core.env_server import create_fastapi_app
from fastapi.responses import JSONResponse

try:
    from models import CrucibleAction, CrucibleObservation
    from server.crucible_environment import CrucibleEnvironment
except ImportError:
    from crucible_env.models import CrucibleAction, CrucibleObservation
    from crucible_env.server.crucible_environment import CrucibleEnvironment

use_architect = os.getenv("USE_ARCHITECT", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "episode_logs"

def health():
    episodes_logged = 0
    try:
        if LOG_DIR.exists():
            episodes_logged = sum(
                1
                for p in LOG_DIR.glob("*.json")
                if p.name not in {"full_run.json", "baseline.json"}
            )
    except Exception:
        episodes_logged = 0

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "environment": "crucible",
            "version": "0.1.0",
            "agents": ["executor", "arbiter", "architect"],
            "domain": "procurement",
            "episodes_logged": episodes_logged,
        },
    )


def metrics():
    payload = {
        "total_episodes": 0,
        "avg_reward": 0.0,
        "latest_reward": 0.0,
        "breakthrough_count": 0,
        "architect_calibration": 0.0,
        "in_band_rate": 0.0,
    }
    try:
        full_run = LOG_DIR / "full_run.json"
        if not full_run.exists():
            return JSONResponse(status_code=200, content=payload)

        with full_run.open(encoding="utf-8") as f:
            history = json.load(f)
        if not history:
            return JSONResponse(status_code=200, content=payload)

        rewards = [float(h.get("final_reward", 0.0)) for h in history]
        in_band = [1 if h.get("in_band") else 0 for h in history]
        arch_history = [h for h in history if h.get("architect_active")]
        arch_in_band = [1 if h.get("in_band") else 0 for h in arch_history]

        payload = {
            "total_episodes": len(history),
            "avg_reward": round(sum(rewards) / len(rewards), 2),
            "latest_reward": round(rewards[-1], 2),
            "breakthrough_count": sum(1 for h in history if h.get("is_breakthrough")),
            "architect_calibration": round(
                sum(arch_in_band) / len(arch_in_band), 2
            )
            if arch_in_band
            else 0.0,
            "in_band_rate": round(sum(in_band) / len(in_band), 2) if in_band else 0.0,
        }
    except Exception:
        pass
    return JSONResponse(status_code=200, content=payload)


def env_factory():
    return CrucibleEnvironment(use_architect=use_architect)


app = create_fastapi_app(env_factory, CrucibleAction, CrucibleObservation)
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/metrics", metrics, methods=["GET"])
