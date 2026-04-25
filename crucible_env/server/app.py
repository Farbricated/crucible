import os

from openenv.core.env_server import create_fastapi_app

from models import CrucibleAction, CrucibleObservation
from server.crucible_environment import CrucibleEnvironment

use_architect = os.getenv("USE_ARCHITECT", "false").lower() == "true"

env = CrucibleEnvironment(use_architect=use_architect)
app = create_fastapi_app(env, CrucibleAction, CrucibleObservation)
