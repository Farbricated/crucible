"""Deploy the crucible_env package to a HuggingFace Space.

Prereqs:
  pip install huggingface_hub
  export HF_TOKEN=hf_xxx   (or set in .env)

Usage:
  python scripts/deploy_hf_space.py --username YOUR_HF_USERNAME --space crucible-env

What this does:
  1. Creates (or reuses) a HuggingFace Space named <username>/<space>.
  2. Uploads the entire crucible_env/ folder to the Space.
  3. Triggers a build; the Space will be live at:
       https://<username>-<space>.hf.space
     with endpoints /health, /metrics, /reset, /step, /state, /docs.

After deploy, edit README.md (top-level and crucible_env/) and replace
YOUR_USERNAME with your actual HF username.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy CRUCIBLE env to a HuggingFace Space")
    parser.add_argument("--username", required=True, help="Your HF username")
    parser.add_argument("--space", default="crucible-env", help="Space name (default: crucible-env)")
    parser.add_argument("--private", action="store_true", help="Create as private Space")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set. export HF_TOKEN=hf_xxx first.")
        return 1

    repo_id = f"{args.username}/{args.space}"
    env_dir = Path(__file__).resolve().parents[1] / "crucible_env"

    if not env_dir.is_dir():
        print(f"ERROR: crucible_env/ directory not found at {env_dir}")
        return 1

    api = HfApi(token=token)

    print(f"[1/3] Creating or reusing Space '{repo_id}' ...")
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    print(f"[2/3] Uploading crucible_env/ to Space '{repo_id}' ...")
    api.upload_folder(
        folder_path=str(env_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy CRUCIBLE v0.3.0: 4 agents, shocks, adversarial, multi-jurisdiction",
        ignore_patterns=[
            "__pycache__/*", "*.pyc", ".venv/*", "uv.lock",
            "data/episode_logs/*", "plots/*",
        ],
    )

    live_url = f"https://{args.username}-{args.space}.hf.space"
    print("[3/3] Deploy triggered. Space URL:")
    print(f"  {live_url}")
    print(f"  {live_url}/health")
    print(f"  {live_url}/docs")
    print()
    print("Now update README.md links to replace YOUR_USERNAME with:", args.username)
    return 0


if __name__ == "__main__":
    sys.exit(main())
