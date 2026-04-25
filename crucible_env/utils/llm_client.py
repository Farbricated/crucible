"""
Unified LLM Client for CRUCIBLE.

Priority order:
  1. HuggingFace Inference API  (uses HF_TOKEN — spends your $30 HF credit)
  2. Anthropic Claude            (uses ANTHROPIC_API_KEY — fallback)

Set environment variables:
  HF_TOKEN=hf_xxx           → uses HuggingFace serverless inference
  ANTHROPIC_API_KEY=sk-xxx  → fallback if HF not available
  LLM_BACKEND=hf|anthropic  → force a specific backend

HuggingFace model options (good quality, low cost):
  - meta-llama/Meta-Llama-3.1-8B-Instruct   (~$0.0002/1K tokens)
  - Qwen/Qwen2.5-7B-Instruct                (~$0.0002/1K tokens)
  - mistralai/Mistral-7B-Instruct-v0.3      (~$0.0002/1K tokens)
  - meta-llama/Meta-Llama-3.3-70B-Instruct  (~$0.0009/1K tokens, better quality)

With $30 HF credit and Llama-3.1-8B:
  ~150,000 episodes at 1K tokens each — effectively unlimited for this hackathon.
"""

import os
import json
import re
import time
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")  # auto | hf | anthropic

# HF model — change to 70B for higher quality if budget allows
HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Anthropic model (fallback)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

MAX_RETRIES = 3
_hf_client = None
_anthropic_client = None


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(
            model=HF_MODEL,
            token=HF_TOKEN or None,
        )
    return _hf_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY or None)
    return _anthropic_client


def _backend() -> str:
    """Determine which backend to use."""
    if LLM_BACKEND == "hf":
        return "hf"
    if LLM_BACKEND == "anthropic":
        return "anthropic"
    # Auto: prefer HF if token available
    if HF_TOKEN:
        return "hf"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    # Last resort: try HF without token (rate-limited but works for light use)
    return "hf"


def complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.3,
) -> str:
    """
    Call the active LLM backend and return the raw text response.
    Falls back to Anthropic if HF fails.
    """
    backend = _backend()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if backend == "hf":
                return _complete_hf(system_prompt, user_prompt, max_tokens, temperature)
            else:
                return _complete_anthropic(system_prompt, user_prompt, max_tokens)
        except Exception as exc:
            last_error = exc
            # On HF failure, try Anthropic once
            if backend == "hf" and ANTHROPIC_API_KEY and attempt == 1:
                try:
                    return _complete_anthropic(system_prompt, user_prompt, max_tokens)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(0.8 * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")


def _complete_hf(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = _get_hf_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def _complete_anthropic(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def active_backend() -> str:
    """Return a human-readable string showing which backend is active."""
    b = _backend()
    if b == "hf":
        model = HF_MODEL.split("/")[-1]
        return f"HuggingFace ({model})"
    return f"Anthropic ({ANTHROPIC_MODEL})"


def backend_info() -> dict:
    """Return backend configuration as a dict (for logging/dashboard)."""
    return {
        "backend": _backend(),
        "hf_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "active": active_backend(),
    }
