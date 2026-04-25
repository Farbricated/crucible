"""
Unified LLM Client for CRUCIBLE.

Priority order:
  1. Groq API                    (uses GROQ_API_KEY — fastest inference)
  2. HuggingFace Inference API   (uses HF_TOKEN)
  3. Anthropic Claude            (uses ANTHROPIC_API_KEY — fallback)

Set environment variables:
  GROQ_API_KEY=gsk_xxx       → uses Groq chat completions
  HF_TOKEN=hf_xxx            → uses HuggingFace serverless inference
  ANTHROPIC_API_KEY=sk-xxx   → fallback if others unavailable
  LLM_BACKEND=groq|hf|anthropic|auto
"""

import os
import json
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")  # auto | groq | hf | anthropic

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# HF model — change to 70B for higher quality if budget allows
HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Anthropic model (fallback)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

MAX_RETRIES = 3
_hf_client = None
_anthropic_client = None
_groq_client = None

# ─────────────────────────────────────────────────────────────
# Live call-counter (visible on dashboard)
# ─────────────────────────────────────────────────────────────
_call_stats: dict = {
    "groq":      {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "hf":        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "anthropic": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}


def get_call_stats() -> dict:
    """Return a copy of the per-backend call statistics."""
    return {k: dict(v) for k, v in _call_stats.items()}


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(
            model=HF_MODEL,
            token=HF_TOKEN or None,
        )
    return _hf_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        import groq as groq_sdk
        _groq_client = groq_sdk.Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY or None)
    return _anthropic_client


def _backend() -> str:
    """Determine which backend to use."""
    if LLM_BACKEND == "groq":
        return "groq"
    if LLM_BACKEND == "hf":
        return "hf"
    if LLM_BACKEND == "anthropic":
        return "anthropic"
    # Auto: prefer Groq if key available
    if GROQ_API_KEY:
        return "groq"
    # Next prefer HF if token available
    if HF_TOKEN:
        return "hf"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    # Last resort: try Groq first, then HF without token
    return "groq"


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
            if backend == "groq":
                return _complete_groq(system_prompt, user_prompt, max_tokens, temperature)
            if backend == "hf":
                return _complete_hf(system_prompt, user_prompt, max_tokens, temperature)
            else:
                return _complete_anthropic(system_prompt, user_prompt, max_tokens)
        except Exception as exc:
            last_error = exc
            # On Groq failure, try HF then Anthropic once
            if backend == "groq" and attempt == 1:
                if HF_TOKEN:
                    try:
                        return _complete_hf(system_prompt, user_prompt, max_tokens, temperature)
                    except Exception:
                        pass
                if ANTHROPIC_API_KEY:
                    try:
                        return _complete_anthropic(system_prompt, user_prompt, max_tokens)
                    except Exception:
                        pass
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
    _call_stats["hf"]["calls"] += 1
    if hasattr(response, "usage") and response.usage:
        _call_stats["hf"]["prompt_tokens"]     += getattr(response.usage, "prompt_tokens", 0) or 0
        _call_stats["hf"]["completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
        _call_stats["hf"]["total_tokens"]      += getattr(response.usage, "total_tokens", 0) or 0
    return response.choices[0].message.content.strip()


def _complete_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Track usage
    _call_stats["groq"]["calls"] += 1
    if resp.usage:
        _call_stats["groq"]["prompt_tokens"]     += resp.usage.prompt_tokens or 0
        _call_stats["groq"]["completion_tokens"] += resp.usage.completion_tokens or 0
        _call_stats["groq"]["total_tokens"]      += resp.usage.total_tokens or 0

    return resp.choices[0].message.content.strip()


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
    _call_stats["anthropic"]["calls"] += 1
    if hasattr(response, "usage") and response.usage:
        _call_stats["anthropic"]["prompt_tokens"]     += getattr(response.usage, "input_tokens", 0) or 0
        _call_stats["anthropic"]["completion_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
        _call_stats["anthropic"]["total_tokens"]      += (
            (getattr(response.usage, "input_tokens", 0) or 0) +
            (getattr(response.usage, "output_tokens", 0) or 0)
        )
    return response.content[0].text.strip()


def active_backend() -> str:
    """Return a human-readable string showing which backend is active."""
    b = _backend()
    if b == "groq":
        return f"Groq ({GROQ_MODEL})"
    if b == "hf":
        model = HF_MODEL.split("/")[-1]
        return f"HuggingFace ({model})"
    return f"Anthropic ({ANTHROPIC_MODEL})"


def backend_info() -> dict:
    """Return backend configuration as a dict (for logging/dashboard)."""
    return {
        "backend": _backend(),
        "groq_model": GROQ_MODEL,
        "groq_key_set": bool(GROQ_API_KEY),
        "hf_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "active": active_backend(),
    }
