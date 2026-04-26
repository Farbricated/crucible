"""
Canonical LLM client for CRUCIBLE.

Two backends, selected by LLM_BACKEND env var:
  groq  — Groq SDK (llama-3.3-70b-versatile). DEV mode.
  hf    — HuggingFace Inference API. SUBMISSION mode.

Key behaviours:
  - Per-agent max_tokens enforced (Arbiter=450, Executor=600, Architect=400, Vendor=500)
  - Groq 429: parse exact retry-after from error message, sleep that duration, max 5 retries
  - Groq daily budget: module-level counter raises BudgetExhausted before the call if over
  - HF 503: sleep 20s and retry up to 3 times (model loading)
  - Every call logged to data/groq_logs/YYYY-MM-DD.jsonl
"""

import datetime
import json
import re
import time
from pathlib import Path

import requests

from core.config import (
    AGENT_MAX_TOKENS,
    GROQ_API_KEY,
    GROQ_DAILY_LIMIT,
    GROQ_MODEL,
    HF_MODEL,
    HF_TOKEN,
    LLM_BACKEND,
)

# ── Daily token budget ────────────────────────────────────────
TOKENS_USED_TODAY: int = 0
_budget_date: datetime.date | None = None

LOG_DIR = Path("data/groq_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class BudgetExhausted(RuntimeError):
    """Raised when the Groq daily token budget would be exceeded."""


class LLMError(RuntimeError):
    """Raised for unrecoverable LLM call failures."""


# ── Public API ────────────────────────────────────────────────

def call_llm(messages: list[dict], agent: str, system: str) -> str:
    """
    Call the active backend.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."} dicts
        agent:    one of "arbiter", "executor", "architect", "vendor"
        system:   static system prompt string (never an f-string)
    Returns:
        The assistant's text response.
    """
    if agent not in AGENT_MAX_TOKENS:
        raise ValueError(f"Unknown agent '{agent}'. Must be one of {list(AGENT_MAX_TOKENS)}")
    max_tokens = AGENT_MAX_TOKENS[agent]

    if LLM_BACKEND == "hf":
        return _call_hf(messages, system, max_tokens, agent)
    return _call_groq(messages, system, max_tokens, agent)


# ── Groq backend ──────────────────────────────────────────────

def _call_groq(
    messages: list[dict],
    system: str,
    max_tokens: int,
    agent: str,
    attempt: int = 0,
) -> str:
    global TOKENS_USED_TODAY, _budget_date

    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Export it or add to .env.")

    # Reset counter at midnight
    today = datetime.date.today()
    if _budget_date != today:
        TOKENS_USED_TODAY = 0
        _budget_date = today

    # Rough input estimate: 1 token ≈ 4 chars
    estimated_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4
    estimated_total = estimated_in + max_tokens
    if TOKENS_USED_TODAY + estimated_total > GROQ_DAILY_LIMIT:
        remaining = GROQ_DAILY_LIMIT - TOKENS_USED_TODAY
        raise BudgetExhausted(
            f"Groq daily budget exhausted — {TOKENS_USED_TODAY:,}/{GROQ_DAILY_LIMIT:,} used. "
            f"Remaining: {remaining:,} tokens. Resets at midnight UTC."
        )

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        used = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
        TOKENS_USED_TODAY += used
        _log(agent, resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0,
             latency_ms, "ok")
        return resp.choices[0].message.content

    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        err = str(exc)
        if "429" in err and attempt < 5:
            wait = _parse_wait(err)
            _log(agent, 0, 0, latency_ms, f"429_retry_in_{wait:.1f}s")
            print(f"[Groq 429] Agent={agent} attempt={attempt+1}/5 — sleeping {wait:.1f}s")
            time.sleep(wait)
            return _call_groq(messages, system, max_tokens, agent, attempt + 1)
        _log(agent, 0, 0, latency_ms, f"error:{err[:80]}")
        raise LLMError(f"Groq call failed (agent={agent}, attempt={attempt}): {err}") from exc


def _parse_wait(err: str) -> float:
    """Extract retry-after seconds from a Groq 429 error message."""
    m = re.search(r"try again in ([\d.]+)([sm])", err, re.IGNORECASE)
    if not m:
        return 10.0
    val, unit = float(m.group(1)), m.group(2)
    return val * 60 if unit == "m" else val


# ── HF Inference API backend ──────────────────────────────────

def _call_hf(
    messages: list[dict],
    system: str,
    max_tokens: int,
    agent: str,
) -> str:
    if not HF_TOKEN:
        raise LLMError("HF_TOKEN is not set. Export it or add to .env.")

    url = (
        "https://api-inference.huggingface.co/models/"
        f"{HF_MODEL}/v1/chat/completions"
    )
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    t0 = time.monotonic()
    for attempt in range(3):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 503:
            _log(agent, 0, 0, latency_ms, f"hf_503_retry_{attempt+1}")
            print(f"[HF 503] Model loading — sleeping 20s (attempt {attempt+1}/3)")
            time.sleep(20)
            continue
        if not r.ok:
            _log(agent, 0, 0, latency_ms, f"hf_error_{r.status_code}")
            raise LLMError(f"HF API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        _log(agent, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
             latency_ms, "ok")
        return content

    raise LLMError("HF Inference API unavailable after 3 retries (503 model loading)")


# ── Call logger ───────────────────────────────────────────────

def _log(
    agent: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    status: str,
) -> None:
    entry = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "agent": agent,
        "backend": LLM_BACKEND,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "status": status,
        "daily_total": TOKENS_USED_TODAY,
    }
    log_file = LOG_DIR / f"{datetime.date.today()}.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # never crash the episode just because logging failed


def get_budget_status() -> dict:
    """Return current daily token usage."""
    return {
        "budget": GROQ_DAILY_LIMIT,
        "used": TOKENS_USED_TODAY,
        "remaining": max(0, GROQ_DAILY_LIMIT - TOKENS_USED_TODAY),
        "pct_used": round(TOKENS_USED_TODAY / GROQ_DAILY_LIMIT * 100, 1),
    }
