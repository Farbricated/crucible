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
  GROQ_TPM_LIMIT=6000        → tokens-per-minute cap (default 6000)
"""

import os
import re
import time
import threading
from collections import deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")  # auto | groq | hf | anthropic

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# HF model — change to 70B for higher quality if budget allows
HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Anthropic model (fallback)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

MAX_RETRIES      = 5   # HF / Anthropic retry ceiling
GROQ_MAX_RETRIES = 5   # Groq: 429 + general error retries
_GROQ_BACKOFF_BASE = 2.0   # seconds — doubles each attempt, capped at 32s
_hf_client = None
_anthropic_client = None
_groq_client = None

# ─────────────────────────────────────────────────────────────
# Daily token budget (450K = 500K TPD limit minus 50K buffer)
# ─────────────────────────────────────────────────────────────
_GROQ_DAILY_BUDGET: int = int(os.getenv("GROQ_DAILY_BUDGET", "450000"))
_daily_tokens_used: int = 0
_daily_reset_date = None        # date object; reset counter when date changes
_daily_lock = threading.Lock()


def _check_and_record_daily(tokens: int, record: bool = False) -> None:
    """
    Pre-call: raise BudgetExceeded if daily limit would be breached.
    Post-call (record=True): add actual tokens to today's counter.
    Thread-safe.
    """
    global _daily_tokens_used, _daily_reset_date
    from datetime import date
    with _daily_lock:
        today = date.today()
        if _daily_reset_date != today:
            _daily_tokens_used = 0
            _daily_reset_date  = today
        if record:
            _daily_tokens_used += tokens
            return
        remaining = _GROQ_DAILY_BUDGET - _daily_tokens_used
        if tokens > remaining:
            raise RuntimeError(
                f"[BudgetExceeded] Groq daily budget exhausted: "
                f"{_daily_tokens_used:,}/{_GROQ_DAILY_BUDGET:,} tokens used today. "
                f"Remaining: {remaining:,}. Reset at midnight UTC."
            )


def get_daily_budget_status() -> dict:
    """Return current daily token usage for monitoring."""
    global _daily_tokens_used
    return {
        "budget": _GROQ_DAILY_BUDGET,
        "used": _daily_tokens_used,
        "remaining": max(0, _GROQ_DAILY_BUDGET - _daily_tokens_used),
        "pct_used": round(_daily_tokens_used / _GROQ_DAILY_BUDGET * 100, 1),
    }

# ─────────────────────────────────────────────────────────────
# Live call-counter (visible on dashboard)
# ─────────────────────────────────────────────────────────────
_call_stats: dict = {
    "groq":      {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "rate_limit_waits": 0, "wait_seconds": 0.0},
    "hf":        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                  "rate_limit_waits": 0, "wait_seconds": 0.0},
    "anthropic": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                  "rate_limit_waits": 0, "wait_seconds": 0.0},
}


def get_call_stats() -> dict:
    """Return a copy of the per-backend call statistics."""
    return {k: dict(v) for k, v in _call_stats.items()}


# ─────────────────────────────────────────────────────────────
# Token-aware sliding-window rate limiter for Groq
# ─────────────────────────────────────────────────────────────
_GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "6000"))
# Use 85% of the limit as a safety margin — stops us from ever touching 6000
_GROQ_TPM_EFFECTIVE = int(_GROQ_TPM_LIMIT * 0.85)   # 5100 tokens/min default


class _SlidingWindowRateLimiter:
    """
    Sliding 60-second window rate limiter.
    Tracks actual tokens consumed and pre-emptively sleeps when the
    budget for the next request would overflow the effective limit.
    """

    def __init__(self, tpm_effective: int):
        self.tpm_effective = tpm_effective
        self._window: deque = deque()   # (monotonic_timestamp, tokens)
        self._lock = threading.Lock()

    def _purge(self):
        cutoff = time.monotonic() - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _used(self) -> int:
        self._purge()
        return sum(t for _, t in self._window)

    def acquire(self, estimated_tokens: int):
        """Block until the window has room for estimated_tokens."""
        with self._lock:
            while True:
                used = self._used()
                if used + estimated_tokens <= self.tpm_effective:
                    break
                # Wait until the oldest entry slides out of the 60s window
                wait = (self._window[0][0] + 60.0) - time.monotonic() + 0.2
                if wait > 0:
                    print(
                        f"[RateLimiter] TPM budget {used}/{self.tpm_effective} — "
                        f"pre-emptive sleep {wait:.1f}s"
                    )
                    _call_stats["groq"]["rate_limit_waits"] += 1
                    _call_stats["groq"]["wait_seconds"]     += wait
                    time.sleep(wait)

    def record(self, tokens: int):
        with self._lock:
            self._window.append((time.monotonic(), tokens))


_groq_limiter = _SlidingWindowRateLimiter(_GROQ_TPM_EFFECTIVE)


# ─────────────────────────────────────────────────────────────
# 429 retry-after parser
# ─────────────────────────────────────────────────────────────
def _parse_groq_retry_after(error_msg: str) -> float:
    """
    Extract the suggested wait time from a Groq rate-limit error message.
    Format: '... Please try again in 12.345s.'
    Falls back to 10s if not parseable.
    """
    match = re.search(r"[Pp]lease try again in ([\d.]+)s", error_msg)
    if match:
        return float(match.group(1)) + 0.5   # small safety buffer
    # Secondary pattern: 'in Xm Ys'
    match2 = re.search(r"in (\d+)m([\d.]+)s", error_msg)
    if match2:
        return int(match2.group(1)) * 60 + float(match2.group(2)) + 0.5
    return 10.0   # conservative fallback


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
    Groq handles its own retries + rate limiting internally.
    HF and Anthropic fall back to each other on failure.
    """
    backend = _backend()

    if backend == "groq":
        # _complete_groq manages its own retry/rate-limit loop
        try:
            return _complete_groq(system_prompt, user_prompt, max_tokens, temperature)
        except Exception as exc:
            # Final fallback chain if Groq exhausts all retries
            if HF_TOKEN:
                try:
                    return _complete_hf(system_prompt, user_prompt, max_tokens, temperature)
                except Exception:
                    pass
            if ANTHROPIC_API_KEY:
                return _complete_anthropic(system_prompt, user_prompt, max_tokens)
            raise

    # ── HF / Anthropic path (simple retry) ──────────────────
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if backend == "hf":
                return _complete_hf(system_prompt, user_prompt, max_tokens, temperature)
            return _complete_anthropic(system_prompt, user_prompt, max_tokens)
        except Exception as exc:
            last_error = exc
            if backend == "hf" and ANTHROPIC_API_KEY and attempt == 1:
                try:
                    return _complete_anthropic(system_prompt, user_prompt, max_tokens)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(min(32.0, _GROQ_BACKOFF_BASE * (2 ** (attempt - 1))))

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


def _estimate_tokens(system_prompt: str, user_prompt: str, max_tokens: int) -> int:
    """
    Rough token estimate before the call so the rate limiter can pre-check budget.
    Rule of thumb: 1 token ≈ 4 chars.  Add max_tokens for worst-case output.
    """
    input_chars = len(system_prompt) + len(user_prompt)
    return (input_chars // 4) + max_tokens


def _complete_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    # ── Pre-flight: daily budget check then TPM rate limiter ─
    estimated = _estimate_tokens(system_prompt, user_prompt, max_tokens)
    _check_and_record_daily(estimated)          # raises if over daily budget
    _groq_limiter.acquire(estimated)            # blocks until per-minute window clears

    client = _get_groq_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Record actual tokens in sliding window and daily counter
            actual_tokens = resp.usage.total_tokens if resp.usage else estimated
            _groq_limiter.record(actual_tokens)
            _check_and_record_daily(actual_tokens, record=True)

            # Update global stats
            _call_stats["groq"]["calls"] += 1
            if resp.usage:
                _call_stats["groq"]["prompt_tokens"]     += resp.usage.prompt_tokens or 0
                _call_stats["groq"]["completion_tokens"] += resp.usage.completion_tokens or 0
                _call_stats["groq"]["total_tokens"]      += resp.usage.total_tokens or 0

            return resp.choices[0].message.content.strip()

        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

            if is_rate_limit:
                wait = _parse_groq_retry_after(err_str)
                _call_stats["groq"]["rate_limit_waits"] += 1
                _call_stats["groq"]["wait_seconds"]     += wait
                print(
                    f"[Groq 429] Rate limited (attempt {attempt}/{GROQ_MAX_RETRIES}). "
                    f"Waiting {wait:.1f}s as instructed..."
                )
                time.sleep(wait)
            else:
                # Non-rate-limit error: exponential backoff (2s base, 32s cap, 5 retries)
                if attempt >= GROQ_MAX_RETRIES:
                    raise
                backoff = min(32.0, _GROQ_BACKOFF_BASE * (2 ** (attempt - 1)))
                time.sleep(backoff)

    raise RuntimeError(f"Groq call failed after {GROQ_MAX_RETRIES} attempts")


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
    budget = get_daily_budget_status()
    return {
        "backend": _backend(),
        "groq_model": GROQ_MODEL,
        "groq_key_set": bool(GROQ_API_KEY),
        "groq_tpm_limit": _GROQ_TPM_LIMIT,
        "groq_tpm_effective": _GROQ_TPM_EFFECTIVE,
        "groq_window_used": _groq_limiter._used(),
        "groq_daily_budget": budget["budget"],
        "groq_daily_used": budget["used"],
        "groq_daily_remaining": budget["remaining"],
        "groq_daily_pct": budget["pct_used"],
        "hf_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "active": active_backend(),
    }
