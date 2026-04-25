"""Test the token-aware rate limiter and retry-after parser."""
import time
from utils.llm_client import (
    _parse_groq_retry_after, _estimate_tokens,
    _groq_limiter, backend_info, get_call_stats, active_backend,
    _GROQ_TPM_LIMIT, _GROQ_TPM_EFFECTIVE,
)

print("=== _parse_groq_retry_after ===")
cases = [
    ("Rate limit reached ... Please try again in 12.345s.", 12.845),
    ("please try again in 5s",                              5.5),
    ("Rate limit ... Please try again in 1m3.5s",           64.0),
    ("some other error with no time",                       10.0),
]
all_ok = True
for msg, expected in cases:
    got = _parse_groq_retry_after(msg)
    ok  = abs(got - expected) < 1.0
    all_ok = all_ok and ok
    print(f"  [{'OK' if ok else 'FAIL'}] got={got:.2f}  expected~{expected:.2f}")

print()
print("=== _estimate_tokens ===")
est = _estimate_tokens("system prompt here", "user message here", 300)
print(f"  Short prompt + max_tokens=300 -> estimate={est}")
assert 300 <= est <= 400, f"Unexpected estimate: {est}"
print("  OK")

print()
print("=== Sliding-window rate limiter ===")
# Use a tiny limit for a non-blocking test
_groq_limiter.tpm_effective = 200
from collections import deque
_groq_limiter._window = deque()   # reset

_groq_limiter.record(180)         # fill 180/200
t0 = time.monotonic()
_groq_limiter.acquire(10)         # 190/200 — should pass immediately
elapsed = time.monotonic() - t0
print(f"  acquire(10) after 180 used: {elapsed:.3f}s (expect <0.1s)")
assert elapsed < 0.5, "Should not have waited!"

print()
print("=== backend_info ===")
info = backend_info()
for k, v in info.items():
    print(f"  {k}: {v}")

print()
print(f"GROQ_TPM_LIMIT    : {_GROQ_TPM_LIMIT}")
print(f"GROQ_TPM_EFFECTIVE: {_GROQ_TPM_EFFECTIVE}  (85% safety margin)")
print(f"Active backend    : {active_backend()}")
print()
if all_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — check output above")
