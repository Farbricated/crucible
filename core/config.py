"""
Central configuration for CRUCIBLE.

All API keys and runtime settings live here.
Never hardcode keys anywhere else — import from this module.

Set these environment variables (or put them in .env):
  GROQ_API_KEY=gsk_xxx
  HF_TOKEN=hf_xxx
  LLM_BACKEND=groq        # "groq" | "hf"
  GROQ_DAILY_LIMIT=480000
"""
import os

# LLM credentials
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", "")

# Backend selector: "groq" for dev, "hf" for submission
LLM_BACKEND: str  = os.getenv("LLM_BACKEND", "groq")

# Models
# DEV  (Groq):  8b-instant — fast, low token cost, won't exhaust 480K daily budget
# SUBMISSION (HF): 70B — better quality, separate HF credits, set LLM_BACKEND=hf
GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_MODEL: str     = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

# Daily token budget (Groq free tier = 500K/day; leave 20K buffer)
# 70b-versatile real daily cap is ~100K tokens; leave 10K buffer
GROQ_DAILY_LIMIT: int = int(os.getenv("GROQ_DAILY_LIMIT", "90000"))

# Per-agent max_tokens (enforce strictly to avoid burning budget)
AGENT_MAX_TOKENS: dict[str, int] = {
    "arbiter":   450,
    "executor":  600,
    "architect": 400,
    "vendor":    500,
}
