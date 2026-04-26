# CRUCIBLE Reusable Prompt Pack

## Cursor Chat Prompt (new feature/file)

I'm building CRUCIBLE for the OpenEnv Hackathon (Meta PyTorch × Scaler, Bangalore, Apr 25–26 2026).

Context:
- 4-agent procurement compliance RL env: Executor, Arbiter (frozen), Architect, Vendor
- Plus a RegulationShock engine that fires mid-episode 30% of the time
- OpenEnv v0.2.3, FastAPI server, deployed to HuggingFace Spaces
- DEV: Groq API (llama-3.1-8b-instant, 500K TPD free tier)
- SUBMISSION: HF Inference API ($30 credit, llama-3.1-8b-instruct)

Real usage numbers from my Groq logs:
- 756 requests in 13 hours, 58.2% success, 41.8% rate-limited (429)
- Avg input: 1,209 tokens | Avg output: 381 tokens
- Crash hours: 18:00 (176K tokens) and 22:00 (301K tokens) — both hit 500K TPD ceiling
- Rate limit is TPD (daily), not TPM — retry-after is in the error message

Task: [DESCRIBE WHAT YOU WANT HERE]

Requirements:
1. OpenEnv base class (Environment) — implement reset(), step(), state() only
2. Client/server separation — client.py never imports from server/
3. Static system prompts only — all dynamic content in user turn
4. max_tokens: Arbiter=450, Executor=600, Architect=400, Vendor=500
5. Groq 429 handling: parse exact wait time from error with regex r"try again in ([\d.]+)([sm])", convert to seconds, sleep that exact duration before retry
6. JSON-only responses from all agents — "Return JSON only." at end of every system prompt
7. Pydantic models for all inputs/outputs
8. For submission mode: swap groq.Groq() for HF Inference API call, same prompt structure

Judging criteria to keep in mind:
- 40% environment innovation (adversarial Vendor, self-improving Architect, RegulationShocks, multi-jurisdiction)
- 30% storytelling (Streamlit dashboard, README, counterfactual consequences from Arbiter)
- 20% reward improvement evidence (before/after plots, +78% from 0.433?0.771 over 80 episodes)
- 10% pipeline quality (composable reward formula, GRPO-ready, token-aware rate limiter)

---

## Agent System Prompts

### agents/arbiter.py
```python
ARBITER_SYSTEM = """Score a procurement compliance analysis on 5 axes.

Weights: correctness=0.35, completeness=0.25, reasoning_transparency=0.20, efficiency=0.10, generalization_signal=0.10

Each axis: float 0.0–1.0. Produce weighted_total and a 1-sentence consequence_if_approved.

Return JSON only:
{"correctness":0.0,"completeness":0.0,"reasoning_transparency":0.0,"efficiency":0.0,"generalization_signal":0.0,"weighted_total":0.0,"consequence_if_approved":"..."}"""
```

### agents/executor.py
```python
EXECUTOR_SYSTEM = """Procurement compliance analyst at AXIOM Corp (aerospace/defense).

Detect violations. Frameworks: FAR, DFARS, EU Directive 2014/24/EU.
Common violations: expired SAM.gov registration, undisclosed OCI, missing mandatory clauses, ITAR breach, defective cost pricing, debarred vendor, TINA threshold breach.

Return JSON only:
{"decision":"COMPLIANT|NON-COMPLIANT","violations_found":["FAR X.XXX — reason"],"reasoning":"...","confidence":0.0}"""
```

### agents/architect.py
```python
ARCHITECT_SYSTEM = """Generate the next compliance training task targeting the agent's weakest axis.

Learning band: 0.45–0.70. Last score < 0.45 ? reduce difficulty. Last score > 0.70 ? escalate.
Escalation: hide violations in boilerplate, chain two conflicting frameworks, correct clause number with wrong threshold, vendor with undisclosed prior violations.

Return JSON only:
{"task_id":"...","scenario":"...","difficulty":"easy|medium|hard|expert","target_axis":"...","jurisdiction":"FAR|DFARS|EU","violations_to_hide":["..."]}"""
```

### agents/vendor.py
```python
VENDOR_SYSTEM = """Adversarial vendor. Write a contract that looks compliant but hides real violations.

Concealment: correct FAR clause number with wrong threshold, list many clauses to obscure a missing one, vague language instead of specific required terms, reference expired registrations without flagging them, verbal approval instead of required written consent.

Return JSON only:
{"contract_text":"...","hidden_violations":["..."],"concealment_techniques_used":["..."]}"""
```

---

## Dual-Mode LLM Client Prompt

```text
Write core/llm_client.py with two backends selectable by LLM_BACKEND env var:

BACKEND = os.getenv("LLM_BACKEND", "groq")  # "groq" | "hf"

groq backend:
- Use groq.Groq() SDK
- Model: llama-3.1-8b-instant
- On 429: parse wait seconds from error message using re.search(r"try again in ([\d.]+)([sm])", str(e))
  Convert: if unit=="m" multiply by 60. Sleep that exact duration. Retry up to 5 times.
- Module-level TOKENS_USED = 0, DAILY_LIMIT = 480000 (leave 20K buffer from 500K)
  Before each call: if TOKENS_USED + estimated_tokens > DAILY_LIMIT raise BudgetExhausted
  After each success: TOKENS_USED += input_tokens + output_tokens
- Per-call max_tokens from caller, never default to None

hf backend:
- Use requests.post to https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct
- Auth: Bearer HF_TOKEN from env
- Same message format as Groq (messages list with role/content)
- Retry on 503 (model loading) with 20s sleep, max 3 retries

Both backends:
- Accept: model_messages: list[dict], max_tokens: int, system: str
- Return: str (the assistant's text content)
- Log each call: {backend, input_tokens, output_tokens, latency_ms, status} to data/groq_logs/{date}.jsonl
- Raise clean LLMError on unrecoverable failure

Pydantic model for log entries.
```
