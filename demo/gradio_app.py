"""CRUCIBLE Gradio dashboard — run via: python main.py dashboard

10-panel mission-control dashboard covering all judging criteria:
 1. Reward Curve (interactive Plotly)
 2. Adversarial Arms Race (Vendor vs Executor)
 3. Shock Adaptation Timeline
 4. Axis × Difficulty Heatmap
 5. Jurisdiction Comparison
 6. Agent Status Cards
 7. Episode Walkthrough (qualitative before/after)
 8. Counterfactual Consequence Explorer
 9. Prior Work Comparison Table
10. Run Control + Backend Status
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.config import GROQ_DAILY_LIMIT, GROQ_MODEL, HF_MODEL, LLM_BACKEND

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "data", "episode_logs")
FULL_LOG = os.path.join(LOG_DIR, "full_run.json")
GROQ_LOG_DIR = os.path.join(ROOT, "data", "groq_logs")

COMMANDS: dict[str, str] = {
    "Single Episode": "episode",
    "Baseline": "baseline",
    "Train": "train",
    "Architect": "architect",
    "Adversarial": "adversarial",
    "Shock": "shock",
    "EU": "eu",
    "Full Pipeline": "full",
    "Plots": "plots",
}

# ── Colors ────────────────────────────────────────────────────
C = {
    "bg": "#0f172a", "card": "#1e293b", "accent": "#10b981",
    "red": "#ef4444", "orange": "#f97316", "blue": "#60a5fa",
    "gold": "#fbbf24", "purple": "#a78bfa", "gray": "#94a3b8",
    "green": "#4ade80", "white": "#f1f5f9",
}

PLOTLY_TEMPLATE = "plotly_dark"


def _load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    return out


def load_history() -> list[dict[str, Any]]:
    if os.path.exists(FULL_LOG):
        try:
            with open(FULL_LOG, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


# ── Panel 1: Reward Curve ─────────────────────────────────────
def build_reward_curve(history: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No episode data yet", template=PLOTLY_TEMPLATE)
        return fig

    df = pd.DataFrame(history).copy()
    df["episode_num"] = range(1, len(df) + 1)
    for col in ("final_reward", "score_1", "score_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rolling = df["final_reward"].rolling(5, min_periods=1).mean()

    if "score_1" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["episode_num"], y=df["score_1"], mode="lines",
            name="Attempt 1", line=dict(color=C["gray"], width=1, dash="dot"), opacity=0.6))

    fig.add_trace(go.Scatter(
        x=df["episode_num"], y=df["final_reward"], mode="lines+markers",
        name="Final Reward", line=dict(color=C["green"], width=2),
        marker=dict(size=4)))

    fig.add_trace(go.Scatter(
        x=df["episode_num"], y=rolling, mode="lines",
        name="Rolling Avg (5)", line=dict(color=C["red"], width=3)))

    # Architect activation line
    arch_eps = df[df.get("architect_active", pd.Series(dtype=bool)) == True]
    if len(arch_eps) > 0:
        first_arch = arch_eps["episode_num"].iloc[0]
        fig.add_vline(x=first_arch, line=dict(color=C["orange"], width=2, dash="dash"),
                      annotation_text="Architect ON", annotation_position="top right")

    # Breakthroughs
    bt = df[df.get("is_breakthrough", pd.Series(dtype=bool)) == True]
    if len(bt) > 0:
        fig.add_trace(go.Scatter(
            x=bt["episode_num"], y=bt["final_reward"], mode="markers",
            name="Breakthrough", marker=dict(symbol="star", size=14, color=C["gold"],
                                             line=dict(width=1, color="black"))))

    fig.add_hrect(y0=0.45, y1=0.70, fillcolor="rgba(16,185,129,0.10)", line_width=0,
                  annotation_text="Learning Band", annotation_position="top left")
    fig.update_layout(
        title="📈 Executor Reward Curve — Self-Improving Curriculum",
        xaxis_title="Episode", yaxis_title="Reward (0–1)", yaxis_range=[0, 1.05],
        template=PLOTLY_TEMPLATE, height=400, margin=dict(t=60, l=40, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ── Panel 2: Adversarial Arms Race ───────────────────────────
def build_adversarial_chart(history: list[dict]) -> go.Figure:
    adv = [r for r in history if r.get("adversarial") and r.get("vendor_reward") is not None]
    fig = go.Figure()
    if not adv:
        fig.update_layout(title="No adversarial episodes yet", template=PLOTLY_TEMPLATE)
        return fig

    eps = [r["episode"] for r in adv]
    vr = [r["vendor_reward"] for r in adv]
    er = [r["final_reward"] for r in adv]

    fig.add_trace(go.Scatter(x=eps, y=er, mode="lines+markers", name="Executor",
                             line=dict(color=C["green"], width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=eps, y=vr, mode="lines+markers", name="Vendor",
                             line=dict(color=C["red"], width=2), marker=dict(size=6, symbol="x")))
    fig.add_hline(y=0.5, line=dict(color=C["gray"], dash="dash", width=1))

    fig.update_layout(
        title="⚔️ Adversarial Arms Race — Vendor vs Executor",
        xaxis_title="Episode", yaxis_title="Reward", yaxis_range=[0, 1.05],
        template=PLOTLY_TEMPLATE, height=350, margin=dict(t=60, l=40, r=20, b=40))
    return fig


# ── Panel 3: Shock Adaptation ─────────────────────────────────
def build_shock_chart(history: list[dict]) -> go.Figure:
    shocks = [r for r in history if r.get("shock_fired")]
    fig = go.Figure()
    if not shocks:
        fig.update_layout(title="No shock episodes yet", template=PLOTLY_TEMPLATE)
        return fig

    eps = [r["episode"] for r in shocks]
    adapted = [1 if r.get("shock_adapted") else 0 for r in shocks]
    rolling = []
    for i in range(len(adapted)):
        vals = adapted[max(0, i - 4):i + 1]
        rolling.append(sum(vals) / len(vals))

    fig.add_trace(go.Bar(x=eps, y=adapted, name="Adapted",
                         marker_color=[C["green"] if a else C["red"] for a in adapted], opacity=0.5))
    fig.add_trace(go.Scatter(x=eps, y=rolling, mode="lines", name="Rolling Rate (5)",
                             line=dict(color=C["blue"], width=3)))
    fig.add_hline(y=0.5, line=dict(color=C["gray"], dash="dash"))

    fig.update_layout(
        title="⚡ Regulation Shock Adaptation Rate",
        xaxis_title="Episode (shock only)", yaxis_title="Adaptation", yaxis_range=[0, 1.1],
        template=PLOTLY_TEMPLATE, height=350, margin=dict(t=60, l=40, r=20, b=40))
    return fig


# ── Panel 4: Axis Heatmap ─────────────────────────────────────
def build_heatmap(history: list[dict]) -> go.Figure:
    axes = ["correctness", "completeness", "reasoning_transparency", "efficiency", "generalization_signal"]
    diffs = ["easy", "medium", "hard", "expert"]
    matrix = np.zeros((len(diffs), len(axes)))
    totals = {d: 0 for d in diffs}

    for r in history:
        d, a = r.get("difficulty", ""), r.get("weakest_axis", "")
        if d in diffs and a in axes:
            matrix[diffs.index(d)][axes.index(a)] += 1
            totals[d] += 1

    for i, d in enumerate(diffs):
        if totals[d] > 0:
            matrix[i] /= totals[d]

    short = ["Correct.", "Complete.", "Reasoning", "Effic.", "General."]
    text = [[f"{matrix[i][j]:.0%}" for j in range(len(axes))] for i in range(len(diffs))]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=short, y=[d.upper() for d in diffs],
        colorscale="YlOrRd", zmin=0, zmax=0.5, text=text, texttemplate="%{text}"))
    fig.update_layout(
        title="🔥 Failure Heatmap — Axis × Difficulty",
        template=PLOTLY_TEMPLATE, height=300, margin=dict(t=60, l=60, r=20, b=40))
    return fig


# ── Panel 5: Reward Decomposition ─────────────────────────────
def build_decomposition_chart(history: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No data yet", template=PLOTLY_TEMPLATE)
        return fig

    eps = [r["episode"] for r in history]
    decomp = [r.get("reward_decomposition", {}) for r in history]
    base = [d.get("base_score", r["score_2"]) for d, r in zip(decomp, history)]
    delta = [d.get("delta_bonus", 0) for d in decomp]
    coherence = [d.get("coherence_bonus", 0) for d in decomp]
    shock = [d.get("shock_bonus", 0) for d in decomp]

    fig.add_trace(go.Scatter(x=eps, y=base, fill="tozeroy", name="Base (5-axis)",
                             line=dict(color=C["green"]), fillcolor="rgba(16,185,129,0.3)"))
    cum = [b + d for b, d in zip(base, delta)]
    fig.add_trace(go.Scatter(x=eps, y=cum, fill="tonexty", name="+Delta",
                             line=dict(color=C["blue"]), fillcolor="rgba(96,165,250,0.3)"))
    cum2 = [c + co for c, co in zip(cum, coherence)]
    fig.add_trace(go.Scatter(x=eps, y=cum2, fill="tonexty", name="+Coherence",
                             line=dict(color=C["purple"]), fillcolor="rgba(167,139,250,0.3)"))
    cum3 = [c + s for c, s in zip(cum2, shock)]
    fig.add_trace(go.Scatter(x=eps, y=cum3, fill="tonexty", name="+Shock",
                             line=dict(color=C["gold"]), fillcolor="rgba(251,191,36,0.3)"))
    finals = [r["final_reward"] for r in history]
    fig.add_trace(go.Scatter(x=eps, y=finals, mode="lines", name="Final Reward",
                             line=dict(color=C["red"], width=2)))
    fig.update_layout(
        title="📊 Reward Decomposition — Where Does the Signal Come From?",
        xaxis_title="Episode", yaxis_title="Reward Components", yaxis_range=[0, 1.15],
        template=PLOTLY_TEMPLATE, height=350, margin=dict(t=60, l=40, r=20, b=40))
    return fig


# ── Panel 6: Jurisdiction Comparison ──────────────────────────
def build_jurisdiction_chart(history: list[dict]) -> go.Figure:
    from collections import defaultdict
    jur_rewards = defaultdict(list)
    for r in history:
        jur_rewards[r.get("jurisdiction", "FAR")].append(r["final_reward"])

    fig = go.Figure()
    colors_map = {"FAR": C["green"], "DFARS": C["blue"], "EU": C["orange"]}
    for jur in sorted(jur_rewards):
        vals = jur_rewards[jur]
        avg = sum(vals) / len(vals)
        fig.add_trace(go.Bar(
            x=[jur], y=[avg], name=jur,
            marker_color=colors_map.get(jur, C["purple"]),
            text=[f"n={len(vals)}<br>{avg:.3f}"], textposition="outside"))

    fig.add_hline(y=0.45, line=dict(color=C["blue"], dash="dash"), annotation_text="Band floor")
    fig.add_hline(y=0.70, line=dict(color=C["green"], dash="dash"), annotation_text="Band ceiling")
    fig.update_layout(
        title="🌍 Cross-Jurisdiction Generalization",
        yaxis_range=[0, 1.0], yaxis_title="Avg Reward",
        template=PLOTLY_TEMPLATE, height=300, margin=dict(t=60, l=40, r=20, b=40),
        showlegend=False)
    return fig


# ── Panel 6: Agent Status Cards ──────────────────────────────
def build_agent_cards(history: list[dict]) -> str:
    n = len(history)
    if not n:
        return "### No episodes run yet"

    df = pd.DataFrame(history)
    avg_rw = float(df["final_reward"].mean())
    last_rw = float(df["final_reward"].iloc[-1])
    shocks = int(df["shock_fired"].fillna(False).sum()) if "shock_fired" in df.columns else 0
    shock_adapted = int(df["shock_adapted"].fillna(False).sum()) if "shock_adapted" in df.columns else 0
    bts = int(df["is_breakthrough"].fillna(False).sum()) if "is_breakthrough" in df.columns else 0
    adv_eps = int(df["adversarial"].fillna(False).sum()) if "adversarial" in df.columns else 0

    first10 = df["final_reward"].iloc[:10].mean() if n >= 10 else avg_rw
    last10 = df["final_reward"].iloc[-10:].mean() if n >= 10 else avg_rw
    improvement = ((last10 - first10) / first10 * 100) if first10 > 0 else 0

    return f"""
### 🟢 Executor (Compliance Analyst)
- Episodes: **{n}** | Avg Reward: **{avg_rw:.3f}** | Latest: **{last_rw:.3f}**
- Improvement: **+{improvement:.0f}%** ({first10:.3f} → {last10:.3f})
- Breakthroughs: **{bts}**

### 🔵 Arbiter (Frozen Judge)
- 5-axis scorer: correctness (35%) · completeness (25%) · reasoning (20%) · efficiency (10%) · generalization (10%)
- Status: **FROZEN** — never adapts, never goes easy

### 🟡 Architect (Curriculum Generator)
- Active from episode **51** | Calibration target: **0.45–0.70**
- Generates tasks targeting Executor's weakest axis

### 🔴 Vendor (Adversarial Red-Team)
- Adversarial episodes: **{adv_eps}** | Concealment techniques: **10+**
- Reward = 1.0 − executor.correctness (zero-sum)

### ⚡ Regulation Shock Engine
- Shocks fired: **{shocks}** | Adapted: **{shock_adapted}** ({shock_adapted/max(1,shocks)*100:.0f}%)
"""


# ── Panel 7: Episode Walkthrough ─────────────────────────────
def build_walkthrough(history: list[dict]) -> str:
    if len(history) < 10:
        return "### Need ≥10 episodes for walkthrough"

    early = history[2]
    late = history[-3]

    def _ep_text(r: dict, label: str) -> str:
        shock = "⚡ YES" if r.get("shock_fired") else "No"
        adapted = "✅ Adapted" if r.get("shock_adapted") else "❌ Not adapted" if r.get("shock_fired") else "N/A"
        adv = "⚔️ YES" if r.get("adversarial") else "No"
        cons = r.get("consequence_if_approved", "—")
        return f"""
#### {label} — Episode {r.get('episode', '?')}
| Metric | Value |
|---|---|
| Task | `{r.get('task_id', '?')}` |
| Difficulty | **{r.get('difficulty', '?')}** |
| Jurisdiction | **{r.get('jurisdiction', '?')}** |
| Attempt 1 Score | {r.get('score_1', 0):.3f} |
| Attempt 2 Score | {r.get('score_2', 0):.3f} |
| Delta (learning) | {r.get('delta', 0):+.3f} |
| **Final Reward** | **{r.get('final_reward', 0):.3f}** |
| Weakest Axis | `{r.get('weakest_axis', '?')}` |
| Adversarial | {adv} |
| Shock Fired | {shock} |
| Shock Adapted | {adapted} |
| Breakthrough | {'🌟 YES' if r.get('is_breakthrough') else 'No'} |
| Consequence | _{cons}_ |
"""

    return (
        "## 📖 Episode Walkthrough — Before vs After\n\n"
        + _ep_text(early, "🔴 BEFORE Training (Early)")
        + "\n---\n"
        + _ep_text(late, "🟢 AFTER Training (Late)")
        + f"\n\n> **Improvement:** {early.get('final_reward',0):.3f} → {late.get('final_reward',0):.3f} "
        f"(**+{((late.get('final_reward',0)-early.get('final_reward',0))/max(0.01,early.get('final_reward',0))*100):.0f}%**)"
    )


# ── Panel 8: Counterfactual Explorer ──────────────────────────
def build_counterfactual_table(history: list[dict]) -> pd.DataFrame:
    rows = []
    for r in history[-20:]:
        rows.append({
            "Episode": r.get("episode", ""),
            "Reward": round(r.get("final_reward", 0), 3),
            "Decision Quality": "✅ Correct" if r.get("final_reward", 0) > 0.6 else "⚠️ Weak" if r.get("final_reward", 0) > 0.4 else "❌ Failed",
            "Consequence": r.get("consequence_if_approved", "—"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Panel 9: Prior Work Comparison ────────────────────────────
PRIOR_WORK_MD = """
## 🏆 Prior Work Comparison — Why CRUCIBLE is Novel

| Feature | SWE-bench | WebArena | ALFWorld | ProcureBench | **CRUCIBLE** |
|---|---|---|---|---|---|
| Domain | Code repair | Web tasks | Household | Procurement | **Procurement** |
| Self-improving curriculum | ❌ | ❌ | ❌ | ❌ | **✅ Architect** |
| Adversarial agent | ❌ | ❌ | ❌ | ❌ | **✅ Vendor** |
| Mid-episode distribution shift | ❌ | ❌ | ❌ | ❌ | **✅ Reg. Shocks** |
| Multi-jurisdiction | ❌ | ❌ | ❌ | ❌ | **✅ FAR+DFARS+EU** |
| Counterfactual consequences | ❌ | ❌ | ❌ | ❌ | **✅ Arbiter** |
| Multi-axis scoring | ❌ | Binary | Binary | Binary | **✅ 5-axis weighted** |
| Learning band control | ❌ | ❌ | ❌ | ❌ | **✅ 0.45–0.70** |
| Zero-sum agent dynamics | ❌ | ❌ | ❌ | ❌ | **✅** |
| Task ceiling | Fixed | Fixed | Fixed | Fixed | **∞ No ceiling** |

> **No prior RL environment combines self-improving curriculum + adversarial dynamics + regulatory distribution shift + multi-jurisdiction generalization in a single coherent task.**
"""


# ── Backend Status ────────────────────────────────────────────
def _backend_status_text() -> str:
    backend = (os.getenv("LLM_BACKEND") or LLM_BACKEND or "groq").strip().lower()
    if backend == "hf":
        model = os.getenv("HF_MODEL") or HF_MODEL
        return f"### LLM Backend\n- Backend: **HF**\n- Model: **`{model}`**"

    model = os.getenv("GROQ_MODEL") or GROQ_MODEL
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    log_file = os.path.join(GROQ_LOG_DIR, f"{today}.jsonl")
    used = 0
    if os.path.exists(log_file):
        try:
            with open(log_file, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    used = max(used, int(obj.get("daily_total", used)))
        except Exception:
            used = 0
    remaining = max(0, GROQ_DAILY_LIMIT - used)
    pct = (used / GROQ_DAILY_LIMIT * 100) if GROQ_DAILY_LIMIT else 0
    return (
        "### LLM Backend\n"
        f"- Backend: **Groq**\n"
        f"- Model: **`{model}`**\n"
        f"- Daily usage: **{used:,}/{GROQ_DAILY_LIMIT:,}** ({pct:.1f}%)\n"
        f"- Remaining: **{remaining:,}**"
    )


# ── Run Command ───────────────────────────────────────────────
def run_command(label: str):
    cmd = COMMANDS.get(label, "episode")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    for k, v in _load_env().items():
        env.setdefault(k, v)

    proc = subprocess.run(
        [sys.executable, "main.py", cmd], cwd=ROOT,
        capture_output=True, text=True, env=env)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    status = f"Command: python main.py {cmd}\nExit code: {proc.returncode}"
    return status, output[-15000:]


# ── Refresh All ───────────────────────────────────────────────
def refresh_all():
    h = load_history()
    return (
        build_reward_curve(h),
        build_adversarial_chart(h),
        build_shock_chart(h),
        build_heatmap(h),
        build_jurisdiction_chart(h),
        build_decomposition_chart(h),
        build_agent_cards(h),
        build_walkthrough(h),
        build_counterfactual_table(h),
        _backend_status_text(),
    )


def run_and_refresh(label: str):
    status, output = run_command(label)
    results = refresh_all()
    return (status, output) + results


# ── Build App ─────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1400px !important; }
.panel-title { font-size: 1.1em; font-weight: 600; }
"""


def make_app() -> gr.Blocks:
    with gr.Blocks(
        title="CRUCIBLE — Mission Control",
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# 🔥 CRUCIBLE — Mission Control Dashboard\n"
            "*Self-Improving Multi-Agent RL for Procurement Compliance — "
            "Team Digital Yudh · OpenEnv Hackathon 2026*"
        )

        # ── Row 1: Controls ──────────────────────────────────
        with gr.Row():
            cmd = gr.Dropdown(list(COMMANDS.keys()), value="Single Episode", label="Command", scale=2)
            run_btn = gr.Button("▶ Run", variant="primary", scale=1)
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        status = gr.Textbox(label="Run Status", lines=2, visible=True)

        # ── Row 2: Main reward curve (full width) ────────────
        reward_curve = gr.Plot(label="Reward Curve")

        # ── Row 3: Adversarial + Shock ───────────────────────
        with gr.Row():
            adv_chart = gr.Plot(label="Adversarial Arms Race")
            shock_chart = gr.Plot(label="Shock Adaptation")

        # ── Row 4: Heatmap + Jurisdiction ────────────────────
        with gr.Row():
            heatmap = gr.Plot(label="Failure Heatmap")
            jur_chart = gr.Plot(label="Jurisdiction Comparison")

        # ── Row 5: Reward Decomposition (full width) ─────────
        decomp_chart = gr.Plot(label="Reward Decomposition")

        # ── Tabs: Deep Dive ──────────────────────────────────
        with gr.Tabs():
            with gr.Tab("🤖 Agent Status"):
                agent_cards = gr.Markdown()
                backend_md = gr.Markdown()

            with gr.Tab("📖 Episode Walkthrough"):
                walkthrough_md = gr.Markdown()

            with gr.Tab("💀 Counterfactual Explorer"):
                gr.Markdown("### What happens when the Executor gets it wrong?\n"
                            "The Arbiter generates a simulated consequence for every decision.")
                cf_table = gr.Dataframe(label="Recent Counterfactuals", wrap=True)

            with gr.Tab("🏆 Prior Work Comparison"):
                gr.Markdown(PRIOR_WORK_MD)

            with gr.Tab("📋 Command Output"):
                output_box = gr.Textbox(label="Command Output", lines=18)

        # ── Wiring ───────────────────────────────────────────
        all_outputs = [
            reward_curve, adv_chart, shock_chart, heatmap, jur_chart,
            decomp_chart,
            agent_cards, walkthrough_md, cf_table, backend_md,
        ]

        run_btn.click(
            run_and_refresh, inputs=[cmd],
            outputs=[status, output_box] + all_outputs,
        )

        refresh_btn.click(refresh_all, outputs=all_outputs)

        # Initial load
        demo.load(refresh_all, outputs=all_outputs)

    return demo


def launch() -> None:
    app = make_app()
    # On hosted runtimes (e.g. HF Spaces), localhost reachability checks can fail.
    # share=True avoids startup abort while still binding the expected host/port.
    app.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False, share=True)


if __name__ == "__main__":
    launch()
