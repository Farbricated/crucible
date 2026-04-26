"""CRUCIBLE Gradio dashboard — run via: python main.py dashboard"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
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

    out: list[dict[str, Any]] = []
    if os.path.exists(LOG_DIR):
        for fn in sorted(os.listdir(LOG_DIR)):
            if fn.startswith("ep_") and fn.endswith(".json") and fn not in ("full_run.json", "baseline.json"):
                try:
                    with open(os.path.join(LOG_DIR, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception:
                    pass
    return out


def build_metrics_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return (
            "### CRUCIBLE Status\n"
            "- Episodes: **0**\n"
            "- Avg reward: **—**\n"
            "- Latest reward: **—**\n"
            "- Shocks: **0**"
        )

    df = pd.DataFrame(history)
    for col in ("final_reward", "score_1", "delta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    episodes = len(df)
    avg_reward = float(df["final_reward"].mean())
    latest = float(df["final_reward"].iloc[-1])
    shocks = int(df["shock_fired"].fillna(False).sum()) if "shock_fired" in df.columns else 0

    return (
        "### CRUCIBLE Status\n"
        f"- Episodes: **{episodes}**\n"
        f"- Avg reward: **{avg_reward:.3f}**\n"
        f"- Latest reward: **{latest:.3f}**\n"
        f"- Shocks: **{shocks}**"
    )


def _backend_status_text() -> str:
    backend = (os.getenv("LLM_BACKEND") or LLM_BACKEND or "groq").strip().lower()
    if backend == "hf":
        model = os.getenv("HF_MODEL") or HF_MODEL
        return (
            "### LLM Backend\n"
            "- Backend: **HF**\n"
            f"- Model: **`{model}`**\n"
            "- Groq usage: **N/A**"
        )

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
        "- Backend: **Groq**\n"
        f"- Model: **`{model}`**\n"
        f"- Daily usage: **{used:,}/{GROQ_DAILY_LIMIT:,}** ({pct:.1f}%)\n"
        f"- Remaining: **{remaining:,}**"
    )


def build_curve(history: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No episode data yet")
        return fig

    df = pd.DataFrame(history).copy()
    df["episode_num"] = range(1, len(df) + 1)
    for col in ("final_reward", "score_1"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rolling = df["final_reward"].rolling(5, min_periods=1).mean()
    if "score_1" in df.columns:
        fig.add_trace(go.Scatter(x=df["episode_num"], y=df["score_1"], mode="lines", name="Attempt 1"))
    fig.add_trace(go.Scatter(x=df["episode_num"], y=df["final_reward"], mode="lines", name="Final reward"))
    fig.add_trace(go.Scatter(x=df["episode_num"], y=rolling, mode="lines", name="Rolling 5"))
    fig.add_hrect(y0=0.45, y1=0.70, fillcolor="rgba(16,185,129,0.12)", line_width=0)
    fig.update_layout(
        title="Reward Curve",
        xaxis_title="Episode",
        yaxis_title="Reward",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        height=350,
        margin=dict(t=50, l=20, r=20, b=40),
    )
    return fig


def run_command(label: str) -> tuple[str, str, str, go.Figure, pd.DataFrame]:
    cmd = COMMANDS.get(label, "episode")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    for k, v in _load_env().items():
        env.setdefault(k, v)

    proc = subprocess.run(
        [sys.executable, "main.py", cmd],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    status = f"Command: python main.py {cmd}\nExit code: {proc.returncode}"

    history = load_history()
    metrics = build_metrics_text(history)
    backend_text = _backend_status_text()
    fig = build_curve(history)
    table = pd.DataFrame(history).tail(20)
    return status, output[-20000:], metrics, fig, table


def refresh_dashboard() -> tuple[str, str, go.Figure, pd.DataFrame]:
    history = load_history()
    return (
        build_metrics_text(history),
        _backend_status_text(),
        build_curve(history),
        pd.DataFrame(history).tail(20),
    )


def make_app() -> gr.Blocks:
    with gr.Blocks(title="CRUCIBLE Gradio Dashboard") as demo:
        gr.Markdown("# CRUCIBLE — Gradio Mission Control")
        gr.Markdown("Run backend episodes/training and monitor rewards, logs, and Groq budget.")

        with gr.Row():
            cmd = gr.Dropdown(list(COMMANDS.keys()), value="Single Episode", label="Command")
            run_btn = gr.Button("Run Command", variant="primary")
            refresh_btn = gr.Button("Refresh Metrics")

        status = gr.Textbox(label="Run Status", lines=2)
        with gr.Row():
            metrics = gr.Markdown()
            backend = gr.Markdown()
        curve = gr.Plot(label="Reward Curve")
        with gr.Tab("Recent Episodes"):
            table = gr.Dataframe(label="Recent Episodes", wrap=True)
        with gr.Tab("Command Output"):
            output = gr.Textbox(label="Command Output", lines=18)

        run_btn.click(
            run_command,
            inputs=[cmd],
            outputs=[status, output, metrics, curve, table],
        )

        refresh_btn.click(
            refresh_dashboard,
            outputs=[metrics, backend, curve, table],
        )

        # Initial load
        demo.load(refresh_dashboard, outputs=[metrics, backend, curve, table])

    return demo


def launch() -> None:
    app = make_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)


if __name__ == "__main__":
    launch()

