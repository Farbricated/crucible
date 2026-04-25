"""
CRUCIBLE Demo Dashboard
Run with: streamlit run demo/dashboard.py

Shows:
  Panel 1 — Executor reward curve
  Panel 2 — Architect calibration accuracy
  Panel 3 — Architect reasoning (live)
  Panel 4 — Lineage map
"""

import json
import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

LOG_DIR = "data/episode_logs"
FULL_LOG = os.path.join(LOG_DIR, "full_run.json")

st.set_page_config(
    page_title="CRUCIBLE — Live Training Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 CRUCIBLE — Live Training Dashboard")
st.caption("Self-Improving RL Environment | AXIOM Corporation | OpenEnv Hackathon 2026")

# ── Auto-refresh ─────────────────────────────────────────────
refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if refresh:
    time.sleep(5)
    st.rerun()

# ── Load data ─────────────────────────────────────────────────
@st.cache_data(ttl=3)
def load_log():
    if not os.path.exists(FULL_LOG):
        return []
    with open(FULL_LOG) as f:
        return json.load(f)

@st.cache_data(ttl=3)
def load_episode_files():
    episodes = []
    if not os.path.exists(LOG_DIR):
        return episodes
    for fname in sorted(os.listdir(LOG_DIR)):
        if fname.startswith("ep-") and fname.endswith(".json"):
            with open(os.path.join(LOG_DIR, fname)) as f:
                try:
                    episodes.append(json.load(f))
                except Exception:
                    pass
    return episodes

history = load_log()
episodes = load_episode_files()

if not history:
    st.warning("No training data yet. Run `python training/grpo_loop.py 2` to start training.")
    st.stop()

df = pd.DataFrame(history)
df["episode_num"] = range(1, len(df) + 1)

# ── Summary metrics ───────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Episodes", len(df))
col2.metric("Avg Reward", f"{df['final_reward'].mean():.3f}")
col3.metric("Latest Reward", f"{df['final_reward'].iloc[-1]:.3f}")
col4.metric("Breakthroughs", int(df['is_breakthrough'].sum()))

if "in_band" in df.columns:
    in_band_rate = df[df["architect_active"]]["in_band"].mean() if df["architect_active"].any() else 0.0
    col5.metric("Arch Calibration", f"{in_band_rate:.1%}")
else:
    col5.metric("Arch Calibration", "N/A")

st.divider()

# ── Panel 1: Reward curve ─────────────────────────────────────
st.subheader("📈 Panel 1 — Executor Reward Curve")

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df["episode_num"], y=df["score_1"],
    mode="lines", name="Attempt 1", line=dict(color="#6366f1", dash="dot"), opacity=0.6
))
fig1.add_trace(go.Scatter(
    x=df["episode_num"], y=df["final_reward"],
    mode="lines+markers", name="Final Reward", line=dict(color="#10b981", width=2)
))

# Architect activation line
arch_start = df[df["architect_active"]]["episode_num"].min() if df["architect_active"].any() else None
if arch_start and not pd.isna(arch_start):
    fig1.add_vline(
        x=arch_start, line_dash="dash", line_color="orange",
        annotation_text="Architect Activated", annotation_position="top right"
    )

# Breakthroughs
breakthroughs = df[df["is_breakthrough"] == True]
if not breakthroughs.empty:
    fig1.add_trace(go.Scatter(
        x=breakthroughs["episode_num"], y=breakthroughs["final_reward"],
        mode="markers", name="Breakthrough 🎯",
        marker=dict(color="gold", size=12, symbol="star")
    ))

fig1.add_hrect(y0=0.45, y1=0.70, fillcolor="green", opacity=0.1,
               annotation_text="Learning Band (0.45–0.70)")
fig1.update_layout(
    xaxis_title="Episode", yaxis_title="Reward",
    yaxis=dict(range=[0, 1.05]),
    height=350, legend=dict(orientation="h")
)
st.plotly_chart(fig1, use_container_width=True)

# ── Panel 2: Architect calibration ───────────────────────────
st.subheader("🎯 Panel 2 — Architect Calibration Accuracy")

arch_df = df[df["architect_active"] == True].copy()
if arch_df.empty:
    st.info("Architect not active yet. Activates in Phase 3.")
else:
    arch_df["calibration_rolling"] = arch_df["in_band"].rolling(5, min_periods=1).mean()

    fig2 = go.Figure()
    fig2.add_hline(y=0.27, line_dash="dot", line_color="red",
                   annotation_text="Random baseline (~27%)")
    fig2.add_trace(go.Scatter(
        x=arch_df["episode_num"], y=arch_df["calibration_rolling"],
        mode="lines+markers", name="Calibration Accuracy (rolling 5)",
        line=dict(color="orange", width=2)
    ))
    fig2.add_hrect(y0=0.60, y1=1.0, fillcolor="green", opacity=0.08,
                   annotation_text="Target zone (>60%)")
    fig2.update_layout(
        xaxis_title="Episode", yaxis_title="% Tasks in 0.45–0.70 Band",
        yaxis=dict(range=[0, 1.05]), height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Panel 3: Architect reasoning ─────────────────────────────
st.subheader("🧠 Panel 3 — Architect Reasoning (Latest)")

arch_episodes = [e for e in episodes if e.get("architect_output")]
if arch_episodes:
    latest = arch_episodes[-1]
    arch_out = latest["architect_output"]
    ep_id = latest["episode_id"]

    with st.container():
        st.markdown(f"**Episode:** `{ep_id}`")
        cols = st.columns(3)
        cols[0].markdown(f"**Domain:** {arch_out.get('domain', 'N/A')}")
        cols[1].markdown(f"**Difficulty:** {arch_out.get('difficulty', 'N/A')}")
        cols[2].markdown(f"**Target Axis:** `{arch_out.get('target_axis', 'N/A')}`")

        st.info(f"💭 **Architect Reasoning:**\n\n{arch_out.get('architect_reasoning', 'N/A')}")
        st.markdown(f"**Lineage:** `{arch_out.get('lineage_id', 'N/A')}`")
        with st.expander("Generated Task"):
            st.markdown(f"**Scenario:** {arch_out.get('scenario_context', '')}")
            st.code(arch_out.get('contract_text', ''), language="text")
else:
    st.info("Architect hasn't generated tasks yet. Activates in Phase 3 after 3+ episodes.")

# ── Panel 4: Lineage map ──────────────────────────────────────
st.subheader("🔗 Panel 4 — Task Lineage (Failure → Generated Task)")

lineage_data = []
for ep in episodes:
    arch = ep.get("architect_output")
    if arch and arch.get("lineage_id"):
        lineage_data.append({
            "Source (Failure)": arch["lineage_id"][:20],
            "Generated Task": arch["task_id"][:20],
            "Target Axis": arch["target_axis"],
            "Difficulty": arch["difficulty"],
        })

if lineage_data:
    lineage_df = pd.DataFrame(lineage_data)
    st.dataframe(lineage_df, use_container_width=True, hide_index=True)

    # Breakthrough events
    bt_episodes = [e for e in episodes if e.get("failure_record", {}).get("breakthrough")]
    if bt_episodes:
        st.success(f"🎯 {len(bt_episodes)} Breakthrough Events — Executor solved tasks it previously failed!")
        for ep in bt_episodes[-3:]:
            fr = ep["failure_record"]
            st.markdown(f"- `{fr['task_id']}` | Score jumped to {fr['attempt_2_score']:.3f} | Axis: {fr['weakest_axis']}")
else:
    st.info("Lineage map populates once Architect generates tasks (Phase 3).")

# ── Raw episode table ─────────────────────────────────────────
with st.expander("📋 Full Episode Log"):
    display_cols = ["episode_num", "task_id", "difficulty", "score_1", "final_reward",
                    "delta", "weakest_axis", "is_breakthrough", "architect_active"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].sort_values("episode_num", ascending=False),
                 use_container_width=True, hide_index=True)
