"""
CRUCIBLE Demo Dashboard
Run with: streamlit run demo/dashboard.py

Panels:
  1 — Executor reward curve
  2 — Architect calibration accuracy
  3 — Adversarial Arms Race (Vendor vs Executor)
  4 — Regulation Shock Adaptation
  5 — Cross-Jurisdiction Generalization
  6 — Axis Failure Heatmap
  7 — Counterfactual Consequence Explorer
  8 — Architect reasoning (live)
  9 — Task Lineage map
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

LOG_DIR = "data/episode_logs"
FULL_LOG = os.path.join(LOG_DIR, "full_run.json")
PLOTS_DIR = "plots"

st.set_page_config(
    page_title="CRUCIBLE — Live Training Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("CRUCIBLE — Live Training Dashboard")

# ── LLM Backend info ─────────────────────────────────────────
try:
    from utils.llm_client import active_backend, backend_info, get_call_stats
    _binfo = backend_info()
    _stats = get_call_stats()
    _backend_label = f"**{_binfo['active']}**"
    _groq_calls  = _stats["groq"]["calls"]
    _groq_tokens = _stats["groq"]["total_tokens"]
except Exception:
    _binfo = {}
    _stats = {}
    _backend_label = "unknown"
    _groq_calls = 0
    _groq_tokens = 0

_is_groq = _binfo.get("backend") == "groq"
_backend_color = "#22c55e" if _is_groq else "#f59e0b"
_backend_icon  = "⚡" if _is_groq else "🤗"

st.markdown(
    f"""
    <div style="background:{_backend_color}22; border-left:4px solid {_backend_color};
                padding:10px 16px; border-radius:6px; margin-bottom:8px;">
      <span style="font-size:1.15rem; font-weight:700; color:{_backend_color};">
        {_backend_icon} Active LLM Backend: {_backend_label}
      </span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      API Key set: {'✅ Yes' if _binfo.get('groq_key_set') else '❌ No'}
      &nbsp;&nbsp;|&nbsp;&nbsp;
      Calls this session: <b>{_groq_calls}</b>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      Tokens used: <b>{_groq_tokens:,}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Self-Improving RL Environment | AXIOM Corporation | OpenEnv Hackathon 2026 "
    "| Adversarial Vendor · Regulation Shocks · Multi-Jurisdiction · Architect Curriculum"
)

# ── Refresh controls ─────────────────────────────────────────
refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh interval (sec)", 3, 30, 5)
if refresh:
    # Keep rerun cadence explicit to avoid continuous fast full-script loops.
    st.sidebar.caption(f"Auto-refresh enabled every {refresh_interval}s.")
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )
if st.sidebar.button("Refresh now"):
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
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Episodes", len(df))
col2.metric("Avg Reward", f"{df['final_reward'].mean():.3f}")
col3.metric("Latest Reward", f"{df['final_reward'].iloc[-1]:.3f}")
col4.metric("Breakthroughs", int(df['is_breakthrough'].sum()))

if "in_band" in df.columns and "architect_active" in df.columns:
    arch_rows = df[df["architect_active"] == True]
    in_band_rate = arch_rows["in_band"].mean() if not arch_rows.empty else 0.0
    col5.metric("Arch Calibration", f"{in_band_rate:.1%}")
else:
    col5.metric("Arch Calibration", "N/A")

if "adversarial" in df.columns:
    adv_count = int(df["adversarial"].fillna(False).sum())
    col6.metric("Adversarial Eps", adv_count)
else:
    col6.metric("Adversarial Eps", "0")

if "shock_fired" in df.columns:
    shock_count = int(df["shock_fired"].fillna(False).sum())
    adapt_rate = df[df["shock_fired"] == True]["shock_adapted"].mean() if shock_count > 0 else 0.0
    col7.metric("Shocks Fired", shock_count, delta=f"{adapt_rate:.0%} adapted")
else:
    col7.metric("Shocks Fired", "0")

st.divider()

# ── Panel 0: LLM Backend Breakdown ───────────────────────────
st.subheader("⚡ Panel 0 — LLM Backend Usage (Live)")

if "llm_backend" in df.columns:
    backend_counts = df["llm_backend"].value_counts().reset_index()
    backend_counts.columns = ["Backend", "Episodes"]

    llm_model_series = df["llm_model"].value_counts().reset_index() if "llm_model" in df.columns else None

    b_col1, b_col2, b_col3 = st.columns(3)

    groq_eps   = int(df[df["llm_backend"] == "groq"].shape[0])
    hf_eps     = int(df[df["llm_backend"] == "hf"].shape[0])
    other_eps  = int(df[~df["llm_backend"].isin(["groq", "hf"])].shape[0])

    b_col1.metric("Groq Episodes",        groq_eps,  delta="⚡ fastest" if groq_eps > 0 else None)
    b_col2.metric("HuggingFace Episodes", hf_eps)
    b_col3.metric("Other Episodes",       other_eps)

    if not backend_counts.empty:
        fig0 = go.Figure(go.Pie(
            labels=backend_counts["Backend"],
            values=backend_counts["Episodes"],
            hole=0.45,
            marker_colors=["#22c55e" if b == "groq" else "#f59e0b" if b == "hf" else "#6366f1"
                           for b in backend_counts["Backend"]],
            textinfo="label+percent+value",
        ))
        fig0.update_layout(
            title="Episodes by LLM Backend",
            height=280,
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=True,
        )
        st.plotly_chart(fig0, use_container_width=True)

    if llm_model_series is not None and not llm_model_series.empty:
        llm_model_series.columns = ["Model", "Episodes"]
        st.dataframe(llm_model_series, use_container_width=True, hide_index=True)

    # Live session tokens from call stats
    if _groq_calls > 0 or _groq_tokens > 0:
        st.markdown(
            f"**This session (in-memory):** Groq calls = `{_groq_calls}` | "
            f"Groq tokens = `{_groq_tokens:,}`  "
            f"*(visit [console.groq.com/usage](https://console.groq.com/usage) for full history)*"
        )
else:
    st.info(
        "Run a live episode to see backend breakdown. "
        "Re-run training with `python main.py baseline` then refresh."
    )

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

# ── Panel 3: Adversarial Arms Race ───────────────────────────
st.subheader("Panel 3 — Adversarial Arms Race (Vendor vs Executor)")

adv_df = df[df.get("adversarial", pd.Series([False]*len(df))).fillna(False) == True].copy() if "adversarial" in df.columns else pd.DataFrame()
if not adv_df.empty and "vendor_reward" in adv_df.columns:
    adv_df = adv_df.dropna(subset=["vendor_reward"])

if not adv_df.empty:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=adv_df["episode_num"], y=adv_df["final_reward"],
        mode="lines+markers", name="Executor reward",
        line=dict(color="#10b981", width=2), marker=dict(size=5)
    ))
    fig3.add_trace(go.Scatter(
        x=adv_df["episode_num"], y=adv_df["vendor_reward"],
        mode="lines+markers", name="Vendor reward (concealment)",
        line=dict(color="#ef4444", width=2), marker=dict(symbol="x", size=6)
    ))
    fig3.add_hrect(y0=0.0, y1=0.5, fillcolor="green", opacity=0.05,
                   annotation_text="Executor winning zone")
    fig3.add_hrect(y0=0.5, y1=1.0, fillcolor="red", opacity=0.05,
                   annotation_text="Vendor winning zone")
    fig3.update_layout(
        xaxis_title="Episode", yaxis_title="Reward (0.0 - 1.0)",
        yaxis=dict(range=[0, 1.05]), height=320, legend=dict(orientation="h")
    )
    st.plotly_chart(fig3, use_container_width=True)

    vendor_wins = (adv_df["vendor_reward"] > adv_df["final_reward"]).sum()
    exec_wins = len(adv_df) - vendor_wins
    c1, c2, c3 = st.columns(3)
    c1.metric("Adversarial Episodes", len(adv_df))
    c2.metric("Executor Wins", exec_wins, delta=f"{exec_wins/len(adv_df):.0%}")
    c3.metric("Vendor Wins", vendor_wins, delta=f"-{vendor_wins/len(adv_df):.0%}")
else:
    st.info("Run adversarial episodes with `python main.py adversarial` to see the arms race.")
    img_path = os.path.join(PLOTS_DIR, "adversarial_arms_race.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Adversarial Arms Race (demo data)", use_column_width=True)

st.divider()

# ── Panel 4: Regulation Shock Adaptation ─────────────────────
st.subheader("Panel 4 — Regulation Shock Adaptation")

if "shock_fired" in df.columns:
    shock_df = df[df["shock_fired"] == True].copy()
    if not shock_df.empty and "shock_adapted" in shock_df.columns:
        shock_df = shock_df.dropna(subset=["shock_adapted"])
        shock_df["adapted_int"] = shock_df["shock_adapted"].astype(int)
        shock_df["rolling_adapt"] = shock_df["adapted_int"].rolling(5, min_periods=1).mean()

        fig4 = go.Figure()
        fig4.add_bar(
            x=shock_df["episode_num"],
            y=shock_df["adapted_int"],
            name="Adapted (1) / Missed (0)",
            marker_color=["#10b981" if a else "#ef4444" for a in shock_df["shock_adapted"]],
            opacity=0.5,
        )
        fig4.add_trace(go.Scatter(
            x=shock_df["episode_num"], y=shock_df["rolling_adapt"],
            mode="lines", name="Rolling adaptation rate (5 ep)",
            line=dict(color="#3b82f6", width=2.5)
        ))
        fig4.add_hline(y=0.5, line_dash="dot", line_color="gray",
                       annotation_text="50% baseline")
        fig4.update_layout(
            xaxis_title="Episode (shock episodes only)",
            yaxis_title="Adaptation Rate", height=300,
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig4, use_container_width=True)

        adapt_rate = shock_df["shock_adapted"].mean()
        st.metric("Overall Shock Adaptation Rate", f"{adapt_rate:.1%}",
                  delta="vs 0% untrained baseline")
    else:
        st.info("No shock adaptation data yet.")
else:
    img_path = os.path.join(PLOTS_DIR, "shock_adaptation.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Regulation Shock Adaptation (demo data)", use_column_width=True)
    else:
        st.info("Run shock episodes with `python main.py shock` to see adaptation data.")

st.divider()

# ── Panel 5: Cross-Jurisdiction Generalization ────────────────
st.subheader("Panel 5 — Cross-Jurisdiction Generalization (FAR / DFARS / EU)")

if "jurisdiction" in df.columns:
    jur_df = df.groupby("jurisdiction")["final_reward"].agg(["mean", "count", "std"]).reset_index()
    jur_df.columns = ["Jurisdiction", "Avg Reward", "Episodes", "Std Dev"]
    jur_df["Avg Reward"] = jur_df["Avg Reward"].round(4)
    jur_df["Std Dev"] = jur_df["Std Dev"].round(4)

    color_map = {"FAR": "#4ade80", "DFARS": "#60a5fa", "EU": "#f97316"}
    fig5 = go.Figure()
    for _, row in jur_df.iterrows():
        fig5.add_bar(
            x=[row["Jurisdiction"]],
            y=[row["Avg Reward"]],
            name=row["Jurisdiction"],
            marker_color=color_map.get(row["Jurisdiction"], "#a78bfa"),
            error_y=dict(type="data", array=[row["Std Dev"]], visible=True),
            text=[f"n={row['Episodes']}\n{row['Avg Reward']:.3f}"],
            textposition="outside",
        )
    fig5.add_hline(y=0.45, line_dash="dash", line_color="blue",
                   annotation_text="Learning band floor")
    fig5.add_hline(y=0.70, line_dash="dash", line_color="green",
                   annotation_text="Learning band ceiling")
    fig5.update_layout(
        yaxis=dict(range=[0, 1.05]),
        yaxis_title="Average Final Reward",
        xaxis_title="Jurisdiction",
        height=350, showlegend=False
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(jur_df, use_container_width=True, hide_index=True)
else:
    img_path = os.path.join(PLOTS_DIR, "jurisdiction_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Jurisdiction Comparison (demo data)", use_column_width=True)

st.divider()

# ── Panel 6: Axis Failure Heatmap ────────────────────────────
st.subheader("Panel 6 — Where Does AXIOM Keep Failing? (Axis × Difficulty Heatmap)")

AXES = ["correctness", "completeness", "reasoning_transparency", "efficiency", "generalization_signal"]
DIFFICULTIES_ORDER = ["easy", "medium", "hard", "expert"]

if "weakest_axis" in df.columns and "difficulty" in df.columns:
    heatmap_data = {}
    for diff in DIFFICULTIES_ORDER:
        sub = df[df["difficulty"] == diff]
        total = len(sub) or 1
        row = {}
        for ax in AXES:
            row[ax] = round((sub["weakest_axis"] == ax).sum() / total, 3)
        heatmap_data[diff] = row

    heat_df = pd.DataFrame(heatmap_data).T
    short_axes = ["correct.", "complete.", "reasoning", "effic.", "general."]
    heat_df.columns = short_axes
    heat_df.index.name = "Difficulty"

    fig6 = px.imshow(
        heat_df,
        color_continuous_scale="YlOrRd",
        zmin=0, zmax=0.5,
        labels=dict(color="Fail Rate"),
        title="Failure Rate per Axis × Difficulty",
        text_auto=".0%",
        aspect="auto",
    )
    fig6.update_layout(height=300)
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Red = where the Executor consistently fails. Architect targets these cells.")
else:
    img_path = os.path.join(PLOTS_DIR, "axis_heatmap.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Axis Failure Heatmap (demo data)", use_column_width=True)

st.divider()

# ── Panel 7: Counterfactual Consequence Explorer ──────────────
st.subheader("Panel 7 — Counterfactual Consequences (What Happens if Wrong Decision Stands?)")

if "consequence_if_approved" in df.columns and episodes:
    consequence_data = []
    for ep in episodes[-20:]:
        s2 = ep.get("score_2", {})
        consequence = s2.get("consequence_if_approved", "") or df[
            df["task_id"] == ep.get("task", {}).get("task_id", "")
        ]["consequence_if_approved"].values
        if hasattr(consequence, "__iter__") and not isinstance(consequence, str):
            consequence = consequence[0] if len(consequence) > 0 else ""
        if consequence:
            consequence_data.append({
                "Episode": ep.get("episode_id", "")[:20],
                "Score": ep.get("score_2", {}).get("weighted_total", 0.0),
                "Decision": ep.get("score_2", {}).get("feedback", "")[:60],
                "Consequence if Wrong": str(consequence)[:120],
            })

    if consequence_data:
        cdf = pd.DataFrame(consequence_data)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        # Fall back to full_run.json data
        conq_df = df[df["consequence_if_approved"].notna()][
            ["episode_num", "task_id", "final_reward", "consequence_if_approved"]
        ].tail(15)
        if not conq_df.empty:
            st.dataframe(conq_df.rename(columns={
                "episode_num": "Episode",
                "task_id": "Task",
                "final_reward": "Reward",
                "consequence_if_approved": "Consequence if Wrong",
            }), use_container_width=True, hide_index=True)
else:
    st.info("Counterfactual data will populate after running real episodes (reward analysis requires Arbiter output).")

st.divider()

# ── Panel 8: Architect reasoning ─────────────────────────────
st.subheader("Panel 8 — Architect Reasoning (Latest Generated Task)")

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

        st.info(f"Architect Reasoning:\n\n{arch_out.get('architect_reasoning', 'N/A')}")
        st.markdown(f"**Lineage:** `{arch_out.get('lineage_id', 'N/A')}`")
        with st.expander("Generated Task Contract"):
            st.markdown(f"**Scenario:** {arch_out.get('scenario_context', '')}")
            st.code(arch_out.get('contract_text', ''), language="text")
else:
    st.info("Architect hasn't generated tasks yet. Activates in Phase 3 after 3+ episodes.")

st.divider()

# ── Panel 9: Lineage map ──────────────────────────────────────
st.subheader("Panel 9 — Task Lineage (Failure → Architect-Generated Task)")

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

    bt_episodes = [e for e in episodes if e.get("failure_record", {}).get("breakthrough")]
    if bt_episodes:
        st.success(f"{len(bt_episodes)} Breakthrough Events — Executor solved tasks it previously failed!")
        for ep in bt_episodes[-3:]:
            fr = ep["failure_record"]
            st.markdown(f"- `{fr['task_id']}` | Score jumped to {fr['attempt_2_score']:.3f} | Axis: {fr['weakest_axis']}")
else:
    st.info("Lineage map populates once Architect generates tasks (Phase 3).")

st.divider()

# ── Panel 10: Curriculum Diversity ───────────────────────────
st.subheader("Panel 10 — Curriculum Diversity Score (Architect Coverage)")

if "diversity_score" in df.columns:
    div_df = df.dropna(subset=["diversity_score"])
    if not div_df.empty:
        div_df["rolling_diversity"] = div_df["diversity_score"].rolling(5, min_periods=1).mean()

        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=div_df["episode_num"], y=div_df["diversity_score"],
            mode="markers", name="Per-episode diversity",
            marker=dict(color="#a78bfa", size=4, opacity=0.5)
        ))
        fig10.add_trace(go.Scatter(
            x=div_df["episode_num"], y=div_df["rolling_diversity"],
            mode="lines", name="Rolling avg (5 ep)",
            line=dict(color="#7c3aed", width=2.5)
        ))
        fig10.update_layout(
            xaxis_title="Episode",
            yaxis_title="Diversity Score (0=repetitive, 1=novel)",
            yaxis=dict(range=[0, 1.05]),
            height=280, legend=dict(orientation="h")
        )
        st.plotly_chart(fig10, use_container_width=True)
        avg_div = div_df["diversity_score"].mean()
        st.metric("Avg Curriculum Diversity", f"{avg_div:.3f}",
                  delta="Higher = more varied tasks, harder to game")
else:
    st.info("Diversity scores populate after running real episodes with `python main.py baseline`.")

st.divider()

# ── Raw episode table ─────────────────────────────────────────
with st.expander("Full Episode Log"):
    display_cols = [
        "episode_num", "task_id", "domain", "jurisdiction", "difficulty",
        "score_1", "final_reward", "delta", "weakest_axis",
        "is_breakthrough", "architect_active", "adversarial", "shock_fired",
        "llm_backend", "llm_model",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].sort_values("episode_num", ascending=False),
                 use_container_width=True, hide_index=True)
