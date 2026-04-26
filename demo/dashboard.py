"""
CRUCIBLE — Immersive Mission Control Dashboard
Run: streamlit run demo/dashboard.py
"""

import json
import os
import sys
import time
import subprocess
import threading
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

LOG_DIR   = "data/episode_logs"
FULL_LOG  = os.path.join(LOG_DIR, "full_run.json")
PLOTS_DIR = "plots"

# Resolve project root so subprocess calls work from any cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = {
    "⚡ Single Episode":   ("episode",    "Run one compliance episode (quick test ~30 s)"),
    "📊 Baseline Run":     ("baseline",   "Phase 1 — 10 episodes, no curriculum (1–2 min)"),
    "🦹 Adversarial":      ("adversarial","Vendor crafts hidden violations, Executor must catch"),
    "⚡ Shock Episode":    ("shock",      "Mid-episode regulation change injected"),
    "🇪🇺 EU Jurisdiction": ("eu",         "Test cross-jurisdiction with EU Directive 2014/24/EU"),
    "🔥 Full Pipeline":    ("full",       "Baseline + train + architect + adv + shock (5–8 min)"),
    "📈 Regenerate Plots": ("plots",      "Rebuild all plots/ from current episode logs"),
}


_active_proc: list = []  # holds the single Popen object so Stop can kill it


def _run_command_streaming(cmd_arg: str, output_lines: list, done_flag: list):
    """Run `python main.py <cmd_arg>` in the project root, append stdout lines."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _ROOT
    proc = subprocess.Popen(
        [sys.executable, "main.py", cmd_arg],
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    _active_proc.clear()
    _active_proc.append(proc)
    for line in proc.stdout:
        output_lines.append(line.rstrip())
        if len(output_lines) > 300:
            output_lines.pop(0)
    proc.wait()
    _active_proc.clear()
    done_flag.append(proc.returncode)


st.set_page_config(
    page_title="CRUCIBLE — Mission Control",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark base */
  .stApp { background: #0d1117; color: #c9d1d9; }
  section[data-testid="stSidebar"] { background: #161b22 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #8b949e !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #58a6ff !important; font-size: 1.6rem !important;
  }

  /* Agent cards */
  .agent-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 18px;
    height: 130px;
  }
  .agent-card.active  { border-color: #3fb950; box-shadow: 0 0 12px #3fb95044; }
  .agent-card.hunting { border-color: #f85149; box-shadow: 0 0 12px #f8514944; }
  .agent-card.frozen  { border-color: #58a6ff; box-shadow: 0 0 10px #58a6ff33; }
  .agent-card.standby { border-color: #e3b341; box-shadow: 0 0 10px #e3b34133; }
  .agent-name  { font-size: 1.0rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px; }
  .agent-stat  { font-size: 0.82rem; color: #8b949e; margin: 2px 0; }
  .agent-badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-top: 6px;
  }
  .badge-active  { background: #1a4731; color: #3fb950; }
  .badge-frozen  { background: #0d2a4a; color: #58a6ff; }
  .badge-hunting { background: #3d1a1a; color: #f85149; }
  .badge-standby { background: #3d2e00; color: #e3b341; }

  /* Hero title */
  .hero-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: 0.04em;
    background: linear-gradient(90deg, #58a6ff, #3fb950, #e3b341);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-sub { color: #8b949e; font-size: 0.95rem; margin-top: -8px; }

  /* Badge pills */
  .tech-badge {
    display: inline-block; background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; padding: 3px 12px; font-size: 0.78rem;
    color: #c9d1d9; margin: 2px;
  }

  /* Panel headers */
  .panel-header {
    font-size: 1.05rem; font-weight: 700; color: #c9d1d9;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-bottom: 12px;
  }

  /* Episode feed cards */
  .ep-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0; font-size: 0.83rem;
  }
  .ep-card.good  { border-left: 3px solid #3fb950; }
  .ep-card.mid   { border-left: 3px solid #e3b341; }
  .ep-card.bad   { border-left: 3px solid #f85149; }

  /* Ticker */
  .ticker-wrap {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 8px 16px; font-family: monospace; font-size: 0.82rem;
    color: #3fb950; overflow: hidden; white-space: nowrap;
  }

  /* Section divider */
  hr { border-color: #21262d !important; }

  /* Tab overrides */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; }
  .stTabs [aria-selected="true"] { color: #58a6ff !important; }

  /* Plotly dark */
  .js-plotly-plot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────
for _k, _v in {
    "running": False,
    "output_lines": [],
    "done_flag": [],
    "active_cmd": "",
    "last_exit": None,
    "proc_pid": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_sec  = st.slider("Interval (sec)", 5, 60, 10)
    if auto_refresh:
        st.markdown(
            f"<script>setTimeout(()=>window.location.reload(),{refresh_sec*1000})</script>",
            unsafe_allow_html=True,
        )
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### ▶ Run Commands")

    # Check if a background job just finished
    if st.session_state.running and st.session_state.done_flag:
        st.session_state.last_exit = st.session_state.done_flag[0]
        st.session_state.running = False
        st.cache_data.clear()

    if st.session_state.running:
        st.markdown(
            f'<div style="color:#e3b341;font-size:0.85rem;padding:6px 0">'
            f'⏳ Running: <b>{st.session_state.active_cmd}</b></div>',
            unsafe_allow_html=True,
        )
        if st.button("🛑 Stop", use_container_width=True):
            if _active_proc:
                try:
                    _active_proc[0].terminate()
                    _active_proc[0].wait(timeout=3)
                except Exception:
                    try:
                        _active_proc[0].kill()
                    except Exception:
                        pass
                _active_proc.clear()
            st.session_state.running = False
            st.session_state.output_lines.append("— stopped by user —")
    else:
        if st.session_state.last_exit is not None:
            color = "#3fb950" if st.session_state.last_exit == 0 else "#f85149"
            label = "✅ Finished" if st.session_state.last_exit == 0 else f"❌ Exit {st.session_state.last_exit}"
            st.markdown(
                f'<div style="color:{color};font-size:0.82rem;padding-bottom:6px">{label}</div>',
                unsafe_allow_html=True,
            )

        for btn_label, (cmd_arg, tooltip) in COMMANDS.items():
            if st.button(btn_label, use_container_width=True, help=tooltip,
                         disabled=st.session_state.running):
                st.session_state.output_lines = [f"$ python main.py {cmd_arg}"]
                st.session_state.done_flag = []
                st.session_state.running = True
                st.session_state.active_cmd = cmd_arg
                st.session_state.last_exit = None
                t = threading.Thread(
                    target=_run_command_streaming,
                    args=(cmd_arg,
                          st.session_state.output_lines,
                          st.session_state.done_flag),
                    daemon=True,
                )
                t.start()
                st.rerun()

        if st.session_state.output_lines:
            if st.button("🗑 Clear Log", use_container_width=True):
                st.session_state.output_lines = []
                st.session_state.last_exit = None

    # Live terminal output in sidebar
    if st.session_state.output_lines:
        st.divider()
        st.markdown("### 🖥 Live Output")
        output_text = "\n".join(st.session_state.output_lines[-60:])
        st.code(output_text, language="bash")
        if st.session_state.running:
            time.sleep(1)
            st.rerun()

    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown("[Groq Usage Console](https://console.groq.com/usage)")
    st.markdown("[HuggingFace Space](https://huggingface.co/spaces/Flake56/crucible-env)")
    st.markdown("[Live API Docs](https://Flake56-crucible-env.hf.space/docs)")

# ── Load data ─────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_log():
    """Load full_run.json; if absent/empty, reconstruct from per-episode files."""
    if os.path.exists(FULL_LOG):
        try:
            with open(FULL_LOG, encoding="utf-8") as f:
                data = json.load(f)
            if data:
                return data
        except Exception:
            pass
    # Fallback: stitch together individual ep-*.json files
    records = []
    if os.path.exists(LOG_DIR):
        for fn in sorted(os.listdir(LOG_DIR)):
            if fn.startswith("ep_") and fn.endswith(".json") and fn != "full_run.json":
                try:
                    with open(os.path.join(LOG_DIR, fn), encoding="utf-8") as f:
                        records.append(json.load(f))
                except Exception:
                    pass
    return records


@st.cache_data(ttl=5)
def load_episodes():
    eps = []
    if not os.path.exists(LOG_DIR):
        return eps
    for fn in sorted(os.listdir(LOG_DIR)):
        if fn.startswith("ep-") and fn.endswith(".json"):
            try:
                with open(os.path.join(LOG_DIR, fn), encoding="utf-8") as f:
                    eps.append(json.load(f))
            except Exception:
                pass
    return eps


def _safe_str(val, default: str = "?", maxlen: int = 0) -> str:
    """Return val as a string, never None; optionally truncate."""
    s = str(val) if val is not None else default
    return s[:maxlen] if maxlen else s


history  = load_log()
episodes = load_episodes()

# ── LLM backend info ──────────────────────────────────────────
try:
    from utils.llm_client import backend_info, get_call_stats
    _binfo  = backend_info()
    _stats  = get_call_stats()
    _groq_calls   = _stats["groq"]["calls"]
    _groq_tokens  = _stats["groq"]["total_tokens"]
    _groq_rl      = _stats["groq"].get("rate_limit_waits", 0)
    _groq_wait_s  = _stats["groq"].get("wait_seconds", 0.0)
    _window_used  = _binfo.get("groq_window_used", 0)
    _tpm_eff      = _binfo.get("groq_tpm_effective", 5100)
    _is_groq      = _binfo.get("backend") == "groq"
    _active_label = _binfo.get("active", "Unknown")
except Exception:
    _binfo = {"groq_key_set": False, "backend": "unknown", "active": "Unknown"}
    _is_groq = False; _active_label = "Unknown"
    _groq_calls = _groq_tokens = _groq_rl = 0
    _groq_wait_s = _window_used = 0.0; _tpm_eff = 5100

_bc   = "#3fb950" if _is_groq else "#e3b341"
_icon = "⚡" if _is_groq else "🤗"
_rl_c = "#f85149" if _groq_rl > 3 else "#8b949e"
_tpm_pct = min(100, int(_window_used / _tpm_eff * 100))
_tpm_bar_color = "#f85149" if _tpm_pct > 80 else "#e3b341" if _tpm_pct > 50 else "#3fb950"

# ── HERO ──────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔥 CRUCIBLE</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Self-Improving Multi-Agent RL for Government Procurement Compliance · OpenEnv Hackathon 2026</p>',
    unsafe_allow_html=True,
)

badges = ["⚡ Groq API", "🤗 HuggingFace", "🔥 PyTorch", "🌐 OpenEnv",
          "🇺🇸 FAR/DFARS", "🇪🇺 EU Directive", "🦹 Adversarial RL", "📋 Regulation Shocks"]
st.markdown("".join(f'<span class="tech-badge">{b}</span>' for b in badges), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Groq live banner ──────────────────────────────────────────
_tpm_filled = "█" * (_tpm_pct // 10) + "░" * (10 - _tpm_pct // 10)
st.markdown(f"""
<div style="background:#161b22; border:1px solid {_bc}; border-left:4px solid {_bc};
            border-radius:8px; padding:10px 18px; margin-bottom:8px; font-size:0.9rem;">
  <span style="font-weight:700; color:{_bc}; font-size:1.05rem;">
    {_icon} {_active_label}
  </span>
  &nbsp;·&nbsp; Key: {'✅' if _binfo.get('groq_key_set') else '❌'}
  &nbsp;·&nbsp; Session calls: <b style="color:#c9d1d9">{_groq_calls}</b>
  &nbsp;·&nbsp; Tokens used: <b style="color:#c9d1d9">{_groq_tokens:,}</b>
  &nbsp;·&nbsp; TPM: <span style="font-family:monospace; color:{_tpm_bar_color}">{_tpm_filled}</span>
    <span style="color:{_tpm_bar_color}"> {_window_used}/{_tpm_eff}</span>
  &nbsp;·&nbsp; <span style="color:{_rl_c}">Rate-limit waits: {_groq_rl} ({_groq_wait_s:.0f}s)</span>
</div>
""", unsafe_allow_html=True)

# ── Ticker ────────────────────────────────────────────────────
if history:
    last = history[-1]
    ticker_items = []
    for r in history[-8:]:
        ep   = r.get("episode", 0)
        rwd  = float(r.get("final_reward", 0) or 0)
        jur  = _safe_str(r.get("jurisdiction", "FAR"))
        axis = _safe_str(r.get("weakest_axis"), "?", maxlen=12)
        ep_fmt = f"{int(ep):04d}" if str(ep).isdigit() or isinstance(ep, int) else str(ep)
        ticker_items.append(f"► ep-{ep_fmt} reward={rwd:.3f} [{jur}] weak={axis}")
    if _groq_tokens > 0:
        ticker_items.append(f"► Groq {_groq_tokens:,} tokens consumed this session")
    st.markdown(
        f'<div class="ticker-wrap">{" &nbsp;&nbsp;&nbsp; ".join(ticker_items)}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Agent Status Cards ────────────────────────────────────────
if not history:
    df = pd.DataFrame()
    _ep_count = 0; _avg_rwd = 0; _last_rwd = 0; _bt = 0
else:
    df = pd.DataFrame(history)
    df["episode_num"] = range(1, len(df) + 1)
    _ep_count = len(df)
    _avg_rwd  = df["final_reward"].mean()
    _last_rwd = df["final_reward"].iloc[-1]
    _bt       = int(df["is_breakthrough"].sum()) if "is_breakthrough" in df.columns else 0

_adv_active  = not df.empty and df.get("adversarial", pd.Series(dtype=bool)).fillna(False).any()
_arch_active = not df.empty and df.get("architect_active", pd.Series(dtype=bool)).fillna(False).any()
_shock_count = int(df["shock_fired"].fillna(False).sum()) if "shock_fired" in df.columns and not df.empty else 0
_exec_calls  = _ep_count * 4  # 2 attempts × executor+arbiter

st.markdown('<p class="panel-header">🤖 Agent Status</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="agent-card active">
      <div class="agent-name" style="color:#3fb950">⚡ EXECUTOR</div>
      <div class="agent-stat">Episodes completed: {_ep_count}</div>
      <div class="agent-stat">LLM calls: ~{_exec_calls}</div>
      <div class="agent-stat">Last reward: {_last_rwd:.3f}</div>
      <span class="agent-badge badge-active">ACTIVE</span>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="agent-card frozen">
      <div class="agent-name" style="color:#58a6ff">🏛️ ARBITER</div>
      <div class="agent-stat">Scores issued: {_ep_count * 2}</div>
      <div class="agent-stat">Avg score: {_avg_rwd:.3f}</div>
      <div class="agent-stat">Shocks judged: {_shock_count}</div>
      <span class="agent-badge badge-frozen">FROZEN</span>
    </div>""", unsafe_allow_html=True)

with c3:
    _arch_tasks = int(df[df.get("architect_active", pd.Series(dtype=bool)).fillna(False)].shape[0]) if not df.empty and "architect_active" in df.columns else 0
    _arch_state = "ACTIVE" if _arch_active else "STANDBY"
    _arch_cls   = "active" if _arch_active else "standby"
    _arch_badge = "badge-active" if _arch_active else "badge-standby"
    st.markdown(f"""
    <div class="agent-card {_arch_cls}">
      <div class="agent-name" style="color:#e3b341">🏗️ ARCHITECT</div>
      <div class="agent-stat">Tasks generated: {_arch_tasks}</div>
      <div class="agent-stat">Breakthroughs: {_bt}</div>
      <div class="agent-stat">Phase 3 activates at ep 51</div>
      <span class="agent-badge {_arch_badge}">{_arch_state}</span>
    </div>""", unsafe_allow_html=True)

with c4:
    _adv_eps = int(df["adversarial"].fillna(False).sum()) if "adversarial" in df.columns and not df.empty else 0
    _vnd_state = "HUNTING" if _adv_active else "IDLE"
    _vnd_cls   = "hunting" if _adv_active else "standby"
    _vnd_badge = "badge-hunting" if _adv_active else "badge-standby"
    st.markdown(f"""
    <div class="agent-card {_vnd_cls}">
      <div class="agent-name" style="color:#f85149">🦹 VENDOR</div>
      <div class="agent-stat">Adversarial eps: {_adv_eps}</div>
      <div class="agent-stat">Zero-sum dynamics</div>
      <div class="agent-stat">Concealment: active</div>
      <span class="agent-badge {_vnd_badge}">{_vnd_state}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Top Metrics Row ───────────────────────────────────────────
if not df.empty:
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    _trend = round(_avg_rwd - df["final_reward"].iloc[max(0,len(df)-11):max(1,len(df)-1)].mean(), 3) if len(df) > 5 else None
    m1.metric("Episodes", _ep_count)
    m2.metric("Avg Reward",    f"{_avg_rwd:.3f}", delta=f"{_trend:+.3f}" if _trend else None)
    m3.metric("Latest Reward", f"{_last_rwd:.3f}")
    m4.metric("Breakthroughs", _bt)
    _arch_rows = df[df.get("architect_active", pd.Series(dtype=bool)).fillna(False) == True] if "architect_active" in df.columns else pd.DataFrame()
    _cal = f"{_arch_rows['in_band'].mean():.0%}" if not _arch_rows.empty and "in_band" in _arch_rows.columns else "—"
    m5.metric("Arch Calibration", _cal)
    m6.metric("Adversarial Eps", _adv_eps)
    m7.metric("Shocks Fired", _shock_count,
              delta=f"{df[df['shock_fired']==True]['shock_adapted'].mean():.0%} adapted" if _shock_count > 0 and "shock_adapted" in df.columns else None)

    st.divider()

# ── Stop here if no data ──────────────────────────────────────
if df.empty:
    st.warning("No training data yet. Run `python main.py baseline` to start, then refresh.")
    st.stop()

# ── MAIN CHARTS ───────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown('<p class="panel-header">📈 Executor Reward Curve</p>', unsafe_allow_html=True)

    rolling = df["final_reward"].rolling(5, min_periods=1).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["episode_num"], y=df["score_1"],
        mode="lines", name="Attempt 1",
        line=dict(color="#8b949e", dash="dot", width=1), opacity=0.6,
    ))
    fig1.add_trace(go.Scatter(
        x=df["episode_num"], y=df["final_reward"],
        mode="lines", name="Final Reward",
        line=dict(color="#3fb950", width=1.5), opacity=0.7,
        fill="tozeroy", fillcolor="rgba(63,185,80,0.06)",
    ))
    fig1.add_trace(go.Scatter(
        x=df["episode_num"], y=rolling,
        mode="lines", name="Rolling avg (5)",
        line=dict(color="#58a6ff", width=2.5),
    ))

    arch_start = df[df["architect_active"] == True]["episode_num"].min() if "architect_active" in df.columns and df["architect_active"].any() else None
    if arch_start and not pd.isna(arch_start):
        fig1.add_vline(x=arch_start, line_dash="dash", line_color="#e3b341",
                       annotation_text="Architect ▶", annotation_font_color="#e3b341",
                       annotation_position="top right")

    if "is_breakthrough" in df.columns:
        bt_df = df[df["is_breakthrough"] == True]
        if not bt_df.empty:
            fig1.add_trace(go.Scatter(
                x=bt_df["episode_num"], y=bt_df["final_reward"],
                mode="markers", name="Breakthrough ★",
                marker=dict(color="#e3b341", size=14, symbol="star",
                            line=dict(color="#0d1117", width=1)),
            ))

    fig1.add_hrect(y0=0.45, y1=0.70, fillcolor="#3fb950", opacity=0.06,
                   annotation_text="Learning Band", annotation_font_color="#3fb950",
                   annotation_font_size=11)
    fig1.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        xaxis=dict(color="#8b949e", gridcolor="#21262d", title="Episode"),
        yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.05], title="Reward"),
        legend=dict(orientation="h", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    st.markdown('<p class="panel-header">🎯 Score Gauge</p>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(_avg_rwd, 3),
        delta={"reference": 0.45, "increasing": {"color": "#3fb950"}, "decreasing": {"color": "#f85149"}},
        number={"font": {"color": "#58a6ff", "size": 42}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": "#3fb950" if _avg_rwd >= 0.70 else "#e3b341" if _avg_rwd >= 0.45 else "#f85149"},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 0.45],  "color": "#3d1a1a"},
                {"range": [0.45, 0.70], "color": "#1e2a0e"},
                {"range": [0.70, 1.0],  "color": "#0d2a1a"},
            ],
            "threshold": {"line": {"color": "#58a6ff", "width": 3}, "value": 0.70},
        },
        title={"text": "Avg Final Reward", "font": {"color": "#8b949e", "size": 13}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#161b22", font=dict(color="#c9d1d9"),
        margin=dict(t=20, b=10, l=20, r=20), height=240,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Before/After comparison
    if len(df) >= 10:
        st.markdown('<p class="panel-header" style="margin-top:8px">📊 Before vs After</p>', unsafe_allow_html=True)
        first5 = df["final_reward"].head(5).mean()
        last5  = df["final_reward"].tail(5).mean()
        delta  = round(last5 - first5, 3)
        b1, b2 = st.columns(2)
        b1.metric("First 5 eps", f"{first5:.3f}", delta=None)
        b2.metric("Last 5 eps",  f"{last5:.3f}",  delta=f"{delta:+.3f}")

st.divider()

# ── Groq Backend + Axis Heatmap row ───────────────────────────
g1, g2 = st.columns([1, 2])

with g1:
    st.markdown('<p class="panel-header">⚡ LLM Backend Breakdown</p>', unsafe_allow_html=True)

    if "llm_backend" in df.columns:
        bc = df["llm_backend"].value_counts().reset_index()
        bc.columns = ["Backend", "Episodes"]
        colors_map = {"groq": "#3fb950", "hf": "#e3b341", "anthropic": "#58a6ff"}
        fig0 = go.Figure(go.Pie(
            labels=bc["Backend"], values=bc["Episodes"],
            hole=0.52,
            marker_colors=[colors_map.get(b, "#8b949e") for b in bc["Backend"]],
            textinfo="label+percent",
            textfont=dict(size=11, color="#c9d1d9"),
        ))
        fig0.update_layout(
            paper_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
            margin=dict(t=10, b=10, l=10, r=10), height=200,
            showlegend=False,
        )
        st.plotly_chart(fig0, use_container_width=True)
        groq_n = int(df[df["llm_backend"] == "groq"].shape[0]) if "llm_backend" in df.columns else 0
        st.caption(f"Groq: **{groq_n}** eps · **{_groq_tokens:,}** tokens · {_groq_calls} calls")
    else:
        _img = os.path.join(PLOTS_DIR, "training_reward_curve.png")
        if os.path.exists(_img):
            st.image(_img, use_container_width=True)

with g2:
    st.markdown('<p class="panel-header">🔥 Axis Failure Heatmap — Where AXIOM Keeps Failing</p>', unsafe_allow_html=True)

    AXES = ["correctness", "completeness", "reasoning_transparency", "efficiency", "generalization_signal"]
    DIFF_ORDER = ["easy", "medium", "hard", "expert"]

    if "weakest_axis" in df.columns and "difficulty" in df.columns:
        hdata = {}
        for diff in DIFF_ORDER:
            sub = df[df["difficulty"] == diff]
            tot = len(sub) or 1
            hdata[diff] = {ax: round((sub["weakest_axis"] == ax).sum() / tot, 3) for ax in AXES}
        heat_df = pd.DataFrame(hdata).T
        heat_df.columns = ["correct.", "complete.", "reasoning", "effic.", "general."]
        heat_df.index.name = "Difficulty"
        fig6 = px.imshow(heat_df, color_continuous_scale="YlOrRd", zmin=0, zmax=0.5,
                         text_auto=".0%", aspect="auto",
                         labels=dict(color="Fail Rate"))
        fig6.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", size=11),
            coloraxis_colorbar=dict(
                tickfont=dict(color="#c9d1d9"),
                title=dict(text="Fail Rate", font=dict(color="#c9d1d9")),
            ),
            margin=dict(t=10, b=10, l=10, r=10), height=220,
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.caption("Red = consistent failure zone. Architect generates tasks targeting these cells.")
    else:
        _img = os.path.join(PLOTS_DIR, "axis_heatmap.png")
        if os.path.exists(_img):
            st.image(_img, caption="Axis Heatmap (demo)", use_container_width=True)

st.divider()

# ── Tabbed deep-dive panels ───────────────────────────────────
st.markdown('<p class="panel-header" style="font-size:1.15rem">🔍 Deep Dive</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚔️ Arms Race", "⚡ Shocks", "🌍 Jurisdiction", "🏗️ Architect", "📋 Diversity"
])

# ── Tab 1: Adversarial Arms Race ─────────────────────────────
with tab1:
    adv_df = df[df.get("adversarial", pd.Series(dtype=bool)).fillna(False) == True].copy() if "adversarial" in df.columns else pd.DataFrame()
    if not adv_df.empty and "vendor_reward" in adv_df.columns:
        adv_df = adv_df.dropna(subset=["vendor_reward"])

    if not adv_df.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=adv_df["episode_num"], y=adv_df["final_reward"],
            mode="lines+markers", name="Executor",
            line=dict(color="#3fb950", width=2), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
        ))
        fig3.add_trace(go.Scatter(
            x=adv_df["episode_num"], y=adv_df["vendor_reward"],
            mode="lines+markers", name="Vendor (concealment)",
            line=dict(color="#f85149", width=2), marker=dict(symbol="x", size=6),
        ))
        fig3.add_hrect(y0=0.5, y1=1.0, fillcolor="#f85149", opacity=0.04, annotation_text="Vendor winning zone", annotation_font_color="#f85149", annotation_font_size=10)
        fig3.add_hrect(y0=0.0, y1=0.5, fillcolor="#3fb950", opacity=0.04, annotation_text="Executor winning zone", annotation_font_color="#3fb950", annotation_font_size=10)
        fig3.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(color="#8b949e", gridcolor="#21262d", title="Episode"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.05]),
            legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10), height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)
        vw = int((adv_df["vendor_reward"] > adv_df["final_reward"]).sum())
        ew = len(adv_df) - vw
        a1, a2, a3 = st.columns(3)
        a1.metric("Adversarial Episodes", len(adv_df))
        a2.metric("Executor Wins ✅", ew, delta=f"{ew/len(adv_df):.0%}")
        a3.metric("Vendor Wins 🦹", vw, delta=f"-{vw/len(adv_df):.0%}")
    else:
        _img = os.path.join(PLOTS_DIR, "adversarial_arms_race.png")
        if os.path.exists(_img):
            st.image(_img, caption="Adversarial Arms Race (demo data — run `python main.py adversarial` to replace)", use_container_width=True)
        else:
            st.info("Run `python main.py adversarial` to populate this panel.")

# ── Tab 2: Regulation Shocks ─────────────────────────────────
with tab2:
    shock_df = df[df.get("shock_fired", pd.Series(dtype=bool)).fillna(False) == True].copy() if "shock_fired" in df.columns else pd.DataFrame()
    if not shock_df.empty and "shock_adapted" in shock_df.columns:
        shock_df = shock_df.dropna(subset=["shock_adapted"])
        shock_df["adapted_int"] = shock_df["shock_adapted"].astype(int)
        shock_df["rolling"] = shock_df["adapted_int"].rolling(5, min_periods=1).mean()
        fig4 = go.Figure()
        fig4.add_bar(x=shock_df["episode_num"], y=shock_df["adapted_int"],
                     name="Adapted/Missed",
                     marker_color=["#3fb950" if a else "#f85149" for a in shock_df["shock_adapted"]],
                     opacity=0.5)
        fig4.add_trace(go.Scatter(
            x=shock_df["episode_num"], y=shock_df["rolling"],
            mode="lines", name="Rolling adapt rate (5 ep)",
            line=dict(color="#58a6ff", width=2.5),
        ))
        fig4.add_hline(y=0.5, line_dash="dot", line_color="#8b949e", annotation_text="50% baseline")
        fig4.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d"),
            legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10), height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.metric("Overall Shock Adaptation Rate",
                  f"{shock_df['shock_adapted'].mean():.1%}",
                  delta="vs 0% untrained baseline")
    else:
        _img = os.path.join(PLOTS_DIR, "shock_adaptation.png")
        if os.path.exists(_img):
            st.image(_img, caption="Regulation Shock Adaptation (demo data)", use_container_width=True)
        else:
            st.info("Run `python main.py shock` to populate this panel.")

# ── Tab 3: Jurisdiction ───────────────────────────────────────
with tab3:
    jur_icons = {"FAR": "🇺🇸 FAR", "DFARS": "🛡️ DFARS", "EU": "🇪🇺 EU"}
    if "jurisdiction" in df.columns:
        jur_g = df.groupby("jurisdiction")["final_reward"].agg(["mean", "count", "std"]).reset_index()
        jur_g.columns = ["Jurisdiction", "Avg Reward", "Episodes", "Std Dev"]
        jur_g["Label"] = jur_g["Jurisdiction"].map(lambda j: jur_icons.get(j, j))
        color_map = {"FAR": "#3fb950", "DFARS": "#58a6ff", "EU": "#e3b341"}
        fig5 = go.Figure()
        for _, row in jur_g.iterrows():
            fig5.add_bar(
                x=[row["Label"]], y=[row["Avg Reward"]],
                name=row["Jurisdiction"],
                marker_color=color_map.get(row["Jurisdiction"], "#8b949e"),
                error_y=dict(type="data", array=[row["Std Dev"]], visible=True,
                             color="#8b949e", thickness=1.5),
                text=[f"n={int(row['Episodes'])}<br>{row['Avg Reward']:.3f}"],
                textposition="outside", textfont=dict(color="#c9d1d9"),
            )
        fig5.add_hline(y=0.45, line_dash="dash", line_color="#8b949e", annotation_text="Floor")
        fig5.add_hline(y=0.70, line_dash="dash", line_color="#3fb950", annotation_text="Target")
        fig5.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.1]),
            showlegend=False, margin=dict(t=20, b=10), height=320,
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(jur_g[["Label","Avg Reward","Episodes","Std Dev"]].rename(columns={"Label":"Jurisdiction"}),
                     use_container_width=True, hide_index=True)
    else:
        _img = os.path.join(PLOTS_DIR, "jurisdiction_comparison.png")
        if os.path.exists(_img):
            st.image(_img, caption="Jurisdiction Comparison (demo data)", use_container_width=True)

# ── Tab 4: Architect ──────────────────────────────────────────
with tab4:
    arch_df = df[df.get("architect_active", pd.Series(dtype=bool)).fillna(False) == True].copy() if "architect_active" in df.columns else pd.DataFrame()
    if not arch_df.empty and "in_band" in arch_df.columns:
        arch_df["calib_rolling"] = arch_df["in_band"].rolling(5, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_hline(y=0.27, line_dash="dot", line_color="#f85149",
                       annotation_text="Random baseline (27%)", annotation_font_color="#f85149", annotation_font_size=10)
        fig2.add_trace(go.Scatter(
            x=arch_df["episode_num"], y=arch_df["calib_rolling"],
            mode="lines+markers", name="Calibration (rolling 5)",
            line=dict(color="#e3b341", width=2.5), marker=dict(size=5),
        ))
        fig2.add_hrect(y0=0.60, y1=1.0, fillcolor="#3fb950", opacity=0.06,
                       annotation_text="Target >60%", annotation_font_color="#3fb950")
        fig2.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.05]),
            margin=dict(t=10, b=10), height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        _img = os.path.join(PLOTS_DIR, "architect_calibration.png")
        if os.path.exists(_img):
            st.image(_img, caption="Architect Calibration (demo data — activates at ep 51)", use_container_width=True)

    # Architect episode reasoning
    arch_eps = [e for e in episodes if e.get("architect_output")]
    if arch_eps:
        latest = arch_eps[-1]
        ao = latest["architect_output"]
        st.markdown(f"**Latest generated task:** `{ao.get('task_id','?')}` · Axis: `{ao.get('target_axis','?')}` · Difficulty: `{ao.get('difficulty','?')}`")
        st.info(f"Architect reasoning: {ao.get('architect_reasoning','')[:300]}")
    else:
        st.caption("Architect reasoning appears here after Phase 3 activates (ep 51+).")

# ── Tab 5: Diversity ──────────────────────────────────────────
with tab5:
    if "diversity_score" in df.columns:
        div_df = df.dropna(subset=["diversity_score"])
        div_df["rolling_div"] = div_df["diversity_score"].rolling(5, min_periods=1).mean()
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=div_df["episode_num"], y=div_df["diversity_score"],
            mode="markers", name="Per-episode",
            marker=dict(color="#a78bfa", size=4, opacity=0.5),
        ))
        fig10.add_trace(go.Scatter(
            x=div_df["episode_num"], y=div_df["rolling_div"],
            mode="lines", name="Rolling avg (5)",
            line=dict(color="#7c3aed", width=2.5),
        ))
        fig10.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, 1.05]),
            legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10), height=280,
        )
        st.plotly_chart(fig10, use_container_width=True)
        st.metric("Avg Curriculum Diversity", f"{div_df['diversity_score'].mean():.3f}",
                  delta="Higher = harder to game, more generalizable")
    else:
        st.info("Diversity data populates after running real episodes.")

st.divider()

# ── Live Episode Feed ──────────────────────────────────────────
st.markdown('<p class="panel-header">🔴 Live Episode Feed (Last 8)</p>', unsafe_allow_html=True)

feed_data = history[-8:][::-1]  # newest first
for r in feed_data:
    rwd   = float(r.get("final_reward") or 0)
    ep_n  = r.get("episode", 0)
    ep_fmt = f"{int(ep_n):04d}" if isinstance(ep_n, (int, float)) else _safe_str(ep_n)
    task  = _safe_str(r.get("task_id"), "?", maxlen=22)
    jur   = _safe_str(r.get("jurisdiction"), "FAR")
    diff  = _safe_str(r.get("difficulty"), "?")
    axis  = _safe_str(r.get("weakest_axis"), "?")
    delta = float(r.get("delta") or 0)
    shock = " ⚡shock" if r.get("shock_fired") else ""
    adv   = " 🦹adv" if r.get("adversarial") else ""
    bt    = " ★BT" if r.get("is_breakthrough") else ""
    cls   = "good" if rwd >= 0.70 else "mid" if rwd >= 0.45 else "bad"
    color = "#3fb950" if rwd >= 0.70 else "#e3b341" if rwd >= 0.45 else "#f85149"
    ts    = _safe_str(r.get("timestamp"), "")[:16].replace("T", " ")
    st.markdown(f"""
    <div class="ep-card {cls}">
      <span style="color:{color}; font-weight:700;">ep-{ep_fmt}</span>
      &nbsp; reward=<b style="color:{color}">{rwd:.3f}</b>
      &nbsp; Δ<span style="color:{'#3fb950' if delta>=0 else '#f85149'}">{delta:+.3f}</span>
      &nbsp;·&nbsp; {jur} · {diff} · weak=<code style="color:#8b949e">{axis}</code>
      <span style="color:#e3b341">{shock}{adv}{bt}</span>
      &nbsp;·&nbsp; task=<code style="font-size:0.78rem; color:#8b949e">{task}</code>
      <span style="float:right; color:#484f58; font-size:0.75rem">{ts}</span>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Counterfactual Explorer ────────────────────────────────────
st.markdown('<p class="panel-header">🔮 Counterfactual Consequence Explorer</p>', unsafe_allow_html=True)

if "consequence_if_approved" in df.columns:
    conq = df[df["consequence_if_approved"].notna()][
        ["episode_num", "task_id", "jurisdiction", "final_reward", "consequence_if_approved"]
    ].tail(10).rename(columns={
        "episode_num": "Episode", "task_id": "Task", "jurisdiction": "Jurisdiction",
        "final_reward": "Reward", "consequence_if_approved": "If Wrong Decision Stood",
    })
    st.dataframe(conq, use_container_width=True, hide_index=True)
else:
    st.caption("Counterfactual data populates after running real episodes with Arbiter output.")

st.divider()

# ── Raw log expander ──────────────────────────────────────────
with st.expander("📂 Full Episode Log (raw)"):
    show_cols = [c for c in [
        "episode_num", "task_id", "domain", "jurisdiction", "difficulty",
        "score_1", "final_reward", "delta", "weakest_axis",
        "is_breakthrough", "adversarial", "shock_fired", "llm_backend", "llm_model",
    ] if c in df.columns]
    st.dataframe(df[show_cols].sort_values("episode_num", ascending=False),
                 use_container_width=True, hide_index=True)
    # Download button
    csv = df[show_cols].to_csv(index=False)
    st.download_button("⬇ Download CSV", csv, "crucible_episodes.csv", "text/csv")

st.markdown(
    "<div style='text-align:center; color:#484f58; font-size:0.78rem; margin-top:20px;'>"
    "CRUCIBLE · OpenEnv Hackathon 2026 · AXIOM Corporation · Powered by Groq + HuggingFace"
    "</div>",
    unsafe_allow_html=True,
)
