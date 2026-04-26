"""CRUCIBLE dashboard — Run: streamlit run demo/dashboard.py"""

import json, os, sys, time, subprocess, threading
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR  = os.path.join(ROOT, "data", "episode_logs")
FULL_LOG = os.path.join(LOG_DIR, "full_run.json")
sys.path.insert(0, ROOT)

# ── Command registry ───────────────────────────────────────────
# Groups: label → (cmd_key, description)
CMD_GROUPS = {
    "Quick Test": [
        ("Single Episode", "episode",    "Run one episode end-to-end (~1m)"),
    ],
    "Training": [
        ("Baseline",       "baseline",   "10 episodes, no curriculum (~3m)"),
        ("Full Pipeline",  "full",       "Baseline → Train → Architect → all modes (~10m)"),
    ],
    "Special Modes": [
        ("Adversarial",    "adversarial","Vendor hides violations; Executor must find them (~1m)"),
        ("Shock",          "shock",      "Mid-episode regulation change injected (~1m)"),
        ("EU",             "eu",         "EU Directive 2014/24/EU cross-jurisdiction test (~1m)"),
    ],
    "Utilities": [
        ("Plots",          "plots",      "Regenerate charts from saved logs (~15s)"),
    ],
}

# Conservative defaults — always slightly over actual so bar rarely goes orange on first run.
# Self-calibration kicks in after the first completed run of each command.
EST_DEFAULT = {
    "episode": 75, "baseline": 210, "adversarial": 75,
    "shock": 75, "eu": 75, "full": 660, "plots": 20,
}

PROC = []   # holds active Popen


# ── Helpers ────────────────────────────────────────────────────
def _s(v, default="?", maxlen=0):
    out = str(v) if v is not None else default
    return out[:maxlen] if maxlen else out


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(dtype=object, index=df.index)


def _fmt(sec):
    """Format seconds as human-readable duration."""
    if sec is None: return "—"
    m, s = divmod(max(0, int(sec)), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m:02d}m"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"


def _load_env():
    out = {}
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p): return out
    for raw in open(p, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k: out[k] = v
    return out


def _run(cmd_arg, output_lines, done_flag):
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    for k, v in _load_env().items():
        env.setdefault(k, v)
    proc = subprocess.Popen(
        [sys.executable, "main.py", cmd_arg],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    PROC.clear(); PROC.append(proc)
    for line in proc.stdout:
        output_lines.append(line.rstrip())
        if len(output_lines) > 250: output_lines.pop(0)
    proc.wait()
    PROC.clear()
    done_flag.append(proc.returncode)


@st.cache_data(ttl=5)
def load_log():
    if os.path.exists(FULL_LOG):
        try:
            with open(FULL_LOG, encoding="utf-8") as f:
                data = json.load(f)
            if data: return data
        except Exception: pass
    out = []
    if os.path.exists(LOG_DIR):
        for fn in sorted(os.listdir(LOG_DIR)):
            if fn.startswith("ep_") and fn.endswith(".json") and fn not in ("full_run.json","baseline.json"):
                try:
                    with open(os.path.join(LOG_DIR, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception: pass
    return out


def _chart(fig, height=300, **kw):
    xo = kw.pop("xaxis", {}); yo = kw.pop("yaxis", {})
    xa = dict(gridcolor="rgba(255,255,255,0.04)", linecolor="#1e2a3a", color="#8b949e"); xa.update(xo)
    ya = dict(gridcolor="rgba(255,255,255,0.04)", linecolor="#1e2a3a", color="#8b949e"); ya.update(yo)
    fig.update_layout(
        paper_bgcolor="#111720", plot_bgcolor="#090d14",
        font=dict(color="#8b949e", family="JetBrains Mono", size=10),
        xaxis=xa, yaxis=ya,
        legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)", x=0, y=-0.2, font=dict(size=10)),
        margin=dict(t=14, b=36, l=8, r=8), height=height,
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#2b3648",
                        font=dict(family="JetBrains Mono", size=11)),
        **kw,
    )


def _est(cmd):
    """Best estimate: last actual × 1.15 (gives 15% buffer), floored at default."""
    actual = st.session_state.actual_times.get(cmd)
    if actual:
        return max(EST_DEFAULT.get(cmd, 90), int(actual * 1.15))
    return EST_DEFAULT.get(cmd, 90)


# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="CRUCIBLE", page_icon="🔥", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background:#0b1017; color:#e6edf3; font-family:Inter,sans-serif; }
section[data-testid="stSidebar"] { background:#070c12!important; border-right:1px solid #1f2a3a; }

div[data-testid="metric-container"] {
  background:#111720; border:1px solid #2b3648; border-radius:10px; padding:12px 14px;
}
div[data-testid="metric-container"] label {
  color:#7c899d!important; font-size:.68rem!important;
  letter-spacing:.08em!important; text-transform:uppercase!important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:"JetBrains Mono",monospace!important; color:#e6edf3!important;
}

div.stButton > button {
  background:#111720!important; border:1px solid #2b3648!important;
  color:#cbd5e1!important; border-radius:8px!important; text-align:left!important;
}
div.stButton > button:hover {
  border-color:#00c9e8!important; color:#00c9e8!important;
  background:rgba(0,201,232,.06)!important;
}
div.stButton > button:disabled { opacity:.35!important; cursor:not-allowed!important; }

/* Primary action button (first-run highlight) */
div.stButton.primary-btn > button {
  border-color:#00c9e8!important; color:#00c9e8!important;
  background:rgba(0,201,232,.08)!important;
}

.sec {
  font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:#7c899d; border-bottom:1px solid #1f2a3a; padding-bottom:7px; margin-bottom:10px;
}
.group-label {
  font-size:.6rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  color:#4a566a; margin:10px 0 4px 2px;
}
.card { background:#111720; border:1px solid #2b3648; border-radius:10px; padding:12px; }
.mono { font-family:"JetBrains Mono",monospace; }
.muted { color:#7c899d; }

/* Progress bar animations */
@keyframes pulse-bar {
  0%   { opacity:1; }
  50%  { opacity:0.3; }
  100% { opacity:1; }
}
.bar-pulse { animation:pulse-bar 1.1s ease-in-out infinite; }

@keyframes shimmer {
  0%   { transform:translateX(-100%); }
  100% { transform:translateX(400%); }
}
.bar-shimmer::after {
  content:""; position:absolute; top:0; left:0; width:30%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent);
  animation:shimmer 1.5s ease-in-out infinite;
}

/* Episode feed */
.feed-row {
  background:#111720; border:1px solid #2b3648; border-radius:10px;
  padding:9px 12px; margin:3px 0;
  font-family:"JetBrains Mono",monospace; font-size:.74rem;
  display:flex; align-items:center; gap:8px; flex-wrap:nowrap;
}
.feed-badge {
  font-size:.58rem; font-weight:700; padding:2px 6px;
  border-radius:4px; letter-spacing:.04em; white-space:nowrap; flex-shrink:0;
}
.feed-meta { color:#7c899d; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for k, v in dict(
    running=False, output_lines=[], done_flag=[],
    active_cmd="", last_exit=None, cmd_start=None,
    actual_times={},          # cmd -> last actual seconds (self-calibration)
    _toast_pending=False,
    _live_last_refresh=0.0,   # timestamp of last auto-refresh
).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.25rem;font-weight:800;color:#00c9e8;'
        'letter-spacing:.04em">🔥 CRUCIBLE</div>'
        '<div style="font-size:.65rem;color:#4a566a;margin-top:2px">'
        'AI Compliance Training Loop</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Detect subprocess finishing ────────────────────────────
    if st.session_state.running and st.session_state.done_flag:
        if st.session_state.cmd_start:
            actual = int(time.time() - st.session_state.cmd_start)
            st.session_state.actual_times[st.session_state.active_cmd] = actual
        st.session_state.last_exit      = st.session_state.done_flag[0]
        st.session_state.running        = False
        st.session_state.cmd_start      = None
        st.session_state._toast_pending = True
        st.cache_data.clear()

    # ── Toast on completion ────────────────────────────────────
    if st.session_state._toast_pending:
        st.session_state._toast_pending = False
        code = st.session_state.last_exit
        if code == 0:
            actual = st.session_state.actual_times.get(st.session_state.active_cmd)
            msg    = f"Done in {_fmt(actual)}" if actual else "Done"
            st.toast(msg, icon="✅")
        else:
            st.toast(f"Exited with code {code}", icon="🔴")

    # ── Running: progress panel ────────────────────────────────
    if st.session_state.running:
        elapsed  = int(time.time() - st.session_state.cmd_start) if st.session_state.cmd_start else 0
        est      = _est(st.session_state.active_cmd)
        exceeded = elapsed >= est
        pct      = min(100, int(elapsed / est * 100)) if est > 0 else 0

        # Lookup display label for the running cmd
        run_label = st.session_state.active_cmd
        for grp in CMD_GROUPS.values():
            for lbl, key, _ in grp:
                if key == st.session_state.active_cmd:
                    run_label = lbl
                    break

        if exceeded:
            overtime  = elapsed - est
            bar_style = "position:absolute;top:0;left:0;width:100%;height:100%;background:linear-gradient(90deg,#f59e0b,#ef4444);"
            bar_class = "bar-pulse"
            status    = f'<span style="color:#f59e0b;font-weight:700">+{_fmt(overtime)} over</span>'
            rem_txt   = "—"
        else:
            bar_style = f"position:absolute;top:0;left:0;width:{pct}%;height:100%;background:#00c9e8;"
            bar_class = "bar-shimmer"
            status    = f'<span style="color:#00c9e8">{pct}%</span>'
            rem_txt   = _fmt(est - elapsed)

        calibrated = st.session_state.actual_times.get(st.session_state.active_cmd)
        cal_note   = "" if calibrated else '<span style="color:#4a566a"> · first run</span>'

        st.markdown(
            f'<div class="mono" style="font-size:.8rem;color:#f59e0b;margin-bottom:6px">'
            f'▶ {run_label}</div>'
            f'<div class="card mono" style="font-size:.68rem">'
            # row 1: elapsed / remaining
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
            f'  <span class="muted">Elapsed</span>'
            f'  <b style="color:#e6edf3">{_fmt(elapsed)}</b>'
            f'  <span class="muted">Remaining</span>'
            f'  <b style="color:#e6edf3">{rem_txt}</b>'
            f'</div>'
            # progress bar
            f'<div style="position:relative;background:#1f2a3a;height:7px;border-radius:4px;overflow:hidden">'
            f'  <div class="{bar_class}" style="{bar_style}"></div>'
            f'</div>'
            # row 2: est / status
            f'<div style="display:flex;justify-content:space-between;margin-top:5px">'
            f'  <span class="muted">est {_fmt(est)}{cal_note}</span>'
            f'  {status}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⏹ Stop", use_container_width=True, key="sb_stop"):
            if PROC:
                try: PROC[0].terminate(); PROC[0].wait(timeout=3)
                except Exception:
                    try: PROC[0].kill()
                    except Exception: pass
                PROC.clear()
            st.session_state.running   = False
            st.session_state.cmd_start = None
            st.session_state.output_lines.append("-- stopped --")

    # ── Idle: last run status + command buttons ────────────────
    else:
        if st.session_state.last_exit is not None:
            c       = "#10b981" if st.session_state.last_exit == 0 else "#f43f5e"
            icon    = "✓" if st.session_state.last_exit == 0 else "✗"
            t       = "Done" if st.session_state.last_exit == 0 else f"Exit {st.session_state.last_exit}"
            actual  = st.session_state.actual_times.get(st.session_state.active_cmd)
            t_str   = f" · {_fmt(actual)}" if actual else ""
            st.markdown(
                f'<div class="mono" style="color:{c};font-size:.78rem;margin-bottom:6px">'
                f'{icon} {t}{t_str}</div>',
                unsafe_allow_html=True,
            )

        has_data = bool(load_log())

        for group_name, cmds in CMD_GROUPS.items():
            st.markdown(f'<div class="group-label">{group_name}</div>', unsafe_allow_html=True)
            for label, cmd, desc in cmds:
                actual  = st.session_state.actual_times.get(cmd)
                est_sec = _est(cmd)
                tooltip = f"{desc}\n\nest: {_fmt(est_sec)}" + (f"\nlast: {_fmt(actual)}" if actual else " (first run)")

                # Highlight "Single Episode" if no data yet
                is_primary = (cmd == "episode" and not has_data)
                prefix = "▶ " if is_primary else ""

                if st.button(
                    f"{prefix}{label}",
                    use_container_width=True,
                    help=tooltip,
                    key=f"sb_cmd_{cmd}",
                    disabled=st.session_state.running,
                ):
                    st.session_state.output_lines = [f"$ python main.py {cmd}"]
                    st.session_state.done_flag    = []
                    st.session_state.running      = True
                    st.session_state.active_cmd   = cmd
                    st.session_state.last_exit    = None
                    st.session_state.cmd_start    = time.time()
                    threading.Thread(
                        target=_run,
                        args=(cmd, st.session_state.output_lines, st.session_state.done_flag),
                        daemon=True,
                    ).start()
                    st.rerun()

    st.divider()

    # ── Auto-refresh (non-blocking poll) ──────────────────────
    ca, cb = st.columns([1, 2])
    live     = ca.checkbox("Live", value=False, key="sb_live")
    interval = cb.slider("", 5, 60, 10, label_visibility="collapsed", key="sb_interval")

    if st.button("↺ Refresh", use_container_width=True, key="sb_refresh"):
        st.cache_data.clear(); st.rerun()

    # ── Live output log ───────────────────────────────────────
    if st.session_state.output_lines:
        st.divider()
        st.code("\n".join(st.session_state.output_lines[-30:]), language="bash")

    # ── Rerun scheduling ──────────────────────────────────────
    if st.session_state.running:
        # Fast poll while a command is running
        time.sleep(0.8)
        st.rerun()
    elif live:
        # Non-blocking live refresh: poll every 1s, refresh cache when interval elapsed
        now = time.time()
        since_last = now - st.session_state._live_last_refresh
        if since_last >= interval:
            st.session_state._live_last_refresh = now
            st.cache_data.clear()
            st.rerun()
        else:
            time.sleep(1.0)
            st.rerun()


# ── Data ──────────────────────────────────────────────────────
history = load_log()
if history:
    df = pd.DataFrame(history)
    df["episode_num"] = range(1, len(df) + 1)
    for c in ("final_reward", "score_1", "delta"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
else:
    df = pd.DataFrame()

# ── Header ────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
    '<span style="font-size:1.4rem;font-weight:800;color:#00c9e8">CRUCIBLE</span>'
    '<span style="font-size:.7rem;color:#4a566a;font-family:\'JetBrains Mono\',monospace">'
    'AI Compliance Training Loop</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── No-data state ─────────────────────────────────────────────
if df.empty:
    st.markdown(
        '<div class="card" style="margin-top:24px;padding:28px;text-align:center">'
        '<div style="font-size:2rem;margin-bottom:12px">🔥</div>'
        '<div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:8px">'
        'No episodes yet</div>'
        '<div style="font-size:.82rem;color:#7c899d">'
        'Start with <b style="color:#00c9e8">Single Episode</b> in the sidebar '
        'to run a quick test, then use <b style="color:#00c9e8">Baseline</b> or '
        '<b style="color:#00c9e8">Full Pipeline</b> for training.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

ep_count    = len(df)
avg_reward  = float(df["final_reward"].mean())
last_reward = float(df["final_reward"].iloc[-1])
shock_count = int(_col(df, "shock_fired").fillna(False).sum())
trend = None
if len(df) > 5:
    prev  = df["final_reward"].iloc[max(0, len(df)-11):max(1, len(df)-1)].mean()
    trend = round(avg_reward - prev, 3)

# ── Metrics ───────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Episodes",   ep_count)
m2.metric("Avg Reward", f"{avg_reward:.3f}", delta=f"{trend:+.3f}" if trend else None)
m3.metric("Latest",     f"{last_reward:.3f}")
m4.metric("Shocks",     shock_count)
st.divider()

# ── Charts ────────────────────────────────────────────────────
cl, cr = st.columns([3, 2])
with cl:
    st.markdown('<div class="sec">Reward Curve</div>', unsafe_allow_html=True)
    rolling = df["final_reward"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    if "score_1" in df.columns:
        fig.add_trace(go.Scatter(x=df["episode_num"], y=df["score_1"],
            mode="lines", name="Attempt 1",
            line=dict(color="rgba(124,137,157,0.4)", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df["episode_num"], y=df["final_reward"],
        mode="lines", name="Final",
        line=dict(color="rgba(0,201,232,0.45)", width=1.2),
        fill="tozeroy", fillcolor="rgba(0,201,232,0.05)"))
    fig.add_trace(go.Scatter(x=df["episode_num"], y=rolling,
        mode="lines", name="Rolling 5",
        line=dict(color="#00c9e8", width=2.4),
        fill="tozeroy", fillcolor="rgba(0,201,232,0.08)"))
    fig.add_hrect(y0=0.45, y1=0.70, fillcolor="rgba(16,185,129,0.05)")
    am = _col(df, "architect_active").fillna(False)
    if am.any():
        ax = df[am]["episode_num"].min()
        if not pd.isna(ax):
            fig.add_vline(x=ax, line_dash="dash", line_color="#10b981",
                          annotation_text="Architect on", annotation_font_color="#10b981")
    bt = df[_col(df, "is_breakthrough").fillna(False)]
    if not bt.empty:
        fig.add_trace(go.Scatter(x=bt["episode_num"], y=bt["final_reward"],
            mode="markers", name="Breakthrough",
            marker=dict(color="#f59e0b", symbol="star", size=12,
                        line=dict(color="#090d14", width=1))))
    _chart(fig, height=320,
           xaxis=dict(title="Episode"),
           yaxis=dict(title="Reward", range=[0, 1.05]))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.markdown('<div class="sec">Score</div>', unsafe_allow_html=True)
    sc = "#10b981" if avg_reward >= 0.70 else "#f59e0b" if avg_reward >= 0.45 else "#f43f5e"
    grade = "GOOD" if avg_reward >= 0.70 else "OK" if avg_reward >= 0.45 else "WEAK"

    # Score bar with true-position threshold ticks
    st.markdown(
        f'<div class="card">'
        f'<div class="mono" style="font-size:2.4rem;font-weight:700;color:{sc};text-align:center">'
        f'{avg_reward:.3f}</div>'
        f'<div style="text-align:center;margin:-4px 0 10px">'
        f'<span style="font-size:.6rem;font-weight:700;letter-spacing:.1em;'
        f'background:rgba(0,0,0,.3);border:1px solid {sc};color:{sc};'
        f'padding:2px 8px;border-radius:4px">{grade}</span>'
        f'</div>'
        # bar container
        f'<div style="position:relative;background:#1f2a3a;height:10px;'
        f'border-radius:6px;overflow:visible;margin-bottom:18px">'
        # filled portion
        f'<div style="position:absolute;top:0;left:0;width:{int(avg_reward*100)}%;'
        f'height:10px;border-radius:6px;background:{sc}"></div>'
        # 0.45 tick
        f'<div style="position:absolute;top:-4px;left:45%;width:1px;height:18px;'
        f'background:#4a566a"></div>'
        # 0.70 tick
        f'<div style="position:absolute;top:-4px;left:70%;width:1px;height:18px;'
        f'background:#4a566a"></div>'
        f'</div>'
        # labels at correct positions
        f'<div class="mono muted" style="position:relative;font-size:.6rem;height:12px;'
        f'margin-top:-12px;margin-bottom:8px">'
        f'<span style="position:absolute;left:0;transform:translateX(0)">0.0</span>'
        f'<span style="position:absolute;left:45%;transform:translateX(-50%)">0.45</span>'
        f'<span style="position:absolute;left:70%;transform:translateX(-50%)">0.70</span>'
        f'<span style="position:absolute;right:0;transform:translateX(0)">1.0</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if len(df) >= 10:
        f5 = df["final_reward"].head(5).mean()
        l5 = df["final_reward"].tail(5).mean()
        x1, x2 = st.columns(2)
        x1.metric("First 5", f"{f5:.3f}")
        x2.metric("Last 5",  f"{l5:.3f}", delta=f"{l5-f5:+.3f}")

st.divider()

# ── Episode feed ──────────────────────────────────────────────
fc, _ = st.columns([2, 1])
with fc:
    st.markdown('<div class="sec">Recent Episodes</div>', unsafe_allow_html=True)
    for r in history[-12:][::-1]:
        rwd      = float(r.get("final_reward") or 0)
        ep       = r.get("episode", 0)
        ep_s     = f"{int(ep):04d}" if isinstance(ep, (int, float)) else _s(ep)
        jur      = _s(r.get("jurisdiction"), "FAR")
        diff     = _s(r.get("difficulty"), "?")
        axis     = _s(r.get("weakest_axis"), "?", 12)
        delta    = float(r.get("delta") or 0)
        ts       = _s(r.get("timestamp"), "")[:16].replace("T", " ")
        rc       = "#10b981" if rwd >= 0.70 else "#f59e0b" if rwd >= 0.45 else "#f43f5e"
        dc       = "#10b981" if delta >= 0 else "#f43f5e"
        is_shock = bool(r.get("shock_fired"))
        is_adv   = bool(r.get("adversarial"))

        badges = ""
        if is_shock:
            badges += ('<span class="feed-badge" '
                       'style="background:rgba(245,158,11,.15);color:#f59e0b">⚡ SHOCK</span>')
        if is_adv:
            badges += ('<span class="feed-badge" '
                       'style="background:rgba(239,68,68,.15);color:#ef4444">⚔ ADV</span>')

        st.markdown(
            f'<div class="feed-row" style="border-left:3px solid {rc}">'
            f'  <span style="color:{rc};font-weight:700;min-width:64px">ep-{ep_s}</span>'
            f'  <span style="color:{rc};min-width:44px">{rwd:.3f}</span>'
            f'  <span style="color:{dc};min-width:52px">{delta:+.3f}</span>'
            f'  <span class="feed-meta">· {jur} · {diff} · {axis}</span>'
            f'  {badges}'
            f'  <span class="muted" style="margin-left:auto;white-space:nowrap;'
            f'font-size:.68rem">{ts}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Raw log ───────────────────────────────────────────────────
with st.expander("Raw log"):
    show = [c for c in ["episode_num","task_id","jurisdiction","difficulty",
                         "score_1","final_reward","delta","weakest_axis",
                         "adversarial","shock_fired"] if c in df.columns]
    st.dataframe(df[show].sort_values("episode_num", ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df[show].to_csv(index=False),
                       "crucible_episodes.csv", "text/csv", key="sb_download")
