"""
AI Signal Codex — Portfolio Analytics Dashboard
================================================
Interactive Streamlit dashboard for AI-driven signal portfolio monitoring.
Redesigned with Plotly charts, tab-based navigation, and auto-generated insights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# ─── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Signal Codex",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* tighter metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d24 100%);
        border: 1px solid #2d333b;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }

    /* insight cards */
    .insight-card {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .insight-card.warn { border-left-color: #d29922; }
    .insight-card.bad  { border-left-color: #f85149; }
    .insight-card.good { border-left-color: #3fb950; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=30),
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.15),
)

ANN = 252.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS (cached 5 min)
# ═══════════════════════════════════════════════════════════════════════════════
def _csv(name: str, idx: str | None = None, dates: list[str] | None = None) -> pd.DataFrame | None:
    path = CSV_DIR / name
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=dates or [])
    if idx and idx in df.columns:
        df = df.set_index(idx)
    return df


def _report(name: str, dates: list[str] | None = None) -> pd.DataFrame | None:
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=dates or [])


@st.cache_data(ttl=300)
def load_perf():
    return _csv("daily_performance.csv", idx="date", dates=["date", "rebalance_date"])

@st.cache_data(ttl=300)
def load_weights():
    return _csv("portfolio_weights.csv", idx="date", dates=["date"])

@st.cache_data(ttl=300)
def load_bm_weights():
    return _csv("benchmark_weights.csv", idx="date", dates=["date"])

@st.cache_data(ttl=300)
def load_fi():
    return _csv("feature_importance.csv")

@st.cache_data(ttl=300)
def load_ic():
    return _csv("ic_series.csv", dates=["date"])

@st.cache_data(ttl=300)
def load_tilt():
    return _csv("style_sector_tilt.csv", dates=["date"])

@st.cache_data(ttl=300)
def load_regime():
    return _csv("monthly_regime.csv")

@st.cache_data(ttl=300)
def load_group_attr():
    return _csv("group_attribution.csv", idx="date", dates=["date"])

@st.cache_data(ttl=300)
def load_li_attr():
    return _csv("li_attribution.csv", dates=["date"])

@st.cache_data(ttl=300)
def load_model_struct():
    return _csv("model_structure.csv", dates=["retrain_date"])

@st.cache_data(ttl=300)
def load_scores():
    return _csv("stock_scores.csv", idx="date", dates=["date"])

@st.cache_data(ttl=300)
def load_shap():
    return _csv("stock_shap_attribution.csv", dates=["date"])

@st.cache_data(ttl=300)
def load_ow_expl():
    return _report("lightgbm_monthly_ow_explanations.csv")

@st.cache_data(ttl=300)
def load_policy_compare():
    return _report("lightgbm_latest_policy_compare.csv", dates=["date"])

@st.cache_data(ttl=300)
def load_overall_metrics():
    return _report("lightgbm_overall_metrics.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def _pct(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}%}"


def _f(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def _insight(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="insight-card {kind}">{text}</div>', unsafe_allow_html=True)


def _trim(df: pd.DataFrame, start, end, col: str | None = None):
    """Filter DataFrame by date range on index or column."""
    if start is None or end is None:
        return df
    if col:
        return df.loc[(df[col] >= start) & (df[col] <= end)].copy()
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def metrics(perf: pd.DataFrame) -> dict[str, float]:
    port = perf["fund_daily_return"].fillna(0.0)
    bm = perf["bm_daily_return"].fillna(0.0)
    active = perf["active_daily_return"].fillna(0.0)
    ann_ret = float(port.mean() * ANN)
    ann_vol = float(port.std(ddof=0) * np.sqrt(ANN))
    bm_ret = float(bm.mean() * ANN)
    bm_vol = float(bm.std(ddof=0) * np.sqrt(ANN))
    active_ret = float(active.mean() * ANN)
    te = float(active.std(ddof=0) * np.sqrt(ANN))
    sharpe = ann_ret / ann_vol if ann_vol else float("nan")
    bm_sharpe = bm_ret / bm_vol if bm_vol else float("nan")
    ir = active_ret / te if te else float("nan")
    cum = perf["fund_cumulative"]
    dd = (cum / cum.cummax()) - 1.0
    total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1.0) if len(cum) > 1 else 0.0
    total_bm = float(perf["bm_cumulative"].iloc[-1] / perf["bm_cumulative"].iloc[0] - 1.0) if len(perf) > 1 else 0.0
    # calmar = annual return / |max drawdown|
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")
    # sortino
    downside = port[port < 0]
    down_vol = float(downside.std(ddof=0) * np.sqrt(ANN)) if len(downside) > 0 else float("nan")
    sortino = ann_ret / down_vol if down_vol and down_vol > 0 else float("nan")

    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        bm_ret=bm_ret, bm_vol=bm_vol, bm_sharpe=bm_sharpe,
        active_ret=active_ret, te=te, ir=ir,
        max_dd=max_dd, total_ret=total_ret, total_bm_ret=total_bm,
        win_rate=float((active > 0).mean()),
        n_days=len(perf), n_years=float(len(perf) / ANN),
        calmar=calmar, sortino=sortino, down_vol=down_vol,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def sidebar():
    st.sidebar.markdown("## 📡 AI Signal Codex")
    st.sidebar.caption("LightGBM multi-factor quant portfolio")

    perf = load_perf()
    if perf is None or perf.empty:
        st.sidebar.error("No data — run pipeline first")
        return None, None

    mn, mx = perf.index.min().date(), perf.index.max().date()
    st.sidebar.markdown(f"**Data range** `{mn}` → `{mx}`")

    date_range = st.sidebar.date_input(
        "Analysis period",
        value=(mn, mx), min_value=mn, max_value=mx,
    )
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = pd.Timestamp(mn), pd.Timestamp(mx)

    # quick stats in sidebar
    p = _trim(perf, start, end)
    m = metrics(p)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Stats**")
    st.sidebar.markdown(
        f"- Return: **{_pct(m['ann_ret'])}** (BM {_pct(m['bm_ret'])})\n"
        f"- Sharpe: **{_f(m['sharpe'])}** | IR: **{_f(m['ir'])}**\n"
        f"- Max DD: **{_pct(m['max_dd'])}**\n"
        f"- Win Rate: **{_pct(m['win_rate'], 1)}**"
    )

    return start, end


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def tab_overview(start, end):
    perf = load_perf()
    if perf is None or perf.empty:
        st.error("daily_performance.csv not found. Run pipeline first.")
        return
    perf = _trim(perf, start, end)
    m = metrics(perf)
    ic = load_ic()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Ann. Return", _pct(m["ann_ret"]), f"{m['active_ret']:+.2%} vs BM")
    c2.metric("Sharpe", _f(m["sharpe"]), f"BM {_f(m['bm_sharpe'])}")
    c3.metric("Info Ratio", _f(m["ir"]))
    c4.metric("Sortino", _f(m["sortino"]))
    c5.metric("Calmar", _f(m["calmar"]))
    c6.metric("Max DD", _pct(m["max_dd"]))
    mean_ic = ic["IC"].mean() if ic is not None and not ic.empty else float("nan")
    c7.metric("Avg IC", _f(mean_ic, 4))

    # ── auto insights ────────────────────────────────────────────────────────
    st.markdown("#### Auto Insights")
    col_ins_l, col_ins_r = st.columns(2)
    with col_ins_l:
        if m["ir"] > 0.5:
            _insight(f"IR {_f(m['ir'])} — 강한 알파 생성 능력. 시그널 품질 우수.", "good")
        elif m["ir"] > 0:
            _insight(f"IR {_f(m['ir'])} — 양(+)의 알파이나 개선 여지 있음.", "info")
        else:
            _insight(f"IR {_f(m['ir'])} — 음(-)의 알파. 시그널 점검 필요.", "bad")

        if m["max_dd"] < -0.15:
            _insight(f"Max DD {_pct(m['max_dd'])} — 15% 초과 낙폭 발생. 리스크 관리 검토.", "bad")
        elif m["max_dd"] < -0.10:
            _insight(f"Max DD {_pct(m['max_dd'])} — 10~15% 낙폭. 모니터링 권장.", "warn")
        else:
            _insight(f"Max DD {_pct(m['max_dd'])} — 양호한 리스크 수준.", "good")
    with col_ins_r:
        if m["win_rate"] > 0.55:
            _insight(f"Win Rate {_pct(m['win_rate'],1)} — 55% 이상. 일관된 방향성.", "good")
        else:
            _insight(f"Win Rate {_pct(m['win_rate'],1)} — 낮은 승률은 tail risk 전략일 수 있음.", "warn")

        if not np.isnan(mean_ic) and mean_ic > 0.05:
            _insight(f"Mean IC {_f(mean_ic,4)} — 5% 이상이면 실전 가능 수준.", "good")
        elif not np.isnan(mean_ic) and mean_ic > 0:
            _insight(f"Mean IC {_f(mean_ic,4)} — 양(+)이나 강한 시그널은 아님.", "info")

    # ── cumulative return chart ──────────────────────────────────────────────
    st.markdown("#### Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index, y=perf["fund_cumulative"],
        name="Fund", line=dict(color="#58a6ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=perf.index, y=perf["bm_cumulative"],
        name="Benchmark", line=dict(color="#8b949e", width=1.5, dash="dot"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Fund vs Benchmark (Cumulative)")
    st.plotly_chart(fig, use_container_width=True)

    # ── drawdown chart ───────────────────────────────────────────────────────
    dd = (perf["fund_cumulative"] / perf["fund_cumulative"].cummax()) - 1.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=perf.index, y=dd, fill="tozeroy",
        line=dict(color="#f85149", width=1),
        fillcolor="rgba(248,81,73,0.2)", name="Drawdown",
    ))
    fig_dd.update_layout(**PLOTLY_LAYOUT, title="Drawdown", yaxis_tickformat=".1%")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── summary table ────────────────────────────────────────────────────────
    st.markdown("#### Summary Statistics")
    summary = pd.DataFrame({
        "Metric": [
            "Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Calmar Ratio", "Total Return", "Max Drawdown",
            "Active Return", "Tracking Error", "Info Ratio",
            "Win Rate (daily)", "Period (yrs)",
        ],
        "Fund": [
            _pct(m["ann_ret"]), _pct(m["ann_vol"]), _f(m["sharpe"]), _f(m["sortino"]),
            _f(m["calmar"]), _pct(m["total_ret"]), _pct(m["max_dd"]),
            _pct(m["active_ret"]), _pct(m["te"]), _f(m["ir"]),
            _pct(m["win_rate"], 1), _f(m["n_years"], 1),
        ],
        "Benchmark": [
            _pct(m["bm_ret"]), _pct(m["bm_vol"]), _f(m["bm_sharpe"]), "—",
            "—", _pct(m["total_bm_ret"]), "—",
            "—", "—", "—",
            "—", "—",
        ],
    })
    st.dataframe(summary.set_index("Metric"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: RETURNS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def tab_returns(start, end):
    perf = load_perf()
    if perf is None or perf.empty:
        return
    perf = _trim(perf, start, end)
    active = perf["active_daily_return"]

    # ── rolling IR ───────────────────────────────────────────────────────────
    window = st.slider("Rolling window (days)", 63, 504, 252, 21, key="roll_win")
    min_per = max(5, window // 4)
    rolling_mean = active.rolling(window, min_periods=min_per).mean() * ANN
    rolling_std = active.rolling(window, min_periods=min_per).std(ddof=0) * np.sqrt(ANN)
    rolling_ir = rolling_mean / rolling_std

    fig_ir = go.Figure()
    fig_ir.add_trace(go.Scatter(
        x=rolling_ir.index, y=rolling_ir,
        line=dict(color="#d2a8ff", width=2), name="Rolling IR",
    ))
    fig_ir.add_hline(y=0, line_dash="dash", line_color="#484f58")
    fig_ir.add_hline(y=0.5, line_dash="dot", line_color="#3fb950", annotation_text="IR=0.5")
    fig_ir.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {window}-day IR")
    st.plotly_chart(fig_ir, use_container_width=True)

    # ── rolling sharpe & vol ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        port_ret = perf["fund_daily_return"].fillna(0)
        r_sharpe = (port_ret.rolling(window, min_periods=min_per).mean() * ANN) / \
                   (port_ret.rolling(window, min_periods=min_per).std(ddof=0) * np.sqrt(ANN))
        fig_sh = go.Figure()
        fig_sh.add_trace(go.Scatter(x=r_sharpe.index, y=r_sharpe, line=dict(color="#79c0ff", width=2), name="Rolling Sharpe"))
        fig_sh.add_hline(y=1.0, line_dash="dot", line_color="#3fb950", annotation_text="Sharpe=1")
        fig_sh.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {window}-day Sharpe")
        st.plotly_chart(fig_sh, use_container_width=True)
    with col_r:
        r_vol = port_ret.rolling(window, min_periods=min_per).std(ddof=0) * np.sqrt(ANN)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=r_vol.index, y=r_vol, fill="tozeroy", line=dict(color="#f0883e", width=1.5), fillcolor="rgba(240,136,62,0.15)", name="Vol"))
        fig_vol.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {window}-day Annualized Vol", yaxis_tickformat=".1%")
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── monthly heatmap ──────────────────────────────────────────────────────
    st.markdown("#### Monthly Return Heatmap")
    for label, column, cmap in [("Fund", "fund_daily_return", "RdYlGn"), ("Active", "active_daily_return", "RdBu")]:
        monthly = perf[column].resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
        pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }).pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        fig_hm = px.imshow(
            pivot.values, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index],
            color_continuous_scale=cmap, aspect="auto",
            labels=dict(color="Return"), text_auto=".1%",
        )
        fig_hm.update_layout(**PLOTLY_LAYOUT, title=f"{label} Monthly Returns", height=max(200, 60 * len(pivot)))
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
def tab_portfolio(start, end):
    weights = load_weights()
    bm = load_bm_weights()
    if weights is None or weights.empty or bm is None or bm.empty:
        st.error("Weight data not found.")
        return

    cutoff = weights.index.max() - pd.DateOffset(months=12)
    hw = weights.loc[weights.index >= cutoff]
    hb = bm.loc[bm.index >= cutoff]
    dates = sorted(hw.index.unique())

    selected_date = st.selectbox(
        "Rebalance date", dates[::-1],
        format_func=lambda v: pd.Timestamp(v).strftime("%Y-%m-%d"),
    )

    fund_row = hw.loc[selected_date].sort_values(ascending=False)
    bm_row = hb.loc[selected_date].reindex(fund_row.index).fillna(0.0)
    comp = pd.DataFrame({
        "Fund": fund_row, "Benchmark": bm_row, "Active": fund_row - bm_row,
    }).sort_values("Active", ascending=False)

    # ── KPI row ──────────────────────────────────────────────────────────────
    active_share = 0.5 * comp["Active"].abs().sum()
    n_ow = (comp["Active"] > 1e-6).sum()
    n_uw = (comp["Active"] < -1e-6).sum()
    top_ow_name = comp.index[0] if not comp.empty else "—"
    max_aw = comp["Active"].iloc[0] if not comp.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Share", _pct(active_share))
    c2.metric("OW / UW stocks", f"{n_ow} / {n_uw}")
    c3.metric("Top OW", f"{top_ow_name} ({_pct(max_aw)})")
    c4.metric("Holdings", str(len(comp[comp["Fund"] > 1e-6])))

    # ── active weight waterfall ──────────────────────────────────────────────
    st.markdown("#### Active Weight — Top 20")
    top20 = comp.head(10).append(comp.tail(10)) if len(comp) > 20 else comp
    top20 = pd.concat([comp.head(10), comp.tail(10)]) if len(comp) > 20 else comp
    top20 = top20.drop_duplicates()
    colors = ["#3fb950" if v > 0 else "#f85149" for v in top20["Active"]]
    fig_aw = go.Figure(go.Bar(
        x=top20.index, y=top20["Active"],
        marker_color=colors,
    ))
    fig_aw.update_layout(**PLOTLY_LAYOUT, title="Active Weights (Top OW + Top UW)", yaxis_tickformat=".1%")
    st.plotly_chart(fig_aw, use_container_width=True)

    # ── portfolio table ──────────────────────────────────────────────────────
    with st.expander("Full portfolio weights", expanded=False):
        display = comp.copy()
        display["Fund"] = display["Fund"].map("{:.2%}".format)
        display["Benchmark"] = display["Benchmark"].map("{:.2%}".format)
        display["Active"] = display["Active"].map("{:+.2%}".format)
        st.dataframe(display, use_container_width=True, height=400)

    # ── 12-month weight history ──────────────────────────────────────────────
    st.markdown("#### 12-Month Weight History (Top 10)")
    top_names = comp.index[:10].tolist()
    fig_wh = go.Figure()
    for name in top_names:
        fig_wh.add_trace(go.Scatter(x=hw.index, y=hw[name], mode="lines", name=name))
    fig_wh.update_layout(**PLOTLY_LAYOUT, title="Fund Weight History", yaxis_tickformat=".1%")
    st.plotly_chart(fig_wh, use_container_width=True)

    # ── latest inference policy compare ──────────────────────────────────────
    lpc = load_policy_compare()
    if lpc is not None and not lpc.empty:
        st.markdown("#### Latest Inference — Policy Comparison")
        compare_date = lpc["date"].max()
        block = lpc[lpc["date"] == compare_date]
        policies = block["policy"].unique().tolist()
        if len(policies) >= 2:
            tabs_policy = st.tabs(policies[:2])
            for tab, pol in zip(tabs_policy, policies[:2]):
                with tab:
                    pdf = block[block["policy"] == pol].sort_values("active_weight", ascending=False)
                    a_share = 0.5 * pdf["active_weight"].abs().sum()
                    st.caption(f"Active Share: {_pct(a_share)} | Max |AW|: {_pct(pdf['active_weight'].abs().max())}")
                    cols_show = [c for c in ["asset", "fund_weight", "benchmark_weight", "active_weight", "prediction", "sector"] if c in pdf.columns]
                    st.dataframe(pdf[cols_show].head(15), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: STOCK DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
def tab_stock_dive(start, end):
    weights = load_weights()
    bm = load_bm_weights()
    scores = load_scores()
    shap_attr = load_shap()

    if weights is None or weights.empty:
        st.error("No portfolio weight data.")
        return

    latest = weights.index.max()
    cutoff = latest - pd.DateOffset(months=12)
    all_tickers = sorted(weights.columns.tolist())

    # default: top 5 by latest active weight
    if bm is not None and not bm.empty:
        latest_fund = weights.loc[latest]
        latest_bm = bm.loc[bm.index <= latest].iloc[-1].reindex(latest_fund.index).fillna(0)
        top5 = (latest_fund - latest_bm).sort_values(ascending=False).head(5).index.tolist()
    else:
        top5 = all_tickers[:5]

    selected = st.multiselect("Select stocks", all_tickers, default=top5)
    if not selected:
        st.info("Pick at least one stock.")
        return

    # ── weight history ───────────────────────────────────────────────────────
    st.markdown("#### Weight History (12M)")
    fund_12m = weights.loc[weights.index >= cutoff, selected]
    fig_w = go.Figure()
    for col in fund_12m.columns:
        fig_w.add_trace(go.Scatter(x=fund_12m.index, y=fund_12m[col], name=col))
    fig_w.update_layout(**PLOTLY_LAYOUT, title="Fund Weight", yaxis_tickformat=".1%")
    st.plotly_chart(fig_w, use_container_width=True)

    if bm is not None and not bm.empty:
        active_12m = fund_12m - bm.loc[bm.index >= cutoff, selected].reindex(fund_12m.index).fillna(0)
        fig_aw = go.Figure()
        for col in active_12m.columns:
            fig_aw.add_trace(go.Scatter(x=active_12m.index, y=active_12m[col], name=col))
        fig_aw.update_layout(**PLOTLY_LAYOUT, title="Active Weight", yaxis_tickformat=".1%")
        st.plotly_chart(fig_aw, use_container_width=True)

    # ── score history ────────────────────────────────────────────────────────
    if scores is not None and not scores.empty:
        valid_cols = [c for c in selected if c in scores.columns]
        score_12m = scores.loc[scores.index >= cutoff, valid_cols]
        if not score_12m.empty:
            st.markdown("#### Model Score (Prediction) History")
            fig_s = go.Figure()
            for col in score_12m.columns:
                fig_s.add_trace(go.Scatter(x=score_12m.index, y=score_12m[col], name=col))
            fig_s.update_layout(**PLOTLY_LAYOUT, title="Score")
            st.plotly_chart(fig_s, use_container_width=True)

            # snapshot table
            snap_score = score_12m.iloc[-1]
            snap_fund = fund_12m.iloc[-1].reindex(snap_score.index).fillna(0)
            snap_bm_vals = bm.loc[bm.index <= latest, snap_score.index].iloc[-1].fillna(0) if bm is not None and not bm.empty else pd.Series(0, index=snap_score.index)
            snap = pd.DataFrame({
                "Score": snap_score,
                "Fund Wt": snap_fund,
                "BM Wt": snap_bm_vals,
                "Active": snap_fund - snap_bm_vals,
            }).sort_values("Active", ascending=False)
            st.dataframe(snap.style.format({
                "Score": "{:.4f}", "Fund Wt": "{:.2%}", "BM Wt": "{:.2%}", "Active": "{:+.2%}",
            }), use_container_width=True)

    # ── SHAP factor attribution ──────────────────────────────────────────────
    if shap_attr is not None and not shap_attr.empty:
        st.markdown("#### Score Factor Attribution (SHAP)")
        s12 = shap_attr[(shap_attr["date"] >= cutoff) & (shap_attr["ticker"].isin(selected))]
        if not s12.empty:
            chosen = st.selectbox("Stock for factor drilldown", sorted(s12["ticker"].unique()))
            sv = s12[s12["ticker"] == chosen].set_index("date")
            groups = [c for c in ["Accounting", "Price", "Sellside", "Conditioning", "Factor"] if c in sv.columns]
            fig_shap = go.Figure()
            for g in groups:
                fig_shap.add_trace(go.Scatter(x=sv.index, y=sv[g], stackgroup="one", name=g))
            fig_shap.update_layout(**PLOTLY_LAYOUT, title=f"{chosen} — SHAP Factor Decomposition")
            st.plotly_chart(fig_shap, use_container_width=True)

            # total score line
            if "total" in sv.columns:
                fig_tot = go.Figure()
                fig_tot.add_trace(go.Scatter(x=sv.index, y=sv["total"], line=dict(color="#58a6ff", width=2), name="Total Score"))
                fig_tot.add_hline(y=0, line_dash="dash", line_color="#484f58")
                fig_tot.update_layout(**PLOTLY_LAYOUT, title=f"{chosen} — Total SHAP Score")
                st.plotly_chart(fig_tot, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5: SECTOR & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
def tab_sector_style(start, end):
    tilt = load_tilt()
    if tilt is None or tilt.empty:
        st.error("No sector/style data.")
        return
    tilt = _trim(tilt, start, end, col="date")

    sector_cols = [c for c in tilt.columns if c.startswith("sector_")]
    style_cols = [c for c in tilt.columns if c.startswith("style_")]

    # ── sector active weights ────────────────────────────────────────────────
    st.markdown("#### Sector Active Weights Over Time")
    fig_sec = go.Figure()
    for col in sector_cols:
        label = col.replace("sector_", "")
        fig_sec.add_trace(go.Scatter(x=tilt["date"], y=tilt[col], name=label))
    fig_sec.add_hline(y=0, line_dash="dash", line_color="#484f58")
    fig_sec.update_layout(**PLOTLY_LAYOUT, title="Sector Active Tilt", yaxis_tickformat=".1%")
    st.plotly_chart(fig_sec, use_container_width=True)

    # ── style active exposures ───────────────────────────────────────────────
    st.markdown("#### Style Active Exposures Over Time")
    fig_sty = go.Figure()
    for col in style_cols:
        label = col.replace("style_", "")
        fig_sty.add_trace(go.Scatter(x=tilt["date"], y=tilt[col], name=label))
    fig_sty.add_hline(y=0, line_dash="dash", line_color="#484f58")
    fig_sty.update_layout(**PLOTLY_LAYOUT, title="Style Active Tilt")
    st.plotly_chart(fig_sty, use_container_width=True)

    # ── latest snapshot ──────────────────────────────────────────────────────
    st.markdown("#### Latest Tilt Snapshot")
    latest = tilt.iloc[-1]
    col_l, col_r = st.columns(2)
    with col_l:
        sec_data = latest[sector_cols].rename(lambda c: c.replace("sector_", ""))
        sec_data = sec_data[sec_data.abs() > 1e-6].sort_values(ascending=True)
        colors = ["#3fb950" if v > 0 else "#f85149" for v in sec_data]
        fig_snap_sec = go.Figure(go.Bar(y=sec_data.index, x=sec_data.values, orientation="h", marker_color=colors))
        fig_snap_sec.update_layout(**PLOTLY_LAYOUT, title="Sector Tilts", xaxis_tickformat=".1%", height=350)
        st.plotly_chart(fig_snap_sec, use_container_width=True)
    with col_r:
        sty_data = latest[style_cols].rename(lambda c: c.replace("style_", ""))
        sty_data = sty_data[sty_data.abs() > 1e-6].sort_values(ascending=True)
        colors = ["#3fb950" if v > 0 else "#f85149" for v in sty_data]
        fig_snap_sty = go.Figure(go.Bar(y=sty_data.index, x=sty_data.values, orientation="h", marker_color=colors))
        fig_snap_sty.update_layout(**PLOTLY_LAYOUT, title="Style Tilts", height=350)
        st.plotly_chart(fig_snap_sty, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6: MODEL & SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
def tab_model_signal(start, end):
    ic = load_ic()
    fi = load_fi()
    ga = load_group_attr()
    li = load_li_attr()

    # ── IC ───────────────────────────────────────────────────────────────────
    if ic is not None and not ic.empty:
        ic_f = _trim(ic, start, end, col="date")
        st.markdown("#### Information Coefficient (IC)")
        c1, c2, c3, c4 = st.columns(4)
        ic_mean = ic_f["IC"].mean()
        ic_std = ic_f["IC"].std()
        icir = ic_mean / ic_std if ic_std > 0 else float("nan")
        c1.metric("Mean IC", _f(ic_mean, 4))
        c2.metric("IC Std", _f(ic_std, 4))
        c3.metric("ICIR", _f(icir))
        c4.metric("IC Hit Rate", _pct((ic_f["IC"] > 0).mean(), 0))

        colors_ic = ["#3fb950" if v > 0 else "#f85149" for v in ic_f["IC"]]
        fig_ic = go.Figure(go.Bar(x=ic_f["date"], y=ic_f["IC"], marker_color=colors_ic))
        fig_ic.add_hline(y=ic_mean, line_dash="dash", line_color="#d2a8ff", annotation_text=f"mean={ic_mean:.4f}")
        fig_ic.update_layout(**PLOTLY_LAYOUT, title="Monthly IC")
        st.plotly_chart(fig_ic, use_container_width=True)

        # insight
        if icir > 0.5:
            _insight(f"ICIR {_f(icir)} — 시그널의 안정성이 높음 (ICIR>0.5는 실전 배포 가능 수준)", "good")
        elif icir > 0:
            _insight(f"ICIR {_f(icir)} — 양(+)이나 시그널 노이즈가 높은 편", "warn")

    # ── feature importance ───────────────────────────────────────────────────
    if fi is not None and not fi.empty:
        st.markdown("#### Feature Importance")
        top_n = st.slider("Top N features", 10, 50, 25, key="fi_topn")
        top_fi = fi.head(top_n).sort_values("importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            y=top_fi["feature"], x=top_fi["importance"],
            orientation="h",
            marker_color=top_fi["group"].map({
                "Price": "#79c0ff", "Accounting": "#3fb950",
                "Sellside": "#d2a8ff", "Conditioning": "#f0883e", "Factor": "#f85149",
            }).fillna("#8b949e"),
            text=top_fi["group"], textposition="inside",
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, title=f"Top {top_n} Features", height=max(400, top_n * 22))
        st.plotly_chart(fig_fi, use_container_width=True)

        # group aggregation
        grp = fi.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
        fig_grp = px.pie(grp, values="importance", names="group", title="Feature Group Share",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_grp.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_grp, use_container_width=True)

    # ── group attribution over time ──────────────────────────────────────────
    if ga is not None and not ga.empty:
        st.markdown("#### Group Attribution Over Time")
        fig_ga = go.Figure()
        for col in ga.columns:
            fig_ga.add_trace(go.Scatter(x=ga.index, y=ga[col], stackgroup="one", name=col))
        fig_ga.update_layout(**PLOTLY_LAYOUT, title="Category-level SHAP Attribution")
        st.plotly_chart(fig_ga, use_container_width=True)

    # ── Li et al. decomposition ──────────────────────────────────────────────
    if li is not None and not li.empty:
        st.markdown("#### Linear vs Nonlinear Decomposition (Li et al.)")
        li_f = _trim(li, start, end, col="date")
        fig_li = go.Figure()
        for col, color in [("linear_ratio", "#79c0ff"), ("marginal_nl_ratio", "#d2a8ff"), ("interaction_ratio", "#f0883e")]:
            if col in li_f.columns:
                fig_li.add_trace(go.Bar(x=li_f["date"], y=li_f[col], name=col.replace("_ratio", "").replace("_", " ").title(), marker_color=color))
        fig_li.update_layout(**PLOTLY_LAYOUT, barmode="stack", title="Linear / Marginal NL / Interaction Ratio", yaxis_tickformat=".0%")
        st.plotly_chart(fig_li, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Linear", _pct(li_f["linear_ratio"].mean()))
        c2.metric("Avg Marginal NL", _pct(li_f["marginal_nl_ratio"].mean()))
        c3.metric("Avg Interaction", _pct(li_f["interaction_ratio"].mean()))

        nl_total = li_f["marginal_nl_ratio"].mean() + li_f["interaction_ratio"].mean()
        _insight(
            f"비선형 비율 평균 {_pct(nl_total)} — "
            + ("모델이 비선형 패턴을 활발히 활용 중. GBM 구조 적합." if nl_total > 0.5
               else "선형 효과 비중이 높음. 단순 linear factor model과 차별화 제한적일 수 있음."),
            "good" if nl_total > 0.5 else "warn",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7: REGIME & EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def tab_regime(start, end):
    regime = load_regime()
    expl = load_ow_expl()

    if regime is not None and not regime.empty:
        st.markdown("#### Market Regime Timeline")
        # color-coded regime indicator
        regime_colors = {
            "Bullish": "#3fb950", "Bearish": "#f85149", "Sideways": "#d29922",
        }
        fig_reg = go.Figure()
        for _, row in regime.iterrows():
            direction = row["market_direction"]
            color = regime_colors.get(direction, "#8b949e")
            fig_reg.add_trace(go.Bar(
                x=[row["year_month"]], y=[row["total_active_share"]],
                marker_color=color, name=direction,
                showlegend=False,
                hovertext=f"{direction} / {row['volatility_regime']}<br>OW: {row['n_ow_stocks']} | UW: {row['n_uw_stocks']}",
            ))
        fig_reg.update_layout(**PLOTLY_LAYOUT, title="Active Share by Regime", yaxis_tickformat=".1%",
                              xaxis_title="Month", yaxis_title="Active Share")
        st.plotly_chart(fig_reg, use_container_width=True)

        # regime table
        display_cols = ["year_month", "market_direction", "volatility_regime", "sector_rotation",
                        "n_ow_stocks", "n_uw_stocks", "total_active_share"]
        display_cols = [c for c in display_cols if c in regime.columns]
        st.dataframe(regime[display_cols], use_container_width=True)

    if expl is not None and not expl.empty:
        st.markdown("#### Position Explanations")
        months = expl["year_month"].tolist()
        month = st.selectbox("Month", months, index=len(months) - 1)
        row = expl[expl["year_month"] == month].iloc[0]

        # regime header card
        st.markdown(f"""
        > **{row['regime_label']}** — {row['regime_reason']}
        >
        > Dominant categories: {row['dominant_category_effects']}
        """)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Sector tilt:** {row['sector_tilt']}")
            st.markdown(f"**Style tilt:** {row['style_tilt']}")
        with col_r:
            st.markdown(f"**OW:** {row['n_ow_stocks']} stocks | **UW:** {row['n_uw_stocks']} stocks")
            st.markdown(f"**Rebal date:** {row['rebal_date']}")

        # OW/UW details
        ow_tab, uw_tab = st.tabs(["Overweight Details", "Underweight Details"])
        with ow_tab:
            for item in str(row.get("top_ow_details", "")).split(" | "):
                if item.strip():
                    st.markdown(f"- {item.strip()}")
        with uw_tab:
            for item in str(row.get("top_uw_details", "")).split(" | "):
                if item.strip():
                    st.markdown(f"- {item.strip()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 8: MODEL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
def tab_model_structure(start, end):
    ms = load_model_struct()
    if ms is None or ms.empty:
        st.error("No model structure data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Retrains", str(len(ms)))
    c2.metric("Avg Trees", _f(ms["n_trees"].mean(), 0))
    c3.metric("Avg Features Used", _f(ms["n_unique_features_used"].mean(), 0))
    c4.metric("Avg Tree Depth", _f(ms["avg_tree_depth"].mean(), 1))

    # depth / trees over time
    if "retrain_date" in ms.columns and len(ms) > 1:
        fig_ms = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ms.add_trace(go.Bar(
            x=ms["retrain_date"], y=ms["avg_tree_depth"],
            name="Avg Depth", marker_color="#79c0ff",
        ), secondary_y=False)
        fig_ms.add_trace(go.Scatter(
            x=ms["retrain_date"], y=ms["n_unique_features_used"],
            name="Features Used", line=dict(color="#d2a8ff", width=2),
        ), secondary_y=True)
        fig_ms.update_layout(**PLOTLY_LAYOUT, title="Model Complexity Over Retrains")
        fig_ms.update_yaxes(title_text="Avg Depth", secondary_y=False)
        fig_ms.update_yaxes(title_text="Features Used", secondary_y=True)
        st.plotly_chart(fig_ms, use_container_width=True)

    # top split features
    st.markdown("#### Top Split Features per Retrain")
    if "top_split_features" in ms.columns:
        for _, row in ms.iterrows():
            date_str = pd.Timestamp(row["retrain_date"]).strftime("%Y-%m-%d") if pd.notna(row["retrain_date"]) else "?"
            with st.expander(f"{date_str} — depth {row['avg_tree_depth']:.1f}, {row['n_trees']} trees"):
                st.code(row["top_split_features"], language=None)

    st.markdown("#### Raw Data")
    st.dataframe(ms, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 9: RISK MONITOR (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
def tab_risk(start, end):
    """Added risk monitoring page — not in the original."""
    perf = load_perf()
    if perf is None or perf.empty:
        return
    perf = _trim(perf, start, end)

    st.markdown("#### Risk Dashboard")

    port = perf["fund_daily_return"].fillna(0)
    bm = perf["bm_daily_return"].fillna(0)
    active = perf["active_daily_return"].fillna(0)

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    var_95 = float(np.percentile(port, 5))
    var_99 = float(np.percentile(port, 1))
    cvar_95 = float(port[port <= var_95].mean()) if len(port[port <= var_95]) > 0 else var_95
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR (95%)", _pct(var_95))
    c2.metric("VaR (99%)", _pct(var_99))
    c3.metric("CVaR (95%)", _pct(cvar_95))
    c4.metric("Worst Day", _pct(float(port.min())))

    # ── return distribution ──────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=port, nbinsx=60, name="Fund", marker_color="#58a6ff", opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=bm, nbinsx=60, name="Benchmark", marker_color="#8b949e", opacity=0.5))
        fig_dist.add_vline(x=var_95, line_dash="dash", line_color="#f85149", annotation_text="VaR 95%")
        fig_dist.update_layout(**PLOTLY_LAYOUT, barmode="overlay", title="Daily Return Distribution", xaxis_tickformat=".1%")
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_r:
        fig_qq = go.Figure()
        sorted_ret = np.sort(port.values)
        n = len(sorted_ret)
        theoretical = np.random.normal(port.mean(), port.std(), n)
        theoretical.sort()
        fig_qq.add_trace(go.Scatter(x=theoretical, y=sorted_ret, mode="markers", marker=dict(size=3, color="#58a6ff"), name="Returns"))
        min_v, max_v = min(theoretical.min(), sorted_ret.min()), max(theoretical.max(), sorted_ret.max())
        fig_qq.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#f85149", dash="dash"), name="Normal"))
        fig_qq.update_layout(**PLOTLY_LAYOUT, title="Q-Q Plot vs Normal")
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── rolling beta ─────────────────────────────────────────────────────────
    st.markdown("#### Rolling Market Beta (126-day)")
    window_beta = 126
    cov_rolling = port.rolling(window_beta).cov(bm)
    var_rolling = bm.rolling(window_beta).var()
    beta_rolling = cov_rolling / var_rolling
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(x=beta_rolling.index, y=beta_rolling, line=dict(color="#d2a8ff", width=2), name="Beta"))
    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#484f58", annotation_text="β=1")
    fig_beta.update_layout(**PLOTLY_LAYOUT, title="Rolling 126-day Beta to Benchmark")
    st.plotly_chart(fig_beta, use_container_width=True)

    # ── tail analysis ────────────────────────────────────────────────────────
    st.markdown("#### Tail Analysis")
    worst_days = port.nsmallest(10)
    best_days = port.nlargest(10)
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("**10 Worst Days**")
        wd = pd.DataFrame({
            "Date": worst_days.index.strftime("%Y-%m-%d"),
            "Fund": worst_days.values,
            "BM": bm.loc[worst_days.index].values,
            "Active": active.loc[worst_days.index].values,
        })
        st.dataframe(wd.style.format({"Fund": "{:.2%}", "BM": "{:.2%}", "Active": "{:+.2%}"}), use_container_width=True)
    with col_r2:
        st.markdown("**10 Best Days**")
        bd = pd.DataFrame({
            "Date": best_days.index.strftime("%Y-%m-%d"),
            "Fund": best_days.values,
            "BM": bm.loc[best_days.index].values,
            "Active": active.loc[best_days.index].values,
        })
        st.dataframe(bd.style.format({"Fund": "{:.2%}", "BM": "{:.2%}", "Active": "{:+.2%}"}), use_container_width=True)

    # insights
    skew = float(port.skew())
    kurt = float(port.kurtosis())
    _insight(f"Skewness: {skew:.3f} | Kurtosis: {kurt:.2f} — " +
             ("좌측 꼬리 위험 주의 (negative skew)" if skew < -0.5 else
              "꼬리 분포 정상 범위" if abs(skew) <= 0.5 else
              "우측 꼬리 수익 편향 (positive skew)"),
             "warn" if skew < -0.5 else "good")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    start, end = sidebar()

    if not (CSV_DIR / "daily_performance.csv").exists():
        st.error("No CSV data. Run: `python run_pipeline.py --config config.yaml --stage full`")
        return

    tabs = st.tabs([
        "📊 Overview",
        "📈 Returns",
        "💼 Portfolio",
        "🔍 Stock Dive",
        "🏭 Sector & Style",
        "🤖 Model & Signal",
        "🌦️ Regime",
        "⚙️ Model Structure",
        "🛡️ Risk",
    ])

    with tabs[0]:
        tab_overview(start, end)
    with tabs[1]:
        tab_returns(start, end)
    with tabs[2]:
        tab_portfolio(start, end)
    with tabs[3]:
        tab_stock_dive(start, end)
    with tabs[4]:
        tab_sector_style(start, end)
    with tabs[5]:
        tab_model_signal(start, end)
    with tabs[6]:
        tab_regime(start, end)
    with tabs[7]:
        tab_model_structure(start, end)
    with tabs[8]:
        tab_risk(start, end)


if __name__ == "__main__":
    main()
