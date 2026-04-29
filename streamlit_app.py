from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = APP_ROOT / "outputs"
TRADING_DAYS = 252

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from src.metadata import TICKER_META
except Exception:
    TICKER_META = {}


st.set_page_config(
    page_title="AI Signal Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    div[data-testid="stDataFrame"] {font-size: 0.85rem;}
    @media (max-width: 720px) {
        .block-container {padding-left: 0.75rem; padding-right: 0.75rem;}
        div[data-testid="stMetricValue"] {font-size: 1.05rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def _compound_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _annualized_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    total = (1.0 + returns).prod()
    return float(total ** (TRADING_DAYS / len(returns)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _sharpe(returns: pd.Series) -> float:
    ann_vol = _annualized_vol(returns)
    if pd.isna(ann_vol) or ann_vol <= 0:
        return np.nan
    return float(_annualized_return(returns) / ann_vol)


def _max_drawdown(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    curve = (1.0 + returns).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def _display_table(
    df: pd.DataFrame,
    pct_cols: Iterable[str] = (),
    num_cols: Iterable[str] = (),
    height: Optional[int] = None,
) -> None:
    view = df.copy()
    for col in pct_cols:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(_format_pct)
    for col in num_cols:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(_format_num)
    st.dataframe(view, width="stretch", height=height)


def _read_date_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0 or df[col].isna().all():
            df[col] = converted
    return df


def _file_snapshot(run_dir: Path) -> float:
    paths = list((run_dir / "csv").glob("*")) + list(run_dir.glob("*.pkl"))
    if not paths:
        return 0.0
    return max(p.stat().st_mtime for p in paths if p.exists())


@st.cache_data(ttl=15, show_spinner=False)
def discover_runs() -> List[str]:
    runs: List[Path] = []
    if (OUTPUTS_ROOT / "csv" / "daily_performance.csv").exists():
        runs.append(OUTPUTS_ROOT)
    if OUTPUTS_ROOT.exists():
        runs.extend(
            p
            for p in OUTPUTS_ROOT.iterdir()
            if p.is_dir() and (p / "csv" / "daily_performance.csv").exists()
        )
    runs = sorted(set(runs), key=lambda p: _file_snapshot(p), reverse=True)
    return [str(p) for p in runs]


@st.cache_data(show_spinner=False)
def load_run_data(run_dir_str: str, snapshot: float) -> Dict[str, object]:
    run_dir = Path(run_dir_str)
    csv_dir = run_dir / "csv"
    data: Dict[str, object] = {"run_dir": run_dir, "csv_dir": csv_dir}

    for key, name in {
        "performance": "daily_performance.csv",
        "portfolio_weights": "portfolio_weights.csv",
        "benchmark_weights": "benchmark_weights.csv",
        "desired_weights": "desired_weights.csv",
        "daily_weights": "daily_weights.csv",
        "execution": "execution_diagnostics.csv",
        "ic": "ic_series.csv",
    }.items():
        path = csv_dir / name
        data[key] = _read_date_csv(path) if path.exists() else pd.DataFrame()

    for key, name in {
        "feature_diagnostics": "feature_diagnostics.csv",
        "feature_group_diagnostics": "feature_group_diagnostics.csv",
        "ow_feature_scores": "ow_feature_scores.csv",
    }.items():
        path = csv_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            data[key] = df
        else:
            data[key] = pd.DataFrame()

    selected_path = csv_dir / "selected_features.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        data["selected_features"] = selected.get("feature", pd.Series(dtype=str)).dropna().astype(str).tolist()
    else:
        data["selected_features"] = []

    groups_path = csv_dir / "compact_feature_groups.json"
    if groups_path.exists():
        data["feature_groups"] = json.loads(groups_path.read_text(encoding="utf-8"))
    else:
        data["feature_groups"] = {}

    data["snapshot"] = snapshot
    return data


@st.cache_resource(show_spinner=False)
def load_backtest_result(pkl_path_str: str, snapshot: float) -> Tuple[object, Optional[str]]:
    path = Path(pkl_path_str)
    if not path.exists():
        return None, None
    try:
        with path.open("rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, str(exc)


def _schedule_refresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    seconds = max(int(seconds), 10)
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        height=0,
    )


def overall_metrics(perf: pd.DataFrame) -> Dict[str, float]:
    fund = perf["fund_daily_return"]
    bm = perf["bm_daily_return"]
    active = fund - bm
    active_vol = _annualized_vol(active)
    return {
        "fund_return": _compound_return(fund),
        "bm_return": _compound_return(bm),
        "active_return": _compound_return(fund) - _compound_return(bm),
        "fund_vol": _annualized_vol(fund),
        "fund_sharpe": _sharpe(fund),
        "tracking_error": active_vol,
        "information_ratio": _annualized_return(active) / active_vol if pd.notna(active_vol) and active_vol > 0 else np.nan,
        "max_drawdown": _max_drawdown(fund),
        "latest_turnover": perf["turnover"].iloc[-1] if "turnover" in perf.columns else np.nan,
    }


def build_period_table(perf: pd.DataFrame) -> pd.DataFrame:
    latest = perf.index.max()
    windows = [
        ("1D", perf.tail(1)),
        ("1W", perf.tail(5)),
        ("1M", perf.tail(21)),
        ("3M", perf.tail(63)),
        ("6M", perf.tail(126)),
        ("MTD", perf[perf.index.to_period("M") == latest.to_period("M")]),
        ("YTD", perf[perf.index.year == latest.year]),
    ]
    rows = []
    for label, window in windows:
        if window.empty:
            continue
        fund = window["fund_daily_return"]
        bm = window["bm_daily_return"]
        active = fund - bm
        include_risk = label == "6M" or (label == "YTD" and len(window) >= 126)
        active_vol = _annualized_vol(active) if include_risk else np.nan
        rows.append(
            {
                "기간": label,
                "시작일": window.index.min().strftime("%Y-%m-%d"),
                "종료일": window.index.max().strftime("%Y-%m-%d"),
                "포트폴리오 수익률": _compound_return(fund),
                "BM 수익률": _compound_return(bm),
                "초과 수익률": _compound_return(fund) - _compound_return(bm),
                "변동성": _annualized_vol(fund) if include_risk else np.nan,
                "샤프": _sharpe(fund) if include_risk else np.nan,
                "TE": active_vol,
                "IR": _annualized_return(active) / active_vol if pd.notna(active_vol) and active_vol > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_annual_table(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, window in perf.groupby(perf.index.year):
        fund = window["fund_daily_return"]
        bm = window["bm_daily_return"]
        active = fund - bm
        te = _annualized_vol(active)
        rows.append(
            {
                "연도": int(year),
                "포트폴리오 수익률": _compound_return(fund),
                "BM 수익률": _compound_return(bm),
                "초과 수익률": _compound_return(fund) - _compound_return(bm),
                "변동성": _annualized_vol(fund),
                "샤프": _sharpe(fund),
                "TE": te,
                "IR": _annualized_return(active) / te if pd.notna(te) and te > 0 else np.nan,
                "MDD": _max_drawdown(fund),
                "리밸런싱 수": int(window["is_rebalance"].sum()) if "is_rebalance" in window else 0,
                "턴오버 합계": float(window["turnover"].sum()) if "turnover" in window else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("연도", ascending=False)


def cumulative_figure(perf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fund_curve = perf["fund_cumulative"] if "fund_cumulative" in perf else (1 + perf["fund_daily_return"]).cumprod()
    bm_curve = perf["bm_cumulative"] if "bm_cumulative" in perf else (1 + perf["bm_daily_return"]).cumprod()
    fig.add_trace(go.Scatter(x=perf.index, y=fund_curve - 1.0, name="Portfolio", line=dict(width=2.4)))
    fig.add_trace(go.Scatter(x=perf.index, y=bm_curve - 1.0, name="Benchmark", line=dict(width=2.0)))
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def drawdown_figure(perf: pd.DataFrame) -> go.Figure:
    fund = (1.0 + perf["fund_daily_return"]).cumprod()
    bm = (1.0 + perf["bm_daily_return"]).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=perf.index, y=fund / fund.cummax() - 1.0, name="Portfolio"))
    fig.add_trace(go.Scatter(x=perf.index, y=bm / bm.cummax() - 1.0, name="Benchmark"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=True)
    fig.update_yaxes(tickformat=".0%")
    return fig


def _row_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    date = pd.Timestamp(date)
    if date in df.index:
        row = df.loc[date]
    else:
        eligible = df.index[df.index <= date]
        row = df.loc[eligible.max()] if len(eligible) else df.iloc[0]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return pd.to_numeric(row, errors="coerce")


def active_weight_table(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    desired: pd.DataFrame,
    date: pd.Timestamp,
    threshold: float,
) -> pd.DataFrame:
    w = _row_on_or_before(portfolio, date)
    bm = _row_on_or_before(benchmark, date)
    if bm.empty:
        bm = pd.Series(1.0 / len(w), index=w.index)
    bm = bm.reindex(w.index).fillna(0.0)
    desired_row = _row_on_or_before(desired, date).reindex(w.index) if not desired.empty else pd.Series(index=w.index, dtype=float)
    active = w - bm
    rows = []
    for ticker in w.index:
        aw = active.get(ticker, np.nan)
        if pd.isna(aw) or abs(aw) < threshold:
            continue
        meta = TICKER_META.get(ticker, {})
        rows.append(
            {
                "Ticker": ticker,
                "Side": "OW" if aw > 0 else "UW",
                "Portfolio Weight": w.get(ticker, np.nan),
                "BM Weight": bm.get(ticker, np.nan),
                "Active Weight": aw,
                "Desired Weight": desired_row.get(ticker, np.nan),
                "Sector": meta.get("sector", ""),
                "Style": meta.get("style", ""),
                "Sub": meta.get("sub", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("Active Weight", ascending=False)


def active_weight_figure(active_df: pd.DataFrame, top_n: int) -> go.Figure:
    ow = active_df[active_df["Side"] == "OW"].head(top_n)
    uw = active_df[active_df["Side"] == "UW"].tail(top_n)
    plot_df = pd.concat([ow, uw], axis=0).sort_values("Active Weight")
    fig = go.Figure()
    colors = np.where(plot_df["Active Weight"] >= 0, "#1f77b4", "#d62728")
    fig.add_trace(
        go.Bar(
            x=plot_df["Active Weight"],
            y=plot_df["Ticker"],
            orientation="h",
            marker_color=colors,
            text=[_format_pct(v) for v in plot_df["Active Weight"]],
            textposition="auto",
        )
    )
    fig.update_layout(height=max(320, 24 * len(plot_df)), margin=dict(l=10, r=10, t=20, b=10))
    fig.update_xaxes(tickformat=".1%")
    return fig


def feature_group_lookup(groups: Dict[str, List[str]]) -> Dict[str, str]:
    lookup = {}
    for group, features in groups.items():
        for feature in features:
            lookup[feature] = group
    return lookup


def model_feature_importance(result: object, group_lookup: Dict[str, str]) -> pd.DataFrame:
    if result is None or not getattr(result, "models", None):
        return pd.DataFrame()
    rows = []
    default_features = list(getattr(result, "feature_names", []) or [])
    for model_date, model in getattr(result, "models", {}).items():
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            continue
        active_features = list(getattr(model, "_active_features", default_features) or default_features)
        if len(importances) != len(active_features):
            continue
        total = float(np.sum(importances))
        if total <= 0:
            total = 1.0
        for feature, importance in zip(active_features, importances):
            rows.append(
                {
                    "date": pd.Timestamp(model_date),
                    "feature": feature,
                    "group": group_lookup.get(feature, "Other"),
                    "importance": float(importance) / total,
                    "raw_importance": float(importance),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby(["feature", "group"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            latest_importance=("importance", "last"),
            model_count=("importance", "count"),
        )
        .sort_values("mean_importance", ascending=False)
    )


def feature_target_payoff(result: object, features: List[str], group_lookup: Dict[str, str]) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    panel = getattr(result, "panel", None)
    targets = getattr(result, "targets", None)
    if panel is None or targets is None or panel.empty or targets.empty:
        return pd.DataFrame()
    if not isinstance(panel.index, pd.MultiIndex):
        return pd.DataFrame()

    date_level = "date" if "date" in panel.index.names else 0
    panel_dates = pd.Index(panel.index.get_level_values(date_level).unique())
    dates = sorted(pd.Index(targets.index).intersection(panel_dates))
    rows = []

    for feature in features:
        if feature not in panel.columns:
            continue
        ic_values = []
        payoff_values = []
        for date in dates:
            try:
                x = panel.xs(date, level=date_level)[feature]
            except KeyError:
                continue
            y = targets.loc[date]
            aligned = pd.concat([x.rename("feature"), y.rename("target")], axis=1).dropna()
            if len(aligned) < 8:
                continue
            ranked_ic = aligned["feature"].rank().corr(aligned["target"].rank())
            std = aligned["feature"].std(ddof=0)
            z = (aligned["feature"] - aligned["feature"].mean()) / std if std and std > 0 else aligned["feature"] * 0.0
            payoff = float((z * aligned["target"]).mean())
            if pd.notna(ranked_ic):
                ic_values.append(float(ranked_ic))
            if pd.notna(payoff):
                payoff_values.append(payoff)
        if not ic_values and not payoff_values:
            continue
        rows.append(
            {
                "feature": feature,
                "group": group_lookup.get(feature, "Other"),
                "mean_ic": float(np.mean(ic_values)) if ic_values else np.nan,
                "ic_hit_rate": float(np.mean(np.array(ic_values) > 0)) if ic_values else np.nan,
                "target_payoff": float(np.mean(payoff_values)) if payoff_values else np.nan,
                "sample_count": len(ic_values),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def feature_diagnostics(result: object, selected_features: List[str], groups: Dict[str, List[str]]) -> pd.DataFrame:
    group_lookup = feature_group_lookup(groups)
    importance = model_feature_importance(result, group_lookup)
    payoff = feature_target_payoff(result, selected_features, group_lookup)

    base = pd.DataFrame({"feature": selected_features})
    base["group"] = base["feature"].map(group_lookup).fillna("Other")
    if not importance.empty:
        base = base.merge(importance.drop(columns=["group"], errors="ignore"), on="feature", how="left")
    if not payoff.empty:
        base = base.merge(payoff.drop(columns=["group"], errors="ignore"), on="feature", how="left")
    for col in ["mean_importance", "latest_importance", "mean_ic", "ic_hit_rate", "target_payoff"]:
        if col not in base:
            base[col] = np.nan
    base["helped_returns"] = np.where(
        (base["mean_ic"].fillna(0.0) > 0.0) & (base["target_payoff"].fillna(0.0) > 0.0),
        "Yes",
        "No",
    )
    return base.sort_values(["mean_importance", "mean_ic"], ascending=[False, False])


def group_diagnostics(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    return (
        feature_df.groupby("group", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_importance=("mean_importance", "sum"),
            mean_ic=("mean_ic", "mean"),
            target_payoff=("target_payoff", "mean"),
            helpful_count=("helped_returns", lambda x: int((x == "Yes").sum())),
        )
        .sort_values("mean_importance", ascending=False)
    )


def ow_feature_score_table(
    result: object,
    active_df: pd.DataFrame,
    date: pd.Timestamp,
    selected_features: List[str],
    groups: Dict[str, List[str]],
    top_stocks: int,
    top_features: int,
) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    panel = getattr(result, "panel", None)
    if panel is None or panel.empty or not isinstance(panel.index, pd.MultiIndex):
        return pd.DataFrame()

    date_level = "date" if "date" in panel.index.names else 0
    group_lookup = feature_group_lookup(groups)
    predictions = getattr(result, "predictions", None)

    try:
        date_panel = panel.xs(pd.Timestamp(date), level=date_level)
    except KeyError:
        panel_dates = pd.Index(panel.index.get_level_values(date_level).unique())
        eligible = panel_dates[panel_dates <= pd.Timestamp(date)]
        if len(eligible) == 0:
            return pd.DataFrame()
        date = eligible.max()
        date_panel = panel.xs(date, level=date_level)

    ow_tickers = active_df[active_df["Side"] == "OW"].head(top_stocks)["Ticker"].tolist()
    rows = []
    available_features = [f for f in selected_features if f in date_panel.columns]
    for ticker in ow_tickers:
        if ticker not in date_panel.index:
            continue
        scores = pd.to_numeric(date_panel.loc[ticker, available_features], errors="coerce").dropna()
        scores = scores.sort_values(ascending=False).head(top_features)
        active_weight = active_df.loc[active_df["Ticker"] == ticker, "Active Weight"].iloc[0]
        pred_score = np.nan
        if predictions is not None and pd.Timestamp(date) in predictions.index and ticker in predictions.columns:
            pred_score = predictions.loc[pd.Timestamp(date), ticker]
        for rank, (feature, score) in enumerate(scores.items(), start=1):
            rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Active Weight": active_weight,
                    "Prediction Score": pred_score,
                    "Rank": rank,
                    "Feature": feature,
                    "Group": group_lookup.get(feature, "Other"),
                    "Feature Score": float(score),
                }
            )
    return pd.DataFrame(rows)


def compact_top_feature_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in score_df.groupby("Ticker", sort=False):
        top_features = " | ".join(
            f"{row.Feature}({_format_num(row['Feature Score'])})"
            for _, row in grp.sort_values("Rank").iterrows()
        )
        first = grp.iloc[0]
        rows.append(
            {
                "Ticker": ticker,
                "Active Weight": first["Active Weight"],
                "Prediction Score": first["Prediction Score"],
                "Top Feature Scores": top_features,
            }
        )
    return pd.DataFrame(rows)


def precomputed_ow_feature_score_table(
    score_df: pd.DataFrame,
    active_df: pd.DataFrame,
    date: pd.Timestamp,
    top_stocks: int,
    top_features: int,
) -> pd.DataFrame:
    if score_df.empty or "date" not in score_df.columns:
        return pd.DataFrame()
    date = pd.Timestamp(date)
    ow_tickers = active_df[active_df["Side"] == "OW"].head(top_stocks)["Ticker"].tolist()
    view = score_df[
        (score_df["date"] == date)
        & (score_df["Ticker"].isin(ow_tickers))
        & (pd.to_numeric(score_df["Rank"], errors="coerce") <= top_features)
    ].copy()
    if view.empty:
        return view
    view["Ticker"] = pd.Categorical(view["Ticker"], categories=ow_tickers, ordered=True)
    return view.sort_values(["Ticker", "Rank"]).reset_index(drop=True)


run_options = discover_runs()
if not run_options:
    st.error("outputs 폴더에서 daily_performance.csv를 찾지 못했습니다.")
    st.stop()

run_labels = {Path(p).name if Path(p) != OUTPUTS_ROOT else "outputs": p for p in run_options}
default_label = "final_v3_current" if "final_v3_current" in run_labels else next(iter(run_labels))

with st.sidebar:
    st.header("설정")
    run_label = st.selectbox("결과 폴더", list(run_labels.keys()), index=list(run_labels.keys()).index(default_label))
    auto_refresh = st.toggle("자동 새로고침", value=True)
    refresh_seconds = st.number_input("새로고침 초", min_value=10, max_value=3600, value=60, step=10)
    active_threshold = st.number_input("Active weight 기준", min_value=0.0, max_value=0.10, value=0.002, step=0.001, format="%.3f")
    top_n = st.slider("표시 종목 수", min_value=5, max_value=30, value=10, step=1)
    top_feature_count = st.slider("종목별 feature 수", min_value=3, max_value=12, value=5, step=1)

_schedule_refresh(auto_refresh, int(refresh_seconds))

run_dir = Path(run_labels[run_label])
snapshot = _file_snapshot(run_dir)
data = load_run_data(str(run_dir), snapshot)
perf = data["performance"]
portfolio = data["portfolio_weights"]
benchmark = data["benchmark_weights"]
desired = data["desired_weights"]
selected_features = data["selected_features"]
feature_groups = data["feature_groups"]
precomputed_feature_df = data["feature_diagnostics"]
precomputed_group_df = data["feature_group_diagnostics"]
precomputed_ow_scores = data["ow_feature_scores"]

if perf.empty:
    st.error("daily_performance.csv가 비어 있습니다.")
    st.stop()

pkl_path = run_dir / "backtest_result_redesign.pkl"
result, pkl_error = load_backtest_result(str(pkl_path), snapshot)

st.title("AI Signal Dashboard")
st.caption(f"{run_dir} · {perf.index.max().strftime('%Y-%m-%d')}")

metrics = overall_metrics(perf)
metric_cols = st.columns(4)
metric_cols[0].metric("누적 수익률", _format_pct(metrics["fund_return"]), _format_pct(metrics["active_return"]))
metric_cols[1].metric("BM 수익률", _format_pct(metrics["bm_return"]))
metric_cols[2].metric("변동성 / 샤프", f"{_format_pct(metrics['fund_vol'])} / {_format_num(metrics['fund_sharpe'])}")
metric_cols[3].metric("MDD", _format_pct(metrics["max_drawdown"]))

if pkl_error:
    st.warning(f"pkl 분석 로드 실패: {pkl_error}")

tabs = st.tabs(["성과", "리밸런싱 OW/UW", "Feature 진단", "OW 점수"])

with tabs[0]:
    st.subheader("성과")
    st.plotly_chart(cumulative_figure(perf), width="stretch")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 기간별 성과")
        period_df = build_period_table(perf)
        _display_table(
            period_df,
            pct_cols=["포트폴리오 수익률", "BM 수익률", "초과 수익률", "변동성", "TE"],
            num_cols=["샤프", "IR"],
            height=305,
        )
    with col_b:
        st.markdown("#### 연도별 성과")
        annual_df = build_annual_table(perf)
        _display_table(
            annual_df,
            pct_cols=["포트폴리오 수익률", "BM 수익률", "초과 수익률", "변동성", "TE", "MDD", "턴오버 합계"],
            num_cols=["샤프", "IR"],
            height=305,
        )

    st.markdown("#### Drawdown")
    st.plotly_chart(drawdown_figure(perf), width="stretch")

with tabs[1]:
    st.subheader("리밸런싱 OW/UW")
    rebal_dates = list(portfolio.index.sort_values(ascending=False))
    if not rebal_dates:
        st.info("portfolio_weights.csv에서 리밸런싱 날짜를 찾지 못했습니다.")
    else:
        date_label = st.selectbox("리밸런싱 날짜", [d.strftime("%Y-%m-%d") for d in rebal_dates])
        selected_date = pd.Timestamp(date_label)
        active_df = active_weight_table(portfolio, benchmark, desired, selected_date, active_threshold)
        if active_df.empty:
            st.info("선택한 기준을 넘는 OW/UW 종목이 없습니다.")
        else:
            st.plotly_chart(active_weight_figure(active_df, top_n), width="stretch")
            col_ow, col_uw = st.columns(2)
            with col_ow:
                st.markdown("#### OW")
                _display_table(
                    active_df[active_df["Side"] == "OW"].head(top_n),
                    pct_cols=["Portfolio Weight", "BM Weight", "Active Weight", "Desired Weight"],
                    height=360,
                )
            with col_uw:
                st.markdown("#### UW")
                _display_table(
                    active_df[active_df["Side"] == "UW"].tail(top_n).sort_values("Active Weight"),
                    pct_cols=["Portfolio Weight", "BM Weight", "Active Weight", "Desired Weight"],
                    height=360,
                )
            st.markdown("#### 전체")
            _display_table(
                active_df,
                pct_cols=["Portfolio Weight", "BM Weight", "Active Weight", "Desired Weight"],
                height=430,
            )

with tabs[2]:
    st.subheader("Feature 진단")
    feature_df = precomputed_feature_df.copy()
    if feature_df.empty:
        feature_df = feature_diagnostics(result, selected_features, feature_groups)
    if feature_df.empty:
        selected_view = pd.DataFrame({"feature": selected_features})
        selected_view["group"] = selected_view["feature"].map(feature_group_lookup(feature_groups)).fillna("Other")
        st.dataframe(selected_view, width="stretch")
    else:
        group_df = precomputed_group_df.copy()
        if group_df.empty:
            group_df = group_diagnostics(feature_df)
        col_g, col_f = st.columns([1, 2])
        with col_g:
            st.markdown("#### 그룹")
            _display_table(
                group_df,
                pct_cols=["mean_importance", "mean_ic", "target_payoff"],
                num_cols=["feature_count", "helpful_count"],
                height=300,
            )
        with col_f:
            top_importance = feature_df.head(20).sort_values("mean_importance")
            fig = go.Figure(
                go.Bar(
                    x=top_importance["mean_importance"],
                    y=top_importance["feature"],
                    orientation="h",
                    text=[_format_pct(v) for v in top_importance["mean_importance"]],
                    textposition="auto",
                )
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            fig.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig, width="stretch")

        view = feature_df.rename(
            columns={
                "feature": "Feature",
                "group": "Group",
                "mean_importance": "Mean Importance",
                "latest_importance": "Latest Importance",
                "mean_ic": "Mean IC",
                "ic_hit_rate": "IC Hit Rate",
                "target_payoff": "Target Payoff",
                "sample_count": "Samples",
                "helped_returns": "Helped Returns",
            }
        )
        _display_table(
            view,
            pct_cols=["Mean Importance", "Latest Importance", "Mean IC", "IC Hit Rate", "Target Payoff"],
            num_cols=["Samples"],
            height=520,
        )

with tabs[3]:
    st.subheader("OW 점수")
    rebal_dates = list(portfolio.index.sort_values(ascending=False))
    if not rebal_dates:
        st.info("portfolio_weights.csv에서 리밸런싱 날짜를 찾지 못했습니다.")
    else:
        date_label = st.selectbox(
            "OW 점수 날짜",
            [d.strftime("%Y-%m-%d") for d in rebal_dates],
            key="ow_score_date",
        )
        selected_date = pd.Timestamp(date_label)
        active_df = active_weight_table(portfolio, benchmark, desired, selected_date, active_threshold)
        score_df = ow_feature_score_table(
            result=result,
            active_df=active_df,
            date=selected_date,
            selected_features=selected_features,
            groups=feature_groups,
            top_stocks=top_n,
            top_features=top_feature_count,
        )
        if score_df.empty:
            score_df = precomputed_ow_feature_score_table(
                score_df=precomputed_ow_scores,
                active_df=active_df,
                date=selected_date,
                top_stocks=top_n,
                top_features=top_feature_count,
            )
        if score_df.empty:
            st.info("OW feature score 데이터를 찾지 못했습니다.")
        else:
            st.markdown("#### 종목별 요약")
            _display_table(
                compact_top_feature_summary(score_df),
                pct_cols=["Active Weight"],
                num_cols=["Prediction Score"],
                height=320,
            )
            st.markdown("#### 상세")
            _display_table(
                score_df,
                pct_cols=["Active Weight"],
                num_cols=["Prediction Score", "Feature Score"],
                height=520,
            )
