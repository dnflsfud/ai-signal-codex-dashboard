"""LightGBM-focused reporting outputs and dashboard inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import LOGGER, save_csv

PRIMARY_PORTFOLIO_TYPE = "decile"

STYLE_GROUPS = {
    "momentum": [("ret_21d", 1.0), ("ret_63d", 1.0), ("mom_63_ex_21", 1.0), ("mom_126_ex_21", 1.0), ("sector_rel_ret_21d", 1.0), ("sector_rel_ret_63d", 1.0)],
    "value": [("fcf_yield", 1.0), ("target_price_upside", 1.0), ("pe_level", -1.0), ("px_bps_level", -1.0), ("ev_to_ebitda_level", -1.0)],
    "quality": [("roe_level", 1.0), ("oper_margin_level", 1.0), ("gross_margin_level", 1.0)],
    "growth": [("eps_delta_63d", 1.0), ("eps_delta_252d", 1.0), ("sales_delta_63d", 1.0), ("sales_delta_252d", 1.0)],
    "sentiment": [("news_sent_mean_21d", 1.0), ("rec_cons_delta_21d", 1.0), ("eps_revision_delta_21d", 1.0), ("sales_revision_delta_21d", 1.0), ("sent_mom_mean_21d", 1.0)],
    "size": [("log_mktcap", 1.0)],
}

FEATURE_LABELS = {
    "ret_21d": "21-day return strength",
    "ret_63d": "63-day return strength",
    "mom_63_ex_21": "medium-term momentum",
    "mom_126_ex_21": "long-term momentum",
    "eps_delta_63d": "EPS 3M growth",
    "eps_delta_252d": "EPS 1Y growth",
    "sales_delta_63d": "Sales 3M growth",
    "sales_delta_252d": "Sales 1Y growth",
    "rec_cons_delta_21d": "analyst recommendation improvement",
    "eps_revision_delta_21d": "EPS revision momentum",
    "sales_revision_delta_21d": "Sales revision momentum",
    "target_price_upside": "target price upside",
    "news_sent_mean_21d": "news sentiment trend",
    "sector_rel_ret_21d": "sector-relative 21D return",
    "sector_rel_ret_63d": "sector-relative 63D return",
    "roe_level": "high ROE",
    "oper_margin_level": "high operating margin",
    "fcf_yield": "FCF yield",
    "log_mktcap": "large-cap support",
}

REGIME_FEATURES = [
    "mxwd_ret_21d",
    "vix_level_z_252d",
    "growth_minus_value_ret_21d",
    "quality_minus_hibeta_ret_21d",
    "sector_eq_ret_21d",
    "sector_dispersion_21d",
    "infotech_minus_health_ret_21d",
    "financials_minus_utilities_ret_21d",
    "energy_minus_consdisc_ret_21d",
    "wti_ret_21d",
]


def _safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _performance_metrics(daily_perf: pd.DataFrame) -> pd.DataFrame:
    fund = daily_perf["fund_daily_net_return"].fillna(0.0)
    bm = daily_perf["bm_daily_return"].fillna(0.0)
    active = fund - bm
    ann_factor = 252.0
    ann_return = float(fund.mean() * ann_factor)
    ann_vol = float(fund.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    bm_ann = float(bm.mean() * ann_factor)
    tracking = float(active.std(ddof=0) * np.sqrt(ann_factor))
    ir = float((active.mean() * ann_factor) / tracking) if tracking > 0 else float("nan")
    rows = [
        {"metric": "fund_total_return", "value": float((1.0 + fund).prod() - 1.0)},
        {"metric": "fund_annualized_return", "value": ann_return},
        {"metric": "fund_annualized_vol", "value": ann_vol},
        {"metric": "fund_sharpe", "value": sharpe},
        {"metric": "bm_total_return", "value": float((1.0 + bm).prod() - 1.0)},
        {"metric": "bm_annualized_return", "value": bm_ann},
        {"metric": "active_total_return", "value": float((1.0 + active).prod() - 1.0)},
        {"metric": "active_annualized_return", "value": float(active.mean() * ann_factor)},
        {"metric": "tracking_error", "value": tracking},
        {"metric": "information_ratio", "value": ir},
    ]
    return pd.DataFrame(rows)


def _build_daily_lightgbm_performance(bundle: dict[str, Any], output_paths: dict[str, Path]) -> pd.DataFrame:
    holdings = _safe_read_csv(output_paths["portfolios"] / "monthly_holdings_lightgbm.csv", parse_dates=["date"])
    holdings = holdings[holdings["portfolio_type"] == PRIMARY_PORTFOLIO_TYPE].copy()
    if holdings.empty:
        raise ValueError("No LightGBM decile holdings found.")

    prices = bundle["prices"]
    daily_returns = bundle["daily_returns_from_px"]
    factor_returns = bundle["factor_returns"]
    proxy = "MXWD"

    monthly_dates = sorted(pd.Timestamp(date) for date in holdings["date"].unique())
    business_days = bundle["business_days"]
    rows = []
    previous_weights = None
    for idx, rebalance_date in enumerate(monthly_dates):
        weights = holdings[holdings["date"] == rebalance_date].set_index("asset")["weight"].astype(float)
        start_loc = business_days.searchsorted(rebalance_date, side="right")
        if start_loc >= len(business_days):
            continue
        start_date = pd.Timestamp(business_days[start_loc])
        end_date = monthly_dates[idx + 1] if idx + 1 < len(monthly_dates) else pd.Timestamp(business_days[-1])
        segment_days = business_days[(business_days >= start_date) & (business_days <= end_date)]
        turnover = float(weights.abs().sum()) if previous_weights is None else float((weights.reindex(weights.index.union(previous_weights.index)).fillna(0.0) - previous_weights.reindex(weights.index.union(previous_weights.index)).fillna(0.0)).abs().sum())
        daily_tc = turnover * 0.001
        for j, current_date in enumerate(segment_days):
            fund_ret = float((weights * daily_returns.loc[current_date, weights.index]).sum())
            bm_ret = float(factor_returns.loc[current_date, proxy]) if pd.notna(factor_returns.loc[current_date, proxy]) else 0.0
            rows.append({
                "date": current_date,
                "fund_daily_gross_return": fund_ret,
                "fund_daily_net_return": fund_ret - (daily_tc if j == 0 else 0.0),
                "bm_daily_return": bm_ret,
                "active_daily_return": (fund_ret - (daily_tc if j == 0 else 0.0)) - bm_ret,
                "rebalance_date": rebalance_date,
            })
        previous_weights = weights
    daily_perf = pd.DataFrame(rows).sort_values("date")
    daily_perf["fund_cum_nav"] = (1.0 + daily_perf["fund_daily_net_return"]).cumprod()
    daily_perf["bm_cum_nav"] = (1.0 + daily_perf["bm_daily_return"]).cumprod()
    save_csv(daily_perf, output_paths["reports"] / "lightgbm_daily_performance.csv", index=False)
    save_csv(_performance_metrics(daily_perf), output_paths["reports"] / "lightgbm_overall_metrics.csv", index=False)
    return daily_perf


def _build_recent_weights_report(output_paths: dict[str, Path]) -> pd.DataFrame:
    holdings = _safe_read_csv(output_paths["portfolios"] / "monthly_holdings_lightgbm.csv", parse_dates=["date"])
    holdings = holdings[holdings["portfolio_type"] == PRIMARY_PORTFOLIO_TYPE].copy()
    latest_path = output_paths["portfolios"] / "latest_holdings_lightgbm.csv"
    if latest_path.exists():
        latest = _safe_read_csv(latest_path, parse_dates=["date"])
        latest["portfolio_type"] = PRIMARY_PORTFOLIO_TYPE
        holdings = pd.concat([holdings, latest], ignore_index=True)
    recent_months = sorted(pd.to_datetime(holdings["date"]).drop_duplicates())[-6:]
    recent = holdings[pd.to_datetime(holdings["date"]).isin(recent_months)].copy()
    recent["position_side"] = np.where(recent["weight"] > 0, "OW", "UW")
    recent = recent.sort_values(["date", "weight"], ascending=[True, False])
    save_csv(recent, output_paths["reports"] / "lightgbm_recent_weights_6m.csv", index=False)
    return recent


def _build_model_structure_report(output_paths: dict[str, Path]) -> pd.DataFrame:
    report = pd.read_json(output_paths["models"] / "training_report_lightgbm.json")
    if report.empty:
        return report
    flattened_rows = []
    for row in report.to_dict(orient="records"):
        configs = row.get("kept_config_details", [])
        for cfg in configs:
            params = cfg.get("params", {})
            flattened_rows.append({
                "date": row.get("date"),
                "validation_rank_ic": row.get("validation_rank_ic"),
                "retained_configs": row.get("retained_configs"),
                "learning_rate": params.get("learning_rate"),
                "num_leaves": params.get("num_leaves"),
                "min_data_in_leaf": params.get("min_data_in_leaf"),
                "feature_fraction": params.get("feature_fraction"),
                "n_estimators": params.get("n_estimators"),
                "config_score": cfg.get("score"),
            })
    structure = pd.DataFrame(flattened_rows)
    save_csv(structure, output_paths["reports"] / "lightgbm_model_structure.csv", index=False)
    return structure


def _build_attribution_reports(output_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _safe_read_csv(output_paths["attribution"] / "attribution_summary_lightgbm.csv")
    comp = _safe_read_csv(output_paths["attribution"] / "component_predictions_lightgbm.csv", parse_dates=["date"])
    monthly_category = []
    for date, group in comp.groupby("date"):
        monthly_category.append({
            "date": date,
            "stock_price_effect_mean": float(group["category_stock_price_total_effect"].mean()),
            "stock_sector_relative_effect_mean": float(group["category_stock_sector_relative_total_effect"].mean()),
            "fundamental_effect_mean": float(group["category_fundamental_valuation_total_effect"].mean()),
            "sentiment_effect_mean": float(group["category_sentiment_analyst_total_effect"].mean()),
            "macro_effect_mean": float(group["category_global_macro_factor_total_effect"].mean()),
            "sector_regime_effect_mean": float(group["category_global_sector_regime_total_effect"].mean()),
        })
    monthly_category_df = pd.DataFrame(monthly_category).sort_values("date")
    save_csv(summary, output_paths["reports"] / "lightgbm_attribution_overview.csv", index=False)
    save_csv(monthly_category_df, output_paths["reports"] / "lightgbm_monthly_category_attribution.csv", index=False)
    return summary, monthly_category_df


def _infer_regime(row: pd.Series) -> tuple[str, str]:
    # Risk-off is checked first: elevated volatility with negative equity momentum
    # is the strongest macro signal and should not be masked by sector rotations.
    if row["mxwd_ret_21d"] < 0 and row["vix_level_z_252d"] > 0.5:
        return "Risk-off", "Global equity momentum was negative and volatility regime was elevated."
    if row["mxwd_ret_21d"] > 0 and row["vix_level_z_252d"] < 0 and row["growth_minus_value_ret_21d"] > 0:
        return "Risk-on growth", "Global equities were positive, volatility was subdued, and growth leadership dominated value."
    if row["energy_minus_consdisc_ret_21d"] > 0 and row["wti_ret_21d"] > 0:
        return "Commodity rotation", "Energy outperformed consumer discretionary alongside positive oil momentum."
    if row["financials_minus_utilities_ret_21d"] > 0 and row["mxwd_ret_21d"] > 0:
        return "Cyclical reflation", "Financials outperformed defensives with positive global equity momentum, consistent with a cyclical risk appetite."
    return "Mixed rotation", "Macro and sector regime signals were mixed rather than strongly directional."


def _adaptive_threshold(feature_row: pd.Series, feature_names: list[str], direction: str) -> float:
    """Compute a data-driven threshold based on the dispersion of the current
    feature values instead of relying on a static 0.15 cutoff.  This adapts
    to cross-sectional environments where overall feature magnitudes may be
    compressed or expanded."""
    vals = [float(feature_row.get(n, np.nan)) for n in feature_names]
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 3:
        return 0.15  # fallback
    std = float(np.std(vals, ddof=0))
    if std < 1e-8:
        return 0.15
    # Use 0.5 sigma as the significance threshold
    return max(0.5 * std, 0.05)


def _stock_reason_strings(feature_row: pd.Series) -> str:
    candidates = {name: float(feature_row.get(name, np.nan)) for name in FEATURE_LABELS}
    threshold = _adaptive_threshold(feature_row, list(FEATURE_LABELS.keys()), "positive")
    positives = {k: v for k, v in candidates.items() if np.isfinite(v) and v > threshold}
    ranked = sorted(positives.items(), key=lambda item: item[1], reverse=True)[:3]
    if not ranked:
        return "No single stock feature dominated; the OW decision was driven by combined tree interactions."
    return "; ".join(f"{FEATURE_LABELS[name]} ({value:.2f})" for name, value in ranked)


def _stock_uw_reason_strings(feature_row: pd.Series) -> str:
    candidates = {name: float(feature_row.get(name, np.nan)) for name in FEATURE_LABELS}
    threshold = _adaptive_threshold(feature_row, list(FEATURE_LABELS.keys()), "negative")
    negatives = {k: v for k, v in candidates.items() if np.isfinite(v) and v < -threshold}
    ranked = sorted(negatives.items(), key=lambda item: item[1])[:3]
    if not ranked:
        return "No single stock feature dominated; the UW decision was driven by combined tree interactions."
    return "; ".join(f"{FEATURE_LABELS[name]} ({value:.2f})" for name, value in ranked)


def _build_monthly_uw_explanations(bundle: dict[str, Any], output_paths: dict[str, Path]) -> pd.DataFrame:
    weights = _safe_read_csv(output_paths["reports"] / "lightgbm_recent_weights_6m.csv", parse_dates=["date"])
    feature_panel = pd.read_parquet(output_paths["features"] / "panel_monthly_features_model.parquet").reset_index()
    category_monthly = _safe_read_csv(output_paths["reports"] / "lightgbm_monthly_category_attribution.csv", parse_dates=["date"])
    feature_panel = feature_panel.merge(category_monthly, on="date", how="left")
    rows = []
    for date in sorted(weights["date"].drop_duplicates()):
        month_weights = weights[(weights["date"] == date) & (weights["weight"] < 0)].copy().sort_values("weight").head(5)
        month_feature_slice = feature_panel[feature_panel["date"] == date].copy()
        if month_feature_slice.empty:
            continue
        regime_source = month_feature_slice.iloc[0]
        regime_label, regime_reason = _infer_regime(regime_source)
        top_categories = {
            "stock_price": float(regime_source.get("stock_price_effect_mean", np.nan)),
            "stock_sector_relative": float(regime_source.get("stock_sector_relative_effect_mean", np.nan)),
            "fundamental": float(regime_source.get("fundamental_effect_mean", np.nan)),
            "sentiment": float(regime_source.get("sentiment_effect_mean", np.nan)),
            "macro": float(regime_source.get("macro_effect_mean", np.nan)),
            "sector_regime": float(regime_source.get("sector_regime_effect_mean", np.nan)),
        }
        top_category_text = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(top_categories.items(), key=lambda item: abs(item[1]), reverse=True)[:3])
        for row in month_weights.itertuples(index=False):
            stock_features = month_feature_slice[month_feature_slice["asset"] == row.asset]
            if stock_features.empty:
                continue
            reason = _stock_uw_reason_strings(stock_features.iloc[0])
            rows.append({
                "date": date,
                "asset": row.asset,
                "weight": row.weight,
                "score": row.score,
                "sector": row.sector,
                "regime_label": regime_label,
                "regime_reason": regime_reason,
                "dominant_category_effects": top_category_text,
                "uw_reason": reason,
            })
    explanations = pd.DataFrame(rows)
    save_csv(explanations, output_paths["reports"] / "lightgbm_monthly_uw_explanations.csv", index=False)
    return explanations


def _build_monthly_explanations(bundle: dict[str, Any], output_paths: dict[str, Path]) -> pd.DataFrame:
    weights = _safe_read_csv(output_paths["reports"] / "lightgbm_recent_weights_6m.csv", parse_dates=["date"])
    feature_panel = pd.read_parquet(output_paths["features"] / "panel_monthly_features_model.parquet").reset_index()
    category_monthly = _safe_read_csv(output_paths["reports"] / "lightgbm_monthly_category_attribution.csv", parse_dates=["date"])
    feature_panel = feature_panel.merge(category_monthly, on="date", how="left")
    rows = []
    for date in sorted(weights["date"].drop_duplicates()):
        month_weights = weights[(weights["date"] == date) & (weights["weight"] > 0)].copy().sort_values("weight", ascending=False).head(5)
        month_feature_slice = feature_panel[feature_panel["date"] == date].copy()
        if month_feature_slice.empty:
            continue
        regime_source = month_feature_slice.iloc[0]
        regime_label, regime_reason = _infer_regime(regime_source)
        top_categories = {
            "stock_price": float(regime_source.get("stock_price_effect_mean", np.nan)),
            "stock_sector_relative": float(regime_source.get("stock_sector_relative_effect_mean", np.nan)),
            "fundamental": float(regime_source.get("fundamental_effect_mean", np.nan)),
            "sentiment": float(regime_source.get("sentiment_effect_mean", np.nan)),
            "macro": float(regime_source.get("macro_effect_mean", np.nan)),
            "sector_regime": float(regime_source.get("sector_regime_effect_mean", np.nan)),
        }
        top_category_text = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(top_categories.items(), key=lambda item: abs(item[1]), reverse=True)[:3])
        for row in month_weights.itertuples(index=False):
            stock_features = month_feature_slice[month_feature_slice["asset"] == row.asset]
            if stock_features.empty:
                continue
            reason = _stock_reason_strings(stock_features.iloc[0])
            rows.append({
                "date": date,
                "asset": row.asset,
                "weight": row.weight,
                "score": row.score,
                "sector": row.sector,
                "regime_label": regime_label,
                "regime_reason": regime_reason,
                "dominant_category_effects": top_category_text,
                "ow_reason": reason,
            })
    explanations = pd.DataFrame(rows)
    save_csv(explanations, output_paths["reports"] / "lightgbm_monthly_ow_explanations.csv", index=False)
    return explanations



def _build_style_sector_summary(feature_payload: dict[str, Any], output_paths: dict[str, Path]) -> pd.DataFrame:
    holdings = _safe_read_csv(output_paths["portfolios"] / "monthly_holdings_lightgbm.csv", parse_dates=["date"])
    holdings = holdings[holdings["portfolio_type"] == PRIMARY_PORTFOLIO_TYPE].copy()
    latest_path = output_paths["portfolios"] / "latest_holdings_lightgbm.csv"
    if latest_path.exists():
        latest = _safe_read_csv(latest_path, parse_dates=["date"])
        latest["portfolio_type"] = PRIMARY_PORTFOLIO_TYPE
        holdings = pd.concat([holdings, latest], ignore_index=True)
    monthly_features = pd.read_parquet(output_paths["features"] / "panel_monthly_features_model.parquet").reset_index()
    rows = []
    for date, month_holdings in holdings.groupby("date"):
        month_features = monthly_features[monthly_features["date"] == date].set_index("asset")
        if month_features.empty:
            continue
        sector_weights = month_holdings.groupby("sector")["weight"].sum().sort_values(ascending=False)
        abs_sector_weights = month_holdings.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
        style_scores = {}
        denom = month_holdings["weight"].abs().sum()
        for style_name, definitions in STYLE_GROUPS.items():
            total = 0.0
            for feature_name, sign in definitions:
                if feature_name not in month_features.columns:
                    continue
                aligned = month_holdings.set_index("asset")["weight"].reindex(month_features.index).fillna(0.0)
                total += float((aligned * (month_features[feature_name].fillna(0.0) * sign)).sum())
            style_scores[style_name] = total / denom if denom > 0 else float('nan')
        positive_styles = [f"{k}:{v:.3f}" for k, v in sorted(style_scores.items(), key=lambda item: item[1], reverse=True)[:3]]
        negative_styles = [f"{k}:{v:.3f}" for k, v in sorted(style_scores.items(), key=lambda item: item[1])[:3]]
        rows.append({
            "date": date,
            "top_long_sectors": ', '.join(f"{idx}:{val:.3f}" for idx, val in sector_weights.head(3).items()),
            "top_gross_sectors": ', '.join(f"{idx}:{val:.3f}" for idx, val in abs_sector_weights.head(3).items()),
            "positive_style_tilts": '; '.join(positive_styles),
            "negative_style_tilts": '; '.join(negative_styles),
            **style_scores,
        })
    summary = pd.DataFrame(rows).sort_values("date")
    save_csv(summary, output_paths["reports"] / "lightgbm_rebalance_style_sector_summary.csv", index=False)
    return summary
def generate_lightgbm_reporting(bundle: dict[str, Any], feature_payload: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    daily_perf = _build_daily_lightgbm_performance(bundle, output_paths)
    recent_weights = _build_recent_weights_report(output_paths)
    model_structure = _build_model_structure_report(output_paths)
    attribution_summary, monthly_category = _build_attribution_reports(output_paths)
    style_sector_summary = _build_style_sector_summary(feature_payload, output_paths)
    explanations = _build_monthly_explanations(bundle, output_paths)
    uw_explanations = _build_monthly_uw_explanations(bundle, output_paths)
    LOGGER.info("Generated LightGBM reporting outputs.")
    return {
        "daily_performance": daily_perf,
        "recent_weights": recent_weights,
        "model_structure": model_structure,
        "attribution_summary": attribution_summary,
        "monthly_category": monthly_category,
        "style_sector_summary": style_sector_summary,
        "explanations": explanations,
        "uw_explanations": uw_explanations,
    }








