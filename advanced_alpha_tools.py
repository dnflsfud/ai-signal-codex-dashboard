"""Advanced target and feature engineering helpers for the alpha engine."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def safe_scaled_change(panel: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Stable change scaled by the absolute lagged level."""
    lagged = panel.shift(periods)
    return panel.subtract(lagged).divide(lagged.abs().clip(lower=1.0e-6))


def time_series_zscore(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series z-score per asset."""
    min_obs = max(2, min(window, max(10, window // 3)))
    mean = panel.rolling(window, min_periods=min_obs).mean()
    std = panel.rolling(window, min_periods=min_obs).std(ddof=0)
    return panel.subtract(mean).divide(std.replace(0.0, np.nan))


def rolling_downside_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling downside volatility."""
    downside = returns.where(returns < 0.0, 0.0)
    return downside.rolling(window, min_periods=max(2, min(window, max(10, window // 3)))).std(ddof=0) * np.sqrt(252.0)


def rolling_upside_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling upside volatility."""
    upside = returns.where(returns > 0.0, 0.0)
    return upside.rolling(window, min_periods=max(2, min(window, max(10, window // 3)))).std(ddof=0) * np.sqrt(252.0)


def rolling_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling drawdown from the trailing peak."""
    min_obs = max(2, min(window, max(10, window // 3)))
    rolling_peak = prices.rolling(window, min_periods=min_obs).max()
    return prices.divide(rolling_peak) - 1.0


def broadcast_date_series(series: pd.Series, assets: Sequence[str]) -> pd.DataFrame:
    """Repeat a date-level series across all assets."""
    values = np.repeat(series.to_numpy(dtype=float).reshape(-1, 1), len(assets), axis=1)
    return pd.DataFrame(values, index=series.index, columns=list(assets), dtype=float)


def add_frame(feature_frames: Dict[str, pd.DataFrame], name: str, frame: pd.DataFrame) -> None:
    """Store a frame with infinities removed."""
    feature_frames[name] = frame.replace([np.inf, -np.inf], np.nan)


def add_level_and_change_family(
    feature_frames: Dict[str, pd.DataFrame],
    name: str,
    panel: pd.DataFrame,
    windows: Sequence[int],
) -> None:
    """Add stable multi-horizon change and z-score features."""
    for window in windows:
        add_frame(feature_frames, f"{name}_chg_{window}d", safe_scaled_change(panel, window))
        add_frame(feature_frames, f"{name}_tsz_{window}d", time_series_zscore(panel, window))
    if len(windows) >= 2:
        short, medium = windows[0], windows[1]
        add_frame(
            feature_frames,
            f"{name}_accel_{short}_{medium}",
            safe_scaled_change(panel, short) - safe_scaled_change(panel, medium),
        )


def add_price_feature_family(
    feature_frames: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    residual_daily_returns: pd.DataFrame,
) -> None:
    """Create richer price and risk features."""
    for window in (1, 5, 10, 21, 63, 126, 252):
        add_frame(feature_frames, f"price_return_{window}d", prices.divide(prices.shift(window)) - 1.0)
    add_frame(feature_frames, "price_reversal_5d", -(prices.divide(prices.shift(5)) - 1.0))
    add_frame(feature_frames, "price_reversal_10d", -(prices.divide(prices.shift(10)) - 1.0))
    add_frame(feature_frames, "price_momentum_21_126", (prices.divide(prices.shift(21)) - 1.0) - (prices.divide(prices.shift(126)) - 1.0))
    add_frame(feature_frames, "price_momentum_63_252", (prices.divide(prices.shift(63)) - 1.0) - (prices.divide(prices.shift(252)) - 1.0))

    for window in (10, 20, 63, 126):
        add_frame(feature_frames, f"price_vol_{window}d", daily_returns.rolling(window, min_periods=max(2, min(window, max(10, window // 3)))).std(ddof=0) * np.sqrt(252.0))
        add_frame(feature_frames, f"price_downside_vol_{window}d", rolling_downside_vol(daily_returns, window))
        add_frame(feature_frames, f"price_upside_vol_{window}d", rolling_upside_vol(daily_returns, window))
    for window in (63, 126):
        add_frame(feature_frames, f"price_drawdown_{window}d", rolling_drawdown(prices, window))
        add_frame(feature_frames, f"residual_vol_{window}d", residual_daily_returns.rolling(window, min_periods=max(2, min(window, max(10, window // 3)))).std(ddof=0) * np.sqrt(252.0))
        add_frame(feature_frames, f"residual_return_{window}d", residual_daily_returns.rolling(window, min_periods=max(2, min(window, max(10, window // 3)))).sum())

    benchmark_21 = benchmark_returns.rolling(21, min_periods=10).sum()
    benchmark_63 = benchmark_returns.rolling(63, min_periods=20).sum()
    add_frame(feature_frames, "price_alpha_vs_benchmark_21d", (prices.divide(prices.shift(21)) - 1.0).subtract(benchmark_21, axis=0))
    add_frame(feature_frames, "price_alpha_vs_benchmark_63d", (prices.divide(prices.shift(63)) - 1.0).subtract(benchmark_63, axis=0))


def compute_pca_residual_daily_returns(
    daily_returns: pd.DataFrame,
    tradability_mask: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute ex-ante daily residual returns via rolling PCA."""
    target_cfg = config["target"]
    pca_cfg = target_cfg.get("pca", {})
    window = int(pca_cfg.get("window_bdays", 252))
    min_history = int(pca_cfg.get("min_history_bdays", 126))
    min_coverage = float(pca_cfg.get("min_coverage_ratio", 0.8))
    min_assets = int(pca_cfg.get("min_assets", 10))
    n_components_requested = int(pca_cfg.get("n_components", 20))

    residual = pd.DataFrame(np.nan, index=daily_returns.index, columns=daily_returns.columns, dtype=float)
    returns = daily_returns.copy()

    for idx in range(len(returns)):
        if idx < min_history:
            continue
        current_date = returns.index[idx]
        history_start = max(0, idx - window)
        history = returns.iloc[history_start:idx]
        current_mask = tradability_mask.iloc[idx].fillna(False)
        tradable_assets = current_mask[current_mask].index.tolist()
        if len(tradable_assets) < min_assets:
            continue
        coverage = history.loc[:, tradable_assets].notna().mean(axis=0)
        eligible_assets = coverage[coverage >= min_coverage].index.tolist()
        if len(eligible_assets) < min_assets:
            continue

        X = history.loc[:, eligible_assets].fillna(0.0).to_numpy(dtype=float)
        mean_vector = X.mean(axis=0, keepdims=True)
        centered = X - mean_vector
        max_components = min(n_components_requested, len(eligible_assets) - 1, len(history) - 1)
        if max_components < 1:
            residual.loc[current_date, eligible_assets] = returns.loc[current_date, eligible_assets].to_numpy(dtype=float)
            continue

        pca = PCA(n_components=max_components, svd_solver="full", random_state=0)
        pca.fit(centered)
        current_row = returns.loc[current_date, eligible_assets].fillna(0.0).to_numpy(dtype=float).reshape(1, -1)
        current_centered = current_row - mean_vector
        reconstructed = pca.inverse_transform(pca.transform(current_centered))
        residual.loc[current_date, eligible_assets] = (current_centered - reconstructed).ravel()

    return residual


def build_pca_specific_return_target(
    daily_returns: pd.DataFrame,
    tradability_mask: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a forward 20-business-day residual-return target from rolling PCA residuals."""
    residual_daily_returns = compute_pca_residual_daily_returns(daily_returns, tradability_mask, config)
    horizon = int(config["target"]["horizon_bdays"])
    execution_lag = int(config["target"]["execution_lag_bdays"])

    forward_specific = pd.DataFrame(0.0, index=residual_daily_returns.index, columns=residual_daily_returns.columns)
    for step in range(execution_lag, execution_lag + horizon):
        forward_specific = forward_specific.add(residual_daily_returns.shift(-step), fill_value=np.nan)
    return forward_specific, residual_daily_returns


def add_factor_conditioning_features(
    feature_frames: Dict[str, pd.DataFrame],
    factor_returns_df: pd.DataFrame | None,
    factor_price_df: pd.DataFrame | None,
    assets: Sequence[str],
    business_days: pd.DatetimeIndex,
    lag_bdays: int,
    prefix: str,
) -> None:
    """Create date-level factor and market regime features."""
    if factor_returns_df is None and factor_price_df is None:
        return

    if factor_returns_df is not None:
        factor_returns = factor_returns_df.copy()
        factor_returns.columns = [str(col).strip() for col in factor_returns.columns]
        date_col = factor_returns.columns[0]
        factor_returns[date_col] = pd.to_datetime(factor_returns[date_col], errors="coerce")
        factor_returns = factor_returns.dropna(subset=[date_col]).set_index(date_col).sort_index()
        factor_returns = factor_returns.apply(pd.to_numeric, errors="coerce").reindex(business_days).shift(lag_bdays)
    else:
        factor_returns = pd.DataFrame(index=business_days)

    if factor_price_df is not None:
        factor_prices = factor_price_df.copy()
        factor_prices.columns = [str(col).strip() for col in factor_prices.columns]
        date_col = factor_prices.columns[0]
        factor_prices[date_col] = pd.to_datetime(factor_prices[date_col], errors="coerce")
        factor_prices = factor_prices.dropna(subset=[date_col]).set_index(date_col).sort_index()
        factor_prices = factor_prices.apply(pd.to_numeric, errors="coerce").reindex(business_days).shift(lag_bdays)
    else:
        factor_prices = pd.DataFrame(index=business_days)

    for column in factor_returns.columns:
        series = factor_returns[column]
        add_frame(feature_frames, f"{prefix}{column}_ret_5d", broadcast_date_series(series.rolling(5, min_periods=3).sum(), assets))
        add_frame(feature_frames, f"{prefix}{column}_ret_21d", broadcast_date_series(series.rolling(21, min_periods=10).sum(), assets))
        add_frame(feature_frames, f"{prefix}{column}_ret_63d", broadcast_date_series(series.rolling(63, min_periods=20).sum(), assets))
        add_frame(feature_frames, f"{prefix}{column}_vol_21d", broadcast_date_series(series.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0), assets))
        add_frame(feature_frames, f"{prefix}{column}_vol_63d", broadcast_date_series(series.rolling(63, min_periods=20).std(ddof=0) * np.sqrt(252.0), assets))
        add_frame(feature_frames, f"{prefix}{column}_z_63d", broadcast_date_series((series - series.rolling(63, min_periods=20).mean()).divide(series.rolling(63, min_periods=20).std(ddof=0).replace(0.0, np.nan)), assets))

    for column in factor_prices.columns:
        price_series = factor_prices[column]
        add_frame(feature_frames, f"{prefix}{column}_drawdown_63d", broadcast_date_series(price_series.divide(price_series.rolling(63, min_periods=20).max()) - 1.0, assets))


def add_conditioning_state_features(
    feature_frames: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    market_caps: pd.DataFrame,
    revision_panel: pd.DataFrame,
    sentiment_panel: pd.DataFrame,
    assets: Sequence[str],
    prefix: str,
) -> None:
    """Create date-level conditioning features from market state and calendar state."""
    benchmark_weights = market_caps.divide(market_caps.sum(axis=1), axis=0).fillna(0.0)
    benchmark_returns = (benchmark_weights.shift(1).fillna(0.0) * daily_returns.fillna(0.0)).sum(axis=1)
    breadth_21 = (prices.divide(prices.shift(21)) - 1.0).gt(0.0).mean(axis=1)
    breadth_63 = (prices.divide(prices.shift(63)) - 1.0).gt(0.0).mean(axis=1)
    xs_dispersion = daily_returns.std(axis=1, ddof=0)
    market_vol_21 = benchmark_returns.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    market_vol_63 = benchmark_returns.rolling(63, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    market_ret_21 = benchmark_returns.rolling(21, min_periods=10).sum()
    market_ret_63 = benchmark_returns.rolling(63, min_periods=20).sum()
    top5_concentration = benchmark_weights.apply(lambda row: row.nlargest(min(5, len(row))).sum(), axis=1)
    revision_breadth = revision_panel.gt(0.0).mean(axis=1)
    sentiment_breadth = sentiment_panel.gt(0.0).mean(axis=1)
    market_corr_proxy = benchmark_returns.rolling(63, min_periods=20).std(ddof=0).divide(
        daily_returns.rolling(63, min_periods=20).std(ddof=0).mean(axis=1).replace(0.0, np.nan)
    )

    calendar = pd.DataFrame(index=prices.index)
    calendar["month_sin"] = np.sin(2.0 * np.pi * prices.index.month / 12.0)
    calendar["month_cos"] = np.cos(2.0 * np.pi * prices.index.month / 12.0)
    calendar["quarter_sin"] = np.sin(2.0 * np.pi * prices.index.quarter / 4.0)
    calendar["quarter_cos"] = np.cos(2.0 * np.pi * prices.index.quarter / 4.0)
    calendar["earnings_season_flag"] = prices.index.month.isin([1, 4, 7, 10]).astype(float)
    calendar["month_end_flag"] = prices.index.is_month_end.astype(float)
    calendar["year_end_flag"] = ((prices.index.month == 12) & (prices.index.day >= 15)).astype(float)

    date_level_series = {
        f"{prefix}benchmark_ret_21d": market_ret_21,
        f"{prefix}benchmark_ret_63d": market_ret_63,
        f"{prefix}benchmark_vol_21d": market_vol_21,
        f"{prefix}benchmark_vol_63d": market_vol_63,
        f"{prefix}breadth_21d": breadth_21,
        f"{prefix}breadth_63d": breadth_63,
        f"{prefix}xs_dispersion": xs_dispersion,
        f"{prefix}top5_concentration": top5_concentration,
        f"{prefix}revision_breadth": revision_breadth,
        f"{prefix}sentiment_breadth": sentiment_breadth,
        f"{prefix}corr_proxy": market_corr_proxy,
    }
    date_level_series.update({f"{prefix}{name}": calendar[name] for name in calendar.columns})

    for name, series in date_level_series.items():
        add_frame(feature_frames, name, broadcast_date_series(series, assets))


def add_engineered_features(
    feature_frames: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    residual_daily_returns: pd.DataFrame,
    assets: Sequence[str],
    sheets: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Expand the feature set with richer accounting, price, sellside, and conditioning features."""
    market_cap = feature_frames["cur_mkt_cap"]
    benchmark_weights = market_cap.divide(market_cap.sum(axis=1), axis=0).fillna(0.0)
    benchmark_returns = (benchmark_weights.shift(1).fillna(0.0) * daily_returns.fillna(0.0)).sum(axis=1)

    add_price_feature_family(feature_frames, prices, daily_returns, benchmark_returns, residual_daily_returns)

    accounting_like = [
        "best_eps",
        "best_sales",
        "best_calculated_fcf",
        "best_capex",
        "best_roe",
        "best_gross_margin",
        "oper_margin",
        "best_pe_ratio",
        "best_peg_ratio",
        "best_px_bps_ratio",
        "best_ev_to_best_ebitda",
    ]
    for name in accounting_like:
        add_level_and_change_family(feature_frames, name, feature_frames[name], windows=(21, 63, 252))

    sellside_like = [
        "eqy_rec_cons",
        "factset_eps_revision",
        "factset_sales_revision",
        "factset_tg_price",
        "news_sentiment_daily_avg",
        "sent_trend_momentum",
        "sent_trend_21d",
    ]
    for name in sellside_like:
        add_level_and_change_family(feature_frames, name, feature_frames[name], windows=(5, 21, 63))

    add_frame(feature_frames, "target_price_premium", feature_frames["factset_tg_price"].divide(prices) - 1.0)
    add_frame(feature_frames, "target_price_premium_chg_21d", safe_scaled_change(feature_frames["factset_tg_price"].divide(prices) - 1.0, 21))
    add_frame(feature_frames, "target_price_premium_chg_63d", safe_scaled_change(feature_frames["factset_tg_price"].divide(prices) - 1.0, 63))
    add_frame(feature_frames, "fcf_yield", feature_frames["best_calculated_fcf"].divide(market_cap.replace(0.0, np.nan)))
    add_frame(feature_frames, "sales_to_mkt_cap", feature_frames["best_sales"].divide(market_cap.replace(0.0, np.nan)))
    add_frame(feature_frames, "capex_to_mkt_cap", feature_frames["best_capex"].divide(market_cap.replace(0.0, np.nan)))
    add_frame(feature_frames, "eps_to_price", feature_frames["best_eps"].divide(prices.replace(0.0, np.nan)))

    add_frame(feature_frames, "revision_combo", feature_frames["factset_eps_revision"] + feature_frames["factset_sales_revision"])
    add_frame(feature_frames, "revision_diff", feature_frames["factset_eps_revision"] - feature_frames["factset_sales_revision"])
    add_frame(feature_frames, "sentiment_revision_combo", feature_frames["news_sentiment_daily_avg"] * feature_frames["factset_eps_revision"])
    add_frame(feature_frames, "analyst_sentiment_combo", feature_frames["eqy_rec_cons"] * feature_frames["news_sentiment_daily_avg"])
    add_frame(feature_frames, "trend_sentiment_gap", feature_frames["sent_trend_momentum"] - feature_frames["sent_trend_21d"])

    macro_prefix = str(config["data"].get("factor_feature_prefix", "macro_"))
    add_factor_conditioning_features(
        feature_frames=feature_frames,
        factor_returns_df=sheets.get(config["data"].get("factor_returns_sheet")),
        factor_price_df=sheets.get(config["data"].get("factor_price_sheet")),
        assets=assets,
        business_days=prices.index,
        lag_bdays=int(config["features"]["lag_bdays"]),
        prefix=macro_prefix,
    )
    add_conditioning_state_features(
        feature_frames=feature_frames,
        prices=prices,
        daily_returns=daily_returns,
        market_caps=market_cap,
        revision_panel=feature_frames["factset_eps_revision"],
        sentiment_panel=feature_frames["news_sentiment_daily_avg"],
        assets=assets,
        prefix=macro_prefix,
    )
    return feature_frames


def fit_linear_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    alpha: float,
) -> np.ndarray:
    """Fit a linear ridge baseline for attribution diagnostics."""
    median_values = X_train.median(axis=0)
    X_train_filled = X_train.fillna(median_values).fillna(0.0)
    X_pred_filled = X_pred.fillna(median_values).fillna(0.0)
    model = Ridge(alpha=alpha)
    model.fit(X_train_filled, y_train)
    return model.predict(X_pred_filled)


def compute_nonlinear_share(raw_prediction: pd.Series, linear_prediction: pd.Series) -> Dict[str, float]:
    """Estimate the share of prediction variance not captured by the linear baseline."""
    aligned = pd.concat([raw_prediction, linear_prediction], axis=1).dropna()
    aligned.columns = ["raw", "linear"]
    if aligned.empty:
        return {"linear_projection_coef": float("nan"), "linear_share": float("nan"), "nonlinear_share": float("nan"), "tree_linear_correlation": float("nan")}

    raw = aligned["raw"].to_numpy(dtype=float)
    linear = aligned["linear"].to_numpy(dtype=float)
    denom = float(np.dot(linear, linear))
    coef = float(np.dot(raw, linear) / denom) if denom > 1.0e-12 else 0.0
    projected_linear = coef * linear
    residual = raw - projected_linear
    raw_var = float(np.var(raw))
    linear_var = float(np.var(projected_linear))
    nonlinear_var = float(np.var(residual))
    if raw_var <= 1.0e-12:
        linear_share = float("nan")
        nonlinear_share = float("nan")
    else:
        linear_share = linear_var / raw_var
        nonlinear_share = nonlinear_var / raw_var
    return {
        "linear_projection_coef": coef,
        "linear_share": linear_share,
        "nonlinear_share": nonlinear_share,
        "tree_linear_correlation": float(np.corrcoef(raw, linear)[0, 1]) if len(aligned) > 1 else float("nan"),
    }



