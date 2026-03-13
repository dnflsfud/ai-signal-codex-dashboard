"""ML alpha engine for 20-business-day specific return forecasting."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from advanced_alpha_tools import (
    add_engineered_features,
    build_pca_specific_return_target,
    compute_nonlinear_share,
    fit_linear_baseline,
)


LOGGER = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping.")
    return config


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "alpha_engine.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def excel_serial_to_datetime(values: pd.Series) -> pd.Series:
    series = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    serial_mask = numeric.between(10000, 100000)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric.loc[serial_mask], unit="D")
    return parsed


def build_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    frame = universe_df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    required = {"Ticker", "Name", "Sector", "Status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Universe sheet is missing columns: {sorted(missing)}")
    frame["short_ticker"] = frame["Ticker"].astype(str).str.split().str[0]
    frame["Name"] = frame["Name"].astype(str).str.strip()
    frame["Sector"] = frame["Sector"].astype(str).str.strip()
    frame["Status"] = frame["Status"].astype(str).str.strip()
    if not frame["short_ticker"].is_unique:
        raise AssertionError("Short tickers must be unique.")
    return frame.loc[:, ["Ticker", "short_ticker", "Name", "Sector", "Status"]]


def build_business_days(business_days_df: pd.DataFrame) -> pd.DatetimeIndex:
    first_col = business_days_df.columns[0]
    dates = excel_serial_to_datetime(business_days_df[first_col]).dropna()
    dates = pd.DatetimeIndex(dates.sort_values().drop_duplicates())
    if not dates.is_monotonic_increasing:
        raise AssertionError("Business-day calendar must be sorted.")
    return dates


def prepare_wide_panel(
    raw_df: pd.DataFrame,
    asset_columns: Sequence[str],
    rename_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    frame = raw_df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    date_col = frame.columns[0]
    frame[date_col] = excel_serial_to_datetime(frame[date_col])
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep="last")
    rename_map = rename_map or {}
    new_columns = {
        column: rename_map.get(str(column).strip(), str(column).strip())
        for column in frame.columns[1:]
    }
    panel = frame.rename(columns=new_columns).set_index(date_col)
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.loc[:, [col for col in panel.columns if col in set(asset_columns)]]
    panel = panel.reindex(columns=list(asset_columns))
    panel = panel.loc[~panel.index.duplicated(keep="last")].sort_index()
    panel.index.name = "date"
    return panel


def load_source_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    workbook_path = Path(config["paths"]["workbook_path"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    data_cfg = config["data"]
    sheet_names: List[str] = [
        data_cfg["universe_sheet"],
        data_cfg["business_days_sheet"],
        data_cfg["price_sheet"],
    ]
    sheet_names.extend(data_cfg["wide_feature_sheets"].keys())
    sheet_names.extend(data_cfg["name_feature_sheets"].keys())
    sheet_names.extend(data_cfg["factset_feature_sheets"].keys())
    if data_cfg.get("factor_returns_sheet"):
        sheet_names.append(data_cfg["factor_returns_sheet"])
    if data_cfg.get("factor_price_sheet"):
        sheet_names.append(data_cfg["factor_price_sheet"])
    LOGGER.info("Reading %s sheets from %s", len(sheet_names), workbook_path)
    return pd.read_excel(workbook_path, sheet_name=sheet_names, engine="openpyxl")


def align_to_business_days(panel: pd.DataFrame, business_days: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = panel.reindex(business_days)
    aligned.index.name = "date"
    return aligned


def winsorize_series(series: pd.Series, clip_z: float) -> pd.Series:
    if series.count() < 3:
        return series
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std <= 0:
        return series
    return mean + (((series - mean) / std).clip(-clip_z, clip_z) * std)


def winsorize_cross_sectional_frame(frame: pd.DataFrame, clip_z: float) -> pd.DataFrame:
    if clip_z <= 0:
        return frame
    return frame.apply(lambda row: winsorize_series(row, clip_z), axis=1)


def zscore_series(series: pd.Series) -> pd.Series:
    if series.count() < 2:
        return pd.Series(np.nan, index=series.index)
    std = series.std(ddof=0)
    if pd.isna(std) or std <= 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_rebalance_dates(business_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    grouped = pd.Series(business_days, index=business_days).groupby(business_days.to_period("W-FRI"))
    return pd.DatetimeIndex(grouped.max().sort_values().values)


def build_retrain_dates(rebalance_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    grouped = pd.Series(rebalance_dates, index=rebalance_dates).groupby(rebalance_dates.to_period("Q"))
    return pd.DatetimeIndex(grouped.min().sort_values().values)


def build_tradability_mask(
    prices: pd.DataFrame,
    business_days: pd.DatetimeIndex,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    threshold = float(config["tradability"]["return_abs_threshold"])
    manual_starts = {
        str(asset): pd.Timestamp(value)
        for asset, value in config["tradability"].get("manual_start_dates", {}).items()
    }
    returns = prices.pct_change()
    mask = pd.DataFrame(False, index=business_days, columns=prices.columns)
    start_dates: Dict[str, str] = {}
    for asset in prices.columns:
        inferred = returns.index[(returns[asset].abs() > threshold).fillna(False)].min()
        start_date = manual_starts.get(asset, inferred)
        if pd.isna(start_date):
            LOGGER.warning("Asset %s never becomes tradable under the configured rule.", asset)
            continue
        start_ts = pd.Timestamp(start_date)
        mask.loc[mask.index >= start_ts, asset] = True
        start_dates[asset] = start_ts.strftime("%Y-%m-%d")
    return mask, start_dates


def compute_equal_weight_universe_return(
    daily_returns: pd.DataFrame,
    tradability_mask: pd.DataFrame,
) -> pd.Series:
    return daily_returns.where(tradability_mask).mean(axis=1, skipna=True)


def compute_rolling_beta(
    asset_returns: pd.DataFrame,
    universe_returns: pd.Series,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    target_cfg = config["target"]
    covariance = asset_returns.rolling(
        window=int(target_cfg["beta_window_bdays"]),
        min_periods=int(target_cfg["beta_min_periods"]),
    ).cov(universe_returns)
    variance = universe_returns.rolling(
        window=int(target_cfg["beta_window_bdays"]),
        min_periods=int(target_cfg["beta_min_periods"]),
    ).var()
    beta = covariance.divide(variance, axis=0)
    return beta.clip(lower=-float(target_cfg["beta_clip"]), upper=float(target_cfg["beta_clip"]))


def compute_specific_return_target(
    prices: pd.DataFrame,
    universe_returns: pd.Series,
    betas: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    horizon = int(config["target"]["horizon_bdays"])
    execution_lag = int(config["target"]["execution_lag_bdays"])
    asset_forward = prices.shift(-(horizon + execution_lag)).divide(prices.shift(-execution_lag)) - 1.0
    universe_index = (1.0 + universe_returns.fillna(0.0)).cumprod()
    universe_forward = universe_index.shift(-(horizon + execution_lag)).divide(
        universe_index.shift(-execution_lag)
    ) - 1.0
    specific = asset_forward.subtract(betas.mul(universe_forward, axis=0), axis=0)
    return winsorize_cross_sectional_frame(specific, float(config["target"]["winsorize_z"]))

def create_sector_dummies(
    business_days: pd.DatetimeIndex,
    universe: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    sector_matrix = pd.get_dummies(
        universe.set_index("short_ticker")["Sector"],
        prefix="sector",
        dtype=float,
    )
    dummies: Dict[str, pd.DataFrame] = {}
    for column in sector_matrix.columns:
        dummies[column] = pd.DataFrame(
            np.repeat(sector_matrix[[column]].T.values, len(business_days), axis=0),
            index=business_days,
            columns=sector_matrix.index,
            dtype=float,
        )
    return dummies



def build_macro_feature_frames(
    factor_returns_df: pd.DataFrame,
    assets: Sequence[str],
    business_days: pd.DatetimeIndex,
    lag_bdays: int,
    prefix: str,
) -> Dict[str, pd.DataFrame]:
    factor_frame = factor_returns_df.copy()
    factor_frame.columns = [str(col).strip() for col in factor_frame.columns]
    date_col = factor_frame.columns[0]
    factor_frame[date_col] = excel_serial_to_datetime(factor_frame[date_col])
    factor_frame = factor_frame.dropna(subset=[date_col]).sort_values(date_col)
    factor_frame = factor_frame.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)
    factor_frame = factor_frame.apply(pd.to_numeric, errors="coerce")
    factor_frame = factor_frame.reindex(business_days).shift(lag_bdays)
    factor_frame.index.name = "date"

    macro_frames: Dict[str, pd.DataFrame] = {}
    for column in factor_frame.columns:
        feature_name = f"{prefix}{column}"
        repeated = np.repeat(factor_frame[[column]].to_numpy(dtype=float), len(assets), axis=1)
        macro_frames[feature_name] = pd.DataFrame(repeated, index=business_days, columns=list(assets), dtype=float)
    return macro_frames
def build_feature_frames(
    sheets: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
    business_days: pd.DatetimeIndex,
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    residual_daily_returns: pd.DataFrame,
    betas: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    assets = universe["short_ticker"].tolist()
    name_to_ticker = dict(zip(universe["Name"], universe["short_ticker"]))
    data_cfg = config["data"]
    feature_frames: Dict[str, pd.DataFrame] = {}

    for sheet_name, alias in data_cfg["wide_feature_sheets"].items():
        feature_frames[alias] = align_to_business_days(
            prepare_wide_panel(sheets[sheet_name], assets),
            business_days,
        )
    for sheet_name, alias in data_cfg["name_feature_sheets"].items():
        feature_frames[alias] = align_to_business_days(
            prepare_wide_panel(sheets[sheet_name], assets, rename_map=name_to_ticker),
            business_days,
        )
    for sheet_name, alias in data_cfg["factset_feature_sheets"].items():
        feature_frames[alias] = align_to_business_days(
            prepare_wide_panel(sheets[sheet_name], assets),
            business_days,
        )

    feature_frames["return_5d"] = prices.divide(prices.shift(5)) - 1.0
    feature_frames["return_20d"] = prices.divide(prices.shift(20)) - 1.0
    feature_frames["return_60d"] = prices.divide(prices.shift(60)) - 1.0
    feature_frames["vol_20d"] = daily_returns.rolling(20).std(ddof=0) * np.sqrt(252.0)
    feature_frames["vol_60d"] = daily_returns.rolling(60).std(ddof=0) * np.sqrt(252.0)
    feature_frames["universe_beta"] = betas

    market_cap = feature_frames["cur_mkt_cap"]
    feature_frames["log_mkt_cap"] = np.log(market_cap.where(market_cap > 0))
    feature_frames["target_price_premium"] = feature_frames["factset_tg_price"].divide(prices) - 1.0
    feature_frames["fcf_yield"] = feature_frames["best_calculated_fcf"].divide(market_cap)
    feature_frames["sales_to_mkt_cap"] = feature_frames["best_sales"].divide(market_cap)
    feature_frames["capex_to_mkt_cap"] = feature_frames["best_capex"].divide(market_cap)

    for column_name, panel in create_sector_dummies(business_days, universe).items():
        feature_frames[column_name] = panel

    lag_bdays = int(config["features"]["lag_bdays"])
    macro_prefix = str(data_cfg.get("factor_feature_prefix", "macro_"))
    if data_cfg.get("factor_returns_sheet") and data_cfg["factor_returns_sheet"] in sheets:
        feature_frames.update(
            build_macro_feature_frames(
                factor_returns_df=sheets[data_cfg["factor_returns_sheet"]],
                assets=assets,
                business_days=business_days,
                lag_bdays=lag_bdays,
                prefix=macro_prefix,
            )
        )

    feature_frames = add_engineered_features(
        feature_frames=feature_frames,
        prices=prices,
        daily_returns=daily_returns,
        residual_daily_returns=residual_daily_returns,
        assets=assets,
        sheets=sheets,
        config=config,
    )

    for name, frame in list(feature_frames.items()):
        if not name.startswith("sector_") and not name.startswith(macro_prefix):
            feature_frames[name] = frame.shift(lag_bdays)
    return feature_frames


def panel_to_long(feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for feature_name, frame in feature_frames.items():
        stacked = frame.stack(dropna=False).rename(feature_name)
        stacked.index.names = ["date", "asset"]
        parts.append(stacked)
    return pd.concat(parts, axis=1).sort_index()


def cross_sectionally_standardize_features(
    feature_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    clip_z = float(config["features"]["cross_sectional_winsorize_z"])
    macro_prefix = str(config["data"].get("factor_feature_prefix", "macro_"))
    macro_columns = [column for column in feature_panel.columns if str(column).startswith(macro_prefix)]
    cross_sectional_columns = [column for column in feature_panel.columns if column not in macro_columns]

    standardized = feature_panel.copy()
    for _, indices in standardized.groupby(level="date").groups.items():
        if cross_sectional_columns:
            block = standardized.loc[indices, cross_sectional_columns]
            clipped = block.apply(lambda col: winsorize_series(col, clip_z))
            standardized.loc[indices, cross_sectional_columns] = clipped.apply(zscore_series)
    return standardized


def build_dataset(config: Mapping[str, Any]) -> Tuple[pd.DataFrame, List[str], pd.DatetimeIndex, pd.DatetimeIndex]:
    sheets = load_source_data(config)
    data_cfg = config["data"]
    universe = build_universe(sheets[data_cfg["universe_sheet"]])
    business_days = build_business_days(sheets[data_cfg["business_days_sheet"]])
    assets = universe["short_ticker"].tolist()

    prices = align_to_business_days(
        prepare_wide_panel(sheets[data_cfg["price_sheet"]], assets),
        business_days,
    )
    rebalance_dates = build_rebalance_dates(business_days)
    daily_returns = prices.pct_change()
    tradability_mask, start_dates = build_tradability_mask(prices, business_days, config)
    universe_returns = compute_equal_weight_universe_return(daily_returns, tradability_mask)
    betas = compute_rolling_beta(daily_returns, universe_returns, config)
    specific_target, residual_daily_returns = build_pca_specific_return_target(
        daily_returns=daily_returns,
        tradability_mask=tradability_mask,
        config=config,
    )
    specific_target = winsorize_cross_sectional_frame(specific_target, float(config["target"]["winsorize_z"]))

    feature_frames = build_feature_frames(
        sheets=sheets,
        universe=universe,
        business_days=business_days,
        prices=prices,
        daily_returns=daily_returns,
        residual_daily_returns=residual_daily_returns,
        betas=betas,
        config=config,
    )
    weekly_feature_frames = {name: frame.reindex(rebalance_dates) for name, frame in feature_frames.items()}
    feature_panel = cross_sectionally_standardize_features(panel_to_long(weekly_feature_frames), config)

    target_panel = specific_target.reindex(rebalance_dates).stack(dropna=False).rename("target_specific_return")
    target_panel.index.names = ["date", "asset"]
    tradable_panel = tradability_mask.reindex(rebalance_dates).stack(dropna=False).rename("is_tradable")
    tradable_panel.index.names = ["date", "asset"]

    sector_map = universe.set_index("short_ticker")["Sector"]
    name_map = universe.set_index("short_ticker")["Name"]
    meta_panel = pd.DataFrame(
        {
            "sector": feature_panel.index.get_level_values("asset").map(sector_map),
            "asset_name": feature_panel.index.get_level_values("asset").map(name_map),
        },
        index=feature_panel.index,
    )

    dataset = pd.concat([feature_panel, target_panel, tradable_panel, meta_panel], axis=1)
    dataset["feature_count"] = dataset[feature_panel.columns].notna().sum(axis=1)
    dataset["is_tradable"] = dataset["is_tradable"].fillna(False)
    dataset["is_target_available"] = dataset["target_specific_return"].notna()
    min_features_required = int(config["model"]["min_features_required"])
    dataset = dataset[(dataset["is_tradable"]) & (dataset["feature_count"] >= min_features_required)]
    dataset.attrs["tradability_start_dates"] = start_dates
    return dataset, list(feature_panel.columns), rebalance_dates, build_retrain_dates(rebalance_dates)


def mean_daily_spearman_ic(frame: pd.DataFrame, prediction_col: str, target_col: str) -> float:
    correlations: List[float] = []
    for _, group in frame.groupby(level="date"):
        subset = group[[prediction_col, target_col]].dropna()
        if len(subset) < 3:
            continue
        corr = subset[prediction_col].corr(subset[target_col], method="spearman")
        if pd.notna(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else float("nan")


def build_param_grid(param_grid_cfg: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid_cfg.keys())
    values = [list(param_grid_cfg[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_walk_forward_splits(
    train_dates: pd.DatetimeIndex,
    rebalance_positions: Mapping[pd.Timestamp, int],
    config: Mapping[str, Any],
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    val_cfg = config["model"]["validation"]
    min_train = int(val_cfg["min_train_rebalances"])
    val_window = int(val_cfg["validation_window_rebalances"])
    n_splits = int(val_cfg["n_splits"])
    purge_periods = int(val_cfg["purge_rebalances"])
    dates = pd.DatetimeIndex(sorted(train_dates))
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    end_idx = len(dates)

    while len(splits) < n_splits:
        val_start = end_idx - val_window
        if val_start <= 0:
            break
        val_dates = dates[val_start:end_idx]
        val_start_pos = rebalance_positions[pd.Timestamp(val_dates[0])]
        eligible_train = [
            date
            for date in dates[:val_start]
            if rebalance_positions[pd.Timestamp(date)] <= val_start_pos - purge_periods
        ]
        if len(eligible_train) < min_train:
            break
        splits.append((pd.DatetimeIndex(eligible_train), pd.DatetimeIndex(val_dates)))
        end_idx = val_start
    return list(reversed(splits))


def fit_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Mapping[str, Any],
    random_state: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=random_state, loss="squared_error", early_stopping=False, **params)
    model.fit(X_train, y_train)
    return model


def tune_model_for_retrain(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    train_dates: pd.DatetimeIndex,
    rebalance_positions: Mapping[pd.Timestamp, int],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DatetimeIndex]:
    param_grid = build_param_grid(config["model"]["param_grid"])
    random_state = int(config["model"]["random_state"])
    splits = build_walk_forward_splits(train_dates, rebalance_positions, config)
    if not splits:
        LOGGER.warning("Insufficient history for validation splits. Falling back to the first parameter set.")
        return param_grid[0], {"mean_ic": np.nan, "rmse": np.nan, "splits": 0}, pd.DatetimeIndex([])

    best_params = param_grid[0]
    best_score = -np.inf
    best_rmse = np.inf
    for params in param_grid:
        split_ics: List[float] = []
        split_rmses: List[float] = []
        for split_train_dates, split_val_dates in splits:
            fit_frame = training_frame.loc[training_frame.index.get_level_values("date").isin(split_train_dates)]
            val_frame = training_frame.loc[training_frame.index.get_level_values("date").isin(split_val_dates)]
            model = fit_single_model(
                fit_frame.loc[:, feature_columns],
                fit_frame["target_specific_return"],
                params=params,
                random_state=random_state,
            )
            scored = val_frame.loc[:, ["target_specific_return"]].copy()
            scored["prediction"] = model.predict(val_frame.loc[:, feature_columns])
            split_ics.append(mean_daily_spearman_ic(scored, "prediction", "target_specific_return"))
            split_rmses.append(float(np.sqrt(mean_squared_error(scored["target_specific_return"], scored["prediction"]))))
        mean_ic = float(np.nanmean(split_ics))
        mean_rmse = float(np.nanmean(split_rmses))
        if (mean_ic > best_score) or (np.isclose(mean_ic, best_score, equal_nan=False) and mean_rmse < best_rmse):
            best_score = mean_ic
            best_rmse = mean_rmse
            best_params = params
    return best_params, {"mean_ic": best_score, "rmse": best_rmse, "splits": len(splits)}, splits[-1][1]

def compute_feature_importance(
    model: HistGradientBoostingRegressor,
    validation_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int,
) -> pd.DataFrame:
    if validation_frame.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    importance = permutation_importance(
        model,
        validation_frame.loc[:, feature_columns],
        validation_frame["target_specific_return"],
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def maybe_generate_shap_summary(
    model: HistGradientBoostingRegressor,
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    shap_cfg = config["outputs"]["shap"]
    status: Dict[str, Any] = {"enabled": bool(shap_cfg["enabled"]), "saved": False}
    if not shap_cfg["enabled"]:
        return status
    try:
        import matplotlib.pyplot as plt
        import shap
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SHAP is unavailable: %s", exc)
        status["reason"] = str(exc)
        return status

    sample_size = min(int(shap_cfg["max_rows"]), len(training_frame))
    if sample_size < 25:
        status["reason"] = "Not enough rows for a stable SHAP summary."
        return status

    shap_frame = training_frame.loc[:, feature_columns].sample(
        n=sample_size,
        random_state=int(config["model"]["random_state"]),
    )
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(shap_frame)
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).to_csv(output_dir / "shap_feature_summary.csv", index=False)
        shap.summary_plot(shap_values, shap_frame, show=False, max_display=int(shap_cfg["max_features"]))
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close("all")
        status["saved"] = True
        return status
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping SHAP summary because the explainer was unstable: %s", exc)
        status["reason"] = str(exc)
        return status


def build_training_cutoff_date(
    retrain_date: pd.Timestamp,
    rebalance_dates: pd.DatetimeIndex,
    rebalance_positions: Mapping[pd.Timestamp, int],
    config: Mapping[str, Any],
) -> pd.Timestamp | None:
    horizon = int(config["target"]["horizon_bdays"])
    execution_lag = int(config["target"]["execution_lag_bdays"])
    rebalance_gap = max(1, int(np.ceil((horizon + execution_lag + 1) / 5.0)))
    retrain_pos = rebalance_positions[retrain_date]
    eligible = [date for date in rebalance_dates if rebalance_positions[date] <= retrain_pos - rebalance_gap]
    return pd.Timestamp(eligible[-1]) if eligible else None


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value)} is not JSON serializable.")


def train_alpha_engine(config: Mapping[str, Any]) -> Dict[str, Any]:
    results_dir = Path(config["paths"]["results_dir"])
    setup_logging(results_dir)
    LOGGER.info("Building the weekly alpha dataset.")

    dataset, feature_columns, rebalance_dates, retrain_dates = build_dataset(config)
    rebalance_positions = {pd.Timestamp(date): idx for idx, date in enumerate(rebalance_dates)}
    random_state = int(config["model"]["random_state"])
    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: List[pd.DataFrame] = []
    importance_history: List[pd.DataFrame] = []
    retrain_records: List[Dict[str, Any]] = []
    latest_model: HistGradientBoostingRegressor | None = None
    latest_training_frame = pd.DataFrame()

    for idx, retrain_date in enumerate(retrain_dates):
        cutoff_date = build_training_cutoff_date(retrain_date, rebalance_dates, rebalance_positions, config)
        if cutoff_date is None:
            LOGGER.info("Skipping retrain date %s because history is not long enough.", retrain_date.date())
            continue
        training_frame = dataset.loc[dataset.index.get_level_values("date") <= cutoff_date].copy()
        training_frame = training_frame[training_frame["is_target_available"]].copy()
        if training_frame.empty:
            continue

        unique_train_dates = pd.DatetimeIndex(sorted(training_frame.index.get_level_values("date").unique()))
        max_train_rebalances = int(config["model"].get("max_train_rebalances", 0))
        if max_train_rebalances > 0 and len(unique_train_dates) > max_train_rebalances:
            keep_dates = unique_train_dates[-max_train_rebalances:]
            training_frame = training_frame.loc[training_frame.index.get_level_values("date").isin(keep_dates)].copy()
            unique_train_dates = pd.DatetimeIndex(sorted(training_frame.index.get_level_values("date").unique()))
        best_params, validation_summary, last_validation_dates = tune_model_for_retrain(
            training_frame=training_frame,
            feature_columns=feature_columns,
            train_dates=unique_train_dates,
            rebalance_positions=rebalance_positions,
            config=config,
        )
        model = fit_single_model(
            training_frame.loc[:, feature_columns],
            training_frame["target_specific_return"],
            params=best_params,
            random_state=random_state,
        )
        latest_model = model
        latest_training_frame = training_frame

        model_path = model_dir / f"alpha_model_{retrain_date.strftime('%Y%m%d')}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": list(feature_columns),
                    "retrain_date": retrain_date.strftime("%Y-%m-%d"),
                    "params": best_params,
                },
                handle,
            )

        next_retrain = retrain_dates[idx + 1] if idx + 1 < len(retrain_dates) else None
        prediction_mask = dataset.index.get_level_values("date") >= retrain_date
        if next_retrain is not None:
            prediction_mask &= dataset.index.get_level_values("date") < next_retrain
        prediction_frame = dataset.loc[prediction_mask].copy()
        if prediction_frame.empty:
            continue
        prediction_frame["raw_prediction"] = model.predict(prediction_frame.loc[:, feature_columns])
        prediction_frame["linear_prediction"] = fit_linear_baseline(
            X_train=training_frame.loc[:, feature_columns],
            y_train=training_frame["target_specific_return"],
            X_pred=prediction_frame.loc[:, feature_columns],
            alpha=float(config["model"].get("linear_baseline_alpha", 1.0)),
        )
        prediction_frame["ai_specific_prediction"] = prediction_frame["raw_prediction"] - prediction_frame["linear_prediction"]
        prediction_frame["alpha_zscore"] = prediction_frame.groupby(level="date")["raw_prediction"].transform(zscore_series)
        prediction_frame["model_retrain_date"] = retrain_date.strftime("%Y-%m-%d")
        all_predictions.append(prediction_frame)

        if len(last_validation_dates) > 0:
            val_frame = training_frame.loc[training_frame.index.get_level_values("date").isin(last_validation_dates)]
            importance = compute_feature_importance(model, val_frame, feature_columns, random_state)
            if not importance.empty:
                importance["retrain_date"] = retrain_date.strftime("%Y-%m-%d")
                importance_history.append(importance)

        retrain_records.append(
            {
                "retrain_date": retrain_date.strftime("%Y-%m-%d"),
                "training_cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "training_rows": int(len(training_frame)),
                "training_rebalance_dates": int(len(unique_train_dates)),
                "best_params": best_params,
                "validation_mean_ic": validation_summary["mean_ic"],
                "validation_rmse": validation_summary["rmse"],
                "validation_splits": validation_summary["splits"],
                "model_path": str(model_path),
            }
        )

    if not all_predictions:
        raise RuntimeError("No quarterly models were trained. Check the history and config constraints.")

    predictions = pd.concat(all_predictions).sort_index()
    prediction_export = predictions.reset_index().loc[
        :,
        [
            "date",
            "asset",
            "asset_name",
            "sector",
            "model_retrain_date",
            "raw_prediction",
            "linear_prediction",
            "ai_specific_prediction",
            "alpha_zscore",
            "target_specific_return",
            "feature_count",
            "is_tradable",
        ],
    ]
    prediction_export.to_csv(results_dir / "alpha_predictions.csv", index=False)

    if importance_history:
        importance_frame = pd.concat(importance_history, ignore_index=True)
        importance_frame.to_csv(results_dir / "feature_importance_history.csv", index=False)
        aggregate_importance = (
            importance_frame.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_std=("importance_mean", "std"),
                observations=("importance_mean", "count"),
            )
            .sort_values("importance_mean", ascending=False)
        )
        aggregate_importance.to_csv(results_dir / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "observations"]).to_csv(
            results_dir / "feature_importance.csv", index=False
        )

    shap_status = {"enabled": False, "saved": False}
    if latest_model is not None and not latest_training_frame.empty:
        shap_status = maybe_generate_shap_summary(
            model=latest_model,
            training_frame=latest_training_frame,
            feature_columns=feature_columns,
            output_dir=results_dir,
            config=config,
        )

    scored_predictions = predictions.dropna(subset=["target_specific_return"]).copy()
    nonlinear_diagnostics = compute_nonlinear_share(
        raw_prediction=scored_predictions["raw_prediction"],
        linear_prediction=scored_predictions["linear_prediction"],
    )
    diagnostics = {
        "prediction_rows": int(len(prediction_export)),
        "prediction_dates": int(prediction_export["date"].nunique()),
        "feature_columns": list(feature_columns),
        "retrain_records": retrain_records,
        "tradability_start_dates": dataset.attrs.get("tradability_start_dates", {}),
        "overall_mean_daily_ic": mean_daily_spearman_ic(scored_predictions, "raw_prediction", "target_specific_return"),
        "overall_linear_mean_daily_ic": mean_daily_spearman_ic(scored_predictions, "linear_prediction", "target_specific_return"),
        "overall_ai_specific_mean_daily_ic": mean_daily_spearman_ic(scored_predictions, "ai_specific_prediction", "target_specific_return"),
        "overall_rmse": float(np.sqrt(mean_squared_error(scored_predictions["target_specific_return"], scored_predictions["raw_prediction"])))
        if not scored_predictions.empty
        else float("nan"),
        "overall_linear_rmse": float(np.sqrt(mean_squared_error(scored_predictions["target_specific_return"], scored_predictions["linear_prediction"])))
        if not scored_predictions.empty
        else float("nan"),
        "nonlinear_diagnostics": nonlinear_diagnostics,
        "shap_status": shap_status,
    }
    with (results_dir / "alpha_engine_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False, default=json_default)

    LOGGER.info("Saved predictions to %s", results_dir / "alpha_predictions.csv")
    LOGGER.info("Saved aggregated feature importance to %s", results_dir / "feature_importance.csv")
    return {
        "results_dir": str(results_dir),
        "predictions_path": str(results_dir / "alpha_predictions.csv"),
        "feature_importance_path": str(results_dir / "feature_importance.csv"),
        "diagnostics_path": str(results_dir / "alpha_engine_diagnostics.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ML alpha engine.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_alpha_engine(config)


if __name__ == "__main__":
    main()




















