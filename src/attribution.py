"""Paper-style functional attribution for LightGBM and FFNN."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.model_training import FeedForwardNet, _fit_pipeline
from src.utils import LOGGER, save_csv, save_json


COMPONENT_COLUMNS = [
    "total_prediction",
    "linear_component",
    "nonlinear_component",
    "interaction_component",
    "sector_etf_relative_component",
    "sector_etf_regime_component",
]


def _build_full_frame(feature_payload: dict[str, Any], target_payload: dict[str, Any]) -> pd.DataFrame:
    feature_panel = feature_payload["model_panel"].reset_index()
    target_name = target_payload["target_name"]
    target_panel = target_payload["target_panel"][[target_name]].reset_index().rename(columns={target_name: "target"})
    frame = feature_panel.merge(target_panel, on=["date", "asset"], how="inner")
    return frame.dropna(subset=["target"]).copy()


def _fit_ffnn_artifact(train_df: pd.DataFrame, feature_names: list[str], params: dict[str, Any], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feature_names]))
    y_train = train_df["target"].to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model = FeedForwardNet(x_tensor.shape[1], int(params["hidden_dim"]), float(params["dropout"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(int(params["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    return {"imputer": imputer, "scaler": scaler, "model": model}


def _predict_ffnn_ensemble(artifacts: list[dict[str, Any]], frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    preds = []
    for artifact in artifacts:
        X = artifact["scaler"].transform(artifact["imputer"].transform(frame[feature_names]))
        x_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds.append(artifact["model"](x_tensor).cpu().numpy())
    return np.mean(preds, axis=0)


def _fit_lightgbm_ensemble(train_df: pd.DataFrame, feature_names: list[str], kept_configs: list[dict[str, Any]]) -> list[Any]:
    return [
        _fit_pipeline(LGBMRegressor(**item["params"]), train_df[feature_names], train_df["target"], standardize=False)
        for item in kept_configs
    ]


def _predict_lightgbm_ensemble(models: list[Any], frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return np.mean([model.predict(frame[feature_names]) for model in models], axis=0)


def _prepare_predict_fn(model_name: str, train_df: pd.DataFrame, feature_names: list[str], kept_configs: list[dict[str, Any]], seed: int) -> Callable[[pd.DataFrame], np.ndarray]:
    if model_name == "lightgbm":
        models = _fit_lightgbm_ensemble(train_df, feature_names, kept_configs)
        return lambda frame: _predict_lightgbm_ensemble(models, frame, feature_names)
    if model_name == "ffnn":
        artifacts = [_fit_ffnn_artifact(train_df, feature_names, item["params"], seed + idx) for idx, item in enumerate(kept_configs)]
        return lambda frame: _predict_ffnn_ensemble(artifacts, frame, feature_names)
    raise KeyError(model_name)


def _get_training_window(full_frame: pd.DataFrame, pred_date: pd.Timestamp, train_window_months: int) -> pd.DataFrame:
    months = sorted(pd.Timestamp(date) for date in full_frame["date"].unique())
    month_idx = months.index(pd.Timestamp(pred_date))
    if month_idx < train_window_months:
        return pd.DataFrame()
    train_months = months[month_idx - train_window_months:month_idx]
    return full_frame[full_frame["date"].isin(train_months)].copy()


def _fit_partial_dependence_curve(predict_fn: Callable[[pd.DataFrame], np.ndarray], background: pd.DataFrame, feature: str, grid_points: int) -> dict[str, Any] | None:
    valid = background[feature].dropna()
    if valid.nunique() < 2:
        return None
    low = float(valid.min())
    high = float(valid.max())
    if np.isclose(low, high):
        return None
    grid = np.linspace(low, high, grid_points)
    curve = np.zeros(grid_points, dtype=float)
    for idx, value in enumerate(grid):
        modified = background.copy()
        modified[feature] = value
        curve[idx] = float(np.mean(predict_fn(modified)))
    linear_coef = np.polyfit(grid, curve, deg=1)
    linear_curve = np.polyval(linear_coef, grid)
    fill_value = float(valid.median())
    mean_total = float(np.mean(np.interp(background[feature].fillna(fill_value).to_numpy(dtype=float), grid, curve)))
    mean_linear = float(np.mean(np.interp(background[feature].fillna(fill_value).to_numpy(dtype=float), grid, linear_curve)))
    return {
        "grid": grid,
        "curve": curve,
        "linear_curve": linear_curve,
        "fill_value": fill_value,
        "mean_total": mean_total,
        "mean_linear": mean_linear,
    }


def _cross_sectional_ic(frame: pd.DataFrame, column: str) -> float:
    subset = frame[[column, "target"]].dropna()
    if len(subset) < 5:
        return float("nan")
    return float(subset[column].corr(subset["target"], method="spearman"))


def _monthly_long_short_return(scores: pd.Series, realized: pd.Series) -> float:
    ranked = scores.dropna().sort_values()
    if len(ranked) < 6:
        return float("nan")
    bucket = max(int(np.ceil(0.10 * len(ranked))), 3)
    short_assets = ranked.head(bucket).index
    long_assets = ranked.tail(bucket).index
    return float(realized.reindex(long_assets).mean() - realized.reindex(short_assets).mean())


def _monthly_turnover(scores: pd.Series, previous_weights: pd.Series | None) -> tuple[float, pd.Series]:
    ranked = scores.dropna().sort_values()
    if len(ranked) < 6:
        return float("nan"), pd.Series(dtype=float)
    bucket = max(int(np.ceil(0.10 * len(ranked))), 3)
    short_assets = ranked.head(bucket).index
    long_assets = ranked.tail(bucket).index
    weights = pd.Series(0.0, index=ranked.index)
    weights.loc[long_assets] = 1.0 / bucket
    weights.loc[short_assets] = -1.0 / bucket
    if previous_weights is None:
        turnover = float(weights.abs().sum())
    else:
        union = sorted(set(weights.index).union(previous_weights.index))
        turnover = 0.5 * float((weights.reindex(union).fillna(0.0) - previous_weights.reindex(union).fillna(0.0)).abs().sum())
    return turnover, weights


def _metric_summary(component_predictions: pd.DataFrame, column: str) -> dict[str, Any]:
    monthly_ics = []
    monthly_returns = []
    monthly_turnovers = []
    previous_weights = None
    for _, group in component_predictions.groupby("date"):
        ic = _cross_sectional_ic(group, column)
        if np.isfinite(ic):
            monthly_ics.append(ic)
        monthly_ret = _monthly_long_short_return(group.set_index("asset")[column], group.set_index("asset")["target"])
        if np.isfinite(monthly_ret):
            monthly_returns.append(monthly_ret)
        turnover, previous_weights = _monthly_turnover(group.set_index("asset")[column], previous_weights)
        if np.isfinite(turnover):
            monthly_turnovers.append(turnover)
    series = component_predictions[column].dropna()
    total_series = component_predictions["total_prediction"].dropna()
    variance_share = float(series.var(ddof=0) / total_series.var(ddof=0)) if len(series) > 1 and total_series.var(ddof=0) > 0 else float("nan")
    return {
        "mean_rank_ic": float(np.nanmean(monthly_ics)) if monthly_ics else float("nan"),
        "positive_ic_ratio": float(np.mean(np.array(monthly_ics) > 0.0)) if monthly_ics else float("nan"),
        "mean_long_short_return": float(np.nanmean(monthly_returns)) if monthly_returns else float("nan"),
        "positive_return_ratio": float(np.mean(np.array(monthly_returns) > 0.0)) if monthly_returns else float("nan"),
        "mean_turnover": float(np.nanmean(monthly_turnovers)) if monthly_turnovers else float("nan"),
        "months": int(component_predictions["date"].nunique()),
        "variance_share": variance_share,
    }


def _compute_month_components(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    pred_month: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_names: list[str],
    feature_dict: pd.DataFrame,
    grid_points: int,
    background_rows: int,
    seed: int,
) -> pd.DataFrame:
    pred_features = pred_month[feature_names].copy()
    total_prediction = pd.Series(predict_fn(pred_features), index=pred_month.index, dtype=float)
    background = train_df[feature_names].copy()
    if len(background) > background_rows:
        background = background.sample(n=background_rows, random_state=seed)
    intercept = float(np.mean(predict_fn(background.copy())))

    linear_sum = pd.Series(0.0, index=pred_month.index, dtype=float)
    nonlinear_sum = pd.Series(0.0, index=pred_month.index, dtype=float)
    additive_total_sum = pd.Series(0.0, index=pred_month.index, dtype=float)
    category_additive_total: dict[str, pd.Series] = {category: pd.Series(0.0, index=pred_month.index, dtype=float) for category in feature_dict["category"].unique()}
    category_linear: dict[str, pd.Series] = {category: pd.Series(0.0, index=pred_month.index, dtype=float) for category in feature_dict["category"].unique()}
    category_nonlinear_add: dict[str, pd.Series] = {category: pd.Series(0.0, index=pred_month.index, dtype=float) for category in feature_dict["category"].unique()}

    for feature in feature_names:
        pd_curve = _fit_partial_dependence_curve(predict_fn, background, feature, grid_points)
        if pd_curve is None:
            continue
        x_pred = pred_features[feature].fillna(pd_curve["fill_value"]).to_numpy(dtype=float)
        total_vals = np.interp(x_pred, pd_curve["grid"], pd_curve["curve"]) - pd_curve["mean_total"]
        linear_vals = np.interp(x_pred, pd_curve["grid"], pd_curve["linear_curve"]) - pd_curve["mean_linear"]
        nonlinear_vals = total_vals - linear_vals

        total_series = pd.Series(total_vals, index=pred_month.index, dtype=float)
        linear_series = pd.Series(linear_vals, index=pred_month.index, dtype=float)
        nonlinear_series = pd.Series(nonlinear_vals, index=pred_month.index, dtype=float)

        additive_total_sum = additive_total_sum.add(total_series, fill_value=0.0)
        linear_sum = linear_sum.add(linear_series, fill_value=0.0)
        nonlinear_sum = nonlinear_sum.add(nonlinear_series, fill_value=0.0)

        category = str(feature_dict.loc[feature, "category"])
        category_additive_total[category] = category_additive_total[category].add(total_series, fill_value=0.0)
        category_linear[category] = category_linear[category].add(linear_series, fill_value=0.0)
        category_nonlinear_add[category] = category_nonlinear_add[category].add(nonlinear_series, fill_value=0.0)

    linear_component = intercept + linear_sum
    nonlinear_component = nonlinear_sum
    interaction_component = total_prediction - intercept - additive_total_sum

    component_frame = pd.DataFrame(
        {
            "total_prediction": total_prediction,
            "linear_component": linear_component,
            "nonlinear_component": nonlinear_component,
            "interaction_component": interaction_component,
        }
    )

    for category in feature_dict["category"].unique():
        category_features = feature_dict.index[feature_dict["category"] == category].tolist()
        category_effect = pd.Series(0.0, index=pred_month.index, dtype=float)
        if category_features:
            modified = pred_features.copy()
            reference_values = background[category_features].median(axis=0)
            for feature in category_features:
                modified[feature] = reference_values[feature]
            category_effect = total_prediction - pd.Series(predict_fn(modified), index=pred_month.index, dtype=float)
        category_interaction = category_effect - category_additive_total[category]
        category_nonlinear_total = category_effect - category_linear[category]
        safe_name = category.lower()
        component_frame[f"category_{safe_name}_total_effect"] = category_effect
        component_frame[f"category_{safe_name}_linear_effect"] = category_linear[category]
        component_frame[f"category_{safe_name}_nonlinear_effect"] = category_nonlinear_total
        component_frame[f"category_{safe_name}_interaction_effect"] = category_interaction

    component_frame["sector_etf_relative_component"] = component_frame["category_stock_sector_relative_total_effect"]
    component_frame["sector_etf_regime_component"] = component_frame["category_global_sector_regime_total_effect"]
    component_frame["decomposition_residual"] = total_prediction - (
        component_frame["linear_component"] + component_frame["nonlinear_component"] + component_frame["interaction_component"]
    )
    return component_frame


def _summarize_components(component_predictions: pd.DataFrame, feature_dict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in COMPONENT_COLUMNS:
        summary = _metric_summary(component_predictions, component)
        rows.append({
            "row_type": "overall_component",
            "name": component,
            **summary,
            "linear_share": np.nan,
            "nonlinear_share": np.nan,
            "interaction_share": np.nan,
            "total_share": summary["variance_share"],
            "decomposition_residual_max_abs": float(component_predictions["decomposition_residual"].abs().max()),
        })

    for category in feature_dict["category"].unique():
        safe_name = category.lower()
        total_col = f"category_{safe_name}_total_effect"
        linear_col = f"category_{safe_name}_linear_effect"
        nonlinear_col = f"category_{safe_name}_nonlinear_effect"
        interaction_col = f"category_{safe_name}_interaction_effect"
        total_summary = _metric_summary(component_predictions, total_col)
        linear_summary = _metric_summary(component_predictions, linear_col)
        nonlinear_summary = _metric_summary(component_predictions, nonlinear_col)
        interaction_summary = _metric_summary(component_predictions, interaction_col)
        rows.append({
            "row_type": "category",
            "name": category,
            "mean_rank_ic": total_summary["mean_rank_ic"],
            "positive_ic_ratio": total_summary["positive_ic_ratio"],
            "mean_long_short_return": total_summary["mean_long_short_return"],
            "positive_return_ratio": total_summary["positive_return_ratio"],
            "mean_turnover": total_summary["mean_turnover"],
            "months": total_summary["months"],
            "variance_share": total_summary["variance_share"],
            "linear_share": linear_summary["variance_share"],
            "nonlinear_share": nonlinear_summary["variance_share"],
            "interaction_share": interaction_summary["variance_share"],
            "total_share": total_summary["variance_share"],
            "linear_rank_ic": linear_summary["mean_rank_ic"],
            "nonlinear_rank_ic": nonlinear_summary["mean_rank_ic"],
            "interaction_rank_ic": interaction_summary["mean_rank_ic"],
            "decomposition_residual_max_abs": float(component_predictions["decomposition_residual"].abs().max()),
        })
    return pd.DataFrame(rows)


def run_functional_attribution(bundle: dict[str, Any], feature_payload: dict[str, Any], target_payload: dict[str, Any], training_payload: dict[str, Any], portfolio_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Any]) -> dict[str, Any]:
    full_frame = _build_full_frame(feature_payload, target_payload)
    feature_names = feature_payload["feature_names"]
    feature_dict = feature_payload["feature_dict"].set_index("feature")
    reports = training_payload["model_reports"]
    train_window_months = int(config["models"]["train_window_months"])
    seed = int(config["project"]["seed"])
    attr_cfg = config.get("attribution", {})
    grid_points = int(attr_cfg.get("grid_points", 50))
    background_rows = int(attr_cfg.get("background_rows", 32))

    attribution_payload: dict[str, Any] = {}
    for model_name in ["lightgbm", "ffnn"]:
        if model_name not in reports or model_name not in training_payload["predictions_by_model"]:
            continue
        component_frames = []
        for idx, report in enumerate(reports[model_name]):
            pred_date = pd.Timestamp(report["date"])
            pred_month = full_frame[full_frame["date"] == pred_date].copy()
            train_df = _get_training_window(full_frame, pred_date, train_window_months)
            kept_configs = report.get("kept_config_details", [])
            if pred_month.empty or train_df.empty or not kept_configs:
                continue
            predict_fn = _prepare_predict_fn(model_name, train_df, feature_names, kept_configs, seed + idx)
            component_frame = _compute_month_components(
                predict_fn=predict_fn,
                pred_month=pred_month,
                train_df=train_df,
                feature_names=feature_names,
                feature_dict=feature_dict,
                grid_points=grid_points,
                background_rows=background_rows,
                seed=seed + idx,
            )
            component_frame.insert(0, "date", pred_date)
            component_frame.insert(1, "asset", pred_month["asset"].values)
            component_frame.insert(2, "model", model_name)
            component_frame["target"] = pred_month["target"].values
            component_frames.append(component_frame)

        if not component_frames:
            continue
        component_predictions = pd.concat(component_frames, ignore_index=True)
        summary_df = _summarize_components(component_predictions, feature_dict)
        save_csv(component_predictions, output_paths["attribution"] / f"component_predictions_{model_name}.csv", index=False)
        save_csv(summary_df, output_paths["attribution"] / f"attribution_summary_{model_name}.csv", index=False)
        save_json(
            {
                "model": model_name,
                "grid_points": grid_points,
                "background_rows": background_rows,
                "decomposition_residual_max_abs": float(component_predictions["decomposition_residual"].abs().max()),
            },
            output_paths["attribution"] / f"attribution_meta_{model_name}.json",
        )
        attribution_payload[model_name] = {"component_predictions": component_predictions, "summary": summary_df}
        LOGGER.info("Saved attribution outputs for model %s.", model_name)
    return attribution_payload
