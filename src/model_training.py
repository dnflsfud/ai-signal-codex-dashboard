"""Monthly walk-forward model training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

from src.utils import LOGGER, cross_sectional_spearman, save_csv, save_json


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _spearman_from_predictions(pred: pd.Series, target: pd.Series, dates: pd.Series) -> float:
    df = pd.DataFrame({"date": dates.values, "prediction": pred.values, "target": target.values})
    return cross_sectional_spearman(df, "prediction", "target")


def _prepare_rows(feature_panel: pd.DataFrame, target_panel: pd.DataFrame, target_name: str) -> pd.DataFrame:
    panel = feature_panel.join(target_panel[[target_name]], how="inner")
    panel = panel.rename(columns={target_name: "target"}).dropna(subset=["target"])
    panel = panel.reset_index()
    return panel


def _split_months(months: list[pd.Timestamp], train_window: int, val_months: int, seed: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if len(months) != train_window:
        raise ValueError("Training month window size mismatch.")
    # Temporal split: use the last val_months as validation to respect time ordering.
    val_months_list = months[-val_months:]
    train_months_list = months[:-val_months]
    return train_months_list, val_months_list


def _fit_pipeline(model: Any, X_train: pd.DataFrame, y_train: pd.Series, standardize: bool) -> Any:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline


def _fit_ffnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    batch_size: int = 256,
) -> np.ndarray:
    torch.manual_seed(seed)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(imputer.fit_transform(X_train))
    X_pred_arr = scaler.transform(imputer.transform(X_pred))
    y_train_arr = y_train.to_numpy(dtype=np.float32)

    x_tensor = torch.tensor(X_train_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
    pred_tensor = torch.tensor(X_pred_arr, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = FeedForwardNet(x_tensor.shape[1], hidden_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        predictions = model(pred_tensor).cpu().numpy()
    return predictions


def _score_ols(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_names: list[str]) -> tuple[float, Any]:
    model = _fit_pipeline(LinearRegression(), train_df[feature_names], train_df["target"], standardize=True)
    pred = pd.Series(model.predict(val_df[feature_names]), index=val_df.index)
    score = _spearman_from_predictions(pred, val_df["target"], val_df["date"])
    return score, model


def _evaluate_lasso_configs(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_names: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for alpha in config["models"]["lasso_grid"]["alpha"]:
        model = _fit_pipeline(Lasso(alpha=float(alpha), max_iter=10000, random_state=config["project"]["seed"]), train_df[feature_names], train_df["target"], standardize=True)
        pred = pd.Series(model.predict(val_df[feature_names]), index=val_df.index)
        score = _spearman_from_predictions(pred, val_df["target"], val_df["date"])
        results.append({"params": {"alpha": float(alpha)}, "score": score})
    return results


def _evaluate_lgbm_configs(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_names: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["models"]["lightgbm_grid"]
    results = []
    for learning_rate in grid["learning_rate"]:
        for num_leaves in grid["num_leaves"]:
            for min_data_in_leaf in grid["min_data_in_leaf"]:
                for feature_fraction in grid["feature_fraction"]:
                    params = {
                        "learning_rate": learning_rate,
                        "num_leaves": num_leaves,
                        "min_data_in_leaf": min_data_in_leaf,
                        "feature_fraction": feature_fraction,
                        "n_estimators": 300,
                        "subsample": 1.0,
                        "random_state": config["project"]["seed"],
                        "objective": "regression",
                        "verbosity": -1,
                    }
                    model = _fit_pipeline(LGBMRegressor(**params), train_df[feature_names], train_df["target"], standardize=False)
                    pred = pd.Series(model.predict(val_df[feature_names]), index=val_df.index)
                    score = _spearman_from_predictions(pred, val_df["target"], val_df["date"])
                    results.append({"params": params, "score": score})
    return results


def _evaluate_ffnn_configs(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_names: list[str], config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    grid = config["models"]["ff_nn_grid"]
    results = []
    for hidden_dim in grid["hidden_dim"]:
        for dropout in grid["dropout"]:
            for lr in grid["lr"]:
                for weight_decay in grid["weight_decay"]:
                    for epochs in grid["epochs"]:
                        params = {
                            "hidden_dim": hidden_dim,
                            "dropout": dropout,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "epochs": epochs,
                        }
                        pred = _fit_ffnn(train_df[feature_names], train_df["target"], val_df[feature_names], seed=seed, **params)
                        score = _spearman_from_predictions(pd.Series(pred, index=val_df.index), val_df["target"], val_df["date"])
                        results.append({"params": params, "score": score})
    return results


def _retain_top_configs(results: list[dict[str, Any]], top_fraction: float) -> list[dict[str, Any]]:
    if not results:
        return []
    ranked = sorted(results, key=lambda item: item["score"], reverse=True)
    keep = max(1, math.ceil(len(ranked) * top_fraction))
    return ranked[:keep]


def _fit_prediction_ensemble(model_name: str, full_train_df: pd.DataFrame, pred_df: pd.DataFrame, feature_names: list[str], kept: list[dict[str, Any]], config: dict[str, Any], seed: int) -> np.ndarray:
    if model_name == "lasso":
        preds = []
        for item in kept:
            model = _fit_pipeline(Lasso(alpha=float(item["params"]["alpha"]), max_iter=10000, random_state=config["project"]["seed"]), full_train_df[feature_names], full_train_df["target"], standardize=True)
            preds.append(model.predict(pred_df[feature_names]))
        return np.mean(preds, axis=0)
    if model_name == "lightgbm":
        preds = []
        for item in kept:
            model = _fit_pipeline(LGBMRegressor(**item["params"]), full_train_df[feature_names], full_train_df["target"], standardize=False)
            preds.append(model.predict(pred_df[feature_names]))
        return np.mean(preds, axis=0)
    if model_name == "ffnn":
        preds = []
        for idx, item in enumerate(kept):
            preds.append(_fit_ffnn(full_train_df[feature_names], full_train_df["target"], pred_df[feature_names], seed=seed + idx, **item["params"]))
        return np.mean(preds, axis=0)
    raise KeyError(model_name)


def train_models_walk_forward(bundle: dict[str, Any], feature_payload: dict[str, Any], target_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    monthly_features = feature_payload["model_panel"]
    target_panel = target_payload["target_panel"]
    target_name = target_payload["target_name"]
    frame = _prepare_rows(monthly_features, target_panel, target_name)
    feature_names = feature_payload["feature_names"]
    months = sorted(pd.Timestamp(date) for date in frame["date"].unique())

    if config.get("run", {}).get("stage") == "smoke":
        months = months[-int(config["run"].get("smoke_months", 36)):]
        frame = frame[frame["date"].isin(months)].copy()

    train_window = int(config["models"]["train_window_months"])
    val_months = int(config["models"]["validation_months"])
    top_fraction = float(config["models"]["top_config_fraction"])
    seed = int(config["project"]["seed"])
    requested_models = list(config["run"]["models"])

    predictions_by_model: dict[str, pd.DataFrame] = {}
    model_reports: dict[str, Any] = {}

    for model_name in requested_models:
        prediction_rows = []
        diagnostics_rows = []
        for month_idx in range(train_window, len(months)):
            pred_month = months[month_idx]
            window_months = months[month_idx - train_window:month_idx]
            train_months, val_months_list = _split_months(window_months, train_window, val_months, seed + month_idx)
            train_df = frame[frame["date"].isin(train_months)].copy()
            val_df = frame[frame["date"].isin(val_months_list)].copy()
            full_train_df = frame[frame["date"].isin(window_months)].copy()
            pred_df = frame[frame["date"] == pred_month].copy()
            if pred_df.empty:
                continue

            if model_name == "ols":
                val_score, model = _score_ols(train_df, val_df, feature_names)
                pred_values = model.predict(pred_df[feature_names])
                kept = [{"params": {}, "score": val_score}]
            elif model_name == "lasso":
                results = _evaluate_lasso_configs(train_df, val_df, feature_names, config)
                kept = _retain_top_configs(results, top_fraction)
                pred_values = _fit_prediction_ensemble("lasso", full_train_df, pred_df, feature_names, kept, config, seed + month_idx)
                val_score = float(np.mean([item["score"] for item in kept]))
            elif model_name == "lightgbm":
                results = _evaluate_lgbm_configs(train_df, val_df, feature_names, config)
                kept = _retain_top_configs(results, top_fraction)
                pred_values = _fit_prediction_ensemble("lightgbm", full_train_df, pred_df, feature_names, kept, config, seed + month_idx)
                val_score = float(np.mean([item["score"] for item in kept]))
            elif model_name == "ffnn":
                results = _evaluate_ffnn_configs(train_df, val_df, feature_names, config, seed + month_idx)
                kept = _retain_top_configs(results, top_fraction)
                pred_values = _fit_prediction_ensemble("ffnn", full_train_df, pred_df, feature_names, kept, config, seed + month_idx)
                val_score = float(np.mean([item["score"] for item in kept]))
            else:
                raise KeyError(model_name)

            pred_df = pred_df.copy()
            pred_df["prediction"] = pred_values
            pred_df["model"] = model_name
            prediction_rows.append(pred_df[["date", "asset", "target", "prediction", "model"]])
            diagnostics_rows.append({
                "date": pred_month,
                "model": model_name,
                "validation_rank_ic": val_score,
                "retained_configs": len(kept),
                "kept_config_details": kept,
            })

        if prediction_rows:
            prediction_frame = pd.concat(prediction_rows, ignore_index=True)
            save_csv(prediction_frame, output_paths["predictions"] / f"monthly_predictions_{model_name}.csv", index=False)
            predictions_by_model[model_name] = prediction_frame
            model_reports[model_name] = diagnostics_rows
            save_json(diagnostics_rows, output_paths["models"] / f"training_report_{model_name}.json")
            LOGGER.info("Finished monthly walk-forward predictions for model %s.", model_name)

    return {
        "predictions_by_model": predictions_by_model,
        "feature_names": feature_names,
        "model_reports": model_reports,
    }





def generate_latest_inference(feature_payload: dict[str, Any], target_payload: dict[str, Any], training_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    monthly_features = feature_payload["model_panel"]
    target_panel = target_payload["target_panel"]
    target_name = target_payload["target_name"]
    labeled_frame = _prepare_rows(monthly_features, target_panel, target_name)
    all_features = monthly_features.reset_index()
    feature_names = feature_payload["feature_names"]
    all_months = sorted(pd.Timestamp(date) for date in all_features["date"].unique())
    labeled_months = sorted(pd.Timestamp(date) for date in labeled_frame["date"].unique())
    latest_labeled = max(labeled_months)
    infer_months = [month for month in all_months if month > latest_labeled]
    if not infer_months:
        return {}

    train_window = int(config["models"]["train_window_months"])
    seed = int(config["project"]["seed"])
    outputs = {}
    for model_name in config["run"]["models"]:
        reports = training_payload["model_reports"].get(model_name, [])
        if not reports:
            continue
        kept = reports[-1].get("kept_config_details", [])
        if not kept:
            continue
        rows = []
        for idx, infer_month in enumerate(infer_months):
            available_months = [month for month in labeled_months if month < infer_month]
            if len(available_months) < train_window:
                continue
            train_months = available_months[-train_window:]
            full_train_df = labeled_frame[labeled_frame["date"].isin(train_months)].copy()
            pred_df = all_features[all_features["date"] == infer_month].copy()
            if pred_df.empty:
                continue
            if model_name == "ols":
                model = _fit_pipeline(LinearRegression(), full_train_df[feature_names], full_train_df["target"], standardize=True)
                pred_values = model.predict(pred_df[feature_names])
            elif model_name == "lasso":
                pred_values = _fit_prediction_ensemble("lasso", full_train_df, pred_df, feature_names, kept, config, seed + idx)
            elif model_name == "lightgbm":
                pred_values = _fit_prediction_ensemble("lightgbm", full_train_df, pred_df, feature_names, kept, config, seed + idx)
            elif model_name == "ffnn":
                pred_values = _fit_prediction_ensemble("ffnn", full_train_df, pred_df, feature_names, kept, config, seed + idx)
            else:
                continue
            pred_month = pred_df[["date", "asset"]].copy()
            pred_month["prediction"] = pred_values
            pred_month["model"] = model_name
            pred_month["is_latest_inference"] = True
            rows.append(pred_month)
        if rows:
            frame = pd.concat(rows, ignore_index=True)
            save_csv(frame, output_paths["predictions"] / f"latest_inference_{model_name}.csv", index=False)
            outputs[model_name] = frame
    return outputs
