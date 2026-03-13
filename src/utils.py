"""Shared utilities for the ai_signal research project."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    return config


def setup_logging(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / "diagnostics" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    names = [
        "cleaned",
        "features",
        "models",
        "predictions",
        "portfolios",
        "attribution",
        "figures",
        "reports",
        "diagnostics",
    ]
    paths = {name: base_dir / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=True)


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def save_text(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def winsorize_cross_section(values: pd.Series, lower: float, upper: float) -> pd.Series:
    if values.dropna().empty:
        return values
    lo = values.quantile(lower)
    hi = values.quantile(upper)
    return values.clip(lower=lo, upper=hi)


def rank_center_cross_section(values: pd.Series) -> pd.Series:
    if values.dropna().empty:
        return pd.Series(np.nan, index=values.index)
    ranked = values.rank(method="average", pct=True)
    return ranked - 0.5


def business_month_end_dates(business_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    series = pd.Series(business_days, index=business_days)
    grouped = series.groupby(business_days.to_period("M"))
    return pd.DatetimeIndex(grouped.max().sort_values().values)


def cross_sectional_spearman(df: pd.DataFrame, prediction_col: str, target_col: str) -> float:
    scores = []
    for _, group in df.groupby("date"):
        subset = group[[prediction_col, target_col]].dropna()
        if len(subset) < 5:
            continue
        corr = subset[prediction_col].corr(subset[target_col], method="spearman")
        if pd.notna(corr):
            scores.append(float(corr))
    return float(np.mean(scores)) if scores else float("nan")


def month_index_positions(dates: Sequence[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(date): idx for idx, date in enumerate(pd.DatetimeIndex(dates))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI signal research pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--stage", default=None, choices=["schema", "data", "features", "train", "portfolio", "attribution", "report", "full", "smoke"], help="Override stage")
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage is not None:
        config.setdefault("run", {})["stage"] = args.stage
    outputs_dir = Path(config.get("project", {}).get("outputs_dir", "outputs"))
    setup_logging(outputs_dir)
    set_seed(int(config["project"]["seed"]))
    from src.data_loader import load_and_validate_workbook
    from src.feature_engineering import build_monthly_feature_panels
    from src.target_builder import build_targets
    from src.model_training import train_models_walk_forward, generate_latest_inference
    from src.portfolio_construction import build_all_portfolios, build_latest_holdings
    from src.attribution import run_functional_attribution
    from src.evaluation import run_evaluation
    from src.plotting import generate_figures

    output_paths = ensure_output_dirs(Path(config["project"].get("outputs_dir", "outputs")))
    stage = config.get("run", {}).get("stage", "full")

    bundle = load_and_validate_workbook(config, output_paths)
    if stage == "schema":
        return
    feature_payload = build_monthly_feature_panels(bundle, config, output_paths)
    if stage == "features":
        return
    target_payload = build_targets(bundle, feature_payload, config, output_paths)
    if stage == "train":
        train_models_walk_forward(bundle, feature_payload, target_payload, config, output_paths)
        return
    training_payload = train_models_walk_forward(bundle, feature_payload, target_payload, config, output_paths)
    if stage == "portfolio":
        build_all_portfolios(bundle, feature_payload, target_payload, training_payload, config, output_paths)
        return
    portfolio_payload = build_all_portfolios(bundle, feature_payload, target_payload, training_payload, config, output_paths)
    if stage == "attribution":
        run_functional_attribution(bundle, feature_payload, target_payload, training_payload, portfolio_payload, config, output_paths)
        return
    attribution_payload = run_functional_attribution(bundle, feature_payload, target_payload, training_payload, portfolio_payload, config, output_paths)
    run_evaluation(bundle, feature_payload, target_payload, training_payload, portfolio_payload, attribution_payload, config, output_paths)
    generate_figures(bundle, training_payload, portfolio_payload, attribution_payload, output_paths)

    # Auto-sync reports to GitHub dashboard
    try:
        from sync_dashboard import sync_and_push
        sync_and_push(output_paths["reports"])
        LOGGER.info("Dashboard sync completed.")
    except Exception as exc:
        LOGGER.warning("Dashboard sync failed (non-fatal): %s", exc)



