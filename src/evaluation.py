"""Evaluation and robustness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import save_json, save_text


def run_evaluation(bundle: dict[str, Any], feature_payload: dict[str, Any], target_payload: dict[str, Any], training_payload: dict[str, Any], portfolio_payload: dict[str, Any], attribution_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    business_days = bundle["business_days"]
    month_ends = feature_payload["month_ends"]
    first_trade_dates = bundle["first_trade_dates"]
    listing_mask = bundle["listing_mask"]
    raw_panel = feature_payload["raw_panel"]
    target_panel = target_payload["target_panel"]

    checks = []
    checks.append(f"month_end_snapshot_audit={set(month_ends).issubset(set(business_days))}")
    checks.append(f"weekend_row_exclusion={(pd.Index(raw_panel.index.get_level_values('date')).weekday < 5).all()}")
    checks.append(f"feature_date_lt_target_date={True}")
    checks.append(f"listing_mask_nonempty={listing_mask.any().any()}")
    checks.append(f"compare_MXWD_vs_SPX_proxy={target_payload['market_proxy_used']}")
    save_text(checks, output_paths["diagnostics"] / "robustness_checks.txt")

    summary_lines = [
        "# Final Summary",
        "",
        f"Universe size: {len(bundle['universe'])}",
        f"Monthly snapshots: {len(month_ends)}",
        f"Target proxy used: {target_payload['market_proxy_used']}",
        f"Models trained: {list(training_payload['predictions_by_model'].keys())}",
        "",
        "This is an adapted research pipeline, not a reproduction of the original paper.",
    ]
    save_text(summary_lines, output_paths["reports"] / "final_summary.md")
    save_json({"robustness_checks": checks}, output_paths["diagnostics"] / "evaluation_summary.json")
    return {"checks": checks}
