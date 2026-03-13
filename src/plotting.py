"""Figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def generate_figures(bundle: dict[str, Any], training_payload: dict[str, Any], portfolio_payload: dict[str, Any], attribution_payload: dict[str, Any], output_paths: dict[str, Path]) -> None:
    perf = portfolio_payload.get("performance_summary")
    if perf is None or perf.empty:
        return
    for (model, portfolio_type), group in perf.groupby(["model", "portfolio_type"]):
        frame = group.sort_values("date").copy()
        frame["cum_gross"] = (1.0 + frame["gross_return"]).cumprod()
        frame["cum_net"] = (1.0 + frame["net_return"]).cumprod()
        plt.figure(figsize=(10, 4))
        plt.plot(frame["date"], frame["cum_gross"], label="Gross")
        plt.plot(frame["date"], frame["cum_net"], label="Net")
        plt.title(f"{model} {portfolio_type} cumulative returns")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_paths["figures"] / f"cum_returns_{model}_{portfolio_type}.png", dpi=200)
        plt.close()
