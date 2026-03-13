"""Target construction for the ai_signal research project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import save_parquet, save_text


def _compounded_forward_return(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return prices.shift(-horizon).divide(prices) - 1.0


def _rolling_beta(stock_returns: pd.DataFrame, market_returns: pd.Series, window: int) -> pd.DataFrame:
    cov = stock_returns.rolling(window, min_periods=window).cov(market_returns)
    var = market_returns.rolling(window, min_periods=window).var()
    return cov.divide(var, axis=0)


def build_targets(bundle: dict[str, Any], feature_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    prices = bundle["prices"]
    daily_returns = bundle["daily_returns_from_px"]
    listing_mask = bundle["listing_mask"]
    universe = bundle["universe"]
    month_ends = feature_payload["month_ends"]
    factor_prices = bundle["factor_prices"]
    factor_returns = bundle["factor_returns"]
    market_proxy = config["portfolio"]["market_proxy_primary"]
    fallback_proxy = config["portfolio"]["market_proxy_fallback"]
    beta_window = int(config["portfolio"]["beta_window_days"])
    vol_window = int(config["portfolio"]["vol_window_days"])

    market_returns = factor_returns[market_proxy] if market_proxy in factor_returns else factor_returns[fallback_proxy]
    proxy_used = market_proxy if market_proxy in factor_returns else fallback_proxy
    beta = _rolling_beta(daily_returns, market_returns, beta_window)

    future_return_21d = _compounded_forward_return(prices, 21)
    future_market_return_21d = _compounded_forward_return((1.0 + market_returns.fillna(0.0)).cumprod().to_frame(proxy_used), 21)[proxy_used]
    beta_neutral = future_return_21d.subtract(beta.multiply(future_market_return_21d, axis=0), axis=0)

    sector_map = universe.set_index("ticker_short")["Sector"]
    sector_demeaned = beta_neutral.copy()
    for date in sector_demeaned.index:
        row = sector_demeaned.loc[date]
        for sector, members in sector_map.groupby(sector_map).groups.items():
            idx = [asset for asset in members if asset in row.index]
            if not idx:
                continue
            sector_mean = row.loc[idx].mean(skipna=True)
            sector_demeaned.loc[date, idx] = row.loc[idx] - sector_mean

    idio_daily = daily_returns.subtract(beta.multiply(market_returns, axis=0), axis=0)
    idio_vol_63d = idio_daily.rolling(vol_window, min_periods=vol_window).std(ddof=0) * np.sqrt(252.0)
    total_vol_63d = daily_returns.rolling(vol_window, min_periods=vol_window).std(ddof=0) * np.sqrt(252.0)
    scale_vol = idio_vol_63d.where(idio_vol_63d.notna(), total_vol_63d).replace(0.0, np.nan)
    scaled_sector_beta_neutral = sector_demeaned.divide(scale_vol)

    target_variants = {
        "future_return_21d": future_return_21d.reindex(month_ends),
        "beta_neutral_future_return_21d": beta_neutral.reindex(month_ends),
        "sector_beta_neutral_future_return_21d": sector_demeaned.reindex(month_ends),
        "scaled_sector_beta_neutral_future_return_21d": scaled_sector_beta_neutral.reindex(month_ends),
    }

    target_panel = pd.concat(
        [frame.stack(dropna=False).rename(name) for name, frame in target_variants.items()],
        axis=1,
    )
    target_panel.index.names = ["date", "asset"]
    target_panel["listing_mask"] = [bool(listing_mask.loc[idx[0], idx[1]]) for idx in target_panel.index]
    target_panel = target_panel[target_panel["listing_mask"]].drop(columns=["listing_mask"])
    save_parquet(target_panel, output_paths["cleaned"] / "targets_monthly.parquet")

    diagnostics_lines = [
        f"Market proxy used: {proxy_used}",
        f"Month-end target rows: {len(target_panel)}",
        "Saved target variants: future_return_21d, beta_neutral_future_return_21d, sector_beta_neutral_future_return_21d, scaled_sector_beta_neutral_future_return_21d",
    ]
    save_text(diagnostics_lines, output_paths["diagnostics"] / "target_builder.txt")
    return {
        "target_panel": target_panel,
        "target_name": "scaled_sector_beta_neutral_future_return_21d",
        "beta_panel": beta.reindex(month_ends),
        "market_proxy_used": proxy_used,
        "market_returns": market_returns,
    }
