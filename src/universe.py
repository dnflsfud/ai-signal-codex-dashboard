"""Universe metadata helpers."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

SECTOR_RETURN_MAP = {
    "technology": "SEC_InfoTech",
    "information technology": "SEC_InfoTech",
    "info tech": "SEC_InfoTech",
    "health care": "SEC_Health",
    "healthcare": "SEC_Health",
    "financials": "SEC_Financials",
    "consumer discretionary": "SEC_ConsDisc",
    "consumer staples": "SEC_ConsStap",
    "energy": "SEC_Energy",
    "industrials": "SEC_Industrials",
    "materials": "SEC_Materials",
    "utilities": "SEC_Utilities",
    "real estate": "SEC_RealEstate",
    "communication services": "SEC_CommSvc",
}


def map_sector_to_return_column(sector_name: str) -> str:
    key = str(sector_name).strip().lower()
    if key not in SECTOR_RETURN_MAP:
        raise KeyError(f"Unsupported sector mapping: {sector_name}")
    return SECTOR_RETURN_MAP[key]


def prepare_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    frame = universe_df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    required = ["Ticker", "Name", "Sector", "Status"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Universe_Meta missing columns: {missing}")
    frame["ticker_short"] = frame["Ticker"].astype(str).str.split().str[0]
    frame["sector_return_col"] = frame["Sector"].map(map_sector_to_return_column)
    if not frame["ticker_short"].is_unique:
        raise ValueError("Universe short tickers must be unique.")
    return frame.loc[:, ["Ticker", "ticker_short", "Name", "Sector", "Status", "sector_return_col"]]


def detect_first_real_trade_dates(prices: pd.DataFrame) -> pd.Series:
    """Detect first real trading date from first non-zero business-day return, using prior day as listing start."""
    returns = prices.pct_change()
    first_dates = {}
    for column in prices.columns:
        non_zero_idx = returns.index[(returns[column].abs() > 1.0e-12).fillna(False)]
        if len(non_zero_idx) == 0:
            first_dates[column] = pd.NaT
            continue
        first_move_date = pd.Timestamp(non_zero_idx[0])
        loc = prices.index.get_loc(first_move_date)
        first_dates[column] = pd.Timestamp(prices.index[max(0, loc - 1)])
    return pd.Series(first_dates)


def build_listing_mask(prices: pd.DataFrame, first_trade_dates: pd.Series) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for column in prices.columns:
        start = first_trade_dates.get(column)
        if pd.isna(start):
            continue
        mask.loc[mask.index >= pd.Timestamp(start), column] = True
    return mask


def month_end_business_dates(business_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    series = pd.Series(business_days, index=business_days)
    grouped = series.groupby(business_days.to_period("M"))
    return pd.DatetimeIndex(grouped.max().sort_values().values)


def sector_market_cap_weights(universe: pd.DataFrame, market_caps_row: pd.Series) -> pd.Series:
    mapping = universe.set_index("ticker_short")["Sector"]
    sector_caps = market_caps_row.groupby(mapping.reindex(market_caps_row.index)).sum()
    total = float(sector_caps.sum())
    return sector_caps / total if total > 0 else sector_caps * np.nan
