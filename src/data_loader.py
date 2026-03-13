"""Workbook loading and strict schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.universe import build_listing_mask, detect_first_real_trade_dates, prepare_universe
from src.utils import LOGGER, save_parquet, save_text


def excel_serial_to_datetime(values: pd.Series) -> pd.Series:
    series = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series.dropna()):
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
    parsed = pd.to_datetime(series, errors="coerce")
    serial_mask = numeric.between(10000, 100000)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric.loc[serial_mask], unit="D")
    return parsed


def wide_sheet_to_panel(df: pd.DataFrame, expected_assets: list[str], rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    date_col = frame.columns[0]
    frame[date_col] = excel_serial_to_datetime(frame[date_col])
    frame = frame.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    rename_map = rename_map or {}
    frame = frame.rename(columns={col: rename_map.get(str(col).strip(), str(col).strip()) for col in frame.columns[1:]})
    panel = frame.set_index(date_col)
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.loc[:, [col for col in panel.columns if col in set(expected_assets)]].reindex(columns=expected_assets)
    panel.index.name = "date"
    return panel


def _resolve_factor_return_sheet(sheet_names: list[str], candidates: list[str]) -> str:
    found = [name for name in candidates if name in sheet_names]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"Multiple factor return sheets found: {found}")
    raise ValueError(f"None of the factor return sheet candidates found: {candidates}")


def _validate_required_sheets(sheet_names: list[str], required: list[str], factor_candidates: list[str]) -> tuple[list[str], str]:
    missing = [sheet for sheet in required if sheet not in sheet_names]
    factor_sheet = _resolve_factor_return_sheet(sheet_names, factor_candidates)
    return missing, factor_sheet


def _validate_sector_return_columns(factor_return_df: pd.DataFrame, required_sector_cols: list[str]) -> tuple[list[str], list[str]]:
    columns = [str(col).strip() for col in factor_return_df.columns]
    missing = [col for col in required_sector_cols if col not in columns]
    return columns, missing


def load_and_validate_workbook(config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    workbook_path = Path(config["project"]["workbook_path"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    xls = pd.ExcelFile(workbook_path)
    sheet_names = list(xls.sheet_names)
    required = list(config["data"]["required_sheets"])
    factor_candidates = list(config["data"]["factor_return_sheet_candidates"])
    missing_sheets, factor_return_sheet = _validate_required_sheets(sheet_names, required, factor_candidates)

    schema_lines = [f"Workbook: {workbook_path}", f"Available sheets: {sheet_names}"]
    if missing_sheets:
        schema_lines.append(f"Missing required sheets: {missing_sheets}")
        save_text(schema_lines, output_paths["diagnostics"] / "schema_check.txt")
        raise ValueError(f"Missing required sheets: {missing_sheets}")

    schema_lines.append(f"Resolved factor return sheet: {factor_return_sheet}")
    sheets_to_load = required + [factor_return_sheet]
    data = pd.read_excel(workbook_path, sheet_name=sheets_to_load, engine="openpyxl")

    universe = prepare_universe(data["Universe_Meta"])
    expected_assets = universe["ticker_short"].tolist()
    business_days = pd.DatetimeIndex(excel_serial_to_datetime(data["BusinessDays"].iloc[:, 0]).dropna().sort_values().drop_duplicates())

    factor_return_columns, missing_sector_cols = _validate_sector_return_columns(data[factor_return_sheet], list(config["data"]["sector_return_columns"]))
    sector_lines = [
        f"Factor return sheet: {factor_return_sheet}",
        f"Available factor return columns: {factor_return_columns}",
        "Sector return columns are treated as returns, not prices.",
    ]
    if missing_sector_cols:
        sector_lines.append(f"Missing required sector return columns: {missing_sector_cols}")
        save_text(sector_lines, output_paths["diagnostics"] / "sector_etf_detection.txt")
        raise ValueError(f"Missing required sector return columns: {missing_sector_cols}")
    sector_lines.append("All required sector return columns found.")
    save_text(sector_lines, output_paths["diagnostics"] / "sector_etf_detection.txt")

    stock_panels = {}
    rename_names = dict(zip(universe["Name"], universe["ticker_short"]))
    for sheet_name in required:
        if sheet_name in {"Universe_Meta", "BusinessDays", "Summary_Stats"}:
            continue
        if sheet_name in {"Sent_Trend_Momentum_Timeseries", "Sent_Trend_21d_Timeseries"}:
            panel = wide_sheet_to_panel(data[sheet_name], expected_assets, rename_map=rename_names)
        else:
            panel = wide_sheet_to_panel(data[sheet_name], expected_assets)
        panel = panel.reindex(business_days)
        stock_panels[sheet_name] = panel

    factor_returns = pd.read_excel(workbook_path, sheet_name=factor_return_sheet, engine="openpyxl")
    factor_returns.columns = [str(col).strip() for col in factor_returns.columns]
    factor_returns[factor_returns.columns[0]] = excel_serial_to_datetime(factor_returns.iloc[:, 0])
    factor_returns = factor_returns.dropna(subset=[factor_returns.columns[0]]).set_index(factor_returns.columns[0]).sort_index()
    factor_returns = factor_returns.apply(pd.to_numeric, errors="coerce").reindex(business_days)

    factor_prices = pd.read_excel(workbook_path, sheet_name="Factor_PX_LAST", engine="openpyxl")
    factor_prices.columns = [str(col).strip() for col in factor_prices.columns]
    factor_prices[factor_prices.columns[0]] = excel_serial_to_datetime(factor_prices.iloc[:, 0])
    factor_prices = factor_prices.dropna(subset=[factor_prices.columns[0]]).set_index(factor_prices.columns[0]).sort_index()
    factor_prices = factor_prices.apply(pd.to_numeric, errors="coerce").reindex(business_days)

    prices = stock_panels["PX_LAST"].copy()
    daily_returns_from_px = prices.pct_change()
    daily_returns_sheet = stock_panels["Daily_Returns"].copy()
    return_diff = (daily_returns_from_px - daily_returns_sheet).abs()
    max_return_diff = float(return_diff.max().max(skipna=True))
    schema_lines.append(f"Daily_Returns cross-check max absolute diff vs PX_LAST returns: {max_return_diff:.8f}")

    first_trade_dates = detect_first_real_trade_dates(prices)
    listing_mask = build_listing_mask(prices, first_trade_dates)
    schema_lines.append(f"Universe size: {len(expected_assets)}")
    schema_lines.append(f"Business day count: {len(business_days)}")
    schema_lines.append("Validation completed successfully.")
    save_text(schema_lines, output_paths["diagnostics"] / "schema_check.txt")

    cleaned_long = pd.concat(
        {
            "PX_LAST": prices.stack(dropna=False),
            "Daily_Returns_from_PX": daily_returns_from_px.stack(dropna=False),
            "Daily_Returns_sheet": daily_returns_sheet.stack(dropna=False),
            "CUR_MKT_CAP": stock_panels["CUR_MKT_CAP"].stack(dropna=False),
        },
        axis=1,
    ).reset_index()
    cleaned_long.columns = ["date", "asset", "PX_LAST", "Daily_Returns_from_PX", "Daily_Returns_sheet", "CUR_MKT_CAP"]
    cleaned_long["first_trade_date"] = cleaned_long["asset"].map(first_trade_dates)
    cleaned_long["listing_mask"] = [bool(listing_mask.loc[date, asset]) for date, asset in zip(cleaned_long["date"], cleaned_long["asset"])]
    save_parquet(cleaned_long.set_index(["date", "asset"]), output_paths["cleaned"] / "panel_daily.parquet")

    LOGGER.info("Loaded workbook with %s assets and %s business days.", len(expected_assets), len(business_days))
    return {
        "workbook_path": workbook_path,
        "sheet_names": sheet_names,
        "factor_return_sheet": factor_return_sheet,
        "universe": universe,
        "business_days": business_days,
        "stock_panels": stock_panels,
        "factor_returns": factor_returns,
        "factor_prices": factor_prices,
        "prices": prices,
        "daily_returns_from_px": daily_returns_from_px,
        "first_trade_dates": first_trade_dates,
        "listing_mask": listing_mask,
        "month_ends": pd.DatetimeIndex([]),
    }


