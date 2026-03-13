"""Exact 126-feature engineering for the ai_signal research project."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from src.universe import map_sector_to_return_column, month_end_business_dates
from src.utils import LOGGER, rank_center_cross_section, save_csv, save_parquet, winsorize_cross_section

STOCK_PRICE_FEATURES = [
    "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "short_rev_5d", "short_rev_21d",
    "mom_21_ex_5", "mom_63_ex_21", "mom_126_ex_21", "vol_21d", "vol_63d", "vol_126d",
    "downside_vol_21d", "downside_vol_63d", "beta_126d_to_MXWD", "beta_252d_to_MXWD",
    "idio_vol_63d_vs_MXWD", "resid_mom_63d_vs_MXWD", "max_drawdown_63d", "max_drawdown_126d",
    "log_mktcap", "sector_rel_ret_21d", "sector_rel_ret_63d",
]

STOCK_SECTOR_REL_FEATURES = [
    "sector_etf_ret_5d", "sector_etf_ret_21d", "sector_etf_ret_63d", "stock_minus_sector_etf_ret_5d",
    "stock_minus_sector_etf_ret_21d", "stock_minus_sector_etf_ret_63d", "beta_126d_to_sector_etf",
    "resid_mom_63d_vs_sector_etf",
]

FUNDAMENTAL_FEATURES = [
    "eps_level", "eps_delta_63d", "eps_delta_252d", "sales_level", "sales_delta_63d", "sales_delta_252d",
    "pe_level", "pe_delta_63d", "pe_delta_252d", "peg_level", "peg_delta_63d", "peg_delta_252d",
    "fcf_level", "fcf_delta_63d", "fcf_yield", "gross_margin_level", "gross_margin_delta_63d", "gross_margin_delta_252d",
    "oper_margin_level", "oper_margin_delta_63d", "oper_margin_delta_252d", "capex_level", "capex_delta_63d", "capex_to_sales",
    "roe_level", "roe_delta_63d", "roe_delta_252d", "px_bps_level", "px_bps_delta_63d", "px_bps_delta_252d",
    "ev_to_ebitda_level", "ev_to_ebitda_delta_63d", "ev_to_ebitda_delta_252d",
]

SENTIMENT_FEATURES = [
    "news_sent_mean_5d", "news_sent_mean_21d", "news_sent_mean_63d", "news_sent_delta_21d",
    "rec_cons_level", "rec_cons_delta_21d", "rec_cons_delta_63d",
    "sent_mom_level", "sent_mom_mean_5d", "sent_mom_mean_21d", "sent_mom_mean_63d",
    "sent_21d_level", "sent_21d_delta_21d", "sent_21d_delta_63d",
    "eps_revision_level", "eps_revision_delta_21d", "eps_revision_delta_63d", "eps_revision_delta_126d",
    "sales_revision_level", "sales_revision_delta_21d", "sales_revision_delta_63d", "sales_revision_delta_126d",
    "target_price_level", "target_price_upside", "target_price_delta_21d", "target_price_delta_63d",
]

GLOBAL_MACRO_FEATURES = [
    "mxwd_ret_5d", "mxwd_ret_21d", "mxwd_ret_63d", "spx_ret_21d", "ndx_minus_rty_ret_21d", "mxwd_minus_mxef_ret_21d",
    "vix_level_z_252d", "vix_delta_21d", "skew_level_z_252d", "ust_10y_level_z_252d", "ust_10y_delta_21d",
    "ust_curve_10y2y_level", "ust_curve_10y2y_delta_21d", "dxy_ret_21d", "usdkrw_ret_21d", "wti_ret_21d", "gold_ret_21d",
    "growth_minus_value_ret_21d", "quality_minus_hibeta_ret_21d", "gs_ai_ret_21d", "gs_semihw_ret_21d",
    "aaii_bull_minus_bear", "cesi_us_level_z_252d",
]

GLOBAL_SECTOR_REGIME_FEATURES = [
    "sector_eq_ret_5d", "sector_eq_ret_21d", "sector_eq_ret_63d", "sector_dispersion_21d", "sector_dispersion_63d",
    "sector_breadth_pos_21d", "sector_breadth_pos_63d", "consdisc_minus_constap_ret_21d",
    "industrials_minus_utilities_ret_21d", "infotech_minus_health_ret_21d", "financials_minus_utilities_ret_21d",
    "energy_minus_consdisc_ret_21d",
]

ALL_FEATURES = STOCK_PRICE_FEATURES + STOCK_SECTOR_REL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES + GLOBAL_MACRO_FEATURES + GLOBAL_SECTOR_REGIME_FEATURES
CATEGORY_MAP = {name: "stock_price" for name in STOCK_PRICE_FEATURES}
CATEGORY_MAP.update({name: "stock_sector_relative" for name in STOCK_SECTOR_REL_FEATURES})
CATEGORY_MAP.update({name: "fundamental_valuation" for name in FUNDAMENTAL_FEATURES})
CATEGORY_MAP.update({name: "sentiment_analyst" for name in SENTIMENT_FEATURES})
CATEGORY_MAP.update({name: "global_macro_factor" for name in GLOBAL_MACRO_FEATURES})
CATEGORY_MAP.update({name: "global_sector_regime" for name in GLOBAL_SECTOR_REGIME_FEATURES})


def _assert_feature_schema() -> None:
    if len(ALL_FEATURES) != 126:
        raise AssertionError(f"Exact feature count must be 126, got {len(ALL_FEATURES)}")
    if len(set(ALL_FEATURES)) != len(ALL_FEATURES):
        raise AssertionError("Feature names must be unique.")


def compounded_return_from_prices(prices: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    return prices.divide(prices.shift(window)) - 1.0


def compounded_return_from_daily_returns(returns: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    return (1.0 + returns).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0


def scaled_delta(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    lagged = panel.shift(window)
    return panel.subtract(lagged).divide(lagged.abs().clip(lower=1.0e-6))


def rolling_beta(stock_returns: pd.DataFrame, market_returns: pd.Series, window: int) -> pd.DataFrame:
    cov = stock_returns.rolling(window, min_periods=window).cov(market_returns)
    var = market_returns.rolling(window, min_periods=window).var()
    return cov.divide(var, axis=0)


def residual_vol(stock_returns: pd.DataFrame, market_returns: pd.Series, beta: pd.DataFrame, window: int) -> pd.DataFrame:
    residual = stock_returns.subtract(beta.mul(market_returns, axis=0), axis=0)
    return residual.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)


def rolling_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    peak = prices.rolling(window, min_periods=window).max()
    return prices.divide(peak) - 1.0


def _prepare_non_price_stock_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.shift(1)


def _broadcast_global_series(series: pd.Series, assets: list[str]) -> pd.DataFrame:
    values = np.repeat(series.to_numpy(dtype=float).reshape(-1, 1), len(assets), axis=1)
    return pd.DataFrame(values, index=series.index, columns=assets, dtype=float)


def _build_feature_dictionary() -> pd.DataFrame:
    _assert_feature_schema()
    return pd.DataFrame(
        [{"feature": feature, "category": CATEGORY_MAP[feature], "scope": "global" if CATEGORY_MAP[feature].startswith("global") else "stock"} for feature in ALL_FEATURES]
    )


def build_monthly_feature_panels(bundle: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    _assert_feature_schema()
    universe = bundle["universe"]
    assets = universe["ticker_short"].tolist()
    stock_panels = bundle["stock_panels"]
    factor_returns = bundle["factor_returns"]
    factor_prices = bundle["factor_prices"]
    prices = bundle["prices"]
    daily_returns = bundle["daily_returns_from_px"]
    listing_mask = bundle["listing_mask"]
    month_ends = month_end_business_dates(bundle["business_days"])
    bundle["month_ends"] = month_ends

    # Lag all non-price stock features by 1 business day.
    lagged_panels = {name: _prepare_non_price_stock_panel(panel) for name, panel in stock_panels.items() if name not in {"PX_LAST", "Daily_Returns"}}
    market_caps = lagged_panels["CUR_MKT_CAP"]

    sector_return_cols = config["data"]["sector_return_columns"]
    sector_returns = factor_returns.loc[:, sector_return_cols]
    sector_prices = (1.0 + sector_returns.fillna(0.0)).cumprod()
    mxwd = factor_prices["MXWD"] if "MXWD" in factor_prices else (1.0 + factor_returns["MXWD"].fillna(0.0)).cumprod()
    spx = factor_prices["SPX"] if "SPX" in factor_prices else (1.0 + factor_returns["SPX"].fillna(0.0)).cumprod()

    feature_frames: OrderedDict[str, pd.DataFrame] = OrderedDict()

    ret_5 = compounded_return_from_prices(prices, 5)
    ret_21 = compounded_return_from_prices(prices, 21)
    ret_63 = compounded_return_from_prices(prices, 63)
    ret_126 = compounded_return_from_prices(prices, 126)
    ret_252 = compounded_return_from_prices(prices, 252)
    beta_126 = rolling_beta(daily_returns, factor_returns["MXWD"], 126)
    beta_252 = rolling_beta(daily_returns, factor_returns["MXWD"], 252)
    idio_63 = residual_vol(daily_returns, factor_returns["MXWD"], beta_126, 63)
    mxwd_ret_63 = compounded_return_from_daily_returns(factor_returns["MXWD"], 63)

    feature_frames["ret_5d"] = ret_5
    feature_frames["ret_21d"] = ret_21
    feature_frames["ret_63d"] = ret_63
    feature_frames["ret_126d"] = ret_126
    feature_frames["ret_252d"] = ret_252
    feature_frames["short_rev_5d"] = -ret_5
    feature_frames["short_rev_21d"] = -ret_21
    feature_frames["mom_21_ex_5"] = ret_21 - ret_5
    feature_frames["mom_63_ex_21"] = ret_63 - ret_21
    feature_frames["mom_126_ex_21"] = ret_126 - ret_21
    feature_frames["vol_21d"] = daily_returns.rolling(21, min_periods=21).std(ddof=0) * np.sqrt(252.0)
    feature_frames["vol_63d"] = daily_returns.rolling(63, min_periods=63).std(ddof=0) * np.sqrt(252.0)
    feature_frames["vol_126d"] = daily_returns.rolling(126, min_periods=126).std(ddof=0) * np.sqrt(252.0)
    feature_frames["downside_vol_21d"] = daily_returns.where(daily_returns < 0.0, 0.0).rolling(21, min_periods=21).std(ddof=0) * np.sqrt(252.0)
    feature_frames["downside_vol_63d"] = daily_returns.where(daily_returns < 0.0, 0.0).rolling(63, min_periods=63).std(ddof=0) * np.sqrt(252.0)
    feature_frames["beta_126d_to_MXWD"] = beta_126
    feature_frames["beta_252d_to_MXWD"] = beta_252
    feature_frames["idio_vol_63d_vs_MXWD"] = idio_63
    feature_frames["resid_mom_63d_vs_MXWD"] = ret_63 - beta_126.multiply(mxwd_ret_63, axis=0)
    feature_frames["max_drawdown_63d"] = rolling_drawdown(prices, 63)
    feature_frames["max_drawdown_126d"] = rolling_drawdown(prices, 126)
    feature_frames["log_mktcap"] = np.log(market_caps.where(market_caps > 0.0))

    sector_map = universe.set_index("ticker_short")["sector_return_col"]
    sector_ret_5 = compounded_return_from_daily_returns(sector_returns, 5)
    sector_ret_21 = compounded_return_from_daily_returns(sector_returns, 21)
    sector_ret_63 = compounded_return_from_daily_returns(sector_returns, 63)
    beta_sector = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    sector_rel_21 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    sector_rel_63 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    sector_etf_5 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    sector_etf_21 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    sector_etf_63 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    resid_sector_63 = pd.DataFrame(index=prices.index, columns=assets, dtype=float)
    for asset in assets:
        sector_col = sector_map.loc[asset]
        sret = sector_returns[sector_col]
        beta_sector[asset] = rolling_beta(daily_returns[[asset]], sret, 126)[asset]
        sector_etf_5[asset] = sector_ret_5[sector_col]
        sector_etf_21[asset] = sector_ret_21[sector_col]
        sector_etf_63[asset] = sector_ret_63[sector_col]
        sector_rel_21[asset] = (1.0 + ret_21[asset]).divide(1.0 + sector_ret_21[sector_col]) - 1.0
        sector_rel_63[asset] = (1.0 + ret_63[asset]).divide(1.0 + sector_ret_63[sector_col]) - 1.0
        resid_sector_63[asset] = ret_63[asset] - beta_sector[asset] * sector_ret_63[sector_col]

    feature_frames["sector_rel_ret_21d"] = sector_rel_21
    feature_frames["sector_rel_ret_63d"] = sector_rel_63
    feature_frames["sector_etf_ret_5d"] = sector_etf_5
    feature_frames["sector_etf_ret_21d"] = sector_etf_21
    feature_frames["sector_etf_ret_63d"] = sector_etf_63
    feature_frames["stock_minus_sector_etf_ret_5d"] = ret_5 - sector_etf_5
    feature_frames["stock_minus_sector_etf_ret_21d"] = ret_21 - sector_etf_21
    feature_frames["stock_minus_sector_etf_ret_63d"] = ret_63 - sector_etf_63
    feature_frames["beta_126d_to_sector_etf"] = beta_sector
    feature_frames["resid_mom_63d_vs_sector_etf"] = resid_sector_63

    fundamental_specs = {
        "eps": lagged_panels["BEST_EPS"],
        "sales": lagged_panels["BEST_SALES"],
        "pe": lagged_panels["BEST_PE_RATIO"],
        "peg": lagged_panels["BEST_PEG_RATIO"],
        "fcf": lagged_panels["BEST_CALCULATED_FCF"],
        "gross_margin": lagged_panels["BEST_GROSS_MARGIN"],
        "oper_margin": lagged_panels["OPER_MARGIN"],
        "capex": lagged_panels["BEST_CAPEX"],
        "roe": lagged_panels["BEST_ROE"],
        "px_bps": lagged_panels["BEST_PX_BPS_RATIO"],
        "ev_to_ebitda": lagged_panels["BEST_EV_TO_BEST_EBITDA"],
    }
    feature_frames["eps_level"] = fundamental_specs["eps"]
    feature_frames["eps_delta_63d"] = scaled_delta(fundamental_specs["eps"], 63)
    feature_frames["eps_delta_252d"] = scaled_delta(fundamental_specs["eps"], 252)
    feature_frames["sales_level"] = fundamental_specs["sales"]
    feature_frames["sales_delta_63d"] = scaled_delta(fundamental_specs["sales"], 63)
    feature_frames["sales_delta_252d"] = scaled_delta(fundamental_specs["sales"], 252)
    feature_frames["pe_level"] = fundamental_specs["pe"]
    feature_frames["pe_delta_63d"] = scaled_delta(fundamental_specs["pe"], 63)
    feature_frames["pe_delta_252d"] = scaled_delta(fundamental_specs["pe"], 252)
    feature_frames["peg_level"] = fundamental_specs["peg"]
    feature_frames["peg_delta_63d"] = scaled_delta(fundamental_specs["peg"], 63)
    feature_frames["peg_delta_252d"] = scaled_delta(fundamental_specs["peg"], 252)
    feature_frames["fcf_level"] = fundamental_specs["fcf"]
    feature_frames["fcf_delta_63d"] = scaled_delta(fundamental_specs["fcf"], 63)
    feature_frames["fcf_yield"] = fundamental_specs["fcf"].divide(market_caps.replace(0.0, np.nan))
    feature_frames["gross_margin_level"] = fundamental_specs["gross_margin"]
    feature_frames["gross_margin_delta_63d"] = scaled_delta(fundamental_specs["gross_margin"], 63)
    feature_frames["gross_margin_delta_252d"] = scaled_delta(fundamental_specs["gross_margin"], 252)
    feature_frames["oper_margin_level"] = fundamental_specs["oper_margin"]
    feature_frames["oper_margin_delta_63d"] = scaled_delta(fundamental_specs["oper_margin"], 63)
    feature_frames["oper_margin_delta_252d"] = scaled_delta(fundamental_specs["oper_margin"], 252)
    feature_frames["capex_level"] = fundamental_specs["capex"]
    feature_frames["capex_delta_63d"] = scaled_delta(fundamental_specs["capex"], 63)
    feature_frames["capex_to_sales"] = fundamental_specs["capex"].divide(fundamental_specs["sales"].replace(0.0, np.nan))
    feature_frames["roe_level"] = fundamental_specs["roe"]
    feature_frames["roe_delta_63d"] = scaled_delta(fundamental_specs["roe"], 63)
    feature_frames["roe_delta_252d"] = scaled_delta(fundamental_specs["roe"], 252)
    feature_frames["px_bps_level"] = fundamental_specs["px_bps"]
    feature_frames["px_bps_delta_63d"] = scaled_delta(fundamental_specs["px_bps"], 63)
    feature_frames["px_bps_delta_252d"] = scaled_delta(fundamental_specs["px_bps"], 252)
    feature_frames["ev_to_ebitda_level"] = fundamental_specs["ev_to_ebitda"]
    feature_frames["ev_to_ebitda_delta_63d"] = scaled_delta(fundamental_specs["ev_to_ebitda"], 63)
    feature_frames["ev_to_ebitda_delta_252d"] = scaled_delta(fundamental_specs["ev_to_ebitda"], 252)

    news = lagged_panels["NEWS_SENTIMENT_DAILY_AVG"]
    rec_cons = lagged_panels["EQY_REC_CONS"]
    sent_mom = lagged_panels["Sent_Trend_Momentum_Timeseries"]
    sent_21 = lagged_panels["Sent_Trend_21d_Timeseries"]
    eps_rev = lagged_panels["Factset_EPS_Revision"]
    sales_rev = lagged_panels["Factset_Sales_Revision"]
    target_price = lagged_panels["Factset_TG_Price"]

    feature_frames["news_sent_mean_5d"] = news.rolling(5, min_periods=5).mean()
    feature_frames["news_sent_mean_21d"] = news.rolling(21, min_periods=21).mean()
    feature_frames["news_sent_mean_63d"] = news.rolling(63, min_periods=63).mean()
    feature_frames["news_sent_delta_21d"] = news - news.shift(21)
    feature_frames["rec_cons_level"] = rec_cons
    feature_frames["rec_cons_delta_21d"] = rec_cons - rec_cons.shift(21)
    feature_frames["rec_cons_delta_63d"] = rec_cons - rec_cons.shift(63)
    feature_frames["sent_mom_level"] = sent_mom
    feature_frames["sent_mom_mean_5d"] = sent_mom.rolling(5, min_periods=5).mean()
    feature_frames["sent_mom_mean_21d"] = sent_mom.rolling(21, min_periods=21).mean()
    feature_frames["sent_mom_mean_63d"] = sent_mom.rolling(63, min_periods=63).mean()
    feature_frames["sent_21d_level"] = sent_21
    feature_frames["sent_21d_delta_21d"] = sent_21 - sent_21.shift(21)
    feature_frames["sent_21d_delta_63d"] = sent_21 - sent_21.shift(63)
    feature_frames["eps_revision_level"] = eps_rev
    feature_frames["eps_revision_delta_21d"] = eps_rev - eps_rev.shift(21)
    feature_frames["eps_revision_delta_63d"] = eps_rev - eps_rev.shift(63)
    feature_frames["eps_revision_delta_126d"] = eps_rev - eps_rev.shift(126)
    feature_frames["sales_revision_level"] = sales_rev
    feature_frames["sales_revision_delta_21d"] = sales_rev - sales_rev.shift(21)
    feature_frames["sales_revision_delta_63d"] = sales_rev - sales_rev.shift(63)
    feature_frames["sales_revision_delta_126d"] = sales_rev - sales_rev.shift(126)
    feature_frames["target_price_level"] = target_price
    feature_frames["target_price_upside"] = target_price.divide(prices.replace(0.0, np.nan)) - 1.0
    feature_frames["target_price_delta_21d"] = target_price - target_price.shift(21)
    feature_frames["target_price_delta_63d"] = target_price - target_price.shift(63)

    factor_ret = factor_returns
    factor_px = factor_prices
    feature_frames["mxwd_ret_5d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["MXWD"], 5), assets)
    feature_frames["mxwd_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["MXWD"], 21), assets)
    feature_frames["mxwd_ret_63d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["MXWD"], 63), assets)
    feature_frames["spx_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["SPX"], 21), assets)
    feature_frames["ndx_minus_rty_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["NDX"], 21) - compounded_return_from_daily_returns(factor_ret["RTY"], 21), assets)
    feature_frames["mxwd_minus_mxef_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["MXWD"], 21) - compounded_return_from_daily_returns(factor_ret["MXEF"], 21), assets)
    feature_frames["vix_level_z_252d"] = _broadcast_global_series((factor_px["VIX"] - factor_px["VIX"].rolling(252, min_periods=126).mean()).divide(factor_px["VIX"].rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)), assets)
    feature_frames["vix_delta_21d"] = _broadcast_global_series(factor_px["VIX"] - factor_px["VIX"].shift(21), assets)
    feature_frames["skew_level_z_252d"] = _broadcast_global_series((factor_px["SKEW"] - factor_px["SKEW"].rolling(252, min_periods=126).mean()).divide(factor_px["SKEW"].rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)), assets)
    feature_frames["ust_10y_level_z_252d"] = _broadcast_global_series((factor_px["UST_10Y"] - factor_px["UST_10Y"].rolling(252, min_periods=126).mean()).divide(factor_px["UST_10Y"].rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)), assets)
    feature_frames["ust_10y_delta_21d"] = _broadcast_global_series(factor_px["UST_10Y"] - factor_px["UST_10Y"].shift(21), assets)
    curve = factor_px["UST_10Y"] - factor_px["UST_2Y"]
    feature_frames["ust_curve_10y2y_level"] = _broadcast_global_series(curve, assets)
    feature_frames["ust_curve_10y2y_delta_21d"] = _broadcast_global_series(curve - curve.shift(21), assets)
    feature_frames["dxy_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["DXY"], 21), assets)
    feature_frames["usdkrw_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["USDKRW"], 21), assets)
    feature_frames["wti_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["WTI"], 21), assets)
    feature_frames["gold_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["GOLD"], 21), assets)
    feature_frames["growth_minus_value_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["F_Growth"], 21) - compounded_return_from_daily_returns(factor_ret["F_Value"], 21), assets)
    feature_frames["quality_minus_hibeta_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["F_Quality"], 21) - compounded_return_from_daily_returns(factor_ret["F_HiBeta"], 21), assets)
    feature_frames["gs_ai_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["GS_AI"], 21), assets)
    feature_frames["gs_semihw_ret_21d"] = _broadcast_global_series(compounded_return_from_daily_returns(factor_ret["GS_SemiHW"], 21), assets)
    feature_frames["aaii_bull_minus_bear"] = _broadcast_global_series(factor_px["AAII_Bull"] - factor_px["AAII_Bear"], assets)
    feature_frames["cesi_us_level_z_252d"] = _broadcast_global_series((factor_px["CESI_US"] - factor_px["CESI_US"].rolling(252, min_periods=126).mean()).divide(factor_px["CESI_US"].rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)), assets)

    sector_comp_5 = compounded_return_from_daily_returns(sector_returns, 5)
    sector_comp_21 = compounded_return_from_daily_returns(sector_returns, 21)
    sector_comp_63 = compounded_return_from_daily_returns(sector_returns, 63)
    feature_frames["sector_eq_ret_5d"] = _broadcast_global_series(sector_comp_5.mean(axis=1), assets)
    feature_frames["sector_eq_ret_21d"] = _broadcast_global_series(sector_comp_21.mean(axis=1), assets)
    feature_frames["sector_eq_ret_63d"] = _broadcast_global_series(sector_comp_63.mean(axis=1), assets)
    feature_frames["sector_dispersion_21d"] = _broadcast_global_series(sector_comp_21.std(axis=1, ddof=0), assets)
    feature_frames["sector_dispersion_63d"] = _broadcast_global_series(sector_comp_63.std(axis=1, ddof=0), assets)
    feature_frames["sector_breadth_pos_21d"] = _broadcast_global_series(sector_comp_21.gt(0.0).mean(axis=1), assets)
    feature_frames["sector_breadth_pos_63d"] = _broadcast_global_series(sector_comp_63.gt(0.0).mean(axis=1), assets)
    feature_frames["consdisc_minus_constap_ret_21d"] = _broadcast_global_series(sector_comp_21["SEC_ConsDisc"] - sector_comp_21["SEC_ConsStap"], assets)
    feature_frames["industrials_minus_utilities_ret_21d"] = _broadcast_global_series(sector_comp_21["SEC_Industrials"] - sector_comp_21["SEC_Utilities"], assets)
    feature_frames["infotech_minus_health_ret_21d"] = _broadcast_global_series(sector_comp_21["SEC_InfoTech"] - sector_comp_21["SEC_Health"], assets)
    feature_frames["financials_minus_utilities_ret_21d"] = _broadcast_global_series(sector_comp_21["SEC_Financials"] - sector_comp_21["SEC_Utilities"], assets)
    feature_frames["energy_minus_consdisc_ret_21d"] = _broadcast_global_series(sector_comp_21["SEC_Energy"] - sector_comp_21["SEC_ConsDisc"], assets)

    feature_dict = _build_feature_dictionary()
    save_csv(feature_dict, output_paths["features"] / "feature_dictionary.csv", index=False)

    month_end_frames = OrderedDict((name, frame.reindex(month_ends)) for name, frame in feature_frames.items())
    raw_panel = pd.concat([frame.stack(dropna=False).rename(name) for name, frame in month_end_frames.items()], axis=1)
    raw_panel.index.names = ["date", "asset"]
    raw_panel = raw_panel.reindex(columns=ALL_FEATURES)
    save_parquet(raw_panel, output_paths["features"] / "panel_monthly_features_raw.parquet")

    transformed = raw_panel.copy()
    stock_feature_names = STOCK_PRICE_FEATURES + STOCK_SECTOR_REL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES
    global_feature_names = GLOBAL_MACRO_FEATURES + GLOBAL_SECTOR_REGIME_FEATURES
    lower = float(config["preprocessing"]["winsor_lower"])
    upper = float(config["preprocessing"]["winsor_upper"])

    # Forward-fill global features across months before cross-sectional processing,
    # then fill any remaining NaNs with the expanding median to avoid the
    # information-destroying 0.0 fill that previously treated missing data as
    # a neutral regime signal.
    global_block = transformed[global_feature_names].copy()
    global_block = global_block.groupby(level="asset").ffill()
    global_medians = global_block.expanding(min_periods=1).median()
    global_block = global_block.fillna(global_medians).fillna(0.0)
    transformed[global_feature_names] = global_block

    for date, indices in transformed.groupby(level="date").groups.items():
        block = transformed.loc[indices, stock_feature_names]
        ranked_block = block.apply(lambda col: rank_center_cross_section(winsorize_cross_section(col, lower, upper)))
        transformed.loc[indices, stock_feature_names] = ranked_block.fillna(0.0)

    save_parquet(transformed, output_paths["features"] / "panel_monthly_features_model.parquet")
    LOGGER.info("Built monthly features with exact schema count %s.", len(ALL_FEATURES))
    return {
        "month_ends": month_ends,
        "feature_dict": feature_dict,
        "raw_panel": raw_panel,
        "model_panel": transformed,
        "feature_names": ALL_FEATURES,
        "stock_feature_names": stock_feature_names,
        "global_feature_names": global_feature_names,
    }
