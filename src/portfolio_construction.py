"""Long-short monthly portfolio construction."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.universe import sector_market_cap_weights
from src.utils import LOGGER, save_csv


def _sector_neutral_weights(scores: pd.DataFrame, universe: pd.DataFrame, sector_weights: pd.Series, bucket_fraction: float, min_bucket: int) -> pd.Series:
    mapping = universe.set_index("ticker_short")["Sector"]
    weights = pd.Series(0.0, index=scores.index)
    for sector, members in mapping.groupby(mapping):
        sector_assets = [asset for asset in members.index if asset in scores.index]
        if len(sector_assets) < min_bucket * 2:
            continue
        sector_scores = scores.loc[sector_assets].sort_values("prediction")
        bucket_size = max(int(ceil(bucket_fraction * len(sector_assets))), min_bucket)
        short_assets = sector_scores.head(bucket_size).index.tolist()
        long_assets = sector_scores.tail(bucket_size).index.tolist()
        gross = float(sector_weights.get(sector, 0.0))
        if gross <= 0:
            continue

        # True beta-neutral allocation: solve for long_gross such that
        #   long_gross * long_beta == short_gross * short_beta
        # subject to long_gross + short_gross == gross (dollar budget).
        # This yields:  long_gross = gross * short_beta / (long_beta + short_beta)
        #               short_gross = gross * long_beta / (long_beta + short_beta)
        # Then verify the resulting portfolio beta exposure is zero within
        # each sector by weighting individual asset betas, not bucket averages.
        long_betas = scores.loc[long_assets, "beta_to_market"].clip(lower=1.0e-6)
        short_betas = scores.loc[short_assets, "beta_to_market"].clip(lower=1.0e-6)

        # Per-asset beta-proportional weights within each bucket ensure
        # the aggregate long beta exposure equals the aggregate short beta
        # exposure, achieving true beta-neutrality per sector.
        long_beta_sum = float(long_betas.sum())
        short_beta_sum = float(short_betas.sum())
        if long_beta_sum < 1.0e-8 or short_beta_sum < 1.0e-8:
            continue

        # Solve: for each long asset i with beta_i, weight_i propto 1/beta_i
        # so that sum(weight_i * beta_i) = long_gross for all i.
        # Similarly for short assets.  Then set long_gross = short_gross = gross/2
        # and scale so that sum(w_long * beta_long) == sum(w_short * beta_short).
        long_inv_beta = 1.0 / long_betas
        short_inv_beta = 1.0 / short_betas

        # Equal beta contribution: each asset contributes equally to total
        # beta exposure within its bucket.
        raw_long_w = long_inv_beta / float(long_inv_beta.sum())
        raw_short_w = short_inv_beta / float(short_inv_beta.sum())

        # Portfolio beta for each side if we allocated gross/2 to each:
        half_gross = gross / 2.0
        long_port_beta = float((raw_long_w * long_betas).sum())
        short_port_beta = float((raw_short_w * short_betas).sum())

        # Scale so long_gross * long_port_beta == short_gross * short_port_beta
        total_beta = long_port_beta + short_port_beta
        if total_beta < 1.0e-8:
            continue
        long_gross = gross * short_port_beta / total_beta
        short_gross = gross * long_port_beta / total_beta

        weights.loc[long_assets] += raw_long_w * long_gross
        weights.loc[short_assets] -= raw_short_w * short_gross
    return weights


def _scale_to_target_vol(weights: pd.Series, covariance: pd.DataFrame, target_annual_vol: float, leverage_cap: float) -> pd.Series:
    aligned_cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    daily_vol = float(np.sqrt(weights.to_numpy() @ aligned_cov.to_numpy() @ weights.to_numpy()))
    annual_vol = daily_vol * np.sqrt(252.0)
    if annual_vol <= 1.0e-8:
        return weights
    leverage = min(target_annual_vol / annual_vol, leverage_cap)
    return weights * leverage


def _monthly_turnover(current: pd.Series, previous: pd.Series) -> float:
    union = sorted(set(current.index).union(previous.index))
    delta = current.reindex(union).fillna(0.0) - previous.reindex(union).fillna(0.0)
    return float(delta.abs().sum())


def build_all_portfolios(bundle: dict[str, Any], feature_payload: dict[str, Any], target_payload: dict[str, Any], training_payload: dict[str, Any], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, Any]:
    universe = bundle["universe"]
    market_caps = bundle["stock_panels"]["CUR_MKT_CAP"]
    factor_returns = bundle["factor_returns"]
    prices = bundle["prices"]
    daily_returns = bundle["daily_returns_from_px"]
    proxy = target_payload["market_proxy_used"]
    beta_panel = target_payload["beta_panel"]
    results = []
    holdings_frames = []
    sector_diag_frames = []

    for model_name, prediction_frame in training_payload["predictions_by_model"].items():
        model_rows = []
        holdings_rows = []
        sector_rows = []
        previous_weights_decile = pd.Series(dtype=float)
        previous_weights_quintile = pd.Series(dtype=float)
        for date, month_scores in prediction_frame.groupby("date"):
            date = pd.Timestamp(date)
            current_caps = market_caps.loc[date].dropna()
            sector_weights = sector_market_cap_weights(universe, current_caps)
            score_block = month_scores.set_index("asset").copy()
            score_block["beta_to_market"] = beta_panel.loc[date, score_block.index].fillna(1.0)
            weights_decile = _sector_neutral_weights(score_block, universe, sector_weights, float(config["portfolio"]["long_fraction"]), int(config["portfolio"]["minimum_bucket_size"]))
            weights_quintile = _sector_neutral_weights(score_block, universe, sector_weights, 0.20, int(config["portfolio"]["minimum_bucket_size"]))

            cov = daily_returns.loc[:date, score_block.index].tail(63).fillna(0.0).cov(ddof=0)
            weights_decile = _scale_to_target_vol(weights_decile, cov, float(config["portfolio"]["target_annual_vol"]), float(config["portfolio"]["leverage_cap"]))
            weights_quintile = _scale_to_target_vol(weights_quintile, cov, float(config["portfolio"]["target_annual_vol"]), float(config["portfolio"]["leverage_cap"]))

            next_idx = prices.index.searchsorted(date, side="right")
            if next_idx >= len(prices.index):
                continue
            next_date = pd.Timestamp(prices.index[next_idx])
            future_idx = prices.index.searchsorted(next_date, side="left") + 21
            if future_idx >= len(prices.index):
                continue
            end_date = pd.Timestamp(prices.index[future_idx])
            realized = prices.loc[end_date, score_block.index].divide(prices.loc[next_date, score_block.index]) - 1.0
            market_realized = float(((1.0 + factor_returns.loc[next_date:end_date, proxy].fillna(0.0)).prod()) - 1.0)

            for label, weights, previous_weights in [("decile", weights_decile, previous_weights_decile), ("quintile", weights_quintile, previous_weights_quintile)]:
                gross = float((weights * realized).sum())
                turnover = _monthly_turnover(weights, previous_weights)
                tc = turnover * (float(config["portfolio"]["transaction_cost_bps_two_way"]) / 10000.0)
                net = gross - tc
                model_rows.append({
                    "date": date,
                    "model": model_name,
                    "portfolio_type": label,
                    "gross_return": gross,
                    "net_return": net,
                    "market_proxy_return": market_realized,
                    "turnover_two_way": turnover,
                    "transaction_cost": tc,
                    "gross_leverage": float(weights.abs().sum()),
                    "net_exposure": float(weights.sum()),
                })
                for asset, weight in weights.items():
                    if abs(weight) < 1.0e-12:
                        continue
                    holdings_rows.append({
                        "date": date,
                        "model": model_name,
                        "portfolio_type": label,
                        "asset": asset,
                        "weight": float(weight),
                        "score": float(score_block.loc[asset, "prediction"]),
                        "sector": universe.set_index("ticker_short").loc[asset, "Sector"],
                    })
                for sector in sector_weights.index:
                    sector_assets = universe.loc[universe["Sector"] == sector, "ticker_short"]
                    sector_assets = [asset for asset in sector_assets if asset in weights.index]
                    if not sector_assets:
                        continue
                    long_ret = float((weights.loc[sector_assets].clip(lower=0.0) * realized.loc[sector_assets]).sum())
                    short_ret = float((weights.loc[sector_assets].clip(upper=0.0) * realized.loc[sector_assets]).sum())
                    sector_col = universe.loc[universe["Sector"] == sector, "sector_return_col"].iloc[0]
                    sector_series = factor_returns.loc[next_date:end_date, sector_col].fillna(0.0)
                    sector_ret = float((1.0 + sector_series).prod() - 1.0)
                    sector_rows.append({
                        "date": date,
                        "model": model_name,
                        "portfolio_type": label,
                        "sector": sector,
                        "long_sector_excess": long_ret - sector_ret,
                        "short_sector_excess": short_ret + sector_ret,
                        "sector_return": sector_ret,
                    })
                if label == "decile":
                    previous_weights_decile = weights
                else:
                    previous_weights_quintile = weights

        perf = pd.DataFrame(model_rows)
        save_csv(perf, output_paths["portfolios"] / f"performance_{model_name}.csv", index=False)
        save_csv(pd.DataFrame(holdings_rows), output_paths["portfolios"] / f"monthly_holdings_{model_name}.csv", index=False)
        save_csv(pd.DataFrame(sector_rows), output_paths["portfolios"] / f"sector_etf_diagnostics_{model_name}.csv", index=False)
        results.append(perf)
        holdings_frames.append(pd.DataFrame(holdings_rows))
        sector_diag_frames.append(pd.DataFrame(sector_rows))
        LOGGER.info("Constructed long-short portfolios for model %s.", model_name)

    performance_summary = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    save_csv(performance_summary, output_paths["portfolios"] / "performance_summary.csv", index=False)
    return {
        "performance_summary": performance_summary,
        "holdings": holdings_frames,
        "sector_diagnostics": sector_diag_frames,
    }


def build_latest_holdings(bundle: dict[str, Any], target_payload: dict[str, Any], latest_inference: dict[str, pd.DataFrame], config: dict[str, Any], output_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    universe = bundle["universe"]
    market_caps = bundle["stock_panels"]["CUR_MKT_CAP"]
    daily_returns = bundle["daily_returns_from_px"]
    beta_panel = target_payload["beta_panel"]
    output = {}
    for model_name, pred_df in latest_inference.items():
        if pred_df.empty:
            continue
        rows = []
        for date, month_scores in pred_df.groupby("date"):
            date = pd.Timestamp(date)
            current_caps = market_caps.loc[date].dropna()
            sector_weights = sector_market_cap_weights(universe, current_caps)
            score_block = month_scores.set_index("asset").copy()
            score_block["beta_to_market"] = beta_panel.loc[date, score_block.index].fillna(1.0)
            weights = _sector_neutral_weights(score_block, universe, sector_weights, float(config["portfolio"]["long_fraction"]), int(config["portfolio"]["minimum_bucket_size"]))
            cov = daily_returns.loc[:date, score_block.index].tail(63).fillna(0.0).cov(ddof=0)
            weights = _scale_to_target_vol(weights, cov, float(config["portfolio"]["target_annual_vol"]), float(config["portfolio"]["leverage_cap"]))
            for asset, weight in weights.items():
                if abs(weight) < 1.0e-12:
                    continue
                rows.append({
                    "date": date,
                    "model": model_name,
                    "asset": asset,
                    "weight": float(weight),
                    "score": float(score_block.loc[asset, "prediction"]),
                    "sector": universe.set_index("ticker_short").loc[asset, "Sector"],
                    "is_latest_inference": True,
                })
        if rows:
            frame = pd.DataFrame(rows)
            save_csv(frame, output_paths["portfolios"] / f"latest_holdings_{model_name}.csv", index=False)
            output[model_name] = frame
    return output
