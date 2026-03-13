"""Portfolio optimization helpers for the AI signal backtest."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def project_to_capped_simplex(
    weights: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> np.ndarray:
    """Project weights onto a capped simplex with long-only style bounds."""
    clipped = np.clip(np.asarray(weights, dtype=float), lower_bounds, upper_bounds)
    for _ in range(max_iter):
        gap = 1.0 - float(clipped.sum())
        if abs(gap) <= tol:
            return clipped
        if gap > 0:
            free = clipped < (upper_bounds - tol)
            if not np.any(free):
                break
            capacity = upper_bounds[free] - clipped[free]
            alloc = capacity / capacity.sum()
            clipped[free] += gap * alloc
        else:
            free = clipped > (lower_bounds + tol)
            if not np.any(free):
                break
            removable = clipped[free] - lower_bounds[free]
            alloc = removable / removable.sum()
            clipped[free] += gap * alloc
        clipped = np.clip(clipped, lower_bounds, upper_bounds)
    total = clipped.sum()
    if total > 0:
        clipped = clipped / total
    return np.clip(clipped, lower_bounds, upper_bounds)


def enforce_sector_caps(
    weights: pd.Series,
    sectors: pd.Series,
    sector_caps: Mapping[str, float],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    reference_weights: pd.Series,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> pd.Series:
    """Scale down violated sectors and redistribute excess weight."""
    adjusted = weights.copy().astype(float)
    ref = reference_weights.reindex(adjusted.index).fillna(0.0).astype(float)

    for _ in range(max_iter):
        violated = False
        for sector, cap in sector_caps.items():
            members = sectors[sectors == sector].index.intersection(adjusted.index)
            if len(members) == 0:
                continue
            sector_weight = float(adjusted.loc[members].sum())
            if sector_weight <= cap + tol:
                continue
            violated = True
            scale = cap / sector_weight if sector_weight > 0 else 0.0
            adjusted.loc[members] = adjusted.loc[members] * scale

        adjusted = pd.Series(
            project_to_capped_simplex(
                adjusted.to_numpy(),
                lower_bounds.reindex(adjusted.index).to_numpy(),
                upper_bounds.reindex(adjusted.index).to_numpy(),
            ),
            index=adjusted.index,
        )

        sector_ok = True
        for sector, cap in sector_caps.items():
            members = sectors[sectors == sector].index.intersection(adjusted.index)
            if len(members) == 0:
                continue
            if float(adjusted.loc[members].sum()) > cap + tol:
                sector_ok = False
                break
        if sector_ok and not violated:
            break

        gap = 1.0 - float(adjusted.sum())
        if abs(gap) > tol:
            eligible = adjusted.index[adjusted < (upper_bounds.reindex(adjusted.index) - tol)]
            if len(eligible) > 0:
                alloc = ref.reindex(eligible).clip(lower=0.0)
                if float(alloc.sum()) <= 0:
                    alloc = pd.Series(1.0, index=eligible)
                alloc = alloc / float(alloc.sum())
                adjusted.loc[eligible] = adjusted.loc[eligible] + gap * alloc
                adjusted = adjusted.clip(
                    lower=lower_bounds.reindex(adjusted.index),
                    upper=upper_bounds.reindex(adjusted.index),
                )

    adjusted = pd.Series(
        project_to_capped_simplex(
            adjusted.to_numpy(),
            lower_bounds.reindex(adjusted.index).to_numpy(),
            upper_bounds.reindex(adjusted.index).to_numpy(),
        ),
        index=adjusted.index,
    )
    return adjusted


def heuristic_fallback_weights(
    expected_returns: pd.Series,
    benchmark_weights: pd.Series,
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    sectors: pd.Series,
    sector_caps: Mapping[str, float],
    config: Mapping[str, Any],
) -> pd.Series:
    """Construct a benchmark-relative fallback portfolio when optimization fails."""
    tilt_strength = float(config["backtest"]["fallback"]["positive_alpha_tilt_strength"])
    clipped_mu = expected_returns.clip(
        lower=-float(config["backtest"]["expected_return"]["max_abs_expected_return"]),
        upper=float(config["backtest"]["expected_return"]["max_abs_expected_return"]),
    )
    positive_alpha = clipped_mu.clip(lower=0.0)
    if float(positive_alpha.sum()) > 0:
        alpha_mix = positive_alpha / float(positive_alpha.sum())
        raw = ((1.0 - tilt_strength) * benchmark_weights) + (tilt_strength * alpha_mix)
    else:
        raw = benchmark_weights.copy()

    raw = raw.clip(lower=lower_bounds, upper=upper_bounds)
    raw = pd.Series(
        project_to_capped_simplex(raw.to_numpy(), lower_bounds.to_numpy(), upper_bounds.to_numpy()),
        index=raw.index,
    )
    return enforce_sector_caps(raw, sectors, sector_caps, lower_bounds, upper_bounds, benchmark_weights)


def ensure_positive_semidefinite(matrix: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Clip eigenvalues so the covariance matrix remains numerically stable."""
    values, vectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    clipped_values = np.clip(values, floor, None)
    psd = vectors @ np.diag(clipped_values) @ vectors.T
    return pd.DataFrame(psd, index=matrix.index, columns=matrix.columns)


def estimate_covariance(
    returns_window: pd.DataFrame,
    assets: Sequence[str],
    config: Mapping[str, Any],
    shrinkage: float | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Estimate a 20-business-day covariance matrix with a robust primary method."""
    cov_cfg = config["backtest"]["covariance"]
    horizon = int(cov_cfg["horizon_bdays"])
    shrinkage_value = float(cov_cfg["shrinkage"] if shrinkage is None else shrinkage)
    floor = float(cov_cfg["diagonal_floor"])
    method = str(cov_cfg.get("method", "ledoit_wolf"))

    window = returns_window.loc[:, list(assets)].copy().fillna(0.0)
    if len(window) < 2:
        default_var = max(float(cov_cfg["default_annual_vol"] / np.sqrt(252.0)) ** 2, floor)
        diag = np.eye(len(assets)) * default_var * horizon
        cov = pd.DataFrame(diag, index=assets, columns=assets)
        return cov, {"shrinkage": 1.0, "history_rows": int(len(window)), "mode": "default_diagonal"}

    if method == "ledoit_wolf" and len(window) >= int(cov_cfg.get("min_ledoit_wolf_history", 40)):
        lw = LedoitWolf().fit(window.to_numpy(dtype=float))
        lw_cov = pd.DataFrame(lw.covariance_ * horizon, index=assets, columns=assets)
        lw_cov = ensure_positive_semidefinite(lw_cov, floor=floor)
        eigenvalues = np.linalg.eigvalsh(lw_cov.to_numpy(dtype=float))
        positive_eigs = eigenvalues[eigenvalues > floor]
        condition_number = float(positive_eigs.max() / positive_eigs.min()) if len(positive_eigs) > 0 else float("nan")
        return lw_cov, {
            "shrinkage": float(getattr(lw, "shrinkage_", np.nan)),
            "history_rows": int(len(window)),
            "condition_number": condition_number,
            "mode": "ledoit_wolf",
        }

    sample_cov = window.cov(ddof=0) * horizon
    diag_cov = pd.DataFrame(np.diag(np.diag(sample_cov)), index=sample_cov.index, columns=sample_cov.columns)
    shrunk = ((1.0 - shrinkage_value) * sample_cov) + (shrinkage_value * diag_cov)
    shrunk = ensure_positive_semidefinite(shrunk, floor=floor)

    eigenvalues = np.linalg.eigvalsh(shrunk.to_numpy(dtype=float))
    positive_eigs = eigenvalues[eigenvalues > floor]
    condition_number = float(positive_eigs.max() / positive_eigs.min()) if len(positive_eigs) > 0 else float("nan")
    diagnostics = {
        "shrinkage": shrinkage_value,
        "history_rows": int(len(window)),
        "condition_number": condition_number,
        "mode": "shrunk_sample",
    }
    return shrunk, diagnostics


def convert_zscores_to_expected_returns(
    alpha_zscores: pd.Series,
    scale: float,
    config: Mapping[str, Any],
) -> pd.Series:
    """Map cross-sectional z-scores into expected 20-business-day returns."""
    exp_cfg = config["backtest"]["expected_return"]
    z_cap = float(exp_cfg["max_abs_zscore"])
    mu_cap = float(exp_cfg["max_abs_expected_return"])
    multiplier = float(exp_cfg["signal_multiplier"])
    expected = alpha_zscores.clip(lower=-z_cap, upper=z_cap) * scale * multiplier
    return expected.clip(lower=-mu_cap, upper=mu_cap)


def optimize_long_only_mean_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark_weights: pd.Series,
    current_weights: pd.Series,
    sectors: pd.Series,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Solve a constrained long-only mean-variance problem with fallbacks."""
    opt_cfg = config["backtest"]["optimizer"]
    solver_cfg = opt_cfg["solver"]
    max_weight = float(opt_cfg["max_weight"])
    max_active = float(opt_cfg["max_active_overweight"])
    risk_aversion = float(opt_cfg["risk_aversion"])
    turnover_penalty = float(opt_cfg["turnover_penalty"])
    raw_sector_caps = opt_cfg.get("sector_caps", {})
    default_sector_cap = float(opt_cfg.get("default_sector_max_weight", 1.0))
    default_sector_active_buffer = float(opt_cfg.get("sector_max_active_overweight", 0.0))

    assets = expected_returns.index.tolist()
    benchmark = benchmark_weights.reindex(assets).fillna(0.0).astype(float)
    benchmark = benchmark / float(benchmark.sum())
    current = current_weights.reindex(assets).fillna(0.0).astype(float)
    sector_series = sectors.reindex(assets)

    lower_bounds = pd.Series(0.0, index=assets)
    benchmark_upper = benchmark + max_active
    absolute_upper = pd.Series(max_weight, index=assets)
    upper_bounds = pd.concat([benchmark_upper, absolute_upper], axis=1).min(axis=1)
    upper_bounds = pd.concat([upper_bounds, benchmark], axis=1).max(axis=1)
    upper_bounds = upper_bounds.clip(lower=float(opt_cfg["min_upper_bound_floor"]), upper=1.0)

    benchmark_sector_weights = benchmark.groupby(sector_series).sum()
    sector_caps = {}
    for sector_name in sector_series.dropna().unique():
        benchmark_sector = float(benchmark_sector_weights.get(sector_name, 0.0))
        dynamic_cap = max(benchmark_sector, min(default_sector_cap, benchmark_sector + default_sector_active_buffer))
        sector_caps[sector_name] = float(raw_sector_caps.get(sector_name, dynamic_cap))

    fallback_seed = heuristic_fallback_weights(
        expected_returns=expected_returns,
        benchmark_weights=benchmark,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        sectors=sector_series,
        sector_caps=sector_caps,
        config=config,
    )

    bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    covariance_matrix = covariance.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype=float)
    benchmark_array = benchmark.to_numpy(dtype=float)
    current_array = current.to_numpy(dtype=float)
    expected_array = expected_returns.to_numpy(dtype=float)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    for sector, cap in sector_caps.items():
        mask = (sector_series == sector).astype(float).to_numpy()
        if mask.sum() <= 0:
            continue
        constraints.append({
            "type": "ineq",
            "fun": lambda w, m=mask, c=float(cap): float(c - np.dot(m, w)),
        })

    def objective(weights: np.ndarray) -> float:
        active = weights - benchmark_array
        turnover_vector = weights - current_array
        risk_term = 0.5 * risk_aversion * float(active @ covariance_matrix @ active)
        alpha_term = -float(expected_array @ weights)
        turnover_term = 0.5 * turnover_penalty * float(turnover_vector @ turnover_vector)
        return alpha_term + risk_term + turnover_term

    attempts = [
        {"mode": "base", "x0": fallback_seed.to_numpy(dtype=float)},
    ]

    best_result: Dict[str, Any] | None = None
    for attempt in attempts:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step, clipping to bounds",
            )
            result = minimize(
            objective,
            x0=attempt["x0"],
            method=str(solver_cfg["method"]),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(solver_cfg["maxiter"]), "ftol": float(solver_cfg["ftol"]), "disp": False},
        )
        candidate = pd.Series(result.x, index=assets).clip(lower=lower_bounds, upper=upper_bounds)
        candidate = pd.Series(
            project_to_capped_simplex(candidate.to_numpy(), lower_bounds.to_numpy(), upper_bounds.to_numpy()),
            index=assets,
        )
        candidate = enforce_sector_caps(
            candidate,
            sectors=sector_series,
            sector_caps=sector_caps,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            reference_weights=benchmark,
        )
        turnover = 0.5 * float(np.abs(candidate.reindex(assets).to_numpy() - current_array).sum())
        diagnostics = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "mode": attempt["mode"],
            "objective": float(objective(candidate.to_numpy())),
            "turnover": turnover,
        }
        if result.success:
            return {"weights": candidate, "diagnostics": diagnostics}
        if best_result is None:
            best_result = {"weights": candidate, "diagnostics": diagnostics}

    fallback = heuristic_fallback_weights(
        expected_returns=expected_returns,
        benchmark_weights=benchmark,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        sectors=sector_series,
        sector_caps=sector_caps,
        config=config,
    )
    turnover = 0.5 * float(np.abs(fallback.reindex(assets).to_numpy() - current_array).sum())
    fallback_diagnostics = {
        "success": False,
        "status": -1,
        "message": "Optimizer fallback: benchmark-relative heuristic",
        "mode": "heuristic_fallback",
        "objective": float(objective(fallback.to_numpy())),
        "turnover": turnover,
    }
    if best_result is not None:
        fallback_diagnostics["failed_solver_message"] = best_result["diagnostics"]["message"]
    return {"weights": fallback, "diagnostics": fallback_diagnostics}


def compute_turnover(current_weights: pd.Series, previous_weights: pd.Series) -> float:
    """Compute one-way portfolio turnover across the full universe."""
    union = sorted(set(current_weights.index).union(previous_weights.index))
    current = current_weights.reindex(union).fillna(0.0)
    previous = previous_weights.reindex(union).fillna(0.0)
    return 0.5 * float((current - previous).abs().sum())






