# AI Signal Research Project

This project builds a deterministic monthly research pipeline around `ai_signal_data.xlsx` in the current workspace.

## Data handling

- `BusinessDays` is the canonical trading calendar.
- Non-business-day rows are dropped before returns or rolling statistics are computed.
- `PX_LAST` is the primary stock price source.
- `Daily_Returns` is used only as a diagnostic cross-check.
- All non-price stock features are lagged by 1 business day.
- Synthetic flat pre-listing history is masked from `PX_LAST` using the first real trading date per stock.
- Structural missingness is preserved until the monthly transform stage.
- Missing values become 0 only after the monthly transform.
- Month-end business dates are the only model snapshot dates.

## Sector return detection and mapping

The code resolves `Factor_Return` or `Factor_Returns` with strict validation.

Required sector return columns:
- `SEC_InfoTech`
- `SEC_Health`
- `SEC_Financials`
- `SEC_ConsDisc`
- `SEC_ConsStap`
- `SEC_Energy`
- `SEC_Industrials`
- `SEC_Materials`
- `SEC_Utilities`
- `SEC_RealEstate`
- `SEC_CommSvc`

These columns are treated as daily return series, not price series.

Universe sector mapping follows:
- Technology / Information Technology / Info Tech -> `SEC_InfoTech`
- Health Care / Healthcare -> `SEC_Health`
- Financials -> `SEC_Financials`
- Consumer Discretionary -> `SEC_ConsDisc`
- Consumer Staples -> `SEC_ConsStap`
- Energy -> `SEC_Energy`
- Industrials -> `SEC_Industrials`
- Materials -> `SEC_Materials`
- Utilities -> `SEC_Utilities`
- Real Estate -> `SEC_RealEstate`
- Communication Services -> `SEC_CommSvc`

## Exact 126-feature schema

The project enforces exactly 126 monthly features.

- 24 stock price features
- 8 stock sector-ETF-relative features
- 33 fundamental / valuation features
- 26 sentiment / analyst features
- 23 global macro / factor features
- 12 global sector-ETF regime features

The exact feature names are saved to `outputs/features/feature_dictionary.csv`.

Stock-specific features are winsorized cross-sectionally at the 1/99 percentile, ranked to `[0,1]`, centered to `[-0.5,0.5]`, and only then filled with 0.
Global features are not cross-sectionally ranked and are broadcast to all stocks after month-end construction.

## Target construction

The target is built at month-end business dates.

1. `future_return_21d = PX_LAST[t+21] / PX_LAST[t] - 1`
2. beta-neutralization uses trailing 126-day beta to `MXWD`, falling back to `SPX` only if `MXWD` is unavailable
3. the residual is demeaned within sector at each rebalance month
4. the residual is scaled by trailing 63-day idiosyncratic volatility, falling back to trailing total volatility
5. all intermediate target variants are saved in the cleaned outputs

## Model training

Monthly walk-forward training is implemented with:
- 120 monthly observations in the training window
- 20 randomly excluded validation months inside each training window
- fixed random seed
- validation metric = average monthly cross-sectional Spearman rank IC

Implemented model families:
- OLS
- LASSO
- LightGBM
- feed-forward neural network

For LASSO, LightGBM, and FFNN, reduced grids are evaluated and the top 20% of configs by validation IC are retained and averaged.

## Portfolio construction

The project constructs monthly long-short portfolios by:
- ranking cross-sectional scores each month
- taking top decile and bottom decile buckets
- also running quintile sensitivity
- equal-weighting within sector on both long and short sides
- allocating sector gross exposure by current sector market-cap weights, with equal-sector fallback
- making each sector sleeve beta-neutral to `MXWD`
- scaling to 10% annualized volatility
- capping leverage at 2.5x by default
- applying 10 bps transaction cost per unit of two-way turnover

## Functional attribution

For LightGBM and FFNN, the attribution module is built around paper-style component decomposition rather than SHAP as the main method.

Outputs include:
- total prediction
- linear component
- non-linear component
- interaction component

The module saves component predictions and attribution summaries under `outputs/attribution/`.

## Differences from the original paper

This is an adaptation, not a strict reproduction.

Key differences:
- the universe is 50 stocks rather than the original large global setup
- the available sheet structure constrains the exact feature menu
- the attribution code is implemented for the available model outputs and data budget in this environment
- the portfolio stage uses the requested long-short construction rather than the earlier long-only prototype

## Run commands

Schema validation only:

```bash
python run_pipeline.py --config config.yaml --stage schema
```

Build cleaned data and monthly features:

```bash
python run_pipeline.py --config config.yaml --stage features
```

Train models:

```bash
python run_pipeline.py --config config.yaml --stage train
```

Full pipeline:

```bash
python run_pipeline.py --config config.yaml --stage full
```

Smoke run on the most recent configured sample window:

```bash
python run_pipeline.py --config config.yaml --stage smoke
```
