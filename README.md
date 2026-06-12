# AI Signal Codex Dashboard

Streamlit dashboard for the `final_v3_current` AI Signal portfolio results.

## Scope

This repository contains ONLY the dashboard app (`streamlit_app.py`,
`src/metadata.py`) and pre-computed CSV artifacts. The pipeline code that
produces these numbers (feature engine, LightGBM walk-forward trainer,
portfolio simulation) lives in the private main repository and is NOT
mirrored here.

## Included Data

The app reads CSV artifacts under:

```text
outputs/final_v3_current/csv/
```

The large `backtest_result_redesign.pkl` file is intentionally not committed.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Deploy on Streamlit Community Cloud:

1. Connect this GitHub repository.
2. Select branch `main`.
3. Set main file path to `streamlit_app.py`.
4. Deploy.

After deployment, the Streamlit app URL can be opened from a phone on any network.
