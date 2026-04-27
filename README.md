# AI Signal Codex Dashboard

Streamlit dashboard for the `final_v3_current` AI Signal portfolio results.

## Included Data

The app reads CSV artifacts under:

```text
outputs/final_v3_current/csv/
```

Included tables:

- portfolio and benchmark daily performance
- yearly and recent-period performance
- rebalance-date OW/UW active weights
- precomputed feature diagnostics
- precomputed OW stock feature score explanations

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
