from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent / 'outputs'
REPORTS = BASE / 'reports'

st.set_page_config(page_title='AI Signal Codex – LightGBM Fund Dashboard', layout='wide')
st.title('AI Signal Codex – LightGBM Fund Dashboard')


@st.cache_data
def load_csv(path: Path, **kwargs):
    return pd.read_csv(path, **kwargs)


def compounded_rolling(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series.fillna(0.0)).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0


def make_heatmap(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        st.info('No data available.')
        return
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(frame.index))))
    im = ax.imshow(frame.to_numpy(dtype=float), aspect='auto', cmap='RdBu_r')
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels([pd.Timestamp(x).strftime('%Y-%m-%d') for x in frame.index], fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)


def make_bar_chart(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        st.info('No data available.')
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    frame.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Share')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    st.pyplot(fig)


daily = load_csv(REPORTS / 'lightgbm_daily_performance.csv', parse_dates=['date', 'rebalance_date'])
metrics = load_csv(REPORTS / 'lightgbm_overall_metrics.csv')
weights = load_csv(REPORTS / 'lightgbm_recent_weights_6m.csv', parse_dates=['date'])
structure = load_csv(REPORTS / 'lightgbm_model_structure.csv')
attr_summary = load_csv(REPORTS / 'lightgbm_attribution_overview.csv')
month_attr = load_csv(REPORTS / 'lightgbm_monthly_category_attribution.csv', parse_dates=['date'])
style_sector = load_csv(REPORTS / 'lightgbm_rebalance_style_sector_summary.csv', parse_dates=['date'])
explanations = load_csv(REPORTS / 'lightgbm_monthly_ow_explanations.csv', parse_dates=['date'])

metric_map = metrics.set_index('metric')['value']
col1, col2, col3, col4 = st.columns(4)
col1.metric('Fund Total Return', f"{metric_map.get('fund_total_return', float('nan')):.2%}")
col2.metric('Fund Sharpe', f"{metric_map.get('fund_sharpe', float('nan')):.2f}")
col3.metric('Active Total Return', f"{metric_map.get('active_total_return', float('nan')):.2%}")
col4.metric('Information Ratio', f"{metric_map.get('information_ratio', float('nan')):.2f}")

daily = daily.sort_values('date').copy()
daily['drawdown'] = daily['fund_cum_nav'] / daily['fund_cum_nav'].cummax() - 1.0
daily['rolling_sharpe_63d'] = (daily['fund_daily_net_return'].rolling(63).mean() * 252.0) / (daily['fund_daily_net_return'].rolling(63).std(ddof=0) * np.sqrt(252.0))
daily['rolling_active_return_63d'] = compounded_rolling(daily['active_daily_return'], 63)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Daily Performance', 'Recent Weights', 'Model Structure', 'Attribution', 'Style/Sector Lens', 'OW Explanations'])

with tab1:
    st.subheader('Daily Fund vs Benchmark Performance')
    st.line_chart(daily.set_index('date')[['fund_cum_nav', 'bm_cum_nav']])
    st.subheader('Drawdown')
    st.line_chart(daily.set_index('date')[['drawdown']])
    st.subheader('Rolling 63-Day Sharpe')
    st.line_chart(daily.set_index('date')[['rolling_sharpe_63d']])
    st.subheader('Rolling 63-Day Active Return')
    st.line_chart(daily.set_index('date')[['rolling_active_return_63d']])
    st.dataframe(daily.tail(40), use_container_width=True)

with tab2:
    st.subheader('Recent 6-Month Holdings')
    selected_month = st.selectbox('Select month', sorted(weights['date'].dt.strftime('%Y-%m-%d').unique(), reverse=True))
    month_df = weights[weights['date'].dt.strftime('%Y-%m-%d') == selected_month].copy().sort_values('weight', ascending=False)
    st.dataframe(month_df, use_container_width=True)
    heatmap_df = weights.pivot_table(index='date', columns='asset', values='weight', aggfunc='sum').fillna(0.0)
    make_heatmap(heatmap_df, 'Recent 6-Month Weight Heatmap')

with tab3:
    st.subheader('Tree-Based Model Structure')
    st.dataframe(structure, use_container_width=True)
    grouped = structure.groupby(['learning_rate', 'num_leaves', 'min_data_in_leaf']).size().reset_index(name='count')
    st.subheader('Retained Config Frequency')
    st.dataframe(grouped.sort_values('count', ascending=False), use_container_width=True)

with tab4:
    st.subheader('Attribution Summary')
    st.dataframe(attr_summary, use_container_width=True)
    category_df = attr_summary[attr_summary['row_type'] == 'category'].copy()
    share_df = category_df.set_index('name')[['linear_share', 'nonlinear_share', 'interaction_share']].fillna(0.0)
    make_bar_chart(share_df, 'Category Linear / Nonlinear / Interaction Shares')
    st.subheader('Monthly Category Attribution')
    st.line_chart(month_attr.set_index('date'))

with tab5:
    st.subheader('Rebalance Style / Sector Lens')
    st.dataframe(style_sector, use_container_width=True)
    selected_style_month = st.selectbox('Select style/sector month', sorted(style_sector['date'].dt.strftime('%Y-%m-%d').unique(), reverse=True), key='style')
    style_row = style_sector[style_sector['date'].dt.strftime('%Y-%m-%d') == selected_style_month].iloc[0]
    style_cols = ['momentum', 'value', 'quality', 'growth', 'sentiment', 'size']
    style_df = pd.DataFrame({'style': style_cols, 'tilt': [style_row[c] for c in style_cols]}).set_index('style')
    st.bar_chart(style_df)
    st.write('Top long sectors:', style_row['top_long_sectors'])
    st.write('Top gross sectors:', style_row['top_gross_sectors'])
    st.write('Positive style tilts:', style_row['positive_style_tilts'])
    st.write('Negative style tilts:', style_row['negative_style_tilts'])

with tab6:
    st.subheader('Monthly Overweight Explanations')
    selected_month_expl = st.selectbox('Explanation month', sorted(explanations['date'].dt.strftime('%Y-%m-%d').unique(), reverse=True), key='expl')
    expl_df = explanations[explanations['date'].dt.strftime('%Y-%m-%d') == selected_month_expl].copy().sort_values('weight', ascending=False)
    st.dataframe(expl_df, use_container_width=True)
