# ui/backtest_ui.py
"""Streamlit dashboard for walk-forward backtesting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from storage.parquet_store import read

st.set_page_config(page_title="HKJC Backtest Dashboard", layout="wide")
st.title("📈 Backtest Dashboard")

# ── Parameters ────────────────────────────────────────────────────────────────
st.sidebar.header("Backtest Parameters")

model_name = st.sidebar.selectbox("Model", ["lgbm", "xgb", "catboost"])
version    = st.sidebar.text_input("Version", "v1")

start_date = st.sidebar.date_input("Start date",
                                    value=datetime.date(2022, 1, 1))
end_date   = st.sidebar.date_input("End date",
                                    value=datetime.date.today())

bankroll      = st.sidebar.number_input("Starting bankroll (HKD)",
                                         value=cfg.STARTING_BANKROLL, step=1000.0)
kelly_frac    = st.sidebar.slider("Kelly fraction",  0.05, 1.0, cfg.KELLY_FRACTION, 0.05)
min_edge_val  = st.sidebar.slider("Min edge",        0.01, 0.30, cfg.MIN_EDGE,       0.01)

# Pool type filter — only WIN, PLA, QIN, QPL
st.sidebar.subheader("Pool types")
pool_selections = {
    pt: st.sidebar.checkbox(pt, value=True) for pt in cfg.VALID_POOL_TYPES
}
selected_pools = [pt for pt, v in pool_selections.items() if v]

# ── Run button ────────────────────────────────────────────────────────────────
if st.button("▶ Run Backtest", type="primary"):
    feat_path = cfg.PROCESSED_DIR / "features_train.parquet"
    if not feat_path.exists():
        st.error("Feature table not found. Run Features pipeline first.")
    else:
        with st.spinner("Running backtest …"):
            try:
                from models.registry import get_model
                from betting.backtest import run_backtest

                df        = read(feat_path)
                win_m     = get_model(model_name, "win",   version)
                place_m   = get_model(model_name, "place", version)
                result    = run_backtest(
                    features_df=df,
                    win_model=win_m,
                    place_model=place_m,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    starting_bankroll=bankroll,
                    kelly_fraction=kelly_frac,
                    min_edge=min_edge_val,
                    pool_types=selected_pools,
                )
                st.session_state["bt_result"] = result
                st.success("Backtest complete.")
            except Exception as exc:
                st.error(f"Backtest error: {exc}")

# ── Results display ───────────────────────────────────────────────────────────
if "bt_result" in st.session_state:
    r = st.session_state["bt_result"]

    st.markdown("---")
    st.header("Summary Metrics")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total Return",   f"{r.total_return_pct:.2f}%")
    mc2.metric("Sharpe Ratio",   f"{r.sharpe_ratio:.3f}")
    mc3.metric("Max Drawdown",   f"{r.max_drawdown_pct:.2f}%")
    mc4.metric("Win Rate",       f"{r.win_rate:.2%}")
    mc5.metric("Bets/Meeting",   f"{r.bets_per_meeting:.1f}")

    st.markdown("---")
    st.header("Equity Curve")

    fig_eq = go.Figure(
        go.Scatter(
            x=r.cumulative_bankroll.index,
            y=r.cumulative_bankroll.values,
            mode="lines", name="Bankroll"
        )
    )
    fig_eq.update_layout(xaxis_title="Date", yaxis_title="Bankroll (HKD)")
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("---")
    st.header("ROI Breakdown")

    c1, c2, c3 = st.columns(3)
    with c1:
        if not r.roi_by_pool.empty:
            st.subheader("By Pool Type")
            st.bar_chart(r.roi_by_pool)
    with c2:
        if not r.roi_by_venue.empty:
            st.subheader("By Venue")
            st.bar_chart(r.roi_by_venue)
    with c3:
        if not r.roi_by_class.empty:
            st.subheader("By Race Class")
            st.bar_chart(r.roi_by_class)

    st.markdown("---")
    st.header("Bet Log")
    if not r.trades_df.empty:
        st.dataframe(r.trades_df, use_container_width=True)
        csv = r.trades_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "backtest_trades.csv", "text/csv")
    else:
        st.info("No bets were placed in the selected date range and parameters.")