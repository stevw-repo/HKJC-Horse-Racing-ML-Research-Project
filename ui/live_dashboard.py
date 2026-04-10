# ui/live_dashboard.py
"""Streamlit live race-day dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import time

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from storage.parquet_store import read, write
from betting.value_detector import find_value_bets
from betting.kelly import kelly_stake_hkd

st.set_page_config(page_title="HKJC Live Dashboard", layout="wide")
st.title("🏇 Live Race Day Dashboard")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Session Settings")

race_date  = st.sidebar.date_input("Meeting date", value=datetime.date.today())
venue      = st.sidebar.selectbox("Venue", sorted(cfg.VALID_VENUES))
bankroll   = st.sidebar.number_input("Current bankroll (HKD)",
                                      value=cfg.STARTING_BANKROLL, step=100.0)
model_name = st.sidebar.selectbox("Model", ["lgbm", "xgb", "catboost"])
version    = st.sidebar.text_input("Version", "v1")
min_edge   = st.sidebar.slider("Min edge", 0.01, 0.30, cfg.MIN_EDGE, 0.01)
kelly_frac = st.sidebar.slider("Kelly fraction", 0.05, 1.0, cfg.KELLY_FRACTION, 0.05)
auto_ref   = st.sidebar.toggle("Auto-refresh (60s)", value=False)

# ── Scrape controls ───────────────────────────────────────────────────────────
if st.sidebar.button("🔄 Scrape Racecard + Odds Now"):
    with st.spinner("Scraping …"):
        from scrapers.racecard_scraper import RacecardScraper
        from scrapers.odds_scraper import OddsScraper
        try:
            RacecardScraper().scrape(start_date=race_date.isoformat())
            OddsScraper().scrape(start_date=race_date.isoformat())
            st.sidebar.success("Done.")
        except Exception as exc:
            st.sidebar.error(f"Scrape error: {exc}")

# ── Load prediction features ──────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _load_predictions(race_date_str: str, venue_str: str,
                      model_str: str, ver: str):
    from features.pipeline import build_prediction_features
    from models.registry import get_model

    df = build_prediction_features(race_date=race_date_str, venue=venue_str)

    exclude = {"race_date", "horse_id", "race_id", "is_debutant",
               "placing_code", "placing", "target_win", "target_place",
               "dividend_win", "dividend_place"}
    fcols   = [c for c in df.columns if c not in exclude
               and df[c].dtype in (float, int, np.float32,
                                    np.float64, np.int32, np.int64)]

    X       = df[fcols].fillna(0).values
    win_m   = get_model(model_str, "win",   ver)
    place_m = get_model(model_str, "place", ver)
    df["pred_win"]   = win_m.predict_proba(X)
    df["pred_place"] = place_m.predict_proba(X)
    return df, fcols


# ── Main area ─────────────────────────────────────────────────────────────────
date_str = race_date.isoformat()
tag      = date_str.replace("-", "")

try:
    with st.spinner("Loading predictions …"):
        pred_df, feat_cols = _load_predictions(
            date_str, venue, model_name, version
        )
    st.success(f"Loaded {len(pred_df)} runners for {venue} on {date_str}.")
except Exception as exc:
    st.error(f"Could not build predictions: {exc}")
    st.stop()

# ── Race tab strip ────────────────────────────────────────────────────────────
races = sorted(pred_df["race_no"].unique())
if not races:
    st.warning("No races found.")
    st.stop()

tabs = st.tabs([f"R{r}" for r in races])

for tab, race_no in zip(tabs, races):
    with tab:
        race = pred_df[pred_df["race_no"] == race_no].copy()

        # Runner card
        st.subheader(f"Race {race_no} — Runner Summary")
        display_cols = [c for c in ["horse_name", "horse_no", "draw",
                                     "actual_weight_lbs", "win_odds_closing",
                                     "pred_win", "pred_place",
                                     "jockey_name", "trainer_name",
                                     "career_win_rate", "days_since_last_run"]
                        if c in race.columns]
        st.dataframe(
            race[display_cols]
            .sort_values("pred_win", ascending=False)
            .reset_index(drop=True)
            .style.format({
                "pred_win":   "{:.1%}",
                "pred_place": "{:.1%}",
                "career_win_rate": "{:.1%}",
            }),
            use_container_width=True,
        )

        # Value bets
        st.subheader("💡 Value Bet Recommendations")
        odds_fp  = cfg.RAW_ODDS_DIR / f"odds_{tag}.parquet"
        odds_df  = read(odds_fp) if odds_fp.exists() else pd.DataFrame()

        if odds_df.empty:
            st.info("No odds data yet. Scrape odds first.")
        else:
            race_odds = odds_df[
                (odds_df["race_no"]   == race_no) &
                (odds_df["pool_type"].isin(cfg.VALID_POOL_TYPES))
            ]
            win_s = pd.Series(
                race["pred_win"].values,
                index=race["horse_no"].astype(str).str.lstrip("0")
            )
            place_s = pd.Series(
                race["pred_place"].values,
                index=race["horse_no"].astype(str).str.lstrip("0")
            )
            bets = find_value_bets(
                model_win_probs=win_s,
                model_place_probs=place_s,
                odds_df=race_odds,
                field_size=len(race),
                min_edge=min_edge,
            )

            if bets.empty:
                st.info("No value bets found above the minimum edge threshold.")
            else:
                for _, bet in bets.iterrows():
                    stake = kelly_stake_hkd(
                        bet["model_prob"], bet["decimal_odds"],
                        bankroll, kelly_frac
                    )
                    colour = "green" if bet["edge"] >= 0.10 else "orange"
                    st.markdown(
                        f"<div style='padding:8px;border-left:4px solid {colour};"
                        f"margin-bottom:6px;background:#1e1e1e'>"
                        f"<b>{bet['pool_type']}</b> — Combination: <b>{bet['comb_string']}</b><br>"
                        f"Model: {bet['model_prob']:.1%} | Market: {bet['market_prob']:.1%} | "
                        f"Edge: <b>{bet['edge']:.1%}</b> | Odds: {bet['decimal_odds']:.1f}<br>"
                        f"Kelly stake: <b>HKD {stake:,.0f}</b> "
                        f"({stake/bankroll*100:.1f}% of bankroll)"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Result logger
        with st.expander("📝 Log Race Result"):
            result_order = st.text_input(
                "Finishing order (comma-separated horse numbers)",
                key=f"result_{race_no}"
            )
            w_div  = st.number_input("WIN dividend (HKD per 10)", 0.0, key=f"wdiv_{race_no}")
            p_div  = st.number_input("PLA dividend (HKD per 10)", 0.0, key=f"pdiv_{race_no}")
            qin_div = st.number_input("QIN dividend (HKD per 10)", 0.0, key=f"qindiv_{race_no}")
            qpl_div = st.number_input("QPL dividend (HKD per 10)", 0.0, key=f"qpldiv_{race_no}")

            if st.button(f"Save R{race_no} result", key=f"save_{race_no}"):
                log_rows = []
                for pt, div in [("WIN", w_div), ("PLA", p_div),
                                 ("QIN", qin_div), ("QPL", qpl_div)]:
                    if div > 0:
                        log_rows.append({
                            "race_date": date_str,
                            "venue": venue,
                            "race_no": race_no,
                            "pool_type": pt,
                            "winning_combination": result_order,
                            "dividend_hkd": div,
                        })
                if log_rows:
                    from pathlib import Path
                    yr   = race_date.year
                    fp   = cfg.RAW_DIVIDENDS_DIR / f"dividends_{yr}.parquet"
                    from storage.parquet_store import write as ps_write
                    existing = read(fp)
                    combined = pd.concat(
                        [existing, pd.DataFrame(log_rows)], ignore_index=True
                    )
                    ps_write(combined, fp)
                    st.success("Result logged.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()