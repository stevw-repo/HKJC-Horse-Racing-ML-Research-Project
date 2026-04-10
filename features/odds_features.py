# features/odds_features.py
"""
Derive implied-probability features from WIN, PLA, QIN, QPL odds.
Only the four in-scope pool types are processed.
"""
import math

import numpy as np
import pandas as pd

from config import VALID_POOL_TYPES


def add_odds_features(df: pd.DataFrame,
                      odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    df       — main feature DataFrame (one row per runner per race).
    odds_df  — odds store rows (race_date, venue, race_no,
                                pool_type, comb_string, odds_value).
    Returns df with odds feature columns added.
    """
    if odds_df.empty:
        df["win_odds_closing"]          = np.nan
        df["win_implied_prob"]          = np.nan
        df["win_implied_prob_norm"]     = np.nan
        df["place_implied_prob_norm"]   = np.nan
        df["log_win_odds"]              = np.nan
        df["market_rank"]               = np.nan
        df["market_rank_norm"]          = np.nan
        df["odds_drop_pct"]             = np.nan
        return df

    # Only use in-scope pool types
    odds_df = odds_df[odds_df["pool_type"].isin(VALID_POOL_TYPES)].copy()

    # ── WIN odds per runner ───────────────────────────────────────────────────
    win_odds = (
        odds_df[odds_df["pool_type"] == "WIN"]
        .rename(columns={"comb_string": "horse_no_str",
                          "odds_value":  "win_odds_closing"})
        [["race_date", "venue", "race_no", "horse_no_str", "win_odds_closing"]]
    )
    df["horse_no_str"] = df["horse_no"].astype(str).str.zfill(2)
    df = df.merge(win_odds, on=["race_date", "venue", "race_no", "horse_no_str"],
                  how="left")

    df["win_implied_prob"] = 1.0 / df["win_odds_closing"].replace(0, np.nan)

    # Normalise to remove overround
    race_id_cols = ["race_date", "venue", "race_no"]
    total_impl   = df.groupby(race_id_cols)["win_implied_prob"].transform("sum")
    df["win_implied_prob_norm"] = df["win_implied_prob"] / total_impl.replace(0, np.nan)

    df["log_win_odds"]  = df["win_odds_closing"].apply(
        lambda x: math.log(x) if pd.notna(x) and x > 0 else np.nan
    )
    df["market_rank"]   = df.groupby(race_id_cols)["win_odds_closing"].rank(
        method="min", ascending=True
    )
    df["market_rank_norm"] = df["market_rank"] / df["field_size"].replace(0, np.nan)

    # ── PLACE implied prob ────────────────────────────────────────────────────
    pla_odds = (
        odds_df[odds_df["pool_type"] == "PLA"]
        .rename(columns={"comb_string": "horse_no_str",
                          "odds_value":  "place_odds_closing"})
        [["race_date", "venue", "race_no", "horse_no_str", "place_odds_closing"]]
    )
    df = df.merge(pla_odds, on=["race_date", "venue", "race_no", "horse_no_str"],
                  how="left")
    df["place_implied_prob"] = 1.0 / df["place_odds_closing"].replace(0, np.nan)
    total_pla = df.groupby(race_id_cols)["place_implied_prob"].transform("sum")
    df["place_implied_prob_norm"] = (
        df["place_implied_prob"] / total_pla.replace(0, np.nan)
    )

    # ── Odds drop ─────────────────────────────────────────────────────────────
    if "odds_drop_value" in odds_df.columns:
        od = (
            odds_df[odds_df["pool_type"] == "WIN"]
            [["race_date", "venue", "race_no", "comb_string", "odds_drop_value"]]
            .rename(columns={"comb_string": "horse_no_str"})
        )
        df = df.merge(od, on=["race_date", "venue", "race_no", "horse_no_str"],
                      how="left")
        df.rename(columns={"odds_drop_value": "odds_drop_pct"}, inplace=True)
    else:
        df["odds_drop_pct"] = np.nan

    df = df.drop(columns=["horse_no_str", "place_odds_closing",
                           "place_implied_prob"], errors="ignore")
    return df