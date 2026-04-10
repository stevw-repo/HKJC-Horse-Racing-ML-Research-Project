# features/jockey_trainer_features.py
"""
Jockey and trainer form features — computed from the results table
with strict historical lookback to avoid leakage.
"""
import numpy as np
import pandas as pd

from config import MIN_RACES_FOR_RATE


def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain race_date, jockey_name, trainer_name, placing,
    venue, race_class_code columns.
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values("race_date").reset_index(drop=True)

    # ── Jockey features ───────────────────────────────────────────────────────
    df = _add_person_features(df, "jockey_name",
                               prefix="jockey", window_days=60)

    # ── Trainer features ──────────────────────────────────────────────────────
    df = _add_person_features(df, "trainer_name",
                               prefix="trainer", window_days=60)

    # ── Jockey × Trainer combo ────────────────────────────────────────────────
    df["jt_pair"] = df["jockey_name"].astype(str) + "||" + df["trainer_name"].astype(str)
    df = _add_pair_features(df)

    return df


def _add_person_features(df: pd.DataFrame, name_col: str,
                          prefix: str, window_days: int) -> pd.DataFrame:
    win_rate_col    = f"{prefix}_win_rate_{window_days}d"
    place_rate_col  = f"{prefix}_place_rate_{window_days}d"
    season_wr_col   = f"{prefix}_season_win_rate"
    class_wr_col    = f"{prefix}_win_rate_this_class"
    venue_wr_col    = f"{prefix}_win_rate_this_venue"

    df[win_rate_col]   = np.nan
    df[place_rate_col] = np.nan
    df[season_wr_col]  = np.nan
    df[class_wr_col]   = np.nan
    df[venue_wr_col]   = np.nan

    for idx, row in df.iterrows():
        person       = row[name_col]
        current_date = row["race_date"]
        season_start = pd.Timestamp(f"{current_date.year}-09-01")
        if current_date.month < 9:
            season_start = pd.Timestamp(f"{current_date.year - 1}-09-01")
        window_start = current_date - pd.Timedelta(days=window_days)

        mask_person  = df[name_col] == person
        mask_past    = df["race_date"] < current_date

        # Rolling window
        mask_window  = mask_person & mask_past & (df["race_date"] >= window_start)
        subset       = df[mask_window]
        if len(subset) >= MIN_RACES_FOR_RATE:
            df.at[idx, win_rate_col]   = (subset["placing"] == 1).mean()
            df.at[idx, place_rate_col] = (subset["placing"] <= 3).mean()

        # Season
        mask_season  = mask_person & mask_past & (df["race_date"] >= season_start)
        season_sub   = df[mask_season]
        if len(season_sub) >= MIN_RACES_FOR_RATE:
            df.at[idx, season_wr_col] = (season_sub["placing"] == 1).mean()

        # Class
        if "race_class_code" in df.columns:
            mask_class = mask_person & mask_past & (
                df["race_class_code"] == row.get("race_class_code", -1)
            )
            cls_sub = df[mask_class]
            if len(cls_sub) >= MIN_RACES_FOR_RATE:
                df.at[idx, class_wr_col] = (cls_sub["placing"] == 1).mean()

        # Venue
        mask_venue = mask_person & mask_past & (df["venue"] == row["venue"])
        v_sub = df[mask_venue]
        if len(v_sub) >= MIN_RACES_FOR_RATE:
            df.at[idx, venue_wr_col] = (v_sub["placing"] == 1).mean()

    return df


def _add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    df["combo_win_rate"]   = np.nan
    df["combo_place_rate"] = np.nan
    df["combo_ride_count"] = 0

    for idx, row in df.iterrows():
        pair      = row["jt_pair"]
        mask_past = (df["jt_pair"] == pair) & (df["race_date"] < row["race_date"])
        subset    = df[mask_past]
        count     = len(subset)
        df.at[idx, "combo_ride_count"] = count
        if count >= MIN_RACES_FOR_RATE:
            df.at[idx, "combo_win_rate"]   = (subset["placing"] == 1).mean()
            df.at[idx, "combo_place_rate"]  = (subset["placing"] <= 3).mean()

    return df