# features/horse_features.py
"""
Per-runner historical form features.
All lookbacks are strictly prior to the target race — no leakage.
"""
import math
import re

import numpy as np
import pandas as pd

from config import FORM_WINDOW_RACES, MIN_RACES_FOR_RATE


_COUNTRY_MAP = {c: i for i, c in enumerate(
    ["GB", "IRE", "AUS", "NZ", "FR", "USA", "ARG", "SA", "NZ",
     "GER", "JP", "HK", "OTHER"]
)}
_IMPORT_MAP = {"griffin": 0, "pp": 1, "ppg": 2}


def add_horse_features(df: pd.DataFrame,
                       profiles: pd.DataFrame) -> pd.DataFrame:
    """
    df       — merged results/racecard rows, sorted by (horse_id, race_date).
    profiles — horse_profiles DataFrame.
    Returns df with horse feature columns appended.
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    # ── Static profile features ───────────────────────────────────────────────
    if not profiles.empty:
        keep = ["horse_id", "import_year", "country_of_origin",
                "colour", "sex", "import_type", "is_retired"]
        keep = [c for c in keep if c in profiles.columns]
        df = df.merge(profiles[keep], on="horse_id", how="left")
    else:
        df["import_year"]        = np.nan
        df["country_of_origin"]  = None
        df["colour"]             = None
        df["sex"]                = None
        df["import_type"]        = None

    # import_year → years_since_import
    df["race_year"]          = df["race_date"].dt.year
    df["years_since_import"] = df["race_year"] - df["import_year"]

    # Encode
    df["country_code"]  = df["country_of_origin"].map(
        lambda x: _COUNTRY_MAP.get(str(x).upper(), len(_COUNTRY_MAP))
        if pd.notna(x) else -1
    )
    df["import_type_code"] = df["import_type"].map(
        lambda x: _IMPORT_MAP.get(str(x).lower(), 3) if pd.notna(x) else -1
    )

    # ── Rolling form features ─────────────────────────────────────────────────
    for W in FORM_WINDOW_RACES:
        df = _add_rolling_form(df, W)

    # ── Career aggregates ─────────────────────────────────────────────────────
    df = _add_career_stats(df)

    # ── Days-since-last-run ───────────────────────────────────────────────────
    df["days_since_last_run"] = (
        df.groupby("horse_id")["race_date"]
        .transform(lambda s: s.diff().dt.days)
    )
    df["races_in_last_30d"] = _races_in_window(df, 30)
    df["races_in_last_90d"] = _races_in_window(df, 90)

    # ── Weight features ───────────────────────────────────────────────────────
    _add_weight_features(df)

    # ── Gear features ─────────────────────────────────────────────────────────
    _add_gear_features(df)

    # ── Draw features ─────────────────────────────────────────────────────────
    _add_draw_features(df)

    # ── Debutant flag ─────────────────────────────────────────────────────────
    df["is_debutant"] = (df["career_starts"] < 3).fillna(True).astype(int)

    return df


# ── Rolling form helpers ──────────────────────────────────────────────────────

def _add_rolling_form(df: pd.DataFrame, W: int) -> pd.DataFrame:
    """Compute win/place rates and avg placing for last W races."""
    def _rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
        placing = group["placing"].shift(1)  # strict past only
        win   = (placing == 1).astype(float)
        place = (placing <= 3).astype(float)
        lbw   = group["lbw"].shift(1)
        odds  = group["win_odds"].shift(1)

        n = min(W, len(group))

        group[f"win_rate_{W}"]   = win.rolling(W, min_periods=1).mean()
        group[f"place_rate_{W}"] = place.rolling(W, min_periods=1).mean()
        group[f"avg_placing_{W}"]= placing.rolling(W, min_periods=1).mean()
        group[f"avg_lbw_{W}"]    = lbw.rolling(W, min_periods=1).mean()
        group[f"avg_win_odds_{W}"]= odds.rolling(W, min_periods=1).mean()

        # Implied prob vs actual rate
        implied = 1.0 / group[f"avg_win_odds_{W}"].replace(0, np.nan)
        group[f"odds_vs_place_rate_{W}"] = (
            implied / group[f"place_rate_{W}"].replace(0, np.nan)
        )
        return group

    df = df.groupby("horse_id", group_keys=False).apply(_rolling_stats)
    return df


def _add_career_stats(df: pd.DataFrame) -> pd.DataFrame:
    def _career(group: pd.DataFrame) -> pd.DataFrame:
        placing    = group["placing"].shift(1)
        wins       = (placing == 1).expanding().sum()
        places     = (placing <= 3).expanding().sum()
        starts     = placing.notna().expanding().sum()
        group["career_starts"]    = starts
        group["career_wins"]      = wins
        group["career_win_rate"]  = (wins / starts.replace(0, np.nan))
        group["career_place_rate"]= (places / starts.replace(0, np.nan))
        return group

    df = df.groupby("horse_id", group_keys=False).apply(_career)
    return df


def _races_in_window(df: pd.DataFrame, days: int) -> pd.Series:
    """Count races in the last `days` days per horse (exclusive of current)."""
    result = pd.Series(np.nan, index=df.index)
    for horse_id, group in df.groupby("horse_id"):
        for idx, row in group.iterrows():
            cutoff = row["race_date"] - pd.Timedelta(days=days)
            count  = ((group["race_date"] < row["race_date"]) &
                      (group["race_date"] >= cutoff)).sum()
            result.at[idx] = count
    return result


def _add_weight_features(df: pd.DataFrame):
    race_id_cols = ["race_date", "venue", "race_no"]
    df["weight_vs_field_avg"] = (
        df["actual_weight_lbs"] -
        df.groupby(race_id_cols)["actual_weight_lbs"].transform("mean")
    )
    df["horse_weight_change"] = (
        df.groupby("horse_id")["declared_horse_weight_lbs"]
        .transform(lambda s: s.diff())
    )


def _add_gear_features(df: pd.DataFrame):
    if "gear" not in df.columns:
        df["gear"] = ""
    prev_gear = df.groupby("horse_id")["gear"].shift(1).fillna("")
    df["gear_change_flag"] = (df["gear"] != prev_gear).astype(int)
    df["has_blinkers"]     = df["gear"].str.contains("B", na=False).astype(int)
    df["has_tongue_tie"]   = df["gear"].str.contains("TT", na=False).astype(int)
    # Count items marked as first-use (digit "1" following a gear code)
    df["new_gear_count"]   = df["gear"].str.count(r"[A-Z]+1").fillna(0).astype(int)


def _add_draw_features(df: pd.DataFrame):
    df["draw_percentile"] = (
        df["draw"] / df["field_size"].replace(0, np.nan)
    ).fillna(0.5)
    # Historical win rate by (venue, distance_cat, draw bucket)
    # Use available historical data — approximation via groupby mean
    if "distance_cat" in df.columns and "placing" in df.columns:
        df["draw_bucket"] = pd.cut(df["draw_percentile"],
                                   bins=[0, 0.25, 0.5, 0.75, 1.01],
                                   labels=["inside", "midinner", "midouter", "outside"])
        hist = (
            df[df["placing"] == 1]
            .groupby(["venue", "distance_cat", "draw_bucket"])
            .size()
            .div(df.groupby(["venue", "distance_cat", "draw_bucket"]).size())
            .reset_index(name="draw_win_rate_course_dist")
        )
        df = df.merge(hist, on=["venue", "distance_cat", "draw_bucket"], how="left")
    else:
        df["draw_win_rate_course_dist"] = np.nan