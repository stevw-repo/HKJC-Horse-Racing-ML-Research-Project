# features/race_features.py
"""
Race-level features — constant across all runners in a given race.
"""
import math

import pandas as pd


_CLASS_ORDER = {
    "griffin": 0, "class 5": 1, "class 4": 2, "class 3": 3,
    "class 2": 4, "class 1": 5, "listed": 6,
    "group 3": 6, "group 2": 7, "group 1": 8,
}
_GOING_ORDER = {
    "firm": 0, "good to firm": 1, "good": 2,
    "good to yielding": 3, "yielding": 4, "soft": 5, "heavy": 6,
}


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race-level features (operates in-place on a copy)."""
    df = df.copy()

    # Distance category
    df["distance_cat"] = pd.cut(
        df["distance_m"].fillna(0),
        bins=[0, 1200, 1800, 9999],
        labels=["sprint", "middle", "long"],
    ).astype(str)

    # Venue code
    df["venue_code"] = df["venue"].map({"ST": 0, "HV": 1}).fillna(-1).astype(int)

    # Going ordinal
    df["going_code"] = (
        df["going"].str.lower().str.strip()
        .map(_GOING_ORDER).fillna(-1).astype(int)
    )

    # Race class ordinal
    df["race_class_code"] = (
        df["race_class"].str.lower().str.strip()
        .map(_CLASS_ORDER).fillna(-1).astype(int)
    )

    # Log prize
    df["prize_hkd_log"] = (
        df["prize_hkd"].clip(lower=1).apply(lambda x: math.log(x) if pd.notna(x) else 0)
    )

    # Field size
    race_id_cols = ["race_date", "venue", "race_no"]
    df["field_size"] = df.groupby(race_id_cols)["horse_no"].transform("count")

    # Month cyclic
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    month = df["race_date_dt"].dt.month
    df["month_sin"] = (month * 2 * math.pi / 12).apply(math.sin)
    df["month_cos"] = (month * 2 * math.pi / 12).apply(math.cos)

    # Day-of-week cyclic
    dow = df["race_date_dt"].dt.dayofweek
    df["dow_sin"] = (dow * 2 * math.pi / 7).apply(math.sin)
    df["dow_cos"] = (dow * 2 * math.pi / 7).apply(math.cos)

    # Night meeting flag (HV only, typically weeknight)
    df["is_night"] = ((df["venue"] == "HV") &
                      (df["race_date_dt"].dt.dayofweek < 5)).astype(int)

    df = df.drop(columns=["race_date_dt"], errors="ignore")
    return df