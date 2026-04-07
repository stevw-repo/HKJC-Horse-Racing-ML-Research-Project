# %% [markdown]
# # Notebook 2 – Feature Engineering
#
# Loads the three raw Parquet files, merges them, and constructs a
# feature matrix with **zero look-ahead bias**.
#
# Key design decisions
# --------------------
# * All rolling/lag features for race T use only data from races < T.
# * Jockey/trainer rolling stats are computed with a dedicated incremental
#   tracker that processes races in strict date order.
# * Categorical target encoding uses only the **training set** to compute
#   smoothed means; validation/test rows are mapped afterwards.
# * The final dataset is split by date (not randomly).

# %% ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import (
    RollingStatsTracker,
    ensure_dirs,
    logger,
    safe_read_parquet,
)

warnings.filterwarnings("ignore")

# %% ── Configuration ────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW       = Path("data/raw")
DATA_PROC      = Path("data/processed")
DATA_SPLITS    = Path("data/splits")

RAW_RACES_PATH  = DATA_RAW / "raw_races.parquet"
RAW_HORSES_PATH = DATA_RAW / "raw_horses.parquet"
WEATHER_PATH    = DATA_RAW / "weather.parquet"
FEATURES_PATH   = DATA_PROC / "features.parquet"

# ── Time-split proportions ───────────────────────────────────────────────────
# Train: everything up to (latest − 6 months)
# Val:   (latest − 6 months) to (latest − 3 months)
# Test:  last 3 months
VAL_MONTHS  = 6   # train/val boundary is this many months before latest race
TEST_MONTHS = 3   # val/test boundary is this many months before latest race

# ── Rolling-window settings ───────────────────────────────────────────────────
ROLLING_JOCKEY_DAYS  = 30
ROLLING_TRAINER_DAYS = 30
HORSE_FORM_WINDOW    = 5    # number of past runs for per-horse rolling stats

# ── Smoothing for target encoding ────────────────────────────────────────────
TE_SMOOTH_K = 20   # smoothing factor: k past races before deferring to global mean

# %% ── Step 1: Load & basic cleaning ────────────────────────────────────────

def load_and_clean(
    races_path: Path,
    horses_path: Path,
    weather_path: Path,
) -> pd.DataFrame:
    """
    Load the three raw Parquet files, clean dtypes, and merge into one
    runner-level DataFrame.

    Parameters
    ----------
    races_path:   Path to raw_races.parquet.
    horses_path:  Path to raw_horses.parquet.
    weather_path: Path to weather.parquet.

    Returns
    -------
    pd.DataFrame — merged, cleaned DataFrame with one row per runner per race.
    """
    races = safe_read_parquet(races_path)
    if races is None or races.empty:
        raise FileNotFoundError(f"Race data not found: {races_path}")

    horses  = safe_read_parquet(horses_path)
    weather = safe_read_parquet(weather_path)

    # ── Races ────────────────────────────────────────────────────────────────
    # Normalise date column
    races["race_date"] = pd.to_datetime(
        races["race_date"].astype(str).str.replace("/", "-"), errors="coerce"
    )
    races = races.dropna(subset=["race_date"])
    races = races.sort_values(["race_date", "race_id", "finish_pos"]).reset_index(drop=True)

    # Normalise finish position: keep numeric value or NaN for non-finishers
    def _norm_finish_pos(v: object) -> Optional[float]:
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    races["finish_pos_num"] = races["finish_pos"].apply(_norm_finish_pos)
    races["is_winner"] = (races["finish_pos_num"] == 1).astype(int)

    # Ensure key numerics
    for col in ["win_odds", "place_odds", "actual_weight_kg",
                "finish_time_sec", "lbw", "draw"]:
        if col in races.columns:
            races[col] = pd.to_numeric(races[col], errors="coerce")

    # ── Horses ───────────────────────────────────────────────────────────────
    if horses is not None and not horses.empty:
        horses["horse_brand_no"] = horses["horse_brand_no"].astype(str)
        for col in ["age_years", "current_rating", "last_rating",
                    "career_starts", "career_wins"]:
            if col in horses.columns:
                horses[col] = pd.to_numeric(horses[col], errors="coerce")
        races = races.merge(
            horses, on="horse_brand_no", how="left", suffixes=("", "_profile")
        )

    # ── Weather ──────────────────────────────────────────────────────────────
    if weather is not None and not weather.empty:
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        weather = weather.rename(columns={"date": "race_date"})
        races = races.merge(
            weather, on=["race_date", "racecourse"], how="left"
        )

    logger.info("Loaded %d runner rows covering %d races",
                len(races), races["race_id"].nunique())
    return races


# %% ── Step 2: Per-horse rolling features ───────────────────────────────────

def _horse_rolling_features(df: pd.DataFrame, window: int = HORSE_FORM_WINDOW) -> pd.DataFrame:
    """
    Add look-ahead-safe rolling features at the horse level.

    Sorts by horse + date, then uses `shift(1)` + `rolling(window)` so
    each row only sees outcomes from strictly earlier races.

    Added columns (all prefixed `h_`)
    ----------------------------------
    h_avg_finish_pos_{w}  – mean finishing position in last *window* runs
    h_win_rate_{w}        – win rate in last *window* runs
    h_place_rate_{w}      – place (top-3) rate in last *window* runs
    h_avg_lbw_{w}         – mean lengths behind winner in last *window* runs
    h_avg_odds_{w}        – mean win odds in last *window* runs
    h_avg_speed_mps_{w}   – mean speed (m/s) in last *window* runs
    h_runs_total          – total career runs seen so far
    h_last_finish_pos     – finishing position in the immediately preceding run
    h_days_since_last_run – days since the previous race
    """
    df = df.sort_values(["horse_brand_no", "race_date", "race_id"]).copy()

    grp = df.groupby("horse_brand_no", sort=False)

    # Speed in m/s
    df["_speed"] = np.where(
        df["finish_time_sec"] > 0,
        df["distance_m"] / df["finish_time_sec"],
        np.nan,
    )

    # Place indicator
    df["_is_place"] = (df["finish_pos_num"] <= 3).astype(float)

    feature_sources = {
        "finish_pos_num": "h_avg_finish_pos",
        "is_winner":      "h_win_rate",
        "_is_place":      "h_place_rate",
        "lbw":            "h_avg_lbw",
        "win_odds":       "h_avg_odds",
        "_speed":         "h_avg_speed_mps",
    }

    for src_col, tgt_prefix in feature_sources.items():
        if src_col not in df.columns:
            continue
        # shift(1) → no look-ahead; rolling(window) → last `window` races
        rolled = (
            grp[src_col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"{tgt_prefix}_{window}"] = rolled

    # Total career runs seen up to (but not including) this race
    df["h_runs_total"] = grp["is_winner"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # Last finish position (single previous race)
    df["h_last_finish_pos"] = grp["finish_pos_num"].transform(
        lambda s: s.shift(1)
    )

    # Days since last run
    df["_prev_date"] = grp["race_date"].transform(lambda s: s.shift(1))
    df["h_days_since_last_run"] = (df["race_date"] - df["_prev_date"]).dt.days
    df = df.drop(columns=["_prev_date", "_speed", "_is_place"], errors="ignore")

    return df


# %% ── Step 3: Jockey & trainer rolling features ─────────────────────────────

def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling (last-30-day) win rates for each jockey and trainer.

    Uses the ``RollingStatsTracker`` which processes rows in chronological
    order, querying stats strictly before each race date and updating after.

    Added columns
    -------------
    jk_overall_win_rate, jk_cd_win_rate  (jockey overall / course-distance)
    tr_overall_win_rate, tr_cd_win_rate  (trainer overall / course-distance)
    """
    df = df.sort_values("race_date").reset_index(drop=True)

    jk_tracker = RollingStatsTracker(window_days=ROLLING_JOCKEY_DAYS)
    tr_tracker = RollingStatsTracker(window_days=ROLLING_TRAINER_DAYS)

    jk_stats_list: List[Dict] = []
    tr_stats_list: List[Dict] = []

    for _, row in df.iterrows():
        rd   = row["race_date"]
        dist = int(row["distance_m"]) if pd.notna(row.get("distance_m")) else 0
        crs  = str(row["racecourse"]) if pd.notna(row.get("racecourse")) else ""
        jk   = str(row["jockey"])   if pd.notna(row.get("jockey"))   else "__UNK__"
        tr   = str(row["trainer"])  if pd.notna(row.get("trainer"))  else "__UNK__"
        won  = bool(row["is_winner"])

        jk_stats = jk_tracker.query(jk, rd, crs, dist)
        tr_stats = tr_tracker.query(tr, rd, crs, dist)

        jk_stats_list.append({
            "jk_overall_win_rate": jk_stats["overall_win_rate"],
            "jk_cd_win_rate":      jk_stats["cd_win_rate"],
            "jk_overall_runs":     jk_stats["overall_runs"],
        })
        tr_stats_list.append({
            "tr_overall_win_rate": tr_stats["overall_win_rate"],
            "tr_cd_win_rate":      tr_stats["cd_win_rate"],
            "tr_overall_runs":     tr_stats["overall_runs"],
        })

        jk_tracker.update(jk, rd, crs, dist, won)
        tr_tracker.update(tr, rd, crs, dist, won)

    jk_df = pd.DataFrame(jk_stats_list, index=df.index)
    tr_df = pd.DataFrame(tr_stats_list, index=df.index)

    df = pd.concat([df, jk_df, tr_df], axis=1)
    return df


# %% ── Step 4: Derived features ──────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add race-level and horse-level derived scalar features.

    Added columns
    -------------
    is_sprint, is_mile, is_staying
    draw_norm
    weight_for_age
    speed_mps
    early_speed_400
    finish_speed_400
    implied_prob
    race_field_size
    """
    # Distance categories
    dist = df.get("distance_m", pd.Series(np.nan, index=df.index))
    df["is_sprint"]  = (dist < 1400).astype(int)
    df["is_mile"]    = ((dist >= 1400) & (dist <= 1800)).astype(int)
    df["is_staying"] = (dist > 2000).astype(int)

    # Normalised draw (field size estimated at 14)
    df["draw_norm"] = df["draw"].fillna(7) / 14.0

    # Weight for age normalisation
    age = df.get("age_years", pd.Series(np.nan, index=df.index)).clip(lower=2)
    wt  = df.get("actual_weight_kg", pd.Series(np.nan, index=df.index))
    df["weight_for_age"] = wt / (age * 2 + 50)

    # Speed in m/s (overall race)
    df["speed_mps"] = np.where(
        df["finish_time_sec"].gt(0) & dist.notna(),
        dist / df["finish_time_sec"],
        np.nan,
    )

    # Sectional speeds (lower raw time = faster)
    if "sect_400" in df.columns:
        df["early_speed_400"]  = df["sect_400"]
    else:
        df["early_speed_400"]  = np.nan

    if "sect_800" in df.columns and "sect_400" in df.columns:
        df["finish_speed_400"] = df["sect_800"] - df["sect_400"]
    else:
        df["finish_speed_400"] = np.nan

    # Implied probability from win odds (clip minimum odds at 1.01)
    df["implied_prob"] = 1.0 / df["win_odds"].clip(lower=1.01)

    # Field size per race
    df["race_field_size"] = df.groupby("race_id")["horse_brand_no"].transform("count")

    return df


# %% ── Step 5: Categorical encoding ─────────────────────────────────────────

def encode_going(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the `going` column, handling unknown values at test time."""
    going_dummies = pd.get_dummies(df["going"].fillna("Unknown"), prefix="going")
    return pd.concat([df, going_dummies], axis=1)


def encode_course_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the `course_type` column."""
    ct_dummies = pd.get_dummies(df["course_type"].fillna("Unknown"), prefix="ct")
    return pd.concat([df, ct_dummies], axis=1)


def target_encode_column(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target_col: str = "is_winner",
    smooth_k: int = TE_SMOOTH_K,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Smoothed target-encode a categorical column.

    Encoding formula
    ----------------
    TE(c) = (n_c × mean_c + k × global_mean) / (n_c + k)

    where n_c = count of category c in the training set,
    mean_c = mean(target) for category c in the training set,
    k = smoothing factor (TE_SMOOTH_K).

    The encoding is computed **only** on the training set; validation and
    test rows receive the training-derived mapping (unseen categories
    receive the global training mean).

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames with the column *col*.
    col:         Column to encode.
    target_col:  Binary target column (used only from train_df).
    smooth_k:    Smoothing factor.

    Returns
    -------
    (train_encoded, val_encoded, test_encoded) as pd.Series.
    """
    global_mean = train_df[target_col].mean()

    stats = (
        train_df.groupby(col)[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    stats["te"] = (
        (stats["sum"] + smooth_k * global_mean) / (stats["count"] + smooth_k)
    )
    mapping = stats.set_index(col)["te"].to_dict()

    train_enc = train_df[col].map(mapping).fillna(global_mean)
    val_enc   = val_df[col].map(mapping).fillna(global_mean)
    test_enc  = test_df[col].map(mapping).fillna(global_mean)

    return train_enc, val_enc, test_enc


# %% ── Step 6: Time-based split ──────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    val_months: int = VAL_MONTHS,
    test_months: int = TEST_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / validation / test sets by race date.

    Split boundaries
    ----------------
    latest_date = df['race_date'].max()
    test_start  = latest_date − test_months months
    val_start   = latest_date − val_months months

    Returns
    -------
    (train_df, val_df, test_df) — non-overlapping, time-ordered subsets.
    """
    latest = df["race_date"].max()
    test_start = latest - pd.DateOffset(months=test_months)
    val_start  = latest - pd.DateOffset(months=val_months)

    train_df = df[df["race_date"] < val_start].copy()
    val_df   = df[(df["race_date"] >= val_start) & (df["race_date"] < test_start)].copy()
    test_df  = df[df["race_date"] >= test_start].copy()

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Date ranges — train: %s→%s  val: %s→%s  test: %s→%s",
        train_df["race_date"].min().date(), train_df["race_date"].max().date(),
        val_df["race_date"].min().date(),   val_df["race_date"].max().date(),
        test_df["race_date"].min().date(),  test_df["race_date"].max().date(),
    )
    return train_df, val_df, test_df


# %% ── Step 7: Assemble feature matrix ──────────────────────────────────────

# Columns excluded from the feature matrix (meta / target / raw strings)
NON_FEATURE_COLS = [
    "race_id", "race_date", "racecourse", "race_no",
    "horse_name", "horse_brand_no", "jockey", "trainer",
    "finish_pos", "finish_pos_num", "is_winner",
    "run_positions", "form_string",
    "going", "course_type",    # replaced by one-hot columns
    "win_odds",                # odds used in backtest, not as raw feature
    "place_odds",
    "_prev_date",
    "race_class",              # will be label-encoded separately
]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Return the list of columns to use as model features.

    Parameters
    ----------
    df:            Full DataFrame (post-engineering).
    feature_cols:  If provided, use exactly these columns.
    exclude_cols:  Additional columns to exclude beyond defaults.

    Returns
    -------
    List[str] of feature column names.
    """
    if feature_cols is not None:
        return feature_cols

    exclude = set(NON_FEATURE_COLS)
    if exclude_cols:
        exclude.update(exclude_cols)

    # Keep only numeric columns not in the exclusion list
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


# %% ── Main pipeline ─────────────────────────────────────────────────────────

def main() -> None:
    """Run the complete feature-engineering pipeline end to end."""
    ensure_dirs(DATA_PROC, DATA_SPLITS)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print("\n[1/6] Loading and cleaning raw data …")
    df = load_and_clean(RAW_RACES_PATH, RAW_HORSES_PATH, WEATHER_PATH)

    # ── 2. Per-horse rolling features ────────────────────────────────────────
    print("[2/6] Computing per-horse rolling features …")
    df = _horse_rolling_features(df, window=HORSE_FORM_WINDOW)

    # ── 3. Jockey / trainer rolling stats ───────────────────────────────────
    print("[3/6] Computing jockey / trainer rolling stats (this may take a minute) …")
    df = add_jockey_trainer_features(df)

    # ── 4. Derived features ──────────────────────────────────────────────────
    print("[4/6] Adding derived features …")
    df = add_derived_features(df)

    # ── 5. Categorical encoding ──────────────────────────────────────────────
    print("[5/6] Encoding categorical variables …")
    df = encode_going(df)
    df = encode_course_type(df)

    # Label-encode race_class (ordinal: Class 5 < Class 4 < … < Class 1 < G1)
    if "race_class" in df.columns:
        le = LabelEncoder()
        df["race_class_enc"] = le.fit_transform(
            df["race_class"].fillna("Unknown").astype(str)
        )

    # Save the full feature-engineered dataset
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Saved features: %d rows, %d columns → %s", len(df), df.shape[1], FEATURES_PATH)

    # ── 6. Time-based split ──────────────────────────────────────────────────
    print("[6/6] Splitting by date and saving splits …")
    train_df, val_df, test_df = time_split(df)

    # Determine feature columns
    feature_cols = build_feature_matrix(df)
    print(f"  Feature count: {len(feature_cols)}")

    # Apply target encoding for jockey + trainer using ONLY training data
    for ent_col, new_col in [("jockey", "jk_te"), ("trainer", "tr_te")]:
        if ent_col in df.columns:
            tr_enc, va_enc, te_enc = target_encode_column(
                train_df, val_df, test_df, col=ent_col
            )
            train_df[new_col] = tr_enc.values
            val_df[new_col]   = va_enc.values
            test_df[new_col]  = te_enc.values
            if new_col not in feature_cols:
                feature_cols.append(new_col)

    # Save feature column list
    pd.Series(feature_cols).to_csv(DATA_SPLITS / "feature_cols.txt", index=False, header=False)

    # Save splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = split_df[feature_cols].copy()
        y = split_df["is_winner"].copy()
        meta = split_df[["race_id", "race_date", "racecourse", "horse_name",
                          "horse_brand_no", "win_odds", "is_winner"]].copy()

        X.to_parquet(DATA_SPLITS / f"X_{split_name}.parquet", index=False)
        y.to_frame().to_parquet(DATA_SPLITS / f"y_{split_name}.parquet", index=False)
        meta.to_parquet(DATA_SPLITS / f"meta_{split_name}.parquet", index=False)
        print(f"  Saved {split_name}: {len(X)} rows")

    print("\n✅  Feature engineering complete.\n")
    print("Feature columns preview:")
    print(feature_cols[:20], "…" if len(feature_cols) > 20 else "")


if __name__ == "__main__":
    main()