# features/pipeline.py
"""
Master feature assembly pipeline.
Joins all raw data sources and computes all feature columns.
"""
import logging

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from storage.parquet_store import read, write
from features.race_features import add_race_features
from features.horse_features import add_horse_features
from features.jockey_trainer_features import add_jockey_trainer_features
from features.odds_features import add_odds_features

logger = logging.getLogger(__name__)


def build_training_features(results_start: str = None,
                             results_end: str = None) -> pd.DataFrame:
    """
    Build the full flat feature table from all historical raw stores.
    Saves to data/processed/features_train.parquet and returns DataFrame.
    """
    from config import (RAW_RESULTS_DIR, RAW_HORSES_DIR, RAW_ODDS_DIR,
                        RAW_DIVIDENDS_DIR, DIVIDENDS_CUTOFF_YEAR)

    logger.info("Loading raw results …")
    results_df = read(RAW_RESULTS_DIR)
    if results_df.empty:
        raise RuntimeError("No results data found. Run the results scraper first.")

    results_df["race_date"] = pd.to_datetime(results_df["race_date"])
    if results_start:
        results_df = results_df[results_df["race_date"] >= results_start]
    if results_end:
        results_df = results_df[results_df["race_date"] <= results_end]

    # Build targets
    results_df["target_win"]   = (results_df["placing"] == 1).astype(int)
    results_df["target_place"] = (results_df["placing"] <= 3).astype(int)

    # Race ID
    results_df["race_id"] = (
        results_df["venue"] + "_" +
        results_df["race_date"].dt.strftime("%Y%m%d") + "_" +
        results_df["race_no"].astype(str)
    )

    # Filter invalid rows
    results_df = results_df[
        results_df["placing_code"].isna() &          # remove WV/ML etc.
        results_df["placing"].notna()
    ]
    race_sizes = results_df.groupby("race_id")["horse_no"].transform("count")
    results_df = results_df[race_sizes >= 4]

    logger.info("Applying race-level features …")
    df = add_race_features(results_df)

    logger.info("Loading horse profiles …")
    profiles = read(RAW_HORSES_DIR / "horse_profiles.parquet")

    logger.info("Applying horse features …")
    df = add_horse_features(df, profiles)

    logger.info("Applying jockey/trainer features …")
    df = add_jockey_trainer_features(df)

    logger.info("Loading odds …")
    odds_df = read(RAW_ODDS_DIR)

    logger.info("Applying odds features …")
    df = add_odds_features(df, odds_df)

    # Merge dividends (for backtest reference — not model features)
    logger.info("Loading dividends …")
    divs_df = read(RAW_DIVIDENDS_DIR)
    if not divs_df.empty:
        divs_df["race_date"] = pd.to_datetime(divs_df["race_date"])
        win_divs = divs_df[divs_df["pool_type"] == "WIN"].rename(
            columns={"dividend_hkd": "dividend_win",
                     "winning_combination": "win_combination"}
        )[["race_date", "venue", "race_no", "win_combination", "dividend_win"]]
        place_divs = divs_df[divs_df["pool_type"] == "PLA"].rename(
            columns={"dividend_hkd": "dividend_place",
                     "winning_combination": "place_combination"}
        )[["race_date", "venue", "race_no", "place_combination", "dividend_place"]]
        df = df.merge(win_divs,   on=["race_date", "venue", "race_no"], how="left")
        df = df.merge(place_divs, on=["race_date", "venue", "race_no"], how="left")
    else:
        df["dividend_win"]   = np.nan
        df["dividend_place"] = np.nan

    out_path = PROCESSED_DIR / "features_train.parquet"
    write(df, out_path)
    logger.info("features_train.parquet written — %d rows.", len(df))
    return df


def build_prediction_features(race_date: str, venue: str) -> pd.DataFrame:
    """
    Build features for today's runners using the racecard + live odds.
    Saves to data/processed/features_predict.parquet and returns DataFrame.
    """
    from config import (RAW_RACECARDS_DIR, RAW_HORSES_DIR, RAW_ODDS_DIR,
                        RAW_RESULTS_DIR)

    tag        = race_date.replace("-", "")
    card_path  = RAW_RACECARDS_DIR / f"racecard_{tag}.parquet"
    racecard   = read(card_path)
    if racecard.empty:
        raise RuntimeError(
            f"No racecard found for {race_date}. Run the racecard scraper first."
        )

    racecard = racecard[racecard["venue"] == venue]
    racecard["race_date"] = pd.to_datetime(race_date)
    racecard["placing"]   = np.nan  # unknown for today

    # Align column names to results schema
    col_renames = {
        "horse_name_en": "horse_name",
        "barrier_draw":  "draw",
        "handicap_weight": "actual_weight_lbs",
    }
    racecard = racecard.rename(columns={k: v for k, v in col_renames.items()
                                        if k in racecard.columns})

    # Fill missing standard columns
    for col in ["placing_code", "lbw", "win_odds", "gear",
                "declared_horse_weight_lbs", "prize_hkd",
                "race_class", "going", "distance_m"]:
        if col not in racecard.columns:
            racecard[col] = None

    # Supplement with historical results for form features
    hist_df = read(RAW_RESULTS_DIR)
    if not hist_df.empty:
        hist_df["race_date"] = pd.to_datetime(hist_df["race_date"])
        hist_df = hist_df[hist_df["race_date"] < pd.Timestamp(race_date)]

    combined = pd.concat([hist_df, racecard], ignore_index=True) if not hist_df.empty else racecard

    df = add_race_features(combined)

    profiles = read(RAW_HORSES_DIR / "horse_profiles.parquet")
    df = add_horse_features(df, profiles)
    df = add_jockey_trainer_features(df)

    odds_df = read(RAW_ODDS_DIR / f"odds_{tag}.parquet")
    df = add_odds_features(df, odds_df)

    # Return only today's runners
    df = df[df["race_date"] == pd.Timestamp(race_date)].copy()
    df["race_id"] = (venue + "_" + race_date.replace("-", "") + "_" +
                     df["race_no"].astype(str))

    out_path = PROCESSED_DIR / "features_predict.parquet"
    write(df, out_path)
    logger.info("features_predict.parquet written — %d runners.", len(df))
    return df