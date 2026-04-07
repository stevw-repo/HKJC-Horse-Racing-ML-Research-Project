"""
04_predict_upcoming.py
======================
Predict win probabilities for horses in an upcoming HKJC race card.

Workflow
--------
1. Scrape the race card (entries) for a given date / course / race number.
2. Load the historical feature dataset (data/processed/features.parquet)
   to reconstruct rolling/lag state for every horse, jockey, and trainer
   as of the race date.
3. Build a feature row per horse using the same transformations as
   Notebook 2, but WITHOUT a target column (race hasn't happened yet).
4. Load the best saved model and predict calibrated win probabilities.
5. Print a ranked card and (optionally) flag value bets.

Usage
-----
# Predict all races on a given day at Sha Tin:
    python 04_predict_upcoming.py --date 2025-06-15 --course ST

# Predict a specific race:
    python 04_predict_upcoming.py --date 2025-06-15 --course HV --race 3

# Show value bets only (EV > 5 %):
    python 04_predict_upcoming.py --date 2025-06-15 --course ST --bets-only
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils import (
    RollingStatsTracker,
    build_session,
    convert_distance_beaten,
    ensure_dirs,
    expected_value,
    kelly_bet_fraction,
    logger,
    safe_get,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors Notebook 2 — keep these in sync)
# ─────────────────────────────────────────────────────────────────────────────

DATA_RAW     = Path("data/raw")
DATA_PROC    = Path("data/processed")
DATA_SPLITS  = Path("data/splits")
DATA_MODELS  = Path("data/models")
DATA_RESULTS = Path("data/results")

# Which saved model to load for predictions.
# Change to "LightGBM.joblib", "Ensemble(XGB+LGB+RF).joblib", etc.
DEFAULT_MODEL_FILE = "Ensemble(XGB+LGB+RF).joblib"

# Fallback order if the default isn't found:
MODEL_FALLBACK_ORDER = [
    "Ensemble(XGB+LGB+RF).joblib",
    "XGBoost.joblib",
    "LightGBM.joblib",
    "RandomForest.joblib",
    "LogisticRegression.joblib",
]

# Feature-engineering constants (must match Notebook 2)
HORSE_FORM_WINDOW    = 5
ROLLING_JOCKEY_DAYS  = 30
ROLLING_TRAINER_DAYS = 30
TE_SMOOTH_K          = 20

# Betting filter thresholds
BET_EV_THRESHOLD = 0.05
BET_MIN_PROB     = 0.15
KELLY_FRAC       = 0.25
KELLY_MAX        = 0.05

# HKJC race card URL
RACECARD_URL_TPL = (
    "https://racing.hkjc.com/racing/information/English/"
    "Racing/RaceCard.aspx?RaceDate={date}&Racecourse={course}&RaceNo={race_no}"
)

MAX_RACE_NO = 11
SLEEP_SECS  = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Scrape the race card
# ─────────────────────────────────────────────────────────────────────────────

def _text(tag: Any) -> str:
    return tag.get_text(strip=True) if tag else ""


def scrape_race_card(
    session: Any,
    race_date: date,
    course: str,
    race_no: int,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Scrape the HKJC race card for one race.

    Returns
    -------
    (race_meta, runners)
    race_meta  – dict with distance_m, going, course_type, race_class, prize_hkd.
    runners    – list of dicts, one per declared runner (horse_name,
                 horse_brand_no, jockey, trainer, draw, declared_weight_kg,
                 win_odds).
    Returns (None, []) if the page is unavailable or has no table.
    """
    from bs4 import BeautifulSoup
    import re

    date_str = race_date.strftime("%Y/%m/%d")
    url = RACECARD_URL_TPL.format(date=date_str, course=course, race_no=race_no)
    resp = safe_get(session, url, sleep_secs=SLEEP_SECS)
    if resp is None:
        return None, []

    soup = BeautifulSoup(resp.text, "lxml")
    full_text = soup.get_text(separator=" ")

    # ── Race metadata ────────────────────────────────────────────────────────
    meta: Dict[str, Any] = {
        "race_date":   race_date,
        "racecourse":  course,
        "race_no":     race_no,
        "distance_m":  None,
        "going":       None,
        "course_type": None,
        "race_class":  None,
        "prize_hkd":   None,
    }

    m = re.search(r"(\d{3,4})\s*[Mm](?:etres?)?", full_text)
    if m:
        meta["distance_m"] = int(m.group(1))

    for token in ["Good To Firm", "Firm", "Good", "Good To Yielding",
                  "Yielding", "Soft", "Heavy", "Fast"]:
        if token.lower() in full_text.lower():
            meta["going"] = token
            break

    if re.search(r"all.weather|awt", full_text, re.I):
        meta["course_type"] = "AWT"
    elif "turf" in full_text.lower():
        meta["course_type"] = "Turf"

    m = re.search(r"Class\s+(\d)|Griffin|International", full_text, re.I)
    if m:
        meta["race_class"] = m.group(0).strip()

    m = re.search(r"HK\$\s*([\d,]+)", full_text)
    if m:
        meta["prize_hkd"] = int(m.group(1).replace(",", ""))

    # ── Runner table ─────────────────────────────────────────────────────────
    # Find the table most likely to contain horse entries
    runners: List[Dict] = []
    for table in soup.find_all("table"):
        headers = [_text(th).lower() for th in table.find_all("th")]
        if not (any("horse" in h for h in headers) and
                any("jockey" in h for h in headers)):
            continue

        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Build column index map
        header_cells = rows[0].find_all(["th", "td"])
        cmap: Dict[str, int] = {}
        col_patterns = {
            "horse_no":    r"no\.?$",
            "horse_name":  r"horse",
            "jockey":      r"jockey",
            "trainer":     r"trainer",
            "draw":        r"^dr$|draw",
            "weight":      r"wt|weight",
            "win_odds":    r"win\s*odds|odds",
        }
        for idx, cell in enumerate(header_cells):
            txt = _text(cell).lower()
            for cname, pat in col_patterns.items():
                if cname not in cmap and re.search(pat, txt):
                    cmap[cname] = idx

        def _cell(cells: List, key: str) -> str:
            i = cmap.get(key)
            return _text(cells[i]) if i is not None and i < len(cells) else ""

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            horse_name = _cell(cells, "horse_name")
            if not horse_name:
                continue

            # Extract brand number from <a> href
            horse_cell = cells[cmap["horse_name"]] if "horse_name" in cmap else None
            horse_no = ""
            if horse_cell:
                link = horse_cell.find("a", href=True)
                if link:
                    hm = re.search(r"HorseNo=([^&\"']+)", link["href"], re.I)
                    if hm:
                        horse_no = hm.group(1).strip()
            if not horse_no:
                horse_no = _cell(cells, "horse_no")

            # Weight: card shows declared weight in pounds
            wt_str = _cell(cells, "weight")
            try:
                wt_kg = float(wt_str) * 0.453592
            except ValueError:
                wt_kg = np.nan

            # Draw
            draw_str = _cell(cells, "draw")
            try:
                draw = int(draw_str)
            except ValueError:
                draw = np.nan

            # Win odds (may show "---" before betting opens)
            odds_str = _cell(cells, "win_odds").replace(",", "")
            try:
                win_odds = float(odds_str)
            except ValueError:
                win_odds = np.nan

            runners.append({
                "horse_name":       horse_name,
                "horse_brand_no":   horse_no,
                "jockey":           _cell(cells, "jockey"),
                "trainer":          _cell(cells, "trainer"),
                "draw":             draw,
                "actual_weight_kg": wt_kg,
                "win_odds":         win_odds,
                **meta,
            })
        break  # use the first matching table

    return meta, runners


def scrape_full_day(
    session: Any,
    race_date: date,
    course: str,
    specific_race: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Scrape all races on *race_date* at *course*.

    Parameters
    ----------
    specific_race: If set, only scrape that race number.

    Returns
    -------
    (all_meta, all_runners) — flat lists across races.
    """
    all_meta:    List[Dict] = []
    all_runners: List[Dict] = []

    race_range = [specific_race] if specific_race else range(1, MAX_RACE_NO + 1)

    for race_no in race_range:
        meta, runners = scrape_race_card(session, race_date, course, race_no)
        if meta is None or not runners:
            if race_no > 1 and not specific_race:
                break  # no more races today
            continue
        logger.info("Race card R%d: %d runners  dist=%sm",
                    race_no, len(runners), meta.get("distance_m"))
        all_meta.append(meta)
        all_runners.extend(runners)

    return all_meta, all_runners


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Rebuild rolling state from history
# ─────────────────────────────────────────────────────────────────────────────

def build_horse_state(
    history_df: pd.DataFrame,
    horse_brand_no: str,
    as_of: pd.Timestamp,
    window: int = HORSE_FORM_WINDOW,
) -> Dict[str, Any]:
    """
    Compute per-horse rolling features using only races strictly before *as_of*.

    Parameters
    ----------
    history_df:    Full historical features DataFrame.
    horse_brand_no: The horse's HKJC brand number.
    as_of:         Race date to predict (exclusive upper bound).
    window:        Number of past runs to use.

    Returns
    -------
    Dict of feature name → value.
    """
    mask = (
        (history_df["horse_brand_no"] == horse_brand_no) &
        (history_df["race_date"] < as_of)
    )
    h = history_df[mask].sort_values("race_date")

    if h.empty:
        # Horse with no recorded history
        return {
            f"h_avg_finish_pos_{window}": np.nan,
            f"h_win_rate_{window}":        0.0,
            f"h_place_rate_{window}":      0.0,
            f"h_avg_lbw_{window}":         np.nan,
            f"h_avg_odds_{window}":        np.nan,
            f"h_avg_speed_mps_{window}":   np.nan,
            "h_runs_total":               0,
            "h_last_finish_pos":           np.nan,
            "h_days_since_last_run":       np.nan,
        }

    recent = h.tail(window)

    avg_fp      = recent["finish_pos_num"].mean()
    win_rate    = float(recent["is_winner"].mean()) if "is_winner" in recent else 0.0
    place_rate  = float((recent["finish_pos_num"] <= 3).mean())
    avg_lbw     = recent["lbw"].mean() if "lbw" in recent else np.nan
    avg_odds    = recent["win_odds"].mean() if "win_odds" in recent else np.nan

    # Speed
    if "finish_time_sec" in recent.columns and "distance_m" in recent.columns:
        speeds = np.where(
            recent["finish_time_sec"].gt(0),
            recent["distance_m"] / recent["finish_time_sec"],
            np.nan,
        )
        avg_speed = float(np.nanmean(speeds)) if not np.all(np.isnan(speeds)) else np.nan
    else:
        avg_speed = np.nan

    last_row = h.iloc[-1]
    last_fp  = float(last_row["finish_pos_num"]) if pd.notna(last_row.get("finish_pos_num")) else np.nan
    days_off = (as_of - last_row["race_date"]).days

    return {
        f"h_avg_finish_pos_{window}": avg_fp,
        f"h_win_rate_{window}":       win_rate,
        f"h_place_rate_{window}":     place_rate,
        f"h_avg_lbw_{window}":        avg_lbw,
        f"h_avg_odds_{window}":       avg_odds,
        f"h_avg_speed_mps_{window}":  avg_speed,
        "h_runs_total":               len(h),
        "h_last_finish_pos":          last_fp,
        "h_days_since_last_run":      float(days_off),
    }


def build_tracker_state(
    history_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> Tuple[RollingStatsTracker, RollingStatsTracker]:
    """
    Replay the full race history up to (but not including) *as_of* to
    produce populated jockey and trainer trackers.

    This is the same logic as ``add_jockey_trainer_features()`` in
    Notebook 2, but we stop at the prediction date.

    Returns
    -------
    (jockey_tracker, trainer_tracker)
    """
    jk_tracker = RollingStatsTracker(window_days=ROLLING_JOCKEY_DAYS)
    tr_tracker = RollingStatsTracker(window_days=ROLLING_TRAINER_DAYS)

    past = history_df[history_df["race_date"] < as_of].sort_values("race_date")

    for _, row in past.iterrows():
        rd   = row["race_date"]
        dist = int(row["distance_m"]) if pd.notna(row.get("distance_m")) else 0
        crs  = str(row["racecourse"]) if pd.notna(row.get("racecourse")) else ""
        jk   = str(row["jockey"])    if pd.notna(row.get("jockey"))    else "__UNK__"
        tr   = str(row["trainer"])   if pd.notna(row.get("trainer"))   else "__UNK__"
        won  = bool(row.get("is_winner", 0))
        jk_tracker.update(jk, rd, crs, dist, won)
        tr_tracker.update(tr, rd, crs, dist, won)

    logger.info("Trackers populated from %d historical rows", len(past))
    return jk_tracker, tr_tracker


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Build inference feature rows
# ─────────────────────────────────────────────────────────────────────────────

def _going_dummies(going: Optional[str], all_goings: List[str]) -> Dict[str, int]:
    """One-hot encode going into the same columns as training."""
    going_clean = going if going else "Unknown"
    return {f"going_{g}": int(going_clean == g) for g in all_goings}


def _course_type_dummies(ct: Optional[str], all_cts: List[str]) -> Dict[str, int]:
    """One-hot encode course_type into the same columns as training."""
    ct_clean = ct if ct else "Unknown"
    return {f"ct_{c}": int(ct_clean == c) for c in all_cts}


def build_inference_rows(
    runners: List[Dict],
    history_df: pd.DataFrame,
    horses_df: Optional[pd.DataFrame],
    weather_row: Optional[pd.Series],
    jk_tracker: RollingStatsTracker,
    tr_tracker: RollingStatsTracker,
    feature_cols: List[str],
    te_mapping_jk: Dict[str, float],
    te_mapping_tr: Dict[str, float],
    te_global_mean: float,
    all_goings: List[str],
    all_course_types: List[str],
) -> pd.DataFrame:
    """
    Construct a feature DataFrame for a list of runners in an upcoming race.

    Each row has the same feature columns as the training matrix so it can
    be passed directly to the saved model.

    Parameters
    ----------
    runners:        List of runner dicts from ``scrape_race_card()``.
    history_df:     Full historical features DataFrame (processed).
    horses_df:      Optional horse-profile DataFrame (for static attributes).
    weather_row:    Optional Series with weather columns for the race date.
    jk_tracker:     Jockey rolling-stats tracker populated up to race date.
    tr_tracker:     Trainer rolling-stats tracker populated up to race date.
    feature_cols:   Ordered list of feature column names (from training split).
    te_mapping_*:   Jockey / trainer target-encoding maps from training set.
    te_global_mean: Global training-set win-rate mean (fallback for unseen entities).
    all_goings:     List of going categories seen during training (for OHE).
    all_course_types: List of course_type categories seen during training.

    Returns
    -------
    pd.DataFrame with columns = feature_cols, one row per runner.
    """
    if not runners:
        return pd.DataFrame(columns=feature_cols)

    race_date = pd.Timestamp(runners[0]["race_date"])
    course    = runners[0]["racecourse"]
    dist      = runners[0].get("distance_m") or 0
    field_size = len(runners)

    rows: List[Dict[str, Any]] = []

    for runner in runners:
        horse_no = str(runner.get("horse_brand_no", ""))
        jockey   = str(runner.get("jockey", "__UNK__"))
        trainer  = str(runner.get("trainer", "__UNK__"))
        rec: Dict[str, Any] = {}

        # ── Per-horse rolling features ────────────────────────────────────
        horse_feats = build_horse_state(history_df, horse_no, race_date, HORSE_FORM_WINDOW)
        rec.update(horse_feats)

        # ── Static horse profile ──────────────────────────────────────────
        if horses_df is not None and not horses_df.empty:
            row_h = horses_df[horses_df["horse_brand_no"] == horse_no]
            if not row_h.empty:
                r = row_h.iloc[0]
                rec["age_years"]       = float(r.get("age_years", np.nan) or np.nan)
                rec["current_rating"]  = float(r.get("current_rating", np.nan) or np.nan)
                rec["last_rating"]     = float(r.get("last_rating", np.nan) or np.nan)
                rec["career_starts"]   = float(r.get("career_starts", np.nan) or np.nan)
                rec["career_wins"]     = float(r.get("career_wins", np.nan) or np.nan)
            else:
                for c in ["age_years", "current_rating", "last_rating",
                          "career_starts", "career_wins"]:
                    rec[c] = np.nan
        else:
            for c in ["age_years", "current_rating", "last_rating",
                      "career_starts", "career_wins"]:
                rec[c] = np.nan

        # ── Jockey / trainer rolling stats ────────────────────────────────
        jk_stats = jk_tracker.query(jockey, race_date, course, dist)
        tr_stats = tr_tracker.query(trainer, race_date, course, dist)
        rec["jk_overall_win_rate"] = jk_stats["overall_win_rate"]
        rec["jk_cd_win_rate"]      = jk_stats["cd_win_rate"]
        rec["jk_overall_runs"]     = jk_stats["overall_runs"]
        rec["tr_overall_win_rate"] = tr_stats["overall_win_rate"]
        rec["tr_cd_win_rate"]      = tr_stats["cd_win_rate"]
        rec["tr_overall_runs"]     = tr_stats["overall_runs"]

        # ── Target encoding (jockey / trainer) ────────────────────────────
        rec["jk_te"] = te_mapping_jk.get(jockey, te_global_mean)
        rec["tr_te"] = te_mapping_tr.get(trainer, te_global_mean)

        # ── Derived features ──────────────────────────────────────────────
        rec["is_sprint"]  = int(dist < 1400)
        rec["is_mile"]    = int(1400 <= dist <= 1800)
        rec["is_staying"] = int(dist > 2000)

        draw = runner.get("draw", np.nan)
        rec["draw_norm"] = float(draw) / 14.0 if pd.notna(draw) else 0.5

        age = rec.get("age_years") or np.nan
        wt  = runner.get("actual_weight_kg", np.nan)
        if pd.notna(age) and pd.notna(wt) and age > 0:
            rec["weight_for_age"] = float(wt) / (float(age) * 2 + 50)
        else:
            rec["weight_for_age"] = np.nan

        # Speed / sectional: unknown before the race (fill NaN)
        rec["speed_mps"]        = np.nan
        rec["early_speed_400"]  = np.nan
        rec["finish_speed_400"] = np.nan

        # Implied prob from current market odds
        odds = runner.get("win_odds", np.nan)
        rec["implied_prob"]   = 1.0 / max(float(odds), 1.01) if pd.notna(odds) else np.nan
        rec["race_field_size"] = float(field_size)

        # ── Weather ───────────────────────────────────────────────────────
        if weather_row is not None:
            rec["max_temp_c"]       = float(weather_row.get("max_temp_c",    np.nan))
            rec["min_temp_c"]       = float(weather_row.get("min_temp_c",    np.nan))
            rec["rainfall_mm"]      = float(weather_row.get("rainfall_mm",   0.0))
            rec["avg_humidity_pct"] = float(weather_row.get("avg_humidity_pct", np.nan))
            rec["grass_temp_min_c"] = float(weather_row.get("grass_temp_min_c", np.nan))
        else:
            for wc in ["max_temp_c", "min_temp_c", "rainfall_mm",
                       "avg_humidity_pct", "grass_temp_min_c"]:
                rec[wc] = np.nan

        # ── Going / course_type one-hot ───────────────────────────────────
        going  = runner.get("going")
        ct     = runner.get("course_type")
        rec.update(_going_dummies(going, all_goings))
        rec.update(_course_type_dummies(ct, all_course_types))

        # ── Race class encoding (simple label) ───────────────────────────
        rc_str = str(runner.get("race_class", "Unknown") or "Unknown")
        # Map to a rough integer; same order as LabelEncoder in training
        _rc_map = {
            "Class 5": 0, "Class 4": 1, "Class 3": 2,
            "Class 2": 3, "Class 1": 4, "Griffin": 5,
            "International": 6, "Unknown": 3,
        }
        rec["race_class_enc"] = _rc_map.get(rc_str, 3)

        # ── Raw runner fields ─────────────────────────────────────────────
        rec["actual_weight_kg"] = float(wt) if pd.notna(wt) else np.nan
        rec["draw"]             = float(draw) if pd.notna(draw) else np.nan
        rec["distance_m"]       = float(dist)
        rec["win_odds"]         = float(odds) if pd.notna(odds) else np.nan  # kept for output

        # Store identity info (not in feature matrix, but needed for display)
        rec["_horse_name"]    = runner.get("horse_name", "")
        rec["_horse_brand"]   = horse_no
        rec["_jockey"]        = jockey
        rec["_trainer"]       = trainer
        rec["_draw"]          = draw
        rec["_odds"]          = odds

        rows.append(rec)

    all_df = pd.DataFrame(rows)

    # Align to feature_cols: add missing cols as NaN, drop extras
    for c in feature_cols:
        if c not in all_df.columns:
            all_df[c] = np.nan

    return all_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Load model and predict
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path = DATA_MODELS) -> Tuple[Any, str]:
    """
    Load the best available saved model.

    Tries DEFAULT_MODEL_FILE first, then falls back through
    MODEL_FALLBACK_ORDER.

    Returns
    -------
    (model_object, model_filename)
    """
    candidates = [DEFAULT_MODEL_FILE] + [
        f for f in MODEL_FALLBACK_ORDER if f != DEFAULT_MODEL_FILE
    ]
    for fname in candidates:
        path = model_dir / fname
        if path.exists():
            model = joblib.load(path)
            logger.info("Loaded model: %s", fname)
            return model, fname
    raise FileNotFoundError(
        f"No saved model found in {model_dir}.\n"
        f"Run 03_model_training_evaluation.py first."
    )


def predict(
    model: Any,
    X: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Run inference on a feature DataFrame.

    Handles NaN imputation (median from each column, falling back to 0)
    and column ordering.

    Returns
    -------
    np.ndarray of shape (n_runners,) — calibrated win probabilities.
    """
    X_aligned = X[feature_cols].copy()

    # Impute NaN with column median (same logic should be applied during
    # training preprocessing; here we use a simple fallback)
    for col in X_aligned.columns:
        if X_aligned[col].isna().any():
            median = X_aligned[col].median()
            X_aligned[col] = X_aligned[col].fillna(median if pd.notna(median) else 0.0)

    arr = X_aligned.values.astype(np.float32)
    probs = model.predict_proba(arr)[:, 1]
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Helpers to load supporting data
# ─────────────────────────────────────────────────────────────────────────────

def load_te_mappings(
    train_meta_path: Path,
    features_path: Path,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Reconstruct jockey and trainer target-encoding maps from the training set.

    If the training metadata file exists, uses it.
    Falls back to the full features dataset.

    Returns
    -------
    (te_mapping_jk, te_mapping_tr, global_mean)
    """
    src_path = train_meta_path if train_meta_path.exists() else features_path
    df = pd.read_parquet(src_path)

    # Use only training rows (dates before val_start)
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"])
        latest  = df["race_date"].max()
        val_cut = latest - pd.DateOffset(months=6)
        train   = df[df["race_date"] < val_cut]
    else:
        train = df

    global_mean = float(train["is_winner"].mean()) if "is_winner" in train else 0.09
    k = TE_SMOOTH_K

    def _build_map(col: str) -> Dict[str, float]:
        if col not in train.columns:
            return {}
        stats = train.groupby(col)["is_winner"].agg(["sum", "count"])
        stats["te"] = (stats["sum"] + k * global_mean) / (stats["count"] + k)
        return stats["te"].to_dict()

    return _build_map("jockey"), _build_map("trainer"), global_mean


def load_going_and_ct_categories(features_path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract the going and course_type categories used during training
    (identified by one-hot column prefixes in the feature file).
    """
    df = pd.read_parquet(features_path)
    goings = sorted({c[len("going_"):] for c in df.columns if c.startswith("going_")})
    cts    = sorted({c[len("ct_"):] for c in df.columns if c.startswith("ct_")})
    return goings, cts


def load_weather_for_date(
    weather_path: Path,
    race_date: date,
    course: str,
) -> Optional[pd.Series]:
    """Load a single weather row for the given race date and course."""
    if not weather_path.exists():
        return None
    wdf = pd.read_parquet(weather_path)
    wdf["date"] = pd.to_datetime(wdf.get("date", wdf.get("race_date")))
    mask = (wdf["date"].dt.date == race_date) & (wdf["racecourse"] == course)
    if mask.any():
        return wdf[mask].iloc[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Format and print predictions
# ─────────────────────────────────────────────────────────────────────────────

def print_race_predictions(
    runners:     List[Dict],
    probs:       np.ndarray,
    race_no:     int,
    race_meta:   Optional[Dict],
    bets_only:   bool = False,
) -> pd.DataFrame:
    """
    Print a formatted prediction table for one race.

    Also computes EV and suggested Kelly stake (% of bankroll) for each
    horse where odds are available.

    Parameters
    ----------
    runners:   Runner dicts from scraper.
    probs:     Calibrated win probabilities (same order as runners).
    race_no:   Race number (for display).
    race_meta: Dict with distance_m, going, course_type (for header).
    bets_only: If True, only print rows where EV > BET_EV_THRESHOLD.

    Returns
    -------
    pd.DataFrame — the full ranked table (regardless of bets_only filter).
    """
    records = []
    for runner, p in zip(runners, probs):
        odds = runner.get("_odds", runner.get("win_odds", np.nan))
        ev   = expected_value(p, float(odds)) if pd.notna(odds) and odds > 1 else np.nan
        ks   = kelly_bet_fraction(p, float(odds), KELLY_FRAC, KELLY_MAX) * 100 \
               if pd.notna(odds) and odds > 1 else np.nan

        records.append({
            "Draw":     runner.get("_draw", runner.get("draw", "")),
            "Horse":    runner.get("_horse_name", runner.get("horse_name", "")),
            "Jockey":   runner.get("_jockey", runner.get("jockey", "")),
            "Trainer":  runner.get("_trainer", runner.get("trainer", "")),
            "Odds":     f"{odds:.1f}" if pd.notna(odds) else "---",
            "Win Prob": f"{p * 100:.1f}%",
            "Impl Prob":f"{(1/odds*100):.1f}%" if pd.notna(odds) and odds > 1 else "---",
            "EV":       f"{ev:.3f}" if pd.notna(ev) else "---",
            "Kelly %":  f"{ks:.2f}%" if pd.notna(ks) else "---",
            "Bet?":     "✅ YES" if (pd.notna(ev) and ev > BET_EV_THRESHOLD
                                      and p > BET_MIN_PROB) else "",
            "_prob":    p,
            "_ev":      ev if pd.notna(ev) else -999,
        })

    result_df = (
        pd.DataFrame(records)
        .sort_values("_prob", ascending=False)
        .reset_index(drop=True)
    )
    result_df.index += 1   # 1-based rank

    # ── Print header ──────────────────────────────────────────────────────────
    if race_meta:
        dist  = race_meta.get("distance_m", "?")
        going = race_meta.get("going",       "?")
        ct    = race_meta.get("course_type", "?")
        cls   = race_meta.get("race_class",  "?")
        print(f"\n{'─'*80}")
        print(f"  RACE {race_no:>2}  |  {dist}m  {ct}  |  Going: {going}  |  Class: {cls}")
        print(f"{'─'*80}")

    display_cols = ["Draw", "Horse", "Jockey", "Win Prob", "Impl Prob",
                    "Odds", "EV", "Kelly %", "Bet?"]
    display_df = result_df[display_cols]

    if bets_only:
        display_df = display_df[result_df["_ev"] > BET_EV_THRESHOLD]

    if display_df.empty:
        print("  (No value bets found for this race)")
    else:
        print(display_df.to_string())

    return result_df


def print_day_summary(
    all_predictions: List[pd.DataFrame],
    race_date: date,
    course: str,
) -> None:
    """Print a brief end-of-day summary: value bets across all races."""
    print(f"\n{'='*80}")
    print(f"  VALUE BET SUMMARY  —  {race_date}  {course}")
    print(f"{'='*80}")

    has_bets = False
    for race_no, df in enumerate(all_predictions, 1):
        bets = df[df["Bet?"] == "✅ YES"]
        if not bets.empty:
            has_bets = True
            for _, row in bets.iterrows():
                print(
                    f"  R{race_no}  {row['Horse']:<28}  odds={row['Odds']:<6}"
                    f"  prob={row['Win Prob']:<7}  EV={row['EV']:<7}"
                    f"  stake={row['Kelly %']} of bankroll"
                )

    if not has_bets:
        print("  No value bets identified today.")
        print(f"  (EV threshold = {BET_EV_THRESHOLD:.0%},  min prob = {BET_MIN_PROB:.0%})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    race_date: date,
    course: str,
    specific_race: Optional[int],
    bets_only: bool,
) -> None:
    """
    Full inference pipeline for one race day.
    """
    ensure_dirs(DATA_RESULTS)

    print(f"\n{'='*80}")
    print(f"  HKJC PREDICTION ENGINE")
    print(f"  Date: {race_date}   Course: {course}"
          + (f"   Race: {specific_race}" if specific_race else "   All races"))
    print(f"{'='*80}\n")

    session = build_session()

    # ── Load supporting data ────────────────────────────────────────────────
    print("[1/5] Loading historical data …")
    features_path = DATA_PROC / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            "data/processed/features.parquet not found.\n"
            "Run 02_feature_engineering.py first."
        )
    history_df = pd.read_parquet(features_path)
    history_df["race_date"] = pd.to_datetime(history_df["race_date"])

    horses_path = DATA_RAW / "raw_horses.parquet"
    horses_df   = pd.read_parquet(horses_path) if horses_path.exists() else None
    if horses_df is not None:
        horses_df["horse_brand_no"] = horses_df["horse_brand_no"].astype(str)

    # Feature column list
    fc_path = DATA_SPLITS / "feature_cols.txt"
    if not fc_path.exists():
        raise FileNotFoundError(
            "data/splits/feature_cols.txt not found.\n"
            "Run 02_feature_engineering.py first."
        )
    feature_cols = pd.read_csv(fc_path, header=None)[0].tolist()

    # Target encoding maps
    te_jk, te_tr, te_global = load_te_mappings(
        DATA_SPLITS / "meta_train.parquet", features_path
    )

    # Going / course_type OHE categories
    all_goings, all_cts = load_going_and_ct_categories(features_path)

    # Weather
    weather_row = load_weather_for_date(
        DATA_RAW / "weather.parquet", race_date, course
    )
    if weather_row is None:
        print("  ⚠  No weather data for this date — weather features will be NaN.")

    # ── Build tracker state ─────────────────────────────────────────────────
    print("[2/5] Rebuilding jockey / trainer rolling stats …")
    as_of       = pd.Timestamp(race_date)
    jk_tracker, tr_tracker = build_tracker_state(history_df, as_of)

    # ── Load model ──────────────────────────────────────────────────────────
    print("[3/5] Loading model …")
    model, model_name = load_model()
    print(f"  Using: {model_name}")

    # ── Scrape race card ────────────────────────────────────────────────────
    print(f"[4/5] Scraping race card for {race_date} {course} …")
    all_race_metas, all_runners = scrape_full_day(
        session, race_date, course, specific_race
    )

    if not all_runners:
        print(
            "\n⚠  No runners found for this date/course.\n"
            "   Possible reasons:\n"
            "   • The race card is not yet published (check HKJC website).\n"
            "   • HKJC changed the page structure — inspect resp.text in\n"
            "     scrape_race_card() and update the column regex patterns.\n"
            "   • The page is JavaScript-rendered — swap safe_get() for a\n"
            "     Selenium/Playwright call.\n"
        )
        return

    print(f"  Found {len(all_runners)} declared runners across "
          f"{len(all_race_metas)} race(s)")

    # ── Build features and predict ──────────────────────────────────────────
    print("[5/5] Building features and predicting …\n")

    # Group runners by race_no
    from itertools import groupby
    sorted_runners = sorted(all_runners, key=lambda r: r.get("race_no", 0))
    all_predictions: List[pd.DataFrame] = []

    for rno, group in groupby(sorted_runners, key=lambda r: r.get("race_no", 0)):
        race_runners = list(group)
        race_meta    = next(
            (m for m in all_race_metas if m.get("race_no") == rno), None
        )

        # Build feature matrix
        X_df = build_inference_rows(
            runners=race_runners,
            history_df=history_df,
            horses_df=horses_df,
            weather_row=weather_row,
            jk_tracker=jk_tracker,
            tr_tracker=tr_tracker,
            feature_cols=feature_cols,
            te_mapping_jk=te_jk,
            te_mapping_tr=te_tr,
            te_global_mean=te_global,
            all_goings=all_goings,
            all_course_types=all_cts,
        )

        # Run model
        probs = predict(model, X_df, feature_cols)

        # Display results
        pred_table = print_race_predictions(
            race_runners, probs, rno, race_meta, bets_only
        )
        all_predictions.append(pred_table)

        # Save per-race CSV
        out_path = (
            DATA_RESULTS
            / f"predictions_{race_date.isoformat()}_{course}_R{rno}.csv"
        )
        pred_table.drop(columns=["_prob", "_ev"], errors="ignore").to_csv(
            out_path, index=False
        )

    # ── End-of-day value-bet summary ─────────────────────────────────────────
    print_day_summary(all_predictions, race_date, course)

    print(f"✅  Done. CSV predictions saved to {DATA_RESULTS}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict upcoming HKJC race card.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", "-d",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="Race date, e.g. 2025-06-15  (default: today)",
    )
    parser.add_argument(
        "--course", "-c",
        choices=["ST", "HV"],
        default="ST",
        help="Racecourse: ST (Sha Tin) or HV (Happy Valley)  (default: ST)",
    )
    parser.add_argument(
        "--race", "-r",
        type=int,
        default=None,
        help="Specific race number to predict (default: all races on the day)",
    )
    parser.add_argument(
        "--bets-only",
        action="store_true",
        default=False,
        help="Only print horses with positive expected value",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=f"Model filename to load from data/models/ "
             f"(default: {DEFAULT_MODEL_FILE})",
    )

    args = parser.parse_args()

    if args.model:
        DEFAULT_MODEL_FILE = args.model

    main(
        race_date=args.date,
        course=args.course,
        specific_race=args.race,
        bets_only=args.bets_only,
    )