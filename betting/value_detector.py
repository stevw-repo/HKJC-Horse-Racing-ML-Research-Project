# betting/value_detector.py
"""
Identify positive-expected-value bets across WIN, PLA, QIN, QPL.
"""
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd

from config import MIN_EDGE, VALID_POOL_TYPES


def find_value_bets(
    model_win_probs:   pd.Series,   # index = horse_id or horse_no
    model_place_probs: pd.Series,
    odds_df:           pd.DataFrame,
    field_size:        int,
    min_edge:          float = MIN_EDGE,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      pool_type, comb_string, model_prob, market_prob, edge,
      recommended_stake_fraction
    """
    records = []

    # ── WIN ───────────────────────────────────────────────────────────────────
    win_odds = odds_df[odds_df["pool_type"] == "WIN"].copy()
    for _, row in win_odds.iterrows():
        cs    = row["comb_string"]
        horse = _lookup(cs, model_win_probs)
        if horse is None:
            continue
        model_p  = float(model_win_probs.get(horse, np.nan))
        odds_val = float(row["odds_value"])
        if pd.isna(model_p) or odds_val <= 0:
            continue
        market_p = 1.0 / odds_val
        edge     = model_p - market_p
        if edge >= min_edge:
            records.append({
                "pool_type":   "WIN",
                "comb_string": cs,
                "model_prob":  round(model_p, 4),
                "market_prob": round(market_p, 4),
                "edge":        round(edge, 4),
                "decimal_odds":odds_val,
            })

    # ── PLA ───────────────────────────────────────────────────────────────────
    pla_odds = odds_df[odds_df["pool_type"] == "PLA"].copy()
    for _, row in pla_odds.iterrows():
        cs    = row["comb_string"]
        horse = _lookup(cs, model_place_probs)
        if horse is None:
            continue
        model_p  = float(model_place_probs.get(horse, np.nan))
        odds_val = float(row["odds_value"])
        if pd.isna(model_p) or odds_val <= 0:
            continue
        market_p = 1.0 / odds_val
        edge     = model_p - market_p
        if edge >= min_edge:
            records.append({
                "pool_type":   "PLA",
                "comb_string": cs,
                "model_prob":  round(model_p, 4),
                "market_prob": round(market_p, 4),
                "edge":        round(edge, 4),
                "decimal_odds":odds_val,
            })

    # ── QIN ───────────────────────────────────────────────────────────────────
    qin_odds = odds_df[odds_df["pool_type"] == "QIN"].copy()
    for _, row in qin_odds.iterrows():
        cs   = row["comb_string"]
        pair = _parse_pair(cs)
        if pair is None:
            continue
        h1, h2   = pair
        pw1      = float(model_win_probs.get(h1, np.nan))
        pw2      = float(model_win_probs.get(h2, np.nan))
        pp1      = float(model_place_probs.get(h1, np.nan))
        pp2      = float(model_place_probs.get(h2, np.nan))
        if any(pd.isna(v) for v in [pw1, pw2, pp1, pp2]):
            continue
        model_p  = _qin_prob(pw1, pw2, pp1, pp2)
        odds_val = float(row["odds_value"])
        if odds_val <= 0:
            continue
        market_p = 1.0 / odds_val
        edge     = model_p - market_p
        if edge >= min_edge:
            records.append({
                "pool_type":   "QIN",
                "comb_string": cs,
                "model_prob":  round(model_p, 4),
                "market_prob": round(market_p, 4),
                "edge":        round(edge, 4),
                "decimal_odds":odds_val,
            })

    # ── QPL ───────────────────────────────────────────────────────────────────
    qpl_odds = odds_df[odds_df["pool_type"] == "QPL"].copy()
    for _, row in qpl_odds.iterrows():
        cs   = row["comb_string"]
        pair = _parse_pair(cs)
        if pair is None:
            continue
        h1, h2   = pair
        pp1      = float(model_place_probs.get(h1, np.nan))
        pp2      = float(model_place_probs.get(h2, np.nan))
        if any(pd.isna(v) for v in [pp1, pp2]):
            continue
        model_p  = _qpl_prob(pp1, pp2)
        odds_val = float(row["odds_value"])
        if odds_val <= 0:
            continue
        market_p = 1.0 / odds_val
        edge     = model_p - market_p
        if edge >= min_edge:
            records.append({
                "pool_type":   "QPL",
                "comb_string": cs,
                "model_prob":  round(model_p, 4),
                "market_prob": round(market_p, 4),
                "edge":        round(edge, 4),
                "decimal_odds":odds_val,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


# ── Pair probability approximations ──────────────────────────────────────────

def _qin_prob(pw1: float, pw2: float, pp1: float, pp2: float) -> float:
    """P(i and j are 1st and 2nd in any order)."""
    return pw1 * max(pp2 - pw2, 0) + pw2 * max(pp1 - pw1, 0)


def _qpl_prob(pp1: float, pp2: float,
              correction: float = 0.85) -> float:
    """P(both i and j finish top-3).  correction accounts for dependence."""
    return min(pp1 * pp2 * correction, 1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lookup(comb_string: str, probs: pd.Series):
    cs = str(comb_string).strip().lstrip("0")
    for key in probs.index:
        if str(key).strip().lstrip("0") == cs:
            return key
    return None


def _parse_pair(comb_string: str):
    parts = [p.strip() for p in str(comb_string).split(",")]
    if len(parts) != 2:
        return None
    return parts[0].lstrip("0") or "0", parts[1].lstrip("0") or "0"