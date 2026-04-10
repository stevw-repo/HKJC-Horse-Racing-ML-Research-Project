# betting/backtest.py
"""
Walk-forward P&L backtest.
Only races where race_year >= DIVIDENDS_CUTOFF_YEAR are included in P&L.
"""
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import (DIVIDENDS_CUTOFF_YEAR, KELLY_FRACTION, MAX_BET_FRACTION,
                    MIN_EDGE, STARTING_BANKROLL, VALID_POOL_TYPES)
from betting.bet_types import payout, calculate_roi
from betting.kelly import kelly_stake_hkd
from betting.value_detector import find_value_bets
from models.base_model import BaseModel
from storage.parquet_store import read

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades_df:            pd.DataFrame
    daily_pnl:            pd.Series
    cumulative_bankroll:  pd.Series
    total_return_pct:     float
    roi_by_pool:          pd.Series
    roi_by_class:         pd.Series
    roi_by_venue:         pd.Series
    sharpe_ratio:         float
    max_drawdown_pct:     float
    win_rate:             float
    avg_odds_bet:         float
    bets_per_meeting:     float

    def summary(self) -> str:
        lines = [
            f"=== Backtest Summary ===",
            f"Total return:   {self.total_return_pct:.2f}%",
            f"Sharpe ratio:   {self.sharpe_ratio:.3f}",
            f"Max drawdown:   {self.max_drawdown_pct:.2f}%",
            f"Win rate:       {self.win_rate:.2%}",
            f"Avg odds bet:   {self.avg_odds_bet:.2f}",
            f"Bets/meeting:   {self.bets_per_meeting:.1f}",
            f"\nROI by pool:\n{self.roi_by_pool.to_string()}",
        ]
        return "\n".join(lines)


def run_backtest(
    features_df:       pd.DataFrame,
    win_model:         BaseModel,
    place_model:       BaseModel,
    start_date:        str,
    end_date:          str,
    starting_bankroll: float = STARTING_BANKROLL,
    kelly_fraction:    float = KELLY_FRACTION,
    min_edge:          float = MIN_EDGE,
    pool_types:        List[str] = None,
) -> BacktestResult:
    if pool_types is None:
        pool_types = VALID_POOL_TYPES

    # Only include in-scope pool types
    pool_types = [p for p in pool_types if p in VALID_POOL_TYPES]

    features_df = features_df.copy()
    features_df["race_date"] = pd.to_datetime(features_df["race_date"])
    features_df = features_df[
        (features_df["race_date"] >= start_date) &
        (features_df["race_date"] <= end_date)
    ]

    # Exclude races before dividend cutoff from P&L
    features_df = features_df[
        features_df["race_date"].dt.year >= DIVIDENDS_CUTOFF_YEAR
    ]

    exclude_cols = {"target_win", "target_place", "race_date", "horse_id",
                    "race_id", "is_debutant", "placing_code",
                    "dividend_win", "dividend_place", "placing"}
    feature_cols = [c for c in features_df.columns if c not in exclude_cols
                    and features_df[c].dtype in (float, int, np.float32,
                                                  np.float64, np.int32, np.int64)]

    bankroll   = starting_bankroll
    trade_rows = []
    daily_pnl  = {}

    meetings = sorted(features_df["race_date"].dt.date.unique())

    for meeting_date in meetings:
        day_df = features_df[
            features_df["race_date"].dt.date == meeting_date
        ].copy()

        # Predictions
        X_feats = day_df[feature_cols].fillna(0).values
        try:
            win_probs   = win_model.predict_proba(X_feats)
            place_probs = place_model.predict_proba(X_feats)
        except Exception as exc:
            logger.warning("Predict failed %s: %s", meeting_date, exc)
            continue

        day_df["pred_win"]   = win_probs
        day_df["pred_place"] = place_probs

        # Load odds for this date
        from config import RAW_ODDS_DIR
        tag      = str(meeting_date).replace("-", "")
        odds_fp  = RAW_ODDS_DIR / f"odds_{tag}.parquet"
        odds_df  = read(odds_fp) if odds_fp.exists() else pd.DataFrame()

        day_pnl = 0.0
        for race_no in day_df["race_no"].unique():
            race = day_df[day_df["race_no"] == race_no]
            venue = race["venue"].iloc[0]

            if odds_df.empty:
                continue
            race_odds = odds_df[
                (odds_df["race_no"] == race_no) &
                (odds_df["pool_type"].isin(pool_types))
            ]
            if race_odds.empty:
                continue

            # Build probability series keyed by horse_no
            win_s   = pd.Series(race["pred_win"].values,
                                 index=race["horse_no"].astype(str).str.lstrip("0"))
            place_s = pd.Series(race["pred_place"].values,
                                 index=race["horse_no"].astype(str).str.lstrip("0"))

            bets = find_value_bets(
                model_win_probs=win_s,
                model_place_probs=place_s,
                odds_df=race_odds,
                field_size=len(race),
                min_edge=min_edge,
            )

            for _, bet in bets.iterrows():
                stake = kelly_stake_hkd(bet["model_prob"],
                                         bet["decimal_odds"],
                                         bankroll, kelly_fraction)
                if stake < 1.0:
                    continue

                # Look up actual dividend
                div = _lookup_dividend(
                    features_df=day_df,
                    race_no=race_no,
                    pool_type=bet["pool_type"],
                    comb_string=bet["comb_string"],
                    race_date=str(meeting_date),
                    venue=venue,
                )

                gross  = payout(bet["pool_type"], div, stake) if div else 0.0
                net_pl = gross - stake

                bankroll  += net_pl
                day_pnl   += net_pl

                trade_rows.append({
                    "date":        str(meeting_date),
                    "venue":       venue,
                    "race_no":     race_no,
                    "pool_type":   bet["pool_type"],
                    "comb_string": bet["comb_string"],
                    "model_prob":  bet["model_prob"],
                    "market_prob": bet["market_prob"],
                    "edge":        bet["edge"],
                    "decimal_odds":bet["decimal_odds"],
                    "stake":       round(stake, 2),
                    "dividend":    div,
                    "gross":       round(gross, 2),
                    "net_pnl":     round(net_pl, 2),
                    "bankroll":    round(bankroll, 2),
                    "race_class":  race["race_class"].iloc[0] if "race_class" in race else None,
                })

        daily_pnl[str(meeting_date)] = day_pnl

    trades_df = pd.DataFrame(trade_rows)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    daily_series = pd.Series(daily_pnl)
    cum_bankroll = starting_bankroll + daily_series.cumsum()
    total_return = (bankroll - starting_bankroll) / starting_bankroll * 100

    if not trades_df.empty:
        roi_by_pool  = (
            trades_df.groupby("pool_type")
            .apply(lambda g: calculate_roi(g["gross"].sum(), g["stake"].sum()))
        )
        roi_by_class = (
            trades_df.groupby("race_class")
            .apply(lambda g: calculate_roi(g["gross"].sum(), g["stake"].sum()))
        )
        roi_by_venue = (
            trades_df.groupby("venue")
            .apply(lambda g: calculate_roi(g["gross"].sum(), g["stake"].sum()))
        )
        win_rate     = (trades_df["net_pnl"] > 0).mean()
        avg_odds     = trades_df["decimal_odds"].mean()
        bets_per_mtg = len(trades_df) / max(len(meetings), 1)
    else:
        roi_by_pool = roi_by_class = roi_by_venue = pd.Series(dtype=float)
        win_rate    = avg_odds = bets_per_mtg = 0.0

    sharpe = _sharpe(daily_series)
    mdd    = _max_drawdown(cum_bankroll)

    return BacktestResult(
        trades_df=trades_df,
        daily_pnl=daily_series,
        cumulative_bankroll=cum_bankroll,
        total_return_pct=total_return,
        roi_by_pool=roi_by_pool,
        roi_by_class=roi_by_class,
        roi_by_venue=roi_by_venue,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd,
        win_rate=win_rate,
        avg_odds_bet=avg_odds,
        bets_per_meeting=bets_per_mtg,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lookup_dividend(features_df, race_no, pool_type, comb_string,
                     race_date, venue) -> float | None:
    """Pull the stored dividend from the features table or dividends store."""
    from config import RAW_DIVIDENDS_DIR
    year  = int(race_date[:4])
    div_fp = RAW_DIVIDENDS_DIR / f"dividends_{year}.parquet"
    if not div_fp.exists():
        return None
    div_df = read(div_fp)
    if div_df.empty:
        return None
    mask = (
        (div_df["race_date"] == race_date) &
        (div_df["venue"]     == venue) &
        (div_df["race_no"]   == race_no) &
        (div_df["pool_type"] == pool_type) &
        (div_df["winning_combination"].astype(str).str.replace(" ", "")
         == str(comb_string).replace(" ", ""))
    )
    rows = div_df[mask]
    if rows.empty:
        return None
    return rows["dividend_hkd"].iloc[0]


def _sharpe(daily_pnl: pd.Series, periods_per_year: int = 52) -> float:
    if daily_pnl.empty or daily_pnl.std() == 0:
        return 0.0
    return (daily_pnl.mean() / daily_pnl.std()) * (periods_per_year ** 0.5)


def _max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return 0.0
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max.replace(0, np.nan) * 100
    return float(drawdown.min())