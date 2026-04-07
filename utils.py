"""
utils.py
========
Shared utilities for the HKJC Horse Racing Prediction project.

Covers:
  • HTTP session with retries and back-off
  • Distance-beaten string → float conversion
  • Scratch/DNF detection
  • Checkpoint persistence (resume interrupted scrapes)
  • Fractional-Kelly bet-sizing and expected-value helpers
  • Incremental jockey / trainer rolling-statistics tracker
  • Ranking metrics: MAP@k, NDCG@k
  • Betting backtest engine
  • Parquet / directory helpers
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hkjc")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────────────────────

def build_session(
    retries: int = 3,
    backoff_factor: float = 1.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Create a ``requests.Session`` with automatic retries and exponential
    back-off.

    Parameters
    ----------
    retries:
        Maximum number of retry attempts per request.
    backoff_factor:
        Factor applied between retries: wait = backoff_factor × 2^(n-1).
    status_forcelist:
        HTTP status codes that trigger a retry.

    Returns
    -------
    requests.Session
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    )
    return session


def safe_get(
    session: requests.Session,
    url: str,
    sleep_secs: float = 1.0,
    timeout: int = 25,
    **kwargs: Any,
) -> Optional[requests.Response]:
    """
    Perform a polite GET request.

    Sleeps for ``sleep_secs`` before sending, so consecutive calls respect
    the server.  Returns ``None`` on any failure or non-200 status.
    """
    time.sleep(sleep_secs)
    try:
        resp = session.get(url, timeout=timeout, **kwargs)
        if resp.status_code == 200:
            return resp
        logger.debug("HTTP %s for URL: %s", resp.status_code, url)
        return None
    except requests.RequestException as exc:
        logger.warning("Request failed — %s  URL: %s", exc, url)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Distance-beaten conversion
# ─────────────────────────────────────────────────────────────────────────────

#: Lookup table for textual beaten-distance tokens used on HKJC pages.
_DIST_MAP: Dict[str, float] = {
    "SH": 0.05,  "SHORT HEAD": 0.05,
    "NOSE": 0.05, "NSE": 0.05,
    "HD": 0.10,  "HEAD": 0.10,
    "NK": 0.25,  "NECK": 0.25,
    "1/4": 0.25,
    "1/2": 0.50,
    "3/4": 0.75,
    "1": 1.00,
    "1-1/4": 1.25, "11/4": 1.25,
    "1-1/2": 1.50, "11/2": 1.50,
    "1-3/4": 1.75, "13/4": 1.75,
    "2": 2.00,
    "2-1/4": 2.25, "21/4": 2.25,
    "2-1/2": 2.50, "21/2": 2.50,
    "2-3/4": 2.75, "23/4": 2.75,
    "3": 3.00,
    "3-1/2": 3.50, "31/2": 3.50,
    "4": 4.00,
    "5": 5.00,
    "6": 6.00,
    "7": 7.00,
    "8": 8.00,
    "9": 9.00,
    "10": 10.00,
    "---": np.nan,   # winner's own row
    "DH":  0.0,      # dead heat
    "WO":  np.nan,   # walkover
    "":    np.nan,
}

#: Non-finish codes that appear in the Place column.
SCRATCH_CODES: frozenset[str] = frozenset(
    {"WX", "WV", "WV-A", "DNF", "DISQ", "FELL", "UR", "PU", "SCR", "WD", "---", "RO"}
)


def convert_distance_beaten(raw: Any) -> float:
    """
    Convert a raw HKJC beaten-distance token to a float (in horse-lengths).

    Handles textual shortcuts (SH, NOSE, HD, NK), plain fractions (1/2),
    compound fractions (1-3/4), plain numbers ("5"), and special codes.

    Parameters
    ----------
    raw:
        The raw cell value from the scraper.

    Returns
    -------
    float — NaN for winner rows or un-parseable values.

    Examples
    --------
    >>> convert_distance_beaten("SH")
    0.05
    >>> convert_distance_beaten("1-3/4")
    1.75
    >>> convert_distance_beaten("---")
    nan
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    s = (
        str(raw)
        .strip()
        .upper()
        .replace("\xa0", "")
        .replace("\u00a0", "")
        .replace(" ", "")
    )
    if s in _DIST_MAP:
        return _DIST_MAP[s]
    # Direct numeric
    try:
        return float(s)
    except ValueError:
        pass
    # Compound fraction  N-M/D  (e.g. "3-1/4")
    m = re.match(r"^(\d+)[–\-](\d+)/(\d+)$", s)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return whole + num / den
    # Plain fraction  M/D  (e.g. "1/2")
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    logger.debug("Unparseable distance_beaten value: %r", raw)
    return np.nan


def is_valid_finisher(value: Any) -> bool:
    """
    Return ``True`` if the runner finished the race (not scratched/DNF/etc.).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() not in SCRATCH_CODES


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint manager
# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    """
    Persist scraping progress in a JSON file so interrupted runs can resume.

    Usage
    -----
    >>> ckpt = Checkpoint("data/raw/scraping_checkpoint.json")
    >>> ckpt.is_done("2024-09-01_ST_1")
    False
    >>> ckpt.mark_done("2024-09-01_ST_1")
    >>> ckpt.is_done("2024-09-01_ST_1")
    True
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = self._load()

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"done": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Persist a key-value pair."""
        self._data[key] = value
        self._save()

    def mark_done(self, item: str) -> None:
        """Mark an arbitrary string item as completed."""
        done: List[str] = self._data.setdefault("done", [])
        if item not in done:
            done.append(item)
            self._save()

    def is_done(self, item: str) -> bool:
        """Check whether an item has been marked done."""
        return item in self._data.get("done", [])


# ─────────────────────────────────────────────────────────────────────────────
# Bet-sizing helpers
# ─────────────────────────────────────────────────────────────────────────────

def expected_value(probability: float, decimal_odds: float) -> float:
    """
    Compute the expected value of a win bet.

    EV = p × (b – 1) – (1 – p),  where b = decimal_odds.

    A positive EV means the bet has a long-run mathematical edge.
    """
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def kelly_bet_fraction(
    probability: float,
    decimal_odds: float,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """
    Compute the fractional-Kelly stake as a proportion of current bankroll.

    Full Kelly: f* = (b·p − q) / b   where b = decimal_odds − 1.
    Fractional Kelly: stake_fraction = f* × kelly_fraction.

    Parameters
    ----------
    probability:
        Calibrated win probability (0–1).
    decimal_odds:
        Decimal odds (e.g. 4.5 means a 1-unit bet returns 4.5 units gross).
    kelly_fraction:
        Scaling factor for the full Kelly (default 0.25 = quarter-Kelly).
    max_fraction:
        Hard cap on the fraction of bankroll per single bet (default 5 %).

    Returns
    -------
    float — fraction of bankroll to stake, clipped to [0, max_fraction].
    """
    b = decimal_odds - 1.0
    if b <= 0.0 or probability <= 0.0:
        return 0.0
    q = 1.0 - probability
    full_kelly = (b * probability - q) / b
    if full_kelly <= 0.0:
        return 0.0
    return float(min(full_kelly * kelly_fraction, max_fraction))


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-statistics tracker  (jockey / trainer)
# ─────────────────────────────────────────────────────────────────────────────

class RollingStatsTracker:
    """
    Efficiently compute rolling win-rate statistics for an entity
    (jockey or trainer) by processing races in strict chronological order.

    No look-ahead: ``query()`` returns stats from races *strictly before*
    the query date; ``update()`` records the outcome *after* querying.

    Parameters
    ----------
    window_days:
        Rolling window width in calendar days (default 30).

    Usage
    -----
    >>> tracker = RollingStatsTracker(window_days=30)
    >>> for _, row in df.sort_values("race_date").iterrows():
    ...     stats = tracker.query(row.jockey, row.race_date, row.racecourse, row.distance_m)
    ...     # attach stats to feature dict …
    ...     tracker.update(row.jockey, row.race_date, row.racecourse,
    ...                    row.distance_m, bool(row.is_winner))
    """

    def __init__(self, window_days: int = 30) -> None:
        self.window_days = window_days
        # entity → [(date, course, distance_m, won), …]
        self._history: Dict[str, List[Tuple[pd.Timestamp, str, int, bool]]] = {}

    # ── private ──────────────────────────────────────────────────────────────

    def _prune(self, entity: str, ref_date: pd.Timestamp) -> None:
        """Drop records older than the rolling window."""
        cutoff = ref_date - pd.Timedelta(days=self.window_days)
        if entity in self._history:
            self._history[entity] = [
                r for r in self._history[entity] if r[0] >= cutoff
            ]

    # ── public API ───────────────────────────────────────────────────────────

    def query(
        self,
        entity: str,
        race_date: pd.Timestamp,
        course: Optional[str] = None,
        distance: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Return rolling stats for *entity* using only races before *race_date*.

        Returned keys: ``overall_runs``, ``overall_wins``,
        ``overall_win_rate``, ``cd_runs``, ``cd_wins``, ``cd_win_rate``
        (where *cd* = same course + distance combination).
        """
        self._prune(entity, race_date)
        hist = [r for r in self._history.get(entity, []) if r[0] < race_date]

        overall_runs = len(hist)
        overall_wins = int(sum(r[3] for r in hist))

        cd = hist
        if course is not None:
            cd = [r for r in cd if r[1] == course]
        if distance is not None:
            cd = [r for r in cd if r[2] == distance]
        cd_runs = len(cd)
        cd_wins = int(sum(r[3] for r in cd))

        return {
            "overall_runs":     overall_runs,
            "overall_wins":     overall_wins,
            "overall_win_rate": overall_wins / overall_runs if overall_runs else 0.0,
            "cd_runs":          cd_runs,
            "cd_wins":          cd_wins,
            "cd_win_rate":      cd_wins  / cd_runs     if cd_runs     else 0.0,
        }

    def update(
        self,
        entity: str,
        race_date: pd.Timestamp,
        course: str,
        distance: int,
        won: bool,
    ) -> None:
        """Record a race outcome for the entity."""
        self._history.setdefault(entity, []).append(
            (race_date, course, distance, won)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics: MAP@k, NDCG@k
# ─────────────────────────────────────────────────────────────────────────────

def _ap_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Average precision at k for a single query (race)."""
    if labels.sum() == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    hits = precision_sum = 0.0
    for rank, idx in enumerate(top_k_idx, 1):
        if labels[idx]:
            hits += 1.0
            precision_sum += hits / rank
    return float(precision_sum / min(labels.sum(), k))


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Normalised Discounted Cumulative Gain at k for a single race."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    dcg  = sum(labels[i] / np.log2(r + 1) for r, i in enumerate(top_k_idx, 1))
    ideal = np.sort(labels)[::-1][:k]
    idcg  = sum(v    / np.log2(r + 1) for r, v in enumerate(ideal, 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_ranking_metrics(
    df: pd.DataFrame,
    race_id_col: str,
    label_col: str,
    score_col: str,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Compute mean AP@k and mean NDCG@k across all races in *df*.

    Parameters
    ----------
    df:
        Prediction DataFrame (one row per horse per race).
    race_id_col:
        Column that identifies the race (grouping key).
    label_col:
        Binary target column (1 = winner).
    score_col:
        Predicted probability column.
    k:
        Cut-off rank.

    Returns
    -------
    (mean_ap_at_k, mean_ndcg_at_k)
    """
    maps, ndcgs = [], []
    for _, grp in df.groupby(race_id_col):
        lbl = grp[label_col].to_numpy(dtype=float)
        scr = grp[score_col].to_numpy(dtype=float)
        if lbl.sum() == 0:
            continue
        maps.append(_ap_at_k(lbl, scr, k))
        ndcgs.append(_ndcg_at_k(lbl, scr, k))
    return (
        float(np.mean(maps))  if maps  else 0.0,
        float(np.mean(ndcgs)) if ndcgs else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Betting backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_betting_backtest(
    predictions_df: pd.DataFrame,
    *,
    prob_col: str = "win_prob",
    odds_col: str = "win_odds",
    label_col: str = "is_winner",
    race_id_col: str = "race_id",
    horse_col: str = "horse_name",
    ev_threshold: float = 0.05,
    min_prob: float = 0.15,
    top_n_in_race: int = 2,
    kelly_frac: float = 0.25,
    kelly_max: float = 0.05,
    starting_bankroll: float = 1_000.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate a fractional-Kelly betting strategy on the test set.

    Betting rules (all must hold):
    1. EV  = p × (odds − 1) − (1 − p)  >  ev_threshold
    2. p   > min_prob
    3. Horse ranks within the top *top_n_in_race* predictions in its race.

    Stake per bet:
        stake = kelly_bet_fraction(p, odds, kelly_frac, kelly_max) × bankroll

    Parameters
    ----------
    predictions_df:
        Must contain *prob_col*, *odds_col*, *label_col*, *race_id_col*.
    ev_threshold:
        Minimum EV to place a bet (default 0.05 = 5 % edge).
    min_prob:
        Minimum predicted probability to consider a bet (default 0.15).
    top_n_in_race:
        Only bet on the N highest-probability horses in each race.
    kelly_frac:
        Fraction of full Kelly (default 0.25 = quarter-Kelly).
    kelly_max:
        Maximum stake as fraction of bankroll (default 5 %).
    starting_bankroll:
        Initial bankroll in arbitrary units (default 1 000).

    Returns
    -------
    (transaction_log, metrics_dict)

    *transaction_log* — one row per bet placed.
    *metrics_dict* — summary statistics.
    """
    df = predictions_df.copy().reset_index(drop=True)

    # Rank horses within each race by predicted probability (descending)
    df["_rank_in_race"] = (
        df.groupby(race_id_col)[prob_col]
          .rank(ascending=False, method="first")
    )

    bankroll = starting_bankroll
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        p    = float(row[prob_col])
        odds = float(row[odds_col]) if not pd.isna(row[odds_col]) else np.nan
        if np.isnan(odds) or odds <= 1.0 or np.isnan(p):
            continue

        ev = expected_value(p, odds)

        if (
            ev          > ev_threshold
            and p       > min_prob
            and row["_rank_in_race"] <= top_n_in_race
        ):
            stake_frac = kelly_bet_fraction(p, odds, kelly_frac, kelly_max)
            stake      = stake_frac * bankroll
            won        = int(row[label_col]) == 1
            profit     = stake * (odds - 1.0) if won else -stake
            bankroll  += profit

            records.append(
                {
                    race_id_col: row[race_id_col],
                    "horse_name": row.get(horse_col, ""),
                    "win_prob":   p,
                    "odds":       odds,
                    "ev":         ev,
                    "stake":      stake,
                    "won":        won,
                    "profit":     profit,
                    "bankroll":   bankroll,
                }
            )

    if not records:
        return pd.DataFrame(), {
            "total_bets":    0,
            "bet_win_rate":  0.0,
            "total_staked":  0.0,
            "net_profit":    0.0,
            "roi_pct":       0.0,
            "final_bankroll": starting_bankroll,
            "sharpe":        0.0,
        }

    log            = pd.DataFrame(records)
    total_bets     = len(log)
    total_staked   = float(log["stake"].sum())
    net_profit     = float(log["profit"].sum())
    bet_win_rate   = float(log["won"].mean())
    roi_pct        = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0

    # Per-race returns → Sharpe ratio
    race_pnl  = log.groupby(race_id_col)["profit"].sum()
    sharpe    = (
        float(race_pnl.mean() / race_pnl.std())
        if len(race_pnl) > 1 and race_pnl.std() > 0
        else 0.0
    )

    metrics = {
        "total_bets":     total_bets,
        "bet_win_rate":   bet_win_rate,
        "total_staked":   total_staked,
        "net_profit":     net_profit,
        "roi_pct":        roi_pct,
        "final_bankroll": float(bankroll),
        "sharpe":         sharpe,
    }
    return log, metrics


# ─────────────────────────────────────────────────────────────────────────────
# File / directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str | Path) -> None:
    """Create all listed directories (including parents) if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def safe_read_parquet(path: str | Path) -> Optional[pd.DataFrame]:
    """Read a Parquet file and return a DataFrame, or ``None`` if not found."""
    p = Path(path)
    if not p.exists():
        logger.warning("Parquet file not found: %s", p)
        return None
    return pd.read_parquet(p)