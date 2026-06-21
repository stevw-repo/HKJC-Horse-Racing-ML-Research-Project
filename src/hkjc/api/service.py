"""Data-access layer for the API (M6): read the DuckDB views + persisted snapshots.

Pure reads -- no training or scraping happens in a request. Backtest/leaderboard/staking come
from the ``processed/`` snapshots that ``run_backtest`` / ``run_leaderboard`` / ``run_sweep``
persist; health/races come live from DuckDB; the race-day card is a mock until M7.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from hkjc.api.schemas import (
    BacktestResponse,
    LeaderboardRow,
    RaceDayRace,
    RaceDayResponse,
    RaceDayRunner,
    RaceSummary,
    StakingRow,
)
from hkjc.common.config import AppConfig, get_config
from hkjc.data import pipeline

_RACES_SQL = """
SELECT r.race_date, r.venue, r.race_no, r.distance_m, r.going, COALESCE(c.field_size, 0)
FROM races r
LEFT JOIN (
    SELECT race_date, venue, race_no, count(*) AS field_size FROM results GROUP BY 1, 2, 3
) c ON c.race_date = r.race_date AND c.venue = r.venue AND c.race_no = r.race_no
ORDER BY r.race_date DESC, r.race_no
LIMIT ?
"""


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def health(cfg: AppConfig | None = None) -> dict[str, Any]:
    """Stored-data coverage + manifest size (the data/scraper-health dashboard)."""
    return pipeline.coverage_summary(cfg or get_config())


def backtest(cfg: AppConfig | None = None) -> BacktestResponse | None:
    """The honest walk-forward backtest snapshot, or None if not generated yet."""
    cfg = cfg or get_config()
    data = _read_json(cfg.paths.processed_dir / "backtest" / "result.json")
    return BacktestResponse.model_validate(data) if data is not None else None


def leaderboard(cfg: AppConfig | None = None) -> list[LeaderboardRow]:
    """The model-zoo leaderboard snapshot (empty if `hkjc train` has not run)."""
    cfg = cfg or get_config()
    data = _read_json(cfg.paths.processed_dir / "experiments" / "leaderboard.json")
    return [LeaderboardRow.model_validate(row) for row in data] if data else []


def staking(cfg: AppConfig | None = None) -> list[StakingRow]:
    """The M5 staking sweep grid (empty if `hkjc risk sweep` has not run)."""
    cfg = cfg or get_config()
    path = cfg.paths.processed_dir / "risk" / "staking_sweep.parquet"
    if not path.is_file():
        return []
    return [StakingRow.model_validate(row) for row in pl.read_parquet(path).to_dicts()]


def races(limit: int = 50, cfg: AppConfig | None = None) -> list[RaceSummary]:
    """The most recent stored races (data-health / explorer context)."""
    cfg = cfg or get_config()
    if not cfg.paths.duckdb_path.is_file():
        return []
    con = duckdb.connect(str(cfg.paths.duckdb_path), read_only=True)
    try:
        rows = con.execute(_RACES_SQL, [limit]).fetchall()
    finally:
        con.close()
    return [
        RaceSummary(
            race_date=str(d),
            venue=str(v),
            race_no=int(rn),
            distance=int(dist) if dist is not None else None,
            going=str(going) if going is not None else None,
            field_size=int(fs),
        )
        for d, v, rn, dist, going, fs in rows
    ]


def raceday(cfg: AppConfig | None = None) -> RaceDayResponse:
    """The latest race-day recommendation card from `hkjc race-day` (M7), or a mock if none.

    Recommendations only -- the platform never places a bet (hard invariant)."""
    cfg = cfg or get_config()
    folder = cfg.paths.processed_dir / "raceday"
    files = sorted(folder.glob("*.json")) if folder.exists() else []
    if files:
        data = _read_json(files[-1])
        if data is not None:
            return RaceDayResponse.model_validate({**data, "mock": False})
    runners = [
        RaceDayRunner(
            saddle=4,
            horse_id="HK_2025_X001",
            name="Sample Flyer",
            win_prob=0.28,
            place_prob=0.61,
            win_odds=4.5,
            ev=0.26,
            stake=120.0,
        ),
        RaceDayRunner(
            saddle=1,
            horse_id="HK_2025_X002",
            name="Placeholder Bay",
            win_prob=0.15,
            place_prob=0.44,
            win_odds=5.5,
            ev=-0.18,
            stake=0.0,
        ),
    ]
    return RaceDayResponse(
        mock=True,
        race_date="2026-06-24",
        venue="HV",
        model_name="logit",
        has_live_odds=False,
        note="Mocked card -- run `hkjc race-day --date YYYY-MM-DD --venue ST` to populate.",
        races=[RaceDayRace(race_no=1, status="DECLARED", runners=runners)],
    )
