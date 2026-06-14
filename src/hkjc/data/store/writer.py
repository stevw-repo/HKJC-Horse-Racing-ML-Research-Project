"""Write parsed meeting results to partitioned raw Parquet + expose DuckDB views.

Raw layout (immutable):
``data/raw/<table>/season=YYYY-YY/venue=XX/date=YYYY-MM-DD/<table>.parquet``.
Partition columns are also kept inside the files, so processed views are a plain recursive
``read_parquet`` glob (no hive dependency).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from hkjc.data.models import MeetingResults

TABLES = ("races", "results", "dividends")

_RACES_SCHEMA: dict[str, pl.DataType] = {
    "race_date": pl.Date(),
    "venue": pl.String(),
    "race_no": pl.Int64(),
    "race_index": pl.Int64(),
    "race_class": pl.String(),
    "distance_m": pl.Int64(),
    "rating_band": pl.String(),
    "race_name": pl.String(),
    "prize_hkd": pl.Int64(),
    "going": pl.String(),
    "course": pl.String(),
    "surface": pl.String(),
    "final_time_s": pl.Float64(),
}
_RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "race_date": pl.Date(),
    "venue": pl.String(),
    "race_no": pl.Int64(),
    "finish_pos": pl.Int64(),
    "finish_pos_raw": pl.String(),
    "dead_heat": pl.Boolean(),
    "saddle": pl.Int64(),
    "horse_id": pl.String(),
    "horse_name": pl.String(),
    "jockey_code": pl.String(),
    "trainer_code": pl.String(),
    "actual_weight": pl.Int64(),
    "declared_weight": pl.Int64(),
    "draw": pl.Int64(),
    "lbw_raw": pl.String(),
    "running_position_raw": pl.String(),
    "finish_time_s": pl.Float64(),
    "win_odds": pl.Float64(),
}
_DIVIDENDS_SCHEMA: dict[str, pl.DataType] = {
    "race_date": pl.Date(),
    "venue": pl.String(),
    "race_no": pl.Int64(),
    "pool": pl.String(),
    "combination": pl.String(),
    "dividend": pl.Float64(),
}
_SCHEMAS = {"races": _RACES_SCHEMA, "results": _RESULTS_SCHEMA, "dividends": _DIVIDENDS_SCHEMA}


def season_label(day: date) -> str:
    """HK racing season label (Sep-Jul). E.g. 2026-06-03 -> ``2025-26``."""
    start = day.year if day.month >= 9 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _flatten(
    meeting: MeetingResults,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    races: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    dividends: list[dict[str, Any]] = []
    key = {"race_date": meeting.race_date, "venue": meeting.venue}
    for race in meeting.races:
        rk = {**key, "race_no": race.race_no}
        races.append(
            {
                **rk,
                "race_index": race.race_index,
                "race_class": race.race_class,
                "distance_m": race.distance_m,
                "rating_band": race.rating_band,
                "race_name": race.race_name,
                "prize_hkd": race.prize_hkd,
                "going": race.going,
                "course": race.course,
                "surface": race.surface,
                "final_time_s": race.final_time_s,
            }
        )
        for runner in race.runners:
            results.append({**rk, **runner.model_dump()})
        for div in race.dividends:
            dividends.append({**rk, **div.model_dump()})
    return races, results, dividends


def _meeting_dir(raw_dir: Path, table: str, day: date, venue: str) -> Path:
    return (
        raw_dir / table / f"season={season_label(day)}" / f"venue={venue}" / f"date={day:%Y-%m-%d}"
    )


def write_meeting(raw_dir: Path, meeting: MeetingResults) -> dict[str, int]:
    """Write a meeting's races/results/dividends to partitioned Parquet; return row counts."""
    flat = dict(zip(TABLES, _flatten(meeting), strict=True))
    counts: dict[str, int] = {}
    for table, rows in flat.items():
        directory = _meeting_dir(raw_dir, table, meeting.race_date, meeting.venue)
        directory.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(rows, schema=_SCHEMAS[table])
        frame.write_parquet(directory / f"{table}.parquet")
        counts[table] = frame.height
    return counts


def refresh_views(con: Any, raw_dir: Path) -> None:
    """(Re)create DuckDB views over the raw Parquet for each table that has data."""
    for table in TABLES:
        if not any((raw_dir / table).rglob("*.parquet")):
            continue
        glob = f"{(raw_dir / table).as_posix()}/**/*.parquet"
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
        )
