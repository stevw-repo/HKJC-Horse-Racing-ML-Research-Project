"""Typed records produced by the parsers (pre-storage).

These are the parsed, validated shapes the scraper emits before they are flattened to
Parquet. Market data (``win_odds`` = starting-price = closing line) is captured here but
must stay walled off from fundamental model features (PLAN.md §1B).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class RunnerResult(BaseModel):
    """One finishing-position row from a results page."""

    model_config = ConfigDict(extra="forbid")

    finish_pos: int | None  # parsed integer placing; None if non-numeric (e.g. WV/DISQ)
    finish_pos_raw: str
    dead_heat: bool = False
    saddle: int
    horse_id: str | None
    horse_name: str
    jockey_code: str | None
    trainer_code: str | None
    actual_weight: int | None  # lbs carried
    declared_weight: int | None  # declared horse weight (lbs)
    draw: int | None
    lbw_raw: str | None  # lengths-behind-winner, raw (e.g. "1-1/2", "N", "---")
    running_position_raw: str | None
    finish_time_s: float | None
    win_odds: float | None  # starting price (MARKET data; never a fundamental feature)


class Dividend(BaseModel):
    """One pool dividend row (generic across WIN..QUARTET → exotics-ready)."""

    model_config = ConfigDict(extra="forbid")

    pool: str  # WIN, PLACE, QUINELLA, QUINELLA PLACE, FORECAST, TIERCE, TRIO, FIRST 4, QUARTET, ...
    combination: str  # e.g. "3", "3,12", "3,12,6"
    dividend: float  # HK$ per winning unit


class RaceResult(BaseModel):
    """A single race: meta + finishing order + full dividend table."""

    model_config = ConfigDict(extra="forbid")

    race_date: date
    venue: str
    race_no: int
    race_index: int | None = None  # HKJC global race index, e.g. (745)
    race_class: str | None = None
    distance_m: int | None = None
    rating_band: str | None = None
    race_name: str | None = None
    prize_hkd: int | None = None
    going: str | None = None
    course: str | None = None  # e.g. 'TURF - "C" Course'
    surface: str | None = None  # TURF / ALL WEATHER TRACK
    final_time_s: float | None = None
    runners: list[RunnerResult]
    dividends: list[Dividend]


class MeetingResults(BaseModel):
    """All races of one meeting (date + venue)."""

    model_config = ConfigDict(extra="forbid")

    race_date: date
    venue: str
    races: list[RaceResult]
