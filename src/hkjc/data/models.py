"""Typed records produced by the parsers (pre-storage).

These are the parsed, validated shapes the scraper emits before they are flattened to
Parquet. Market data (``win_odds`` = starting-price = closing line) is captured here but
must stay walled off from fundamental model features (PLAN.md §1B).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


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


class HorseFormRun(BaseModel):
    """One past run from a horse's form-records table (historical pre-race signal)."""

    model_config = ConfigDict(extra="forbid")

    race_index: int | None  # joins to RaceResult.race_index
    finish_pos: int | None
    finish_pos_raw: str
    run_date: date | None
    venue: str | None  # ST / HV (other = overseas, kept verbatim)
    track: str | None  # Turf / AWT
    course: str | None  # e.g. "C"
    distance_m: int | None
    going: str | None  # going code (GF, G, ...)
    race_class: str | None
    draw: int | None
    rating: int | None  # official rating at the time of that run
    jockey_code: str | None
    trainer_code: str | None
    lbw_raw: str | None
    win_odds: float | None
    actual_weight: int | None
    running_position_raw: str | None
    finish_time_s: float | None
    declared_weight: int | None
    gear: str | None


class HorseProfile(BaseModel):
    """Horse profile: locked bio block (PLAN.md §0) + form records."""

    model_config = ConfigDict(extra="forbid")

    horse_id: str
    name: str | None = None
    brand: str | None = None  # e.g. H447
    country_of_origin: str | None = None
    age: int | None = None  # current age (age-at-race is derived from run dates)
    colour: str | None = None
    sex: str | None = None
    import_type: str | None = None
    sire: str | None = None
    dam: str | None = None
    dams_sire: str | None = None
    owner: str | None = None
    trainer: str | None = None
    current_rating: int | None = None
    season_start_rating: int | None = None
    season_stakes: int | None = None
    total_stakes: int | None = None
    form: list[HorseFormRun] = Field(default_factory=list)


class PersonProfile(BaseModel):
    """Jockey or trainer current-season profile (strike-rate inputs)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    role: str  # "jockey" | "trainer"
    name: str | None = None
    age: int | None = None
    nationality: str | None = None
    season: str | None = None  # e.g. "25/26"
    up_to_date: date | None = None  # "Up to Race Meeting of ..."
    wins: int | None = None
    seconds: int | None = None
    thirds: int | None = None
    fourths: int | None = None
    total_starts: int | None = None  # rides (jockey) / runners (trainer)
    win_pct: float | None = None  # percentage value, e.g. 16.55
    stakes: int | None = None
    wins_last10: int | None = None
