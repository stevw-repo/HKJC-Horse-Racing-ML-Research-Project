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
    jockey_code: str | None  # None for retired persons (unlinked on old pages); see jockey_name
    jockey_name: str | None = None
    trainer_code: str | None
    trainer_name: str | None = None
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
    rail: str | None = None  # rail position from the course token, e.g. "C", "A+3" (#3)
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
    age: int | None = None  # current age as scraped (None once retired); see birth_year
    birth_year: int | None = (
        None  # = scrape_year - age (calendar convention); anchor for age-at-race
    )
    colour: str | None = None
    sex: str | None = None
    import_type: str | None = None
    sire: str | None = None
    dam: str | None = None
    dams_sire: str | None = None
    owner: str | None = None
    trainer: str | None = None
    current_rating: int | None = None  # active horses
    last_rating: int | None = None  # retired horses show "Last Rating" instead
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


class WeatherDaily(BaseModel):
    """One day's HKO daily-climate reading at a station, mapped to a venue."""

    model_config = ConfigDict(extra="forbid")

    date: date
    station: str  # HKO / SHA
    venue: str  # HV (HKO) / ST (SHA)
    mean_temp: float | None = None
    max_temp: float | None = None
    min_temp: float | None = None


class PublicHoliday(BaseModel):
    """One HK public holiday (gov.hk open data, source #14)."""

    model_config = ConfigDict(extra="forbid")

    date: date
    name: str


class BarrierTrialRun(BaseModel):
    """One horse's run in a barrier trial heat (pre-race signal, source #4)."""

    model_config = ConfigDict(extra="forbid")

    trial_date: date
    batch: int | None = None
    location: str | None = None  # raw, e.g. "SHA TIN ALL WEATHER TRACK"
    venue: str | None = None  # ST / HV / CH (Conghua)
    surface: str | None = None  # TURF / ALL WEATHER TRACK
    distance_m: int | None = None
    going: str | None = None
    batch_time_s: float | None = None
    horse_id: str | None = None
    horse_name: str
    jockey: str | None = None  # name only (trials don't link jockey/trainer ids)
    trainer: str | None = None
    draw: int | None = None
    gear: str | None = None
    lbw_raw: str | None = None
    running_position_raw: str | None = None
    time_s: float | None = None
    result: str | None = None  # e.g. "Passed"
    comment: str | None = None


class TrackworkRecord(BaseModel):
    """One horse's morning trackwork on a given day (source #5, JSON endpoint)."""

    model_config = ConfigDict(extra="forbid")

    work_date: date
    horse_name: str
    trainer: str | None = None  # name (the JSON feed carries names, not ids)
    work_type: str | None = None  # e.g. Trotting / Gallop / Swimming
    racecourse_track: str | None = None  # e.g. "Conghua TroR"
    workouts: str | None = None  # e.g. "TroR TroR Canter (R.B.)"
    gear: str | None = None


class SectionalSplit(BaseModel):
    """One runner's split for one section of a race (source #7, displaysectionaltime).

    ``section_index`` is 1-based from the start. ``section_time_s`` is the time to run that
    section (the section times sum to ``final_time_s``); ``split_200m_s`` is the per-200m pace
    within the section (the blue ``sectional_200`` value; absent for the first section). The
    home-grown speed figures (M2/M3) consume these splits.
    """

    model_config = ConfigDict(extra="forbid")

    saddle: int
    horse_id: str | None
    finishing_order: int | None
    section_index: int
    running_position: int | None  # position at the section marker
    margin_raw: str | None  # margin behind leader at the marker, e.g. "1-1/4"
    section_time_s: float | None  # time to run this section (sums to final_time_s)
    split_200m_s: float | None  # per-200m pace within the section
    final_time_s: float | None  # the runner's overall finishing time
