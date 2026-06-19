"""M1 scrape orchestration: fetch -> parse -> store, idempotently.

A meeting whose date is in the past is *frozen*: once its meeting URL is recorded in the
manifest, re-runs skip it entirely (zero network fetches, zero new rows).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import duckdb

from hkjc.common.config import AppConfig, get_config
from hkjc.common.time import now_hkt
from hkjc.data.holidays import parse_holidays
from hkjc.data.models import MeetingResults, WeatherDaily
from hkjc.data.parse.fixtures import parse_fixture_meeting_days
from hkjc.data.parse.profiles import parse_horse_profile, parse_person_profile
from hkjc.data.parse.results import (
    count_races,
    parse_meeting_dates,
    parse_race_result,
    parse_venue,
)
from hkjc.data.parse.sectionals import parse_race_sectionals
from hkjc.data.parse.text import parse_comments_on_running
from hkjc.data.parse.trials import parse_barrier_trials
from hkjc.data.scrape.client import Fetcher
from hkjc.data.store.manifest import Manifest
from hkjc.data.store.writer import (
    refresh_views,
    season_label,
    write_comments,
    write_holidays,
    write_horse_profile,
    write_meeting,
    write_person_profile,
    write_sectionals,
    write_trackwork,
    write_trials,
    write_weather,
)
from hkjc.data.trackwork import (
    parse_trackwork_dates,
    parse_trackwork_page,
    racing_origin,
    trackwork_url,
)
from hkjc.data.weather.hko import DATATYPES, VENUE_STATION, parse_climate_csv, weather_url

DEFAULT_RATE_PER_SEC = 5.0
DEFAULT_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class ScrapeReport:
    """Outcome of scraping one meeting date."""

    race_date: date
    venue: str | None
    races: int
    fetched: int  # network GETs (cache hits and skips excluded)
    skipped: bool  # whole meeting skipped via manifest
    rows: dict[str, int] = field(default_factory=dict)


def meeting_url(base: str, day: date) -> str:
    return f"{base}/resultsall?racedate={day:%Y/%m/%d}"


def race_url(base: str, day: date, venue: str, race_no: int) -> str:
    return f"{base}/localresults?racedate={day:%Y/%m/%d}&Racecourse={venue}&RaceNo={race_no}"


def scrape_meeting(
    day: date,
    *,
    cfg: AppConfig | None = None,
    fetcher: Fetcher | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
) -> ScrapeReport:
    """Scrape one meeting's full results into raw Parquet + the manifest."""
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    frozen = day < now_hkt().date()

    owns_manifest = manifest is None
    manifest = manifest or Manifest(cfg.paths.duckdb_path)
    fetcher = fetcher or Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    try:
        m_url = meeting_url(base, day)
        if frozen and not force and manifest.has(m_url):
            return ScrapeReport(day, venue=None, races=0, fetched=0, skipped=True)

        m_res = fetcher.fetch(m_url)
        fetched = 0 if m_res.from_cache else 1
        venue = parse_venue(m_res.text)
        if venue is None:  # no meeting on this date
            if frozen:
                manifest.record(m_url, "resultsall", m_res.content_hash, m_res.status, 0)
            return ScrapeReport(day, venue=None, races=0, fetched=fetched, skipped=False)

        n_races = count_races(m_res.text)
        urls = [race_url(base, day, venue, no) for no in range(1, n_races + 1)]
        race_results = fetcher.fetch_many(urls)
        fetched += sum(0 if r.from_cache else 1 for r in race_results)

        races = []
        for no, result in zip(range(1, n_races + 1), race_results, strict=True):
            race = parse_race_result(result.text, day, venue, no)
            races.append(race)
            manifest.record(
                urls[no - 1], "localresults", result.content_hash, result.status, len(race.runners)
            )

        meeting = MeetingResults(race_date=day, venue=venue, races=races)
        counts = write_meeting(cfg.paths.raw_dir, meeting)
        refresh_views(manifest.con, cfg.paths.raw_dir)
        # Record the meeting URL last: its presence means the meeting is fully stored.
        manifest.record(m_url, "resultsall", m_res.content_hash, m_res.status, n_races)
        return ScrapeReport(
            day, venue=venue, races=len(races), fetched=fetched, skipped=False, rows=counts
        )
    finally:
        if owns_manifest:
            manifest.close()


def list_meeting_dates(cfg: AppConfig | None = None, fetcher: Fetcher | None = None) -> list[date]:
    """Recent meeting dates from the results ``selectId`` dropdown (~2 seasons only).

    For full historical enumeration use :func:`list_fixture_dates` (the dropdown does not
    list dates older than ~2 seasons, even though the result pages themselves exist).
    """
    cfg = cfg or get_config()
    fetcher = fetcher or Fetcher(cfg.paths.cache_dir, use_cache=False)
    landing = fetcher.fetch(f"{cfg.sources.hkjc_base_url}/localresults")
    return sorted(parse_meeting_dates(landing.text))


DEFAULT_BACKFILL_START = date(2006, 9, 1)  # ~earliest fixtures-calendar coverage


def fixture_url(base: str, year: int, month: int) -> str:
    return f"{base}/fixture?calyear={year}&calmonth={month:02d}"


def _iter_year_months(start: date, end: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def list_fixture_dates(
    *,
    cfg: AppConfig | None = None,
    start: date = DEFAULT_BACKFILL_START,
    end: date | None = None,
    fetcher: Fetcher | None = None,
) -> list[date]:
    """Enumerate every meeting date in ``[start, end]`` from the fixtures calendar.

    Authoritative back to ~2006 via ``fixture?calyear=Y&calmonth=M`` (one request per month).
    """
    cfg = cfg or get_config()
    end = end or now_hkt().date()
    fetcher = fetcher or Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    months = _iter_year_months(start, end)
    results = fetcher.fetch_many([fixture_url(cfg.sources.hkjc_base_url, y, m) for y, m in months])
    dates: list[date] = []
    for (year, month), result in zip(months, results, strict=True):
        for day in parse_fixture_meeting_days(result.text):
            try:
                meeting_day = date(year, month, day)
            except ValueError:
                continue
            if start <= meeting_day <= end:
                dates.append(meeting_day)
    return sorted(set(dates))


def backfill(
    *,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    since: date | None = None,
    force: bool = False,
    on_meeting: Callable[[ScrapeReport], None] | None = None,
) -> list[ScrapeReport]:
    """Scrape every meeting from the fixtures calendar (idempotent, back to ~2006).

    ``since`` sets the start of enumeration (default ~2006-09); ``limit`` keeps the newest N.
    """
    cfg = cfg or get_config()
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    dates = list_fixture_dates(cfg=cfg, start=since or DEFAULT_BACKFILL_START, fetcher=fetcher)
    if limit is not None:
        dates = dates[-limit:]

    reports: list[ScrapeReport] = []
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for day in dates:
            report = scrape_meeting(day, cfg=cfg, fetcher=fetcher, manifest=manifest, force=force)
            reports.append(report)
            if on_meeting is not None:
                on_meeting(report)
    return reports


def horse_url(base: str, horse_id: str) -> str:
    return f"{base}/horse?horseid={horse_id}"


def stored_horse_ids(cfg: AppConfig | None = None) -> list[str]:
    """Distinct horse ids seen in stored results (the set worth profiling)."""
    cfg = cfg or get_config()
    if not cfg.paths.duckdb_path.is_file():
        return []
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    try:
        rows = con.execute(
            "SELECT DISTINCT horse_id FROM results WHERE horse_id IS NOT NULL ORDER BY horse_id"
        ).fetchall()
        return [str(r[0]) for r in rows]
    except duckdb.Error:
        return []
    finally:
        con.close()


def scrape_horses(
    *,
    cfg: AppConfig | None = None,
    horse_ids: list[str] | None = None,
    limit: int | None = None,
    on_horse: Callable[[str, int], None] | None = None,
) -> dict[str, int]:
    """Scrape horse profiles (bio + form) for ``horse_ids`` (default: all seen in results).

    Profiles are mutable, so this always refetches (the on-disk cache keeps it polite
    within a session); each per-horse file is overwritten.
    """
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    ids = horse_ids if horse_ids is not None else stored_horse_ids(cfg)
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        return {"horses": 0, "form_rows": 0}

    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    results = fetcher.fetch_many([horse_url(base, hid) for hid in ids])
    total_form = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        as_of = now_hkt().date()
        for hid, result in zip(ids, results, strict=True):
            profile = parse_horse_profile(result.text, hid, as_of=as_of)
            n_form = write_horse_profile(cfg.paths.raw_dir, profile)
            total_form += n_form
            manifest.record(result.url, "horse", result.content_hash, result.status, n_form)
            if on_horse is not None:
                on_horse(hid, n_form)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"horses": len(ids), "form_rows": total_form}


def person_url(base: str, role: str, code: str) -> str:
    if role == "jockey":
        return f"{base}/jockeyprofile?jockeyid={code}&Season=Current"
    return f"{base}/trainerprofile?trainerid={code}&Season=Current"


def stored_person_codes(cfg: AppConfig | None = None) -> list[tuple[str, str]]:
    """Distinct (role, code) jockey/trainer pairs seen in stored results."""
    cfg = cfg or get_config()
    if not cfg.paths.duckdb_path.is_file():
        return []
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    pairs: list[tuple[str, str]] = []
    try:
        for role, column in (("jockey", "jockey_code"), ("trainer", "trainer_code")):
            try:
                rows = con.execute(
                    f"SELECT DISTINCT {column} FROM results "
                    f"WHERE {column} IS NOT NULL ORDER BY {column}"
                ).fetchall()
            except duckdb.Error:
                continue
            pairs.extend((role, str(r[0])) for r in rows)
    finally:
        con.close()
    return pairs


def scrape_people(
    *,
    cfg: AppConfig | None = None,
    codes: list[tuple[str, str]] | None = None,
    limit: int | None = None,
    on_person: Callable[[str, str], None] | None = None,
) -> dict[str, int]:
    """Scrape jockey/trainer profiles for ``(role, code)`` pairs (default: all in results)."""
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    pairs = codes if codes is not None else stored_person_codes(cfg)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        return {"people": 0, "jockeys": 0, "trainers": 0}

    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    results = fetcher.fetch_many([person_url(base, role, code) for role, code in pairs])
    jockeys = 0
    trainers = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for (role, code), result in zip(pairs, results, strict=True):
            write_person_profile(cfg.paths.raw_dir, parse_person_profile(result.text, code, role))
            manifest.record(result.url, role, result.content_hash, result.status, 1)
            if role == "jockey":
                jockeys += 1
            else:
                trainers += 1
            if on_person is not None:
                on_person(role, code)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"people": len(pairs), "jockeys": jockeys, "trainers": trainers}


def ingest_weather(*, cfg: AppConfig | None = None, since_year: int = 2000) -> dict[str, int]:
    """Ingest HKO daily-climate temperature series for both venues (mapped stations).

    Each station's full history is one request; rows older than ``since_year`` are dropped.
    """
    cfg = cfg or get_config()
    api = cfg.sources.hko_weather_api
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    data_types = list(DATATYPES.items())  # [(CLMTEMP, mean_temp), ...]
    records: list[WeatherDaily] = []
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for venue, station in VENUE_STATION.items():
            results = fetcher.fetch_many([weather_url(api, station, dt) for dt, _ in data_types])
            series: dict[str, dict[date, float]] = {}
            for (_dt, field), result in zip(data_types, results, strict=True):
                series[field] = parse_climate_csv(result.text)
                manifest.record(
                    result.url,
                    "hko_weather",
                    result.content_hash,
                    result.status,
                    len(series[field]),
                )
            all_dates: set[date] = set()
            for values in series.values():
                all_dates.update(values)
            for day in sorted(all_dates):
                if day.year < since_year:
                    continue
                records.append(
                    WeatherDaily(
                        date=day,
                        station=station,
                        venue=venue,
                        mean_temp=series["mean_temp"].get(day),
                        max_temp=series["max_temp"].get(day),
                        min_temp=series["min_temp"].get(day),
                    )
                )
        n_rows = write_weather(cfg.paths.raw_dir, records)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"weather_rows": n_rows, "stations": len(VENUE_STATION)}


def trial_url(base: str, day: date) -> str:
    return f"{base}/btresult?date={day:%Y/%m/%d}"


def list_trial_dates(cfg: AppConfig | None = None, fetcher: Fetcher | None = None) -> list[date]:
    """Fetch the barrier-trial landing page and return all trial dates (ascending)."""
    cfg = cfg or get_config()
    fetcher = fetcher or Fetcher(cfg.paths.cache_dir, use_cache=False)
    landing = fetcher.fetch(f"{cfg.sources.hkjc_base_url}/btresult")
    return sorted(parse_meeting_dates(landing.text))


def scrape_trials(
    *,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    since: date | None = None,
    force: bool = False,
    on_date: Callable[[date, int], None] | None = None,
) -> dict[str, int]:
    """Scrape barrier-trial results per date (idempotent; past dates are frozen)."""
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    dates = list_trial_dates(cfg, fetcher)
    if since is not None:
        dates = [d for d in dates if d >= since]
    if limit is not None:
        dates = dates[-limit:]

    total_rows = 0
    scraped = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for day in dates:
            url = trial_url(base, day)
            if day < now_hkt().date() and not force and manifest.has(url):
                continue
            result = fetcher.fetch(url)
            n_rows = write_trials(cfg.paths.raw_dir, day, parse_barrier_trials(result.text, day))
            manifest.record(url, "btresult", result.content_hash, result.status, n_rows)
            total_rows += n_rows
            scraped += 1
            if on_date is not None:
                on_date(day, n_rows)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"trial_dates": scraped, "trial_rows": total_rows}


def scrape_trackwork(
    *,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    since: date | None = None,
    force: bool = False,
    max_pages: int = 60,
    on_date: Callable[[date, int], None] | None = None,
) -> dict[str, int]:
    """Scrape trackwork (paginated JSON) per available date (idempotent on frozen dates)."""
    cfg = cfg or get_config()
    origin = racing_origin(cfg.sources.hkjc_base_url)
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    landing = fetcher.fetch(f"{cfg.sources.hkjc_base_url}/trackworksearch")
    dates = parse_trackwork_dates(landing.text)
    if since is not None:
        dates = [d for d in dates if d >= since]
    if limit is not None:
        dates = dates[-limit:]

    total_rows = 0
    scraped = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for day in dates:
            marker = trackwork_url(origin, day, 0)  # stable per-date manifest key
            if day < now_hkt().date() and not force and manifest.has(marker):
                continue
            records = []
            content_hash = ""
            status = 0
            page = 1
            for _ in range(max_pages):
                result = fetcher.fetch(trackwork_url(origin, day, page))
                content_hash, status = result.content_hash, result.status
                next_page, page_records = parse_trackwork_page(result.text, day)
                records.extend(page_records)
                if not page_records or next_page <= page:
                    break
                page = next_page
            n_rows = write_trackwork(cfg.paths.raw_dir, day, records)
            manifest.record(marker, "trackwork", content_hash, status, n_rows)
            total_rows += n_rows
            scraped += 1
            if on_date is not None:
                on_date(day, n_rows)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"trackwork_dates": scraped, "trackwork_rows": total_rows}


def sectional_url(base: str, day: date, race_no: int) -> str:
    return f"{base}/displaysectionaltime?racedate={day:%d/%m/%Y}&RaceNo={race_no}"


def stored_meeting_races(cfg: AppConfig | None = None) -> list[tuple[date, str, list[int]]]:
    """Stored meetings as (date, venue, sorted race numbers) from the races view."""
    cfg = cfg or get_config()
    if not cfg.paths.duckdb_path.is_file():
        return []
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    try:
        rows = con.execute(
            "SELECT race_date, venue, race_no FROM races ORDER BY race_date, venue, race_no"
        ).fetchall()
    except duckdb.Error:
        return []
    finally:
        con.close()
    meetings: dict[tuple[date, str], list[int]] = {}
    for race_date, venue, race_no in rows:
        meetings.setdefault((race_date, venue), []).append(int(race_no))
    return [(d, v, nos) for (d, v), nos in meetings.items()]


def scrape_sectionals(
    *,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    since: date | None = None,
    force: bool = False,
    on_meeting: Callable[[date, str, int], None] | None = None,
) -> dict[str, int]:
    """Scrape per-race sectional times for stored meetings (#7).

    The ``displaysectionaltime`` page is one race; we fetch every race of each stored meeting
    and store the splits meeting-partitioned. Idempotent: a frozen (past) meeting recorded in
    the manifest is skipped.
    """
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    meetings = sorted(stored_meeting_races(cfg))
    if since is not None:
        meetings = [m for m in meetings if m[0] >= since]
    if limit is not None:
        meetings = meetings[-limit:]

    total_rows = 0
    scraped = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for day, venue, race_nos in meetings:
            marker = sectional_url(base, day, 0)  # per-meeting idempotency marker
            if day < now_hkt().date() and not force and manifest.has(marker):
                continue
            results = fetcher.fetch_many([sectional_url(base, day, n) for n in race_nos])
            rows: list[dict[str, Any]] = []
            for race_no, result in zip(race_nos, results, strict=True):
                rows.extend(
                    {"race_no": race_no, **split.model_dump()}
                    for split in parse_race_sectionals(result.text)
                )
            n = write_sectionals(cfg.paths.raw_dir, day, venue, rows)
            last = results[-1] if results else None
            manifest.record(
                marker,
                "sectionals",
                last.content_hash if last else "",
                last.status if last else 0,
                n,
            )
            total_rows += n
            scraped += 1
            if on_meeting is not None:
                on_meeting(day, venue, n)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"meetings": scraped, "sectional_rows": total_rows}


def comments_url(base: str, day: date, race_no: int) -> str:
    # corunning honours ``Date=YYYYMMDD`` (NOT ``racedate=YYYY/MM/DD``, which it silently
    # ignores -> always the latest meeting); there is no ``Racecourse`` param (one per date).
    return f"{base}/corunning?Date={day:%Y%m%d}&RaceNo={race_no}"


def scrape_text(
    *,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    since: date | None = None,
    force: bool = False,
    on_meeting: Callable[[date, str, int], None] | None = None,
) -> dict[str, int]:
    """Scrape English comments-on-running (#9) per race for each stored meeting.

    Lagged text (PLAN.md §1C): the meeting date is the text_event_time. Idempotent: a frozen
    past meeting recorded in the manifest is skipped. (The meeting-level report pages --
    racereportfull / veterinaryrecord / exceptionalfactors -- do **not** reliably honour the
    date param, so only the reliable per-runner comments are captured.)
    """
    cfg = cfg or get_config()
    base = cfg.sources.hkjc_base_url
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    meetings = sorted(stored_meeting_races(cfg))
    if since is not None:
        meetings = [m for m in meetings if m[0] >= since]
    if limit is not None:
        meetings = meetings[-limit:]

    total_comments = 0
    scraped = 0
    with Manifest(cfg.paths.duckdb_path) as manifest:
        for day, venue, race_nos in meetings:
            marker = f"{base}/corunning?Date={day:%Y%m%d}"
            if day < now_hkt().date() and not force and manifest.has(marker):
                continue
            cor = fetcher.fetch_many([comments_url(base, day, n) for n in race_nos])
            crows: list[dict[str, Any]] = []
            for race_no, res in zip(race_nos, cor, strict=True):
                crows.extend(
                    {"race_no": race_no, **c.model_dump()}
                    for c in parse_comments_on_running(res.text)
                )
            nc = write_comments(cfg.paths.raw_dir, day, venue, crows)
            last = cor[0] if cor else None
            manifest.record(
                marker, "text", last.content_hash if last else "", last.status if last else 0, nc
            )
            total_comments += nc
            scraped += 1
            if on_meeting is not None:
                on_meeting(day, venue, nc)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"meetings": scraped, "comments": total_comments}


def ingest_holidays(*, cfg: AppConfig | None = None) -> dict[str, int]:
    """Ingest the gov.hk public-holiday calendar (#14)."""
    cfg = cfg or get_config()
    fetcher = Fetcher(
        cfg.paths.cache_dir, rate_per_sec=DEFAULT_RATE_PER_SEC, concurrency=DEFAULT_CONCURRENCY
    )
    result = fetcher.fetch(cfg.sources.gov_holidays_url)
    n_rows = write_holidays(cfg.paths.raw_dir, parse_holidays(result.text))
    with Manifest(cfg.paths.duckdb_path) as manifest:
        manifest.record(result.url, "gov_holidays", result.content_hash, result.status, n_rows)
        refresh_views(manifest.con, cfg.paths.raw_dir)
    return {"holidays": n_rows}


def coverage_summary(cfg: AppConfig | None = None) -> dict[str, Any]:
    """Summarize stored coverage from the DuckDB views (for the data-health report)."""
    cfg = cfg or get_config()
    summary: dict[str, Any] = {
        "races_rows": 0,
        "results_rows": 0,
        "dividends_rows": 0,
        "horses_rows": 0,
        "horse_form_rows": 0,
        "people_rows": 0,
        "weather_rows": 0,
        "public_holidays_rows": 0,
        "barrier_trials_rows": 0,
        "trackwork_rows": 0,
        "sectionals_rows": 0,
        "comments_on_running_rows": 0,
        "meetings": 0,
        "manifest_urls": 0,
        "date_min": None,
        "date_max": None,
        "seasons": {},
    }
    if not cfg.paths.duckdb_path.is_file():
        return summary
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    try:
        for table in (
            "races",
            "results",
            "dividends",
            "horses",
            "horse_form",
            "people",
            "weather",
            "public_holidays",
            "barrier_trials",
            "trackwork",
            "sectionals",
            "comments_on_running",
        ):
            try:
                row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
                summary[f"{table}_rows"] = int(row[0]) if row else 0
            except duckdb.Error:
                pass
        try:
            meetings = con.execute(
                "SELECT race_date, venue FROM races GROUP BY race_date, venue ORDER BY race_date"
            ).fetchall()
            summary["meetings"] = len(meetings)
            if meetings:
                summary["date_min"] = str(meetings[0][0])
                summary["date_max"] = str(meetings[-1][0])
                seasons: dict[str, int] = {}
                for race_date, _venue in meetings:
                    label = season_label(race_date)
                    seasons[label] = seasons.get(label, 0) + 1
                summary["seasons"] = seasons
        except duckdb.Error:
            pass
        try:
            row = con.execute("SELECT count(*) FROM _scrape_manifest").fetchone()
            summary["manifest_urls"] = int(row[0]) if row else 0
        except duckdb.Error:
            pass
    finally:
        con.close()
    return summary
