# %% [markdown]
# # Notebook 1 – Scrape & Build Raw Dataset
#
# **Phases:**
# 1. Scrape every HKJC race result from 2024-09-01 to yesterday.
# 2. Scrape individual horse-profile pages for all unique horses.
# 3. Fetch daily weather data from Hong Kong Observatory Open Data API.
#
# Outputs (in `data/raw/`):
# - `raw_races.parquet`   – one row per runner per race
# - `raw_horses.parquet`  – one row per unique horse
# - `weather.parquet`     – one row per (date, racecourse)
# - `scraping_checkpoint.json` – progress file; delete to re-scrape from scratch

# %% ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import re
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Project utilities (same directory)
from utils import (
    Checkpoint,
    build_session,
    convert_distance_beaten,
    ensure_dirs,
    is_valid_finisher,
    logger,
    safe_get,
)

warnings.filterwarnings("ignore")

# %% ── Configuration ────────────────────────────────────────────────────────

# ── Date range ───────────────────────────────────────────────────────────────
START_DATE   = date(2024, 9, 1)
END_DATE     = date.today() - timedelta(days=1)   # yesterday

# ── HKJC URLs ────────────────────────────────────────────────────────────────
# Primary URL format (newer site):
RESULTS_URL_TPL = (
    "https://racing.hkjc.com/racing/en-us/local/information/"
    "localresults?racedate={date}&Racecourse={course}&RaceNo={race_no}"
    #"localResults.aspx?RaceDate={date}&Racecourse={course}&RaceNo={race_no}"
)

# Legacy URL format (fallback):
RESULTS_URL_LEGACY_TPL = (
    "https://racing.hkjc.com/racing/information/english/"
    "Racing/LocalResults.aspx?RaceDate={date}&Racecourse={course}&RaceNo={race_no}"
)
HORSE_PROFILE_URL_TPL = (
    "https://racing.hkjc.com/en-us/local/information/"
    "horse?horseid={horse_no}"
    # "Horse/Horse.aspx?HorseNo={horse_no}"
)
# https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_2024_K284

# ── HKO Weather API ───────────────────────────────────────────────────────────
HKO_BASE = "https://data.weather.gov.hk/weatherAPI/opendata/climate.php"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW        = Path("data/raw")
CHECKPOINT_PATH = DATA_RAW / "scraping_checkpoint.json"
RAW_RACES_PATH  = DATA_RAW / "raw_races.parquet"
RAW_HORSES_PATH = DATA_RAW / "raw_horses.parquet"
WEATHER_PATH    = DATA_RAW / "weather.parquet"

# ── Scraper settings ─────────────────────────────────────────────────────────
SLEEP_BETWEEN_REQS = 1.0   # seconds – be polite
COURSES            = ["ST", "HV"]
MAX_RACE_NO        = 11

# %% ── HTML-parsing helpers ─────────────────────────────────────────────────

def _text(tag: Any) -> str:
    """Return stripped inner text of a BS4 tag (or '' if None)."""
    return tag.get_text(strip=True) if tag else ""


def _find_results_table(soup: BeautifulSoup) -> Optional[Any]:
    """
    Locate the main runner-results table on the HKJC results page.

    Strategy
    --------
    1. Look for a <table> whose headers contain both 'Horse' and 'Jockey'.
    2. Fall back to any <table> with > 5 columns if step 1 fails.
    """
    for table in soup.find_all("table"):
        headers = [_text(th).lower() for th in table.find_all("th")]
        if any("horse" in h for h in headers) and any("jockey" in h for h in headers):
            return table
        # some pages use <td> in the first row as headers
        first_row = table.find("tr")
        if first_row:
            cells = [_text(td).lower() for td in first_row.find_all(["td", "th"])]
            if any("horse" in c for c in cells) and any("jockey" in c for c in cells):
                return table
    # Last-resort: largest table on page
    tables = soup.find_all("table")
    if tables:
        return max(tables, key=lambda t: len(t.find_all("tr")))
    return None


def _extract_column_map(header_row: Any) -> Dict[str, int]:
    """
    Build a {canonical_name: column_index} map from a table header row.

    Canonical names: place, horse_no, horse_name, jockey, trainer,
    actual_weight_lbs, draw, win_odds, place_odds, lbw, run_pos, finish_time.
    """
    cells = [_text(c).lower() for c in header_row.find_all(["th", "td"])]
    cmap: Dict[str, int] = {}

    patterns = {
        "place":              r"pl[ac]",
        "horse_no":           r"no\.?$|horse\s*no",
        "horse_name":         r"horse$|horse\s*name",
        "jockey":             r"jockey",
        "trainer":            r"trainer",
        "actual_weight_lbs":  r"act\.?\s*wt|actual\s*wt|weight",
        "draw":               r"^dr$|draw",
        "win_odds":           r"win\s*odds|wn\s*odds|odds$",
        "place_odds":         r"pl[ac][e]?\s*odds",
        "lbw":                r"lbw|behind|beaten|margin",
        "run_pos":            r"running|run\s*pos",
        "finish_time":        r"time|finish\s*time",
        "sect_400":           r"400m?|s400",
        "sect_800":           r"800m?|s800",
        "sect_1200":          r"1200m?|s1200",
    }
    for col_name, pattern in patterns.items():
        for idx, cell_text in enumerate(cells):
            if re.search(pattern, cell_text):
                cmap[col_name] = idx
                break
    return cmap


def _parse_race_meta(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract race-level metadata (class, distance, going, course_type, prize)
    from anywhere on the HKJC results page.

    Returns a dict with keys:
    race_class, distance_m, going, course_type, prize_hkd.
    Unextracted fields are set to None.
    """
    full_text = soup.get_text(separator=" ")

    meta: Dict[str, Any] = {
        "race_class":  None,
        "distance_m":  None,
        "going":       None,
        "course_type": None,
        "prize_hkd":   None,
    }

    # Race class  (Class 1 … Class 5, Griffin, International)
    m = re.search(r"Class\s+(\d)|Griffin|International\s+Group", full_text, re.I)
    if m:
        meta["race_class"] = m.group(0).strip()

    # Distance  (e.g. "1000 M", "1200M", "2000 Metres")
    m = re.search(r"(\d{3,4})\s*[Mm](?:etres?)?", full_text)
    if m:
        meta["distance_m"] = int(m.group(1))

    # Going
    going_tokens = [
        "Good To Firm", "Good to Firm", "Firm",
        "Good", "Good To Yielding", "Good to Yielding",
        "Yielding", "Soft", "Heavy", "Fast",
    ]
    for token in going_tokens:
        if token.lower() in full_text.lower():
            meta["going"] = token
            break

    # Course type
    if re.search(r"all.weather|awt", full_text, re.I):
        meta["course_type"] = "AWT"
    elif "turf" in full_text.lower():
        meta["course_type"] = "Turf"

    # Prize (HK$)
    m = re.search(r"HK\$\s*([\d,]+)", full_text)
    if m:
        meta["prize_hkd"] = int(m.group(1).replace(",", ""))

    return meta


def parse_race_page(
    html: str,
    race_date: str,
    racecourse: str,
    race_no: int,
) -> List[Dict[str, Any]]:
    """
    Parse one HKJC local-results page and return a list of runner dicts.

    Each dict contains both race-level metadata and runner-level data.

    Parameters
    ----------
    html:       Raw HTML string from the page.
    race_date:  Date string "YYYY/MM/DD".
    racecourse: "ST" or "HV".
    race_no:    Race number (1–11).

    Returns
    -------
    List[Dict] – one dict per runner; empty list if the table was not found.
    """
    soup = BeautifulSoup(html, "lxml")
    race_id = f"{race_date.replace('/', '-')}_{racecourse}_{race_no}"

    # ── Race metadata ────────────────────────────────────────────────────────
    meta = _parse_race_meta(soup)

    # ── Results table ────────────────────────────────────────────────────────
    table = _find_results_table(soup)
    if table is None:
        logger.debug("No results table found for %s", race_id)
        return []

    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    # Detect header row (first row that contains <th> or has column-like text)
    header_row = rows[0]
    cmap = _extract_column_map(header_row)
    if not cmap:
        # Try second row
        cmap = _extract_column_map(rows[1]) if len(rows) > 1 else {}

    if "horse_name" not in cmap:
        logger.debug("Could not parse column map for %s — cmap: %s", race_id, cmap)
        return []

    def _cell(cells: List, key: str, default: str = "") -> str:
        idx = cmap.get(key)
        if idx is None or idx >= len(cells):
            return default
        return _text(cells[idx])

    runners: List[Dict[str, Any]] = []

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        place_raw = _cell(cells, "place")
        if not place_raw:
            continue  # skip spacer rows

        horse_cell = cells[cmap.get("horse_name", 1)] if cmap.get("horse_name") is not None else None
        horse_name = _text(horse_cell) if horse_cell else ""

        # Extract horse brand number from the <a href> in the horse cell
        horse_no = ""
        if horse_cell:
            link = horse_cell.find("a", href=True)
            if link:
                m = re.search(r"HorseNo=([^&\"']+)", link["href"], re.I)
                if m:
                    horse_no = m.group(1).strip()
            if not horse_no:
                # Fallback: try the "No." column
                horse_no = _cell(cells, "horse_no")

        # Finish position
        finish_pos: Any = None
        if is_valid_finisher(place_raw):
            try:
                finish_pos = int(place_raw)
            except ValueError:
                finish_pos = place_raw
        else:
            finish_pos = place_raw   # keep WX/DNF/etc.

        # Numeric odds (handle "SCR" etc.)
        def _to_float(s: str) -> Optional[float]:
            try:
                return float(s.replace(",", ""))
            except (ValueError, AttributeError):
                return None

        win_odds   = _to_float(_cell(cells, "win_odds"))
        place_odds = _to_float(_cell(cells, "place_odds"))

        # Actual weight  (HKJC reports in pounds; convert to kg)
        wt_str = _cell(cells, "actual_weight_lbs")
        wt_lbs: Optional[float] = _to_float(wt_str)
        wt_kg = round(wt_lbs * 0.453592, 2) if wt_lbs is not None else None

        # Draw
        draw_str = _cell(cells, "draw")
        draw: Optional[int] = None
        try:
            draw = int(draw_str)
        except ValueError:
            pass

        # LBW
        lbw = convert_distance_beaten(_cell(cells, "lbw"))

        # Finish time (first number on the row that looks like a race time)
        ft_str = _cell(cells, "finish_time")
        finish_time: Optional[float] = _to_float(ft_str)

        # Sectional times
        sect_400  = _to_float(_cell(cells, "sect_400"))
        sect_800  = _to_float(_cell(cells, "sect_800"))
        sect_1200 = _to_float(_cell(cells, "sect_1200"))

        # Running positions (stored as raw string)
        run_pos = _cell(cells, "run_pos")

        runners.append(
            {
                "race_id":       race_id,
                "race_date":     race_date,
                "racecourse":    racecourse,
                "race_no":       race_no,
                **meta,
                "horse_name":    horse_name,
                "horse_brand_no": horse_no,
                "jockey":        _cell(cells, "jockey"),
                "trainer":       _cell(cells, "trainer"),
                "draw":          draw,
                "actual_weight_kg": wt_kg,
                "win_odds":      win_odds,
                "place_odds":    place_odds,
                "finish_pos":    finish_pos,
                "lbw":           lbw,
                "finish_time_sec": finish_time,
                "sect_400":      sect_400,
                "sect_800":      sect_800,
                "sect_1200":     sect_1200,
                "run_positions": run_pos,
            }
        )

    return runners


# %% ── Race-results scraper ─────────────────────────────────────────────────

def scrape_all_races(
    session: requests.Session,
    checkpoint: Checkpoint,
    start: date,
    end: date,
    courses: List[str] = COURSES,
    max_race_no: int = MAX_RACE_NO,
) -> pd.DataFrame:
    """
    Scrape all race results between *start* and *end* (inclusive).

    Skips already-completed items (per checkpoint).  Saves intermediate
    progress to ``RAW_RACES_PATH`` every time a new date is completed.

    Parameters
    ----------
    session:     Authenticated requests Session.
    checkpoint:  Checkpoint instance for resuming.
    start, end:  Inclusive date range to scrape.

    Returns
    -------
    pd.DataFrame with one row per runner per race.
    """
    # Load previously scraped data if it exists
    existing: List[pd.DataFrame] = []
    if RAW_RACES_PATH.exists():
        existing.append(pd.read_parquet(RAW_RACES_PATH))

    all_runners: List[Dict] = []

    all_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    for race_date in tqdm(all_dates, desc="Dates", unit="day"):
        date_str = race_date.strftime("%Y/%m/%d")
        date_key = race_date.isoformat()

        for course in courses:
            for race_no in range(1, max_race_no + 1):
                item_key = f"{date_key}_{course}_{race_no}"
                if checkpoint.is_done(item_key):
                    continue

                url = RESULTS_URL_TPL.format(
                    date=date_str, course=course, race_no=race_no
                )
                resp = safe_get(session, url, sleep_secs=SLEEP_BETWEEN_REQS)

                if resp is None:
                    # No race at this slot – mark done and continue
                    checkpoint.mark_done(item_key)
                    continue

                runners = parse_race_page(resp.text, date_str, course, race_no)

                if not runners:
                    # Empty table → this race number doesn't exist; stop for this course/date
                    checkpoint.mark_done(item_key)
                    if race_no > 1:
                        # Once races dry up, subsequent numbers are also absent
                        for remaining in range(race_no + 1, max_race_no + 1):
                            checkpoint.mark_done(f"{date_key}_{course}_{remaining}")
                    break

                all_runners.extend(runners)
                checkpoint.mark_done(item_key)
                logger.info("Scraped %s %s R%s — %d runners", date_str, course, race_no, len(runners))

        # Save progress after each date
        if all_runners:
            batch = pd.DataFrame(all_runners)
            if existing:
                combined = pd.concat(existing + [batch], ignore_index=True)
            else:
                combined = batch
            combined.to_parquet(RAW_RACES_PATH, index=False)
            existing = [combined]
            all_runners = []

    return pd.read_parquet(RAW_RACES_PATH) if RAW_RACES_PATH.exists() else pd.DataFrame()


# %% ── Horse-profile scraper ────────────────────────────────────────────────

def parse_horse_profile(html: str, horse_no: str) -> Dict[str, Any]:
    """
    Parse a HKJC horse-profile page and return a dict of horse attributes.

    Extracted fields
    ----------------
    horse_brand_no, country_of_origin, age_years, sex, colour,
    current_rating, last_rating, career_starts, career_wins, form_string.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")

    rec: Dict[str, Any] = {
        "horse_brand_no":  horse_no,
        "country_of_origin": None,
        "age_years":       None,
        "sex":             None,
        "colour":          None,
        "current_rating":  None,
        "last_rating":     None,
        "career_starts":   None,
        "career_wins":     None,
        "form_string":     None,
    }

    # Helper: extract value following a label in a table row
    def _lookup(label_pattern: str, src_text: str) -> Optional[str]:
        m = re.search(label_pattern + r"\s*[:\s]+([^\n\r]{1,40})", src_text, re.I)
        return m.group(1).strip() if m else None

    # Country of origin
    val = _lookup(r"Country of Origin", text)
    if val:
        rec["country_of_origin"] = val.split()[0]

    # Age
    val = _lookup(r"Age", text)
    if val:
        try:
            rec["age_years"] = int(re.search(r"\d+", val).group())
        except (AttributeError, ValueError):
            pass

    # Sex
    val = _lookup(r"Sex", text)
    if val:
        rec["sex"] = val.split()[0]

    # Colour
    val = _lookup(r"Colour", text)
    if val:
        rec["colour"] = val.strip()

    # Ratings
    m = re.search(r"Current\s+Rating\s*[:\s]+(\d+)", text, re.I)
    if m:
        rec["current_rating"] = int(m.group(1))

    m = re.search(r"Last\s+(?:Season\s+)?Rating\s*[:\s]+(\d+)", text, re.I)
    if m:
        rec["last_rating"] = int(m.group(1))

    # Career record  "Starts: 24  Wins: 5"
    m = re.search(r"(?:1st|Wins?)[^\d]*(\d+)[^:]*(?:Starts?|Races?)[^\d]*(\d+)", text, re.I)
    if m:
        rec["career_wins"]   = int(m.group(1))
        rec["career_starts"] = int(m.group(2))
    else:
        m2 = re.search(r"Starts?\s*[:\s]+(\d+)", text, re.I)
        if m2:
            rec["career_starts"] = int(m2.group(1))
        m3 = re.search(r"Wins?\s*[:\s]+(\d+)", text, re.I)
        if m3:
            rec["career_wins"] = int(m3.group(1))

    # Form string  "1 2 0 3 4" or "1/2/0/3"
    m = re.search(r"Form\s*[:\s]+([\d /\-WFURPDs]{3,30})", text, re.I)
    if m:
        rec["form_string"] = m.group(1).strip()

    return rec


def scrape_horse_profiles(
    session: requests.Session,
    checkpoint: Checkpoint,
    horse_nos: List[str],
) -> pd.DataFrame:
    """
    Scrape horse-profile pages for all unique horses in *horse_nos*.

    Skips horses already in the checkpoint.  Saves incrementally to
    ``RAW_HORSES_PATH``.

    Parameters
    ----------
    session:    HTTP session.
    checkpoint: Shared checkpoint (uses prefix ``horse_``).
    horse_nos:  List of HKJC horse brand numbers.

    Returns
    -------
    pd.DataFrame with one row per horse.
    """
    existing: List[pd.DataFrame] = []
    if RAW_HORSES_PATH.exists():
        existing.append(pd.read_parquet(RAW_HORSES_PATH))

    batch: List[Dict] = []

    for horse_no in tqdm(horse_nos, desc="Horse profiles", unit="horse"):
        if not horse_no or horse_no == "nan":
            continue
        ck_key = f"horse_{horse_no}"
        if checkpoint.is_done(ck_key):
            continue

        url  = HORSE_PROFILE_URL_TPL.format(horse_no=horse_no)
        resp = safe_get(session, url, sleep_secs=SLEEP_BETWEEN_REQS)

        if resp is None:
            logger.debug("No profile for horse %s", horse_no)
            checkpoint.mark_done(ck_key)
            continue

        profile = parse_horse_profile(resp.text, horse_no)
        batch.append(profile)
        checkpoint.mark_done(ck_key)

        # Save every 100 horses
        if len(batch) >= 100:
            _flush_horses(batch, existing)
            existing = [pd.read_parquet(RAW_HORSES_PATH)]
            batch = []

    if batch:
        _flush_horses(batch, existing)

    return pd.read_parquet(RAW_HORSES_PATH) if RAW_HORSES_PATH.exists() else pd.DataFrame()


def _flush_horses(batch: List[Dict], existing: List[pd.DataFrame]) -> None:
    """Append *batch* to existing data and save to Parquet."""
    new_df = pd.DataFrame(batch)
    if existing:
        combined = pd.concat(existing + [new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["horse_brand_no"])
    else:
        combined = new_df
    combined.to_parquet(RAW_HORSES_PATH, index=False)


# %% ── Weather scraper ──────────────────────────────────────────────────────

# HKO station codes closest to each racecourse:
#   SHA → Sha Tin  (for ST races)
#   KLT → King's Park (proxy for HV / Happy Valley)
#   HKO → Main observatory (used as fallback)
STATION_MAP = {"ST": "SHA", "HV": "KLT"}


def _fetch_hko_monthly(
    session: requests.Session,
    year: int,
    month: int,
    data_type: str,
    station: str = "HKO",
) -> pd.DataFrame:
    """
    Fetch one month of HKO climate data for a given data type and station.

    Known data types
    ----------------
    CLMMAXTMP  – daily maximum temperature (°C)
    CLMMINTMP  – daily minimum temperature (°C)
    CLMRAIN    – daily rainfall (mm)
    CLMMAXRH   – daily maximum relative humidity (%)
    CLMMINRH   – daily minimum relative humidity (%)
    CLMGRSMIN  – daily minimum grass temperature (°C)

    Returns
    -------
    pd.DataFrame with columns [date, value].
    """
    params = {
        "dataType": data_type,
        "year":     year,
        "month":    f"{month:02d}",
        "lang":     "en",
    }
    try:
        resp = safe_get(session, HKO_BASE, sleep_secs=0.5, params=params)
        if resp is None:
            return pd.DataFrame(columns=["date", "value"])

        payload = resp.json()
        records = []

        # HKO returns either {"data": [...]} or a list
        data_list = payload.get("data", payload) if isinstance(payload, dict) else payload

        for entry in data_list:
            if isinstance(entry, dict):
                stn = entry.get("stn", entry.get("station", ""))
                if stn and stn.upper() != station.upper() and station != "HKO":
                    # Try to match station; accept if no station field
                    if not any(c.isdigit() for c in stn):
                        continue
                day = entry.get("day", entry.get("d", None))
                val = entry.get("value", entry.get("v", None))
                if day is not None and val is not None:
                    try:
                        dt = date(year, month, int(day))
                        records.append({"date": dt, "value": float(val)})
                    except (ValueError, TypeError):
                        pass

        return pd.DataFrame(records)

    except Exception as exc:
        logger.warning("HKO fetch failed (%s, %s, %s): %s", data_type, year, month, exc)
        return pd.DataFrame(columns=["date", "value"])


def scrape_weather(
    session: requests.Session,
    race_dates: List[date],
    courses: List[str] = COURSES,
) -> pd.DataFrame:
    """
    Fetch daily weather data for each race date and racecourse.

    Parameters
    ----------
    session:     HTTP session.
    race_dates:  Sorted list of unique race dates.
    courses:     List of racecourse codes.

    Returns
    -------
    pd.DataFrame with columns:
      date, racecourse, max_temp_c, min_temp_c,
      rainfall_mm, avg_humidity_pct, grass_temp_min_c.
    """
    # Identify year-month pairs needed
    ym_pairs = sorted({(d.year, d.month) for d in race_dates})

    # Data type → column name
    dtypes = {
        "CLMMAXTMP": "max_temp_c",
        "CLMMINTMP": "min_temp_c",
        "CLMRAIN":   "rainfall_mm",
        "CLMMAXRH":  "max_humidity_pct",
        "CLMMINRH":  "min_humidity_pct",
        "CLMGRSMIN": "grass_temp_min_c",
    }

    # Fetch data at HKO main station (used for all courses as baseline)
    monthly_cache: Dict[Tuple[int, int, str], pd.DataFrame] = {}

    for (yr, mo) in tqdm(ym_pairs, desc="Weather months"):
        for dtype in dtypes:
            key = (yr, mo, dtype)
            if key not in monthly_cache:
                df = _fetch_hko_monthly(session, yr, mo, dtype, station="HKO")
                monthly_cache[key] = df

    # Build per-date lookup
    date_lookup: Dict[Tuple[date, str], Dict[str, Any]] = {}
    for (yr, mo, dtype) in monthly_cache:
        df = monthly_cache[(yr, mo, dtype)]
        col = dtypes[dtype]
        for _, row in df.iterrows():
            d = row["date"]
            for course in courses:
                key = (d, course)
                date_lookup.setdefault(key, {"date": d, "racecourse": course})
                date_lookup[key][col] = row["value"]

    # Compute avg_humidity_pct
    records = []
    for (d, course), vals in date_lookup.items():
        if d in [r for r in race_dates]:
            max_rh = vals.get("max_humidity_pct")
            min_rh = vals.get("min_humidity_pct")
            if max_rh is not None and min_rh is not None:
                vals["avg_humidity_pct"] = (max_rh + min_rh) / 2.0
            records.append(vals)

    if not records:
        logger.warning("No weather records built; check HKO API response format.")
        return pd.DataFrame()

    weather_df = pd.DataFrame(records)
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df.sort_values(["date", "racecourse"]).reset_index(drop=True)

    # Keep only race dates
    race_date_set = {pd.Timestamp(d) for d in race_dates}
    weather_df = weather_df[weather_df["date"].isin(race_date_set)]

    weather_df.to_parquet(WEATHER_PATH, index=False)
    logger.info("Saved weather data: %s rows → %s", len(weather_df), WEATHER_PATH)
    return weather_df


# %% ── Main orchestration ───────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full scraping pipeline.

    Order of operations
    -------------------
    1. Scrape race results for all dates in [START_DATE, END_DATE].
    2. Collect unique horse brand numbers from the results.
    3. Scrape horse profiles.
    4. Scrape weather data for all race dates.
    """
    ensure_dirs(DATA_RAW)
    session    = build_session()
    checkpoint = Checkpoint(CHECKPOINT_PATH)

    print(f"\n{'='*60}")
    print("PHASE 1 — Race Results")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Courses    : {COURSES}")
    print(f"{'='*60}\n")

    races_df = scrape_all_races(
        session, checkpoint, START_DATE, END_DATE, COURSES, MAX_RACE_NO
    )
    print(f"\nRace results: {len(races_df):,} runner rows saved to {RAW_RACES_PATH}")

    print(f"\n{'='*60}")
    print("PHASE 2 — Horse Profiles")
    print(f"{'='*60}\n")

    unique_horses = (
        races_df["horse_brand_no"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    print(f"Unique horse brand numbers: {len(unique_horses)}")

    horses_df = scrape_horse_profiles(session, checkpoint, unique_horses)
    print(f"Horse profiles: {len(horses_df):,} rows saved to {RAW_HORSES_PATH}")

    print(f"\n{'='*60}")
    print("PHASE 3 — Weather Data")
    print(f"{'='*60}\n")

    race_dates = (
        races_df["race_date"]
        .dropna()
        .astype(str)
        .apply(lambda s: date.fromisoformat(s.replace("/", "-")))
        .unique()
        .tolist()
    )
    weather_df = scrape_weather(session, race_dates)
    print(f"Weather rows: {len(weather_df)} saved to {WEATHER_PATH}")

    print("\n✅  All scraping complete.")


if __name__ == "__main__":
    main()