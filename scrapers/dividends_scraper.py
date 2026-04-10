# scrapers/dividends_scraper.py
import datetime
import logging
import re

import pandas as pd

from config import (DIVIDENDS_CUTOFF_YEAR, RAW_DIVIDENDS_DIR,
                    RESULTS_BASE_URL, RESULTS_START_DATE, VALID_POOL_TYPES,
                    VALID_VENUES)
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import read, write

logger = logging.getLogger(__name__)


class DividendsScraper(BaseScraper):
    """
    Scrape WIN, PLA, QIN, QPL dividends.

    Races with race_year < DIVIDENDS_CUTOFF_YEAR are recorded but all
    dividend values are stored as NaN (page structure is unreliable
    for those years).
    """

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        start = datetime.date.fromisoformat(start_date or RESULTS_START_DATE)
        end   = datetime.date.fromisoformat(
            end_date or datetime.date.today().isoformat()
        )
        existing = self._existing_dates()
        cursor   = start
        while cursor <= end:
            if cursor.isoformat() not in existing:
                try:
                    self._scrape_date(cursor)
                except Exception as exc:
                    logger.warning("Dividends skip %s: %s", cursor, exc)
            cursor += datetime.timedelta(days=1)

    # ── Private ───────────────────────────────────────────────────────────────

    def _existing_dates(self) -> set:
        dates: set = set()
        for fp in RAW_DIVIDENDS_DIR.glob("dividends_*.parquet"):
            try:
                df = read(fp)
                dates.update(
                    pd.to_datetime(df["race_date"]).dt.date.astype(str).tolist()
                )
            except Exception:
                pass
        return dates

    def _scrape_date(self, race_date: datetime.date):
        rows = []
        for venue in sorted(VALID_VENUES):
            for race_no in range(1, 13):
                try:
                    r = self._scrape_race(race_date, venue, race_no)
                    if not r:
                        break
                    rows.extend(r)
                except Exception:
                    break

        if not rows:
            return
        df   = pd.DataFrame(rows)
        year = race_date.year
        fp   = RAW_DIVIDENDS_DIR / f"dividends_{year}.parquet"
        if fp.exists():
            existing_df = read(fp)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["race_date", "venue", "race_no", "pool_type",
                        "winning_combination"]
            )
        write(df, fp)
        logger.info("Dividends saved — %d rows — %s.", len(rows), race_date)

    def _scrape_race(self, race_date: datetime.date, venue: str,
                     race_no: int) -> list:
        year = race_date.year

        # Before cutoff: store placeholder NaN rows so date is marked as scraped
        if year < DIVIDENDS_CUTOFF_YEAR:
            return [{
                "race_date":           race_date.isoformat(),
                "venue":               venue,
                "race_no":             race_no,
                "pool_type":           pt,
                "winning_combination": None,
                "dividend_hkd":        None,
            } for pt in VALID_POOL_TYPES]

        params = {
            "racedate":   race_date.strftime("%Y/%m/%d"),
            "Racecourse": venue,
            "RaceNo":     str(race_no),
        }
        soup = self.get_html(RESULTS_BASE_URL, params=params)

        # Find dividend table (typically after the performance table)
        tables = soup.find_all("table")
        div_table = None
        for t in tables:
            headers = [th.get_text(strip=True).upper()
                       for th in t.find_all("th")]
            if any("DIV" in h or "POOL" in h for h in headers):
                div_table = t
                break

        if not div_table:
            return []

        rows = []
        for tr in div_table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            pool_raw = cells[0].strip().upper()
            # Map PLA/PLACE variants
            pool_type = _normalise_pool(pool_raw)
            if pool_type not in VALID_POOL_TYPES:
                continue
            rows.append({
                "race_date":           race_date.isoformat(),
                "venue":               venue,
                "race_no":             race_no,
                "pool_type":           pool_type,
                "winning_combination": cells[1].strip(),
                "dividend_hkd":        _safe_float(
                    re.sub(r"[^\d.]", "", cells[2])
                ),
            })
        return rows


def _normalise_pool(raw: str) -> str:
    mapping = {
        "WIN":   "WIN",
        "W":     "WIN",
        "PLACE": "PLA",
        "PLA":   "PLA",
        "P":     "PLA",
        "QIN":   "QIN",
        "Q":     "QIN",
        "QPL":   "QPL",
        "QP":    "QPL",
    }
    return mapping.get(raw.upper(), raw.upper())


def _safe_float(val):
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return None