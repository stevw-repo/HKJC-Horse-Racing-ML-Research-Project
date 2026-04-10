# scrapers/horse_scraper.py
import logging
import re

import pandas as pd

from config import HORSE_BASE_URL, RAW_HORSES_DIR, RAW_RESULTS_DIR
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import read, upsert, write

logger = logging.getLogger(__name__)


class HorseScraper(BaseScraper):
    """Scrape horse profiles for every unique horse_id found in results."""

    def scrape(self, **kwargs):
        horse_ids  = self._collect_horse_ids()
        profiled   = self._already_profiled()
        to_scrape  = horse_ids - profiled
        logger.info("%d new horse IDs to scrape.", len(to_scrape))

        rows = []
        for horse_id in sorted(to_scrape):
            try:
                row = self._scrape_horse(horse_id)
                if row:
                    rows.append(row)
            except Exception as exc:
                logger.warning("Skip %s: %s", horse_id, exc)

        if rows:
            df = pd.DataFrame(rows)
            upsert(df, RAW_HORSES_DIR / "horse_profiles.parquet", key_cols=["horse_id"])
            logger.info("Upserted %d horse profiles.", len(rows))

    # ── Private ───────────────────────────────────────────────────────────────

    def _collect_horse_ids(self) -> set:
        ids: set = set()
        for fp in RAW_RESULTS_DIR.glob("results_*.parquet"):
            try:
                df = read(fp)
                if "horse_id" in df.columns:
                    ids.update(df["horse_id"].dropna().unique().tolist())
            except Exception:
                pass
        return ids

    def _already_profiled(self) -> set:
        fp = RAW_HORSES_DIR / "horse_profiles.parquet"
        if not fp.exists():
            return set()
        try:
            df = read(fp)
            return set(df["horse_id"].dropna().tolist())
        except Exception:
            return set()

    def _scrape_horse(self, horse_id: str) -> dict | None:
        for path_suffix in [f"horse?horseid={horse_id}",
                             f"otherhorse?horseid={horse_id}"]:
            try:
                url  = f"{HORSE_BASE_URL}/{path_suffix}"
                soup = self.get_html(url)
                if soup.find("div", string=re.compile(r"not.*found|no.*record", re.I)):
                    continue
                return self._parse_profile(soup, horse_id,
                                           is_retired="otherhorse" in path_suffix)
            except Exception:
                continue
        return None

    @staticmethod
    def _parse_profile(soup, horse_id: str, is_retired: bool) -> dict:
        text = soup.get_text(" ", strip=True)

        def _find(pattern: str):
            m = re.search(pattern, text, re.I)
            return m.group(1).strip() if m else None

        # Parse import_year from horse_id: HK_{year}_{code}
        parts      = horse_id.split("_")
        import_year = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None

        return {
            "horse_id":         horse_id,
            "horse_name_en":    _find(r"([A-Z][A-Z\s']+)\s+(?:Colour|Country)"),
            "import_year":      import_year,
            "country_of_origin":_find(r"Country of Origin\s*:?\s*([A-Z]{2,3})"),
            "colour":           _find(r"Colour\s*:?\s*(\w+)"),
            "sex":              _find(r"Sex\s*:?\s*(\w+)"),
            "import_type":      _find(r"Type\s*:?\s*(\w+)"),
            "total_stakes_hkd": _safe_float(_find(r"Total Prize\s*[:\$]?\s*([\d,]+)")),
            "last_rating":      _safe_int(_find(r"Current Rating\s*:?\s*(\d+)")),
            "sire":             _find(r"Sire\s*:?\s*([A-Z][A-Z\s']+?)(?:\s+Dam|$)"),
            "dam":              _find(r"Dam\s*:?\s*([A-Z][A-Z\s']+?)(?:\s+Dam's Sire|$)"),
            "dam_sire":         _find(r"Dam's Sire\s*:?\s*([A-Z][A-Z\s']+?)(?:\s|$)"),
            "is_retired":       is_retired,
            "last_scraped_date":pd.Timestamp.today().date().isoformat(),
        }


def _safe_int(val):
    try:
        return int(str(val).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _safe_float(val):
    try:
        return float(str(val).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None