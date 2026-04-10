# scrapers/jockey_trainer_scraper.py
import datetime
import logging
import re

import pandas as pd

from config import HORSE_BASE_URL, RAW_HORSES_DIR
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import write

logger = logging.getLogger(__name__)

_RANKING_URLS = {
    "jockey":  f"{HORSE_BASE_URL}/jockey-ranking",
    "trainer": f"{HORSE_BASE_URL}/trainer-ranking",
}


class JockeyTrainerScraper(BaseScraper):
    """Scrape season-to-date jockey and trainer ranking snapshots."""

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        today   = datetime.date.today()
        rows_j  = self._scrape_rankings("jockey", today)
        rows_t  = self._scrape_rankings("trainer", today)

        if rows_j or rows_t:
            df      = pd.DataFrame(rows_j + rows_t)
            tag     = today.isoformat().replace("-", "")
            write(df, RAW_HORSES_DIR / f"jockey_trainer_stats_{tag}.parquet")
            logger.info("Saved jockey/trainer stats — %d rows.", len(df))

    def _scrape_rankings(self, role: str, snapshot_date: datetime.date) -> list:
        url  = _RANKING_URLS[role]
        try:
            soup = self.get_html(url)
        except Exception as exc:
            logger.warning("Could not fetch %s rankings: %s", role, exc)
            return []

        rows  = []
        table = soup.find("table")
        if not table:
            return []
        headers = [th.get_text(strip=True).lower()
                   for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells:
                continue
            row = {"role": role, "snapshot_date": snapshot_date.isoformat()}
            for i, col in enumerate(headers):
                row[col] = cells[i] if i < len(cells) else None
            rows.append(row)
        return rows