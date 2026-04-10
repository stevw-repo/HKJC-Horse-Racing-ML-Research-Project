# scrapers/odds_scraper.py
import datetime
import logging

import pandas as pd

from config import RAW_ODDS_DIR, VALID_POOL_TYPES, VALID_VENUES
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import write

logger = logging.getLogger(__name__)

# Only query the four in-scope pool types
_GQL_ODDS = """
query raceOdds($date: String!, $venueCode: String!, $raceNo: Int!) {
  raceOdds(input: {
    date: $date,
    venueCode: $venueCode,
    raceNo: $raceNo,
    oddsTypes: ["WIN", "PLA", "QIN", "QPL"]
  }) {
    poolType
    poolInvestment
    oddsNodes {
      combString
      oddsValue
      hotFavourite
      oddsDropValue
    }
  }
}
"""

_GQL_WIN_ODDS = """
query raceMeetingWinOdds($date: String!, $venueCode: String!) {
  raceMeetings(input: { date: $date, venueCode: $venueCode }) {
    venueCode
    raceDate
    races {
      raceNo
      runners {
        horseNo
        winOdds
      }
    }
  }
}
"""


class OddsScraper(BaseScraper):
    """Scrape WIN, PLA, QIN, QPL odds for a given meeting date."""

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        race_date = start_date or datetime.date.today().isoformat()
        rows      = []
        for venue in sorted(VALID_VENUES):
            for race_no in range(1, 13):
                try:
                    race_rows = self._fetch_race_odds(race_date, venue, race_no)
                    if not race_rows:
                        break
                    rows.extend(race_rows)
                except Exception as exc:
                    logger.debug("Odds %s %s R%d: %s", venue, race_date, race_no, exc)
                    break

        if rows:
            df  = pd.DataFrame(rows)
            # Keep only valid pool types
            df  = df[df["pool_type"].isin(VALID_POOL_TYPES)]
            tag = race_date.replace("-", "")
            fp  = RAW_ODDS_DIR / f"odds_{tag}.parquet"
            write(df, fp)
            logger.info("Odds saved — %d rows for %s.", len(df), race_date)

    def _fetch_race_odds(self, race_date: str, venue: str, race_no: int) -> list:
        data    = self.post_graphql(
            _GQL_ODDS,
            {"date": race_date, "venueCode": venue, "raceNo": race_no}
        )
        pools   = (data.get("data") or {}).get("raceOdds") or []
        now_utc = pd.Timestamp.utcnow()
        rows    = []
        for pool in pools:
            pool_type = pool.get("poolType", "")
            if pool_type not in VALID_POOL_TYPES:
                continue
            for node in pool.get("oddsNodes") or []:
                rows.append({
                    "race_date":           race_date,
                    "venue":               venue,
                    "race_no":             race_no,
                    "pool_type":           pool_type,
                    "comb_string":         node.get("combString"),
                    "odds_value":          _safe_float(node.get("oddsValue")),
                    "hot_favourite":       bool(node.get("hotFavourite", False)),
                    "odds_drop_value":     _safe_float(node.get("oddsDropValue")),
                    "pool_investment_hkd": _safe_float(pool.get("poolInvestment")),
                    "snapshot_time":       now_utc,
                })
        return rows


def _safe_float(val):
    try:
        return float(str(val).replace(",", ""))
    except (TypeError, ValueError):
        return None