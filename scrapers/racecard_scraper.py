# scrapers/racecard_scraper.py
import datetime
import logging

import pandas as pd

from config import RAW_RACECARDS_DIR, VALID_VENUES
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import write

logger = logging.getLogger(__name__)

_GQL_RACE_MEETINGS = """
query raceMeetings($date: String!, $venueCode: String!) {
  raceMeetings(input: { date: $date, venueCode: $venueCode }) {
    venueCode
    raceDate
    races {
      raceNo
      raceNameEn
      distance
      raceClass
      going
      courseType
      postTime
      runners {
        horseNo
        horseNameEn
        horseId
        jockeyCode
        trainerCode
        barrierDraw
        handicapWeight
        currentRating
        gearInfo
        allowance
        last6RunEn
      }
    }
  }
}
"""


class RacecardScraper(BaseScraper):
    """Fetch the runner list for a meeting using the HKJC GraphQL API."""

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        race_date = start_date or datetime.date.today().isoformat()
        rows = []
        for venue in sorted(VALID_VENUES):
            try:
                rows.extend(self._fetch_meeting(race_date, venue))
            except Exception as exc:
                logger.warning("Racecard failed %s %s: %s", venue, race_date, exc)

        if rows:
            df = pd.DataFrame(rows)
            date_tag = race_date.replace("-", "")
            write(df, RAW_RACECARDS_DIR / f"racecard_{date_tag}.parquet")
            logger.info("Racecard saved — %d runners for %s.", len(df), race_date)

    def _fetch_meeting(self, race_date: str, venue: str) -> list:
        data   = self.post_graphql(_GQL_RACE_MEETINGS,
                                   {"date": race_date, "venueCode": venue})
        mtgs   = (data.get("data") or {}).get("raceMeetings") or []
        rows   = []
        for mtg in mtgs:
            for race in mtg.get("races") or []:
                for runner in race.get("runners") or []:
                    rows.append({
                        "race_date":      race_date,
                        "venue":          mtg.get("venueCode", venue),
                        "race_no":        race.get("raceNo"),
                        "race_name_en":   race.get("raceNameEn"),
                        "distance_m":     race.get("distance"),
                        "race_class":     race.get("raceClass"),
                        "going":          race.get("going"),
                        "course_type":    race.get("courseType"),
                        "post_time":      race.get("postTime"),
                        "horse_no":       runner.get("horseNo"),
                        "horse_name_en":  runner.get("horseNameEn"),
                        "horse_id":       (runner.get("horseId") or "").upper(),
                        "jockey_code":    runner.get("jockeyCode"),
                        "trainer_code":   runner.get("trainerCode"),
                        "barrier_draw":   runner.get("barrierDraw"),
                        "handicap_weight":runner.get("handicapWeight"),
                        "current_rating": runner.get("currentRating"),
                        "gear_info":      runner.get("gearInfo"),
                        "allowance":      runner.get("allowance"),
                        "last_6_run":     runner.get("last6RunEn"),
                    })
        return rows