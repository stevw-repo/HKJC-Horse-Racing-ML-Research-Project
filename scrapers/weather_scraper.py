# scrapers/weather_scraper.py
import datetime
import logging

import pandas as pd
import requests

from config import RAW_WEATHER_DIR, WEATHER_API_URL
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import write

logger = logging.getLogger(__name__)


class WeatherScraper(BaseScraper):
    """Fetch daily weather summaries from the HK Observatory Open Data API."""

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        race_date = start_date or datetime.date.today().isoformat()
        try:
            row = self._fetch_weather(race_date)
            if row:
                df  = pd.DataFrame([row])
                tag = race_date.replace("-", "")
                write(df, RAW_WEATHER_DIR / f"weather_{tag}.parquet")
                logger.info("Weather saved for %s.", race_date)
        except Exception as exc:
            logger.warning("Weather fetch failed %s: %s", race_date, exc)

    def _fetch_weather(self, race_date: str) -> dict | None:
        params = {"dataType": "DailyExtract", "lang": "en", "rformat": "json"}
        try:
            resp = self._get_with_retry(WEATHER_API_URL, params=params)
            data = resp.json()
        except Exception as exc:
            logger.warning("Weather API error: %s", exc)
            return None

        record = data.get("data") or {}
        return {
            "race_date":    race_date,
            "temp_max_c":   record.get("dailyMax", {}).get("value"),
            "temp_min_c":   record.get("dailyMin", {}).get("value"),
            "humidity_pct": record.get("dailyMeanRH", {}).get("value"),
            "rainfall_mm":  record.get("dailyRainfall", {}).get("value"),
        }