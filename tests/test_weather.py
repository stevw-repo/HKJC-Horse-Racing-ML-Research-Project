"""Tests for HKO daily-climate parsing and weather storage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from hkjc.data.models import WeatherDaily
from hkjc.data.store.writer import refresh_views, write_weather
from hkjc.data.weather.hko import VENUE_STATION, parse_climate_csv, weather_url

CSV = (
    '﻿"Daily Mean Temperature (C) at the Hong Kong Observatory"\n'
    '"bilingual title"\n'
    "Year,Month,Day,Value,Completeness\n"
    "2026,5,1,24.0,C\n"
    "2026,5,2,25.2,C\n"
    "2026,5,3,***,*\n"  # unavailable -> skipped
    '"*** unavailable"\n'
    '"C data Complete"\n'
)


def test_parse_climate_csv() -> None:
    parsed = parse_climate_csv(CSV)
    assert parsed[date(2026, 5, 1)] == 24.0
    assert parsed[date(2026, 5, 2)] == 25.2
    assert date(2026, 5, 3) not in parsed  # "***" unavailable is dropped
    assert len(parsed) == 2


def test_weather_url_and_station_mapping() -> None:
    assert VENUE_STATION == {"HV": "HKO", "ST": "SHA"}
    url = weather_url("https://data.weather.gov.hk/weatherAPI/opendata", "SHA", "CLMMAXT")
    assert "dataType=CLMMAXT" in url
    assert "station=SHA" in url


def test_write_weather_and_view(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    n = write_weather(
        raw,
        [
            WeatherDaily(
                date=date(2026, 5, 1),
                station="SHA",
                venue="ST",
                mean_temp=23.2,
                max_temp=28.0,
                min_temp=20.0,
            ),
            WeatherDaily(date=date(2026, 5, 1), station="HKO", venue="HV", mean_temp=24.0),
        ],
    )
    assert n == 2
    con = duckdb.connect()
    try:
        refresh_views(con, raw)
        row = con.execute("SELECT mean_temp FROM weather WHERE venue = 'ST'").fetchone()
        assert row is not None
        assert row[0] == 23.2
    finally:
        con.close()
