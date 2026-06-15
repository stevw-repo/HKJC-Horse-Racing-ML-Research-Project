"""Tests for the gov.hk public-holiday parser and storage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from hkjc.data.holidays import parse_holidays
from hkjc.data.store.writer import refresh_views, write_holidays

JSON = """
{"vcalendar": [{"vevent": [
  {"dtstart": ["20250101", {"value": "DATE"}], "summary": "The first day of January"},
  {"dtstart": ["20250129", {"value": "DATE"}], "summary": "Lunar New Year"},
  {"summary": "missing dtstart"},
  {"dtstart": ["bad"], "summary": "unparseable date"}
]}]}
"""


def test_parse_holidays_skips_invalid() -> None:
    holidays = parse_holidays("﻿" + JSON)  # the live feed is served with a UTF-8 BOM
    assert len(holidays) == 2  # the two valid events; missing/unparseable dropped
    assert holidays[0].date == date(2025, 1, 1)
    assert holidays[0].name == "The first day of January"
    assert holidays[1].date == date(2025, 1, 29)


def test_write_holidays_and_view(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    assert write_holidays(raw, parse_holidays(JSON)) == 2
    con = duckdb.connect()
    try:
        refresh_views(con, raw)
        count = con.execute("SELECT count(*) FROM public_holidays").fetchone()
        assert count is not None
        assert count[0] == 2
        name = con.execute("SELECT name FROM public_holidays WHERE date = '2025-01-01'").fetchone()
        assert name is not None
        assert name[0] == "The first day of January"
    finally:
        con.close()
