"""Tests for trackwork JSON parsing, endpoint helpers, and storage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from hkjc.data.models import TrackworkRecord
from hkjc.data.store.writer import refresh_views, write_trackwork
from hkjc.data.trackwork import (
    parse_trackwork_dates,
    parse_trackwork_page,
    racing_origin,
    trackwork_key,
    trackwork_url,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
PAGE_JSON = (FIXTURES / "trackwork_2026-06-13_p1.json").read_text(encoding="utf-8")


def test_parse_trackwork_page() -> None:
    next_page, records = parse_trackwork_page(PAGE_JSON, date(2026, 6, 13))
    assert next_page == 2
    assert len(records) == 50
    first = records[0]
    assert first.work_date == date(2026, 6, 13)
    assert first.horse_name == "A TIME FOR US"
    assert first.trainer == "K W Lui"
    assert first.work_type == "Trotting"
    assert first.racecourse_track == "Conghua TroR"
    assert first.workouts == "TroR TroR Canter (R.B.)"
    assert first.gear == "H"


def test_trackwork_key_and_url() -> None:
    assert trackwork_key(date(2026, 6, 13)) == "202606131E"
    origin = racing_origin("https://racing.hkjc.com/en-us/local/information")
    assert origin == "https://racing.hkjc.com"
    assert trackwork_url(origin, date(2026, 6, 13), 1) == (
        "https://racing.hkjc.com/racing/information/json/"
        "TrackworkOneDayRecords/202606131E.aspx?PageNum=1"
    )


def test_parse_trackwork_dates() -> None:
    html = (
        '<select name="OptOneDay">'
        "<option>15/6/2026</option><option>9/6/2026</option><option>foo</option>"
        "</select>"
    )
    dates = parse_trackwork_dates(html)
    assert dates == [date(2026, 6, 9), date(2026, 6, 15)]  # sorted, 'foo' dropped


def test_write_trackwork_and_view(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    n = write_trackwork(
        raw,
        date(2026, 6, 13),
        [
            TrackworkRecord(
                work_date=date(2026, 6, 13),
                horse_name="A TIME FOR US",
                trainer="K W Lui",
                work_type="Trotting",
                racecourse_track="Conghua TroR",
                workouts="TroR TroR Canter (R.B.)",
                gear="H",
            )
        ],
    )
    assert n == 1
    con = duckdb.connect()
    try:
        refresh_views(con, raw)
        row = con.execute(
            "SELECT work_type FROM trackwork WHERE horse_name = 'A TIME FOR US'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Trotting"
    finally:
        con.close()
