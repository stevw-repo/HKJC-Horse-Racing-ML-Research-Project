"""Offline test for the fixtures-calendar meeting-day enumerator."""

from __future__ import annotations

from pathlib import Path

from hkjc.data.parse.fixtures import parse_fixture_meeting_days

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
HTML = (FIXTURES / "fixture_2010-12.html").read_text(encoding="utf-8")


def test_parse_fixture_meeting_days() -> None:
    # December 2010 raced on these days (verified against the live calendar).
    assert parse_fixture_meeting_days(HTML) == [1, 4, 8, 12, 15, 19, 23, 27]
