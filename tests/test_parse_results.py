"""Offline regression tests for the results parser, against checked-in fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hkjc.data.parse.results import (
    count_races,
    parse_meeting_dates,
    parse_race_result,
    parse_venue,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
RACE_HTML = (FIXTURES / "localresults_2026-06-03_HV_R1.html").read_text(encoding="utf-8")
MEETING_HTML = (FIXTURES / "resultsall_2026-06-03_HV.html").read_text(encoding="utf-8")


def test_parse_meeting_dates() -> None:
    dates = parse_meeting_dates(RACE_HTML)
    assert date(2026, 6, 3) in dates
    assert len(dates) > 100  # dropdown spans ~2 seasons
    assert all(isinstance(d, date) for d in dates)


def test_meeting_venue_and_race_count() -> None:
    assert parse_venue(MEETING_HTML) == "HV"
    assert count_races(MEETING_HTML) == 9


def test_parse_race_meta() -> None:
    race = parse_race_result(RACE_HTML, date(2026, 6, 3), "HV", 1)
    assert race.race_no == 1
    assert race.distance_m == 1650
    assert race.race_class == "Class 5"
    assert race.going == "GOOD TO FIRM"
    assert race.surface == "TURF"
    assert race.rail == "C"  # draw-bias input (#3), parsed from the course token
    assert race.prize_hkd == 875000
    assert race.final_time_s is not None and abs(race.final_time_s - 98.85) < 0.01


def test_parse_race_winner() -> None:
    race = parse_race_result(RACE_HTML, date(2026, 6, 3), "HV", 1)
    assert len(race.runners) == 12
    winner = race.runners[0]
    assert winner.finish_pos == 1
    assert winner.saddle == 3
    assert winner.horse_id == "HK_2022_H447"
    assert winner.horse_name == "FAMILY FORTUNE"
    assert winner.jockey_code == "MOJ"
    assert winner.trainer_code == "FC"
    assert winner.actual_weight == 133
    assert winner.draw == 1
    assert winner.win_odds == 2.7  # SP = market closing line
    assert winner.finish_time_s is not None and abs(winner.finish_time_s - 98.85) < 0.01


def test_parse_dividends_full_pool_table() -> None:
    race = parse_race_result(RACE_HTML, date(2026, 6, 3), "HV", 1)
    pools = {d.pool for d in race.dividends}
    assert {"WIN", "PLACE", "QUINELLA", "TRIO", "QUARTET"} <= pools
    win = next(d for d in race.dividends if d.pool == "WIN")
    assert win.combination == "3"
    assert win.dividend == 27.0
    # PLACE pays the placed horses (carry-forward pool label across rows).
    place = [d for d in race.dividends if d.pool == "PLACE"]
    assert len(place) == 3
