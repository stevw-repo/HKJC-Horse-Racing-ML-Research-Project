"""Offline regression tests for the horse profile parser, against a checked-in fixture."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hkjc.data.parse.profiles import parse_horse_profile

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
HTML = (FIXTURES / "horse_HK_2022_H447.html").read_text(encoding="utf-8")


def test_horse_bio_locked_block() -> None:
    p = parse_horse_profile(HTML, "HK_2022_H447")
    assert p.name == "FAMILY FORTUNE"
    assert p.brand == "H447"
    assert p.country_of_origin == "NZ"
    assert p.age == 5
    assert p.colour == "Bay"
    assert p.sex == "Gelding"
    assert p.import_type == "PPG"
    assert p.sire == "Derryn"
    assert p.dam == "Waiana Gold"
    assert p.dams_sire == "Gold Centre"
    assert p.owner is not None and p.owner.startswith("Tony Stradmoor")
    assert p.current_rating == 43
    assert p.season_start_rating == 27
    assert p.season_stakes == 1885625
    assert p.total_stakes == 2205000


def test_horse_form_records() -> None:
    p = parse_horse_profile(HTML, "HK_2022_H447")
    assert len(p.form) >= 10
    latest = p.form[0]
    assert latest.race_index == 745  # joins to RaceResult.race_index
    assert latest.finish_pos == 1
    assert latest.run_date == date(2026, 6, 3)
    assert latest.venue == "HV"
    assert latest.track == "Turf"
    assert latest.course == "C"
    assert latest.distance_m == 1650
    assert latest.going == "GF"
    assert latest.race_class == "5"
    assert latest.draw == 1
    assert latest.rating == 37  # rating at the time of that run (not the current 43)
    assert latest.rating != p.current_rating
    assert latest.win_odds == 2.7
    assert latest.actual_weight == 133
    assert latest.declared_weight == 995
    assert latest.gear == "TT"
    assert latest.finish_time_s is not None and abs(latest.finish_time_s - 98.85) < 0.01
