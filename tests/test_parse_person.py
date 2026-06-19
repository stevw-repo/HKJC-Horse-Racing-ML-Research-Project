"""Offline regression tests for the jockey/trainer profile parser."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hkjc.data.parse.profiles import parse_person_profile

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
JOCKEY = (FIXTURES / "jockey_MOJ.html").read_text(encoding="utf-8")
TRAINER = (FIXTURES / "trainer_FC.html").read_text(encoding="utf-8")


def test_jockey_profile() -> None:
    p = parse_person_profile(JOCKEY, "MOJ", "jockey")
    assert p.role == "jockey"
    assert p.code == "MOJ"
    assert p.name == "Joao Moreira"
    assert p.age == 42
    assert p.nationality == "BRZ"
    assert p.season == "25/26"
    assert p.up_to_date == date(2026, 6, 13)
    assert p.wins == 23
    assert p.seconds == 10
    assert p.thirds == 9
    assert p.fourths == 7
    assert p.total_starts == 139  # rides
    assert p.win_pct == 16.55
    assert p.stakes == 29200683
    assert p.wins_last10 == 13


def test_trainer_profile() -> None:
    p = parse_person_profile(TRAINER, "FC", "trainer")
    assert p.role == "trainer"
    assert p.code == "FC"
    assert p.name == "Caspar Fownes"
    assert p.age == 58
    assert p.season == "25/26"
    assert p.wins == 62
    assert p.seconds == 41
    assert p.thirds == 40
    assert p.total_starts == 517  # runners
    assert p.win_pct == 11.99
    assert p.stakes == 77022930
