"""Tests for canonical identifiers."""

from __future__ import annotations

from datetime import date

import pytest

from hkjc.common.keys import RaceKey, is_valid_horse_id


def test_race_key_string_form() -> None:
    rk = RaceKey(race_date=date(2026, 6, 21), venue="ST", race_no=1)
    assert rk.as_str() == "2026-06-21_ST_R1"


def test_race_key_rejects_unknown_venue() -> None:
    with pytest.raises(ValueError, match="venue"):
        RaceKey(race_date=date(2026, 6, 21), venue="XX", race_no=1)


def test_race_key_rejects_out_of_range_race_no() -> None:
    with pytest.raises(ValueError, match="race_no"):
        RaceKey(race_date=date(2026, 6, 21), venue="HV", race_no=0)


def test_horse_id_validation() -> None:
    assert is_valid_horse_id("HK_2021_D123")
    assert not is_valid_horse_id("HK_21_D123")
    assert not is_valid_horse_id("2021_D123")
    assert not is_valid_horse_id("HK_2021_d123")  # must be uppercase
