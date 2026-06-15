"""Tests for the store layer: season labels, Parquet write + DuckDB views, manifest."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from hkjc.data.models import (
    Dividend,
    HorseFormRun,
    HorseProfile,
    MeetingResults,
    RaceResult,
    RunnerResult,
)
from hkjc.data.store.manifest import Manifest
from hkjc.data.store.writer import (
    refresh_views,
    season_label,
    write_horse_profile,
    write_meeting,
)


def _scalar(con: Any, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row[0]


def _meeting() -> MeetingResults:
    race = RaceResult(
        race_date=date(2026, 6, 3),
        venue="HV",
        race_no=1,
        distance_m=1650,
        race_class="Class 5",
        going="GOOD TO FIRM",
        surface="TURF",
        final_time_s=98.85,
        runners=[
            RunnerResult(
                finish_pos=1,
                finish_pos_raw="1",
                saddle=3,
                horse_id="HK_2022_H447",
                horse_name="FAMILY FORTUNE",
                jockey_code="MOJ",
                trainer_code="FC",
                actual_weight=133,
                declared_weight=995,
                draw=1,
                lbw_raw="---",
                running_position_raw="7761",
                finish_time_s=98.85,
                win_odds=2.7,
            ),
            RunnerResult(
                finish_pos=2,
                finish_pos_raw="2",
                saddle=12,
                horse_id="HK_2023_J493",
                horse_name="THE WAY WE WIN",
                jockey_code="MHT",
                trainer_code="RW",
                actual_weight=113,
                declared_weight=1076,
                draw=6,
                lbw_raw="N",
                running_position_raw="1112",
                finish_time_s=98.90,
                win_odds=38.0,
            ),
        ],
        dividends=[
            Dividend(pool="WIN", combination="3", dividend=27.0),
            Dividend(pool="PLACE", combination="3", dividend=13.0),
        ],
    )
    return MeetingResults(race_date=date(2026, 6, 3), venue="HV", races=[race])


def test_season_label() -> None:
    assert season_label(date(2026, 6, 3)) == "2025-26"
    assert season_label(date(2025, 9, 1)) == "2025-26"
    assert season_label(date(2025, 8, 31)) == "2024-25"


def test_write_meeting_partitions_and_views(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    counts = write_meeting(raw, _meeting())
    assert counts == {"races": 1, "results": 2, "dividends": 2}
    expected = raw / "races" / "season=2025-26" / "venue=HV" / "date=2026-06-03" / "races.parquet"
    assert expected.is_file()

    con = duckdb.connect()
    try:
        refresh_views(con, raw)
        assert _scalar(con, "SELECT count(*) FROM results") == 2
        assert _scalar(con, "SELECT count(*) FROM races") == 1
        assert _scalar(con, "SELECT dividend FROM dividends WHERE pool = 'WIN'") == 27.0
        assert _scalar(con, "SELECT win_odds FROM results WHERE horse_id = 'HK_2022_H447'") == 2.7
    finally:
        con.close()


def test_write_horse_profile_and_views(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    profile = HorseProfile(
        horse_id="HK_2022_H447",
        name="FAMILY FORTUNE",
        brand="H447",
        country_of_origin="NZ",
        age=5,
        colour="Bay",
        sex="Gelding",
        sire="Derryn",
        dam="Waiana Gold",
        dams_sire="Gold Centre",
        current_rating=43,
        season_start_rating=27,
        form=[
            HorseFormRun(
                race_index=745,
                finish_pos=1,
                finish_pos_raw="1",
                run_date=date(2026, 6, 3),
                venue="HV",
                track="Turf",
                course="C",
                distance_m=1650,
                going="GF",
                race_class="5",
                draw=1,
                rating=37,
                jockey_code="MOJ",
                trainer_code="FC",
                lbw_raw="N",
                win_odds=2.7,
                actual_weight=133,
                running_position_raw="7 7 6 1",
                finish_time_s=98.85,
                declared_weight=995,
                gear="TT",
            )
        ],
    )
    assert write_horse_profile(raw, profile) == 1

    con = duckdb.connect()
    try:
        refresh_views(con, raw)
        assert _scalar(con, "SELECT current_rating FROM horses") == 43
        assert _scalar(con, "SELECT rating FROM horse_form WHERE race_index = 745") == 37
    finally:
        con.close()


def test_manifest_idempotency(tmp_path: Path) -> None:
    with Manifest(tmp_path / "hkjc.duckdb") as manifest:
        assert not manifest.has("u1")
        manifest.record("u1", "localresults", "hash1", 200, 12)
        assert manifest.has("u1")
        assert manifest.count() == 1
        # Re-recording the same URL upserts, never duplicates.
        manifest.record("u1", "localresults", "hash2", 200, 13)
        assert manifest.count() == 1
