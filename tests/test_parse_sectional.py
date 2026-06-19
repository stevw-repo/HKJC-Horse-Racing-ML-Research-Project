"""Offline regression test for the sectional-time parser (source #7)."""

from __future__ import annotations

from pathlib import Path

from hkjc.data.parse.sectionals import parse_race_sectionals

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
HTML = (FIXTURES / "sectional_2026_06_13_ST_R1.html").read_text(encoding="utf-8")


def test_parses_runner_sections() -> None:
    rows = parse_race_sectionals(HTML)
    assert len(rows) == 21  # 7 runners x 3 timed sections (a sprint)
    assert len({r.saddle for r in rows}) == 7

    winner = [r for r in rows if r.finishing_order == 1]
    winner.sort(key=lambda r: r.section_index)
    assert [r.section_index for r in winner] == [1, 2, 3]
    assert winner[0].horse_id == "HK_2025_L441"
    assert winner[0].saddle == 6
    assert winner[0].running_position == 1  # led from the start
    # Bold value is the section time; first section has no 200m split.
    assert winner[0].section_time_s is not None and abs(winner[0].section_time_s - 13.37) < 1e-6
    assert winner[0].split_200m_s is None
    assert winner[1].section_time_s is not None and abs(winner[1].section_time_s - 20.84) < 1e-6
    assert winner[1].split_200m_s is not None and abs(winner[1].split_200m_s - 10.37) < 1e-6


def test_section_times_sum_to_final() -> None:
    rows = parse_race_sectionals(HTML)
    by_runner: dict[int, list[float]] = {}
    finals: dict[int, float] = {}
    for r in rows:
        if r.section_time_s is not None:
            by_runner.setdefault(r.saddle, []).append(r.section_time_s)
        if r.final_time_s is not None:
            finals[r.saddle] = r.final_time_s
    for saddle, sections in by_runner.items():
        assert abs(sum(sections) - finals[saddle]) < 0.05  # sections partition the race
