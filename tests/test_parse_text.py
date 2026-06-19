"""Offline tests for the race-text parsers (source #9)."""

from __future__ import annotations

from pathlib import Path

from hkjc.data.parse.text import parse_comments_on_running, parse_race_text

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
CORUNNING = (FIXTURES / "corunning_2026_06_13_ST_R1.html").read_text(encoding="utf-8")


def test_parse_comments_on_running() -> None:
    comments = parse_comments_on_running(CORUNNING)
    assert len(comments) == 7  # 7 runners in this race
    winner = next(c for c in comments if c.placing == 1)
    assert winner.saddle == 6
    assert winner.horse_id == "HK_2025_L441"
    assert winner.comment.startswith("Shifted in at start")
    # Every parsed comment is non-empty text.
    assert all(c.comment for c in comments)


def test_parse_race_text_extracts_blob() -> None:
    html = (
        '<html><body><table class="table_bd">'
        "<tr><td>Pla.</td><td>HorseNo</td><td>Incident</td></tr>"
        "<tr><td>2</td><td>1</td><td>Near the 800M was steadied to avoid another runner.</td></tr>"
        "</table><script>ignore()</script></body></html>"
    )
    blobs = parse_race_text(html, "racereportfull")
    assert len(blobs) == 1
    assert blobs[0].source == "racereportfull"
    assert "steadied to avoid" in blobs[0].text
    assert "ignore()" not in blobs[0].text  # script stripped
