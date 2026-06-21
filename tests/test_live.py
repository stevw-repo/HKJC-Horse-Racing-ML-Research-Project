"""Offline tests for the live GraphQL layer (M7), using captured fixtures (CI-safe)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from hkjc.data.live.graphql import CARD_QUERY, ODDS_QUERY, parse_card, parse_odds
from hkjc.data.live.raceday import card_to_spine

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"


def _meeting(name: str) -> dict[str, Any]:
    raw: Any = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    meeting: dict[str, Any] = raw["data"]["raceMeetings"][0]
    return meeting


def test_registered_queries_are_racing_operations() -> None:
    # The gateway whitelists exact query strings; both must stay the captured `racing` ops.
    assert CARD_QUERY.startswith("query racing(") and "raceMeetings" in CARD_QUERY
    assert "pmPools" in ODDS_QUERY and "oddsNodes" in ODDS_QUERY


def test_parse_card() -> None:
    card = parse_card(_meeting("graphql_card_ST_2026_06_21.json"), date(2026, 6, 21), "ST")
    assert card.status == "DEFINED"
    assert card.venue == "ST"
    assert len(card.races) == 11
    runner = card.races[0].runners[0]
    assert runner.horse_id is not None and runner.horse_id.startswith("HK_")  # canonical id
    assert runner.saddle == 1
    assert runner.jockey_code and runner.trainer_code


def test_parse_odds() -> None:
    pools = parse_odds(_meeting("graphql_odds_S5_live.json")["pmPools"], race_no=4)
    assert {p.pool_type for p in pools} == {"WIN", "PLA"}
    win = next(p for p in pools if p.pool_type == "WIN")
    assert win.last_update_time  # the dedup key
    node = win.nodes[0]
    assert node.saddle is not None and node.odds is not None


def test_card_to_spine_keeps_declared_runners() -> None:
    card = parse_card(_meeting("graphql_card_ST_2026_06_21.json"), date(2026, 6, 21), "ST")
    spine = card_to_spine(card)
    assert spine.height > 0
    assert {"race_date", "venue", "race_no", "saddle", "horse_id", "_card_rating"} <= set(
        spine.columns
    )
    declared = sum(
        1 for r in card.races for rr in r.runners if (rr.status or "").lower() == "declared"
    )
    assert spine.height == declared
