"""Parsed shapes for the live GraphQL feed (M7).

The ``racing`` operation returns runner/odds fields as strings (``"1"``, ``"052"``, ``"8.7"``);
these models carry the cast, typed values. ``horse.id`` is already the canonical
``HK_YYYY_XXXX`` id, so live runners reconcile to the stored history with no mapping.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class CardRunner(BaseModel):
    """One declared runner on an upcoming card (GraphQL B2)."""

    saddle: int | None
    horse_id: str | None  # horse.id == HK_YYYY_XXXX
    horse_code: str | None
    name_en: str | None
    status: str | None
    draw: int | None
    handicap_weight: int | None
    current_rating: int | None
    intl_rating: int | None
    gear: str | None
    last6run: str | None
    win_odds: float | None
    jockey_code: str | None
    jockey_name: str | None
    trainer_code: str | None
    trainer_name: str | None


class RaceCard(BaseModel):
    """One race on the card."""

    race_no: int
    status: str | None
    runners: list[CardRunner]


class MeetingCard(BaseModel):
    """An upcoming meeting's full card."""

    race_date: date
    venue: str
    status: str | None
    total_investment: float | None
    races: list[RaceCard]


class OddsNode(BaseModel):
    """One combination's live odds (GraphQL B1)."""

    comb: str  # combString, e.g. "07"
    saddle: int | None
    odds: float | None
    hot_fav: bool
    odds_drop: float | None


class PoolOdds(BaseModel):
    """A WIN/PLACE pool's live odds at one refresh, keyed on ``last_update_time``."""

    race_no: int
    pool_type: str  # WIN | PLA
    status: str | None
    sell_status: str | None
    last_update_time: str | None
    nodes: list[OddsNode]
