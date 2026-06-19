"""Canonical identifiers (PLAN.md §4).

``race_key = (race_date, venue in {ST, HV}, race_no)``; ``horse_id = HK_YYYY_XXXX``;
jockey/trainer codes are short upstream strings used as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

VENUES: tuple[str, ...] = ("ST", "HV")
MAX_RACE_NO = 14  # HK cards run up to ~11-12 races; allow headroom.

HORSE_ID_RE = re.compile(r"^HK_\d{4}_[A-Z0-9]{4}$")


@dataclass(frozen=True, slots=True)
class RaceKey:
    """A meeting-race identifier: date + venue + race number."""

    race_date: date
    venue: str
    race_no: int

    def __post_init__(self) -> None:
        if self.venue not in VENUES:
            msg = f"venue must be one of {VENUES}, got {self.venue!r}"
            raise ValueError(msg)
        if not 1 <= self.race_no <= MAX_RACE_NO:
            msg = f"race_no must be in 1..{MAX_RACE_NO}, got {self.race_no}"
            raise ValueError(msg)

    def as_str(self) -> str:
        """Stable string form, e.g. ``2026-06-21_ST_R1``."""
        return f"{self.race_date:%Y-%m-%d}_{self.venue}_R{self.race_no}"


def is_valid_horse_id(value: str) -> bool:
    """Return True if ``value`` matches the HKJC horse-id format ``HK_YYYY_XXXX``."""
    return bool(HORSE_ID_RE.match(value))
