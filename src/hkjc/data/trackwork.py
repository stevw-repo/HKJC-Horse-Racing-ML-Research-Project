"""HKJC trackwork (#5): a JSON endpoint discovered via the browser.

The ``trackworksearch`` page's "Search" loads ``trackworkonedayresult``, which fetches
``/racing/information/json/TrackworkOneDayRecords/<YYYYMMDD>1E.aspx?PageNum=N`` (``1E`` =
session 1, English) returning ``{next, Records:[{Horse,Trainer,Type,Racecourse_Track,
Workouts,Gear}]}``, paginated for infinite scroll. The records carry names, not ids.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from selectolax.parser import HTMLParser

from hkjc.data.models import TrackworkRecord
from hkjc.data.parse.common import clean, parse_dmy_flex

SESSION_SUFFIX = "1E"  # session 1, English


def racing_origin(base_url: str) -> str:
    """Scheme+host of the racing site, e.g. ``https://racing.hkjc.com``."""
    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}"


def trackwork_key(day: date) -> str:
    return f"{day:%Y%m%d}{SESSION_SUFFIX}"


def trackwork_url(origin: str, day: date, page: int) -> str:
    key = trackwork_key(day)
    return f"{origin}/racing/information/json/TrackworkOneDayRecords/{key}.aspx?PageNum={page}"


def parse_trackwork_dates(html: str) -> list[date]:
    """Return available trackwork dates from the ``trackworksearch`` dropdown."""
    tree = HTMLParser(html)
    dates = [parse_dmy_flex(clean(option.text())) for option in tree.css("option")]
    return sorted({d for d in dates if d is not None})


def parse_trackwork_page(json_text: str, work_date: date) -> tuple[int, list[TrackworkRecord]]:
    """Parse one JSON page into ``(next_page, records)``; ``next_page`` 0 means no more."""
    data: Any = json.loads(json_text)
    next_page = int(data.get("next") or 0) if isinstance(data, dict) else 0
    raw_records = data.get("Records", []) if isinstance(data, dict) else []
    records: list[TrackworkRecord] = []
    for row in raw_records:
        name = clean(str(row.get("Horse", "")))
        if not name:
            continue
        records.append(
            TrackworkRecord(
                work_date=work_date,
                horse_name=name,
                trainer=clean(str(row.get("Trainer", ""))) or None,
                work_type=clean(str(row.get("Type", ""))) or None,
                racecourse_track=clean(str(row.get("Racecourse_Track", ""))) or None,
                workouts=clean(str(row.get("Workouts", ""))) or None,
                gear=clean(str(row.get("Gear", ""))) or None,
            )
        )
    return next_page, records
