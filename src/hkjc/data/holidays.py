"""HK public-holiday calendar (gov.hk open data, source #14).

The feed (``.../common/ical/en.json``) is iCalendar-as-JSON:
``vcalendar[0].vevent[]`` with ``dtstart = ["YYYYMMDD", {...}]`` and ``summary``. It spans
roughly the current +/- ~1 year, so very old seasons are not covered.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from hkjc.data.models import PublicHoliday


def parse_holidays(text: str) -> list[PublicHoliday]:
    """Parse the gov.hk iCal-JSON holiday feed into ``PublicHoliday`` records."""
    data: Any = json.loads(text.lstrip("﻿"))  # the gov.hk feed is served with a BOM
    calendars = data.get("vcalendar") if isinstance(data, dict) else None
    events = calendars[0].get("vevent", []) if calendars else []
    holidays: list[PublicHoliday] = []
    for event in events:
        dtstart = event.get("dtstart")
        summary = event.get("summary")
        if not dtstart or not summary:
            continue
        raw = dtstart[0] if isinstance(dtstart, list) else dtstart
        try:
            day = date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
        except (ValueError, TypeError, IndexError):
            continue
        holidays.append(PublicHoliday(date=day, name=str(summary)))
    return holidays
