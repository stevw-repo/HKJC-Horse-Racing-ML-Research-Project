"""Parser for the HKJC fixtures calendar — historical meeting-date enumeration.

``fixture?calyear=YYYY&calmonth=MM`` returns a month grid. Meeting days are
``<td class="calendar">`` cells with the day-of-month in a leading ``<span>``; non-meeting
days are bare ``<td>``. This is the authoritative way to enumerate meeting dates back to
~2006 — the results ``selectId`` dropdown only lists the most recent ~2 seasons.
"""

from __future__ import annotations

from selectolax.parser import HTMLParser


def parse_fixture_meeting_days(html: str) -> list[int]:
    """Return the day-of-month numbers that have a race meeting in a fixtures month page."""
    tree = HTMLParser(html)
    days: set[int] = set()
    for cell in tree.css("td"):
        if "calendar" not in (cell.attributes.get("class") or "").split():
            continue
        span = cell.css_first("span")
        text = span.text(strip=True) if span is not None else ""
        if text.isdigit():
            days.add(int(text))
    return sorted(days)
