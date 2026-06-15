"""Shared parsing helpers (text cleaning, number/time parsing, id extraction)."""

from __future__ import annotations

import re
from datetime import date

from selectolax.parser import Node

_ID_PARAM_RE = {
    "horseid": re.compile(r"horseid=([A-Za-z0-9_]+)", re.IGNORECASE),
    "jockeyid": re.compile(r"jockeyid=([A-Za-z0-9_]+)", re.IGNORECASE),
    "trainerid": re.compile(r"trainerid=([A-Za-z0-9_]+)", re.IGNORECASE),
}
_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2})\.(\d{1,2})$")
_TIME_DOTS_RE = re.compile(r"^(\d+)\.(\d{2})\.(\d{1,2})$")  # form-table form, e.g. 1.38.85
_DATE_DMY_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_DATE_DMY2_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$")  # 2-digit year, e.g. 03/06/26


def clean(text: str | None) -> str:
    """Collapse whitespace (incl. non-breaking spaces) and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def id_from_node(node: Node, param: str) -> str | None:
    """Extract an HKJC id (e.g. ``horseid``) from the first anchor under ``node``."""
    pattern = _ID_PARAM_RE[param]
    for anchor in node.css("a"):
        href = anchor.attributes.get("href") or ""
        match = pattern.search(href)
        if match:
            return match.group(1)
    return None


def to_int(text: str | None) -> int | None:
    """Parse an integer, tolerating commas/spaces; return None if not numeric."""
    cleaned = clean(text).replace(",", "")
    match = re.search(r"-?\d+", cleaned)
    return int(match.group()) if match else None


def to_float(text: str | None) -> float | None:
    """Parse a float, tolerating thousands separators; return None if not numeric."""
    cleaned = clean(text).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_time_to_seconds(text: str | None) -> float | None:
    """Parse a race time into seconds: ``1:38.85``, ``58.50``, or ``1.38.85`` (form table)."""
    cleaned = clean(text)
    match = _TIME_RE.match(cleaned)
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        return minutes * 60 + int(match.group(2)) + int(match.group(3).ljust(2, "0")) / 100
    dots = _TIME_DOTS_RE.match(cleaned)
    if dots:
        return int(dots.group(1)) * 60 + int(dots.group(2)) + int(dots.group(3).ljust(2, "0")) / 100
    return None


def parse_dmy(text: str) -> date | None:
    """Parse a ``DD/MM/YYYY`` date (HKJC dropdown format)."""
    match = _DATE_DMY_RE.match(clean(text))
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    return date(year, month, day)


def parse_dmy2(text: str) -> date | None:
    """Parse a ``DD/MM/YY`` date (form-table format); 2-digit year is 2000+YY."""
    match = _DATE_DMY2_RE.match(clean(text))
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    return date(2000 + year, month, day)
