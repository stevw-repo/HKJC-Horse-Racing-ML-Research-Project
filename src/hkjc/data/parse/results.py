"""Parser for HKJC results pages.

- :func:`parse_meeting_dates` reads the ``selectId`` dropdown (meeting enumeration).
- :func:`parse_venue` / :func:`count_races` read the meeting ``resultsall`` page.
- :func:`parse_race_result` reads a per-race ``localresults`` page into a
  :class:`~hkjc.data.models.RaceResult` (the per-race page is the only one carrying
  Win Odds, finish time, running positions, and the canonical horse/jockey/trainer ids).
"""

from __future__ import annotations

import re
from datetime import date
from typing import TypedDict

from selectolax.parser import HTMLParser, Node

from hkjc.data.models import Dividend, RaceResult, RunnerResult
from hkjc.data.parse.common import (
    clean,
    id_from_node,
    parse_dmy,
    parse_time_to_seconds,
    to_int,
)

VENUE_BY_NAME = {"happy valley": "HV", "sha tin": "ST"}
_CLASS_LINE_RE = re.compile(r"\b\d+\s*M\b", re.IGNORECASE)
_RACE_HEADER_RE = re.compile(r"RACE\s+(\d+)\s*\((\d+)\)", re.IGNORECASE)
_DISTANCE_RE = re.compile(r"(\d+)\s*M", re.IGNORECASE)
_RATING_RE = re.compile(r"\(([^)]*)\)")


class _Meta(TypedDict, total=False):
    race_index: int | None
    race_class: str | None
    distance_m: int | None
    rating_band: str | None
    race_name: str | None
    prize_hkd: int | None
    going: str | None
    course: str | None
    rail: str | None
    surface: str | None


# --------------------------------------------------------------------------- #
# Meeting enumeration helpers
# --------------------------------------------------------------------------- #
def parse_meeting_dates(html: str) -> list[date]:
    """Return all meeting dates from the results ``selectId`` dropdown."""
    tree = HTMLParser(html)
    for select in tree.css("select"):
        ident = select.attributes.get("id") or select.attributes.get("name")
        if ident == "selectId":
            dates = [parse_dmy(option.text()) for option in select.css("option")]
            return [d for d in dates if d is not None]
    return []


def parse_venue(html: str) -> str | None:
    """Return the venue code (ST/HV) from a meeting/race page header."""
    tree = HTMLParser(html)
    node = tree.css_first(".js_racecard")
    text = clean(node.text() if node else tree.text()).lower()
    for name, code in VENUE_BY_NAME.items():
        if name in text:
            return code
    return None


def count_races(html: str) -> int:
    """Return the number of races in a meeting ``resultsall`` page."""
    tree = HTMLParser(html)
    n = sum(1 for tb in tree.css("table") if "result" in (tb.attributes.get("class") or "").split())
    if n == 0:
        n = len({int(m.group(1)) for m in _RACE_HEADER_RE.finditer(tree.text())})
    return n


# --------------------------------------------------------------------------- #
# Per-race parsing
# --------------------------------------------------------------------------- #
def _canon_column(header: str) -> str:
    h = clean(header).lower()
    if h.startswith("pla"):
        return "pos"
    if h.startswith("horse no") or h.startswith("h.no") or h == "h. no":
        return "saddle"
    if h == "horse":
        return "horse"
    if h.startswith("jockey"):
        return "jockey"
    if h.startswith("trainer"):
        return "trainer"
    if h.startswith("act"):
        return "act_wt"
    if "declar" in h:
        return "declar_wt"
    if h.startswith("dr"):
        return "draw"
    if "lbw" in h:
        return "lbw"
    if "running" in h:
        return "running"
    if "finish time" in h:
        return "finish_time"
    if "win odds" in h:
        return "win_odds"
    return ""


def _cells(row: Node) -> list[Node]:
    return row.css("td,th")


def _node_text(node: Node | None) -> str:
    return clean(node.text()) if node is not None else ""


def _find_results_table(tables: list[Node]) -> Node | None:
    for table in tables:
        for row in table.css("tr"):
            joined = " ".join(clean(c.text()) for c in _cells(row))
            if "Win Odds" in joined and "Pla" in joined:
                return table
    return None


def _parse_runners(tables: list[Node]) -> list[RunnerResult]:
    table = _find_results_table(tables)
    if table is None:
        return []
    rows = table.css("tr")
    cols: dict[str, int] = {}
    data_start = 0
    for i, row in enumerate(rows):
        mapping = {_canon_column(c.text()): idx for idx, c in enumerate(_cells(row))}
        mapping.pop("", None)
        if "win_odds" in mapping and "pos" in mapping:
            cols = mapping
            data_start = i + 1
            break
    if not cols:
        return []

    runners: list[RunnerResult] = []
    for row in rows[data_start:]:
        cells = _cells(row)
        by_key: dict[str, Node] = {key: cells[idx] for key, idx in cols.items() if idx < len(cells)}

        saddle = to_int(_node_text(by_key.get("saddle")))
        horse_node = by_key.get("horse")
        horse_name = ""
        horse_id: str | None = None
        if horse_node is not None:
            anchor = horse_node.css_first("a")
            horse_name = (
                clean(anchor.text())
                if anchor
                else re.sub(r"\(.*?\)\s*$", "", _node_text(horse_node))
            )
            horse_id = id_from_node(horse_node, "horseid")
        if saddle is None or not horse_name:
            continue  # skip non-runner / spacer rows

        pos_raw = _node_text(by_key.get("pos"))
        jockey_node = by_key.get("jockey")
        trainer_node = by_key.get("trainer")
        runners.append(
            RunnerResult(
                finish_pos=to_int(pos_raw),
                finish_pos_raw=pos_raw,
                dead_heat="DH" in pos_raw.upper(),
                saddle=saddle,
                horse_id=horse_id,
                horse_name=horse_name,
                jockey_code=id_from_node(jockey_node, "jockeyid")
                if jockey_node is not None
                else None,
                trainer_code=(
                    id_from_node(trainer_node, "trainerid") if trainer_node is not None else None
                ),
                actual_weight=to_int(_node_text(by_key.get("act_wt"))),
                declared_weight=to_int(_node_text(by_key.get("declar_wt"))),
                draw=to_int(_node_text(by_key.get("draw"))),
                lbw_raw=_node_text(by_key.get("lbw")) or None,
                running_position_raw=_node_text(by_key.get("running")) or None,
                finish_time_s=parse_time_to_seconds(_node_text(by_key.get("finish_time"))),
                win_odds=_odds(_node_text(by_key.get("win_odds"))),
            )
        )
    return runners


def _odds(raw: str) -> float | None:
    cleaned = clean(raw).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_dividend_table(tables: list[Node]) -> Node | None:
    for table in tables:
        rows = table.css("tr")
        if rows and clean(rows[0].text()) == "Dividend":
            return table
    return None


def _parse_dividends(tables: list[Node]) -> list[Dividend]:
    table = _find_dividend_table(tables)
    if table is None:
        return []
    dividends: list[Dividend] = []
    current_pool: str | None = None
    for row in table.css("tr"):
        cells = [clean(c.text()) for c in _cells(row)]
        cells = [c for c in cells if c != ""]
        if len(cells) >= 3:
            pool, combination, value = cells[0], cells[1], cells[2]
            current_pool = pool
        elif len(cells) == 2 and current_pool is not None:
            pool, combination, value = current_pool, cells[0], cells[1]
        else:
            continue
        amount = _odds(value)
        if amount is None or pool in {"Pool", "Dividend"}:
            continue
        dividends.append(Dividend(pool=pool, combination=combination, dividend=amount))
    return dividends


def _find_meta_table(tables: list[Node]) -> Node | None:
    for table in tables:
        rows = table.css("tr")
        if rows:
            first = _cells(rows[0])
            if first and _RACE_HEADER_RE.search(clean(first[0].text())):
                return table
    return None


def _parse_meta(tables: list[Node]) -> _Meta:
    table = _find_meta_table(tables)
    meta: _Meta = {}
    if table is None:
        return meta
    rows = table.css("tr")
    col0: list[str] = []
    for row in rows:
        cells = [clean(c.text()) for c in _cells(row)]
        if cells:
            col0.append(cells[0])
        # labelled fields live in the 2nd/3rd columns
        for j, cell in enumerate(cells):
            label = cell.rstrip(" :")
            value = cells[j + 1] if j + 1 < len(cells) else ""
            if label == "Going":
                meta["going"] = value or None
            elif label == "Course":
                meta["course"] = value or None
                meta["surface"] = _surface(value)
                rail_match = re.search(r'"([^"]+)"', value)  # rail token, e.g. "C" / "A+3"
                meta["rail"] = rail_match.group(1) if rail_match else None

    header = next((c for c in col0 if _RACE_HEADER_RE.search(c)), "")
    if (m := _RACE_HEADER_RE.search(header)) is not None:
        meta["race_index"] = int(m.group(2))

    class_line = next((c for c in col0 if _CLASS_LINE_RE.search(c)), None)
    if class_line:
        parts = [p.strip() for p in class_line.split(" - ")]
        meta["race_class"] = parts[0] or None
        if (dm := _DISTANCE_RE.search(class_line)) is not None:
            meta["distance_m"] = int(dm.group(1))
        if (rm := _RATING_RE.search(class_line)) is not None:
            meta["rating_band"] = rm.group(1) or None

    prize = next((c for c in col0 if c.upper().startswith("HK$")), None)
    if prize:
        meta["prize_hkd"] = to_int(prize)

    name = next(
        (
            c
            for c in col0
            if c
            and c == c.upper()
            and not _RACE_HEADER_RE.search(c)
            and not _CLASS_LINE_RE.search(c)
            and not c.upper().startswith("HK$")
            and any(ch.isalpha() for ch in c)
        ),
        None,
    )
    if name:
        meta["race_name"] = name
    return meta


def _surface(course: str) -> str | None:
    upper = course.upper()
    if "ALL WEATHER" in upper or "AWT" in upper:
        return "ALL WEATHER TRACK"
    if "TURF" in upper:
        return "TURF"
    return None


def parse_race_result(html: str, race_date: date, venue: str, race_no: int) -> RaceResult:
    """Parse a per-race ``localresults`` page into a :class:`RaceResult`."""
    tables = HTMLParser(html).css("table")
    runners = _parse_runners(tables)
    dividends = _parse_dividends(tables)
    meta = _parse_meta(tables)
    final_time = next((r.finish_time_s for r in runners if r.finish_pos == 1), None)
    return RaceResult(
        race_date=race_date,
        venue=venue,
        race_no=race_no,
        final_time_s=final_time,
        runners=runners,
        dividends=dividends,
        **meta,
    )
