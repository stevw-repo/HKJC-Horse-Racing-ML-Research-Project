"""Parser for HKJC barrier-trial results (``btresult``, source #4).

The page repeats, per heat: a batch header (``Batch N - <LOCATION SURFACE> - <DIST>m``),
a going/time line, then a ``bigborder`` table of runners. Trial dates are enumerated from
the ``selectId`` dropdown (reuse :func:`hkjc.data.parse.results.parse_meeting_dates`); a
given date is fetched with ``btresult?date=YYYY/MM/DD``.
"""

from __future__ import annotations

import re
from datetime import date

from selectolax.parser import HTMLParser, Node

from hkjc.data.models import BarrierTrialRun
from hkjc.data.parse.common import clean, id_from_node, parse_time_to_seconds, to_int

_BATCH_RE = re.compile(r"Batch\s+(\d+)\s*-\s*(.+?)\s*-\s*(\d+)\s*m", re.IGNORECASE)
_GOING_RE = re.compile(r"Going:\s*(.+?)\s+Time:", re.IGNORECASE)
_TIME_RE = re.compile(r"\bTime:\s*([\d.:]+)")
_VENUE_BY_NAME = {"sha tin": "ST", "happy valley": "HV", "conghua": "CH"}


def _node_text(node: Node | None) -> str:
    return clean(node.text()) if node is not None else ""


def _venue_surface(location: str) -> tuple[str | None, str | None]:
    upper = location.upper()
    venue = next((code for name, code in _VENUE_BY_NAME.items() if name in location.lower()), None)
    if "ALL WEATHER" in upper or "AWT" in upper:
        surface = "ALL WEATHER TRACK"
    elif "TURF" in upper:
        surface = "TURF"
    else:
        surface = None
    return venue, surface


class _Batch:
    __slots__ = ("batch", "batch_time_s", "distance_m", "going", "location", "surface", "venue")

    def __init__(self, context: str) -> None:
        self.batch: int | None = None
        self.location: str | None = None
        self.venue: str | None = None
        self.surface: str | None = None
        self.distance_m: int | None = None
        self.going: str | None = None
        self.batch_time_s: float | None = None
        if (m := _BATCH_RE.search(context)) is not None:
            self.batch = int(m.group(1))
            self.location = clean(m.group(2))
            self.venue, self.surface = _venue_surface(self.location)
            self.distance_m = int(m.group(3))
        if (g := _GOING_RE.search(context)) is not None:
            self.going = clean(g.group(1)) or None
        if (t := _TIME_RE.search(context)) is not None:
            self.batch_time_s = parse_time_to_seconds(t.group(1))


_COLUMNS = {
    "horse": "horse",
    "jockey": "jockey",
    "trainer": "trainer",
    "draw": "draw",
    "gear": "gear",
    "lbw": "lbw",
    "running position": "running",
    "time": "time",
    "result": "result",
    "comment": "comment",
}


def _runners(table: Node, trial_date: date, batch: _Batch) -> list[BarrierTrialRun]:
    rows = table.css("tr")
    if not rows:
        return []
    header = [clean(c.text()).lower() for c in rows[0].css("td,th")]
    cols = {_COLUMNS[h]: i for i, h in enumerate(header) if h in _COLUMNS}
    if "horse" not in cols:
        return []

    runs: list[BarrierTrialRun] = []
    for row in rows[1:]:
        cells = row.css("td,th")
        by_key = {key: cells[i] for key, i in cols.items() if i < len(cells)}
        horse_node = by_key.get("horse")
        if horse_node is None:
            continue
        anchor = horse_node.css_first("a")
        name = (
            clean(anchor.text()) if anchor else re.sub(r"\(.*?\)\s*$", "", _node_text(horse_node))
        )
        if not name:
            continue
        runs.append(
            BarrierTrialRun(
                trial_date=trial_date,
                batch=batch.batch,
                location=batch.location,
                venue=batch.venue,
                surface=batch.surface,
                distance_m=batch.distance_m,
                going=batch.going,
                batch_time_s=batch.batch_time_s,
                horse_id=id_from_node(horse_node, "horseid"),
                horse_name=name,
                jockey=_node_text(by_key.get("jockey")) or None,
                trainer=_node_text(by_key.get("trainer")) or None,
                draw=to_int(_node_text(by_key.get("draw"))),
                gear=_node_text(by_key.get("gear")) or None,
                lbw_raw=_node_text(by_key.get("lbw")) or None,
                running_position_raw=_node_text(by_key.get("running")) or None,
                time_s=parse_time_to_seconds(_node_text(by_key.get("time"))),
                result=_node_text(by_key.get("result")) or None,
                comment=_node_text(by_key.get("comment")) or None,
            )
        )
    return runs


def parse_barrier_trials(html: str, trial_date: date) -> list[BarrierTrialRun]:
    """Parse a ``btresult`` page (one trial date) into per-runner trial records."""
    tree = HTMLParser(html)
    runs: list[BarrierTrialRun] = []
    context = ""
    for table in tree.css("table"):
        if "bigborder" in (table.attributes.get("class") or "").split():
            runs.extend(_runners(table, trial_date, _Batch(context)))
            context = ""
        else:
            context += " " + clean(table.text())
    return runs
