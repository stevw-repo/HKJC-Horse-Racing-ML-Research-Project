"""Parser for the HKJC ``displaysectionaltime`` page (source #7).

One page = one race. The grid carries, per runner and per 200m section: running position,
margin behind the leader, the cumulative time to the section marker, and the section split
itself (shown in a ``color_blue2`` span; the first section has only a cumulative time). These
splits are the basis for the home-grown speed figures.

Page quirks vs ``localresults``: the URL date is ``DD/MM/YYYY`` and there is no ``Racecourse``
param (one meeting per date). Empty sections render a ``blank_img.gif`` placeholder.
"""

from __future__ import annotations

from selectolax.parser import HTMLParser, Node

from hkjc.data.models import SectionalSplit
from hkjc.data.parse.common import clean, id_from_node, parse_time_to_seconds, to_int

MAX_SECTIONS = 6


def _race_table(tree: HTMLParser) -> Node | None:
    for table in tree.css("table"):
        classes = (table.attributes.get("class") or "").split()
        if "race_table" in classes:
            return table
    return None


def _first_number(text: str) -> float | None:
    """Parse the leading time token of a cell's text (``"20.84 10.37"`` -> 20.84)."""
    parts = clean(text).split()
    return parse_time_to_seconds(parts[0]) if parts else None


def _parse_section(
    cell: Node,
) -> tuple[int | None, str | None, float | None, float | None] | None:
    """Return (position, margin_raw, section_time_s, split_200m_s) or None if the section is
    absent. The bold value is the section time; the ``color_blue2`` value is the 200m pace."""
    paras = cell.css("p")
    if not paras:  # empty section -> blank_img placeholder
        return None
    pos_span = paras[0].css_first("span.f_fl")
    margin_node = paras[0].css_first("i")
    position = to_int(pos_span.text()) if pos_span is not None else None
    margin = clean(margin_node.text()) if margin_node is not None else None
    section_time = _first_number(paras[1].text()) if len(paras) > 1 else None
    split_span = paras[1].css_first("span.color_blue2") if len(paras) > 1 else None
    split_200m = _first_number(split_span.text()) if split_span is not None else None
    return position, margin, section_time, split_200m


def parse_race_sectionals(html: str) -> list[SectionalSplit]:
    """Parse one race's sectional-time page into per-runner per-section rows."""
    tree = HTMLParser(html)
    table = _race_table(tree)
    if table is None:
        return []
    rows: list[SectionalSplit] = []
    for tr in table.css("tr"):
        cells = tr.css("td")
        if len(cells) < MAX_SECTIONS + 4:  # header / spacer rows
            continue
        horse_id = id_from_node(cells[2], "horseid")
        if horse_id is None and to_int(cells[1].text()) is None:
            continue  # not a runner row
        saddle = to_int(cells[1].text())
        finishing_order = to_int(cells[0].text())
        final_time_s = parse_time_to_seconds(cells[-1].text())
        for index in range(1, MAX_SECTIONS + 1):
            parsed = _parse_section(cells[2 + index])
            if parsed is None:
                continue
            position, margin, section_time, split_200m = parsed
            rows.append(
                SectionalSplit(
                    saddle=saddle if saddle is not None else 0,
                    horse_id=horse_id,
                    finishing_order=finishing_order,
                    section_index=index,
                    running_position=position,
                    margin_raw=margin,
                    section_time_s=section_time,
                    split_200m_s=split_200m,
                    final_time_s=final_time_s,
                )
            )
    return rows
