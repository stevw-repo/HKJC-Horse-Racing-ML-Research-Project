"""Parser for HKJC horse profile pages: the locked bio block + form records.

The form-records table is the historical pre-race signal (rating/gear/going/draw per past
run) that race cards would provide pre-race but which HKJC removes after each meeting.
Each form row carries the ``RaceIndex`` that joins back to a stored ``RaceResult``.
"""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser, Node

from hkjc.data.models import HorseFormRun, HorseProfile
from hkjc.data.parse.common import (
    clean,
    id_from_node,
    parse_dmy2,
    parse_time_to_seconds,
    to_float,
    to_int,
)


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _tables_with_class(tree: HTMLParser, cls: str) -> list[Node]:
    return [tb for tb in tree.css("table") if cls in (tb.attributes.get("class") or "").split()]


def _slash(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = [p.strip() for p in value.split("/")]
    first = parts[0] or None
    second = parts[1].strip() if len(parts) > 1 else None
    return first, (second or None)


def _bio_pairs(tree: HTMLParser) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for table in _tables_with_class(tree, "table_eng_text"):
        for row in table.css("tr"):
            cells = [clean(c.text()) for c in row.css("td,th")]
            if len(cells) >= 3 and cells[1] == ":":
                pairs[_norm(cells[0])] = cells[2]
    return pairs


def _horse_name(tree: HTMLParser) -> str | None:
    title = tree.css_first("title")
    if title is None:
        return None
    return clean(title.text()).split(" - ")[0] or None


def _form_canon(header: str) -> str:
    h = clean(header).lower()
    compact = h.replace(" ", "")
    if compact == "raceindex":
        return "race_index"
    if h.startswith("pla"):
        return "pos"
    if h == "date":
        return "date"
    if "track" in h or h.startswith("rc"):
        return "rctc"
    if h.startswith("dist"):
        return "dist"
    if h == "g":
        return "going"
    if "class" in h:
        return "race_class"
    if h.startswith("dr"):
        return "draw"
    if h.startswith("rtg") or "rating" in h:
        return "rating"
    if h.startswith("trainer"):
        return "trainer"
    if h.startswith("jockey"):
        return "jockey"
    if "lbw" in h:
        return "lbw"
    if "win odds" in h:
        return "win_odds"
    if h.startswith("act"):
        return "act_wt"
    if "running" in h:
        return "running"
    if "finish time" in h:
        return "finish_time"
    if "declar" in h:
        return "declar_wt"
    if "gear" in h:
        return "gear"
    return ""


def _node_text(node: Node | None) -> str:
    return clean(node.text()) if node is not None else ""


def _parse_form(tree: HTMLParser) -> list[HorseFormRun]:
    tables = _tables_with_class(tree, "bigborder")
    if not tables:
        return []
    rows = tables[0].css("tr")
    cols: dict[str, int] = {}
    data_start = 0
    for i, row in enumerate(rows):
        mapping = {_form_canon(c.text()): idx for idx, c in enumerate(row.css("td,th"))}
        mapping.pop("", None)
        if "race_index" in mapping and "pos" in mapping:
            cols = mapping
            data_start = i + 1
            break
    if not cols:
        return []

    runs: list[HorseFormRun] = []
    for row in rows[data_start:]:
        cells = row.css("td,th")
        if len(cells) < 5:  # season-header / spacer rows
            continue
        by_key = {key: cells[idx] for key, idx in cols.items() if idx < len(cells)}
        race_index = to_int(_node_text(by_key.get("race_index")))
        pos_raw = _node_text(by_key.get("pos"))
        if race_index is None and not pos_raw:
            continue
        venue, track = _slash(_node_text(by_key.get("rctc")))
        course = None
        rctc_parts = _node_text(by_key.get("rctc")).split("/")
        if len(rctc_parts) > 2:
            course = clean(rctc_parts[2]).strip('"') or None
        runs.append(
            HorseFormRun(
                race_index=race_index,
                finish_pos=to_int(pos_raw),
                finish_pos_raw=pos_raw,
                run_date=parse_dmy2(_node_text(by_key.get("date"))),
                venue=venue,
                track=track,
                course=course,
                distance_m=to_int(_node_text(by_key.get("dist"))),
                going=_node_text(by_key.get("going")) or None,
                race_class=_node_text(by_key.get("race_class")) or None,
                draw=to_int(_node_text(by_key.get("draw"))),
                rating=to_int(_node_text(by_key.get("rating"))),
                jockey_code=id_from_node(by_key["jockey"], "jockeyid")
                if "jockey" in by_key
                else None,
                trainer_code=(
                    id_from_node(by_key["trainer"], "trainerid") if "trainer" in by_key else None
                ),
                lbw_raw=_node_text(by_key.get("lbw")) or None,
                win_odds=to_float(_node_text(by_key.get("win_odds"))),
                actual_weight=to_int(_node_text(by_key.get("act_wt"))),
                running_position_raw=_node_text(by_key.get("running")) or None,
                finish_time_s=parse_time_to_seconds(_node_text(by_key.get("finish_time"))),
                declared_weight=to_int(_node_text(by_key.get("declar_wt"))),
                gear=_node_text(by_key.get("gear")) or None,
            )
        )
    return runs


def parse_horse_profile(html: str, horse_id: str) -> HorseProfile:
    """Parse a horse profile page into a :class:`HorseProfile`."""
    tree = HTMLParser(html)
    bio = _bio_pairs(tree)
    country, age = _slash(bio.get("countryoforiginage"))
    colour, sex = _slash(bio.get("coloursex"))
    return HorseProfile(
        horse_id=horse_id,
        name=_horse_name(tree),
        brand=horse_id.rsplit("_", 1)[-1],
        country_of_origin=country,
        age=to_int(age),
        colour=colour,
        sex=sex,
        import_type=bio.get("importtype") or None,
        sire=bio.get("sire") or None,
        dam=bio.get("dam") or None,
        dams_sire=bio.get("damssire") or None,
        owner=bio.get("owner") or None,
        trainer=bio.get("trainer") or None,
        current_rating=to_int(bio.get("currentrating")),
        season_start_rating=to_int(bio.get("startofseasonrating")),
        season_stakes=to_int(bio.get("seasonstakes")),
        total_stakes=to_int(bio.get("totalstakes")),
        form=_parse_form(tree),
    )
