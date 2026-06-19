"""Parsers for HKJC English race text (source #9): comments-on-running + report blobs.

``corunning`` is a clean per-runner table (Placing / HorseNo / Horse / Jockey / Gear /
Comment) -- the structured lagged signal the NLP pipeline is built on. The prose reports
(``racereportfull`` stewards' report, ``veterinaryrecord``, ``exceptionalfactors``) are
captured as best-effort text blobs for the corpus. All of it is **lagged** (PLAN.md §1C): the
meeting date is the text_event_time, valid only for a horse's later races.
"""

from __future__ import annotations

from selectolax.parser import HTMLParser, Node

from hkjc.data.models import CommentOnRunning, RaceText
from hkjc.data.parse.common import clean, id_from_node, to_int


def _tables_with_class(tree: HTMLParser, cls: str) -> list[Node]:
    return [tb for tb in tree.css("table") if cls in (tb.attributes.get("class") or "").split()]


def _canon(header: str) -> str:
    h = header.lower()
    if "comment" in h:
        return "comment"
    if "plac" in h:
        return "placing"
    if "horse no" in h or h.strip() == "horseno":
        return "horseno"
    if "horse" in h:
        return "horse"
    if "jockey" in h:
        return "jockey"
    if "gear" in h:
        return "gear"
    return ""


def parse_comments_on_running(html: str) -> list[CommentOnRunning]:
    """Parse the ``corunning`` page into one comment per runner."""
    tree = HTMLParser(html)
    for table in _tables_with_class(tree, "table_bd"):
        rows = table.css("tr")
        if not rows:
            continue
        header = {_canon(clean(c.text())): i for i, c in enumerate(rows[0].css("td,th"))}
        header.pop("", None)
        if "comment" not in header:
            continue
        out: list[CommentOnRunning] = []
        for tr in rows[1:]:
            cells = tr.css("td,th")
            if len(cells) <= header["comment"]:
                continue
            comment = clean(cells[header["comment"]].text())
            if not comment or comment == "--":
                continue
            horse_cell = cells[header["horse"]] if "horse" in header else None
            out.append(
                CommentOnRunning(
                    saddle=to_int(cells[header["horseno"]].text()) if "horseno" in header else None,
                    horse_id=id_from_node(horse_cell, "horseid") if horse_cell else None,
                    placing=to_int(cells[header["placing"]].text())
                    if "placing" in header
                    else None,
                    jockey=clean(cells[header["jockey"]].text()) or None
                    if "jockey" in header
                    else None,
                    gear=clean(cells[header["gear"]].text()) or None if "gear" in header else None,
                    comment=comment,
                )
            )
        return out
    return []


def parse_race_text(html: str, source: str) -> list[RaceText]:
    """Best-effort extraction of a report page's content tables as one text blob."""
    tree = HTMLParser(html)
    for tag in tree.css("script, style"):
        tag.decompose()
    blocks = [clean(tb.text()) for tb in _tables_with_class(tree, "table_bd")]
    text = " ".join(b for b in blocks if len(b) > 40)
    if not text:
        return []
    return [RaceText(source=source, race_no=None, horse_id=None, text=text)]
