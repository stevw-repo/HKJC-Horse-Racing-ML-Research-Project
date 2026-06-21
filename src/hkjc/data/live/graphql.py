"""Live HKJC odds via the public GraphQL gateway (M7).

The gateway whitelists **exact registered query strings** (an arbitrary hand-written ``racing``
query is rejected with ``WHITELIST_ERROR``), so :data:`CARD_QUERY` / :data:`ODDS_QUERY` are the
byte-for-byte operations the bet.hkjc.com app sends, and :data:`HEADERS` mirrors its request --
do not "tidy" them. The feed serves only current/upcoming meetings and falls back to the
currently-live meeting when a requested ``(date, venueCode)`` has no defined meeting, so callers
validate ``MeetingCard.status``.

Reads only -- this client fetches odds and cards; it never submits a wager.
"""

from __future__ import annotations

import json
from datetime import date
from types import TracebackType
from typing import Any

import httpx

from hkjc.common.config import get_config
from hkjc.data.live.models import CardRunner, MeetingCard, OddsNode, PoolOdds, RaceCard

HEADERS = {
    "accept": "*/*",
    "accept-language": "en,zh-TW;q=0.9,zh;q=0.8",
    "content-type": "application/json",
    "origin": "https://bet.hkjc.com",
    "referer": "https://bet.hkjc.com/",
    "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": (
        "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Mobile Safari/537.36"
    ),
}

# Exact whitelisted operations (captured from bet.hkjc.com). Byte-for-byte -- do not reformat.
CARD_QUERY = "query racing($date: String, $venueCode: String) {\n  raceMeetings(date: $date, venueCode: $venueCode) {\n    status\n    currentNumberOfRace\n    totalInvestment\n    races {\n      no\n      status\n      runners {\n        id\n        no\n        standbyNo\n        status\n        name_ch\n        name_en\n        horse {\n          id\n          code\n        }\n        color\n        barrierDrawNumber\n        handicapWeight\n        currentWeight\n        currentRating\n        internationalRating\n        gearInfo\n        racingColorFileName\n        allowance\n        trainerPreference\n        last6run\n        saddleClothNo\n        trumpCard\n        priority\n        finalPosition\n        deadHeat\n        winOdds\n        jockey {\n          code\n          name_en\n          name_ch\n        }\n        trainer {\n          code\n          name_en\n          name_ch\n        }\n      }\n    }\n  }\n}"  # noqa: E501
ODDS_QUERY = "query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {\n  raceMeetings(date: $date, venueCode: $venueCode) {\n    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {\n      id\n      status\n      sellStatus\n      oddsType\n      lastUpdateTime\n      guarantee\n      minTicketCost\n      name_en\n      name_ch\n      leg {\n        number\n        races\n      }\n      cWinSelections {\n        composite\n        name_ch\n        name_en\n        starters\n      }\n      oddsNodes {\n        combString\n        oddsValue\n        hotFavourite\n        oddsDropValue\n        bankerOdds {\n          combString\n          oddsValue\n        }\n      }\n    }\n  }\n}"  # noqa: E501


class GraphQLError(RuntimeError):
    """The gateway returned a GraphQL ``errors`` payload (e.g. WHITELIST_ERROR)."""


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


class LiveClient:
    """Synchronous client for the whitelisted ``racing`` operations."""

    def __init__(self, url: str | None = None, *, timeout: float = 20.0) -> None:
        self._url = url or get_config().sources.hkjc_graphql_url
        self._client = httpx.Client(headers=HEADERS, timeout=timeout)

    def __enter__(self) -> LiveClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _post(self, variables: dict[str, Any], query: str) -> list[dict[str, Any]]:
        body = {"operationName": "racing", "variables": variables, "query": query}
        resp = self._client.post(self._url, content=json.dumps(body))
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise GraphQLError(str(payload["errors"]))
        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        return list(meetings)

    def card(self, day: date, venue: str) -> MeetingCard | None:
        """Fetch an upcoming meeting's full card (B2), or None if no meeting is served."""
        meetings = self._post({"date": day.isoformat(), "venueCode": venue}, CARD_QUERY)
        return parse_card(meetings[0], day, venue) if meetings else None

    def odds(
        self, day: date, venue: str, race_no: int, odds_types: tuple[str, ...] = ("WIN", "PLA")
    ) -> list[PoolOdds]:
        """Fetch live WIN/PLACE odds for one race (B1)."""
        meetings = self._post(
            {
                "date": day.isoformat(),
                "venueCode": venue,
                "raceNo": race_no,
                "oddsTypes": list(odds_types),
            },
            ODDS_QUERY,
        )
        if not meetings:
            return []
        return parse_odds(meetings[0].get("pmPools") or [], race_no)


def parse_card(meeting: dict[str, Any], day: date, venue: str) -> MeetingCard:
    """Parse a raw ``raceMeetings`` node into a :class:`MeetingCard`."""
    races = [
        RaceCard(
            race_no=int(race["no"]),
            status=race.get("status"),
            runners=[_runner(r) for r in (race.get("runners") or [])],
        )
        for race in (meeting.get("races") or [])
    ]
    return MeetingCard(
        race_date=day,
        venue=venue,
        status=meeting.get("status"),
        total_investment=_to_float(meeting.get("totalInvestment")),
        races=races,
    )


def _runner(r: dict[str, Any]) -> CardRunner:
    horse = r.get("horse") or {}
    jockey = r.get("jockey") or {}
    trainer = r.get("trainer") or {}
    return CardRunner(
        saddle=_to_int(r.get("no")),
        horse_id=horse.get("id"),
        horse_code=horse.get("code"),
        name_en=r.get("name_en"),
        status=r.get("status"),
        draw=_to_int(r.get("barrierDrawNumber")),
        handicap_weight=_to_int(r.get("handicapWeight")),
        current_rating=_to_int(r.get("currentRating")),
        intl_rating=_to_int(r.get("internationalRating")),
        gear=r.get("gearInfo") or None,
        last6run=r.get("last6run") or None,
        win_odds=_to_float(r.get("winOdds")),
        jockey_code=jockey.get("code"),
        jockey_name=jockey.get("name_en"),
        trainer_code=trainer.get("code"),
        trainer_name=trainer.get("name_en"),
    )


def parse_odds(pools: list[dict[str, Any]], race_no: int) -> list[PoolOdds]:
    """Parse raw ``pmPools`` nodes into typed :class:`PoolOdds`."""
    out: list[PoolOdds] = []
    for pool in pools:
        nodes = [
            OddsNode(
                comb=str(n.get("combString")),
                saddle=_to_int(n.get("combString")),
                odds=_to_float(n.get("oddsValue")),
                hot_fav=bool(n.get("hotFavourite")),
                odds_drop=_to_float(n.get("oddsDropValue")),
            )
            for n in (pool.get("oddsNodes") or [])
        ]
        out.append(
            PoolOdds(
                race_no=race_no,
                pool_type=str(pool.get("oddsType")),
                status=pool.get("status"),
                sell_status=pool.get("sellStatus"),
                last_update_time=pool.get("lastUpdateTime"),
                nodes=nodes,
            )
        )
    return out
