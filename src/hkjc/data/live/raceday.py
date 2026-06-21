"""Race-day pipeline (M7): card -> forward features -> model -> blend live odds -> Kelly recs.

Fetches the upcoming card (GraphQL B2), builds as-of features for its runners, predicts WIN
(softmax) + PLACE (Harville) with the persisted production model, and -- when betting is open --
blends the live WIN odds (B1), flags positive-EV value, and sizes stakes with the M5 Kelly
machinery. It emits a recommendation card and **never places a bet**.

The whitelisted card query carries no race-level distance/going, so those features are left null
(median-imputed by the model); the horse-keyed backbone (form, rating, connections, bio) drives
the prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import polars as pl

from hkjc.backtest.pari_mutuel import places_for_field_size
from hkjc.common.config import AppConfig, get_config
from hkjc.common.time import now_hkt
from hkjc.data.live.graphql import LiveClient
from hkjc.data.live.models import MeetingCard
from hkjc.features.build import build_forward_features
from hkjc.features.design import build_design
from hkjc.models.base import FloatArray, group_codes
from hkjc.models.blend import blend_probs
from hkjc.models.persist import ProductionModel, load_production_model
from hkjc.models.place import harville_place_probs
from hkjc.risk.staking import StakingConfig, desired_stakes, round_stakes, scale_to_cap

_SPINE_SCHEMA = {
    "race_date": pl.Date,
    "venue": pl.String,
    "race_no": pl.Int64,
    "saddle": pl.Int64,
    "horse_id": pl.String,
    "jockey_code": pl.String,
    "jockey_name": pl.String,
    "trainer_code": pl.String,
    "trainer_name": pl.String,
    "draw": pl.Int64,
    "_card_rating": pl.Int64,
    "distance_m": pl.Int64,
    "going": pl.String,
    "surface": pl.String,
    "rail": pl.String,
    "race_class": pl.String,
    "race_index": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class RunnerRec:
    saddle: int | None
    horse_id: str | None
    name: str | None
    win_prob: float
    place_prob: float
    win_odds: float | None
    ev: float | None
    stake: float


@dataclass(frozen=True, slots=True)
class RaceRec:
    race_no: int
    status: str | None
    runners: list[RunnerRec]


@dataclass(frozen=True, slots=True)
class RaceDayCard:
    race_date: str
    venue: str
    model_name: str
    generated_at: str
    has_live_odds: bool
    note: str
    races: list[RaceRec]


def card_to_spine(card: MeetingCard) -> pl.DataFrame:
    """Declared runners -> the runner-spine columns the forward-feature builder needs."""
    rows: list[dict[str, object]] = []
    for race in card.races:
        for r in race.runners:
            if (r.status or "").lower() != "declared" or not r.horse_id:
                continue  # drop standby / scratched
            rows.append(
                {
                    "race_date": card.race_date,
                    "venue": card.venue,
                    "race_no": race.race_no,
                    "saddle": r.saddle,
                    "horse_id": r.horse_id,
                    "jockey_code": r.jockey_code,
                    "jockey_name": r.jockey_name,
                    "trainer_code": r.trainer_code,
                    "trainer_name": r.trainer_name,
                    "draw": r.draw,
                    "_card_rating": r.current_rating,
                    "distance_m": None,
                    "going": None,
                    "surface": None,
                    "rail": None,
                    "race_class": None,
                    "race_index": None,
                }
            )
    return pl.DataFrame(rows, schema=_SPINE_SCHEMA)


def _market_probs(feats: pl.DataFrame, odds_by_key: dict[tuple[int, int], float]) -> FloatArray:
    keys = zip(feats["race_no"].to_list(), feats["saddle"].to_list(), strict=True)
    inv = np.array(
        [1.0 / odds_by_key[k] if odds_by_key.get(k, 0.0) > 0 else np.nan for k in keys],
        dtype=np.float64,
    )
    return inv  # un-normalised; blend_probs renormalises within race


def run_raceday(
    cfg: AppConfig | None = None,
    *,
    day: date,
    venue: str,
    model_name: str = "logit",
    fetch_odds: bool = True,
    persist: bool = True,
) -> RaceDayCard:
    """Build the race-day recommendation card for an upcoming meeting."""
    cfg = cfg or get_config()
    prod = load_production_model(cfg, model_name=model_name)
    with LiveClient() as client:
        card = client.card(day, venue)
        if card is None or not card.races:
            msg = f"No card served for {venue} {day} (status none)."
            raise RuntimeError(msg)
        spine = card_to_spine(card)
        if spine.is_empty():
            msg = f"{venue} {day} has no declared runners yet (card status {card.status})."
            raise RuntimeError(msg)
        # Sort race-contiguous: Harville (in _predict) slices per race by adjacency.
        feats = build_forward_features(spine, cfg).sort(["race_no", "saddle"])
        wp, pp, codes = _predict(feats, prod, cfg)

        odds_by_key: dict[tuple[int, int], float] = {}
        if fetch_odds:
            for race_no in sorted({int(n) for n in feats["race_no"].to_list()}):
                for pool in client.odds(day, venue, race_no):
                    if pool.pool_type == "WIN":
                        for node in pool.nodes:
                            if node.saddle is not None and node.odds:
                                odds_by_key[(race_no, node.saddle)] = node.odds

    return _assemble(cfg, card, feats, wp, pp, codes, odds_by_key, model_name, persist=persist)


def _predict(
    feats: pl.DataFrame, prod: ProductionModel, cfg: AppConfig
) -> tuple[FloatArray, FloatArray, np.typing.NDArray[np.int64]]:
    design = build_design(feats)
    x = design.numeric() if prod.design == "numeric" else design.x
    race_no = feats["race_no"].to_numpy().astype(np.int64)
    codes, ng = group_codes(race_no)
    wp = prod.model.win_probs(x, codes)
    field = np.bincount(codes, minlength=ng)
    rules = cfg.backtest.place_rules.places_by_field_size
    npg = np.array([places_for_field_size(int(f), rules) for f in field], dtype=np.int64)
    pp = harville_place_probs(wp, codes, ng, npg)
    return wp, pp, codes


def _assemble(
    cfg: AppConfig,
    card: MeetingCard,
    feats: pl.DataFrame,
    wp: FloatArray,
    pp: FloatArray,
    codes: np.typing.NDArray[np.int64],
    odds_by_key: dict[tuple[int, int], float],
    model_name: str,
    *,
    persist: bool,
) -> RaceDayCard:
    ng = int(codes.max()) + 1 if codes.size else 0
    market = _market_probs(feats, odds_by_key)
    blended = blend_probs(wp, market, cfg.models.market_blend_weight, codes, ng)
    odds = np.array(
        [
            odds_by_key.get((int(rn), int(sd)), np.nan)
            for rn, sd in zip(feats["race_no"], feats["saddle"], strict=True)
        ],
        dtype=np.float64,
    )
    ev = blended * odds - 1.0
    stakes = _stakes(cfg, blended, odds, codes, ng)

    names = {
        (int(r.race_no), int(rr.saddle or -1)): rr.name_en for r in card.races for rr in r.runners
    }
    race_no = feats["race_no"].to_numpy().astype(np.int64)
    saddle = feats["saddle"].to_numpy()
    horse = feats["horse_id"].to_list()
    by_race: dict[int, list[RunnerRec]] = {}
    for i in range(len(horse)):
        rn = int(race_no[i])
        sd = int(saddle[i]) if saddle[i] is not None else None
        has_odds = bool(np.isfinite(odds[i]) and odds[i] > 0)
        by_race.setdefault(rn, []).append(
            RunnerRec(
                saddle=sd,
                horse_id=horse[i],
                name=names.get((rn, sd if sd is not None else -1)),
                win_prob=round(float(wp[i]), 4),
                place_prob=round(float(pp[i]), 4),
                win_odds=round(float(odds[i]), 1) if has_odds else None,
                ev=round(float(ev[i]), 4) if has_odds else None,
                stake=round(float(stakes[i]), 1),
            )
        )
    status_by_race = {int(r.race_no): r.status for r in card.races}
    races = [
        RaceRec(
            race_no=rn,
            status=status_by_race.get(rn),
            runners=sorted(by_race[rn], key=lambda x: -x.win_prob),
        )
        for rn in sorted(by_race)
    ]
    has_odds = bool(odds_by_key)
    out = RaceDayCard(
        race_date=card.race_date.isoformat(),
        venue=card.venue,
        model_name=model_name,
        generated_at=now_hkt().isoformat(timespec="seconds"),
        has_live_odds=has_odds,
        note=(
            "Recommendations only -- the platform never places a bet."
            if has_odds
            else "Model-only (betting not open yet); value + stakes appear once live odds flow."
        ),
        races=races,
    )
    if persist:
        path = cfg.paths.processed_dir / "raceday" / f"{out.race_date}_{out.venue}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(out), indent=2), encoding="utf-8")
    return out


def _stakes(
    cfg: AppConfig,
    blended: FloatArray,
    odds: FloatArray,
    codes: np.typing.NDArray[np.int64],
    ng: int,
) -> FloatArray:
    """Per-race fractional-Kelly stakes on +EV runners (reuses the M5 staking machinery)."""
    policy = StakingConfig(
        method="kelly_fractional",
        correlated=cfg.risk.correlated_kelly,
        kelly_lambda=0.25,
        ev_threshold=cfg.risk.ev_threshold,
        per_race_cap=cfg.risk.per_race_cap,
        per_day_cap=cfg.risk.per_day_cap,
        bet_unit=cfg.risk.bet_unit,
        min_bet=cfg.risk.min_bet,
    )
    out = np.zeros(blended.shape, dtype=np.float64)
    for g in range(ng):
        mask = codes == g
        if not mask.any() or not np.any(np.isfinite(odds[mask])):
            continue
        cash = desired_stakes(blended[mask], odds[mask], cfg.risk.bankroll, policy, win=True)
        cash = scale_to_cap(cash, policy.per_race_cap * cfg.risk.bankroll)
        out[mask] = round_stakes(cash, unit=policy.bet_unit, min_bet=policy.min_bet)
    return out
