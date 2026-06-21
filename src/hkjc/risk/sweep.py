"""Staking sweep orchestrator (M5, PLAN.md §2 M5).

Reuses the backtest's walk-forward OOS predictions once, builds a day-ordered card, then runs
every ``(staking policy x bankroll)`` cell through :func:`~hkjc.risk.simulate.simulate_path`.
The bettor's WIN probability is the model blended with the market (the value lens that needs a
price -- the only lens where staking is meaningful); the PLACE line, which HKJC does not
publish historically, is **constructed** from the market WIN probabilities via Harville and the
place takeout. Payouts always use the stored final dividends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hkjc.backtest.engine import walk_forward_oos
from hkjc.common.config import AppConfig, get_config
from hkjc.models.base import FloatArray, group_codes
from hkjc.models.place import harville_place_probs
from hkjc.risk.rebate import RebateRule
from hkjc.risk.simulate import RacePanel, StakingOutcome, simulate_path
from hkjc.risk.staking import StakingConfig

REBATE_THRESHOLD = 10000.0  # HK$ losing turnover per betline before any rebate (PLAN.md §1F)


@dataclass(frozen=True, slots=True)
class SweepResult:
    """All sweep cells plus provenance for the report."""

    outcomes: list[StakingOutcome]
    n_oos_races: int
    test_span: tuple[str, str]
    feature_version: str
    market_weight: float


def _staking(
    cfg: AppConfig,
    method: str,
    *,
    correlated: bool = True,
    kelly_lambda: float = 1.0,
    flat_stake: float = 10.0,
    fixed_fraction: float = 0.02,
) -> StakingConfig:
    """A :class:`StakingConfig` with the shared caps/EV/rounding pulled from config."""
    return StakingConfig(
        method=method,
        correlated=correlated,
        kelly_lambda=kelly_lambda,
        flat_stake=flat_stake,
        fixed_fraction=fixed_fraction,
        ev_threshold=cfg.risk.ev_threshold,
        per_race_cap=cfg.risk.per_race_cap,
        per_day_cap=cfg.risk.per_day_cap,
        bet_unit=cfg.risk.bet_unit,
        min_bet=cfg.risk.min_bet,
    )


def _policy_grid(cfg: AppConfig) -> list[StakingConfig]:
    """The staking policies to compare (PLAN.md §5.2): flat, fixed-fraction, full and
    fractional Kelly -- the Kelly methods in both naive and correlated variants."""
    grid: list[StakingConfig] = [
        _staking(cfg, "flat", flat_stake=cfg.risk.min_bet),
        _staking(cfg, "fixed_fraction", fixed_fraction=cfg.risk.fixed_fraction),
    ]
    for correlated in (False, True):
        grid.append(_staking(cfg, "kelly_full", correlated=correlated))
        for lam in cfg.risk.kelly_fractions:
            grid.append(_staking(cfg, "kelly_fractional", correlated=correlated, kelly_lambda=lam))
    return grid


def _build_card(
    cfg: AppConfig, *, l2: float, min_train_seasons: int
) -> tuple[list[list[RacePanel]], SweepResult]:
    """Fit the walk-forward, derive blended WIN probs + the constructed PLACE line, and group
    the OOS runners into a calendar-ordered ``[day][race]`` card."""
    wf = walk_forward_oos(cfg, l2=l2, min_train_seasons=min_train_seasons)
    a = wf.arrays
    oos = wf.oos
    wp = wf.win_prob
    pp = wf.place_prob

    market_weight = cfg.models.market_blend_weight
    market = a.market_prob[oos]
    mkt = np.where(np.isfinite(market) & (market > 0.0), market, wp)
    p_win = (1.0 - market_weight) * wp + market_weight * mkt

    # Constructed PLACE line: Harville on the (renormalised) market WIN probs, priced at the
    # place takeout -- HKJC publishes no historical place-odds line to bet against.
    race_id = a.race_id[oos]
    codes, ng = group_codes(race_id)
    npg = np.zeros(ng, dtype=np.int64)
    npg[codes] = a.n_places[oos]
    race_sum = np.zeros(ng, dtype=np.float64)
    np.add.at(race_sum, codes, mkt)
    mkt_norm = mkt / np.where(race_sum[codes] > 0.0, race_sum[codes], 1.0)
    place_prob_mkt = harville_place_probs(mkt_norm, codes, ng, npg)
    b_place = (1.0 - cfg.backtest.takeout_place) / np.clip(place_prob_mkt, 1e-9, None)

    df = pl.DataFrame(
        {
            "race_date": a.race_date[oos],
            "race_id": race_id,
            "p_win": p_win,
            "b_win": a.win_odds[oos],
            "won": a.y[oos],
            "win_div": a.win_div[oos],
            "p_place": pp,
            "b_place": b_place,
            "placed": a.placed[oos],
            "place_div": a.place_div[oos],
        }
    ).sort(["race_date", "race_id"])

    card: list[list[RacePanel]] = []
    for day_df in df.partition_by("race_date", maintain_order=True):
        races: list[RacePanel] = []
        for rf in day_df.partition_by("race_id", maintain_order=True):
            races.append(
                RacePanel(
                    p_win=_col(rf, "p_win"),
                    b_win=_col(rf, "b_win"),
                    won=_col(rf, "won"),
                    win_div=_col(rf, "win_div"),
                    p_place=_col(rf, "p_place"),
                    b_place=_col(rf, "b_place"),
                    placed=_col(rf, "placed"),
                    place_div=_col(rf, "place_div"),
                )
            )
        card.append(races)

    seasons = a.season[oos]
    meta = SweepResult(
        outcomes=[],
        n_oos_races=ng,
        test_span=(str(seasons[0]), str(seasons[-1])),
        feature_version=cfg.features.feature_version,
        market_weight=market_weight,
    )
    return card, meta


def _col(frame: pl.DataFrame, name: str) -> FloatArray:
    return frame[name].to_numpy().astype(np.float64)


def run_sweep(
    cfg: AppConfig | None = None,
    *,
    bankrolls: list[float] | None = None,
    pools: tuple[str, ...] = ("win", "place"),
    rebate_rate: float = 0.0,
    l2: float = 1.0,
    min_train_seasons: int = 1,
    seed: int = 0,
) -> SweepResult:
    """Run the full staking sweep across policies and bankrolls."""
    cfg = cfg or get_config()
    bankrolls = bankrolls if bankrolls is not None else list(cfg.risk.multi_bankroll)
    card, meta = _build_card(cfg, l2=l2, min_train_seasons=min_train_seasons)
    rebate = RebateRule(rate=rebate_rate, threshold=REBATE_THRESHOLD)

    outcomes: list[StakingOutcome] = []
    for bankroll in bankrolls:
        for policy in _policy_grid(cfg):
            outcomes.append(
                simulate_path(card, policy, bankroll, pools=pools, rebate=rebate, seed=seed)
            )
    return SweepResult(
        outcomes=outcomes,
        n_oos_races=meta.n_oos_races,
        test_span=meta.test_span,
        feature_version=meta.feature_version,
        market_weight=meta.market_weight,
    )
