"""Staking policies + exposure caps + legal rounding (M5, PLAN.md §2 M5, §5.2).

A :class:`StakingConfig` turns one race's model/odds into **desired cash stakes** per runner,
for each of the four staking methods:

* ``flat`` -- a constant cash stake on every positive-EV bet.
* ``fixed_fraction`` -- a constant fraction of bankroll on every positive-EV bet.
* ``kelly_full`` / ``kelly_fractional`` -- log-optimal sizing (``kelly.py``), scaled by
  ``kelly_lambda`` (1.0 = full; the fractional grid {0.05..0.5} otherwise). WIN uses the
  exact correlated solver when ``correlated`` is set, else the naive per-bet sizer; PLACE
  always uses per-bet Kelly (the correlated solver is WIN-only -- see ``kelly.py``).

Every method only bets runners clearing the EV gate (``p*b - 1 >= ev_threshold``, the >=5%
net-takeout edge of PLAN.md §5.2). The bankroll exposure caps (per-race 10%, per-day 25%) and
the legal HK$10 rounding are applied by the simulator via :func:`scale_to_cap` and
:func:`round_stakes`; rounding is **last**, so the HK$10 granularity loss (a headline M5
finding, PLAN.md §1F) is what actually reaches the pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hkjc.backtest.pari_mutuel import round_stake
from hkjc.models.base import FloatArray
from hkjc.risk.kelly import naive_kelly, simultaneous_kelly_win

BoolArray = npt.NDArray[np.bool_]
STAKING_METHODS = ("flat", "fixed_fraction", "kelly_full", "kelly_fractional")


@dataclass(frozen=True, slots=True)
class StakingConfig:
    """One staking policy. ``kelly_lambda`` is the fractional-Kelly multiplier (1.0 = full)."""

    method: str
    correlated: bool = True
    kelly_lambda: float = 1.0
    flat_stake: float = 10.0
    fixed_fraction: float = 0.02
    ev_threshold: float = 0.05
    per_race_cap: float = 0.10
    per_day_cap: float = 0.25
    bet_unit: float = 10.0
    min_bet: float = 10.0

    def __post_init__(self) -> None:
        if self.method not in STAKING_METHODS:
            msg = f"unknown staking method {self.method!r}; expected one of {STAKING_METHODS}"
            raise ValueError(msg)


def ev_selection(p: FloatArray, b: FloatArray, ev_threshold: float) -> BoolArray:
    """Boolean mask of runners clearing the EV gate ``p*b - 1 >= ev_threshold``."""
    p = np.asarray(p, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(b) & (b > 1.0) & (p > 0.0)
    with np.errstate(invalid="ignore"):
        edge = np.where(valid, p * b - 1.0, -np.inf)
    mask: BoolArray = valid & (edge >= ev_threshold)
    return mask


def _kelly_stakes(
    p: FloatArray, b: FloatArray, sel: BoolArray, bankroll: float, cfg: StakingConfig, *, win: bool
) -> FloatArray:
    p_sel = np.where(sel, np.asarray(p, dtype=np.float64), 0.0)
    frac = simultaneous_kelly_win(p_sel, b) if (win and cfg.correlated) else naive_kelly(p_sel, b)
    return cfg.kelly_lambda * frac * bankroll


def desired_stakes(
    p: FloatArray, b: FloatArray, bankroll: float, cfg: StakingConfig, *, win: bool = True
) -> FloatArray:
    """Desired cash stakes per runner for one race+pool, before caps and rounding.

    ``p`` is the bettor's probability (model-only or market-blended), ``b`` the decimal odds
    (WIN: the SP line; PLACE: the place dividend / unit). Set ``win=False`` for PLACE so the
    correlated WIN solver is bypassed (PLACE is sized per-bet)."""
    b = np.asarray(b, dtype=np.float64)
    sel = ev_selection(p, b, cfg.ev_threshold)
    if not sel.any():
        return np.zeros(np.asarray(p).shape, dtype=np.float64)
    if cfg.method == "flat":
        return np.where(sel, cfg.flat_stake, 0.0)
    if cfg.method == "fixed_fraction":
        return np.where(sel, cfg.fixed_fraction * bankroll, 0.0)
    return _kelly_stakes(p, b, sel, bankroll, cfg, win=win)


def scale_to_cap(stakes: FloatArray, cap_cash: float) -> FloatArray:
    """Proportionally scale ``stakes`` down so their sum does not exceed ``cap_cash``."""
    stakes = np.asarray(stakes, dtype=np.float64)
    total = float(stakes.sum())
    if total <= cap_cash or total <= 0.0:
        return stakes
    return stakes * (cap_cash / total)


def round_stakes(stakes: FloatArray, *, unit: float, min_bet: float) -> FloatArray:
    """Round each stake to a legal multiple of ``unit`` (sub-``min_bet`` -> 0)."""
    stakes = np.asarray(stakes, dtype=np.float64)
    return np.array(
        [round_stake(float(s), unit=unit, min_bet=min_bet) for s in stakes], dtype=np.float64
    )
