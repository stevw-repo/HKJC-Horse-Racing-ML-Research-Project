"""Bootstrap confidence intervals for ROI (PLAN.md §1F, §2).

Resamples *races* (not bets) with replacement to separate skill from luck -- the unit of
independence is the race, and a race contributes a (profit, stake) pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.models.base import FloatArray


@dataclass(frozen=True, slots=True)
class RoiCI:
    """Point ROI plus a bootstrap percentile confidence interval."""

    roi: float
    lo: float
    hi: float
    n_races: int


def bootstrap_roi_ci(
    race_profit: FloatArray,
    race_stake: FloatArray,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> RoiCI:
    """Percentile CI for ROI = sum(profit)/sum(stake), resampling races with replacement."""
    profit = np.asarray(race_profit, dtype=np.float64)
    stake = np.asarray(race_stake, dtype=np.float64)
    n = profit.size
    total_stake = float(stake.sum())
    point = float(profit.sum() / total_stake) if total_stake > 0 else 0.0
    if n == 0 or total_stake <= 0:
        return RoiCI(point, point, point, n)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        s = float(stake[idx].sum())
        rois[i] = float(profit[idx].sum() / s) if s > 0 else 0.0
    lo = float(np.quantile(rois, alpha / 2))
    hi = float(np.quantile(rois, 1 - alpha / 2))
    return RoiCI(point, lo, hi, n)
