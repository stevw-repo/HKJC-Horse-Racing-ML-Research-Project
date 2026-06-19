"""Backtest metrics: ROI, Sharpe, within-race log-loss, and calibration bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.models.base import FloatArray, IntArray


def roi(profit: float, staked: float) -> float:
    """Return on turnover = total profit / total staked (0 when nothing staked)."""
    return profit / staked if staked > 0 else 0.0


def sharpe(returns: FloatArray) -> float:
    """Unitless Sharpe of per-bet net returns (mean / std). 0 if degenerate."""
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    return float(returns.mean()) / std if std > 1e-12 else 0.0


def win_log_loss(win_probs: FloatArray, won: FloatArray, codes: IntArray, n_groups: int) -> float:
    """Mean within-race negative log-likelihood of the actual winner(s)."""
    if win_probs.size == 0:
        return 0.0
    logp = np.log(np.clip(win_probs, 1e-12, 1.0))
    per_group_ll = np.bincount(codes, weights=won * logp, minlength=n_groups)
    winners = np.bincount(codes, weights=won, minlength=n_groups)
    mask = winners > 0
    if not mask.any():
        return 0.0
    return float(-(per_group_ll[mask] / winners[mask]).mean())


def brier_win(win_probs: FloatArray, won: FloatArray) -> float:
    """Mean squared error of the WIN probabilities."""
    if win_probs.size == 0:
        return 0.0
    return float(np.mean((win_probs - won) ** 2))


def top1_hit_rate(win_probs: FloatArray, won: FloatArray, codes: IntArray, n_groups: int) -> float:
    """Fraction of races whose highest-probability runner actually won."""
    hits = 0
    total = 0
    _uniq, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    for start, count in zip(starts, counts, strict=True):
        sl = slice(int(start), int(start) + int(count))
        if won[sl].sum() <= 0:
            continue
        total += 1
        if won[sl][int(np.argmax(win_probs[sl]))] > 0:
            hits += 1
    return hits / total if total else 0.0


@dataclass(frozen=True, slots=True)
class CalibrationBins:
    """Reliability-diagram data: predicted vs observed WIN rate per probability bin."""

    bin_mid: list[float]
    pred_mean: list[float]
    obs_rate: list[float]
    count: list[int]


def expected_calibration_error(prob: FloatArray, outcome: FloatArray, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rate across probability bins (ECE)."""
    if prob.size == 0:
        return 0.0
    bins = calibration_bins(prob, outcome, n_bins)
    total = float(sum(bins.count))
    if total == 0:
        return 0.0
    return sum(
        c / total * abs(p - o)
        for p, o, c in zip(bins.pred_mean, bins.obs_rate, bins.count, strict=True)
    )


def calibration_bins(prob: FloatArray, outcome: FloatArray, n_bins: int = 10) -> CalibrationBins:
    """Bin predictions into ``n_bins`` equal-width buckets over [0,1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    mids, preds, obs, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        mids.append(float((edges[b] + edges[b + 1]) / 2))
        preds.append(float(prob[mask].mean()))
        obs.append(float(outcome[mask].mean()))
        counts.append(c)
    return CalibrationBins(mids, preds, obs, counts)
