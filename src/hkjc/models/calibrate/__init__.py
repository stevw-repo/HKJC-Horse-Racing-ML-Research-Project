"""Probability calibration (PLAN.md §1E).

Within-race softmax is the **primary** calibration (it makes each race's probabilities sum to
1); these are the **secondary**, post-hoc layers:

* ``fit_temperature`` -- a single scalar ``T`` that divides the logits before the within-race
  softmax (preserves the ranking, only sharpens/softens). The natural choice here.
* ``fit_isotonic`` / ``fit_platt`` -- pooled monotonic / logistic maps on the WIN probability,
  followed by within-race renormalisation.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from hkjc.backtest.metrics import win_log_loss
from hkjc.models.base import FloatArray, IntArray, softmax_by_group


def fit_temperature(eta: FloatArray, codes: IntArray, n_groups: int, y: FloatArray) -> float:
    """Fit the temperature that minimises within-race NLL (bounded 1-D search)."""

    def nll(temp: float) -> float:
        probs = softmax_by_group(eta / temp, codes, n_groups)
        return win_log_loss(probs, y, codes, n_groups)

    result = optimize.minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def renormalize(probs: FloatArray, codes: IntArray, n_groups: int) -> FloatArray:
    """Rescale so each race's probabilities sum to 1."""
    sums = np.bincount(codes, weights=probs, minlength=n_groups)
    return probs / np.where(sums[codes] > 0, sums[codes], 1.0)


def fit_isotonic(probs: FloatArray, y: FloatArray) -> IsotonicRegression:
    """Pooled isotonic map predicted WIN prob -> empirical (monotonic, clipped to [0,1])."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs, y)
    return iso


def fit_platt(probs: FloatArray, y: FloatArray) -> LogisticRegression:
    """Pooled Platt (logistic) map on the logit of the WIN probability."""
    logit = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression()
    lr.fit(logit.reshape(-1, 1), y.astype(int))
    return lr
