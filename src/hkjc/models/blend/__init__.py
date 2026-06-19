"""Market-blend stage (PLAN.md §1A): combine model and market WIN probabilities.

The blend ``(1-w) * model + w * market`` is the Benter-style shrinkage toward the closing
line. ``w`` is a tunable hyperparameter (M3/Optuna); ``w=0`` is the pure model, ``w=1`` the
market. The blended probabilities are renormalised within each race.
"""

from __future__ import annotations

import numpy as np

from hkjc.models.base import FloatArray, IntArray


def blend_probs(
    model_probs: FloatArray,
    market_probs: FloatArray,
    weight: float,
    codes: IntArray,
    n_groups: int,
) -> FloatArray:
    """Blend model and market WIN probabilities and renormalise within race.

    Missing market probabilities fall back to the model's, so a runner with no SP is left
    on the model alone rather than dropped.
    """
    market = np.where(np.isfinite(market_probs), market_probs, model_probs)
    blended = (1.0 - weight) * model_probs + weight * market
    sums = np.bincount(codes, weights=blended, minlength=n_groups)
    return blended / np.where(sums[codes] > 0, sums[codes], 1.0)
