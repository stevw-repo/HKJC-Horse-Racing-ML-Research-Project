"""The canonical model interface (PLAN.md §1D).

Every model emits one latent **Plackett-Luce strength** per runner; everything else is
derived from it:

* WIN  = within-race softmax over strengths (Benter-style conditional logit).
* PLACE = Harville recursion over strengths (``models.place``).
* Exotics (v2+) = Monte-Carlo orderings over the same strengths.

Models work on a flat design matrix ``X`` (one row per runner) plus a ``groups`` array that
assigns each row to a race. ``log_strength`` returns the latent log-strength ``eta_i``; the
strength is ``exp(eta_i)`` and WIN probabilities are the per-race softmax of ``eta``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def group_codes(groups: npt.ArrayLike) -> tuple[IntArray, int]:
    """Map arbitrary group labels to contiguous codes ``0..G-1``; return (codes, G)."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    codes = inverse.astype(np.int64).ravel()
    return codes, int(codes.max()) + 1 if codes.size else 0


def softmax_by_group(eta: FloatArray, codes: IntArray, n_groups: int) -> FloatArray:
    """Numerically stable within-group softmax (per-race WIN probabilities)."""
    if eta.size == 0:
        return eta.astype(np.float64)
    maxs = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(maxs, codes, eta)
    shifted = np.exp(eta - maxs[codes])
    sums = np.bincount(codes, weights=shifted, minlength=n_groups)
    return shifted / sums[codes]


class ProbabilityModel(ABC):
    """Abstract base: race -> per-runner PL strength vector."""

    name: str = "base"

    @abstractmethod
    def fit(self, x: FloatArray, groups: npt.ArrayLike, y: FloatArray) -> ProbabilityModel:
        """Fit on design matrix ``x`` with per-row race ``groups`` and win labels ``y``."""

    @abstractmethod
    def log_strength(self, x: FloatArray) -> FloatArray:
        """Return the latent log-strength ``eta_i`` per row."""

    def win_probs(self, x: FloatArray, groups: npt.ArrayLike) -> FloatArray:
        """Within-race softmax over strengths = WIN probabilities."""
        codes, n = group_codes(groups)
        return softmax_by_group(self.log_strength(x), codes, n)
