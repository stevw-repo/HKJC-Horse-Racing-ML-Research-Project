"""Ensemble model: average member WIN probabilities within each race."""

from __future__ import annotations

import numpy as np

from hkjc.models.base import FloatArray, ProbabilityModel, group_codes


class EnsembleModel(ProbabilityModel):
    """Averages member models' within-race WIN probabilities (a robust soft vote)."""

    name = "ensemble"

    def __init__(self, members: list[ProbabilityModel]) -> None:
        if not members:
            msg = "EnsembleModel needs at least one member."
            raise ValueError(msg)
        self.members = members

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> EnsembleModel:
        for member in self.members:
            member.fit(x, groups, y)
        return self

    def win_probs(self, x: FloatArray, groups: np.typing.ArrayLike) -> FloatArray:
        probs = np.mean([m.win_probs(x, groups) for m in self.members], axis=0)
        # Renormalise within race so each race's probabilities sum to 1.
        codes, n = group_codes(groups)
        sums = np.bincount(codes, weights=probs, minlength=n)
        return np.asarray(probs / sums[codes], dtype=np.float64)

    def log_strength(self, x: FloatArray) -> FloatArray:
        # Fallback (the engine uses win_probs): mean of standardised member log-strengths.
        cols = []
        for m in self.members:
            s = m.log_strength(x)
            std = s.std()
            cols.append((s - s.mean()) / std if std > 1e-12 else s - s.mean())
        return np.asarray(np.mean(cols, axis=0), dtype=np.float64)
