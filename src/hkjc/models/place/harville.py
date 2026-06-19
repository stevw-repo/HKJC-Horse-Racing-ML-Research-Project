"""Harville PLACE recursion (PLAN.md §1D).

Given per-runner WIN probabilities ``p`` (= softmax of PL strengths), the Harville model
assumes the finishing order is drawn by repeatedly sampling without replacement in
proportion to ``p``. ``harville_topk`` returns ``P(runner finishes in the top k)`` using
closed forms for ``k <= 3`` (HK pays at most 3 places). Harville is known to *overstate*
favourites' place chances; that bias is exactly why M3 compares Henery / Lo-Bacon-Shek and a
directly-modelled PLACE head.
"""

from __future__ import annotations

import numpy as np

from hkjc.models.base import FloatArray, IntArray


def harville_topk(p: FloatArray, k: int) -> FloatArray:
    """``P(in top k)`` per runner for a single race (probabilities ``p`` need not sum to 1)."""
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if k <= 0 or p.size == 0 or total <= 0:
        return np.zeros_like(p)
    p = p / total
    out = p.copy()  # P(finish 1st)
    if k >= 2:
        r = p / np.clip(1.0 - p, 1e-12, None)
        out = out + p * (r.sum() - r)  # P(finish 2nd)
    if k >= 3:
        denom = 1.0 - p[:, None] - p[None, :]
        safe = np.where(denom > 1e-12, denom, 1.0)  # diagonal (1-2p) can be <=0; avoid /0
        m = (r[:, None] * p[None, :]) / safe
        np.fill_diagonal(m, 0.0)
        m = np.where(denom > 1e-12, m, 0.0)
        # P(finish 3rd)_i = p_i * sum_{a!=i,b!=i,a!=b} r_a p_b/(1-p_a-p_b)
        out = out + p * (m.sum() - m.sum(axis=1) - m.sum(axis=0))
    return np.clip(out, 0.0, 1.0)


def harville_place_probs(
    p: FloatArray, codes: IntArray, n_groups: int, n_places: IntArray
) -> FloatArray:
    """Vectorised PLACE probabilities across races.

    ``codes`` assigns each row to a race and **must be contiguous** (rows of a race adjacent,
    which the feature frame guarantees by sorting on the race key). ``n_places[g]`` is the
    paid-place count for race ``g``.
    """
    out = np.zeros_like(np.asarray(p, dtype=np.float64))
    if out.size == 0:
        return out
    uniq, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    for code, start, count in zip(uniq, starts, counts, strict=True):
        sl = slice(int(start), int(start) + int(count))
        out[sl] = harville_topk(p[sl], int(n_places[int(code)]))
    return out
