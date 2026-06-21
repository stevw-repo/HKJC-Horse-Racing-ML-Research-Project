"""Kelly stake sizing (M5, PLAN.md §2 M5, §5.2).

Three log-optimal sizers, all returning **bankroll fractions** (not cash):

* :func:`kelly_fraction` -- the textbook single-bet Kelly ``f* = (p*b - 1)/(b - 1)`` for a
  bet at decimal odds ``b`` (payout per unit *including* stake) and win probability ``p``.
* :func:`naive_kelly` -- per-bet single Kelly applied independently to every runner. This is
  the ``correlated_kelly = false`` baseline: it ignores that, within one race, at most one WIN
  bet can come in. Because mutually exclusive bets partially **hedge** each other, the naive
  sizer is sub-optimal -- it *under*-deploys, forgoing the hedge when several runners are
  simultaneously +EV, and it never bets a runner that looks -EV in isolation even when doing
  so would raise log-growth. The exact solver captures both.
* :func:`simultaneous_kelly_win` -- the **exact** correlated solution for the mutually
  exclusive WIN outcomes of a single race: the stake vector that maximises ``E[log wealth]``
  over the race's outcomes, holding the rest as cash. With one candidate it reduces exactly
  to :func:`kelly_fraction`.

The closed form for a *given* bet set ``S`` is ``f_i = p_i - R/b_i`` with reserve
``R = (1 - sum_{S} p_i) / (1 - sum_{S} 1/b_i)`` (the optimal cash fraction). The optimal set
is an upper-contour set in expected value ``p_i*b_i`` (a runner is bet iff ``p_i*b_i > R``).
Crucially ``R`` can fall *below* 1, so a runner that is mildly -EV in isolation
(``p_i*b_i < 1``) can still belong in the optimal set as a hedge -- the bet set is **not**
simply "the +EV runners". We therefore evaluate every value-sorted prefix and keep the one
with the highest exact ``E[log wealth]`` -- O(field^2), trivially cheap for <=14 runners and
free of any greedy stopping-rule doubt.

PLACE Kelly uses :func:`kelly_fraction` per runner (see ``staking.py``): the exact correlated
solver is **WIN-only**, because place outcomes are *positively* correlated (several horses
place) and the place dividend is not known at bet time -- documented approximations, not the
clean mutually-exclusive WIN structure.
"""

from __future__ import annotations

import numpy as np

from hkjc.models.base import FloatArray


def kelly_fraction(p: float, b: float) -> float:
    """Single-bet Kelly fraction ``(p*b - 1)/(b - 1)``; 0 when there is no edge.

    ``b`` is decimal odds (payout per unit incl. stake), so net odds are ``b - 1``. Returns a
    fraction in ``[0, 1]``; non-finite inputs, ``b <= 1`` or non-positive edge -> 0.
    """
    if not (np.isfinite(p) and np.isfinite(b)) or b <= 1.0 or p <= 0.0:
        return 0.0
    edge = p * b - 1.0
    if edge <= 0.0:
        return 0.0
    return float(min(edge / (b - 1.0), 1.0))


def naive_kelly(p: FloatArray, b: FloatArray) -> FloatArray:
    """Per-bet single Kelly for every runner, ignoring within-race mutual exclusivity."""
    p = np.asarray(p, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(b) & (b > 1.0) & (p > 0.0)
    edge = p * b - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(valid & (edge > 0.0), edge / (b - 1.0), 0.0)
    clipped: FloatArray = np.clip(np.nan_to_num(f, nan=0.0), 0.0, 1.0)
    return clipped


def expected_log_wealth(f: FloatArray, p: FloatArray, b: FloatArray) -> float:
    """``E[log wealth]`` of WIN stake-fractions ``f`` over one race's outcomes.

    Outcome "runner j wins" (probability ``p_j``) leaves wealth ``1 - sum(f) + f_j*b_j``;
    unbet runners (``f_j = 0``) collapse to the cash term ``1 - sum(f)``. Returns ``-inf`` if
    the stakes are infeasible (``sum(f) >= 1``)."""
    f = np.asarray(f, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    staked = float(f.sum())
    cash = 1.0 - staked
    if cash <= 0.0:
        return float("-inf")
    bet = f > 0.0
    p_bet = float(p[bet].sum())
    wealth_win = cash + f[bet] * b[bet]
    if np.any(wealth_win <= 0.0):
        return float("-inf")
    total = float(np.dot(p[bet], np.log(wealth_win)))
    p_none = max(1.0 - p_bet, 0.0)
    if p_none > 0.0:
        total += p_none * np.log(cash)
    return total


def simultaneous_kelly_win(p: FloatArray, b: FloatArray) -> FloatArray:
    """Exact correlated Kelly for the mutually exclusive WIN outcomes of one race.

    Returns the ``E[log wealth]``-maximising stake-fraction vector aligned to the inputs
    (zeros for unbet runners). Reduces to :func:`kelly_fraction` for a single candidate.
    """
    p = np.asarray(p, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = p.size
    best = np.zeros(n, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(b) & (b > 1.0) & (p > 0.0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return best

    order = idx[np.argsort(-(p[idx] * b[idx]))]  # by expected value, descending
    best_val = 0.0  # E[log 1] of betting nothing
    sum_p = 0.0
    sum_inv_b = 0.0
    for t in range(order.size):
        k = order[t]
        sum_p += float(p[k])
        sum_inv_b += 1.0 / float(b[k])
        denom = 1.0 - sum_inv_b
        if denom <= 1e-12:
            continue  # the subset's implied probabilities sum to >=1 -> reserve undefined
        reserve = (1.0 - sum_p) / denom
        if reserve <= 0.0:
            continue
        prefix = order[: t + 1]
        f_prefix = p[prefix] - reserve / b[prefix]
        if np.any(f_prefix < -1e-12):
            continue  # this prefix is not a valid all-positive bet set; a later one may be
        cand = np.zeros(n, dtype=np.float64)
        cand[prefix] = np.clip(f_prefix, 0.0, None)
        val = expected_log_wealth(cand, p, b)
        if val > best_val + 1e-15:
            best_val = val
            best = cand
    return best
