"""Tests for the conditional-logit PL-strength baseline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hkjc.models.base import group_codes, softmax_by_group
from hkjc.models.logit import ConditionalLogit

_Sim = tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]


def _simulate(n_races: int, field: int, beta: npt.NDArray[np.float64], seed: int) -> _Sim:
    rng = np.random.default_rng(seed)
    n = n_races * field
    x = rng.normal(size=(n, beta.size))
    groups = np.repeat(np.arange(n_races), field)
    codes, ng = group_codes(groups)
    probs = softmax_by_group(x @ beta, codes, ng)
    y = np.zeros(n)
    for g in range(ng):
        idx = np.where(codes == g)[0]
        winner = rng.choice(idx, p=probs[idx] / probs[idx].sum())
        y[winner] = 1.0
    return x, groups, y


def test_recovers_coefficient_signs() -> None:
    beta_true = np.array([1.5, -1.0])
    x, groups, y = _simulate(2000, 8, beta_true, seed=7)
    model = ConditionalLogit(l2=0.01).fit(x, groups, y)
    coef = model.coefficients
    assert coef[0] > 0.3  # positive feature
    assert coef[1] < -0.2  # negative feature


def test_win_probs_sum_to_one_per_race() -> None:
    beta_true = np.array([1.0, -0.5])
    x, groups, y = _simulate(50, 10, beta_true, seed=1)
    model = ConditionalLogit().fit(x, groups, y)
    probs = model.win_probs(x, groups)
    codes, ng = group_codes(groups)
    sums = np.bincount(codes, weights=probs)
    np.testing.assert_allclose(sums, np.ones(ng), atol=1e-9)


def test_handles_nan_via_imputation() -> None:
    beta_true = np.array([1.0, -0.5])
    x, groups, y = _simulate(100, 6, beta_true, seed=3)
    x[::10, 0] = np.nan  # scatter some missing values
    model = ConditionalLogit().fit(x, groups, y)
    probs = model.win_probs(x, groups)
    assert np.all(np.isfinite(probs))
