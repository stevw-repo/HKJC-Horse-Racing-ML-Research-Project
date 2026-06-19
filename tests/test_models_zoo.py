"""Smoke + sanity tests for the M3 model zoo (CPU-forced, tiny synthetic data)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["HKJC_FORCE_CPU"] = "1"  # keep tests off the GPU / CI-safe

from hkjc.models.base import group_codes, softmax_by_group
from hkjc.models.ensemble import EnsembleModel
from hkjc.models.gbm import (
    CatBoostModel,
    LambdaMARTModel,
    LightGBMModel,
    XGBoostModel,
)
from hkjc.models.nn import FTTransformerModel, MLPModel

_Data = tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]


def _synthetic(n_races: int = 300, field: int = 6, seed: int = 0) -> _Data:
    rng = np.random.default_rng(seed)
    n = n_races * field
    num = rng.normal(size=(n, 3))
    cat = rng.integers(0, 5, size=(n, 1)).astype(float)
    x = np.column_stack([num, cat])
    groups = np.repeat(np.arange(n_races), field)
    codes, ng = group_codes(groups)
    strength = 1.5 * num[:, 0] - 1.0 * num[:, 1] + 0.5 * (cat[:, 0] == 0)
    probs = softmax_by_group(strength, codes, ng)
    y = np.zeros(n)
    for g in range(ng):
        idx = np.where(codes == g)[0]
        y[rng.choice(idx, p=probs[idx] / probs[idx].sum())] = 1.0
    return x, groups, y, [3]  # last column is categorical


def _assert_valid_win_probs(model: object, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.win_probs(x, groups))  # type: ignore[attr-defined]
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0) & (probs <= 1))
    codes, ng = group_codes(groups)
    sums = np.bincount(codes, weights=probs, minlength=ng)
    np.testing.assert_allclose(sums, np.ones(ng), atol=1e-6)
    return probs


def _top1(probs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    from hkjc.backtest.metrics import top1_hit_rate

    codes, ng = group_codes(groups)
    return top1_hit_rate(probs, y, codes, ng)


@pytest.mark.parametrize(
    "factory",
    [
        lambda c: LightGBMModel(cat_indices=c, n_estimators=60),
        lambda c: LambdaMARTModel(cat_indices=c, n_estimators=60),
        lambda c: XGBoostModel(cat_indices=c, n_estimators=60, use_gpu=False),
        lambda c: CatBoostModel(cat_indices=c, n_estimators=80, use_gpu=False),
    ],
)
def test_gbm_models_fit_and_rank(factory: object) -> None:
    x, groups, y, cats = _synthetic()
    model = factory(cats).fit(x, groups, y)  # type: ignore[operator]
    probs = _assert_valid_win_probs(model, x, groups)
    assert _top1(probs, y, groups) > 0.30  # learns signal (random ~0.167 for field 6)


def test_ensemble_averages_and_normalises() -> None:
    x, groups, y, cats = _synthetic()
    model = EnsembleModel(
        [
            LightGBMModel(cat_indices=cats, n_estimators=40),
            XGBoostModel(cat_indices=cats, n_estimators=40, use_gpu=False),
        ]
    ).fit(x, groups, y)
    probs = _assert_valid_win_probs(model, x, groups)
    assert _top1(probs, y, groups) > 0.30


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MLPModel(epochs=15, races_per_batch=64, use_gpu=False, seed=0),
        lambda: FTTransformerModel(epochs=15, races_per_batch=64, use_gpu=False, seed=0),
    ],
)
def test_nn_models_fit(factory: object) -> None:
    x, groups, y, _cats = _synthetic()
    x_num = x[:, :3]  # NNs use numeric only
    model = factory().fit(x_num, groups, y)  # type: ignore[operator]
    _assert_valid_win_probs(model, x_num, groups)
