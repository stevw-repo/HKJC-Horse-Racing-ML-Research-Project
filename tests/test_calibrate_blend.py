"""Tests for the design matrix, calibration layers, and the market blend (M3)."""

from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.backtest.metrics import win_log_loss
from hkjc.features.design import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_design
from hkjc.models.base import group_codes, softmax_by_group
from hkjc.models.blend import blend_probs
from hkjc.models.calibrate import fit_isotonic, fit_platt, fit_temperature, renormalize


def test_build_design_shapes_and_encoding() -> None:
    n = 12
    data: dict[str, list[float] | list[str]] = {
        c: [float(i) for i in range(n)] for c in NUMERIC_FEATURES
    }
    for c in CATEGORICAL_FEATURES:
        data[c] = [f"v{i % 3}" for i in range(n)]
    design = build_design(pl.DataFrame(data))
    assert design.x.shape == (n, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    assert design.numeric_indices == list(range(len(NUMERIC_FEATURES)))
    assert len(design.categorical_indices) == len(CATEGORICAL_FEATURES)
    cat0 = design.x[:, design.categorical_indices[0]]
    assert set(np.unique(cat0).tolist()).issubset({0.0, 1.0, 2.0})
    assert np.all(np.isfinite(design.x[:, design.categorical_indices]))  # nulls -> own code


def _races(
    n_races: int = 200, field: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n = n_races * field
    groups = np.repeat(np.arange(n_races), field)
    codes, ng = group_codes(groups)
    eta = rng.normal(size=n)
    probs = softmax_by_group(eta, codes, ng)
    y = np.zeros(n)
    for g in range(ng):
        idx = np.where(codes == g)[0]
        y[rng.choice(idx, p=probs[idx] / probs[idx].sum())] = 1.0
    return eta, y, codes, ng


def test_temperature_reduces_overconfident_loss() -> None:
    eta, y, codes, ng = _races()
    over = 3.0 * eta  # overconfident logits
    raw = win_log_loss(softmax_by_group(over, codes, ng), y, codes, ng)
    temp = fit_temperature(over, codes, ng, y)
    cal = win_log_loss(softmax_by_group(over / temp, codes, ng), y, codes, ng)
    assert temp > 1.0  # needs softening
    assert cal <= raw + 1e-9


def test_renormalize_sums_to_one() -> None:
    _eta, _y, codes, ng = _races(50, 6)
    probs = np.random.default_rng(1).random(codes.size)
    out = renormalize(probs, codes, ng)
    sums = np.bincount(codes, weights=out, minlength=ng)
    np.testing.assert_allclose(sums, np.ones(ng), atol=1e-9)


def test_isotonic_and_platt_produce_valid_probs() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    iso = fit_isotonic(p, y)
    platt = fit_platt(p, y)
    iso_out = iso.predict(p)
    assert np.all((iso_out >= 0) & (iso_out <= 1))
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt_out = platt.predict_proba(logit.reshape(-1, 1))[:, 1]
    assert np.all((platt_out >= 0) & (platt_out <= 1))


def test_blend_endpoints_and_fallback() -> None:
    codes = np.array([0, 0, 0, 1, 1, 1])
    ng = 2
    model = np.array([0.5, 0.3, 0.2, 0.6, 0.3, 0.1])
    market = np.array([0.2, 0.3, 0.5, np.nan, np.nan, np.nan])  # race 1 has no SP
    only_model = blend_probs(model, market, 0.0, codes, ng)
    np.testing.assert_allclose(only_model, model, atol=1e-9)
    blended = blend_probs(model, market, 1.0, codes, ng)
    # Race 1 has no market -> falls back to the model (renormalised, so unchanged).
    np.testing.assert_allclose(blended[3:], model[3:], atol=1e-9)
    for sl in (slice(0, 3), slice(3, 6)):
        assert abs(blended[sl].sum() - 1.0) < 1e-9
