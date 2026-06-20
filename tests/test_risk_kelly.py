"""Property + unit tests for Kelly sizing and staking policies (M5, PLAN.md §2 M5).

The headline correctness check cross-validates the closed-form correlated Kelly solver against
a generic SLSQP optimiser on the *same* ``E[log wealth]`` objective: since that objective is
concave, the two must agree on the optimum value.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from hkjc.risk import kelly, staking

prob = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
odds = st.floats(min_value=1.05, max_value=40.0, allow_nan=False, allow_infinity=False)


@st.composite
def race(draw: st.DrawFn, max_n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A within-race draw: win probs summing to <1, with decimal odds per runner."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    raws = draw(st.lists(st.floats(0.05, 1.0), min_size=n, max_size=n))
    scale = draw(st.floats(0.5, 0.98))
    total = sum(raws)
    p = np.array([r / total * scale for r in raws], dtype=np.float64)
    b = np.array(draw(st.lists(odds, min_size=n, max_size=n)), dtype=np.float64)
    return p, b


def _scipy_optimum(p: np.ndarray, b: np.ndarray) -> float:
    """Max E[log wealth] found by a generic concave optimiser (the reference value)."""
    n = p.size

    def neg(f: np.ndarray) -> float:
        return -kelly.expected_log_wealth(f, p, b)

    cons = [{"type": "ineq", "fun": lambda f: 1.0 - float(f.sum()) - 1e-9}]
    bounds = [(0.0, 1.0)] * n
    best = 0.0  # betting nothing is always feasible -> E[log 1] = 0
    for level in (0.0, 0.02, 0.08):
        x0 = np.full(n, level)
        if x0.sum() >= 1.0:
            continue
        res = minimize(
            neg, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12}
        )
        if res.success:
            best = max(best, -float(res.fun))
    return best


# --------------------------------------------------------------------------- #
# Single-bet Kelly
# --------------------------------------------------------------------------- #
@given(p=prob, b=odds)
def test_kelly_fraction_identity(p: float, b: float) -> None:
    f = kelly.kelly_fraction(p, b)
    assert 0.0 <= f <= 1.0
    if p * b > 1.0:
        assert math.isclose(f, min((p * b - 1.0) / (b - 1.0), 1.0), rel_tol=1e-9, abs_tol=1e-12)
    else:
        assert f == 0.0


def test_kelly_fraction_rejects_garbage() -> None:
    assert kelly.kelly_fraction(0.5, 1.0) == 0.0  # no net odds
    assert kelly.kelly_fraction(float("nan"), 3.0) == 0.0
    assert kelly.kelly_fraction(0.5, float("inf")) == 0.0
    assert kelly.kelly_fraction(0.0, 3.0) == 0.0


# --------------------------------------------------------------------------- #
# Correlated (simultaneous) Kelly
# --------------------------------------------------------------------------- #
@given(p=prob, b=odds)
def test_single_candidate_reduces_to_kelly(p: float, b: float) -> None:
    f = kelly.simultaneous_kelly_win(np.array([p]), np.array([b]))
    assert math.isclose(float(f[0]), kelly.kelly_fraction(p, b), rel_tol=1e-9, abs_tol=1e-12)


@given(data=race())
def test_simultaneous_is_feasible(data: tuple[np.ndarray, np.ndarray]) -> None:
    p, b = data
    f = kelly.simultaneous_kelly_win(p, b)
    assert np.all(f >= 0.0)
    assert float(f.sum()) < 1.0 + 1e-9  # a positive cash reserve is always kept


@given(data=race())
@settings(max_examples=60, deadline=None)
def test_simultaneous_matches_scipy_optimum(data: tuple[np.ndarray, np.ndarray]) -> None:
    p, b = data
    f = kelly.simultaneous_kelly_win(p, b)
    ours = kelly.expected_log_wealth(f, p, b)
    ref = _scipy_optimum(p, b)
    assert ours >= ref - 1e-6  # the closed form is at least as good as the numeric optimum
    assert ours <= ref + 1e-3  # ... and not implausibly better (sanity)


@given(data=race())
def test_correlated_at_least_as_good_as_naive(data: tuple[np.ndarray, np.ndarray]) -> None:
    p, b = data
    naive = kelly.naive_kelly(p, b)
    if float(naive.sum()) >= 1.0:
        return  # naive is infeasible here; nothing to compare
    corr = kelly.simultaneous_kelly_win(p, b)
    assert kelly.expected_log_wealth(corr, p, b) >= kelly.expected_log_wealth(naive, p, b) - 1e-9


def test_no_edge_means_no_bet() -> None:
    # Every runner is -EV (p*b < 1) -> stake nothing.
    p = np.array([0.3, 0.2, 0.1])
    b = np.array([2.0, 3.0, 5.0])
    assert np.all(kelly.simultaneous_kelly_win(p, b) == 0.0)
    assert np.all(kelly.naive_kelly(p, b) == 0.0)


def test_correlated_deploys_more_than_naive_when_hedging() -> None:
    # Two +EV mutually exclusive horses: the hedge lets the exact solver deploy more.
    p = np.array([0.5, 0.4])
    b = np.array([2.5, 3.0])
    naive = kelly.naive_kelly(p, b)
    corr = kelly.simultaneous_kelly_win(p, b)
    assert float(corr.sum()) > float(naive.sum())
    assert kelly.expected_log_wealth(corr, p, b) > kelly.expected_log_wealth(naive, p, b)


def test_optimal_set_can_include_mildly_negative_ev_hedge() -> None:
    # h0 is -EV standalone (p*b = 0.875) yet belongs in the joint optimum as a hedge.
    p = np.array([0.4375, 0.4375])
    b = np.array([2.0, 3.0])
    f = kelly.simultaneous_kelly_win(p, b)
    assert f[0] > 0.0 and f[1] > 0.0
    assert kelly.expected_log_wealth(f, p, b) >= _scipy_optimum(p, b) - 1e-7


# --------------------------------------------------------------------------- #
# Staking policies
# --------------------------------------------------------------------------- #
def test_ev_selection_gate() -> None:
    p = np.array([0.5, 0.4, 0.2])
    b = np.array([2.5, 2.0, 2.0])  # edges: +0.25, -0.20, -0.60
    sel = staking.ev_selection(p, b, ev_threshold=0.05)
    assert sel.tolist() == [True, False, False]


def test_flat_and_fixed_fraction() -> None:
    p = np.array([0.5, 0.1])
    b = np.array([2.5, 2.0])  # only runner 0 is +EV
    flat = staking.StakingConfig(method="flat", flat_stake=10.0)
    assert staking.desired_stakes(p, b, 1000.0, flat).tolist() == [10.0, 0.0]
    frac = staking.StakingConfig(method="fixed_fraction", fixed_fraction=0.02)
    assert staking.desired_stakes(p, b, 1000.0, frac).tolist() == [20.0, 0.0]


def test_fractional_kelly_scales_full() -> None:
    p = np.array([0.5, 0.4])
    b = np.array([2.5, 3.0])
    full = staking.StakingConfig(method="kelly_full", correlated=True, kelly_lambda=1.0)
    half = staking.StakingConfig(method="kelly_fractional", correlated=True, kelly_lambda=0.5)
    s_full = staking.desired_stakes(p, b, 1000.0, full)
    s_half = staking.desired_stakes(p, b, 1000.0, half)
    assert np.allclose(s_half, 0.5 * s_full)


def test_kelly_method_matches_solver() -> None:
    p = np.array([0.5, 0.4])
    b = np.array([2.5, 3.0])
    cfg = staking.StakingConfig(method="kelly_full", correlated=True)
    expected = kelly.simultaneous_kelly_win(p, b) * 1000.0
    assert np.allclose(staking.desired_stakes(p, b, 1000.0, cfg), expected)


@given(
    stakes=st.lists(st.floats(0.0, 500.0, allow_nan=False), min_size=1, max_size=8),
    cap=st.floats(10.0, 200.0, allow_nan=False),
)
def test_scale_to_cap_respects_cap(stakes: list[float], cap: float) -> None:
    arr = np.array(stakes, dtype=np.float64)
    scaled = staking.scale_to_cap(arr, cap)
    assert float(scaled.sum()) <= cap + 1e-6
    if float(arr.sum()) <= cap:
        assert np.allclose(scaled, arr)  # untouched when already under the cap


@given(stakes=st.lists(st.floats(0.0, 1000.0, allow_nan=False), min_size=1, max_size=8))
def test_round_stakes_are_legal(stakes: list[float]) -> None:
    rounded = staking.round_stakes(np.array(stakes), unit=10.0, min_bet=10.0)
    for s in rounded:
        assert s == 0.0 or math.isclose(s % 10.0, 0.0, abs_tol=1e-9)


def test_unknown_method_raises() -> None:
    try:
        staking.StakingConfig(method="martingale")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown staking method")
