"""Tests for the Harville PLACE recursion (PLAN.md §1D)."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from hkjc.models.base import group_codes
from hkjc.models.place import harville_place_probs, harville_topk

strengths = st.lists(
    st.floats(min_value=0.01, max_value=20.0, allow_nan=False), min_size=4, max_size=14
)


@given(s=strengths, k=st.integers(min_value=1, max_value=3))
def test_topk_sums_to_k(s: list[float], k: int) -> None:
    # Identity: expected number of runners in the top k == k.
    p = np.asarray(s) / np.sum(s)
    topk = harville_topk(p, k)
    assert np.all(topk >= -1e-9) and np.all(topk <= 1.0 + 1e-9)
    assert np.isclose(topk.sum(), k, atol=1e-6)


@given(s=strengths)
def test_topk_is_monotonic_in_strength(s: list[float]) -> None:
    p = np.asarray(s) / np.sum(s)
    ordered = np.sort(p)
    assume(ordered[-1] > ordered[-2] + 1e-6)  # unique favourite (avoid tie ambiguity)
    topk = harville_topk(p, 3)
    # The strongest runner is the most likely to place.
    assert int(np.argmax(topk)) == int(np.argmax(p))


def test_topk_k1_equals_win_prob() -> None:
    p = np.array([0.5, 0.3, 0.2])
    assert np.allclose(harville_topk(p, 1), p)


def test_vectorised_matches_per_race() -> None:
    p = np.array([0.4, 0.3, 0.2, 0.1, 0.5, 0.5])
    race = np.array([0, 0, 0, 0, 1, 1])
    codes, n = group_codes(race)
    nplaces = np.array([3, 2], dtype=np.int64)
    out = harville_place_probs(p, codes, n, nplaces)
    np.testing.assert_allclose(out[:4], harville_topk(p[:4], 3))
    np.testing.assert_allclose(out[4:], harville_topk(p[4:], 2))


def test_zero_places_returns_zero() -> None:
    p = np.array([0.6, 0.4])
    assert np.allclose(harville_topk(p, 0), 0.0)
