"""Property + unit tests for the pari-mutuel dividend math (PLAN.md §1I)."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from hkjc.backtest import pari_mutuel as pm
from hkjc.common.config import PlaceRule

RULES = [
    PlaceRule(min_runners=7, places=3),
    PlaceRule(min_runners=5, places=2),
    PlaceRule(min_runners=1, places=0),
]

pos = st.floats(min_value=1.0, max_value=1e7, allow_nan=False, allow_infinity=False)
takeout_st = st.floats(min_value=0.0, max_value=0.3, allow_nan=False)


def test_places_by_field_size() -> None:
    for n in range(0, 21):
        expected = 3 if n >= 7 else 2 if n >= 5 else 0
        assert pm.places_for_field_size(n, RULES) == expected


@given(amount=st.floats(min_value=0.0, max_value=1e6, allow_nan=False))
def test_round_stake_is_legal_multiple(amount: float) -> None:
    stake = pm.round_stake(amount, unit=10.0, min_bet=10.0)
    assert stake >= 0.0
    assert math.isclose(stake % 10.0, 0.0, abs_tol=1e-9) or math.isclose(
        stake % 10.0, 10.0, abs_tol=1e-9
    )
    if amount < 10.0:
        assert stake == 0.0
    else:
        assert abs(stake - amount) <= 5.0 + 1e-9  # within half a unit


@given(pool=pos, amount=pos, takeout=takeout_st)
def test_win_dividend_conserves_pool(pool: float, amount: float, takeout: float) -> None:
    # Total paid to winning backers == net pool (single winner).
    div = pm.win_dividend(pool, amount, takeout, unit=10.0)
    paid = div * amount / 10.0
    assert math.isclose(paid, pm.net_pool(pool, takeout), rel_tol=1e-9)


@given(pool=pos, amount=pos, takeout=takeout_st)
def test_dead_heat_halves_dividend(pool: float, amount: float, takeout: float) -> None:
    solo = pm.win_dividend(pool, amount, takeout, n_winners=1)
    dh = pm.win_dividend(pool, amount, takeout, n_winners=2)
    assert math.isclose(dh, solo / 2.0, rel_tol=1e-9)


@given(pool=pos, a1=pos, a2=pos, takeout=takeout_st)
def test_win_dividend_monotonic_in_amount(
    pool: float, a1: float, a2: float, takeout: float
) -> None:
    # More money on the winner -> smaller dividend.
    lo, hi = sorted((a1, a2))
    if math.isclose(lo, hi):
        return
    assert pm.win_dividend(pool, lo, takeout) >= pm.win_dividend(pool, hi, takeout)


@given(pool=pos, amount=pos, takeout=takeout_st)
def test_place_dividend_conserves_position_share(
    pool: float, amount: float, takeout: float
) -> None:
    div = pm.place_dividend(pool, amount, n_places=3, takeout=takeout)
    paid = div * amount / 10.0
    assert math.isclose(paid, pm.net_pool(pool, takeout) / 3.0, rel_tol=1e-9)


@given(
    pool=pos,
    frac=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    stake=pos,
    takeout=takeout_st,
)
def test_pool_impact_dilutes_winning_dividend(
    pool: float, frac: float, stake: float, takeout: float
) -> None:
    # The amount already on the winner is a share of the pool (amount <= pool, physical).
    # Our own winning stake joins the pool and the winner side -> the dividend can only dilute.
    amount = pool * frac
    base = pm.win_dividend(pool, amount, takeout)
    impacted = pm.pool_impact_dividend(pool, amount, stake, takeout, won=True)
    assert impacted <= base + 1e-9


def test_bet_profit() -> None:
    assert pm.bet_profit(10.0, 35.0, won=True) == 25.0  # $10 at div 35/HK$10 -> +25
    assert pm.bet_profit(10.0, 35.0, won=False) == -10.0
    assert pm.profit_at_decimal_odds(10.0, 3.5, won=True) == 25.0
    assert pm.profit_at_decimal_odds(10.0, 3.5, won=False) == -10.0
