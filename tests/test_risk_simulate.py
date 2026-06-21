"""Tests for the rebate rule and the day-ordered bankroll simulator (M5, PLAN.md §2 M5)."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hkjc.risk.rebate import RebateRule
from hkjc.risk.simulate import RacePanel, simulate_path
from hkjc.risk.staking import StakingConfig


def _panel(
    p_win: list[float],
    b_win: list[float],
    won: list[float],
    win_div: list[float],
) -> RacePanel:
    """A WIN-only race panel (PLACE inputs zeroed out so no place bet is taken)."""
    n = len(p_win)
    return RacePanel(
        p_win=np.array(p_win, dtype=np.float64),
        b_win=np.array(b_win, dtype=np.float64),
        won=np.array(won, dtype=np.float64),
        win_div=np.array(win_div, dtype=np.float64),
        p_place=np.zeros(n, dtype=np.float64),
        b_place=np.zeros(n, dtype=np.float64),
        placed=np.zeros(n, dtype=np.float64),
        place_div=np.full(n, np.nan, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Rebate rule
# --------------------------------------------------------------------------- #
@given(
    losing=st.floats(0.0, 1e6, allow_nan=False),
    rate=st.floats(0.0, 0.5, allow_nan=False),
)
def test_rebate_monotone_and_thresholded(losing: float, rate: float) -> None:
    rule = RebateRule(rate=rate, threshold=10000.0)
    credit = rule.credit(losing)
    assert credit >= 0.0
    assert rule.triggered(losing) == (losing > 10000.0)
    if losing <= 10000.0 or rate == 0.0:
        assert credit == 0.0
    assert rule.credit(losing + 1000.0) >= credit - 1e-6  # non-decreasing in losing turnover


def test_rebate_credit_is_rate_times_excess() -> None:
    rule = RebateRule(rate=0.10, threshold=10000.0)
    assert math.isclose(rule.credit(15000.0), 0.10 * 5000.0)
    assert rule.credit(10000.0) == 0.0


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
FLAT = StakingConfig(method="flat", flat_stake=10.0, ev_threshold=0.05)


def test_wealth_is_conserved() -> None:
    # terminal bankroll must equal start + summed profit, exactly, on any card.
    card = [
        [_panel([0.5, 0.5], [3.0, 3.0], [1.0, 0.0], [30.0, math.nan])],
        [_panel([0.6, 0.3], [2.5, 4.0], [0.0, 1.0], [math.nan, 40.0])],
        [_panel([0.4], [3.5], [0.0], [math.nan])],
    ]
    out = simulate_path(card, FLAT, 1000.0, pools=("win",))
    assert math.isclose(
        out.terminal_bankroll, 1000.0 + out.total_profit, rel_tol=1e-9, abs_tol=1e-6
    )


def test_winning_card_grows_bankroll() -> None:
    card = [[_panel([0.5], [3.0], [1.0], [30.0])]]  # +EV, wins, pays 3x
    out = simulate_path(card, FLAT, 1000.0, pools=("win",))
    assert out.n_bets == 1
    assert out.total_profit > 0.0
    assert out.terminal_bankroll > 1000.0


def test_losing_card_drawsdown() -> None:
    card = [[_panel([0.5], [3.0], [0.0], [math.nan])] for _ in range(5)]  # +EV pick, loses daily
    out = simulate_path(card, FLAT, 1000.0, pools=("win",))
    assert out.total_profit < 0.0
    assert out.terminal_bankroll < 1000.0
    assert out.max_drawdown > 0.0


def test_ruin_is_flagged() -> None:
    # A +EV pick that loses every day: with flat HK$100 stakes (re-capped to 10% of a shrinking
    # bankroll) the equity bleeds below the 10% ruin floor over a long losing run.
    heavy = StakingConfig(method="flat", flat_stake=100.0, ev_threshold=0.05)
    card = [[_panel([0.5], [3.0], [0.0], [math.nan])] for _ in range(40)]
    out = simulate_path(card, heavy, 1000.0, pools=("win",), ruin_floor=0.10)
    assert out.ruined is True
    assert out.terminal_bankroll < 100.0  # below 10% of the HK$1,000 start
    assert out.ruin_prob > 0.0


def test_no_positive_ev_means_no_bets() -> None:
    card = [[_panel([0.3, 0.2], [2.0, 3.0], [1.0, 0.0], [20.0, math.nan])]]  # all -EV
    out = simulate_path(card, FLAT, 1000.0, pools=("win",))
    assert out.n_bets == 0
    assert out.total_staked == 0.0
    assert out.terminal_bankroll == 1000.0


def test_per_day_cap_limits_exposure() -> None:
    # Fixed-fraction 50% across two +EV runners would want 100% of bankroll; the 25% day cap
    # (and HK$10 rounding) must hold staked <= 25% of the start-of-day bankroll.
    cfg = StakingConfig(method="fixed_fraction", fixed_fraction=0.5, ev_threshold=0.05)
    card = [[_panel([0.5, 0.5], [3.0, 3.0], [0.0, 0.0], [math.nan, math.nan])]]
    out = simulate_path(card, cfg, 1000.0, pools=("win",))
    assert out.total_staked <= 0.25 * 1000.0 + 1e-9
