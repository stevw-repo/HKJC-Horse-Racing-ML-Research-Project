"""Pari-mutuel dividend mathematics (PLAN.md §1F, §1I).

Pure, side-effect-free functions, property-tested in ``tests/``. Two distinct uses:

* **Decision/payout in the backtest** uses the *stored final dividends* (the honest pool
  truth -- everyone is paid the closing dividend regardless of when they bet), via
  :func:`bet_profit`.
* **Reconstruction / pool-impact / M7** uses :func:`win_dividend` / :func:`place_dividend`
  to compute a dividend from pool sizes (incl. dead-heat splitting and our own stake's
  dilution), for when pool totals are available.

Money convention: a *dividend* is HK$ returned per winning ``unit`` (HK$10 in HK), and it
*includes* the stake. ``win_odds`` (decimal SP) ~= dividend / unit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from hkjc.common.config import PlaceRule


def places_for_field_size(field_size: int, rules: Sequence[PlaceRule]) -> int:
    """Paid place positions for a field size, per the configured rules (3@7+, 2@5-6, 0@<5)."""
    best = 0
    best_threshold = -1
    for rule in rules:
        if field_size >= rule.min_runners and rule.min_runners > best_threshold:
            best = rule.places
            best_threshold = rule.min_runners
    return best


def round_stake(amount: float, unit: float = 10.0, min_bet: float = 10.0) -> float:
    """Round a desired stake to a legal multiple of ``unit``; below ``min_bet`` -> 0.

    Always returns an exact multiple of ``unit`` (the HK$10 granularity that makes small
    Kelly edges round to 0/1 units -- a headline effect, PLAN.md §1F).
    """
    if amount < min_bet:
        return 0.0
    units = math.floor(amount / unit + 0.5)
    stake = units * unit
    return stake if stake >= min_bet else 0.0


def net_pool(gross_pool: float, takeout: float) -> float:
    """Pool remaining for distribution after takeout."""
    return gross_pool * (1.0 - takeout)


def win_dividend(
    gross_pool: float,
    amount_on_winner: float,
    takeout: float,
    n_winners: int = 1,
    unit: float = 10.0,
) -> float:
    """WIN dividend per ``unit``. With a ``n_winners``-way dead-heat the net pool is split
    equally between the dead-heating horses' backers (conserves the pool)."""
    if amount_on_winner <= 0 or n_winners < 1:
        return 0.0
    share = net_pool(gross_pool, takeout) / n_winners
    return unit * share / amount_on_winner


def place_dividend(
    gross_place_pool: float,
    amount_on_horse: float,
    n_places: int,
    takeout: float,
    n_dead_heat: int = 1,
    unit: float = 10.0,
) -> float:
    """PLACE dividend per ``unit`` for one placed horse. The net place pool is divided into
    ``n_places`` equal position-shares; a position shared by ``n_dead_heat`` horses splits
    its share further."""
    if amount_on_horse <= 0 or n_places < 1 or n_dead_heat < 1:
        return 0.0
    share = net_pool(gross_place_pool, takeout) / n_places / n_dead_heat
    return unit * share / amount_on_horse


def pool_impact_dividend(
    gross_pool: float,
    amount_on_winner: float,
    my_stake: float,
    takeout: float,
    won: bool,
    n_winners: int = 1,
    unit: float = 10.0,
) -> float:
    """Dividend after adding our own ``my_stake`` to the pool (and, if ``won``, to the
    winner side) -- the pool-dilution effect (~0 at HK$1,000; material at HK$100k)."""
    new_gross = gross_pool + my_stake
    new_winner = amount_on_winner + (my_stake if won else 0.0)
    return win_dividend(new_gross, new_winner, takeout, n_winners=n_winners, unit=unit)


def bet_profit(stake: float, dividend: float, won: bool, unit: float = 10.0) -> float:
    """Net profit of a flat bet given the realised per-``unit`` dividend.

    ``dividend`` includes the stake, so gross return = ``stake * dividend / unit`` and profit
    is that minus the stake (or ``-stake`` if the bet lost).
    """
    if not won:
        return -stake
    return stake * dividend / unit - stake


def profit_at_decimal_odds(stake: float, decimal_odds: float, won: bool) -> float:
    """Net profit of a flat bet priced at decimal odds (SP). Payout = stake*odds incl. stake."""
    if not won:
        return -stake
    return stake * (decimal_odds - 1.0)
