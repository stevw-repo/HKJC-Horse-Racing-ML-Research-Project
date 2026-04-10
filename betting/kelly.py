# betting/kelly.py
"""
Fractional Kelly criterion stake sizing.
"""
from config import KELLY_FRACTION, MAX_BET_FRACTION


def kelly_stake_fraction(
    model_prob:   float,
    decimal_odds: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_fraction:   float = MAX_BET_FRACTION,
) -> float:
    """
    Return the fraction of bankroll to bet.

    Full Kelly: f = (b*p - q) / b
      b = decimal_odds - 1
      p = model_prob
      q = 1 - p

    Returns KELLY_FRACTION * full_kelly, capped at MAX_BET_FRACTION.
    Returns 0.0 if the calculated stake is non-positive (no edge).
    """
    if decimal_odds <= 1.0 or model_prob <= 0.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(kelly_fraction * full_kelly, max_fraction)


def kelly_stake_hkd(
    model_prob:    float,
    decimal_odds:  float,
    bankroll_hkd:  float,
    kelly_fraction: float = KELLY_FRACTION,
    max_fraction:   float = MAX_BET_FRACTION,
) -> float:
    """Return the absolute HKD stake amount."""
    fraction = kelly_stake_fraction(model_prob, decimal_odds,
                                    kelly_fraction, max_fraction)
    return fraction * bankroll_hkd