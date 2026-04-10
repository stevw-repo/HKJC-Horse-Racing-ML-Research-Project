# betting/bet_types.py
"""
Payout calculations for the four in-scope HKJC pool types.
All calculations use a HKD 10 unit stake (HKJC standard).
"""
from config import VALID_POOL_TYPES

UNIT_STAKE = 10.0  # HKD


def win_payout(dividend: float, stake: float) -> float:
    """Gross return for a WIN bet.  Dividend is per HKD 10 unit."""
    if dividend is None or dividend <= 0:
        return 0.0
    return stake * (dividend / UNIT_STAKE)


def place_payout(dividend: float, stake: float) -> float:
    """Gross return for a PLACE (PLA) bet."""
    return win_payout(dividend, stake)


def qin_payout(dividend: float, stake: float) -> float:
    """Gross return for a QIN (Quinella) bet.
    Pays if the selected pair finish 1st and 2nd in any order."""
    return win_payout(dividend, stake)


def qpl_payout(dividend: float, stake: float) -> float:
    """Gross return for a QPL (Quinella Place) bet.
    Pays if the selected pair both finish in the top 3 in any order."""
    return win_payout(dividend, stake)


def payout(pool_type: str, dividend: float, stake: float) -> float:
    """Dispatch payout calculation by pool type."""
    if pool_type not in VALID_POOL_TYPES:
        raise ValueError(
            f"Pool type '{pool_type}' not in scope. Valid: {VALID_POOL_TYPES}"
        )
    mapping = {
        "WIN": win_payout,
        "PLA": place_payout,
        "QIN": qin_payout,
        "QPL": qpl_payout,
    }
    return mapping[pool_type](dividend, stake)


def calculate_roi(gross_return: float, stake: float) -> float:
    """Return (gross_return - stake) / stake."""
    if stake <= 0:
        return 0.0
    return (gross_return - stake) / stake