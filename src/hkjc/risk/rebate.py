"""HKJC losing-turnover rebate -- the *mechanism*, parameterised (M5, PLAN.md §1F, §5.2).

HKJC refunds a fraction of **losing** turnover once it crosses a per-betline threshold
(``rebate_rate = 0`` at HK$1,000 because the HK$10,000 threshold is never reached there).
The exact published rebate schedule is **not** encoded here -- per the project's golden rules
we do not fabricate financial specifics. Instead :class:`RebateRule` models a configurable
rate applied to losing turnover *above* the threshold, so the sweep can show two honest
things: how often the threshold is crossed at each bankroll (zero at HK$1k, frequent at
HK$100k), and -- under any assumed rate -- how much that would dent the effective takeout.

A "betline" here is one ``(day, pool)`` bucket; the threshold is in HK$ of losing stake.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RebateRule:
    """Configurable losing-turnover rebate. Default rate 0 reproduces the real HK$1k case."""

    rate: float = 0.0
    threshold: float = 10000.0

    def triggered(self, losing_turnover: float) -> bool:
        """Whether losing turnover on a betline has crossed the rebate threshold."""
        return losing_turnover > self.threshold

    def credit(self, losing_turnover: float) -> float:
        """Rebate cash on a betline: ``rate * (losing_turnover - threshold)`` above the
        threshold, else 0. Monotone non-decreasing in losing turnover."""
        if self.rate <= 0.0 or losing_turnover <= self.threshold:
            return 0.0
        return self.rate * (losing_turnover - self.threshold)
