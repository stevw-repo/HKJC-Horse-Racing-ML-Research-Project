"""Time helpers. HKJC races run on Hong Kong Time (HKT, UTC+8, no DST).

All timestamps in the platform are HKT-aware; this module is the single source of
that timezone. Requires the ``tzdata`` package on Windows (declared as a dependency).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

HKT = ZoneInfo("Asia/Hong_Kong")


def now_hkt() -> datetime:
    """Return the current time as an HKT-aware :class:`datetime`."""
    return datetime.now(tz=HKT)
