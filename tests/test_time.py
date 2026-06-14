"""Tests for HKT time helpers (also verifies tzdata is available on Windows)."""

from __future__ import annotations

from hkjc.common.time import now_hkt


def test_now_hkt_is_tz_aware_utc_plus_8() -> None:
    dt = now_hkt()
    offset = dt.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 8 * 3600
