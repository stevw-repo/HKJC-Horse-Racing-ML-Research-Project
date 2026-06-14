"""Pipeline idempotency test using a fake fetcher (no network)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from hkjc.common.config import AppConfig, Paths
from hkjc.common.time import HKT
from hkjc.data import pipeline
from hkjc.data.scrape.client import Fetcher, FetchResult
from hkjc.data.store.manifest import Manifest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hkjc"
MEETING_HTML = (FIXTURES / "resultsall_2026-06-03_HV.html").read_text(encoding="utf-8")
RACE_HTML = (FIXTURES / "localresults_2026-06-03_HV_R1.html").read_text(encoding="utf-8")


class _FakeFetcher(Fetcher):
    """Returns canned fixture HTML; counts network calls; never touches the network."""

    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self.network_calls = 0

    def fetch(self, url: str) -> FetchResult:
        self.network_calls += 1
        return FetchResult(
            url, 200, MEETING_HTML, "h", from_cache=False, fetched_at=datetime.now(HKT)
        )

    def fetch_many(self, urls: list[str]) -> list[FetchResult]:
        self.network_calls += len(urls)
        return [
            FetchResult(u, 200, RACE_HTML, "h", from_cache=False, fetched_at=datetime.now(HKT))
            for u in urls
        ]


def test_scrape_meeting_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Make the meeting deterministically "frozen" regardless of the real clock.
    monkeypatch.setattr(pipeline, "now_hkt", lambda: datetime(2026, 7, 1, tzinfo=HKT))

    cfg = AppConfig(paths=Paths(data_root=tmp_path))
    fetcher = _FakeFetcher(tmp_path / "cache")

    with Manifest(cfg.paths.duckdb_path) as manifest:
        first = pipeline.scrape_meeting(
            date(2026, 6, 3), cfg=cfg, fetcher=fetcher, manifest=manifest
        )
        assert first.venue == "HV"
        assert first.races == 9
        assert not first.skipped
        assert fetcher.network_calls == 10  # 1 meeting page + 9 race pages
        assert first.rows["races"] == 9

        # Re-running a frozen, already-stored meeting fetches nothing.
        second = pipeline.scrape_meeting(
            date(2026, 6, 3), cfg=cfg, fetcher=fetcher, manifest=manifest
        )
        assert second.skipped
        assert second.fetched == 0
        assert fetcher.network_calls == 10  # unchanged
