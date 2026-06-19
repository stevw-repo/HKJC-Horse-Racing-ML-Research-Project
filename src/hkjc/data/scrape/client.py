"""Polite async HTTP fetcher: concurrency-capped, rate-limited, retrying, cached.

The on-disk cache (one file per URL) keeps re-runs off the network; the scrape manifest
(:mod:`hkjc.data.store.manifest`) provides data-level idempotency on top of it.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hkjc.common.time import now_hkt

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 hkjc-research",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The outcome of fetching one URL."""

    url: str
    status: int
    text: str
    content_hash: str
    from_cache: bool
    fetched_at: datetime


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{_sha1(url)}.html"


class Fetcher:
    """Async fetcher with a shared rate limit, concurrency cap, retries, and cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        rate_per_sec: float = 5.0,
        concurrency: int = 4,
        timeout: float = 25.0,
        use_cache: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.use_cache = use_cache
        self._rate_per_sec = max(rate_per_sec, 0.1)
        self._concurrency = concurrency

    def _read_cache(self, url: str) -> FetchResult | None:
        path = _cache_path(self.cache_dir, url)
        if self.use_cache and path.is_file():
            text = path.read_text(encoding="utf-8")
            return FetchResult(url, 200, text, _sha1(text), from_cache=True, fetched_at=now_hkt())
        return None

    def _write_cache(self, url: str, text: str) -> None:
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _cache_path(self.cache_dir, url).write_text(text, encoding="utf-8")

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        limiter: AsyncLimiter,
        url: str,
    ) -> FetchResult:
        cached = self._read_cache(url)
        if cached is not None:
            return cached
        async with sem, limiter:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception_type(httpx.HTTPError),
                reraise=True,
            ):
                with attempt:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    text = resp.text
                    self._write_cache(url, text)
                    return FetchResult(
                        url,
                        resp.status_code,
                        text,
                        _sha1(text),
                        from_cache=False,
                        fetched_at=now_hkt(),
                    )
        raise RuntimeError("unreachable")  # pragma: no cover

    async def afetch_many(self, urls: list[str]) -> list[FetchResult]:
        sem = asyncio.Semaphore(self._concurrency)
        limiter = AsyncLimiter(self._rate_per_sec, 1.0)
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, timeout=self.timeout, follow_redirects=True
        ) as client:
            return await asyncio.gather(
                *(self._fetch_one(client, sem, limiter, url) for url in urls)
            )

    def fetch_many(self, urls: list[str]) -> list[FetchResult]:
        """Fetch many URLs concurrently (sync entry point)."""
        return asyncio.run(self.afetch_many(urls))

    def fetch(self, url: str) -> FetchResult:
        """Fetch a single URL (sync entry point)."""
        return self.fetch_many([url])[0]
