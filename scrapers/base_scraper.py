# scrapers/base_scraper.py
import logging
import time
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup

from config import (BACKOFF_FACTOR, GRAPHQL_URL, HEADERS,
                    MAX_RETRIES, REQUEST_DELAY_SEC, REQUEST_TIMEOUT_SEC)

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base providing HTTP session, retry logic, and HTML/GraphQL helpers."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_html(self, url: str, params: dict = None) -> BeautifulSoup:
        resp = self._get_with_retry(url, params=params)
        return BeautifulSoup(resp.text, "lxml")

    def post_graphql(self, query: str, variables: dict) -> dict:
        resp = self._post_with_retry(GRAPHQL_URL, json={"query": query,
                                                         "variables": variables})
        return resp.json()

    # ── Retry wrappers ────────────────────────────────────────────────────────

    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        return self._request_with_retry("GET", url, **kwargs)

    def _post_with_retry(self, url: str, **kwargs) -> requests.Response:
        return self._request_with_retry("POST", url, **kwargs)

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY_SEC)
                resp = self.session.request(
                    method, url, timeout=REQUEST_TIMEOUT_SEC, **kwargs
                )
                if resp.status_code in (429, 503):
                    wait = REQUEST_DELAY_SEC * (BACKOFF_FACTOR ** attempt)
                    logger.warning("HTTP %d — back-off %.1fs", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                wait = REQUEST_DELAY_SEC * (BACKOFF_FACTOR ** attempt)
                logger.warning("Attempt %d/%d failed: %s. Retry in %.1fs",
                               attempt + 1, MAX_RETRIES, exc, wait)
                time.sleep(wait)
        raise RuntimeError(
            f"All {MAX_RETRIES} retries failed for {method} {url}"
        ) from last_exc

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def scrape(self, **kwargs):
        ...