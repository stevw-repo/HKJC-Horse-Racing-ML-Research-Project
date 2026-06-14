"""The ``_scrape_manifest`` table (idempotency + provenance), backed by DuckDB.

A URL recorded here with a frozen (final) target lets the crawler skip it on re-runs,
which is what makes a re-crawl fetch zero new rows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from hkjc.common.time import now_hkt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS _scrape_manifest (
    url TEXT PRIMARY KEY,
    source TEXT,
    fetched_at TIMESTAMP,
    content_hash TEXT,
    status INTEGER,
    n_rows INTEGER
);
"""


class Manifest:
    """Thin wrapper over a DuckDB connection holding the scrape manifest."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con: Any = duckdb.connect(str(db_path))
        self.con.execute(_SCHEMA)

    def has(self, url: str) -> bool:
        """Return True if ``url`` has already been recorded."""
        row = self.con.execute("SELECT 1 FROM _scrape_manifest WHERE url = ?", [url]).fetchone()
        return row is not None

    def record(
        self,
        url: str,
        source: str,
        content_hash: str,
        status: int,
        n_rows: int,
        fetched_at: datetime | None = None,
    ) -> None:
        """Insert or update a manifest entry for ``url``."""
        self.con.execute(
            """
            INSERT INTO _scrape_manifest VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (url) DO UPDATE SET
                source = excluded.source,
                fetched_at = excluded.fetched_at,
                content_hash = excluded.content_hash,
                status = excluded.status,
                n_rows = excluded.n_rows
            """,
            [url, source, fetched_at or now_hkt(), content_hash, status, n_rows],
        )

    def count(self) -> int:
        """Return the number of recorded URLs."""
        row = self.con.execute("SELECT count(*) FROM _scrape_manifest").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
