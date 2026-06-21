"""Live-odds snapshot logger (M7).

Polls the WIN/PLACE pools for a meeting's races and appends snapshots **deduplicated on
``lastUpdateTime``** -- the gateway re-serves the same timestamp until odds actually move, so we
store one batch per genuine refresh. Append-only: the ``live_odds_snapshots`` view accumulates
the intraday series for later market-microstructure study. Logs odds; never places a bet.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date

import duckdb

from hkjc.common.config import AppConfig, get_config
from hkjc.common.time import now_hkt
from hkjc.data.live.graphql import LiveClient
from hkjc.data.store.writer import refresh_views, write_live_odds


def log_odds(
    cfg: AppConfig | None = None,
    *,
    day: date,
    venue: str,
    race_nos: list[int] | None = None,
    interval: float = 30.0,
    rounds: int = 1,
    on_round: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Poll live WIN/PLACE odds ``rounds`` times (``interval`` s apart) and store new snapshots."""
    cfg = cfg or get_config()
    total = 0
    seen: set[tuple[int, str, str | None]] = set()
    with LiveClient() as client:
        if race_nos is None:
            card = client.card(day, venue)
            race_nos = [r.race_no for r in card.races] if card else []
        for cycle in range(rounds):
            snapshot_ts = now_hkt().isoformat(timespec="milliseconds")
            rows: list[dict[str, object]] = []
            for race_no in race_nos:
                for pool in client.odds(day, venue, race_no):
                    key = (race_no, pool.pool_type, pool.last_update_time)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.extend(
                        {
                            "snapshot_ts": snapshot_ts,
                            "race_no": race_no,
                            "pool_type": pool.pool_type,
                            "comb": node.comb,
                            "saddle": node.saddle,
                            "odds_value": node.odds,
                            "odds_drop": node.odds_drop,
                            "hot_fav": node.hot_fav,
                            "status": pool.status,
                            "sell_status": pool.sell_status,
                            "last_update_time": pool.last_update_time,
                        }
                        for node in pool.nodes
                    )
            total += write_live_odds(cfg.paths.raw_dir, day, venue, rows)
            if on_round is not None:
                on_round(cycle, len(rows))
            if cycle < rounds - 1:
                time.sleep(interval)

    con = duckdb.connect(str(cfg.paths.duckdb_path))
    try:
        refresh_views(con, cfg.paths.raw_dir)
    finally:
        con.close()
    return {"snapshots": total, "rounds": rounds}
