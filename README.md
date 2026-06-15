# HKJC Horse-Racing ML Research Platform

A **local, single-user research platform** that predicts HKJC **WIN + PLACE** probabilities
for Sha Tin (`ST`) and Happy Valley (`HV`) races, detects value against the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> It is a methodology / research sandbox first, with eventual real-money use — and it
> **recommends only; it never places bets.**

See [`PLAN.md`](PLAN.md) for the authoritative build plan (critique, phased roadmap,
data scope, schema, scraper strategy) and [`CLAUDE.md`](CLAUDE.md) for the working
conventions and current state of this repo.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Foundations: env, config, logging, storage layout, tooling, CLI | ✅ done |
| **M1** | Incremental scraper + storage | ✅ built — run `hkjc backfill` to populate |
| M2 | Features + baseline conditional-logit + honest backtest | ◻ next |
| M3 | Model zoo + calibration + market blend | — |
| M4 | NLP track (English) | — |
| M5 | Risk / staking sweeps | — |
| M6 | UI (React + FastAPI) | — |
| M7 | Live ops + odds logging | — |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # create .venv and install pinned deps from uv.lock
uv run hkjc --help      # CLI entry point
uv run hkjc doctor      # resolved config, paths, locked data scope
uv run pytest           # test suite
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy             # strict type-check
uv run pre-commit install   # one-time: git hooks
```

## Data layer (M1)

Scrape HKJC + open data into a local DuckDB + partitioned-Parquet lake. Every fetch is
recorded in a `_scrape_manifest`, so re-runs are **idempotent** (frozen past pages fetch
zero rows). Politeness: concurrency-capped, rate-limited, retried, cached.

```bash
# Historical results backfill (fixtures calendar enumerates meetings back to ~2006).
uv run hkjc backfill --since 2024-09-01      # bound it; omit --since for full history (~2006)
uv run hkjc scrape --date 2026-06-03         # a single meeting

# Profiles + alternative sources (run after results so ids/dates are known).
uv run hkjc scrape-horses                    # horse bio + form records
uv run hkjc scrape-people                    # jockey + trainer season stats
uv run hkjc scrape-weather --since-year 2006 # HKO daily-climate temperatures
uv run hkjc scrape-trials                    # barrier trials
uv run hkjc scrape-trackwork                 # trackwork (gallop) records
uv run hkjc scrape-holidays                  # HK public holidays

uv run hkjc data-health                      # coverage report (meetings/races/rows by season)
```

**What's collected:** results (finish order, SP/win-odds, running positions, sectionals,
full WIN→QUARTET dividends), going + rail position, horse profiles (locked bio block +
per-run form), jockey/trainer season stats, HKO daily temperatures, barrier trials,
trackwork, public holidays. See `PLAN.md` §0 for the locked scope (alt sources
3,4,5,7,9,11,14; #6 vet-list and #8 gear-change declarations were dropped as forward-only).

> A full ~20-season backfill is a multi-hour crawl (~1,600 meetings × ~11 pages). It's
> idempotent and cached, so bound the first run with `--since` or run it overnight.

## Configuration

YAML under [`config/`](config/), loaded and validated via `pydantic-settings`
(`src/hkjc/common/config.py`). `HKJC_`-prefixed env vars override YAML (nested via `__`,
e.g. `HKJC_RISK__BANKROLL=5000`). `config/local.yaml` is a gitignored override layer.

## Layout

```
config/            # YAML: paths, sources, features, risk, backtest, models
src/hkjc/
  common/          # config, logging, keys, time (HKT)
  data/            # scrape · parse · store (DuckDB+Parquet) · weather · holidays · live(M7)
  features/ models/ risk/ backtest/ experiments/ api/   # M2+ (skeletons)
  cli.py           # Typer entry point (`hkjc`)
tests/             # pytest suite
fixtures/          # checked-in HTML/JSON for offline parser tests
data/              # gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```
