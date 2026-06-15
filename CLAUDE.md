# CLAUDE.md — working guide for this repo

Context for AI assistants (and humans) working on the HKJC horse-racing ML research
platform. Read this plus **[`PLAN.md`](PLAN.md)** (the authoritative build plan) before
making changes. `prompt.md` is the original brief that produced the plan.

## What this is

A **local, single-user** research platform that predicts HKJC **WIN + PLACE**
probabilities for Sha Tin (`ST`) and Happy Valley (`HV`), detects value vs the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> **Hard invariant: it recommends only — it NEVER places bets.** No code path may submit
> a wager to HKJC or anywhere else.

## Golden rules (domain invariants — do not violate)

- **Never place bets.** Output recommendations only.
- **Pari-mutuel, not fixed odds.** WIN/PLACE are pools; displayed odds move until close;
  the final dividend isn't known at bet time. EV is net of **~17.5% takeout** (rebate 0
  at HK$1,000). Model the pool, not fixed odds.
- **Market wall.** SP / win-odds on the results page is the **closing line = market data**.
  Never use it as a fundamental pre-race feature (leakage). `features.groups.market.wall`
  marks this; keep market features separate from fundamentals.
- **Lagged text only.** Stewards' reports / comments-on-running describe the race that just
  happened → only valid as features for a horse's **prior** runs (`text_event_time <
  race_off_time`). Pre-race vet-list may be a feature; post-race vet notes are lagged.
- **As-of features only.** Every feature must be computable from `event_time ≤
  race_off_time`. A leakage canary (M2) must score ≈0 importance and ≈0 backtest ROI.
- **Honest backtest.** Report **two ROIs**: conservative model-only and upper-bound
  market-blended (final-dividend honesty caveat, PLAN.md §1A).
- **Pedigree is HKJC on-site only.** No external pedigree DBs. The external WBRR list is
  not used (but the per-runner `internationalRating` field is captured).
- All times are **HKT** (UTC+8). Use `hkjc.common.time` (needs `tzdata` on Windows).

## Commands

```bash
uv sync                      # install pinned deps + editable package (creates .venv)
uv run hkjc --help           # CLI
uv run hkjc doctor           # show resolved config, paths, locked data scope
uv run pytest                # tests
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy                  # strict type-check
uv run pre-commit install    # one-time: enable git hooks
```

CI (`.github/workflows/ci.yml`) runs ruff + ruff-format-check + mypy + pytest on
ubuntu-latest with `uv sync --frozen`. Keep `uv.lock` committed and in sync.

## Layout

```
config/            YAML config (one top-level key per file): paths, sources, features, risk, backtest, models
src/hkjc/
  common/          config (pydantic-settings), logging (structlog), keys, time (HKT)
  data/            scrape · parse · store (DuckDB+Parquet) · live (GraphQL, M7) · weather (HKO CSV)
  features/        as-of feature store (M2) + nlp/ (English, M4)
  models/          ProbabilityModel zoo: logit (M2); gbm/nn/ensemble/calibrate/place/blend (M3)
  risk/            Kelly variants, caps, legal rounding (M5)
  backtest/        walk-forward engine, pari-mutuel sim, metrics, bootstrap (M2)
  experiments/     MLflow(local) + Optuna + leaderboard (M3)
  api/             FastAPI (M6)
  cli.py           Typer entry point (`hkjc`)
tests/             pytest suite
fixtures/          checked-in sample HTML/JSON for offline parser tests (M1+)
data/              gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```

Note: leaf model subpackages (`models/logit`, `models/gbm`, …) and `models/base.py`
(the `ProbabilityModel` interface emitting a per-runner **Plackett–Luce strength
vector**) are created when M2/M3 land — they don't exist yet.

## Configuration

`src/hkjc/common/config.py` loads every `config/*.yaml`, merges them, and validates into
`AppConfig` (pydantic-settings). Precedence: **explicit kwargs > `HKJC_` env vars
(nested via `__`) > merged YAML**. Example override: `HKJC_RISK__BANKROLL=5000`.
`config/local.yaml` is a gitignored override layer. Use `get_config()` (cached).

Config sub-models use `extra="forbid"` — adding a YAML key requires adding the field to
the model (and vice versa). Tests in `tests/test_config.py` assert the **locked data
scope** (PLAN.md §0); update them if the scope intentionally changes.

## Conventions

- Python 3.12, `from __future__ import annotations` at the top of every module.
- **mypy strict** must pass; **ruff** (line length 100) lints and formats.
- Keep **user-facing CLI strings ASCII** (Windows console can mangle em/en-dashes; ruff
  flags ambiguous dashes via RUF00x). Docstrings/comments may use em-dashes (`—`).
- Add new runtime deps at the milestone that first needs them (don't front-load the whole
  stack); pin via `uv add` so `uv.lock` updates. Dev tools live in the `dev` group.
- Tests live in `tests/`, named `test_*.py`, fully typed (`-> None`). Property tests
  (hypothesis) are reserved for the backtest/Kelly/dividend math (M2+).

## M1 scraper (in progress)

The **results** pipeline is built and pilot-validated. Flow: `data/pipeline.py`
orchestrates `scrape/client.py` (async fetcher: cache + retry + rate-limit + concurrency)
→ `parse/results.py` (per-race page is the source of truth — only it carries Win Odds,
finish time, running positions, and horse/jockey/trainer ids; the meeting `resultsall`
page is abridged, used only for venue + race-count enumeration) → `store/writer.py`
(partitioned raw Parquet + DuckDB views) + `store/manifest.py` (`_scrape_manifest` for
idempotency). Parsed shapes live in `data/models.py`.

**Horse profiles** are also built (`parse/profiles.py` → `store.write_horse_profile`):
the locked bio block (PLAN.md §0) + the form-records table (per-run rating/gear/going/draw,
with a `RaceIndex` that joins to `races`). Form records are how we reconstruct historical
pre-race fundamentals, because **race cards are forward-only** — HKJC removes a meeting's
card once results publish, so cards cannot be backfilled (capture upcoming cards going
forward, for race-day/M7).

CLI: `hkjc scrape --date YYYY-MM-DD [--force]`, `hkjc backfill [--limit N] [--since ...]`,
`hkjc scrape-horses [--limit N]` (profiles all horses seen in results), `hkjc data-health`.
Meeting URLs (`resultsall?racedate=`) discover venue + race count without a `Racecourse`
param; per-race URLs are `localresults?...&RaceNo=N`. Re-running a **frozen** (past) meeting
recorded in the manifest fetches 0 rows; horse profiles are **mutable** and refetched.

**Open item before the full backfill:** the results `selectId` dropdown only lists ~2
seasons (newest 2026, oldest 2024-07-27). Deep historical backfill needs a date-candidate
generator (Wed + weekend race days, probe + record empties) or a season param.

**Remaining M1:** jockey/trainer profiles, enabled pre-race alt sources (#3 going/rail,
#4 trials, #5 trackwork, #6 vet-list, #8 gear, #14 holidays; #11 pedigree is captured via
horse bio), HKO weather CSV, forward race-card capture, then the full backfill + a richer
coverage/gap report.

## Milestone status

M0 (foundations) is **done**. M1 (scraper + storage) is **underway** (results pipeline +
pilot landed; see above). Each milestone has an explicit exit criterion in PLAN.md §2;
treat it as the definition of done.
