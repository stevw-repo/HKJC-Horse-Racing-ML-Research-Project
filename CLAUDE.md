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
  race_off_time`).
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

**Jockey/trainer profiles** are built too (`parse_person_profile` → `people` view): one
parser for both roles capturing current-season strike-rate inputs (wins/2-3-4ths, rides or
runners, win %, stakes).

**HKO weather** (`data/weather/hko.py` → `weather` view): daily mean/max/min temperature
per station (full history is one request each), mapped HKO→HV and SHA→ST. Only temperature
is exposed per-station (no humidity/rainfall). Note HKO's **current-month lag** — daily
climate publishes after the month completes, so very recent meetings have no weather yet.

CLI: `hkjc scrape --date YYYY-MM-DD [--force]`, `hkjc backfill [--limit N] [--since ...]`,
`hkjc scrape-horses [--limit N]`, `hkjc scrape-people [--limit N]`,
`hkjc scrape-weather [--since-year Y]`, `hkjc scrape-holidays`,
`hkjc scrape-trials [--limit N]`, `hkjc scrape-trackwork [--limit N]`, `hkjc data-health`.
**Public holidays** (#14) are ingested from gov.hk open data (`data/holidays.py` →
`public_holidays` view); the feed is served with a BOM and spans only ~current +/- 1 year.
**Barrier trials** (#4, `parse/trials.py` → `barrier_trials` view): per-batch runs from
`btresult?date=YYYY/MM/DD` (param is `date`, not `racedate`); trials run at ST/HV/Conghua
and do **not** link jockey/trainer ids, so those are stored as names.
**Trackwork** (#5, `data/trackwork.py` → `trackwork` view): a paginated JSON endpoint
(`/racing/information/json/TrackworkOneDayRecords/<YYYYMMDD>1E.aspx?PageNum=N`) discovered
via the browser — the `trackworksearch` page is JS-driven, so the data isn't in its HTML.
Records carry names, not ids.
Meeting URLs (`resultsall?racedate=`) discover venue + race count without a `Racecourse`
param; per-race URLs are `localresults?...&RaceNo=N`. Re-running a **frozen** (past) meeting
recorded in the manifest fetches 0 rows; horse profiles are **mutable** and refetched.

**Historical enumeration (RESOLVED 2026-06-15):** the results `selectId` dropdown only lists
~2 recent seasons, but the **fixtures calendar** covers the full history —
`fixture?calyear=Y&calmonth=M` (`parse/fixtures.py`, `<td class="calendar">` day cells) lists
meeting days back to ~2006. `pipeline.list_fixture_dates` drives `hkjc backfill` (default
start ~2006-09; `--since` to bound). Caveat: on old result pages HKJC only hyperlinks
**currently-active** jockeys/trainers, so `jockey_code`/`trainer_code` are partial for old
meetings — the always-present `jockey_name`/`trainer_name` text is captured alongside.

**Going/rail** (#3): `going` and `rail` are first-class on the `races` view — going comes
straight from the results meta, and `rail` is parsed from the course token (`TURF - "C"
Course` → `C`; AWT has none). HKJC publishes no going-stick / separate rail page, so this
is the complete #3 signal from results.

**Scope change 2026-06-15:** #6 vet-list and #8 gear-change declarations were **dropped**
(forward-only / tied to declared starters; not historically backfillable). Per-run gear is
still captured via horse-form. Race cards are **kept** (needed for M7 race-day predictions).

**Remaining M1:** the historical data layer is complete (results, profiles, weather,
holidays, trials, trackwork, going/rail). Left: **forward race-card capture** (with M7
live-ops, when a meeting is upcoming) and the **full backfill** + a richer coverage/gap
report. The enabled alternative sources are now: going/rail #3, barrier trials #4,
trackwork #5, sectional archive #7, racing news #9 (M4 NLP), pedigree #11, holidays #14.

## Milestone status

M0 (foundations) and **M1 (scraper + storage) are implementation-complete**: every locked
source has a parser + DuckDB view + offline fixture test, idempotency is proven, historical
enumeration reaches ~2006, and `hkjc data-health` reports coverage. The one operational
step left is **running the full backfill** (`hkjc backfill` — a multi-hour crawl the user
kicks off; nothing else depends on it being run first). Forward race-card capture is parked
with M7. Each milestone's exit criterion is in PLAN.md §2 — treat it as the definition of done.

## Next: M2 (features + baseline + honest backtest)

Start here when resuming. Build against the DuckDB views the scraper populates
(`races`, `results`, `dividends`, `horses`, `horse_form`, `people`, `weather`,
`barrier_trials`, `trackwork`, `public_holidays`). M2 scope (PLAN.md §2):
- **As-of feature store** in `features/` — every feature computable from `event_time ≤
  race_off_time`; honor the **market wall** (SP is market data) and **lagged-text** rules.
- **Leakage canary** (shuffle/future sentinel) that must score ≈0 — the highest-ROI test.
- **PL-strength conditional-logit baseline** (`models/base.py` = `ProbabilityModel` →
  per-runner strength vector; `models/logit/`); Harville PLACE.
- **Walk-forward backtest** in `backtest/` with takeout (~17.5%), HK$10 rounding,
  pool-impact, place-count-by-field-size + dead-heat logic; bootstrap CIs; **two ROIs**
  (model-only + market-blended). Property-test the dividend/Kelly/place math (hypothesis).
- Leaf model subpackages and `models/base.py` do **not** exist yet — create them in M2.
  Add ML deps (numpy/scipy/scikit-learn/polars-as-needed) via `uv add` at this milestone.
- Exit criterion: end-to-end walk-forward backtest of the baseline with honest ROI/Sharpe
  + calibration plots + CIs, and the leakage canary scoring ≈0.
