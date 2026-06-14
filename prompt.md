## Your role

You are my senior collaborator — quant researcher, ML engineer, and software architect at once. **Your first deliverable is a PLAN, not an implementation.** Do not write the full codebase until I approve the plan. End your response with: (a) a phased roadmap, (b) a proposed repository structure and tech stack, (c) a data schema and scraper strategy, and (d) an explicit "What I need from you" list. Ask clarifying questions before finalizing if anything is ambiguous.

## What I'm building

A research platform that predicts **HKJC local horse races (Sha Tin + Happy Valley only)** and recommends **value bets with principled stake sizing**. It is **both a methodology/research project and something I intend to bet real money with eventually, leaning research-sandbox.** It must let me experiment with and rigorously compare many models, datasets, features, and quant/risk techniques. It recommends *what to bet and how much* only — it must **never place bets.**

Pipeline:
`scrape (incremental) → store → feature-engineer (incl. NLP on text) → train/compare a model zoo → predict WIN & PLACE probabilities → calibrate → blend with market → detect value vs live odds → size stakes (Kelly variants) → backtest honestly → surface in a React + FastAPI UI`

## Confirmed decisions

- **Goal:** research/methodology **and** eventual real-money betting, leaning research-sandbox.
- **Bet types (v1):** **WIN + PLACE only.** Architect the probability core so exotics can be added later (see "Future-proofing").
- **Venues:** Sha Tin (`ST`) + Happy Valley (`HV`) local races only.
- **Historical depth:** **as many seasons as can be cleanly collected** (season ≈ Sep–Jul). Report the earliest reliably-parseable date and ingest from there to present.
- **Data sources:** **official HKJC for race related data, Hong Kong Observatory for weather data and other appropriate sources** (no paid feeds). You should also recommend 15 additional alternative data sources for me to choose from.
- **Text / NLP:** **yes** — NLP on HKJC text (stewards' reports, comments on running, vet records, etc.) as a feature track.
- **Live odds:** **poll the public GraphQL odds API** (spec below) on race days; for backtests use **final dividends** (I have these).
- **Odds-drift:** **not a trainable v1 feature** (no historical intra-betting snapshots exist). BUT the live API exposes a per-runner `oddsDropValue`, so (a) **log live odds snapshots** from day one for future use, and (b) the live `oddsDropValue` may be used as a real-time input/sanity-check at bet time.
- **Modeling:** a **model zoo** behind one interface; **blend model probabilities with public/market-implied probabilities** (tunable weight); support **both** deriving PLACE probabilities from WIN probabilities **and** modeling PLACE directly, then compare.
- **Evaluation:** strict **time-ordered walk-forward** (no shuffling/lookahead); **prioritize ROI / Sharpe-like metrics** (alongside calibration/log-loss).
- **Risk/staking:** sweep **multiple fractional-Kelly settings and multiple exposure caps**; **starting bankroll = HK$1,000** (bankroll is a parameter so larger bankrolls can be *simulated* in backtests).
- **Stack:** **Python everywhere**; **database = your pick (recommended: DuckDB + Parquet raw layer)**; **UI = React + FastAPI**; **local single-user only.**
- **Operations:** **automated race-day scrape with incremental "missing-data-only" fetching**, plus automated predictions for the day's card.
- **Framing:** I accept that ~17.5% Win/Place takeout with **no rebate at HK$1,000** (rebate needs ≥HK$10,000 losing per betline) makes a real edge genuinely hard; **success = sound methodology, good calibration, and honest edge estimation, not guaranteed profit.**

## Environment & constraints

- Windows 11, single user, local. NVIDIA **RTX 4060 Laptop GPU (8 GB VRAM)**, Intel **i9-13900**, **64 GB RAM**.
- Right-size everything: GPU-accelerated gradient boosting and modest tabular NNs are fine; nothing needing >~8 GB VRAM. Dataset is modest (tens of thousands of races); 64 GB RAM is ample.
- Propose and justify every library; prefer mature, well-documented, free/open-source tools.
- Scraper must be **polite and resilient**: rate limiting, exponential backoff, local caching, **idempotent incremental crawling** (never re-fetch stored data).

## Verified HKJC data sources

### A. Server-rendered HTML — scrapeable with plain HTTP (`requests`/`httpx` + parser), no browser

- **Results** (finishing order, horse/jockey/trainer IDs, act./declared weight, draw, running positions, finish time, win odds, **full dividend table for all pools**):
  `https://racing.hkjc.com/en-us/local/information/localresults?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`
  (whole meeting: `.../resultsall?racedate=YYYY/MM/DD&Racecourse=ST|HV`)
- **Race card / entries:** `.../local/information/racecard?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`
- **Sectional times:** `.../local/information/displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N`
- **Horse profile:** `.../local/information/horse?horseid=HK_YYYY_XXXX`
- **Jockey profile:** `.../local/information/jockeyprofile?jockeyid=XXX&Season=Current`
- **Trainer profile:** `.../local/information/trainerprofile?trainerid=XXX&Season=Current`
- **Draw statistics:** `.../local/information/draw` · **Barrier trials:** `.../local/information/btresult` · **Trackwork:** `.../local/information/trackworksearch` · **Fixtures (to enumerate dates):** `.../local/information/fixture`
- **NLP text:** Comments on Running (`.../corunning?Date=YYYYMMDD&RaceNo=N`), Racing Incident Report / stewards' (`.../racereportfull`), Past Incidents Extract (`.../racereportext`), Veterinary Records (`.../veterinaryrecord`, `.../overecord`), Exceptional Factors (`.../exceptionalfactors`), Form Line Report (`.../formline`).
- Static/media assets are served from `consvc.hkjc.com`.

Verify: PLACE pays first 3 with 7+ runners, first 2 with 5–6; handle field-size-dependent place count. Win dividend is per HK$10 unit. Confirm whether older seasons use legacy URL/markup needing a separate parser.

### B. Live odds API (public GraphQL — captured & decoded; ready to build against)

- **Endpoint:** `POST https://info.cld.hkjc.com/graphql/base/` · `Content-Type: application/json`
- **Auth:** none (no cookies/token). Send headers `origin: https://bet.hkjc.com`, `referer: https://bet.hkjc.com/`, and a normal `User-Agent` to avoid being filtered.
- **Venue codes:** `ST`, `HV`. **Date:** `YYYY-MM-DD`.
- During live betting, pool `status` is `OPEN` / `sellStatus` `SELL` and `oddsValue` updates each refresh; **key stored snapshots on `lastUpdateTime`** to dedupe and to measure refresh cadence empirically.
- Polling cadence (recommended default): every ~30s through the final ~10 min; make the bet decision at a fixed cutoff ~90–120s before scheduled `postTime` on the latest snapshot.

**(B1) Live WIN/PLACE odds** — `operationName: "racing"`, variables `{"date":"2026-06-13","venueCode":"ST","raceNo":11,"oddsTypes":["WIN","PLA"]}`:
```graphql
query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id status sellStatus oddsType lastUpdateTime guarantee minTicketCost
      name_en name_ch
      leg { number races }
      cWinSelections { composite name_ch name_en starters }
      oddsNodes { combString oddsValue hotFavourite oddsDropValue bankerOdds { combString oddsValue } }
    }
  }
}
```
`oddsNodes[].combString` = zero-padded saddle number (e.g. "07"); `oddsValue` = decimal odds (string); `oddsDropValue` = recent shortening magnitude (live drift signal); `hotFavourite` = bool.

**(B2) Full card + win odds** — `operationName: "racing"`, variables `{date, venueCode}`. Returns `raceMeetings[].races[].runners[]` with: `id, no` (saddle), `status, name_en/ch, horse{id,code}, barrierDrawNumber, handicapWeight, currentWeight, currentRating, internationalRating, gearInfo, allowance, last6run, finalPosition, deadHeat, winOdds, jockey{code,name_en/ch}, trainer{code,name_en/ch}`. (Also available: `wageringFieldSize, postTime, distance, go_en, raceClass_en` etc. at race/meeting level.)

**(B3) Pool investments (liquidity / dilution)** — `operationName: "racing"`, variables `{date, venueCode, raceNo, oddsTypes:[…]}`. Returns `totalInvestment` + `poolInvs[]{ oddsType, investment, status, sellStatus, lastUpdateTime }`. Use pool size to parameterize the pari-mutuel pool-impact model and as a liquidity feature. (Captured example: WIN ≈ HK$31.6M, PLACE ≈ HK$29.4M for one race — confirming negligible dilution at HK$1,000 stakes.)

Minimal poller example to adapt:
```python
import httpx
H = {"Content-Type":"application/json","Origin":"https://bet.hkjc.com",
     "Referer":"https://bet.hkjc.com/","User-Agent":"Mozilla/5.0"}
QUERY = "..."  # B1 query string above
def poll(date, venue, race_no):
    body = {"operationName":"racing","query":QUERY,
            "variables":{"date":date,"venueCode":venue,"raceNo":race_no,"oddsTypes":["WIN","PLA"]}}
    r = httpx.post("https://info.cld.hkjc.com/graphql/base/", json=body, headers=H, timeout=15)
    r.raise_for_status()
    return r.json()["data"]["raceMeetings"][0]["pmPools"]
```

## Domain realities you MUST design around (do not hand-wave)

1. **Pari-mutuel, not fixed odds.** Win/Place are pools; displayed odds are estimates that move until close; the final dividend isn't known at bet time; my stake dilutes the pool. Model this correctly — not as fixed-odds.
2. **Takeout ≈ 17.5%** on Win/Place (higher on exotics). EV must be net of takeout.
3. **Rebate = 0% for me** (needs ≥HK$10,000 losing per betline). Keep `rebate_rate` configurable (default 0) to study its effect at scale.
4. **Pool impact:** model it for correctness; expect ~zero effect at HK$1,000 (pools ~HK$30M).
5. **Minimum bet HK$10, in HK$10 units.** Round stakes to legal increments. At a HK$1,000 bankroll this is coarse (1% per unit) and will often force over-betting small edges or skipping them — **surface this as a result.**
6. **The public is informative.** Blend model and market-implied probabilities (Benter-style); the blend weight is a tunable, evaluated hyperparameter.

## Data layer & storage

Per source: exact fields, access method, and what you need from me. Build separate **raw** and **processed** layers, **idempotent incremental re-scraping**, and **data versioning**. **Recommended (justify or counter): DuckDB analytical store + partitioned Parquet raw layer**, plus a lightweight table for live-odds snapshot logging on race days. Canonical keys from HKJC IDs (horse `HK_YYYY_XXXX`, jockey/trainer codes, race key = date+course+race-no). Reconcile the GraphQL feed and the HTML results page (same IDs) into one model.

## Feature engineering (strict leakage controls — only pre-race info)

Horse/jockey/trainer form and strike rates (overall and split by course/distance/going/class), recent speed figures and sectionals, draw bias by course/distance, weight and weight changes, days since last run, class moves, gear changes, going/track condition, weather, seasonality, day vs night, field size, and other appropriate features. **Market-derived features** (early/declared odds, live `oddsDropValue`) kept clearly separate from the live odds used at the bet moment.

**NLP / text track:** parse stewards' reports, comments on running, vet records, and incident extracts into structured signals — trouble-in-running / "unlucky" flags, veterinary concerns, "won easing down" signals — via keyword/rule extraction, sentiment, and/or embeddings. Treat as an ablatable feature group so I can measure its contribution.

## Modeling & experimentation (comparison is first-class)

- Frame WIN prediction as a **single-winner-among-N-runners** problem. Build a **model zoo behind one common interface**:
  - Conditional / multinomial logit (Benter-style) and regularized variants.
  - Gradient boosting — XGBoost / LightGBM / CatBoost — incl. **learning-to-rank** (e.g. LambdaMART).
  - Tabular NNs sized for 8 GB VRAM (MLP, TabNet, FT-Transformer).
  - Ensembles / stacking.
- **PLACE (top-N) probabilities:** support **both** derivation from WIN probabilities (**Harville**, plus **Henery** / **Lo–Bacon–Shek**) **and** direct modeling; compare.
- **Calibration** (isotonic / Platt / temperature) is a primary metric.
- **Market-blend** step with a tunable weight.
- **Experiment tracking** (recommend MLflow or W&B) + **hyperparameter search** (recommend Optuna): every dataset × feature-set × model × technique combination logged, reproducible, comparable on a leaderboard.

## Backtesting & evaluation (honesty is the whole point)

- Strict **time-ordered, walk-forward** with realistic retraining cadence. No shuffling/lookahead/post-race leakage.
- Simulate realistic betting: **takeout**, configurable **rebate** (default 0), **pari-mutuel pool impact**, **HK$10 min-bet rounding**, and the gap between bet-time odds and final dividend.
- **Metrics:** primary = **ROI and Sharpe-like ratios** (plus turnover, hit rate, max drawdown, volatility, bankroll trajectories); also log-loss and calibration. **Bootstrap confidence intervals** to separate skill from luck.
- Clean **A/B comparison** of any two end-to-end configurations.

## Value detection & bet pricing

Calibrated probabilities → fair odds; compare to live odds for **edge / EV** per runner and per pool, net of takeout and pool dilution; transparent, configurable value threshold.

## Risk management & staking (a core research goal)

Implement and compare: flat, fixed-fraction, full Kelly, **fractional Kelly across a configurable grid**, and **Kelly for simultaneous/correlated bets** across runners within a race and across concurrent races; note drawdown-aware / risk-parity variants. Enforce configurable **bankroll (default HK$1,000)**, **per-race and per-day exposure caps (sweep settings)**, and legal stake rounding. All risk parameters first-class and configurable.

## UI / UX — React + FastAPI (intuitive is required)

FastAPI backend exposing data, predictions, value/staking, and backtest engines; React frontend with:
- **Race-day dashboard** — today's cards, model probabilities, live odds, detected value, recommended stakes (bet/no-bet + rounding logic visible).
- **Model / experiment comparison** — leaderboards, calibration plots, ablations.
- **Backtest explorer** — bankroll curves, drawdowns, ROI/Sharpe with CIs, filters by venue/class/distance/going.
- **Data & scraper health** — coverage, gaps, last-run status, live-odds log status.

## Engineering expectations

Clean, modular, documented, **config-driven** (YAML/TOML); reproducible (pinned env, fixed seeds, data versioning); structured logging; **tests** for scraper parsers, feature pipeline, and especially the backtest/Kelly math; clear separation of `data / features / models / risk / backtest / api / ui`.

## Future-proofing (v2+, plan for but don't build now)

Design the probability core so a **finishing-order distribution layer (Plackett–Luce, generalizing Harville)** can sit on top to price exotics — Quinella / Quinella Place first, then Forecast/Tierce, Trio, First 4, Quartet — via scoring/Monte-Carlo over finishing orders. The live odds API (B1) already accepts `oddsTypes` like `QIN, QPL, FCT, TCE, TRI, FF, QTT` and the dividend table is fully scraped, so keep pool/dividend modeling generic enough to extend.

## Deliver in THIS response (plan only — do not build yet)

1. A concise critique/validation of this plan and any **better approaches** you'd recommend.
2. A **phased roadmap** with milestones (e.g., M1 incremental scraper + storage; M2 features + baseline conditional-logit + honest backtest; M3 model zoo + calibration + market blend; M4 NLP track; M5 risk/staking sweeps; M6 React+FastAPI UI; M7 experiment tooling + live-odds polling).
3. A **proposed repository structure** and core tech-stack choices (with brief justifications), right-sized to my hardware.
4. The proposed **data schema** and **scraper strategy** per source (HTTP-parse vs GraphQL), including the incremental/idempotent crawl design and how you'll enumerate historical race dates.
5. **An explicit "What I need from you" list**, grouped **must-have to start** vs **nice-to-have later**. The live-odds API is already provided (Section B), so focus on: (a) whether *you* (Claude Code) have internet access — if not, which **sample HTML pages** I should save (one results page + one race card) so you can write parsers offline; (b) confirmation of bankroll/risk defaults; (c) any decisions still open.
6. A short list of **open questions and assumptions** you're making.
