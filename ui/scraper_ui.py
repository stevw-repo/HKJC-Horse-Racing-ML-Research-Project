# ui/scraper_ui.py
"""Streamlit dashboard for data scraping and dataset inspection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import importlib
import io

import pandas as pd
import streamlit as st

import config as cfg
from storage.parquet_store import summary, read, list_dates

st.set_page_config(page_title="HKJC Scraper Dashboard", layout="wide")
st.title("📥 HKJC Scraper Dashboard")


# ── Sidebar — global config overrides ────────────────────────────────────────
st.sidebar.header("Configuration")

results_start = st.sidebar.date_input(
    "Results start date",
    value=datetime.date.fromisoformat(cfg.RESULTS_START_DATE),
)

# KEY CONTROL — adjustable dividend cutoff year
div_cutoff = st.sidebar.number_input(
    "Dividend cutoff year",
    min_value=2000,
    max_value=datetime.date.today().year,
    value=cfg.DIVIDENDS_CUTOFF_YEAR,
    step=1,
    help=(
        "Races before this year are scraped for form features "
        "but dividends are stored as NaN.  Adjust to probe "
        "how far back reliable dividend data exists."
    ),
)
# Apply runtime override (without modifying the file)
cfg.DIVIDENDS_CUTOFF_YEAR = int(div_cutoff)

# ── Date range for scraping ───────────────────────────────────────────────────
st.header("Scraper Controls")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=results_start)
with col2:
    end_date = st.date_input("To",   value=datetime.date.today())

# ── Scraper selector ──────────────────────────────────────────────────────────
st.subheader("Select scrapers to run")
scraper_options = {
    "Results":            ("scrapers.results_scraper",        "ResultsScraper"),
    "Racecard":           ("scrapers.racecard_scraper",       "RacecardScraper"),
    "Horse Profiles":     ("scrapers.horse_scraper",          "HorseScraper"),
    "Jockey / Trainer":   ("scrapers.jockey_trainer_scraper", "JockeyTrainerScraper"),
    "Odds":               ("scrapers.odds_scraper",           "OddsScraper"),
    "Dividends (WIN/PLA/QIN/QPL)": ("scrapers.dividends_scraper", "DividendsScraper"),
    "Weather":            ("scrapers.weather_scraper",        "WeatherScraper"),
}

selected = {
    name: st.checkbox(name, value=(name in ["Results", "Dividends (WIN/PLA/QIN/QPL)"]))
    for name in scraper_options
}

log_area = st.empty()
log_buffer: list = []


def _append_log(msg: str):
    log_buffer.append(msg)
    log_area.text_area("Log output", value="\n".join(log_buffer),
                        height=200, key=f"log_{len(log_buffer)}")


if st.button("▶ Run Selected Scrapers", type="primary"):
    for name, (mod_path, cls_name) in scraper_options.items():
        if not selected.get(name):
            continue
        _append_log(f"[INFO] Starting {name} …")
        try:
            mod     = importlib.import_module(mod_path)
            scraper = getattr(mod, cls_name)()
            scraper.scrape(start_date=start_date.isoformat(),
                           end_date=end_date.isoformat())
            _append_log(f"[OK]   {name} completed.")
        except Exception as exc:
            _append_log(f"[ERR]  {name} failed: {exc}")

    st.success("Scraping run finished.")

st.markdown("---")

# ── Dataset Inspector ─────────────────────────────────────────────────────────
st.header("📊 Dataset Inspector")

stores = {
    "Results":   cfg.RAW_RESULTS_DIR,
    "Racecards": cfg.RAW_RACECARDS_DIR,
    "Horses":    cfg.RAW_HORSES_DIR,
    "Odds":      cfg.RAW_ODDS_DIR,
    "Dividends": cfg.RAW_DIVIDENDS_DIR,
    "Weather":   cfg.RAW_WEATHER_DIR,
    "Processed": cfg.PROCESSED_DIR,
}

if st.button("🔍 Refresh Dataset Summaries"):
    for store_name, store_path in stores.items():
        s = summary(store_path)
        with st.expander(f"{store_name} — {s.get('rows', 0):,} rows"):
            st.json(s)

st.markdown("---")

# ── Missing date finder ───────────────────────────────────────────────────────
st.header("🗓 Missing Date Finder")

if st.button("Find missing race dates in Results store"):
    present = set(list_dates(cfg.RAW_RESULTS_DIR))
    all_dates = pd.date_range(
        start=results_start, end=datetime.date.today(), freq="D"
    )
    missing = [str(d.date()) for d in all_dates if str(d.date()) not in present]
    st.write(f"Dates in range with no results data: **{len(missing)}**")
    if missing:
        st.dataframe(pd.DataFrame({"missing_date": missing}))

st.markdown("---")

# ── Raw data viewer ───────────────────────────────────────────────────────────
st.header("📂 Raw Data Viewer")

viewer_store = st.selectbox("Choose store", list(stores.keys()))
viewer_path  = stores[viewer_store]

if st.button("Load data"):
    df = read(viewer_path)
    if df.empty:
        st.warning("No data found in this store.")
    else:
        # Filters
        filter_cols = st.columns(3)
        with filter_cols[0]:
            if "race_date" in df.columns:
                min_d = pd.to_datetime(df["race_date"]).min().date()
                max_d = pd.to_datetime(df["race_date"]).max().date()
                date_filter = st.date_input("Filter date", value=(min_d, max_d))
                if len(date_filter) == 2:
                    df = df[(pd.to_datetime(df["race_date"]).dt.date >= date_filter[0]) &
                            (pd.to_datetime(df["race_date"]).dt.date <= date_filter[1])]
        with filter_cols[1]:
            if "venue" in df.columns:
                venues = st.multiselect("Venue", sorted(df["venue"].dropna().unique()),
                                        default=sorted(df["venue"].dropna().unique()))
                if venues:
                    df = df[df["venue"].isin(venues)]
        with filter_cols[2]:
            if "pool_type" in df.columns:
                pools = st.multiselect(
                    "Pool type (in-scope only)",
                    cfg.VALID_POOL_TYPES,
                    default=cfg.VALID_POOL_TYPES,
                )
                df = df[df["pool_type"].isin(pools)]

        st.dataframe(df.head(500), use_container_width=True)
        st.caption(f"Showing first 500 of {len(df):,} rows.")