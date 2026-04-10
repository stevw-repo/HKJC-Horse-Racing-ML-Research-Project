# config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
DATA_DIR          = BASE_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
MODELS_DIR        = BASE_DIR / "models_saved"

RAW_RESULTS_DIR   = RAW_DIR / "results"
RAW_RACECARDS_DIR = RAW_DIR / "racecards"
RAW_HORSES_DIR    = RAW_DIR / "horses"
RAW_ODDS_DIR      = RAW_DIR / "odds"
RAW_DIVIDENDS_DIR = RAW_DIR / "dividends"
RAW_WEATHER_DIR   = RAW_DIR / "weather"

# ── Date ranges ───────────────────────────────────────────────────────────────
RESULTS_START_DATE    = "2010-09-01"

# Earliest year from which DIVIDENDS are scraped and used in P&L.
# Races before this year are kept for feature engineering but dividends
# are stored as NaN.  Adjust freely to probe data availability.
DIVIDENDS_CUTOFF_YEAR = 2010          # <-- key tunable

# ── Scraper behaviour ─────────────────────────────────────────────────────────
REQUEST_DELAY_SEC   = 1.5
MAX_RETRIES         = 5
BACKOFF_FACTOR      = 2.0
REQUEST_TIMEOUT_SEC = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "en,zh-TW;q=0.9",
}

# ── Endpoints ─────────────────────────────────────────────────────────────────
GRAPHQL_URL       = "https://info.cld.hkjc.com/graphql/base/"
RESULTS_BASE_URL  = "https://racing.hkjc.com/en-us/local/information/localresults"
HORSE_BASE_URL    = "https://racing.hkjc.com/en-us/local/information"
BTRESULT_BASE_URL = "https://racing.hkjc.com/en-us/local/information/btresult"
WEATHER_API_URL   = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"

# ── Feature engineering ───────────────────────────────────────────────────────
FORM_WINDOW_RACES  = [3, 5, 10]
MIN_RACES_FOR_RATE = 5

# ── Model training ────────────────────────────────────────────────────────────
TRAIN_CUTOFF_DATE = "2024-06-30"
VALID_CUTOFF_DATE = "2025-06-30"
RANDOM_SEED       = 42

DEFAULT_LGBM_PARAMS = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
}

# ── Betting / backtesting ─────────────────────────────────────────────────────
KELLY_FRACTION    = 0.25
MAX_BET_FRACTION  = 0.05
MIN_EDGE          = 0.05
STARTING_BANKROLL = 10_000.0

# ── In-scope pool types (WIN, PLACE, QIN, QPL only) ──────────────────────────
VALID_POOL_TYPES = ["WIN", "PLA", "QIN", "QPL"]

# ── Venue codes ───────────────────────────────────────────────────────────────
VALID_VENUES = {"ST", "HV"}

# ── Create all directories on import ─────────────────────────────────────────
for _d in [RAW_RESULTS_DIR, RAW_RACECARDS_DIR, RAW_HORSES_DIR,
           RAW_ODDS_DIR, RAW_DIVIDENDS_DIR, RAW_WEATHER_DIR,
           PROCESSED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)