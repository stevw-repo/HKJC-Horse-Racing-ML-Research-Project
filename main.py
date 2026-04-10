# main.py
import argparse
import datetime
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HKJC Predictor CLI")
    parser.add_argument("--mode", required=True,
                        choices=["scrape", "features", "train", "backtest",
                                 "predict", "ui", "inspect"])
    parser.add_argument("--scraper", default="all",
                        choices=["results", "racecard", "horses",
                                 "jockey_trainer", "odds", "dividends",
                                 "weather", "all"])
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm", "xgb", "catboost", "nn", "ensemble"])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dashboard", default="live",
                        choices=["scraper", "training", "backtest", "live"])
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    dispatch = {
        "scrape":   _run_scraper,
        "features": _run_features,
        "train":    _run_train,
        "backtest": _run_backtest,
        "predict":  _run_predict,
        "ui":       _run_ui,
        "inspect":  _run_inspect,
    }
    dispatch[args.mode](args)


# ── Mode handlers ─────────────────────────────────────────────────────────────

def _run_scraper(args):
    from config import RESULTS_START_DATE
    start = args.start or RESULTS_START_DATE
    end   = args.end   or datetime.date.today().isoformat()

    scraper_map = {
        "results":       ("scrapers.results_scraper",       "ResultsScraper"),
        "racecard":      ("scrapers.racecard_scraper",      "RacecardScraper"),
        "horses":        ("scrapers.horse_scraper",         "HorseScraper"),
        "jockey_trainer":("scrapers.jockey_trainer_scraper","JockeyTrainerScraper"),
        "odds":          ("scrapers.odds_scraper",          "OddsScraper"),
        "dividends":     ("scrapers.dividends_scraper",     "DividendsScraper"),
        "weather":       ("scrapers.weather_scraper",       "WeatherScraper"),
    }

    selected = list(scraper_map.keys()) if args.scraper == "all" else [args.scraper]
    for key in selected:
        module_path, class_name = scraper_map[key]
        import importlib
        mod = importlib.import_module(module_path)
        scraper = getattr(mod, class_name)()
        logger.info("Running %s …", class_name)
        scraper.scrape(start_date=start, end_date=end)
        logger.info("%s completed.", class_name)


def _run_features(args):
    from features.pipeline import build_training_features
    logger.info("Building training feature table …")
    df = build_training_features()
    logger.info("Done — %d rows, %d columns.", len(df), len(df.columns))


def _run_train(args):
    import pandas as pd
    from config import TRAIN_CUTOFF_DATE, VALID_CUTOFF_DATE, PROCESSED_DIR, MODELS_DIR
    from storage.parquet_store import read
    from models.registry import get_model

    feat_path = PROCESSED_DIR / "features_train.parquet"
    if not feat_path.exists():
        from features.pipeline import build_training_features
        df = build_training_features()
    else:
        df = read(feat_path)

    df["race_date"] = pd.to_datetime(df["race_date"])
    exclude = {"target_win", "target_place", "race_date", "horse_id",
               "race_id", "is_debutant", "placing_code"}
    feature_cols = [c for c in df.columns if c not in exclude]

    train_df = df[df["race_date"] <= TRAIN_CUTOFF_DATE]
    valid_df  = df[(df["race_date"] > TRAIN_CUTOFF_DATE) &
                   (df["race_date"] <= VALID_CUTOFF_DATE)]

    for target in ["win", "place"]:
        logger.info("Training %s/%s …", args.model, target)
        model = get_model(args.model, target, args.version)
        model.fit(
            train_df[feature_cols].values, train_df[f"target_{target}"].values,
            valid_df[feature_cols].values, valid_df[f"target_{target}"].values,
        )
        save_path = MODELS_DIR / f"{args.model}_{target}_{args.version}.pkl"
        model.save(save_path)
        logger.info("Saved → %s", save_path)


def _run_backtest(args):
    from config import PROCESSED_DIR, STARTING_BANKROLL, KELLY_FRACTION, MIN_EDGE
    from storage.parquet_store import read
    from models.registry import get_model
    from betting.backtest import run_backtest

    df          = read(PROCESSED_DIR / "features_train.parquet")
    win_model   = get_model(args.model, "win",   args.version)
    place_model = get_model(args.model, "place", args.version)
    result      = run_backtest(
        features_df=df,
        win_model=win_model,
        place_model=place_model,
        start_date=args.start or "2022-01-01",
        end_date=args.end     or "2025-12-31",
        starting_bankroll=STARTING_BANKROLL,
        kelly_fraction=KELLY_FRACTION,
        min_edge=MIN_EDGE,
    )
    print(result.summary())


def _run_predict(args):
    from config import VALID_VENUES
    from features.pipeline import build_prediction_features
    race_date = args.start or datetime.date.today().isoformat()
    for venue in sorted(VALID_VENUES):
        try:
            df = build_prediction_features(race_date=race_date, venue=venue)
            logger.info("%s %s — %d runners loaded.", venue, race_date, len(df))
        except Exception as exc:
            logger.warning("Could not build features for %s: %s", venue, exc)


def _run_ui(args):
    dashboard_map = {
        "scraper":  "ui/scraper_ui.py",
        "training": "ui/training_ui.py",
        "backtest": "ui/backtest_ui.py",
        "live":     "ui/live_dashboard.py",
    }
    script = dashboard_map[args.dashboard]
    subprocess.run([sys.executable, "-m", "streamlit", "run", script])


def _run_inspect():
    from config import (RAW_RESULTS_DIR, RAW_RACECARDS_DIR, RAW_HORSES_DIR,
                        RAW_ODDS_DIR, RAW_DIVIDENDS_DIR, RAW_WEATHER_DIR,
                        PROCESSED_DIR)
    from storage.parquet_store import summary

    stores = {
        "results":   RAW_RESULTS_DIR,
        "racecards": RAW_RACECARDS_DIR,
        "horses":    RAW_HORSES_DIR,
        "odds":      RAW_ODDS_DIR,
        "dividends": RAW_DIVIDENDS_DIR,
        "weather":   RAW_WEATHER_DIR,
        "processed": PROCESSED_DIR,
    }
    for name, path in stores.items():
        print(f"\n=== {name} ===")
        for k, v in summary(path).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()