"""Typer CLI entry point: ``hkjc``.

M0 provides ``version`` and ``doctor`` (working) plus stubs for the pipeline commands
that are implemented in later milestones.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer

from hkjc import __version__
from hkjc.common.config import get_config
from hkjc.common.logging import configure_logging

app = typer.Typer(
    name="hkjc",
    help=(
        "HKJC horse-racing ML research platform - recommends WIN/PLACE value bets "
        "with principled stake sizing. It never places bets."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    show_version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
    log_level: Annotated[
        str, typer.Option(help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    ] = "INFO",
    json_logs: Annotated[
        bool, typer.Option("--json-logs/--no-json-logs", help="Emit JSON logs instead of console.")
    ] = False,
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(level=log_level, json_logs=json_logs)


@app.command()
def version() -> None:
    """Print the installed hkjc version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Show resolved configuration, paths, and the locked data scope."""
    cfg = get_config()
    typer.echo(f"hkjc {__version__}\n")

    typer.echo("Paths:")
    typer.echo(f"  data_root  : {cfg.paths.data_root}")
    typer.echo(f"  raw        : {cfg.paths.raw_dir}")
    typer.echo(f"  processed  : {cfg.paths.processed_dir}")
    typer.echo(f"  duckdb     : {cfg.paths.duckdb_path}")
    typer.echo(f"  fixtures   : {cfg.paths.fixtures_dir}\n")

    typer.echo("Risk:")
    typer.echo(f"  bankroll   : {cfg.risk.currency} {cfg.risk.bankroll:,.0f}")
    typer.echo(f"  rebate     : {cfg.risk.rebate_rate}")
    typer.echo(f"  ev_thresh  : {cfg.risk.ev_threshold}")
    typer.echo(f"  kelly grid : {cfg.risk.kelly_fractions}\n")

    typer.echo("Backtest:")
    typer.echo(f"  takeout WIN: {cfg.backtest.takeout_win}")
    typer.echo(f"  takeout PLA: {cfg.backtest.takeout_place}\n")

    typer.echo("Enabled alternative data sources:")
    typer.echo(f"  {', '.join(cfg.sources.enabled_alt_sources)}")


def _todo(command: str, milestone: str) -> None:
    typer.echo(f"`{command}` is planned for {milestone} - not implemented yet.")


@app.command()
def scrape(
    date_str: Annotated[str, typer.Option("--date", help="Meeting date, YYYY-MM-DD.")],
    force: Annotated[bool, typer.Option(help="Re-fetch even if already stored.")] = False,
) -> None:
    """Scrape one meeting's full results into raw Parquet + the manifest (M1)."""
    from hkjc.data import pipeline

    report = pipeline.scrape_meeting(date.fromisoformat(date_str), force=force)
    if report.skipped:
        typer.echo(f"{report.race_date}: already stored - skipped (0 fetches).")
    elif report.venue is None:
        typer.echo(f"{report.race_date}: no meeting found.")
    else:
        typer.echo(
            f"{report.race_date} {report.venue}: {report.races} races, "
            f"{report.fetched} fetched, rows={report.rows}"
        )


@app.command()
def backfill(
    limit: Annotated[int | None, typer.Option(help="Only the newest N meetings.")] = None,
    since: Annotated[str | None, typer.Option(help="Only meetings on/after YYYY-MM-DD.")] = None,
    force: Annotated[bool, typer.Option(help="Re-fetch even if already stored.")] = False,
) -> None:
    """Backfill meetings from the results dropdown, idempotently (M1)."""
    from hkjc.data import pipeline

    def _progress(rep: pipeline.ScrapeReport) -> None:
        if not rep.skipped and rep.venue:
            typer.echo(f"  {rep.race_date} {rep.venue}: {rep.races} races ({rep.fetched} fetched)")

    reports = pipeline.backfill(
        limit=limit,
        since=date.fromisoformat(since) if since else None,
        force=force,
        on_meeting=_progress,
    )
    scraped = sum(1 for r in reports if not r.skipped and r.venue)
    skipped = sum(1 for r in reports if r.skipped)
    fetched = sum(r.fetched for r in reports)
    typer.echo(
        f"Backfill: {len(reports)} dates — {scraped} scraped, {skipped} skipped, {fetched} fetches."
    )


@app.command(name="scrape-horses")
def scrape_horses(
    limit: Annotated[int | None, typer.Option(help="Only the first N horses.")] = None,
) -> None:
    """Scrape horse profiles (bio + form) for horses seen in stored results (M1)."""
    from hkjc.data import pipeline

    summary = pipeline.scrape_horses(limit=limit)
    typer.echo(f"Horses: {summary['horses']} profiles, {summary['form_rows']} form rows.")


@app.command(name="scrape-people")
def scrape_people(
    limit: Annotated[int | None, typer.Option(help="Only the first N jockeys/trainers.")] = None,
) -> None:
    """Scrape jockey + trainer profiles for people seen in stored results (M1)."""
    from hkjc.data import pipeline

    summary = pipeline.scrape_people(limit=limit)
    typer.echo(
        f"People: {summary['people']} profiles "
        f"({summary['jockeys']} jockeys, {summary['trainers']} trainers)."
    )


@app.command(name="scrape-weather")
def scrape_weather(
    since_year: Annotated[int, typer.Option(help="Keep daily climate from this year on.")] = 2000,
) -> None:
    """Ingest HKO daily-climate temperature for both venues' stations (M1)."""
    from hkjc.data import pipeline

    summary = pipeline.ingest_weather(since_year=since_year)
    typer.echo(
        f"Weather: {summary['weather_rows']} daily rows across {summary['stations']} stations."
    )


@app.command(name="scrape-trials")
def scrape_trials(
    limit: Annotated[int | None, typer.Option(help="Only the newest N trial dates.")] = None,
    since: Annotated[str | None, typer.Option(help="Only trials on/after YYYY-MM-DD.")] = None,
) -> None:
    """Scrape barrier-trial results, idempotently (M1)."""
    from hkjc.data import pipeline

    summary = pipeline.scrape_trials(
        limit=limit, since=date.fromisoformat(since) if since else None
    )
    typer.echo(f"Trials: {summary['trial_dates']} dates, {summary['trial_rows']} runs.")


@app.command(name="scrape-holidays")
def scrape_holidays() -> None:
    """Ingest the HK public-holiday calendar from gov.hk open data (M1)."""
    from hkjc.data import pipeline

    summary = pipeline.ingest_holidays()
    typer.echo(f"Holidays: {summary['holidays']} dates.")


@app.command(name="data-health")
def data_health() -> None:
    """Show stored data coverage and the manifest size (M1)."""
    from hkjc.data import pipeline

    s = pipeline.coverage_summary()
    typer.echo("Coverage:")
    typer.echo(f"  meetings     : {s['meetings']}  ({s['date_min']} -> {s['date_max']})")
    typer.echo(f"  races        : {s['races_rows']}")
    typer.echo(f"  runners      : {s['results_rows']}")
    typer.echo(f"  dividends    : {s['dividends_rows']}")
    typer.echo(f"  horses       : {s['horses_rows']}")
    typer.echo(f"  horse form   : {s['horse_form_rows']}")
    typer.echo(f"  people       : {s['people_rows']}")
    typer.echo(f"  weather      : {s['weather_rows']}")
    typer.echo(f"  holidays     : {s['public_holidays_rows']}")
    typer.echo(f"  trials       : {s['barrier_trials_rows']}")
    typer.echo(f"  manifest urls: {s['manifest_urls']}")
    for season, n in sorted(s["seasons"].items()):
        typer.echo(f"    season {season}: {n} meetings")


@app.command()
def backtest() -> None:
    """Run an honest, time-ordered walk-forward backtest (M2)."""
    _todo("backtest", "M2 (features + baseline + backtest)")


@app.command()
def train() -> None:
    """Train and compare the model zoo (M3)."""
    _todo("train", "M3 (model zoo + calibration + blend)")


@app.command()
def predict() -> None:
    """Predict WIN/PLACE probabilities for a race card (M2/M7)."""
    _todo("predict", "M2/M7")


@app.command()
def poll() -> None:
    """Poll the live GraphQL odds API and log snapshots (M7)."""
    _todo("poll", "M7 (live ops + odds logging)")


if __name__ == "__main__":  # pragma: no cover
    app()
