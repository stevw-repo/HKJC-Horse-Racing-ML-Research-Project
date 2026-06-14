"""Typer CLI entry point: ``hkjc``.

M0 provides ``version`` and ``doctor`` (working) plus stubs for the pipeline commands
that are implemented in later milestones.
"""

from __future__ import annotations

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
    typer.echo(f"`{command}` is planned for {milestone} — not implemented yet.")


@app.command()
def scrape() -> None:
    """Incrementally scrape HKJC results/cards/profiles + HKO weather (M1)."""
    _todo("scrape", "M1 (scraper + storage)")


@app.command()
def backfill() -> None:
    """Backfill all modern-markup seasons (M1)."""
    _todo("backfill", "M1 (scraper + storage)")


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
