"""Smoke tests for the Typer CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from hkjc import __version__
from hkjc.cli import app

runner = CliRunner()


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Enabled alternative data sources" in result.stdout


def test_predict_stub_runs() -> None:
    result = runner.invoke(app, ["predict"])
    assert result.exit_code == 0
    assert "M2/M7" in result.stdout


def test_train_help_runs() -> None:
    # `train` now runs the real leaderboard, so only smoke its wiring via --help (no training).
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "leaderboard" in result.stdout.lower()


def test_scrape_requires_date() -> None:
    result = runner.invoke(app, ["scrape"])
    assert result.exit_code != 0  # --date is required


def test_data_health_runs() -> None:
    result = runner.invoke(app, ["data-health"])
    assert result.exit_code == 0
    assert "Coverage" in result.stdout
