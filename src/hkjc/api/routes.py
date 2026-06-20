"""API routes (M6). One router under ``/api``; every handler delegates to the service layer."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hkjc.api import service
from hkjc.api.schemas import (
    BacktestResponse,
    LeaderboardRow,
    RaceDayResponse,
    RaceSummary,
    StakingRow,
)

router = APIRouter(prefix="/api")


@router.get("/ping")
def ping() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@router.get("/health")
def health() -> dict[str, Any]:
    """Stored-data coverage + manifest size (data/scraper-health dashboard)."""
    return service.health()


@router.get("/backtest", response_model=BacktestResponse)
def backtest() -> BacktestResponse:
    """The honest walk-forward backtest (M2)."""
    result = service.backtest()
    if result is None:
        raise HTTPException(
            status_code=404, detail="No backtest snapshot yet; run `hkjc backtest`."
        )
    return result


@router.get("/leaderboard", response_model=list[LeaderboardRow])
def leaderboard() -> list[LeaderboardRow]:
    """The model-zoo leaderboard (M3). Empty until `hkjc train` has run."""
    return service.leaderboard()


@router.get("/staking", response_model=list[StakingRow])
def staking() -> list[StakingRow]:
    """The staking sweep grid (M5). Empty until `hkjc risk sweep` has run."""
    return service.staking()


@router.get("/races", response_model=list[RaceSummary])
def races(limit: int = Query(default=50, ge=1, le=500)) -> list[RaceSummary]:
    """The most recent stored races."""
    return service.races(limit)


@router.get("/raceday", response_model=RaceDayResponse)
def raceday() -> RaceDayResponse:
    """A mocked upcoming card with value-staking recommendations (real arrives at M7)."""
    return service.raceday()
