"""API tests (M6): FastAPI TestClient against the read-only endpoints.

Written to pass on a fresh checkout with no ``data/``: endpoints degrade to empty/zeros and the
backtest snapshot may be absent (404). The tests assert the *contract*, not generated data.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hkjc.api.app import create_app

client = TestClient(create_app())


def test_ping() -> None:
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_raceday_shape() -> None:
    # mock=True on a fresh checkout; mock=False once `hkjc race-day` has written a real card.
    response = client.get("/api/raceday")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["mock"], bool)
    assert {"race_date", "venue", "model_name", "has_live_odds", "races"} <= body.keys()
    assert body["races"] and body["races"][0]["runners"]


def test_health_shape() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    for key in ("meetings", "results_rows", "seasons"):
        assert key in body


def test_races_list() -> None:
    response = client.get("/api/races", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) <= 5
    for race in body:
        assert {"race_date", "venue", "race_no", "field_size"} <= race.keys()


def test_races_limit_is_validated() -> None:
    assert client.get("/api/races", params={"limit": 0}).status_code == 422


def test_staking_list() -> None:
    response = client.get("/api/staking")
    assert response.status_code == 200
    rows = response.json()
    assert isinstance(rows, list)
    if rows:
        assert {"bankroll", "policy", "roi", "ruined"} <= rows[0].keys()


def test_leaderboard_list() -> None:
    response = client.get("/api/leaderboard")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_backtest_present_or_404() -> None:
    response = client.get("/api/backtest")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        body = response.json()
        assert "policies" in body and "calibration" in body
        assert "model_win" in body["policies"]
