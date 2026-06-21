"""FastAPI application factory (M6).

Local, single-user, read-only API surfacing M2-M5 output to the React dashboards. It
**recommends only -- it never places a bet** (the platform's hard invariant carries into the
UI: there is no write/bet endpoint).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hkjc.api.routes import router
from hkjc.common.config import AppConfig, get_config

SUMMARY = "HKJC WIN/PLACE research dashboards. Recommends only; never places bets."


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI app: CORS for the Vite dev server + the read-only /api router."""
    cfg = cfg or get_config()
    app = FastAPI(title="HKJC Research API", version="0.1.0", summary=SUMMARY)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
