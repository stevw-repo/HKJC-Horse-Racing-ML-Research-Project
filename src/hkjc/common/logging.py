"""Structured logging via ``structlog``.

Call :func:`configure_logging` once at process start (the CLI does this in its
callback). Modules obtain a logger with ``structlog.get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import Processor


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog for console (default) or JSON output to stderr."""
    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
