"""HKJC horse-racing ML research platform.

Predicts WIN/PLACE probabilities for Sha Tin and Happy Valley races, detects value
against the live pari-mutuel odds, and sizes stakes with Kelly variants. It recommends
only — it never places bets.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hkjc")
except PackageNotFoundError:  # pragma: no cover - only when running from a non-installed tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
