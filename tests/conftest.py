"""Pytest session setup.

The M3 zoo pulls torch + lightgbm + xgboost + catboost + scipy + scikit-learn into one
process, each bundling its own OpenMP runtime. On Windows that duplicate-runtime load can
abort the interpreter; opt into the standard guard *before* any of them import. (Harmless when
there is no duplication.) ``setdefault`` keeps it overridable by the environment / CI.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
