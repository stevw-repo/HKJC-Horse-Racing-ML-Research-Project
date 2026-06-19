"""GPU detection with a hard CPU fallback.

The user's box has an RTX 4060, but CI is CPU-only, so every learner must run both ways.
``gpu_available()`` is the single switch; set ``HKJC_FORCE_CPU=1`` to force CPU anywhere.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if a CUDA GPU is usable and CPU is not forced via ``HKJC_FORCE_CPU``."""
    if os.environ.get("HKJC_FORCE_CPU", "") not in ("", "0", "false", "False"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
