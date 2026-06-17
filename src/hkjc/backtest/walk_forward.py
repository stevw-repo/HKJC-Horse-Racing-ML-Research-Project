"""Walk-forward splitting (PLAN.md §2): strictly time-ordered, season-expanding folds.

The unit of time is the HK racing season. For each test season we train on *all* prior
seasons (expanding window, ``retrain_cadence: per_season``), so no future information ever
reaches a fit. This is the temporal backbone the leakage canary defends.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def iter_season_splits(
    seasons: npt.NDArray[np.str_], min_train_seasons: int = 1
) -> Iterator[tuple[str, BoolArray, BoolArray]]:
    """Yield ``(test_season, train_mask, test_mask)`` for each testable season in order.

    The first ``min_train_seasons`` seasons are training-only (never tested). Masks index the
    full row array passed in.
    """
    order = sorted(set(seasons.tolist()))
    for i in range(min_train_seasons, len(order)):
        test_season = order[i]
        train_seasons = order[:i]
        train_mask = np.isin(seasons, train_seasons)
        test_mask = seasons == test_season
        if train_mask.any() and test_mask.any():
            yield test_season, train_mask, test_mask
