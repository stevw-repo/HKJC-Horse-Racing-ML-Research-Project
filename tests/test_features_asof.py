"""As-of correctness tests for the feature builder (PLAN.md §1H): the leakage-critical
guarantee that horse-history features use *strictly prior* runs only."""

from __future__ import annotations

from datetime import date

import polars as pl

from hkjc.features.build import _horse_history, lbw_to_lengths


def test_lbw_to_lengths() -> None:
    assert lbw_to_lengths("-") == 0.0
    assert lbw_to_lengths("1-1/2") == 1.5
    assert lbw_to_lengths("3/4") == 0.75
    assert lbw_to_lengths("2-3/4") == 2.75
    assert lbw_to_lengths("10") == 10.0
    assert lbw_to_lengths("NK") == 0.3
    assert lbw_to_lengths("SH") == 0.05
    assert lbw_to_lengths("DIST") == 30.0
    assert lbw_to_lengths(None) is None
    assert lbw_to_lengths("garbage") is None


def _horse_frame() -> pl.DataFrame:
    # One horse, three dated runs; won race 1 and race 3.
    return pl.DataFrame(
        {
            "horse_id": ["H1", "H1", "H1"],
            "race_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "race_no": [1, 1, 1],
            "season": ["2019-20", "2019-20", "2019-20"],
            "won": [1, 0, 1],
            "placed": [1, 0, 1],
            "finish_pos": [1, 5, 1],
            "lbw_len": [0.0, 3.0, 0.0],
            "speed": [16.0, 15.0, 16.5],
            "dist_band": [1200, 1200, 1200],
            "going": ["G", "G", "G"],
        }
    )


def test_career_features_are_strictly_prior() -> None:
    out = _horse_history(_horse_frame()).sort("race_date")
    # career_run_number counts only *previous* runs (0 on debut).
    assert out["career_run_number"].to_list() == [0, 1, 2]
    # win_rate_prior must exclude the current race's own result.
    win_rate = out["win_rate_prior"].to_list()
    assert win_rate[0] is None  # debut: no prior runs
    assert win_rate[1] == 1.0  # after 1 prior run (a win)
    assert win_rate[2] == 0.5  # after 2 prior runs (1 win of 2)
    # days since last run is the gap to the previous run, not 0 for the current.
    assert out["days_since_last_run"].to_list() == [None, 31, 29]


def test_recent_form_excludes_current_run() -> None:
    out = _horse_history(_horse_frame()).sort("race_date")
    # avg_finish_last3 on race 3 averages prior finishes {1, 5} = 3.0, not including the 1.
    assert out["avg_finish_last3"].to_list()[2] == 3.0
