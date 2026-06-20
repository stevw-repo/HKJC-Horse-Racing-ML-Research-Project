"""Staking comparison report: table, CSV/Parquet, and a ROI figure (M5, PLAN.md §2 M5).

The deliverable of M5 -- a side-by-side of every staking method at every bankroll, with the
two structural findings called out: the HK$10 **granularity** loss (large at HK$1k, ~0 at
HK$100k) and the HK$10k **rebate-threshold** crossings (never at HK$1k, frequent at HK$100k).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from hkjc.common.config import AppConfig
from hkjc.risk.simulate import StakingOutcome
from hkjc.risk.sweep import SweepResult


def _label(o: StakingOutcome) -> str:
    """Compact, ASCII policy label for tables and plots."""
    if o.method == "flat":
        return "flat"
    if o.method == "fixed_fraction":
        return "fixed_frac"
    tag = "corr" if o.correlated else "naive"
    if o.method == "kelly_full":
        return f"kelly_full/{tag}"
    return f"kelly_{o.kelly_lambda:.2f}/{tag}"


def outcomes_to_frame(outcomes: list[StakingOutcome]) -> pl.DataFrame:
    """Tidy one-row-per-cell frame (one row per policy x bankroll)."""
    return pl.DataFrame(
        {
            "bankroll": [o.bankroll0 for o in outcomes],
            "policy": [_label(o) for o in outcomes],
            "n_bets": [o.n_bets for o in outcomes],
            "staked": [round(o.total_staked, 2) for o in outcomes],
            "roi": [round(o.roi, 4) for o in outcomes],
            "roi_lo": [round(o.roi_lo, 4) for o in outcomes],
            "roi_hi": [round(o.roi_hi, 4) for o in outcomes],
            "terminal": [round(o.terminal_bankroll, 2) for o in outcomes],
            "max_dd": [round(o.max_drawdown, 4) for o in outcomes],
            "sharpe": [round(o.sharpe, 4) for o in outcomes],
            "round_loss": [round(o.rounding_loss, 2) for o in outcomes],
            "rebate_days": [o.rebate_days for o in outcomes],
            "ruin_prob": [round(o.ruin_prob, 4) for o in outcomes],
            "ruined": [o.ruined for o in outcomes],
        }
    )


def _plot_roi(df: pl.DataFrame, out_dir: str) -> str:
    """Grouped bar of ROI (%) per policy, one series per bankroll."""
    bankrolls = sorted(df["bankroll"].unique().to_list())
    labels = df.filter(pl.col("bankroll") == bankrolls[0])["policy"].to_list()
    width = 0.8 / max(len(bankrolls), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    for i, bk in enumerate(bankrolls):
        sub = df.filter(pl.col("bankroll") == bk)
        order = dict(zip(sub["policy"], sub["roi"], strict=True))
        heights = [order.get(p, 0.0) * 100.0 for p in labels]
        ax.bar(
            [x + i * width for x in range(len(labels))],
            heights,
            width,
            label=f"HK${int(bk):,}",
        )
    ax.axhline(-17.5, ls="--", color="grey", lw=1, label="-takeout (-17.5%)")
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(labels))])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Model+market WIN/PLACE ROI (%)")
    ax.set_title("Staking sweep: ROI by policy and bankroll (walk-forward OOS)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{out_dir}/staking_roi.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_report(result: SweepResult, cfg: AppConfig) -> dict[str, str]:
    """Persist the sweep frame (CSV + Parquet) and the ROI figure. Returns the paths."""
    out_dir = cfg.paths.processed_dir / "risk"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = outcomes_to_frame(result.outcomes)
    csv_path = out_dir / "staking_sweep.csv"
    pq_path = out_dir / "staking_sweep.parquet"
    df.write_csv(csv_path)
    df.write_parquet(pq_path)
    png_path = _plot_roi(df, str(out_dir))
    return {"csv": str(csv_path), "parquet": str(pq_path), "png": png_path}


def _granularity_line(df: pl.DataFrame) -> str:
    """Share of *intended* stake lost to the HK$10 floor (round_loss / desired), small vs large
    bankroll, for the correlated full-Kelly policy. Bounded [0, 1] -- unlike round_loss/staked,
    which blows up when nearly every stake rounds below the minimum (the HK$1k case)."""
    sub = df.filter(pl.col("policy") == "kelly_full/corr").sort("bankroll")
    if sub.is_empty():
        return "granularity: (no kelly_full/corr cell)"
    parts = []
    for row in sub.iter_rows(named=True):
        desired = row["round_loss"] + row["staked"]
        frac = row["round_loss"] / desired if desired > 0 else 0.0
        parts.append(f"HK${int(row['bankroll']):,}: {frac:.1%}")
    return "granularity (intended stake lost to HK$10 rounding, kelly_full/corr): " + ", ".join(
        parts
    )


def _rebate_line(df: pl.DataFrame) -> str:
    """Days the HK$10k losing-turnover threshold is crossed, small vs large bankroll."""
    sub = df.filter(pl.col("policy") == "kelly_full/corr").sort("bankroll")
    if sub.is_empty():
        return "rebate: (no kelly_full/corr cell)"
    parts = [f"HK${int(r['bankroll']):,}: {r['rebate_days']}" for r in sub.iter_rows(named=True)]
    return "rebate threshold crossed (days, kelly_full/corr): " + ", ".join(parts)


def format_sweep(result: SweepResult) -> str:
    """A console summary: provenance, the per-cell table, and the two headline findings."""
    df = outcomes_to_frame(result.outcomes)
    lo, hi = result.test_span
    lines = [
        f"Staking sweep -- {result.n_oos_races:,} OOS races, seasons {lo}..{hi}, "
        f"feature {result.feature_version}, market_weight {result.market_weight:.2f}",
        "",
    ]
    # ASCII_FULL: the Windows console (cp1252) cannot encode polars' Unicode box-drawing.
    with pl.Config(tbl_formatting="ASCII_FULL", tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        lines.append(str(df))
    lines += [
        "",
        _granularity_line(df),
        _rebate_line(df),
        "",
        "Note: ROI is the model+market value lens (a price is needed to size a bet). Every "
        "method still loses ~ the takeout -- staking changes variance/drawdown, not the edge "
        "(PLAN.md section 1F).",
    ]
    return "\n".join(lines)
