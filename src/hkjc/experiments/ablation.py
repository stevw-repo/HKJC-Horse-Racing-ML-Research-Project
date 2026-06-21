"""NLP ablation (PLAN.md §2 M4 exit criterion).

Runs the *same* model through the *same* honest walk-forward twice -- without and with the
lagged ``nlp_text`` group -- and reports the marginal change in log-loss / top-1 / ROI. That
delta is M4's deliverable: it quantifies whether the comment-on-running signal helps, ablatably.
"""

from __future__ import annotations

from dataclasses import dataclass

from hkjc.backtest.dataset import load_model_data
from hkjc.backtest.engine import BacktestResult
from hkjc.common.config import AppConfig, get_config
from hkjc.experiments.runner import evaluate_model
from hkjc.models.logit import ConditionalLogit


@dataclass(frozen=True, slots=True)
class AblationResult:
    baseline: BacktestResult
    with_nlp: BacktestResult


def run_ablation(
    cfg: AppConfig | None = None,
    *,
    market_weight: float | None = None,
    ev_threshold: float | None = None,
    max_test_seasons: int | None = None,
    seed: int = 0,
) -> AblationResult:
    """Walk-forward the conditional logit with and without the NLP group; return both results."""
    cfg = cfg or get_config()
    market_weight = cfg.models.market_blend_weight if market_weight is None else market_weight
    ev_threshold = cfg.risk.ev_threshold if ev_threshold is None else ev_threshold

    results: list[BacktestResult] = []
    for include_nlp in (False, True):
        data = load_model_data(cfg, include_nlp=include_nlp)
        n_seasons = len(set(data.season.tolist()))
        min_train = 1 if max_test_seasons is None else max(1, n_seasons - max_test_seasons)
        results.append(
            evaluate_model(
                ConditionalLogit,
                data.numeric(),
                data,
                market_weight=market_weight,
                ev_threshold=ev_threshold,
                stake=cfg.risk.min_bet,
                min_train_seasons=min_train,
                seed=seed,
                cfg=cfg,
            )
        )
    return AblationResult(baseline=results[0], with_nlp=results[1])


def format_ablation(result: AblationResult) -> str:
    """ASCII table: baseline vs +NLP on log-loss / top-1 / model-only WIN ROI, with deltas."""
    b, n = result.baseline, result.with_nlp
    rows = [
        ("log-loss", b.win_log_loss, n.win_log_loss, "{:+.4f}"),
        ("top-1 hit", b.top1_hit_rate, n.top1_hit_rate, "{:+.4f}"),
        ("model-only WIN ROI", b.policies["model_win"].roi, n.policies["model_win"].roi, "{:+.2%}"),
        (
            "market-blend WIN ROI",
            b.policies["blend_win"].roi,
            n.policies["blend_win"].roi,
            "{:+.2%}",
        ),
    ]
    out = [
        f"NLP ablation (logit, {b.n_oos_races} OOS races, seasons "
        f"{b.test_span[0]}..{b.test_span[1]}):\n",
        f"  {'metric':<22}{'baseline':>12}{'+nlp':>12}{'delta':>12}",
        "  " + "-" * 58,
    ]
    for name, base_v, nlp_v, fmt in rows:
        is_pct = fmt.endswith("%}")
        bv = f"{base_v:.2%}" if is_pct else f"{base_v:.4f}"
        nv = f"{nlp_v:.2%}" if is_pct else f"{nlp_v:.4f}"
        out.append(f"  {name:<22}{bv:>12}{nv:>12}{fmt.format(nlp_v - base_v):>12}")
    return "\n".join(out)
