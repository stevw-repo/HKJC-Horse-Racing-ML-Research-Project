"""Conditional-logit baseline: a Benter-style Plackett-Luce strength model (PLAN.md §1D).

The per-runner log-strength is linear, ``eta_i = beta . x_i``; WIN probability is the
within-race softmax of ``eta`` (multinomial conditional logit, with the winner as the chosen
alternative). The negative log-likelihood is convex, so it is fit by L-BFGS-B with an
analytic gradient and a small L2 (ridge) penalty. Features are median-imputed and
standardised on the training fold; the fitted statistics are reused at predict time.

Dead-heat wins (two runners labelled ``won=1``) are handled naturally: the per-race winner
count ``S_g`` enters the gradient, so a race simply contributes two unit "wins".
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize

from hkjc.models.base import FloatArray, ProbabilityModel, group_codes, softmax_by_group


class ConditionalLogit(ProbabilityModel):
    """Within-race conditional logit emitting a Plackett-Luce strength vector."""

    name = "conditional_logit"

    def __init__(self, l2: float = 1.0, max_iter: int = 500) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.beta_: FloatArray | None = None
        self.median_: FloatArray | None = None
        self.mean_: FloatArray | None = None
        self.std_: FloatArray | None = None

    # -- preprocessing ------------------------------------------------------- #
    def _fit_scaler(self, x: FloatArray) -> FloatArray:
        with warnings.catch_warnings():  # all-NaN columns (e.g. sparse age) -> median 0
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.nan_to_num(self.median_, nan=0.0)
        xi = self._impute(x)
        self.mean_ = xi.mean(axis=0)
        std = xi.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return (xi - self.mean_) / self.std_

    def _impute(self, x: FloatArray) -> FloatArray:
        assert self.median_ is not None
        out = np.where(np.isnan(x), self.median_, x)
        return np.nan_to_num(out, nan=0.0)

    def _transform(self, x: FloatArray) -> FloatArray:
        assert self.mean_ is not None and self.std_ is not None
        return (self._impute(x) - self.mean_) / self.std_

    # -- fit / predict ------------------------------------------------------- #
    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> ConditionalLogit:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        xs = self._fit_scaler(x)
        codes, n_groups = group_codes(groups)
        winners_per_group = np.bincount(codes, weights=y, minlength=n_groups)
        s_by_row = winners_per_group[codes]
        n_features = xs.shape[1]

        def objective(beta: FloatArray) -> tuple[float, FloatArray]:
            eta = xs @ beta
            probs = softmax_by_group(eta, codes, n_groups)
            # NLL = -sum_i y_i log p_i  (+ L2); winners only contribute to the loss term.
            loss = -float(np.sum(y * np.log(probs + 1e-300))) + self.l2 * float(beta @ beta)
            grad = xs.T @ (s_by_row * probs - y) + 2.0 * self.l2 * beta
            return loss, grad

        result = optimize.minimize(
            objective,
            np.zeros(n_features),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.beta_ = np.asarray(result.x, dtype=np.float64)
        return self

    def log_strength(self, x: FloatArray) -> FloatArray:
        if self.beta_ is None:
            msg = "ConditionalLogit is not fitted."
            raise RuntimeError(msg)
        return self._transform(np.asarray(x, dtype=np.float64)) @ self.beta_

    @property
    def coefficients(self) -> FloatArray:
        """Fitted standardised coefficients (comparable across features)."""
        if self.beta_ is None:
            msg = "ConditionalLogit is not fitted."
            raise RuntimeError(msg)
        return self.beta_
