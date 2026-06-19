"""Tabular NN comparators in PyTorch (PLAN.md §1G).

Both an MLP and an FT-Transformer encoder map a runner's numeric features to a scalar PL
log-strength, trained with the *grouped* within-race conditional-logit (Plackett-Luce) NLL --
the same likelihood as the logit baseline, so WIN/PLACE stay comparable across the zoo. Per
PLAN.md these are comparators, not the workhorse; they run on GPU when available, minibatched
by race to fit the 8 GB card, with early stopping on a validation slice.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from hkjc.models.base import FloatArray, ProbabilityModel
from hkjc.models.device import gpu_available


def _grouped_nll(eta: Tensor, codes: Tensor, y: Tensor, n_groups: int) -> Tensor:
    """Mean within-race negative log-likelihood of the winner(s)."""
    neg_inf = torch.full((n_groups,), -1e30, device=eta.device)
    maxv = neg_inf.scatter_reduce(0, codes, eta, reduce="amax", include_self=True)
    shifted = torch.exp(eta - maxv[codes])
    sums = torch.zeros(n_groups, device=eta.device).index_add(0, codes, shifted)
    log_z = torch.log(sums) + maxv
    ll = (y * (eta - log_z[codes])).sum()
    return -ll / y.sum().clamp(min=1.0)


class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x).squeeze(-1)
        return out


class _FTTransformer(nn.Module):
    """A compact FT-Transformer: each numeric feature becomes a token; a CLS token reads out."""

    def __init__(self, d_in: int, dim: int = 32, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_in, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        encoder = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 2, dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (B, d, dim)
        cls = self.cls.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        scores: Tensor = self.head(out[:, 0, :]).squeeze(-1)
        return scores


class _TorchModel(ProbabilityModel):
    """Shared trainer: standardize, minibatch by race, grouped PL loss, early stop."""

    name = "torch"

    def __init__(
        self,
        epochs: int = 120,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        races_per_batch: int = 1024,
        patience: int = 12,
        seed: int = 0,
        use_gpu: bool | None = None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.races_per_batch = races_per_batch
        self.patience = patience
        self.seed = seed
        self.device = "cuda" if (gpu_available() if use_gpu is None else use_gpu) else "cpu"
        self.median_: FloatArray | None = None
        self.mean_: FloatArray | None = None
        self.std_: FloatArray | None = None
        self.net: nn.Module | None = None

    def _build(self, d_in: int) -> nn.Module:  # overridden by subclasses
        raise NotImplementedError

    def _prep(self, x: FloatArray, fit: bool) -> FloatArray:
        if fit:
            med = np.nanmedian(x, axis=0)
            self.median_ = np.nan_to_num(med, nan=0.0)
        assert self.median_ is not None
        xi = np.nan_to_num(np.where(np.isnan(x), self.median_, x), nan=0.0)
        if fit:
            self.mean_ = xi.mean(axis=0)
            std = xi.std(axis=0)
            self.std_ = np.where(std > 0, std, 1.0)
        assert self.mean_ is not None and self.std_ is not None
        return ((xi - self.mean_) / self.std_).astype(np.float32)

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> _TorchModel:
        torch.manual_seed(self.seed)
        xs = self._prep(np.asarray(x, dtype=np.float64), fit=True)
        g = np.asarray(groups)
        slices = _race_slices(g)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(slices)
        n_val = max(1, int(0.15 * len(slices)))
        val_slices, train_slices = slices[:n_val], slices[n_val:]

        device = torch.device(self.device)
        xt = torch.from_numpy(xs).to(device)
        yt = torch.from_numpy(y.astype(np.float32)).to(device)
        net = self._build(xs.shape[1]).to(device)
        self.net = net
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state: dict[str, Tensor] | None = None
        bad = 0
        for _epoch in range(self.epochs):
            net.train()
            order = rng.permutation(len(train_slices))
            for i in range(0, len(order), self.races_per_batch):
                batch = [train_slices[j] for j in order[i : i + self.races_per_batch]]
                idx, codes = _gather(batch)
                idx_t = torch.from_numpy(idx).to(device)
                codes_t = torch.from_numpy(codes).to(device)
                opt.zero_grad()
                eta = net(xt[idx_t])
                loss = _grouped_nll(eta, codes_t, yt[idx_t], int(codes.max()) + 1)
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
            val = self._val_loss(net, xt, yt, val_slices, device)
            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _val_loss(
        self,
        net: nn.Module,
        xt: Tensor,
        yt: Tensor,
        val_slices: list[tuple[int, int]],
        device: torch.device,
    ) -> float:
        net.eval()
        idx, codes = _gather(val_slices)
        idx_t = torch.from_numpy(idx).to(device)
        codes_t = torch.from_numpy(codes).to(device)
        eta = net(xt[idx_t])
        return float(_grouped_nll(eta, codes_t, yt[idx_t], int(codes.max()) + 1).item())

    @torch.no_grad()
    def log_strength(self, x: FloatArray) -> FloatArray:
        assert self.net is not None
        self.net.eval()
        xs = self._prep(np.asarray(x, dtype=np.float64), fit=False)
        device = torch.device(self.device)
        out = self.net(torch.from_numpy(xs).to(device))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


def _race_slices(groups: np.typing.ArrayLike) -> list[tuple[int, int]]:
    g = np.asarray(groups)
    change = np.empty(g.size, dtype=bool)
    change[0] = True
    change[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(change)
    stops = np.append(starts[1:], g.size)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _gather(
    slices: list[tuple[int, int]],
) -> tuple[np.typing.NDArray[np.int64], np.typing.NDArray[np.int64]]:
    """Row indices for a set of races + local 0..G-1 group codes."""
    idx_parts, code_parts = [], []
    for code, (start, stop) in enumerate(slices):
        idx_parts.append(np.arange(start, stop, dtype=np.int64))
        code_parts.append(np.full(stop - start, code, dtype=np.int64))
    return np.concatenate(idx_parts), np.concatenate(code_parts)


class MLPModel(_TorchModel):
    """Feed-forward tabular network."""

    name = "mlp"

    def __init__(self, hidden: int = 128, dropout: float = 0.2, **kwargs: float | int | bool):
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.hidden = hidden
        self.dropout = dropout

    def _build(self, d_in: int) -> nn.Module:
        return _MLP(d_in, self.hidden, self.dropout)


class FTTransformerModel(_TorchModel):
    """FT-Transformer encoder over numeric feature tokens."""

    name = "ft_transformer"

    def __init__(
        self, dim: int = 32, heads: int = 4, layers: int = 2, **kwargs: float | int | bool
    ):
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.dim = dim
        self.heads = heads
        self.layers = layers

    def _build(self, d_in: int) -> nn.Module:
        return _FTTransformer(d_in, self.dim, self.heads, self.layers)
