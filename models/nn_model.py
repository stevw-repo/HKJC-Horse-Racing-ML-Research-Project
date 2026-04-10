# models/nn_model.py
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class _MLP(object):
    """PyTorch MLP with optional residual connections."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden_dims:
                    layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net    = nn.Sequential(*layers)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(self.net(x)).squeeze(-1)

        self.net    = _Net()
        self.torch  = torch
        self.nn     = nn

    def to(self, device):
        self.net = self.net.to(device)
        return self


class NNModel(BaseModel):
    """PyTorch MLP binary classifier."""

    def __init__(self, hidden_dims=None, dropout=0.3, lr=1e-3,
                 epochs=100, batch_size=2048, params: dict = None):
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self._state      = None
        self._input_dim  = None

    def fit(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._input_dim = X_train.shape[1]
        mlp  = _MLP(self._input_dim, self.hidden_dims, self.dropout).to(device)
        net  = mlp.net

        Xt   = torch.tensor(X_train, dtype=torch.float32)
        yt   = torch.tensor(y_train, dtype=torch.float32)
        Xv   = torch.tensor(X_val,   dtype=torch.float32).to(device)
        yv   = torch.tensor(y_val,   dtype=torch.float32).to(device)

        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)
        opt   = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit  = nn.BCELoss()

        best_val_loss = float("inf")
        patience = 10
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(net(xb), yb)
                loss.backward()
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                val_loss = crit(net(Xv), yv).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.4f)",
                            epoch, best_val_loss)
                break

        if best_state:
            net.load_state_dict(best_state)
        self._state = best_state
        self._net   = net

    def predict_proba(self, X) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Model not trained.")
        import torch
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return self._net(t).numpy()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"state": self._state,
                         "hidden_dims": self.hidden_dims,
                         "dropout":     self.dropout,
                         "input_dim":   self._input_dim}, f)

    @classmethod
    def load(cls, path: Path) -> "NNModel":
        import torch
        import torch.nn as nn
        with open(Path(path), "rb") as f:
            d = pickle.load(f)
        inst = cls(hidden_dims=d["hidden_dims"], dropout=d["dropout"])
        inst._input_dim = d["input_dim"]
        inst._state     = d["state"]
        mlp  = _MLP(d["input_dim"], d["hidden_dims"], d["dropout"])
        mlp.net.load_state_dict(d["state"])
        inst._net = mlp.net
        return inst