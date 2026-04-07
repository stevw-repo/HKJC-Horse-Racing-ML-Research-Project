# %% [markdown]
# # Notebook 3 – Model Training, Calibration & Evaluation
#
# Models trained
# --------------
# 1. Logistic Regression (baseline)
# 2. Random Forest
# 3. XGBoost (Optuna-tuned)
# 4. LightGBM (Optuna-tuned)
# 5. Ensemble (average of XGBoost + LightGBM + RF)
# 6. [Optional] PyTorch 2-layer MLP
#
# Evaluation
# ----------
# * AUC, log-loss, accuracy on the test set.
# * MAP@3, NDCG@3 (ranking metrics per race).
# * Betting backtest using fractional-Kelly stake sizing.

# %% ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠  Optuna not installed. Falling back to default hyperparameters.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils import (
    compute_ranking_metrics,
    ensure_dirs,
    logger,
    run_betting_backtest,
    safe_read_parquet,
)

warnings.filterwarnings("ignore")

# %% ── Configuration ────────────────────────────────────────────────────────

DATA_SPLITS  = Path("data/splits")
DATA_MODELS  = Path("data/models")
DATA_RESULTS = Path("data/results")

# Betting backtest parameters
BET_EV_THRESHOLD  = 0.05
BET_MIN_PROB      = 0.15
BET_TOP_N         = 2
KELLY_FRAC        = 0.25
KELLY_MAX         = 0.05
STARTING_BANKROLL = 1_000.0

# Optuna settings
N_OPTUNA_TRIALS = 50
N_CV_SPLITS     = 3

# %% ── Data loading ──────────────────────────────────────────────────────────

def load_splits() -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    List[str],
]:
    """
    Load pre-built train / val / test splits from Parquet files.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    meta_train, meta_val, meta_test,
    feature_cols
    """
    def _load(split: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        X   = pd.read_parquet(DATA_SPLITS / f"X_{split}.parquet")
        y   = pd.read_parquet(DATA_SPLITS / f"y_{split}.parquet").squeeze()
        meta = pd.read_parquet(DATA_SPLITS / f"meta_{split}.parquet")
        return X.values.astype(np.float32), y.values.astype(int), meta

    X_train, y_train, meta_train = _load("train")
    X_val,   y_val,   meta_val   = _load("val")
    X_test,  y_test,  meta_test  = _load("test")

    # Load feature column names
    fc_path = DATA_SPLITS / "feature_cols.txt"
    feature_cols = pd.read_csv(fc_path, header=None)[0].tolist() if fc_path.exists() else []

    logger.info("Loaded splits — train: %s  val: %s  test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    logger.info("Class balance (train): %.3f positive rate",
                y_train.mean())

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        meta_train, meta_val, meta_test,
        feature_cols,
    )


# %% ── Helper: combined train+val arrays ─────────────────────────────────────

def _merge_train_val(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate training and validation arrays for final model fitting."""
    return (
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )


# %% ── Model 1: Logistic Regression (baseline) ──────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Train a baseline Logistic Regression with StandardScaler.

    Uses ``class_weight='balanced'`` to handle class imbalance.

    Returns
    -------
    sklearn Pipeline (scaler + LogisticRegression).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return pipe


# %% ── Model 2: Random Forest ────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with balanced class weights.

    Returns
    -------
    Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest trained.")
    return rf


# %% ── Model 3: XGBoost (Optuna-tuned) ──────────────────────────────────────

def _xgb_objective(
    trial: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
) -> float:
    """Optuna objective for XGBoost: minimise validation log-loss."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": float((y == 0).sum() / max((y == 1).sum(), 1)),
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
    }
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  verbose=False)
        preds = model.predict_proba(X[va_idx])[:, 1]
        scores.append(log_loss(y[va_idx], preds))
    return float(np.mean(scores))


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> xgb.XGBClassifier:
    """
    Train XGBoost with Optuna hyperparameter search over the train+val data
    using time-series cross-validation.

    Returns
    -------
    Fitted XGBClassifier with the best found hyperparameters.
    """
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    if OPTUNA_AVAILABLE:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: _xgb_objective(trial, X_all, y_all),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info("XGBoost best params: %s", best_params)
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 20, "gamma": 1.0,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }

    best_params.update({
        "scale_pos_weight": float((y_all == 0).sum() / max((y_all == 1).sum(), 1)),
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "tree_method":       "hist",
        "random_state":      42,
        "n_jobs":            -1,
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost trained.")
    return model


# %% ── Model 4: LightGBM (Optuna-tuned) ─────────────────────────────────────

def _lgb_objective(
    trial: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
) -> float:
    """Optuna objective for LightGBM: minimise validation log-loss."""
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 200, 1000, step=100),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "is_unbalance":    True,
        "verbosity":       -1,
        "random_state":    42,
        "n_jobs":          -1,
    }
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        preds = model.predict_proba(X[va_idx])[:, 1]
        scores.append(log_loss(y[va_idx], preds))
    return float(np.mean(scores))


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with Optuna hyperparameter search.

    Returns
    -------
    Fitted LGBMClassifier.
    """
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    if OPTUNA_AVAILABLE:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: _lgb_objective(trial, X_all, y_all),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info("LightGBM best params: %s", best_params)
    else:
        best_params = {
            "n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05,
            "max_depth": 6, "min_child_samples": 20,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }

    best_params.update({
        "is_unbalance": True,
        "verbosity":    -1,
        "random_state": 42,
        "n_jobs":       -1,
    })

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    logger.info("LightGBM trained.")
    return model


# %% ── [Optional] Model 5: Simple Neural Network ────────────────────────────

class _MLP(nn.Module):
    """
    Two-layer feedforward network for binary classification.

    Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU
                  → Dropout → Linear (logit output)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


class TorchMLPWrapper:
    """
    Sklearn-compatible wrapper around ``_MLP`` supporting
    ``fit``, ``predict_proba``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 512,
        pos_weight_value: float = 10.0,
        device: str = "cpu",
    ):
        self.input_dim       = input_dim
        self.hidden_dim      = hidden_dim
        self.dropout         = dropout
        self.lr              = lr
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.pos_weight_val  = pos_weight_value
        self.device          = torch.device(device)
        self.model_: Optional[_MLP] = None
        self.scaler_         = StandardScaler()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TorchMLPWrapper":
        X = self.scaler_.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)

        self.model_ = _MLP(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        pos_weight  = torch.tensor([self.pos_weight_val], device=self.device)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer   = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        dataset    = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                val_ll = ""
                if X_val is not None and y_val is not None:
                    probs  = self.predict_proba(X_val)[:, 1]
                    val_ll = f"  val_logloss={log_loss(y_val, probs):.4f}"
                logger.info("MLP epoch %d/%d  train_loss=%.4f%s",
                            epoch + 1, self.epochs,
                            epoch_loss / len(loader), val_ll)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_scaled = self.scaler_.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_scaled).to(self.device))
            probs  = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])


def train_neural_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> TorchMLPWrapper:
    """
    Train the optional PyTorch MLP.

    Returns
    -------
    Fitted TorchMLPWrapper.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.  Run: pip install torch")

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training MLP on %s", device)

    wrapper = TorchMLPWrapper(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        dropout=0.3,
        lr=1e-3,
        epochs=40,
        batch_size=512,
        pos_weight_value=pos_weight,
        device=device,
    )
    wrapper.fit(X_train, y_train, X_val, y_val)
    return wrapper


# %% ── Probability calibration ───────────────────────────────────────────────

def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling (method='sigmoid') or isotonic regression
    (method='isotonic') to calibrate predicted probabilities.

    Uses the validation set with cv='prefit' (model already fitted).

    Parameters
    ----------
    model:   A fitted classifier with ``predict_proba``.
    X_val:   Validation features.
    y_val:   Validation labels.
    method:  'isotonic' (default) or 'sigmoid'.

    Returns
    -------
    CalibratedClassifierCV fitted on the validation set.
    """
    calibrated = CalibratedClassifierCV(model, cv="prefit", method=method)
    calibrated.fit(X_val, y_val)
    return calibrated


# %% ── Evaluation helpers ────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    model_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compute all evaluation metrics for a single fitted model.

    Returns
    -------
    (metrics_dict, predictions_df)

    *predictions_df* contains race meta + predicted probability, for the
    betting backtest.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Standard classification metrics
    auc      = roc_auc_score(y_test, probs)
    ll       = log_loss(y_test, probs)
    acc      = accuracy_score(y_test, preds)

    # Build prediction DataFrame for ranking metrics & backtest
    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["win_prob"] = probs

    # Ranking metrics
    map3, ndcg3 = compute_ranking_metrics(
        pred_df, "race_id", "is_winner", "win_prob", k=3
    )

    # Betting backtest
    _, bet_metrics = run_betting_backtest(
        pred_df,
        prob_col="win_prob",
        odds_col="win_odds",
        label_col="is_winner",
        race_id_col="race_id",
        horse_col="horse_name",
        ev_threshold=BET_EV_THRESHOLD,
        min_prob=BET_MIN_PROB,
        top_n_in_race=BET_TOP_N,
        kelly_frac=KELLY_FRAC,
        kelly_max=KELLY_MAX,
        starting_bankroll=STARTING_BANKROLL,
    )

    metrics = {
        "model":         model_name,
        "auc":           round(auc,  4),
        "log_loss":      round(ll,   4),
        "accuracy":      round(acc,  4),
        "map@3":         round(map3, 4),
        "ndcg@3":        round(ndcg3,4),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in bet_metrics.items()},
    }

    return metrics, pred_df


# %% ── Feature importance plot ───────────────────────────────────────────────

def plot_feature_importance(
    model: Any,
    feature_cols: List[str],
    title: str,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot top-N feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if clf and hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            logger.info("Cannot extract feature importance for %s", title)
            return
    else:
        logger.info("Cannot extract feature importance for %s", title)
        return

    n = min(top_n, len(feature_cols), len(imp))
    idx = np.argsort(imp)[::-1][:n]

    plt.figure(figsize=(10, n * 0.35 + 1))
    plt.barh(
        [feature_cols[i] if i < len(feature_cols) else f"feat_{i}" for i in idx[::-1]],
        imp[idx[::-1]],
        color="steelblue",
    )
    plt.title(f"Feature Importance – {title}")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# %% ── Summary table ─────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Pretty-print and return a comparison table for all models.

    Parameters
    ----------
    results: List of metrics dicts (one per model).

    Returns
    -------
    pd.DataFrame — one row per model.
    """
    cols = [
        "model", "auc", "log_loss", "accuracy",
        "map@3", "ndcg@3",
        "total_bets", "bet_win_rate",
        "total_staked", "net_profit", "roi_pct",
        "final_bankroll", "sharpe",
    ]
    df = pd.DataFrame(results)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    print("\n" + "=" * 110)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)

    # Highlight best model by ROI
    if "roi_pct" in df.columns and df["roi_pct"].notna().any():
        best_idx = df["roi_pct"].idxmax()
        best_roi = df.loc[best_idx, "roi_pct"]
        best_name = df.loc[best_idx, "model"]
        if best_roi > 0:
            print(f"\n✅  Best model by ROI: {best_name}  (ROI = {best_roi:.2f} %)")
        else:
            print("\n⚠  No model achieved positive ROI.")
            print("   Consider: raising EV threshold, more training data, or additional features.")

    return df


# %% ── Equity curve plot ──────────────────────────────────────────────────────

def plot_equity_curves(
    logs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot bankroll-over-time (equity) curves for all models side by side.

    Parameters
    ----------
    logs: {model_name: transaction_log_DataFrame}
    """
    plt.figure(figsize=(12, 5))
    for model_name, log in logs.items():
        if log.empty:
            continue
        plt.plot(
            range(len(log)),
            log["bankroll"].values,
            label=model_name,
            linewidth=1.5,
        )
    plt.axhline(STARTING_BANKROLL, color="black", linestyle="--",
                linewidth=1, label=f"Starting bankroll ({STARTING_BANKROLL:,.0f})")
    plt.xlabel("Bet number")
    plt.ylabel("Bankroll (units)")
    plt.title("Equity Curves – Betting Backtest (Test Set)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# %% ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full model training, evaluation, and betting backtest pipeline.

    Steps
    -----
    1.  Load splits.
    2.  Train Logistic Regression, Random Forest, XGBoost, LightGBM.
    3.  Optionally train PyTorch MLP.
    4.  Calibrate each model on the validation set.
    5.  Build ensemble (average of XGB + LGB + RF calibrated probabilities).
    6.  Evaluate every model on the test set.
    7.  Run betting backtest on the test set.
    8.  Print summary table and plots.
    9.  Save models and results.
    """
    ensure_dirs(DATA_MODELS, DATA_RESULTS)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/9] Loading splits …")
    (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        meta_train, meta_val, meta_test,
        feature_cols,
    ) = load_splits()

    pos_rate = y_train.mean()
    print(f"  Train positive rate: {pos_rate:.4f}")

    # Guard: if no data in test set, warn
    if len(X_test) == 0:
        print("⚠  Test set is empty. Check your date ranges / data.")
        return

    # ── 2. Train models ───────────────────────────────────────────────────────
    print("\n[2/9] Training Logistic Regression …")
    lr_model = train_logistic_regression(X_train, y_train)

    print("\n[3/9] Training Random Forest …")
    rf_model = train_random_forest(X_train, y_train)

    print("\n[4/9] Training XGBoost (Optuna tuning) …")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    print("\n[5/9] Training LightGBM (Optuna tuning) …")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # ── 3. Calibrate ─────────────────────────────────────────────────────────
    print("\n[6/9] Calibrating models on validation set …")
    lr_cal  = calibrate_model(lr_model,  X_val, y_val)
    rf_cal  = calibrate_model(rf_model,  X_val, y_val)
    xgb_cal = calibrate_model(xgb_model, X_val, y_val)
    lgb_cal = calibrate_model(lgb_model, X_val, y_val)

    models: Dict[str, Any] = {
        "LogisticRegression": lr_cal,
        "RandomForest":       rf_cal,
        "XGBoost":            xgb_cal,
        "LightGBM":           lgb_cal,
    }

    # ── [Optional] Neural net ────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print("\n[Optional] Training PyTorch MLP …")
        try:
            mlp_model = train_neural_net(X_train, y_train, X_val, y_val)
            mlp_cal   = calibrate_model(mlp_model, X_val, y_val)
            models["NeuralNet"] = mlp_cal
        except Exception as exc:
            print(f"  MLP skipped: {exc}")

    # ── 4. Ensemble ───────────────────────────────────────────────────────────
    print("\n[7/9] Building ensemble (XGB + LGB + RF) …")

    class _EnsembleWrapper:
        """Simple average-probability ensemble (no sklearn wrapper needed)."""

        def __init__(self, member_models: List[Any]) -> None:
            self._members = member_models

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            probs = np.mean(
                [m.predict_proba(X)[:, 1] for m in self._members], axis=0
            )
            return np.column_stack([1 - probs, probs])

    ensemble_members = [xgb_cal, lgb_cal, rf_cal]
    ensemble = _EnsembleWrapper(ensemble_members)
    models["Ensemble(XGB+LGB+RF)"] = ensemble

    # ── 5. Evaluate & backtest ────────────────────────────────────────────────
    print("\n[8/9] Evaluating on test set …")
    all_results: List[Dict]        = []
    all_logs:    Dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        print(f"  → {model_name}")
        metrics, pred_df = evaluate_model(
            model, X_test, y_test, meta_test, model_name
        )
        # Re-run backtest to capture transaction log
        _, bet_metrics = run_betting_backtest(
            pred_df,
            ev_threshold=BET_EV_THRESHOLD,
            min_prob=BET_MIN_PROB,
            top_n_in_race=BET_TOP_N,
            kelly_frac=KELLY_FRAC,
            kelly_max=KELLY_MAX,
            starting_bankroll=STARTING_BANKROLL,
        )
        log, _ = run_betting_backtest(
            pred_df,
            ev_threshold=BET_EV_THRESHOLD,
            min_prob=BET_MIN_PROB,
            top_n_in_race=BET_TOP_N,
            kelly_frac=KELLY_FRAC,
            kelly_max=KELLY_MAX,
            starting_bankroll=STARTING_BANKROLL,
        )
        all_results.append(metrics)
        all_logs[model_name] = log

        # Save predictions
        pred_df.to_parquet(
            DATA_RESULTS / f"preds_{model_name.replace(' ', '_')}.parquet",
            index=False,
        )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n[9/9] Summary …")
    summary_df = print_summary_table(all_results)
    summary_df.to_csv(DATA_RESULTS / "model_comparison.csv", index=False)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    plot_equity_curves(all_logs, save_path=DATA_RESULTS / "equity_curves.png")

    # Feature importance for XGBoost
    if feature_cols:
        plot_feature_importance(
            xgb_model, feature_cols, "XGBoost",
            save_path=DATA_RESULTS / "xgb_feature_importance.png",
        )

    # ── 8. Save models ────────────────────────────────────────────────────────
    for model_name, model in models.items():
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = DATA_MODELS / f"{safe_name}.joblib"
        try:
            joblib.dump(model, save_path)
            logger.info("Saved model → %s", save_path)
        except Exception as exc:
            logger.warning("Could not save %s: %s", model_name, exc)

    print("\n✅  Training & evaluation complete.")
    print(f"   Results saved to: {DATA_RESULTS}")
    print(f"   Models saved to:  {DATA_MODELS}")


if __name__ == "__main__":
    main()