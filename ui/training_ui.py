# ui/training_ui.py
"""Streamlit dashboard for model training and evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from storage.parquet_store import read

st.set_page_config(page_title="HKJC Training Dashboard", layout="wide")
st.title("🧠 Model Training Dashboard")

# ── Feature pipeline ──────────────────────────────────────────────────────────
st.header("1. Feature Pipeline")

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("▶ Build / Rebuild features_train.parquet", type="primary"):
        from features.pipeline import build_training_features
        with st.spinner("Building features …"):
            try:
                df = build_training_features()
                st.success(
                    f"Done — {len(df):,} rows, {len(df.columns)} columns. "
                    f"WIN target: {df['target_win'].mean():.1%}  "
                    f"PLACE target: {df['target_place'].mean():.1%}"
                )
            except Exception as exc:
                st.error(f"Error: {exc}")

feat_path = cfg.PROCESSED_DIR / "features_train.parquet"
if feat_path.exists():
    df_meta = read(feat_path)
    if not df_meta.empty:
        st.metric("Rows in feature table", f"{len(df_meta):,}")
        st.metric("Columns", len(df_meta.columns))

st.markdown("---")

# ── Feature importance preview ────────────────────────────────────────────────
st.header("2. Quick Feature Importance (LightGBM)")

if st.button("Show top-30 feature importances"):
    if not feat_path.exists():
        st.warning("Build the feature table first.")
    else:
        with st.spinner("Running quick LightGBM …"):
            try:
                df_feat = read(feat_path)
                exclude = {"target_win", "target_place", "race_date", "horse_id",
                           "race_id", "is_debutant", "placing_code",
                           "dividend_win", "dividend_place", "placing"}
                fcols = [c for c in df_feat.columns if c not in exclude
                         and df_feat[c].dtype in (float, int, np.float32,
                                                   np.float64, np.int32, np.int64)]
                X = df_feat[fcols].fillna(0).values
                y = df_feat["target_win"].values
                from models.lgbm_model import LGBMModel
                m = LGBMModel({"n_estimators": 200, "learning_rate": 0.1})
                split = int(len(X) * 0.8)
                m.fit(X[:split], y[:split], X[split:], y[split:])
                imp = m.feature_importance()
                import plotly.express as px
                fig = px.bar(
                    imp.head(30).reset_index(),
                    x="gain", y="index", orientation="h",
                    labels={"index": "Feature", "gain": "Gain"},
                    title="Top 30 Feature Importances (WIN, quick run)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Error: {exc}")

st.markdown("---")

# ── Model selector + hyperparameters ─────────────────────────────────────────
st.header("3. Train Models")

model_choice = st.selectbox("Model", ["lgbm", "xgb", "catboost", "nn", "ensemble"])
version      = st.text_input("Version tag", value="v1")

with st.expander("Hyperparameter overrides (LightGBM / XGB)"):
    n_est   = st.slider("n_estimators",    100, 5000, 1000, 100)
    lr      = st.slider("learning_rate",  0.001, 0.3,  0.05, 0.005, format="%.3f")
    leaves  = st.slider("num_leaves",     15, 255, 63)
    subsamp = st.slider("subsample",      0.3, 1.0, 0.8, 0.05)

train_cutoff = st.date_input(
    "Train cutoff date",
    value=pd.Timestamp(cfg.TRAIN_CUTOFF_DATE).date()
)
valid_cutoff = st.date_input(
    "Validation cutoff date",
    value=pd.Timestamp(cfg.VALID_CUTOFF_DATE).date()
)

if st.button("▶ Train", type="primary"):
    if not feat_path.exists():
        st.error("Feature table not found — build it first.")
    else:
        df_feat = read(feat_path)
        df_feat["race_date"] = pd.to_datetime(df_feat["race_date"])
        exclude = {"target_win", "target_place", "race_date", "horse_id",
                   "race_id", "is_debutant", "placing_code",
                   "dividend_win", "dividend_place", "placing"}
        fcols = [c for c in df_feat.columns if c not in exclude
                 and df_feat[c].dtype in (float, int, np.float32,
                                           np.float64, np.int32, np.int64)]
        train_df = df_feat[df_feat["race_date"] <= str(train_cutoff)]
        valid_df = df_feat[(df_feat["race_date"] > str(train_cutoff)) &
                           (df_feat["race_date"] <= str(valid_cutoff))]
        X_tr, X_va = train_df[fcols].fillna(0).values, valid_df[fcols].fillna(0).values

        for target in ["win", "place"]:
            st.write(f"Training **{model_choice}** for target=**{target}** …")
            from models.registry import get_model
            params = {"n_estimators": n_est, "learning_rate": lr,
                      "num_leaves": leaves, "subsample": subsamp}
            model = get_model(model_choice, target, version)
            y_tr  = train_df[f"target_{target}"].values
            y_va  = valid_df[f"target_{target}"].values
            try:
                with st.spinner(f"Fitting {target} model …"):
                    model.fit(X_tr, y_tr, X_va, y_va)
                save_path = cfg.MODELS_DIR / f"{model_choice}_{target}_{version}.pkl"
                model.save(save_path)
                st.success(f"Saved → {save_path}")

                # Quick AUC
                from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
                preds = model.predict_proba(X_va)
                auc   = roc_auc_score(y_va, preds)
                ll    = log_loss(y_va, preds)
                bs    = brier_score_loss(y_va, preds)
                st.metric(f"AUC ({target})",   f"{auc:.4f}")
                st.metric(f"Log-loss ({target})",f"{ll:.4f}")
                st.metric(f"Brier ({target})",  f"{bs:.4f}")
            except Exception as exc:
                st.error(f"Training error: {exc}")

st.markdown("---")

# ── Calibration plot ──────────────────────────────────────────────────────────
st.header("4. Calibration Plot")

cal_model_name = st.selectbox("Model for calibration", ["lgbm", "xgb", "catboost"])
cal_target     = st.radio("Target", ["win", "place"], horizontal=True)
cal_version    = st.text_input("Version", value="v1", key="cal_ver")

if st.button("Plot calibration"):
    import plotly.graph_objects as go
    model_path = cfg.MODELS_DIR / f"{cal_model_name}_{cal_target}_{cal_version}.pkl"
    if not model_path.exists():
        st.warning("Train and save the model first.")
    elif not feat_path.exists():
        st.warning("Feature table not found.")
    else:
        df_feat = read(feat_path)
        df_feat["race_date"] = pd.to_datetime(df_feat["race_date"])
        valid_df = df_feat[(df_feat["race_date"] > str(train_cutoff)) &
                           (df_feat["race_date"] <= str(valid_cutoff))]
        exclude = {"target_win", "target_place", "race_date", "horse_id",
                   "race_id", "is_debutant", "placing_code",
                   "dividend_win", "dividend_place", "placing"}
        fcols  = [c for c in df_feat.columns if c not in exclude
                  and df_feat[c].dtype in (float, int, np.float32,
                                            np.float64, np.int32, np.int64)]
        from models.registry import get_model
        m      = get_model(cal_model_name, cal_target, cal_version)
        preds  = m.predict_proba(valid_df[fcols].fillna(0).values)
        actual = valid_df[f"target_{cal_target}"].values

        bins      = np.linspace(0, 1, 11)
        bin_idx   = np.digitize(preds, bins) - 1
        mean_pred = [preds[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                     for i in range(len(bins))]
        mean_act  = [actual[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                     for i in range(len(bins))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=mean_act,
                                  mode="markers+lines", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                  mode="lines", name="Perfect calibration",
                                  line=dict(dash="dash")))
        fig.update_layout(title=f"Calibration — {cal_model_name}/{cal_target}",
                          xaxis_title="Mean predicted probability",
                          yaxis_title="Actual win/place rate")
        st.plotly_chart(fig, use_container_width=True)