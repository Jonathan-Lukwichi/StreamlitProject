from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
from .base import MLSpec, TrainConfig, MLArtifacts
from .utils import train_test_split_time, compute_metrics, permutation_importance

def train_xgb(df: pd.DataFrame, cfg: TrainConfig, params: Dict[str, Any] | None = None) -> MLArtifacts:
    params = params or {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9}

    # --- Data Preparation ---
    df = df.copy()
    if cfg.date_col and cfg.date_col in df.columns:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors='coerce')
        df = df.set_index(cfg.date_col)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex or a valid date_col specified.")

    cols: List[str] = [c for c in cfg.feature_cols if c in df.columns]
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target '{cfg.target_col}' not in dataframe.")
    if not cols:
        raise ValueError("No valid feature columns selected.")

    # Calculate test_size and val_size based on user's split_ratio
    test_size = 1 - cfg.split_ratio
    val_size = test_size * 0.5  # validation is half of test size
    train_df, val_df, test_df = train_test_split_time(df, date_col=None, test_size=test_size, val_size=val_size)
    X_train, y_train = train_df[cols], train_df[cfg.target_col]
    X_val, y_val = val_df[cols], val_df[cfg.target_col]
    X_test,  y_test  = test_df[cols],  test_df[cfg.target_col]

    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        objective="reg:squarederror",
        random_state=cfg.random_state if hasattr(cfg, 'random_state') else 42,
        n_jobs=0
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model)
    ])

    tscv = TimeSeriesSplit(n_splits=max(2, cfg.n_splits))

    # Fit the pipeline, using the validation set for early stopping if the model supports it
    # We must manually scale the validation data for early stopping, as the pipeline's scaler is not fitted yet.
    scaler = pipe.named_steps['scaler']
    scaler.fit(X_train)
    X_val_scaled = scaler.transform(X_val)
    try:
        pipe.fit(X_train, y_train, model__eval_set=[(X_val_scaled, y_val)], model__early_stopping_rounds=10)
    except TypeError as e:
        if "early_stopping_rounds" in str(e):
            # Fallback for older xgboost versions
            pipe.fit(X_train, y_train)
        else:
            raise e

    y_tr_pred = pipe.predict(X_train)
    y_te_pred = pipe.predict(X_test)

    # Compute both train and test metrics
    train_metrics = compute_metrics(y_train, y_tr_pred)
    test_metrics = compute_metrics(y_test, y_te_pred)

    # Calculate accuracy from MAPE
    train_mape = train_metrics.get("mape", float('nan'))
    test_mape = test_metrics.get("mape", float('nan'))
    train_accuracy = 100.0 - train_mape if pd.notna(train_mape) and np.isfinite(train_mape) else 0.0
    test_accuracy = 100.0 - test_mape if pd.notna(test_mape) and np.isfinite(test_mape) else 0.0

    # Combine metrics with prefixes
    metrics = {
        "train_RMSE": train_metrics.get("rmse", float('nan')),
        "train_MAE": train_metrics.get("mae", float('nan')),
        "train_MAPE": train_metrics.get("mape", float('nan')),
        "train_Accuracy": train_accuracy,
        "test_RMSE": test_metrics.get("rmse", float('nan')),
        "test_MAE": test_metrics.get("mae", float('nan')),
        "test_MAPE": test_metrics.get("mape", float('nan')),
        "test_Accuracy": test_accuracy,
    }

    def rmse(y_true, y_pred):
        return compute_metrics(y_true, y_pred)["rmse"]
    importances = permutation_importance(pipe, X_test, y_test, rmse) if len(X_test) else {}

    y_tr_pred_series = pd.Series(y_tr_pred, index=y_train.index)
    y_te_pred_series = pd.Series(y_te_pred, index=y_test.index)

    spec = MLSpec(name="XGBoost", params=params)
    return MLArtifacts(spec, cfg, pipe, pipe, cols, metrics, y_tr_pred_series, y_te_pred_series, y_train, y_test, importances)
