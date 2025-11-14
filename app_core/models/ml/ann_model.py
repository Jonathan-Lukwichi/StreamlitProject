from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from .base import MLSpec, TrainConfig, MLArtifacts
from .utils import compute_metrics, train_test_split_time

def train_ann_tabular(df, cfg: TrainConfig, params: Dict[str, Any] | None = None) -> MLArtifacts:
    params = params or {"layers": [128, 64], "dropout": 0.1, "epochs": 20, "batch_size": 32, "lr": 1e-3}

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

    train_df, _, test_df = train_test_split_time(df, date_col=None, test_size=1 - cfg.split_ratio, val_size=0)
    X_train_df, y_train = train_df[cols], train_df[cfg.target_col]
    X_test_df,  y_test  = test_df[cols],  test_df[cfg.target_col]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    keras.backend.clear_session()
    # Input shape must be a tuple, not an integer
    model = keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],))])
    for h in params["layers"]:
        model.add(keras.layers.Dense(int(h), activation="relu"))
        if params["dropout"] > 0:
            model.add(keras.layers.Dropout(params["dropout"]))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["lr"]), loss="mse")

    es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    model.fit(X_train, y_train.to_numpy(), validation_split=0.1, epochs=params["epochs"], batch_size=params["batch_size"], callbacks=[es], verbose=0)

    y_tr_pred_arr = model.predict(X_train, verbose=0).ravel()
    y_te_pred_arr = model.predict(X_test,  verbose=0).ravel()

    y_tr_pred = pd.Series(y_tr_pred_arr, index=y_train.index)
    y_te_pred = pd.Series(y_te_pred_arr, index=y_test.index)

    # Compute both train and test metrics
    train_metrics = compute_metrics(y_train, y_tr_pred) or {}
    test_metrics = compute_metrics(y_test, y_te_pred) or {}

    # Calculate accuracy from MAPE
    train_mape = train_metrics.get("mape", float('nan'))
    test_mape = test_metrics.get("mape", float('nan'))
    train_accuracy = 100.0 - train_mape if np.isfinite(train_mape) else 0.0
    test_accuracy = 100.0 - test_mape if np.isfinite(test_mape) else 0.0

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

    spec = MLSpec(name="ANN", params=params)
    return MLArtifacts(spec, cfg, model, scaler, cols, metrics, y_tr_pred, y_te_pred, y_train, y_test, importances=None)
