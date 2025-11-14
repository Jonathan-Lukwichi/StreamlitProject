from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .base import MLSpec, TrainConfig, MLArtifacts, HybridArtifacts, BaseHybridModel
from .utils import compute_metrics, permutation_importance, train_test_split_time
from .sequence_builder import build_supervised_sequences

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    _tf_available = True
except ImportError:
    _tf_available = False


def build_lstm_model(config: Dict[str, Any]) -> "tf.keras.Model":
    """
    Builds and compiles a Keras LSTM model based on the provided configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing model parameters:
            - lookback (int): The number of time steps in each input sequence.
            - n_features (int): The number of features in the input data.
            - units (int): The number of units in the LSTM layer.
            - dropout (float): The dropout rate.
            - lr (float): The learning rate for the Adam optimizer.

    Returns:
        A compiled Keras Sequential model.
    """
    if not _tf_available:
        raise ImportError("TensorFlow/Keras is required to build the LSTM model.")

    model = Sequential([
        LSTM(int(config["units"]), input_shape=(config["lookback"], config["n_features"])),
        Dropout(float(config["dropout"])),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=float(config["lr"]))
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def train_lstm_per_horizon(df: pd.DataFrame, cfg: TrainConfig, params: Dict[str, Any]) -> MLArtifacts:
    """
    Trains an LSTM model using the provided configuration and parameters.
    This function now uses the `build_lstm_model` helper.
    """
    if not _tf_available:
        raise ImportError("TensorFlow/Keras is required for this function.")

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
    X_train_raw, y_train_raw = train_df[cols], train_df[cfg.target_col]
    X_test_raw, y_test_raw = test_df[cols], test_df[cfg.target_col]

    # Ensure target is numeric
    y_train_raw = pd.to_numeric(y_train_raw, errors='coerce')
    y_test_raw = pd.to_numeric(y_test_raw, errors='coerce')

    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Build dataframes with scaled features and target for sequence building
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=cols, index=train_df.index)
    train_scaled_df[cfg.target_col] = y_train_raw.values
    test_scaled_df = pd.DataFrame(X_test_scaled, columns=cols, index=test_df.index)
    test_scaled_df[cfg.target_col] = y_test_raw.values

    lookback = int(params.get("lookback", 28))
    X_train, y_train, train_indices = build_supervised_sequences(train_scaled_df, cols, cfg.target_col, lookback)
    X_test, y_test, test_indices = build_supervised_sequences(test_scaled_df, cols, cfg.target_col, lookback)

    # Build and train model
    model_config = {**params, "n_features": len(cols), "lookback": lookback}
    model = build_lstm_model(model_config)
    model.fit(X_train, y_train, epochs=int(params.get("epochs", 20)), batch_size=int(params.get("batch_size", 32)), verbose=0)

    # Predictions and metrics
    y_tr_pred = model.predict(X_train).flatten()
    y_te_pred = model.predict(X_test).flatten()

    # Create Series with correct index
    y_train_actual_series = pd.Series(y_train, index=train_df.index[train_indices])
    y_test_actual_series = pd.Series(y_test, index=test_df.index[test_indices])
    y_train_pred_series = pd.Series(y_tr_pred, index=train_df.index[train_indices])
    y_test_pred_series = pd.Series(y_te_pred, index=test_df.index[test_indices])

    train_metrics = compute_metrics(y_train_actual_series, y_train_pred_series)
    test_metrics = compute_metrics(y_test_actual_series, y_test_pred_series)

    # Build metrics dict with proper naming
    metrics = {}
    for k, v in (train_metrics or {}).items():
        metrics[f"train_{k.upper() if k != 'mape' else 'MAPE'}"] = v
    for k, v in (test_metrics or {}).items():
        metrics[f"test_{k.upper() if k != 'mape' else 'MAPE'}"] = v

    # Calculate accuracy from MAPE
    train_mape = train_metrics.get("mape", float('nan'))
    test_mape = test_metrics.get("mape", float('nan'))
    metrics["train_Accuracy"] = 100.0 - train_mape if np.isfinite(train_mape) else 0.0
    metrics["test_Accuracy"] = 100.0 - test_mape if np.isfinite(test_mape) else 0.0

    spec = MLSpec(name="LSTM", params=params)
    return MLArtifacts(spec, cfg, model, scaler, cols, metrics, y_train_pred_series, y_test_pred_series, y_train_actual_series, y_test_actual_series, importances=None)