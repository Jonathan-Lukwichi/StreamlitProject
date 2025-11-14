from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import random
import os

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def time_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    test_size: int | float = 0.2,
    val_size: int | float = 0.1,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Deterministic time-based split.
    Returns train_idx, val_idx, test_idx (as df indices), with no shuffling.

    test_size / val_size can be fractions (0..1) or absolute row counts.
    """
    if date_col not in df.columns:
        raise KeyError(f"{date_col=} not found in dataframe")

    df = df.sort_values(date_col)
    n = len(df)

    def _as_count(x):
        return int(x) if isinstance(x, int) else int(np.floor(n * float(x)))

    n_test = _as_count(test_size)
    n_val  = _as_count(val_size)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough rows ({n}) for val_size={val_size} and test_size={test_size}"
        )

    train_idx = df.index[:n_train]
    val_idx   = df.index[n_train:n_train + n_val] if n_val > 0 else pd.Index([])
    test_idx  = df.index[n_train + n_val:]
    return train_idx, val_idx, test_idx


def rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    """Compute Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def compute_metrics(y_true, y_pred) -> dict:
    """Computes a dictionary of standard regression metrics."""
    return {"rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred)}

def save_artifacts(obj: Any, path: str):
    """Save an object to a file using joblib."""
    joblib.dump(obj, path)

def load_artifacts(path: str) -> Any:
    """Load an object from a file using joblib."""
    return joblib.load(path)

def permutation_importance(model, X: pd.DataFrame, y: pd.Series, metric_fn) -> Dict[str, float]:
    # Baseline (lower metric is better)
    base = metric_fn(y, model.predict(X))
    imp = {}
    rng = np.random.default_rng(42)
    Xc = X.copy()
    for col in X.columns:
        saved = Xc[col].to_numpy().copy()
        shuffled = saved.copy()
        rng.shuffle(shuffled)
        Xc[col] = shuffled
        score = metric_fn(y, model.predict(Xc))
        imp[col] = score - base
        Xc[col] = saved
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))

def train_test_split_time(
    df: pd.DataFrame,
    date_col: str | None = "Date",
    test_size: int | float = 0.2,
    val_size: int | float = 0.1,
):
    """
    Split a time-ordered dataframe into train/val/test sets without shuffling.
    Returns (train_df, val_df, test_df).
    test_size and val_size can be fractions (0-1) or absolute counts.

    If date_col is None, assumes df has a DatetimeIndex.
    """
    # Handle DatetimeIndex case
    if date_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("If date_col is None, df must have a DatetimeIndex")
        df = df.sort_index()
    else:
        if date_col not in df.columns:
            raise KeyError(f"{date_col} not found in dataframe")
        df = df.sort_values(date_col)

    n = len(df)

    def _as_count(x):
        return int(x) if isinstance(x, int) else int(np.floor(n * float(x)))

    n_test = _as_count(test_size)
    n_val  = _as_count(val_size)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough rows ({n}) for given val/test sizes")

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df

__all__ = [
    "set_seed", "time_split", "rmse", "mae", "mape", "compute_metrics",
    "save_artifacts", "load_artifacts", "permutation_importance",
    "train_test_split_time",
]
