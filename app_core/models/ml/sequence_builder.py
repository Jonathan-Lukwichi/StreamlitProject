import numpy as np
import pandas as pd
from typing import Tuple, List

def build_supervised_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int = 28
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    X shape: (N, lookback, F), y shape: (N,)
    """
    X_list, y_list, idx = [], [], []
    # Ensure all data is numeric (float64) to avoid dtype issues
    values = df[feature_cols + [target_col]].astype(float).to_numpy()
    # use index if datetime; else fallback to 'Date' column
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else pd.RangeIndex(len(df))

    # Convert dates to array for positional indexing (handles non-zero-based indices)
    if isinstance(dates, pd.Series):
        dates = dates.values
    elif isinstance(dates, (pd.DatetimeIndex, pd.RangeIndex)):
        dates = dates.to_numpy() if hasattr(dates, 'to_numpy') else np.array(dates)

    F = len(feature_cols)
    for t in range(lookback, len(df)):
        X_list.append(values[t-lookback:t, :F])
        y_list.append(values[t, F])
        idx.append(dates[t])
    return np.asarray(X_list), np.asarray(y_list), pd.DatetimeIndex(idx)

def chronological_split(n: int, train_ratio: float=0.8):
    n_train = max(1, int(n * train_ratio))
    return slice(0, n_train), slice(n_train, n)
