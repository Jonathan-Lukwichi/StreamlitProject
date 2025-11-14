from __future__ import annotations
from typing import List, Callable, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .sequence_builder import build_supervised_sequences


def build_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Creates calendar-based features from a datetime column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): The name of the column containing datetime objects.

    Returns:
        pd.DataFrame: DataFrame with new calendar features.
    """
    dates = pd.to_datetime(df[date_col])
    cal_df = pd.DataFrame(index=df.index)
    cal_df["month"] = dates.dt.month
    cal_df["day_of_week"] = dates.dt.dayofweek  # Monday=0, Sunday=6
    cal_df["day_of_month"] = dates.dt.day
    cal_df["day_of_year"] = dates.dt.dayofyear
    cal_df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    cal_df["quarter"] = dates.dt.quarter
    cal_df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    return cal_df


def make_lags(df: pd.DataFrame, target: str, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    """
    Creates lag and rolling window features for a target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): The name of the target column.
        lags (List[int]): List of lag periods to create (e.g., [1, 7, 14]).
        rolls (List[int]): List of window sizes for rolling means (e.g., [7, 14]).

    Returns:
        pd.DataFrame: DataFrame with new lag and rolling features.
    """
    df_out = pd.DataFrame(index=df.index)
    for lag in lags:
        df_out[f"{target}_lag_{lag}"] = df[target].shift(lag)

    for roll in rolls:
        df_out[f"{target}_roll_mean_{roll}"] = df[target].shift(1).rolling(window=roll).mean()

    return df_out


def expanding_oof_forecast(
    model_fn: Callable[..., Any],
    series_df: pd.DataFrame,
    y: pd.Series,
    splits: int,
    config: dict,
) -> pd.Series:
    """
    Generates out-of-fold (OOF) predictions using an expanding window approach.
    This is crucial for creating residual features for a stage-2 model without leakage.

    Args:
        model_fn (Callable): A function that takes (X_train, y_train, config) and returns a trained model.
        series_df (pd.DataFrame): DataFrame containing all features.
        y (pd.Series): The target time series.
        splits (int): The number of cross-validation splits.
        config (dict): Configuration dictionary passed to the model_fn.

    Returns:
        pd.Series: A series of out-of-fold predictions, aligned with the original `y` series index.
    """
    tscv = TimeSeriesSplit(n_splits=splits)
    oof_preds = []
    oof_indices = []

    for train_idx, val_idx in tscv.split(series_df):
        # Prepare training and validation data for this fold
        X_train, X_val = series_df.iloc[train_idx], series_df.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # The model function is responsible for its own sequence generation if needed
        model = model_fn(X_train, y_train, config)

        # Prepare validation data for prediction (may need sequencing)
        if config.get("use_sequences", False):
            # For sequence models, we need to build sequences for the validation set
            # Ensure X_val and y_val are DataFrame/Series
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val, name="target")

            df_val_combined = pd.concat([X_val, y_val], axis=1)
            X_val_seq, _, _ = build_supervised_sequences(
                df_val_combined,
                list(X_val.columns),
                y_val.name,
                config["lookback"]
            )
            fold_preds = model.predict(X_val_seq)
        else:
            # For standard tabular models
            fold_preds = model.predict(X_val)

        oof_preds.append(fold_preds.flatten())
        oof_indices.append(y_val.index)

    # Concatenate all OOF predictions and create a pandas Series
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_indices = np.concatenate(oof_indices)

    return pd.Series(all_oof_preds, index=all_oof_indices, name="oof_preds")