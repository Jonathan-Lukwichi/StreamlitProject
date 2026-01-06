# =============================================================================
# app_core/models/ml/recursive_strategy.py
# Recursive (Iterated) Multi-Step Forecasting Strategy
# Academic Reference: Taieb et al. (2012) - A review of multi-step ahead forecasting
# =============================================================================
"""
Recursive Forecasting Strategy for Multi-Horizon Predictions.

In recursive (iterated) forecasting:
1. Train a single model to predict 1-step ahead
2. For multi-step forecasts, feed predictions back as inputs
3. Repeat until desired horizon is reached

Comparison with Direct Strategy:
- Direct: Train separate models for each horizon (h=1, h=2, ..., h=H)
- Recursive: Train one model, iterate predictions

Trade-offs:
- Recursive: More computationally efficient, but errors compound
- Direct: No error propagation, but requires H models

Academic References:
    - Taieb, Bontempi, Atiya & Sorjamaa (2012) - "A review and comparison
      of strategies for multi-step ahead time series forecasting"
    - Ben Taieb & Hyndman (2014) - "A gradient boosting approach to the
      Kaggle load forecasting competition"

Author: HealthForecast AI Research Team
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd


@dataclass
class RecursiveForecastResult:
    """Store recursive forecasting results."""
    predictions: np.ndarray  # Shape: (n_samples, horizon)
    horizons: List[int]
    strategy: str = "recursive"
    iterations: int = 0
    # Per-horizon metrics
    metrics_per_horizon: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Aggregated metrics
    overall_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DirectForecastResult:
    """Store direct forecasting results."""
    predictions: np.ndarray  # Shape: (n_samples, horizon)
    horizons: List[int]
    strategy: str = "direct"
    models_trained: int = 0
    metrics_per_horizon: Dict[int, Dict[str, float]] = field(default_factory=dict)
    overall_metrics: Dict[str, float] = field(default_factory=dict)


class RecursiveForecaster:
    """
    Recursive multi-step forecasting wrapper for any single-step model.

    This class wraps a base model (XGBoost, LSTM, ANN, etc.) and implements
    recursive prediction: predictions are fed back as inputs for subsequent steps.

    Usage:
        base_model = XGBoostPipeline(config)
        recursive = RecursiveForecaster(base_model, lag_features=['ED_lag_1', ...])
        recursive.fit(X_train, y_train)
        predictions = recursive.predict_recursive(X_test, horizon=7)

    Academic Reference:
        Taieb et al. (2012) - A review of multi-step ahead forecasting strategies
    """

    def __init__(
        self,
        base_model: Any,
        lag_feature_names: List[str],
        target_col: str = "Target_1",
        date_col: str = "datetime",
    ):
        """
        Initialize RecursiveForecaster.

        Args:
            base_model: Any trained model with predict() method
            lag_feature_names: Names of lag features (e.g., ['ED_lag_1', 'ED_lag_2', ...])
                              These will be updated during recursive prediction
            target_col: Name of the target column (prediction output)
            date_col: Name of the date column
        """
        self.base_model = base_model
        self.lag_feature_names = sorted(
            lag_feature_names,
            key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
        )
        self.target_col = target_col
        self.date_col = date_col
        self.is_fitted = False

        # Store feature order for reconstruction
        self.feature_names: List[str] = []
        self.n_lags = len(lag_feature_names)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit the base model on training data.

        The base model is trained to predict 1-step ahead (Target_1).
        For recursive forecasting, this model will be applied iteratively.

        Args:
            X_train: Training features
            y_train: Training target (1-step ahead values)
            feature_names: Names of feature columns
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Training results from base model
        """
        self.feature_names = feature_names or []

        # Find indices of lag features for later updates
        self.lag_indices = []
        for lag_name in self.lag_feature_names:
            if lag_name in self.feature_names:
                self.lag_indices.append(self.feature_names.index(lag_name))

        # Train base model
        if hasattr(self.base_model, 'train'):
            result = self.base_model.train(X_train, y_train, X_val, y_val)
        elif hasattr(self.base_model, 'fit'):
            self.base_model.fit(X_train, y_train)
            result = {"status": "fitted"}
        else:
            raise ValueError("Base model must have train() or fit() method")

        self.is_fitted = True
        return result

    def predict_recursive(
        self,
        X: np.ndarray,
        horizon: int = 7,
        return_all_steps: bool = True,
    ) -> RecursiveForecastResult:
        """
        Generate multi-step predictions using recursive strategy.

        At each step h:
        1. Predict y_hat[h] using current features
        2. Shift lag features: lag_1 <- y_hat[h], lag_2 <- lag_1, ...
        3. Repeat until horizon is reached

        Args:
            X: Input features for starting point (shape: n_samples x n_features)
            horizon: Number of steps to forecast
            return_all_steps: If True, return predictions for all horizons

        Returns:
            RecursiveForecastResult with predictions shape (n_samples, horizon)

        Academic Note:
            Errors compound in recursive forecasting. For horizon h, the prediction
            uses h-1 predicted values as inputs, propagating estimation errors.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, horizon))

        # Copy input features (we'll modify them during iteration)
        X_current = X.copy()

        for h in range(horizon):
            # Predict 1-step ahead
            if hasattr(self.base_model, 'predict'):
                y_pred = self.base_model.predict(X_current)
            else:
                raise ValueError("Base model must have predict() method")

            # Flatten if needed
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()

            # Store prediction for this horizon
            predictions[:, h] = y_pred

            # Update lag features for next iteration (shift operation)
            # lag_1 <- current prediction, lag_2 <- lag_1, etc.
            if h < horizon - 1 and len(self.lag_indices) > 0:
                X_current = self._update_lag_features(X_current, y_pred)

        return RecursiveForecastResult(
            predictions=predictions,
            horizons=list(range(1, horizon + 1)),
            strategy="recursive",
            iterations=horizon,
        )

    def _update_lag_features(
        self,
        X: np.ndarray,
        new_value: np.ndarray,
    ) -> np.ndarray:
        """
        Shift lag features and insert new prediction as lag_1.

        Example: If lags are [lag_1, lag_2, lag_3] and new_value=10:
            lag_3 <- lag_2
            lag_2 <- lag_1
            lag_1 <- new_value (10)

        Args:
            X: Current feature matrix
            new_value: Predicted value to insert as lag_1

        Returns:
            Updated feature matrix with shifted lags
        """
        X_updated = X.copy()

        # Shift lag features (from highest lag to lowest)
        for i in range(len(self.lag_indices) - 1, 0, -1):
            current_idx = self.lag_indices[i]
            previous_idx = self.lag_indices[i - 1]
            X_updated[:, current_idx] = X[:, previous_idx]

        # Insert new prediction as lag_1 (lowest lag)
        if len(self.lag_indices) > 0:
            X_updated[:, self.lag_indices[0]] = new_value

        return X_updated


class DirectForecaster:
    """
    Direct multi-step forecasting strategy.

    Trains H separate models, one for each forecast horizon.
    No error propagation, but requires more training.

    Usage:
        direct = DirectForecaster(model_factory, horizons=[1,2,3,4,5,6,7])
        direct.fit(X_train, Y_train_multi)  # Y has columns for each horizon
        predictions = direct.predict(X_test)

    Academic Reference:
        Taieb et al. (2012) - Direct strategy avoids error accumulation
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        horizons: List[int] = None,
    ):
        """
        Initialize DirectForecaster.

        Args:
            model_factory: Callable that returns a new model instance
            horizons: List of horizons to predict (default: [1,2,...,7])
        """
        self.model_factory = model_factory
        self.horizons = horizons or list(range(1, 8))
        self.models: Dict[int, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit separate models for each horizon.

        Args:
            X_train: Training features (shape: n_samples x n_features)
            Y_train: Training targets (shape: n_samples x n_horizons)
            X_val: Validation features (optional)
            Y_val: Validation targets (optional)

        Returns:
            Dictionary with training results per horizon
        """
        results = {}

        for i, h in enumerate(self.horizons):
            # Create fresh model instance
            model = self.model_factory()

            # Get target for this horizon
            y_h = Y_train[:, i] if Y_train.ndim > 1 else Y_train

            # Validation data for this horizon
            y_val_h = None
            if Y_val is not None:
                y_val_h = Y_val[:, i] if Y_val.ndim > 1 else Y_val

            # Train model
            if hasattr(model, 'train'):
                result = model.train(X_train, y_h, X_val, y_val_h)
            elif hasattr(model, 'fit'):
                model.fit(X_train, y_h)
                result = {"status": "fitted"}
            else:
                raise ValueError("Model must have train() or fit() method")

            self.models[h] = model
            results[f"horizon_{h}"] = result

        self.is_fitted = True
        return {"models_trained": len(self.horizons), "per_horizon": results}

    def predict(self, X: np.ndarray) -> DirectForecastResult:
        """
        Generate predictions for all horizons.

        Args:
            X: Input features

        Returns:
            DirectForecastResult with predictions for each horizon
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.horizons)))

        for i, h in enumerate(self.horizons):
            model = self.models[h]
            y_pred = model.predict(X)
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            predictions[:, i] = y_pred

        return DirectForecastResult(
            predictions=predictions,
            horizons=self.horizons,
            strategy="direct",
            models_trained=len(self.horizons),
        )


def compare_strategies(
    X_train: np.ndarray,
    y_train_single: np.ndarray,  # For recursive (1-step target)
    Y_train_multi: np.ndarray,   # For direct (multi-horizon targets)
    X_test: np.ndarray,
    Y_test_multi: np.ndarray,
    base_model: Any,
    model_factory: Callable[[], Any],
    lag_feature_names: List[str],
    feature_names: List[str],
    horizons: List[int] = None,
) -> Dict[str, Any]:
    """
    Compare recursive vs direct forecasting strategies.

    Args:
        X_train, y_train_single: Training data for recursive strategy
        Y_train_multi: Multi-horizon targets for direct strategy
        X_test, Y_test_multi: Test data for evaluation
        base_model: Model instance for recursive strategy
        model_factory: Factory to create models for direct strategy
        lag_feature_names: Names of lag features
        feature_names: All feature names
        horizons: Horizons to compare (default: 1-7)

    Returns:
        Dictionary comparing both strategies with metrics

    Academic Reference:
        Taieb et al. (2012) show that optimal strategy depends on:
        - Forecast horizon
        - Model complexity
        - Data characteristics (noise, trend)
    """
    horizons = horizons or list(range(1, 8))
    n_horizons = len(horizons)

    # === Recursive Strategy ===
    recursive = RecursiveForecaster(
        base_model=base_model,
        lag_feature_names=lag_feature_names,
    )
    recursive.fit(X_train, y_train_single, feature_names=feature_names)
    recursive_result = recursive.predict_recursive(X_test, horizon=n_horizons)

    # === Direct Strategy ===
    direct = DirectForecaster(model_factory=model_factory, horizons=horizons)
    direct.fit(X_train, Y_train_multi)
    direct_result = direct.predict(X_test)

    # === Compute Metrics ===
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard forecasting metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Avoid division by zero in MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    comparison = {
        "recursive": {"per_horizon": {}, "overall": {}},
        "direct": {"per_horizon": {}, "overall": {}},
    }

    # Per-horizon metrics
    recursive_all_mae, recursive_all_rmse = [], []
    direct_all_mae, direct_all_rmse = [], []

    for i, h in enumerate(horizons):
        y_true = Y_test_multi[:, i]

        # Recursive metrics for horizon h
        rec_pred = recursive_result.predictions[:, i]
        rec_metrics = compute_metrics(y_true, rec_pred)
        comparison["recursive"]["per_horizon"][h] = rec_metrics
        recursive_all_mae.append(rec_metrics["MAE"])
        recursive_all_rmse.append(rec_metrics["RMSE"])

        # Direct metrics for horizon h
        dir_pred = direct_result.predictions[:, i]
        dir_metrics = compute_metrics(y_true, dir_pred)
        comparison["direct"]["per_horizon"][h] = dir_metrics
        direct_all_mae.append(dir_metrics["MAE"])
        direct_all_rmse.append(dir_metrics["RMSE"])

    # Overall metrics (average across horizons)
    comparison["recursive"]["overall"] = {
        "Avg_MAE": np.mean(recursive_all_mae),
        "Avg_RMSE": np.mean(recursive_all_rmse),
    }
    comparison["direct"]["overall"] = {
        "Avg_MAE": np.mean(direct_all_mae),
        "Avg_RMSE": np.mean(direct_all_rmse),
    }

    # Determine winner
    rec_avg_mae = comparison["recursive"]["overall"]["Avg_MAE"]
    dir_avg_mae = comparison["direct"]["overall"]["Avg_MAE"]
    comparison["winner"] = "recursive" if rec_avg_mae < dir_avg_mae else "direct"
    comparison["improvement"] = abs(rec_avg_mae - dir_avg_mae) / max(rec_avg_mae, dir_avg_mae) * 100

    return comparison


# =============================================================================
# RECTIFY (RectifY) Strategy - Hybrid Approach
# Academic Reference: Taieb et al. (2012) - Optimal combinations
# =============================================================================
class RectifyForecaster:
    """
    RECTIFY: Combine recursive and direct strategies adaptively.

    For short horizons (h <= threshold), use direct strategy.
    For long horizons (h > threshold), use recursive strategy.

    This exploits:
    - Direct's accuracy at short horizons (no error propagation)
    - Recursive's efficiency at long horizons (one model)

    Academic Reference:
        Ben Taieb et al. (2012) - "Machine learning strategies for
        multi-step-ahead time series forecasting"
    """

    def __init__(
        self,
        recursive_model: Any,
        model_factory: Callable[[], Any],
        lag_feature_names: List[str],
        threshold: int = 3,
        horizons: List[int] = None,
    ):
        """
        Initialize RECTIFY forecaster.

        Args:
            recursive_model: Base model for recursive strategy
            model_factory: Factory for direct strategy models
            lag_feature_names: Lag feature names for recursive updating
            threshold: Horizon threshold (direct if h <= threshold)
            horizons: List of horizons to forecast
        """
        self.threshold = threshold
        self.horizons = horizons or list(range(1, 8))

        self.recursive = RecursiveForecaster(
            base_model=recursive_model,
            lag_feature_names=lag_feature_names,
        )
        self.direct = DirectForecaster(
            model_factory=model_factory,
            horizons=[h for h in self.horizons if h <= threshold],
        )

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train_single: np.ndarray,
        Y_train_multi: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        Y_val_multi: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit both recursive and direct models.

        Args:
            X_train: Training features
            y_train_single: Single-step target for recursive
            Y_train_multi: Multi-horizon targets for direct
            feature_names: Feature column names
            X_val, y_val, Y_val_multi: Validation data (optional)

        Returns:
            Training results for both strategies
        """
        results = {}

        # Fit recursive for all horizons
        rec_result = self.recursive.fit(
            X_train, y_train_single, feature_names, X_val, y_val
        )
        results["recursive"] = rec_result

        # Fit direct for short horizons only
        if len(self.direct.horizons) > 0:
            # Select only the columns for short horizons
            direct_cols = [h - 1 for h in self.direct.horizons]
            Y_direct = Y_train_multi[:, direct_cols]
            Y_val_direct = Y_val_multi[:, direct_cols] if Y_val_multi is not None else None

            dir_result = self.direct.fit(X_train, Y_direct, X_val, Y_val_direct)
            results["direct"] = dir_result

        self.is_fitted = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using RECTIFY strategy.

        Uses direct predictions for h <= threshold,
        recursive predictions for h > threshold.

        Args:
            X: Input features

        Returns:
            Predictions array (n_samples, n_horizons)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.horizons)))

        # Get recursive predictions for all horizons
        rec_result = self.recursive.predict_recursive(X, horizon=len(self.horizons))

        # Get direct predictions for short horizons
        if len(self.direct.horizons) > 0:
            dir_result = self.direct.predict(X)

        # Combine: use direct for short, recursive for long
        for i, h in enumerate(self.horizons):
            if h <= self.threshold and h in self.direct.horizons:
                # Use direct prediction
                dir_idx = self.direct.horizons.index(h)
                predictions[:, i] = dir_result.predictions[:, dir_idx]
            else:
                # Use recursive prediction
                predictions[:, i] = rec_result.predictions[:, i]

        return predictions
