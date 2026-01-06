# =============================================================================
# app_core/models/ml/xgboost_pipeline.py
# XGBoost Implementation for Time Series Forecasting
# Example implementation of BaseMLPipeline
# =============================================================================

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import itertools
from .base_ml_pipeline import BaseMLPipeline


class XGBoostPipeline(BaseMLPipeline):
    """
    XGBoost pipeline for time series forecasting.

    Extends BaseMLPipeline with XGBoost-specific implementation.
    Supports regression with gradient boosting.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost pipeline.

        Args:
            config: Configuration dict with XGBoost hyperparameters
                    - n_estimators: Number of boosting rounds
                    - max_depth: Maximum tree depth
                    - learning_rate (eta): Learning rate
                    - min_child_weight: Minimum sum of instance weight
                    - subsample: Subsample ratio of training instances
                    - colsample_bytree: Subsample ratio of columns
        """
        super().__init__(config)
        self.preprocessing_report = None

    def detect_preprocessing_state(self, X, y=None):
        """
        Check EACH variable individually for preprocessing state.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - optional

        Returns:
            Comprehensive analysis of each feature's preprocessing state
        """
        n_features = X.shape[1]
        feature_analysis = []

        # Check each feature column separately
        for i in range(n_features):
            feature_col = X[:, i]

            col_min, col_max = feature_col.min(), feature_col.max()
            col_mean, col_std = feature_col.mean(), feature_col.std()
            col_range = col_max - col_min

            # Detect scaling type for THIS specific feature
            if -0.05 <= col_min <= 0.05 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[0,1]'
                is_scaled = True
                confidence = 'HIGH'

            elif -1.05 <= col_min <= -0.95 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[-1,1]'
                is_scaled = True
                confidence = 'HIGH'

            elif abs(col_mean) < 0.3 and 0.7 < col_std < 1.3:
                scaling_type = 'StandardScaler'
                is_scaled = True
                confidence = 'MEDIUM'

            elif col_range < 5 and abs(col_mean) < 2:
                scaling_type = 'Custom/Unknown'
                is_scaled = True
                confidence = 'LOW'

            else:
                scaling_type = 'Unscaled'
                is_scaled = False
                confidence = 'HIGH'

            feature_analysis.append({
                'feature_idx': i,
                'is_scaled': is_scaled,
                'scaling_type': scaling_type,
                'confidence': confidence,
                'min': float(col_min),
                'max': float(col_max),
                'mean': float(col_mean),
                'std': float(col_std),
                'range': float(col_range)
            })

        # Check target variable separately
        target_analysis = None
        if y is not None:
            y_min, y_max = y.min(), y.max()
            y_mean, y_std = y.mean(), y.std()
            y_range = y_max - y_min

            if -0.05 <= y_min <= 0.05 and 0.95 <= y_max <= 1.05:
                target_scaling = 'MinMaxScaler[0,1]'
                target_scaled = True
                target_confidence = 'HIGH'
            elif -1.05 <= y_min <= -0.95 and 0.95 <= y_max <= 1.05:
                target_scaling = 'MinMaxScaler[-1,1]'
                target_scaled = True
                target_confidence = 'HIGH'
            elif abs(y_mean) < 0.3 and 0.7 < y_std < 1.3:
                target_scaling = 'StandardScaler'
                target_scaled = True
                target_confidence = 'MEDIUM'
            else:
                target_scaling = 'Unscaled'
                target_scaled = False
                target_confidence = 'HIGH'

            target_analysis = {
                'is_scaled': target_scaled,
                'scaling_type': target_scaling,
                'confidence': target_confidence,
                'min': float(y_min),
                'max': float(y_max),
                'mean': float(y_mean),
                'std': float(y_std),
                'range': float(y_range)
            }

        # Aggregate analysis
        scaled_count = sum(1 for f in feature_analysis if f['is_scaled'])
        high_confidence_count = sum(1 for f in feature_analysis
                                    if f['is_scaled'] and f['confidence'] == 'HIGH')

        # Determine overall state
        if scaled_count == n_features and high_confidence_count >= n_features * 0.8:
            overall_state = 'FULLY_SCALED'
            needs_scaling = False
            recommendation = 'All features appear scaled - SKIP scaling (XGBoost handles scaling well)'
        elif scaled_count == 0:
            overall_state = 'UNSCALED'
            needs_scaling = False  # XGBoost doesn't require scaling
            recommendation = 'No features scaled - OK for XGBoost (tree-based)'
        elif scaled_count < n_features * 0.5:
            overall_state = 'MOSTLY_UNSCALED'
            needs_scaling = False  # XGBoost handles this
            recommendation = 'Majority unscaled - OK for XGBoost'
        else:
            overall_state = 'PARTIALLY_SCALED'
            needs_scaling = False  # Keep as-is for XGBoost
            recommendation = 'Mixed scaling - OK for XGBoost (tree-based model)'

        return {
            'feature_analysis': feature_analysis,
            'target_analysis': target_analysis,
            'scaled_features': scaled_count,
            'total_features': n_features,
            'high_confidence_scaled': high_confidence_count,
            'overall_state': overall_state,
            'needs_scaling': needs_scaling,
            'recommendation': recommendation
        }

    def build_model(self, input_shape: Tuple = None) -> Any:
        """
        Build XGBoost regressor model.

        Args:
            input_shape: Not used for XGBoost (sklearn API)

        Returns:
            XGBoost regressor instance
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        # Extract hyperparameters from config
        params = {
            "n_estimators": int(self.config.get("n_estimators", 300)),
            "max_depth": int(self.config.get("max_depth", 6)),
            "learning_rate": float(self.config.get("eta", 0.1)),
            "min_child_weight": int(self.config.get("min_child_weight", 1)),
            "subsample": float(self.config.get("subsample", 0.8)),
            "colsample_bytree": float(self.config.get("colsample_bytree", 0.8)),
            "random_state": int(self.config.get("random_state", 42)),
            "n_jobs": int(self.config.get("n_jobs", -1)),
            "verbosity": 0,  # Suppress XGBoost warnings
        }

        self.model = XGBRegressor(**params)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train XGBoost model with intelligent preprocessing detection.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history with train and validation metrics
        """
        # Detect preprocessing state
        detection = self.detect_preprocessing_state(X_train, y_train)
        self.preprocessing_report = detection

        print("=" * 70)
        print("🔍 XGBOOST PREPROCESSING DETECTION REPORT")
        print("=" * 70)
        print(f"Overall State: {detection['overall_state']}")
        print(f"Recommendation: {detection['recommendation']}")
        print(f"Scaled Features: {detection['scaled_features']}/{detection['total_features']}")
        print("-" * 70)
        print("ℹ️  XGBoost is tree-based and doesn't require feature scaling")
        print("   Using data as-is for optimal performance")
        print("=" * 70)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Prepare eval set for early stopping
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]

        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.is_trained = True

        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        train_metrics = self.evaluate(y_train, y_train_pred)

        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_metrics = self.evaluate(y_val, y_val_pred)

        # Store training history
        self.training_history = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "n_estimators": self.model.n_estimators,
            "best_iteration": getattr(self.model, "best_iteration", None),
        }

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with XGBoost model.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     param_grid: Dict) -> Dict[str, Any]:
        """
        Grid search for XGBoost hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter bounds from config

        Returns:
            Best parameters and results
        """
        # Define parameter grid from bounds
        n_est_min = param_grid.get("n_estimators_min", 100)
        n_est_max = param_grid.get("n_estimators_max", 1000)
        depth_min = param_grid.get("depth_min", 3)
        depth_max = param_grid.get("depth_max", 12)
        eta_min = param_grid.get("eta_min", 0.01)
        eta_max = param_grid.get("eta_max", 0.3)

        # Create grid (bounded to avoid explosion)
        grid = {
            "n_estimators": [n_est_min, (n_est_min + n_est_max) // 2, n_est_max],
            "max_depth": list(range(depth_min, min(depth_max + 1, depth_min + 4))),
            "eta": [eta_min, (eta_min + eta_max) / 2, eta_max],
        }

        best_score = float("inf")
        best_params = None
        results = []

        # Grid search
        for n_est in grid["n_estimators"]:
            for depth in grid["max_depth"]:
                for eta in grid["eta"]:
                    # Update config
                    self.config.update({
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "eta": eta,
                    })

                    # Build and train
                    self.build_model()
                    self.train(X_train, y_train, X_val, y_val)

                    # Evaluate
                    y_pred = self.predict(X_val)
                    metrics = self.evaluate(y_val, y_pred)

                    # Track results
                    results.append({
                        "params": {"n_estimators": n_est, "max_depth": depth, "eta": eta},
                        "metrics": metrics
                    })

                    # Update best
                    if metrics["RMSE"] < best_score:
                        best_score = metrics["RMSE"]
                        best_params = {"n_estimators": n_est, "max_depth": depth, "eta": eta}

        # Retrain with best params
        self.config.update(best_params)
        self.build_model()
        self.train(X_train, y_train, X_val, y_val)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        }

    def _random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       param_grid: Dict, n_iter: int = 10) -> Dict[str, Any]:
        """
        Random search for XGBoost hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter bounds
            n_iter: Number of random iterations

        Returns:
            Best parameters and results
        """
        import random

        # Get bounds
        n_est_min = param_grid.get("n_estimators_min", 100)
        n_est_max = param_grid.get("n_estimators_max", 1000)
        depth_min = param_grid.get("depth_min", 3)
        depth_max = param_grid.get("depth_max", 12)
        eta_min = param_grid.get("eta_min", 0.01)
        eta_max = param_grid.get("eta_max", 0.3)

        best_score = float("inf")
        best_params = None
        results = []

        for _ in range(n_iter):
            # Random sample
            params = {
                "n_estimators": random.randint(n_est_min, n_est_max),
                "max_depth": random.randint(depth_min, depth_max),
                "eta": random.uniform(eta_min, eta_max),
            }

            # Update and train
            self.config.update(params)
            self.build_model()
            self.train(X_train, y_train, X_val, y_val)

            # Evaluate
            y_pred = self.predict(X_val)
            metrics = self.evaluate(y_val, y_pred)

            results.append({"params": params, "metrics": metrics})

            # Update best
            if metrics["RMSE"] < best_score:
                best_score = metrics["RMSE"]
                best_params = params

        # Retrain with best
        self.config.update(best_params)
        self.build_model()
        self.train(X_train, y_train, X_val, y_val)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        }

    def _bayesian_search(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         param_space: Dict) -> Dict[str, Any]:
        """
        Bayesian optimization for XGBoost (placeholder).

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_space: Parameter space

        Returns:
            Best parameters (currently uses random search as fallback)
        """
        # Placeholder - Use random search for now
        # TODO: Implement with optuna or hyperopt
        return self._random_search(X_train, y_train, X_val, y_val, param_space, n_iter=15)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained to get feature importance")

        importance = self.model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}


# =============================================================================
# TREND-RESIDUAL HYBRID XGBOOST (Step 3.3)
# Academic Reference: Hyndman & Athanasopoulos (2021) - Decomposition methods
# =============================================================================
class TrendXGBoostHybrid:
    """
    Trend-Residual Hybrid XGBoost Model.

    Decomposes time series into trend + residuals, modeling each separately:
    - Trend: Simple model (linear regression or rolling mean)
    - Residuals: XGBoost captures complex patterns

    Final prediction: y_pred = trend_pred + residual_pred

    Advantages:
    - Better extrapolation for trend (XGBoost struggles with trends)
    - XGBoost focuses on detrended patterns
    - Often better for long-horizon forecasts

    Academic Reference:
        - Hyndman & Athanasopoulos (2021) - Forecasting: Principles and Practice
        - Zhang (2003) - Time series forecasting using hybrid ARIMA-neural networks

    Usage:
        hybrid = TrendXGBoostHybrid(xgb_config, trend_method="linear")
        hybrid.fit(X_train, y_train)
        predictions = hybrid.predict(X_test)
    """

    def __init__(
        self,
        xgb_config: Dict[str, Any] = None,
        trend_method: str = "linear",
        detrend_window: int = 7,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 50,
    ):
        """
        Initialize Trend-Residual Hybrid.

        Args:
            xgb_config: XGBoost hyperparameters
            trend_method: Method to extract trend
                - "linear": Linear regression on time index
                - "rolling": Rolling mean (moving average)
                - "polynomial": Polynomial regression (degree 2)
                - "lowess": Locally weighted regression (if available)
            detrend_window: Window size for rolling method
            use_early_stopping: Enable early stopping for XGBoost
            early_stopping_rounds: Rounds for early stopping
        """
        self.xgb_config = xgb_config or {
            "n_estimators": 500,
            "max_depth": 6,
            "eta": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self.trend_method = trend_method
        self.detrend_window = detrend_window
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds

        # Models
        self.trend_model = None
        self.residual_model = None  # XGBoost

        # Training state
        self.is_fitted = False
        self.training_history = {}
        self.trend_info = {}

    def _extract_trend(self, y: np.ndarray, time_index: np.ndarray = None) -> np.ndarray:
        """
        Extract trend component from time series.

        Args:
            y: Target values
            time_index: Time index for regression (if None, uses 0,1,2,...)

        Returns:
            Trend values (same shape as y)
        """
        n = len(y)
        t = time_index if time_index is not None else np.arange(n)

        if self.trend_method == "linear":
            # Linear regression: y = a + b*t
            from sklearn.linear_model import LinearRegression
            self.trend_model = LinearRegression()
            self.trend_model.fit(t.reshape(-1, 1), y)
            trend = self.trend_model.predict(t.reshape(-1, 1))
            self.trend_info = {
                "method": "linear",
                "intercept": float(self.trend_model.intercept_),
                "slope": float(self.trend_model.coef_[0]),
            }

        elif self.trend_method == "rolling":
            # Rolling mean (centered)
            import pandas as pd
            trend = pd.Series(y).rolling(
                window=self.detrend_window, center=True, min_periods=1
            ).mean().values
            self.trend_model = {"method": "rolling", "window": self.detrend_window}
            self.trend_info = {"method": "rolling", "window": self.detrend_window}

        elif self.trend_method == "polynomial":
            # Polynomial regression (degree 2)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline

            self.trend_model = Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("lr", LinearRegression())
            ])
            self.trend_model.fit(t.reshape(-1, 1), y)
            trend = self.trend_model.predict(t.reshape(-1, 1))
            self.trend_info = {
                "method": "polynomial",
                "degree": 2,
                "coefficients": list(self.trend_model.named_steps["lr"].coef_),
            }

        elif self.trend_method == "lowess":
            # Locally Weighted Scatterplot Smoothing
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(y, t, frac=0.1, return_sorted=False)
                trend = smoothed
                self.trend_model = {"method": "lowess", "frac": 0.1}
                self.trend_info = {"method": "lowess", "frac": 0.1}
            except ImportError:
                # Fallback to rolling
                import pandas as pd
                trend = pd.Series(y).rolling(
                    window=self.detrend_window, center=True, min_periods=1
                ).mean().values
                self.trend_model = {"method": "rolling", "window": self.detrend_window}
                self.trend_info = {"method": "rolling (lowess fallback)", "window": self.detrend_window}

        else:
            raise ValueError(f"Unknown trend method: {self.trend_method}")

        return trend

    def _predict_trend(self, time_index: np.ndarray) -> np.ndarray:
        """
        Predict trend for future time points.

        Args:
            time_index: Future time indices

        Returns:
            Predicted trend values
        """
        if self.trend_method in ["linear", "polynomial"]:
            return self.trend_model.predict(time_index.reshape(-1, 1))
        elif self.trend_method in ["rolling", "lowess"]:
            # For rolling/lowess, extrapolate using last values
            # Simple linear extrapolation from last trend values
            return np.full(len(time_index), self.training_history.get("last_trend", 0))
        else:
            raise ValueError(f"Cannot predict trend for method: {self.trend_method}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        time_index: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Fit the hybrid model.

        1. Extract trend from y_train
        2. Compute residuals: residuals = y_train - trend
        3. Train XGBoost on residuals

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            time_index: Time index for trend extraction

        Returns:
            Training history with trend and residual metrics
        """
        n = len(y_train)
        time_idx = time_index if time_index is not None else np.arange(n)

        # Step 1: Extract trend
        trend = self._extract_trend(y_train, time_idx)
        self.training_history["last_trend"] = float(trend[-1])
        self.training_history["last_time_index"] = float(time_idx[-1])

        # Step 2: Compute residuals
        residuals = y_train - trend

        # Step 3: Train XGBoost on residuals
        self.residual_model = XGBoostPipeline(self.xgb_config)

        # Handle validation set
        val_residuals = None
        if X_val is not None and y_val is not None:
            # For validation, we need to extend the trend
            n_val = len(y_val)
            val_time_idx = np.arange(n, n + n_val)
            if self.trend_method in ["linear", "polynomial"]:
                val_trend = self._predict_trend(val_time_idx)
            else:
                # Use last trend value for simple extrapolation
                val_trend = np.full(n_val, trend[-1])
            val_residuals = y_val - val_trend

        # Train XGBoost on residuals
        xgb_history = self.residual_model.train(
            X_train, residuals,
            X_val, val_residuals
        )

        # Store training history
        self.training_history.update({
            "trend_method": self.trend_method,
            "trend_info": self.trend_info,
            "xgb_history": xgb_history,
            "train_trend_mean": float(np.mean(trend)),
            "train_residual_mean": float(np.mean(residuals)),
            "train_residual_std": float(np.std(residuals)),
        })

        # Evaluate on training set
        y_train_pred = self.predict(X_train, np.arange(n))
        train_metrics = self._compute_metrics(y_train, y_train_pred)
        self.training_history["train_metrics"] = train_metrics

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val, np.arange(n, n + len(y_val)))
            val_metrics = self._compute_metrics(y_val, y_val_pred)
            self.training_history["val_metrics"] = val_metrics

        self.is_fitted = True
        return self.training_history

    def predict(
        self,
        X: np.ndarray,
        time_index: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generate predictions by combining trend and residual predictions.

        Args:
            X: Input features
            time_index: Time indices for trend prediction

        Returns:
            Final predictions = trend + residual_prediction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n = len(X)

        # Predict trend
        if time_index is None:
            # Use extrapolated indices from training
            last_idx = self.training_history.get("last_time_index", 0)
            time_index = np.arange(last_idx + 1, last_idx + 1 + n)

        if self.trend_method in ["linear", "polynomial"]:
            trend_pred = self._predict_trend(time_index)
        else:
            # Linear extrapolation for rolling/lowess
            last_trend = self.training_history.get("last_trend", 0)
            trend_pred = np.full(n, last_trend)

        # Predict residuals with XGBoost
        residual_pred = self.residual_model.predict(X)

        # Combine
        return trend_pred + residual_pred

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE with zero handling
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    def get_decomposition(
        self,
        y: np.ndarray,
        time_index: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Get trend-residual decomposition.

        Args:
            y: Target values
            time_index: Time indices

        Returns:
            Dictionary with 'original', 'trend', 'residual'
        """
        n = len(y)
        t = time_index if time_index is not None else np.arange(n)
        trend = self._extract_trend(y, t)
        residual = y - trend

        return {
            "original": y,
            "trend": trend,
            "residual": residual,
            "trend_method": self.trend_method,
        }
