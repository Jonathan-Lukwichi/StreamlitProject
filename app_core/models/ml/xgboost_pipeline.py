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
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history with train and validation metrics
        """
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
