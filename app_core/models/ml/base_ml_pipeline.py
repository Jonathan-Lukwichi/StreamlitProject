# =============================================================================
# app_core/models/ml/base_ml_pipeline.py
# Abstract Base Class for All ML Forecasting Models
# Enforces a unified workflow across LSTM, ANN, XGBoost, and future models
# =============================================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


class BaseMLPipeline(ABC):
    """
    Abstract base class for all ML forecasting models.

    Provides:
    - Generic data preprocessing (scaling, splitting)
    - Standard evaluation metrics (MAE, RMSE, MAPE, Accuracy, R²)
    - Time series cross-validation
    - Hyperparameter search (Grid, Random, Bayesian)
    - Model save/load functionality

    Subclasses must implement:
    - build_model()
    - train()
    - predict()
    - _grid_search()
    - _random_search()
    - _bayesian_search()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML pipeline.

        Args:
            config: Model configuration from model_registry or user input
                    Contains hyperparameters, search settings, etc.
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = {}

    # -------------------------------------------------------------------------
    # Abstract Methods (Must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def build_model(self, input_shape: Tuple) -> Any:
        """
        Build the model architecture.

        Args:
            input_shape: Shape of input data (n_features,) or (timesteps, n_features)

        Returns:
            Compiled model instance
        """
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history/results dictionary
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     param_grid: Dict) -> Dict[str, Any]:
        """
        Model-specific grid search implementation.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter grid for search

        Returns:
            Best parameters and results
        """
        pass

    @abstractmethod
    def _random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       param_grid: Dict, n_iter: int = 10) -> Dict[str, Any]:
        """
        Model-specific random search implementation.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter distributions for search
            n_iter: Number of iterations

        Returns:
            Best parameters and results
        """
        pass

    @abstractmethod
    def _bayesian_search(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         param_space: Dict) -> Dict[str, Any]:
        """
        Model-specific Bayesian optimization implementation.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_space: Parameter space for optimization

        Returns:
            Best parameters and results
        """
        pass

    # -------------------------------------------------------------------------
    # Generic Methods (Shared across all models)
    # -------------------------------------------------------------------------

    def preprocess_data(self, df: pd.DataFrame, target_col: str,
                       feature_cols: list, train_ratio: float = 0.8,
                       scale_features: bool = True) -> Tuple:
        """
        Generic data preprocessing pipeline.

        Args:
            df: Input dataframe
            target_col: Name of target column
            feature_cols: List of feature column names
            train_ratio: Train/test split ratio
            scale_features: Whether to scale features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values

        # Split train/test (time series - no shuffle)
        split_idx = int(len(df) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features if requested
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard forecasting metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics (MAE, RMSE, MAPE, Accuracy, R²)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Ensure arrays are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Calculate metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # MAPE with safety for zero values
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
        accuracy = float(100 - mape)

        try:
            r2 = float(r2_score(y_true, y_pred))
        except:
            r2 = np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Accuracy": accuracy,
            "R2": r2,
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       n_folds: int = 5) -> Dict[str, list]:
        """
        Time series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_folds: Number of CV folds

        Returns:
            Dictionary of metric lists for each fold
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_folds)
        cv_scores = {
            "MAE": [],
            "RMSE": [],
            "MAPE": [],
            "Accuracy": [],
            "R2": []
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train on fold
            self.train(X_train, y_train, X_val, y_val)

            # Predict and evaluate
            y_pred = self.predict(X_val)
            metrics = self.evaluate(y_val, y_pred)

            # Store results
            for key in cv_scores:
                cv_scores[key].append(metrics[key])

        return cv_scores

    def hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             search_method: str = "grid",
                             param_config: Dict = None,
                             n_iter: int = 10) -> Dict[str, Any]:
        """
        Generic hyperparameter search dispatcher.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            search_method: Search strategy ("grid", "random", "bayesian")
            param_config: Parameter configuration
            n_iter: Number of iterations for random search

        Returns:
            Best parameters and search results
        """
        if param_config is None:
            param_config = {}

        method = search_method.lower()

        if "grid" in method:
            return self._grid_search(X_train, y_train, X_val, y_val, param_config)
        elif "random" in method:
            return self._random_search(X_train, y_train, X_val, y_val, param_config, n_iter)
        elif "bayesian" in method:
            return self._bayesian_search(X_train, y_train, X_val, y_val, param_config)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

    def save(self, filepath: str):
        """
        Save model and preprocessing artifacts to disk.

        Args:
            filepath: Path to save model
        """
        import joblib

        save_dict = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
        }

        joblib.dump(save_dict, filepath)

    def load(self, filepath: str):
        """
        Load model and preprocessing artifacts from disk.

        Args:
            filepath: Path to saved model
        """
        import joblib

        save_dict = joblib.load(filepath)

        self.model = save_dict["model"]
        self.scaler = save_dict.get("scaler")
        self.config = save_dict.get("config", {})
        self.is_trained = save_dict.get("is_trained", True)
        self.training_history = save_dict.get("training_history", {})

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary with model details
        """
        return {
            "model_class": self.__class__.__name__,
            "config": self.config,
            "is_trained": self.is_trained,
            "has_scaler": self.scaler is not None,
        }
