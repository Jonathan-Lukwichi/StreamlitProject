# =============================================================================
# app_core/models/ml/stacking.py
# Stacking Generalization for Ensemble Forecasting (Step 4.2)
# Academic Reference: Wolpert (1992) - Stacked Generalization
#                     Breiman (1996) - Stacked Regressions
# =============================================================================
"""
Stacking Generalization for Time Series Forecasting.

Stacking combines predictions from multiple base models using a meta-learner:
1. Train base models (ARIMA, SARIMAX, XGBoost, LSTM, etc.)
2. Generate predictions from each base model
3. Train meta-learner on base predictions to produce final forecast

Key Advantages:
- Combines strengths of different model types
- Reduces variance through ensemble averaging
- Meta-learner learns optimal combination weights

Academic References:
    - Wolpert (1992) - "Stacked Generalization"
    - Breiman (1996) - "Stacked Regressions"
    - van der Laan et al. (2007) - "Super Learner"

Author: HealthForecast AI Research Team
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd


@dataclass
class StackingResult:
    """Store stacking ensemble results."""
    final_predictions: np.ndarray
    base_predictions: Dict[str, np.ndarray]  # Model name -> predictions
    meta_weights: Optional[np.ndarray] = None
    base_model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stacked_metrics: Dict[str, float] = field(default_factory=dict)
    meta_model_type: str = "Ridge"


@dataclass
class BaseModelResult:
    """Store individual base model results."""
    name: str
    predictions: np.ndarray
    model: Any
    metrics: Dict[str, float] = field(default_factory=dict)


class ForecastStacker:
    """
    Stacking Generalization Ensemble for Time Series Forecasting.

    Implements the stacked generalization framework:
    1. Level 0: Multiple base models (heterogeneous ensemble)
    2. Level 1: Meta-learner combines base predictions

    The meta-learner (default: Ridge regression) learns optimal weights
    for combining base model predictions.

    Usage:
        stacker = ForecastStacker(meta_model="Ridge", use_cv=True)
        stacker.add_base_model("XGBoost", xgb_model)
        stacker.add_base_model("LSTM", lstm_model)
        stacker.add_base_model("SARIMAX", sarimax_model)
        stacker.fit(X_train, y_train, X_val, y_val)
        predictions = stacker.predict(X_test)

    Academic Reference:
        Wolpert (1992) - Stacked Generalization
    """

    def __init__(
        self,
        meta_model: str = "Ridge",
        meta_params: Dict[str, Any] = None,
        use_cv: bool = True,
        cv_folds: int = 5,
        include_original_features: bool = False,
        passthrough: bool = False,
    ):
        """
        Initialize ForecastStacker.

        Args:
            meta_model: Type of meta-learner
                - "Ridge": Ridge regression (default, regularized linear)
                - "Linear": Ordinary least squares
                - "Lasso": L1 regularized regression
                - "ElasticNet": L1+L2 regularized
                - "GradientBoosting": Gradient boosted trees
                - "XGBoost": XGBoost regressor
            meta_params: Parameters for meta-model
            use_cv: Use cross-validation for generating meta-features
                    (prevents overfitting to training predictions)
            cv_folds: Number of CV folds for meta-feature generation
            include_original_features: Include original X features in meta-model
            passthrough: Pass original features to meta-learner alongside
                        base predictions (for feature augmentation)
        """
        self.meta_model_type = meta_model
        self.meta_params = meta_params or {}
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.include_original_features = include_original_features
        self.passthrough = passthrough

        # Base models: list of (name, model, trained_flag)
        self.base_models: List[Tuple[str, Any, bool]] = []
        self.meta_model = None
        self.is_fitted = False

        # Store training info
        self.n_base_models = 0
        self.feature_names: List[str] = []
        self.base_model_weights: Optional[np.ndarray] = None

    def add_base_model(
        self,
        name: str,
        model: Any,
        already_trained: bool = False,
    ) -> "ForecastStacker":
        """
        Add a base model to the ensemble.

        Args:
            name: Unique identifier for the model
            model: Model instance with fit() and predict() methods
            already_trained: If True, skip training this model

        Returns:
            Self for method chaining
        """
        # Check for duplicate names
        existing_names = [n for n, _, _ in self.base_models]
        if name in existing_names:
            raise ValueError(f"Model with name '{name}' already exists")

        self.base_models.append((name, model, already_trained))
        self.n_base_models = len(self.base_models)
        return self

    def add_base_predictions(
        self,
        name: str,
        predictions: np.ndarray,
    ) -> "ForecastStacker":
        """
        Add pre-computed predictions as a base "model".

        Useful when base model predictions are already available
        (e.g., from ARIMA/SARIMAX trained separately).

        Args:
            name: Identifier for this prediction source
            predictions: Pre-computed predictions

        Returns:
            Self for method chaining
        """
        # Create a dummy model that just returns the predictions
        class PrecomputedModel:
            def __init__(self, preds):
                self.predictions = preds
                self.current_idx = 0

            def predict(self, X):
                return self.predictions

        self.add_base_model(name, PrecomputedModel(predictions), already_trained=True)
        return self

    def _build_meta_model(self):
        """Build the meta-learner model."""
        if self.meta_model_type == "Ridge":
            from sklearn.linear_model import Ridge
            alpha = self.meta_params.get("alpha", 1.0)
            self.meta_model = Ridge(alpha=alpha)

        elif self.meta_model_type == "Linear":
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()

        elif self.meta_model_type == "Lasso":
            from sklearn.linear_model import Lasso
            alpha = self.meta_params.get("alpha", 1.0)
            self.meta_model = Lasso(alpha=alpha, max_iter=5000)

        elif self.meta_model_type == "ElasticNet":
            from sklearn.linear_model import ElasticNet
            alpha = self.meta_params.get("alpha", 1.0)
            l1_ratio = self.meta_params.get("l1_ratio", 0.5)
            self.meta_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

        elif self.meta_model_type == "GradientBoosting":
            from sklearn.ensemble import GradientBoostingRegressor
            n_estimators = self.meta_params.get("n_estimators", 100)
            max_depth = self.meta_params.get("max_depth", 3)
            self.meta_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

        elif self.meta_model_type == "XGBoost":
            try:
                from xgboost import XGBRegressor
                n_estimators = self.meta_params.get("n_estimators", 100)
                max_depth = self.meta_params.get("max_depth", 3)
                learning_rate = self.meta_params.get("learning_rate", 0.1)
                self.meta_model = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    verbosity=0
                )
            except ImportError:
                # Fallback to GradientBoosting
                from sklearn.ensemble import GradientBoostingRegressor
                self.meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                print("[WARN] XGBoost not available, using GradientBoosting as fallback")

        else:
            raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")

    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_training: bool = True,
    ) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        For training: Uses CV to prevent information leakage
        For prediction: Uses trained models directly

        Args:
            X: Input features
            y: Target values (needed for training base models)
            is_training: If True, train base models and use CV

        Returns:
            Meta-features array (n_samples, n_base_models)
        """
        n_samples = len(X)
        meta_features = np.zeros((n_samples, self.n_base_models))

        for i, (name, model, already_trained) in enumerate(self.base_models):
            if is_training and not already_trained:
                # Train base model
                if hasattr(model, 'train'):
                    model.train(X, y, None, None)
                elif hasattr(model, 'fit'):
                    model.fit(X, y)

                # Mark as trained
                self.base_models[i] = (name, model, True)

            # Generate predictions
            if hasattr(model, 'predict'):
                preds = model.predict(X)
            else:
                raise ValueError(f"Model {name} has no predict() method")

            # Handle NaN values from LSTM lookback
            if isinstance(preds, np.ndarray) and len(preds) == n_samples:
                meta_features[:, i] = np.nan_to_num(preds, nan=np.nanmean(preds))
            else:
                # Predictions might be shorter (e.g., due to lookback)
                # Pad with mean
                if len(preds) < n_samples:
                    padded = np.full(n_samples, np.nanmean(preds))
                    padded[-len(preds):] = preds
                    meta_features[:, i] = padded
                else:
                    meta_features[:, i] = preds[:n_samples]

        return meta_features

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Fit the stacking ensemble.

        1. Train all base models on training data
        2. Generate meta-features (base predictions on validation or CV)
        3. Train meta-learner on meta-features

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, uses CV if not provided)
            y_val: Validation targets

        Returns:
            Training results dictionary
        """
        if self.n_base_models == 0:
            raise ValueError("No base models added. Use add_base_model() first.")

        print("=" * 70)
        print("🔧 STACKING ENSEMBLE - Training")
        print("=" * 70)
        print(f"   Base Models: {self.n_base_models}")
        print(f"   Meta-Learner: {self.meta_model_type}")
        print(f"   Use CV: {self.use_cv}")
        print("-" * 70)

        results = {"base_models": {}, "meta_model": {}}

        # Step 1: Train base models and generate meta-features
        print("📊 Training base models...")

        if X_val is not None and y_val is not None:
            # Use validation set for meta-features
            meta_features_train = self._generate_meta_features(X_train, y_train, is_training=True)
            meta_features_val = self._generate_meta_features(X_val, y_val, is_training=False)
            meta_X = meta_features_val
            meta_y = y_val
        else:
            # Use training predictions directly (less ideal)
            meta_features_train = self._generate_meta_features(X_train, y_train, is_training=True)
            meta_X = meta_features_train
            meta_y = y_train

        # Store feature names for interpretation
        self.feature_names = [name for name, _, _ in self.base_models]

        # Include original features if requested
        if self.include_original_features or self.passthrough:
            if X_val is not None:
                meta_X = np.hstack([meta_X, X_val])
            else:
                meta_X = np.hstack([meta_X, X_train])

        # Step 2: Build and train meta-model
        print("🏗️  Training meta-learner...")
        self._build_meta_model()
        self.meta_model.fit(meta_X, meta_y)

        # Extract weights for interpretability (if linear meta-model)
        if hasattr(self.meta_model, 'coef_'):
            weights = self.meta_model.coef_[:self.n_base_models]
            self.base_model_weights = weights / np.sum(np.abs(weights))  # Normalize
            print(f"   Meta-model weights: {dict(zip(self.feature_names, weights.round(3)))}")
        else:
            self.base_model_weights = None

        # Step 3: Evaluate
        meta_pred = self.meta_model.predict(meta_X)
        stacked_metrics = self._compute_metrics(meta_y, meta_pred)
        results["stacked_metrics"] = stacked_metrics

        print(f"✓ Stacking complete")
        print(f"   Stacked MAE: {stacked_metrics['MAE']:.4f}")
        print(f"   Stacked RMSE: {stacked_metrics['RMSE']:.4f}")
        print("=" * 70)

        self.is_fitted = True
        return results

    def predict(self, X: np.ndarray) -> StackingResult:
        """
        Generate stacked ensemble predictions.

        Args:
            X: Input features

        Returns:
            StackingResult with final and base predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Generate base predictions
        base_preds = {}
        meta_features = np.zeros((len(X), self.n_base_models))

        for i, (name, model, _) in enumerate(self.base_models):
            preds = model.predict(X)
            # Handle NaN
            preds_clean = np.nan_to_num(preds, nan=np.nanmean(preds) if not np.all(np.isnan(preds)) else 0)
            base_preds[name] = preds
            meta_features[:, i] = preds_clean

        # Include original features if used during training
        if self.include_original_features or self.passthrough:
            meta_features = np.hstack([meta_features, X])

        # Meta-model prediction
        final_preds = self.meta_model.predict(meta_features)

        return StackingResult(
            final_predictions=final_preds,
            base_predictions=base_preds,
            meta_weights=self.base_model_weights,
            meta_model_type=self.meta_model_type,
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Handle NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    def get_base_model_contributions(self) -> Dict[str, float]:
        """
        Get relative contribution of each base model.

        For linear meta-models, returns normalized weights.
        For tree-based, returns feature importances.

        Returns:
            Dictionary of model name -> contribution percentage
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        if self.base_model_weights is not None:
            # Normalize to percentages
            weights = np.abs(self.base_model_weights)
            weights = weights / weights.sum() * 100
            return dict(zip(self.feature_names, weights))

        elif hasattr(self.meta_model, 'feature_importances_'):
            # Tree-based importance
            importance = self.meta_model.feature_importances_[:self.n_base_models]
            importance = importance / importance.sum() * 100
            return dict(zip(self.feature_names, importance))

        else:
            # Equal weights if not determinable
            equal = 100 / self.n_base_models
            return {name: equal for name in self.feature_names}


# =============================================================================
# SIMPLE AVERAGING ENSEMBLE (Baseline comparison)
# =============================================================================
class SimpleAverageEnsemble:
    """
    Simple averaging ensemble for comparison with stacking.

    Combines base model predictions using:
    - Simple mean (default)
    - Weighted mean (specify weights)
    - Median (more robust to outliers)

    Academic Reference:
        Bates & Granger (1969) - "The Combination of Forecasts"
    """

    def __init__(
        self,
        method: str = "mean",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize averaging ensemble.

        Args:
            method: Aggregation method ("mean", "median", "weighted")
            weights: Dict of model_name -> weight (for weighted method)
        """
        self.method = method
        self.weights = weights
        self.base_models: Dict[str, Any] = {}

    def add_base_model(self, name: str, model: Any) -> "SimpleAverageEnsemble":
        """Add a base model."""
        self.base_models[name] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []

        for name, model in self.base_models.items():
            preds = model.predict(X)
            predictions.append(preds)

        predictions = np.array(predictions)  # (n_models, n_samples)

        if self.method == "mean":
            return np.nanmean(predictions, axis=0)

        elif self.method == "median":
            return np.nanmedian(predictions, axis=0)

        elif self.method == "weighted":
            if self.weights is None:
                return np.nanmean(predictions, axis=0)

            weighted_sum = np.zeros(predictions.shape[1])
            weight_sum = 0

            for i, name in enumerate(self.base_models.keys()):
                w = self.weights.get(name, 1.0)
                weighted_sum += predictions[i] * w
                weight_sum += w

            return weighted_sum / weight_sum

        else:
            raise ValueError(f"Unknown method: {self.method}")


# =============================================================================
# BLENDING ENSEMBLE (Simplified Stacking)
# =============================================================================
class BlendingEnsemble:
    """
    Blending Ensemble - simplified stacking without CV.

    Uses a holdout set for training the blender instead of CV.
    Faster but requires larger dataset.

    Academic Reference:
        Originated from Netflix Prize competition
    """

    def __init__(
        self,
        blender_type: str = "Ridge",
        holdout_ratio: float = 0.2,
    ):
        """
        Initialize blending ensemble.

        Args:
            blender_type: Type of blender model
            holdout_ratio: Ratio of training data for blending
        """
        self.blender_type = blender_type
        self.holdout_ratio = holdout_ratio
        self.base_models: List[Tuple[str, Any]] = []
        self.blender = None
        self.is_fitted = False

    def add_base_model(self, name: str, model: Any) -> "BlendingEnsemble":
        """Add a base model."""
        self.base_models.append((name, model))
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fit blending ensemble.

        1. Split data into training and blending sets
        2. Train base models on training set
        3. Train blender on base predictions from blending set
        """
        n = len(X_train)
        split_idx = int(n * (1 - self.holdout_ratio))

        X_base = X_train[:split_idx]
        y_base = y_train[:split_idx]
        X_blend = X_train[split_idx:]
        y_blend = y_train[split_idx:]

        # Train base models
        for name, model in self.base_models:
            if hasattr(model, 'train'):
                model.train(X_base, y_base, None, None)
            elif hasattr(model, 'fit'):
                model.fit(X_base, y_base)

        # Generate blend features
        blend_features = np.zeros((len(X_blend), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            preds = model.predict(X_blend)
            blend_features[:, i] = np.nan_to_num(preds, nan=np.nanmean(preds))

        # Train blender
        if self.blender_type == "Ridge":
            from sklearn.linear_model import Ridge
            self.blender = Ridge(alpha=1.0)
        else:
            from sklearn.linear_model import LinearRegression
            self.blender = LinearRegression()

        self.blender.fit(blend_features, y_blend)
        self.is_fitted = True

        return {"status": "fitted", "n_base_models": len(self.base_models)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate blended predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        # Generate base predictions
        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            preds = model.predict(X)
            base_preds[:, i] = np.nan_to_num(preds, nan=np.nanmean(preds))

        # Blend
        return self.blender.predict(base_preds)
