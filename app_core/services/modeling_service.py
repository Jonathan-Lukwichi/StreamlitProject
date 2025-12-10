# =============================================================================
# app_core/services/modeling_service.py
# Modeling Service - Business Logic for ML Model Training
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from .base_service import BaseService, ServiceResult
from app_core.errors import ModelTrainingError


@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str  # 'xgboost', 'lstm', 'ann', 'hybrid_lstm_xgb', etc.
    target_column: str
    horizons: List[int] = None
    hyperparameters: Dict[str, Any] = None
    auto_tune: bool = False
    n_trials: int = 50

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = list(range(1, 8))  # Default: 7 horizons
        if self.hyperparameters is None:
            self.hyperparameters = {}


@dataclass
class ModelResult:
    """Result container for trained model"""
    model_name: str
    model_type: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    feature_importances: Optional[Dict[str, float]] = None
    model_object: Any = None
    training_time: float = 0.0


class ModelingService(BaseService):
    """
    Service for training and evaluating ML models.

    This service encapsulates all model training logic, separating it from
    the UI layer. It provides a clean interface for:
    - Training individual models (XGBoost, LSTM, ANN)
    - Training hybrid models
    - Computing CQR prediction intervals
    - Model comparison and selection

    Usage:
        service = ModelingService()

        # Train a single model
        result = service.train_model(
            X_train, y_train, X_test, y_test,
            config=ModelConfig(model_type='xgboost', target_column='Target_1')
        )

        if result.success:
            model_result = result.data
            print(f"RMSE: {model_result.metrics['rmse']}")
    """

    def __init__(self):
        super().__init__()
        self._models = {}

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ModelConfig,
        X_cal: Optional[pd.DataFrame] = None,
        y_cal: Optional[pd.Series] = None,
    ) -> ServiceResult:
        """
        Train a model with the given configuration.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            config: Model configuration
            X_cal: Calibration features (optional, for CQR)
            y_cal: Calibration targets (optional, for CQR)

        Returns:
            ServiceResult containing ModelResult on success
        """
        operation = f"Training {config.model_type} model"

        def _train():
            import time
            start_time = time.time()

            self._update_progress(10, "Initializing model...")

            # Select and train model based on type
            if config.model_type == 'xgboost':
                result = self._train_xgboost(
                    X_train, y_train, X_test, y_test, config
                )
            elif config.model_type == 'lstm':
                result = self._train_lstm(
                    X_train, y_train, X_test, y_test, config
                )
            elif config.model_type == 'ann':
                result = self._train_ann(
                    X_train, y_train, X_test, y_test, config
                )
            else:
                raise ModelTrainingError(
                    f"Unknown model type: {config.model_type}",
                    model_type=config.model_type,
                )

            result.training_time = time.time() - start_time
            self._update_progress(100, "Training complete")

            return result

        return self.safe_execute(operation, _train)

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ModelConfig,
    ) -> ModelResult:
        """Train XGBoost model"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import xgboost as xgb

        self._update_progress(20, "Configuring XGBoost...")

        # Default hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        params.update(config.hyperparameters)

        self._update_progress(30, "Training XGBoost model...")

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        self._update_progress(70, "Generating predictions...")

        predictions = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        self._update_progress(90, "Extracting feature importances...")

        # Feature importances
        importances = dict(zip(X_train.columns, model.feature_importances_))

        return ModelResult(
            model_name=f"XGBoost_{config.target_column}",
            model_type='xgboost',
            metrics={'rmse': rmse, 'mae': mae, 'mape': mape},
            predictions=predictions,
            feature_importances=importances,
            model_object=model,
        )

    def _train_lstm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ModelConfig,
    ) -> ModelResult:
        """Train LSTM model (placeholder - would use actual LSTM pipeline)"""
        # This would delegate to the actual LSTM pipeline
        # For now, return a placeholder
        raise ModelTrainingError(
            "LSTM training requires TensorFlow. Use the full LSTM pipeline.",
            model_type='lstm',
        )

    def _train_ann(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ModelConfig,
    ) -> ModelResult:
        """Train ANN model (placeholder - would use actual ANN pipeline)"""
        raise ModelTrainingError(
            "ANN training requires TensorFlow. Use the full ANN pipeline.",
            model_type='ann',
        )

    def train_multihorizon(
        self,
        X: pd.DataFrame,
        y_multi: Dict[str, pd.Series],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        config: ModelConfig,
        cal_idx: Optional[np.ndarray] = None,
    ) -> ServiceResult:
        """
        Train models for multiple horizons.

        Args:
            X: Feature matrix
            y_multi: Dictionary mapping horizon names to target series
            train_idx: Training set indices
            test_idx: Test set indices
            config: Model configuration
            cal_idx: Calibration set indices (optional)

        Returns:
            ServiceResult containing dict of horizon -> ModelResult
        """
        operation = f"Training {config.model_type} for {len(y_multi)} horizons"

        def _train_all():
            results = {}
            total_horizons = len(y_multi)

            for i, (horizon_name, y) in enumerate(y_multi.items()):
                progress = int((i / total_horizons) * 100)
                self._update_progress(progress, f"Training {horizon_name}...")

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # Create horizon-specific config
                horizon_config = ModelConfig(
                    model_type=config.model_type,
                    target_column=horizon_name,
                    hyperparameters=config.hyperparameters,
                )

                result = self.train_model(
                    X_train, y_train, X_test, y_test, horizon_config
                )

                if result.success:
                    results[horizon_name] = result.data
                else:
                    self.logger.warning(
                        f"Failed to train {horizon_name}: {result.error}"
                    )

            return results

        return self.safe_execute(operation, _train_all)

    def compare_models(
        self,
        results: Dict[str, ModelResult],
        metric: str = 'rmse'
    ) -> ServiceResult:
        """
        Compare multiple trained models and select the best.

        Args:
            results: Dictionary of model_name -> ModelResult
            metric: Metric to use for comparison ('rmse', 'mae', 'mape')

        Returns:
            ServiceResult containing best model name and comparison table
        """
        def _compare():
            comparison = []
            for name, result in results.items():
                comparison.append({
                    'model': name,
                    'rmse': result.metrics.get('rmse', np.inf),
                    'mae': result.metrics.get('mae', np.inf),
                    'mape': result.metrics.get('mape', np.inf),
                    'training_time': result.training_time,
                })

            df = pd.DataFrame(comparison)
            best_idx = df[metric].idxmin()
            best_model = df.loc[best_idx, 'model']

            return {
                'best_model': best_model,
                'comparison_table': df,
                'metric_used': metric,
            }

        return self.safe_execute("Comparing models", _compare)
