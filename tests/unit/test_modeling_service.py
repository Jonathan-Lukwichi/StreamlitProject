# =============================================================================
# tests/unit/test_modeling_service.py
# Unit Tests for ModelingService
# =============================================================================

import pytest
import pandas as pd
import numpy as np


class TestModelingServiceBasics:
    """Test ModelingService basic functionality"""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_train = 200
        n_test = 50
        n_features = 10

        # Generate synthetic data
        X = np.random.randn(n_train + n_test, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_train + n_test) * 0.3

        X_train = pd.DataFrame(X[:n_train], columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(y[:n_train], name='target')
        X_test = pd.DataFrame(X[n_train:], columns=[f'feature_{i}' for i in range(n_features)])
        y_test = pd.Series(y[n_train:], name='target')

        return X_train, y_train, X_test, y_test

    def test_modeling_service_import(self, mock_streamlit):
        """Test ModelingService can be imported"""
        from app_core.services import ModelingService
        assert ModelingService is not None

    def test_modeling_service_initialization(self, mock_streamlit):
        """Test ModelingService initialization"""
        from app_core.services import ModelingService

        service = ModelingService()
        assert service is not None
        assert service.logger is not None

    def test_model_config_dataclass(self, mock_streamlit):
        """Test ModelConfig dataclass"""
        from app_core.services.modeling_service import ModelConfig

        config = ModelConfig(
            model_type='xgboost',
            target_column='Target_1',
            horizons=[1, 2, 3],
            auto_tune=False,
        )

        assert config.model_type == 'xgboost'
        assert config.target_column == 'Target_1'
        assert len(config.horizons) == 3

    def test_model_config_defaults(self, mock_streamlit):
        """Test ModelConfig default values"""
        from app_core.services.modeling_service import ModelConfig

        config = ModelConfig(
            model_type='xgboost',
            target_column='Target_1',
        )

        # Check defaults
        assert config.horizons == list(range(1, 8))
        assert config.hyperparameters == {}
        assert config.auto_tune == False
        assert config.n_trials == 50

    def test_train_xgboost_model(self, mock_streamlit, sample_training_data):
        """Test XGBoost model training"""
        from app_core.services import ModelingService
        from app_core.services.modeling_service import ModelConfig

        X_train, y_train, X_test, y_test = sample_training_data

        service = ModelingService()
        config = ModelConfig(
            model_type='xgboost',
            target_column='target',
            hyperparameters={'n_estimators': 10, 'max_depth': 3},  # Small for speed
        )

        result = service.train_model(
            X_train, y_train, X_test, y_test, config
        )

        assert result.success, f"Training failed: {result.error}"

        model_result = result.data
        assert model_result is not None
        assert model_result.model_type == 'xgboost'
        assert 'rmse' in model_result.metrics
        assert 'mae' in model_result.metrics
        assert model_result.predictions is not None
        assert len(model_result.predictions) == len(y_test)

    def test_feature_importances_returned(self, mock_streamlit, sample_training_data):
        """Test that feature importances are returned"""
        from app_core.services import ModelingService
        from app_core.services.modeling_service import ModelConfig

        X_train, y_train, X_test, y_test = sample_training_data

        service = ModelingService()
        config = ModelConfig(
            model_type='xgboost',
            target_column='target',
            hyperparameters={'n_estimators': 10, 'max_depth': 3},
        )

        result = service.train_model(
            X_train, y_train, X_test, y_test, config
        )

        assert result.success

        model_result = result.data
        assert model_result.feature_importances is not None
        assert len(model_result.feature_importances) == len(X_train.columns)


class TestModelingServiceResult:
    """Test ServiceResult and ModelResult"""

    def test_service_result_ok(self, mock_streamlit):
        """Test ServiceResult.ok() factory"""
        from app_core.services.base_service import ServiceResult

        result = ServiceResult.ok(data={'key': 'value'})

        assert result.success == True
        assert result.data == {'key': 'value'}
        assert result.error is None

    def test_service_result_fail(self, mock_streamlit):
        """Test ServiceResult.fail() factory"""
        from app_core.services.base_service import ServiceResult

        result = ServiceResult.fail(error="Something went wrong", error_code="ERR_001")

        assert result.success == False
        assert result.error == "Something went wrong"
        assert result.error_code == "ERR_001"

    def test_service_result_bool(self, mock_streamlit):
        """Test ServiceResult boolean behavior"""
        from app_core.services.base_service import ServiceResult

        ok_result = ServiceResult.ok()
        fail_result = ServiceResult.fail("error")

        assert bool(ok_result) == True
        assert bool(fail_result) == False

        # Can use in if statements
        if ok_result:
            passed = True
        else:
            passed = False
        assert passed

    def test_model_result_dataclass(self, mock_streamlit):
        """Test ModelResult dataclass"""
        from app_core.services.modeling_service import ModelResult

        result = ModelResult(
            model_name='XGBoost_Target_1',
            model_type='xgboost',
            metrics={'rmse': 1.5, 'mae': 1.2},
            predictions=np.array([1, 2, 3]),
            feature_importances={'feature_0': 0.5, 'feature_1': 0.3},
            training_time=2.5,
        )

        assert result.model_name == 'XGBoost_Target_1'
        assert result.metrics['rmse'] == 1.5
        assert result.training_time == 2.5


class TestModelingServiceComparison:
    """Test model comparison functionality"""

    def test_compare_models(self, mock_streamlit):
        """Test model comparison"""
        from app_core.services import ModelingService
        from app_core.services.modeling_service import ModelResult

        service = ModelingService()

        # Create mock results
        results = {
            'model_a': ModelResult(
                model_name='model_a',
                model_type='xgboost',
                metrics={'rmse': 2.0, 'mae': 1.5, 'mape': 10.0},
                predictions=np.array([1, 2, 3]),
                training_time=1.0,
            ),
            'model_b': ModelResult(
                model_name='model_b',
                model_type='lstm',
                metrics={'rmse': 1.5, 'mae': 1.2, 'mape': 8.0},
                predictions=np.array([1, 2, 3]),
                training_time=5.0,
            ),
        }

        comparison = service.compare_models(results, metric='rmse')

        assert comparison.success
        assert comparison.data['best_model'] == 'model_b'  # Lower RMSE
        assert comparison.data['metric_used'] == 'rmse'

    def test_compare_models_different_metric(self, mock_streamlit):
        """Test comparison with different metric"""
        from app_core.services import ModelingService
        from app_core.services.modeling_service import ModelResult

        service = ModelingService()

        results = {
            'model_a': ModelResult(
                model_name='model_a',
                model_type='xgboost',
                metrics={'rmse': 2.0, 'mae': 1.0, 'mape': 10.0},  # Best MAE
                predictions=np.array([1, 2, 3]),
                training_time=1.0,
            ),
            'model_b': ModelResult(
                model_name='model_b',
                model_type='lstm',
                metrics={'rmse': 1.5, 'mae': 1.5, 'mape': 8.0},  # Best RMSE
                predictions=np.array([1, 2, 3]),
                training_time=5.0,
            ),
        }

        # Compare by MAE
        comparison = service.compare_models(results, metric='mae')
        assert comparison.data['best_model'] == 'model_a'

        # Compare by RMSE
        comparison = service.compare_models(results, metric='rmse')
        assert comparison.data['best_model'] == 'model_b'
