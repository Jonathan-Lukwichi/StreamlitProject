# =============================================================================
# tests/integration/test_forecast_pipeline.py
# Integration Tests for Forecasting Pipeline
# =============================================================================

import pytest
import pandas as pd
import numpy as np


class TestForecastPipelineIntegration:
    """
    Integration tests for the forecasting pipeline.

    Tests multi-horizon forecasting with:
    1. Model training
    2. Prediction generation
    3. Interval computation (if CQR available)
    """

    @pytest.fixture
    def multihorizon_data(self, mock_streamlit):
        """Generate data for multi-horizon forecasting"""
        np.random.seed(42)
        n = 300

        # Create time series with trend and seasonality
        t = np.arange(n)
        trend = 0.05 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        noise = np.random.randn(n) * 3
        values = 100 + trend + seasonal + noise

        dates = pd.date_range('2022-01-01', periods=n)

        df = pd.DataFrame({'datetime': dates})

        # Create lag features
        for i in range(1, 8):
            df[f'ED_{i}'] = pd.Series(values).shift(i)

        # Create target features
        for i in range(1, 8):
            df[f'Target_{i}'] = pd.Series(values).shift(-i)

        # Add some exogenous features
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)

        return df

    def test_multihorizon_model_training(self, mock_streamlit, multihorizon_data):
        """Test training models for multiple horizons"""
        from app_core.services import DataService, ModelingService
        from app_core.services.modeling_service import ModelConfig

        df = multihorizon_data

        data_service = DataService()
        modeling_service = ModelingService()

        # Split data
        split_result = data_service.compute_temporal_split(
            df=df,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )
        assert split_result.success

        split = split_result.data
        train_idx = split['train_idx']
        test_idx = split['test_idx']

        # Define feature and target columns
        feature_cols = [c for c in df.columns
                       if c.startswith('ED_') or c in ['day_of_week', 'is_weekend']]
        target_cols = [f'Target_{i}' for i in range(1, 4)]  # First 3 horizons

        X_train = df.loc[train_idx, feature_cols]
        X_test = df.loc[test_idx, feature_cols]

        # Train model for each horizon
        horizon_results = {}
        for target_col in target_cols:
            y_train = df.loc[train_idx, target_col]
            y_test = df.loc[test_idx, target_col]

            config = ModelConfig(
                model_type='xgboost',
                target_column=target_col,
                hyperparameters={'n_estimators': 20, 'max_depth': 3},
            )

            result = modeling_service.train_model(
                X_train, y_train, X_test, y_test, config
            )

            if result.success:
                horizon_results[target_col] = result.data

        # Should have trained all horizons
        assert len(horizon_results) == len(target_cols)

        # Each horizon should have valid predictions
        for horizon, result in horizon_results.items():
            assert result.predictions is not None
            assert len(result.predictions) == len(test_idx)
            assert not np.isnan(result.predictions).any()

    def test_forecast_with_calibration_data(self, mock_streamlit, multihorizon_data):
        """Test forecasting with calibration data (for CQR)"""
        from app_core.services import DataService, ModelingService
        from app_core.services.modeling_service import ModelConfig

        df = multihorizon_data

        data_service = DataService()
        modeling_service = ModelingService()

        # Split with calibration set
        split_result = data_service.compute_temporal_split(
            df=df,
            date_column='datetime',
            train_ratio=0.70,
            cal_ratio=0.15,
        )
        assert split_result.success

        split = split_result.data

        # Verify we have all three sets
        assert split['sizes']['train'] > 0
        assert split['sizes']['cal'] > 0
        assert split['sizes']['test'] > 0

        # The calibration set can be used for conformal prediction
        # Just verify the data is properly split
        feature_cols = [c for c in df.columns if c.startswith('ED_')]

        X_train = df.loc[split['train_idx'], feature_cols]
        X_cal = df.loc[split['cal_idx'], feature_cols]
        X_test = df.loc[split['test_idx'], feature_cols]

        assert len(X_train) > len(X_cal)
        assert len(X_cal) > 0
        assert len(X_test) > 0


class TestErrorHandlingIntegration:
    """Test error handling across the pipeline"""

    def test_invalid_data_handling(self, mock_streamlit):
        """Test that invalid data is handled gracefully"""
        from app_core.services import DataService

        service = DataService()

        # Empty DataFrame
        result = service.validate_data(pd.DataFrame())
        # Should either fail gracefully or report issues
        assert result is not None

    def test_missing_columns_handling(self, mock_streamlit):
        """Test handling of missing required columns"""
        from app_core.services import DataService

        service = DataService()

        df = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': [4, 5, 6],
        })

        result = service.validate_data(
            df,
            required_columns=['col_a', 'nonexistent_column']
        )

        # Should report the missing column
        assert not result.success or not result.data.get('valid', True)

    def test_invalid_date_column_handling(self, mock_streamlit):
        """Test handling of invalid date column"""
        from app_core.services import DataService

        service = DataService()

        df = pd.DataFrame({
            'datetime': ['not', 'valid', 'dates'],
            'value': [1, 2, 3],
        })

        result = service.detect_datetime_column(df)
        # Should detect the column but may report parsing issues
        assert result is not None


class TestServiceIntegration:
    """Test integration between different services"""

    def test_data_to_modeling_handoff(self, mock_streamlit):
        """Test data flows correctly from DataService to ModelingService"""
        from app_core.services import DataService, ModelingService
        from app_core.services.modeling_service import ModelConfig

        np.random.seed(42)

        # Create data using DataService patterns
        data_service = DataService()
        modeling_service = ModelingService()

        # Create and validate data
        df = pd.DataFrame({
            'datetime': pd.date_range('2022-01-01', periods=200),
            'value': np.random.randn(200) * 10 + 100,
        })

        validation = data_service.validate_data(df)
        assert validation.success

        # Generate lag features
        lag_result = data_service.generate_lag_features(
            df, target_column='value', n_lags=5, n_horizons=3
        )
        assert lag_result.success

        processed_df = lag_result.data

        # Split data
        split_result = data_service.compute_temporal_split(
            processed_df, date_column='datetime', train_ratio=0.7, cal_ratio=0.15
        )
        assert split_result.success

        split = split_result.data

        # Prepare for modeling
        feature_cols = [f'ED_{i}' for i in range(1, 6)]
        X_train = processed_df.loc[split['train_idx'], feature_cols]
        y_train = processed_df.loc[split['train_idx'], 'Target_1']
        X_test = processed_df.loc[split['test_idx'], feature_cols]
        y_test = processed_df.loc[split['test_idx'], 'Target_1']

        # Train model
        config = ModelConfig(
            model_type='xgboost',
            target_column='Target_1',
            hyperparameters={'n_estimators': 10, 'max_depth': 2},
        )

        model_result = modeling_service.train_model(
            X_train, y_train, X_test, y_test, config
        )

        assert model_result.success
        assert model_result.data.metrics['rmse'] > 0
