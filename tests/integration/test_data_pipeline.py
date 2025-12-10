# =============================================================================
# tests/integration/test_data_pipeline.py
# Integration Tests for Data Pipeline (Upload → Fusion → Processing → Split)
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDataPipelineIntegration:
    """
    Integration tests for the complete data pipeline.

    Tests the flow:
    1. Data validation
    2. Data fusion
    3. Lag feature generation
    4. Temporal split
    """

    @pytest.fixture
    def realistic_patient_data(self):
        """Generate realistic patient data"""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')

        # Realistic arrivals with weekly pattern
        base = 150
        weekly = [1.2, 1.1, 1.0, 1.0, 1.1, 0.8, 0.7]  # Mon-Sun pattern
        arrivals = []

        for i, date in enumerate(dates):
            day = date.dayofweek
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * i / 365)
            noise = np.random.normal(0, 10)
            arrivals.append(int(base * weekly[day] * seasonal + noise))

        return pd.DataFrame({
            'datetime': dates,
            'Total_Arrivals': arrivals,
            'ED_category': np.random.choice(['urgent', 'semi-urgent', 'non-urgent'], 365),
        })

    @pytest.fixture
    def realistic_weather_data(self):
        """Generate realistic weather data"""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')

        # Temperature with seasonal pattern
        day_of_year = np.arange(365)
        temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp += np.random.normal(0, 3, 365)

        return pd.DataFrame({
            'datetime': dates,
            'average_temp': temp.round(1),
            'max_temp': (temp + np.random.uniform(3, 8, 365)).round(1),
            'average_wind': np.random.uniform(5, 25, 365).round(1),
            'total_precipitation': np.random.exponential(2, 365).round(2),
        })

    @pytest.fixture
    def realistic_calendar_data(self):
        """Generate realistic calendar data"""
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')

        df = pd.DataFrame({'datetime': dates})
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['datetime'].dt.month
        df['Holiday'] = 0

        # Add holidays
        holidays = [
            '2022-01-01', '2022-01-17', '2022-02-21', '2022-05-30',
            '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-25'
        ]
        for h in holidays:
            df.loc[df['datetime'] == h, 'Holiday'] = 1

        return df

    def test_full_data_pipeline(
        self,
        mock_streamlit,
        realistic_patient_data,
        realistic_weather_data,
        realistic_calendar_data
    ):
        """Test complete data pipeline from raw data to split"""
        from app_core.services import DataService

        service = DataService()

        # Step 1: Validate each dataset
        patient_valid = service.validate_data(
            realistic_patient_data,
            required_columns=['datetime', 'Total_Arrivals']
        )
        assert patient_valid.success, f"Patient validation failed: {patient_valid.error}"

        weather_valid = service.validate_data(
            realistic_weather_data,
            required_columns=['datetime', 'average_temp']
        )
        assert weather_valid.success, f"Weather validation failed: {weather_valid.error}"

        calendar_valid = service.validate_data(
            realistic_calendar_data,
            required_columns=['datetime', 'day_of_week']
        )
        assert calendar_valid.success, f"Calendar validation failed: {calendar_valid.error}"

        # Step 2: Fuse datasets
        fusion_result = service.fuse_datasets(
            patient_df=realistic_patient_data,
            weather_df=realistic_weather_data,
            calendar_df=realistic_calendar_data,
        )
        assert fusion_result.success, f"Fusion failed: {fusion_result.error}"

        merged_df = fusion_result.data
        assert len(merged_df) > 300  # Should have most of the data

        # Step 3: Generate lag features
        lag_result = service.generate_lag_features(
            df=merged_df,
            target_column='Total_Arrivals',
            n_lags=7,
            n_horizons=7,
        )
        assert lag_result.success, f"Lag generation failed: {lag_result.error}"

        processed_df = lag_result.data

        # Check lag columns exist
        for i in range(1, 8):
            assert f'ED_{i}' in processed_df.columns, f"Missing ED_{i}"
            assert f'Target_{i}' in processed_df.columns, f"Missing Target_{i}"

        # Step 4: Compute temporal split
        split_result = service.compute_temporal_split(
            df=processed_df,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )
        assert split_result.success, f"Split failed: {split_result.error}"

        split_data = split_result.data

        # Verify split integrity
        total = (split_data['sizes']['train'] +
                split_data['sizes']['cal'] +
                split_data['sizes']['test'])
        assert total == len(processed_df)

        # Verify no overlap
        train_set = set(split_data['train_idx'])
        cal_set = set(split_data['cal_idx'])
        test_set = set(split_data['test_idx'])

        assert len(train_set & cal_set) == 0
        assert len(train_set & test_set) == 0
        assert len(cal_set & test_set) == 0

    def test_pipeline_with_missing_data(
        self,
        mock_streamlit,
        realistic_patient_data,
        realistic_weather_data,
        realistic_calendar_data
    ):
        """Test pipeline handles missing data gracefully"""
        from app_core.services import DataService

        service = DataService()

        # Introduce missing values
        patient_with_missing = realistic_patient_data.copy()
        patient_with_missing.loc[10:15, 'Total_Arrivals'] = np.nan

        # Validation should still pass but report warnings
        result = service.validate_data(patient_with_missing)
        assert result.success

        # Should have warnings about missing values
        if result.data.get('warnings'):
            assert any('missing' in str(w).lower() for w in result.data['warnings'])


class TestModelingPipelineIntegration:
    """
    Integration tests for the modeling pipeline.

    Tests the flow:
    1. Prepare features
    2. Split data
    3. Train model
    4. Evaluate
    """

    @pytest.fixture
    def prepared_modeling_data(self, mock_streamlit):
        """Generate prepared data for modeling"""
        np.random.seed(42)
        n = 300

        # Create features
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n) for i in range(10)
        })

        # Create target with some relationship to features
        y = (X['feature_0'] * 2 +
             X['feature_1'] * 0.5 +
             np.random.randn(n) * 0.3)

        # Add date column for splitting
        X['datetime'] = pd.date_range('2022-01-01', periods=n)

        return X, pd.Series(y, name='target')

    def test_model_training_pipeline(self, mock_streamlit, prepared_modeling_data):
        """Test complete model training pipeline"""
        from app_core.services import DataService, ModelingService
        from app_core.services.modeling_service import ModelConfig

        X, y = prepared_modeling_data

        data_service = DataService()
        modeling_service = ModelingService()

        # Step 1: Compute temporal split
        split_result = data_service.compute_temporal_split(
            df=X,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )
        assert split_result.success

        split = split_result.data
        train_idx = split['train_idx']
        test_idx = split['test_idx']

        # Prepare data
        feature_cols = [c for c in X.columns if c.startswith('feature_')]
        X_train = X.loc[train_idx, feature_cols]
        y_train = y.iloc[train_idx]
        X_test = X.loc[test_idx, feature_cols]
        y_test = y.iloc[test_idx]

        # Step 2: Train model
        config = ModelConfig(
            model_type='xgboost',
            target_column='target',
            hyperparameters={'n_estimators': 20, 'max_depth': 3},
        )

        train_result = modeling_service.train_model(
            X_train, y_train, X_test, y_test, config
        )
        assert train_result.success, f"Training failed: {train_result.error}"

        model_result = train_result.data

        # Step 3: Validate results
        assert model_result.metrics['rmse'] > 0
        assert model_result.metrics['mae'] > 0
        assert len(model_result.predictions) == len(y_test)

        # Predictions should be reasonable
        assert not np.isnan(model_result.predictions).any()
        assert not np.isinf(model_result.predictions).any()

    def test_multi_model_comparison(self, mock_streamlit, prepared_modeling_data):
        """Test training and comparing multiple models"""
        from app_core.services import DataService, ModelingService
        from app_core.services.modeling_service import ModelConfig, ModelResult

        X, y = prepared_modeling_data

        data_service = DataService()
        modeling_service = ModelingService()

        # Split data
        split_result = data_service.compute_temporal_split(
            df=X,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )
        split = split_result.data

        feature_cols = [c for c in X.columns if c.startswith('feature_')]
        X_train = X.loc[split['train_idx'], feature_cols]
        y_train = y.iloc[split['train_idx']]
        X_test = X.loc[split['test_idx'], feature_cols]
        y_test = y.iloc[split['test_idx']]

        # Train multiple XGBoost models with different hyperparameters
        configs = [
            ModelConfig(
                model_type='xgboost',
                target_column='target',
                hyperparameters={'n_estimators': 10, 'max_depth': 2},
            ),
            ModelConfig(
                model_type='xgboost',
                target_column='target',
                hyperparameters={'n_estimators': 20, 'max_depth': 4},
            ),
        ]

        results = {}
        for i, config in enumerate(configs):
            result = modeling_service.train_model(
                X_train, y_train, X_test, y_test, config
            )
            if result.success:
                results[f'model_{i}'] = result.data

        assert len(results) >= 1, "At least one model should train successfully"

        # Compare models
        if len(results) >= 2:
            comparison = modeling_service.compare_models(results, metric='rmse')
            assert comparison.success
            assert comparison.data['best_model'] in results
