# =============================================================================
# tests/unit/test_data_service.py
# Unit Tests for DataService
# =============================================================================

import pytest
import pandas as pd
import numpy as np


class TestDataServiceValidation:
    """Test data validation functionality"""

    def test_validate_empty_dataframe_fails(self, mock_streamlit):
        """Empty DataFrame should fail validation"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.validate_data(pd.DataFrame())

        # Should fail due to empty DataFrame
        assert not result.success or not result.data.get('valid', True)

    def test_validate_with_required_columns_success(self, mock_streamlit, sample_patient_df):
        """Validation passes when required columns present"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.validate_data(
            sample_patient_df,
            required_columns=['datetime', 'Total_Arrivals']
        )

        assert result.success
        assert result.data['valid']

    def test_validate_missing_columns_fails(self, mock_streamlit, sample_patient_df):
        """Validation fails when required columns missing"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.validate_data(
            sample_patient_df,
            required_columns=['datetime', 'nonexistent_column']
        )

        # Should fail due to missing column
        assert not result.success or not result.data.get('valid', True)


class TestDataServiceDatetimeDetection:
    """Test datetime column detection"""

    def test_detect_datetime_column(self, mock_streamlit, sample_patient_df):
        """Should detect 'datetime' column"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.detect_datetime_column(sample_patient_df)

        assert result.success
        assert result.data['column'] == 'datetime'

    def test_detect_alternative_date_column(self, mock_streamlit):
        """Should detect columns with date-like names"""
        from app_core.services.data_service import DataService

        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'value': range(10),
        })

        service = DataService()
        result = service.detect_datetime_column(df)

        assert result.success
        assert result.data['column'] == 'Date'


class TestDataServiceTemporalSplit:
    """Test temporal split functionality"""

    def test_temporal_split_ratios(self, mock_streamlit, sample_processed_df):
        """Split indices should match specified ratios"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.compute_temporal_split(
            sample_processed_df,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )

        assert result.success

        data = result.data
        total = data['sizes']['train'] + data['sizes']['cal'] + data['sizes']['test']

        assert total == len(sample_processed_df)
        assert data['sizes']['train'] > data['sizes']['cal']
        assert data['sizes']['train'] > data['sizes']['test']

    def test_temporal_split_no_overlap(self, mock_streamlit, sample_processed_df):
        """Split indices should not overlap"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.compute_temporal_split(
            sample_processed_df,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )

        assert result.success

        train_set = set(result.data['train_idx'])
        cal_set = set(result.data['cal_idx'])
        test_set = set(result.data['test_idx'])

        # No overlap between sets
        assert len(train_set & cal_set) == 0
        assert len(train_set & test_set) == 0
        assert len(cal_set & test_set) == 0

    def test_temporal_ordering(self, mock_streamlit, sample_processed_df):
        """Train indices should come before cal, cal before test"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.compute_temporal_split(
            sample_processed_df,
            date_column='datetime',
            train_ratio=0.7,
            cal_ratio=0.15,
        )

        assert result.success

        train_idx = result.data['train_idx']
        cal_idx = result.data['cal_idx']
        test_idx = result.data['test_idx']

        # Max train < min cal < max cal < min test
        assert train_idx.max() < cal_idx.min()
        assert cal_idx.max() < test_idx.min()


class TestDataServiceLagFeatures:
    """Test lag feature generation"""

    def test_generate_lag_features(self, mock_streamlit, sample_merged_df):
        """Should generate ED_1 to ED_n lag features"""
        from app_core.services.data_service import DataService

        service = DataService()
        result = service.generate_lag_features(
            sample_merged_df,
            target_column='Total_Arrivals',
            n_lags=7,
            n_horizons=7,
        )

        assert result.success

        df = result.data

        # Check lag features exist
        for i in range(1, 8):
            assert f'ED_{i}' in df.columns
            assert f'Target_{i}' in df.columns

        # Check no NaN values
        assert df.isna().sum().sum() == 0
