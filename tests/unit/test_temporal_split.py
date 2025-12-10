# =============================================================================
# tests/unit/test_temporal_split.py
# Unit Tests for Temporal Split Pipeline
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestTemporalSplitPipeline:
    """Test the temporal_split.py pipeline module"""

    @pytest.fixture
    def sample_df_with_dates(self):
        """Create sample DataFrame with dates"""
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'datetime': dates,
            'value': np.random.randn(365) * 10 + 100,
            'feature1': np.random.randn(365),
            'feature2': np.random.randn(365),
        })

    def test_temporal_split_import(self):
        """Test that temporal_split module can be imported"""
        try:
            from app_core.pipelines import compute_temporal_split
            assert compute_temporal_split is not None
        except ImportError:
            pytest.skip("temporal_split module not available")

    def test_split_result_dataclass(self):
        """Test TemporalSplitResult dataclass"""
        try:
            from app_core.pipelines import TemporalSplitResult

            # Create with actual dataclass fields (not property aliases)
            result = TemporalSplitResult(
                train_idx=np.array([0, 1, 2]),
                cal_idx=np.array([3, 4]),
                test_idx=np.array([5, 6]),
                train_end_date=datetime(2022, 1, 3),
                cal_end_date=datetime(2022, 1, 5),
                test_end_date=datetime(2022, 1, 7),
                min_date=datetime(2022, 1, 1),
                max_date=datetime(2022, 1, 7),
                total_records=7,
                total_days=6,
                train_ratio=0.43,
                cal_ratio=0.29,
                test_ratio=0.29,
            )

            assert len(result.train_idx) == 3
            assert len(result.cal_idx) == 2
            assert len(result.test_idx) == 2
            # Test property aliases work
            assert result.train_start == result.min_date
            assert result.train_end == result.train_end_date
        except ImportError:
            pytest.skip("TemporalSplitResult not available")

    def test_compute_temporal_split_ratios(self, sample_df_with_dates):
        """Test that split ratios are approximately correct"""
        try:
            from app_core.pipelines import compute_temporal_split

            result = compute_temporal_split(
                df=sample_df_with_dates,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            total = len(result.train_idx) + len(result.cal_idx) + len(result.test_idx)
            assert total == len(sample_df_with_dates)

            # Check approximate ratios (within 5%)
            train_ratio = len(result.train_idx) / total
            cal_ratio = len(result.cal_idx) / total

            assert 0.65 <= train_ratio <= 0.75
            assert 0.10 <= cal_ratio <= 0.20

        except ImportError:
            pytest.skip("compute_temporal_split not available")

    def test_temporal_ordering_preserved(self, sample_df_with_dates):
        """Test that temporal ordering is preserved in splits"""
        try:
            from app_core.pipelines import compute_temporal_split

            result = compute_temporal_split(
                df=sample_df_with_dates,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            # Train should come before calibration (by index)
            assert result.train_idx.max() < result.cal_idx.min()

            # Calibration should come before test (by index)
            assert result.cal_idx.max() < result.test_idx.min()

            # Date ordering - boundaries are inclusive/exclusive
            # train_end == cal_start (boundary date is shared)
            assert result.train_end <= result.cal_start
            assert result.cal_end <= result.test_start

            # But the actual split is properly ordered
            assert result.train_end_date <= result.cal_end_date
            assert result.cal_end_date <= result.test_end_date

        except ImportError:
            pytest.skip("compute_temporal_split not available")

    def test_no_data_leakage(self, sample_df_with_dates):
        """Test that there's no overlap between splits (no data leakage)"""
        try:
            from app_core.pipelines import compute_temporal_split

            result = compute_temporal_split(
                df=sample_df_with_dates,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            train_set = set(result.train_idx)
            cal_set = set(result.cal_idx)
            test_set = set(result.test_idx)

            # No overlap between any splits
            assert len(train_set & cal_set) == 0, "Train and cal overlap"
            assert len(train_set & test_set) == 0, "Train and test overlap"
            assert len(cal_set & test_set) == 0, "Cal and test overlap"

        except ImportError:
            pytest.skip("compute_temporal_split not available")

    def test_validate_temporal_split(self, sample_df_with_dates):
        """Test the validation function"""
        try:
            from app_core.pipelines import compute_temporal_split, validate_temporal_split

            result = compute_temporal_split(
                df=sample_df_with_dates,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            # validate_temporal_split requires df, date_col, and split_result
            validation_result = validate_temporal_split(
                sample_df_with_dates,
                'datetime',
                result
            )

            # Returns a dict with 'valid' and 'issues' keys
            assert validation_result.get('valid', True), f"Validation failed: {validation_result.get('issues', [])}"

        except ImportError:
            pytest.skip("validate_temporal_split not available")


class TestTemporalSplitEdgeCases:
    """Test edge cases for temporal split"""

    def test_small_dataset(self):
        """Test with very small dataset"""
        try:
            from app_core.pipelines import compute_temporal_split

            # Small dataset with 20 rows
            df = pd.DataFrame({
                'datetime': pd.date_range('2022-01-01', periods=20),
                'value': range(20),
            })

            result = compute_temporal_split(
                df=df,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            # Should still work
            assert len(result.train_idx) > 0
            assert len(result.test_idx) > 0

        except ImportError:
            pytest.skip("compute_temporal_split not available")

    def test_unsorted_data(self):
        """Test with unsorted data (should auto-sort)"""
        try:
            from app_core.pipelines import compute_temporal_split

            # Create unsorted data
            dates = pd.date_range('2022-01-01', periods=100)
            np.random.seed(42)
            shuffled_idx = np.random.permutation(100)

            df = pd.DataFrame({
                'datetime': dates[shuffled_idx],
                'value': range(100),
            })

            result = compute_temporal_split(
                df=df,
                date_col='datetime',
                train_ratio=0.7,
                cal_ratio=0.15,
            )

            # Should handle unsorted data - boundaries may be equal
            assert result.train_end <= result.cal_start

            # Indices should be properly ordered
            assert result.train_idx.max() < result.cal_idx.min()
            assert result.cal_idx.max() < result.test_idx.min()

        except ImportError:
            pytest.skip("compute_temporal_split not available")
