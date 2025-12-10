# =============================================================================
# tests/conftest.py
# Pytest Configuration and Fixtures
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_dates():
    """Generate sample date range"""
    return pd.date_range(start='2023-01-01', periods=365, freq='D')


@pytest.fixture
def sample_patient_df(sample_dates):
    """Generate sample patient arrivals data"""
    np.random.seed(42)
    n = len(sample_dates)

    # Realistic patient arrivals with weekly pattern
    base = 150
    weekly_pattern = np.array([1.2, 1.1, 1.0, 1.0, 1.1, 0.8, 0.7])
    arrivals = []

    for i, date in enumerate(sample_dates):
        day_of_week = date.dayofweek
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
        noise = np.random.normal(0, 10)
        arrivals.append(int(base * weekly_pattern[day_of_week] * seasonal + noise))

    return pd.DataFrame({
        'datetime': sample_dates,
        'Total_Arrivals': arrivals,
    })


@pytest.fixture
def sample_weather_df(sample_dates):
    """Generate sample weather data"""
    np.random.seed(42)
    n = len(sample_dates)

    # Temperature follows seasonal pattern
    day_of_year = np.arange(n)
    avg_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    avg_temp += np.random.normal(0, 3, n)

    return pd.DataFrame({
        'datetime': sample_dates,
        'average_temp': avg_temp.round(1),
        'max_temp': (avg_temp + np.random.uniform(3, 8, n)).round(1),
        'average_wind': np.random.uniform(5, 25, n).round(1),
        'total_precipitation': np.random.exponential(2, n).round(2),
    })


@pytest.fixture
def sample_calendar_df(sample_dates):
    """Generate sample calendar data"""
    df = pd.DataFrame({'datetime': sample_dates})
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['datetime'].dt.month
    df['Holiday'] = 0

    # Mark some holidays
    holidays = ['2023-01-01', '2023-12-25', '2023-07-04']
    for h in holidays:
        df.loc[df['datetime'] == h, 'Holiday'] = 1

    return df


@pytest.fixture
def sample_merged_df(sample_patient_df, sample_weather_df, sample_calendar_df):
    """Generate merged dataset"""
    merged = sample_patient_df.merge(
        sample_weather_df, on='datetime', how='inner'
    ).merge(
        sample_calendar_df, on='datetime', how='inner'
    )
    return merged


@pytest.fixture
def sample_processed_df(sample_merged_df):
    """Generate processed dataset with lag features"""
    df = sample_merged_df.copy()

    # Add lag features
    for i in range(1, 8):
        df[f'ED_{i}'] = df['Total_Arrivals'].shift(i)

    # Add target features
    for i in range(1, 8):
        df[f'Target_{i}'] = df['Total_Arrivals'].shift(-i)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


@pytest.fixture
def sample_train_test_split(sample_processed_df):
    """Generate train/cal/test split indices"""
    n = len(sample_processed_df)
    train_end = int(n * 0.7)
    cal_end = int(n * 0.85)

    return {
        'train_idx': np.arange(0, train_end),
        'cal_idx': np.arange(train_end, cal_end),
        'test_idx': np.arange(cal_end, n),
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing"""
    import sys

    # Create mock streamlit module
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.cache_data = lambda f: f
    mock_st.cache_resource = lambda f: f

    # Store original and replace
    original_st = sys.modules.get('streamlit')
    sys.modules['streamlit'] = mock_st

    yield mock_st

    # Restore original
    if original_st:
        sys.modules['streamlit'] = original_st


@pytest.fixture
def mock_supabase():
    """Mock Supabase client"""
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.execute.return_value.data = []
    mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()
    return mock_client


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_dataframe_equal(df1, df2, check_dtype=False):
    """Assert two DataFrames are equal"""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_series_equal(s1, s2, check_dtype=False):
    """Assert two Series are equal"""
    pd.testing.assert_series_equal(s1, s2, check_dtype=check_dtype)
