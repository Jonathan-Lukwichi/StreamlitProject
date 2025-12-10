# =============================================================================
# app_core/services/data_service.py
# Data Service - Business Logic for Data Operations
# =============================================================================

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .base_service import BaseService, ServiceResult
from app_core.errors import DataValidationError, DataFusionError


class DataService(BaseService):
    """
    Service for data operations.

    Handles:
    - Data validation
    - Data fusion
    - Feature generation
    - Temporal splitting

    Usage:
        service = DataService()

        # Validate uploaded data
        result = service.validate_data(df, required_columns=['datetime', 'value'])

        # Fuse multiple datasets
        result = service.fuse_datasets(patient_df, weather_df, calendar_df)
    """

    def __init__(self):
        super().__init__()

    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        date_column: Optional[str] = None,
    ) -> ServiceResult:
        """
        Validate a DataFrame for required columns and data quality.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            date_column: Name of datetime column to validate

        Returns:
            ServiceResult with validation details
        """
        def _validate():
            issues = []
            warnings = []

            # Check for empty DataFrame
            if df is None or df.empty:
                raise DataValidationError("DataFrame is empty or None")

            # Check required columns
            if required_columns:
                missing = [c for c in required_columns if c not in df.columns]
                if missing:
                    raise DataValidationError(
                        f"Missing required columns: {missing}",
                        column=str(missing),
                    )

            # Validate date column
            if date_column and date_column in df.columns:
                try:
                    pd.to_datetime(df[date_column])
                except Exception as e:
                    issues.append(f"Cannot parse date column '{date_column}': {e}")

            # Check for missing values
            missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
            high_missing = {k: v for k, v in missing_pct.items() if v > 10}
            if high_missing:
                warnings.append(f"High missing values: {high_missing}")

            # Check for duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                warnings.append(f"Found {dup_count} duplicate rows")

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
            }

        return self.safe_execute("Validating data", _validate)

    def detect_datetime_column(self, df: pd.DataFrame) -> ServiceResult:
        """
        Auto-detect the datetime column in a DataFrame.

        Returns:
            ServiceResult with detected column name
        """
        def _detect():
            priority_names = [
                'datetime', 'date', 'timestamp', 'Date', 'ds',
                'time', 'DATE', 'DATETIME'
            ]

            # Check priority names first
            for name in priority_names:
                if name in df.columns:
                    return {'column': name, 'method': 'name_match'}

            # Check column types
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return {'column': col, 'method': 'dtype_detection'}

            # Try parsing columns that look like dates
            for col in df.columns:
                if any(k in col.lower() for k in ['date', 'time', 'stamp']):
                    try:
                        pd.to_datetime(df[col].head(10))
                        return {'column': col, 'method': 'parse_attempt'}
                    except:
                        continue

            return {'column': None, 'method': 'not_found'}

        return self.safe_execute("Detecting datetime column", _detect)

    def fuse_datasets(
        self,
        patient_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        reason_df: Optional[pd.DataFrame] = None,
    ) -> ServiceResult:
        """
        Fuse multiple datasets on datetime keys.

        Args:
            patient_df: Patient arrivals data
            weather_df: Weather data
            calendar_df: Calendar/holiday data
            reason_df: Optional reason for visit data

        Returns:
            ServiceResult with merged DataFrame
        """
        def _fuse():
            self._update_progress(10, "Detecting datetime columns...")

            # Detect datetime columns in each dataset
            dfs = [
                ('patient', patient_df),
                ('weather', weather_df),
                ('calendar', calendar_df),
            ]
            if reason_df is not None:
                dfs.append(('reason', reason_df))

            date_cols = {}
            for name, df in dfs:
                result = self.detect_datetime_column(df)
                if result.success and result.data['column']:
                    date_cols[name] = result.data['column']
                else:
                    raise DataFusionError(
                        f"Cannot detect datetime column in {name} dataset",
                        datasets=[name],
                    )

            self._update_progress(30, "Normalizing datetime columns...")

            # Normalize datetime columns
            normalized = {}
            for name, df in dfs:
                df_copy = df.copy()
                date_col = date_cols[name]
                df_copy['_merge_date'] = pd.to_datetime(
                    df_copy[date_col], errors='coerce'
                )
                df_copy = df_copy.dropna(subset=['_merge_date'])
                normalized[name] = df_copy

            self._update_progress(50, "Merging datasets...")

            # Merge sequentially
            merged = normalized['patient']

            for name in ['weather', 'calendar', 'reason']:
                if name in normalized:
                    self._update_progress(
                        50 + 10 * list(normalized.keys()).index(name),
                        f"Merging {name}..."
                    )
                    merged = merged.merge(
                        normalized[name],
                        on='_merge_date',
                        how='outer',
                        suffixes=('', f'_{name}')
                    )

            # Clean up
            self._update_progress(90, "Cleaning merged data...")
            merged = merged.drop(columns=['_merge_date'], errors='ignore')

            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.duplicated()]

            # Sort by date
            date_col = date_cols['patient']
            if date_col in merged.columns:
                merged = merged.sort_values(date_col).reset_index(drop=True)

            self._update_progress(100, "Fusion complete")

            return merged

        return self.safe_execute("Fusing datasets", _fuse)

    def generate_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        n_lags: int = 7,
        n_horizons: int = 7,
    ) -> ServiceResult:
        """
        Generate lag features and future targets.

        Args:
            df: Input DataFrame
            target_column: Column to create lags for
            n_lags: Number of lag features (ED_1 to ED_n)
            n_horizons: Number of future targets (Target_1 to Target_n)

        Returns:
            ServiceResult with DataFrame containing lag features
        """
        def _generate():
            result = df.copy()

            self._update_progress(20, "Generating lag features...")

            # Lag features (past values)
            for i in range(1, n_lags + 1):
                result[f'ED_{i}'] = result[target_column].shift(i)

            self._update_progress(60, "Generating target columns...")

            # Future targets
            for i in range(1, n_horizons + 1):
                result[f'Target_{i}'] = result[target_column].shift(-i)

            self._update_progress(80, "Dropping edge rows...")

            # Drop rows with NaN from shifts
            result = result.dropna().reset_index(drop=True)

            return result

        return self.safe_execute("Generating lag features", _generate)

    def compute_temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str,
        train_ratio: float = 0.70,
        cal_ratio: float = 0.15,
    ) -> ServiceResult:
        """
        Compute temporal train/calibration/test split.

        Args:
            df: DataFrame to split
            date_column: Datetime column name
            train_ratio: Proportion for training
            cal_ratio: Proportion for calibration

        Returns:
            ServiceResult with split indices
        """
        def _split():
            # Ensure sorted by date
            df_sorted = df.sort_values(date_column).reset_index(drop=True)
            n = len(df_sorted)

            # Calculate split points
            train_end = int(n * train_ratio)
            cal_end = int(n * (train_ratio + cal_ratio))

            # Create indices
            train_idx = np.arange(0, train_end)
            cal_idx = np.arange(train_end, cal_end)
            test_idx = np.arange(cal_end, n)

            # Get date ranges
            dates = pd.to_datetime(df_sorted[date_column])

            return {
                'train_idx': train_idx,
                'cal_idx': cal_idx,
                'test_idx': test_idx,
                'train_dates': (dates.iloc[0], dates.iloc[train_end - 1]),
                'cal_dates': (dates.iloc[train_end], dates.iloc[cal_end - 1]),
                'test_dates': (dates.iloc[cal_end], dates.iloc[-1]),
                'sizes': {
                    'train': len(train_idx),
                    'cal': len(cal_idx),
                    'test': len(test_idx),
                },
            }

        return self.safe_execute("Computing temporal split", _split)
