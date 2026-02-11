# =============================================================================
# app_core/pipelines/temporal_split.py
# Temporal Data Splitting for Time Series Forecasting
# =============================================================================
"""
TEMPORAL SPLITTING MODULE
=========================

This module provides date-based splitting for time series data, ensuring:
1. Chronological order is preserved (no future data leaks into training)
2. Split boundaries are computed automatically from the dataset's date range
3. Works with ANY dataset size (5 years, 24 years, etc.)

WHY TEMPORAL SPLITTING?
-----------------------
In time series forecasting, random splitting is INVALID because:
- It mixes future data with past data during training
- The model "sees" patterns from the future, leading to overly optimistic results
- This is called "data leakage" and invalidates your research

Academic Reference:
    Bergmeir, C., & Benitez, J. M. (2012). "On the use of cross-validation
    for time series predictor evaluation." Information Sciences, 191, 192-213.

WHY 3 SPLITS (Train/Calibration/Test)?
--------------------------------------
- TRAIN: Model learns patterns from this data
- CALIBRATION: Used to calibrate prediction intervals (conformal prediction)
- TEST: Final evaluation - model never sees this until the end

The calibration set is required for Conformal Prediction to provide
mathematically guaranteed coverage for prediction intervals.

Academic Reference:
    Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning
    in a Random World." Springer.

USAGE EXAMPLE
-------------
```python
from app_core.pipelines import compute_temporal_split

# Load your data
df = pd.read_csv("hospital_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Compute splits (automatically detects date range)
split_result = compute_temporal_split(
    df=df,
    date_col='Date',
    train_ratio=0.70,   # 70% for training
    cal_ratio=0.15      # 15% for calibration, 15% for test
)

# Access the results
print(f"Train: {len(split_result.train_idx)} records")
print(f"Calibration: {len(split_result.cal_idx)} records")
print(f"Test: {len(split_result.test_idx)} records")

# Use the indices to split your data
X_train = df.iloc[split_result.train_idx]
X_cal = df.iloc[split_result.cal_idx]
X_test = df.iloc[split_result.test_idx]
```

Author: HealthForecast AI Research Team
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TemporalSplitResult:
    """
    Result of temporal splitting operation.

    This class holds all information about how the data was split,
    making it easy to reproduce the split or understand what happened.

    Attributes
    ----------
    train_idx : np.ndarray
        Array of indices for training data
    cal_idx : np.ndarray
        Array of indices for calibration data
    test_idx : np.ndarray
        Array of indices for test data
    train_end_date : datetime
        The date where training ends (exclusive)
    cal_end_date : datetime
        The date where calibration ends (exclusive)
    test_end_date : datetime
        The last date in the test set
    min_date : datetime
        First date in the dataset
    max_date : datetime
        Last date in the dataset
    total_records : int
        Total number of records in the dataset
    total_days : int
        Total number of days covered by the dataset
    train_ratio : float
        Proportion used for training
    cal_ratio : float
        Proportion used for calibration
    test_ratio : float
        Proportion used for testing
    date_col : str
        Name of the date column used for splitting
    """
    # Indices for each split
    train_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray

    # Date boundaries
    train_end_date: datetime
    cal_end_date: datetime
    test_end_date: datetime
    min_date: datetime
    max_date: datetime

    # Statistics
    total_records: int
    total_days: int
    train_ratio: float
    cal_ratio: float
    test_ratio: float

    # Date column name
    date_col: str = ""

    @property
    def train_records(self) -> int:
        """Number of records in training set."""
        return len(self.train_idx)

    @property
    def cal_records(self) -> int:
        """Number of records in calibration set."""
        return len(self.cal_idx)

    @property
    def test_records(self) -> int:
        """Number of records in test set."""
        return len(self.test_idx)

    # Alias properties for cleaner API
    @property
    def train_start(self) -> datetime:
        """Start date of training set."""
        return self.min_date

    @property
    def train_end(self) -> datetime:
        """End date of training set."""
        return self.train_end_date

    @property
    def cal_start(self) -> datetime:
        """Start date of calibration set."""
        return self.train_end_date

    @property
    def cal_end(self) -> datetime:
        """End date of calibration set."""
        return self.cal_end_date

    @property
    def test_start(self) -> datetime:
        """Start date of test set."""
        return self.cal_end_date

    @property
    def test_end(self) -> datetime:
        """End date of test set."""
        return self.test_end_date

    @property
    def actual_train_ratio(self) -> float:
        """Actual proportion of data in training set."""
        return self.train_records / self.total_records if self.total_records > 0 else 0

    @property
    def actual_cal_ratio(self) -> float:
        """Actual proportion of data in calibration set."""
        return self.cal_records / self.total_records if self.total_records > 0 else 0

    @property
    def actual_test_ratio(self) -> float:
        """Actual proportion of data in test set."""
        return self.test_records / self.total_records if self.total_records > 0 else 0

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary dictionary of the split.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all split information
        """
        return {
            "dataset": {
                "min_date": self.min_date.strftime("%Y-%m-%d"),
                "max_date": self.max_date.strftime("%Y-%m-%d"),
                "total_days": self.total_days,
                "total_records": self.total_records,
            },
            "split_dates": {
                "train_end": self.train_end_date.strftime("%Y-%m-%d"),
                "cal_end": self.cal_end_date.strftime("%Y-%m-%d"),
                "test_end": self.test_end_date.strftime("%Y-%m-%d"),
            },
            "requested_ratios": {
                "train": self.train_ratio,
                "calibration": self.cal_ratio,
                "test": self.test_ratio,
            },
            "actual_ratios": {
                "train": round(self.actual_train_ratio, 4),
                "calibration": round(self.actual_cal_ratio, 4),
                "test": round(self.actual_test_ratio, 4),
            },
            "record_counts": {
                "train": self.train_records,
                "calibration": self.cal_records,
                "test": self.test_records,
            },
        }

    def __repr__(self) -> str:
        return (
            f"TemporalSplitResult(\n"
            f"  Dataset: {self.min_date.strftime('%Y-%m-%d')} to {self.max_date.strftime('%Y-%m-%d')} "
            f"({self.total_days} days, {self.total_records} records)\n"
            f"  Train: {self.train_records} records ({self.actual_train_ratio:.1%}) "
            f"until {self.train_end_date.strftime('%Y-%m-%d')}\n"
            f"  Calibration: {self.cal_records} records ({self.actual_cal_ratio:.1%}) "
            f"until {self.cal_end_date.strftime('%Y-%m-%d')}\n"
            f"  Test: {self.test_records} records ({self.actual_test_ratio:.1%}) "
            f"until {self.test_end_date.strftime('%Y-%m-%d')}\n"
            f")"
        )


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_temporal_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15,
    verbose: bool = False
) -> TemporalSplitResult:
    """
    Compute temporal train/calibration/test split based on dates.

    This function automatically computes split boundaries from the dataset's
    own date range, making it work for ANY dataset size.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing time series data
    date_col : str
        Name of the column containing dates
    train_ratio : float, default=0.70
        Proportion of data for training (0.0 to 1.0)
        Default: 70% for training
    cal_ratio : float, default=0.15
        Proportion of data for calibration (0.0 to 1.0)
        Default: 15% for calibration
        Note: Test ratio = 1.0 - train_ratio - cal_ratio
    verbose : bool, default=False
        If True, print detailed information about the split

    Returns
    -------
    TemporalSplitResult
        Object containing all split information including indices

    Raises
    ------
    ValueError
        If date column doesn't exist or ratios are invalid

    Examples
    --------
    >>> # Basic usage
    >>> result = compute_temporal_split(df, 'Date')
    >>> print(result)

    >>> # Custom ratios: 80% train, 10% cal, 10% test
    >>> result = compute_temporal_split(df, 'Date', train_ratio=0.80, cal_ratio=0.10)

    >>> # Get indices for splitting
    >>> train_data = df.iloc[result.train_idx]
    >>> cal_data = df.iloc[result.cal_idx]
    >>> test_data = df.iloc[result.test_idx]

    Academic Justification
    ----------------------
    Date-based splitting ensures:
    1. No future information leaks into training (Tashman, 2000)
    2. Respects the temporal nature of the data (Bergmeir & Benitez, 2012)
    3. Generalizes to any dataset size (computed from data, not hardcoded)
    """
    # ==========================================================================
    # STEP 1: VALIDATE INPUTS
    # ==========================================================================

    # Check if date column exists
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate ratios
    test_ratio = 1.0 - train_ratio - cal_ratio

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if cal_ratio <= 0 or cal_ratio >= 1:
        raise ValueError(f"cal_ratio must be between 0 and 1, got {cal_ratio}")

    if test_ratio <= 0:
        raise ValueError(
            f"test_ratio (1 - train_ratio - cal_ratio) must be positive. "
            f"Got train_ratio={train_ratio}, cal_ratio={cal_ratio}, "
            f"leaving test_ratio={test_ratio}"
        )

    if abs(train_ratio + cal_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0. Got {train_ratio + cal_ratio + test_ratio}"
        )

    # ==========================================================================
    # STEP 2: PREPARE DATA
    # ==========================================================================

    # Make a copy to avoid modifying original
    df_sorted = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]):
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    # Sort by date (CRITICAL for temporal ordering)
    df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

    # ==========================================================================
    # STEP 3: COMPUTE DATE BOUNDARIES
    # ==========================================================================

    # Get date range from the data itself (NOT hardcoded!)
    min_date = df_sorted[date_col].min()
    max_date = df_sorted[date_col].max()
    total_days = (max_date - min_date).days
    total_records = len(df_sorted)

    if verbose:
        print(f"Dataset date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"Total days: {total_days}, Total records: {total_records}")

    # Compute split dates based on PROPORTIONS of the date range
    # This ensures the same proportions work for ANY dataset
    train_days = int(total_days * train_ratio)
    cal_days = int(total_days * (train_ratio + cal_ratio))

    train_end_date = min_date + timedelta(days=train_days)
    cal_end_date = min_date + timedelta(days=cal_days)
    test_end_date = max_date

    if verbose:
        print(f"\nComputed split dates:")
        print(f"  Train ends:       {train_end_date.strftime('%Y-%m-%d')} (day {train_days})")
        print(f"  Calibration ends: {cal_end_date.strftime('%Y-%m-%d')} (day {cal_days})")
        print(f"  Test ends:        {test_end_date.strftime('%Y-%m-%d')} (day {total_days})")

    # ==========================================================================
    # STEP 4: CREATE INDEX MASKS
    # ==========================================================================

    # Create boolean masks based on dates
    train_mask = df_sorted[date_col] < train_end_date
    cal_mask = (df_sorted[date_col] >= train_end_date) & (df_sorted[date_col] < cal_end_date)
    test_mask = df_sorted[date_col] >= cal_end_date

    # Convert to indices
    train_idx = df_sorted[train_mask].index.values
    cal_idx = df_sorted[cal_mask].index.values
    test_idx = df_sorted[test_mask].index.values

    if verbose:
        print(f"\nRecord counts:")
        print(f"  Train:       {len(train_idx)} records ({len(train_idx)/total_records:.1%})")
        print(f"  Calibration: {len(cal_idx)} records ({len(cal_idx)/total_records:.1%})")
        print(f"  Test:        {len(test_idx)} records ({len(test_idx)/total_records:.1%})")

    # ==========================================================================
    # STEP 5: BUILD RESULT OBJECT
    # ==========================================================================

    result = TemporalSplitResult(
        train_idx=train_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        train_end_date=train_end_date,
        cal_end_date=cal_end_date,
        test_end_date=test_end_date,
        min_date=min_date,
        max_date=max_date,
        total_records=total_records,
        total_days=total_days,
        train_ratio=train_ratio,
        cal_ratio=cal_ratio,
        test_ratio=test_ratio,
        date_col=date_col,
    )

    return result


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_temporal_split(
    df: pd.DataFrame,
    date_col: str,
    split_result: TemporalSplitResult
) -> Dict[str, Any]:
    """
    Validate that a temporal split was done correctly.

    This function checks for common errors in temporal splitting,
    useful for academic research to verify methodology.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe
    date_col : str
        Name of the date column
    split_result : TemporalSplitResult
        The result from compute_temporal_split()

    Returns
    -------
    Dict[str, Any]
        Validation results with 'valid' boolean and any issues found

    Examples
    --------
    >>> result = compute_temporal_split(df, 'Date')
    >>> validation = validate_temporal_split(df, 'Date', result)
    >>> if validation['valid']:
    ...     print("Split is valid!")
    >>> else:
    ...     print(f"Issues found: {validation['issues']}")
    """
    issues = []
    warnings = []

    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # ==========================================================================
    # CHECK 1: No overlap between splits
    # ==========================================================================

    train_set = set(split_result.train_idx)
    cal_set = set(split_result.cal_idx)
    test_set = set(split_result.test_idx)

    train_cal_overlap = train_set & cal_set
    train_test_overlap = train_set & test_set
    cal_test_overlap = cal_set & test_set

    if train_cal_overlap:
        issues.append(f"CRITICAL: {len(train_cal_overlap)} records appear in both train and calibration sets")

    if train_test_overlap:
        issues.append(f"CRITICAL: {len(train_test_overlap)} records appear in both train and test sets")

    if cal_test_overlap:
        issues.append(f"CRITICAL: {len(cal_test_overlap)} records appear in both calibration and test sets")

    # ==========================================================================
    # CHECK 2: All records are assigned
    # ==========================================================================

    total_assigned = len(train_set) + len(cal_set) + len(test_set)
    if total_assigned != split_result.total_records:
        issues.append(
            f"CRITICAL: Only {total_assigned} of {split_result.total_records} records assigned to splits"
        )

    # ==========================================================================
    # CHECK 3: Temporal ordering (no future data in training)
    # ==========================================================================

    if len(split_result.train_idx) > 0 and len(split_result.test_idx) > 0:
        max_train_date = df.iloc[split_result.train_idx][date_col].max()
        min_test_date = df.iloc[split_result.test_idx][date_col].min()

        if max_train_date >= min_test_date:
            issues.append(
                f"CRITICAL: Training data ({max_train_date.strftime('%Y-%m-%d')}) "
                f"overlaps with test data ({min_test_date.strftime('%Y-%m-%d')}). "
                f"This is DATA LEAKAGE!"
            )

    if len(split_result.cal_idx) > 0 and len(split_result.test_idx) > 0:
        max_cal_date = df.iloc[split_result.cal_idx][date_col].max()
        min_test_date = df.iloc[split_result.test_idx][date_col].min()

        if max_cal_date >= min_test_date:
            issues.append(
                f"CRITICAL: Calibration data ({max_cal_date.strftime('%Y-%m-%d')}) "
                f"overlaps with test data ({min_test_date.strftime('%Y-%m-%d')}). "
                f"This is DATA LEAKAGE!"
            )

    # ==========================================================================
    # CHECK 4: Reasonable split sizes
    # ==========================================================================

    if split_result.train_records < 30:
        warnings.append(
            f"WARNING: Training set has only {split_result.train_records} records. "
            f"Consider using more data for training."
        )

    if split_result.cal_records < 10:
        warnings.append(
            f"WARNING: Calibration set has only {split_result.cal_records} records. "
            f"Conformal prediction may be unreliable with small calibration sets."
        )

    if split_result.test_records < 10:
        warnings.append(
            f"WARNING: Test set has only {split_result.test_records} records. "
            f"Evaluation metrics may be unreliable."
        )

    # ==========================================================================
    # BUILD RESULT
    # ==========================================================================

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checks_passed": {
            "no_overlap": len(train_cal_overlap) == 0 and len(train_test_overlap) == 0 and len(cal_test_overlap) == 0,
            "all_assigned": total_assigned == split_result.total_records,
            "temporal_order": len([i for i in issues if "LEAKAGE" in i]) == 0,
        },
        "split_summary": split_result.summary(),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def split_dataframe(
    df: pd.DataFrame,
    split_result: TemporalSplitResult
) -> tuple:
    """
    Split a dataframe using the result from compute_temporal_split.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split
    split_result : TemporalSplitResult
        Result from compute_temporal_split()

    Returns
    -------
    tuple
        (train_df, cal_df, test_df) - Three dataframes

    Examples
    --------
    >>> split_result = compute_temporal_split(df, 'Date')
    >>> train_df, cal_df, test_df = split_dataframe(df, split_result)
    """
    train_df = df.iloc[split_result.train_idx].copy()
    cal_df = df.iloc[split_result.cal_idx].copy()
    test_df = df.iloc[split_result.test_idx].copy()

    return train_df, cal_df, test_df


# =============================================================================
# TIME SERIES CROSS-VALIDATION
# =============================================================================
"""
TIME SERIES CROSS-VALIDATION
=============================

Unlike standard k-fold cross-validation which randomly shuffles data,
time series CV respects temporal ordering to prevent data leakage.

Academic Reference:
    Bergmeir, C., & Benitez, J. M. (2012). "On the use of cross-validation
    for time series predictor evaluation." Information Sciences, 191, 192-213.

EXPANDING WINDOW CV (used here):
--------------------------------
Fold 1: [Train: ████░░░░░░░░] [Val: ██░░░░░░░░]
Fold 2: [Train: ██████░░░░░░] [Val: ██░░░░░░░░]
Fold 3: [Train: ████████░░░░] [Val: ██░░░░░░░░]
Fold 4: [Train: ██████████░░] [Val: ██░░░░░░░░]
Fold 5: [Train: ████████████] [Val: ██░░░░░░░░]

Each fold uses all previous data for training, ensuring no future leakage.
"""

try:
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class CVFoldResult:
    """
    Result for a single cross-validation fold.

    Attributes
    ----------
    fold_idx : int
        Index of the fold (0-based)
    train_idx : np.ndarray
        Array of indices for training data in this fold
    val_idx : np.ndarray
        Array of indices for validation data in this fold
    train_start : datetime
        Start date of training data
    train_end : datetime
        End date of training data
    val_start : datetime
        Start date of validation data
    val_end : datetime
        End date of validation data
    """
    fold_idx: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    @property
    def n_train(self) -> int:
        """Number of training samples in this fold."""
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        """Number of validation samples in this fold."""
        return len(self.val_idx)

    def __repr__(self) -> str:
        return (
            f"CVFold {self.fold_idx + 1}: "
            f"Train[{self.train_start.strftime('%Y-%m-%d')} to {self.train_end.strftime('%Y-%m-%d')}] "
            f"({self.n_train} samples) → "
            f"Val[{self.val_start.strftime('%Y-%m-%d')} to {self.val_end.strftime('%Y-%m-%d')}] "
            f"({self.n_val} samples)"
        )


@dataclass
class CVConfig:
    """
    Configuration for time series cross-validation.

    Attributes
    ----------
    n_splits : int
        Number of CV folds (default: 5)
    gap : int
        Number of time periods between train and validation sets
        to prevent leakage from autocorrelation (default: 0)
    min_train_size : int, optional
        Minimum number of samples required in training set
    test_size : int, optional
        Fixed size for each validation fold (if None, uses equal splits)
    """
    n_splits: int = 5
    gap: int = 0
    min_train_size: Optional[int] = None
    test_size: Optional[int] = None


def create_cv_folds(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int = 5,
    gap: int = 0,
    min_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    verbose: bool = False
) -> List[CVFoldResult]:
    """
    Create time series cross-validation folds using expanding window.

    Uses sklearn's TimeSeriesSplit to ensure temporal ordering is preserved.
    Training sets grow with each fold (expanding window), validation sets
    remain approximately equal in size.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing time series data
    date_col : str
        Name of the column containing dates
    n_splits : int, default=5
        Number of folds to create
    gap : int, default=0
        Number of samples to exclude between train and validation
        to account for autocorrelation in time series
    min_train_size : int, optional
        Minimum number of samples required in training set
        If None, defaults to n_splits (ensuring first fold has data)
    test_size : int, optional
        Fixed number of samples for each validation fold
        If None, splits remaining data equally
    verbose : bool, default=False
        If True, print detailed information about each fold

    Returns
    -------
    List[CVFoldResult]
        List of CVFoldResult objects, one per fold

    Raises
    ------
    ValueError
        If date column doesn't exist or insufficient data for CV
    ImportError
        If sklearn is not installed

    Examples
    --------
    >>> # Basic usage with 5-fold CV
    >>> folds = create_cv_folds(df, 'Date', n_splits=5)
    >>> for fold in folds:
    ...     print(f"Fold {fold.fold_idx}: {fold.n_train} train, {fold.n_val} val")

    >>> # With gap to prevent autocorrelation leakage
    >>> folds = create_cv_folds(df, 'Date', n_splits=5, gap=7)

    >>> # Use folds for training
    >>> for fold in folds:
    ...     X_train = df.iloc[fold.train_idx][features]
    ...     y_train = df.iloc[fold.train_idx][target]
    ...     X_val = df.iloc[fold.val_idx][features]
    ...     y_val = df.iloc[fold.val_idx][target]
    ...     # Train and evaluate model...

    Academic Justification
    ----------------------
    Time series CV prevents data leakage by ensuring:
    1. All training data comes BEFORE validation data
    2. Gap parameter accounts for autocorrelation
    3. Expanding window mimics real-world forecasting scenario
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn is required for time series cross-validation. "
            "Install it with: pip install scikit-learn"
        )

    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    # Prepare data - sort by date
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]):
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

    n_samples = len(df_sorted)

    # Calculate minimum train size if not specified
    if min_train_size is None:
        # Ensure at least n_splits samples in first training fold
        min_train_size = max(n_splits, int(n_samples * 0.1))

    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=None,  # Use expanding window
        test_size=test_size
    )

    folds = []

    for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(df_sorted)):
        # Skip if training set is too small
        if len(train_indices) < min_train_size:
            if verbose:
                print(f"Skipping fold {fold_idx + 1}: insufficient training data "
                      f"({len(train_indices)} < {min_train_size})")
            continue

        # Get date boundaries
        train_dates = df_sorted.iloc[train_indices][date_col]
        val_dates = df_sorted.iloc[val_indices][date_col]

        fold_result = CVFoldResult(
            fold_idx=len(folds),  # Re-number after potential skips
            train_idx=train_indices,
            val_idx=val_indices,
            train_start=train_dates.min(),
            train_end=train_dates.max(),
            val_start=val_dates.min(),
            val_end=val_dates.max(),
        )

        folds.append(fold_result)

        if verbose:
            print(fold_result)

    if len(folds) == 0:
        raise ValueError(
            f"Could not create any CV folds with the given parameters. "
            f"Dataset has {n_samples} samples, min_train_size={min_train_size}. "
            f"Try reducing n_splits or min_train_size."
        )

    return folds


def visualize_cv_folds(
    df: pd.DataFrame,
    date_col: str,
    folds: List[CVFoldResult],
    title: str = "Time Series Cross-Validation Folds"
) -> "go.Figure":
    """
    Create a premium futuristic Plotly visualization of CV fold splits.

    Features a fluorescent neon design with glowing effects and
    professional sci-fi styling matching the HealthForecast AI theme.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe
    date_col : str
        Name of the date column
    folds : List[CVFoldResult]
        List of CV folds from create_cv_folds()
    title : str
        Title for the chart

    Returns
    -------
    go.Figure
        Plotly figure object showing the CV fold structure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for visualization")

    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    total_days = (max_date - min_date).days

    fig = go.Figure()

    # =========================================================================
    # PREMIUM FLUORESCENT COLOR SCHEME
    # =========================================================================
    # Neon cyan for training (matches HealthForecast AI theme)
    train_fill = "rgba(34, 211, 238, 0.75)"      # Cyan fluorescent
    train_line = "rgba(34, 211, 238, 1)"         # Cyan glow border
    train_glow = "rgba(34, 211, 238, 0.3)"       # Outer glow

    # Neon magenta/pink for validation
    val_fill = "rgba(236, 72, 153, 0.75)"        # Magenta fluorescent
    val_line = "rgba(236, 72, 153, 1)"           # Magenta glow border
    val_glow = "rgba(236, 72, 153, 0.3)"         # Outer glow

    # Background grid lines
    grid_color = "rgba(34, 211, 238, 0.1)"
    axis_line_color = "rgba(34, 211, 238, 0.3)"

    for fold in folds:
        fold_label = f"FOLD {fold.fold_idx + 1}"
        train_days = (fold.train_end - fold.train_start).days
        val_days = (fold.val_end - fold.val_start).days
        train_base = (fold.train_start - min_date).days
        val_base = (fold.val_start - min_date).days

        # -----------------------------------------------------------------
        # TRAINING BAR - Neon Cyan with Glow Effect
        # -----------------------------------------------------------------
        # Outer glow layer (slightly larger, transparent)
        fig.add_trace(go.Bar(
            x=[train_days + 4],
            y=[fold_label],
            orientation='h',
            base=[train_base - 2],
            marker=dict(
                color=train_glow,
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Main training bar with gradient effect
        fig.add_trace(go.Bar(
            x=[train_days],
            y=[fold_label],
            orientation='h',
            base=[train_base],
            name="◼ Train" if fold.fold_idx == 0 else None,
            marker=dict(
                color=train_fill,
                line=dict(color=train_line, width=2),
                pattern=dict(shape="", fillmode="replace")
            ),
            text=f"<b>{fold.n_train:,}</b>",
            textposition="inside",
            textfont=dict(
                color="#0f172a",  # Dark text for better contrast on cyan
                size=13,
                family="Segoe UI, sans-serif"
            ),
            showlegend=(fold.fold_idx == 0),
            legendgroup="train",
            hovertemplate=(
                f"<b style='color:#22d3ee'>▶ TRAINING SET</b><br>"
                f"<b>Fold {fold.fold_idx + 1}</b><br><br>"
                f"📅 Period: {fold.train_start.strftime('%b %d, %Y')} → {fold.train_end.strftime('%b %d, %Y')}<br>"
                f"📊 Samples: <b>{fold.n_train:,}</b><br>"
                f"⏱️ Duration: {train_days} days"
                f"<extra></extra>"
            )
        ))

        # -----------------------------------------------------------------
        # VALIDATION BAR - Neon Magenta with Glow Effect
        # -----------------------------------------------------------------
        # Outer glow layer
        fig.add_trace(go.Bar(
            x=[val_days + 4],
            y=[fold_label],
            orientation='h',
            base=[val_base - 2],
            marker=dict(
                color=val_glow,
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Main validation bar
        fig.add_trace(go.Bar(
            x=[val_days],
            y=[fold_label],
            orientation='h',
            base=[val_base],
            name="◼ Validation" if fold.fold_idx == 0 else None,
            marker=dict(
                color=val_fill,
                line=dict(color=val_line, width=2)
            ),
            text=f"<b>{fold.n_val:,}</b>",
            textposition="inside",
            textfont=dict(
                color="#ffffff",  # White text on magenta
                size=13,
                family="Segoe UI, sans-serif"
            ),
            showlegend=(fold.fold_idx == 0),
            legendgroup="val",
            hovertemplate=(
                f"<b style='color:#ec4899'>▶ VALIDATION SET</b><br>"
                f"<b>Fold {fold.fold_idx + 1}</b><br><br>"
                f"📅 Period: {fold.val_start.strftime('%b %d, %Y')} → {fold.val_end.strftime('%b %d, %Y')}<br>"
                f"📊 Samples: <b>{fold.n_val:,}</b><br>"
                f"⏱️ Duration: {val_days} days"
                f"<extra></extra>"
            )
        ))

    # =========================================================================
    # PREMIUM LAYOUT STYLING
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=16,
                color="#22d3ee",
                family="Segoe UI, sans-serif"
            ),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top"
        ),
        barmode='overlay',

        # X-Axis - Futuristic styling (no title to avoid overlap)
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            linewidth=1,
            tickfont=dict(color="#94a3b8", size=10),
            ticksuffix=" days",
            range=[-10, total_days + 50]
        ),

        # Y-Axis - Fold labels
        yaxis=dict(
            title="",
            categoryorder='array',
            categoryarray=[f"FOLD {i + 1}" for i in range(len(folds) - 1, -1, -1)],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            linewidth=1,
            tickfont=dict(
                color="#22d3ee",
                size=13,
                family="Segoe UI, sans-serif"
            )
        ),

        # Legend - Top right corner, compact
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(color="#e2e8f0", size=12),
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="rgba(34, 211, 238, 0.4)",
            borderwidth=1,
            itemsizing="constant"
        ),

        # Chart dimensions - taller bars
        height=max(380, 75 * len(folds) + 80),
        margin=dict(l=90, r=50, t=50, b=40),

        # Dark futuristic background
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",

        # Hover styling
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(34, 211, 238, 0.5)",
            font=dict(color="#e2e8f0", size=12)
        ),

        # Uniform bar gap for thicker bars
        bargap=0.3
    )

    # =========================================================================
    # ANNOTATIONS - Data Range (clean, minimal)
    # =========================================================================
    # Data range annotation (bottom right, below the chart)
    fig.add_annotation(
        x=1.0, y=-0.08,
        xref="paper", yref="paper",
        text=f"{min_date.strftime('%b %Y')} → {max_date.strftime('%b %Y')}",
        showarrow=False,
        font=dict(size=9, color="#64748b"),
        xanchor="right"
    )

    return fig


def evaluate_cv_metrics(
    metrics_per_fold: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across CV folds to compute mean ± std.

    Parameters
    ----------
    metrics_per_fold : List[Dict[str, float]]
        List of metric dictionaries, one per fold.
        Each dict should have keys like 'MAE', 'RMSE', 'MAPE', etc.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with metric names as keys and {'mean': X, 'std': Y} as values.

    Examples
    --------
    >>> fold_metrics = [
    ...     {'MAE': 10.5, 'RMSE': 15.2},
    ...     {'MAE': 11.2, 'RMSE': 16.1},
    ...     {'MAE': 10.8, 'RMSE': 15.5},
    ... ]
    >>> summary = evaluate_cv_metrics(fold_metrics)
    >>> print(f"MAE: {summary['MAE']['mean']:.2f} ± {summary['MAE']['std']:.2f}")
    MAE: 10.83 ± 0.35
    """
    if not metrics_per_fold:
        return {}

    # Get all metric names
    metric_names = set()
    for fold in metrics_per_fold:
        metric_names.update(fold.keys())

    result = {}
    for name in metric_names:
        values = [fold.get(name) for fold in metrics_per_fold if fold.get(name) is not None]
        if values:
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]  # Remove NaN/inf
            if len(arr) > 0:
                result[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'n_folds': len(arr)
                }

    return result


def format_cv_metric(metric_summary: Dict[str, float], precision: int = 2) -> str:
    """
    Format a CV metric summary as "mean ± std" string.

    Parameters
    ----------
    metric_summary : Dict[str, float]
        Dictionary with 'mean' and 'std' keys
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string like "12.34 ± 1.23"
    """
    mean = metric_summary.get('mean', 0)
    std = metric_summary.get('std', 0)
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see a demonstration of temporal splitting.

    Usage:
        python -m app_core.pipelines.temporal_split
    """
    import numpy as np

    print("=" * 70)
    print(" TEMPORAL SPLITTING DEMONSTRATION")
    print("=" * 70)

    # Create sample data (simulating hospital data from 2018-2023)
    np.random.seed(42)

    date_range = pd.date_range(start="2018-01-08", end="2023-01-25", freq="D")
    n_days = len(date_range)

    sample_df = pd.DataFrame({
        "Date": date_range,
        "Patients": np.random.poisson(50, n_days) + np.sin(np.arange(n_days) * 2 * np.pi / 365) * 10,
        "Temperature": np.random.normal(15, 10, n_days),
    })

    print(f"\nSample Dataset:")
    print(f"  Date range: {sample_df['Date'].min().strftime('%Y-%m-%d')} to {sample_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total records: {len(sample_df)}")

    # Compute temporal split with verbose output
    print("\n" + "-" * 70)
    print(" Computing Temporal Split (70% train, 15% calibration, 15% test)")
    print("-" * 70)

    result = compute_temporal_split(
        df=sample_df,
        date_col="Date",
        train_ratio=0.70,
        cal_ratio=0.15,
        verbose=True
    )

    # Print the result object
    print("\n" + "-" * 70)
    print(" Split Result Object")
    print("-" * 70)
    print(result)

    # Validate the split
    print("\n" + "-" * 70)
    print(" Validation")
    print("-" * 70)

    validation = validate_temporal_split(sample_df, "Date", result)

    if validation["valid"]:
        print("✅ Split is VALID - no data leakage detected!")
    else:
        print("❌ Split has ISSUES:")
        for issue in validation["issues"]:
            print(f"   - {issue}")

    if validation["warnings"]:
        print("\n⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")

    print("\n" + "=" * 70)
    print(" DEMONSTRATION COMPLETE")
    print("=" * 70)
