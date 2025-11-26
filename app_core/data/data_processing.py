
"""
data_processing.py — Cleaning & Feature Engineering for ED Forecasting
----------------------------------------------------------------------
Drop-in module for the Streamlit scaffold. It:
- Standardizes column names
- Parses and sorts by date
- Infers/constructs the base ED column (today's arrivals)
- Generates ED lag features (ED_1..ED_7) and target features (Target_1..Target_7)
- Cleans and imputes missing values with sensible defaults
- Adds helpful calendar features (day_of_week, is_weekend, etc.)
- Returns both the processed dataframe and a processing report

Usage (inside Streamlit):
-------------------------
from app_core.data.data_processing import process_dataset

df_proc, report = process_dataset(raw_df, n_lags=7)
st.session_state["processed_df"] = df_proc
st.json(report)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple, Dict, Any, List

# -------------------------------
# Helpers
# -------------------------------

def _clean_col_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[/\\]+", "_", s)        # slashes to underscore
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col_name(c) for c in df.columns]
    return df

def _find_date_like(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Date", "date", "DATE", "ds", "timestamp", "Datetime", "datetime", "Timestamp"]
    existing = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in existing:
            return existing[k.lower()]
    # fallback: look for columns containing "date" or "time"
    for c in df.columns:
        if re.search(r"date|time|timestamp", c, flags=re.IGNORECASE):
            return c
    return None

def _ensure_datetime(df: pd.DataFrame, date_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    df = df.copy()
    if date_col is None:
        # Nothing to parse — just return
        return df, None
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Drop rows with unparseable dates
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    return df, date_col

def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # try numeric conversion
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _guess_ed_column(df: pd.DataFrame) -> Optional[str]:
    # Priority 1: exact 'ED' or similar
    for cand in ["ED", "ed", "Ed", "Arrivals", "arrivals", "patients", "patient_count", "ED_Count"]:
        if cand in df.columns:
            return cand
        # also consider cleaned version
        cleaned = _clean_col_name(cand)
        if cleaned in df.columns:
            return cleaned
    # Priority 2: reconstruct from Target_1 (ED_t = Target_1 at t-1)
    if "Target_1" in df.columns:
        return None  # We'll create ED later from Target_1 if needed
    # Priority 3: reconstruct from ED_1 (ED_t = ED_1 at t+1)
    if "ED_1" in df.columns:
        return None
    return None

def _construct_ed_from_targets_or_lags(df: pd.DataFrame) -> pd.Series | None:
    s = None
    if "Target_1" in df.columns:
        s = df["Target_1"].shift(1)
    elif "ED_1" in df.columns:
        s = df["ED_1"].shift(-1)
    return s

def _clip_negatives(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.mask(series < 0, np.nan)
    return series

def _add_calendar_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col is None:
        return df
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek  # Monday=0,...Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    return df

def _detect_reason_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically detect ALL reason for visit columns in the dataset.

    This function identifies reason columns by excluding known non-reason columns
    (date, weather, calendar, lag/target features) and keeping all numeric columns
    that appear to be medical reasons for visit.

    Supports both:
    - Aggregated categories (Respiratory_Cases, Cardiac_Cases, etc.)
    - Granular columns (Asthma, Pneumonia, Fracture, Chest_Pain, etc.)
    """
    # Define columns that are NOT reason columns
    exclude_patterns = [
        "date", "datetime", "timestamp", "ds", "time",  # Date columns
        "target_", "ed_",  # Lag/target features
        "_lag_",  # Reason lag features (Asthma_lag_1, etc.)
        "day_of_week", "is_weekend", "day_of_month", "week_of_year", "month", "year",  # Calendar features
        "holiday", "moon_phase",  # Calendar features
        "temp", "wind", "mslp", "pressure", "precip", "rain", "snow", "humidity", "dew"  # Weather features
    ]

    reason_cols = []
    for col in df.columns:
        col_lower = col.lower()

        # Skip if matches exclude patterns
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            reason_cols.append(col)

    return reason_cols


# Clinical category mapping for aggregation
CLINICAL_CATEGORIES = {
    "RESPIRATORY": ["asthma", "pneumonia", "shortness_of_breath"],
    "CARDIAC": ["chest_pain", "arrhythmia", "hypertensive_emergency"],
    "TRAUMA": ["fracture", "laceration", "burn", "fall_injury"],
    "GASTROINTESTINAL": ["abdominal_pain", "vomiting", "diarrhea"],
    "INFECTIOUS": ["flu_symptoms", "fever", "viral_infection"],
    "NEUROLOGICAL": ["headache", "dizziness"],
    "OTHER": ["allergic_reaction", "mental_health"],
}


def _create_clinical_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate granular medical reasons into 6 clinical categories.

    This improves forecasting accuracy by:
    - Increasing daily counts (more signal, less noise)
    - Reducing sparsity (fewer zeros)
    - Making ARIMA/SARIMAX more effective

    Categories:
        RESPIRATORY = asthma + pneumonia + shortness_of_breath
        CARDIAC = chest_pain + arrhythmia + hypertensive_emergency
        TRAUMA = fracture + laceration + burn + fall_injury
        GASTROINTESTINAL = abdominal_pain + vomiting + diarrhea
        INFECTIOUS = flu_symptoms + fever + viral_infection
        NEUROLOGICAL = headache + dizziness
        OTHER = allergic_reaction + mental_health

    Returns:
        Tuple of (df with new category columns, list of created category column names)
    """
    df = df.copy()
    created_categories = []

    # Build column name mapping (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}

    for category, reasons in CLINICAL_CATEGORIES.items():
        # Find matching columns for this category
        matching_cols = []
        for reason in reasons:
            # Try exact match first
            if reason in col_map:
                matching_cols.append(col_map[reason])
            # Try with underscores replaced
            elif reason.replace("_", "") in col_map:
                matching_cols.append(col_map[reason.replace("_", "")])

        # Create aggregated column if we found any matching columns
        if matching_cols:
            # Sum the matching columns, treating NaN as 0
            df[category] = df[matching_cols].fillna(0).sum(axis=1)
            created_categories.append(category)

    return df, created_categories


def _generate_lags_and_targets(df: pd.DataFrame, ed_col: str, n_lags: int) -> pd.DataFrame:
    """Generate lag features (ED_1..ED_n) and future targets (Target_1..Target_n) from ED column."""
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"ED_{i}"] = df[ed_col].shift(i)
        df[f"Target_{i}"] = df[ed_col].shift(-i)
    return df

def _generate_multi_lags(df: pd.DataFrame, lag_cols: List[str], n_lags: int) -> pd.DataFrame:
    """Generate lag features for multiple reason columns (e.g., Asthma_lag_1..Asthma_lag_n).

    This creates historical features for each reason column, allowing models to learn
    temporal patterns and autocorrelations in specific medical conditions.
    """
    df = df.copy()
    for col in lag_cols:
        if col in df.columns:
            for i in range(1, n_lags + 1):
                df[f"{col}_lag_{i}"] = df[col].shift(i)
    return df

def _generate_multi_targets(df: pd.DataFrame, target_cols: List[str], n_lags: int) -> pd.DataFrame:
    """Generate future targets for multiple reason columns (e.g., Respiratory_Cases_1..Respiratory_Cases_n)."""
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            for i in range(1, n_lags + 1):
                df[f"{col}_{i}"] = df[col].shift(-i)
    return df

def _impute_numerics(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    df = df.copy()
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # Two-pass imputation: linear time interpolation then back/forward fill
    if len(num_cols) > 0:
        df[num_cols] = df.sort_index()[num_cols].interpolate(method="linear", limit_direction="both")
        df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")
    return df

# -------------------------------
# Public API
# -------------------------------

def _reorder_columns_for_training(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Reorder columns for optimal ML training workflow:
    1. Date columns (datetime, Date)
    2. Calendar columns (day_of_week, holiday, is_weekend, month, year, etc.)
    3. Weather columns (Average_Temp, Max_temp, Average_wind, etc.)
    4. Past Features / Lag columns (ED_1, ED_2, asthma_lag_1, pneumonia_lag_1, etc.)
    5. Future Targets (Target_1, Target_2, asthma_1, pneumonia_1, etc.)

    This structure helps ML models understand: Calendar → Weather → Past → Future
    """
    df = df.copy()
    all_cols = list(df.columns)

    # Define known column categories
    date_keywords = ['date', 'datetime', 'ds', 'timestamp']
    calendar_keywords = ['day_of_week', 'holiday', 'holiday_prev', 'is_weekend',
                        'day_of_month', 'week_of_year', 'month', 'year']
    weather_keywords = ['average_temp', 'max_temp', 'average_wind', 'max_wind',
                       'average_mslp', 'total_precipitation', 'moon_phase']

    # Known reason column names (granular medical categories)
    reason_keywords = [
        'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
        'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
        'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
        'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
    ]

    # Aggregated clinical category names
    clinical_category_keywords = [
        'respiratory', 'cardiac', 'trauma', 'gastrointestinal',
        'infectious', 'neurological', 'other'
    ]

    # Categorize columns
    date_cols = []
    calendar_cols = []
    weather_cols = []
    lag_cols = []
    target_cols = []
    other_cols = []

    for col in all_cols:
        col_lower = col.lower()

        # 1. Date columns
        if col_lower in date_keywords:
            date_cols.append(col)

        # 2. Calendar columns
        elif col_lower in calendar_keywords:
            calendar_cols.append(col)

        # 3. Weather columns
        elif col_lower in weather_keywords:
            weather_cols.append(col)

        # 4. Past Features / Lag columns (ED_1, ED_2, asthma_lag_1, etc.)
        elif '_lag_' in col_lower:
            lag_cols.append(col)
        elif col_lower.startswith('ed_') and len(col_lower) > 3 and col_lower[3:].isdigit():
            lag_cols.append(col)

        # 5. Future Targets (Target_1, Target_2, asthma_1, pneumonia_1, RESPIRATORY_1, etc.)
        elif col_lower.startswith('target_'):
            target_cols.append(col)
        elif '_' in col and col.split('_')[-1].isdigit() and any(keyword in col_lower for keyword in reason_keywords):
            # This catches columns like "asthma_1", "pneumonia_2", etc.
            target_cols.append(col)
        elif '_' in col and col.split('_')[-1].isdigit() and any(keyword in col_lower for keyword in clinical_category_keywords):
            # This catches columns like "RESPIRATORY_1", "CARDIAC_2", etc.
            target_cols.append(col)

        # Current reason columns (base columns without suffixes) - put with other
        elif any(col_lower == keyword for keyword in reason_keywords):
            other_cols.append(col)

        # Current clinical category columns (base columns without suffixes) - put with other
        elif any(col_lower == keyword for keyword in clinical_category_keywords):
            other_cols.append(col)

        # Everything else
        else:
            other_cols.append(col)

    # Sort within categories for consistency
    def sort_by_number(cols):
        """Sort columns by numeric suffix if present."""
        def get_num(c):
            parts = c.replace('_lag_', '_').split('_')
            for p in reversed(parts):
                if p.isdigit():
                    return int(p)
            return 0
        return sorted(cols, key=get_num)

    lag_cols = sort_by_number(lag_cols)
    target_cols = sort_by_number(target_cols)

    # Final order: Date → Calendar → Weather → Past (Lag) → Future (Targets)
    # Note: "other_cols" includes current reason columns - place them before lags
    ordered_cols = date_cols + calendar_cols + weather_cols + other_cols + lag_cols + target_cols

    # Ensure we haven't lost any columns
    if len(ordered_cols) != len(all_cols):
        missing = set(all_cols) - set(ordered_cols)
        ordered_cols.extend(list(missing))

    return df[ordered_cols]


# -------------------------------

def process_dataset(
    df: pd.DataFrame,
    n_lags: int = 7,
    date_col: Optional[str] = None,
    ed_col: Optional[str] = None,
    strict_drop_na_edges: bool = True,
    use_aggregated_categories: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and engineer a dataset for ED forecasting with multi-target support.
    Returns (processed_df, report).
    """
    report: Dict[str, Any] = {
        "steps": [],
        "date_col": None,
        "ed_col": None,
        "created_ed": False,
        "created_lags": [],
        "created_targets": [],
        "created_multi_lags": {},
        "created_multi_targets": {},
        "dropped_rows": 0,
        "notes": [],
        "columns_before": list(df.columns),
        "columns_after": None,
    }

    # 1) Standardize column names
    df = _standardize_columns(df)
    report["steps"].append("standardize_columns")
    # Keep a copy of original index
    original_len = len(df)

    # 2) Date column detection & parsing
    if date_col is None:
        date_col = _find_date_like(df)
    df, date_col = _ensure_datetime(df, date_col)
    report["date_col"] = date_col
    if date_col is None:
        report["notes"].append("No clear date column found; proceeding without time-aware operations.")
    else:
        # Drop duplicated timestamps keeping the first occurrence
        before = len(df)
        df = df.drop_duplicates(subset=[date_col])
        after = len(df)
        if after < before:
            report["notes"].append(f"Removed {before-after} duplicate rows based on {date_col}.")
        df = df.sort_values(date_col).reset_index(drop=True)

    # 3) Coerce numerics (ignore date col)
    df = _coerce_numeric(df, exclude=[date_col] if date_col else [])

    # 4) Identify or create ED column
    ed_guess = ed_col or _guess_ed_column(df)
    created_ed = False
    if ed_guess and ed_guess in df.columns:
        ed_col = ed_guess
    else:
        # Try to construct ED from Target_1 or ED_1
        constructed = _construct_ed_from_targets_or_lags(df)
        if constructed is not None:
            df["ED"] = constructed
            ed_col = "ED"
            created_ed = True
            report["notes"].append("Constructed ED from existing Target_1/ED_1 columns.")
        else:
            # Try last-resort: if any column looks like today's arrivals (integer small-ish)
            # We keep it simple: if only one integer-like column, adopt it.
            int_like = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
            if len(int_like) == 1:
                ed_col = int_like[0]
                report["notes"].append(f"Using '{ed_col}' as ED (last-resort heuristic).")
            else:
                report["notes"].append("Could not infer ED. Please specify ed_col.")
                ed_col = None

    report["ed_col"] = ed_col
    report["created_ed"] = created_ed

    # 5) Clean negatives on ED and targets
    if ed_col is not None and ed_col in df.columns:
        df[ed_col] = _clip_negatives(df[ed_col])

    for c in [c for c in df.columns if c.lower().startswith("target_") or c.lower().startswith("ed_")]:
        df[c] = _clip_negatives(df[c])

    # 6) Generate lags & targets if ED available
    if ed_col is not None and ed_col in df.columns:
        df = _generate_lags_and_targets(df, ed_col=ed_col, n_lags=n_lags)
        report["created_lags"] = [f"ED_{i}" for i in range(1, n_lags+1)]
        report["created_targets"] = [f"Target_{i}" for i in range(1, n_lags+1)]
    else:
        report["notes"].append("Skipped lag/target generation because ED was not identified.")

    # 6a) Create aggregated clinical categories from granular reason columns (if enabled)
    created_categories = []
    if use_aggregated_categories:
        df, created_categories = _create_clinical_categories(df)
        if created_categories:
            report["created_clinical_categories"] = created_categories
            report["notes"].append(f"Created {len(created_categories)} clinical categories: {', '.join(created_categories)}")
            report["steps"].append("create_clinical_categories")
    else:
        report["notes"].append("Aggregated clinical categories disabled - using granular reason columns only.")

    # 6b) Generate multi-lag and multi-target forecasting features for reason columns
    if use_aggregated_categories and created_categories:
        # Use aggregated clinical categories for multi-target forecasting
        reason_cols = created_categories
        report["notes"].append(f"Using aggregated clinical categories for multi-target: {', '.join(reason_cols)}")
    else:
        # Use granular reason columns for multi-target forecasting
        reason_cols = _detect_reason_columns(df)
        # Exclude the ED column itself to avoid duplicate lag features
        if ed_col and ed_col in reason_cols:
            reason_cols.remove(ed_col)
        # Also exclude the aggregated categories if they were created
        reason_cols = [c for c in reason_cols if c not in list(CLINICAL_CATEGORIES.keys())]

    if len(reason_cols) > 0:
        # Generate lag features (past values) - only for granular reasons, not aggregated
        if not use_aggregated_categories:
            df = _generate_multi_lags(df, lag_cols=reason_cols, n_lags=n_lags)
            for col in reason_cols:
                report["created_multi_lags"][col] = [f"{col}_lag_{i}" for i in range(1, n_lags+1)]
            report["notes"].append(f"Generated {n_lags} lag features for {len(reason_cols)} reason columns: {', '.join(reason_cols)}")
            report["steps"].append("generate_multi_lags")

        # Generate target features (future values)
        df = _generate_multi_targets(df, target_cols=reason_cols, n_lags=n_lags)
        for col in reason_cols:
            report["created_multi_targets"][col] = [f"{col}_{i}" for i in range(1, n_lags+1)]
        report["notes"].append(f"Generated {n_lags} future targets for {len(reason_cols)} reason columns: {', '.join(reason_cols)}")
        report["steps"].append("generate_multi_targets")

    # 7) Add calendar features
    df = _add_calendar_features(df, date_col=date_col)
    report["steps"].append("add_calendar_features")

    # 8) Impute numeric columns (non-destructive, time-aware)
    df = _impute_numerics(df, exclude=[date_col] if date_col else [])
    report["steps"].append("impute_numeric_linear_ffill_bfill")

    # 9) Drop edge rows introduced by shifting (optional)
    before_drop = len(df)
    if ed_col is not None and strict_drop_na_edges:
        needed_cols = [ed_col] + [f"ED_{i}" for i in range(1, n_lags+1)] + [f"Target_{i}" for i in range(1, n_lags+1)]
        # Also include multi-lag and multi-target columns for reason data
        for col in reason_cols:
            needed_cols.extend([f"{col}_lag_{i}" for i in range(1, n_lags+1)])
            needed_cols.extend([f"{col}_{i}" for i in range(1, n_lags+1)])
        have_cols = [c for c in needed_cols if c in df.columns]
        df = df.dropna(subset=have_cols)
        report["dropped_rows"] = before_drop - len(df)
        report["steps"].append(f"drop_na_edges({report['dropped_rows']})")

    # 10) Reorder columns: Date, current features, lag features, then target features
    df = _reorder_columns_for_training(df, date_col=date_col)
    report["steps"].append("reorder_columns_for_training")

    report["columns_after"] = list(df.columns)
    return df, report
