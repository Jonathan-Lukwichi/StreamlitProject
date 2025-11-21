
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
    """Detect reason for visit columns in the dataset."""
    reason_candidates = [
        "Respiratory_Cases", "Cardiac_Cases", "Trauma_Cases",
        "Gastrointestinal_Cases", "Infectious_Cases", "Other_Cases", "Total_Arrivals"
    ]
    found = [col for col in reason_candidates if col in df.columns]
    return found

def _generate_lags_and_targets(df: pd.DataFrame, ed_col: str, n_lags: int) -> pd.DataFrame:
    """Generate lag features (ED_1..ED_n) and future targets (Target_1..Target_n) from ED column."""
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"ED_{i}"] = df[ed_col].shift(i)
        df[f"Target_{i}"] = df[ed_col].shift(-i)
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

def process_dataset(
    df: pd.DataFrame,
    n_lags: int = 7,
    date_col: Optional[str] = None,
    ed_col: Optional[str] = None,
    strict_drop_na_edges: bool = True,
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

    # 6b) Generate multi-target forecasting features for reason columns
    reason_cols = _detect_reason_columns(df)
    if len(reason_cols) > 0:
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
        # Also include multi-target columns
        for col in reason_cols:
            needed_cols.extend([f"{col}_{i}" for i in range(1, n_lags+1)])
        have_cols = [c for c in needed_cols if c in df.columns]
        df = df.dropna(subset=have_cols)
        report["dropped_rows"] = before_drop - len(df)
        report["steps"].append(f"drop_na_edges({report['dropped_rows']})")

    report["columns_after"] = list(df.columns)
    return df, report
