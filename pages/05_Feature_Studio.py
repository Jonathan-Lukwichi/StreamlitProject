# =============================================================================
# pages/06_Advanced_Feature_Engineering.py — Advanced Feature Engineering (Flexible, Deduplicated)
# =============================================================================
from __future__ import annotations
import inspect
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
except Exception:
    OneHotEncoder = None
    MinMaxScaler = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header

# Import temporal split module
try:
    from app_core.pipelines import (
        compute_temporal_split,
        validate_temporal_split,
        TemporalSplitResult,
        # Time series cross-validation
        CVFoldResult,
        CVConfig,
        create_cv_folds,
        visualize_cv_folds,
        evaluate_cv_metrics,
        format_cv_metric,
    )
    TEMPORAL_SPLIT_AVAILABLE = True
    CV_AVAILABLE = True
except ImportError:
    TEMPORAL_SPLIT_AVAILABLE = False
    CV_AVAILABLE = False

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        st.session_state.setdefault("feature_engineering", {})

try:
    from app_core.ui.components import header
except Exception:
    def header(title: str, subtitle: str = "", icon: str = ""):
        st.markdown(f"### {icon} {title}")
        if subtitle: st.caption(subtitle)

st.set_page_config(
    page_title="Feature Studio - HealthForecast AI",
    page_icon="🧪",
    layout="wide",
)

# === HELPERS ==================================================================

def render_tab_header(title: str, icon: str = ""):
    """Render a tab title in a blue fluorescent card style."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(6, 78, 145, 0.25) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(34, 211, 238, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <h4 style="
            margin: 0;
            color: #22d3ee;
            font-weight: 700;
            font-size: 1.25rem;
            text-shadow: 0 0 10px rgba(34, 211, 238, 0.3);
        ">{icon} {title}</h4>
    </div>
    """, unsafe_allow_html=True)

DATE_PREFS = ["datetime", "Date", "date", "timestamp", "ds", "time"]
HOLIDAY_PREFS = ["is_holiday","Is_holiday","Holiday","holiday","IsHoliday"]

def _get_processed() -> Optional[pd.DataFrame]:
    for k in ["processed_df", "data_final", "prepared_data", "preprocessed_data"]:
        df = st.session_state.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None

def _guess_date_col(df: pd.DataFrame) -> str:
    for c in DATE_PREFS:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["date","time","stamp"]):
            return c
    return df.columns[0]

def _guess_targets(df: pd.DataFrame) -> List[str]:
    t = [c for c in df.columns if c.lower().startswith("target_")]
    if t: return sorted(t, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    alt = [c for c in df.columns if c.lower() in ("target","y","label")]
    return alt

def _ensure_dt_sorted(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df

def _holiday_series(df: pd.DataFrame) -> pd.Series:
    for c in HOLIDAY_PREFS:
        if c in df.columns:
            s = df[c]
            if s.dtype == bool: return s.astype(int)
            return s.astype(str).str.lower().isin(["1","true","yes","y"]).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def _calendar_frame(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    cal = pd.DataFrame(index=df.index)
    cal["year"] = s.dt.year
    cal["quarter"] = s.dt.quarter
    cal["month"] = s.dt.month
    cal["week_of_year"] = s.dt.isocalendar().week.astype(int)
    cal["day_of_year"] = s.dt.dayofyear
    cal["day_of_month"] = s.dt.day
    cal["day_of_week"] = s.dt.weekday
    cal["is_weekend"] = cal["day_of_week"].isin([5,6]).astype(int)
    cal["is_holiday"] = _holiday_series(df)
    cal["is_month_end"] = s.dt.is_month_end.astype(int)
    cal["is_quarter_end"] = s.dt.is_quarter_end.astype(int)
    cal["is_year_end"] = s.dt.is_year_end.astype(int)
    return cal

def _weekly_harmonics(day_of_week: pd.Series, orders: List[int] = [1,2]) -> pd.DataFrame:
    out = pd.DataFrame(index=day_of_week.index)
    x = day_of_week.astype(float).to_numpy()
    for k in orders:
        out[f"dow_sin_k{k}"] = np.sin(2*np.pi*k*x/7.0)
        out[f"dow_cos_k{k}"] = np.cos(2*np.pi*k*x/7.0)
    return out

def _cyclical_block(df: pd.DataFrame, date_col: str, cal: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    def _sin_cos(series: pd.Series, period: int, name: str):
        x = series.astype(float).to_numpy()
        out[f"{name}_sin"] = np.sin(2*np.pi*x/period)
        out[f"{name}_cos"] = np.cos(2*np.pi*x/period)
    _sin_cos(cal["month"], 12, "month")
    _sin_cos(cal["day_of_week"], 7, "dow")
    _sin_cos(cal["day_of_month"], 31, "dom")
    _sin_cos(cal["day_of_year"], 365, "doy")
    out = pd.concat([out, _weekly_harmonics(cal["day_of_week"], [1,2])], axis=1)
    for b in ["is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"]:
        out[b] = cal[b]
    out["week_of_year"] = cal["week_of_year"]
    return out

# =============================================================================
# FOURIER FEATURES GENERATOR (Step 2.2: Automate from FFT)
# Academic Reference: Box et al. (2015) - Time Series Analysis, Harvey (1989)
# =============================================================================
def generate_fourier_features(
    df: pd.DataFrame,
    date_col: str,
    periods: List[float],
    harmonics_per_period: int = 1,
    prefix: str = "fourier"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate Fourier (sin/cos) features from detected FFT cycle periods.

    Fourier features capture periodic patterns by decomposing seasonality into
    sine and cosine pairs. For period P and harmonic k, features are:
        sin(2πk·t/P) and cos(2πk·t/P)

    Academic Reference:
        - Box, Jenkins, Reinsel & Ljung (2015) - Time Series Analysis
        - Harvey (1989) - Forecasting, Structural Time Series Models

    Args:
        df: Input DataFrame with a date column
        date_col: Name of the date column
        periods: List of cycle periods (in days) detected from FFT
        harmonics_per_period: Number of harmonics (k=1,2,...) per period
        prefix: Prefix for feature names

    Returns:
        Tuple of (DataFrame with Fourier features, dict with feature info)
    """
    work = df.copy()
    feature_info = {
        'fourier_features': [],
        'periods_used': periods,
        'harmonics': harmonics_per_period,
    }

    # Convert date to numeric (days from start)
    dates = pd.to_datetime(work[date_col])
    t = (dates - dates.min()).dt.days.values.astype(float)

    fourier_df = pd.DataFrame(index=work.index)

    for period in periods:
        if period <= 0 or not np.isfinite(period):
            continue

        # Normalize period name for column naming
        period_name = f"{period:.0f}" if period >= 1 else f"{period:.2f}"

        for k in range(1, harmonics_per_period + 1):
            # Fourier terms: sin(2πk·t/P) and cos(2πk·t/P)
            sin_col = f"{prefix}_sin_p{period_name}_k{k}"
            cos_col = f"{prefix}_cos_p{period_name}_k{k}"

            fourier_df[sin_col] = np.sin(2 * np.pi * k * t / period)
            fourier_df[cos_col] = np.cos(2 * np.pi * k * t / period)

            feature_info['fourier_features'].extend([sin_col, cos_col])

    feature_info['total_features'] = len(feature_info['fourier_features'])

    return fourier_df, feature_info


# =============================================================================
# DIFFERENCING FUNCTIONS (Step 3.1: Stationarity Auto-Response)
# Academic Reference: Dickey & Fuller (1979), Box & Jenkins (1976)
# =============================================================================
@dataclass
class DifferencingResult:
    """Store differencing transformation details for inverse transform."""
    original_first_values: Dict[str, float]  # Column: first value for each differenced column
    differencing_order: int
    columns_differenced: List[str]
    applied: bool = True


def apply_differencing(
    df: pd.DataFrame,
    target_columns: List[str],
    order: int = 1,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, DifferencingResult]:
    """
    Apply differencing to make time series stationary.

    Differencing removes trends by computing: y'(t) = y(t) - y(t-1)
    Second-order differencing: y''(t) = y'(t) - y'(t-1)

    Academic Reference:
        - Box & Jenkins (1976) - Time Series Analysis: Forecasting and Control
        - Dickey & Fuller (1979) - Distribution of the Estimators for
          Autoregressive Time Series with a Unit Root

    Args:
        df: Input DataFrame
        target_columns: Columns to difference
        order: Differencing order (1 = first difference, 2 = second difference)
        drop_na: Whether to drop NaN values created by differencing

    Returns:
        Tuple of (differenced DataFrame, DifferencingResult for inverse transform)
    """
    work = df.copy()
    first_values = {}

    for col in target_columns:
        if col not in work.columns:
            continue

        # Store first value(s) needed for inverse transform
        if order == 1:
            first_values[col] = float(work[col].iloc[0])
        elif order == 2:
            first_values[col] = [float(work[col].iloc[0]), float(work[col].iloc[1])]

        # Apply differencing
        for _ in range(order):
            work[col] = work[col].diff()

    # Drop NaN rows created by differencing
    if drop_na:
        work = work.dropna().reset_index(drop=True)

    result = DifferencingResult(
        original_first_values=first_values,
        differencing_order=order,
        columns_differenced=target_columns,
        applied=True
    )

    return work, result


def inverse_differencing(
    differenced_values: np.ndarray,
    diff_result: DifferencingResult,
    column: str,
    original_series: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Inverse differencing to recover original scale predictions.

    For first-order differencing: y(t) = y'(t) + y(t-1)
    For second-order: y(t) = y''(t) + 2*y(t-1) - y(t-2)

    Academic Reference:
        - Box & Jenkins (1976) - Inverse transforms for ARIMA

    Args:
        differenced_values: Differenced predictions
        diff_result: DifferencingResult from apply_differencing()
        column: Name of the column to inverse transform
        original_series: Original undifferenced series (needed for proper reconstruction)

    Returns:
        Reconstructed values in original scale
    """
    if not diff_result.applied or column not in diff_result.columns_differenced:
        return differenced_values

    order = diff_result.differencing_order
    first_vals = diff_result.original_first_values.get(column)

    if first_vals is None:
        return differenced_values

    values = differenced_values.copy()

    if order == 1:
        # Cumulative sum to reverse first difference
        # Need last known value from training set for proper reconstruction
        if original_series is not None and len(original_series) > 0:
            last_known = float(original_series.iloc[-1])
            values = np.cumsum(values) + last_known
        else:
            values = np.cumsum(values) + first_vals

    elif order == 2:
        # Reverse second-order differencing
        if original_series is not None and len(original_series) >= 2:
            last_val = float(original_series.iloc[-1])
            second_last = float(original_series.iloc[-2])
            # First integrate to get first difference
            values = np.cumsum(values)
            # Then integrate again
            reconstructed = np.zeros(len(values))
            reconstructed[0] = values[0] + last_val
            for i in range(1, len(values)):
                reconstructed[i] = values[i] + reconstructed[i-1]
            values = reconstructed
        else:
            # Fallback with stored first values
            values = np.cumsum(np.cumsum(values))

    return values


def get_stationarity_status() -> Dict:
    """
    Get stationarity test results from session state.

    Returns:
        Dict with 'is_stationary', 'p_value', 'adf_statistic', 'tested'
    """
    adf_data = st.session_state.get("adf_stationarity", {})

    if not adf_data:
        return {
            "tested": False,
            "is_stationary": None,
            "p_value": None,
            "adf_statistic": None,
            "recommendation": "Run ADF test in Explore Data page first."
        }

    is_stationary = adf_data.get("is_stationary", True)
    p_value = adf_data.get("p_value", 1.0)

    if is_stationary:
        recommendation = "Series is stationary. No differencing required."
    else:
        recommendation = "Series is non-stationary (p ≥ 0.05). Differencing recommended."

    return {
        "tested": True,
        "is_stationary": is_stationary,
        "p_value": p_value,
        "adf_statistic": adf_data.get("adf_statistic"),
        "series_name": adf_data.get("series_name", "target"),
        "tested_at": adf_data.get("tested_at"),
        "recommendation": recommendation
    }


def get_recommended_periods_from_fft() -> List[Dict]:
    """
    Get recommended periods from FFT analysis stored in session state.

    Returns:
        List of dicts with 'period_days', 'cycle_type', 'amplitude'
    """
    fft_data = st.session_state.get("fft_dominant_cycles", {})
    cycles_list = fft_data.get("cycles_df", [])

    # Add standard periods if not detected
    standard_periods = {
        "weekly": 7.0,
        "bi-weekly": 14.0,
        "monthly": 30.44,  # Average month
        "quarterly": 91.31,
        "yearly": 365.25,
    }

    # Merge detected with standards
    result = []
    detected_types = set()

    for cycle in cycles_list:
        result.append({
            "period_days": cycle.get("period_days", 0),
            "cycle_type": cycle.get("cycle_type", "other"),
            "amplitude": cycle.get("amplitude", 0),
            "source": "FFT detected"
        })
        detected_types.add(cycle.get("cycle_type", ""))

    # Add standard periods not already detected
    for cycle_type, period in standard_periods.items():
        if cycle_type not in detected_types:
            result.append({
                "period_days": period,
                "cycle_type": cycle_type,
                "amplitude": 0,
                "source": "Standard (not detected)"
            })

    return result


def _make_ohe():
    params = set(inspect.signature(OneHotEncoder).parameters.keys())
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if "sparse" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    return OneHotEncoder(handle_unknown="ignore")

def _safe_concat(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two dataframes safely, dropping duplicate columns."""
    common = [c for c in extra.columns if c in base.columns]
    if common:
        extra = extra.drop(columns=common)
    out = pd.concat([base, extra], axis=1)
    # Final deduplication safeguard
    out = out.loc[:, ~out.columns.duplicated()]
    return out

# Legacy function (kept for fallback if temporal_split not available)
def _split_indices_legacy(df: pd.DataFrame, date_col: str, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy ratio-based split (fallback only)."""
    n = len(df)
    n_tr = int(np.floor(n*ratio))
    return np.arange(n_tr), np.arange(n_tr, n)


def _get_temporal_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[TemporalSplitResult]]:
    """
    Compute temporal split using the academically-rigorous pipeline module.

    Returns:
        Tuple of (train_idx, cal_idx, test_idx, split_result)
        If temporal_split module not available, cal_idx will be empty.
    """
    if TEMPORAL_SPLIT_AVAILABLE:
        split_result = compute_temporal_split(
            df=df,
            date_col=date_col,
            train_ratio=train_ratio,
            cal_ratio=cal_ratio,
            verbose=False
        )
        return (
            split_result.train_idx,
            split_result.cal_idx,
            split_result.test_idx,
            split_result
        )
    else:
        # Fallback to legacy 2-way split
        train_idx, test_idx = _split_indices_legacy(df, date_col, train_ratio + cal_ratio)
        return train_idx, np.array([], dtype=int), test_idx, None


# ============================================================================
# TRANSFORMER PERSISTENCE (Academic Best Practice: Save scalers for inference)
# ============================================================================
TRANSFORMERS_DIR = Path("app_core/pipelines/saved_transformers")

def _ensure_transformers_dir():
    """Create directory for saving transformers if it doesn't exist."""
    if not TRANSFORMERS_DIR.exists():
        TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)

def _ensure_transformers_from_cloud():
    """
    Download transformers from Supabase Storage if not present locally.

    This enables the 'Train Locally, Deploy to Cloud' workflow:
    - Transformers (scalers, encoders) fitted during local training
    - Uploaded to Supabase via Cloud Sync tab
    - Downloaded automatically on Streamlit Cloud startup

    Returns:
        int: Number of transformers downloaded (0 if already present or not connected)
    """
    _ensure_transformers_dir()

    # Check if we already have transformers locally
    existing_files = list(TRANSFORMERS_DIR.glob("*.joblib"))
    if existing_files:
        return 0  # Already have local transformers

    # Try to download from cloud
    try:
        from app_core.data.model_storage_service import download_transformers
        results = download_transformers(str(TRANSFORMERS_DIR))

        if "error" in results:
            return 0

        downloaded = sum(1 for key, (success, _) in results.items() if key != "error" and success)
        if downloaded > 0:
            import logging
            logging.getLogger(__name__).info(f"Downloaded {downloaded} transformer(s) from cloud storage")
        return downloaded
    except ImportError:
        # Model storage service not available (e.g., no Supabase connection)
        return 0
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not download transformers from cloud: {e}")
        return 0

# Auto-download transformers on module load (for cloud deployment)
_ensure_transformers_from_cloud()

def _save_transformer(transformer, name: str, variant: str = "A") -> Optional[str]:
    """
    Save a fitted transformer (scaler/encoder) for later inference.

    Academic Reference:
        Prevents data leakage by ensuring production uses the SAME transformer
        fitted on training data only.
    """
    if not JOBLIB_AVAILABLE:
        return None

    _ensure_transformers_dir()
    filepath = TRANSFORMERS_DIR / f"{name}_{variant}.joblib"
    try:
        joblib.dump(transformer, filepath)
        return str(filepath)
    except Exception as e:
        st.warning(f"Could not save transformer {name}: {e}")
        return None

def _load_transformer(name: str, variant: str = "A"):
    """Load a previously saved transformer."""
    if not JOBLIB_AVAILABLE:
        return None

    filepath = TRANSFORMERS_DIR / f"{name}_{variant}.joblib"
    if filepath.exists():
        return joblib.load(filepath)
    return None

# === PIPELINES ================================================================
def _build_variant_A(base, date_col, targets, train_idx, cal_idx, test_idx, apply_scaling=True, save_transformers=True):
    """
    Build Variant A: One-Hot Encoded categorical calendar features.

    Academic Notes:
        - OneHotEncoder fitted ONLY on train_idx (prevents data leakage)
        - MinMaxScaler fitted ONLY on train_idx
        - Transformers saved for inference consistency
    """
    cal = _calendar_frame(base, date_col)
    cat_df = cal[["month","quarter","day_of_week","year"]]

    # Fit OHE on training data ONLY
    ohe = _make_ohe()
    ohe.fit(cat_df.iloc[train_idx])

    # Save OHE for later inference
    if save_transformers:
        _save_transformer(ohe, "ohe", variant="A")

    names = ohe.get_feature_names_out(["month","quarter","day_of_week","year"])
    arr = ohe.transform(cat_df)
    if hasattr(arr, "toarray"): arr = arr.toarray()
    ohe_df = pd.DataFrame(arr, columns=names, index=base.index)
    keep = ["week_of_year","day_of_year","is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"]
    cal_keep = cal[keep]
    A = _safe_concat(base, cal_keep)
    A = _safe_concat(A, ohe_df)

    if apply_scaling and MinMaxScaler is not None:
        exclude = set([date_col] + targets + [c for c in A.columns if c.startswith("ED_")])
        exclude.update({"is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"})
        num_cols = A.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in num_cols if c not in exclude]
        if scale_cols:
            # Fit scaler on training data ONLY (Academic Best Practice)
            scaler = MinMaxScaler().fit(A.loc[train_idx, scale_cols])

            # Save scaler for later inference
            if save_transformers:
                _save_transformer(scaler, "scaler", variant="A")
                # Also save the column names for reference
                _save_transformer(scale_cols, "scale_cols", variant="A")

            # Transform ALL data (train, cal, test) using train-fitted scaler
            A.loc[:, scale_cols] = scaler.transform(A.loc[:, scale_cols])

    A = A.loc[:, ~A.columns.duplicated()]
    return A


def _build_variant_B(base, date_col, targets, train_idx, cal_idx, test_idx, apply_scaling=True, save_transformers=True):
    """
    Build Variant B: Cyclical (sin/cos) encoded calendar features.

    Academic Notes:
        - Cyclical encoding doesn't require fitting (mathematical transform)
        - MinMaxScaler fitted ONLY on train_idx
        - Transformer saved for inference consistency
    """
    cal = _calendar_frame(base, date_col)
    cyc = _cyclical_block(base, date_col, cal)
    B = _safe_concat(base, cyc)

    if apply_scaling and MinMaxScaler is not None:
        exclude = set([date_col] + targets + [c for c in B.columns if c.startswith("ED_")])
        exclude.update({"is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"})
        num_cols = B.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in num_cols if c not in exclude]
        if scale_cols:
            # Fit scaler on training data ONLY (Academic Best Practice)
            scaler = MinMaxScaler().fit(B.loc[train_idx, scale_cols])

            # Save scaler for later inference
            if save_transformers:
                _save_transformer(scaler, "scaler", variant="B")
                _save_transformer(scale_cols, "scale_cols", variant="B")

            # Transform ALL data using train-fitted scaler
            B.loc[:, scale_cols] = scaler.transform(B.loc[:, scale_cols])

    B = B.loc[:, ~B.columns.duplicated()]
    return B

# === PAGE BODY ================================================================
def page_feature_engineering():

    base = _get_processed()
    if base is None or base.empty:
        st.warning("⚠️ No processed dataset found. Run Data Preparation Studio first.")
        return

    date_col = _guess_date_col(base)
    targets = _guess_targets(base)
    base = _ensure_dt_sorted(base, date_col)

    # =========================================================================
    # FEATURE ENGINEERING TABS
    # Academic Reference: Bergmeir & Benítez (2012), Tashman (2000)
    # =========================================================================
    tab_split, tab_cv, tab_fourier, tab_options, tab_generate = st.tabs([
        "📊 Data Split",
        "🔄 Cross-Validation",
        "🌊 Fourier Features",
        "🧪 Options",
        "🚀 Generate"
    ])

    with tab_split:
        render_tab_header("Configure Train/Validation/Test Split", "📊")

        # 3-way split configuration
        split_col1, split_col2 = st.columns(2)

        with split_col1:
            train_ratio = st.slider(
                "Training Set Ratio",
                min_value=0.50,
                max_value=0.80,
                value=0.70,
                step=0.05,
                help="Proportion of data for training (recommended: 70%)"
            )

        with split_col2:
            val_ratio = st.slider(
                "Validation Set Ratio",
                min_value=0.10,
                max_value=0.20,
                value=0.15,
                step=0.05,
                help="Proportion of data for validation/calibration (recommended: 15%)"
            )

        test_ratio = round(1.0 - train_ratio - val_ratio, 2)

        # Validate ratios
        if test_ratio < 0.05:
            st.warning(f"⚠️ Test set too small ({test_ratio:.0%}). Reduce train or validation ratio.")

        # Show split preview
        min_date = base[date_col].min()
        max_date = base[date_col].max()
        total_days = (max_date - min_date).days

        from datetime import timedelta
        train_end = min_date + timedelta(days=int(total_days * train_ratio))
        val_end = train_end + timedelta(days=int(total_days * val_ratio))

        col_prev1, col_prev2, col_prev3 = st.columns(3)
        with col_prev1:
            st.metric("Train", f"{train_ratio:.0%}", f"{min_date.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
        with col_prev2:
            st.metric("Validation", f"{val_ratio:.0%}", f"{train_end.strftime('%Y-%m-%d')} → {val_end.strftime('%Y-%m-%d')}")
        with col_prev3:
            st.metric("Test", f"{test_ratio:.0%}", f"{val_end.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")

        # Compute proper 3-way temporal split
        train_idx, cal_idx, test_idx, split_result = _get_temporal_split(
            base, date_col, train_ratio=train_ratio, cal_ratio=val_ratio
        )

    # =========================================================================
    # CROSS-VALIDATION TAB
    # Academic Reference: Bergmeir & Benítez (2012)
    # =========================================================================
    with tab_cv:
        render_tab_header("Time Series Cross-Validation", "🔄")

        if not CV_AVAILABLE:
            st.warning("⚠️ Cross-validation module not available. Please check installation.")
        else:
            # CV Configuration
            cv_col1, cv_col2 = st.columns(2)

            with cv_col1:
                n_splits = st.slider(
                    "Number of Folds",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Number of train/validation splits. More folds = more robust estimates but slower training."
                )

                gap_days = st.slider(
                    "Gap (days)",
                    min_value=0,
                    max_value=14,
                    value=0,
                    step=1,
                    help="Days between training and validation sets to prevent autocorrelation leakage. "
                         "Recommended: 0-7 days for daily data."
                )

            with cv_col2:
                use_cv = st.checkbox(
                    "Enable Cross-Validation for Model Training",
                    value=True,
                    help="When enabled, models will be trained on all CV folds and metrics will show mean ± std."
                )

                min_train_pct = st.slider(
                    "Minimum Training Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Minimum percentage of data required for training in the first fold."
                )

            # Save CV config to session state
            cv_config = CVConfig(
                n_splits=n_splits,
                gap=gap_days,
                min_train_size=int(len(base) * min_train_pct / 100) if min_train_pct > 0 else None,
                test_size=None
            )
            st.session_state["cv_config"] = cv_config
            st.session_state["cv_enabled"] = use_cv

            # Preview CV folds
            st.markdown("---")
            st.markdown("##### CV Fold Preview")

            try:
                cv_folds = create_cv_folds(
                    df=base,
                    date_col=date_col,
                    n_splits=n_splits,
                    gap=gap_days,
                    min_train_size=cv_config.min_train_size,
                    verbose=False
                )

                # Display fold summary
                fold_data = []
                for fold in cv_folds:
                    fold_data.append({
                        "Fold": fold.fold_idx + 1,
                        "Train Start": fold.train_start.strftime("%Y-%m-%d"),
                        "Train End": fold.train_end.strftime("%Y-%m-%d"),
                        "Train Samples": fold.n_train,
                        "Val Start": fold.val_start.strftime("%Y-%m-%d"),
                        "Val End": fold.val_end.strftime("%Y-%m-%d"),
                        "Val Samples": fold.n_val,
                    })
                fold_df = pd.DataFrame(fold_data)
                st.dataframe(fold_df, use_container_width=True, hide_index=True)

                # Visualization
                fig = visualize_cv_folds(base, date_col, cv_folds, "Time Series Cross-Validation Folds")
                st.plotly_chart(fig, use_container_width=True, key="cv_folds_viz")

                # Summary metrics
                total_train = sum(f.n_train for f in cv_folds)
                total_val = sum(f.n_val for f in cv_folds)
                avg_train = total_train / len(cv_folds) if cv_folds else 0
                avg_val = total_val / len(cv_folds) if cv_folds else 0

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Active Folds", len(cv_folds))
                with metric_cols[1]:
                    st.metric("Avg Train Size", f"{avg_train:,.0f}")
                with metric_cols[2]:
                    st.metric("Avg Val Size", f"{avg_val:,.0f}")
                with metric_cols[3]:
                    st.metric("Total Samples", f"{len(base):,}")

                # Save folds to session state for use in training
                st.session_state["cv_folds"] = cv_folds

                st.success(f"✅ {len(cv_folds)} CV folds configured. Ready for model training with cross-validation.")

            except Exception as e:
                st.error(f"Error creating CV folds: {str(e)}")
                st.session_state["cv_folds"] = None

    # =========================================================================
    # FOURIER FEATURES TAB (Step 2.2: Auto-generate from FFT)
    # Academic Reference: Box et al. (2015), Harvey (1989)
    # =========================================================================
    with tab_fourier:
        render_tab_header("Fourier Features from FFT Analysis", "🌊")

        # Check for FFT data from EDA page
        fft_data = st.session_state.get("fft_dominant_cycles", {})

        if not fft_data:
            st.warning("""
            ⚠️ **No FFT cycles detected yet.**

            Please run FFT analysis in the **Explore Data** page first:
            1. Go to Explore Data (Page 4)
            2. Click the "Advanced Analytics" tab
            3. View the "Dominant Cycles (FFT)" section
            4. FFT cycles will be automatically saved

            Or, you can manually configure Fourier features below.
            """)

            use_manual = st.checkbox("Configure Fourier features manually", value=False)

            if use_manual:
                # Manual period configuration
                st.markdown("##### Select Periods for Fourier Features")

                period_options = {
                    "Weekly (7 days)": 7.0,
                    "Bi-weekly (14 days)": 14.0,
                    "Monthly (30.44 days)": 30.44,
                    "Quarterly (91.31 days)": 91.31,
                    "Yearly (365.25 days)": 365.25,
                }

                selected_periods = st.multiselect(
                    "Standard Periods",
                    options=list(period_options.keys()),
                    default=["Weekly (7 days)"],
                    help="Select standard seasonal periods"
                )

                custom_period = st.number_input(
                    "Custom Period (days)",
                    min_value=1.0,
                    max_value=730.0,
                    value=0.0,
                    step=1.0,
                    help="Add a custom period (0 to skip)"
                )

                periods_to_use = [period_options[p] for p in selected_periods]
                if custom_period > 0:
                    periods_to_use.append(custom_period)
            else:
                periods_to_use = []
        else:
            st.success(f"✅ **FFT cycles detected** — Detected at: {fft_data.get('detected_at', 'unknown')}")

            # Display detected cycles
            cycles_list = fft_data.get("cycles_df", [])
            if cycles_list:
                st.markdown("##### Detected Cycles from FFT")
                cycles_df = pd.DataFrame(cycles_list)
                st.dataframe(cycles_df, use_container_width=True, hide_index=True)

                # Dominant cycle type
                dominant = fft_data.get("dominant_type", "unknown")
                st.markdown(f"**Dominant cycle type:** `{dominant}`")

            # Let user select which cycles to use
            st.markdown("##### Select Cycles for Fourier Features")

            recommended = get_recommended_periods_from_fft()
            recommended_df = pd.DataFrame(recommended)

            if not recommended_df.empty:
                # Filter to detected cycles for auto-selection
                detected_cycles = [c for c in recommended if c.get("source") == "FFT detected"]
                default_periods = [c.get("period_days") for c in detected_cycles[:3]]

                all_periods = recommended_df["period_days"].tolist()
                period_labels = {
                    p: f"{row['cycle_type']} ({p:.1f} days) - {row['source']}"
                    for p, row in zip(recommended_df["period_days"], recommended_df.to_dict("records"))
                }

                selected = st.multiselect(
                    "Cycles to include",
                    options=all_periods,
                    default=default_periods,
                    format_func=lambda x: period_labels.get(x, f"{x:.1f} days"),
                    help="Select which cycle periods to convert to Fourier features"
                )

                periods_to_use = selected
            else:
                periods_to_use = []

        # Harmonics configuration
        st.markdown("---")
        fourier_col1, fourier_col2 = st.columns(2)

        with fourier_col1:
            n_harmonics = st.slider(
                "Harmonics per Period",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                help="Number of harmonic terms (k=1,2,...) per period. Higher values capture sharper patterns."
            )

        with fourier_col2:
            enable_fourier = st.checkbox(
                "Enable Fourier Features for Generation",
                value=len(periods_to_use) > 0,
                help="Include Fourier features in the generated dataset"
            )

        # Preview features to be generated
        if periods_to_use and enable_fourier:
            n_features = len(periods_to_use) * n_harmonics * 2  # sin + cos per harmonic
            st.markdown("---")
            preview_cols = st.columns(3)
            with preview_cols[0]:
                st.metric("Periods Selected", len(periods_to_use))
            with preview_cols[1]:
                st.metric("Harmonics", n_harmonics)
            with preview_cols[2]:
                st.metric("Total Fourier Features", n_features)

            # Show feature names preview
            with st.expander("Preview Feature Names"):
                preview_names = []
                for p in periods_to_use[:3]:  # Show first 3
                    p_name = f"{p:.0f}" if p >= 1 else f"{p:.2f}"
                    for k in range(1, n_harmonics + 1):
                        preview_names.append(f"fourier_sin_p{p_name}_k{k}")
                        preview_names.append(f"fourier_cos_p{p_name}_k{k}")
                if len(periods_to_use) > 3:
                    preview_names.append(f"... (+{(len(periods_to_use)-3)*n_harmonics*2} more)")
                st.code("\n".join(preview_names))

        # Save Fourier config to session state
        st.session_state["fourier_config"] = {
            "enabled": enable_fourier,
            "periods": periods_to_use,
            "n_harmonics": n_harmonics,
            "n_features": len(periods_to_use) * n_harmonics * 2 if periods_to_use else 0,
        }

    with tab_options:
        render_tab_header("Feature Engineering Options", "🧪")

        c1, c2, c3 = st.columns(3)
        with c1:
            skip_engineering = st.checkbox("Skip Feature Engineering", value=False)
        with c2:
            apply_scaling = st.checkbox("Apply Min–Max Scaling", value=True)
        with c3:
            save_transformers = st.checkbox("Save Transformers", value=True,
                                            help="Save scalers/encoders for production inference")

    with tab_generate:
        render_tab_header("Generate Dataset", "🚀")

        run = st.button("🚀 Generate Dataset", type="primary")
        if not run:
            st.dataframe(base.head(10), use_container_width=True)
            return

        with st.spinner("Processing features..."):
            if skip_engineering:
                A_full = base.copy()
                B_full = base.copy()
                variant_used = "Original dataset (no feature engineering)"
            else:
                A_full = _build_variant_A(base, date_col, targets, train_idx, cal_idx, test_idx,
                                          apply_scaling, save_transformers)
                B_full = _build_variant_B(base, date_col, targets, train_idx, cal_idx, test_idx,
                                          apply_scaling, save_transformers)
                variant_used = "Feature engineered (deduplicated)"

            # =================================================================
            # APPLY FOURIER FEATURES (Step 2.2)
            # =================================================================
            fourier_config = st.session_state.get("fourier_config", {})
            fourier_info = None

            if fourier_config.get("enabled", False) and fourier_config.get("periods"):
                fourier_df, fourier_info = generate_fourier_features(
                    df=base,
                    date_col=date_col,
                    periods=fourier_config["periods"],
                    harmonics_per_period=fourier_config.get("n_harmonics", 1),
                    prefix="fourier"
                )

                if not fourier_df.empty:
                    # Add Fourier features to both variants
                    A_full = _safe_concat(A_full, fourier_df)
                    B_full = _safe_concat(B_full, fourier_df)
                    variant_used += f" + {fourier_info['total_features']} Fourier features"

                    # Save Fourier info for reference
                    st.session_state["fourier_features_info"] = fourier_info

            # =================================================================
            # APPLY DIFFERENCING (Step 3.1: Stationarity Auto-Response)
            # =================================================================
            diff_config = st.session_state.get("differencing_config", {})
            diff_result = None

            if diff_config.get("enabled", False) and diff_config.get("target_columns"):
                diff_targets = diff_config["target_columns"]
                diff_order = diff_config.get("order", 1)

                # Apply differencing to both variants
                A_full, diff_result_A = apply_differencing(
                    df=A_full,
                    target_columns=diff_targets,
                    order=diff_order,
                    drop_na=True
                )
                B_full, diff_result_B = apply_differencing(
                    df=B_full,
                    target_columns=diff_targets,
                    order=diff_order,
                    drop_na=True
                )

                diff_result = diff_result_A  # Store for reference
                variant_used += f" + Differencing(order={diff_order})"

                # Update indices to account for dropped rows
                n_dropped = diff_order
                if len(train_idx) > n_dropped:
                    train_idx = train_idx[train_idx >= n_dropped] - n_dropped
                if len(test_idx) > n_dropped:
                    test_idx = test_idx[test_idx >= n_dropped] - n_dropped

                # Save differencing info
                st.session_state["differencing_result"] = {
                    "applied": True,
                    "order": diff_order,
                    "columns": diff_targets,
                    "original_first_values": diff_result.original_first_values,
                    "rows_dropped": n_dropped,
                }

            # Validate temporal split if available
            if TEMPORAL_SPLIT_AVAILABLE and split_result is not None:
                validation_result = validate_temporal_split(
                    df=base,
                    date_col=date_col,
                    split_result=split_result
                )
                if not validation_result["valid"]:
                    st.error(f"⚠️ Temporal split validation failed: {validation_result['issues']}")
                if validation_result.get("warnings"):
                    for warning in validation_result["warnings"]:
                        st.warning(warning)

        # =========================================================================
        # STORE IN SESSION STATE (70/15/15 Train/Val/Test split)
        # =========================================================================
        st.session_state["feature_engineering"] = {
            "summary": {
                "variant_used": variant_used,
                "apply_scaling": apply_scaling,
                "skip_engineering": skip_engineering,
                "train_size": len(train_idx),
                "val_size": len(cal_idx),
                "test_size": len(test_idx),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "transformers_saved": save_transformers and not skip_engineering,
                "fourier_enabled": fourier_info is not None,
                "fourier_features": fourier_info.get("total_features", 0) if fourier_info else 0,
                "differencing_enabled": diff_result is not None,
                "differencing_order": diff_config.get("order", 1) if diff_result else 0,
                "differencing_columns": diff_config.get("target_columns", []) if diff_result else [],
            },
            "A": A_full,
            "B": B_full,
            "train_idx": train_idx,
            "cal_idx": cal_idx,    # Validation/calibration set for early stopping
            "test_idx": test_idx,
            "split_result": split_result,
            "fourier_info": fourier_info,
            "differencing_result": diff_result,
        }

        # =========================================================================
        # DISPLAY RESULTS
        # =========================================================================
        st.success(f"✅ Dataset ready: {variant_used}")

        # Show split summary (Train/Val/Test + optional Fourier)
        if fourier_info:
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        else:
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            col_sum4 = None

        with col_sum1:
            st.metric("Train Records", f"{len(train_idx):,}", f"{len(train_idx)/len(base)*100:.1f}%")
        with col_sum2:
            st.metric("Validation Records", f"{len(cal_idx):,}", f"{len(cal_idx)/len(base)*100:.1f}%")
        with col_sum3:
            st.metric("Test Records", f"{len(test_idx):,}", f"{len(test_idx)/len(base)*100:.1f}%")
        if col_sum4 and fourier_info:
            with col_sum4:
                st.metric("Fourier Features", f"{fourier_info['total_features']}", f"{len(fourier_info['periods_used'])} periods")

        if save_transformers and not skip_engineering:
            st.info(f"💾 Transformers saved to: `{TRANSFORMERS_DIR}/`")

        # Show Fourier features info if generated
        if fourier_info and fourier_info.get("total_features", 0) > 0:
            with st.expander("🌊 Fourier Features Generated", expanded=True):
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown("**Periods Used:**")
                    periods_str = ", ".join([f"{p:.1f} days" for p in fourier_info["periods_used"]])
                    st.code(periods_str)
                with fcol2:
                    st.markdown("**Feature Names:**")
                    feat_list = fourier_info.get("fourier_features", [])
                    if len(feat_list) > 6:
                        st.code("\n".join(feat_list[:6]) + f"\n... (+{len(feat_list)-6} more)")
                    else:
                        st.code("\n".join(feat_list))

        # Show Differencing info if applied (Step 3.1)
        if diff_result is not None:
            with st.expander("📈 Differencing Applied", expanded=True):
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    st.metric("Order", diff_config.get("order", 1))
                with dcol2:
                    st.metric("Columns", len(diff_config.get("target_columns", [])))
                with dcol3:
                    st.metric("Rows Dropped", diff_config.get("order", 1))

                st.markdown("**Differenced Columns:**")
                cols_str = ", ".join(diff_config.get("target_columns", [])[:5])
                if len(diff_config.get("target_columns", [])) > 5:
                    cols_str += f" (+{len(diff_config['target_columns'])-5} more)"
                st.code(cols_str)

                st.warning("""
                ⚠️ **Important:** Predictions will need inverse differencing.
                The original first values have been saved for reconstruction.
                """)

        result_tab1, result_tab2, result_tab3 = st.tabs(["Variant A (Decomp)", "Variant B (Cyclical)", "Split Details"])
        with result_tab1:
            st.dataframe(A_full.head(20), use_container_width=True)
            st.download_button("⬇️ Download Variant A", A_full.to_csv(index=False).encode(), "variant_A.csv", "text/csv")
        with result_tab2:
            st.dataframe(B_full.head(20), use_container_width=True)
            st.download_button("⬇️ Download Variant B", B_full.to_csv(index=False).encode(), "variant_B.csv", "text/csv")
        with result_tab3:
            st.markdown("#### Train/Validation/Test Split Summary")

            # Get date boundaries
            train_dates = base.iloc[train_idx][date_col]
            val_dates = base.iloc[cal_idx][date_col] if len(cal_idx) > 0 else pd.Series()
            test_dates = base.iloc[test_idx][date_col]

            split_summary = {
                "date_column": date_col,
                "train_start": str(train_dates.min()),
                "train_end": str(train_dates.max()),
                "train_records": len(train_idx),
                "train_ratio": f"{train_ratio:.0%}",
            }

            if len(cal_idx) > 0:
                split_summary.update({
                    "val_start": str(val_dates.min()),
                    "val_end": str(val_dates.max()),
                    "val_records": len(cal_idx),
                    "val_ratio": f"{val_ratio:.0%}",
                })

            split_summary.update({
                "test_start": str(test_dates.min()),
                "test_end": str(test_dates.max()),
                "test_records": len(test_idx),
                "test_ratio": f"{test_ratio:.0%}",
            })

            st.json(split_summary)

            st.markdown("---")
            st.markdown("#### Data Leakage Prevention")
            st.success("""
            ✅ **Scalers fitted on TRAIN data only** → Applied to validation and test

            ✅ **Encoders fitted on TRAIN data only** → Applied to validation and test

            ✅ **Validation set for early stopping** → Prevents overfitting during training

            ✅ **Test set untouched until final evaluation** → Unbiased performance estimate
            """)

# === ENTRYPOINT ===============================================================
def main():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    init_state()

    # Results Storage Panel (Supabase persistence)
    render_results_storage_panel(
        page_key="Feature Studio",
    )

    # Auto-load saved results if available
    auto_load_if_available("Feature Studio")

    # Fluorescent effects
    st.markdown("""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR ADVANCED FEATURE ENGINEERING
       ======================================== */

    @keyframes float-orb {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.25;
        }
        50% {
            transform: translate(30px, -30px) scale(1.05);
            opacity: 0.35;
        }
    }

    .fluorescent-orb {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(70px);
    }

    .orb-1 {
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
        top: 15%;
        right: 20%;
        animation: float-orb 25s ease-in-out infinite;
    }

    .orb-2 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
        bottom: 20%;
        left: 15%;
        animation: float-orb 30s ease-in-out infinite;
        animation-delay: 5s;
    }

    @keyframes sparkle {
        0%, 100% {
            opacity: 0;
            transform: scale(0);
        }
        50% {
            opacity: 0.6;
            transform: scale(1);
        }
    }

    .sparkle {
        position: fixed;
        width: 3px;
        height: 3px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
        border-radius: 50%;
        pointer-events: none;
        z-index: 2;
        animation: sparkle 3s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }

    .sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
    .sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
    .sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

    @media (max-width: 768px) {
        .fluorescent-orb {
            width: 200px !important;
            height: 200px !important;
            filter: blur(50px);
        }
        .sparkle {
            display: none;
        }
    }

    /* ========================================
       PREMIUM TAB STYLING - BLUE FLUORESCENT
       ======================================== */

    /* Tab container */
    .stTabs {
        background: transparent;
        margin-top: 1rem;
    }

    /* Tab list (container for all tab buttons) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(6, 78, 145, 0.4), rgba(15, 23, 42, 0.9));
        padding: 0.75rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.15),
            0 4px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    /* Individual tab buttons */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(30, 41, 59, 0.4));
        border-radius: 12px;
        padding: 0 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: rgba(148, 163, 184, 0.9);
        border: 1px solid rgba(59, 130, 246, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Tab button hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(34, 211, 238, 0.1));
        border-color: rgba(59, 130, 246, 0.4);
        color: #e2e8f0;
        transform: translateY(-2px);
        box-shadow:
            0 0 20px rgba(59, 130, 246, 0.3),
            0 4px 12px rgba(59, 130, 246, 0.2);
    }

    /* Active/selected tab */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(34, 211, 238, 0.2)) !important;
        border: 1px solid rgba(59, 130, 246, 0.6) !important;
        color: #f1f5f9 !important;
        box-shadow:
            0 0 25px rgba(59, 130, 246, 0.4),
            0 0 50px rgba(34, 211, 238, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }

    /* Active tab indicator (underline) - hide it */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }

    /* Tab panels (content area) */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
        background: transparent;
    }

    /* Mobile responsive tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
            padding: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            width: 100%;
            justify-content: center;
        }
    }
    </style>

    <!-- Fluorescent Floating Orbs -->
    <div class="fluorescent-orb orb-1"></div>
    <div class="fluorescent-orb orb-2"></div>

    <!-- Sparkle Particles -->
    <div class="sparkle sparkle-1"></div>
    <div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div>
    """, unsafe_allow_html=True)

    # Premium Hero Header
    render_scifi_hero_header(
        title="Feature Studio",
        subtitle="Create temporal and cyclical features. Build predictive signals with lag features and rolling statistics.",
        status="SYSTEM ONLINE"
    )
    page_feature_engineering()

    # =============================================================================
    # PAGE NAVIGATION
    # =============================================================================
    render_page_navigation(5)  # Feature Studio is page index 5

if __name__ == "__main__":
    main()
