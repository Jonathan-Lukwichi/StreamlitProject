# =============================================================================
# pages/02_Data_Studio.py - Unified Data Studio with Stepper UI
# Consolidates: Upload Data, Prepare Data, and Feature Studio
# Academic Best Practice: Guided workflow with data leakage prevention
# =============================================================================
from __future__ import annotations

import re
import os
import inspect
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css, is_quiet_mode
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, CARD_BG, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand, render_cache_management, render_user_preferences
from app_core.ui.page_navigation import render_page_navigation
from app_core.cache import save_to_cache, get_cache_manager
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# Optional imports with fallbacks
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

# =============================================================================
# AUTHENTICATION
# =============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================
try:
    from app_core.api.config_manager import APIConfigManager
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    APIConfigManager = None

try:
    from app_core.data.hospital_service import HospitalDataService, get_hospital_service
    HOSPITAL_SERVICE_AVAILABLE = True
except ImportError:
    HOSPITAL_SERVICE_AVAILABLE = False
    HospitalDataService = None

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        for k, v in {
            "patient_loaded": False, "weather_loaded": False,
            "calendar_loaded": False, "patient_data": None,
            "weather_data": None, "calendar_data": None,
            "reason_data": None, "merged_data": None,
            "processed_df": None, "feature_engineering": {},
            "_nav_intent": None, "data_studio_step": 1,
        }.items():
            st.session_state.setdefault(k, v)

try:
    from app_core.data.fusion import fuse_data
except Exception:
    def fuse_data(patient_df, weather_df, calendar_df, reason_df=None):
        def _key(df):
            for c in ["datetime", "Date", "date", "timestamp", "ds", "time"]:
                if c in df.columns: return c
            return None
        p, w, c = patient_df.copy(), weather_df.copy(), calendar_df.copy()
        r = reason_df.copy() if reason_df is not None else None
        for df in (p, w, c, r):
            if df is not None:
                for k in ["datetime","Date","date","timestamp","ds","time"]:
                    if k in df.columns:
                        df[k] = pd.to_datetime(df[k], errors="coerce")
        pk, wk, ck = _key(p), _key(w), _key(c)
        rk = _key(r) if r is not None else None
        merged = p
        if w is not None and wk: merged = merged.merge(w, left_on=pk, right_on=wk, how="outer")
        if c is not None and ck: merged = merged.merge(c, left_on=pk or wk, right_on=ck, how="outer")
        if r is not None and rk: merged = merged.merge(r, left_on=pk or wk or ck, right_on=rk, how="outer")
        if "Total_Arrivals" in merged.columns and "Target_1" in merged.columns:
            if merged["Total_Arrivals"].equals(merged["Target_1"]):
                merged = merged.drop(columns=["Total_Arrivals"])
        return merged, {}

try:
    from app_core.data import process_dataset
except Exception:
    def process_dataset(df, n_lags, date_col=None, ed_col=None, strict_drop_na_edges=True, use_aggregated_categories=True):
        work = df.copy()
        y = None
        for c in work.columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                y = c; break
        if y:
            for i in range(1, n_lags+1):
                work[f"ED_{i}"] = work[y].shift(i)
                work[f"Target_{i}"] = work[y].shift(-i)
        clinical_categories = {
            "RESPIRATORY": ["asthma", "pneumonia", "shortness_of_breath"],
            "CARDIAC": ["chest_pain", "arrhythmia", "hypertensive_emergency"],
            "TRAUMA": ["fracture", "laceration", "burn", "fall_injury"],
            "GASTROINTESTINAL": ["abdominal_pain", "vomiting", "diarrhea"],
            "INFECTIOUS": ["flu_symptoms", "fever", "viral_infection"],
            "NEUROLOGICAL": ["headache", "dizziness"],
            "OTHER": ["allergic_reaction", "mental_health"],
        }
        col_map = {c.lower(): c for c in work.columns}
        if use_aggregated_categories:
            for category, reasons in clinical_categories.items():
                matching_cols = [col_map[r] for r in reasons if r in col_map]
                if matching_cols:
                    work[category] = work[matching_cols].fillna(0).sum(axis=1)
            category_cols = list(clinical_categories.keys())
            for col in category_cols:
                if col in work.columns:
                    for i in range(1, n_lags+1):
                        work[f"{col}_{i}"] = work[col].shift(-i)
        if strict_drop_na_edges:
            work = work.dropna(how="any")
        return work, {}

try:
    from app_core.pipelines import (
        compute_temporal_split, validate_temporal_split, TemporalSplitResult,
        CVFoldResult, CVConfig, create_cv_folds, visualize_cv_folds,
        evaluate_cv_metrics, format_cv_metric,
    )
    TEMPORAL_SPLIT_AVAILABLE = True
    CV_AVAILABLE = True
except ImportError:
    TEMPORAL_SPLIT_AVAILABLE = False
    CV_AVAILABLE = False

try:
    from app_core.data.ingest import _upload_generic
except Exception:
    def _upload_generic(title: str, key_prefix: str, desired_datetime_name: str):
        up = st.file_uploader(f"{title} (CSV)", type=["csv"], key=f"{key_prefix}_upload")
        if up is not None:
            try:
                df = pd.read_csv(up)
                if desired_datetime_name == "datetime":
                    for c in ["datetime", "timestamp", "Date", "date", "time", "ds"]:
                        if c in df.columns:
                            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
                            break
                else:
                    for c in ["date", "Date", "datetime", "ds"]:
                        if c in df.columns:
                            df["date"] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
                            break
                st.session_state[f"{key_prefix}_data"] = df
                st.session_state[f"{key_prefix}_loaded"] = True
                st.success(f"Loaded {title}: {df.shape[0]:,} rows")
            except Exception as e:
                st.error(f"Failed to load {title}: {e}")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Data Studio - HealthForecast AI",
    page_icon="🔬",
    layout="wide",
)

# =============================================================================
# STEPPER UI COMPONENT
# =============================================================================
def render_stepper(current_step: int, steps: List[Dict[str, str]]) -> int:
    """
    Render a visual stepper component for guided workflow.

    Args:
        current_step: Current step index (1-based)
        steps: List of dicts with 'title', 'icon', 'description'

    Returns:
        Selected step index (1-based)
    """
    n_steps = len(steps)

    # Stepper CSS
    st.markdown(f"""
    <style>
    .stepper-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 1.5rem 0 2rem 0;
        padding: 1rem;
    }}
    .step-item {{
        display: flex;
        flex-direction: column;
        align-items: center;
        cursor: pointer;
        transition: all 0.3s ease;
        padding: 1rem;
        border-radius: 16px;
        min-width: 140px;
        position: relative;
    }}
    .step-item:hover {{
        background: rgba(59, 130, 246, 0.1);
    }}
    .step-circle {{
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }}
    .step-circle.active {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
    }}
    .step-circle.completed {{
        background: {SUCCESS_COLOR};
        border-color: {SUCCESS_COLOR};
    }}
    .step-circle.pending {{
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
    }}
    .step-title {{
        font-weight: 600;
        font-size: 0.9375rem;
        color: {TEXT_COLOR};
        margin-bottom: 0.25rem;
        text-align: center;
    }}
    .step-title.active {{
        color: {PRIMARY_COLOR};
    }}
    .step-description {{
        font-size: 0.75rem;
        color: {SUBTLE_TEXT};
        text-align: center;
        max-width: 120px;
    }}
    .step-connector {{
        width: 80px;
        height: 3px;
        background: rgba(255, 255, 255, 0.1);
        margin: 0 -10px;
        position: relative;
        top: -40px;
    }}
    .step-connector.completed {{
        background: linear-gradient(90deg, {SUCCESS_COLOR}, {PRIMARY_COLOR});
    }}
    .step-connector.active {{
        background: linear-gradient(90deg, {SUCCESS_COLOR}, {PRIMARY_COLOR});
        animation: pulse-connector 2s infinite;
    }}
    @keyframes pulse-connector {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # Render stepper HTML
    stepper_html = '<div class="stepper-container">'

    for i, step in enumerate(steps, 1):
        # Determine step state
        if i < current_step:
            circle_class = "completed"
            title_class = ""
            circle_content = "✓"
        elif i == current_step:
            circle_class = "active"
            title_class = "active"
            circle_content = step['icon']
        else:
            circle_class = "pending"
            title_class = ""
            circle_content = step['icon']

        stepper_html += f"""
        <div class="step-item" onclick="document.getElementById('step_btn_{i}').click()">
            <div class="step-circle {circle_class}">{circle_content}</div>
            <div class="step-title {title_class}">{step['title']}</div>
            <div class="step-description">{step['description']}</div>
        </div>
        """

        # Add connector between steps
        if i < n_steps:
            connector_class = "completed" if i < current_step else ("active" if i == current_step else "")
            stepper_html += f'<div class="step-connector {connector_class}"></div>'

    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)

    # Hidden buttons for step navigation (clicked via JavaScript)
    cols = st.columns(n_steps)
    selected_step = current_step
    for i, col in enumerate(cols, 1):
        with col:
            if st.button(f"Step {i}", key=f"step_btn_{i}", use_container_width=True,
                        type="primary" if i == current_step else "secondary"):
                selected_step = i

    return selected_step

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _clean_datetime_column(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    df = df.copy()
    dt_col = None
    for col in ["datetime", "Date", "date", "timestamp", "ds"]:
        if col in df.columns:
            dt_col = col
            break
    if dt_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_col = col
                break
    if dt_col is not None:
        dt_series = pd.to_datetime(df[dt_col])
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_localize(None)
        df['datetime'] = dt_series
        if dt_col != 'datetime' and dt_col in df.columns:
            df = df.drop(columns=[dt_col])
    return df

def _remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    empty_cols = []
    for col in df.columns:
        if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            empty_cols.append(col)
    if empty_cols:
        df = df.drop(columns=empty_cols)
    return df

def _date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "date", "Date", "timestamp", "ds", "time"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def _find_time_like_col(df: pd.DataFrame) -> Optional[str]:
    priority = ["datetime", "timestamp", "date_time", "date", "ds"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_map: return lower_map[p]
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "stamp" in cl: return c
    return None

def _data_health(df: Optional[pd.DataFrame]):
    if isinstance(df, pd.DataFrame) and not df.empty:
        dcol = _date_col(df)
        if dcol is not None:
            dt = pd.to_datetime(df[dcol], errors="coerce")
            dr = f"{dt.min().date()} to {dt.max().date()}" if dt.notna().any() else "N/A"
        else:
            dr = "N/A"
        st.write(pd.DataFrame({
            "Indicator": ["Rows", "Columns", "Date Range", "Missing Values"],
            "Value": [f"{len(df):,}", f"{df.shape[1]}", dr, f"{int(df.isna().sum().sum()):,}"]
        }))
    else:
        st.caption("No data loaded yet.")

def _fetch_from_api(dataset_type: str, start_date: datetime, end_date: datetime, provider: str = "supabase"):
    if not API_AVAILABLE:
        st.error("API functionality not available.")
        return None
    try:
        config_manager = APIConfigManager()
        if dataset_type == "patient":
            connector = config_manager.get_patient_connector(provider)
            session_key = "patient_data"
        elif dataset_type == "weather":
            connector = config_manager.get_weather_connector(provider)
            session_key = "weather_data"
        elif dataset_type == "calendar":
            connector = config_manager.get_calendar_connector(provider)
            session_key = "calendar_data"
        elif dataset_type == "reason":
            connector = config_manager.get_reason_connector(provider)
            session_key = "reason_data"
        else:
            st.error(f"Unknown dataset type: {dataset_type}")
            return None
        with st.spinner(f"Fetching {dataset_type} data..."):
            df = connector.fetch_data(start_date, end_date)
        if df is not None and not df.empty:
            df = _clean_datetime_column(df, dataset_type)
            df = _remove_empty_columns(df)
            if dataset_type == "reason":
                cols_to_drop = [col for col in df.columns if col.lower() == "total_arrivals"]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
            st.session_state[session_key] = df
            st.session_state[f"{dataset_type}_loaded"] = True
            st.success(f"Fetched {len(df):,} rows")
            return df
        else:
            st.warning("No data found for selected date range")
            return None
    except Exception as e:
        st.error(f"API fetch failed: {str(e)}")
        return None

# =============================================================================
# STEP 1: UPLOAD DATA
# =============================================================================
def render_step_upload():
    """Render Step 1: Upload Data section."""
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 2rem;
         background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
         border: 2px solid rgba(59, 130, 246, 0.3);'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>1</div>
      <h2 style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Upload Data</h2>
      <p style='color: {BODY_TEXT}; max-width: 600px; margin: 0 auto;'>
        Load patient arrivals, weather, calendar, and clinical visit data from files or hospital database
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Hospital Selector
    st.markdown("### Select Hospital")
    available_hospitals = ["Pamplona Spain Hospital"]
    if HOSPITAL_SERVICE_AVAILABLE:
        try:
            hospital_service = get_hospital_service()
            if hospital_service.is_connected():
                available_hospitals = hospital_service.get_available_hospitals()
        except Exception:
            pass

    col_hosp1, col_hosp2, col_hosp3 = st.columns([1, 2, 1])
    with col_hosp2:
        selected_hospital = st.selectbox(
            "Hospital", options=available_hospitals, index=0, key="hospital_selector"
        )
        st.session_state["selected_hospital"] = selected_hospital

        if HOSPITAL_SERVICE_AVAILABLE and API_AVAILABLE:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                hosp_start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="hosp_start")
            with col_date2:
                hosp_end = st.date_input("End Date", value=datetime(2023, 12, 31), key="hosp_end")

            if st.button(f"Load All Data for {selected_hospital}", use_container_width=True, type="primary"):
                start_dt = datetime.combine(hosp_start, datetime.min.time())
                end_dt = datetime.combine(hosp_end, datetime.min.time())
                hospital_service = get_hospital_service()
                progress_bar = st.progress(0, text="Fetching datasets...")
                datasets = ["patient", "weather", "calendar", "reason"]
                success_count = 0
                for i, dataset_type in enumerate(datasets):
                    progress_bar.progress((i) / 4, text=f"Fetching {dataset_type} data...")
                    df = hospital_service.fetch_dataset_by_hospital(
                        dataset_type=dataset_type, hospital_name=selected_hospital,
                        start_date=start_dt, end_date=end_dt,
                    )
                    if df is not None and not df.empty:
                        df = _clean_datetime_column(df, dataset_type)
                        df = _remove_empty_columns(df)
                        if dataset_type == "reason":
                            cols_to_drop = [col for col in df.columns if col.lower() == "total_arrivals"]
                            if cols_to_drop:
                                df = df.drop(columns=cols_to_drop)
                        st.session_state[f"{dataset_type}_data"] = df
                        st.session_state[f"{dataset_type}_loaded"] = True
                        success_count += 1
                        st.success(f"{dataset_type.title()}: {len(df):,} rows")
                    else:
                        st.warning(f"No {dataset_type} data found")
                progress_bar.progress(1.0, text="Complete!")
                if success_count == 4:
                    st.success(f"Successfully loaded all data for **{selected_hospital}**!")
                    st.balloons()

    st.markdown("---")

    # Data Source Cards
    st.markdown("### Data Sources")
    col1, col2, col3, col4 = st.columns(4)

    datasets_config = [
        ("patient", "Patient Data", "Historical arrivals", col1),
        ("weather", "Weather Data", "Meteorological conditions", col2),
        ("calendar", "Calendar Data", "Holidays & events", col3),
        ("reason", "Reason for Visit", "Clinical categories", col4),
    ]

    for ds_type, title, desc, col in datasets_config:
        with col:
            df = st.session_state.get(f"{ds_type}_data")
            is_loaded = df is not None and not getattr(df, "empty", True)

            status_color = SUCCESS_COLOR if is_loaded else WARNING_COLOR
            status_icon = "check_circle" if is_loaded else "hourglass_empty"

            st.markdown(f"""
            <div class='hf-feature-card' style='padding: 1rem; text-align: center;'>
                <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>
                    {'✅' if is_loaded else '⏳'}
                </div>
                <div style='font-weight: 600; margin-bottom: 0.25rem;'>{title}</div>
                <div style='font-size: 0.75rem; color: {SUBTLE_TEXT};'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["Upload", "API"])
            with tab1:
                _upload_generic(f"Upload {title}", ds_type, "datetime")
            with tab2:
                if API_AVAILABLE:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        start = st.date_input("Start", value=datetime(2018, 1, 1), key=f"{ds_type}_start")
                    with col_b:
                        end = st.date_input("End", value=datetime.now(), key=f"{ds_type}_end")
                    if st.button(f"Fetch", key=f"fetch_{ds_type}", use_container_width=True):
                        _fetch_from_api(ds_type, datetime.combine(start, datetime.min.time()),
                                       datetime.combine(end, datetime.min.time()), "supabase")
                else:
                    st.caption("API not available")

            if is_loaded:
                st.metric("Rows", f"{len(df):,}")

    # Status Summary
    all_loaded = all([
        st.session_state.get("patient_data") is not None,
        st.session_state.get("weather_data") is not None,
        st.session_state.get("calendar_data") is not None,
        st.session_state.get("reason_data") is not None,
    ])

    st.markdown("---")
    if all_loaded:
        st.success("All datasets loaded! Click **Step 2** above to continue with data fusion.")
        if st.button("Continue to Step 2: Fuse & Prepare", type="primary", use_container_width=True):
            st.session_state["data_studio_step"] = 2
            st.rerun()
    else:
        st.warning("Please load all four datasets to continue.")

# =============================================================================
# STEP 2: FUSE & PREPARE DATA
# =============================================================================
def render_step_prepare():
    """Render Step 2: Fuse & Prepare Data section."""
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 2rem;
         background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
         border: 2px solid rgba(34, 197, 94, 0.3);'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>2</div>
      <h2 style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Fuse & Prepare</h2>
      <p style='color: {BODY_TEXT}; max-width: 600px; margin: 0 auto;'>
        Merge datasets, generate lag features, and create clinical category targets
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check prerequisites
    patient_df = st.session_state.get("patient_data")
    weather_df = st.session_state.get("weather_data")
    calendar_df = st.session_state.get("calendar_data")
    reason_df = st.session_state.get("reason_data")

    all_ready = all([
        patient_df is not None and not getattr(patient_df, "empty", True),
        weather_df is not None and not getattr(weather_df, "empty", True),
        calendar_df is not None and not getattr(calendar_df, "empty", True),
        reason_df is not None and not getattr(reason_df, "empty", True),
    ])

    if not all_ready:
        st.warning("Please complete Step 1 first - load all four datasets.")
        if st.button("Go back to Step 1", use_container_width=True):
            st.session_state["data_studio_step"] = 1
            st.rerun()
        return

    # Fusion and Preprocessing tabs
    tab_merge, tab_preprocess, tab_preview = st.tabs([
        "Merge Datasets", "Feature Engineering", "Preview Results"
    ])

    with tab_merge:
        st.markdown("### Data Fusion")
        st.info("Merge Patient, Weather, Calendar, and Reason datasets on time keys.")

        # Show detected keys
        st.markdown("**Detected Time Keys:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.code(_find_time_like_col(patient_df) or "—")
            st.caption("Patient")
        with col2:
            st.code(_find_time_like_col(weather_df) or "—")
            st.caption("Weather")
        with col3:
            st.code(_find_time_like_col(calendar_df) or "—")
            st.caption("Calendar")
        with col4:
            st.code(_find_time_like_col(reason_df) or "—")
            st.caption("Reason")

        if st.button("Merge Datasets", type="primary", use_container_width=True):
            with st.spinner("Merging datasets..."):
                try:
                    merged, log = fuse_data(patient_df, weather_df, calendar_df, reason_df)
                    if merged is not None and not merged.empty:
                        st.session_state["merged_data"] = merged
                        st.success(f"Successfully merged {len(merged):,} rows!")
                        st.dataframe(merged.head(10), use_container_width=True)
                    else:
                        st.error("Merge returned empty result.")
                except Exception as e:
                    st.error(f"Fusion failed: {e}")

    with tab_preprocess:
        st.markdown("### Feature Engineering Configuration")

        merged_df = st.session_state.get("merged_data")
        if merged_df is None or getattr(merged_df, "empty", True):
            st.warning("Please merge datasets first in the 'Merge Datasets' tab.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            n_lags = st.number_input("Lags/Targets (days)", min_value=1, max_value=30, value=7)
        with col2:
            strict_drop = st.checkbox("Drop edge rows", value=True)
        with col3:
            use_aggregated = st.checkbox("Aggregated Categories", value=True)

        # Temporal features config
        with st.expander("Advanced Temporal Features", expanded=False):
            enable_temporal = st.checkbox("Enable lag & rolling features", value=True)
            if enable_temporal:
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    max_lag_days = st.slider("Max Lag Days", 1, 21, 14)
                    rolling_window = st.slider("Rolling Window", 3, 21, 7)
                with col_t2:
                    roll_mean = st.checkbox("Rolling Mean", value=True)
                    roll_std = st.checkbox("Rolling Std", value=True)
                    roll_min = st.checkbox("Rolling Min", value=True)
                    roll_max = st.checkbox("Rolling Max", value=True)
                include_stats = []
                if roll_mean: include_stats.append('mean')
                if roll_std: include_stats.append('std')
                if roll_min: include_stats.append('min')
                if roll_max: include_stats.append('max')

        if st.button("Build Features", type="primary", use_container_width=True):
            with st.spinner("Engineering features..."):
                try:
                    processed, _ = process_dataset(
                        merged_df, n_lags=n_lags, date_col=None, ed_col=None,
                        strict_drop_na_edges=strict_drop, use_aggregated_categories=use_aggregated
                    )

                    # Apply temporal features if enabled
                    if enable_temporal:
                        # Find target column
                        target = None
                        for col in processed.columns:
                            if col.lower() in ['total_arrivals', 'ed', 'arrivals', 'patients']:
                                target = col
                                break
                        if target is None:
                            for col in processed.columns:
                                if pd.api.types.is_numeric_dtype(processed[col]) and not col.startswith('Target_'):
                                    target = col
                                    break

                        if target:
                            # Generate lag features
                            for lag in range(1, max_lag_days + 1):
                                processed[f"ED_lag_{lag}"] = processed[target].shift(lag)

                            # Generate rolling features
                            series = processed[target]
                            if 'mean' in include_stats:
                                processed[f"roll_{rolling_window}_mean"] = series.rolling(window=rolling_window, min_periods=1).mean()
                            if 'std' in include_stats:
                                processed[f"roll_{rolling_window}_std"] = series.rolling(window=rolling_window, min_periods=2).std()
                            if 'min' in include_stats:
                                processed[f"roll_{rolling_window}_min"] = series.rolling(window=rolling_window, min_periods=1).min()
                            if 'max' in include_stats:
                                processed[f"roll_{rolling_window}_max"] = series.rolling(window=rolling_window, min_periods=1).max()

                            if strict_drop:
                                processed = processed.dropna(how="any")

                    st.session_state["processed_df"] = processed
                    st.success(f"Features built! {len(processed.columns)} columns, {len(processed):,} rows")

                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

    with tab_preview:
        st.markdown("### Processed Dataset Preview")
        processed = st.session_state.get("processed_df")
        if processed is not None and not getattr(processed, "empty", True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(processed):,}")
            with col2:
                st.metric("Columns", f"{len(processed.columns)}")
            with col3:
                targets = [c for c in processed.columns if c.lower().startswith("target_")]
                st.metric("Targets", len(targets))

            st.dataframe(processed.head(20), use_container_width=True)

            st.success("Data prepared! Click **Step 3** above to configure train/test split and generate final dataset.")
            if st.button("Continue to Step 3: Engineer Features", type="primary", use_container_width=True):
                st.session_state["data_studio_step"] = 3
                st.rerun()
        else:
            st.info("No processed data yet. Please merge and build features first.")

# =============================================================================
# STEP 3: FEATURE ENGINEERING (Train/Test Split, CV, Fourier, etc.)
# =============================================================================
def render_step_features():
    """Render Step 3: Feature Engineering section."""
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 2rem;
         background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
         border: 2px solid rgba(168, 85, 247, 0.3);'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>3</div>
      <h2 style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Engineer Features</h2>
      <p style='color: {BODY_TEXT}; max-width: 600px; margin: 0 auto;'>
        Configure train/test split, cross-validation, Fourier features, and generate final dataset
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check prerequisites
    processed = st.session_state.get("processed_df")
    if processed is None or getattr(processed, "empty", True):
        st.warning("Please complete Step 2 first - prepare and preprocess data.")
        if st.button("Go back to Step 2", use_container_width=True):
            st.session_state["data_studio_step"] = 2
            st.rerun()
        return

    # Find date column and targets
    date_col = _find_time_like_col(processed)
    targets = [c for c in processed.columns if c.lower().startswith("target_")]

    # Ensure datetime sorted
    if date_col:
        processed[date_col] = pd.to_datetime(processed[date_col], errors="coerce")
        processed = processed.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Feature Engineering Tabs
    tab_split, tab_cv, tab_fourier, tab_generate = st.tabs([
        "Train/Test Split", "Cross-Validation", "Fourier Features", "Generate Dataset"
    ])

    with tab_split:
        st.markdown("### Configure Train/Test Split")
        st.info("**Data Leakage Prevention:** Scalers fit on training data only.")

        train_ratio = st.slider("Training Set Ratio", 0.60, 0.90, 0.80, 0.05)
        test_ratio = 1.0 - train_ratio

        if date_col:
            min_date = processed[date_col].min()
            max_date = processed[date_col].max()
            total_days = (max_date - min_date).days
            train_end = min_date + timedelta(days=int(total_days * train_ratio))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train", f"{train_ratio:.0%}", f"{min_date.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            with col2:
                st.metric("Test", f"{test_ratio:.0%}", f"{train_end.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        # Store split config
        st.session_state["split_config"] = {"train_ratio": train_ratio, "test_ratio": test_ratio}

    with tab_cv:
        st.markdown("### Time Series Cross-Validation")
        st.info("**Why Time Series CV?** Standard k-fold shuffles data, causing leakage. TS-CV uses expanding windows.")

        if CV_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                n_splits = st.slider("Number of Folds", 3, 10, 5)
                gap_days = st.slider("Gap (days)", 0, 14, 0)
            with col2:
                use_cv = st.checkbox("Enable CV for Training", value=True)
                min_train_pct = st.slider("Min Train Size (%)", 10, 50, 20, 5)

            cv_config = CVConfig(
                n_splits=n_splits, gap=gap_days,
                min_train_size=int(len(processed) * min_train_pct / 100) if min_train_pct > 0 else None,
                test_size=None
            )
            st.session_state["cv_config"] = cv_config
            st.session_state["cv_enabled"] = use_cv

            if st.button("Preview CV Folds", use_container_width=True):
                try:
                    cv_folds = create_cv_folds(
                        df=processed, date_col=date_col, n_splits=n_splits,
                        gap=gap_days, min_train_size=cv_config.min_train_size, verbose=False
                    )
                    fold_data = [{
                        "Fold": f.fold_idx + 1,
                        "Train": f"{f.train_start.strftime('%Y-%m-%d')} to {f.train_end.strftime('%Y-%m-%d')}",
                        "Train N": f.n_train,
                        "Val": f"{f.val_start.strftime('%Y-%m-%d')} to {f.val_end.strftime('%Y-%m-%d')}",
                        "Val N": f.n_val,
                    } for f in cv_folds]
                    st.dataframe(pd.DataFrame(fold_data), use_container_width=True, hide_index=True)
                    st.session_state["cv_folds"] = cv_folds
                    st.success(f"{len(cv_folds)} CV folds configured!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("CV module not available.")

    with tab_fourier:
        st.markdown("### Fourier Features")
        st.info("Fourier terms capture periodic patterns with sine/cosine pairs.")

        fft_data = st.session_state.get("fft_dominant_cycles", {})
        if fft_data:
            st.success("FFT cycles detected from EDA page!")
            cycles_list = fft_data.get("cycles_df", [])
            if cycles_list:
                st.dataframe(pd.DataFrame(cycles_list), hide_index=True)

        enable_fourier = st.checkbox("Enable Fourier Features", value=False)
        if enable_fourier:
            col1, col2 = st.columns(2)
            with col1:
                periods = st.multiselect(
                    "Periods (days)",
                    options=[7.0, 14.0, 30.44, 91.31, 365.25],
                    default=[7.0],
                    format_func=lambda x: f"{x:.1f} days"
                )
            with col2:
                n_harmonics = st.slider("Harmonics per Period", 1, 5, 1)

            st.session_state["fourier_config"] = {
                "enabled": True, "periods": periods, "n_harmonics": n_harmonics
            }
        else:
            st.session_state["fourier_config"] = {"enabled": False}

    with tab_generate:
        st.markdown("### Generate Final Dataset")

        col1, col2 = st.columns(2)
        with col1:
            apply_scaling = st.checkbox("Apply Min-Max Scaling", value=True)
        with col2:
            skip_engineering = st.checkbox("Skip Feature Engineering", value=False)

        if st.button("Generate Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating dataset..."):
                try:
                    # Get split config
                    split_config = st.session_state.get("split_config", {"train_ratio": 0.8})
                    train_ratio = split_config["train_ratio"]

                    # Compute indices
                    n = len(processed)
                    n_train = int(np.floor(n * train_ratio))
                    train_idx = np.arange(n_train)
                    test_idx = np.arange(n_train, n)

                    # Apply Fourier features if enabled
                    result_df = processed.copy()
                    fourier_config = st.session_state.get("fourier_config", {})
                    fourier_info = None

                    if fourier_config.get("enabled") and date_col:
                        periods = fourier_config.get("periods", [])
                        n_harm = fourier_config.get("n_harmonics", 1)
                        dates = pd.to_datetime(result_df[date_col])
                        t = (dates - dates.min()).dt.days.values.astype(float)

                        fourier_features = []
                        for period in periods:
                            for k in range(1, n_harm + 1):
                                sin_col = f"fourier_sin_p{period:.0f}_k{k}"
                                cos_col = f"fourier_cos_p{period:.0f}_k{k}"
                                result_df[sin_col] = np.sin(2 * np.pi * k * t / period)
                                result_df[cos_col] = np.cos(2 * np.pi * k * t / period)
                                fourier_features.extend([sin_col, cos_col])

                        fourier_info = {"features": fourier_features, "n_features": len(fourier_features)}

                    # Apply scaling if enabled
                    if apply_scaling and not skip_engineering and MinMaxScaler is not None:
                        exclude = set([date_col] + targets + [c for c in result_df.columns if c.startswith("ED_")])
                        num_cols = result_df.select_dtypes(include=[np.number]).columns
                        scale_cols = [c for c in num_cols if c not in exclude]
                        if scale_cols:
                            scaler = MinMaxScaler().fit(result_df.loc[train_idx, scale_cols])
                            result_df.loc[:, scale_cols] = scaler.transform(result_df.loc[:, scale_cols])

                    # Store results
                    st.session_state["feature_engineering"] = {
                        "summary": {
                            "train_size": len(train_idx),
                            "test_size": len(test_idx),
                            "train_ratio": train_ratio,
                            "apply_scaling": apply_scaling,
                            "fourier_enabled": fourier_info is not None,
                        },
                        "A": result_df.copy(),
                        "B": result_df.copy(),
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "cal_idx": np.array([], dtype=int),
                        "fourier_info": fourier_info,
                    }

                    st.success(f"Dataset generated! {len(result_df.columns)} features, {len(train_idx):,} train / {len(test_idx):,} test rows")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train", f"{len(train_idx):,}", f"{train_ratio:.0%}")
                    with col2:
                        st.metric("Test", f"{len(test_idx):,}", f"{1-train_ratio:.0%}")
                    with col3:
                        st.metric("Features", len(result_df.columns))

                    st.dataframe(result_df.head(10), use_container_width=True)

                    # Download button
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Dataset", csv, "engineered_dataset.csv", "text/csv", use_container_width=True)

                    st.success("Data Studio complete! Proceed to **Explore Data** or **Train Models**.")

                except Exception as e:
                    st.error(f"Generation failed: {e}")

# =============================================================================
# MAIN PAGE FUNCTION
# =============================================================================
def page_data_studio():
    # Apply theme and sidebar
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    add_logout_button()
    render_user_preferences()
    render_cache_management()

    # Results Storage Panel
    render_results_storage_panel(
        page_type="data_studio",
        page_title="Data Studio",
        custom_keys=["patient_data", "weather_data", "calendar_data", "reason_data", "processed_df", "feature_engineering"],
    )
    auto_load_if_available("data_studio")

    # Fluorescent effects (respect quiet mode)
    if not is_quiet_mode():
        st.markdown("""
        <style>
        @keyframes float-orb {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.25; }
            50% { transform: translate(30px, -30px) scale(1.05); opacity: 0.35; }
        }
        .fluorescent-orb {
            position: fixed; border-radius: 50%; pointer-events: none;
            z-index: 0; filter: blur(70px);
        }
        .orb-1 {
            width: 350px; height: 350px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
            top: 15%; right: 20%; animation: float-orb 25s ease-in-out infinite;
        }
        .orb-2 {
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
            bottom: 20%; left: 15%; animation: float-orb 30s ease-in-out infinite;
            animation-delay: 5s;
        }
        </style>
        <div class="fluorescent-orb orb-1"></div>
        <div class="fluorescent-orb orb-2"></div>
        """, unsafe_allow_html=True)

    # Hero Header
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem;'>
      <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>&#128300;</div>
      <h1 style='font-size: 2rem; margin-bottom: 0.5rem;'>Data Studio</h1>
      <p style='color: {BODY_TEXT}; max-width: 700px; margin: 0 auto;'>
        Complete data pipeline: Upload, Fuse, Prepare, and Engineer features for ML training
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize step in session state
    if "data_studio_step" not in st.session_state:
        st.session_state["data_studio_step"] = 1

    # Define steps
    steps = [
        {"title": "Upload", "icon": "1", "description": "Load data sources"},
        {"title": "Fuse & Prepare", "icon": "2", "description": "Merge & preprocess"},
        {"title": "Engineer", "icon": "3", "description": "Split & transform"},
    ]

    # Render stepper
    current_step = st.session_state["data_studio_step"]
    selected_step = render_stepper(current_step, steps)

    # Update step if changed
    if selected_step != current_step:
        st.session_state["data_studio_step"] = selected_step
        st.rerun()

    # Render current step content
    st.markdown("---")

    if current_step == 1:
        render_step_upload()
    elif current_step == 2:
        render_step_prepare()
    elif current_step == 3:
        render_step_features()

# =============================================================================
# ENTRYPOINT
# =============================================================================
init_state()
page_data_studio()
render_page_navigation(1)  # Data Studio is page index 1
