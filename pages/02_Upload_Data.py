# =============================================================================
# pages/02_Data_Hub.py — Data Integration Hub (Premium Design)
# Upload and manage patient, weather, and calendar data with premium UI.
# =============================================================================
from __future__ import annotations

import os
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.components import render_scifi_hero_header
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

# Import API components
try:
    from app_core.api.config_manager import APIConfigManager
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    APIConfigManager = None

# Import Hospital Data Service
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
            "patient_loaded": False,
            "weather_loaded": False,
            "calendar_loaded": False,
            "patient_data": None,
            "weather_data": None,
            "calendar_data": None,
            "merged_data": None,
            "_nav_intent": None,
        }.items():
            st.session_state.setdefault(k, v)

# Optional UI helpers
try:
    from app_core.ui.components import header
except Exception:
    def header(title: str, subtitle: str = "", icon: str = ""):
        st.markdown(f"### {icon} {title}")
        if subtitle:
            st.caption(subtitle)

# Ingest helper provided by your core
try:
    from app_core.data.ingest import _upload_generic
except Exception:
    def _upload_generic(title: str, key_prefix: str, desired_datetime_name: str):
        up = st.file_uploader(f"{title} (CSV)", type=["csv"], key=f"{key_prefix}_upload")
        if up is not None:
            try:
                df = pd.read_csv(up)
                if desired_datetime_name == "datetime":
                    # best-effort parse common datetime names
                    for c in ["datetime", "timestamp", "Date", "date", "time", "ds"]:
                        if c in df.columns:
                            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
                            break
                else:
                    # date-only normalization
                    for c in ["date", "Date", "datetime", "ds"]:
                        if c in df.columns:
                            df["date"] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
                            break
                st.session_state[f"{key_prefix}_data"] = df
                st.session_state[f"{key_prefix}_loaded"] = True
                if key_prefix == "patient":  st.session_state["patient_data"] = df
                if key_prefix == "weather":  st.session_state["weather_data"] = df
                if key_prefix == "calendar": st.session_state["calendar_data"] = df
                st.success(f"Loaded {title}: {df.shape[0]:,} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Failed to load {title}: {e}")


def _clean_datetime_column(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Clean datetime columns to ensure timezone-naive datetime64[ns].
    Standardizes all datasets to have a 'datetime' column.
    """
    df = df.copy()

    # Find the datetime column
    dt_col = None
    for col in ["datetime", "Date", "date", "timestamp", "ds"]:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None:
        # Look for any datetime-like column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_col = col
                break

    if dt_col is not None:
        # Convert to datetime and strip timezone
        dt_series = pd.to_datetime(df[dt_col])

        # Remove timezone if present by converting to string and back
        if dt_series.dt.tz is not None:
            # Strip timezone
            dt_series = dt_series.dt.tz_localize(None)

        # Store as 'datetime' column (standardized name)
        df['datetime'] = dt_series

        # Remove old column if it had a different name
        if dt_col != 'datetime' and dt_col in df.columns:
            df = df.drop(columns=[dt_col])

    return df


def _remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are completely empty (all NaN, None, or empty strings).
    This cleans up datasets from Supabase that may have placeholder columns.
    """
    df = df.copy()
    initial_cols = len(df.columns)

    # Find columns that are completely empty
    empty_cols = []
    for col in df.columns:
        # Check if column is all NaN, None, or empty strings
        if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            empty_cols.append(col)

    # Drop empty columns
    if empty_cols:
        df = df.drop(columns=empty_cols)
        st.info(f"🧹 Removed {len(empty_cols)} empty columns: {', '.join(empty_cols)}")

    return df


def _fetch_from_api(dataset_type: str, start_date: datetime, end_date: datetime, provider: str = "supabase"):
    """Fetch data from API and store in session state"""
    if not API_AVAILABLE:
        st.error("API functionality not available. Please check installation.")
        return None

    try:
        config_manager = APIConfigManager()

        # Get appropriate connector
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

        # Fetch data
        with st.spinner(f"Fetching {dataset_type} data from {provider}..."):
            df = connector.fetch_data(start_date, end_date)

        if df is not None and not df.empty:
            # Clean datetime columns to remove timezone and standardize
            df = _clean_datetime_column(df, dataset_type)

            # Remove empty columns from Supabase
            df = _remove_empty_columns(df)

            # Special handling for reason dataset: Remove total_arrivals (duplicates Target_1)
            if dataset_type == "reason":
                # Find and remove total_arrivals column (case-insensitive)
                cols_to_drop = [col for col in df.columns if col.lower() == "total_arrivals"]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    st.info(f"🗑️ Removed '{cols_to_drop[0]}' from reason dataset (duplicates patient_count)")

            st.session_state[session_key] = df
            st.session_state[f"{dataset_type}_loaded"] = True
            st.success(f"✅ Fetched {len(df):,} rows from {provider} (cleaned {len(df.columns)} columns)")
            return df
        else:
            st.warning(f"No data found for selected date range")
            return None

    except Exception as e:
        st.error(f"❌ API fetch failed: {str(e)}")
        return None


# ---- Small utilities ------------------------------------------------------------
def _date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "date", "Date", "timestamp", "ds", "time"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def _data_health(df: Optional[pd.DataFrame]):
    """Lightweight data-health table."""
    if isinstance(df, pd.DataFrame) and not df.empty:
        dcol = _date_col(df)
        if dcol is not None:
            dt = pd.to_datetime(df[dcol], errors="coerce")
            dr = f"{dt.min().date()} → {dt.max().date()}" if dt.notna().any() else "N/A"
        else:
            dr = "N/A"
        st.write(pd.DataFrame({
            "Indicator": ["Rows", "Columns", "Date Range", "Missing Values"],
            "Value": [f"{len(df):,}", f"{df.shape[1]}", dr, f"{int(df.isna().sum().sum()):,}"]
        }))
    else:
        st.caption("No data loaded yet.")

# ---- Context-aware navigation to “Data Preprocessing Studio” --------------------
def _go_preprocessing():
    """Try to open a preprocessing page if available; fall back to your router."""
    st.session_state["_nav_intent"] = "🧠 Data Preprocessing Studio"
    for path in [
        "pages/02_DataFusion.py",          # legacy name in your app
        "pages/02_Data_Fusion.py",         # underscore variant
        "pages/02_Data_Preprocessing.py",  # new canonical option
        "pages/DataPreprocessingStudio.py"
    ]:
        try:
            if os.path.exists(path):
                try:
                    st.switch_page(path)
                    return
                except Exception:
                    pass
        except Exception:
            pass
    st.rerun()

# ---- Page Configuration ----
st.set_page_config(
    page_title="Upload Data - HealthForecast AI",
    page_icon="📤",
    layout="wide",
)

# ---- Page body -----------------------------------------------------------------
def page_data_hub():
    # Apply premium theme and sidebar
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    add_logout_button()

    # Apply fluorescent effects
    st.markdown("""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR DATA HUB
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
        title="Upload Data",
        subtitle="Upload and manage your patient, weather, and calendar data for comprehensive healthcare forecasting.",
        status="SYSTEM ONLINE"
    )

    # Ensure session keys exist
    for k in ["patient_data", "weather_data", "calendar_data", "reason_data", "selected_hospital"]:
        st.session_state.setdefault(k, None)

    # ========================================================================
    # HOSPITAL SELECTOR SECTION
    # ========================================================================
    st.markdown(f"""
    <div class='hf-feature-card' style='margin-bottom: 1.5rem; padding: 1.5rem;'>
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
            <div style='font-size: 2rem;'>🏥</div>
            <div>
                <h3 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Select Hospital</h3>
                <p class='hf-feature-description' style='margin: 0.25rem 0 0 0; font-size: 0.875rem;'>
                    Choose the hospital to load data from
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get available hospitals
    available_hospitals = ["Pamplona Spain Hospital"]  # Default
    if HOSPITAL_SERVICE_AVAILABLE:
        try:
            hospital_service = get_hospital_service()
            if hospital_service.is_connected():
                available_hospitals = hospital_service.get_available_hospitals()
        except Exception:
            pass

    col_hosp1, col_hosp2, col_hosp3 = st.columns([1, 2, 1])
    with col_hosp2:
        # Hospital selector dropdown
        selected_hospital = st.selectbox(
            "🏥 Hospital",
            options=available_hospitals,
            index=0,
            key="hospital_selector",
            help="Select the hospital to fetch data from"
        )
        st.session_state["selected_hospital"] = selected_hospital

        # Quick fetch all button
        if HOSPITAL_SERVICE_AVAILABLE and API_AVAILABLE:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                hosp_start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="hosp_start")
            with col_date2:
                hosp_end = st.date_input("End Date", value=datetime(2023, 12, 31), key="hosp_end")

            if st.button(f"🚀 Load All Data for {selected_hospital}", use_container_width=True, type="primary"):
                start_dt = datetime.combine(hosp_start, datetime.min.time())
                end_dt = datetime.combine(hosp_end, datetime.min.time())

                hospital_service = get_hospital_service()

                progress_bar = st.progress(0, text="Fetching datasets...")
                datasets = ["patient", "weather", "calendar", "reason"]

                success_count = 0
                for i, dataset_type in enumerate(datasets):
                    progress_bar.progress((i) / 4, text=f"Fetching {dataset_type} data...")

                    df = hospital_service.fetch_dataset_by_hospital(
                        dataset_type=dataset_type,
                        hospital_name=selected_hospital,
                        start_date=start_dt,
                        end_date=end_dt,
                    )

                    if df is not None and not df.empty:
                        # Clean datetime columns
                        df = _clean_datetime_column(df, dataset_type)
                        df = _remove_empty_columns(df)

                        # Special handling for reason dataset
                        if dataset_type == "reason":
                            cols_to_drop = [col for col in df.columns if col.lower() == "total_arrivals"]
                            if cols_to_drop:
                                df = df.drop(columns=cols_to_drop)

                        st.session_state[f"{dataset_type}_data"] = df
                        st.session_state[f"{dataset_type}_loaded"] = True
                        success_count += 1
                        st.success(f"✅ {dataset_type.title()}: {len(df):,} rows")
                    else:
                        st.warning(f"⚠️ No {dataset_type} data found")

                progress_bar.progress(1.0, text="Complete!")

                if success_count == 4:
                    st.success(f"✅ Successfully loaded all data for **{selected_hospital}**!")
                    st.balloons()
                elif success_count > 0:
                    st.warning(f"⚠️ Loaded {success_count}/4 datasets for {selected_hospital}")

    st.markdown("---")

    # ========================================================================
    # FOUR-COLUMN GRID LAYOUT FOR DATA UPLOAD CARDS
    # ========================================================================
    col1, col2, col3, col4 = st.columns(4)

    # ========================================================================
    # PATIENT DATA CARD (Column 1)
    # ========================================================================
    with col1:
        st.markdown("""
        <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>📊</div>
                <h2 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Patient Data</h2>
                <p class='hf-feature-description' style='margin: 0.5rem 0 0 0; font-size: 0.8125rem;'>Historical arrival records</p>
            </div>
        """, unsafe_allow_html=True)

        # Tabs for Upload vs API Fetch
        tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 Fetch from API"])

        with tab1:
            _upload_generic("Upload CSV", "patient", "datetime")

        with tab2:
            if API_AVAILABLE:
                st.caption("Fetch patient data from hospital database")
                col_a, col_b = st.columns(2)
                with col_a:
                    start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="patient_start")
                with col_b:
                    end = st.date_input("End Date", value=datetime.now(), key="patient_end")

                if st.button("Fetch Patient Data", key="fetch_patient", use_container_width=True):
                    _fetch_from_api("patient", datetime.combine(start, datetime.min.time()),
                                   datetime.combine(end, datetime.min.time()), "supabase")
            else:
                st.warning("API functionality not available")

        df_patient = st.session_state.get("patient_data")

        if isinstance(df_patient, pd.DataFrame) and not df_patient.empty:
            st.markdown("<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06);'></div>", unsafe_allow_html=True)
            st.metric("Rows", f"{df_patient.shape[0]:,}")
            st.metric("Columns", df_patient.shape[1])

            dcol = _date_col(df_patient)
            if dcol:
                dt = pd.to_datetime(df_patient[dcol], errors="coerce")
                if dt.notna().any():
                    st.caption(f"📅 {dt.min().date()} → {dt.max().date()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # WEATHER DATA CARD (Column 2)
    # ========================================================================
    with col2:
        st.markdown("""
        <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>🌤️</div>
                <h2 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Weather Data</h2>
                <p class='hf-feature-description' style='margin: 0.5rem 0 0 0; font-size: 0.8125rem;'>Meteorological conditions</p>
            </div>
        """, unsafe_allow_html=True)

        # Tabs for Upload vs API Fetch
        tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 Fetch from API"])

        with tab1:
            _upload_generic("Upload CSV", "weather", "datetime")

        with tab2:
            if API_AVAILABLE:
                st.caption("Fetch weather data from hospital database")
                col_a, col_b = st.columns(2)
                with col_a:
                    start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="weather_start")
                with col_b:
                    end = st.date_input("End Date", value=datetime.now(), key="weather_end")

                if st.button("Fetch Weather Data", key="fetch_weather", use_container_width=True):
                    _fetch_from_api("weather", datetime.combine(start, datetime.min.time()),
                                   datetime.combine(end, datetime.min.time()), "supabase")
            else:
                st.warning("API functionality not available")

        df_weather = st.session_state.get("weather_data")

        if isinstance(df_weather, pd.DataFrame) and not df_weather.empty:
            st.markdown("<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06);'></div>", unsafe_allow_html=True)
            st.metric("Rows", f"{df_weather.shape[0]:,}")
            st.metric("Columns", df_weather.shape[1])

            dcol = _date_col(df_weather)
            if dcol:
                dt = pd.to_datetime(df_weather[dcol], errors="coerce")
                if dt.notna().any():
                    st.caption(f"📅 {dt.min().date()} → {dt.max().date()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # CALENDAR DATA CARD (Column 3)
    # ========================================================================
    with col3:
        st.markdown("""
        <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>📅</div>
                <h2 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Calendar Data</h2>
                <p class='hf-feature-description' style='margin: 0.5rem 0 0 0; font-size: 0.8125rem;'>Holidays & events</p>
            </div>
        """, unsafe_allow_html=True)

        # Tabs for Upload vs API Fetch
        tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 Fetch from API"])

        with tab1:
            _upload_generic("Upload CSV", "calendar", "date")

        with tab2:
            if API_AVAILABLE:
                st.caption("Fetch calendar data from hospital database")
                col_a, col_b = st.columns(2)
                with col_a:
                    start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="calendar_start")
                with col_b:
                    end = st.date_input("End Date", value=datetime.now(), key="calendar_end")

                if st.button("Fetch Calendar Data", key="fetch_calendar", use_container_width=True):
                    _fetch_from_api("calendar", datetime.combine(start, datetime.min.time()),
                                   datetime.combine(end, datetime.min.time()), "supabase")
            else:
                st.warning("API functionality not available")

        df_calendar = st.session_state.get("calendar_data")

        if isinstance(df_calendar, pd.DataFrame) and not df_calendar.empty:
            st.markdown("<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06);'></div>", unsafe_allow_html=True)
            st.metric("Rows", f"{df_calendar.shape[0]:,}")
            st.metric("Columns", df_calendar.shape[1])

            dcol = _date_col(df_calendar)
            if dcol:
                dt = pd.to_datetime(df_calendar[dcol], errors="coerce")
                if dt.notna().any():
                    st.caption(f"📅 {dt.min().date()} → {dt.max().date()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # REASON FOR VISIT DATA CARD (Column 4)
    # ========================================================================
    with col4:
        st.markdown("""
        <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>🏥</div>
                <h2 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Reason for Visit</h2>
                <p class='hf-feature-description' style='margin: 0.5rem 0 0 0; font-size: 0.8125rem;'>Visit reasons by category</p>
            </div>
        """, unsafe_allow_html=True)

        # Tabs for Upload vs API Fetch
        tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 Fetch from API"])

        with tab1:
            _upload_generic("Upload CSV", "reason", "datetime")

        with tab2:
            if API_AVAILABLE:
                st.caption("Fetch clinical visit data from hospital database")
                col_a, col_b = st.columns(2)
                with col_a:
                    start = st.date_input("Start Date", value=datetime(2018, 1, 1), key="reason_start")
                with col_b:
                    end = st.date_input("End Date", value=datetime.now(), key="reason_end")

                if st.button("Fetch Reason Data", key="fetch_reason", use_container_width=True):
                    _fetch_from_api("reason", datetime.combine(start, datetime.min.time()),
                                   datetime.combine(end, datetime.min.time()), "supabase")
            else:
                st.warning("API functionality not available")

        df_reason = st.session_state.get("reason_data")

        if isinstance(df_reason, pd.DataFrame) and not df_reason.empty:
            st.markdown("<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06);'></div>", unsafe_allow_html=True)
            st.metric("Rows", f"{df_reason.shape[0]:,}")
            st.metric("Columns", df_reason.shape[1])

            dcol = _date_col(df_reason)
            if dcol:
                dt = pd.to_datetime(df_reason[dcol], errors="coerce")
                if dt.notna().any():
                    st.caption(f"📅 {dt.min().date()} → {dt.max().date()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # DETAILED DATA PREVIEW SECTION (EXPANDABLE)
    # ========================================================================
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Get all loaded datasets
    df_patient = st.session_state.get("patient_data")
    df_weather = st.session_state.get("weather_data")
    df_calendar = st.session_state.get("calendar_data")
    df_reason = st.session_state.get("reason_data")

    # Show detailed previews for loaded data
    if isinstance(df_patient, pd.DataFrame) and not df_patient.empty:
        with st.expander("📊 Patient Data - Detailed View", expanded=False):
            st.dataframe(df_patient.head(20), use_container_width=True)
            st.markdown("**Data Health**")
            _data_health(df_patient)

    if isinstance(df_weather, pd.DataFrame) and not df_weather.empty:
        with st.expander("🌤️ Weather Data - Detailed View", expanded=False):
            st.dataframe(df_weather.head(20), use_container_width=True)
            st.markdown("**Data Health**")
            _data_health(df_weather)

    if isinstance(df_calendar, pd.DataFrame) and not df_calendar.empty:
        with st.expander("📅 Calendar Data - Detailed View", expanded=False):
            st.dataframe(df_calendar.head(20), use_container_width=True)
            st.markdown("**Data Health**")
            _data_health(df_calendar)

    if isinstance(df_reason, pd.DataFrame) and not df_reason.empty:
        with st.expander("🏥 Reason for Visit Data - Detailed View", expanded=False):
            st.dataframe(df_reason.head(20), use_container_width=True)
            st.markdown("**Data Health**")
            _data_health(df_reason)

    # Premium Footer CTA
    all_loaded = all([
        bool(st.session_state.get("patient_data")  is not None and not getattr(st.session_state.get("patient_data"),  "empty", True)),
        bool(st.session_state.get("weather_data")  is not None and not getattr(st.session_state.get("weather_data"),  "empty", True)),
        bool(st.session_state.get("calendar_data") is not None and not getattr(st.session_state.get("calendar_data"), "empty", True)),
        bool(st.session_state.get("reason_data") is not None and not getattr(st.session_state.get("reason_data"), "empty", True)),
    ])

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Status Summary Card
    current_hospital = st.session_state.get("selected_hospital", "No hospital selected")
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem;'>
        <h3 class='hf-feature-title' style='font-size: 1.125rem; margin-bottom: 0.5rem;'>📋 Data Upload Status</h3>
        <p style='color: {PRIMARY_COLOR}; font-size: 0.9rem; margin-bottom: 1rem;'>🏥 {current_hospital}</p>
        <div style='display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap;'>
            <div>
                <div style='font-size: 1.25rem; margin-bottom: 0.25rem;'>
                    {'✅' if st.session_state.get("patient_data") is not None else '⏳'}
                </div>
                <div style='color: {SUBTLE_TEXT}; font-size: 0.8125rem;'>Patient Data</div>
            </div>
            <div>
                <div style='font-size: 1.25rem; margin-bottom: 0.25rem;'>
                    {'✅' if st.session_state.get("weather_data") is not None else '⏳'}
                </div>
                <div style='color: {SUBTLE_TEXT}; font-size: 0.8125rem;'>Weather Data</div>
            </div>
            <div>
                <div style='font-size: 1.25rem; margin-bottom: 0.25rem;'>
                    {'✅' if st.session_state.get("calendar_data") is not None else '⏳'}
                </div>
                <div style='color: {SUBTLE_TEXT}; font-size: 0.8125rem;'>Calendar Data</div>
            </div>
            <div>
                <div style='font-size: 1.25rem; margin-bottom: 0.25rem;'>
                    {'✅' if st.session_state.get("reason_data") is not None else '⏳'}
                </div>
                <div style='color: {SUBTLE_TEXT}; font-size: 0.8125rem;'>Reason for Visit</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


init_state()
page_data_hub()

# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(1)  # Upload Data is page index 1


