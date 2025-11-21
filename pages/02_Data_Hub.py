# =============================================================================
# pages/02_Data_Hub.py — Data Integration Hub (Premium Design)
# Upload and manage patient, weather, and calendar data with premium UI.
# =============================================================================
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

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
    page_title="Data Hub - HealthForecast AI",
    page_icon="📤",
    layout="wide",
)

# ---- Page body -----------------------------------------------------------------
def page_data_hub():
    # Apply premium theme and sidebar
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()

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
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>📤</div>
          <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Data Integration Hub</h1>
          <p class='hf-feature-description' style='font-size: 1rem; max-width: 700px; margin: 0 auto;'>
            Upload and manage your patient, weather, and calendar data
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ensure session keys exist
    for k in ["patient_data", "weather_data", "calendar_data", "reason_data"]:
        st.session_state.setdefault(k, None)

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

        _upload_generic("Upload CSV", "patient", "datetime")
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

        _upload_generic("Upload CSV", "weather", "datetime")
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

        _upload_generic("Upload CSV", "calendar", "date")
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

        _upload_generic("Upload CSV", "reason", "datetime")
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
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem;'>
        <h3 class='hf-feature-title' style='font-size: 1.125rem; margin-bottom: 1rem;'>📋 Data Upload Status</h3>
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

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if all_loaded:
            if st.button("🧠 Continue to Data Preprocessing Studio", use_container_width=True, type="primary"):
                _go_preprocessing()
        else:
            st.button(
                "🧠 Continue to Data Preprocessing Studio",
                use_container_width=True,
                disabled=True,
                help="Upload all four datasets to proceed.",
            )

init_state()
page_data_hub()


