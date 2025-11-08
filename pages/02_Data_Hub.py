# =============================================================================
# pages/02_Data_Hub_enhanced.py — Data Integration Hub (Pro Prototype)
# Matches the dashboard’s clean template: hero strip, tabs for uploads, previews,
# compact data-health summaries, and a context-aware CTA to “Data Preprocessing Studio”.
# Defensive: runs even if some app_core modules are missing.
# =============================================================================
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

# ---- Theme / framework fallbacks ------------------------------------------------
try:
    from app_core.ui.theme import apply_css
    from app_core.ui.theme import (
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
        DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
    )
except Exception:
    PRIMARY_COLOR   = "#2563eb"
    SECONDARY_COLOR = "#8b5cf6"
    SUCCESS_COLOR   = "#10b981"
    WARNING_COLOR   = "#f59e0b"
    DANGER_COLOR    = "#ef4444"
    TEXT_COLOR      = "#0f172a"
    SUBTLE_TEXT     = "#64748b"

    def apply_css():
        st.markdown(
            """
            <style>
            .muted {color:#6b7280}
            .metric-card {border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:#fff;}
            </style>
            """,
            unsafe_allow_html=True,
        )

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

# ---- Local CSS to mirror dashboard style ---------------------------------------
def _datahub_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*='css'] {{ font-family: 'Inter', system-ui, sans-serif; }}

        .hero {{
            border-radius: 20px; padding: 28px; color: #0b1220;
            background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(99,102,241,.08), rgba(20,184,166,.12));
            border: 1px solid rgba(37,99,235,.25);
            box-shadow: 0 20px 25px -5px rgba(0,0,0,.08), 0 10px 10px -5px rgba(0,0,0,.04);
            position: relative; overflow: hidden; margin-bottom: 18px;
        }}
        .hero-title {{
            font-size: 1.8rem; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.02em;
            background: linear-gradient(135deg, {TEXT_COLOR}, {PRIMARY_COLOR});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .hero-sub {{ color: {SUBTLE_TEXT}; font-size: 1.0rem; margin: 0; }}

        .section-card {{ border:1px solid #e5e7eb; border-radius:16px; padding:14px; background:#fff; }}
        .section-title {{ font-weight:800; font-size:1.05rem; margin-bottom:8px; color:{TEXT_COLOR}; }}
        .badge-ok {{ color:{SUCCESS_COLOR}; font-weight:700; }}
        .badge-warn {{ color:{WARNING_COLOR}; font-weight:700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

# ---- Page body -----------------------------------------------------------------
def page_data_hub():
    _datahub_css()

    # HERO (aligned with dashboard styling)
    st.markdown(
        f"""
        <div class='hero'>
          <div class='hero-title'>📤 Data Integration Hub</div>
          <p class='hero-sub'>Upload and validate your source datasets. Patient, weather, and calendar files are supported.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ensure session keys exist
    for k in ["patient_data", "weather_data", "calendar_data"]:
        st.session_state.setdefault(k, None)

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Patient Data", "🌤️ Weather Data", "📅 Calendar Data"])

    with t1:
        _upload_generic("📊 Patient Arrival Data", "patient", "datetime")
        df = st.session_state.get("patient_data")
        st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            _data_health(df)
        else:
            st.info("Upload a Patient dataset to preview.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        _upload_generic("🌤️ Weather Data", "weather", "datetime")
        df = st.session_state.get("weather_data")
        st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            _data_health(df)
        else:
            st.info("Upload a Weather dataset to preview.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        _upload_generic("📅 Calendar Data", "calendar", "date")
        df = st.session_state.get("calendar_data")
        st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            _data_health(df)
        else:
            st.info("Upload a Calendar dataset to preview.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer CTA — enabled only when all three sources are loaded
    all_loaded = all([
        bool(st.session_state.get("patient_data")  is not None and not getattr(st.session_state.get("patient_data"),  "empty", True)),
        bool(st.session_state.get("weather_data")  is not None and not getattr(st.session_state.get("weather_data"),  "empty", True)),
        bool(st.session_state.get("calendar_data") is not None and not getattr(st.session_state.get("calendar_data"), "empty", True)),
    ])

    st.markdown("\n")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if all_loaded:
            if st.button("🧠 Continue to Data Preprocessing Studio", use_container_width=True, type="primary"):
                _go_preprocessing()
        else:
            st.button(
                "🧠 Continue to Data Preprocessing Studio",
                use_container_width=True,
                disabled=True,
                help="Upload all three datasets to proceed.",
            )

# ---- Entrypoint -----------------------------------------------------------------
def main():
    apply_css()
    init_state()
    page_data_hub()

if __name__ == "__main__":
    main()
