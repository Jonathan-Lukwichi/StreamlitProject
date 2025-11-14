from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

from app_core.state.session import init_state

# ---- Small helpers ----

def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "date", "Date", "timestamp", "time", "ds"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(s)))


def _pick_count_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["patient_count", "Target_1", "patients", "count", "y", "value"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# ---- Page Configuration ----
st.set_page_config(
    page_title="Dashboard - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

# ---- Page body ----

def page_dashboard():
    # Apply premium theme and sidebar
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()

    # Apply fluorescent effects
    st.markdown("""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR DASHBOARD
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

    datasets_loaded = sum([
        bool(st.session_state.get("patient_loaded")),
        bool(st.session_state.get("weather_loaded")),
        bool(st.session_state.get("calendar_loaded")),
    ])
    system_active = bool(st.session_state.get("patient_loaded"))

    # Premium Hero Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>📊</div>
          <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Healthcare Forecasting Dashboard</h1>
          <p class='hf-feature-description' style='font-size: 1rem; max-width: 700px; margin: 0 auto;'>
            Real-time analytics and predictive insights for patient arrival forecasting
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "Active" if system_active else "Offline"
        status_icon = "🟢" if system_active else "🔴"
        rgb = "16,185,129" if system_active else "239,68,68"
        color = SUCCESS_COLOR if system_active else DANGER_COLOR
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>System Status</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {status}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Data pipeline connectivity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        df_p = st.session_state.get("patient_data")
        total_patients = 0
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            count_col = _pick_count_col(df_p)
            if count_col:
                total_patients = int(pd.to_numeric(df_p[count_col], errors="coerce").fillna(0).sum())
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Total Patients</div>
              <div class='hf-feature-title' style='font-size: 2rem; background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.75rem 0;'>
                {total_patients:,}
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Sum across loaded history</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Datasets</div>
              <div class='hf-feature-title' style='font-size: 2rem; background: linear-gradient(135deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.75rem 0;'>
                {datasets_loaded}/3
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Patient • Weather • Calendar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        pipeline_complete = st.session_state.get("merged_data") is not None
        status_icon = "🟢" if pipeline_complete else "🟡"
        rgb = "16,185,129" if pipeline_complete else "245,158,11"
        color = SUCCESS_COLOR if pipeline_complete else WARNING_COLOR
        label = "Complete" if pipeline_complete else "Pending"
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Pipeline</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {label}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Data fusion status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # DATA HEALTH & RECENT ACTIVITY
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(f"<h2 class='hf-feature-title' style='margin-bottom: 0.75rem; font-size: 1.5rem;'>📋 Data Health</h2>", unsafe_allow_html=True)
        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            date_col = _pick_date_col(df_p)
            date_range_txt = "N/A"
            if date_col:
                dt = _to_datetime_safe(df_p[date_col])
                if dt.notna().any():
                    date_range_txt = f"{dt.min().date()} → {dt.max().date()}"
            nulls = int(df_p.isna().sum().sum())
            cols = df_p.shape[1]
            rows = len(df_p)
            st.write(
                pd.DataFrame(
                    {"Indicator": ["Rows", "Columns", "Date Range", "Missing Values"],
                     "Value": [f"{rows:,}", f"{cols}", date_range_txt, f"{nulls:,}"]}
                )
            )
        else:
            st.info("Upload patient data to view schema, date range, and nulls.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f"<h2 class='hf-feature-title' style='margin-bottom: 0.75rem; font-size: 1.5rem;'>📈 Recent Activity</h2>", unsafe_allow_html=True)
        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        try:
            if isinstance(df_p, pd.DataFrame) and not df_p.empty:
                date_col = _pick_date_col(df_p)
                y_col = _pick_count_col(df_p)
                if date_col and y_col:
                    dt = _to_datetime_safe(df_p[date_col])
                    s = pd.to_numeric(df_p[y_col], errors="coerce")
                    recent = pd.DataFrame({"dt": dt, "y": s}).dropna().sort_values("dt")
                    if not recent.empty:
                        cutoff = recent["dt"].max() - pd.Timedelta(days=30)
                        recent = recent[recent["dt"] >= cutoff]
                        st.line_chart(recent.set_index("dt")[["y"]])
                    else:
                        st.caption("No valid time series points detected yet.")
                else:
                    st.caption("Waiting for a valid date/count column to render a sparkline.")
            else:
                st.caption("Data not loaded.")
        except Exception as e:
            st.caption(f"Unable to draw sparkline: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # MODEL READINESS
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='hf-feature-title' style='margin-bottom: 0.75rem; font-size: 1.5rem;'>✓ Model Readiness</h2>", unsafe_allow_html=True)
    st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
    checks = [
        ("Patient data loaded", bool(st.session_state.get("patient_loaded"))),
        ("Weather data loaded", bool(st.session_state.get("weather_loaded"))),
        ("Calendar data loaded", bool(st.session_state.get("calendar_loaded"))),
        ("Data fusion complete", st.session_state.get("merged_data") is not None),
    ]
    for label, ok in checks:
        cls = "check" if ok else ("warn" if label != "Data fusion complete" else "fail")
        icon = "✅" if ok else ("⏳" if label != "Data fusion complete" else "⚠️")
        st.markdown(f"<div style='margin: 0.75rem 0;'><span class='{cls}'>{icon}</span> {label}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # QUICK INSIGHTS
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='hf-feature-title' style='margin-bottom: 0.75rem; font-size: 1.5rem;'>💡 Quick Insights</h2>", unsafe_allow_html=True)
    st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
    if isinstance(df_p, pd.DataFrame) and not df_p.empty:
        y_col = _pick_count_col(df_p)
        date_col = _pick_date_col(df_p)
        if y_col and date_col:
            dfq = pd.DataFrame({
                "dt": _to_datetime_safe(df_p[date_col]),
                "y": pd.to_numeric(df_p[y_col], errors="coerce")
            }).dropna().sort_values("dt")
            if not dfq.empty:
                mean_v = float(dfq["y"].mean())
                std_v = float(dfq["y"].std(ddof=0))
                ratio = std_v / mean_v if mean_v > 0 else np.nan
                hint = ""
                if len(dfq) >= 14:
                    try:
                        w = dfq.copy(); w["dow"] = w["dt"].dt.dayofweek
                        wd = w[w["dow"] < 5]["y"].mean(); we = w[w["dow"] >= 5]["y"].mean()
                        if np.isfinite(wd) and np.isfinite(we) and abs(wd - we) / (mean_v + 1e-9) > 0.05:
                            hint = "Possible weekday/weekend seasonality detected."
                    except Exception:
                        pass
                st.markdown(
                    f"<div style='line-height: 1.8;'><strong>Mean arrivals:</strong> {mean_v:,.1f} &nbsp;|&nbsp; <strong>Volatility (σ/μ):</strong> {ratio:,.2f} " + (f"&nbsp;|&nbsp; <strong>Signal:</strong> {hint}" if hint else "") + "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("Not enough valid points for insights.")
        else:
            st.caption("Insights will appear once a date and count column are detected.")
    else:
        st.caption("Load data to compute summary insights.")
    st.markdown("</div>", unsafe_allow_html=True)


init_state()
page_dashboard()
