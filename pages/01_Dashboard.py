from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
import datetime

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

    # Enhanced Dashboard Styling - Power BI Inspired
    st.markdown("""
    <style>
    /* ========================================
       PREMIUM DASHBOARD DESIGN SYSTEM
       Power BI-Inspired Professional Theme
       ======================================== */

    /* --- AMBIENT LIGHTING EFFECTS --- */
    @keyframes float-orb {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.18;
        }
        50% {
            transform: translate(40px, -40px) scale(1.08);
            opacity: 0.28;
        }
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.7; }
    }

    .fluorescent-orb {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(90px);
        mix-blend-mode: screen;
    }

    .orb-1 {
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.3), rgba(99, 102, 241, 0.15) 50%, transparent 70%);
        top: 10%;
        right: 15%;
        animation: float-orb 28s ease-in-out infinite;
    }

    .orb-2 {
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.25), rgba(14, 165, 233, 0.12) 50%, transparent 70%);
        bottom: 15%;
        left: 10%;
        animation: float-orb 35s ease-in-out infinite;
        animation-delay: 7s;
    }

    .orb-3 {
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(147, 51, 234, 0.2), transparent 70%);
        top: 50%;
        left: 50%;
        animation: float-orb 40s ease-in-out infinite;
        animation-delay: 14s;
    }

    @keyframes sparkle {
        0%, 100% {
            opacity: 0;
            transform: scale(0) rotate(0deg);
        }
        50% {
            opacity: 0.8;
            transform: scale(1) rotate(180deg);
        }
    }

    .sparkle {
        position: fixed;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.9), rgba(59, 130, 246, 0.4));
        border-radius: 50%;
        pointer-events: none;
        z-index: 2;
        animation: sparkle 4s ease-in-out infinite;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.6), 0 0 24px rgba(59, 130, 246, 0.3);
    }

    .sparkle-1 { top: 20%; left: 30%; animation-delay: 0s; }
    .sparkle-2 { top: 70%; left: 75%; animation-delay: 1.3s; }
    .sparkle-3 { top: 40%; left: 12%; animation-delay: 2.6s; }
    .sparkle-4 { top: 85%; left: 40%; animation-delay: 3.9s; }

    /* --- ENHANCED KPI CARDS --- */
    .dashboard-kpi-card {
        position: relative;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border-radius: 14px;
        padding: 1.75rem 1.25rem;
        border: 1px solid rgba(59, 130, 246, 0.15);
        box-shadow:
            0 4px 20px rgba(0, 0, 0, 0.25),
            0 0 40px rgba(59, 130, 246, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }

    .dashboard-kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    .dashboard-kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.35);
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.35),
            0 0 60px rgba(59, 130, 246, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .dashboard-kpi-card:hover::before {
        opacity: 1;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: rgba(148, 163, 184, 0.85);
        margin-bottom: 0.875rem;
        display: block;
    }

    .kpi-value {
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0 0.75rem 0;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    .status-badge-enhanced {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.125rem;
        border-radius: 10px;
        font-size: 0.8125rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    .status-badge-enhanced:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* --- SECTION HEADERS --- */
    .section-header {
        position: relative;
        padding-left: 1rem;
        margin-bottom: 1.25rem;
        font-size: 1.375rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.3px;
    }

    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 70%;
        background: linear-gradient(180deg, #3b82f6, #22d3ee);
        border-radius: 2px;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.5);
    }

    /* --- INSIGHTS CARD ENHANCEMENT --- */
    .insights-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.7));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.12);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }

    .insight-metric {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.625rem 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        color: #cbd5e1;
        transition: all 0.3s ease;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .insight-metric:hover {
        background: rgba(59, 130, 246, 0.15);
        transform: translateX(4px);
    }

    .insight-metric strong {
        color: #3b82f6;
    }

    @media (max-width: 768px) {
        .fluorescent-orb {
            width: 250px !important;
            height: 250px !important;
            filter: blur(60px);
        }
        .sparkle {
            display: none;
        }
        .dashboard-kpi-card {
            padding: 1.25rem 1rem;
        }
        .kpi-value {
            font-size: 1.875rem;
        }
    }
    </style>

    <!-- Enhanced Ambient Effects -->
    <div class="fluorescent-orb orb-1"></div>
    <div class="fluorescent-orb orb-2"></div>
    <div class="fluorescent-orb orb-3"></div>

    <!-- Sparkle Particles -->
    <div class="sparkle sparkle-1"></div>
    <div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div>
    <div class="sparkle sparkle-4"></div>
    """, unsafe_allow_html=True)

    datasets_loaded = sum([
        bool(st.session_state.get("patient_loaded")),
        bool(st.session_state.get("weather_loaded")),
        bool(st.session_state.get("calendar_loaded")),
    ])
    system_active = bool(st.session_state.get("patient_loaded"))

    # Compact Hero Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='
            margin-bottom: 1.5rem;
            padding: 1.25rem 2rem;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(59, 130, 246, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 48px rgba(59, 130, 246, 0.1);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: space-between;
        '>
          <div style='
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              height: 2px;
              background: linear-gradient(90deg, transparent, #3b82f6, #22d3ee, #3b82f6, transparent);
          '></div>
          <div style='display: flex; align-items: center; gap: 1.25rem;'>
              <div class='hf-feature-icon' style='
                  font-size: 2.5rem;
                  filter: drop-shadow(0 4px 12px rgba(59, 130, 246, 0.4));
              '>📊</div>
              <div>
                  <h1 class='hf-feature-title' style='
                      font-size: 1.625rem;
                      margin-bottom: 0.25rem;
                      font-weight: 800;
                      letter-spacing: -0.5px;
                      background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      background-clip: text;
                  '>Healthcare Forecasting Dashboard</h1>
                  <p class='hf-feature-description' style='
                      font-size: 0.875rem;
                      margin: 0;
                      color: rgba(203, 213, 225, 0.75);
                      font-weight: 400;
                  '>
                    Real-time analytics and predictive insights
                  </p>
              </div>
          </div>
          <div style='
              font-size: 0.75rem;
              color: rgba(148, 163, 184, 0.7);
              text-align: right;
          '>
              <div>Last Updated</div>
              <div style='color: #22d3ee; font-weight: 600;'>Live</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- OPTIMIZED COMPACT LAYOUT ---
    # System Overview - All in one row
    st.markdown(f"<h2 class='section-header' style='margin-bottom: 1rem;'>📋 System Overview</h2>", unsafe_allow_html=True)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.2, 1.2, 1.2, 1.5])

    with kpi_col1:
        status = "Active" if system_active else "Offline"
        status_icon = "🟢" if system_active else "🔴"
        rgb = "16,185,129" if system_active else "239,68,68"
        color = SUCCESS_COLOR if system_active else DANGER_COLOR
        st.markdown(
            f"""
            <div class='dashboard-kpi-card' style='text-align: center; padding: 1.25rem 1rem;'>
              <span class='kpi-label' style='margin-bottom: 0.5rem;'>System Status</span>
              <div style='margin: 0.75rem 0;'>
                <span class='status-badge-enhanced' style='background:rgba({rgb},.18);color:{color};border:1px solid rgba({rgb},.35);'>
                  {status_icon} {status}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class='dashboard-kpi-card' style='text-align: center; padding: 1.25rem 1rem;'>
              <span class='kpi-label' style='margin-bottom: 0.5rem;'>Datasets Loaded</span>
              <div class='kpi-value' style='font-size: 2rem; margin: 0.5rem 0;'>{datasets_loaded}/3</div>
              <div style='
                  width: 100%;
                  height: 5px;
                  background: rgba(59, 130, 246, 0.15);
                  border-radius: 3px;
                  overflow: hidden;
                  margin-top: 0.25rem;
              '>
                <div style='
                    width: {(datasets_loaded/3)*100}%;
                    height: 100%;
                    background: linear-gradient(90deg, #3b82f6, #22d3ee);
                    transition: width 0.5s ease;
                '></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        pipeline_complete = st.session_state.get("merged_data") is not None
        status_icon = "🟢" if pipeline_complete else "🟡"
        rgb = "16,185,129" if pipeline_complete else "245,158,11"
        color = SUCCESS_COLOR if pipeline_complete else WARNING_COLOR
        label = "Complete" if pipeline_complete else "Pending"
        st.markdown(
            f"""
            <div class='dashboard-kpi-card' style='text-align: center; padding: 1.25rem 1rem;'>
              <span class='kpi-label' style='margin-bottom: 0.5rem;'>Data Pipeline</span>
              <div style='margin: 0.75rem 0;'>
                <span class='status-badge-enhanced' style='background:rgba({rgb},.18);color:{color};border:1px solid rgba({rgb},.35);'>
                  {status_icon} {label}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick Insights in the 4th column
    with kpi_col4:
        st.markdown("<div class='dashboard-kpi-card' style='padding: 1.25rem 1rem;'>", unsafe_allow_html=True)
        st.markdown("<span class='kpi-label' style='margin-bottom: 0.5rem;'>Quick Insights</span>", unsafe_allow_html=True)
        df_p = st.session_state.get("patient_data")
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
                    st.markdown(
                        f"""<div style='font-size: 0.8125rem; line-height: 1.6;'>
                            <div style='margin-bottom: 0.5rem;'>
                                <span style='color: #94a3b8;'>Mean:</span>
                                <strong style='color: #3b82f6;'>{mean_v:,.0f}</strong>
                            </div>
                            <div>
                                <span style='color: #94a3b8;'>Volatility:</span>
                                <strong style='color: #a78bfa;'>{ratio:,.2f}</strong>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("No data", unsafe_allow_html=True)
            else:
                st.caption("No data", unsafe_allow_html=True)
        else:
            st.caption("Load data", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Activity Chart - Full Width Below
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='section-header'>📈 Recent Activity</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='dashboard-kpi-card' style='padding: 1.25rem 1.5rem;'>
        <span class='kpi-label' style='margin-bottom: 1rem;'>Last 30 Days Trend</span>
    """, unsafe_allow_html=True)
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
                    st.line_chart(recent.set_index("dt")[["y"]], use_container_width=True, height=200)
                else:
                    st.caption("No valid time series points detected yet.")
            else:
                st.caption("Waiting for a valid date/count column to render a sparkline.")
        else:
            st.caption("Data not loaded.")
    except Exception as e:
        st.caption(f"Unable to draw sparkline: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


    # ========================================
    # PATIENT ARRIVAL FORECAST SECTION (Horizontal Design)
    # ========================================
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='section-header'>🔮 Patient Arrival Forecast</h2>", unsafe_allow_html=True)

    # Enhanced Forecast CSS - Power BI Style
    st.markdown("""
    <style>
    .forecast-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
        border-radius: 16px;
        padding: 2rem 1.75rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.35),
            0 0 48px rgba(59, 130, 246, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .forecast-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(59, 130, 246, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .forecast-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #3b82f6, #22d3ee, #3b82f6, transparent);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .forecast-main {
        text-align: center;
        padding: 1.25rem 1rem;
    }

    .forecast-date {
        font-size: 1.1875rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .forecast-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.75rem 0;
        line-height: 1;
        letter-spacing: -1px;
        filter: drop-shadow(0 2px 8px rgba(59, 130, 246, 0.3));
    }

    .forecast-label {
        font-size: 0.8125rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .forecast-row {
        display: flex;
        align-items: center;
        padding: 1rem 1.25rem;
        margin: 0.375rem 0;
        border-radius: 10px;
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.08);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .forecast-row:hover {
        background: rgba(59, 130, 246, 0.08);
        border-color: rgba(59, 130, 246, 0.2);
        transform: translateX(6px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .forecast-day {
        flex: 0 0 75px;
        font-weight: 700;
        font-size: 0.9375rem;
        color: #e2e8f0;
        letter-spacing: 0.3px;
    }

    .forecast-date-small {
        flex: 0 0 85px;
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .forecast-predicted {
        flex: 1;
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: -0.3px;
    }

    .forecast-confidence {
        flex: 0 0 120px;
        text-align: right;
    }

    .forecast-accuracy-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #10b981;
        letter-spacing: 0.3px;
    }

    .forecast-mini-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check for trained models
    arima_results = st.session_state.get("arima_mh_results")
    sarimax_results = st.session_state.get("sarimax_results")

    models_available = []
    if arima_results:
        models_available.append("ARIMA")
    if sarimax_results:
        models_available.append("SARIMAX")

    if not models_available:
        st.markdown("<div class='dashboard-kpi-card' style='padding: 2rem; text-align: center;'>", unsafe_allow_html=True)
        st.info("🎯 Train ARIMA or SARIMAX models to view patient arrival forecasts. Visit the Benchmarks or Modeling Hub pages to train models.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # --- CONTROLS ---
        st.markdown("""<div class='dashboard-kpi-card' style='padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;'>
            <span class='kpi-label' style='margin-bottom: 1rem;'>Forecast Configuration</span>
        """, unsafe_allow_html=True)
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1, 1.2, 1])

        with ctrl_col1:
            selected_model = st.selectbox(
                "Select Model",
                options=models_available,
                key="forecast_model_selector",
                help="Choose which model to use for forecasting"
            )

        with ctrl_col2:
            forecast_days = st.selectbox(
                "Forecast Period",
                options=[1, 3, 7],
                format_func=lambda x: f"{x}-Day Forecast",
                key="forecast_period_selector",
                help="Select how many days ahead to forecast"
            )

        # Set date range to full year 2023
        min_date = datetime.date(2023, 1, 1)
        max_date = datetime.date(2023, 12, 31)
        default_date = datetime.date(2023, 12, 31)

        with ctrl_col3:
            selected_date = st.date_input(
                "Forecast From",
                value=default_date,
                min_value=min_date,
                max_value=max_date,
                key="forecast_date_selector",
                help="Select a date from the test set to start the forecast from. The first prediction will be for the next day."
            )

        with ctrl_col4:
            st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
            run_forecast = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


        # --- FORECAST LOGIC & DATA EXTRACTION ---
        if run_forecast or st.session_state.get("forecast_data_cache"):
            if run_forecast:
                # This block runs only when the button is pressed, to update the cache
                model_results = arima_results if selected_model == "ARIMA" else sarimax_results
                F, L, U, test_eval, metrics_df = None, None, None, None, None
                try:
                    if selected_model == "ARIMA":
                        if isinstance(model_results, dict) and "per_h" in model_results:
                            per_h = model_results.get("per_h", {})
                            results_df = model_results.get("results_df")
                            successful = model_results.get("successful", [])
                            if per_h and successful:
                                horizons = sorted(successful)
                                H = len(horizons)
                                y_test_ref = per_h[horizons[0]].get("y_test")
                                T = len(y_test_ref) if y_test_ref is not None else 0
                                if T > 0:
                                    F, L, U = np.full((T, H), np.nan), np.full((T, H), np.nan), np.full((T, H), np.nan)
                                    test_eval_dict = {}
                                    for idx, h in enumerate(horizons):
                                        h_data = per_h[h]
                                        F[:, idx] = h_data.get("forecast", [np.nan]*T)
                                        L[:, idx] = h_data.get("ci_lo", [np.nan]*T)
                                        U[:, idx] = h_data.get("ci_hi", [np.nan]*T)
                                    test_eval = pd.DataFrame(index=y_test_ref.index) # Preserve the date index
                                    metrics_df = results_df
                        else:
                            st.error("ARIMA model results are not in the expected format. Please re-train the model.")
                    else: # SARIMAX
                        from app_core.models.sarimax_pipeline import to_multihorizon_artifacts
                        metrics_df, F, L, U, test_eval, _, _, _ = to_multihorizon_artifacts(model_results)

                    if F is not None and test_eval is not None:
                        st.session_state["forecast_data_cache"] = {
                            "model": selected_model, "F": F, "L": L, "U": U,
                            "test_eval": test_eval, "metrics_df": metrics_df,
                        }
                    else:
                        st.error(f"Failed to extract valid forecast data for {selected_model}.")
                        st.session_state["forecast_data_cache"] = None

                except Exception as e:
                    st.error(f"An error occurred while processing model data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state["forecast_data_cache"] = None

            # --- DISPLAY FORECAST ---
            forecast_cache = st.session_state.get("forecast_data_cache")
            if forecast_cache:
                F, L, U, test_eval, metrics_df = (
                    forecast_cache["F"], forecast_cache["L"], forecast_cache["U"],
                    forecast_cache["test_eval"], forecast_cache["metrics_df"]
                )
                cached_model = forecast_cache["model"]

                # --- Data Handling for Metrics ---
                # Sanitize metrics_df to handle different column names and formats
                accuracy_col = None
                if metrics_df is not None:
                    for col in ["Accuracy_%", "Test_Acc"]:
                        if col in metrics_df.columns:
                            accuracy_col = col
                            break
                    # Sanitize Horizon column to be integer, filtering out non-numeric rows like "Average"
                    if "Horizon" in metrics_df.columns:
                        if metrics_df["Horizon"].dtype == 'object':
                            metrics_df["Horizon"] = metrics_df["Horizon"].str.replace("h=", "")
                        # Convert to numeric, filtering out non-numeric values (like "Average")
                        metrics_df["Horizon"] = pd.to_numeric(metrics_df["Horizon"], errors='coerce')
                        metrics_df = metrics_df.dropna(subset=["Horizon"])
                        metrics_df["Horizon"] = metrics_df["Horizon"].astype(int)

                # Find the index corresponding to the selected date
                forecast_idx = -1
                if isinstance(test_eval.index, pd.DatetimeIndex):
                    try:
                        selected_ts = pd.Timestamp(selected_date)
                        idx_loc = test_eval.index.get_indexer([selected_ts], method='nearest')[0]
                        forecast_idx = int(idx_loc)
                    except (KeyError, IndexError):
                        st.warning("Selected date not found in the test set. Using the last available date.")
                        forecast_idx = len(test_eval) - 1
                else:
                    st.error("Test data is missing a valid date index. Cannot select forecast date.")
                    forecast_idx = len(test_eval) - 1 # Fallback

                if forecast_idx != -1:
                    base_date = test_eval.index[forecast_idx]

                    # --- Horizontal Forecast Timeline ---
                    st.markdown(f"""
                    <div class='forecast-card' style='padding: 1.5rem 2rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
                            <div>
                                <div style='font-size: 1.125rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.25rem;'>
                                    {forecast_days}-Day Forecast Timeline
                                </div>
                                <div style='font-size: 0.8125rem; color: #94a3b8;'>
                                    Model: <span style='color: #22d3ee; font-weight: 600;'>{cached_model}</span> |
                                    From: <span style='color: #60a5fa; font-weight: 600;'>{base_date.strftime("%b %d, %Y")}</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Horizontal forecast cards
                    max_h = F.shape[1]
                    forecast_cols = st.columns(min(forecast_days, max_h, 7))  # Max 7 columns to fit screen

                    for h in range(min(forecast_days, max_h)):
                        forecast_dt = base_date + pd.Timedelta(days=h+1)
                        forecast_val = float(F[forecast_idx, h])
                        lower_val = float(L[forecast_idx, h])
                        upper_val = float(U[forecast_idx, h])

                        # Get accuracy for the current horizon h+1
                        horizon_accuracy = np.nan
                        if metrics_df is not None and accuracy_col is not None and "Horizon" in metrics_df.columns:
                            h_row = metrics_df[metrics_df["Horizon"] == (h + 1)]
                            if not h_row.empty:
                                horizon_accuracy = h_row[accuracy_col].iloc[0]

                        accuracy_text = f"{horizon_accuracy:.0f}%" if pd.notna(horizon_accuracy) else "N/A"

                        # Use modulo to wrap if more than 7 days
                        col_idx = h % 7
                        with forecast_cols[col_idx]:
                            is_primary = (h == 0)
                            bg_color = "rgba(59, 130, 246, 0.12)" if is_primary else "rgba(30, 41, 59, 0.4)"
                            border_color = "rgba(59, 130, 246, 0.35)" if is_primary else "rgba(59, 130, 246, 0.15)"

                            st.markdown(f"""
                            <div style='
                                background: {bg_color};
                                border: 1px solid {border_color};
                                border-radius: 12px;
                                padding: 1rem 0.75rem;
                                text-align: center;
                                transition: all 0.3s ease;
                                margin-bottom: 0.5rem;
                            ' class='forecast-mini-card'>
                                <div style='font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: 600;'>
                                    {forecast_dt.strftime("%a")}
                                </div>
                                <div style='font-size: 0.6875rem; color: #94a3b8; margin-bottom: 0.75rem;'>
                                    {forecast_dt.strftime("%m/%d")}
                                </div>
                                <div style='font-size: {"2.25rem" if is_primary else "1.75rem"}; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0; line-height: 1;'>
                                    {int(forecast_val)}
                                </div>
                                <div style='font-size: 0.625rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem;'>
                                    Patients
                                </div>
                                <div style='font-size: 0.625rem; color: #10b981; background: rgba(16, 185, 129, 0.12); padding: 0.25rem 0.5rem; border-radius: 6px; display: inline-block; margin-bottom: 0.5rem;'>
                                    {accuracy_text}
                                </div>
                                <div style='font-size: 0.625rem; color: #94a3b8;'>
                                    {int(lower_val)}-{int(upper_val)}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)



init_state()
page_dashboard()
