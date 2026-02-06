# =============================================================================
# 01_Dashboard.py - COMMAND CENTER
# Blue Fluorescent Theme with Intelligent Status Tracking
# =============================================================================
"""
COMMAND CENTER - Central Navigation Hub

Features:
- Blue fluorescent color scheme
- Static design (no animations)
- Intelligent status tracking (Pending/Complete for each page)
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# AUTHENTICATION & CORE IMPORTS
# =============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Command Center - HealthForecast AI",
    page_icon="🚀",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# =============================================================================
# APPLY BASE STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# BLUE FLUORESCENT CSS - STATIC DESIGN
# =============================================================================
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   BLUE FLUORESCENT THEME - STATIC DESIGN
   ═══════════════════════════════════════════════════════════════════════════ */

/* Hero Header */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 58, 95, 0.95) 50%, rgba(15, 23, 42, 0.98) 100%);
    border: 1px solid rgba(79, 143, 252, 0.4);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(79, 143, 252, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.hero-eyebrow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: #4f8ffc;
    border-radius: 50%;
    box-shadow: 0 0 10px #4f8ffc, 0 0 20px rgba(79, 143, 252, 0.5);
}

.status-text {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4f8ffc;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: #4f8ffc;
    text-shadow: 0 0 30px rgba(79, 143, 252, 0.5);
    margin-bottom: 0.5rem;
    letter-spacing: 3px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.5rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(79, 143, 252, 0.3);
}

.section-icon {
    font-size: 1.5rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #4f8ffc;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Navigation Cards */
.nav-card {
    position: relative;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
    border: 1px solid rgba(79, 143, 252, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 180px;
    transition: all 0.3s ease;
}

.nav-card:hover {
    border-color: rgba(79, 143, 252, 0.6);
    box-shadow: 0 0 30px rgba(79, 143, 252, 0.15);
    transform: translateY(-3px);
}

.nav-card-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.nav-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}

.nav-card-desc {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.5;
}

/* Status Badges */
.status-badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.35rem 0.6rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.status-badge.complete {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22c55e;
}

.status-badge.pending {
    background: rgba(100, 116, 139, 0.15);
    border: 1px solid rgba(100, 116, 139, 0.3);
    color: #94a3b8;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
    border: 1px solid rgba(79, 143, 252, 0.25);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(79, 143, 252, 0.5);
    box-shadow: 0 0 20px rgba(79, 143, 252, 0.1);
}

.kpi-icon {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #4f8ffc;
}

.kpi-label {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

.kpi-status {
    display: inline-block;
    font-size: 0.6rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    margin-top: 0.5rem;
}

.kpi-status.good { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
.kpi-status.warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.kpi-status.neutral { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(79, 143, 252, 0.15);
}

.footer-text {
    font-size: 0.7rem;
    color: #475569;
    letter-spacing: 2px;
}

.footer-brand {
    font-size: 0.85rem;
    font-weight: 700;
    color: #4f8ffc;
    margin-top: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INTELLIGENT STATUS TRACKING
# =============================================================================
def get_page_statuses() -> Dict[str, bool]:
    """Check completion status for each page based on session state."""

    def safe_check(key: str) -> bool:
        """Safely check if a key exists and has valid data."""
        if key not in st.session_state:
            return False
        val = st.session_state[key]
        if val is None:
            return False
        if isinstance(val, pd.DataFrame):
            return not val.empty
        if isinstance(val, dict):
            return bool(val)
        return bool(val)

    def has_ml_results() -> bool:
        """Check if any ML model results exist."""
        for k in st.session_state.keys():
            if k.startswith("ml_mh_results_"):
                if safe_check(k):
                    return True
        return False

    return {
        "upload": safe_check("raw_data") or safe_check("uploaded_file") or safe_check("patient_data"),
        "prepare": safe_check("processed_df") or safe_check("merged_data"),
        "explore": safe_check("processed_df") or safe_check("merged_data"),
        "baseline": safe_check("arima_mh_results") or safe_check("sarimax_results"),
        "features": safe_check("feature_engineering") or safe_check("fe_data"),
        "selection": safe_check("selected_features") or safe_check("feature_importance"),
        "train": has_ml_results(),
        "results": has_ml_results() or safe_check("arima_mh_results") or safe_check("sarimax_results"),
        "forecast": safe_check("forecast_generated") or has_ml_results(),
        "staff": safe_check("staff_schedule_data") or safe_check("milp_solution"),
        "supply": safe_check("inventory_data") or safe_check("inventory_optimization"),
        "actions": True,  # Always accessible
    }


# =============================================================================
# GET STATUSES
# =============================================================================
page_statuses = get_page_statuses()


# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">
        <div class="status-dot"></div>
        <span class="status-text">System Online</span>
    </div>
    <h1 class="hero-title">COMMAND CENTER</h1>
    <p class="hero-subtitle">
        Central navigation hub for HealthForecast AI. Access all modules from one place.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# KPI STATUS CARDS
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    data_ok = page_statuses["prepare"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">{"✓" if data_ok else "○"}</div>
        <div class="kpi-label">Data Status</div>
        <div class="kpi-status {"good" if data_ok else "neutral"}">{"Loaded" if data_ok else "Not Loaded"}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    feat_ok = page_statuses["features"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🛠️</div>
        <div class="kpi-value">{"✓" if feat_ok else "○"}</div>
        <div class="kpi-label">Features</div>
        <div class="kpi-status {"good" if feat_ok else "neutral"}">{"Ready" if feat_ok else "Not Ready"}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    model_ok = page_statuses["train"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🧠</div>
        <div class="kpi-value">{"✓" if model_ok else "○"}</div>
        <div class="kpi-label">Models</div>
        <div class="kpi-status {"good" if model_ok else "neutral"}">{"Trained" if model_ok else "Not Trained"}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    forecast_ok = page_statuses["forecast"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🔮</div>
        <div class="kpi-value">{"✓" if forecast_ok else "○"}</div>
        <div class="kpi-label">Forecast</div>
        <div class="kpi-status {"good" if forecast_ok else "neutral"}">{"Ready" if forecast_ok else "Not Ready"}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTION FOR STATUS BADGE
# =============================================================================
def status_badge(is_complete: bool) -> str:
    if is_complete:
        return '<div class="status-badge complete">✓ Complete</div>'
    return '<div class="status-badge pending">○ Pending</div>'


# =============================================================================
# DATA ACQUISITION SECTION
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📥</span>
    <span class="section-title">Data Acquisition</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["upload"])}
        <div class="nav-card-icon">📁</div>
        <div class="nav-card-title">Upload Data</div>
        <div class="nav-card-desc">Import patient data from CSV, Excel, or Parquet files.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Upload Data", key="nav_upload", use_container_width=True):
        st.switch_page("pages/02_Upload_Data.py")

with col2:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["prepare"])}
        <div class="nav-card-icon">🔧</div>
        <div class="nav-card-title">Prepare Data</div>
        <div class="nav-card-desc">Clean, transform, and fuse multiple data sources.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Prepare Data", key="nav_prepare", use_container_width=True):
        st.switch_page("pages/03_Prepare_Data.py")

with col3:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["explore"])}
        <div class="nav-card-icon">🔍</div>
        <div class="nav-card-title">Explore Data</div>
        <div class="nav-card-desc">Visualize patterns and discover insights.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Explore Data", key="nav_explore", use_container_width=True):
        st.switch_page("pages/04_Explore_Data.py")


# =============================================================================
# MODEL TRAINING SECTION
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">🧠</span>
    <span class="section-title">Model Training</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["baseline"])}
        <div class="nav-card-icon">📈</div>
        <div class="nav-card-title">Baseline Models</div>
        <div class="nav-card-desc">ARIMA & SARIMAX time series models.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Baseline", key="nav_baseline", use_container_width=True):
        st.switch_page("pages/05_Baseline_Models.py")

with col2:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["features"])}
        <div class="nav-card-icon">🛠️</div>
        <div class="nav-card-title">Feature Studio</div>
        <div class="nav-card-desc">Engineer lag features and targets.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Features", key="nav_features", use_container_width=True):
        st.switch_page("pages/06_Feature_Studio.py")

with col3:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["selection"])}
        <div class="nav-card-icon">🎯</div>
        <div class="nav-card-title">Feature Selection</div>
        <div class="nav-card-desc">Automated feature importance.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Selection", key="nav_selection", use_container_width=True):
        st.switch_page("pages/07_Feature_Selection.py")

with col4:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["train"])}
        <div class="nav-card-icon">🧠</div>
        <div class="nav-card-title">Train Models</div>
        <div class="nav-card-desc">XGBoost, LSTM, ANN, Hybrids.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Training", key="nav_train", use_container_width=True):
        st.switch_page("pages/08_Train_Models.py")


# =============================================================================
# RESULTS & FORECASTING SECTION
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📊</span>
    <span class="section-title">Results & Forecasting</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["results"])}
        <div class="nav-card-icon">📊</div>
        <div class="nav-card-title">Model Results</div>
        <div class="nav-card-desc">Compare model performance, analyze accuracy, and review diagnostics.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Results", key="nav_results", use_container_width=True):
        st.switch_page("pages/09_Model_Results.py")

with col2:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["forecast"])}
        <div class="nav-card-icon">🔮</div>
        <div class="nav-card-title">Patient Forecast</div>
        <div class="nav-card-desc">7-day forecasts with clinical category breakdowns.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Forecast", key="nav_forecast", use_container_width=True):
        st.switch_page("pages/10_Patient_Forecast.py")


# =============================================================================
# OPTIMIZATION SECTION
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">⚙️</span>
    <span class="section-title">Resource Optimization</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["staff"])}
        <div class="nav-card-icon">👥</div>
        <div class="nav-card-title">Staff Planner</div>
        <div class="nav-card-desc">Optimize schedules with MILP solver.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Staff Planner", key="nav_staff", use_container_width=True):
        st.switch_page("pages/11_Staff_Planner.py")

with col2:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["supply"])}
        <div class="nav-card-icon">📦</div>
        <div class="nav-card-title">Supply Planner</div>
        <div class="nav-card-desc">Optimize inventory and safety stock.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Supply Planner", key="nav_supply", use_container_width=True):
        st.switch_page("pages/12_Supply_Planner.py")

with col3:
    st.markdown(f"""
    <div class="nav-card">
        {status_badge(page_statuses["actions"])}
        <div class="nav-card-icon">✅</div>
        <div class="nav-card-title">Action Center</div>
        <div class="nav-card-desc">AI recommendations and priorities.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Action Center", key="nav_actions", use_container_width=True):
        st.switch_page("pages/13_Action_Center.py")


# =============================================================================
# HOME BUTTON
# =============================================================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🏠 Return to Home", key="nav_home", use_container_width=True):
        st.switch_page("Welcome.py")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <div class="footer-text">POWERED BY MACHINE LEARNING & OPERATIONS RESEARCH</div>
    <div class="footer-brand">HealthForecast AI · Command Center</div>
</div>
""", unsafe_allow_html=True)
