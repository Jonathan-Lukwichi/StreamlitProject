# =============================================================================
# 01_Dashboard.py - COMMAND CENTER
# Futuristic Navigation Hub with All Pages
# =============================================================================
"""
COMMAND CENTER - Futuristic Navigation Dashboard

Central hub with stunning visuals and quick access to all application pages.
Features fluorescent/neon design, animated effects, and organized workflow navigation.
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
# FUTURISTIC CSS - FLUORESCENT NEON THEME
# =============================================================================
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   FUTURISTIC FLUORESCENT THEME - COMMAND CENTER
   ═══════════════════════════════════════════════════════════════════════════ */

/* Animated Background Particles */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.3; }
    50% { transform: translateY(-20px) rotate(180deg); opacity: 0.8; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), 0 0 40px rgba(0, 255, 255, 0.1); }
    50% { box-shadow: 0 0 40px rgba(0, 255, 255, 0.5), 0 0 80px rgba(0, 255, 255, 0.2); }
}

@keyframes scan-line {
    0% { top: -10%; }
    100% { top: 110%; }
}

@keyframes text-glow {
    0%, 100% { text-shadow: 0 0 10px currentColor, 0 0 20px currentColor; }
    50% { text-shadow: 0 0 20px currentColor, 0 0 40px currentColor, 0 0 60px currentColor; }
}

@keyframes border-pulse {
    0%, 100% { border-color: rgba(0, 255, 255, 0.3); }
    50% { border-color: rgba(0, 255, 255, 0.8); }
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating Orbs Background */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
    z-index: 0;
    animation: float 20s ease-in-out infinite;
}

.orb-1 {
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0, 255, 255, 0.15), transparent 70%);
    top: 10%; left: 10%;
    animation-delay: 0s;
}

.orb-2 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.12), transparent 70%);
    top: 60%; right: 10%;
    animation-delay: 5s;
}

.orb-3 {
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.15), transparent 70%);
    bottom: 20%; left: 30%;
    animation-delay: 10s;
}

/* Hero Header */
.hero-container {
    position: relative;
    background: linear-gradient(135deg,
        rgba(0, 20, 40, 0.95) 0%,
        rgba(0, 40, 60, 0.9) 50%,
        rgba(0, 20, 40, 0.95) 100%);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 20px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    overflow: hidden;
    animation: border-pulse 3s ease-in-out infinite;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    animation: gradient-shift 3s ease infinite;
    background-size: 200% 200%;
}

.hero-container::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.5), transparent);
    animation: scan-line 4s linear infinite;
}

.hero-eyebrow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.status-dot {
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: pulse-glow 2s ease-in-out infinite;
    box-shadow: 0 0 10px #00ff88;
}

.status-text {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00ffff;
    animation: text-glow 2s ease-in-out infinite;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #00ffff 0%, #00ff88 50%, #00ffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% 200%;
    animation: gradient-shift 4s ease infinite;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.5rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(0, 255, 255, 0.2);
}

.section-icon {
    font-size: 1.5rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00ffff;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Navigation Cards */
.nav-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
}

.nav-card {
    position: relative;
    background: linear-gradient(135deg, rgba(0, 30, 50, 0.8), rgba(0, 20, 40, 0.9));
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
    cursor: pointer;
}

.nav-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.nav-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0, 255, 255, 0.6);
    box-shadow:
        0 10px 40px rgba(0, 255, 255, 0.15),
        0 0 20px rgba(0, 255, 255, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.nav-card:hover::before {
    opacity: 1;
}

.nav-card-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 10px currentColor);
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
    line-height: 1.4;
}

.nav-card-number {
    position: absolute;
    top: 1rem; right: 1rem;
    font-size: 0.65rem;
    font-weight: 800;
    color: rgba(0, 255, 255, 0.4);
    letter-spacing: 1px;
}

/* Category Labels */
.category-label {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(0, 255, 255, 0.05));
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00ffff;
    margin-bottom: 1rem;
}

/* KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(0, 30, 50, 0.8), rgba(0, 20, 40, 0.9));
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(0, 255, 255, 0.5);
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.1);
}

.kpi-icon {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #00ffff;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
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

.kpi-status.good { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.kpi-status.warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.kpi-status.neutral { background: rgba(100, 116, 139, 0.2); color: #94a3b8; }

/* Workflow Timeline */
.workflow-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    position: relative;
    padding: 2rem 0;
}

.workflow-container::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 5%;
    right: 5%;
    height: 2px;
    background: linear-gradient(90deg,
        rgba(0, 255, 255, 0.1),
        rgba(0, 255, 255, 0.5),
        rgba(16, 185, 129, 0.5),
        rgba(139, 92, 246, 0.5),
        rgba(139, 92, 246, 0.1));
    z-index: 0;
}

.workflow-step {
    text-align: center;
    z-index: 1;
    flex: 1;
}

.workflow-number {
    width: 50px; height: 50px;
    margin: 0 auto 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(0, 40, 60, 0.9), rgba(0, 20, 40, 0.95));
    border: 2px solid rgba(0, 255, 255, 0.5);
    border-radius: 50%;
    font-size: 1.25rem;
    font-weight: 800;
    color: #00ffff;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
}

.workflow-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}

.workflow-desc {
    font-size: 0.75rem;
    color: #64748b;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(0, 255, 255, 0.1);
}

.footer-text {
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 2px;
}

.footer-brand {
    font-size: 0.9rem;
    font-weight: 700;
    color: #00ffff;
    margin-top: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .kpi-container { grid-template-columns: repeat(2, 1fr); }
    .workflow-container { flex-direction: column; gap: 1rem; }
    .workflow-container::before { display: none; }
}
</style>

<!-- Floating Orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_system_status() -> Dict[str, Any]:
    """Get current system status from session state."""
    has_data = "processed_df" in st.session_state or "merged_data" in st.session_state
    has_forecast = any(k.startswith("ml_mh_results_") or k in ["arima_mh_results", "sarimax_results"]
                       for k in st.session_state.keys() if st.session_state.get(k))
    has_features = "feature_engineering" in st.session_state

    return {
        "data_loaded": has_data,
        "forecast_ready": has_forecast,
        "features_ready": has_features,
        "health_score": sum([has_data * 35, has_forecast * 40, has_features * 25])
    }


# =============================================================================
# HERO HEADER
# =============================================================================
status = get_system_status()

st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">
        <div class="status-dot"></div>
        <span class="status-text">System Online</span>
    </div>
    <h1 class="hero-title">COMMAND CENTER</h1>
    <p class="hero-subtitle">
        Central navigation hub for HealthForecast AI. Access all modules,
        monitor system status, and streamline your workflow.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# KPI STATUS CARDS
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    data_status = "good" if status["data_loaded"] else "neutral"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">{"✓" if status["data_loaded"] else "○"}</div>
        <div class="kpi-label">Data Status</div>
        <div class="kpi-status {data_status}">{"Loaded" if status["data_loaded"] else "Not Loaded"}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    feat_status = "good" if status["features_ready"] else "neutral"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🛠️</div>
        <div class="kpi-value">{"✓" if status["features_ready"] else "○"}</div>
        <div class="kpi-label">Features</div>
        <div class="kpi-status {feat_status}">{"Ready" if status["features_ready"] else "Not Ready"}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    forecast_status = "good" if status["forecast_ready"] else "neutral"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🔮</div>
        <div class="kpi-value">{"✓" if status["forecast_ready"] else "○"}</div>
        <div class="kpi-label">Forecast</div>
        <div class="kpi-status {forecast_status}">{"Ready" if status["forecast_ready"] else "Not Ready"}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    health_class = "good" if status["health_score"] >= 70 else ("warning" if status["health_score"] >= 35 else "neutral")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-value">{status["health_score"]}%</div>
        <div class="kpi-label">Health Score</div>
        <div class="kpi-status {health_class}">{"Excellent" if status["health_score"] >= 70 else ("Good" if status["health_score"] >= 35 else "Setup Needed")}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# WORKFLOW TIMELINE
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">🔄</span>
    <span class="section-title">Workflow Pipeline</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow-container">
    <div class="workflow-step">
        <div class="workflow-number">1</div>
        <div class="workflow-title">Upload</div>
        <div class="workflow-desc">Import data</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-number">2</div>
        <div class="workflow-title">Prepare</div>
        <div class="workflow-desc">Clean & fuse</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-number">3</div>
        <div class="workflow-title">Explore</div>
        <div class="workflow-desc">Analyze patterns</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-number">4</div>
        <div class="workflow-title">Train</div>
        <div class="workflow-desc">Build models</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-number">5</div>
        <div class="workflow-title">Forecast</div>
        <div class="workflow-desc">Predict demand</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-number">6</div>
        <div class="workflow-title">Optimize</div>
        <div class="workflow-desc">Plan resources</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


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
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 02</div>
        <div class="nav-card-icon">📁</div>
        <div class="nav-card-title">Upload Data</div>
        <div class="nav-card-desc">Import patient data from CSV, Excel, or Parquet files. Connect to external APIs.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Upload Data →", key="nav_upload", use_container_width=True):
        st.switch_page("pages/02_Upload_Data.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 03</div>
        <div class="nav-card-icon">🔧</div>
        <div class="nav-card-title">Prepare Data</div>
        <div class="nav-card-desc">Clean, transform, and fuse multiple data sources. Handle missing values.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Prepare Data →", key="nav_prepare", use_container_width=True):
        st.switch_page("pages/03_Prepare_Data.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 04</div>
        <div class="nav-card-icon">🔍</div>
        <div class="nav-card-title">Explore Data</div>
        <div class="nav-card-desc">Visualize patterns, analyze distributions, and discover insights in your data.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Explore Data →", key="nav_explore", use_container_width=True):
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
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 05</div>
        <div class="nav-card-icon">📈</div>
        <div class="nav-card-title">Baseline Models</div>
        <div class="nav-card-desc">Train ARIMA & SARIMAX statistical time series models.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Baseline →", key="nav_baseline", use_container_width=True):
        st.switch_page("pages/05_Baseline_Models.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 06</div>
        <div class="nav-card-icon">🛠️</div>
        <div class="nav-card-title">Feature Studio</div>
        <div class="nav-card-desc">Engineer lag features, rolling stats, and target variables.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Features →", key="nav_features", use_container_width=True):
        st.switch_page("pages/06_Feature_Studio.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 07</div>
        <div class="nav-card-icon">🎯</div>
        <div class="nav-card-title">Feature Selection</div>
        <div class="nav-card-desc">Automated feature importance analysis and selection.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Selection →", key="nav_selection", use_container_width=True):
        st.switch_page("pages/07_Feature_Selection.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 08</div>
        <div class="nav-card-icon">🧠</div>
        <div class="nav-card-title">Train Models</div>
        <div class="nav-card-desc">XGBoost, LSTM, ANN, Hybrid models with hyperparameter tuning.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Training →", key="nav_train", use_container_width=True):
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
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 09</div>
        <div class="nav-card-icon">📊</div>
        <div class="nav-card-title">Model Results</div>
        <div class="nav-card-desc">Compare model performance metrics, analyze accuracy, and review diagnostics across all trained models.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Results →", key="nav_results", use_container_width=True):
        st.switch_page("pages/09_Model_Results.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 10</div>
        <div class="nav-card-icon">🔮</div>
        <div class="nav-card-title">Patient Forecast</div>
        <div class="nav-card-desc">Generate 7-day patient arrival forecasts with clinical category breakdowns and confidence intervals.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Forecast →", key="nav_forecast", use_container_width=True):
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
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 11</div>
        <div class="nav-card-icon">👥</div>
        <div class="nav-card-title">Staff Planner</div>
        <div class="nav-card-desc">Optimize staff schedules using MILP solver. Balance workloads and reduce labor costs.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Staff Planner →", key="nav_staff", use_container_width=True):
        st.switch_page("pages/11_Staff_Planner.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 12</div>
        <div class="nav-card-icon">📦</div>
        <div class="nav-card-title">Supply Planner</div>
        <div class="nav-card-desc">Optimize inventory levels, calculate safety stock, and manage reorder points.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Supply Planner →", key="nav_supply", use_container_width=True):
        st.switch_page("pages/12_Supply_Planner.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-number">PAGE 13</div>
        <div class="nav-card-icon">✅</div>
        <div class="nav-card-title">Action Center</div>
        <div class="nav-card-desc">View AI-powered recommendations and prioritized actions for operations.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Action Center →", key="nav_actions", use_container_width=True):
        st.switch_page("pages/13_Action_Center.py")


# =============================================================================
# QUICK ACCESS HOME
# =============================================================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;">
            Return to Landing
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.switch_page("Welcome.py")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <div class="footer-text">POWERED BY ADVANCED MACHINE LEARNING & OPERATIONS RESEARCH</div>
    <div class="footer-brand">HealthForecast AI · Command Center v2.0</div>
</div>
""", unsafe_allow_html=True)
