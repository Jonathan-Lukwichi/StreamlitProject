# =============================================================================
# 01_Dashboard.py - Executive Summary Dashboard
# Lightweight overview with links to detailed pages
# =============================================================================
"""
Executive Summary Dashboard - Quick overview of system status.

This is a SUMMARY-ONLY page that displays:
- Key Performance Indicators (KPIs)
- System Health Score
- Quick navigation to detailed pages

For detailed work, use the dedicated pages:
- Patient Forecast (Page 10)
- Staff Planner (Page 11)
- Supply Planner (Page 12)
- Action Center (Page 13)
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
    page_title="Dashboard - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# =============================================================================
# APPLY STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CUSTOM CSS FOR EXECUTIVE DASHBOARD
# =============================================================================
st.markdown("""
<style>
/* Executive Dashboard Styles */
.exec-header {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}

.exec-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.exec-subtitle {
    color: #94a3b8;
    font-size: 1rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    transform: translateY(-2px);
}

.kpi-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f1f5f9;
}

.kpi-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

.kpi-status {
    font-size: 0.7rem;
    margin-top: 0.5rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
}

.status-good { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
.status-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.status-error { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.status-neutral { background: rgba(100, 116, 139, 0.2); color: #94a3b8; }

.nav-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(71, 85, 105, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.nav-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
}

.nav-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}

.nav-card-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}

.health-score {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

.health-value {
    font-size: 3rem;
    font-weight: 800;
}

.health-good { color: #22c55e; }
.health-warning { color: #f59e0b; }
.health-error { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS - GET DATA FROM SESSION STATE
# =============================================================================
def get_forecast_summary() -> Dict[str, Any]:
    """Get forecast summary from session state."""
    result = {
        "has_data": False,
        "total_week": 0,
        "avg_daily": 0,
        "model_name": "None",
        "status": "No forecast available"
    }

    # Check for any forecast results
    for key in ["arima_mh_results", "sarimax_results"]:
        if key in st.session_state and st.session_state[key]:
            data = st.session_state[key]
            if "per_h" in data and data["per_h"]:
                result["has_data"] = True
                result["model_name"] = "ARIMA" if "arima" in key else "SARIMAX"
                # Get forecasts
                forecasts = []
                for h, h_data in data["per_h"].items():
                    if "forecast" in h_data and h_data["forecast"] is not None:
                        forecasts.extend(h_data["forecast"][:7])
                if forecasts:
                    result["total_week"] = sum(forecasts[:7])
                    result["avg_daily"] = result["total_week"] / 7
                    result["status"] = f"{result['model_name']} model"
                break

    # Check ML models
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_") and st.session_state[key]:
            data = st.session_state[key]
            if "per_h" in data and data["per_h"]:
                result["has_data"] = True
                result["model_name"] = key.replace("ml_mh_results_", "")
                forecasts = []
                for h, h_data in data["per_h"].items():
                    if "forecast" in h_data and h_data["forecast"] is not None:
                        forecasts.extend(h_data["forecast"][:7])
                if forecasts:
                    result["total_week"] = sum(forecasts[:7])
                    result["avg_daily"] = result["total_week"] / 7
                    result["status"] = f"{result['model_name']} model"
                break

    return result


def get_staff_summary() -> Dict[str, Any]:
    """Get staff summary from session state."""
    result = {
        "has_data": False,
        "utilization": 0,
        "status": "No data loaded"
    }

    if "staff_schedule_data" in st.session_state and st.session_state["staff_schedule_data"] is not None:
        result["has_data"] = True
        data = st.session_state["staff_schedule_data"]
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Calculate utilization if available
            if "utilization" in data.columns:
                result["utilization"] = data["utilization"].mean()
            result["status"] = "Data loaded"

    if "milp_solution" in st.session_state and st.session_state["milp_solution"]:
        result["has_data"] = True
        result["status"] = "Optimized"

    return result


def get_inventory_summary() -> Dict[str, Any]:
    """Get inventory summary from session state."""
    result = {
        "has_data": False,
        "service_level": 0,
        "critical_items": 0,
        "status": "No data loaded"
    }

    if "inventory_data" in st.session_state and st.session_state["inventory_data"] is not None:
        result["has_data"] = True
        data = st.session_state["inventory_data"]
        if isinstance(data, pd.DataFrame) and not data.empty:
            result["service_level"] = 95  # Default
            result["status"] = "Data loaded"

    if "inventory_optimization" in st.session_state and st.session_state["inventory_optimization"]:
        result["has_data"] = True
        result["status"] = "Optimized"

    return result


def calculate_health_score(forecast: Dict, staff: Dict, inventory: Dict) -> int:
    """Calculate overall system health score (0-100)."""
    score = 0

    # Forecast component (40 points)
    if forecast["has_data"]:
        score += 40

    # Staff component (30 points)
    if staff["has_data"]:
        score += 30

    # Inventory component (30 points)
    if inventory["has_data"]:
        score += 30

    return score


# =============================================================================
# GET CURRENT STATE
# =============================================================================
forecast_summary = get_forecast_summary()
staff_summary = get_staff_summary()
inventory_summary = get_inventory_summary()
health_score = calculate_health_score(forecast_summary, staff_summary, inventory_summary)


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="exec-header">
    <div class="exec-title">Executive Dashboard</div>
    <div class="exec-subtitle">Quick overview of HealthForecast AI system status</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SYSTEM HEALTH SCORE
# =============================================================================
col_health, col_spacer = st.columns([1, 2])

with col_health:
    health_class = "health-good" if health_score >= 70 else ("health-warning" if health_score >= 40 else "health-error")
    health_text = "Excellent" if health_score >= 70 else ("Good" if health_score >= 40 else "Needs Attention")

    st.markdown(f"""
    <div class="health-score">
        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">System Health</div>
        <div class="health-value {health_class}">{health_score}%</div>
        <div style="color: #64748b; font-size: 0.85rem;">{health_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =============================================================================
# KPI CARDS
# =============================================================================
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    forecast_status = "status-good" if forecast_summary["has_data"] else "status-neutral"
    forecast_value = f"{int(forecast_summary['total_week']):,}" if forecast_summary["has_data"] else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🔮</div>
        <div class="kpi-value">{forecast_value}</div>
        <div class="kpi-label">7-Day Forecast</div>
        <div class="kpi-status {forecast_status}">{forecast_summary['status']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    staff_status = "status-good" if staff_summary["has_data"] else "status-neutral"
    staff_value = f"{staff_summary['utilization']:.0f}%" if staff_summary["has_data"] and staff_summary["utilization"] > 0 else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-value">{staff_value}</div>
        <div class="kpi-label">Staff Utilization</div>
        <div class="kpi-status {staff_status}">{staff_summary['status']}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    inv_status = "status-good" if inventory_summary["has_data"] else "status-neutral"
    inv_value = f"{inventory_summary['service_level']:.0f}%" if inventory_summary["has_data"] else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📦</div>
        <div class="kpi-value">{inv_value}</div>
        <div class="kpi-label">Service Level</div>
        <div class="kpi-status {inv_status}">{inventory_summary['status']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    data_ready = forecast_summary["has_data"] or staff_summary["has_data"] or inventory_summary["has_data"]
    data_status = "status-good" if data_ready else "status-warning"
    data_text = "Ready" if data_ready else "Load Data"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-value">{"✓" if data_ready else "○"}</div>
        <div class="kpi-label">Data Status</div>
        <div class="kpi-status {data_status}">{data_text}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# QUICK ACTIONS - NAVIGATION TO DETAILED PAGES
# =============================================================================
st.markdown("---")
st.markdown("### 🚀 Quick Actions")
st.markdown("*Click to navigate to detailed pages*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-title">🔮 Patient Forecast</div>
        <div class="nav-card-desc">View 7-day patient arrival predictions with category breakdowns and confidence intervals.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Patient Forecast →", key="nav_forecast", use_container_width=True):
        st.switch_page("pages/10_Patient_Forecast.py")

    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-title">📦 Supply Planner</div>
        <div class="nav-card-desc">Optimize inventory levels, calculate safety stock, and manage reorder points.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Supply Planner →", key="nav_supply", use_container_width=True):
        st.switch_page("pages/12_Supply_Planner.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-title">👥 Staff Planner</div>
        <div class="nav-card-desc">Optimize staff schedules, balance workloads, and reduce labor costs with MILP solver.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Staff Planner →", key="nav_staff", use_container_width=True):
        st.switch_page("pages/11_Staff_Planner.py")

    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-title">✅ Action Center</div>
        <div class="nav-card-desc">View prioritized recommendations and AI-powered insights for operational decisions.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Action Center →", key="nav_actions", use_container_width=True):
        st.switch_page("pages/13_Action_Center.py")


# =============================================================================
# WORKFLOW GUIDE
# =============================================================================
st.markdown("---")
st.markdown("### 📋 Workflow Guide")

workflow_cols = st.columns(4)

with workflow_cols[0]:
    st.markdown("""
    **Step 1: Data**
    - Upload patient data
    - Prepare & clean data
    - Explore patterns
    """)
    if st.button("📁 Upload Data", key="wf_upload", use_container_width=True):
        st.switch_page("pages/02_Upload_Data.py")

with workflow_cols[1]:
    st.markdown("""
    **Step 2: Features**
    - Engineer features
    - Select best features
    - Validate inputs
    """)
    if st.button("🛠️ Feature Studio", key="wf_features", use_container_width=True):
        st.switch_page("pages/06_Feature_Studio.py")

with workflow_cols[2]:
    st.markdown("""
    **Step 3: Train**
    - Train baseline models
    - Train ML models
    - Compare results
    """)
    if st.button("🧠 Train Models", key="wf_train", use_container_width=True):
        st.switch_page("pages/08_Train_Models.py")

with workflow_cols[3]:
    st.markdown("""
    **Step 4: Optimize**
    - Forecast patients
    - Optimize staff
    - Plan inventory
    """)
    if st.button("🔮 Forecast", key="wf_forecast", use_container_width=True):
        st.switch_page("pages/10_Patient_Forecast.py")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem;">
    HealthForecast AI · Executive Dashboard · Quick Overview Mode
</div>
""", unsafe_allow_html=True)
