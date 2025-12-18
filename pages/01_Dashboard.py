# =============================================================================
# 14_Executive_Dashboard.py - Unified Hospital Operations Dashboard
# Integrates: Patient Forecast, Staff Planner (Staff Planner),
#             Supply Planner (Supply Planner), Decision Command Center (Action Center)
# =============================================================================
"""
Executive Dashboard - Unified view combining all operational data

Tab Structure:
1. Executive Summary - High-level KPIs, financial impact, system health
2. Patient Forecast - 7-day forecast visualization from Patient Forecast
3. Staffing Overview - Staff metrics and optimization status from Staff Planner
4. Inventory Status - Inventory metrics and optimization from Supply Planner
5. Action Items - Prioritized actions and recommendations

Data Sources:
- Forecasts: Session state from Patient Forecast
- Staff data: Session state from Staff Planner (Staff Planner)
- Inventory data: Session state from Supply Planner (Supply Planner)
- All extracted via standardized functions
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.page_navigation import render_page_navigation

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Dashboard - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CUSTOM CSS - Premium Executive Dashboard Design
# =============================================================================
st.markdown(f"""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR EXECUTIVE DASHBOARD
   ======================================== */

@keyframes float-orb {{
    0%, 100% {{
        transform: translate(0, 0) scale(1);
        opacity: 0.25;
    }}
    50% {{
        transform: translate(30px, -30px) scale(1.05);
        opacity: 0.35;
    }}
}}

.fluorescent-orb {{
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}}

.orb-1 {{
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 10%;
    right: 15%;
    animation: float-orb 25s ease-in-out infinite;
}}

.orb-2 {{
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.2), transparent 70%);
    bottom: 15%;
    left: 10%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}}

.orb-3 {{
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.18), transparent 70%);
    top: 50%;
    left: 50%;
    animation: float-orb 35s ease-in-out infinite;
    animation-delay: 10s;
}}

@keyframes sparkle {{
    0%, 100% {{
        opacity: 0;
        transform: scale(0);
    }}
    50% {{
        opacity: 0.6;
        transform: scale(1);
    }}
}}

.sparkle {{
    position: fixed;
    width: 3px;
    height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}}

.sparkle-1 {{ top: 20%; left: 30%; animation-delay: 0s; }}
.sparkle-2 {{ top: 60%; left: 75%; animation-delay: 1s; }}
.sparkle-3 {{ top: 40%; left: 20%; animation-delay: 2s; }}
.sparkle-4 {{ top: 75%; left: 45%; animation-delay: 1.5s; }}

/* ========================================
   CUSTOM TAB STYLING (Executive Design)
   ======================================== */

.stTabs {{
    background: transparent;
    margin-top: 1rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.7), rgba(5, 8, 22, 0.6));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(59, 130, 246, 0.25);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}

.stTabs [data-baseweb="tab"] {{
    height: 50px;
    background: transparent;
    border-radius: 12px;
    padding: 0 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: {BODY_TEXT};
    border: 1px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 197, 94, 0.1));
    border-color: rgba(59, 130, 246, 0.3);
    color: {TEXT_COLOR};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(34, 197, 94, 0.2)) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    color: {TEXT_COLOR} !important;
    box-shadow:
        0 0 20px rgba(59, 130, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent;
}}

.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
}}

/* ========================================
   EXECUTIVE KPI CARDS
   ======================================== */

.exec-kpi-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    height: 100%;
}}

.exec-kpi-card:hover {{
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
}}

.exec-kpi-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color, #3b82f6), var(--accent-end, #22d3ee));
}}

.exec-kpi-icon {{
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}}

.exec-kpi-value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.25rem 0;
    line-height: 1.2;
}}

.exec-kpi-label {{
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 600;
}}

.exec-kpi-sublabel {{
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}}

/* Status Badges */
.status-good {{
    color: #22c55e;
    background: rgba(34, 197, 94, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}}

.status-warning {{
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}}

.status-alert {{
    color: #ef4444;
    background: rgba(239, 68, 68, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}}

/* Action Card */
.action-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border-left: 4px solid;
    transition: all 0.2s ease;
}}

.action-card:hover {{
    transform: translateX(5px);
}}

.action-urgent {{
    border-left-color: #ef4444;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(15, 23, 42, 0.95));
}}

.action-important {{
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(15, 23, 42, 0.95));
}}

.action-normal {{
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(15, 23, 42, 0.95));
}}

.action-icon {{
    font-size: 1.5rem;
    flex-shrink: 0;
}}

.action-content {{
    flex: 1;
}}

.action-title {{
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
    font-size: 0.95rem;
}}

.action-description {{
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}}

/* Data Status Box */
.data-status-box {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}}

.data-status-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(100, 116, 139, 0.2);
}}

.data-status-item:last-child {{
    border-bottom: none;
}}

/* Section Header */
.section-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
}}

.section-icon {{
    font-size: 1.5rem;
}}

.section-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
}}

/* Savings Card */
.savings-card {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
    border: 2px solid rgba(34, 197, 94, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}}

.savings-value {{
    font-size: 2.5rem;
    font-weight: 800;
    color: #22c55e;
    text-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
}}

.savings-label {{
    font-size: 1rem;
    color: #86efac;
    font-weight: 600;
}}

/* Day Card for Forecast */
.day-card {{
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}}

.day-card:hover {{
    border-color: rgba(59, 130, 246, 0.5);
}}

.day-card.today {{
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
}}

.day-card.high-volume {{
    border-color: #f59e0b;
    background: rgba(245, 158, 11, 0.1);
}}

.day-name {{
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
}}

.day-date {{
    font-size: 0.75rem;
    color: #64748b;
}}

.day-patients {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0.5rem 0;
}}

.day-label {{
    font-size: 0.7rem;
    color: #64748b;
}}

/* Quick Stats Grid */
.quick-stat {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(100, 116, 139, 0.2);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}}

.quick-stat-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
}}

.quick-stat-label {{
    font-size: 0.75rem;
    color: #94a3b8;
}}

/* Mobile Responsive */
@media (max-width: 768px) {{
    .fluorescent-orb {{
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }}
    .sparkle {{
        display: none;
    }}
    .exec-kpi-value {{ font-size: 1.5rem; }}
    .savings-value {{ font-size: 2rem; }}
    .day-patients {{ font-size: 1.25rem; }}
}}

/* ========================================
   DARK BLUE FLUORESCENT SIDEBAR STATUS CARDS
   ======================================== */

@keyframes sidebar-card-pulse {{
    0%, 100% {{
        box-shadow:
            0 0 10px rgba(6, 78, 145, 0.4),
            0 0 20px rgba(6, 78, 145, 0.2),
            inset 0 1px 0 rgba(34, 211, 238, 0.1);
    }}
    50% {{
        box-shadow:
            0 0 15px rgba(6, 78, 145, 0.6),
            0 0 30px rgba(6, 78, 145, 0.3),
            inset 0 1px 0 rgba(34, 211, 238, 0.2);
    }}
}}

@keyframes status-glow {{
    0%, 100% {{ text-shadow: 0 0 5px currentColor; }}
    50% {{ text-shadow: 0 0 15px currentColor, 0 0 25px currentColor; }}
}}

.sidebar-status-container {{
    background: linear-gradient(135deg, rgba(6, 78, 145, 0.15) 0%, rgba(15, 23, 42, 0.98) 50%, rgba(6, 78, 145, 0.1) 100%);
    border: 1px solid rgba(34, 211, 238, 0.3);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
    animation: sidebar-card-pulse 4s ease-in-out infinite;
}}

.sidebar-status-title {{
    text-align: center;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #22d3ee;
    margin-bottom: 0.75rem;
    text-shadow: 0 0 10px rgba(34, 211, 238, 0.5);
}}

.sidebar-status-card {{
    background: linear-gradient(135deg, #064e91 0%, #0a3d6e 40%, #041e42 100%);
    border: 1px solid rgba(34, 211, 238, 0.25);
    border-radius: 10px;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.3s ease;
}}

.sidebar-status-card:hover {{
    transform: translateX(3px);
    border-color: rgba(34, 211, 238, 0.5);
    box-shadow: 0 0 15px rgba(6, 78, 145, 0.4);
}}

.sidebar-status-card:last-child {{
    margin-bottom: 0;
}}

.sidebar-status-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.sidebar-status-value {{
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}}

.status-active {{
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.4);
    animation: status-glow 2s ease-in-out infinite;
}}

.status-inactive {{
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

.status-pending {{
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}}

/* Savings Card */
.sidebar-savings-card {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(6, 78, 145, 0.2) 50%, rgba(16, 185, 129, 0.1) 100%);
    border: 2px solid rgba(34, 197, 94, 0.4);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    animation: sidebar-card-pulse 4s ease-in-out infinite;
}}

.sidebar-savings-value {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #22c55e;
    text-shadow: 0 0 15px rgba(34, 197, 94, 0.5);
    animation: status-glow 2s ease-in-out infinite;
}}

.sidebar-savings-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: #86efac;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* Activity Tracker */
.sidebar-activity-container {{
    background: linear-gradient(135deg, rgba(6, 78, 145, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 12px;
    padding: 0.75rem;
    margin-top: 0.75rem;
}}

.sidebar-activity-title {{
    font-size: 0.7rem;
    font-weight: 700;
    color: #22d3ee;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.sidebar-activity-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.65rem;
    color: #94a3b8;
    border-bottom: 1px solid rgba(100, 116, 139, 0.15);
}}

.sidebar-activity-item:last-child {{
    border-bottom: none;
}}

.activity-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}}

.activity-dot.active {{
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
}}

.activity-dot.pending {{
    background: #fbbf24;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
}}

.activity-dot.inactive {{
    background: #64748b;
}}

/* Quick Links Card */
.sidebar-links-container {{
    background: linear-gradient(135deg, rgba(6, 78, 145, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 12px;
    padding: 0.75rem;
}}

.sidebar-link-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.5rem;
    margin-bottom: 0.25rem;
    border-radius: 8px;
    font-size: 0.7rem;
    color: #22d3ee;
    font-weight: 600;
    transition: all 0.2s ease;
    cursor: pointer;
}}

.sidebar-link-item:hover {{
    background: rgba(6, 78, 145, 0.3);
    transform: translateX(3px);
}}

.sidebar-link-item:last-child {{
    margin-bottom: 0;
}}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
<div class="fluorescent-orb orb-3"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
<div class="sparkle sparkle-4"></div>
""", unsafe_allow_html=True)


# =============================================================================
# DATA EXTRACTION FUNCTIONS - Aligned with Pages 10, 11, 12
# =============================================================================

def get_forecast_data() -> Dict[str, Any]:
    """
    Extract forecast data from session state.

    PRIORITY ORDER (synced with Patient Forecast Forecast Hub):
    1. forecast_hub_demand - Primary source from Patient Forecast
    2. active_forecast - Alternative key from Forecast Hub
    3. ML models (fallback)
    4. ARIMA (fallback)
    """
    result = {
        "has_forecast": False,
        "source": None,
        "model_type": None,
        "forecasts": [],
        "dates": [],
        "horizon": 0,
        "avg_daily": 0,
        "total_week": 0,
        "peak_day": None,
        "peak_patients": 0,
        "low_day": None,
        "low_patients": float('inf'),
        "trend": "stable",
        "confidence": "Unknown",
        "accuracy": None,
        "timestamp": None,
        "synced_with_page10": False,
    }

    # PRIORITY 1: Check forecast_hub_demand (from Patient Forecast - Forecast Hub)
    hub_demand = st.session_state.get("forecast_hub_demand")
    if hub_demand and isinstance(hub_demand, dict):
        forecasts = hub_demand.get("forecast", [])
        if forecasts and len(forecasts) > 0:
            result["has_forecast"] = True
            result["source"] = hub_demand.get("model", "Forecast Hub")
            result["model_type"] = hub_demand.get("model_type", "Unknown")
            result["forecasts"] = [max(0, float(f)) for f in forecasts[:7]]
            result["dates"] = hub_demand.get("dates", [])
            result["horizon"] = len(result["forecasts"])
            result["accuracy"] = hub_demand.get("avg_accuracy")
            result["timestamp"] = hub_demand.get("timestamp")
            result["synced_with_page10"] = True

    # PRIORITY 2: Check active_forecast (alternative from Patient Forecast)
    if not result["has_forecast"]:
        active = st.session_state.get("active_forecast")
        if active and isinstance(active, dict):
            forecasts = active.get("forecasts", [])
            if forecasts and len(forecasts) > 0:
                result["has_forecast"] = True
                result["source"] = active.get("model", "Active Forecast")
                result["model_type"] = active.get("model_type", "Unknown")
                result["forecasts"] = [max(0, float(f)) for f in forecasts[:7]]
                result["dates"] = active.get("dates", [])
                result["horizon"] = len(result["forecasts"])
                result["accuracy"] = active.get("accuracy")
                result["timestamp"] = active.get("updated_at")
                result["synced_with_page10"] = True

    # PRIORITY 3: Check ML models from Modeling Hub (fallback)
    if not result["has_forecast"]:
        for model in ["XGBoost", "LSTM", "ANN"]:
            key = f"ml_mh_results_{model}"
            ml_results = st.session_state.get(key)
            if ml_results and isinstance(ml_results, dict):
                per_h = ml_results.get("per_h", {})
                if per_h:
                    try:
                        horizons = sorted([int(h) for h in per_h.keys()])
                        forecasts = []
                        for h in horizons[:7]:
                            h_data = per_h.get(h, {})
                            pred = h_data.get("y_test_pred") or h_data.get("forecast") or h_data.get("predictions")
                            if pred is not None:
                                arr = pred.values if hasattr(pred, 'values') else np.array(pred)
                                if len(arr) > 0:
                                    forecasts.append(max(0, float(arr[-1])))
                        if forecasts:
                            result["has_forecast"] = True
                            result["source"] = f"ML ({model})"
                            result["model_type"] = "ML"
                            result["forecasts"] = forecasts
                            result["horizon"] = len(forecasts)
                            result["synced_with_page10"] = False
                            break
                    except Exception:
                        continue

    # PRIORITY 4: Check ARIMA (fallback)
    if not result["has_forecast"]:
        arima = st.session_state.get("arima_mh_results")
        if arima and isinstance(arima, dict):
            per_h = arima.get("per_h", {})
            if per_h:
                try:
                    horizons = sorted([int(h) for h in per_h.keys()])
                    forecasts = []
                    for h in horizons[:7]:
                        fc = per_h.get(h, {}).get("forecast")
                        if fc is not None:
                            arr = fc.values if hasattr(fc, 'values') else np.array(fc)
                            if len(arr) > 0:
                                forecasts.append(max(0, float(arr[-1])))
                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = "ARIMA"
                        result["model_type"] = "Statistical"
                        result["forecasts"] = forecasts
                        result["horizon"] = len(forecasts)
                        result["synced_with_page10"] = False
                except Exception:
                    pass

    # Calculate statistics if forecast available
    if result["has_forecast"] and result["forecasts"]:
        forecasts = result["forecasts"]
        result["total_week"] = int(sum(forecasts))
        result["avg_daily"] = int(np.mean(forecasts))

        for i, p in enumerate(forecasts):
            if p > result["peak_patients"]:
                result["peak_patients"] = int(p)
                result["peak_day"] = i
            if p < result["low_patients"]:
                result["low_patients"] = int(p)
                result["low_day"] = i

        # Determine trend
        if len(forecasts) >= 3:
            first_half = np.mean(forecasts[:len(forecasts)//2])
            second_half = np.mean(forecasts[len(forecasts)//2:])
            if second_half > first_half * 1.1:
                result["trend"] = "increasing"
            elif second_half < first_half * 0.9:
                result["trend"] = "decreasing"

        # Get confidence from RPIW if available
        rpiw = st.session_state.get("forecast_rpiw")
        if rpiw is not None:
            try:
                rpiw_val = float(rpiw)
                if rpiw_val < 15:
                    result["confidence"] = "High"
                elif rpiw_val < 30:
                    result["confidence"] = "Medium"
                else:
                    result["confidence"] = "Low"
            except:
                pass

        # Generate dates if not available
        if not result["dates"]:
            today = datetime.now().date()
            result["dates"] = [(today + timedelta(days=i+1)).strftime("%a %m/%d") for i in range(result["horizon"])]

    return result


def get_staff_insights() -> Dict[str, Any]:
    """
    Extract staffing data from Staff Planner session state.
    Reads: staff_stats, cost_params, optimized_results, staff_data_loaded
    """
    result = {
        "has_data": False,
        "data_loaded": False,
        "optimization_done": False,
        # Current stats
        "avg_doctors": 0,
        "avg_nurses": 0,
        "avg_support": 0,
        "avg_utilization": 0,
        "total_overtime": 0,
        "shortage_days": 0,
        # Financial
        "daily_labor_cost": 0,
        "weekly_labor_cost": 0,
        "coverage_percent": 0,
        # Optimized (if available)
        "opt_doctors": 0,
        "opt_nurses": 0,
        "opt_support": 0,
        "opt_weekly_cost": 0,
        "weekly_savings": 0,
        "recommendations": [],
    }

    # Check if data is loaded
    result["data_loaded"] = st.session_state.get("staff_data_loaded", False)
    result["optimization_done"] = st.session_state.get("optimization_done", False)

    if not result["data_loaded"]:
        return result

    result["has_data"] = True

    # Get staff stats from Staff Planner
    staff_stats = st.session_state.get("staff_stats", {})
    cost_params = st.session_state.get("cost_params", {})

    # Current values - ensure non-negative
    result["avg_doctors"] = max(0, staff_stats.get("avg_doctors", 0))
    result["avg_nurses"] = max(0, staff_stats.get("avg_nurses", 0))
    result["avg_support"] = max(0, staff_stats.get("avg_support", 0))
    result["avg_utilization"] = max(0, min(100, staff_stats.get("avg_utilization", 0)))
    result["total_overtime"] = max(0, staff_stats.get("total_overtime", 0))
    result["shortage_days"] = max(0, staff_stats.get("shortage_days", 0))

    # Calculate costs
    shift_hours = cost_params.get("shift_hours", 8)
    doc_hourly = cost_params.get("doctor_hourly", 150)
    nurse_hourly = cost_params.get("nurse_hourly", 50)
    support_hourly = cost_params.get("support_hourly", 25)

    daily_cost = (
        result["avg_doctors"] * doc_hourly * shift_hours +
        result["avg_nurses"] * nurse_hourly * shift_hours +
        result["avg_support"] * support_hourly * shift_hours
    )
    result["daily_labor_cost"] = daily_cost
    result["weekly_labor_cost"] = daily_cost * 7

    # Coverage estimate based on utilization
    result["coverage_percent"] = result["avg_utilization"]

    # Check for optimization results
    opt_results = st.session_state.get("optimized_results")
    if opt_results and isinstance(opt_results, dict) and opt_results.get("success", False):
        result["opt_doctors"] = opt_results.get("avg_opt_doctors", 0)
        result["opt_nurses"] = opt_results.get("avg_opt_nurses", 0)
        result["opt_support"] = opt_results.get("avg_opt_support", 0)
        result["opt_weekly_cost"] = opt_results.get("optimized_weekly_cost", 0)
        result["weekly_savings"] = opt_results.get("weekly_savings", 0)

    # Generate recommendations
    if result["avg_utilization"] < 80:
        result["recommendations"].append({
            "priority": "high",
            "text": f"Staff utilization is {result['avg_utilization']:.0f}% - consider optimizing schedules"
        })

    if result["shortage_days"] > 5:
        result["recommendations"].append({
            "priority": "urgent",
            "text": f"{result['shortage_days']} shortage days detected - urgent staffing action needed"
        })

    if result["total_overtime"] > 100:
        result["recommendations"].append({
            "priority": "important",
            "text": f"High overtime ({result['total_overtime']:.0f}h) - review staffing levels"
        })

    return result


def get_inventory_insights() -> Dict[str, Any]:
    """
    Extract inventory data from Supply Planner session state.
    Reads: inventory_stats, inv_cost_params, inv_optimized_results, inventory_data_loaded
    """
    result = {
        "has_data": False,
        "data_loaded": False,
        "optimization_done": False,
        # Current stats
        "avg_gloves_usage": 0,
        "avg_ppe_usage": 0,
        "avg_medication_usage": 0,
        "avg_stockout_risk": 0,
        "restock_events": 0,
        "service_level": 100,
        # Financial
        "daily_inventory_cost": 0,
        "weekly_inventory_cost": 0,
        # Counts
        "items_tracked": 3,  # Gloves, PPE, Medications
        "items_ok": 0,
        "items_low": 0,
        "items_critical": 0,
        # Optimized (if available)
        "opt_weekly_cost": 0,
        "weekly_savings": 0,
        "reorder_needed": [],
    }

    # Check if data is loaded
    result["data_loaded"] = st.session_state.get("inventory_data_loaded", False)
    result["optimization_done"] = st.session_state.get("inv_optimization_done", False)

    if not result["data_loaded"]:
        return result

    result["has_data"] = True

    # Get inventory stats from Supply Planner
    inv_stats = st.session_state.get("inventory_stats", {})
    cost_params = st.session_state.get("inv_cost_params", {})

    # Current values - ensure non-negative
    result["avg_gloves_usage"] = max(0, inv_stats.get("avg_gloves_usage", 0))
    result["avg_ppe_usage"] = max(0, inv_stats.get("avg_ppe_usage", 0))
    result["avg_medication_usage"] = max(0, inv_stats.get("avg_medication_usage", 0))
    result["avg_stockout_risk"] = max(0, min(100, inv_stats.get("avg_stockout_risk", 0)))
    result["restock_events"] = max(0, inv_stats.get("restock_events", 0))
    result["service_level"] = 100 - result["avg_stockout_risk"]

    # Calculate costs
    gloves_cost = result["avg_gloves_usage"] * cost_params.get("gloves_unit_cost", 15)
    ppe_cost = result["avg_ppe_usage"] * cost_params.get("ppe_unit_cost", 45)
    med_cost = result["avg_medication_usage"] * cost_params.get("medication_unit_cost", 8)

    result["daily_inventory_cost"] = gloves_cost + ppe_cost + med_cost
    result["weekly_inventory_cost"] = result["daily_inventory_cost"] * 7

    # Determine item status based on stockout risk
    if result["avg_stockout_risk"] > 10:
        result["items_critical"] = 1
        result["items_low"] = 1
        result["items_ok"] = 1
    elif result["avg_stockout_risk"] > 5:
        result["items_critical"] = 0
        result["items_low"] = 2
        result["items_ok"] = 1
    else:
        result["items_critical"] = 0
        result["items_low"] = 0
        result["items_ok"] = 3

    # Check for optimization results
    opt_results = st.session_state.get("inv_optimized_results")
    if opt_results and isinstance(opt_results, dict) and opt_results.get("success", False):
        result["opt_weekly_cost"] = opt_results.get("optimized_weekly_cost", 0)
        result["weekly_savings"] = opt_results.get("weekly_savings", 0)

    # Generate reorder alerts if stockout risk is high
    if result["avg_stockout_risk"] > 10:
        result["reorder_needed"].append({
            "name": "Critical Items",
            "current": "Low",
            "needed": "Restock ASAP",
            "urgency": "CRITICAL"
        })

    return result


def calculate_financial_impact(forecast: Dict, staffing: Dict, inventory: Dict) -> Dict[str, Any]:
    """Calculate total financial impact from all optimizations."""
    result = {
        "staffing_savings": 0,
        "inventory_savings": 0,
        "efficiency_gains": 0,
        "total_monthly_savings": 0,
        "annual_savings": 0,
        "roi_percent": 0,
        "breakdown": [],
    }

    # Staffing savings
    if staffing["optimization_done"] and staffing["weekly_savings"] > 0:
        monthly_staff_savings = staffing["weekly_savings"] * 4.3
        result["staffing_savings"] = monthly_staff_savings
        result["breakdown"].append({
            "category": "Staff Planner",
            "savings": monthly_staff_savings,
            "description": "Optimized staff allocation based on forecast"
        })

    # Inventory savings
    if inventory["optimization_done"] and inventory["weekly_savings"] > 0:
        monthly_inv_savings = inventory["weekly_savings"] * 4.3
        result["inventory_savings"] = monthly_inv_savings
        result["breakdown"].append({
            "category": "Supply Planner",
            "savings": monthly_inv_savings,
            "description": "Optimized ordering and reduced stockouts"
        })

    # Efficiency gains from forecasting
    if forecast["has_forecast"]:
        weekly_patients = forecast["total_week"]
        avg_cost_per_patient = 150
        monthly_patients = weekly_patients * 4.3
        efficiency = monthly_patients * avg_cost_per_patient * 0.05  # 5% efficiency gain
        result["efficiency_gains"] = efficiency
        result["breakdown"].append({
            "category": "Forecast Accuracy",
            "savings": efficiency,
            "description": "Better resource planning from accurate forecasts"
        })

    result["total_monthly_savings"] = (
        result["staffing_savings"] +
        result["inventory_savings"] +
        result["efficiency_gains"]
    )
    result["annual_savings"] = result["total_monthly_savings"] * 12

    # ROI calculation (assume $500/month system cost)
    system_cost = 500
    if result["total_monthly_savings"] > 0:
        result["roi_percent"] = ((result["total_monthly_savings"] - system_cost) / system_cost) * 100

    return result


def generate_action_items(forecast: Dict, staffing: Dict, inventory: Dict) -> List[Dict[str, Any]]:
    """Generate prioritized action items based on current data."""
    actions = []

    # Forecast actions
    if not forecast["has_forecast"]:
        actions.append({
            "priority": "important",
            "icon": "📊",
            "title": "No Patient Forecast Available",
            "description": "Go to Train Models to train models, then Patient Forecast to generate predictions.",
            "category": "System"
        })
    else:
        if forecast["trend"] == "increasing":
            actions.append({
                "priority": "important",
                "icon": "📈",
                "title": "Patient Volume Increasing",
                "description": f"Peak expected: {forecast['peak_patients']} patients. Prepare additional resources.",
                "category": "Planning"
            })

        if forecast["confidence"] == "Low":
            actions.append({
                "priority": "normal",
                "icon": "🔄",
                "title": "Low Forecast Confidence",
                "description": "Consider training more models or using longer historical data.",
                "category": "Planning"
            })

    # Staffing actions
    if not staffing["data_loaded"]:
        actions.append({
            "priority": "normal",
            "icon": "👥",
            "title": "Load Staff Data",
            "description": "Go to Staff Planner (Staff Planner) to load and analyze staffing data.",
            "category": "System"
        })
    else:
        if staffing["shortage_days"] > 0:
            actions.append({
                "priority": "urgent",
                "icon": "⚠️",
                "title": f"{staffing['shortage_days']} Shortage Days",
                "description": "Some periods lack adequate coverage. Review and adjust schedules.",
                "category": "Staffing"
            })

        if staffing["avg_utilization"] < 80:
            actions.append({
                "priority": "important",
                "icon": "📉",
                "title": f"Low Utilization ({staffing['avg_utilization']:.0f}%)",
                "description": "Staff utilization below target. Consider schedule optimization.",
                "category": "Staffing"
            })

        if not staffing["optimization_done"]:
            actions.append({
                "priority": "normal",
                "icon": "🚀",
                "title": "Run Staff Optimization",
                "description": "Apply forecast-based optimization in Staff Planner page.",
                "category": "System"
            })

    # Inventory actions
    if not inventory["data_loaded"]:
        actions.append({
            "priority": "normal",
            "icon": "📦",
            "title": "Load Inventory Data",
            "description": "Go to Supply Planner (Supply Planner) to load and analyze inventory.",
            "category": "System"
        })
    else:
        if inventory["items_critical"] > 0:
            actions.append({
                "priority": "urgent",
                "icon": "🚨",
                "title": f"{inventory['items_critical']} Critical Items",
                "description": "Some inventory items at critical levels. Reorder immediately.",
                "category": "Inventory"
            })

        if inventory["avg_stockout_risk"] > 5:
            actions.append({
                "priority": "important",
                "icon": "📦",
                "title": f"Stockout Risk: {inventory['avg_stockout_risk']:.1f}%",
                "description": "Monitor inventory levels closely and consider increasing safety stock.",
                "category": "Inventory"
            })

        if not inventory["optimization_done"]:
            actions.append({
                "priority": "normal",
                "icon": "🚀",
                "title": "Run Inventory Optimization",
                "description": "Apply forecast-based optimization in Supply Planner page.",
                "category": "System"
            })

    # Sort by priority
    priority_order = {"urgent": 0, "important": 1, "normal": 2}
    actions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return actions


# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1.5rem;'>
      <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>📊</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Dashboard</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Unified operational view • Patient forecasts, staffing, inventory & financial insights in one place
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Current date display
today = datetime.now()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 1.5rem;'>
    <span style='color: #64748b; font-size: 0.9rem;'>
        📅 Week of {today.strftime('%B %d, %Y')} | Last updated: {today.strftime('%I:%M %p')}
    </span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# GET ALL DATA
# =============================================================================
forecast = get_forecast_data()
staffing = get_staff_insights()
inventory = get_inventory_insights()
financial = calculate_financial_impact(forecast, staffing, inventory)
actions = generate_action_items(forecast, staffing, inventory)




# =============================================================================
# WORKFLOW PROGRESS INDICATOR - Shows user journey through the system
# =============================================================================
# Determine step completion status
step1_done = st.session_state.get("patient_loaded", False) or st.session_state.get("merged_data") is not None
step2_done = forecast["has_forecast"]
step3_done = forecast["has_forecast"]
step4_done = staffing["optimization_done"] or inventory["optimization_done"]
step5_done = len([a for a in actions if a["priority"] == "urgent"]) == 0

# Calculate completion percentage
steps_done = sum([step1_done, step2_done, step4_done, step5_done])
completion_pct = int((steps_done / 4) * 100)

# Step styling
s1_color = "#22c55e" if step1_done else "#64748b"
s1_icon = "✅" if step1_done else "1️⃣"
s2_color = "#22c55e" if step2_done else "#64748b"
s2_icon = "✅" if step2_done else "2️⃣"
s3_color = "#22c55e" if step3_done else "#64748b"
s3_icon = "✅" if step3_done else "3️⃣"
s4_color = "#22c55e" if step4_done else "#64748b"
s4_icon = "✅" if step4_done else "4️⃣"
s5_color = "#22c55e" if step5_done else "#64748b"
s5_icon = "✅" if step5_done else "5️⃣"

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 16px; padding: 1.25rem; margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; font-weight: 600; color: #94a3b8;">WORKFLOW PROGRESS</span>
        <span style="font-size: 0.85rem; font-weight: 700; color: #22c55e;">{completion_pct}% Complete</span>
    </div>
    <div style="background: #1e293b; border-radius: 8px; height: 8px; margin-bottom: 1.25rem; overflow: hidden;">
        <div style="background: linear-gradient(90deg, #22c55e, #3b82f6); height: 100%; width: {completion_pct}%;
                    border-radius: 8px; transition: width 0.5s ease;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{s1_icon}</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: {s1_color};">Upload</div>
            <div style="font-size: 0.65rem; color: #64748b;">Data</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{s2_icon}</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: {s2_color};">Train</div>
            <div style="font-size: 0.65rem; color: #64748b;">Models</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{s3_icon}</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: {s3_color};">Generate</div>
            <div style="font-size: 0.65rem; color: #64748b;">Forecast</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{s4_icon}</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: {s4_color};">Optimize</div>
            <div style="font-size: 0.65rem; color: #64748b;">Resources</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{s5_icon}</div>
            <div style="font-size: 0.7rem; font-weight: 600; color: {s5_color};">Take</div>
            <div style="font-size: 0.65rem; color: #64748b;">Action</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# FORECAST SYNC STATUS - Synced with Patient Forecast page
# =============================================================================
sync_col1, sync_col2 = st.columns([3, 1])
with sync_col1:
    if forecast["has_forecast"]:
        if forecast["synced_with_page10"]:
            sync_icon = "✅"
            sync_status = "Synced with Forecast Hub"
            sync_color = "#22c55e"
        else:
            sync_icon = "⚠️"
            sync_status = "Using Direct Model Results"
            sync_color = "#eab308"

        model_info = f"{forecast['source']}"
        if forecast.get("model_type"):
            model_info += f" ({forecast['model_type']})"
        if forecast.get("accuracy"):
            model_info += f" • Accuracy: {forecast['accuracy']:.1f}%"
        if forecast.get("timestamp"):
            model_info += f" • Updated: {forecast['timestamp'][:16]}"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                    border: 1px solid {sync_color}40; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{sync_icon}</span>
                <div>
                    <div style="font-weight: 600; color: {sync_color};">{sync_status}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{model_info}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
                    border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">❌</span>
                <div>
                    <div style="font-weight: 600; color: #ef4444;">No Forecast Data Available</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Run models in Patient Forecast page to enable insights</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with sync_col2:
    if st.button("🔄 Refresh Data", type="secondary", use_container_width=True, key="refresh_forecast_exec"):
        st.rerun()


# =============================================================================
# DATA STATUS BAR
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📡</span>
    <span class="section-title">System Status</span>
</div>
""", unsafe_allow_html=True)

status_cols = st.columns(4)

with status_cols[0]:
    fc_status = "✅ Available" if forecast["has_forecast"] else "❌ Not Available"
    fc_class = "status-good" if forecast["has_forecast"] else "status-alert"
    fc_detail = f"Source: {forecast['source']}" if forecast["has_forecast"] else "Run Patient Forecast"
    st.markdown(f"""
    <div class="data-status-box">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">🔮 Forecast</div>
        <span class="{fc_class}">{fc_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{fc_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[1]:
    st_status = "✅ Loaded" if staffing["data_loaded"] else "❌ Not Loaded"
    st_class = "status-good" if staffing["data_loaded"] else "status-alert"
    st_detail = f"Utilization: {staffing['avg_utilization']:.0f}%" if staffing["data_loaded"] else "Load in Staff Planner"
    st.markdown(f"""
    <div class="data-status-box">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">👥 Staff Data</div>
        <span class="{st_class}">{st_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{st_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[2]:
    inv_status = "✅ Loaded" if inventory["data_loaded"] else "❌ Not Loaded"
    inv_class = "status-good" if inventory["data_loaded"] else "status-alert"
    inv_detail = f"Service: {inventory['service_level']:.0f}%" if inventory["data_loaded"] else "Load in Supply Planner"
    st.markdown(f"""
    <div class="data-status-box">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">📦 Inventory Data</div>
        <span class="{inv_class}">{inv_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{inv_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[3]:
    opt_count = sum([staffing["optimization_done"], inventory["optimization_done"]])
    opt_status = f"✅ {opt_count}/2 Complete" if opt_count > 0 else "⚠️ None Run"
    opt_class = "status-good" if opt_count == 2 else "status-warning" if opt_count == 1 else "status-alert"
    st.markdown(f"""
    <div class="data-status-box">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">🚀 Optimizations</div>
        <span class="{opt_class}">{opt_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Staff & Inventory</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN TABS
# =============================================================================
tab_overview, tab_forecast, tab_staff, tab_inventory, tab_actions = st.tabs([
    "📊 Executive Summary",
    "🔮 Patient Forecast",
    "👥 Staffing Overview",
    "📦 Inventory Status",
    "✅ Action Items"
])


# =============================================================================
# TAB 1: EXECUTIVE SUMMARY
# =============================================================================
with tab_overview:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-title">This Week at a Glance</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards Row
    kpi_cols = st.columns(5)

    with kpi_cols[0]:
        patients = forecast["total_week"] if forecast["has_forecast"] else "—"
        trend_icon = "📈" if forecast["trend"] == "increasing" else ("📉" if forecast["trend"] == "decreasing" else "➡️")
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #3b82f6; --accent-end: #60a5fa;">
            <div class="exec-kpi-icon">🏥</div>
            <div class="exec-kpi-value">{patients}</div>
            <div class="exec-kpi-label">Expected Patients</div>
            <div class="exec-kpi-sublabel">{trend_icon} {forecast["trend"].title()} trend</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        avg = forecast["avg_daily"] if forecast["has_forecast"] else "—"
        conf_class = "status-good" if forecast["confidence"] == "High" else ("status-warning" if forecast["confidence"] == "Medium" else "status-alert")
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #8b5cf6; --accent-end: #a78bfa;">
            <div class="exec-kpi-icon">📈</div>
            <div class="exec-kpi-value">{avg}</div>
            <div class="exec-kpi-label">Avg Daily Patients</div>
            <div class="exec-kpi-sublabel"><span class="{conf_class}">{forecast["confidence"]} Confidence</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        util = f"{staffing['avg_utilization']:.0f}%" if staffing["has_data"] else "—"
        util_class = "status-good" if staffing["avg_utilization"] >= 80 else ("status-warning" if staffing["avg_utilization"] >= 60 else "status-alert")
        util_text = "✅ Good" if staffing["avg_utilization"] >= 80 else ("⚠️ Low" if staffing["has_data"] else "Not loaded")
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #22c55e; --accent-end: #4ade80;">
            <div class="exec-kpi-icon">👥</div>
            <div class="exec-kpi-value">{util}</div>
            <div class="exec-kpi-label">Staff Utilization</div>
            <div class="exec-kpi-sublabel"><span class="{util_class if staffing['has_data'] else ''}">{util_text}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        if inventory["has_data"]:
            if inventory["items_critical"] > 0:
                supply_status = f"🚨 {inventory['items_critical']} Critical"
                supply_class = "status-alert"
            elif inventory["items_low"] > 0:
                supply_status = f"⚠️ {inventory['items_low']} Low"
                supply_class = "status-warning"
            else:
                supply_status = "✅ All OK"
                supply_class = "status-good"
        else:
            supply_status = "Not loaded"
            supply_class = ""
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #f59e0b; --accent-end: #fbbf24;">
            <div class="exec-kpi-icon">📦</div>
            <div class="exec-kpi-value">{inventory["items_tracked"]}</div>
            <div class="exec-kpi-label">Supply Items</div>
            <div class="exec-kpi-sublabel"><span class="{supply_class}">{supply_status}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[4]:
        savings = f"${financial['total_monthly_savings']:,.0f}" if financial["total_monthly_savings"] > 0 else "—"
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #10b981; --accent-end: #34d399;">
            <div class="exec-kpi-icon">💰</div>
            <div class="exec-kpi-value" style="color: #22c55e;">{savings}</div>
            <div class="exec-kpi-label">Monthly Savings</div>
            <div class="exec-kpi-sublabel">From optimization</div>
        </div>
        """, unsafe_allow_html=True)

    # Financial Impact Section
    if financial["total_monthly_savings"] > 0:
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">💰</span>
            <span class="section-title">Financial Impact</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-label">ESTIMATED MONTHLY SAVINGS</div>
            <div class="savings-value">${financial['total_monthly_savings']:,.0f}</div>
            <div style="color: #86efac; margin-top: 0.5rem;">
                Annual projection: ${financial['annual_savings']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if financial["breakdown"]:
            fin_cols = st.columns(len(financial["breakdown"]))
            for idx, item in enumerate(financial["breakdown"]):
                with fin_cols[idx]:
                    st.markdown(f"""
                    <div class="action-card action-normal">
                        <div class="action-icon">💵</div>
                        <div class="action-content">
                            <div class="action-title">{item['category']}</div>
                            <div class="action-description">${item['savings']:,.0f}/month</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Quick Actions summary
    if actions:
        st.markdown("---")
        urgent_count = sum(1 for a in actions if a["priority"] == "urgent")
        important_count = sum(1 for a in actions if a["priority"] == "important")

        if urgent_count > 0:
            st.error(f"⚠️ **{urgent_count} urgent action(s)** require immediate attention. See Action Items tab.")
        elif important_count > 0:
            st.warning(f"📋 **{important_count} important action(s)** pending. See Action Items tab.")


# =============================================================================
# TAB 2: PATIENT FORECAST
# =============================================================================
with tab_forecast:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🔮</span>
        <span class="section-title">7-Day Patient Forecast</span>
    </div>
    """, unsafe_allow_html=True)

    if forecast["has_forecast"]:
        # Show sync-aware success message
        if forecast["synced_with_page10"]:
            st.success(f"✅ **Synced with Forecast Hub** — {forecast['source']} | Confidence: {forecast['confidence']}")
        else:
            st.info(f"📊 Using direct model results: **{forecast['source']}** | Confidence: {forecast['confidence']}\n\n*Run forecasts in Patient Forecast for full sync.*")

        # Day cards
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        num_days = min(7, len(forecast["forecasts"]))
        day_cols = st.columns(num_days)

        for i in range(num_days):
            with day_cols[i]:
                patients = forecast["forecasts"][i]
                day_date = today + timedelta(days=i+1)
                is_peak = i == forecast["peak_day"]

                card_class = "high-volume" if is_peak else ""
                peak_badge = "<span class='status-warning' style='font-size: 0.7rem;'>Peak Day</span>" if is_peak else ""

                st.markdown(f"""
                <div class="day-card {card_class}">
                    <div class="day-name">{day_names[day_date.weekday()]}</div>
                    <div class="day-date">{day_date.strftime('%b %d')}</div>
                    <div class="day-patients">{int(patients)}</div>
                    <div class="day-label">patients</div>
                    {peak_badge}
                </div>
                """, unsafe_allow_html=True)

        # Chart
        st.markdown("#### Weekly Volume Distribution")

        fig = go.Figure()
        dates = [(today + timedelta(days=i+1)).strftime('%a\n%b %d') for i in range(len(forecast["forecasts"]))]
        colors = ['#f59e0b' if i == forecast["peak_day"] else '#3b82f6' for i in range(len(forecast["forecasts"]))]

        fig.add_trace(go.Bar(
            x=dates,
            y=forecast["forecasts"],
            marker_color=colors,
            text=[int(p) for p in forecast["forecasts"]],
            textposition='auto',
        ))

        fig.add_hline(
            y=forecast["avg_daily"],
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Avg: {forecast['avg_daily']}"
        )

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Expected Patients",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        sum_cols = st.columns(4)
        with sum_cols[0]:
            st.metric("Total Patients", f"{forecast['total_week']:,}")
        with sum_cols[1]:
            st.metric("Daily Average", f"{forecast['avg_daily']:,}")
        with sum_cols[2]:
            st.metric("Peak Day", f"{forecast['peak_patients']:,} patients")
        with sum_cols[3]:
            st.metric("Forecast Horizon", f"{forecast['horizon']} days")

    else:
        st.warning("""
        **📊 No Patient Forecast Available**

        To see predicted patient volumes:
        1. Go to **Modeling Hub** (Train Models) and train a forecasting model
        2. Visit **Forecast Hub** (Patient Forecast) to generate predictions
        3. Return here to see your weekly planning dashboard

        **Tip:** Use the "🔄 Refresh Data" button above to sync after running forecasts.
        """)


# =============================================================================
# TAB 3: STAFFING OVERVIEW
# =============================================================================
with tab_staff:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">👥</span>
        <span class="section-title">Staffing Overview</span>
    </div>
    """, unsafe_allow_html=True)

    if staffing["has_data"]:
        # Current Stats
        staff_cols = st.columns(4)

        with staff_cols[0]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-icon">👨‍⚕️</div>
                <div class="exec-kpi-value">{staffing['avg_doctors']:.0f}</div>
                <div class="exec-kpi-label">Avg Doctors/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[1]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #22c55e;">
                <div class="exec-kpi-icon">👩‍⚕️</div>
                <div class="exec-kpi-value">{staffing['avg_nurses']:.0f}</div>
                <div class="exec-kpi-label">Avg Nurses/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[2]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #f97316;">
                <div class="exec-kpi-icon">🏥</div>
                <div class="exec-kpi-value">{staffing['avg_support']:.0f}</div>
                <div class="exec-kpi-label">Avg Support/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[3]:
            util_color = "#22c55e" if staffing['avg_utilization'] >= 80 else "#f59e0b"
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {util_color};">
                <div class="exec-kpi-icon">📊</div>
                <div class="exec-kpi-value" style="color: {util_color};">{staffing['avg_utilization']:.0f}%</div>
                <div class="exec-kpi-label">Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        # Financial
        st.markdown("#### 💰 Labor Costs")
        cost_cols = st.columns(3)

        with cost_cols[0]:
            st.metric("Daily Labor Cost", f"${staffing['daily_labor_cost']:,.0f}")
        with cost_cols[1]:
            st.metric("Weekly Labor Cost", f"${staffing['weekly_labor_cost']:,.0f}")
        with cost_cols[2]:
            if staffing["optimization_done"] and staffing["weekly_savings"] > 0:
                st.metric("Weekly Savings", f"${staffing['weekly_savings']:,.0f}", delta="Optimized")
            else:
                st.metric("Optimization", "Not Run", delta="Go to Staff Planner")

        # Utilization gauge
        st.markdown("#### 📊 Staff Utilization")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=staffing['avg_utilization'],
            title={'text': "Staff Utilization Rate"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#22c55e" if staffing['avg_utilization'] >= 80 else "#f59e0b"},
                'steps': [
                    {'range': [0, 60], 'color': "rgba(239,68,68,0.2)"},
                    {'range': [60, 80], 'color': "rgba(234,179,8,0.2)"},
                    {'range': [80, 100], 'color': "rgba(34,197,94,0.2)"},
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 85}
            }
        ))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        if staffing["recommendations"]:
            st.markdown("#### 💡 Recommendations")
            for rec in staffing["recommendations"]:
                priority_class = "action-urgent" if rec["priority"] == "urgent" else "action-important"
                icon = "🚨" if rec["priority"] == "urgent" else "⚠️"
                st.markdown(f"""
                <div class="action-card {priority_class}">
                    <div class="action-icon">{icon}</div>
                    <div class="action-content">
                        <div class="action-description">{rec['text']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("""
        **👥 Staff Data Not Loaded**

        To see staffing insights:
        1. Go to **Staff Planner Optimization** (Staff Planner)
        2. Click "Load Staff Data" to fetch from Supabase
        3. Return here to see staffing overview
        """)


# =============================================================================
# TAB 4: INVENTORY STATUS
# =============================================================================
with tab_inventory:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📦</span>
        <span class="section-title">Inventory Status</span>
    </div>
    """, unsafe_allow_html=True)

    if inventory["has_data"]:
        # Status Overview
        inv_cols = st.columns(4)

        with inv_cols[0]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #a855f7;">
                <div class="exec-kpi-icon">🧤</div>
                <div class="exec-kpi-value">{inventory['avg_gloves_usage']:.0f}</div>
                <div class="exec-kpi-label">Gloves/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[1]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-icon">🛡️</div>
                <div class="exec-kpi-value">{inventory['avg_ppe_usage']:.0f}</div>
                <div class="exec-kpi-label">PPE Sets/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[2]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #22c55e;">
                <div class="exec-kpi-icon">💊</div>
                <div class="exec-kpi-value">{inventory['avg_medication_usage']:.0f}</div>
                <div class="exec-kpi-label">Medications/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[3]:
            risk_color = "#ef4444" if inventory['avg_stockout_risk'] > 10 else "#f59e0b" if inventory['avg_stockout_risk'] > 5 else "#22c55e"
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {risk_color};">
                <div class="exec-kpi-icon">⚠️</div>
                <div class="exec-kpi-value" style="color: {risk_color};">{inventory['avg_stockout_risk']:.1f}%</div>
                <div class="exec-kpi-label">Stockout Risk</div>
            </div>
            """, unsafe_allow_html=True)

        # Service Level Gauge
        st.markdown("#### 📊 Service Level")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=inventory["service_level"],
            title={'text': "Service Level"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#22c55e" if inventory["service_level"] >= 95 else "#f59e0b"},
                'steps': [
                    {'range': [0, 90], 'color': "rgba(239,68,68,0.2)"},
                    {'range': [90, 95], 'color': "rgba(234,179,8,0.2)"},
                    {'range': [95, 100], 'color': "rgba(34,197,94,0.2)"},
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 98}
            }
        ))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Cost metrics
        cost_cols = st.columns(3)
        with cost_cols[0]:
            st.metric("Daily Inventory Cost", f"${inventory['daily_inventory_cost']:,.0f}")
        with cost_cols[1]:
            st.metric("Weekly Inventory Cost", f"${inventory['weekly_inventory_cost']:,.0f}")
        with cost_cols[2]:
            if inventory["optimization_done"] and inventory["weekly_savings"] > 0:
                st.metric("Weekly Savings", f"${inventory['weekly_savings']:,.0f}", delta="Optimized")
            else:
                st.metric("Optimization", "Not Run", delta="Go to Supply Planner")

    else:
        st.info("""
        **📦 Inventory Data Not Loaded**

        To see inventory insights:
        1. Go to **Supply Planner Optimization** (Supply Planner)
        2. Click "Load Inventory Data" to fetch from Supabase
        3. Return here to see inventory overview
        """)


# =============================================================================
# TAB 5: ACTION ITEMS
# =============================================================================
with tab_actions:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">✅</span>
        <span class="section-title">Action Items for This Week</span>
    </div>
    """, unsafe_allow_html=True)

    if actions:
        # Count by priority
        urgent_count = sum(1 for a in actions if a["priority"] == "urgent")
        important_count = sum(1 for a in actions if a["priority"] == "important")
        normal_count = sum(1 for a in actions if a["priority"] == "normal")

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #ef4444;">
                <div class="exec-kpi-value" style="color: #ef4444;">{urgent_count}</div>
                <div class="exec-kpi-label">Urgent</div>
            </div>
            """, unsafe_allow_html=True)
        with summary_cols[1]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #f59e0b;">
                <div class="exec-kpi-value" style="color: #f59e0b;">{important_count}</div>
                <div class="exec-kpi-label">Important</div>
            </div>
            """, unsafe_allow_html=True)
        with summary_cols[2]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-value" style="color: #3b82f6;">{normal_count}</div>
                <div class="exec-kpi-label">Normal</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Action cards
        col1, col2 = st.columns(2)

        for idx, action in enumerate(actions[:10]):
            with col1 if idx % 2 == 0 else col2:
                priority_class = f"action-{action['priority']}"
                st.markdown(f"""
                <div class="action-card {priority_class}">
                    <div class="action-icon">{action['icon']}</div>
                    <div class="action-content">
                        <div class="action-title">{action['title']}</div>
                        <div class="action-description">{action['description']}</div>
                        <div style="margin-top: 0.5rem;">
                            <span style="background: rgba(100, 116, 139, 0.3); padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; color: #94a3b8;">{action['category']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ All systems running smoothly! No urgent actions required.")


# =============================================================================
# SIDEBAR - Dark Blue Fluorescent Status Tracker
# =============================================================================
with st.sidebar:
    st.markdown("---")

    # -------------------------------------------------------------------------
    # QUICK STATUS - Dark Blue Fluorescent Cards
    # -------------------------------------------------------------------------
    # Determine status classes for each item
    forecast_status = "status-active" if forecast['has_forecast'] else "status-inactive"
    forecast_icon = "✅" if forecast['has_forecast'] else "❌"
    forecast_text = forecast['source'][:15] + "..." if forecast['has_forecast'] and len(forecast.get('source', '')) > 15 else (forecast.get('source', 'N/A') if forecast['has_forecast'] else "Not available")

    staff_data_status = "status-active" if staffing['data_loaded'] else "status-inactive"
    staff_data_icon = "✅" if staffing['data_loaded'] else "❌"
    staff_data_text = "Loaded" if staffing['data_loaded'] else "Not loaded"

    inv_data_status = "status-active" if inventory['data_loaded'] else "status-inactive"
    inv_data_icon = "✅" if inventory['data_loaded'] else "❌"
    inv_data_text = "Loaded" if inventory['data_loaded'] else "Not loaded"

    staff_opt_status = "status-active" if staffing['optimization_done'] else "status-inactive"
    staff_opt_icon = "✅" if staffing['optimization_done'] else "❌"
    staff_opt_text = "Optimized" if staffing['optimization_done'] else "Pending"

    inv_opt_status = "status-active" if inventory['optimization_done'] else "status-inactive"
    inv_opt_icon = "✅" if inventory['optimization_done'] else "❌"
    inv_opt_text = "Optimized" if inventory['optimization_done'] else "Pending"

    st.markdown(f"""
    <div class="sidebar-status-container">
        <div class="sidebar-status-title">📊 Quick Status</div>

        <div class="sidebar-status-card">
            <span class="sidebar-status-label">🔮 Forecast</span>
            <span class="sidebar-status-value {forecast_status}">{forecast_icon} {forecast_text}</span>
        </div>

        <div class="sidebar-status-card">
            <span class="sidebar-status-label">👥 Staff Data</span>
            <span class="sidebar-status-value {staff_data_status}">{staff_data_icon} {staff_data_text}</span>
        </div>

        <div class="sidebar-status-card">
            <span class="sidebar-status-label">📦 Inventory</span>
            <span class="sidebar-status-value {inv_data_status}">{inv_data_icon} {inv_data_text}</span>
        </div>

        <div class="sidebar-status-card">
            <span class="sidebar-status-label">🚀 Staff Opt.</span>
            <span class="sidebar-status-value {staff_opt_status}">{staff_opt_icon} {staff_opt_text}</span>
        </div>

        <div class="sidebar-status-card">
            <span class="sidebar-status-label">📈 Inv. Opt.</span>
            <span class="sidebar-status-value {inv_opt_status}">{inv_opt_icon} {inv_opt_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SAVINGS CARD - Show if optimizations have generated savings
    # -------------------------------------------------------------------------
    if financial["total_monthly_savings"] > 0:
        st.markdown(f"""
        <div class="sidebar-savings-card">
            <div class="sidebar-savings-label">💰 Monthly Savings</div>
            <div class="sidebar-savings-value">${financial['total_monthly_savings']:,.0f}</div>
            <div style="font-size: 0.65rem; color: #94a3b8; margin-top: 0.25rem;">
                Annual: ${financial['annual_savings']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ACTIVITY TRACKER - Real-time app activity monitoring
    # -------------------------------------------------------------------------
    # Determine activity status
    data_uploaded = st.session_state.get("patient_loaded", False) or st.session_state.get("merged_data") is not None
    models_trained = forecast["has_forecast"]
    forecast_generated = forecast["synced_with_page10"] if forecast["has_forecast"] else False
    resources_optimized = staffing["optimization_done"] or inventory["optimization_done"]
    actions_pending = len([a for a in actions if a["priority"] == "urgent"]) > 0

    # Activity dots
    upload_dot = "active" if data_uploaded else "inactive"
    train_dot = "active" if models_trained else ("pending" if data_uploaded else "inactive")
    forecast_dot = "active" if forecast_generated else ("pending" if models_trained else "inactive")
    optimize_dot = "active" if resources_optimized else ("pending" if forecast_generated else "inactive")
    action_dot = "active" if not actions_pending else "pending"

    st.markdown(f"""
    <div class="sidebar-activity-container">
        <div class="sidebar-activity-title">⚡ Activity Tracker</div>

        <div class="sidebar-activity-item">
            <span class="activity-dot {upload_dot}"></span>
            <span>Data Upload</span>
            <span style="margin-left: auto; color: {'#22c55e' if data_uploaded else '#64748b'};">{'Done' if data_uploaded else 'Pending'}</span>
        </div>

        <div class="sidebar-activity-item">
            <span class="activity-dot {train_dot}"></span>
            <span>Model Training</span>
            <span style="margin-left: auto; color: {'#22c55e' if models_trained else '#64748b'};">{'Done' if models_trained else 'Pending'}</span>
        </div>

        <div class="sidebar-activity-item">
            <span class="activity-dot {forecast_dot}"></span>
            <span>Forecast Generated</span>
            <span style="margin-left: auto; color: {'#22c55e' if forecast_generated else '#64748b'};">{'Done' if forecast_generated else 'Pending'}</span>
        </div>

        <div class="sidebar-activity-item">
            <span class="activity-dot {optimize_dot}"></span>
            <span>Resource Optimization</span>
            <span style="margin-left: auto; color: {'#22c55e' if resources_optimized else '#64748b'};">{'Done' if resources_optimized else 'Pending'}</span>
        </div>

        <div class="sidebar-activity-item">
            <span class="activity-dot {action_dot}"></span>
            <span>Urgent Actions</span>
            <span style="margin-left: auto; color: {'#22c55e' if not actions_pending else '#fbbf24'};">{'Clear' if not actions_pending else 'Review'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # QUICK LINKS - Styled navigation links
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="sidebar-links-container">
        <div class="sidebar-status-title" style="font-size: 0.75rem; margin-bottom: 0.5rem;">🔗 Quick Links</div>
        <div class="sidebar-link-item">🧠 Train Models → Modeling Hub</div>
        <div class="sidebar-link-item">🔮 Forecast → Patient Forecast</div>
        <div class="sidebar-link-item">👥 Staff → Staff Planner</div>
        <div class="sidebar-link-item">📦 Supply → Supply Planner</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(0)  # Dashboard is page index 0

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Dashboard</strong> | HealthForecast AI<br>
    <em>Data-driven decisions for better patient care</em>
</div>
""", unsafe_allow_html=True)
