# =============================================================================
# 13_Action_Center.py — Enhanced Hospital Manager Decision Dashboard
# 4-Tab Structure: Staff Actions | Supply Actions | Combined Dashboard | Export
# Integrates ML Priority Classifier, Financial Impact, Export Capabilities
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import io
import json

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Action Center - HealthForecast AI",
    page_icon="🏥",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# IMPORT ML PRIORITY CLASSIFIER
# =============================================================================
try:
    from app_core.classification.action_priority_model import (
        ActionPriorityClassifier,
        ActionPriorityFeatures,
        ActionPriority,
    )
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================
CLINICAL_CATEGORIES = [
    "RESPIRATORY", "CARDIAC", "TRAUMA", "GASTROINTESTINAL",
    "INFECTIOUS", "NEUROLOGICAL", "OTHER"
]

PRIORITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f59e0b",
    "MEDIUM": "#3b82f6",
    "LOW": "#22c55e",
}

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(f"""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR ACTION CENTER
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
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}}

.orb-2 {{
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.2), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
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
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(34, 197, 94, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
}}

.sparkle-1 {{ top: 25%; left: 35%; animation-delay: 0s; }}
.sparkle-2 {{ top: 65%; left: 70%; animation-delay: 1s; }}
.sparkle-3 {{ top: 45%; left: 15%; animation-delay: 2s; }}

/* ========================================
   CUSTOM TAB STYLING
   ======================================== */

.stTabs {{
    background: transparent;
    margin-top: 1rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(34, 197, 94, 0.2);
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
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.1));
    border-color: rgba(34, 197, 94, 0.3);
    color: {TEXT_COLOR};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(59, 130, 246, 0.2)) !important;
    border: 1px solid rgba(34, 197, 94, 0.5) !important;
    color: {TEXT_COLOR} !important;
    box-shadow:
        0 0 20px rgba(34, 197, 94, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}

/* ========================================
   KPI CARDS
   ======================================== */

.cmd-kpi-card {{
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

.cmd-kpi-card:hover {{
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
}}

.cmd-kpi-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color, #3b82f6), var(--accent-end, #22d3ee));
}}

.cmd-kpi-icon {{
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}}

.cmd-kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.25rem 0;
}}

.cmd-kpi-label {{
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 600;
}}

.cmd-kpi-sublabel {{
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

/* Priority Cards */
.priority-card {{
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

.priority-card:hover {{
    transform: translateX(5px);
}}

.priority-critical {{
    border-left-color: #ef4444;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(15, 23, 42, 0.95));
}}

.priority-high {{
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(15, 23, 42, 0.95));
}}

.priority-medium {{
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(15, 23, 42, 0.95));
}}

.priority-low {{
    border-left-color: #22c55e;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(15, 23, 42, 0.95));
}}

.priority-icon {{
    font-size: 1.5rem;
    flex-shrink: 0;
}}

.priority-content {{
    flex: 1;
}}

.priority-title {{
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
    font-size: 0.95rem;
}}

.priority-description {{
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}}

.priority-meta {{
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.5rem;
}}

/* Section Header */
.section-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(34, 197, 94, 0.3);
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

/* Export Card */
.export-card {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(99, 102, 241, 0.1));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

/* Reorder Alert */
.reorder-alert {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}}

/* Mobile Responsive - Tablet */
@media (max-width: 768px) {{
    .fluorescent-orb {{
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }}
    .sparkle {{
        display: none;
    }}

    /* KPI Cards */
    .cmd-kpi-card {{
        padding: 1rem;
        border-radius: 12px;
    }}
    .cmd-kpi-icon {{ font-size: 2rem; }}
    .cmd-kpi-value {{ font-size: 1.5rem; }}
    .cmd-kpi-label {{ font-size: 0.8rem; }}
    .cmd-kpi-sublabel {{ font-size: 0.7rem; }}

    /* Savings Card */
    .savings-value {{ font-size: 2rem; }}
    .savings-label {{ font-size: 0.85rem; }}
    .savings-card {{ padding: 1rem; border-radius: 12px; }}

    /* Priority Cards */
    .priority-card {{
        padding: 0.875rem 1rem;
        border-radius: 10px;
    }}
    .priority-icon {{ font-size: 1.25rem; }}
    .priority-title {{ font-size: 0.875rem; }}
    .priority-description {{ font-size: 0.8rem; }}
    .priority-meta {{ font-size: 0.7rem; }}

    /* Section Headers */
    .section-header {{
        margin: 1rem 0 0.75rem 0;
    }}
    .section-icon {{ font-size: 1.25rem; }}
    .section-title {{ font-size: 1.1rem; }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        padding: 0 1rem;
        font-size: 0.85rem;
    }}
}}

/* Mobile Responsive - Phone */
@media (max-width: 480px) {{
    /* KPI Cards - Compact */
    .cmd-kpi-card {{
        padding: 0.875rem;
    }}
    .cmd-kpi-icon {{ font-size: 1.75rem; }}
    .cmd-kpi-value {{ font-size: 1.25rem; }}
    .cmd-kpi-label {{ font-size: 0.75rem; }}

    /* Savings Card */
    .savings-value {{ font-size: 1.5rem; }}
    .savings-card {{ padding: 0.875rem; }}

    /* Priority Cards */
    .priority-card {{
        padding: 0.75rem;
        gap: 0.75rem;
    }}
    .priority-icon {{ font-size: 1.1rem; }}
    .priority-title {{ font-size: 0.8rem; }}
    .priority-description {{ font-size: 0.75rem; }}

    /* Section Headers */
    .section-icon {{ font-size: 1.1rem; }}
    .section-title {{ font-size: 1rem; }}

    /* Export Card */
    .export-card {{ padding: 1rem; }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        height: 36px;
        padding: 0 0.75rem;
        font-size: 0.75rem;
    }}
}}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
""", unsafe_allow_html=True)


# =============================================================================
# ML PRIORITY CLASSIFIER INSTANCE
# =============================================================================
@st.cache_resource
def get_priority_classifier() -> Optional[ActionPriorityClassifier]:
    """Get or create the priority classifier instance."""
    if ML_CLASSIFIER_AVAILABLE:
        return ActionPriorityClassifier()
    return None

priority_classifier = get_priority_classifier()


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def get_forecast_data() -> Dict[str, Any]:
    """
    Extract forecast data from session state.
    Priority: forecast_hub_demand → active_forecast → ML models → ARIMA
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
        "category_forecasts": {},
    }

    # PRIORITY 1: Check forecast_hub_demand (from Page 10)
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
            result["category_forecasts"] = hub_demand.get("category_forecasts", {})

    # PRIORITY 2: Check active_forecast
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

    # PRIORITY 3: Check ML models
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
                            pred = h_data.get("y_test_pred") or h_data.get("forecast")
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

        # Confidence from RPIW
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
            result["dates"] = [
                (today + timedelta(days=i+1)).strftime("%a %m/%d")
                for i in range(result["horizon"])
            ]

    return result


def get_staff_data() -> Dict[str, Any]:
    """
    Extract staffing data from Page 11 (Staff Planner) session state.
    Reads: staff_data_loaded, staff_stats, cost_params, optimized_results,
           staff_optimization_results (MILP), staff_constraint_config
    """
    result = {
        "has_data": False,
        "data_loaded": False,
        "optimization_done": False,
        "milp_used": False,
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
        # Schedule data for export
        "schedule_df": None,
        "recommendations": [],
        "actions": [],
    }

    # Check if data is loaded
    result["data_loaded"] = st.session_state.get("staff_data_loaded", False)
    result["optimization_done"] = st.session_state.get("optimization_done", False)

    if not result["data_loaded"]:
        return result

    result["has_data"] = True

    # Get staff stats from Page 11
    staff_stats = st.session_state.get("staff_stats", {})
    cost_params = st.session_state.get("cost_params", {})

    # Current values
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
    result["coverage_percent"] = result["avg_utilization"]

    # Check for MILP optimization results (new enhanced page)
    milp_results = st.session_state.get("staff_optimization_results")
    if milp_results and isinstance(milp_results, dict):
        if milp_results.get("status") == "Optimal":
            result["milp_used"] = True
            result["optimization_done"] = True
            result["opt_weekly_cost"] = milp_results.get("total_cost", 0)
            # Calculate savings
            if result["weekly_labor_cost"] > 0:
                result["weekly_savings"] = max(0, result["weekly_labor_cost"] - result["opt_weekly_cost"])

            # Extract schedule if available
            if "schedule" in milp_results:
                result["schedule_df"] = milp_results["schedule"]

    # Check legacy optimization results
    if not result["milp_used"]:
        opt_results = st.session_state.get("optimized_results")
        if opt_results and isinstance(opt_results, dict) and opt_results.get("success", False):
            result["optimization_done"] = True
            result["opt_doctors"] = opt_results.get("avg_opt_doctors", 0)
            result["opt_nurses"] = opt_results.get("avg_opt_nurses", 0)
            result["opt_support"] = opt_results.get("avg_opt_support", 0)
            result["opt_weekly_cost"] = opt_results.get("optimized_weekly_cost", 0)
            result["weekly_savings"] = opt_results.get("weekly_savings", 0)

    # Generate actions using ML classifier if available
    result["actions"] = generate_staff_actions(result)

    return result


def get_inventory_data() -> Dict[str, Any]:
    """
    Extract inventory data from Page 12 (Supply Planner) session state.
    Reads: inventory_data_loaded, inventory_stats, inv_cost_params,
           inv_optimized_results, inventory_optimization_results (MILP)
    """
    result = {
        "has_data": False,
        "data_loaded": False,
        "optimization_done": False,
        "milp_used": False,
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
        "items_tracked": 3,
        "items_ok": 0,
        "items_low": 0,
        "items_critical": 0,
        # Optimized (if available)
        "opt_weekly_cost": 0,
        "weekly_savings": 0,
        # Order schedule for export
        "order_schedule_df": None,
        "reorder_alerts": [],
        "actions": [],
    }

    # Check if data is loaded
    result["data_loaded"] = st.session_state.get("inventory_data_loaded", False)
    result["optimization_done"] = st.session_state.get("inv_optimization_done", False)

    if not result["data_loaded"]:
        return result

    result["has_data"] = True

    # Get inventory stats from Page 12
    inv_stats = st.session_state.get("inventory_stats", {})
    cost_params = st.session_state.get("inv_cost_params", {})

    # Current values
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

    # Determine item status
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

    # Check for MILP optimization results (new enhanced page)
    milp_results = st.session_state.get("inventory_optimization_results")
    if milp_results and isinstance(milp_results, dict):
        if milp_results.get("status") in ["Optimal", "Feasible"]:
            result["milp_used"] = True
            result["optimization_done"] = True
            result["opt_weekly_cost"] = milp_results.get("total_cost", 0)
            if result["weekly_inventory_cost"] > 0:
                result["weekly_savings"] = max(0, result["weekly_inventory_cost"] - result["opt_weekly_cost"])

            # Extract order schedule if available
            if "order_schedule" in milp_results:
                result["order_schedule_df"] = milp_results["order_schedule"]

            # Extract reorder alerts
            if "reorder_alerts" in milp_results:
                result["reorder_alerts"] = milp_results["reorder_alerts"]

    # Check legacy optimization results
    if not result["milp_used"]:
        opt_results = st.session_state.get("inv_optimized_results")
        if opt_results and isinstance(opt_results, dict) and opt_results.get("success", False):
            result["optimization_done"] = True
            result["opt_weekly_cost"] = opt_results.get("optimized_weekly_cost", 0)
            result["weekly_savings"] = opt_results.get("weekly_savings", 0)

    # Generate reorder alerts if stockout risk is high
    if result["avg_stockout_risk"] > 10 and not result["reorder_alerts"]:
        result["reorder_alerts"].append({
            "item": "Critical Items",
            "current_stock": "Low",
            "reorder_qty": "Immediate restock needed",
            "urgency": "CRITICAL",
            "days_until_stockout": 2,
        })

    # Generate actions using ML classifier if available
    result["actions"] = generate_inventory_actions(result)

    return result


# =============================================================================
# ML-BASED ACTION GENERATION
# =============================================================================

def generate_staff_actions(staff_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate staff-related actions with ML priority classification."""
    actions = []

    if not staff_data["has_data"]:
        return [{
            "priority": "MEDIUM",
            "icon": "👥",
            "title": "Load Staff Data",
            "description": "Go to Staff Planner (Page 11) to load and analyze staffing data.",
            "category": "System",
            "cost_impact": 0,
        }]

    # Shortage days action
    if staff_data["shortage_days"] > 0:
        priority = classify_action_priority(
            cost_impact_pct=15.0,  # Significant impact
            shortage_days=staff_data["shortage_days"],
            service_level_gap=10.0,
            time_sensitivity=1,  # Urgent
            staff_type="staff",
        )
        actions.append({
            "priority": priority,
            "icon": "⚠️",
            "title": f"{staff_data['shortage_days']} Shortage Days Detected",
            "description": "Some periods lack adequate coverage. Review and adjust schedules immediately.",
            "category": "Staffing",
            "cost_impact": staff_data["shortage_days"] * 2000,  # Estimated cost per shortage day
        })

    # Utilization action
    if staff_data["avg_utilization"] < 80:
        gap = 80 - staff_data["avg_utilization"]
        priority = classify_action_priority(
            cost_impact_pct=gap * 0.5,  # Impact scales with gap
            shortage_days=0,
            service_level_gap=gap,
            time_sensitivity=2,  # Soon
        )
        actions.append({
            "priority": priority,
            "icon": "📉",
            "title": f"Low Staff Utilization ({staff_data['avg_utilization']:.0f}%)",
            "description": "Staff utilization below target. Consider schedule optimization.",
            "category": "Staffing",
            "cost_impact": staff_data["weekly_labor_cost"] * (gap / 100),
        })

    # Overtime action
    if staff_data["total_overtime"] > 100:
        priority = classify_action_priority(
            cost_impact_pct=10.0,
            shortage_days=0,
            service_level_gap=0,
            time_sensitivity=2,
        )
        actions.append({
            "priority": priority,
            "icon": "⏰",
            "title": f"High Overtime ({staff_data['total_overtime']:.0f}h)",
            "description": "Review staffing levels to reduce overtime costs.",
            "category": "Staffing",
            "cost_impact": staff_data["total_overtime"] * 75,  # Overtime premium estimate
        })

    # Optimization not run
    if not staff_data["optimization_done"]:
        actions.append({
            "priority": "MEDIUM",
            "icon": "🚀",
            "title": "Run Staff Optimization",
            "description": "Apply forecast-based optimization in Staff Planner page.",
            "category": "System",
            "cost_impact": 0,
        })

    return actions


def generate_inventory_actions(inv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate inventory-related actions with ML priority classification."""
    actions = []

    if not inv_data["has_data"]:
        return [{
            "priority": "MEDIUM",
            "icon": "📦",
            "title": "Load Inventory Data",
            "description": "Go to Supply Planner (Page 12) to load and analyze inventory.",
            "category": "System",
            "cost_impact": 0,
        }]

    # Critical items action
    if inv_data["items_critical"] > 0:
        priority = classify_action_priority(
            cost_impact_pct=25.0,  # High impact
            shortage_days=inv_data["items_critical"] * 3,
            service_level_gap=15.0,
            time_sensitivity=0,  # Immediate
            staff_type="inventory",
        )
        actions.append({
            "priority": priority,
            "icon": "🚨",
            "title": f"{inv_data['items_critical']} Critical Items",
            "description": "Some inventory items at critical levels. Reorder immediately.",
            "category": "Inventory",
            "cost_impact": inv_data["items_critical"] * 5000,  # Stockout cost estimate
        })

    # Stockout risk action
    if inv_data["avg_stockout_risk"] > 5:
        priority = classify_action_priority(
            cost_impact_pct=inv_data["avg_stockout_risk"],
            shortage_days=0,
            service_level_gap=inv_data["avg_stockout_risk"],
            time_sensitivity=1 if inv_data["avg_stockout_risk"] > 10 else 2,
            staff_type="inventory",
        )
        actions.append({
            "priority": priority,
            "icon": "📦",
            "title": f"Stockout Risk: {inv_data['avg_stockout_risk']:.1f}%",
            "description": "Monitor inventory levels closely. Consider increasing safety stock.",
            "category": "Inventory",
            "cost_impact": inv_data["weekly_inventory_cost"] * (inv_data["avg_stockout_risk"] / 100),
        })

    # Service level action
    if inv_data["service_level"] < 95:
        gap = 95 - inv_data["service_level"]
        priority = classify_action_priority(
            cost_impact_pct=gap * 2,
            shortage_days=0,
            service_level_gap=gap,
            time_sensitivity=2,
            staff_type="inventory",
        )
        actions.append({
            "priority": priority,
            "icon": "📊",
            "title": f"Service Level Below Target ({inv_data['service_level']:.0f}%)",
            "description": "Target is 95%. Review safety stock levels and reorder points.",
            "category": "Inventory",
            "cost_impact": inv_data["weekly_inventory_cost"] * (gap / 100) * 2,
        })

    # Process reorder alerts
    for alert in inv_data.get("reorder_alerts", []):
        priority = "CRITICAL" if alert.get("urgency") == "CRITICAL" else "HIGH"
        actions.append({
            "priority": priority,
            "icon": "🔔",
            "title": f"Reorder Alert: {alert.get('item', 'Unknown')}",
            "description": f"Current stock low. Order {alert.get('reorder_qty', 'ASAP')}.",
            "category": "Reorder",
            "cost_impact": 0,
            "days_until_stockout": alert.get("days_until_stockout", 0),
        })

    # Optimization not run
    if not inv_data["optimization_done"]:
        actions.append({
            "priority": "MEDIUM",
            "icon": "🚀",
            "title": "Run Inventory Optimization",
            "description": "Apply EOQ/Safety Stock optimization in Supply Planner page.",
            "category": "System",
            "cost_impact": 0,
        })

    return actions


def classify_action_priority(
    cost_impact_pct: float,
    shortage_days: int,
    service_level_gap: float,
    time_sensitivity: int = 2,
    staff_type: str = "staff",
) -> str:
    """
    Classify action priority using ML model or rule-based fallback.

    Args:
        cost_impact_pct: Estimated cost impact as percentage
        shortage_days: Number of shortage days
        service_level_gap: Gap from target service level
        time_sensitivity: 0=immediate, 1=urgent, 2=soon, 3=planned
        staff_type: "staff" or "inventory"

    Returns:
        Priority level: CRITICAL, HIGH, MEDIUM, or LOW
    """
    if ML_CLASSIFIER_AVAILABLE and priority_classifier is not None:
        # Map function parameters to ActionPriorityFeatures fields
        # time_sensitivity: 0=immediate, 1=urgent, 2=soon, 3=planned → urgency_score: 0-1
        urgency_score = max(0.0, min(1.0, 1.0 - (time_sensitivity / 3.0)))
        # service_level_gap → service_level (invert: gap=0 means level=1.0)
        service_level = max(0.0, min(1.0, 1.0 - (service_level_gap / 100.0)))
        # category_priority based on staff_type
        category_priority = 1.0 if staff_type == "staff" else 0.8

        features = ActionPriorityFeatures(
            cost_impact_pct=cost_impact_pct,
            shortage_days=shortage_days,
            urgency_score=urgency_score,
            service_level=service_level,
            category_priority=category_priority,
        )
        priority_result = priority_classifier.predict_single(features)
        return priority_result.value if hasattr(priority_result, 'value') else str(priority_result)
    else:
        # Rule-based fallback
        if cost_impact_pct > 20 or shortage_days >= 5 or time_sensitivity == 0:
            return "CRITICAL"
        elif cost_impact_pct > 10 or shortage_days >= 3 or service_level_gap > 10:
            return "HIGH"
        elif cost_impact_pct > 5 or service_level_gap > 5:
            return "MEDIUM"
        else:
            return "LOW"


# =============================================================================
# FINANCIAL IMPACT CALCULATION
# =============================================================================

def calculate_financial_impact(
    forecast: Dict,
    staffing: Dict,
    inventory: Dict
) -> Dict[str, Any]:
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
            "category": "Staff Scheduling",
            "savings": monthly_staff_savings,
            "description": "Optimized staff allocation based on forecast",
            "method": "MILP" if staffing["milp_used"] else "Heuristic",
        })

    # Inventory savings
    if inventory["optimization_done"] and inventory["weekly_savings"] > 0:
        monthly_inv_savings = inventory["weekly_savings"] * 4.3
        result["inventory_savings"] = monthly_inv_savings
        result["breakdown"].append({
            "category": "Inventory Management",
            "savings": monthly_inv_savings,
            "description": "EOQ + Safety Stock optimization",
            "method": "MILP" if inventory["milp_used"] else "EOQ",
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
            "description": "Better resource planning from accurate forecasts",
            "method": forecast["source"],
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


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def generate_staff_schedule_csv(staff_data: Dict) -> str:
    """Generate CSV string for staff schedule export."""
    if staff_data.get("schedule_df") is not None:
        return staff_data["schedule_df"].to_csv(index=False)

    # Generate from current data
    today = datetime.now().date()
    rows = []
    for i in range(7):
        day_date = today + timedelta(days=i)
        rows.append({
            "Date": day_date.strftime("%Y-%m-%d"),
            "Day": day_date.strftime("%A"),
            "Doctors": int(staff_data["avg_doctors"]),
            "Nurses": int(staff_data["avg_nurses"]),
            "Support": int(staff_data["avg_support"]),
            "Total_Staff": int(staff_data["avg_doctors"] + staff_data["avg_nurses"] + staff_data["avg_support"]),
            "Estimated_Cost": f"${staff_data['daily_labor_cost']:,.0f}",
            "Utilization": f"{staff_data['avg_utilization']:.0f}%",
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_order_schedule_csv(inv_data: Dict) -> str:
    """Generate CSV string for order schedule export."""
    if inv_data.get("order_schedule_df") is not None:
        return inv_data["order_schedule_df"].to_csv(index=False)

    # Generate from current data
    today = datetime.now().date()
    rows = []

    items = [
        ("Gloves", inv_data["avg_gloves_usage"], 15),
        ("PPE Sets", inv_data["avg_ppe_usage"], 45),
        ("Medications", inv_data["avg_medication_usage"], 8),
    ]

    for item_name, daily_usage, unit_cost in items:
        # Calculate EOQ-based reorder
        annual_demand = daily_usage * 365
        ordering_cost = 50
        holding_rate = 0.20
        if annual_demand > 0:
            eoq = np.sqrt((2 * ordering_cost * annual_demand) / (unit_cost * holding_rate))
            days_between_orders = eoq / daily_usage if daily_usage > 0 else 30
            next_order = today + timedelta(days=int(days_between_orders))
        else:
            eoq = 0
            next_order = today + timedelta(days=30)

        rows.append({
            "Item": item_name,
            "Daily_Usage": f"{daily_usage:.0f}",
            "Unit_Cost": f"${unit_cost:.2f}",
            "EOQ": f"{eoq:.0f}",
            "Next_Order_Date": next_order.strftime("%Y-%m-%d"),
            "Estimated_Cost": f"${eoq * unit_cost:,.0f}",
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_actions_csv(staff_actions: List, inv_actions: List) -> str:
    """Generate CSV string for all actions export."""
    rows = []

    for action in staff_actions:
        rows.append({
            "Type": "Staff",
            "Priority": action["priority"],
            "Title": action["title"],
            "Description": action["description"],
            "Category": action["category"],
            "Estimated_Cost_Impact": f"${action.get('cost_impact', 0):,.0f}",
        })

    for action in inv_actions:
        rows.append({
            "Type": "Inventory",
            "Priority": action["priority"],
            "Title": action["title"],
            "Description": action["description"],
            "Category": action["category"],
            "Estimated_Cost_Impact": f"${action.get('cost_impact', 0):,.0f}",
        })

    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    rows.sort(key=lambda x: priority_order.get(x["Priority"], 4))

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_markdown_report(
    forecast: Dict,
    staffing: Dict,
    inventory: Dict,
    financial: Dict,
) -> str:
    """Generate comprehensive Markdown report."""
    today = datetime.now()
    report = f"""# HealthForecast AI - Action Center Report
**Generated:** {today.strftime('%B %d, %Y at %I:%M %p')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Forecast Period** | 7 Days |
| **Expected Patients** | {forecast['total_week'] if forecast['has_forecast'] else 'N/A'} |
| **Staff Utilization** | {staffing['avg_utilization']:.0f}% |
| **Service Level** | {inventory['service_level']:.0f}% |
| **Est. Monthly Savings** | ${financial['total_monthly_savings']:,.0f} |

---

## Patient Forecast

"""

    if forecast["has_forecast"]:
        report += f"""**Model:** {forecast['source']}
**Trend:** {forecast['trend'].title()}
**Confidence:** {forecast['confidence']}

| Day | Expected Patients |
|-----|------------------|
"""
        for i, patients in enumerate(forecast["forecasts"]):
            day_date = today + timedelta(days=i+1)
            peak_marker = " (Peak)" if i == forecast["peak_day"] else ""
            report += f"| {day_date.strftime('%a %b %d')} | {int(patients)}{peak_marker} |\n"
    else:
        report += "*No forecast data available. Run models in Page 10.*\n"

    report += """
---

## Staffing Overview

"""

    if staffing["has_data"]:
        report += f"""| Role | Average/Day |
|------|-------------|
| Doctors | {staffing['avg_doctors']:.0f} |
| Nurses | {staffing['avg_nurses']:.0f} |
| Support | {staffing['avg_support']:.0f} |

**Utilization:** {staffing['avg_utilization']:.0f}%
**Weekly Labor Cost:** ${staffing['weekly_labor_cost']:,.0f}
**Optimization Status:** {'Complete' if staffing['optimization_done'] else 'Not Run'}
"""
        if staffing["optimization_done"] and staffing["weekly_savings"] > 0:
            report += f"**Weekly Savings:** ${staffing['weekly_savings']:,.0f}\n"
    else:
        report += "*Staff data not loaded. Load in Page 11.*\n"

    report += """
---

## Inventory Status

"""

    if inventory["has_data"]:
        report += f"""| Item | Daily Usage |
|------|-------------|
| Gloves | {inventory['avg_gloves_usage']:.0f} |
| PPE Sets | {inventory['avg_ppe_usage']:.0f} |
| Medications | {inventory['avg_medication_usage']:.0f} |

**Service Level:** {inventory['service_level']:.0f}%
**Stockout Risk:** {inventory['avg_stockout_risk']:.1f}%
**Weekly Inventory Cost:** ${inventory['weekly_inventory_cost']:,.0f}
**Optimization Status:** {'Complete' if inventory['optimization_done'] else 'Not Run'}
"""
        if inventory["optimization_done"] and inventory["weekly_savings"] > 0:
            report += f"**Weekly Savings:** ${inventory['weekly_savings']:,.0f}\n"
    else:
        report += "*Inventory data not loaded. Load in Page 12.*\n"

    report += """
---

## Financial Impact

"""

    if financial["total_monthly_savings"] > 0:
        report += f"""**Monthly Savings:** ${financial['total_monthly_savings']:,.0f}
**Annual Projection:** ${financial['annual_savings']:,.0f}
**ROI:** {financial['roi_percent']:.0f}%

### Savings Breakdown

| Category | Monthly Savings | Method |
|----------|-----------------|--------|
"""
        for item in financial["breakdown"]:
            report += f"| {item['category']} | ${item['savings']:,.0f} | {item.get('method', 'N/A')} |\n"
    else:
        report += "*Run optimizations in Pages 11 and 12 to see potential savings.*\n"

    report += """
---

## Action Items

"""

    all_actions = staffing.get("actions", []) + inventory.get("actions", [])
    if all_actions:
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_actions.sort(key=lambda x: priority_order.get(x["priority"], 4))

        for action in all_actions:
            priority_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "📋", "LOW": "✅"}.get(action["priority"], "📋")
            report += f"""### {priority_emoji} [{action['priority']}] {action['title']}

{action['description']}

- **Category:** {action['category']}
- **Est. Cost Impact:** ${action.get('cost_impact', 0):,.0f}

"""
    else:
        report += "*No action items at this time.*\n"

    report += """
---

*Report generated by HealthForecast AI Action Center*
"""

    return report


# =============================================================================
# HERO HEADER
# =============================================================================
render_scifi_hero_header(
    title="Action Center",
    subtitle="Unified dashboard with ML-powered recommendations and export capabilities",
    status="SYSTEM ONLINE"
)

# Current date display
today = datetime.now()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 1.5rem;'>
    <span style='color: #64748b; font-size: 0.9rem;'>
        Week of {today.strftime('%B %d, %Y')} | Last updated: {today.strftime('%I:%M %p')}
    </span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# GET ALL DATA
# =============================================================================
forecast = get_forecast_data()
staffing = get_staff_data()
inventory = get_inventory_data()
financial = calculate_financial_impact(forecast, staffing, inventory)


# =============================================================================
# DATA STATUS BAR
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📡</span>
    <span class="section-title">Data Status</span>
</div>
""", unsafe_allow_html=True)

status_cols = st.columns(5)

with status_cols[0]:
    fc_status = "Available" if forecast["has_forecast"] else "Not Available"
    fc_icon = "✅" if forecast["has_forecast"] else "❌"
    fc_class = "status-good" if forecast["has_forecast"] else "status-alert"
    fc_detail = f"{forecast['source']}" if forecast["has_forecast"] else "Run Page 10"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1rem;">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">🔮 Forecast</div>
        <span class="{fc_class}">{fc_icon} {fc_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{fc_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[1]:
    st_status = "Loaded" if staffing["data_loaded"] else "Not Loaded"
    st_icon = "✅" if staffing["data_loaded"] else "❌"
    st_class = "status-good" if staffing["data_loaded"] else "status-alert"
    st_detail = f"Util: {staffing['avg_utilization']:.0f}%" if staffing["data_loaded"] else "Load in Page 11"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1rem;">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">👥 Staff</div>
        <span class="{st_class}">{st_icon} {st_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{st_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[2]:
    inv_status = "Loaded" if inventory["data_loaded"] else "Not Loaded"
    inv_icon = "✅" if inventory["data_loaded"] else "❌"
    inv_class = "status-good" if inventory["data_loaded"] else "status-alert"
    inv_detail = f"SL: {inventory['service_level']:.0f}%" if inventory["data_loaded"] else "Load in Page 12"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1rem;">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">📦 Inventory</div>
        <span class="{inv_class}">{inv_icon} {inv_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{inv_detail}</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[3]:
    opt_count = sum([staffing["optimization_done"], inventory["optimization_done"]])
    opt_status = f"{opt_count}/2 Complete" if opt_count > 0 else "None Run"
    opt_icon = "✅" if opt_count == 2 else "⚠️" if opt_count == 1 else "❌"
    opt_class = "status-good" if opt_count == 2 else "status-warning" if opt_count == 1 else "status-alert"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1rem;">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">🚀 Optimizations</div>
        <span class="{opt_class}">{opt_icon} {opt_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Staff & Inventory</div>
    </div>
    """, unsafe_allow_html=True)

with status_cols[4]:
    ml_status = "Active" if ML_CLASSIFIER_AVAILABLE else "Rule-Based"
    ml_icon = "🤖" if ML_CLASSIFIER_AVAILABLE else "📋"
    ml_class = "status-good" if ML_CLASSIFIER_AVAILABLE else "status-warning"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1rem;">
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">{ml_icon} Priority Model</div>
        <span class="{ml_class}">{ml_status}</span>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Action classification</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN TABS (4 Tabs)
# =============================================================================
tab_staff, tab_supply, tab_dashboard, tab_export = st.tabs([
    "👥 Staff Actions",
    "📦 Supply Actions",
    "📊 Combined Dashboard",
    "📤 Export & Reports"
])


# =============================================================================
# TAB 1: STAFF ACTIONS
# =============================================================================
with tab_staff:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">👥</span>
        <span class="section-title">Staff Actions & Recommendations</span>
    </div>
    """, unsafe_allow_html=True)

    if staffing["has_data"]:
        # Quick Stats Row
        staff_kpi_cols = st.columns(4)

        with staff_kpi_cols[0]:
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: #3b82f6; --accent-end: #60a5fa;">
                <div class="cmd-kpi-icon">👨‍⚕️</div>
                <div class="cmd-kpi-value">{staffing['avg_doctors']:.0f}</div>
                <div class="cmd-kpi-label">Avg Doctors/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_kpi_cols[1]:
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: #22c55e; --accent-end: #4ade80;">
                <div class="cmd-kpi-icon">👩‍⚕️</div>
                <div class="cmd-kpi-value">{staffing['avg_nurses']:.0f}</div>
                <div class="cmd-kpi-label">Avg Nurses/Day</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_kpi_cols[2]:
            util_color = "#22c55e" if staffing['avg_utilization'] >= 80 else "#f59e0b"
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: {util_color}; --accent-end: {util_color};">
                <div class="cmd-kpi-icon">📊</div>
                <div class="cmd-kpi-value" style="color: {util_color};">{staffing['avg_utilization']:.0f}%</div>
                <div class="cmd-kpi-label">Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_kpi_cols[3]:
            savings_color = "#22c55e" if staffing['weekly_savings'] > 0 else "#64748b"
            savings_val = f"${staffing['weekly_savings']:,.0f}" if staffing['weekly_savings'] > 0 else "—"
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: {savings_color}; --accent-end: {savings_color};">
                <div class="cmd-kpi-icon">💰</div>
                <div class="cmd-kpi-value" style="color: {savings_color};">{savings_val}</div>
                <div class="cmd-kpi-label">Weekly Savings</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Staff Actions List
        staff_actions = staffing.get("actions", [])

        if staff_actions:
            # Priority summary
            priority_counts = {}
            for action in staff_actions:
                p = action["priority"]
                priority_counts[p] = priority_counts.get(p, 0) + 1

            summary_cols = st.columns(4)
            for idx, priority in enumerate(["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
                with summary_cols[idx]:
                    count = priority_counts.get(priority, 0)
                    color = PRIORITY_COLORS.get(priority, "#64748b")
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{count}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">{priority}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Action cards
            for action in staff_actions:
                priority_val = str(action['priority']).lower() if action.get('priority') else 'medium'
                priority_class = f"priority-{priority_val}"
                priority_color = PRIORITY_COLORS.get(action["priority"], PRIORITY_COLORS.get(priority_val, "#3b82f6"))
                st.markdown(f"""
                <div class="priority-card {priority_class}">
                    <div class="priority-icon">{action['icon']}</div>
                    <div class="priority-content">
                        <div class="priority-title">{action['title']}</div>
                        <div class="priority-description">{action['description']}</div>
                        <div class="priority-meta">
                            <span style="background: {priority_color}20; color: {priority_color}; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{priority_val.title()}</span>
                            <span style="margin-left: 0.5rem; color: #64748b;">{action['category']}</span>
                            {f'<span style="margin-left: 0.5rem; color: #94a3b8;">Est. Impact: ${action.get("cost_impact", 0):,.0f}</span>' if action.get('cost_impact', 0) > 0 else ''}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All systems running smoothly. No urgent staff actions required.")

    else:
        st.info("""
        **Staff Data Not Loaded**

        To see staffing insights and actions:
        1. Go to **Staff Planner** (Page 11)
        2. Click "Load Staff Data"
        3. Run optimization
        4. Return here to see recommendations
        """)


# =============================================================================
# TAB 2: SUPPLY ACTIONS
# =============================================================================
with tab_supply:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📦</span>
        <span class="section-title">Supply Actions & Reorder Alerts</span>
    </div>
    """, unsafe_allow_html=True)

    if inventory["has_data"]:
        # Quick Stats Row
        inv_kpi_cols = st.columns(4)

        with inv_kpi_cols[0]:
            service_color = "#22c55e" if inventory['service_level'] >= 95 else "#f59e0b"
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: {service_color}; --accent-end: {service_color};">
                <div class="cmd-kpi-icon">📊</div>
                <div class="cmd-kpi-value" style="color: {service_color};">{inventory['service_level']:.0f}%</div>
                <div class="cmd-kpi-label">Service Level</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_kpi_cols[1]:
            risk_color = "#ef4444" if inventory['avg_stockout_risk'] > 10 else "#f59e0b" if inventory['avg_stockout_risk'] > 5 else "#22c55e"
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: {risk_color}; --accent-end: {risk_color};">
                <div class="cmd-kpi-icon">⚠️</div>
                <div class="cmd-kpi-value" style="color: {risk_color};">{inventory['avg_stockout_risk']:.1f}%</div>
                <div class="cmd-kpi-label">Stockout Risk</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_kpi_cols[2]:
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: #a855f7; --accent-end: #c084fc;">
                <div class="cmd-kpi-icon">📦</div>
                <div class="cmd-kpi-value">{inventory['items_tracked']}</div>
                <div class="cmd-kpi-label">Items Tracked</div>
                <div class="cmd-kpi-sublabel">{inventory['items_critical']} critical, {inventory['items_low']} low</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_kpi_cols[3]:
            savings_color = "#22c55e" if inventory['weekly_savings'] > 0 else "#64748b"
            savings_val = f"${inventory['weekly_savings']:,.0f}" if inventory['weekly_savings'] > 0 else "—"
            st.markdown(f"""
            <div class="cmd-kpi-card" style="--accent-color: {savings_color}; --accent-end: {savings_color};">
                <div class="cmd-kpi-icon">💰</div>
                <div class="cmd-kpi-value" style="color: {savings_color};">{savings_val}</div>
                <div class="cmd-kpi-label">Weekly Savings</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Reorder Alerts Section
        reorder_alerts = inventory.get("reorder_alerts", [])
        if reorder_alerts:
            st.markdown("#### 🔔 Reorder Alerts")
            for alert in reorder_alerts:
                urgency_color = "#ef4444" if alert.get("urgency") == "CRITICAL" else "#f59e0b"
                days_text = f" ({alert.get('days_until_stockout', 0)} days until stockout)" if alert.get('days_until_stockout') else ""
                st.markdown(f"""
                <div class="reorder-alert">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 600; color: #f1f5f9;">{alert.get('item', 'Unknown Item')}</span>
                            <span style="margin-left: 0.5rem; background: {urgency_color}20; color: {urgency_color}; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{alert.get('urgency', 'HIGH')}</span>
                        </div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">{days_text}</span>
                    </div>
                    <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.85rem;">
                        Current: {alert.get('current_stock', 'Unknown')} | Order: {alert.get('reorder_qty', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

        # Supply Actions List
        inv_actions = inventory.get("actions", [])

        if inv_actions:
            # Priority summary
            priority_counts = {}
            for action in inv_actions:
                p = action["priority"]
                priority_counts[p] = priority_counts.get(p, 0) + 1

            summary_cols = st.columns(4)
            for idx, priority in enumerate(["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
                with summary_cols[idx]:
                    count = priority_counts.get(priority, 0)
                    color = PRIORITY_COLORS.get(priority, "#64748b")
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{count}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">{priority}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Action cards
            for action in inv_actions:
                priority_val = str(action['priority']).lower() if action.get('priority') else 'medium'
                priority_class = f"priority-{priority_val}"
                priority_color = PRIORITY_COLORS.get(action["priority"], PRIORITY_COLORS.get(priority_val, "#3b82f6"))
                st.markdown(f"""
                <div class="priority-card {priority_class}">
                    <div class="priority-icon">{action['icon']}</div>
                    <div class="priority-content">
                        <div class="priority-title">{action['title']}</div>
                        <div class="priority-description">{action['description']}</div>
                        <div class="priority-meta">
                            <span style="background: {priority_color}20; color: {priority_color}; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{priority_val.title()}</span>
                            <span style="margin-left: 0.5rem; color: #64748b;">{action['category']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All inventory levels healthy. No urgent supply actions required.")

    else:
        st.info("""
        **Inventory Data Not Loaded**

        To see inventory insights and actions:
        1. Go to **Supply Planner** (Page 12)
        2. Click "Load Inventory Data"
        3. Run optimization (EOQ + Safety Stock)
        4. Return here to see recommendations
        """)


# =============================================================================
# TAB 3: COMBINED DASHBOARD
# =============================================================================
with tab_dashboard:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-title">Combined Operations Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    # Refresh button
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data", type="secondary", key="refresh_dashboard"):
            st.rerun()

    # Combined KPIs
    kpi_cols = st.columns(5)

    with kpi_cols[0]:
        patients = forecast["total_week"] if forecast["has_forecast"] else "—"
        trend_icon = "📈" if forecast["trend"] == "increasing" else ("📉" if forecast["trend"] == "decreasing" else "➡️")
        st.markdown(f"""
        <div class="cmd-kpi-card" style="--accent-color: #3b82f6; --accent-end: #60a5fa;">
            <div class="cmd-kpi-icon">🏥</div>
            <div class="cmd-kpi-value">{patients}</div>
            <div class="cmd-kpi-label">Expected Patients</div>
            <div class="cmd-kpi-sublabel">{trend_icon} {forecast["trend"].title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        util = f"{staffing['avg_utilization']:.0f}%" if staffing["has_data"] else "—"
        util_color = "#22c55e" if staffing["avg_utilization"] >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="cmd-kpi-card" style="--accent-color: {util_color}; --accent-end: {util_color};">
            <div class="cmd-kpi-icon">👥</div>
            <div class="cmd-kpi-value" style="color: {util_color};">{util}</div>
            <div class="cmd-kpi-label">Staff Utilization</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        service = f"{inventory['service_level']:.0f}%" if inventory["has_data"] else "—"
        service_color = "#22c55e" if inventory["service_level"] >= 95 else "#f59e0b"
        st.markdown(f"""
        <div class="cmd-kpi-card" style="--accent-color: {service_color}; --accent-end: {service_color};">
            <div class="cmd-kpi-icon">📦</div>
            <div class="cmd-kpi-value" style="color: {service_color};">{service}</div>
            <div class="cmd-kpi-label">Service Level</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        weekly_cost = staffing["weekly_labor_cost"] + inventory["weekly_inventory_cost"]
        st.markdown(f"""
        <div class="cmd-kpi-card" style="--accent-color: #f59e0b; --accent-end: #fbbf24;">
            <div class="cmd-kpi-icon">💵</div>
            <div class="cmd-kpi-value">${weekly_cost:,.0f}</div>
            <div class="cmd-kpi-label">Weekly Cost</div>
            <div class="cmd-kpi-sublabel">Staff + Inventory</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[4]:
        planning_complete = staffing["optimization_done"] and inventory["optimization_done"]
        status_text = "✓ Complete" if planning_complete else "In Progress"
        status_color = "#22c55e" if planning_complete else "#f59e0b"
        st.markdown(f"""
        <div class="cmd-kpi-card" style="--accent-color: {status_color}; --accent-end: {status_color};">
            <div class="cmd-kpi-icon">📋</div>
            <div class="cmd-kpi-value" style="color: {status_color};">{status_text}</div>
            <div class="cmd-kpi-label">Planning Status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Resource Allocation Summary (replaces Financial Impact)
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-title">Resource Allocation Summary</span>
    </div>
    """, unsafe_allow_html=True)

    alloc_cols = st.columns(3)
    with alloc_cols[0]:
        staff_status = "✓ Optimized" if staffing["optimization_done"] else "⏳ Pending"
        staff_color = "#22c55e" if staffing["optimization_done"] else "#94a3b8"
        st.markdown(f"""
        <div class="priority-card priority-low" style="border-left: 3px solid {staff_color};">
            <div class="priority-icon">👥</div>
            <div class="priority-content">
                <div class="priority-title">Staff Planning</div>
                <div class="priority-description" style="color: {staff_color};">{staff_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with alloc_cols[1]:
        inv_status = "✓ Optimized" if inventory["optimization_done"] else "⏳ Pending"
        inv_color = "#22c55e" if inventory["optimization_done"] else "#94a3b8"
        st.markdown(f"""
        <div class="priority-card priority-low" style="border-left: 3px solid {inv_color};">
            <div class="priority-icon">📦</div>
            <div class="priority-content">
                <div class="priority-title">Inventory Planning</div>
                <div class="priority-description" style="color: {inv_color};">{inv_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with alloc_cols[2]:
        forecast_source = forecast.get("source", "None") if forecast["has_forecast"] else "No Forecast"
        fc_color = "#22c55e" if forecast["has_forecast"] else "#94a3b8"
        st.markdown(f"""
        <div class="priority-card priority-low" style="border-left: 3px solid {fc_color};">
            <div class="priority-icon">📈</div>
            <div class="priority-content">
                <div class="priority-title">Forecast Basis</div>
                <div class="priority-description" style="color: {fc_color};">{forecast_source}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Combined Actions Summary
    all_actions = staffing.get("actions", []) + inventory.get("actions", [])
    if all_actions:
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">✅</span>
            <span class="section-title">All Action Items</span>
        </div>
        """, unsafe_allow_html=True)

        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_actions.sort(key=lambda x: priority_order.get(x["priority"], 4))

        # Summary counts
        total_actions = len(all_actions)
        critical_count = sum(1 for a in all_actions if a["priority"] == "CRITICAL")
        high_count = sum(1 for a in all_actions if a["priority"] == "HIGH")

        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
            <span style="background: rgba(239, 68, 68, 0.15); color: #ef4444; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                {critical_count} Critical
            </span>
            <span style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                {high_count} High
            </span>
            <span style="background: rgba(59, 130, 246, 0.15); color: #3b82f6; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                {total_actions} Total
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Show top actions
        for action in all_actions[:6]:
            priority_val = str(action['priority']).lower() if action.get('priority') else 'medium'
            priority_class = f"priority-{priority_val}"
            priority_color = PRIORITY_COLORS.get(action["priority"], PRIORITY_COLORS.get(priority_val, "#3b82f6"))
            st.markdown(f"""
            <div class="priority-card {priority_class}">
                <div class="priority-icon">{action['icon']}</div>
                <div class="priority-content">
                    <div class="priority-title">{action['title']}</div>
                    <div class="priority-description">{action['description']}</div>
                    <div class="priority-meta">
                        <span style="background: {priority_color}20; color: {priority_color}; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{priority_val.title()}</span>
                        <span style="margin-left: 0.5rem; color: #64748b;">{action['category']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(all_actions) > 6:
            st.info(f"Showing 6 of {len(all_actions)} actions. See Staff/Supply tabs for full list.")

    else:
        st.success("All systems running smoothly. No urgent actions required.")


# =============================================================================
# TAB 4: EXPORT & REPORTS
# =============================================================================
with tab_export:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📤</span>
        <span class="section-title">Export & Reports</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Export your optimization results and action items in various formats for sharing,
    documentation, or integration with other systems.
    """)

    # Export Cards
    export_cols = st.columns(2)

    with export_cols[0]:
        st.markdown("""
        <div class="export-card">
            <h4 style="color: #f1f5f9; margin-bottom: 1rem;">📊 CSV Exports</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Download structured data for spreadsheet analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Staff Schedule Export
        if staffing["has_data"]:
            staff_csv = generate_staff_schedule_csv(staffing)
            st.download_button(
                label="📥 Download Staff Schedule",
                data=staff_csv,
                file_name=f"staff_schedule_{today.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_staff_csv",
            )
        else:
            st.button("📥 Staff Schedule (Load data first)", disabled=True, use_container_width=True, key="export_staff_disabled")

        # Order Schedule Export
        if inventory["has_data"]:
            order_csv = generate_order_schedule_csv(inventory)
            st.download_button(
                label="📥 Download Order Schedule",
                data=order_csv,
                file_name=f"order_schedule_{today.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_order_csv",
            )
        else:
            st.button("📥 Order Schedule (Load data first)", disabled=True, use_container_width=True, key="export_order_disabled")

        # Actions Export
        all_actions = staffing.get("actions", []) + inventory.get("actions", [])
        if all_actions:
            actions_csv = generate_actions_csv(staffing.get("actions", []), inventory.get("actions", []))
            st.download_button(
                label="📥 Download All Actions",
                data=actions_csv,
                file_name=f"action_items_{today.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_actions_csv",
            )
        else:
            st.button("📥 Actions (No actions available)", disabled=True, use_container_width=True, key="export_actions_disabled")

    with export_cols[1]:
        st.markdown("""
        <div class="export-card">
            <h4 style="color: #f1f5f9; margin-bottom: 1rem;">📝 Reports</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Comprehensive summary reports for stakeholders</p>
        </div>
        """, unsafe_allow_html=True)

        # Markdown Report Export
        markdown_report = generate_markdown_report(forecast, staffing, inventory, financial)
        st.download_button(
            label="📥 Download Full Report (Markdown)",
            data=markdown_report,
            file_name=f"action_center_report_{today.strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="export_markdown",
        )

        # JSON Export
        export_json = {
            "generated_at": today.isoformat(),
            "forecast": {
                "has_data": forecast["has_forecast"],
                "source": forecast["source"],
                "total_week": forecast["total_week"],
                "avg_daily": forecast["avg_daily"],
                "trend": forecast["trend"],
                "confidence": forecast["confidence"],
                "forecasts": forecast["forecasts"],
            },
            "staffing": {
                "has_data": staffing["has_data"],
                "avg_doctors": staffing["avg_doctors"],
                "avg_nurses": staffing["avg_nurses"],
                "avg_support": staffing["avg_support"],
                "utilization": staffing["avg_utilization"],
                "weekly_cost": staffing["weekly_labor_cost"],
                "optimization_done": staffing["optimization_done"],
                "weekly_savings": staffing["weekly_savings"],
            },
            "inventory": {
                "has_data": inventory["has_data"],
                "service_level": inventory["service_level"],
                "stockout_risk": inventory["avg_stockout_risk"],
                "weekly_cost": inventory["weekly_inventory_cost"],
                "optimization_done": inventory["optimization_done"],
                "weekly_savings": inventory["weekly_savings"],
            },
            "financial": {
                "monthly_savings": financial["total_monthly_savings"],
                "annual_savings": financial["annual_savings"],
                "roi_percent": financial["roi_percent"],
                "breakdown": financial["breakdown"],
            },
            "actions": {
                "staff_actions": [
                    {"priority": a["priority"], "title": a["title"], "category": a["category"]}
                    for a in staffing.get("actions", [])
                ],
                "inventory_actions": [
                    {"priority": a["priority"], "title": a["title"], "category": a["category"]}
                    for a in inventory.get("actions", [])
                ],
            },
        }

        st.download_button(
            label="📥 Download Data (JSON)",
            data=json.dumps(export_json, indent=2),
            file_name=f"action_center_data_{today.strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
            key="export_json",
        )

    st.markdown("---")

    # Report Preview
    with st.expander("📄 Preview Report", expanded=False):
        st.markdown(markdown_report)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Quick Status")

    st.markdown(f"**Forecast:** {'✅' if forecast['has_forecast'] else '❌'} {forecast['source'] if forecast['has_forecast'] else 'N/A'}")
    st.markdown(f"**Staff Data:** {'✅' if staffing['data_loaded'] else '❌'} {'Loaded' if staffing['data_loaded'] else 'Not loaded'}")
    st.markdown(f"**Inventory:** {'✅' if inventory['data_loaded'] else '❌'} {'Loaded' if inventory['data_loaded'] else 'Not loaded'}")
    st.markdown(f"**Staff Opt:** {'✅' if staffing['optimization_done'] else '❌'} {'MILP' if staffing['milp_used'] else 'Heuristic' if staffing['optimization_done'] else 'Not run'}")
    st.markdown(f"**Inv Opt:** {'✅' if inventory['optimization_done'] else '❌'} {'MILP' if inventory['milp_used'] else 'EOQ' if inventory['optimization_done'] else 'Not run'}")

    if financial["total_monthly_savings"] > 0:
        st.markdown("---")
        st.markdown("### 💰 Savings")
        st.markdown(f"**Monthly:** ${financial['total_monthly_savings']:,.0f}")
        st.markdown(f"**Annual:** ${financial['annual_savings']:,.0f}")
        st.markdown(f"**ROI:** {financial['roi_percent']:.0f}%")

    st.markdown("---")
    st.markdown("### 🤖 ML Status")
    if ML_CLASSIFIER_AVAILABLE:
        st.success("Priority Classifier: Active")
        if priority_classifier and priority_classifier.is_trained:
            st.info("Model: Trained")
        else:
            st.info("Model: Rule-based (training data needed)")
    else:
        st.warning("Using rule-based priority classification")


# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(12)  # Action Center is page index 12

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Action Center</strong> | HealthForecast AI<br>
    <em>ML-powered recommendations for better patient care</em>
</div>
""", unsafe_allow_html=True)
