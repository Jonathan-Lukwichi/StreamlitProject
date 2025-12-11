# =============================================================================
# 13_Decision_Command_Center.py — AI-Enhanced Decision Support Dashboard
# A unified command center for hospital managers with real-time insights
# Integrates: Forecasting, Staff Scheduling, Inventory Management
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Decision Command Center - HealthForecast AI",
    page_icon="🧭",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# CUSTOM CSS - Premium Dashboard Styling
# =============================================================================
st.markdown("""
<style>
/* ========================================
   DECISION COMMAND CENTER - PREMIUM STYLES
   ======================================== */

/* Fluorescent Effects */
@keyframes float-orb {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.2; }
    50% { transform: translate(20px, -20px) scale(1.05); opacity: 0.3; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.5); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.fluorescent-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(80px);
}

.orb-1 {
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.2), transparent 70%);
    top: 10%;
    right: 15%;
    animation: float-orb 30s ease-in-out infinite;
}

.orb-2 {
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.15), transparent 70%);
    bottom: 15%;
    left: 10%;
    animation: float-orb 25s ease-in-out infinite;
    animation-delay: 5s;
}

.orb-3 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.15), transparent 70%);
    top: 50%;
    left: 50%;
    animation: float-orb 35s ease-in-out infinite;
    animation-delay: 10s;
}

/* Executive KPI Card */
.exec-kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.exec-kpi-card:hover {
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
}

.exec-kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color, #3b82f6), var(--accent-color-end, #22d3ee));
}

.exec-kpi-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.5));
}

.exec-kpi-value {
    font-size: 2.25rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.5rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.exec-kpi-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.exec-kpi-sublabel {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Alert Card */
.alert-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-left: 4px solid;
    transition: all 0.2s ease;
}

.alert-card:hover {
    transform: translateX(5px);
}

.alert-critical {
    border-left-color: #ef4444;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
}

.alert-warning {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(15, 23, 42, 0.95));
}

.alert-info {
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(15, 23, 42, 0.95));
}

.alert-success {
    border-left-color: #22c55e;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.95));
}

.alert-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}

.alert-content {
    flex: 1;
}

.alert-title {
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}

.alert-description {
    font-size: 0.85rem;
    color: #94a3b8;
}

/* Health Score Gauge */
.health-gauge {
    position: relative;
    width: 200px;
    height: 200px;
    margin: 0 auto;
}

/* Recommendation Card */
.rec-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.rec-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
}

.rec-priority {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
}

.rec-priority-high {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.rec-priority-medium {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.rec-priority-low {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
}

.rec-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}

.rec-description {
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.5;
}

.rec-impact {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
    font-size: 0.8rem;
    color: #64748b;
}

/* Status Badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-optimal {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-critical {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Section Header */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
}

.section-icon {
    font-size: 1.5rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Insight Box */
.insight-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

.insight-box-title {
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.insight-box-content {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .fluorescent-orb { display: none; }
    .exec-kpi-value { font-size: 1.75rem; }
    .exec-kpi-card { padding: 1rem; }
}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
<div class="fluorescent-orb orb-3"></div>
""", unsafe_allow_html=True)


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

@dataclass
class SystemHealth:
    """Overall system health assessment."""
    score: float  # 0-100
    status: str   # OPTIMAL, WARNING, CRITICAL
    forecast_status: str
    staffing_status: str
    inventory_status: str
    alerts: List[Dict[str, Any]]


def get_all_trained_models() -> Dict[str, Dict[str, Any]]:
    """Get all trained models from session state."""
    models = {}

    # ARIMA
    if "arima_mh_results" in st.session_state and st.session_state["arima_mh_results"]:
        data = st.session_state["arima_mh_results"]
        results_df = data.get("results_df")
        if results_df is not None and not results_df.empty:
            mae_cols = [c for c in results_df.columns if "MAE" in c.upper()]
            rmse_cols = [c for c in results_df.columns if "RMSE" in c.upper()]
            models["ARIMA"] = {
                "type": "Statistical",
                "mae": results_df[mae_cols[0]].mean() if mae_cols else None,
                "rmse": results_df[rmse_cols[0]].mean() if rmse_cols else None,
                "horizons": data.get("successful", []),
            }

    # SARIMAX
    if "sarimax_results" in st.session_state and st.session_state["sarimax_results"]:
        data = st.session_state["sarimax_results"]
        results_df = data.get("results_df")
        if results_df is not None and not results_df.empty:
            mae_cols = [c for c in results_df.columns if "MAE" in c.upper()]
            rmse_cols = [c for c in results_df.columns if "RMSE" in c.upper()]
            models["SARIMAX"] = {
                "type": "Statistical",
                "mae": results_df[mae_cols[0]].mean() if mae_cols else None,
                "rmse": results_df[rmse_cols[0]].mean() if rmse_cols else None,
                "horizons": data.get("successful", []),
            }

    # ML Models
    for model_name in ["XGBoost", "LSTM", "ANN", "RandomForest", "LightGBM"]:
        key = f"ml_mh_results_{model_name}"
        if key in st.session_state and st.session_state[key]:
            data = st.session_state[key]
            results_df = data.get("results_df")
            if results_df is not None and not results_df.empty:
                mae_cols = [c for c in results_df.columns if "MAE" in c.upper()]
                rmse_cols = [c for c in results_df.columns if "RMSE" in c.upper()]
                models[model_name] = {
                    "type": "ML",
                    "mae": results_df[mae_cols[0]].mean() if mae_cols else None,
                    "rmse": results_df[rmse_cols[0]].mean() if rmse_cols else None,
                    "horizons": data.get("successful", list(data.get("per_h", {}).keys())),
                }

    # Check for optimized models
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        opt_key = f"opt_results_{model_type}"
        if opt_key in st.session_state and st.session_state[opt_key]:
            opt_data = st.session_state[opt_key]
            if model_type in models:
                models[model_type]["optimized"] = True
                models[model_type]["best_score"] = opt_data.get("best_score")
                models[model_type]["best_params"] = opt_data.get("best_params", {})

    return models


def get_best_model() -> Tuple[Optional[str], Optional[Dict]]:
    """Get the best performing model based on MAE."""
    models = get_all_trained_models()
    if not models:
        return None, None

    # Find model with lowest MAE
    best_name = None
    best_data = None
    best_mae = float('inf')

    for name, data in models.items():
        mae = data.get("mae")
        if mae is not None and mae < best_mae:
            best_mae = mae
            best_name = name
            best_data = data

    return best_name, best_data


def get_forecast_summary() -> Dict[str, Any]:
    """Get forecast summary from Forecast Hub."""
    summary = {
        "has_forecast": False,
        "model": None,
        "uncertainty_level": None,
        "rpiw": None,
        "recommended_method": None,
        "forecast_values": [],
    }

    # Check Forecast Hub data
    if "forecast_hub_demand" in st.session_state:
        data = st.session_state["forecast_hub_demand"]
        if data:
            summary["has_forecast"] = True
            summary["model"] = data.get("model", "Unknown")
            summary["forecast_values"] = data.get("forecast", [])

    # Get uncertainty info
    summary["uncertainty_level"] = st.session_state.get("forecast_uncertainty_level", "Unknown")
    summary["rpiw"] = st.session_state.get("forecast_rpiw")
    summary["recommended_method"] = st.session_state.get("recommended_optimization_method", "Not determined")

    return summary


def get_staffing_summary() -> Dict[str, Any]:
    """Get staffing optimization summary."""
    summary = {
        "has_optimization": False,
        "total_staff": 0,
        "total_cost": 0,
        "coverage_rate": 0,
        "understaffed_periods": 0,
        "overstaffed_periods": 0,
    }

    opt_result = st.session_state.get("optimization_result")
    if opt_result:
        summary["has_optimization"] = True

        # Extract from result object or dict
        if hasattr(opt_result, "total_regular_cost"):
            summary["total_cost"] = (
                getattr(opt_result, "total_regular_cost", 0) +
                getattr(opt_result, "total_overtime_cost", 0) +
                getattr(opt_result, "total_understaff_penalty", 0)
            )
            summary["coverage_rate"] = getattr(opt_result, "coverage_rate", 0)
            summary["understaffed_periods"] = getattr(opt_result, "understaffed_periods", 0)
            summary["overstaffed_periods"] = getattr(opt_result, "overstaffed_periods", 0)

            # Count total staff from schedule
            schedule = getattr(opt_result, "schedule_df", None)
            if schedule is not None:
                staff_cols = [c for c in schedule.columns if c not in ["Date", "Hour", "Period", "Demand"]]
                summary["total_staff"] = schedule[staff_cols].sum().sum() if staff_cols else 0
        elif isinstance(opt_result, dict):
            summary["total_cost"] = opt_result.get("total_cost", 0)
            summary["coverage_rate"] = opt_result.get("coverage_rate", 0)

    return summary


def get_inventory_summary() -> Dict[str, Any]:
    """Get inventory optimization summary."""
    summary = {
        "has_data": False,
        "total_items": 0,
        "critical_alerts": 0,
        "eoq_total_cost": 0,
        "milp_total_cost": 0,
        "current_levels": {},
        "reorder_points": {},
    }

    # Check inventory items
    items = st.session_state.get("inventory_items", [])
    if items:
        summary["has_data"] = True
        summary["total_items"] = len(items)

    # Current inventory levels
    current_inv = st.session_state.get("current_inventory", {})
    summary["current_levels"] = current_inv

    # EOQ Results
    eoq_results = st.session_state.get("eoq_results")
    if eoq_results:
        summary["eoq_total_cost"] = sum(r.total_annual_cost for r in eoq_results)
        for r in eoq_results:
            summary["reorder_points"][r.item_id] = r.reorder_point

    # MILP Results
    milp_results = st.session_state.get("milp_results")
    if milp_results and hasattr(milp_results, "status") and milp_results.status == "Optimal":
        summary["milp_total_cost"] = milp_results.total_cost
        summary["reorder_points"].update(milp_results.reorder_points)

    # Count critical alerts
    for item_id, current in current_inv.items():
        rop = summary["reorder_points"].get(item_id, 0)
        if current <= rop * 0.5:
            summary["critical_alerts"] += 1

    return summary


def calculate_system_health() -> SystemHealth:
    """Calculate overall system health score."""
    alerts = []
    scores = []

    # Forecast Health
    forecast = get_forecast_summary()
    if forecast["has_forecast"]:
        forecast_score = 80
        if forecast["uncertainty_level"] == "LOW":
            forecast_score = 95
        elif forecast["uncertainty_level"] == "MEDIUM":
            forecast_score = 75
        elif forecast["uncertainty_level"] == "HIGH":
            forecast_score = 55
            alerts.append({
                "type": "warning",
                "icon": "⚠️",
                "title": "High Forecast Uncertainty",
                "description": f"RPIW is {forecast['rpiw']:.1f}% - consider Two-Stage Stochastic optimization"
            })
        scores.append(forecast_score)
        forecast_status = "OPTIMAL" if forecast_score >= 80 else ("WARNING" if forecast_score >= 60 else "CRITICAL")
    else:
        scores.append(50)
        forecast_status = "WARNING"
        alerts.append({
            "type": "info",
            "icon": "📊",
            "title": "No Forecast Available",
            "description": "Generate forecasts in the Modeling Hub to enable predictions"
        })

    # Staffing Health
    staffing = get_staffing_summary()
    if staffing["has_optimization"]:
        staffing_score = min(100, staffing["coverage_rate"])
        if staffing["understaffed_periods"] > 5:
            staffing_score -= 20
            alerts.append({
                "type": "critical",
                "icon": "🚨",
                "title": "Staffing Gaps Detected",
                "description": f"{staffing['understaffed_periods']} periods are understaffed"
            })
        scores.append(staffing_score)
        staffing_status = "OPTIMAL" if staffing_score >= 90 else ("WARNING" if staffing_score >= 70 else "CRITICAL")
    else:
        scores.append(60)
        staffing_status = "WARNING"
        alerts.append({
            "type": "info",
            "icon": "👥",
            "title": "No Staffing Optimization",
            "description": "Run staff scheduling optimization to optimize workforce"
        })

    # Inventory Health
    inventory = get_inventory_summary()
    if inventory["has_data"]:
        inventory_score = 90
        if inventory["critical_alerts"] > 0:
            inventory_score -= inventory["critical_alerts"] * 15
            alerts.append({
                "type": "critical",
                "icon": "📦",
                "title": f"{inventory['critical_alerts']} Critical Inventory Alerts",
                "description": "Some items are below 50% of reorder point"
            })
        scores.append(max(inventory_score, 30))
        inventory_status = "OPTIMAL" if inventory_score >= 80 else ("WARNING" if inventory_score >= 50 else "CRITICAL")
    else:
        scores.append(70)
        inventory_status = "WARNING"
        alerts.append({
            "type": "info",
            "icon": "📋",
            "title": "Inventory Not Configured",
            "description": "Configure inventory items and run EOQ/MILP optimization"
        })

    # Calculate overall score
    overall_score = np.mean(scores)
    overall_status = "OPTIMAL" if overall_score >= 80 else ("WARNING" if overall_score >= 60 else "CRITICAL")

    return SystemHealth(
        score=overall_score,
        status=overall_status,
        forecast_status=forecast_status,
        staffing_status=staffing_status,
        inventory_status=inventory_status,
        alerts=alerts
    )


def generate_recommendations() -> List[Dict[str, Any]]:
    """Generate AI-powered recommendations based on system state."""
    recommendations = []

    # Forecast Recommendations
    models = get_all_trained_models()
    forecast = get_forecast_summary()

    if not models:
        recommendations.append({
            "priority": "high",
            "category": "Forecasting",
            "title": "Train Forecasting Models",
            "description": "No models detected. Train ARIMA, SARIMAX, or ML models in the Modeling Hub to enable demand forecasting.",
            "impact": "Enables data-driven staffing and inventory decisions",
            "action": "Go to Modeling Hub → Select model → Train"
        })
    elif len(models) < 3:
        recommendations.append({
            "priority": "medium",
            "category": "Forecasting",
            "title": "Train Additional Models for Comparison",
            "description": f"Only {len(models)} model(s) trained. Training multiple models allows better comparison and selection.",
            "impact": "Improves forecast accuracy by 5-15%",
            "action": "Train XGBoost, LSTM, and statistical models"
        })

    # Check for hyperparameter optimization
    has_optimized = any(m.get("optimized") for m in models.values())
    if models and not has_optimized:
        recommendations.append({
            "priority": "medium",
            "category": "Forecasting",
            "title": "Run Hyperparameter Optimization",
            "description": "ML models can be improved with hyperparameter tuning using Grid Search, Random Search, or Bayesian optimization.",
            "impact": "Potential 10-20% improvement in model performance",
            "action": "Modeling Hub → Hyperparameter Optimization tab"
        })

    # Uncertainty-based recommendations
    if forecast["uncertainty_level"] == "HIGH":
        recommendations.append({
            "priority": "high",
            "category": "Optimization",
            "title": "Use Two-Stage Stochastic Optimization",
            "description": f"High uncertainty (RPIW: {forecast['rpiw']:.1f}%) requires stochastic optimization to handle demand variability.",
            "impact": "Better handling of demand uncertainty, reduced risk",
            "action": "Use scenario-based optimization for staffing and inventory"
        })
    elif forecast["uncertainty_level"] == "MEDIUM":
        recommendations.append({
            "priority": "medium",
            "category": "Optimization",
            "title": "Consider Robust Optimization",
            "description": "Medium uncertainty suggests using robust optimization with adjustable budget parameter Γ.",
            "impact": "Balances cost efficiency with protection against uncertainty",
            "action": "Enable robust optimization in Staff Scheduling"
        })

    # Staffing Recommendations
    staffing = get_staffing_summary()
    if staffing["has_optimization"]:
        if staffing["coverage_rate"] < 95:
            recommendations.append({
                "priority": "high",
                "category": "Staffing",
                "title": "Improve Staff Coverage",
                "description": f"Current coverage rate is {staffing['coverage_rate']:.1f}%. Consider adding flexible staff or adjusting shift patterns.",
                "impact": f"Increase coverage to 95%+ for better patient care",
                "action": "Re-run optimization with increased staff budget"
            })
        if staffing["overstaffed_periods"] > 10:
            recommendations.append({
                "priority": "low",
                "category": "Staffing",
                "title": "Reduce Overstaffing",
                "description": f"{staffing['overstaffed_periods']} periods have excess staff. Consider reducing or reassigning.",
                "impact": f"Potential cost savings of 5-10%",
                "action": "Review schedule and adjust staffing levels"
            })

    # Inventory Recommendations
    inventory = get_inventory_summary()
    if inventory["critical_alerts"] > 0:
        recommendations.append({
            "priority": "high",
            "category": "Inventory",
            "title": "Urgent Reorder Required",
            "description": f"{inventory['critical_alerts']} item(s) are critically low and need immediate reorder.",
            "impact": "Prevent stockouts and service disruption",
            "action": "Review Results & Alerts tab in Inventory Management"
        })

    if inventory["has_data"] and not inventory["reorder_points"]:
        recommendations.append({
            "priority": "medium",
            "category": "Inventory",
            "title": "Run EOQ/MILP Optimization",
            "description": "Calculate optimal order quantities and reorder points to minimize inventory costs.",
            "impact": "Reduce inventory costs by 15-25%",
            "action": "Inventory Management → EOQ Analysis or MILP Optimization"
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return recommendations


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧭</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Decision Command Center</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        AI-Enhanced Decision Support for Hospital Operations<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Real-time insights from Forecasting, Staffing & Inventory</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXECUTIVE SUMMARY (Always Visible)
# =============================================================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📊</span>
    <span class="section-title">Executive Summary</span>
</div>
""", unsafe_allow_html=True)

# Calculate system health
health = calculate_system_health()

# Health Score and Status Cards Row
exec_cols = st.columns([1.5, 1, 1, 1, 1])

with exec_cols[0]:
    # Health Score Gauge
    score_color = "#22c55e" if health.score >= 80 else ("#f59e0b" if health.score >= 60 else "#ef4444")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health.score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Health", 'font': {'size': 16, 'color': '#94a3b8'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#f1f5f9'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': score_color},
            'bgcolor': "rgba(30, 41, 59, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#f1f5f9", 'width': 3},
                'thickness': 0.8,
                'value': health.score
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f1f5f9'},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with exec_cols[1]:
    status_class = {"OPTIMAL": "status-optimal", "WARNING": "status-warning", "CRITICAL": "status-critical"}
    status_icon = {"OPTIMAL": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}

    st.markdown(f"""
    <div class="exec-kpi-card" style="--accent-color: #3b82f6; --accent-color-end: #22d3ee;">
        <div class="exec-kpi-icon">📈</div>
        <div class="exec-kpi-label">Forecasting</div>
        <div style="margin-top: 0.5rem;">
            <span class="status-badge {status_class[health.forecast_status]}">
                {status_icon[health.forecast_status]} {health.forecast_status}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[2]:
    st.markdown(f"""
    <div class="exec-kpi-card" style="--accent-color: #a855f7; --accent-color-end: #8b5cf6;">
        <div class="exec-kpi-icon">👥</div>
        <div class="exec-kpi-label">Staffing</div>
        <div style="margin-top: 0.5rem;">
            <span class="status-badge {status_class[health.staffing_status]}">
                {status_icon[health.staffing_status]} {health.staffing_status}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[3]:
    st.markdown(f"""
    <div class="exec-kpi-card" style="--accent-color: #f59e0b; --accent-color-end: #f97316;">
        <div class="exec-kpi-icon">📦</div>
        <div class="exec-kpi-label">Inventory</div>
        <div style="margin-top: 0.5rem;">
            <span class="status-badge {status_class[health.inventory_status]}">
                {status_icon[health.inventory_status]} {health.inventory_status}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[4]:
    best_model, best_data = get_best_model()
    if best_model:
        mae_display = f"{best_data['mae']:.2f}" if best_data.get('mae') else "—"
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #22c55e; --accent-color-end: #10b981;">
            <div class="exec-kpi-icon">🏆</div>
            <div class="exec-kpi-label">Best Model</div>
            <div class="exec-kpi-value" style="font-size: 1.25rem;">{best_model}</div>
            <div class="exec-kpi-sublabel">MAE: {mae_display}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="exec-kpi-card" style="--accent-color: #64748b; --accent-color-end: #475569;">
            <div class="exec-kpi-icon">🏆</div>
            <div class="exec-kpi-label">Best Model</div>
            <div class="exec-kpi-value" style="font-size: 1.25rem; color: #64748b;">—</div>
            <div class="exec-kpi-sublabel">No models trained</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ALERTS SECTION
# =============================================================================
if health.alerts:
    st.markdown("---")
    st.markdown("#### 🚨 Active Alerts")

    alert_cols = st.columns(2)
    for idx, alert in enumerate(health.alerts[:4]):  # Show max 4 alerts
        with alert_cols[idx % 2]:
            alert_class = f"alert-{alert['type']}"
            st.markdown(f"""
            <div class="alert-card {alert_class}">
                <div class="alert-icon">{alert['icon']}</div>
                <div class="alert-content">
                    <div class="alert-title">{alert['title']}</div>
                    <div class="alert-description">{alert['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TABS FOR DETAILED VIEWS
# =============================================================================
st.markdown("---")

tab_forecast, tab_staff, tab_inventory, tab_recommend, tab_explain = st.tabs([
    "📈 Forecasting Insights",
    "👥 Staffing Overview",
    "📦 Inventory Status",
    "💡 AI Recommendations",
    "🧩 Explainability"
])


# =============================================================================
# TAB 1: FORECASTING INSIGHTS
# =============================================================================
with tab_forecast:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-title">Forecasting Insights</span>
    </div>
    """, unsafe_allow_html=True)

    models = get_all_trained_models()
    forecast = get_forecast_summary()

    if models:
        # Model Summary Cards
        model_cols = st.columns(4)

        with model_cols[0]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-icon">🤖</div>
                <div class="exec-kpi-value">{len(models)}</div>
                <div class="exec-kpi-label">Models Trained</div>
            </div>
            """, unsafe_allow_html=True)

        with model_cols[1]:
            stat_count = sum(1 for m in models.values() if m["type"] == "Statistical")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #a855f7;">
                <div class="exec-kpi-icon">📊</div>
                <div class="exec-kpi-value">{stat_count}</div>
                <div class="exec-kpi-label">Statistical Models</div>
            </div>
            """, unsafe_allow_html=True)

        with model_cols[2]:
            ml_count = sum(1 for m in models.values() if m["type"] == "ML")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #22c55e;">
                <div class="exec-kpi-icon">🧠</div>
                <div class="exec-kpi-value">{ml_count}</div>
                <div class="exec-kpi-label">ML Models</div>
            </div>
            """, unsafe_allow_html=True)

        with model_cols[3]:
            opt_count = sum(1 for m in models.values() if m.get("optimized"))
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #f59e0b;">
                <div class="exec-kpi-icon">⚡</div>
                <div class="exec-kpi-value">{opt_count}</div>
                <div class="exec-kpi-label">Optimized Models</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Model Comparison Table
        st.markdown("#### Model Performance Comparison")

        model_data = []
        for name, data in models.items():
            model_data.append({
                "Model": name,
                "Type": data["type"],
                "MAE": round(data["mae"], 2) if data.get("mae") else "—",
                "RMSE": round(data["rmse"], 2) if data.get("rmse") else "—",
                "Horizons": len(data.get("horizons", [])),
                "Optimized": "⚡ Yes" if data.get("optimized") else "—"
            })

        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        # MAE Comparison Chart
        if any(d.get("mae") for d in models.values()):
            st.markdown("#### Model MAE Comparison")

            chart_data = [(name, data["mae"]) for name, data in models.items() if data.get("mae")]
            chart_data.sort(key=lambda x: x[1])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d[0] for d in chart_data],
                y=[d[1] for d in chart_data],
                marker_color=['#22c55e' if i == 0 else '#3b82f6' for i in range(len(chart_data))],
                text=[f"{d[1]:.2f}" for d in chart_data],
                textposition='auto'
            ))
            fig.update_layout(
                title="Mean Absolute Error by Model (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="MAE",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        # Uncertainty Analysis Summary
        st.markdown("---")
        st.markdown("#### Uncertainty Analysis")

        uncert_cols = st.columns(3)

        with uncert_cols[0]:
            level = forecast.get("uncertainty_level", "Unknown")
            level_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(level, "#64748b")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {level_color};">
                <div class="exec-kpi-label">Uncertainty Level</div>
                <div class="exec-kpi-value" style="color: {level_color};">{level}</div>
            </div>
            """, unsafe_allow_html=True)

        with uncert_cols[1]:
            rpiw = forecast.get("rpiw")
            rpiw_display = f"{rpiw:.1f}%" if rpiw else "—"
            st.markdown(f"""
            <div class="exec-kpi-card">
                <div class="exec-kpi-label">RPIW</div>
                <div class="exec-kpi-value">{rpiw_display}</div>
                <div class="exec-kpi-sublabel">Relative Prediction Interval Width</div>
            </div>
            """, unsafe_allow_html=True)

        with uncert_cols[2]:
            method = forecast.get("recommended_method", "—")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #8b5cf6;">
                <div class="exec-kpi-label">Recommended Method</div>
                <div class="exec-kpi-value" style="font-size: 1rem;">{method}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("""
        **No models trained yet.**

        To see forecasting insights:
        1. Go to **Modeling Hub** (page 08)
        2. Train ARIMA, SARIMAX, or ML models
        3. Return here to view insights
        """)


# =============================================================================
# TAB 2: STAFFING OVERVIEW
# =============================================================================
with tab_staff:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">👥</span>
        <span class="section-title">Staffing Overview</span>
    </div>
    """, unsafe_allow_html=True)

    staffing = get_staffing_summary()

    if staffing["has_optimization"]:
        # KPI Cards
        staff_cols = st.columns(4)

        with staff_cols[0]:
            coverage = staffing["coverage_rate"]
            cov_color = "#22c55e" if coverage >= 95 else ("#f59e0b" if coverage >= 80 else "#ef4444")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {cov_color};">
                <div class="exec-kpi-icon">📊</div>
                <div class="exec-kpi-value">{coverage:.1f}%</div>
                <div class="exec-kpi-label">Coverage Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[1]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-icon">💰</div>
                <div class="exec-kpi-value">${staffing['total_cost']:,.0f}</div>
                <div class="exec-kpi-label">Total Cost</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[2]:
            under = staffing["understaffed_periods"]
            under_color = "#22c55e" if under == 0 else ("#f59e0b" if under <= 5 else "#ef4444")
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {under_color};">
                <div class="exec-kpi-icon">⚠️</div>
                <div class="exec-kpi-value">{under}</div>
                <div class="exec-kpi-label">Understaffed Periods</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[3]:
            over = staffing["overstaffed_periods"]
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #64748b;">
                <div class="exec-kpi-icon">📈</div>
                <div class="exec-kpi-value">{over}</div>
                <div class="exec-kpi-label">Overstaffed Periods</div>
            </div>
            """, unsafe_allow_html=True)

        # Schedule Preview
        opt_result = st.session_state.get("optimization_result")
        if opt_result and hasattr(opt_result, "schedule_df"):
            schedule = opt_result.schedule_df
            if schedule is not None and not schedule.empty:
                st.markdown("---")
                st.markdown("#### Schedule Preview (First 10 Periods)")
                st.dataframe(schedule.head(10), use_container_width=True, hide_index=True)

    else:
        st.info("""
        **No staffing optimization run yet.**

        To see staffing insights:
        1. Go to **Staff Scheduling Optimization** (page 11)
        2. Configure parameters and run optimization
        3. Return here to view results
        """)


# =============================================================================
# TAB 3: INVENTORY STATUS
# =============================================================================
with tab_inventory:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📦</span>
        <span class="section-title">Inventory Status</span>
    </div>
    """, unsafe_allow_html=True)

    inventory = get_inventory_summary()

    if inventory["has_data"]:
        # KPI Cards
        inv_cols = st.columns(4)

        with inv_cols[0]:
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #3b82f6;">
                <div class="exec-kpi-icon">📋</div>
                <div class="exec-kpi-value">{inventory['total_items']}</div>
                <div class="exec-kpi-label">Items Tracked</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[1]:
            alert_color = "#22c55e" if inventory["critical_alerts"] == 0 else "#ef4444"
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: {alert_color};">
                <div class="exec-kpi-icon">🚨</div>
                <div class="exec-kpi-value" style="color: {alert_color};">{inventory['critical_alerts']}</div>
                <div class="exec-kpi-label">Critical Alerts</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[2]:
            eoq_cost = inventory.get("eoq_total_cost", 0)
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #22c55e;">
                <div class="exec-kpi-icon">📐</div>
                <div class="exec-kpi-value">${eoq_cost:,.0f}</div>
                <div class="exec-kpi-label">EOQ Annual Cost</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[3]:
            milp_cost = inventory.get("milp_total_cost", 0)
            st.markdown(f"""
            <div class="exec-kpi-card" style="--accent-color: #a855f7;">
                <div class="exec-kpi-icon">🔧</div>
                <div class="exec-kpi-value">${milp_cost:,.0f}</div>
                <div class="exec-kpi-label">MILP Period Cost</div>
            </div>
            """, unsafe_allow_html=True)

        # Current Inventory Levels
        if inventory["current_levels"] and inventory["reorder_points"]:
            st.markdown("---")
            st.markdown("#### Inventory Level Status")

            # Create status chart
            items = st.session_state.get("inventory_items", [])
            status_data = []

            for item in items:
                current = inventory["current_levels"].get(item.item_id, 0)
                rop = inventory["reorder_points"].get(item.item_id, 0)

                status = "OK"
                color = "#22c55e"
                if current <= rop * 0.5:
                    status = "CRITICAL"
                    color = "#ef4444"
                elif current <= rop * 0.75:
                    status = "LOW"
                    color = "#f97316"
                elif current <= rop:
                    status = "REORDER"
                    color = "#facc15"

                status_data.append({
                    "Item": item.name[:20],
                    "Current": current,
                    "Reorder Point": rop,
                    "Status": status,
                    "Color": color
                })

            if status_data:
                status_df = pd.DataFrame(status_data)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Current Level',
                    x=status_df["Item"],
                    y=status_df["Current"],
                    marker_color=status_df["Color"].tolist()
                ))
                fig.add_trace(go.Scatter(
                    name='Reorder Point',
                    x=status_df["Item"],
                    y=status_df["Reorder Point"],
                    mode='markers+lines',
                    marker=dict(symbol='line-ew', size=15, color='#ef4444'),
                    line=dict(color='#ef4444', dash='dash')
                ))
                fig.update_layout(
                    title="Current Inventory vs Reorder Points",
                    xaxis_title="Item",
                    yaxis_title="Units",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("""
        **No inventory data configured yet.**

        To see inventory insights:
        1. Go to **Inventory Management Optimization** (page 12)
        2. Load inventory data and configure items
        3. Run EOQ or MILP optimization
        4. Return here to view status
        """)


# =============================================================================
# TAB 4: AI RECOMMENDATIONS
# =============================================================================
with tab_recommend:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">💡</span>
        <span class="section-title">AI-Powered Recommendations</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <div class="insight-box-title">
            🤖 Intelligent Analysis
        </div>
        <div class="insight-box-content">
            Based on your current system state, forecast accuracy, staffing optimization,
            and inventory levels, here are prioritized recommendations to improve hospital operations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    recommendations = generate_recommendations()

    if recommendations:
        for rec in recommendations:
            priority_class = f"rec-priority-{rec['priority']}"
            priority_label = rec['priority'].upper()

            st.markdown(f"""
            <div class="rec-card">
                <span class="rec-priority {priority_class}">{priority_label} PRIORITY</span>
                <span style="float: right; color: #64748b; font-size: 0.8rem;">📁 {rec['category']}</span>
                <div class="rec-title">{rec['title']}</div>
                <div class="rec-description">{rec['description']}</div>
                <div class="rec-impact">
                    <strong>💫 Impact:</strong> {rec['impact']}<br>
                    <strong>🎯 Action:</strong> {rec['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ All systems are optimally configured! No immediate recommendations.")


# =============================================================================
# TAB 5: EXPLAINABILITY
# =============================================================================
with tab_explain:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🧩</span>
        <span class="section-title">Model Explainability</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <div class="insight-box-title">
            🔍 Understanding Model Decisions
        </div>
        <div class="insight-box-content">
            This section provides insights into how forecasting models make predictions,
            using SHAP (SHapley Additive Explanations) for global feature importance
            and LIME (Local Interpretable Model-agnostic Explanations) for individual predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    models = get_all_trained_models()
    ml_models = {k: v for k, v in models.items() if v["type"] == "ML"}

    if ml_models:
        st.markdown("#### Available ML Models for Explainability")

        selected_model = st.selectbox(
            "Select model to explain:",
            options=list(ml_models.keys()),
            key="explain_model_select"
        )

        explain_method = st.radio(
            "Explanation method:",
            ["SHAP (Global Feature Importance)", "LIME (Local Instance Explanation)"],
            horizontal=True
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Model Info")
            model_info = ml_models[selected_model]
            st.markdown(f"""
            - **Type:** {model_info['type']}
            - **MAE:** {model_info.get('mae', '—')}
            - **RMSE:** {model_info.get('rmse', '—')}
            - **Optimized:** {'Yes' if model_info.get('optimized') else 'No'}
            """)

        with col2:
            if st.button("🚀 Generate Explanation", type="primary"):
                st.info("Explainability analysis will be available in a future update. This requires SHAP/LIME libraries and model artifacts.")

        st.markdown("---")

        # Placeholder for explainability visuals
        if "shap_fig" in st.session_state and "SHAP" in explain_method:
            st.plotly_chart(st.session_state["shap_fig"], use_container_width=True)
        elif "lime_fig" in st.session_state and "LIME" in explain_method:
            st.plotly_chart(st.session_state["lime_fig"], use_container_width=True)
        else:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 2rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div style="color: #94a3b8;">
                    Click "Generate Explanation" to create feature importance visualizations.<br>
                    <em>Requires trained ML model with feature data.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("""
        **No ML models available for explainability.**

        Train XGBoost, LSTM, or ANN models in the Modeling Hub to enable
        SHAP and LIME explanations.
        """)


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Decision Command Center</strong><br>
    AI-Enhanced Decision Support for Hospital Operations<br>
    <em>Integrating Forecasting, Staffing & Inventory Optimization</em>
</div>
""", unsafe_allow_html=True)
