# =============================================================================
# 13_Decision_Command_Center.py — Hospital Manager Decision Dashboard
# Plain-language operational insights for healthcare administrators
# No ML knowledge required - just actionable information
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
    page_title="Hospital Command Center - HealthForecast AI",
    page_icon="🏥",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# CUSTOM CSS - Clean, Professional Hospital Dashboard
# =============================================================================
st.markdown("""
<style>
/* ========================================
   HOSPITAL COMMAND CENTER - CLEAN STYLES
   ======================================== */

/* Executive Summary Card */
.exec-summary-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    height: 100%;
}

.exec-summary-card:hover {
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
}

.exec-summary-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color, #3b82f6), var(--accent-end, #22d3ee));
}

.card-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.25rem 0;
}

.card-label {
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 600;
}

.card-sublabel {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Status Indicators */
.status-good {
    color: #22c55e;
    background: rgba(34, 197, 94, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-warning {
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-alert {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.15);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Action Card */
.action-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border-left: 4px solid;
    transition: all 0.2s ease;
}

.action-card:hover {
    transform: translateX(5px);
}

.action-urgent {
    border-left-color: #ef4444;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(15, 23, 42, 0.95));
}

.action-important {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(15, 23, 42, 0.95));
}

.action-normal {
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(15, 23, 42, 0.95));
}

.action-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}

.action-content {
    flex: 1;
}

.action-title {
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
    font-size: 0.95rem;
}

.action-description {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* Savings Card */
.savings-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
    border: 2px solid rgba(34, 197, 94, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

.savings-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #22c55e;
    text-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
}

.savings-label {
    font-size: 1rem;
    color: #86efac;
    font-weight: 600;
}

/* Week Day Card */
.day-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.day-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
}

.day-card.today {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
}

.day-card.high-volume {
    border-color: #f59e0b;
    background: rgba(245, 158, 11, 0.1);
}

.day-name {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
}

.day-date {
    font-size: 0.75rem;
    color: #64748b;
}

.day-patients {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0.5rem 0;
}

.day-label {
    font-size: 0.7rem;
    color: #64748b;
}

/* Section Header */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
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
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

.insight-title {
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.insight-content {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Supply Status */
.supply-ok { color: #22c55e; }
.supply-low { color: #f59e0b; }
.supply-critical { color: #ef4444; }

/* Mobile Responsive */
@media (max-width: 768px) {
    .card-value { font-size: 1.5rem; }
    .savings-value { font-size: 2rem; }
    .day-patients { font-size: 1.25rem; }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA EXTRACTION FUNCTIONS (from pages 08-12)
# =============================================================================

def get_weekly_patient_forecast() -> Dict[str, Any]:
    """Extract 7-day patient forecast in plain terms."""
    result = {
        "has_forecast": False,
        "daily_patients": [],
        "total_week": 0,
        "avg_daily": 0,
        "peak_day": None,
        "peak_patients": 0,
        "low_day": None,
        "low_patients": float('inf'),
        "trend": "stable",
        "confidence": "Unknown",
    }

    # Try Forecast Hub first
    if "forecast_hub_demand" in st.session_state:
        data = st.session_state["forecast_hub_demand"]
        if data and "forecast" in data:
            forecast = list(data["forecast"])[:7]
            if forecast:
                result["has_forecast"] = True
                result["daily_patients"] = [round(p) for p in forecast]
                result["total_week"] = sum(result["daily_patients"])
                result["avg_daily"] = round(np.mean(forecast))

                # Find peak and low days
                for i, p in enumerate(forecast):
                    if p > result["peak_patients"]:
                        result["peak_patients"] = round(p)
                        result["peak_day"] = i
                    if p < result["low_patients"]:
                        result["low_patients"] = round(p)
                        result["low_day"] = i

                # Determine trend
                if len(forecast) >= 3:
                    first_half = np.mean(forecast[:len(forecast)//2])
                    second_half = np.mean(forecast[len(forecast)//2:])
                    if second_half > first_half * 1.1:
                        result["trend"] = "increasing"
                    elif second_half < first_half * 0.9:
                        result["trend"] = "decreasing"

                # Get confidence level (translate RPIW to plain language)
                rpiw = st.session_state.get("forecast_rpiw")
                if rpiw:
                    if rpiw < 15:
                        result["confidence"] = "High"
                    elif rpiw < 30:
                        result["confidence"] = "Medium"
                    else:
                        result["confidence"] = "Low"

    # Fallback to ML models
    if not result["has_forecast"]:
        for model in ["XGBoost", "LSTM", "ANN", "ARIMA", "SARIMAX"]:
            key = f"ml_mh_results_{model}" if model in ["XGBoost", "LSTM", "ANN"] else f"{model.lower()}_mh_results"
            if key in st.session_state:
                data = st.session_state[key]
                if data and "per_h" in data:
                    per_h = data["per_h"]
                    if per_h:
                        forecast = []
                        for h in range(1, 8):
                            if h in per_h and "y_test_pred" in per_h[h]:
                                pred = per_h[h]["y_test_pred"]
                                if hasattr(pred, "__len__") and len(pred) > 0:
                                    forecast.append(float(pred[-1]))

                        if forecast:
                            result["has_forecast"] = True
                            result["daily_patients"] = [round(p) for p in forecast[:7]]
                            result["total_week"] = sum(result["daily_patients"])
                            result["avg_daily"] = round(np.mean(forecast[:7]))
                            result["confidence"] = "Medium"
                            break

    return result


def get_staffing_insights() -> Dict[str, Any]:
    """Extract staffing optimization results in plain terms."""
    result = {
        "has_data": False,
        "total_staff_needed": 0,
        "shifts_covered": 0,
        "shifts_understaffed": 0,
        "overtime_hours": 0,
        "total_labor_cost": 0,
        "coverage_percent": 0,
        "cost_per_patient": 0,
        "recommendations": [],
    }

    opt_result = st.session_state.get("optimization_result")
    if opt_result:
        result["has_data"] = True

        if hasattr(opt_result, "total_regular_cost"):
            result["total_labor_cost"] = (
                getattr(opt_result, "total_regular_cost", 0) +
                getattr(opt_result, "total_overtime_cost", 0)
            )
            result["coverage_percent"] = getattr(opt_result, "coverage_rate", 0)
            result["shifts_understaffed"] = getattr(opt_result, "understaffed_periods", 0)

            # Get schedule details
            schedule = getattr(opt_result, "schedule_df", None)
            if schedule is not None and not schedule.empty:
                staff_cols = [c for c in schedule.columns if c not in ["Date", "Hour", "Period", "Demand"]]
                if staff_cols:
                    result["total_staff_needed"] = int(schedule[staff_cols].sum().sum())
                    result["shifts_covered"] = len(schedule) - result["shifts_understaffed"]

        elif isinstance(opt_result, dict):
            result["total_labor_cost"] = opt_result.get("total_cost", 0)
            result["coverage_percent"] = opt_result.get("coverage_rate", 0)

        # Generate recommendations
        if result["coverage_percent"] < 95:
            result["recommendations"].append({
                "priority": "high",
                "text": f"Coverage is at {result['coverage_percent']:.0f}% - consider adding {max(1, result['shifts_understaffed'])} more staff"
            })

        if result["shifts_understaffed"] > 5:
            result["recommendations"].append({
                "priority": "urgent",
                "text": f"{result['shifts_understaffed']} shifts are understaffed - urgent action needed"
            })

    return result


def get_inventory_insights() -> Dict[str, Any]:
    """Extract inventory status in plain terms."""
    result = {
        "has_data": False,
        "items_tracked": 0,
        "items_ok": 0,
        "items_low": 0,
        "items_critical": 0,
        "reorder_needed": [],
        "total_inventory_cost": 0,
        "potential_savings": 0,
        "stockout_risk": "Low",
    }

    items = st.session_state.get("inventory_items", [])
    current_inv = st.session_state.get("current_inventory", {})
    eoq_results = st.session_state.get("eoq_results")
    milp_results = st.session_state.get("milp_results")

    if items:
        result["has_data"] = True
        result["items_tracked"] = len(items)

        # Get reorder points
        reorder_points = {}
        if milp_results and hasattr(milp_results, "reorder_points"):
            reorder_points = milp_results.reorder_points
        elif eoq_results:
            for r in eoq_results:
                reorder_points[r.item_id] = r.reorder_point

        # Check each item status
        for item in items:
            current = current_inv.get(item.item_id, 0)
            rop = reorder_points.get(item.item_id, 0)

            if rop > 0:
                if current <= rop * 0.5:
                    result["items_critical"] += 1
                    result["reorder_needed"].append({
                        "name": item.name,
                        "current": current,
                        "needed": round(rop * 2),
                        "urgency": "CRITICAL"
                    })
                elif current <= rop:
                    result["items_low"] += 1
                    result["reorder_needed"].append({
                        "name": item.name,
                        "current": current,
                        "needed": round(rop * 1.5),
                        "urgency": "LOW"
                    })
                else:
                    result["items_ok"] += 1
            else:
                result["items_ok"] += 1

        # Get costs
        if eoq_results:
            result["total_inventory_cost"] = sum(r.total_annual_cost for r in eoq_results)

        if milp_results and hasattr(milp_results, "total_cost"):
            result["potential_savings"] = max(0, result["total_inventory_cost"] - milp_results.total_cost * 12)

        # Overall risk
        if result["items_critical"] > 0:
            result["stockout_risk"] = "High"
        elif result["items_low"] > 2:
            result["stockout_risk"] = "Medium"

    return result


def calculate_financial_impact() -> Dict[str, Any]:
    """Calculate financial savings and ROI from using the system."""
    result = {
        "staffing_savings": 0,
        "inventory_savings": 0,
        "efficiency_gains": 0,
        "total_monthly_savings": 0,
        "annual_savings": 0,
        "roi_percent": 0,
        "breakdown": [],
    }

    # Staffing savings calculation
    staffing = get_staffing_insights()
    if staffing["has_data"]:
        # Assume 15% efficiency gain from optimized scheduling
        baseline_cost = staffing["total_labor_cost"] / 0.85  # What it would cost without optimization
        result["staffing_savings"] = baseline_cost - staffing["total_labor_cost"]
        result["breakdown"].append({
            "category": "Staff Scheduling Optimization",
            "savings": result["staffing_savings"],
            "description": "Reduced overtime and better shift allocation"
        })

    # Inventory savings
    inventory = get_inventory_insights()
    if inventory["has_data"] and inventory["potential_savings"] > 0:
        result["inventory_savings"] = inventory["potential_savings"] / 12  # Monthly
        result["breakdown"].append({
            "category": "Inventory Optimization",
            "savings": result["inventory_savings"],
            "description": "Optimal order quantities and reduced holding costs"
        })

    # Efficiency gains from accurate forecasting
    forecast = get_weekly_patient_forecast()
    if forecast["has_forecast"]:
        # Estimate: Better forecasting saves ~5% in resource allocation
        weekly_patients = forecast["total_week"]
        avg_cost_per_patient = 150  # Rough estimate
        monthly_patients = weekly_patients * 4.3
        result["efficiency_gains"] = monthly_patients * avg_cost_per_patient * 0.05
        result["breakdown"].append({
            "category": "Demand Forecasting Accuracy",
            "savings": result["efficiency_gains"],
            "description": "Better resource preparation and reduced waste"
        })

    result["total_monthly_savings"] = (
        result["staffing_savings"] +
        result["inventory_savings"] +
        result["efficiency_gains"]
    )
    result["annual_savings"] = result["total_monthly_savings"] * 12

    # Assume system costs ~$500/month
    system_cost = 500
    if result["total_monthly_savings"] > 0:
        result["roi_percent"] = ((result["total_monthly_savings"] - system_cost) / system_cost) * 100

    return result


def generate_action_items() -> List[Dict[str, Any]]:
    """Generate prioritized action items for hospital manager."""
    actions = []

    forecast = get_weekly_patient_forecast()
    staffing = get_staffing_insights()
    inventory = get_inventory_insights()

    # Forecast-based actions
    if forecast["has_forecast"]:
        if forecast["trend"] == "increasing":
            actions.append({
                "priority": "important",
                "icon": "📈",
                "title": "Patient Volume Increasing",
                "description": f"Expect {forecast['peak_patients']} patients on peak day. Consider alerting additional on-call staff.",
                "category": "Staffing"
            })

        if forecast["confidence"] == "Low":
            actions.append({
                "priority": "normal",
                "icon": "🔄",
                "title": "Forecast Confidence is Low",
                "description": "Predictions have wider uncertainty. Prepare contingency plans for both high and low scenarios.",
                "category": "Planning"
            })
    else:
        actions.append({
            "priority": "important",
            "icon": "📊",
            "title": "No Patient Forecast Available",
            "description": "Run the forecasting models in Modeling Hub to enable data-driven planning.",
            "category": "System"
        })

    # Staffing actions
    if staffing["has_data"]:
        if staffing["shifts_understaffed"] > 0:
            actions.append({
                "priority": "urgent",
                "icon": "👥",
                "title": f"{staffing['shifts_understaffed']} Shifts Understaffed",
                "description": "Some periods don't have adequate coverage. Review schedule and assign additional staff.",
                "category": "Staffing"
            })

        if staffing["coverage_percent"] < 90:
            actions.append({
                "priority": "urgent",
                "icon": "⚠️",
                "title": f"Staff Coverage Only {staffing['coverage_percent']:.0f}%",
                "description": "Target is 95%+ coverage. Consider hiring temporary staff or adjusting shifts.",
                "category": "Staffing"
            })
    else:
        actions.append({
            "priority": "normal",
            "icon": "📋",
            "title": "Run Staff Optimization",
            "description": "Go to Staff Scheduling page to optimize your workforce allocation based on patient forecast.",
            "category": "System"
        })

    # Inventory actions
    if inventory["has_data"]:
        if inventory["items_critical"] > 0:
            for item in inventory["reorder_needed"]:
                if item["urgency"] == "CRITICAL":
                    actions.append({
                        "priority": "urgent",
                        "icon": "🚨",
                        "title": f"CRITICAL: {item['name']} Running Out",
                        "description": f"Only {item['current']} units left. Order {item['needed']} units immediately.",
                        "category": "Inventory"
                    })

        if inventory["items_low"] > 0:
            low_items = [i["name"] for i in inventory["reorder_needed"] if i["urgency"] == "LOW"]
            if low_items:
                actions.append({
                    "priority": "important",
                    "icon": "📦",
                    "title": f"{len(low_items)} Items Need Reordering",
                    "description": f"Low stock: {', '.join(low_items[:3])}{'...' if len(low_items) > 3 else ''}",
                    "category": "Inventory"
                })

    # Sort by priority
    priority_order = {"urgent": 0, "important": 1, "normal": 2}
    actions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return actions


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🏥</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 0.5rem;'>Hospital Command Center</h1>
      <p class='hf-feature-description' style='font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
        Your weekly operations dashboard<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Plan staff, supplies, and resources based on predicted patient volumes</span>
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
# EXECUTIVE SUMMARY ROW
# =============================================================================
forecast = get_weekly_patient_forecast()
staffing = get_staffing_insights()
inventory = get_inventory_insights()
financial = calculate_financial_impact()

st.markdown("""
<div class="section-header">
    <span class="section-icon">📊</span>
    <span class="section-title">This Week at a Glance</span>
</div>
""", unsafe_allow_html=True)

exec_cols = st.columns(5)

with exec_cols[0]:
    patients = forecast["total_week"] if forecast["has_forecast"] else "—"
    trend_icon = "📈" if forecast["trend"] == "increasing" else ("📉" if forecast["trend"] == "decreasing" else "➡️")
    st.markdown(f"""
    <div class="exec-summary-card" style="--accent-color: #3b82f6; --accent-end: #60a5fa;">
        <div class="card-icon">🏥</div>
        <div class="card-value">{patients}</div>
        <div class="card-label">Expected Patients</div>
        <div class="card-sublabel">{trend_icon} {forecast["trend"].title()} trend</div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[1]:
    avg = forecast["avg_daily"] if forecast["has_forecast"] else "—"
    conf_class = "status-good" if forecast["confidence"] == "High" else ("status-warning" if forecast["confidence"] == "Medium" else "status-alert")
    st.markdown(f"""
    <div class="exec-summary-card" style="--accent-color: #8b5cf6; --accent-end: #a78bfa;">
        <div class="card-icon">📊</div>
        <div class="card-value">{avg}</div>
        <div class="card-label">Avg Daily Patients</div>
        <div class="card-sublabel"><span class="{conf_class}">{forecast["confidence"]} Confidence</span></div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[2]:
    coverage = f"{staffing['coverage_percent']:.0f}%" if staffing["has_data"] else "—"
    cov_class = "status-good" if staffing["coverage_percent"] >= 95 else ("status-warning" if staffing["coverage_percent"] >= 80 else "status-alert")
    st.markdown(f"""
    <div class="exec-summary-card" style="--accent-color: #22c55e; --accent-end: #4ade80;">
        <div class="card-icon">👥</div>
        <div class="card-value">{coverage}</div>
        <div class="card-label">Staff Coverage</div>
        <div class="card-sublabel"><span class="{cov_class if staffing['has_data'] else ''}">{"✅ Good" if staffing["coverage_percent"] >= 95 else ("⚠️ Needs attention" if staffing["has_data"] else "Not configured")}</span></div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[3]:
    supply_status = "✅ OK" if inventory["items_critical"] == 0 and inventory["items_low"] <= 1 else (f"⚠️ {inventory['items_low']} Low" if inventory["items_critical"] == 0 else f"🚨 {inventory['items_critical']} Critical")
    supply_class = "status-good" if inventory["items_critical"] == 0 and inventory["items_low"] <= 1 else ("status-warning" if inventory["items_critical"] == 0 else "status-alert")
    st.markdown(f"""
    <div class="exec-summary-card" style="--accent-color: #f59e0b; --accent-end: #fbbf24;">
        <div class="card-icon">📦</div>
        <div class="card-value">{inventory["items_tracked"]}</div>
        <div class="card-label">Supply Items</div>
        <div class="card-sublabel"><span class="{supply_class if inventory['has_data'] else ''}">{supply_status if inventory["has_data"] else "Not configured"}</span></div>
    </div>
    """, unsafe_allow_html=True)

with exec_cols[4]:
    savings = f"${financial['total_monthly_savings']:,.0f}" if financial["total_monthly_savings"] > 0 else "—"
    st.markdown(f"""
    <div class="exec-summary-card" style="--accent-color: #10b981; --accent-end: #34d399;">
        <div class="card-icon">💰</div>
        <div class="card-value" style="color: #22c55e;">{savings}</div>
        <div class="card-label">Monthly Savings</div>
        <div class="card-sublabel">From optimization</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 7-DAY PATIENT FORECAST
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="section-header">
    <span class="section-icon">📅</span>
    <span class="section-title">7-Day Patient Forecast</span>
</div>
""", unsafe_allow_html=True)

if forecast["has_forecast"]:
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 What This Means For You</div>
        <div class="insight-content">
            Plan your staff schedules and supply orders based on these predicted patient volumes.
            Higher patient days need more nurses on duty and adequate supplies ready.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Day cards
    day_cols = st.columns(7)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for i, col in enumerate(day_cols):
        with col:
            if i < len(forecast["daily_patients"]):
                patients = forecast["daily_patients"][i]
                day_date = today + timedelta(days=i)
                is_today = i == 0
                is_peak = i == forecast["peak_day"]

                card_class = "today" if is_today else ("high-volume" if is_peak else "")

                st.markdown(f"""
                <div class="day-card {card_class}">
                    <div class="day-name">{day_names[day_date.weekday()]}</div>
                    <div class="day-date">{day_date.strftime('%b %d')}</div>
                    <div class="day-patients">{patients}</div>
                    <div class="day-label">patients</div>
                    {"<span class='status-warning' style='font-size: 0.7rem;'>Peak Day</span>" if is_peak else ""}
                </div>
                """, unsafe_allow_html=True)

    # Weekly summary chart
    st.markdown("#### Weekly Volume Distribution")

    fig = go.Figure()

    dates = [(today + timedelta(days=i)).strftime('%a\n%b %d') for i in range(len(forecast["daily_patients"]))]
    colors = ['#f59e0b' if i == forecast["peak_day"] else '#3b82f6' for i in range(len(forecast["daily_patients"]))]

    fig.add_trace(go.Bar(
        x=dates,
        y=forecast["daily_patients"],
        marker_color=colors,
        text=forecast["daily_patients"],
        textposition='auto',
        hovertemplate='%{x}<br>%{y} patients<extra></extra>'
    ))

    # Add average line
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
        height=300,
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("""
    **📊 No Patient Forecast Available**

    To see predicted patient volumes:
    1. Go to **Modeling Hub** (page 08) and train a forecasting model
    2. Then visit **Forecast Hub** (page 10) to generate predictions
    3. Return here to see your weekly planning dashboard
    """)


# =============================================================================
# ACTION ITEMS
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="section-header">
    <span class="section-icon">✅</span>
    <span class="section-title">Action Items for This Week</span>
</div>
""", unsafe_allow_html=True)

actions = generate_action_items()

if actions:
    col1, col2 = st.columns(2)

    for idx, action in enumerate(actions[:6]):  # Show max 6 actions
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
# DETAILED TABS
# =============================================================================
st.markdown("---")

tab_staff, tab_supplies, tab_financial, tab_weekly = st.tabs([
    "👥 Staffing Plan",
    "📦 Supply Status",
    "💰 Financial Impact",
    "📋 Weekly Report"
])


# =============================================================================
# TAB 1: STAFFING PLAN
# =============================================================================
with tab_staff:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">👥</span>
        <span class="section-title">Staffing Recommendations</span>
    </div>
    """, unsafe_allow_html=True)

    if staffing["has_data"]:
        # KPI cards
        staff_cols = st.columns(4)

        with staff_cols[0]:
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: #3b82f6;">
                <div class="card-icon">👥</div>
                <div class="card-value">{staffing['total_staff_needed']}</div>
                <div class="card-label">Total Staff Hours</div>
                <div class="card-sublabel">This week</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[1]:
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: #22c55e;">
                <div class="card-icon">✅</div>
                <div class="card-value">{staffing['shifts_covered']}</div>
                <div class="card-label">Shifts Covered</div>
                <div class="card-sublabel">Properly staffed</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[2]:
            under_color = "#22c55e" if staffing['shifts_understaffed'] == 0 else "#ef4444"
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: {under_color};">
                <div class="card-icon">{"✅" if staffing['shifts_understaffed'] == 0 else "⚠️"}</div>
                <div class="card-value" style="color: {under_color};">{staffing['shifts_understaffed']}</div>
                <div class="card-label">Gaps to Fill</div>
                <div class="card-sublabel">Understaffed periods</div>
            </div>
            """, unsafe_allow_html=True)

        with staff_cols[3]:
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: #f59e0b;">
                <div class="card-icon">💰</div>
                <div class="card-value">${staffing['total_labor_cost']:,.0f}</div>
                <div class="card-label">Labor Cost</div>
                <div class="card-sublabel">Optimized schedule</div>
            </div>
            """, unsafe_allow_html=True)

        # Recommendations
        if staffing["recommendations"]:
            st.markdown("#### 💡 Recommendations")
            for rec in staffing["recommendations"]:
                priority_class = "action-urgent" if rec["priority"] == "urgent" else "action-important"
                st.markdown(f"""
                <div class="action-card {priority_class}">
                    <div class="action-icon">{"🚨" if rec["priority"] == "urgent" else "⚠️"}</div>
                    <div class="action-content">
                        <div class="action-description">{rec['text']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Link to detailed scheduling
        st.markdown("---")
        st.info("📋 **Need to adjust the schedule?** Go to **Staff Scheduling Optimization** (page 11) to modify staff assignments.")

    else:
        st.info("""
        **👥 Staff Scheduling Not Configured**

        To optimize your staffing:
        1. First, generate a patient forecast in **Modeling Hub**
        2. Then go to **Staff Scheduling Optimization** (page 11)
        3. Configure your staff types and run optimization
        4. Return here to see staffing recommendations
        """)


# =============================================================================
# TAB 2: SUPPLY STATUS
# =============================================================================
with tab_supplies:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📦</span>
        <span class="section-title">Supply & Inventory Status</span>
    </div>
    """, unsafe_allow_html=True)

    if inventory["has_data"]:
        # Status overview
        inv_cols = st.columns(4)

        with inv_cols[0]:
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: #3b82f6;">
                <div class="card-icon">📋</div>
                <div class="card-value">{inventory['items_tracked']}</div>
                <div class="card-label">Items Tracked</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[1]:
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: #22c55e;">
                <div class="card-icon">✅</div>
                <div class="card-value" style="color: #22c55e;">{inventory['items_ok']}</div>
                <div class="card-label">Stocked OK</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[2]:
            low_color = "#f59e0b" if inventory['items_low'] > 0 else "#22c55e"
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: {low_color};">
                <div class="card-icon">{"⚠️" if inventory['items_low'] > 0 else "✅"}</div>
                <div class="card-value" style="color: {low_color};">{inventory['items_low']}</div>
                <div class="card-label">Running Low</div>
            </div>
            """, unsafe_allow_html=True)

        with inv_cols[3]:
            crit_color = "#ef4444" if inventory['items_critical'] > 0 else "#22c55e"
            st.markdown(f"""
            <div class="exec-summary-card" style="--accent-color: {crit_color};">
                <div class="card-icon">{"🚨" if inventory['items_critical'] > 0 else "✅"}</div>
                <div class="card-value" style="color: {crit_color};">{inventory['items_critical']}</div>
                <div class="card-label">Critical</div>
            </div>
            """, unsafe_allow_html=True)

        # Reorder alerts
        if inventory["reorder_needed"]:
            st.markdown("#### 🚨 Items Needing Reorder")

            for item in inventory["reorder_needed"]:
                urgency_class = "action-urgent" if item["urgency"] == "CRITICAL" else "action-important"
                icon = "🚨" if item["urgency"] == "CRITICAL" else "⚠️"

                st.markdown(f"""
                <div class="action-card {urgency_class}">
                    <div class="action-icon">{icon}</div>
                    <div class="action-content">
                        <div class="action-title">{item['name']}</div>
                        <div class="action-description">
                            Current stock: <strong>{item['current']}</strong> units |
                            Recommended order: <strong>{item['needed']}</strong> units
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ All inventory levels are healthy. No immediate reorders needed.")

        st.markdown("---")
        st.info("📦 **Need to manage inventory?** Go to **Inventory Management** (page 12) for detailed EOQ analysis and order scheduling.")

    else:
        st.info("""
        **📦 Inventory Not Configured**

        To track your supplies:
        1. Go to **Inventory Management Optimization** (page 12)
        2. Load your inventory data or configure items
        3. Run EOQ/MILP optimization
        4. Return here to see supply status
        """)


# =============================================================================
# TAB 3: FINANCIAL IMPACT
# =============================================================================
with tab_financial:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">💰</span>
        <span class="section-title">Financial Impact & Savings</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 How This System Saves Money</div>
        <div class="insight-content">
            By accurately predicting patient volumes, you can optimize staff schedules (avoiding overtime),
            order supplies more efficiently (reducing waste), and prepare resources proactively (improving patient outcomes).
        </div>
    </div>
    """, unsafe_allow_html=True)

    if financial["total_monthly_savings"] > 0:
        # Big savings display
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-label">ESTIMATED MONTHLY SAVINGS</div>
            <div class="savings-value">${financial['total_monthly_savings']:,.0f}</div>
            <div style="color: #86efac; margin-top: 0.5rem;">
                Annual projection: ${financial['annual_savings']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Savings Breakdown")

        # Breakdown chart
        if financial["breakdown"]:
            categories = [b["category"] for b in financial["breakdown"]]
            values = [b["savings"] for b in financial["breakdown"]]

            fig = go.Figure(data=[go.Pie(
                labels=categories,
                values=values,
                hole=0.4,
                marker_colors=['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
            )])

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detail cards
            for breakdown in financial["breakdown"]:
                st.markdown(f"""
                <div class="action-card action-normal">
                    <div class="action-icon">💵</div>
                    <div class="action-content">
                        <div class="action-title">{breakdown['category']}: ${breakdown['savings']:,.0f}/month</div>
                        <div class="action-description">{breakdown['description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ROI
        if financial["roi_percent"] > 0:
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 1rem; text-align: center; margin-top: 1rem;">
                <div style="color: #94a3b8; font-size: 0.9rem;">Return on Investment</div>
                <div style="color: #22c55e; font-size: 2rem; font-weight: 700;">{financial['roi_percent']:.0f}%</div>
                <div style="color: #64748b; font-size: 0.8rem;">System pays for itself {12 / max(1, financial['roi_percent'] / 100):.1f}x per year</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("""
        **💰 Financial Impact Calculation**

        To see your potential savings:
        1. Run patient forecasting models
        2. Optimize staff scheduling
        3. Configure inventory management

        The system will calculate your savings from:
        - Reduced overtime costs
        - Optimized inventory ordering
        - Better resource utilization
        """)


# =============================================================================
# TAB 4: WEEKLY REPORT
# =============================================================================
with tab_weekly:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📋</span>
        <span class="section-title">Weekly Operations Summary</span>
    </div>
    """, unsafe_allow_html=True)

    # Generate report content
    report_date = today.strftime("%B %d, %Y")

    st.markdown(f"### Week of {report_date}")

    # Patient Volume Section
    st.markdown("#### 🏥 Patient Volume Outlook")
    if forecast["has_forecast"]:
        st.markdown(f"""
        - **Total expected patients this week:** {forecast['total_week']}
        - **Average daily patients:** {forecast['avg_daily']}
        - **Peak day:** {day_names[(today + timedelta(days=forecast['peak_day'])).weekday()]} with {forecast['peak_patients']} patients
        - **Forecast confidence:** {forecast['confidence']}
        - **Volume trend:** {forecast['trend'].title()}
        """)
    else:
        st.markdown("*No forecast data available. Run forecasting models to enable predictions.*")

    # Staffing Section
    st.markdown("#### 👥 Staffing Status")
    if staffing["has_data"]:
        st.markdown(f"""
        - **Staff coverage rate:** {staffing['coverage_percent']:.1f}%
        - **Total staff hours needed:** {staffing['total_staff_needed']}
        - **Understaffed periods:** {staffing['shifts_understaffed']}
        - **Projected labor cost:** ${staffing['total_labor_cost']:,.0f}
        """)
    else:
        st.markdown("*Run staff scheduling optimization to see staffing plan.*")

    # Inventory Section
    st.markdown("#### 📦 Supply Status")
    if inventory["has_data"]:
        st.markdown(f"""
        - **Items tracked:** {inventory['items_tracked']}
        - **Items in good stock:** {inventory['items_ok']}
        - **Items running low:** {inventory['items_low']}
        - **Critical items:** {inventory['items_critical']}
        - **Stockout risk level:** {inventory['stockout_risk']}
        """)

        if inventory["reorder_needed"]:
            st.markdown("**Items to reorder:**")
            for item in inventory["reorder_needed"]:
                st.markdown(f"- {item['name']}: Order {item['needed']} units ({item['urgency']})")
    else:
        st.markdown("*Configure inventory management to track supplies.*")

    # Financial Section
    st.markdown("#### 💰 Financial Summary")
    if financial["total_monthly_savings"] > 0:
        st.markdown(f"""
        - **Estimated monthly savings:** ${financial['total_monthly_savings']:,.0f}
        - **Projected annual savings:** ${financial['annual_savings']:,.0f}
        - **ROI:** {financial['roi_percent']:.0f}%
        """)
    else:
        st.markdown("*Complete system setup to calculate financial impact.*")

    # Export option
    st.markdown("---")

    # Create downloadable report
    report_text = f"""
WEEKLY OPERATIONS REPORT
========================
Hospital Command Center - HealthForecast AI
Generated: {today.strftime('%Y-%m-%d %H:%M')}

PATIENT VOLUME
--------------
Total Expected: {forecast['total_week'] if forecast['has_forecast'] else 'N/A'}
Daily Average: {forecast['avg_daily'] if forecast['has_forecast'] else 'N/A'}
Peak Day: {day_names[(today + timedelta(days=forecast['peak_day'])).weekday()] if forecast['has_forecast'] and forecast['peak_day'] is not None else 'N/A'}
Confidence: {forecast['confidence']}

STAFFING
--------
Coverage Rate: {staffing['coverage_percent']:.1f}%
Staff Hours: {staffing['total_staff_needed']}
Gaps to Fill: {staffing['shifts_understaffed']}
Labor Cost: ${staffing['total_labor_cost']:,.0f}

INVENTORY
---------
Items Tracked: {inventory['items_tracked']}
Items OK: {inventory['items_ok']}
Low Stock: {inventory['items_low']}
Critical: {inventory['items_critical']}

FINANCIAL IMPACT
----------------
Monthly Savings: ${financial['total_monthly_savings']:,.0f}
Annual Projection: ${financial['annual_savings']:,.0f}
    """

    st.download_button(
        "📥 Download Weekly Report",
        data=report_text,
        file_name=f"hospital_weekly_report_{today.strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Hospital Command Center</strong> | HealthForecast AI<br>
    <em>Data-driven decisions for better patient care</em><br>
    <span style='font-size: 0.75rem;'>Questions? Contact your IT administrator</span>
</div>
""", unsafe_allow_html=True)
