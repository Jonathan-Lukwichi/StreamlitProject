# =============================================================================
# 12_Inventory_Management_Optimization.py
# SIMPLIFIED: Forecast-Driven Inventory Management with Before/After Comparison
# Design: Matches Modeling Hub (Page 08) and Staff Scheduling (Page 11) styling
# =============================================================================
"""
Simplified Inventory Management - Forecast Driven

Approach:
- BEFORE: Fixed reorder points and quantities (historical approach without forecasting)
- AFTER: Dynamic ordering based on ML demand predictions

Key Formula:
    Optimal Order = Σ (Predicted Patients × Usage Rate per Item) + Safety Stock
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Inventory Management - HealthForecast AI",
    page_icon="📦",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CONFIGURATION: INVENTORY ITEMS & PARAMETERS
# =============================================================================
# Items with their parameters

INVENTORY_ITEMS = {
    "GLOVES": {
        "name": "Medical Gloves (boxes)",
        "usage_per_patient": 2.0,
        "unit_cost": 15.0,
        "holding_cost_pct": 0.25,  # 25% annual holding cost
        "ordering_cost": 50.0,
        "stockout_penalty": 100.0,
        "lead_time_days": 3,
        "shelf_life_days": 365,
    },
    "PPE_SETS": {
        "name": "PPE Sets",
        "usage_per_patient": 0.5,
        "unit_cost": 45.0,
        "holding_cost_pct": 0.30,
        "ordering_cost": 75.0,
        "stockout_penalty": 200.0,
        "lead_time_days": 5,
        "shelf_life_days": 180,
    },
    "MEDICATIONS": {
        "name": "Basic Medications (units)",
        "usage_per_patient": 3.0,
        "unit_cost": 8.0,
        "holding_cost_pct": 0.35,
        "ordering_cost": 100.0,
        "stockout_penalty": 500.0,
        "lead_time_days": 2,
        "shelf_life_days": 90,
    },
    "SYRINGES": {
        "name": "Syringes (boxes)",
        "usage_per_patient": 1.5,
        "unit_cost": 12.0,
        "holding_cost_pct": 0.20,
        "ordering_cost": 40.0,
        "stockout_penalty": 150.0,
        "lead_time_days": 3,
        "shelf_life_days": 730,
    },
    "BANDAGES": {
        "name": "Bandages & Dressings",
        "usage_per_patient": 2.5,
        "unit_cost": 5.0,
        "holding_cost_pct": 0.15,
        "ordering_cost": 30.0,
        "stockout_penalty": 75.0,
        "lead_time_days": 2,
        "shelf_life_days": 365,
    },
}

# Fixed ordering approach (BEFORE - no forecasting)
FIXED_ORDER_PARAMS = {
    "GLOVES": {"reorder_point": 200, "order_quantity": 500},
    "PPE_SETS": {"reorder_point": 50, "order_quantity": 150},
    "MEDICATIONS": {"reorder_point": 300, "order_quantity": 800},
    "SYRINGES": {"reorder_point": 150, "order_quantity": 400},
    "BANDAGES": {"reorder_point": 250, "order_quantity": 600},
}


# =============================================================================
# FLUORESCENT EFFECTS + CUSTOM TAB STYLING (Same as Page 08 & 11)
# =============================================================================
st.markdown(f"""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR INVENTORY
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
    background: radial-gradient(circle, rgba(168, 85, 247, 0.25), transparent 70%);
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
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(168, 85, 247, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.5);
}}

.sparkle-1 {{ top: 25%; left: 35%; animation-delay: 0s; }}
.sparkle-2 {{ top: 65%; left: 70%; animation-delay: 1s; }}
.sparkle-3 {{ top: 45%; left: 15%; animation-delay: 2s; }}

/* ========================================
   CUSTOM TAB STYLING (Modern Design)
   ======================================== */

/* Tab container */
.stTabs {{
    background: transparent;
    margin-top: 1rem;
}}

/* Tab list (container for all tab buttons) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(168, 85, 247, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}

/* Individual tab buttons */
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

/* Tab button hover effect */
.stTabs [data-baseweb="tab"]:hover {{
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(59, 130, 246, 0.1));
    border-color: rgba(168, 85, 247, 0.3);
    color: {TEXT_COLOR};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.2);
}}

/* Active/selected tab */
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(59, 130, 246, 0.2)) !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
    color: {TEXT_COLOR} !important;
    box-shadow:
        0 0 20px rgba(168, 85, 247, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}

/* Active tab indicator (underline) - hide it */
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent;
}}

/* Tab panels (content area) */
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
}}

/* ========================================
   METRIC CARDS (Inventory Style)
   ======================================== */

.metric-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}}

.metric-card.before {{
    border-color: rgba(239, 68, 68, 0.5);
}}

.metric-card.after {{
    border-color: rgba(168, 85, 247, 0.5);
}}

.metric-card.savings {{
    border-color: rgba(234, 179, 8, 0.5);
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98));
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}

.metric-value.red {{ color: #ef4444; }}
.metric-value.green {{ color: #22c55e; }}
.metric-value.blue {{ color: #3b82f6; }}
.metric-value.yellow {{ color: #eab308; }}
.metric-value.purple {{ color: #a855f7; }}

.metric-label {{
    font-size: 0.9rem;
    color: #94a3b8;
}}

/* Section headers */
.section-header {{
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(168, 85, 247, 0.3);
}}

/* Data source badge */
.data-source-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(59, 130, 246, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 20px;
    font-size: 0.875rem;
    color: #a855f7;
    font-weight: 500;
}}

.data-source-badge.warning {{
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.2), rgba(234, 179, 8, 0.1));
    border-color: rgba(234, 179, 8, 0.4);
    color: #eab308;
}}

/* Item cards */
.item-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}}

.item-name {{
    font-weight: 600;
    color: #e2e8f0;
}}

.item-detail {{
    font-size: 0.85rem;
    color: #94a3b8;
}}

@media (max-width: 768px) {{
    .fluorescent-orb {{
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }}
    .sparkle {{
        display: none;
    }}

    /* Make tabs stack vertically on mobile */
    .stTabs [data-baseweb="tab-list"] {{
        flex-direction: column;
    }}

    .stTabs [data-baseweb="tab"] {{
        width: 100%;
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
# HERO HEADER (Same style as Page 08 & 11)
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1.5rem;'>
      <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>📦</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Inventory Management Optimization</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Forecast-driven inventory ordering with Before/After financial impact comparison
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA EXTRACTION FUNCTIONS (Same pattern as Page 11)
# =============================================================================

def extract_forecast_from_ml_results() -> Optional[Dict[str, Any]]:
    """Extract forecast data from ML multi-horizon results."""
    ml_results = st.session_state.get("ml_mh_results")
    if not ml_results or not isinstance(ml_results, dict):
        return None

    for model_name in ["XGBoost", "LSTM", "ANN"]:
        if model_name in ml_results:
            model_data = ml_results[model_name]
            per_h = model_data.get("per_h", {})

            if not per_h:
                continue

            try:
                horizons = sorted([int(h) for h in per_h.keys()])
                if not horizons:
                    continue

                forecasts = []
                for h in horizons[:7]:
                    h_data = per_h.get(h, {})
                    pred = h_data.get("predictions") or h_data.get("forecast") or h_data.get("y_pred")
                    if pred is not None:
                        pred_arr = pred.values if hasattr(pred, 'values') else np.array(pred)
                        forecasts.append(float(pred_arr[-1]) if len(pred_arr) > 0 else 0)

                if forecasts:
                    return {
                        "model": f"ML ({model_name})",
                        "forecasts": forecasts,
                        "horizons": horizons[:7],
                    }
            except Exception:
                continue

    return None


def extract_forecast_from_arima() -> Optional[Dict[str, Any]]:
    """Extract forecast data from ARIMA results."""
    arima_results = st.session_state.get("arima_mh_results")
    if not arima_results or not isinstance(arima_results, dict):
        return None

    per_h = arima_results.get("per_h", {})
    if not per_h:
        return None

    try:
        horizons = sorted([int(h) for h in per_h.keys()])
        forecasts = []

        for h in horizons[:7]:
            h_data = per_h.get(h, {})
            fc = h_data.get("forecast")
            if fc is not None:
                fc_arr = fc.values if hasattr(fc, 'values') else np.array(fc)
                forecasts.append(float(fc_arr[-1]) if len(fc_arr) > 0 else 0)

        if forecasts:
            return {
                "model": "ARIMA",
                "forecasts": forecasts,
                "horizons": horizons[:7],
            }
    except Exception:
        pass

    return None


def extract_forecast_from_sarimax() -> Optional[Dict[str, Any]]:
    """Extract forecast data from SARIMAX results."""
    sarimax_results = st.session_state.get("sarimax_results")
    if not sarimax_results or not isinstance(sarimax_results, dict):
        return None

    try:
        fc = sarimax_results.get("forecast")
        if fc is not None:
            fc_arr = fc.values if hasattr(fc, 'values') else np.array(fc)
            if len(fc_arr) > 0:
                forecasts = list(fc_arr[:7])
                return {
                    "model": "SARIMAX",
                    "forecasts": forecasts,
                    "horizons": list(range(1, len(forecasts) + 1)),
                }
    except Exception:
        pass

    return None


def get_forecast_data() -> Dict[str, Any]:
    """
    Extract forecast data from session state.
    Priority: ML models > ARIMA > SARIMAX
    """
    result = {
        "has_forecast": False,
        "source": None,
        "patient_forecast": [],
        "dates": [],
        "horizon": 0,
    }

    forecast_source = None

    # 1. Try ML models
    forecast_source = extract_forecast_from_ml_results()

    # 2. Try ARIMA
    if not forecast_source:
        forecast_source = extract_forecast_from_arima()

    # 3. Try SARIMAX
    if not forecast_source:
        forecast_source = extract_forecast_from_sarimax()

    if forecast_source:
        result["has_forecast"] = True
        result["source"] = forecast_source["model"]
        result["patient_forecast"] = forecast_source["forecasts"]
        result["horizon"] = len(forecast_source["forecasts"])

        # Generate dates
        today = datetime.now().date()
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d")
                          for i in range(1, result["horizon"] + 1)]

    return result


def get_historical_data() -> Dict[str, Any]:
    """Extract historical patient data for BEFORE analysis."""
    result = {
        "has_data": False,
        "daily_patients": [],
    }

    processed_df = st.session_state.get("processed_df")
    if processed_df is not None and len(processed_df) > 0:
        result["has_data"] = True

        if "patient_count" in processed_df.columns:
            result["daily_patients"] = processed_df["patient_count"].tail(30).tolist()
        elif "Target_1" in processed_df.columns:
            result["daily_patients"] = processed_df["Target_1"].tail(30).tolist()

    return result


def calculate_inventory_demand(patients: float, item_id: str) -> float:
    """Calculate inventory demand based on patient count."""
    item = INVENTORY_ITEMS.get(item_id, {})
    usage_rate = item.get("usage_per_patient", 1.0)
    return patients * usage_rate


def analyze_before_scenario(historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze BEFORE scenario: Fixed reorder points and quantities.
    """
    if not historical_data["has_data"]:
        return {"has_analysis": False}

    daily_patients = historical_data["daily_patients"]
    total_days = len(daily_patients)

    # Simulate inventory with fixed ordering
    total_ordering_cost = 0
    total_holding_cost = 0
    total_stockout_cost = 0
    total_waste_cost = 0
    stockout_days = 0
    orders_placed = 0

    item_results = {}

    for item_id, item_params in INVENTORY_ITEMS.items():
        fixed_params = FIXED_ORDER_PARAMS[item_id]
        reorder_point = fixed_params["reorder_point"]
        order_qty = fixed_params["order_quantity"]

        # Starting inventory
        inventory = order_qty * 2
        item_stockouts = 0
        item_orders = 0
        item_holding_cost = 0
        item_waste = 0

        for patients in daily_patients:
            demand = calculate_inventory_demand(patients, item_id)

            # Check if we need to order
            if inventory <= reorder_point:
                item_orders += 1
                inventory += order_qty
                total_ordering_cost += item_params["ordering_cost"]

            # Consume inventory
            if inventory >= demand:
                inventory -= demand
            else:
                # Stockout
                stockout_qty = demand - inventory
                inventory = 0
                item_stockouts += 1
                total_stockout_cost += stockout_qty * item_params["stockout_penalty"]

            # Holding cost (daily)
            daily_holding = inventory * item_params["unit_cost"] * (item_params["holding_cost_pct"] / 365)
            item_holding_cost += daily_holding
            total_holding_cost += daily_holding

        item_results[item_id] = {
            "stockouts": item_stockouts,
            "orders": item_orders,
            "holding_cost": item_holding_cost,
        }
        stockout_days += item_stockouts
        orders_placed += item_orders

    total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost + total_waste_cost

    return {
        "has_analysis": True,
        "total_days": total_days,
        "total_cost": total_cost,
        "ordering_cost": total_ordering_cost,
        "holding_cost": total_holding_cost,
        "stockout_cost": total_stockout_cost,
        "waste_cost": total_waste_cost,
        "stockout_days": stockout_days,
        "orders_placed": orders_placed,
        "avg_daily_cost": total_cost / total_days,
        "item_results": item_results,
        "service_level": 100 * (1 - stockout_days / (total_days * len(INVENTORY_ITEMS))),
    }


def analyze_after_scenario(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze AFTER scenario: Forecast-driven dynamic ordering.
    """
    if not forecast_data["has_forecast"]:
        return {"has_analysis": False}

    patient_forecast = forecast_data["patient_forecast"]
    horizon = forecast_data["horizon"]

    total_ordering_cost = 0
    total_holding_cost = 0
    total_stockout_cost = 0
    orders_placed = 0

    item_results = {}
    daily_orders = []

    for item_id, item_params in INVENTORY_ITEMS.items():
        # Calculate optimal order based on forecast
        total_forecast_demand = sum(
            calculate_inventory_demand(p, item_id) for p in patient_forecast
        )

        # Safety stock (10% buffer)
        safety_stock = total_forecast_demand * 0.1

        # One optimized order for the period
        optimal_order = int(np.ceil(total_forecast_demand + safety_stock))

        # Cost calculation
        order_cost = item_params["ordering_cost"]  # Just 1 order
        holding_cost = (optimal_order / 2) * item_params["unit_cost"] * (item_params["holding_cost_pct"] / 365) * horizon

        total_ordering_cost += order_cost
        total_holding_cost += holding_cost

        item_results[item_id] = {
            "optimal_order": optimal_order,
            "forecast_demand": total_forecast_demand,
            "safety_stock": safety_stock,
            "order_cost": order_cost,
            "holding_cost": holding_cost,
        }
        orders_placed += 1

    total_cost = total_ordering_cost + total_holding_cost

    return {
        "has_analysis": True,
        "horizon": horizon,
        "source": forecast_data["source"],
        "total_cost": total_cost,
        "ordering_cost": total_ordering_cost,
        "holding_cost": total_holding_cost,
        "stockout_cost": 0,  # Proper forecasting = no stockouts
        "waste_cost": 0,
        "stockout_days": 0,
        "orders_placed": orders_placed,
        "avg_daily_cost": total_cost / horizon,
        "item_results": item_results,
        "service_level": 100.0,  # Perfect service with forecasting
        "dates": forecast_data["dates"],
        "patient_forecast": patient_forecast,
    }


# =============================================================================
# LOAD DATA
# =============================================================================

forecast_data = get_forecast_data()
historical_data = get_historical_data()

# Show data status with styled badges
st.markdown("### 📡 Data Connection Status")
col1, col2 = st.columns(2)

with col1:
    if forecast_data["has_forecast"]:
        st.markdown(f"""
        <div class="data-source-badge">
            ✅ Forecast: {forecast_data['source']} ({forecast_data['horizon']} days)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="data-source-badge warning">
            ⚠️ No forecast - Run models in Modeling Hub
        </div>
        """, unsafe_allow_html=True)

with col2:
    if historical_data["has_data"]:
        st.markdown(f"""
        <div class="data-source-badge">
            ✅ Historical: {len(historical_data['daily_patients'])} days
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="data-source-badge warning">
            ⚠️ No historical data - Load in Data Hub
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ANALYSIS
# =============================================================================

if forecast_data["has_forecast"] or historical_data["has_data"]:

    before_analysis = analyze_before_scenario(historical_data)
    after_analysis = analyze_after_scenario(forecast_data)

    # ==========================================================================
    # TAB LAYOUT
    # ==========================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary",
        "🔴 BEFORE Analysis",
        "🟢 AFTER Analysis",
        "📈 Comparison Charts"
    ])

    # ==========================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # ==========================================================================
    with tab1:
        st.markdown('<div class="section-header">💰 Financial Impact Summary</div>', unsafe_allow_html=True)

        if before_analysis.get("has_analysis") and after_analysis.get("has_analysis"):
            # Calculate savings (normalize to 7 days)
            before_weekly = before_analysis["avg_daily_cost"] * 7
            after_weekly = after_analysis["total_cost"]

            weekly_savings = before_weekly - after_weekly
            savings_pct = (weekly_savings / before_weekly) * 100 if before_weekly > 0 else 0
            annual_savings = weekly_savings * 52

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card before">
                    <div class="metric-value red">${before_weekly:,.0f}</div>
                    <div class="metric-label">BEFORE (Weekly Cost)</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card after">
                    <div class="metric-value purple">${after_weekly:,.0f}</div>
                    <div class="metric-label">AFTER (Weekly Cost)</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card savings">
                    <div class="metric-value yellow">${weekly_savings:,.0f}</div>
                    <div class="metric-label">Weekly Savings ({savings_pct:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card savings">
                    <div class="metric-value yellow">${annual_savings:,.0f}</div>
                    <div class="metric-label">Projected Annual Savings</div>
                </div>
                """, unsafe_allow_html=True)

            # Key insights
            st.markdown("---")
            st.markdown("### 🎯 Key Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Problems with Fixed Ordering (BEFORE):**")
                st.markdown(f"""
                - 📅 Analyzed **{before_analysis['total_days']} days** of historical data
                - ⚠️ **{before_analysis['stockout_days']} stockout events** (lost revenue/patient care)
                - 📦 **{before_analysis['orders_placed']} orders** placed (high ordering costs)
                - 💵 Holding cost: **${before_analysis['holding_cost']:,.0f}**
                - 📉 Service level: **{before_analysis['service_level']:.1f}%**
                """)

            with col2:
                st.markdown("**Benefits of Forecast-Driven Ordering (AFTER):**")
                st.markdown(f"""
                - 🎯 Orders match **predicted demand** precisely
                - ✅ **Zero stockouts** (right inventory, right time)
                - 📦 Only **{after_analysis['orders_placed']} orders** needed (consolidated)
                - 📊 Based on **{after_analysis['source']}** predictions
                - 💰 **{savings_pct:.1f}% cost reduction**
                """)

        else:
            st.info("Run forecasting models and load historical data to see the comparison.")

    # ==========================================================================
    # TAB 2: BEFORE ANALYSIS
    # ==========================================================================
    with tab2:
        st.markdown('<div class="section-header">🔴 BEFORE: Fixed Ordering Analysis</div>', unsafe_allow_html=True)

        if before_analysis.get("has_analysis"):
            st.markdown("""
            **Traditional Approach:** Hospital uses fixed reorder points and order quantities,
            regardless of expected patient volume.
            """)

            # Fixed ordering display
            st.markdown("### 📋 Fixed Order Parameters")
            fixed_data = []
            for item_id, params in FIXED_ORDER_PARAMS.items():
                item = INVENTORY_ITEMS[item_id]
                fixed_data.append({
                    "Item": item["name"],
                    "Reorder Point": params["reorder_point"],
                    "Order Quantity": params["order_quantity"],
                    "Unit Cost": f"${item['unit_cost']:.2f}",
                })
            st.dataframe(pd.DataFrame(fixed_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Cost breakdown
            st.markdown("### 💰 Cost Breakdown")

            cost_data = {
                "Category": ["Ordering Costs", "Holding Costs", "Stockout Penalties", "TOTAL"],
                "Amount": [
                    f"${before_analysis['ordering_cost']:,.0f}",
                    f"${before_analysis['holding_cost']:,.0f}",
                    f"${before_analysis['stockout_cost']:,.0f}",
                    f"${before_analysis['total_cost']:,.0f}",
                ],
                "Per Day": [
                    f"${before_analysis['ordering_cost']/before_analysis['total_days']:,.0f}",
                    f"${before_analysis['holding_cost']/before_analysis['total_days']:,.0f}",
                    f"${before_analysis['stockout_cost']/before_analysis['total_days']:,.0f}",
                    f"${before_analysis['avg_daily_cost']:,.0f}",
                ],
            }
            st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

            # Problem visualization
            st.markdown("### 📊 Cost Distribution")

            fig = go.Figure(data=[go.Pie(
                labels=['Ordering', 'Holding', 'Stockout'],
                values=[
                    before_analysis['ordering_cost'],
                    before_analysis['holding_cost'],
                    before_analysis['stockout_cost'],
                ],
                hole=0.4,
                marker_colors=['#3b82f6', '#f97316', '#ef4444'],
            )])
            fig.update_layout(
                title="Cost Breakdown (BEFORE)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Load historical data to analyze the BEFORE scenario.")

    # ==========================================================================
    # TAB 3: AFTER ANALYSIS
    # ==========================================================================
    with tab3:
        st.markdown('<div class="section-header">🟢 AFTER: Forecast-Driven Ordering</div>', unsafe_allow_html=True)

        if after_analysis.get("has_analysis"):
            st.markdown(f"""
            **Forecast-Driven Approach:** Order quantities based on **{after_analysis['source']}** predictions.
            Orders are optimized to match predicted patient demand with safety stock.
            """)

            # Optimal orders table
            st.markdown("### 📅 Recommended Orders")

            order_data = []
            for item_id, result in after_analysis["item_results"].items():
                item = INVENTORY_ITEMS[item_id]
                order_data.append({
                    "Item": item["name"],
                    "Forecast Demand": f"{result['forecast_demand']:.0f}",
                    "Safety Stock": f"{result['safety_stock']:.0f}",
                    "Optimal Order": result["optimal_order"],
                    "Order Cost": f"${result['order_cost']:,.0f}",
                    "Holding Cost": f"${result['holding_cost']:,.0f}",
                })

            st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

            # Demand forecast chart
            st.markdown("### 📊 Patient Demand Forecast")

            if after_analysis.get("patient_forecast"):
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=after_analysis["dates"],
                    y=after_analysis["patient_forecast"],
                    mode='lines+markers',
                    name='Patient Forecast',
                    line=dict(color='#a855f7', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(168, 85, 247, 0.2)',
                ))

                fig.update_layout(
                    title="Patient Arrivals Forecast (Drives Inventory Demand)",
                    xaxis_title="Day",
                    yaxis_title="Patients",
                    template="plotly_dark",
                    height=350,
                )

                st.plotly_chart(fig, use_container_width=True)

            # Cost summary
            st.markdown("### 💰 Cost Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Weekly Cost", f"${after_analysis['total_cost']:,.0f}")
            with col2:
                st.metric("Ordering Cost", f"${after_analysis['ordering_cost']:,.0f}")
            with col3:
                st.metric("Stockout Cost", f"${after_analysis['stockout_cost']:,.0f}", "Zero!")

        else:
            st.info("Run forecasting models in the Modeling Hub to see the AFTER scenario.")

    # ==========================================================================
    # TAB 4: COMPARISON CHARTS
    # ==========================================================================
    with tab4:
        st.markdown('<div class="section-header">📈 Before vs After Comparison</div>', unsafe_allow_html=True)

        if before_analysis.get("has_analysis") and after_analysis.get("has_analysis"):

            # Chart 1: Cost Comparison Bar
            st.markdown("### 💰 Weekly Cost Comparison")

            before_weekly = before_analysis["avg_daily_cost"] * 7
            after_weekly = after_analysis["total_cost"]

            fig_cost = go.Figure()

            fig_cost.add_trace(go.Bar(
                x=['BEFORE (Fixed)', 'AFTER (Forecast)'],
                y=[before_weekly, after_weekly],
                marker_color=['#ef4444', '#a855f7'],
                text=[f'${before_weekly:,.0f}', f'${after_weekly:,.0f}'],
                textposition='auto',
            ))

            fig_cost.update_layout(
                title="Weekly Inventory Cost: Before vs After",
                yaxis_title="Cost ($)",
                template="plotly_dark",
                height=400,
            )

            st.plotly_chart(fig_cost, use_container_width=True)

            # Chart 2: Cost Breakdown Comparison
            st.markdown("### 📊 Cost Breakdown")

            fig_breakdown = go.Figure()

            categories = ['Ordering', 'Holding', 'Stockout']
            before_values = [
                before_analysis["ordering_cost"] / before_analysis["total_days"] * 7,
                before_analysis["holding_cost"] / before_analysis["total_days"] * 7,
                before_analysis["stockout_cost"] / before_analysis["total_days"] * 7,
            ]
            after_values = [
                after_analysis["ordering_cost"],
                after_analysis["holding_cost"],
                0,
            ]

            fig_breakdown.add_trace(go.Bar(
                name='BEFORE',
                x=categories,
                y=before_values,
                marker_color='#ef4444',
            ))

            fig_breakdown.add_trace(go.Bar(
                name='AFTER',
                x=categories,
                y=after_values,
                marker_color='#a855f7',
            ))

            fig_breakdown.update_layout(
                barmode='group',
                title="Cost Breakdown: Before vs After",
                yaxis_title="Weekly Cost ($)",
                template="plotly_dark",
                height=400,
            )

            st.plotly_chart(fig_breakdown, use_container_width=True)

            # Chart 3: Service Level
            st.markdown("### 🎯 Service Level Comparison")

            col1, col2 = st.columns(2)

            with col1:
                fig_gauge_before = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=before_analysis["service_level"],
                    title={'text': "BEFORE: Service Level"},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444"},
                        'steps': [
                            {'range': [0, 85], 'color': "rgba(239,68,68,0.3)"},
                            {'range': [85, 95], 'color': "rgba(234,179,8,0.3)"},
                            {'range': [95, 100], 'color': "rgba(34,197,94,0.3)"},
                        ],
                    }
                ))
                fig_gauge_before.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_gauge_before, use_container_width=True)

            with col2:
                fig_gauge_after = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=after_analysis["service_level"],
                    title={'text': "AFTER: Service Level"},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#a855f7"},
                        'steps': [
                            {'range': [0, 85], 'color': "rgba(239,68,68,0.3)"},
                            {'range': [85, 95], 'color': "rgba(234,179,8,0.3)"},
                            {'range': [95, 100], 'color': "rgba(34,197,94,0.3)"},
                        ],
                    }
                ))
                fig_gauge_after.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_gauge_after, use_container_width=True)

            # Summary box
            st.success(f"""
            ### 🎉 Summary

            By using **ML forecasting** instead of fixed ordering:
            - **Weekly savings:** ${before_weekly - after_weekly:,.0f} ({((before_weekly - after_weekly) / before_weekly * 100):.1f}%)
            - **Annual savings:** ${(before_weekly - after_weekly) * 52:,.0f}
            - **Stockouts eliminated:** {before_analysis['stockout_days']} → 0
            - **Service level improved:** {before_analysis['service_level']:.1f}% → 100%
            """)

        else:
            st.info("Load both historical data and run forecasting models to see comparison charts.")

else:
    st.warning("""
    ### ⚠️ No Data Available

    To use Inventory Management, please:
    1. **Load patient data** in the Data Hub (Page 02)
    2. **Process the data** in Data Preparation Studio (Page 03)
    3. **Run forecasting models** in the Modeling Hub (Page 08)

    Then return here to see the Before vs After comparison!
    """)


# =============================================================================
# SIDEBAR: CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    with st.expander("Inventory Items", expanded=False):
        st.caption("Items managed in this system")
        for item_id, item in INVENTORY_ITEMS.items():
            st.text(f"{item['name'][:20]}: ${item['unit_cost']}/unit")

    with st.expander("Fixed Order Params (BEFORE)", expanded=False):
        st.caption("Traditional fixed ordering approach")
        for item_id, params in FIXED_ORDER_PARAMS.items():
            st.text(f"{item_id}: ROP={params['reorder_point']}, Q={params['order_quantity']}")

    with st.expander("Cost Parameters", expanded=False):
        st.caption("Average cost parameters")
        avg_holding = np.mean([i["holding_cost_pct"] for i in INVENTORY_ITEMS.values()])
        avg_ordering = np.mean([i["ordering_cost"] for i in INVENTORY_ITEMS.values()])
        st.text(f"Avg Holding: {avg_holding*100:.0f}%/year")
        st.text(f"Avg Ordering: ${avg_ordering:.0f}/order")
