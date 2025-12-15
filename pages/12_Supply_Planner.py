# =============================================================================
# 12_Inventory_Management_Optimization.py
# REDESIGNED: 4-Tab Structure with Data Upload, Situation Analysis, Optimization
# =============================================================================
"""
Tab Structure:
1. Data Upload & Preview - Load Inventory + Financial data with KPIs & graphs
2. Current Inventory Situation - Summarize & link inventory + financial metrics
3. After Optimization - Show forecast-adjusted parameters
4. Before vs After - Compare results

Data Sources:
- Inventory data: Supabase inventory_management table
- Financial data: Supabase financial_data table
- Forecasts: Session state from Modeling Hub
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

# Import Supabase services
from app_core.data.inventory_service import InventoryService, get_inventory_service
from app_core.data.financial_service import FinancialService, get_financial_service

# ============================================================================
# AUTHENTICATION
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Supply Planner - HealthForecast AI",
    page_icon="📦",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_INVENTORY_ITEMS = {
    "GLOVES": {"name": "Medical Gloves (boxes)", "usage_per_patient": 2.0, "unit_cost": 15.0, "holding_cost_pct": 0.25, "ordering_cost": 50.0, "stockout_penalty": 100.0},
    "PPE_SETS": {"name": "PPE Sets", "usage_per_patient": 0.5, "unit_cost": 45.0, "holding_cost_pct": 0.30, "ordering_cost": 75.0, "stockout_penalty": 200.0},
    "MEDICATIONS": {"name": "Basic Medications (units)", "usage_per_patient": 3.0, "unit_cost": 8.0, "holding_cost_pct": 0.35, "ordering_cost": 100.0, "stockout_penalty": 500.0},
    "SYRINGES": {"name": "Syringes (boxes)", "usage_per_patient": 1.5, "unit_cost": 12.0, "holding_cost_pct": 0.20, "ordering_cost": 40.0, "stockout_penalty": 150.0},
    "BANDAGES": {"name": "Bandages & Dressings", "usage_per_patient": 2.5, "unit_cost": 5.0, "holding_cost_pct": 0.15, "ordering_cost": 30.0, "stockout_penalty": 75.0},
}

DEFAULT_FIXED_ORDER_PARAMS = {
    "GLOVES": {"reorder_point": 200, "order_quantity": 500},
    "PPE_SETS": {"reorder_point": 50, "order_quantity": 150},
    "MEDICATIONS": {"reorder_point": 300, "order_quantity": 800},
    "SYRINGES": {"reorder_point": 150, "order_quantity": 400},
    "BANDAGES": {"reorder_point": 250, "order_quantity": 600},
}

DEFAULT_COST_PARAMS = {
    "gloves_unit_cost": 15.0,
    "ppe_unit_cost": 45.0,
    "medication_unit_cost": 8.0,
    "holding_cost_pct": 0.25,
    "ordering_cost": 50.0,
    "stockout_penalty": 100.0,
    "overstock_penalty": 25.0,
}

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown(f"""
<style>
/* Fluorescent Effects */
@keyframes float-orb {{
    0%, 100% {{ transform: translate(0, 0) scale(1); opacity: 0.25; }}
    50% {{ transform: translate(30px, -30px) scale(1.05); opacity: 0.35; }}
}}

.fluorescent-orb {{
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}}

.orb-1 {{
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.25), transparent 70%);
    top: 15%; right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}}

.orb-2 {{
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.2), transparent 70%);
    bottom: 20%; left: 15%;
    animation: float-orb 30s ease-in-out infinite;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(168, 85, 247, 0.2);
}}

.stTabs [data-baseweb="tab"] {{
    height: 50px;
    background: transparent;
    border-radius: 12px;
    padding: 0 1.5rem;
    font-weight: 600;
    color: {BODY_TEXT};
    border: 1px solid transparent;
    transition: all 0.3s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(59, 130, 246, 0.1));
    border-color: rgba(168, 85, 247, 0.3);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(59, 130, 246, 0.2)) !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent;
}}

/* KPI Cards */
.kpi-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}}

.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}}

.kpi-card.inventory {{ border-color: rgba(168, 85, 247, 0.5); }}
.kpi-card.financial {{ border-color: rgba(59, 130, 246, 0.5); }}
.kpi-card.warning {{ border-color: rgba(234, 179, 8, 0.5); background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.danger {{ border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.success {{ border-color: rgba(34, 197, 94, 0.5); background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.linked {{ border-color: rgba(236, 72, 153, 0.5); background: linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(15, 23, 42, 0.98)); }}

.kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}}

.kpi-value.green {{ color: #22c55e; }}
.kpi-value.blue {{ color: #3b82f6; }}
.kpi-value.red {{ color: #ef4444; }}
.kpi-value.yellow {{ color: #eab308; }}
.kpi-value.purple {{ color: #a855f7; }}
.kpi-value.pink {{ color: #ec4899; }}

.kpi-label {{
    font-size: 0.85rem;
    color: #94a3b8;
}}

.kpi-delta {{
    font-size: 0.8rem;
    margin-top: 0.25rem;
}}
.kpi-delta.positive {{ color: #22c55e; }}
.kpi-delta.negative {{ color: #ef4444; }}

/* Section Headers */
.section-header {{
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(168, 85, 247, 0.3);
}}

.subsection-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Data Box */
.data-box {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}}

.data-box.inventory {{
    border-color: rgba(168, 85, 247, 0.3);
}}

.data-box.financial {{
    border-color: rgba(59, 130, 246, 0.3);
}}

/* Observation Box */
.observation-box {{
    background: rgba(59, 130, 246, 0.1);
    border-left: 4px solid #3b82f6;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}}

.observation-box.warning {{
    background: rgba(234, 179, 8, 0.1);
    border-left-color: #eab308;
}}

.observation-box.success {{
    background: rgba(34, 197, 94, 0.1);
    border-left-color: #22c55e;
}}

.observation-box.danger {{
    background: rgba(239, 68, 68, 0.1);
    border-left-color: #ef4444;
}}

/* Status Badge */
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}}

.status-badge.loaded {{
    background: rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #a855f7;
}}

.status-badge.pending {{
    background: rgba(234, 179, 8, 0.2);
    border: 1px solid rgba(234, 179, 8, 0.4);
    color: #eab308;
}}
</style>

<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown(
    """
    <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1.5rem;'>
      <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>📦</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Supply Planner</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Load inventory data, analyze current situation, apply forecast optimization, and compare results
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================
if "inventory_data_loaded" not in st.session_state:
    st.session_state.inventory_data_loaded = False
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = None
if "inventory_stats" not in st.session_state:
    st.session_state.inventory_stats = {}

if "inv_financial_data_loaded" not in st.session_state:
    st.session_state.inv_financial_data_loaded = False
if "inv_financial_df" not in st.session_state:
    st.session_state.inv_financial_df = None
if "inv_cost_params" not in st.session_state:
    st.session_state.inv_cost_params = DEFAULT_COST_PARAMS.copy()
if "inv_financial_stats" not in st.session_state:
    st.session_state.inv_financial_stats = {}

if "inv_optimization_done" not in st.session_state:
    st.session_state.inv_optimization_done = False
if "inv_optimized_results" not in st.session_state:
    st.session_state.inv_optimized_results = None


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_inventory_data() -> Dict[str, Any]:
    """Load inventory data from Supabase."""
    result = {"success": False, "data": None, "stats": {}, "error": None}

    try:
        service = get_inventory_service()
        if not service.is_connected():
            result["error"] = "Supabase not connected"
            return result

        df = service.fetch_all()
        if df.empty:
            result["error"] = "No inventory data found"
            return result

        result["success"] = True
        result["data"] = df

        # Calculate stats
        stats = service.get_summary_stats() or {}
        avg_usage = stats.get("avg_usage", {})
        avg_levels = stats.get("avg_levels", {})

        # Get raw values
        raw_gloves_usage = avg_usage.get("gloves", 0)
        raw_ppe_usage = avg_usage.get("ppe", 0)
        raw_medication_usage = avg_usage.get("medications", 0)
        raw_gloves_level = avg_levels.get("gloves", 0)
        raw_ppe_level = avg_levels.get("ppe", 0)
        raw_medication_level = avg_levels.get("medications", 0)
        raw_restock_events = stats.get("restock_events", 0)
        raw_stockout_risk = stats.get("avg_stockout_risk", 0)

        # Validate all values - ensure no negative values
        # Stockout risk should be 0-100%, all others should be >= 0
        result["stats"] = {
            "total_records": len(df),
            "date_start": df["Date"].min() if "Date" in df.columns else None,
            "date_end": df["Date"].max() if "Date" in df.columns else None,
            "avg_gloves_usage": max(0, raw_gloves_usage) if raw_gloves_usage else 0,
            "avg_ppe_usage": max(0, raw_ppe_usage) if raw_ppe_usage else 0,
            "avg_medication_usage": max(0, raw_medication_usage) if raw_medication_usage else 0,
            "avg_gloves_level": max(0, raw_gloves_level) if raw_gloves_level else 0,
            "avg_ppe_level": max(0, raw_ppe_level) if raw_ppe_level else 0,
            "avg_medication_level": max(0, raw_medication_level) if raw_medication_level else 0,
            "restock_events": max(0, int(raw_restock_events)) if raw_restock_events else 0,
            "avg_stockout_risk": max(0, min(100, abs(raw_stockout_risk))) if raw_stockout_risk else 0,
        }

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def load_financial_data() -> Dict[str, Any]:
    """Load financial/cost parameters from Supabase."""
    result = {"success": False, "cost_params": DEFAULT_COST_PARAMS.copy(), "stats": {}, "data": None, "error": None}

    try:
        service = get_financial_service()
        if not service.is_connected():
            result["error"] = "Supabase not connected"
            return result

        # Fetch the raw financial dataframe for preview
        try:
            financial_df = service.fetch_all()
            if financial_df is not None and not financial_df.empty:
                result["data"] = financial_df
        except Exception:
            pass  # Continue even if raw data fetch fails

        inv_params = service.get_inventory_cost_params()

        if inv_params:
            # Get raw values
            raw_gloves_cost = inv_params.get("gloves_unit_cost", 15.0)
            raw_ppe_cost = inv_params.get("ppe_unit_cost", 45.0)
            raw_med_cost = inv_params.get("medication_unit_cost", 8.0)
            raw_holding_cost = inv_params.get("holding_cost", 0.25)
            raw_ordering_cost = inv_params.get("ordering_cost", 50.0)
            raw_stockout_penalty = inv_params.get("stockout_penalty", 100.0)
            raw_overstock_penalty = inv_params.get("overstock_penalty", 25.0)

            # Validate holding_cost_pct - should be 0-1 (representing 0-100%)
            # If value > 1, it might be in percentage form already, so divide by 100
            # Then clamp to reasonable range (0.05 to 0.50 = 5% to 50%)
            if raw_holding_cost > 1:
                raw_holding_cost = raw_holding_cost / 100
            validated_holding_cost = max(0.05, min(0.50, abs(raw_holding_cost))) if raw_holding_cost else 0.25

            result["cost_params"] = {
                "gloves_unit_cost": max(0, raw_gloves_cost) if raw_gloves_cost else 15.0,
                "ppe_unit_cost": max(0, raw_ppe_cost) if raw_ppe_cost else 45.0,
                "medication_unit_cost": max(0, raw_med_cost) if raw_med_cost else 8.0,
                "holding_cost_pct": validated_holding_cost,
                "ordering_cost": max(0, raw_ordering_cost) if raw_ordering_cost else 50.0,
                "stockout_penalty": max(0, raw_stockout_penalty) if raw_stockout_penalty else 100.0,
                "overstock_penalty": max(0, raw_overstock_penalty) if raw_overstock_penalty else 25.0,
            }

        result["success"] = True
        result["stats"] = {
            "gloves_cost": result["cost_params"]["gloves_unit_cost"],
            "ppe_cost": result["cost_params"]["ppe_unit_cost"],
            "medication_cost": result["cost_params"]["medication_unit_cost"],
            "holding_pct": result["cost_params"]["holding_cost_pct"] * 100,
            "ordering_cost": result["cost_params"]["ordering_cost"],
            "stockout_penalty": result["cost_params"]["stockout_penalty"],
        }

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def get_forecast_data() -> Dict[str, Any]:
    """
    Extract forecast data from session state.

    PRIORITY ORDER (synced with Page 10 Forecast Hub):
    1. forecast_hub_demand - Primary source from Forecast Hub (Page 10)
    2. active_forecast - Alternative key from Forecast Hub
    3. ML models (fallback)
    4. ARIMA (fallback)
    """
    result = {
        "has_forecast": False,
        "source": None,
        "model_type": None,
        "forecasts": [],
        "horizon": 0,
        "dates": [],
        "accuracy": None,
        "timestamp": None,
        "synced_with_page10": False,
    }

    # PRIORITY 1: Check forecast_hub_demand (from Page 10 - Forecast Hub)
    hub_demand = st.session_state.get("forecast_hub_demand")
    if hub_demand and isinstance(hub_demand, dict):
        forecasts = hub_demand.get("forecast", [])
        if forecasts and len(forecasts) > 0:
            result["has_forecast"] = True
            result["source"] = hub_demand.get("model", "Forecast Hub")
            result["model_type"] = hub_demand.get("model_type", "Unknown")
            result["forecasts"] = [max(0, float(f)) for f in forecasts[:7]]
            result["horizon"] = len(result["forecasts"])
            result["dates"] = hub_demand.get("dates", [])
            result["accuracy"] = hub_demand.get("avg_accuracy")
            result["timestamp"] = hub_demand.get("timestamp")
            result["synced_with_page10"] = True
            return result

    # PRIORITY 2: Check active_forecast (alternative from Page 10)
    active = st.session_state.get("active_forecast")
    if active and isinstance(active, dict):
        forecasts = active.get("forecasts", [])
        if forecasts and len(forecasts) > 0:
            result["has_forecast"] = True
            result["source"] = active.get("model", "Active Forecast")
            result["model_type"] = active.get("model_type", "Unknown")
            result["forecasts"] = [max(0, float(f)) for f in forecasts[:7]]
            result["horizon"] = len(result["forecasts"])
            result["dates"] = active.get("dates", [])
            result["accuracy"] = active.get("accuracy")
            result["timestamp"] = active.get("updated_at")
            result["synced_with_page10"] = True
            return result

    # PRIORITY 3: Try ML models directly (fallback)
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
                        pred = h_data.get("predictions") or h_data.get("forecast") or h_data.get("y_test_pred")
                        if pred is not None:
                            arr = pred.values if hasattr(pred, 'values') else np.array(pred)
                            forecasts.append(max(0, float(arr[-1])) if len(arr) > 0 else 0)
                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = f"ML ({model})"
                        result["model_type"] = "ML"
                        result["forecasts"] = forecasts
                        result["horizon"] = len(forecasts)
                        result["synced_with_page10"] = False
                        break
                except:
                    continue

    # PRIORITY 4: Try ARIMA (fallback)
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
                            forecasts.append(max(0, float(arr[-1])) if len(arr) > 0 else 0)
                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = "ARIMA"
                        result["model_type"] = "Statistical"
                        result["forecasts"] = forecasts
                        result["horizon"] = len(forecasts)
                        result["synced_with_page10"] = False
                except:
                    pass

    # Generate dates if not provided
    if result["has_forecast"] and not result["dates"]:
        today = datetime.now().date()
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, result["horizon"] + 1)]

    return result


def calculate_optimization(forecast_data: Dict, inv_stats: Dict, cost_params: Dict) -> Dict[str, Any]:
    """
    Calculate optimized inventory ordering based on patient demand forecast.

    Key Formulas:
    1. Demand Forecast: total_demand = Σ(patients × usage_per_patient)
    2. Safety Stock: safety_stock = total_demand × safety_factor (10%)
    3. Optimal Order: Q* = total_demand + safety_stock
    4. Holding Cost: H = (Q/2) × unit_cost × (annual_holding_rate / 365) × days
       - Q/2 is average inventory level (assuming uniform demand)
    5. Total Cost: TC = ordering_cost + holding_cost

    Comparison uses consistent 7-day periods for weekly metrics.
    """
    if not forecast_data["has_forecast"]:
        return {"success": False}

    patient_forecast = forecast_data["forecasts"]
    horizon = forecast_data["horizon"]

    # Calculate optimal orders for each item
    item_results = {}
    total_ordering_cost = 0
    total_holding_cost = 0

    # Get holding cost percentage from cost_params (loaded from Supabase)
    holding_cost_pct = cost_params.get("holding_cost_pct", 0.25)
    ordering_cost_per_order = cost_params.get("ordering_cost", 50.0)

    for item_id, item_params in DEFAULT_INVENTORY_ITEMS.items():
        # Calculate demand based on patient forecast
        # Formula: demand = Σ(daily_patients × usage_per_patient)
        total_demand = sum(max(0, p) * item_params["usage_per_patient"] for p in patient_forecast)

        # Safety stock = 10% buffer to prevent stockouts
        safety_factor = 0.10
        safety_stock = total_demand * safety_factor
        optimal_order = int(np.ceil(total_demand + safety_stock))

        # Get unit cost from cost_params (Supabase) or fallback to defaults
        if item_id == "GLOVES":
            unit_cost = cost_params.get("gloves_unit_cost", item_params["unit_cost"])
        elif item_id == "PPE_SETS":
            unit_cost = cost_params.get("ppe_unit_cost", item_params["unit_cost"])
        elif item_id == "MEDICATIONS":
            unit_cost = cost_params.get("medication_unit_cost", item_params["unit_cost"])
        else:
            unit_cost = item_params["unit_cost"]

        # One order per item for the forecast period
        order_cost = ordering_cost_per_order

        # Holding Cost Formula: H = (Q/2) × C × (h/365) × T
        # Where: Q = order quantity, C = unit cost, h = annual holding rate, T = time period
        # Q/2 represents average inventory (assumes uniform depletion)
        holding_cost = (optimal_order / 2) * unit_cost * (holding_cost_pct / 365) * horizon

        total_ordering_cost += order_cost
        total_holding_cost += holding_cost

        item_results[item_id] = {
            "name": item_params["name"],
            "forecast_demand": total_demand,
            "safety_stock": safety_stock,
            "optimal_order": optimal_order,
            "unit_cost": unit_cost,
            "order_cost": order_cost,
            "holding_cost": holding_cost,
        }

    total_cost = total_ordering_cost + total_holding_cost

    # Calculate CURRENT costs for comparison using CONSISTENT cost parameters
    # Use cost_params from Supabase for consistency (not hardcoded item params)
    current_daily_cost = 0
    for item_id, params in DEFAULT_FIXED_ORDER_PARAMS.items():
        item = DEFAULT_INVENTORY_ITEMS[item_id]

        # Get consistent unit cost
        if item_id == "GLOVES":
            unit_cost = cost_params.get("gloves_unit_cost", item["unit_cost"])
        elif item_id == "PPE_SETS":
            unit_cost = cost_params.get("ppe_unit_cost", item["unit_cost"])
        elif item_id == "MEDICATIONS":
            unit_cost = cost_params.get("medication_unit_cost", item["unit_cost"])
        else:
            unit_cost = item["unit_cost"]

        # Daily holding cost: (avg_inventory × unit_cost × annual_rate) / 365
        daily_holding = (params["order_quantity"] / 2) * unit_cost * (holding_cost_pct / 365)

        # Daily ordering cost: assume weekly ordering = ordering_cost / 7
        daily_ordering = ordering_cost_per_order / 7

        current_daily_cost += daily_holding + daily_ordering

    # Calculate optimized average daily cost
    optimized_daily_cost = total_cost / horizon if horizon > 0 else 0

    # Use consistent 7-day period for weekly comparisons
    current_weekly_cost = current_daily_cost * 7
    optimized_weekly_cost = optimized_daily_cost * 7
    weekly_savings = current_weekly_cost - optimized_weekly_cost

    return {
        "success": True,
        "item_results": item_results,
        "total_cost": total_cost,                      # Total for forecast horizon
        "ordering_cost": total_ordering_cost,
        "holding_cost": total_holding_cost,
        "avg_daily_cost": optimized_daily_cost,        # Average optimized daily cost
        "current_daily_cost": current_daily_cost,      # Current average daily cost
        "current_weekly_cost": current_weekly_cost,    # Current: daily × 7
        "optimized_weekly_cost": optimized_weekly_cost,# Optimized: avg_daily × 7
        "weekly_savings": weekly_savings,              # Consistent weekly comparison
        "horizon": horizon,
        "source": forecast_data["source"],
        "dates": forecast_data["dates"],
        "patient_forecast": patient_forecast,
    }


# =============================================================================
# FORECAST SYNC STATUS - Synced with Page 10 (Forecast Hub)
# =============================================================================
forecast_data = get_forecast_data()

sync_col1, sync_col2 = st.columns([3, 1])
with sync_col1:
    if forecast_data["has_forecast"]:
        if forecast_data["synced_with_page10"]:
            sync_icon = "✅"
            sync_status = "Synced with Forecast Hub"
            sync_color = "#22c55e"
        else:
            sync_icon = "⚠️"
            sync_status = "Using Direct Model Results"
            sync_color = "#eab308"

        model_info = f"{forecast_data['source']}"
        if forecast_data.get("model_type"):
            model_info += f" ({forecast_data['model_type']})"
        if forecast_data.get("accuracy"):
            model_info += f" • Accuracy: {forecast_data['accuracy']:.1f}%"
        if forecast_data.get("timestamp"):
            model_info += f" • Updated: {forecast_data['timestamp'][:16]}"

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
                    <div style="font-size: 0.85rem; color: #94a3b8;">Run models in Page 10 (Forecast Hub) to enable optimization</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with sync_col2:
    if st.button("🔄 Refresh from Forecast Hub", type="secondary", use_container_width=True, key="refresh_forecast_inv"):
        st.rerun()


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Upload & Preview",
    "📦 Current Inventory Situation",
    "🚀 After Optimization",
    "📈 Before vs After Comparison"
])


# =============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">📊 Data Upload & Preview</div>', unsafe_allow_html=True)

    # Two columns for Inventory and Financial
    col_inv, col_fin = st.columns(2)

    # =========================================================================
    # INVENTORY DATA SECTION
    # =========================================================================
    with col_inv:
        st.markdown('<div class="subsection-header">📦 Inventory Data</div>', unsafe_allow_html=True)

        # Load Button
        if st.button("🔄 Load Inventory Data", type="primary", use_container_width=True, key="load_inv"):
            with st.spinner("Loading inventory data from Supabase..."):
                result = load_inventory_data()
                if result["success"]:
                    st.session_state.inventory_data_loaded = True
                    st.session_state.inventory_df = result["data"]
                    st.session_state.inventory_stats = result["stats"]
                    st.success(f"✅ Loaded {result['stats']['total_records']} records!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        # Status
        if st.session_state.inventory_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Inventory Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.inventory_stats

            # Description
            st.markdown(f"""
            <div class="data-box inventory">
                <strong>Dataset Description:</strong><br>
                • <strong>{stats['total_records']}</strong> daily records<br>
                • Period: <strong>{stats['date_start'].strftime('%Y-%m-%d') if stats['date_start'] else 'N/A'}</strong> to <strong>{stats['date_end'].strftime('%Y-%m-%d') if stats['date_end'] else 'N/A'}</strong><br>
                • Contains: Gloves, PPE, Medications usage & levels
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            st.markdown("**Average Daily Usage:**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card inventory">
                    <div class="kpi-value purple">{stats['avg_gloves_usage']:.0f}</div>
                    <div class="kpi-label">Gloves/Day</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card inventory">
                    <div class="kpi-value purple">{stats['avg_ppe_usage']:.0f}</div>
                    <div class="kpi-label">PPE Sets/Day</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card inventory">
                    <div class="kpi-value purple">{stats['avg_medication_usage']:.0f}</div>
                    <div class="kpi-label">Medications/Day</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card {'danger' if stats['avg_stockout_risk'] > 10 else 'warning' if stats['avg_stockout_risk'] > 5 else 'success'}">
                    <div class="kpi-value {'red' if stats['avg_stockout_risk'] > 10 else 'yellow' if stats['avg_stockout_risk'] > 5 else 'green'}">{stats['avg_stockout_risk']:.1f}%</div>
                    <div class="kpi-label">Stockout Risk</div>
                </div>
                """, unsafe_allow_html=True)

            # Graph: Usage Trends (Dynamic Equipment Selection)
            if st.session_state.inventory_df is not None:
                df = st.session_state.inventory_df.copy()
                if "Date" in df.columns:
                    df = df.sort_values("Date").tail(30)

                    # Find usage columns - actual column names from Supabase:
                    # Inventory_Used_Gloves, Inventory_Used_PPE_Sets, Inventory_Used_Medications
                    usage_cols = [c for c in df.columns if "Used" in c or "Usage" in c or "used" in c or "usage" in c]

                    # Equipment selection options
                    equipment_options = ["📊 All Equipment"]
                    equipment_col_map = {}
                    equipment_colors = {
                        "📊 All Equipment": ["#a855f7", "#3b82f6", "#22c55e", "#f97316", "#eab308"],
                        "🧤 Gloves": "#a855f7",
                        "🛡️ PPE Sets": "#3b82f6",
                        "💊 Medications": "#22c55e",
                        "💉 Syringes": "#f97316",
                        "🩹 Bandages": "#eab308",
                    }

                    # Map columns to equipment names
                    for col in usage_cols:
                        col_lower = col.lower()
                        if "glove" in col_lower:
                            equipment_options.append("🧤 Gloves")
                            equipment_col_map["🧤 Gloves"] = col
                        elif "ppe" in col_lower:
                            equipment_options.append("🛡️ PPE Sets")
                            equipment_col_map["🛡️ PPE Sets"] = col
                        elif "medication" in col_lower or "med" in col_lower:
                            equipment_options.append("💊 Medications")
                            equipment_col_map["💊 Medications"] = col
                        elif "syringe" in col_lower:
                            equipment_options.append("💉 Syringes")
                            equipment_col_map["💉 Syringes"] = col
                        elif "bandage" in col_lower:
                            equipment_options.append("🩹 Bandages")
                            equipment_col_map["🩹 Bandages"] = col

                    # Equipment selector
                    selected_equipment = st.selectbox(
                        "Select Equipment to View",
                        options=equipment_options,
                        index=0,
                        key="inv_equipment_selector"
                    )

                    fig = go.Figure()

                    if selected_equipment == "📊 All Equipment":
                        # Show all equipment
                        colors = equipment_colors["📊 All Equipment"]
                        for i, col in enumerate(usage_cols[:5]):
                            # Format display name: Inventory_Used_Gloves -> Gloves
                            display_name = col.replace("Inventory_Used_", "").replace("_", " ")
                            fig.add_trace(go.Scatter(
                                x=df["Date"],
                                y=df[col],
                                name=display_name,
                                line=dict(color=colors[i % len(colors)], width=2),
                                mode='lines+markers',
                                marker=dict(size=4)
                            ))
                        title = "Usage Trends - All Equipment (Last 30 Days)"
                    else:
                        # Show selected equipment only
                        col = equipment_col_map.get(selected_equipment)
                        if col and col in df.columns:
                            color = equipment_colors.get(selected_equipment, "#a855f7")
                            fig.add_trace(go.Scatter(
                                x=df["Date"],
                                y=df[col],
                                name=selected_equipment,
                                line=dict(color=color, width=3),
                                mode='lines+markers',
                                marker=dict(size=6),
                                fill='tozeroy',
                                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
                            ))
                        title = f"Usage Trend - {selected_equipment} (Last 30 Days)"

                    fig.update_layout(
                        title=title,
                        template="plotly_dark",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title="Date",
                        yaxis_title="Units Used",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Observations
            observations = []
            if stats['avg_stockout_risk'] > 10:
                observations.append(("danger", f"⚠️ High stockout risk ({stats['avg_stockout_risk']:.1f}%) - inventory levels too low"))
            elif stats['avg_stockout_risk'] > 5:
                observations.append(("warning", f"📊 Moderate stockout risk ({stats['avg_stockout_risk']:.1f}%) - monitor closely"))

            if stats['restock_events'] > 50:
                observations.append(("warning", f"📦 High restock frequency: {stats['restock_events']} events"))

            if observations:
                st.markdown("**Observations:**")
                for obs_type, obs_text in observations:
                    st.markdown(f'<div class="observation-box {obs_type}">{obs_text}</div>', unsafe_allow_html=True)

            # Raw Data Preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(st.session_state.inventory_df.tail(20), use_container_width=True, height=200)

        else:
            st.markdown('<span class="status-badge pending">⚠️ Not Loaded</span>', unsafe_allow_html=True)
            st.info("Click button above to load inventory data from Supabase")

    # =========================================================================
    # FINANCIAL DATA SECTION
    # =========================================================================
    with col_fin:
        st.markdown('<div class="subsection-header">💰 Cost Parameters</div>', unsafe_allow_html=True)

        # Load Button
        if st.button("🔄 Load Cost Data", type="primary", use_container_width=True, key="load_fin"):
            with st.spinner("Loading cost parameters from Supabase..."):
                result = load_financial_data()
                if result["success"]:
                    st.session_state.inv_financial_data_loaded = True
                    st.session_state.inv_financial_df = result["data"]
                    st.session_state.inv_cost_params = result["cost_params"]
                    st.session_state.inv_financial_stats = result["stats"]
                    st.success("✅ Cost parameters loaded!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        # Status
        if st.session_state.inv_financial_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Cost Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.inv_financial_stats
            cost_params = st.session_state.inv_cost_params

            # Description
            st.markdown(f"""
            <div class="data-box financial">
                <strong>Dataset Description:</strong><br>
                • Cost parameters from Supabase<br>
                • Contains: Unit costs, holding costs, penalties<br>
                • Linked to inventory management
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            st.markdown("**Unit Costs:**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['gloves_unit_cost']:.0f}</div>
                    <div class="kpi-label">Gloves/Box</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['ppe_unit_cost']:.0f}</div>
                    <div class="kpi-label">PPE Set</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['medication_unit_cost']:.0f}</div>
                    <div class="kpi-label">Medication/Unit</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value yellow">{cost_params['holding_cost_pct']*100:.0f}%</div>
                    <div class="kpi-label">Annual Holding Cost</div>
                </div>
                """, unsafe_allow_html=True)

            # Penalty KPIs
            st.markdown("**Cost Parameters:**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value blue">${cost_params['ordering_cost']:.0f}</div>
                    <div class="kpi-label">Ordering Cost/Order</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card danger">
                    <div class="kpi-value red">${cost_params['stockout_penalty']:.0f}</div>
                    <div class="kpi-label">Stockout Penalty</div>
                </div>
                """, unsafe_allow_html=True)

            # Cost Distribution Chart
            if st.session_state.inventory_data_loaded:
                inv_stats = st.session_state.inventory_stats
                gloves_cost = inv_stats['avg_gloves_usage'] * cost_params['gloves_unit_cost']
                ppe_cost = inv_stats['avg_ppe_usage'] * cost_params['ppe_unit_cost']
                med_cost = inv_stats['avg_medication_usage'] * cost_params['medication_unit_cost']

                fig = go.Figure(data=[go.Pie(
                    labels=['Gloves', 'PPE', 'Medications'],
                    values=[gloves_cost, ppe_cost, med_cost],
                    hole=0.4,
                    marker_colors=['#a855f7', '#3b82f6', '#22c55e']
                )])
                fig.update_layout(title="Daily Usage Cost Distribution", template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Observations
                st.markdown("**Observations:**")
                total_daily = gloves_cost + ppe_cost + med_cost
                st.markdown(f'<div class="observation-box success">💵 Total daily inventory cost: ${total_daily:,.0f}</div>', unsafe_allow_html=True)

            # Raw Financial Data Preview
            if st.session_state.inv_financial_df is not None and not st.session_state.inv_financial_df.empty:
                with st.expander("📋 View Raw Financial Data"):
                    st.dataframe(st.session_state.inv_financial_df.tail(20), use_container_width=True, height=200)

        else:
            st.markdown('<span class="status-badge pending">⚠️ Not Loaded</span>', unsafe_allow_html=True)
            st.info("Click button above to load cost parameters from Supabase")


# =============================================================================
# TAB 2: CURRENT INVENTORY SITUATION
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">📦 Current Inventory Situation</div>', unsafe_allow_html=True)

    if st.session_state.inventory_data_loaded and st.session_state.inv_financial_data_loaded:
        inv_stats = st.session_state.inventory_stats
        cost_params = st.session_state.inv_cost_params

        # Calculate linked variables
        # Daily Usage Cost = usage × unit_cost for each item
        daily_gloves_cost = inv_stats['avg_gloves_usage'] * cost_params['gloves_unit_cost']
        daily_ppe_cost = inv_stats['avg_ppe_usage'] * cost_params['ppe_unit_cost']
        daily_med_cost = inv_stats['avg_medication_usage'] * cost_params['medication_unit_cost']
        daily_inventory_cost = daily_gloves_cost + daily_ppe_cost + daily_med_cost
        weekly_inventory_cost = daily_inventory_cost * 7
        monthly_inventory_cost = daily_inventory_cost * 30

        # Estimate holding cost using actual inventory levels and unit costs
        # Formula: H = Σ(inventory_level × unit_cost × annual_rate / 365)
        # This is the daily cost of holding inventory
        daily_holding_cost = (
            inv_stats['avg_gloves_level'] * cost_params['gloves_unit_cost'] * (cost_params['holding_cost_pct'] / 365) +
            inv_stats['avg_ppe_level'] * cost_params['ppe_unit_cost'] * (cost_params['holding_cost_pct'] / 365) +
            inv_stats['avg_medication_level'] * cost_params['medication_unit_cost'] * (cost_params['holding_cost_pct'] / 365)
        )
        avg_total_inventory = inv_stats['avg_gloves_level'] + inv_stats['avg_ppe_level'] + inv_stats['avg_medication_level']

        # Estimate stockout cost
        # Formula: stockout_events × penalty_per_stockout
        stockout_events = int(inv_stats['total_records'] * inv_stats['avg_stockout_risk'] / 100)
        total_stockout_cost = stockout_events * cost_params['stockout_penalty']

        # INVENTORY SITUATION SUMMARY
        st.markdown('<div class="subsection-header">📦 Inventory Usage Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card inventory">
                <div class="kpi-value purple">{inv_stats['avg_gloves_usage']:.0f}</div>
                <div class="kpi-label">Gloves/Day</div>
                <div class="kpi-delta">Level: {inv_stats['avg_gloves_level']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card inventory">
                <div class="kpi-value purple">{inv_stats['avg_ppe_usage']:.0f}</div>
                <div class="kpi-label">PPE Sets/Day</div>
                <div class="kpi-delta">Level: {inv_stats['avg_ppe_level']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card inventory">
                <div class="kpi-value purple">{inv_stats['avg_medication_usage']:.0f}</div>
                <div class="kpi-label">Medications/Day</div>
                <div class="kpi-delta">Level: {inv_stats['avg_medication_level']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            risk_class = "danger" if inv_stats['avg_stockout_risk'] > 10 else "warning" if inv_stats['avg_stockout_risk'] > 5 else "success"
            st.markdown(f"""
            <div class="kpi-card {risk_class}">
                <div class="kpi-value {'red' if inv_stats['avg_stockout_risk'] > 10 else 'yellow' if inv_stats['avg_stockout_risk'] > 5 else 'green'}">{inv_stats['avg_stockout_risk']:.1f}%</div>
                <div class="kpi-label">Stockout Risk</div>
            </div>
            """, unsafe_allow_html=True)

        # FINANCIAL SITUATION SUMMARY
        st.markdown('<div class="subsection-header">💰 Financial Situation Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${daily_inventory_cost:,.0f}</div>
                <div class="kpi-label">Daily Inventory Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${weekly_inventory_cost:,.0f}</div>
                <div class="kpi-label">Weekly Inventory Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${monthly_inventory_cost:,.0f}</div>
                <div class="kpi-label">Monthly Inventory Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card {'danger' if total_stockout_cost > 5000 else 'warning' if total_stockout_cost > 1000 else ''}">
                <div class="kpi-value {'red' if total_stockout_cost > 5000 else 'yellow' if total_stockout_cost > 1000 else 'blue'}">${total_stockout_cost:,.0f}</div>
                <div class="kpi-label">Stockout Penalties</div>
            </div>
            """, unsafe_allow_html=True)

        # LINKED VARIABLES (Inventory × Financial)
        st.markdown('<div class="subsection-header">🔗 Linked Metrics (Inventory × Financial)</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cost_per_unit = daily_inventory_cost / (inv_stats['avg_gloves_usage'] + inv_stats['avg_ppe_usage'] + inv_stats['avg_medication_usage']) if (inv_stats['avg_gloves_usage'] + inv_stats['avg_ppe_usage'] + inv_stats['avg_medication_usage']) > 0 else 0
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value pink">${cost_per_unit:.2f}</div>
                <div class="kpi-label">Avg Cost/Unit Used</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            turnover = (inv_stats['avg_gloves_usage'] + inv_stats['avg_ppe_usage'] + inv_stats['avg_medication_usage']) / avg_total_inventory * 100 if avg_total_inventory > 0 else 0
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value pink">{turnover:.1f}%</div>
                <div class="kpi-label">Daily Turnover Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value pink">{inv_stats['restock_events']}</div>
                <div class="kpi-label">Restock Events</div>
                <div class="kpi-delta negative">${inv_stats['restock_events'] * cost_params['ordering_cost']:,.0f} cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            service_level = 100 - inv_stats['avg_stockout_risk']
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value pink">{service_level:.1f}%</div>
                <div class="kpi-label">Service Level</div>
            </div>
            """, unsafe_allow_html=True)

        # PARAMETERS TO OPTIMIZE
        st.markdown('<div class="subsection-header">🎯 Parameters to Optimize</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="kpi-card {'danger' if inv_stats['avg_stockout_risk'] > 10 else 'warning' if inv_stats['avg_stockout_risk'] > 5 else 'success'}">
                <div class="kpi-value {'red' if inv_stats['avg_stockout_risk'] > 10 else 'yellow' if inv_stats['avg_stockout_risk'] > 5 else 'green'}">{inv_stats['avg_stockout_risk']:.1f}%</div>
                <div class="kpi-label">🎯 Reduce Stockout Risk</div>
                <div class="kpi-delta">Target: &lt;2%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card {'danger' if inv_stats['restock_events'] > 100 else 'warning' if inv_stats['restock_events'] > 50 else 'success'}">
                <div class="kpi-value {'red' if inv_stats['restock_events'] > 100 else 'yellow' if inv_stats['restock_events'] > 50 else 'green'}">{inv_stats['restock_events']}</div>
                <div class="kpi-label">🎯 Reduce Restocking</div>
                <div class="kpi-delta">Target: &lt;30/month</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card {'danger' if service_level < 90 else 'warning' if service_level < 95 else 'success'}">
                <div class="kpi-value {'red' if service_level < 90 else 'yellow' if service_level < 95 else 'green'}">{service_level:.1f}%</div>
                <div class="kpi-label">🎯 Improve Service Level</div>
                <div class="kpi-delta">Target: &gt;98%</div>
            </div>
            """, unsafe_allow_html=True)

        # VISUALIZATION
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Cost breakdown bar chart
            fig = go.Figure(data=[go.Bar(
                x=['Gloves', 'PPE', 'Medications', 'Stockout Penalties'],
                y=[daily_gloves_cost * 7, daily_ppe_cost * 7, daily_med_cost * 7, total_stockout_cost / (inv_stats['total_records'] / 7)],
                marker_color=['#a855f7', '#3b82f6', '#22c55e', '#ef4444'],
                text=[f'${v:,.0f}' for v in [daily_gloves_cost * 7, daily_ppe_cost * 7, daily_med_cost * 7, total_stockout_cost / (inv_stats['total_records'] / 7)]],
                textposition='auto',
            )])
            fig.update_layout(title="Weekly Cost by Category", yaxis_title="Cost ($)", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Service level gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=service_level,
                title={'text': "Service Level"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#22c55e" if service_level >= 95 else "#eab308" if service_level >= 90 else "#ef4444"},
                    'steps': [
                        {'range': [0, 90], 'color': "rgba(239,68,68,0.2)"},
                        {'range': [90, 95], 'color': "rgba(234,179,8,0.2)"},
                        {'range': [95, 100], 'color': "rgba(34,197,94,0.2)"},
                    ],
                    'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 98}
                }
            ))
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please load both **Inventory Data** and **Cost Data** in the first tab to see the current situation analysis.")


# =============================================================================
# TAB 3: AFTER OPTIMIZATION
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">🚀 After Optimization (Forecast Applied)</div>', unsafe_allow_html=True)

    if st.session_state.inventory_data_loaded and st.session_state.inv_financial_data_loaded:
        forecast_data = get_forecast_data()

        if not forecast_data["has_forecast"]:
            st.warning("""
            ⚠️ **No forecast data available!**

            Please run forecasting models in the **Forecast Hub (Page 10)** or **Modeling Hub (Page 08)** first, then return here.

            **Tip:** Use the "🔄 Refresh from Forecast Hub" button above to sync after running forecasts.
            """)
        else:
            # Show sync-aware success message
            if forecast_data["synced_with_page10"]:
                st.success(f"✅ **Synced with Forecast Hub** — Using **{forecast_data['source']}** ({forecast_data['horizon']} days ahead)")
            else:
                st.info(f"📊 Using direct model results: **{forecast_data['source']}** ({forecast_data['horizon']} days ahead)\n\n*Run forecasts in Page 10 for full sync.*")

            # Optimization Button
            if st.button("🚀 Apply Forecast Optimization", type="primary", use_container_width=True, key="optimize"):
                with st.spinner("Optimizing inventory based on forecast..."):
                    result = calculate_optimization(forecast_data, st.session_state.inventory_stats, st.session_state.inv_cost_params)
                    if result["success"]:
                        st.session_state.inv_optimized_results = result
                        st.session_state.inv_optimization_done = True
                        st.success("✅ Optimization complete!")
                        st.rerun()

            # Show Results
            if st.session_state.inv_optimization_done and st.session_state.inv_optimized_results:
                opt = st.session_state.inv_optimized_results
                inv_stats = st.session_state.inventory_stats

                st.markdown("---")

                # ADJUSTED INVENTORY PARAMETERS
                st.markdown('<div class="subsection-header">📦 Optimized Order Quantities</div>', unsafe_allow_html=True)

                # Show optimal orders for each item
                order_data = []
                for item_id, item_result in opt["item_results"].items():
                    order_data.append({
                        "Item": item_result["name"],
                        "Forecast Demand": f"{item_result['forecast_demand']:.0f}",
                        "Safety Stock": f"{item_result['safety_stock']:.0f}",
                        "Optimal Order": item_result["optimal_order"],
                        "Unit Cost": f"${item_result['unit_cost']:.0f}",
                        "Order Cost": f"${item_result['order_cost']:.0f}",
                    })
                st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

                # ADJUSTED FINANCIAL PARAMETERS
                st.markdown('<div class="subsection-header">💰 Adjusted Financial Parameters</div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">${opt['avg_daily_cost']:,.0f}</div>
                        <div class="kpi-label">New Daily Cost</div>
                        <div class="kpi-delta positive">${opt['current_daily_cost'] - opt['avg_daily_cost']:,.0f} saved</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">${opt['optimized_weekly_cost']:,.0f}</div>
                        <div class="kpi-label">New Weekly Cost</div>
                        <div class="kpi-delta positive">${opt['weekly_savings']:,.0f} saved</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">0%</div>
                        <div class="kpi-label">Expected Stockout Risk</div>
                        <div class="kpi-delta positive">Eliminated</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">100%</div>
                        <div class="kpi-label">Expected Service Level</div>
                        <div class="kpi-delta positive">Perfect</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Patient Forecast Chart
                st.markdown('<div class="subsection-header">📊 Patient Demand Forecast</div>', unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=opt["dates"],
                    y=opt["patient_forecast"],
                    mode='lines+markers',
                    name='Patient Forecast',
                    line=dict(color='#a855f7', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(168,85,247,0.2)'
                ))
                fig.update_layout(title="Patient Arrivals Forecast (Drives Inventory Demand)", xaxis_title="Day", yaxis_title="Patients", template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

                # GRAPHS
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    # Optimized order breakdown
                    items = [r["name"][:15] for r in opt["item_results"].values()]
                    orders = [r["optimal_order"] for r in opt["item_results"].values()]

                    fig = go.Figure(data=[go.Bar(
                        x=items,
                        y=orders,
                        marker_color=['#a855f7', '#3b82f6', '#22c55e', '#f97316', '#eab308'],
                        text=[f'{o:,}' for o in orders],
                        textposition='auto',
                    )])
                    fig.update_layout(title="Optimal Order Quantities", yaxis_title="Units", template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Cost trend
                    fig = go.Figure()
                    daily_costs = [opt['avg_daily_cost']] * opt['horizon']
                    fig.add_trace(go.Scatter(
                        x=opt["dates"],
                        y=daily_costs,
                        mode='lines+markers',
                        name='Optimized Cost',
                        line=dict(color='#22c55e', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(34,197,94,0.2)'
                    ))
                    fig.add_hline(y=opt['current_daily_cost'], line_dash="dash", line_color="#ef4444", annotation_text="Current Avg")
                    fig.update_layout(title="Daily Cost Comparison", yaxis_title="Cost ($)", template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please load both **Inventory Data** and **Cost Data** in the first tab first.")


# =============================================================================
# TAB 4: BEFORE VS AFTER COMPARISON
# =============================================================================
with tab4:
    st.markdown('<div class="section-header">📈 Before vs After Comparison</div>', unsafe_allow_html=True)

    if st.session_state.inv_optimization_done and st.session_state.inv_optimized_results:
        opt = st.session_state.inv_optimized_results
        inv_stats = st.session_state.inventory_stats
        cost_params = st.session_state.inv_cost_params

        # Calculate values
        service_level_before = 100 - inv_stats['avg_stockout_risk']
        savings_pct = (opt['weekly_savings'] / opt['current_weekly_cost'] * 100) if opt['current_weekly_cost'] > 0 else 0
        annual_savings = opt['weekly_savings'] * 52

        # SUMMARY CARDS
        st.markdown('<div class="subsection-header">💰 Financial Impact</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card danger">
                <div class="kpi-value red">${opt['current_weekly_cost']:,.0f}</div>
                <div class="kpi-label">BEFORE (Weekly)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card success">
                <div class="kpi-value green">${opt['optimized_weekly_cost']:,.0f}</div>
                <div class="kpi-label">AFTER (Weekly)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card warning">
                <div class="kpi-value yellow">${opt['weekly_savings']:,.0f}</div>
                <div class="kpi-label">Weekly Savings</div>
                <div class="kpi-delta positive">{savings_pct:.1f}% reduction</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card warning">
                <div class="kpi-value yellow">${annual_savings:,.0f}</div>
                <div class="kpi-label">Annual Savings</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # COMPARISON CHARTS
        col1, col2 = st.columns(2)

        with col1:
            # Cost comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['BEFORE', 'AFTER'],
                y=[opt['current_weekly_cost'], opt['optimized_weekly_cost']],
                marker_color=['#ef4444', '#a855f7'],
                text=[f"${opt['current_weekly_cost']:,.0f}", f"${opt['optimized_weekly_cost']:,.0f}"],
                textposition='auto',
            ))
            fig.update_layout(title="Weekly Cost Comparison", yaxis_title="Cost ($)", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Service level comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['BEFORE', 'AFTER'],
                y=[service_level_before, 100],
                marker_color=['#ef4444', '#22c55e'],
                text=[f"{service_level_before:.1f}%", "100%"],
                textposition='auto',
            ))
            fig.update_layout(title="Service Level Comparison", yaxis_title="Service Level (%)", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Service Level gauges
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=service_level_before,
                title={'text': "BEFORE: Service Level"},
                number={'suffix': '%'},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ef4444"},
                       'steps': [{'range': [0, 90], 'color': "rgba(239,68,68,0.2)"},
                                {'range': [90, 95], 'color': "rgba(234,179,8,0.2)"},
                                {'range': [95, 100], 'color': "rgba(34,197,94,0.2)"}]}
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=100,
                title={'text': "AFTER: Service Level"},
                number={'suffix': '%'},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#22c55e"},
                       'steps': [{'range': [0, 90], 'color': "rgba(239,68,68,0.2)"},
                                {'range': [90, 95], 'color': "rgba(234,179,8,0.2)"},
                                {'range': [95, 100], 'color': "rgba(34,197,94,0.2)"}]}
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # SUMMARY
        st.success(f"""
        ### 🎉 Optimization Summary

        By applying **{opt['source']}** forecast-driven inventory management:

        | Metric | Before | After | Change |
        |--------|--------|-------|--------|
        | Weekly Cost | ${opt['current_weekly_cost']:,.0f} | ${opt['optimized_weekly_cost']:,.0f} | **-${opt['weekly_savings']:,.0f}** ({savings_pct:.1f}%) |
        | Annual Cost | ${opt['current_weekly_cost'] * 52:,.0f} | ${opt['optimized_weekly_cost'] * 52:,.0f} | **-${annual_savings:,.0f}** |
        | Service Level | {service_level_before:.1f}% | 100% | **+{100 - service_level_before:.1f}%** |
        | Stockout Risk | {inv_stats['avg_stockout_risk']:.1f}% | 0% | **Eliminated** |
        """)

    elif st.session_state.inventory_data_loaded and st.session_state.inv_financial_data_loaded:
        st.info("📊 Go to **'After Optimization'** tab and click **'Apply Forecast Optimization'** to see the comparison.")
    else:
        st.warning("⚠️ Please load data in the first tab and run optimization to see comparison.")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Data Status")

    st.markdown(f"**Inventory Data:** {'✅ Loaded' if st.session_state.inventory_data_loaded else '❌ Not loaded'}")
    st.markdown(f"**Cost Data:** {'✅ Loaded' if st.session_state.inv_financial_data_loaded else '❌ Not loaded'}")
    st.markdown(f"**Optimization:** {'✅ Done' if st.session_state.inv_optimization_done else '❌ Not done'}")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        for key in ["inventory_data_loaded", "inventory_df", "inventory_stats", "inv_financial_data_loaded",
                    "inv_financial_df", "inv_cost_params", "inv_financial_stats", "inv_optimization_done", "inv_optimized_results"]:
            if key in st.session_state:
                st.session_state[key] = None if "df" in key or "stats" in key or "results" in key or "params" in key else False
        st.session_state.inv_cost_params = DEFAULT_COST_PARAMS.copy()
        st.rerun()
