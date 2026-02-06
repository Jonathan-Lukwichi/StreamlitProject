# =============================================================================
# 12_Supply_Planner.py
# ENHANCED: 5-Tab Structure with EOQ, Statistical Safety Stock & MILP
# =============================================================================
"""
Tab Structure:
1. Data Upload & Preview - Load Inventory + Cost data with KPIs & graphs
2. Current Inventory Situation - Summarize usage metrics
3. After Optimization - EOQ/MILP optimization with statistical safety stock
4. Before vs After - Compare results with AI insights
5. Constraint Configuration - Full inventory constraint editor

Key Enhancements:
- Statistical Safety Stock: SS = Z × σ × √L (replaces fixed 10%)
- Economic Order Quantity: EOQ = √((2 × D × S) / H)
- MILP inventory optimization integration
- Configurable service levels and constraints
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
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header

# Import Supabase services
from app_core.data.inventory_service import InventoryService, get_inventory_service

# Import AI Insight components
from app_core.ui.insight_components import (
    inject_insight_styles,
    render_supply_optimization_insights
)
from app_core.ai.recommendation_engine import SupplyRecommendationEngine

# Import optimization modules
from app_core.optimization.constraint_config import (
    InventoryConstraintConfig,
    DEFAULT_INVENTORY_CONSTRAINTS,
    SERVICE_LEVEL_Z_SCORES,
    get_z_score,
)
from app_core.optimization.scenario_manager import ScenarioManager
from app_core.optimization.inventory_optimizer import (
    calculate_eoq,
    solve_milp_inventory,
    convert_patient_forecast_to_inventory_demand,
    generate_reorder_alerts,
    calculate_rpiw,
    get_optimization_recommendation,
    DEFAULT_HEALTHCARE_ITEMS,
    HealthcareInventoryItem,
    ItemCriticality,
    EOQResult,
    MILPResult,
)

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
DEFAULT_INVENTORY_CONFIG = {
    "GLOVES": {"name": "Medical Gloves (boxes)", "usage_per_patient": 2.0, "unit_cost": 15.0, "lead_time": 2},
    "PPE_SETS": {"name": "PPE Sets", "usage_per_patient": 0.5, "unit_cost": 45.0, "lead_time": 5},
    "MEDICATIONS": {"name": "Emergency Medications", "usage_per_patient": 0.8, "unit_cost": 85.0, "lead_time": 3},
    "SYRINGES": {"name": "Syringes (boxes)", "usage_per_patient": 1.5, "unit_cost": 12.0, "lead_time": 2},
    "BANDAGES": {"name": "Bandages & Dressings", "usage_per_patient": 1.0, "unit_cost": 18.0, "lead_time": 2},
    "IV_FLUIDS": {"name": "IV Fluids (units)", "usage_per_patient": 0.6, "unit_cost": 8.0, "lead_time": 3},
}

DEFAULT_COST_PARAMS = {
    "holding_cost_pct": 0.20,
    "ordering_cost": 50.0,
    "stockout_penalty": 100.0,
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
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 15%; right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}}

.orb-2 {{
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.2), transparent 70%);
    bottom: 20%; left: 15%;
    animation: float-orb 30s ease-in-out infinite;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(59, 130, 246, 0.2);
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
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(168, 85, 247, 0.1));
    border-color: rgba(59, 130, 246, 0.3);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(168, 85, 247, 0.2)) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
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

.kpi-card.inventory {{ border-color: rgba(59, 130, 246, 0.5); }}
.kpi-card.cost {{ border-color: rgba(168, 85, 247, 0.5); }}
.kpi-card.warning {{ border-color: rgba(234, 179, 8, 0.5); background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.danger {{ border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.success {{ border-color: rgba(34, 197, 94, 0.5); background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.constraint {{ border-color: rgba(6, 182, 212, 0.5); background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(15, 23, 42, 0.98)); }}

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
.kpi-value.cyan {{ color: #06b6d4; }}

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
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
}}

.subsection-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1rem 0 0.75rem 0;
}}

/* Data Box */
.data-box {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}}

.data-box.inventory {{ border-color: rgba(59, 130, 246, 0.3); }}
.data-box.cost {{ border-color: rgba(168, 85, 247, 0.3); }}
.data-box.constraint {{ border-color: rgba(6, 182, 212, 0.3); }}

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
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22c55e;
}}

.status-badge.pending {{
    background: rgba(234, 179, 8, 0.2);
    border: 1px solid rgba(234, 179, 8, 0.4);
    color: #eab308;
}}

/* Constraint Section */
.constraint-section {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.05), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}}

.constraint-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #06b6d4;
    margin-bottom: 1rem;
}}

/* Formula Box */
.formula-box {{
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    margin: 0.5rem 0;
}}

/* Alert Card */
.alert-card {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
}}

.alert-card.warning {{
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.95));
    border-color: rgba(234, 179, 8, 0.4);
}}

.alert-card.critical {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.95));
    border-color: rgba(239, 68, 68, 0.6);
}}
</style>

<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HERO HEADER
# =============================================================================
render_scifi_hero_header(
    title="Supply Planner",
    subtitle="EOQ optimization with statistical safety stock (SS = Z × σ × √L)",
    status="EOQ + MILP ACTIVE"
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
if "inv_cost_params" not in st.session_state:
    st.session_state.inv_cost_params = DEFAULT_COST_PARAMS.copy()

if "inv_optimization_done" not in st.session_state:
    st.session_state.inv_optimization_done = False
if "inv_optimized_results" not in st.session_state:
    st.session_state.inv_optimized_results = None

# Constraint configuration
if "inventory_constraint_config" not in st.session_state:
    st.session_state.inventory_constraint_config = DEFAULT_INVENTORY_CONSTRAINTS

# Scenario management
if "inventory_scenarios" not in st.session_state:
    st.session_state.inventory_scenarios = ScenarioManager(scenario_type="inventory")

# AI Insight Mode
if "supply_insight_mode" not in st.session_state:
    st.session_state.supply_insight_mode = "rule_based"
if "supply_llm_api_key" not in st.session_state:
    st.session_state.supply_llm_api_key = None
if "supply_api_provider" not in st.session_state:
    st.session_state.supply_api_provider = "anthropic"


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

        df = service.fetch_inventory_data()
        if df.empty:
            result["error"] = "No inventory data found"
            return result

        result["success"] = True
        result["data"] = df

        # Calculate statistics
        result["stats"] = {
            "total_records": len(df),
            "date_start": df["Date"].min() if "Date" in df.columns else None,
            "date_end": df["Date"].max() if "Date" in df.columns else None,
            "avg_gloves_daily": df["Gloves_Used"].mean() if "Gloves_Used" in df.columns else 0,
            "avg_ppe_daily": df["PPE_Sets_Used"].mean() if "PPE_Sets_Used" in df.columns else 0,
            "avg_meds_daily": df["Medications_Used"].mean() if "Medications_Used" in df.columns else 0,
            "std_gloves": df["Gloves_Used"].std() if "Gloves_Used" in df.columns else 0,
            "std_ppe": df["PPE_Sets_Used"].std() if "PPE_Sets_Used" in df.columns else 0,
            "std_meds": df["Medications_Used"].std() if "Medications_Used" in df.columns else 0,
            "stockout_events": df["Stockout_Event"].sum() if "Stockout_Event" in df.columns else 0,
            "avg_service_level": df["Service_Level"].mean() * 100 if "Service_Level" in df.columns else 95,
        }

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def get_forecast_data() -> Dict[str, Any]:
    """Get forecast data from session state (synced with Page 10)."""
    result = {
        "has_forecast": False,
        "source": None,
        "forecasts": [],
        "dates": [],
        "horizon": 0,
        "accuracy": None,
        "synced_with_page10": False,
    }

    # Priority 1: forecast_hub_demand
    if "forecast_hub_demand" in st.session_state and st.session_state.forecast_hub_demand:
        hub_data = st.session_state.forecast_hub_demand
        if isinstance(hub_data, dict) and "forecasts" in hub_data:
            result["has_forecast"] = True
            result["source"] = hub_data.get("model_name", "Forecast Hub")
            result["forecasts"] = hub_data.get("forecasts", [])
            result["horizon"] = len(result["forecasts"])
            result["accuracy"] = hub_data.get("accuracy")
            result["synced_with_page10"] = True
            return result

    # Priority 2: active_forecast
    if "active_forecast" in st.session_state and st.session_state.active_forecast:
        af = st.session_state.active_forecast
        if isinstance(af, dict) and "forecasts" in af:
            result["has_forecast"] = True
            result["source"] = af.get("model_name", "Active Forecast")
            result["forecasts"] = af.get("forecasts", [])
            result["horizon"] = len(result["forecasts"])
            result["accuracy"] = af.get("accuracy")
            result["synced_with_page10"] = True
            return result

    # Priority 3: ML model results
    for model_key in ["ml_mh_results_XGBoost", "ml_mh_results_LSTM", "ml_mh_results_ANN"]:
        if model_key in st.session_state and st.session_state[model_key]:
            ml_results = st.session_state[model_key]
            if isinstance(ml_results, dict):
                best_h = ml_results.get("best_horizon", 1)
                per_h = ml_results.get("per_h", {})
                if best_h in per_h:
                    h_data = per_h[best_h]
                    if "forecast" in h_data:
                        result["has_forecast"] = True
                        result["source"] = f"{model_key.split('_')[-1]} (H={best_h})"
                        result["forecasts"] = list(h_data["forecast"])
                        result["horizon"] = len(result["forecasts"])
                        return result

    # Generate dates
    today = datetime.now().date()
    if result["has_forecast"]:
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, result["horizon"] + 1)]

    return result


def calculate_statistical_safety_stock(
    demand_std: float,
    lead_time: int,
    service_level: float = 0.95
) -> float:
    """
    Calculate safety stock using statistical formula.

    SS = Z × σ × √L

    Where:
        Z = z-score for target service level
        σ = standard deviation of demand
        L = lead time in days

    Args:
        demand_std: Standard deviation of daily demand
        lead_time: Lead time in days
        service_level: Target service level (0.0 to 1.0)

    Returns:
        Safety stock quantity
    """
    z = get_z_score(service_level)
    return z * demand_std * np.sqrt(lead_time)


def calculate_eoq_formula(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_rate: float,
    unit_cost: float
) -> float:
    """
    Calculate Economic Order Quantity.

    EOQ = √((2 × D × S) / H)

    Where:
        D = annual demand
        S = ordering cost per order
        H = annual holding cost per unit = unit_cost × holding_cost_rate

    Args:
        annual_demand: Annual demand in units
        ordering_cost: Fixed cost per order ($)
        holding_cost_rate: Annual holding cost as fraction of unit cost
        unit_cost: Cost per unit ($)

    Returns:
        Economic order quantity
    """
    h = unit_cost * holding_cost_rate
    if h <= 0 or annual_demand <= 0:
        return 0.0
    return np.sqrt((2 * ordering_cost * annual_demand) / h)


def calculate_inventory_optimization(
    forecast_data: Dict,
    inventory_stats: Dict,
    cost_params: Dict,
    constraints: InventoryConstraintConfig,
) -> Dict[str, Any]:
    """
    Calculate optimized inventory using EOQ and statistical safety stock.

    Enhancements over old version:
    1. Statistical safety stock: SS = Z × σ × √L
    2. EOQ calculation: √((2 × D × S) / H)
    3. Configurable service levels
    """
    if not forecast_data["has_forecast"]:
        return {"success": False, "error": "No forecast available"}

    patient_forecast = forecast_data["forecasts"]
    horizon = len(patient_forecast)

    # Get cost parameters
    holding_cost_rate = constraints.holding_cost_rate
    ordering_cost = cost_params.get("ordering_cost", 50.0)
    service_level = constraints.target_service_level

    item_results = {}
    total_ordering_cost = 0
    total_holding_cost = 0
    total_purchase_cost = 0

    for item_id, item_config in DEFAULT_INVENTORY_CONFIG.items():
        # Calculate demand based on patient forecast
        usage_rate = item_config["usage_per_patient"]
        daily_demands = [max(0, p) * usage_rate for p in patient_forecast]
        total_demand = sum(daily_demands)
        avg_daily_demand = np.mean(daily_demands) if daily_demands else 0

        # Get demand standard deviation from historical data or estimate
        if item_id == "GLOVES" and inventory_stats.get("std_gloves", 0) > 0:
            demand_std = inventory_stats["std_gloves"]
        elif item_id == "PPE_SETS" and inventory_stats.get("std_ppe", 0) > 0:
            demand_std = inventory_stats["std_ppe"]
        elif item_id == "MEDICATIONS" and inventory_stats.get("std_meds", 0) > 0:
            demand_std = inventory_stats["std_meds"]
        else:
            # Estimate std as 20% of average demand
            demand_std = avg_daily_demand * 0.20

        # Get lead time
        lead_time = constraints.get_lead_time(item_id, item_config.get("lead_time", 3))
        unit_cost = item_config["unit_cost"]

        # Calculate STATISTICAL SAFETY STOCK: SS = Z × σ × √L
        safety_stock = calculate_statistical_safety_stock(
            demand_std=demand_std,
            lead_time=lead_time,
            service_level=service_level
        )

        # Calculate EOQ: √((2 × D × S) / H)
        annual_demand = avg_daily_demand * 365
        eoq = calculate_eoq_formula(
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_cost_rate,
            unit_cost=unit_cost
        )

        # Calculate reorder point
        reorder_point = (avg_daily_demand * lead_time) + safety_stock

        # Optimal order = max(EOQ, total_demand + safety_stock for horizon)
        horizon_order = total_demand + safety_stock
        optimal_order = max(eoq, horizon_order) if constraints.optimization_method == "hybrid" else horizon_order

        # Apply min/max constraints
        min_qty = constraints.get_min_order_qty(item_id, 1)
        max_qty = constraints.get_max_order_qty(item_id, 10000)
        optimal_order = max(min_qty, min(max_qty, int(np.ceil(optimal_order))))

        # Cost calculations
        num_orders = max(1, np.ceil(total_demand / eoq)) if eoq > 0 else 1
        item_ordering_cost = num_orders * ordering_cost
        item_holding_cost = (optimal_order / 2 + safety_stock) * unit_cost * (holding_cost_rate / 365) * horizon
        item_purchase_cost = optimal_order * unit_cost

        total_ordering_cost += item_ordering_cost
        total_holding_cost += item_holding_cost
        total_purchase_cost += item_purchase_cost

        item_results[item_id] = {
            "name": item_config["name"],
            "forecast_demand": total_demand,
            "avg_daily_demand": avg_daily_demand,
            "demand_std": demand_std,
            "safety_stock": safety_stock,
            "eoq": eoq,
            "reorder_point": reorder_point,
            "optimal_order": optimal_order,
            "unit_cost": unit_cost,
            "lead_time": lead_time,
            "ordering_cost": item_ordering_cost,
            "holding_cost": item_holding_cost,
            "purchase_cost": item_purchase_cost,
            "service_level": service_level,
        }

    total_cost = total_ordering_cost + total_holding_cost + total_purchase_cost

    # Calculate current costs for comparison (from old method)
    current_daily_cost = 0
    for item_id, item_config in DEFAULT_INVENTORY_CONFIG.items():
        unit_cost = item_config["unit_cost"]
        avg_usage = inventory_stats.get(f"avg_{item_id.lower()}_daily", item_config["usage_per_patient"] * 100)
        # Old method: 10% safety stock
        old_safety = avg_usage * 0.10 * horizon
        old_order = avg_usage * horizon + old_safety
        current_daily_cost += old_order * unit_cost / horizon

    current_weekly_cost = current_daily_cost * 7
    optimized_daily_cost = total_cost / horizon if horizon > 0 else 0
    optimized_weekly_cost = optimized_daily_cost * 7
    weekly_savings = current_weekly_cost - optimized_weekly_cost

    return {
        "success": True,
        "method": constraints.optimization_method,
        "item_results": item_results,
        "total_cost": total_cost,
        "ordering_cost": total_ordering_cost,
        "holding_cost": total_holding_cost,
        "purchase_cost": total_purchase_cost,
        "avg_daily_cost": optimized_daily_cost,
        "current_daily_cost": current_daily_cost,
        "current_weekly_cost": current_weekly_cost,
        "optimized_weekly_cost": optimized_weekly_cost,
        "weekly_savings": weekly_savings,
        "horizon": horizon,
        "source": forecast_data["source"],
        "dates": forecast_data.get("dates", []),
        "patient_forecast": patient_forecast,
        "service_level": service_level * 100,
        "safety_factor_z": constraints.safety_factor,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_kpi_card(label: str, value: str, card_class: str = "", value_class: str = "blue", delta: str = None, delta_positive: bool = True):
    """Render a KPI card with optional delta."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card {card_class}">
        <div class="kpi-value {value_class}">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FORECAST SYNC STATUS
# =============================================================================
forecast_data = get_forecast_data()

st.markdown("""
<div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(168, 85, 247, 0.05));
            border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;'>
""", unsafe_allow_html=True)

sync_col1, sync_col2 = st.columns([3, 1])

with sync_col1:
    if forecast_data["has_forecast"]:
        sync_badge = "🔗 Synced" if forecast_data.get("synced_with_page10") else "⚠️ Fallback"
        sync_color = "#22c55e" if forecast_data.get("synced_with_page10") else "#f59e0b"
        acc_text = f" | Accuracy: {forecast_data['accuracy']:.1f}%" if forecast_data.get("accuracy") else ""

        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.25rem;'>📦 INVENTORY OPTIMIZATION SOURCE</div>
        <div style='display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;'>
            <span style='font-weight: 700; color: #e2e8f0; font-size: 1.1rem;'>{forecast_data['source']}</span>
            <span style='background: rgba({34 if forecast_data.get("synced_with_page10") else 245}, {197 if forecast_data.get("synced_with_page10") else 158}, {94 if forecast_data.get("synced_with_page10") else 11}, 0.15);
                        color: {sync_color}; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                {sync_badge}
            </span>
            <span style='background: rgba(168, 85, 247, 0.15); color: #a855f7; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                📐 EOQ + Statistical SS
            </span>
            <span style='color: #64748b; font-size: 0.8rem;'>{forecast_data['horizon']} days ahead{acc_text}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.25rem;'>📦 INVENTORY OPTIMIZATION SOURCE</div>
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <span style='font-weight: 600; color: #ef4444;'>❌ No Forecast Available</span>
            <span style='color: #64748b; font-size: 0.8rem;'>Go to Page 10 (Forecast Hub) to generate forecasts</span>
        </div>
        """, unsafe_allow_html=True)

with sync_col2:
    if st.button("🔄 Refresh", type="secondary", use_container_width=True, key="refresh_forecast_supply"):
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Upload",
    "📦 Current Situation",
    "🚀 Optimization",
    "📈 Comparison",
    "⚙️ Constraints"
])


# =============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">📊 Data Upload & Preview</div>', unsafe_allow_html=True)

    col_inv, col_cost = st.columns(2)

    # Inventory Data Section
    with col_inv:
        st.markdown('<div class="subsection-header">📦 Inventory Usage Data</div>', unsafe_allow_html=True)

        if st.button("🔄 Load Inventory Data", type="primary", use_container_width=True, key="load_inventory"):
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

        if st.session_state.inventory_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Inventory Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.inventory_stats

            st.markdown(f"""
            <div class="data-box inventory">
                <strong>Dataset:</strong> {stats['total_records']} records<br>
                <strong>Period:</strong> {stats.get('date_start', 'N/A')} to {stats.get('date_end', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                render_kpi_card("Avg Gloves/Day", f"{stats['avg_gloves_daily']:.0f}", "inventory", "blue")
            with kpi_cols[1]:
                render_kpi_card("Avg PPE/Day", f"{stats['avg_ppe_daily']:.1f}", "inventory", "blue")
            with kpi_cols[2]:
                render_kpi_card("Avg Meds/Day", f"{stats['avg_meds_daily']:.1f}", "inventory", "blue")
            with kpi_cols[3]:
                sl = stats.get('avg_service_level', 95)
                sl_class = "success" if sl >= 95 else "warning" if sl >= 90 else "danger"
                sl_color = "green" if sl >= 95 else "yellow" if sl >= 90 else "red"
                render_kpi_card("Service Level", f"{sl:.1f}%", sl_class, sl_color)

            # Show demand statistics for safety stock calculation
            st.markdown('<div class="subsection-header">📊 Demand Variability (for Safety Stock)</div>', unsafe_allow_html=True)
            std_cols = st.columns(3)
            with std_cols[0]:
                render_kpi_card("σ Gloves", f"{stats.get('std_gloves', 0):.1f}", "constraint", "cyan")
            with std_cols[1]:
                render_kpi_card("σ PPE", f"{stats.get('std_ppe', 0):.2f}", "constraint", "cyan")
            with std_cols[2]:
                render_kpi_card("σ Meds", f"{stats.get('std_meds', 0):.2f}", "constraint", "cyan")
        else:
            st.markdown('<span class="status-badge pending">⏳ Not Loaded</span>', unsafe_allow_html=True)

    # Cost Parameters Section
    with col_cost:
        st.markdown('<div class="subsection-header">💰 Cost Parameters</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="data-box cost">
            <strong>Current Settings:</strong><br>
            Holding Cost Rate: {st.session_state.inv_cost_params['holding_cost_pct'] * 100:.0f}%/year<br>
            Ordering Cost: ${st.session_state.inv_cost_params['ordering_cost']:.0f}/order<br>
            Stockout Penalty: ${st.session_state.inv_cost_params['stockout_penalty']:.0f}/unit
        </div>
        """, unsafe_allow_html=True)

        # Formula reference
        st.markdown('<div class="subsection-header">📐 Formulas Used</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
            <strong>Safety Stock:</strong> SS = Z × σ × √L<br>
            <small>Z = service level z-score, σ = demand std dev, L = lead time</small>
        </div>
        <div class="formula-box">
            <strong>EOQ:</strong> Q* = √((2 × D × S) / H)<br>
            <small>D = annual demand, S = ordering cost, H = holding cost</small>
        </div>
        <div class="formula-box">
            <strong>Reorder Point:</strong> ROP = (d × L) + SS<br>
            <small>d = avg daily demand, L = lead time</small>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: CURRENT INVENTORY SITUATION
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">📦 Current Inventory Situation</div>', unsafe_allow_html=True)

    if not st.session_state.inventory_data_loaded:
        st.warning("⚠️ Please load inventory data in Tab 1 first.")
    else:
        stats = st.session_state.inventory_stats
        params = st.session_state.inv_cost_params

        # Usage Summary
        st.markdown('<div class="subsection-header">📊 Daily Usage Summary</div>', unsafe_allow_html=True)

        usage_cols = st.columns(4)
        with usage_cols[0]:
            render_kpi_card("Gloves/Day", f"{stats['avg_gloves_daily']:.0f}", "inventory", "blue")
        with usage_cols[1]:
            render_kpi_card("PPE Sets/Day", f"{stats['avg_ppe_daily']:.1f}", "inventory", "blue")
        with usage_cols[2]:
            render_kpi_card("Medications/Day", f"{stats['avg_meds_daily']:.1f}", "inventory", "purple")
        with usage_cols[3]:
            stockouts = stats.get('stockout_events', 0)
            card_class = "danger" if stockouts > 5 else "warning" if stockouts > 0 else "success"
            render_kpi_card("Stockout Events", f"{stockouts}", card_class, "red" if stockouts > 5 else "yellow")

        # Cost Situation
        st.markdown('<div class="subsection-header">💰 Cost Situation</div>', unsafe_allow_html=True)

        # Estimate current costs using old method (10% safety stock)
        current_daily_cost = 0
        for item_id, item_config in DEFAULT_INVENTORY_CONFIG.items():
            unit_cost = item_config["unit_cost"]
            usage = stats.get(f"avg_{item_id.lower()}_daily", item_config["usage_per_patient"] * 100)
            daily_holding = usage * unit_cost * (params['holding_cost_pct'] / 365)
            daily_ordering = params['ordering_cost'] / 7
            current_daily_cost += daily_holding + daily_ordering

        cost_cols = st.columns(4)
        with cost_cols[0]:
            render_kpi_card("Est. Daily Cost", f"${current_daily_cost:,.0f}", "cost", "purple")
        with cost_cols[1]:
            render_kpi_card("Weekly Cost", f"${current_daily_cost * 7:,.0f}", "cost", "purple")
        with cost_cols[2]:
            render_kpi_card("Holding Rate", f"{params['holding_cost_pct'] * 100:.0f}%", "cost", "blue")
        with cost_cols[3]:
            sl = stats.get('avg_service_level', 95)
            render_kpi_card("Service Level", f"{sl:.1f}%", "success" if sl >= 95 else "warning", "green" if sl >= 95 else "yellow")


# =============================================================================
# TAB 3: AFTER OPTIMIZATION
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">🚀 After Optimization</div>', unsafe_allow_html=True)

    if not forecast_data["has_forecast"]:
        st.warning("⚠️ No forecast available. Please generate forecasts in Page 10 first.")
    elif not st.session_state.inventory_data_loaded:
        st.warning("⚠️ Please load inventory data in Tab 1 first.")
    else:
        constraints = st.session_state.inventory_constraint_config

        # Service level selector
        col_sl, col_method, col_run = st.columns([2, 2, 1])

        with col_sl:
            service_level_options = constraints.get_service_level_options()
            sl_labels = [opt["label"] for opt in service_level_options]
            current_idx = 1  # Default to 95%
            for i, opt in enumerate(service_level_options):
                if abs(opt["level"] - constraints.target_service_level) < 0.01:
                    current_idx = i
                    break

            selected_sl = st.selectbox(
                "Target Service Level",
                options=sl_labels,
                index=current_idx,
                key="service_level_select"
            )
            selected_opt = service_level_options[sl_labels.index(selected_sl)]
            constraints.target_service_level = selected_opt["level"]
            constraints.safety_factor = selected_opt["z_score"]

            st.markdown(f"""
            <div style='font-size: 0.8rem; color: #94a3b8;'>
                Z-score: {selected_opt['z_score']:.3f} → SS = {selected_opt['z_score']:.3f} × σ × √L
            </div>
            """, unsafe_allow_html=True)

        with col_method:
            method = st.radio(
                "Optimization Method",
                ["hybrid", "eoq_only", "milp_full"],
                format_func=lambda x: {"hybrid": "📐 Hybrid (EOQ + Horizon)", "eoq_only": "📊 EOQ Only", "milp_full": "🧮 MILP Full"}[x],
                horizontal=True,
                key="opt_method_select"
            )
            constraints.optimization_method = method

        with col_run:
            run_optimization = st.button("🚀 Optimize", type="primary", use_container_width=True)

        if run_optimization:
            with st.spinner("Running EOQ optimization with statistical safety stock..."):
                results = calculate_inventory_optimization(
                    forecast_data,
                    st.session_state.inventory_stats,
                    st.session_state.inv_cost_params,
                    constraints
                )

                if results["success"]:
                    st.session_state.inv_optimization_done = True
                    st.session_state.inv_optimized_results = results
                    st.success(f"✅ Optimization complete! Service Level: {results['service_level']:.1f}%")
                else:
                    st.error(f"❌ {results.get('error', 'Unknown error')}")

        # Display results
        if st.session_state.inv_optimization_done and st.session_state.inv_optimized_results:
            results = st.session_state.inv_optimized_results

            st.markdown('<div class="subsection-header">📦 Optimized Order Quantities</div>', unsafe_allow_html=True)

            # Results table
            results_data = []
            for item_id, item_data in results["item_results"].items():
                results_data.append({
                    "Item": item_data["name"],
                    "Demand": f"{item_data['forecast_demand']:.0f}",
                    "σ (Std Dev)": f"{item_data['demand_std']:.1f}",
                    "Safety Stock": f"{item_data['safety_stock']:.0f}",
                    "EOQ": f"{item_data['eoq']:.0f}",
                    "Reorder Point": f"{item_data['reorder_point']:.0f}",
                    "Order Qty": f"{item_data['optimal_order']}",
                    "Total Cost": f"${item_data['ordering_cost'] + item_data['holding_cost'] + item_data['purchase_cost']:,.0f}",
                })

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, hide_index=True, use_container_width=True)

            # Cost breakdown
            st.markdown('<div class="subsection-header">💰 Cost Breakdown</div>', unsafe_allow_html=True)

            cost_cols = st.columns(4)
            with cost_cols[0]:
                render_kpi_card("Ordering Cost", f"${results['ordering_cost']:,.0f}", "cost", "blue")
            with cost_cols[1]:
                render_kpi_card("Holding Cost", f"${results['holding_cost']:,.0f}", "cost", "purple")
            with cost_cols[2]:
                render_kpi_card("Purchase Cost", f"${results['purchase_cost']:,.0f}", "cost", "blue")
            with cost_cols[3]:
                render_kpi_card("Total Cost", f"${results['total_cost']:,.0f}", "success", "green")

            # Charts
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Order quantities chart
                fig_orders = go.Figure(data=[
                    go.Bar(
                        x=[r["Item"][:15] for r in results_data],
                        y=[results["item_results"][item_id]["optimal_order"] for item_id in results["item_results"]],
                        marker_color="#3b82f6",
                        text=[results["item_results"][item_id]["optimal_order"] for item_id in results["item_results"]],
                        textposition="outside"
                    )
                ])
                fig_orders.update_layout(
                    title="Optimal Order Quantities",
                    template="plotly_dark",
                    height=300,
                    yaxis_title="Units",
                )
                st.plotly_chart(fig_orders, use_container_width=True)

            with col_chart2:
                # Safety stock chart
                fig_ss = go.Figure(data=[
                    go.Bar(
                        x=[r["Item"][:15] for r in results_data],
                        y=[results["item_results"][item_id]["safety_stock"] for item_id in results["item_results"]],
                        marker_color="#a855f7",
                        text=[f"{results['item_results'][item_id]['safety_stock']:.0f}" for item_id in results["item_results"]],
                        textposition="outside"
                    )
                ])
                fig_ss.update_layout(
                    title=f"Safety Stock (Z={results['safety_factor_z']:.2f} for {results['service_level']:.0f}% SL)",
                    template="plotly_dark",
                    height=300,
                    yaxis_title="Units",
                )
                st.plotly_chart(fig_ss, use_container_width=True)


# =============================================================================
# TAB 4: BEFORE VS AFTER COMPARISON
# =============================================================================
with tab4:
    st.markdown('<div class="section-header">📈 Before vs After Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.inv_optimization_done:
        st.warning("⚠️ Please run optimization in Tab 3 first.")
    else:
        results = st.session_state.inv_optimized_results

        # Financial Impact
        st.markdown('<div class="subsection-header">💰 Financial Impact</div>', unsafe_allow_html=True)

        impact_cols = st.columns(4)
        with impact_cols[0]:
            render_kpi_card("Current Weekly", f"${results['current_weekly_cost']:,.0f}", "cost", "blue")
        with impact_cols[1]:
            render_kpi_card("Optimized Weekly", f"${results['optimized_weekly_cost']:,.0f}", "success", "green")
        with impact_cols[2]:
            savings = results['weekly_savings']
            savings_class = "success" if savings > 0 else "danger"
            render_kpi_card("Weekly Savings", f"${savings:,.0f}", savings_class, "green" if savings > 0 else "red")
        with impact_cols[3]:
            render_kpi_card("Service Level", f"{results['service_level']:.1f}%", "success", "green")

        # Comparison chart
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_cost = go.Figure(data=[
                go.Bar(
                    x=["Current (10% SS)", f"Optimized (Z={results['safety_factor_z']:.2f})"],
                    y=[results['current_weekly_cost'], results['optimized_weekly_cost']],
                    marker_color=["#3b82f6", "#22c55e"],
                    text=[f"${results['current_weekly_cost']:,.0f}", f"${results['optimized_weekly_cost']:,.0f}"],
                    textposition="outside"
                )
            ])
            fig_cost.update_layout(
                title="Weekly Cost Comparison",
                template="plotly_dark",
                height=300,
                yaxis_title="Cost ($)",
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        with col_chart2:
            # Method comparison info
            st.markdown(f"""
            <div class="data-box">
                <strong>Optimization Method:</strong> {results['method'].replace('_', ' ').title()}<br>
                <strong>Service Level:</strong> {results['service_level']:.1f}%<br>
                <strong>Safety Factor (Z):</strong> {results['safety_factor_z']:.3f}<br>
                <strong>Forecast Horizon:</strong> {results['horizon']} days<br>
                <strong>Forecast Source:</strong> {results['source']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="formula-box" style="margin-top: 1rem;">
                <strong>Old Method:</strong> SS = 10% × Demand (fixed)<br>
                <strong>New Method:</strong> SS = Z × σ × √L (statistical)
            </div>
            """, unsafe_allow_html=True)

        # AI Insights
        st.markdown('<div class="subsection-header">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)

        inject_insight_styles()

        insight_col1, insight_col2, insight_col3 = st.columns([2, 2, 1])
        with insight_col1:
            mode = st.radio(
                "Insight Mode",
                ["rule_based", "llm"],
                format_func=lambda x: "📊 Rule-Based Analysis" if x == "rule_based" else "🧠 LLM-Enhanced",
                horizontal=True,
                key="supply_insight_mode_selector"
            )
            st.session_state.supply_insight_mode = mode

        with insight_col2:
            if mode == "llm":
                provider = st.selectbox(
                    "Provider",
                    ["anthropic", "openai"],
                    format_func=lambda x: "Anthropic (Claude)" if x == "anthropic" else "OpenAI (GPT)",
                    key="supply_provider_select"
                )
                st.session_state.supply_api_provider = provider

        with insight_col3:
            if mode == "llm":
                api_key = st.text_input("API Key", type="password", key="supply_api_key_input")
                st.session_state.supply_llm_api_key = api_key

        # Build insight data
        before_data = {
            "weekly_cost": results['current_weekly_cost'],
            "service_level": 95.0,  # Old assumed service level
            "stockout_risk": 5.0,  # Estimated stockout risk with 10% safety stock
            "safety_stock_method": "Fixed 10%",
        }

        after_data = {
            "weekly_cost": results['optimized_weekly_cost'],
            "service_level": results['service_level'],
            "stockout_risk": (1 - results['service_level'] / 100) * 100,
            "safety_stock_method": f"Statistical (Z={results['safety_factor_z']:.2f})",
        }

        render_supply_optimization_insights(
            before_data=before_data,
            after_data=after_data,
            mode=st.session_state.supply_insight_mode,
            api_key=st.session_state.supply_llm_api_key,
            provider=st.session_state.supply_api_provider
        )


# =============================================================================
# TAB 5: CONSTRAINT CONFIGURATION
# =============================================================================
with tab5:
    st.markdown('<div class="section-header">⚙️ Constraint Configuration</div>', unsafe_allow_html=True)

    constraints = st.session_state.inventory_constraint_config

    # Service Level Configuration
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">🎯 Service Level Configuration</div>', unsafe_allow_html=True)

    sl_col1, sl_col2 = st.columns(2)
    with sl_col1:
        service_level = st.slider(
            "Target Service Level (%)",
            min_value=90.0,
            max_value=99.9,
            value=constraints.target_service_level * 100,
            step=0.5,
            key="sl_slider"
        )

        # Auto-calculate Z-score
        z_score = get_z_score(service_level / 100)
        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #94a3b8;'>
            Corresponding Z-score: <strong>{z_score:.3f}</strong><br>
            <small>Formula: SS = {z_score:.3f} × σ × √L</small>
        </div>
        """, unsafe_allow_html=True)

    with sl_col2:
        st.markdown("""
        **Service Level Guide:**
        - 90%: Standard items (Z=1.28)
        - 95%: Important items (Z=1.65)
        - 99%: Critical items (Z=2.33)
        - 99.5%: Life-critical (Z=2.58)
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Storage & Budget Constraints
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">📦 Storage & Budget Constraints</div>', unsafe_allow_html=True)

    storage_col1, storage_col2, storage_col3 = st.columns(3)
    with storage_col1:
        storage_capacity = st.number_input(
            "Storage Capacity (m³)",
            min_value=100,
            max_value=10000,
            value=int(constraints.storage_capacity),
            step=100,
            key="storage_input"
        )
    with storage_col2:
        daily_budget = st.number_input(
            "Daily Budget ($)",
            min_value=0,
            max_value=100000,
            value=int(constraints.daily_budget or 0),
            step=500,
            key="daily_budget_input",
            help="Set to 0 for unlimited"
        )
    with storage_col3:
        holding_rate = st.slider(
            "Holding Cost Rate (%/year)",
            min_value=10,
            max_value=40,
            value=int(constraints.holding_cost_rate * 100),
            step=5,
            key="holding_rate_slider"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Lead Times
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">🚚 Lead Times (days)</div>', unsafe_allow_html=True)

    lead_cols = st.columns(6)
    lead_times = {}
    for i, (item_id, item_config) in enumerate(DEFAULT_INVENTORY_CONFIG.items()):
        with lead_cols[i % 6]:
            lt = st.number_input(
                f"{item_config['name'][:12]}",
                min_value=1,
                max_value=30,
                value=constraints.get_lead_time(item_id, item_config["lead_time"]),
                key=f"lt_{item_id}"
            )
            lead_times[item_id] = lt
    st.markdown('</div>', unsafe_allow_html=True)

    # Solver Settings
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">🧮 Optimization Settings</div>', unsafe_allow_html=True)

    solver_col1, solver_col2, solver_col3 = st.columns(3)
    with solver_col1:
        opt_method = st.selectbox(
            "Optimization Method",
            ["hybrid", "eoq_only", "milp_full"],
            format_func=lambda x: {"hybrid": "Hybrid (EOQ + Demand)", "eoq_only": "EOQ Only", "milp_full": "MILP Full"}[x],
            index=["hybrid", "eoq_only", "milp_full"].index(constraints.optimization_method),
            key="opt_method_config"
        )
    with solver_col2:
        planning_horizon = st.slider(
            "Planning Horizon (days)",
            min_value=7,
            max_value=90,
            value=constraints.planning_horizon,
            step=7,
            key="horizon_slider"
        )
    with solver_col3:
        time_limit = st.slider(
            "Solver Time Limit (s)",
            min_value=10,
            max_value=300,
            value=constraints.solve_time_limit,
            step=10,
            key="time_limit_inv_slider"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Save/Load buttons
    save_col1, save_col2 = st.columns(2)

    with save_col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True, key="save_inv_config"):
            new_config = InventoryConstraintConfig(
                storage_capacity=float(storage_capacity),
                daily_budget=float(daily_budget) if daily_budget > 0 else None,
                target_service_level=service_level / 100,
                safety_factor=get_z_score(service_level / 100),
                lead_times=lead_times,
                holding_cost_rate=holding_rate / 100,
                optimization_method=opt_method,
                planning_horizon=planning_horizon,
                solve_time_limit=time_limit,
            )

            is_valid, errors = new_config.validate()
            if is_valid:
                st.session_state.inventory_constraint_config = new_config
                st.success("✅ Configuration saved!")
            else:
                for error in errors:
                    st.error(f"❌ {error}")

    with save_col2:
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True, key="reset_inv_config"):
            st.session_state.inventory_constraint_config = DEFAULT_INVENTORY_CONSTRAINTS
            st.success("✅ Reset to defaults!")
            st.rerun()


# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(current_page_index=12)
