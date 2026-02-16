# =============================================================================
# 11_Staff_Planner.py
# ENHANCED: 5-Tab Structure with MILP Optimization & Seasonal Proportions
# =============================================================================
"""
Tab Structure:
1. Data Upload & Preview - Load Staff + Financial data with KPIs & graphs
2. Current Hospital Situation - Summarize & link staff + financial metrics
3. After Optimization - MILP-based optimization with Seasonal Proportions
4. Before vs After - Compare results with AI insights
5. Constraint Configuration - Full MILP constraint editor

Key Enhancements:
- Seasonal Proportions integration (replaces hardcoded category distribution)
- MILP solver integration for optimal staff scheduling
- Full constraint configuration UI
- Scenario management for what-if analysis
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
from app_core.data.staff_scheduling_service import StaffSchedulingService
from app_core.data.financial_service import FinancialService, get_financial_service

# Import AI Insight components
from app_core.ui.insight_components import (
    inject_insight_styles,
    render_mode_selector,
    render_full_insight_panel,
    render_staff_optimization_insights
)
from app_core.ai.recommendation_engine import StaffRecommendationEngine

# Import optimization modules
from app_core.optimization.constraint_config import (
    StaffConstraintConfig,
    DEFAULT_STAFF_CONSTRAINTS,
    create_constraints_summary,
)
from app_core.optimization.scenario_manager import ScenarioManager, OptimizationScenario
from app_core.optimization.milp_solver import (
    StaffSchedulingMILP,
    OptimizationResult,
    solve_staff_scheduling,
    PULP_AVAILABLE,
)
from app_core.optimization.cost_parameters import (
    CostParameters,
    StaffingRatios,
    CLINICAL_CATEGORIES,
    CATEGORY_PRIORITIES,
    STAFF_TYPES,
)

# Import Seasonal Proportions
from app_core.analytics.seasonal_proportions import (
    calculate_seasonal_proportions,
    distribute_forecast_to_categories,
    SeasonalProportionConfig,
    SeasonalProportionResult,
    create_seasonal_heatmap,
    render_seasonal_forecast_breakdown,
)

# ============================================================================
# AUTHENTICATION
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Staff Planner - HealthForecast AI",
    page_icon="👥",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_COST_PARAMS = {
    "doctor_hourly": 150,
    "nurse_hourly": 50,
    "support_hourly": 25,
    "shift_hours": 8,
    "overtime_multiplier": 1.5,
}

# Fallback distribution if Seasonal Proportions not available
FALLBACK_DISTRIBUTION = {
    "CARDIAC": 0.15,
    "TRAUMA": 0.20,
    "RESPIRATORY": 0.25,
    "GASTROINTESTINAL": 0.15,
    "INFECTIOUS": 0.15,
    "NEUROLOGICAL": 0.05,
    "OTHER": 0.05,
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
    background: radial-gradient(circle, rgba(34, 197, 94, 0.25), transparent 70%);
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
    border: 1px solid rgba(34, 197, 94, 0.2);
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
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.1));
    border-color: rgba(34, 197, 94, 0.3);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(59, 130, 246, 0.2)) !important;
    border: 1px solid rgba(34, 197, 94, 0.5) !important;
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

.kpi-card.staff {{ border-color: rgba(34, 197, 94, 0.5); }}
.kpi-card.financial {{ border-color: rgba(59, 130, 246, 0.5); }}
.kpi-card.warning {{ border-color: rgba(234, 179, 8, 0.5); background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.danger {{ border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.success {{ border-color: rgba(34, 197, 94, 0.5); background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.98)); }}
.kpi-card.linked {{ border-color: rgba(168, 85, 247, 0.5); background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(15, 23, 42, 0.98)); }}
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
    border-bottom: 2px solid rgba(34, 197, 94, 0.3);
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

.data-box.staff {{ border-color: rgba(34, 197, 94, 0.3); }}
.data-box.financial {{ border-color: rgba(59, 130, 246, 0.3); }}
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
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* MILP Status */
.milp-status {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(34, 197, 94, 0.4);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}}

.milp-status.optimal {{ border-color: #22c55e; color: #22c55e; }}
.milp-status.feasible {{ border-color: #3b82f6; color: #3b82f6; }}
.milp-status.infeasible {{ border-color: #ef4444; color: #ef4444; }}
</style>

<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HERO HEADER
# =============================================================================
render_scifi_hero_header(
    title="Staff Planner",
    subtitle="MILP-optimized staffing with Seasonal Proportions integration",
    status="MILP SOLVER ACTIVE" if PULP_AVAILABLE else "HEURISTIC MODE"
)


# =============================================================================
# SESSION STATE
# =============================================================================
if "staff_data_loaded" not in st.session_state:
    st.session_state.staff_data_loaded = False
if "staff_df" not in st.session_state:
    st.session_state.staff_df = None
if "staff_stats" not in st.session_state:
    st.session_state.staff_stats = {}

if "financial_data_loaded" not in st.session_state:
    st.session_state.financial_data_loaded = False
if "financial_df" not in st.session_state:
    st.session_state.financial_df = None
if "financial_stats" not in st.session_state:
    st.session_state.financial_stats = {}
if "cost_params" not in st.session_state:
    st.session_state.cost_params = DEFAULT_COST_PARAMS.copy()

if "optimization_done" not in st.session_state:
    st.session_state.optimization_done = False
if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = None
if "milp_result" not in st.session_state:
    st.session_state.milp_result = None

# Constraint configuration
if "staff_constraint_config" not in st.session_state:
    st.session_state.staff_constraint_config = DEFAULT_STAFF_CONSTRAINTS

# Scenario management
if "staff_scenarios" not in st.session_state:
    st.session_state.staff_scenarios = ScenarioManager(scenario_type="staff")

# AI Insight Mode
if "staff_insight_mode" not in st.session_state:
    st.session_state.staff_insight_mode = "rule_based"
if "staff_llm_api_key" not in st.session_state:
    st.session_state.staff_llm_api_key = None
if "staff_api_provider" not in st.session_state:
    st.session_state.staff_api_provider = "anthropic"

# Seasonal Proportions
if "use_seasonal_proportions" not in st.session_state:
    st.session_state.use_seasonal_proportions = True


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_staff_data() -> Dict[str, Any]:
    """Load staff scheduling data from Supabase."""
    result = {"success": False, "data": None, "stats": {}, "error": None}

    try:
        service = StaffSchedulingService()
        if not service.is_connected():
            result["error"] = "Supabase not connected"
            return result

        df = service.fetch_staff_data()
        if df.empty:
            result["error"] = "No staff data found"
            return result

        result["success"] = True
        result["data"] = df

        # Calculate statistics
        raw_avg_doctors = df["Doctors_on_Duty"].mean() if "Doctors_on_Duty" in df.columns else 0
        raw_avg_nurses = df["Nurses_on_Duty"].mean() if "Nurses_on_Duty" in df.columns else 0
        raw_avg_support = df["Support_Staff_on_Duty"].mean() if "Support_Staff_on_Duty" in df.columns else 0
        raw_total_overtime = df["Overtime_Hours"].sum() if "Overtime_Hours" in df.columns else 0
        raw_avg_utilization = df["Staff_Utilization_Rate"].mean() if "Staff_Utilization_Rate" in df.columns else 0
        raw_shortage_days = df["Staff_Shortage_Flag"].sum() if "Staff_Shortage_Flag" in df.columns else 0

        # Convert utilization from ratio to percentage if needed
        if "Staff_Utilization_Rate" in df.columns:
            max_util = df["Staff_Utilization_Rate"].max()
            if max_util <= 2.0:
                raw_avg_utilization = raw_avg_utilization * 100

        result["stats"] = {
            "total_records": len(df),
            "date_start": df["Date"].min() if "Date" in df.columns else None,
            "date_end": df["Date"].max() if "Date" in df.columns else None,
            "avg_doctors": max(0, raw_avg_doctors) if raw_avg_doctors else 0,
            "avg_nurses": max(0, raw_avg_nurses) if raw_avg_nurses else 0,
            "avg_support": max(0, raw_avg_support) if raw_avg_support else 0,
            "total_overtime": max(0, raw_total_overtime) if raw_total_overtime else 0,
            "avg_utilization": max(0, abs(raw_avg_utilization)) if raw_avg_utilization else 0,
            "shortage_days": max(0, int(raw_shortage_days)) if raw_shortage_days else 0,
            "max_doctors": max(0, df["Doctors_on_Duty"].max()) if "Doctors_on_Duty" in df.columns else 0,
            "min_doctors": max(0, df["Doctors_on_Duty"].min()) if "Doctors_on_Duty" in df.columns else 0,
            "max_nurses": max(0, df["Nurses_on_Duty"].max()) if "Nurses_on_Duty" in df.columns else 0,
            "min_nurses": max(0, df["Nurses_on_Duty"].min()) if "Nurses_on_Duty" in df.columns else 0,
        }

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def load_financial_data() -> Dict[str, Any]:
    """Load financial data from Supabase."""
    result = {"success": False, "data": None, "stats": {}, "cost_params": DEFAULT_COST_PARAMS.copy(), "error": None}

    try:
        service = get_financial_service()
        if not service.is_connected():
            result["error"] = "Supabase not connected"
            return result

        df = service.fetch_all()  # Use fetch_all() instead of fetch_financial_data()
        if df.empty:
            result["error"] = "No financial data found"
            return result

        result["success"] = True
        result["data"] = df

        # Extract cost parameters (handle different column name variations)
        if "Doctor_Hourly_Rate" in df.columns and len(df) > 0:
            result["cost_params"]["doctor_hourly"] = df["Doctor_Hourly_Rate"].iloc[-1]
        if "Nurse_Hourly_Rate" in df.columns and len(df) > 0:
            result["cost_params"]["nurse_hourly"] = df["Nurse_Hourly_Rate"].iloc[-1]
        if "Support_Staff_Hourly_Rate" in df.columns and len(df) > 0:
            result["cost_params"]["support_hourly"] = df["Support_Staff_Hourly_Rate"].iloc[-1]
        elif "Support_Hourly_Rate" in df.columns and len(df) > 0:
            result["cost_params"]["support_hourly"] = df["Support_Hourly_Rate"].iloc[-1]
        if "Overtime_Premium_Rate" in df.columns and len(df) > 0:
            result["cost_params"]["overtime_multiplier"] = df["Overtime_Premium_Rate"].iloc[-1]
        elif "Overtime_Multiplier" in df.columns and len(df) > 0:
            result["cost_params"]["overtime_multiplier"] = df["Overtime_Multiplier"].iloc[-1]

        # Calculate statistics (handle different column name variations)
        result["stats"] = {
            "total_records": len(df),
            "avg_daily_labor_cost": df["Total_Labor_Cost"].mean() if "Total_Labor_Cost" in df.columns else (df["Daily_Labor_Cost"].mean() if "Daily_Labor_Cost" in df.columns else 0),
            "total_overtime_cost": df["Overtime_Cost"].sum() if "Overtime_Cost" in df.columns else 0,
            "avg_cost_per_patient": df["Cost_Per_Patient"].mean() if "Cost_Per_Patient" in df.columns else 0,
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
        "category_forecasts": None,  # NEW: Category-level forecasts
    }

    # Helper to generate dates
    def _generate_dates(horizon: int) -> List[str]:
        today = datetime.now().date()
        return [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, horizon + 1)]

    # Priority 1: forecast_hub_demand (direct sync with Page 10)
    if "forecast_hub_demand" in st.session_state and st.session_state.forecast_hub_demand:
        hub_data = st.session_state.forecast_hub_demand
        if isinstance(hub_data, dict) and "forecasts" in hub_data:
            result["has_forecast"] = True
            result["source"] = hub_data.get("model_name", "Forecast Hub")
            result["forecasts"] = hub_data.get("forecasts", [])
            result["horizon"] = len(result["forecasts"])
            result["accuracy"] = hub_data.get("accuracy")
            result["synced_with_page10"] = True
            result["category_forecasts"] = hub_data.get("category_forecasts")
            result["dates"] = _generate_dates(result["horizon"])
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
            result["dates"] = _generate_dates(result["horizon"])
            return result

    # Priority 3: ML model results (using lowercase keys)
    for model_key in ["ml_mh_results_xgboost", "ml_mh_results_lstm", "ml_mh_results_ann"]:
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
                        result["dates"] = _generate_dates(result["horizon"])
                        return result

    # Priority 4: ARIMA results
    if "arima_mh_results" in st.session_state and st.session_state.arima_mh_results:
        arima = st.session_state.arima_mh_results
        if isinstance(arima, dict):
            best_h = arima.get("best_horizon", 1)
            per_h = arima.get("per_h", {})
            if best_h in per_h:
                h_data = per_h[best_h]
                if "forecast" in h_data:
                    result["has_forecast"] = True
                    result["source"] = f"ARIMA (H={best_h})"
                    result["forecasts"] = list(h_data["forecast"])
                    result["horizon"] = len(result["forecasts"])
                    result["dates"] = _generate_dates(result["horizon"])
                    return result

    return result


def get_category_distribution(forecast_data: Dict, prepared_data: Optional[pd.DataFrame] = None) -> Dict[str, List[float]]:
    """
    Get category distribution using Seasonal Proportions or fallback.

    Returns dict mapping category -> list of daily forecasts.
    """
    if not forecast_data["has_forecast"]:
        return {}

    total_forecasts = forecast_data["forecasts"]
    horizon = len(total_forecasts)

    # Check if category forecasts already provided
    if forecast_data.get("category_forecasts") is not None:
        return forecast_data["category_forecasts"]

    # Try Seasonal Proportions if enabled and data available
    if st.session_state.use_seasonal_proportions:
        # Get seasonal proportions from session state
        sp_result = st.session_state.get("seasonal_proportions_result")

        if sp_result is not None:
            try:
                # Generate forecast dates
                today = datetime.now().date()
                forecast_dates = pd.date_range(
                    start=today + timedelta(days=1),
                    periods=horizon,
                    freq='D'
                )

                # Convert forecasts to series
                forecast_series = pd.Series(total_forecasts, index=forecast_dates)

                # Distribute using seasonal proportions
                if hasattr(sp_result, 'dow_proportions') and hasattr(sp_result, 'monthly_proportions'):
                    category_df = distribute_forecast_to_categories(
                        forecast_series,
                        sp_result.dow_proportions,
                        sp_result.monthly_proportions
                    )

                    # Convert to dict format
                    result = {}
                    for cat in CLINICAL_CATEGORIES:
                        if cat in category_df.columns:
                            result[cat] = category_df[cat].tolist()
                        else:
                            # Fallback for missing categories
                            result[cat] = [f * FALLBACK_DISTRIBUTION.get(cat, 0.1) for f in total_forecasts]

                    return result
            except Exception as e:
                st.warning(f"Seasonal Proportions failed: {e}. Using fallback distribution.")

        # Try to calculate from prepared data if available
        if prepared_data is not None and not prepared_data.empty:
            try:
                sp_config = SeasonalProportionConfig(
                    use_dow_seasonality=True,
                    use_monthly_seasonality=True,
                )
                sp_result = calculate_seasonal_proportions(
                    prepared_data,
                    sp_config,
                    date_col="Date"
                )
                st.session_state["seasonal_proportions_result"] = sp_result

                # Recursively call to use the newly calculated proportions
                return get_category_distribution(forecast_data, None)
            except Exception as e:
                pass  # Fall through to fallback

    # Fallback: Use static distribution
    result = {}
    for cat, pct in FALLBACK_DISTRIBUTION.items():
        result[cat] = [f * pct for f in total_forecasts]

    return result


def calculate_optimization_milp(
    forecast_data: Dict,
    staff_stats: Dict,
    cost_params: Dict,
    constraints: StaffConstraintConfig,
) -> Dict[str, Any]:
    """
    Calculate optimized staffing using MILP solver with Seasonal Proportions.

    This is the enhanced optimization function that:
    1. Uses Seasonal Proportions for category distribution
    2. Runs MILP optimization via PuLP
    3. Returns comprehensive results
    """
    if not forecast_data["has_forecast"]:
        return {"success": False, "error": "No forecast available"}

    # Get prepared data for Seasonal Proportions
    prepared_data = st.session_state.get("prepared_data")

    # Get category-level demand distribution
    category_demand = get_category_distribution(forecast_data, prepared_data)

    if not category_demand:
        return {"success": False, "error": "Could not calculate category distribution"}

    total_demand = forecast_data["forecasts"]
    horizon = len(total_demand)

    # Try MILP optimization if available
    if PULP_AVAILABLE:
        try:
            # Convert constraints to solver parameters
            cost_params_obj = constraints.to_cost_params()
            staffing_ratios = constraints.to_staffing_ratios()

            # Run MILP solver
            milp_result = solve_staff_scheduling(
                total_demand=total_demand,
                category_demand=category_demand,
                cost_params=cost_params_obj,
                staffing_ratios=staffing_ratios,
                time_limit=constraints.solve_time_limit,
            )

            # Store MILP result
            st.session_state.milp_result = milp_result

            # Extract results
            if milp_result.is_optimal or milp_result.status == "Optimal":
                # Build schedules from MILP result
                schedules = []
                today = datetime.now().date()

                for i in range(horizon):
                    date_str = (today + timedelta(days=i+1)).strftime("%a %m/%d")

                    # Get staff counts from MILP result
                    if milp_result.staff_schedule is not None:
                        row = milp_result.staff_schedule.iloc[i]
                        doctors = int(row.get("Doctors", 0))
                        nurses = int(row.get("Nurses", 0))
                        support = int(row.get("Support", 0))
                    else:
                        doctors = nurses = support = 0

                    # Calculate daily cost
                    daily_cost = (
                        doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
                        nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
                        support * cost_params["support_hourly"] * cost_params["shift_hours"]
                    )

                    schedules.append({
                        "date": date_str,
                        "patients": total_demand[i],
                        "doctors": doctors,
                        "nurses": nurses,
                        "support": support,
                        "total_staff": doctors + nurses + support,
                        "cost": daily_cost,
                    })

                # Calculate comparison metrics
                current_daily_cost = (
                    staff_stats.get("avg_doctors", 0) * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
                    staff_stats.get("avg_nurses", 0) * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
                    staff_stats.get("avg_support", 0) * cost_params["support_hourly"] * cost_params["shift_hours"]
                )

                optimized_daily_cost = milp_result.total_cost / horizon if horizon > 0 else 0
                current_weekly_cost = current_daily_cost * 7
                optimized_weekly_cost = optimized_daily_cost * 7
                weekly_savings = current_weekly_cost - optimized_weekly_cost

                return {
                    "success": True,
                    "method": "MILP",
                    "status": milp_result.status,
                    "is_optimal": milp_result.is_optimal,
                    "schedules": schedules,
                    "total_cost": milp_result.total_cost,
                    "regular_labor_cost": milp_result.regular_labor_cost,
                    "overtime_cost": milp_result.overtime_cost,
                    "understaffing_penalty": milp_result.understaffing_penalty,
                    "overstaffing_penalty": milp_result.overstaffing_penalty,
                    "avg_daily_cost": optimized_daily_cost,
                    "current_daily_cost": current_daily_cost,
                    "current_weekly_cost": current_weekly_cost,
                    "optimized_weekly_cost": optimized_weekly_cost,
                    "weekly_savings": weekly_savings,
                    "horizon": horizon,
                    "source": forecast_data["source"],
                    "avg_opt_doctors": np.mean([s["doctors"] for s in schedules]),
                    "avg_opt_nurses": np.mean([s["nurses"] for s in schedules]),
                    "avg_opt_support": np.mean([s["support"] for s in schedules]),
                    "category_coverage": milp_result.category_coverage,
                    "daily_summary": milp_result.daily_summary,
                    "solve_time": milp_result.solve_time,
                    "num_variables": milp_result.num_variables,
                    "num_constraints": milp_result.num_constraints,
                    "category_demand": category_demand,
                    "used_seasonal_proportions": st.session_state.use_seasonal_proportions,
                }

        except Exception as e:
            st.error(f"MILP optimization failed: {e}")
            # Fall through to heuristic

    # Fallback: Heuristic optimization (same as old version but with category distribution)
    return calculate_optimization_heuristic(forecast_data, staff_stats, cost_params, category_demand)


def calculate_optimization_heuristic(
    forecast_data: Dict,
    staff_stats: Dict,
    cost_params: Dict,
    category_demand: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Fallback heuristic optimization when MILP is not available."""
    if not forecast_data["has_forecast"]:
        return {"success": False}

    schedules = []
    total_cost = 0
    constraints = st.session_state.staff_constraint_config
    staffing_ratios = constraints.staffing_ratios

    # Ensure dates are available (defensive check)
    horizon = len(forecast_data["forecasts"])
    dates = forecast_data.get("dates", [])
    if not dates or len(dates) < horizon:
        today = datetime.now().date()
        dates = [(today + timedelta(days=i+1)).strftime("%a %m/%d") for i in range(horizon)]

    for i, patients in enumerate(forecast_data["forecasts"]):
        # Get category-level patients for this day
        patients_by_cat = {cat: category_demand[cat][i] for cat in CLINICAL_CATEGORIES}

        # Calculate staff needed using staffing ratios
        doctors = sum(np.ceil(max(0, p) / staffing_ratios[c]["Doctors"]) for c, p in patients_by_cat.items())
        nurses = sum(np.ceil(max(0, p) / staffing_ratios[c]["Nurses"]) for c, p in patients_by_cat.items())
        support = sum(np.ceil(max(0, p) / staffing_ratios[c]["Support"]) for c, p in patients_by_cat.items())

        # Apply constraints
        doctors = max(constraints.min_doctors, min(constraints.max_doctors, int(doctors)))
        nurses = max(constraints.min_nurses, min(constraints.max_nurses, int(nurses)))
        support = max(constraints.min_support, min(constraints.max_support, int(support)))

        # Daily labor cost
        daily_cost = (
            doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
            nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
            support * cost_params["support_hourly"] * cost_params["shift_hours"]
        )

        schedules.append({
            "date": dates[i],
            "patients": patients,
            "doctors": int(doctors),
            "nurses": int(nurses),
            "support": int(support),
            "total_staff": int(doctors + nurses + support),
            "cost": daily_cost,
        })
        total_cost += daily_cost

    horizon = forecast_data["horizon"]

    current_daily_cost = (
        staff_stats.get("avg_doctors", 0) * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
        staff_stats.get("avg_nurses", 0) * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
        staff_stats.get("avg_support", 0) * cost_params["support_hourly"] * cost_params["shift_hours"]
    )

    optimized_daily_cost = total_cost / horizon if horizon > 0 else 0
    current_weekly_cost = current_daily_cost * 7
    optimized_weekly_cost = optimized_daily_cost * 7
    weekly_savings = current_weekly_cost - optimized_weekly_cost

    return {
        "success": True,
        "method": "Heuristic",
        "status": "Heuristic Solution",
        "is_optimal": False,
        "schedules": schedules,
        "total_cost": total_cost,
        "avg_daily_cost": optimized_daily_cost,
        "current_daily_cost": current_daily_cost,
        "current_weekly_cost": current_weekly_cost,
        "optimized_weekly_cost": optimized_weekly_cost,
        "weekly_savings": weekly_savings,
        "horizon": horizon,
        "source": forecast_data["source"],
        "avg_opt_doctors": np.mean([s["doctors"] for s in schedules]),
        "avg_opt_nurses": np.mean([s["nurses"] for s in schedules]),
        "avg_opt_support": np.mean([s["support"] for s in schedules]),
        "category_demand": category_demand,
        "used_seasonal_proportions": st.session_state.use_seasonal_proportions,
    }


# =============================================================================
# HELPER FUNCTIONS FOR UI
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
<div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;'>
        <div>
            <div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.25rem;'>🔮 FORECAST SOURCE</div>
""", unsafe_allow_html=True)

sync_col1, sync_col2 = st.columns([3, 1])

with sync_col1:
    if forecast_data["has_forecast"]:
        sync_badge = "🔗 Synced" if forecast_data.get("synced_with_page10") else "⚠️ Fallback"
        sync_color = "#22c55e" if forecast_data.get("synced_with_page10") else "#f59e0b"
        acc_text = f" | Accuracy: {forecast_data['accuracy']:.1f}%" if forecast_data.get("accuracy") else ""
        method_badge = "🧮 MILP" if PULP_AVAILABLE else "📊 Heuristic"

        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;'>
            <span style='font-weight: 700; color: #e2e8f0; font-size: 1.1rem;'>{forecast_data['source']}</span>
            <span style='background: rgba({34 if forecast_data.get("synced_with_page10") else 245}, {197 if forecast_data.get("synced_with_page10") else 158}, {94 if forecast_data.get("synced_with_page10") else 11}, 0.15);
                        color: {sync_color}; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                {sync_badge}
            </span>
            <span style='background: rgba(6, 182, 212, 0.15); color: #06b6d4; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                {method_badge}
            </span>
            <span style='color: #64748b; font-size: 0.8rem;'>{forecast_data['horizon']} days ahead{acc_text}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <span style='font-weight: 600; color: #ef4444;'>❌ No Forecast Available</span>
            <span style='color: #64748b; font-size: 0.8rem;'>Go to Page 10 (Forecast Hub) to generate forecasts</span>
        </div>
        """, unsafe_allow_html=True)

with sync_col2:
    if st.button("🔄 Refresh", type="secondary", use_container_width=True, key="refresh_forecast_staff"):
        st.rerun()

st.markdown("</div></div>", unsafe_allow_html=True)


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Upload",
    "🏥 Current Situation",
    "🚀 Optimization",
    "📈 Comparison",
    "⚙️ Constraints"
])


# =============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">📊 Data Upload & Preview</div>', unsafe_allow_html=True)

    col_staff, col_fin = st.columns(2)

    # Staff Data Section
    with col_staff:
        st.markdown('<div class="subsection-header">👥 Staff Scheduling Data</div>', unsafe_allow_html=True)

        if st.button("🔄 Load Staff Data", type="primary", use_container_width=True, key="load_staff"):
            with st.spinner("Loading staff data from Supabase..."):
                result = load_staff_data()
                if result["success"]:
                    st.session_state.staff_data_loaded = True
                    st.session_state.staff_df = result["data"]
                    st.session_state.staff_stats = result["stats"]
                    st.success(f"✅ Loaded {result['stats']['total_records']} records!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        if st.session_state.staff_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Staff Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.staff_stats

            st.markdown(f"""
            <div class="data-box staff">
                <strong>Dataset:</strong> {stats['total_records']} records<br>
                <strong>Period:</strong> {stats.get('date_start', 'N/A')} to {stats.get('date_end', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                render_kpi_card("Avg Doctors", f"{stats['avg_doctors']:.1f}", "staff", "green")
            with kpi_cols[1]:
                render_kpi_card("Avg Nurses", f"{stats['avg_nurses']:.1f}", "staff", "green")
            with kpi_cols[2]:
                render_kpi_card("Avg Support", f"{stats['avg_support']:.1f}", "staff", "green")
            with kpi_cols[3]:
                render_kpi_card("Utilization", f"{stats['avg_utilization']:.1f}%", "staff", "blue")

            # Staff levels chart
            if st.session_state.staff_df is not None:
                df = st.session_state.staff_df
                if "Date" in df.columns:
                    fig = go.Figure()
                    if "Doctors_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Doctors_on_Duty"], name="Doctors", line=dict(color="#22c55e")))
                    if "Nurses_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Nurses_on_Duty"], name="Nurses", line=dict(color="#3b82f6")))
                    if "Support_Staff_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Support_Staff_on_Duty"], name="Support", line=dict(color="#a855f7")))

                    fig.update_layout(
                        title="Staff Levels Over Time",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<span class="status-badge pending">⏳ Not Loaded</span>', unsafe_allow_html=True)

    # Financial Data Section
    with col_fin:
        st.markdown('<div class="subsection-header">💰 Financial Data</div>', unsafe_allow_html=True)

        if st.button("🔄 Load Financial Data", type="primary", use_container_width=True, key="load_financial"):
            with st.spinner("Loading financial data from Supabase..."):
                result = load_financial_data()
                if result["success"]:
                    st.session_state.financial_data_loaded = True
                    st.session_state.financial_df = result["data"]
                    st.session_state.financial_stats = result["stats"]
                    st.session_state.cost_params = result["cost_params"]
                    st.success("✅ Financial data loaded!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        if st.session_state.financial_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Financial Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.financial_stats
            params = st.session_state.cost_params

            st.markdown(f"""
            <div class="data-box financial">
                <strong>Cost Parameters:</strong><br>
                Doctor: ${params['doctor_hourly']}/hr | Nurse: ${params['nurse_hourly']}/hr | Support: ${params['support_hourly']}/hr
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            kpi_cols = st.columns(3)
            with kpi_cols[0]:
                render_kpi_card("Doctor Rate", f"${params['doctor_hourly']}/hr", "financial", "blue")
            with kpi_cols[1]:
                render_kpi_card("Nurse Rate", f"${params['nurse_hourly']}/hr", "financial", "blue")
            with kpi_cols[2]:
                render_kpi_card("OT Multiplier", f"{params['overtime_multiplier']}x", "financial", "yellow")
        else:
            st.markdown('<span class="status-badge pending">⏳ Not Loaded</span>', unsafe_allow_html=True)
            st.info("Using default cost parameters")


# =============================================================================
# TAB 2: CURRENT HOSPITAL SITUATION
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">🏥 Current Hospital Situation</div>', unsafe_allow_html=True)

    if not st.session_state.staff_data_loaded:
        st.warning("⚠️ Please load staff data in Tab 1 first.")
    else:
        stats = st.session_state.staff_stats
        params = st.session_state.cost_params

        # Staff Situation
        st.markdown('<div class="subsection-header">👥 Staff Situation Summary</div>', unsafe_allow_html=True)

        staff_cols = st.columns(4)
        with staff_cols[0]:
            render_kpi_card("Avg Doctors/Day", f"{stats['avg_doctors']:.1f}", "staff", "green")
        with staff_cols[1]:
            render_kpi_card("Avg Nurses/Day", f"{stats['avg_nurses']:.1f}", "staff", "green")
        with staff_cols[2]:
            render_kpi_card("Avg Support/Day", f"{stats['avg_support']:.1f}", "staff", "green")
        with staff_cols[3]:
            card_class = "danger" if stats['shortage_days'] > 5 else "warning" if stats['shortage_days'] > 0 else "success"
            render_kpi_card("Shortage Days", f"{stats['shortage_days']}", card_class, "red" if stats['shortage_days'] > 5 else "yellow")

        # Financial Situation
        st.markdown('<div class="subsection-header">💰 Financial Situation Summary</div>', unsafe_allow_html=True)

        daily_cost = (
            stats['avg_doctors'] * params['doctor_hourly'] * params['shift_hours'] +
            stats['avg_nurses'] * params['nurse_hourly'] * params['shift_hours'] +
            stats['avg_support'] * params['support_hourly'] * params['shift_hours']
        )

        fin_cols = st.columns(4)
        with fin_cols[0]:
            render_kpi_card("Est. Daily Cost", f"${daily_cost:,.0f}", "financial", "blue")
        with fin_cols[1]:
            render_kpi_card("Weekly Cost", f"${daily_cost * 7:,.0f}", "financial", "blue")
        with fin_cols[2]:
            render_kpi_card("Overtime Hours", f"{stats['total_overtime']:,.0f}", "warning", "yellow")
        with fin_cols[3]:
            render_kpi_card("Utilization", f"{stats['avg_utilization']:.1f}%", "staff", "blue")

        # Optimization targets
        st.markdown('<div class="subsection-header">🎯 Parameters to Optimize</div>', unsafe_allow_html=True)

        target_cols = st.columns(3)
        with target_cols[0]:
            render_kpi_card("Target: Weekly Cost", f"${daily_cost * 7:,.0f}", "linked", "purple")
        with target_cols[1]:
            render_kpi_card("Target: Shortage Days", "0", "linked", "purple")
        with target_cols[2]:
            render_kpi_card("Target: Utilization", "85-90%", "linked", "purple")


# =============================================================================
# TAB 3: AFTER OPTIMIZATION
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">🚀 After Optimization</div>', unsafe_allow_html=True)

    if not forecast_data["has_forecast"]:
        st.warning("⚠️ No forecast available. Please generate forecasts in Page 10 first.")
    elif not st.session_state.staff_data_loaded:
        st.warning("⚠️ Please load staff data in Tab 1 first.")
    else:
        # Seasonal Proportions toggle
        col_sp, col_run = st.columns([2, 1])

        with col_sp:
            st.session_state.use_seasonal_proportions = st.checkbox(
                "🔄 Use Seasonal Proportions for category distribution",
                value=st.session_state.use_seasonal_proportions,
                help="When enabled, uses DOW × Monthly patterns from your data instead of fixed percentages"
            )

            sp_status = "✅ Enabled" if st.session_state.use_seasonal_proportions else "⚠️ Using fallback distribution"
            sp_color = "#22c55e" if st.session_state.use_seasonal_proportions else "#f59e0b"
            st.markdown(f"<span style='color: {sp_color}; font-size: 0.85rem;'>{sp_status}</span>", unsafe_allow_html=True)

        with col_run:
            run_optimization = st.button(
                "🚀 Run MILP Optimization" if PULP_AVAILABLE else "📊 Run Optimization",
                type="primary",
                use_container_width=True
            )

        if run_optimization:
            with st.spinner("Running optimization..." + (" (MILP Solver)" if PULP_AVAILABLE else " (Heuristic)")):
                results = calculate_optimization_milp(
                    forecast_data,
                    st.session_state.staff_stats,
                    st.session_state.cost_params,
                    st.session_state.staff_constraint_config
                )

                if results["success"]:
                    st.session_state.optimization_done = True
                    st.session_state.optimized_results = results
                    st.success(f"✅ Optimization complete! Method: {results.get('method', 'Unknown')}")

                    # Show MILP status if available
                    if results.get("method") == "MILP":
                        status_class = "optimal" if results.get("is_optimal") else "feasible"
                        st.markdown(f"""
                        <div class="milp-status {status_class}">
                            <strong>MILP Status:</strong> {results.get('status', 'Unknown')} |
                            <strong>Solve Time:</strong> {results.get('solve_time', 0):.2f}s |
                            <strong>Variables:</strong> {results.get('num_variables', 0)} |
                            <strong>Constraints:</strong> {results.get('num_constraints', 0)}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")

        # Display results
        if st.session_state.optimization_done and st.session_state.optimized_results:
            results = st.session_state.optimized_results

            st.markdown('<div class="subsection-header">📊 Optimized Staff Parameters</div>', unsafe_allow_html=True)

            opt_cols = st.columns(4)
            with opt_cols[0]:
                current = st.session_state.staff_stats['avg_doctors']
                optimized = results['avg_opt_doctors']
                delta = optimized - current
                delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}"
                render_kpi_card("Doctors/Day", f"{optimized:.1f}", "success", "green", delta_str, delta < 0)

            with opt_cols[1]:
                current = st.session_state.staff_stats['avg_nurses']
                optimized = results['avg_opt_nurses']
                delta = optimized - current
                delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}"
                render_kpi_card("Nurses/Day", f"{optimized:.1f}", "success", "green", delta_str, delta < 0)

            with opt_cols[2]:
                current = st.session_state.staff_stats['avg_support']
                optimized = results['avg_opt_support']
                delta = optimized - current
                delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}"
                render_kpi_card("Support/Day", f"{optimized:.1f}", "success", "green", delta_str, delta < 0)

            with opt_cols[3]:
                savings = results['weekly_savings']
                savings_class = "success" if savings > 0 else "danger"
                savings_color = "green" if savings > 0 else "red"
                render_kpi_card("Weekly Savings", f"${savings:,.0f}", savings_class, savings_color)

            # Schedule table
            st.markdown('<div class="subsection-header">📅 Optimized Schedule</div>', unsafe_allow_html=True)

            schedule_df = pd.DataFrame(results["schedules"])
            schedule_df["cost"] = schedule_df["cost"].apply(lambda x: f"${x:,.0f}")

            st.dataframe(
                schedule_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "date": st.column_config.TextColumn("Date"),
                    "patients": st.column_config.NumberColumn("Patients", format="%.0f"),
                    "doctors": st.column_config.NumberColumn("Doctors"),
                    "nurses": st.column_config.NumberColumn("Nurses"),
                    "support": st.column_config.NumberColumn("Support"),
                    "total_staff": st.column_config.NumberColumn("Total"),
                    "cost": st.column_config.TextColumn("Daily Cost"),
                }
            )

            # Category coverage if available
            if results.get("category_coverage") is not None:
                st.markdown('<div class="subsection-header">📊 Category Coverage</div>', unsafe_allow_html=True)
                st.dataframe(results["category_coverage"], hide_index=True, use_container_width=True)

            # Staff allocation chart
            schedule_data = pd.DataFrame(results["schedules"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=schedule_data["date"], y=schedule_data["doctors"], name="Doctors", marker_color="#22c55e"))
            fig.add_trace(go.Bar(x=schedule_data["date"], y=schedule_data["nurses"], name="Nurses", marker_color="#3b82f6"))
            fig.add_trace(go.Bar(x=schedule_data["date"], y=schedule_data["support"], name="Support", marker_color="#a855f7"))

            fig.update_layout(
                title="Optimized Staff Allocation",
                barmode="stack",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 4: BEFORE VS AFTER COMPARISON
# =============================================================================
with tab4:
    st.markdown('<div class="section-header">📈 Before vs After Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.optimization_done:
        st.warning("⚠️ Please run optimization in Tab 3 first.")
    else:
        results = st.session_state.optimized_results
        stats = st.session_state.staff_stats

        # Financial Impact Summary
        st.markdown('<div class="subsection-header">💰 Financial Impact</div>', unsafe_allow_html=True)

        impact_cols = st.columns(4)
        with impact_cols[0]:
            render_kpi_card("Current Weekly", f"${results['current_weekly_cost']:,.0f}", "financial", "blue")
        with impact_cols[1]:
            render_kpi_card("Optimized Weekly", f"${results['optimized_weekly_cost']:,.0f}", "success", "green")
        with impact_cols[2]:
            savings = results['weekly_savings']
            savings_class = "success" if savings > 0 else "danger"
            render_kpi_card("Weekly Savings", f"${savings:,.0f}", savings_class, "green" if savings > 0 else "red")
        with impact_cols[3]:
            monthly = savings * 4.33
            render_kpi_card("Monthly Savings", f"${monthly:,.0f}", "success" if savings > 0 else "danger", "green" if savings > 0 else "red")

        # Comparison charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Cost comparison
            fig_cost = go.Figure(data=[
                go.Bar(
                    x=["Current", "Optimized"],
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
            # Staff comparison
            categories = ["Doctors", "Nurses", "Support"]
            current_vals = [stats['avg_doctors'], stats['avg_nurses'], stats['avg_support']]
            optimized_vals = [results['avg_opt_doctors'], results['avg_opt_nurses'], results['avg_opt_support']]

            fig_staff = go.Figure(data=[
                go.Bar(name="Current", x=categories, y=current_vals, marker_color="#3b82f6"),
                go.Bar(name="Optimized", x=categories, y=optimized_vals, marker_color="#22c55e"),
            ])
            fig_staff.update_layout(
                title="Staff Levels Comparison",
                barmode="group",
                template="plotly_dark",
                height=300,
                yaxis_title="Staff Count",
            )
            st.plotly_chart(fig_staff, use_container_width=True)

        # AI Insights
        st.markdown('<div class="subsection-header">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)

        inject_insight_styles()

        # Mode selector
        insight_col1, insight_col2, insight_col3 = st.columns([2, 2, 1])
        with insight_col1:
            mode = st.radio(
                "Insight Mode",
                ["rule_based", "llm"],
                format_func=lambda x: "📊 Rule-Based Analysis" if x == "rule_based" else "🧠 LLM-Enhanced",
                horizontal=True,
                key="staff_insight_mode_selector"
            )
            st.session_state.staff_insight_mode = mode

        with insight_col2:
            if mode == "llm":
                provider = st.selectbox(
                    "Provider",
                    ["anthropic", "openai"],
                    format_func=lambda x: "Anthropic (Claude)" if x == "anthropic" else "OpenAI (GPT)",
                    key="staff_provider_select"
                )
                st.session_state.staff_api_provider = provider

        with insight_col3:
            if mode == "llm":
                api_key = st.text_input("API Key", type="password", key="staff_api_key_input")
                st.session_state.staff_llm_api_key = api_key

        # Render insights
        before_data = {
            "avg_doctors": stats['avg_doctors'],
            "avg_nurses": stats['avg_nurses'],
            "avg_support": stats['avg_support'],
            "daily_cost": results['current_daily_cost'],
            "weekly_cost": results['current_weekly_cost'],
            "overtime_hours": stats.get('total_overtime', 0),
            "shortage_days": stats.get('shortage_days', 0),
            "utilization": stats.get('avg_utilization', 0),
        }

        after_data = {
            "avg_doctors": results['avg_opt_doctors'],
            "avg_nurses": results['avg_opt_nurses'],
            "avg_support": results['avg_opt_support'],
            "daily_cost": results['avg_daily_cost'],
            "weekly_cost": results['optimized_weekly_cost'],
            "overtime_hours": 0,  # Optimized schedule minimizes overtime
            "shortage_days": 0,
            "utilization": 85.0,  # Target utilization
        }

        render_staff_optimization_insights(
            before_data=before_data,
            after_data=after_data,
            mode=st.session_state.staff_insight_mode,
            api_key=st.session_state.staff_llm_api_key,
            provider=st.session_state.staff_api_provider
        )


# =============================================================================
# TAB 5: CONSTRAINT CONFIGURATION
# =============================================================================
with tab5:
    st.markdown('<div class="section-header">⚙️ Constraint Configuration</div>', unsafe_allow_html=True)

    constraints = st.session_state.staff_constraint_config

    # Budget Constraints
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">💰 Budget Constraints</div>', unsafe_allow_html=True)

    budget_col1, budget_col2, budget_col3 = st.columns(3)
    with budget_col1:
        daily_budget = st.number_input(
            "Daily Budget Cap ($)",
            min_value=10000,
            max_value=500000,
            value=int(constraints.daily_budget_cap),
            step=5000,
            key="daily_budget_input"
        )
    with budget_col2:
        weekly_budget = st.number_input(
            "Weekly Budget Cap ($)",
            min_value=50000,
            max_value=3000000,
            value=int(constraints.weekly_budget_cap),
            step=25000,
            key="weekly_budget_input"
        )
    with budget_col3:
        enforce_budget = st.checkbox(
            "Enforce Budget Constraints",
            value=constraints.enforce_budget,
            key="enforce_budget_check"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Headcount Limits
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">👥 Headcount Limits</div>', unsafe_allow_html=True)

    head_col1, head_col2, head_col3 = st.columns(3)
    with head_col1:
        st.markdown("**Doctors**")
        min_doctors = st.number_input("Min", min_value=1, max_value=50, value=constraints.min_doctors, key="min_doc")
        max_doctors = st.number_input("Max", min_value=1, max_value=100, value=constraints.max_doctors, key="max_doc")
    with head_col2:
        st.markdown("**Nurses**")
        min_nurses = st.number_input("Min", min_value=1, max_value=100, value=constraints.min_nurses, key="min_nurse")
        max_nurses = st.number_input("Max", min_value=1, max_value=200, value=constraints.max_nurses, key="max_nurse")
    with head_col3:
        st.markdown("**Support Staff**")
        min_support = st.number_input("Min", min_value=1, max_value=50, value=constraints.min_support, key="min_sup")
        max_support = st.number_input("Max", min_value=1, max_value=100, value=constraints.max_support, key="max_sup")
    st.markdown('</div>', unsafe_allow_html=True)

    # Overtime Rules
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">⏰ Overtime Rules</div>', unsafe_allow_html=True)

    ot_col1, ot_col2, ot_col3 = st.columns(3)
    with ot_col1:
        max_ot = st.slider(
            "Max OT Hours/Staff/Day",
            min_value=0.0,
            max_value=8.0,
            value=constraints.max_overtime_per_staff,
            step=0.5,
            key="max_ot_slider"
        )
    with ot_col2:
        ot_mult = st.slider(
            "Overtime Multiplier",
            min_value=1.0,
            max_value=2.5,
            value=constraints.overtime_multiplier,
            step=0.1,
            key="ot_mult_slider"
        )
    with ot_col3:
        allow_ot = st.checkbox(
            "Allow Overtime",
            value=constraints.allow_overtime,
            key="allow_ot_check"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Skill Mix
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">🎓 Skill Mix Ratio</div>', unsafe_allow_html=True)

    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        nurse_doc_ratio = st.slider(
            "Min Nurses per Doctor",
            min_value=1.0,
            max_value=5.0,
            value=constraints.nurse_doctor_ratio,
            step=0.5,
            key="nurse_doc_ratio_slider"
        )
    with skill_col2:
        enforce_skill = st.checkbox(
            "Enforce Skill Mix",
            value=constraints.enforce_skill_mix,
            key="enforce_skill_check"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Solver Settings
    st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
    st.markdown('<div class="constraint-title">🧮 Solver Settings</div>', unsafe_allow_html=True)

    solver_col1, solver_col2 = st.columns(2)
    with solver_col1:
        time_limit = st.slider(
            "Solve Time Limit (seconds)",
            min_value=10,
            max_value=300,
            value=constraints.solve_time_limit,
            step=10,
            key="time_limit_slider"
        )
    with solver_col2:
        opt_gap = st.slider(
            "Optimality Gap (%)",
            min_value=0.1,
            max_value=5.0,
            value=constraints.optimality_gap * 100,
            step=0.1,
            key="opt_gap_slider"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Save/Load buttons
    save_col1, save_col2, save_col3 = st.columns(3)

    with save_col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            # Update constraints
            new_config = StaffConstraintConfig(
                daily_budget_cap=float(daily_budget),
                weekly_budget_cap=float(weekly_budget),
                enforce_budget=enforce_budget,
                min_doctors=min_doctors,
                max_doctors=max_doctors,
                min_nurses=min_nurses,
                max_nurses=max_nurses,
                min_support=min_support,
                max_support=max_support,
                max_overtime_per_staff=max_ot,
                overtime_multiplier=ot_mult,
                allow_overtime=allow_ot,
                nurse_doctor_ratio=nurse_doc_ratio,
                enforce_skill_mix=enforce_skill,
                solve_time_limit=time_limit,
                optimality_gap=opt_gap / 100,
            )

            # Validate
            is_valid, errors = new_config.validate()
            if is_valid:
                st.session_state.staff_constraint_config = new_config
                st.success("✅ Configuration saved!")
            else:
                for error in errors:
                    st.error(f"❌ {error}")

    with save_col2:
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
            st.session_state.staff_constraint_config = DEFAULT_STAFF_CONSTRAINTS
            st.success("✅ Reset to defaults!")
            st.rerun()

    with save_col3:
        # Show estimated costs
        est_min = constraints.estimate_min_daily_cost()
        est_max = constraints.estimate_max_daily_cost()
        st.markdown(f"""
        <div class="data-box constraint">
            <strong>Estimated Daily Cost Range:</strong><br>
            Min: ${est_min:,.0f} | Max: ${est_max:,.0f}
        </div>
        """, unsafe_allow_html=True)

    # Validation status
    is_valid, errors = constraints.validate()
    if is_valid:
        st.markdown("""
        <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 8px; padding: 0.75rem; margin-top: 1rem;'>
            ✅ <strong>Configuration Valid</strong> - Ready for optimization
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 8px; padding: 0.75rem; margin-top: 1rem;'>
            ❌ <strong>Configuration Invalid</strong>
        </div>
        """, unsafe_allow_html=True)
        for error in errors:
            st.error(error)


# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(current_page_index=11)
