# =============================================================================
# app_core/state/unified_dashboard_state.py
# Unified State Management for the Premium Command Center Dashboard
# =============================================================================
"""
Centralized state management for the unified dashboard.
Provides typed dataclasses and extraction functions for all dashboard data.
"""
from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


# =============================================================================
# STATE DATACLASSES
# =============================================================================

@dataclass
class ForecastState:
    """Unified forecast state extracted from all model sources."""
    has_forecast: bool = False
    source: str = ""                      # Model name (e.g., "XGBoost", "ARIMA")
    model_type: str = ""                  # "Statistical" or "ML"
    forecasts: List[float] = field(default_factory=list)  # 7-day values
    lower_bounds: List[float] = field(default_factory=list)
    upper_bounds: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)        # Date labels
    horizon: int = 0
    total_week: float = 0
    avg_daily: float = 0
    peak_day: int = 0                     # Index of peak day (0-6)
    peak_patients: float = 0
    trend: str = "stable"                 # "up", "down", "stable"
    confidence: str = "Unknown"           # "High", "Medium", "Low"
    avg_accuracy: float = 0
    avg_mape: float = 0
    synced_with_page10: bool = False
    timestamp: Optional[str] = None


@dataclass
class StaffState:
    """Unified staff data state."""
    data_loaded: bool = False
    optimization_done: bool = False
    # Current metrics
    avg_doctors: float = 0
    avg_nurses: float = 0
    avg_support: float = 0
    avg_utilization: float = 0
    total_overtime: float = 0
    shortage_days: int = 0
    daily_labor_cost: float = 0
    weekly_labor_cost: float = 0
    monthly_labor_cost: float = 0
    # Optimized values (after optimization)
    opt_doctors: float = 0
    opt_nurses: float = 0
    opt_support: float = 0
    opt_utilization: float = 95.0  # Target
    opt_weekly_cost: float = 0
    weekly_savings: float = 0
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class InventoryState:
    """Unified inventory data state."""
    data_loaded: bool = False
    optimization_done: bool = False
    # Current metrics
    avg_gloves_usage: float = 0
    avg_ppe_usage: float = 0
    avg_medication_usage: float = 0
    current_gloves_level: float = 0
    current_ppe_level: float = 0
    current_medication_level: float = 0
    avg_stockout_risk: float = 0
    service_level: float = 100
    restock_events: int = 0
    daily_inventory_cost: float = 0
    weekly_inventory_cost: float = 0
    # Item status counts
    items_ok: int = 0
    items_low: int = 0
    items_critical: int = 0
    # Optimized values
    opt_weekly_cost: float = 0
    weekly_savings: float = 0
    opt_service_level: float = 100


@dataclass
class ActionItem:
    """Single action item for the Actions tab."""
    priority: str                         # "urgent", "important", "normal"
    icon: str
    title: str
    description: str
    category: str                         # "Staff", "Inventory", "System", "Planning"
    page_ref: Optional[str] = None        # Reference to related page
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UnifiedDashboardState:
    """Master state container for the unified dashboard."""
    forecast: ForecastState = field(default_factory=ForecastState)
    staff: StaffState = field(default_factory=StaffState)
    inventory: InventoryState = field(default_factory=InventoryState)
    actions: List[ActionItem] = field(default_factory=list)

    # Dashboard metadata
    active_tab: int = 0
    last_refresh: Optional[datetime] = None

    # Financial aggregates
    total_weekly_savings: float = 0
    total_monthly_savings: float = 0
    annual_savings: float = 0

    # System health (0-100)
    system_health_score: float = 0


# =============================================================================
# STATE INITIALIZATION
# =============================================================================

def initialize_dashboard_state() -> None:
    """Initialize unified dashboard state in session."""
    if "unified_dashboard" not in st.session_state:
        st.session_state.unified_dashboard = UnifiedDashboardState()


def get_dashboard_state() -> UnifiedDashboardState:
    """Get current dashboard state."""
    initialize_dashboard_state()
    return st.session_state.unified_dashboard


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_forecast_state() -> ForecastState:
    """
    Extract forecast data from session state with 4-tier priority.

    Priority Order:
    1. forecast_hub_demand - Primary from Patient Forecast page
    2. active_forecast - Alternative from Forecast Hub
    3. ML models - Direct from Train Models results
    4. ARIMA/SARIMAX - Statistical model fallback
    """
    state = ForecastState()

    # Priority 1: forecast_hub_demand (from Page 10)
    hub_demand = st.session_state.get("forecast_hub_demand")
    if hub_demand and isinstance(hub_demand, dict):
        forecasts = hub_demand.get("forecast", [])
        if forecasts and len(forecasts) > 0:
            state.has_forecast = True
            state.source = hub_demand.get("model", "Forecast Hub")
            state.model_type = hub_demand.get("model_type", "Unknown")
            state.forecasts = [max(0, float(f)) for f in forecasts[:7]]
            state.lower_bounds = hub_demand.get("lower_bounds", [])
            state.upper_bounds = hub_demand.get("upper_bounds", [])
            state.dates = hub_demand.get("dates", [])
            state.horizon = len(state.forecasts)
            state.total_week = sum(state.forecasts)
            state.avg_daily = state.total_week / max(1, len(state.forecasts))
            state.peak_day = int(np.argmax(state.forecasts)) if state.forecasts else 0
            state.peak_patients = max(state.forecasts) if state.forecasts else 0
            state.avg_accuracy = hub_demand.get("avg_accuracy", 0)
            state.avg_mape = hub_demand.get("avg_mape", 0)
            state.synced_with_page10 = True
            state.timestamp = hub_demand.get("timestamp")

            # Calculate trend
            if len(state.forecasts) >= 2:
                first_half = sum(state.forecasts[:3])
                second_half = sum(state.forecasts[3:])
                if second_half > first_half * 1.1:
                    state.trend = "up"
                elif second_half < first_half * 0.9:
                    state.trend = "down"
                else:
                    state.trend = "stable"

            # Set confidence based on accuracy
            if state.avg_accuracy >= 90:
                state.confidence = "High"
            elif state.avg_accuracy >= 75:
                state.confidence = "Medium"
            else:
                state.confidence = "Low"

            return state

    # Priority 2: active_forecast
    active = st.session_state.get("active_forecast")
    if active and isinstance(active, dict):
        forecasts = active.get("forecast", active.get("forecasts", []))
        if forecasts:
            state.has_forecast = True
            state.source = active.get("model", "Active Forecast")
            state.model_type = active.get("model_type", "Unknown")
            state.forecasts = [max(0, float(f)) for f in forecasts[:7]]
            state.horizon = len(state.forecasts)
            state.total_week = sum(state.forecasts)
            state.avg_daily = state.total_week / max(1, len(state.forecasts))
            state.peak_day = int(np.argmax(state.forecasts)) if state.forecasts else 0
            state.peak_patients = max(state.forecasts) if state.forecasts else 0
            return state

    # Priority 3: ML model results (using lowercase keys)
    for model_key in ["ml_mh_results_xgboost", "ml_mh_results_lstm", "ml_mh_results_ann"]:
        ml_results = st.session_state.get(model_key)
        if ml_results and isinstance(ml_results, dict):
            per_h = ml_results.get("per_h", {})
            if per_h:
                # Extract first horizon forecast
                forecasts = []
                for h in range(1, 8):
                    h_data = per_h.get(h, per_h.get(str(h), {}))
                    if h_data:
                        pred = h_data.get("forecast", h_data.get("predictions", h_data.get("y_pred")))
                        if pred is not None:
                            if hasattr(pred, '__len__') and len(pred) > 0:
                                forecasts.append(float(pred[-1]))
                            else:
                                forecasts.append(float(pred))

                if forecasts:
                    state.has_forecast = True
                    state.source = model_key.replace("ml_mh_results_", "")
                    state.model_type = "ML"
                    state.forecasts = forecasts[:7]
                    state.horizon = len(state.forecasts)
                    state.total_week = sum(state.forecasts)
                    state.avg_daily = state.total_week / max(1, len(state.forecasts))
                    state.peak_day = int(np.argmax(state.forecasts)) if state.forecasts else 0
                    state.peak_patients = max(state.forecasts) if state.forecasts else 0
                    return state

    # Priority 4: ARIMA/SARIMAX
    for stat_key in ["arima_mh_results", "sarimax_results"]:
        stat_results = st.session_state.get(stat_key)
        if stat_results and isinstance(stat_results, dict):
            per_h = stat_results.get("per_h", {})
            if per_h:
                forecasts = []
                for h in range(1, 8):
                    h_data = per_h.get(h, per_h.get(str(h), {}))
                    if h_data:
                        pred = h_data.get("forecast", h_data.get("predictions"))
                        if pred is not None:
                            if hasattr(pred, '__len__') and len(pred) > 0:
                                forecasts.append(float(pred[-1]))
                            else:
                                forecasts.append(float(pred))

                if forecasts:
                    state.has_forecast = True
                    state.source = "ARIMA" if "arima" in stat_key else "SARIMAX"
                    state.model_type = "Statistical"
                    state.forecasts = forecasts[:7]
                    state.horizon = len(state.forecasts)
                    state.total_week = sum(state.forecasts)
                    state.avg_daily = state.total_week / max(1, len(state.forecasts))
                    state.peak_day = int(np.argmax(state.forecasts)) if state.forecasts else 0
                    state.peak_patients = max(state.forecasts) if state.forecasts else 0
                    return state

    return state


def extract_staff_state() -> StaffState:
    """Extract staff data from session state."""
    state = StaffState()

    state.data_loaded = st.session_state.get("staff_data_loaded", False)
    state.optimization_done = st.session_state.get("optimization_done", False)

    if state.data_loaded:
        staff_stats = st.session_state.get("staff_stats", {})
        cost_params = st.session_state.get("cost_params", {})

        # Current metrics
        state.avg_doctors = max(0, staff_stats.get("avg_doctors", 0))
        state.avg_nurses = max(0, staff_stats.get("avg_nurses", 0))
        state.avg_support = max(0, staff_stats.get("avg_support", 0))
        state.avg_utilization = max(0, min(100, staff_stats.get("avg_utilization", 0)))
        state.total_overtime = max(0, staff_stats.get("total_overtime", 0))
        state.shortage_days = max(0, staff_stats.get("shortage_days", 0))

        # Calculate costs
        doctor_rate = cost_params.get("doctor_hourly", 150)
        nurse_rate = cost_params.get("nurse_hourly", 50)
        support_rate = cost_params.get("support_hourly", 25)
        shift_hours = cost_params.get("shift_hours", 8)

        state.daily_labor_cost = (
            state.avg_doctors * doctor_rate * shift_hours +
            state.avg_nurses * nurse_rate * shift_hours +
            state.avg_support * support_rate * shift_hours
        )
        state.weekly_labor_cost = state.daily_labor_cost * 7
        state.monthly_labor_cost = state.daily_labor_cost * 30

        # Optimized values (if optimization was run)
        if state.optimization_done:
            opt_results = st.session_state.get("optimized_results", {})
            state.opt_doctors = opt_results.get("avg_doctors", state.avg_doctors)
            state.opt_nurses = opt_results.get("avg_nurses", state.avg_nurses)
            state.opt_support = opt_results.get("avg_support", state.avg_support)
            state.opt_weekly_cost = opt_results.get("total_weekly_cost", state.weekly_labor_cost)
            state.weekly_savings = max(0, state.weekly_labor_cost - state.opt_weekly_cost)

        # Generate recommendations
        if state.avg_utilization < 70:
            state.recommendations.append("Staff utilization below 70% - consider reducing headcount")
        if state.shortage_days > 2:
            state.recommendations.append(f"{state.shortage_days} shortage days detected - review staffing levels")
        if state.total_overtime > 40:
            state.recommendations.append("High overtime hours - consider hiring additional staff")

    return state


def extract_inventory_state() -> InventoryState:
    """Extract inventory data from session state."""
    state = InventoryState()

    state.data_loaded = st.session_state.get("inventory_data_loaded", False)
    state.optimization_done = st.session_state.get("inv_optimization_done", False)

    if state.data_loaded:
        inv_stats = st.session_state.get("inventory_stats", {})
        cost_params = st.session_state.get("inv_cost_params", {})

        # Usage metrics
        state.avg_gloves_usage = max(0, inv_stats.get("avg_gloves_usage", 0))
        state.avg_ppe_usage = max(0, inv_stats.get("avg_ppe_usage", 0))
        state.avg_medication_usage = max(0, inv_stats.get("avg_medication_usage", 0))

        # Current levels
        state.current_gloves_level = max(0, inv_stats.get("current_gloves_level", 0))
        state.current_ppe_level = max(0, inv_stats.get("current_ppe_level", 0))
        state.current_medication_level = max(0, inv_stats.get("current_medication_level", 0))

        # Risk metrics
        state.avg_stockout_risk = max(0, min(100, inv_stats.get("avg_stockout_risk", 0)))
        state.service_level = 100 - state.avg_stockout_risk
        state.restock_events = max(0, inv_stats.get("restock_events", 0))

        # Calculate costs
        gloves_cost = cost_params.get("gloves_unit_cost", 15)
        ppe_cost = cost_params.get("ppe_unit_cost", 45)
        med_cost = cost_params.get("medication_unit_cost", 8)

        state.daily_inventory_cost = (
            state.avg_gloves_usage * gloves_cost +
            state.avg_ppe_usage * ppe_cost +
            state.avg_medication_usage * med_cost
        )
        state.weekly_inventory_cost = state.daily_inventory_cost * 7

        # Item status
        # Items OK: all three above threshold
        # Items Low: usage > 80% of level
        # Items Critical: usage > 95% of level
        for usage, level, name in [
            (state.avg_gloves_usage, state.current_gloves_level, "Gloves"),
            (state.avg_ppe_usage, state.current_ppe_level, "PPE"),
            (state.avg_medication_usage, state.current_medication_level, "Meds"),
        ]:
            if level > 0:
                ratio = usage / level
                if ratio > 0.95:
                    state.items_critical += 1
                elif ratio > 0.8:
                    state.items_low += 1
                else:
                    state.items_ok += 1

        # Optimized values
        if state.optimization_done:
            opt_results = st.session_state.get("inv_optimized_results", {})
            state.opt_weekly_cost = opt_results.get("total_weekly_cost", state.weekly_inventory_cost)
            state.weekly_savings = max(0, state.weekly_inventory_cost - state.opt_weekly_cost)
            state.opt_service_level = 100  # Target after optimization

    return state


def generate_action_items(
    forecast: ForecastState,
    staff: StaffState,
    inventory: InventoryState
) -> List[ActionItem]:
    """Generate prioritized action items based on current state."""
    actions: List[ActionItem] = []

    # System actions (always check)
    if not forecast.has_forecast:
        actions.append(ActionItem(
            priority="urgent",
            icon="🔮",
            title="No Forecast Available",
            description="Train a model in the Train Models page to generate patient forecasts.",
            category="System",
            page_ref="pages/08_Train_Models.py"
        ))

    if not staff.data_loaded:
        actions.append(ActionItem(
            priority="important",
            icon="👥",
            title="Load Staff Data",
            description="Staff data not loaded. Click to load from database.",
            category="Staff"
        ))

    if not inventory.data_loaded:
        actions.append(ActionItem(
            priority="important",
            icon="📦",
            title="Load Inventory Data",
            description="Inventory data not loaded. Click to load from database.",
            category="Inventory"
        ))

    # Forecast-based actions
    if forecast.has_forecast:
        if forecast.peak_patients > forecast.avg_daily * 1.3:
            peak_day_name = forecast.dates[forecast.peak_day] if forecast.dates else f"Day {forecast.peak_day + 1}"
            actions.append(ActionItem(
                priority="important",
                icon="📈",
                title=f"High Patient Volume Expected",
                description=f"Peak of {int(forecast.peak_patients)} patients predicted for {peak_day_name}. Review staffing.",
                category="Planning"
            ))

        if forecast.trend == "up":
            actions.append(ActionItem(
                priority="normal",
                icon="📊",
                title="Upward Trend Detected",
                description="Patient volume trending upward. Consider proactive resource allocation.",
                category="Planning"
            ))

    # Staff-based actions
    if staff.data_loaded:
        if staff.shortage_days > 0:
            actions.append(ActionItem(
                priority="urgent",
                icon="⚠️",
                title=f"{staff.shortage_days} Staff Shortage Days",
                description="Historical staffing shortages detected. Review and optimize schedules.",
                category="Staff",
                page_ref="pages/11_Staff_Planner.py"
            ))

        if staff.avg_utilization < 70:
            actions.append(ActionItem(
                priority="important",
                icon="📉",
                title="Low Staff Utilization",
                description=f"Current utilization at {staff.avg_utilization:.1f}%. Consider optimization.",
                category="Staff"
            ))

        if staff.total_overtime > 40:
            actions.append(ActionItem(
                priority="important",
                icon="⏰",
                title="High Overtime Hours",
                description=f"{staff.total_overtime:.0f} overtime hours recorded. Review workload distribution.",
                category="Staff"
            ))

        if not staff.optimization_done:
            actions.append(ActionItem(
                priority="normal",
                icon="🚀",
                title="Run Staff Optimization",
                description="Optimize staffing based on forecast to reduce costs.",
                category="Staff"
            ))

    # Inventory-based actions
    if inventory.data_loaded:
        if inventory.items_critical > 0:
            actions.append(ActionItem(
                priority="urgent",
                icon="🔴",
                title=f"{inventory.items_critical} Items at Critical Level",
                description="Immediate restock required to prevent stockouts.",
                category="Inventory",
                page_ref="pages/12_Supply_Planner.py"
            ))

        if inventory.items_low > 0:
            actions.append(ActionItem(
                priority="important",
                icon="🟡",
                title=f"{inventory.items_low} Items Running Low",
                description="Review inventory levels and plan restocking.",
                category="Inventory"
            ))

        if inventory.avg_stockout_risk > 10:
            actions.append(ActionItem(
                priority="important",
                icon="📦",
                title="Elevated Stockout Risk",
                description=f"Stockout risk at {inventory.avg_stockout_risk:.1f}%. Review safety stock levels.",
                category="Inventory"
            ))

        if not inventory.optimization_done:
            actions.append(ActionItem(
                priority="normal",
                icon="🚀",
                title="Run Supply Optimization",
                description="Optimize inventory ordering based on forecast demand.",
                category="Inventory"
            ))

    # Sort by priority
    priority_order = {"urgent": 0, "important": 1, "normal": 2}
    actions.sort(key=lambda x: priority_order.get(x.priority, 3))

    return actions


def calculate_system_health(
    forecast: ForecastState,
    staff: StaffState,
    inventory: InventoryState
) -> float:
    """Calculate overall system health score (0-100)."""
    scores = []
    weights = []

    # Forecast health (20% weight)
    if forecast.has_forecast:
        forecast_score = min(100, forecast.avg_accuracy) if forecast.avg_accuracy > 0 else 50
        scores.append(forecast_score)
        weights.append(0.2)

    # Staff health (40% weight)
    if staff.data_loaded:
        # Utilization score (target: 85-95%)
        if 85 <= staff.avg_utilization <= 95:
            util_score = 100
        elif staff.avg_utilization < 85:
            util_score = max(0, 100 - (85 - staff.avg_utilization) * 2)
        else:
            util_score = max(0, 100 - (staff.avg_utilization - 95) * 3)

        # Shortage penalty
        shortage_penalty = min(30, staff.shortage_days * 10)

        staff_score = max(0, util_score - shortage_penalty)
        scores.append(staff_score)
        weights.append(0.4)

    # Inventory health (40% weight)
    if inventory.data_loaded:
        inv_score = inventory.service_level  # 0-100

        # Critical items penalty
        if inventory.items_critical > 0:
            inv_score = max(0, inv_score - 20)
        if inventory.items_low > 0:
            inv_score = max(0, inv_score - 10)

        scores.append(inv_score)
        weights.append(0.4)

    # Calculate weighted average
    if scores and weights:
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight

    return 0


# =============================================================================
# MAIN REFRESH FUNCTION
# =============================================================================

def refresh_dashboard_state() -> UnifiedDashboardState:
    """Refresh all data from source pages and return updated state."""
    state = get_dashboard_state()

    # Extract from existing session state keys
    state.forecast = extract_forecast_state()
    state.staff = extract_staff_state()
    state.inventory = extract_inventory_state()
    state.actions = generate_action_items(state.forecast, state.staff, state.inventory)

    # Calculate financial aggregates
    state.total_weekly_savings = state.staff.weekly_savings + state.inventory.weekly_savings
    state.total_monthly_savings = state.total_weekly_savings * 4.33  # Average weeks per month
    state.annual_savings = state.total_monthly_savings * 12

    # Calculate system health
    state.system_health_score = calculate_system_health(
        state.forecast, state.staff, state.inventory
    )

    # Update timestamp
    state.last_refresh = datetime.now()

    # Save back to session state
    st.session_state.unified_dashboard = state

    return state


def sync_tabs_on_update(updated_source: str) -> None:
    """
    Synchronize data across tabs when one source updates.

    Args:
        updated_source: "forecast", "staff", or "inventory"
    """
    # Simply refresh the entire state
    refresh_dashboard_state()
