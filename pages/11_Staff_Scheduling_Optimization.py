# =============================================================================
# 11_Staff_Scheduling_Optimization.py
# Staff Scheduling Optimization with MILP Solver
# Implements Deterministic MILP with Clinical Category Integration
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from typing import Optional, Dict, List

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.data.staff_scheduling_service import (
    StaffSchedulingService,
    fetch_staff_scheduling_data,
    check_supabase_connection
)

# Import financial service for cost parameters
try:
    from app_core.data.financial_service import (
        FinancialService,
        get_financial_service,
        check_financial_connection
    )
    FINANCIAL_SERVICE_AVAILABLE = True
except ImportError:
    FINANCIAL_SERVICE_AVAILABLE = False

# Import optimization module
try:
    from app_core.optimization import (
        CostParameters,
        StaffingRatios,
        DEFAULT_COST_PARAMS,
        DEFAULT_STAFFING_RATIOS,
        CLINICAL_CATEGORIES,
        CATEGORY_PRIORITIES,
        StaffSchedulingMILP,
        OptimizationResult,
        solve_staff_scheduling,
        SolutionAnalyzer,
        analyze_historical_costs,
        compare_before_after,
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    OPTIMIZATION_ERROR = str(e)

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Staff Scheduling Optimization - HealthForecast AI",
    page_icon="👥",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# CUSTOM CSS FOR OPTIMIZATION PAGE
# =============================================================================
st.markdown("""
<style>
/* Optimization cards */
.opt-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.opt-card-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.opt-metric {
    text-align: center;
    padding: 0.75rem;
}

.opt-metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #22c55e;
}

.opt-metric-value.negative {
    color: #ef4444;
}

.opt-metric-value.neutral {
    color: #60a5fa;
}

.opt-metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Comparison table */
.comparison-improved {
    color: #22c55e;
    font-weight: 600;
}

.comparison-worse {
    color: #ef4444;
    font-weight: 600;
}

/* Staff metric cards */
.staff-metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.staff-metric-card:hover {
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
}

.staff-metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 0.25rem;
}

.staff-metric-label {
    font-size: 0.875rem;
    color: #94a3b8;
}

.connection-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-connected {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-disconnected {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Category badges */
.category-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.125rem;
}

.category-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.category-medium { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
.category-low { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>👥</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Staff Scheduling Optimization</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Deterministic MILP Optimization with Clinical Category Integration<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Minimize costs while meeting patient demand across all clinical categories</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check if optimization module is available
if not OPTIMIZATION_AVAILABLE:
    st.error(f"Optimization module not available: {OPTIMIZATION_ERROR}")
    st.info("Please install PuLP: `pip install pulp`")
    st.stop()

# =============================================================================
# SUPABASE CONNECTION STATUS
# =============================================================================
is_connected = check_supabase_connection()

col1, col2 = st.columns([3, 1])
with col2:
    if is_connected:
        st.markdown(
            '<div class="connection-status status-connected">🟢 Supabase Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="connection-status status-disconnected">🔴 Supabase Disconnected</div>',
            unsafe_allow_html=True
        )

# =============================================================================
# TABS FOR ORGANIZATION
# =============================================================================
tab_data, tab_config, tab_optimize, tab_results = st.tabs([
    "📊 Data Source",
    "⚙️ Configuration",
    "🚀 Optimize",
    "📈 Results & Comparison"
])

# Initialize session state
if "staff_df" not in st.session_state:
    st.session_state["staff_df"] = None
if "optimization_result" not in st.session_state:
    st.session_state["optimization_result"] = None
if "historical_analysis" not in st.session_state:
    st.session_state["historical_analysis"] = None
if "cost_params" not in st.session_state:
    st.session_state["cost_params"] = DEFAULT_COST_PARAMS
if "staffing_ratios" not in st.session_state:
    st.session_state["staffing_ratios"] = DEFAULT_STAFFING_RATIOS


# =============================================================================
# UNIFIED FORECAST DETECTOR
# Detects forecast results from ALL trained models:
# - Forecast Hub (10): Generated forecasts using rolling/walk-forward approach
# - Modeling Hub (08): XGBoost, LSTM, ANN, LightGBM, RandomForest, etc.
# - Benchmarks (05): ARIMA, SARIMAX (single & multi-target)
# - Dashboard (01): Quick forecasts
# =============================================================================
def detect_forecast_sources() -> list:
    """
    Dynamically scan session state for ALL available forecast sources.
    This function finds any trained model results regardless of model name.

    Returns:
        List of dicts with source info and data
    """
    sources = []
    found_keys = set()  # Track found keys to avoid duplicates

    # -------------------------------------------------------------------------
    # FORECAST HUB (10_Forecast.py) - PRIORITY SOURCE
    # These are actual generated forecasts ready for consumption
    # -------------------------------------------------------------------------
    if "forecast_hub_demand" not in found_keys:
        forecast_hub = st.session_state.get("forecast_hub_demand")
        if forecast_hub is not None:
            model_name = forecast_hub.get("model", "Unknown")
            sources.append({
                "name": f"Forecast Hub: {model_name}",
                "key": "forecast_hub_demand",
                "data": forecast_hub,
                "type": "forecast_hub"
            })
            found_keys.add("forecast_hub_demand")

    # Also check for forecast_hub_results (full results)
    if "forecast_hub_results" not in found_keys:
        forecast_hub_full = st.session_state.get("forecast_hub_results")
        if forecast_hub_full is not None and "forecast_hub_demand" not in found_keys:
            model_name = forecast_hub_full.get("model_name", "Unknown")
            sources.append({
                "name": f"Forecast Hub: {model_name} (Full)",
                "key": "forecast_hub_results",
                "data": forecast_hub_full,
                "type": "forecast_hub_full"
            })
            found_keys.add("forecast_hub_results")

    # -------------------------------------------------------------------------
    # DYNAMIC SCAN: Find ALL ml_mh_results_* keys (any model name)
    # -------------------------------------------------------------------------
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_") and key not in found_keys:
            model_name = key.replace("ml_mh_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                sources.append({
                    "name": f"ML: {model_name}",
                    "key": key,
                    "data": data,
                    "type": "ml"
                })
                found_keys.add(key)

    # -------------------------------------------------------------------------
    # DYNAMIC SCAN: Find ALL opt_results_* keys (optimized models)
    # -------------------------------------------------------------------------
    for key in st.session_state.keys():
        if key.startswith("opt_results_") and key not in found_keys:
            model_name = key.replace("opt_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                sources.append({
                    "name": f"ML Optimized: {model_name}",
                    "key": key,
                    "data": data,
                    "type": "ml_optimized"
                })
                found_keys.add(key)

    # -------------------------------------------------------------------------
    # DYNAMIC SCAN: Find ALL backtest_results_* keys
    # -------------------------------------------------------------------------
    for key in st.session_state.keys():
        if key.startswith("backtest_results_") and key not in found_keys:
            model_name = key.replace("backtest_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                sources.append({
                    "name": f"ML Backtest: {model_name}",
                    "key": key,
                    "data": data,
                    "type": "ml_backtest"
                })
                found_keys.add(key)

    # -------------------------------------------------------------------------
    # ML MODELS - Standard keys (from 08_Modeling_Hub.py)
    # -------------------------------------------------------------------------

    # Single ML model results (generic)
    if "ml_mh_results" not in found_keys:
        ml_results = st.session_state.get("ml_mh_results")
        if ml_results is not None:
            sources.append({
                "name": "ML Model (Latest)",
                "key": "ml_mh_results",
                "data": ml_results,
                "type": "ml"
            })
            found_keys.add("ml_mh_results")

    # All models combined results
    if "opt_all_models_results" not in found_keys:
        all_models = st.session_state.get("opt_all_models_results")
        if all_models is not None:
            sources.append({
                "name": "ML: All Models Combined",
                "key": "opt_all_models_results",
                "data": all_models,
                "type": "ml_all"
            })
            found_keys.add("opt_all_models_results")

    # -------------------------------------------------------------------------
    # STATISTICAL MODELS (from 05_Benchmarks.py)
    # -------------------------------------------------------------------------

    # SARIMAX single target
    if "sarimax_results" not in found_keys:
        sarimax_results = st.session_state.get("sarimax_results")
        if sarimax_results is not None:
            sources.append({
                "name": "SARIMAX (Single Target)",
                "key": "sarimax_results",
                "data": sarimax_results,
                "type": "sarimax"
            })
            found_keys.add("sarimax_results")

    # ARIMA single target
    if "arima_mh_results" not in found_keys:
        arima_results = st.session_state.get("arima_mh_results")
        if arima_results is not None:
            sources.append({
                "name": "ARIMA (Single Target)",
                "key": "arima_mh_results",
                "data": arima_results,
                "type": "arima"
            })
            found_keys.add("arima_mh_results")

    # SARIMAX multi-target
    if "sarimax_multi_target_results" not in found_keys:
        sarimax_multi = st.session_state.get("sarimax_multi_target_results")
        if sarimax_multi is not None:
            sources.append({
                "name": "SARIMAX (Multi-Target)",
                "key": "sarimax_multi_target_results",
                "data": sarimax_multi,
                "type": "sarimax_multi"
            })
            found_keys.add("sarimax_multi_target_results")

    # ARIMA multi-target
    if "arima_multi_target_results" not in found_keys:
        arima_multi = st.session_state.get("arima_multi_target_results")
        if arima_multi is not None:
            sources.append({
                "name": "ARIMA (Multi-Target)",
                "key": "arima_multi_target_results",
                "data": arima_multi,
                "type": "arima_multi"
            })
            found_keys.add("arima_multi_target_results")

    # -------------------------------------------------------------------------
    # DASHBOARD FORECASTS (from 01_Dashboard.py)
    # -------------------------------------------------------------------------

    # Dashboard forecast results
    if "dashboard_forecast" not in found_keys:
        dash_forecast = st.session_state.get("dashboard_forecast")
        if dash_forecast is not None:
            sources.append({
                "name": "Dashboard Forecast",
                "key": "dashboard_forecast",
                "data": dash_forecast,
                "type": "dashboard"
            })
            found_keys.add("dashboard_forecast")

    # -------------------------------------------------------------------------
    # LEGACY KEYS (backwards compatibility)
    # -------------------------------------------------------------------------
    if "forecast_results" not in found_keys:
        legacy_results = st.session_state.get("forecast_results")
        if legacy_results is not None:
            sources.append({
                "name": "Forecast Results (Legacy)",
                "key": "forecast_results",
                "data": legacy_results,
                "type": "legacy"
            })
            found_keys.add("forecast_results")

    if "multi_target_results" not in found_keys:
        multi_target = st.session_state.get("multi_target_results")
        if multi_target is not None:
            sources.append({
                "name": "Multi-Target Results (Legacy)",
                "key": "multi_target_results",
                "data": multi_target,
                "type": "legacy_multi"
            })
            found_keys.add("multi_target_results")

    # -------------------------------------------------------------------------
    # CATCH-ALL: Scan for any other forecast-related keys
    # -------------------------------------------------------------------------
    forecast_patterns = ["forecast", "prediction", "model_results"]
    for key in st.session_state.keys():
        if key not in found_keys:
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in forecast_patterns):
                data = st.session_state.get(key)
                # Only add if it looks like forecast data (dict with results)
                if isinstance(data, dict) and len(data) > 0:
                    sources.append({
                        "name": f"Other: {key}",
                        "key": key,
                        "data": data,
                        "type": "other"
                    })
                    found_keys.add(key)

    return sources


def extract_demand_from_source(source: dict, horizon: int = 7) -> tuple:
    """
    Extract total demand forecast and category proportions from a source.

    Returns:
        (total_demand_list, category_proportions_dict, source_info_str)
    """
    data = source["data"]
    source_type = source["type"]
    source_name = source["name"]

    total_demand = None
    category_proportions = None
    info = ""

    try:
        # ---------------------------------------------------------------------
        # FORECAST HUB (10_Forecast.py) - PRIORITY SOURCE
        # Standardized format with direct forecast values
        # ---------------------------------------------------------------------
        if source_type == "forecast_hub":
            # forecast_hub_demand format: {model, forecast, dates, targets}
            if isinstance(data, dict):
                if "forecast" in data:
                    total_demand = list(data["forecast"])[:horizon]
                    info = f"Using forecast from {source_name}"

                # Try to get category proportions from targets
                if "targets" in data:
                    targets = data["targets"]
                    # Look for clinical category targets
                    for cat in CLINICAL_CATEGORIES:
                        for target_name, values in targets.items():
                            if cat.lower() in target_name.lower():
                                if category_proportions is None:
                                    category_proportions = {}
                                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                                    category_proportions[cat] = np.mean(values)

        elif source_type == "forecast_hub_full":
            # forecast_hub_results format: {model_name, forecast_values, target_forecasts, ...}
            if isinstance(data, dict):
                if "forecast_values" in data:
                    total_demand = list(data["forecast_values"])[:horizon]
                    info = f"Using forecast from {source_name}"

                # Try to get category proportions from target_forecasts
                if "target_forecasts" in data:
                    targets = data["target_forecasts"]
                    for cat in CLINICAL_CATEGORIES:
                        for target_name, values in targets.items():
                            if cat.lower() in target_name.lower():
                                if category_proportions is None:
                                    category_proportions = {}
                                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                                    category_proportions[cat] = np.mean(values)

        # ---------------------------------------------------------------------
        # ML Models (typically have 'predictions' or 'forecast' keys)
        # Includes: ml, ml_optimized, ml_all, ml_backtest, dashboard, other
        # ---------------------------------------------------------------------
        elif source_type in ["ml", "ml_optimized", "ml_all", "ml_backtest", "dashboard", "other"]:
            # ML results structure varies, try common patterns
            if isinstance(data, dict):
                # Check for predictions array
                if "predictions" in data:
                    preds = data["predictions"]
                    if isinstance(preds, (list, np.ndarray)):
                        total_demand = list(preds)[:horizon]
                        info = f"Using predictions from {source_name}"

                # Check for forecast key
                elif "forecast" in data:
                    forecast = data["forecast"]
                    if isinstance(forecast, (list, np.ndarray)):
                        total_demand = list(forecast)[:horizon]
                        info = f"Using forecast from {source_name}"

                # Check for test_predictions (from backtesting)
                elif "test_predictions" in data:
                    preds = data["test_predictions"]
                    if hasattr(preds, "values"):
                        total_demand = list(preds.values)[:horizon]
                    else:
                        total_demand = list(preds)[:horizon]
                    info = f"Using test predictions from {source_name}"

                # Multi-target ML results - look for Target_1
                elif any(k.startswith("Target_") for k in data.keys()):
                    for target_key in ["Target_1", "target_1", "Total_Arrivals"]:
                        if target_key in data:
                            target_data = data[target_key]
                            if isinstance(target_data, dict) and "forecast" in target_data:
                                total_demand = list(target_data["forecast"])[:horizon]
                            elif isinstance(target_data, dict) and "predictions" in target_data:
                                total_demand = list(target_data["predictions"])[:horizon]
                            info = f"Using {target_key} from {source_name}"
                            break

        # ---------------------------------------------------------------------
        # SARIMAX / ARIMA Single Target
        # ---------------------------------------------------------------------
        elif source_type in ["sarimax", "arima"]:
            if isinstance(data, dict):
                # Check for forecast in results
                if "forecast" in data:
                    fc = data["forecast"]
                    if hasattr(fc, "values"):
                        total_demand = list(fc.values)[:horizon]
                    elif isinstance(fc, (list, np.ndarray)):
                        total_demand = list(fc)[:horizon]
                    info = f"Using forecast from {source_name}"

                # Check for results_df with predictions
                elif "results_df" in data:
                    results_df = data["results_df"]
                    if hasattr(results_df, "columns"):
                        # Look for prediction column
                        pred_cols = [c for c in results_df.columns if "pred" in c.lower() or "forecast" in c.lower()]
                        if pred_cols:
                            total_demand = list(results_df[pred_cols[0]].tail(horizon).values)
                            info = f"Using {pred_cols[0]} from {source_name}"

                # Check for future_forecast (out-of-sample)
                elif "future_forecast" in data:
                    ff = data["future_forecast"]
                    if hasattr(ff, "values"):
                        total_demand = list(ff.values)[:horizon]
                    elif isinstance(ff, (list, np.ndarray)):
                        total_demand = list(ff)[:horizon]
                    info = f"Using future forecast from {source_name}"

        # ---------------------------------------------------------------------
        # Multi-Target SARIMAX / ARIMA
        # ---------------------------------------------------------------------
        elif source_type in ["sarimax_multi", "arima_multi"]:
            if isinstance(data, dict):
                # Look for Target_1 or first successful target
                target_keys = [k for k in data.keys() if k.startswith("Target_")]

                # Also check for successful_targets list
                successful = data.get("successful_targets", [])
                if successful:
                    target_keys = successful

                if target_keys:
                    first_target = target_keys[0] if isinstance(target_keys[0], str) else f"Target_{target_keys[0]}"
                    target_data = data.get(first_target, data.get("Target_1"))

                    if isinstance(target_data, dict):
                        if "forecast" in target_data:
                            fc = target_data["forecast"]
                            if hasattr(fc, "values"):
                                total_demand = list(fc.values)[:horizon]
                            else:
                                total_demand = list(fc)[:horizon]
                        elif "predictions" in target_data:
                            preds = target_data["predictions"]
                            if hasattr(preds, "values"):
                                total_demand = list(preds.values)[:horizon]
                            else:
                                total_demand = list(preds)[:horizon]
                        info = f"Using {first_target} from {source_name}"

        # ---------------------------------------------------------------------
        # Legacy format
        # ---------------------------------------------------------------------
        elif source_type in ["legacy", "legacy_multi"]:
            if isinstance(data, dict):
                if "Target_1" in data:
                    target_1 = data["Target_1"]
                    if isinstance(target_1, dict) and "forecast" in target_1:
                        total_demand = list(target_1["forecast"])[:horizon]
                        info = f"Using Target_1 from {source_name}"
                elif "forecast" in data:
                    total_demand = list(data["forecast"])[:horizon]
                    info = f"Using forecast from {source_name}"

        # Pad demand if shorter than horizon
        if total_demand is not None and len(total_demand) < horizon:
            last_val = total_demand[-1] if total_demand else 100
            while len(total_demand) < horizon:
                total_demand.append(last_val)

        # Try to get category proportions from processed_df
        processed_df = st.session_state.get("processed_df")
        if processed_df is not None and total_demand is not None:
            cat_cols_upper = {c.upper(): c for c in processed_df.columns}
            category_proportions = {}
            total_cat = 0

            for cat in CLINICAL_CATEGORIES:
                matching_cols = [cat_cols_upper[k] for k in cat_cols_upper if cat in k]
                if matching_cols:
                    cat_sum = processed_df[matching_cols].sum().sum()
                    category_proportions[cat] = cat_sum
                    total_cat += cat_sum

            if total_cat > 0:
                category_proportions = {k: v/total_cat for k, v in category_proportions.items()}
            else:
                category_proportions = None

        # Fallback: equal proportions
        if category_proportions is None:
            category_proportions = {cat: 1/len(CLINICAL_CATEGORIES) for cat in CLINICAL_CATEGORIES}

    except Exception as e:
        info = f"Error extracting from {source_name}: {str(e)}"
        total_demand = None

    return total_demand, category_proportions, info

# =============================================================================
# TAB 1: DATA SOURCE
# =============================================================================
with tab_data:
    st.markdown("### 📊 Staff Data Source")

    data_source = st.radio(
        "Select data source:",
        ["🔗 Supabase (Cloud Database)", "📂 Upload File (CSV/Excel)"],
        horizontal=True,
        index=0 if is_connected else 1
    )

    staff_df = None

    if "Supabase" in data_source:
        if is_connected:
            with st.expander("📊 Supabase Data Options", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    fetch_mode = st.selectbox(
                        "Fetch mode",
                        ["All Data", "Date Range", "Latest N Days"]
                    )

                with col2:
                    if fetch_mode == "Date Range":
                        start_date = st.date_input("Start date", date.today() - timedelta(days=365))
                    elif fetch_mode == "Latest N Days":
                        n_days = st.number_input("Number of days", min_value=7, max_value=365, value=90)
                    else:
                        st.info("Fetching all available data")

                with col3:
                    if fetch_mode == "Date Range":
                        end_date = st.date_input("End date", date.today())

                if st.button("🔄 Fetch Data from Supabase", type="primary"):
                    with st.spinner("Fetching data from Supabase..."):
                        service = StaffSchedulingService()

                        if fetch_mode == "All Data":
                            staff_df = service.fetch_staff_data()
                        elif fetch_mode == "Date Range":
                            staff_df = service.fetch_staff_data(start_date, end_date)
                        else:
                            staff_df = pd.DataFrame(service.get_latest_records(n_days))
                            if not staff_df.empty:
                                column_mapping = {
                                    "date": "Date",
                                    "doctors_on_duty": "Doctors_on_Duty",
                                    "nurses_on_duty": "Nurses_on_Duty",
                                    "support_staff_on_duty": "Support_Staff_on_Duty",
                                    "overtime_hours": "Overtime_Hours",
                                    "average_shift_length_hours": "Average_Shift_Length_Hours",
                                    "staff_shortage_flag": "Staff_Shortage_Flag",
                                    "staff_utilization_rate": "Staff_Utilization_Rate"
                                }
                                staff_df = staff_df.rename(columns=column_mapping)

                        if staff_df is not None and not staff_df.empty:
                            st.session_state["staff_df"] = staff_df
                            st.success(f"✅ Loaded {len(staff_df)} records from Supabase")
                        else:
                            st.warning("No data found in Supabase.")

            if "staff_df" in st.session_state and st.session_state["staff_df"] is not None:
                staff_df = st.session_state["staff_df"]
        else:
            st.warning("⚠️ Supabase is not connected. Please configure credentials.")

    else:
        uploaded_file = st.file_uploader("📂 Upload staff roster (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    staff_df = pd.read_csv(uploaded_file)
                else:
                    staff_df = pd.read_excel(uploaded_file)

                st.session_state["staff_df"] = staff_df
                st.success(f"✅ Loaded {len(staff_df)} records from file")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        elif "staff_df" in st.session_state:
            staff_df = st.session_state["staff_df"]

    # Display data overview
    if st.session_state.get("staff_df") is not None:
        staff_df = st.session_state["staff_df"]
        st.divider()
        st.markdown("### 📊 Staff Data Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if "Doctors_on_Duty" in staff_df.columns:
                min_doc = int(staff_df["Doctors_on_Duty"].min())
                max_doc = int(staff_df["Doctors_on_Duty"].max())
                st.markdown(f"""
                <div class="staff-metric-card">
                    <div class="staff-metric-value">{min_doc}-{max_doc}</div>
                    <div class="staff-metric-label">Doctors/Day</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if "Nurses_on_Duty" in staff_df.columns:
                min_nurse = int(staff_df["Nurses_on_Duty"].min())
                max_nurse = int(staff_df["Nurses_on_Duty"].max())
                st.markdown(f"""
                <div class="staff-metric-card">
                    <div class="staff-metric-value">{min_nurse}-{max_nurse}</div>
                    <div class="staff-metric-label">Nurses/Day</div>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            if "Support_Staff_on_Duty" in staff_df.columns:
                min_support = int(staff_df["Support_Staff_on_Duty"].min())
                max_support = int(staff_df["Support_Staff_on_Duty"].max())
                st.markdown(f"""
                <div class="staff-metric-card">
                    <div class="staff-metric-value">{min_support}-{max_support}</div>
                    <div class="staff-metric-label">Support Staff</div>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            if "Overtime_Hours" in staff_df.columns:
                total_overtime = int(staff_df["Overtime_Hours"].sum())
                st.markdown(f"""
                <div class="staff-metric-card">
                    <div class="staff-metric-value">{total_overtime:,}h</div>
                    <div class="staff-metric-label">Total Overtime</div>
                </div>
                """, unsafe_allow_html=True)

        with col5:
            if "Staff_Shortage_Flag" in staff_df.columns:
                shortage_days = int(staff_df["Staff_Shortage_Flag"].sum())
                total_days = len(staff_df)
                st.markdown(f"""
                <div class="staff-metric-card">
                    <div class="staff-metric-value">{shortage_days}/{total_days}</div>
                    <div class="staff-metric-label">Shortage Days</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📋 Data Preview (First 10 Rows)", expanded=False):
            st.dataframe(staff_df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(staff_df):,} total records")

# =============================================================================
# TAB 2: CONFIGURATION
# =============================================================================
with tab_config:
    st.markdown("### ⚙️ Optimization Configuration")

    # Financial Data Integration
    if FINANCIAL_SERVICE_AVAILABLE:
        st.markdown("#### 📊 Load Cost Parameters from Financial Data")

        fin_col1, fin_col2 = st.columns([3, 1])

        with fin_col1:
            use_financial_data = st.checkbox(
                "Load real cost parameters from Supabase Financial Data",
                value=False,
                help="Use historical financial data to auto-populate cost parameters"
            )

        with fin_col2:
            if use_financial_data:
                if st.button("📥 Load Financial Data", type="secondary"):
                    with st.spinner("Loading financial data..."):
                        fin_service = get_financial_service()
                        if fin_service.is_connected():
                            labor_params = fin_service.get_labor_cost_params()
                            if labor_params:
                                st.session_state["financial_labor_params"] = labor_params
                                st.success(f"✅ Loaded financial parameters!")
                            else:
                                st.warning("No financial data found. Using defaults.")
                        else:
                            st.error("Financial service not connected.")

        # Show loaded financial params
        if "financial_labor_params" in st.session_state and use_financial_data:
            params = st.session_state["financial_labor_params"]
            st.info(f"""
            **Loaded from Financial Data:**
            - Doctor Rate: ${params.get('doctor_hourly_rate', 150):.2f}/hr
            - Nurse Rate: ${params.get('nurse_hourly_rate', 45):.2f}/hr
            - Support Rate: ${params.get('support_hourly_rate', 25):.2f}/hr
            - Overtime Multiplier: {params.get('overtime_multiplier', 1.5):.2f}x
            """)

        st.divider()

    col_cost, col_ratio = st.columns(2)

    with col_cost:
        st.markdown("#### 💰 Cost Parameters (USD)")

        # Use financial data if available and enabled
        fin_params = st.session_state.get("financial_labor_params", {})
        use_fin = FINANCIAL_SERVICE_AVAILABLE and st.session_state.get("use_financial_data", False) and fin_params

        with st.expander("Hourly Rates", expanded=True):
            doctor_rate = st.number_input(
                "Doctor Hourly Rate ($)",
                min_value=50.0, max_value=1000.0,
                value=float(fin_params.get("doctor_hourly_rate", 150.0)) if fin_params else 150.0,
                step=10.0
            )
            nurse_rate = st.number_input(
                "Nurse Hourly Rate ($)",
                min_value=20.0, max_value=500.0,
                value=float(fin_params.get("nurse_hourly_rate", 45.0)) if fin_params else 45.0,
                step=5.0
            )
            support_rate = st.number_input(
                "Support Staff Hourly Rate ($)",
                min_value=10.0, max_value=300.0,
                value=float(fin_params.get("support_hourly_rate", 25.0)) if fin_params else 25.0,
                step=5.0
            )

        with st.expander("Multipliers & Penalties", expanded=True):
            overtime_mult = st.number_input(
                "Overtime Multiplier",
                min_value=1.0, max_value=3.0,
                value=float(fin_params.get("overtime_multiplier", 1.5)) if fin_params else 1.5,
                step=0.1
            )
            agency_mult = st.number_input(
                "Agency Staff Multiplier",
                min_value=1.0, max_value=4.0, value=2.0, step=0.1
            )
            understaffing_penalty = st.number_input(
                "Understaffing Penalty ($/patient-hr)",
                min_value=50.0, max_value=100000.0,
                value=float(fin_params.get("understaffing_penalty", 200.0)) if fin_params else 200.0,
                step=25.0
            )
            overstaffing_penalty = st.number_input(
                "Overstaffing Penalty ($/staff-hr)",
                min_value=1.0, max_value=10000.0,
                value=float(fin_params.get("overstaffing_penalty", 15.0)) if fin_params else 15.0,
                step=1.0
            )

        with st.expander("Shift Configuration", expanded=False):
            shift_length = st.number_input(
                "Shift Length (hours)",
                min_value=4.0, max_value=12.0, value=8.0, step=1.0
            )
            max_overtime = st.number_input(
                "Max Overtime per Staff (hours/day)",
                min_value=0.0, max_value=8.0, value=4.0, step=0.5
            )

        # Update cost params
        cost_params = CostParameters(
            doctor_hourly_rate=doctor_rate,
            nurse_hourly_rate=nurse_rate,
            support_hourly_rate=support_rate,
            overtime_multiplier=overtime_mult,
            agency_multiplier=agency_mult,
            understaffing_penalty_base=understaffing_penalty,
            overstaffing_penalty=overstaffing_penalty,
            shift_length=shift_length,
            max_overtime_hours=max_overtime,
        )
        st.session_state["cost_params"] = cost_params

    with col_ratio:
        st.markdown("#### 👥 Staffing Ratios (Patients per Staff)")

        staffing_ratios = StaffingRatios()

        with st.expander("Staff Availability Constraints", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                min_doctors = st.number_input("Min Doctors", min_value=1, max_value=50, value=5)
                min_nurses = st.number_input("Min Nurses", min_value=1, max_value=100, value=15)
                min_support = st.number_input("Min Support", min_value=1, max_value=50, value=8)
            with col2:
                max_doctors = st.number_input("Max Doctors", min_value=5, max_value=100, value=20)
                max_nurses = st.number_input("Max Nurses", min_value=10, max_value=200, value=50)
                max_support = st.number_input("Max Support", min_value=5, max_value=100, value=30)

            nurse_ratio = st.number_input(
                "Nurse-to-Doctor Ratio",
                min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                help="Minimum nurses per doctor"
            )

        staffing_ratios.min_doctors = min_doctors
        staffing_ratios.max_doctors = max_doctors
        staffing_ratios.min_nurses = min_nurses
        staffing_ratios.max_nurses = max_nurses
        staffing_ratios.min_support = min_support
        staffing_ratios.max_support = max_support
        staffing_ratios.nurse_to_doctor_ratio = nurse_ratio

        with st.expander("Category-Specific Ratios", expanded=False):
            st.markdown("**Patients per Staff Member (by Category)**")
            st.caption("Lower ratio = more staff needed (higher acuity)")

            for category in CLINICAL_CATEGORIES:
                st.markdown(f"**{category}** (Priority: {CATEGORY_PRIORITIES[category]})")
                col1, col2, col3 = st.columns(3)
                with col1:
                    doc_ratio = st.number_input(
                        f"Doc 1:{staffing_ratios.get_ratio(category, 'Doctors'):.0f}",
                        min_value=1, max_value=20,
                        value=int(staffing_ratios.get_ratio(category, "Doctors")),
                        key=f"doc_{category}"
                    )
                    staffing_ratios.set_ratio(category, "Doctors", doc_ratio)
                with col2:
                    nurse_ratio_cat = st.number_input(
                        f"Nurse 1:{staffing_ratios.get_ratio(category, 'Nurses'):.0f}",
                        min_value=1, max_value=15,
                        value=int(staffing_ratios.get_ratio(category, "Nurses")),
                        key=f"nurse_{category}"
                    )
                    staffing_ratios.set_ratio(category, "Nurses", nurse_ratio_cat)
                with col3:
                    support_ratio = st.number_input(
                        f"Support 1:{staffing_ratios.get_ratio(category, 'Support'):.0f}",
                        min_value=1, max_value=25,
                        value=int(staffing_ratios.get_ratio(category, "Support")),
                        key=f"support_{category}"
                    )
                    staffing_ratios.set_ratio(category, "Support", support_ratio)

        st.session_state["staffing_ratios"] = staffing_ratios

    # Display current configuration summary
    st.divider()
    st.markdown("### 📋 Configuration Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost Parameters (USD)**")
        st.dataframe(
            pd.DataFrame([cost_params.to_dict()]).T.reset_index().rename(
                columns={"index": "Parameter", 0: "Value"}
            ),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("**Staffing Ratios by Category**")
        st.dataframe(staffing_ratios.to_dataframe(), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: OPTIMIZE
# =============================================================================
with tab_optimize:
    st.markdown("### 🚀 Run Optimization")

    staff_df = st.session_state.get("staff_df")

    if staff_df is None or staff_df.empty:
        st.warning("⚠️ Please load staff data first (Tab 1: Data Source)")
    else:
        # Demand source selection
        st.markdown("#### 📈 Demand Forecast Source")

        demand_source = st.radio(
            "Select demand source:",
            [
                "🔮 ML Forecast (from Dashboard/Modeling Hub)",
                "📊 Historical Average (from Supabase data)",
                "✏️ Manual Input"
            ],
            horizontal=True
        )

        planning_horizon = st.slider("Planning Horizon (days)", min_value=3, max_value=14, value=7)

        # Get demand based on source
        total_demand = None
        category_demand = None
        category_proportions = None

        if "ML Forecast" in demand_source:
            # Use unified forecast detector
            forecast_sources = detect_forecast_sources()

            if forecast_sources:
                st.success(f"✅ {len(forecast_sources)} forecast source(s) detected!")

                # Show available sources in expander
                with st.expander("📊 Available Forecast Sources", expanded=True):
                    source_names = [s["name"] for s in forecast_sources]
                    st.markdown("**Detected sources:**")
                    for i, name in enumerate(source_names, 1):
                        st.markdown(f"  {i}. {name}")

                # Let user select source
                selected_source_name = st.selectbox(
                    "Select forecast source to use:",
                    options=[s["name"] for s in forecast_sources],
                    index=0,
                    key="forecast_source_selector"
                )

                # Find selected source
                selected_source = next(s for s in forecast_sources if s["name"] == selected_source_name)

                # Extract demand from selected source
                total_demand, category_proportions, extract_info = extract_demand_from_source(
                    selected_source, horizon=planning_horizon
                )

                if total_demand is not None:
                    st.info(f"ℹ️ {extract_info}")
                    st.write(f"**Total Demand Forecast ({planning_horizon} days):** {[round(d, 1) for d in total_demand]}")

                    # Show category proportions
                    if category_proportions:
                        st.write("**Category Proportions:**")
                        prop_df = pd.DataFrame([{k: f"{v:.1%}" for k, v in category_proportions.items()}])
                        st.dataframe(prop_df, hide_index=True)
                else:
                    st.warning(f"Could not extract forecast from {selected_source_name}. {extract_info}")
                    st.info("Try selecting a different source or use Historical Average.")
            else:
                st.warning("No forecast found. Please run forecasting first in:")
                st.markdown("""
                - **Dashboard** (01_Dashboard.py) - Quick forecasts
                - **Benchmarks** (05_Benchmarks.py) - ARIMA/SARIMAX models
                - **Modeling Hub** (08_Modeling_Hub.py) - XGBoost, LSTM, ANN
                """)
                st.info("Using historical average as fallback.")
                demand_source = "📊 Historical Average"

        if "Historical Average" in demand_source:
            # Calculate from Supabase data
            avg_doctors = staff_df["Doctors_on_Duty"].mean() if "Doctors_on_Duty" in staff_df.columns else 10
            avg_nurses = staff_df["Nurses_on_Duty"].mean() if "Nurses_on_Duty" in staff_df.columns else 25
            avg_support = staff_df["Support_Staff_on_Duty"].mean() if "Support_Staff_on_Duty" in staff_df.columns else 15

            # Estimate patients using average ratios
            estimated_daily = (
                avg_doctors * 8 +
                avg_nurses * 4 +
                avg_support * 10
            ) / 3

            total_demand = [estimated_daily] * planning_horizon
            st.info(f"Estimated daily demand from historical staffing: ~{estimated_daily:.0f} patients")

            # Default equal proportions
            category_proportions = {cat: 1/len(CLINICAL_CATEGORIES) for cat in CLINICAL_CATEGORIES}

        if "Manual Input" in demand_source:
            st.markdown("**Enter Daily Demand:**")

            col1, col2 = st.columns(2)

            with col1:
                total_demand = []
                for day in range(planning_horizon):
                    demand = st.number_input(
                        f"Day {day + 1} Total Patients",
                        min_value=10, max_value=500, value=100,
                        key=f"demand_day_{day}"
                    )
                    total_demand.append(demand)

            with col2:
                st.markdown("**Category Proportions (must sum to 1):**")
                category_proportions = {}
                remaining = 1.0

                for i, cat in enumerate(CLINICAL_CATEGORIES[:-1]):
                    prop = st.slider(
                        f"{cat}",
                        min_value=0.0, max_value=remaining, value=min(0.14, remaining),
                        step=0.01, key=f"prop_{cat}"
                    )
                    category_proportions[cat] = prop
                    remaining -= prop

                # Last category gets the remainder
                category_proportions[CLINICAL_CATEGORIES[-1]] = max(0, remaining)
                st.write(f"{CLINICAL_CATEGORIES[-1]}: {remaining:.2f}")

        st.divider()

        # Optimization execution
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Solver Settings")
            time_limit = st.number_input("Time Limit (seconds)", min_value=10, max_value=300, value=60)
            gap_tolerance = st.number_input("Optimality Gap (%)", min_value=0.1, max_value=10.0, value=1.0) / 100

            run_optimization = st.button("🚀 Run MILP Optimization", type="primary", use_container_width=True)

        with col2:
            if run_optimization and total_demand is not None:
                with st.spinner("Running MILP optimization..."):
                    try:
                        # Run optimization
                        solver = StaffSchedulingMILP(
                            cost_params=st.session_state["cost_params"],
                            staffing_ratios=st.session_state["staffing_ratios"],
                        )

                        solver.set_demand_forecast(
                            total_demand=total_demand,
                            category_proportions=category_proportions,
                            category_demand=category_demand,
                        )

                        solver.build_model()
                        result = solver.solve(time_limit=time_limit, gap=gap_tolerance)

                        st.session_state["optimization_result"] = result

                        # Also run historical analysis for comparison
                        historical = analyze_historical_costs(
                            staff_df,
                            cost_params=st.session_state["cost_params"],
                            n_days=planning_horizon
                        )
                        st.session_state["historical_analysis"] = historical

                        if result.is_optimal:
                            st.success(f"✅ Optimization Complete! Status: {result.status}")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ Optimization finished with status: {result.status}")

                        # Show quick summary
                        st.markdown("#### Quick Summary")
                        summary = result.get_summary()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Cost", summary["Total Cost"])
                        with col2:
                            st.metric("Solve Time", summary["Solve Time"])
                        with col3:
                            st.metric("Status", summary["Status"])

                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# =============================================================================
# TAB 4: RESULTS & COMPARISON
# =============================================================================
with tab_results:
    st.markdown("### 📈 Optimization Results & Before/After Comparison")

    result = st.session_state.get("optimization_result")
    historical = st.session_state.get("historical_analysis")

    if result is None:
        st.info("Run optimization first (Tab 3: Optimize) to see results.")
    else:
        # Before/After Comparison
        st.markdown("## 🔄 Before vs After Comparison")

        if historical is not None:
            analyzer = SolutionAnalyzer(st.session_state["cost_params"])
            comparison_df = analyzer.compare_before_after(historical, result)

            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)

            cost_savings = historical.total_cost - result.total_cost
            savings_pct = (cost_savings / historical.total_cost * 100) if historical.total_cost > 0 else 0

            with col1:
                savings_class = "negative" if cost_savings < 0 else ""
                st.markdown(f"""
                <div class="opt-card">
                    <div class="opt-card-header">💰 Cost Savings</div>
                    <div class="opt-metric">
                        <div class="opt-metric-value {savings_class}">${cost_savings:,.0f}</div>
                        <div class="opt-metric-label">{savings_pct:+.1f}% vs Historical</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="opt-card">
                    <div class="opt-card-header">📊 Before (Historical)</div>
                    <div class="opt-metric">
                        <div class="opt-metric-value neutral">${historical.total_cost:,.0f}</div>
                        <div class="opt-metric-label">Total Cost</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="opt-card">
                    <div class="opt-card-header">✅ After (Optimized)</div>
                    <div class="opt-metric">
                        <div class="opt-metric-value">${result.total_cost:,.0f}</div>
                        <div class="opt-metric-label">Total Cost</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                avg_coverage = result.daily_summary["Coverage_%"].mean() if result.daily_summary is not None else 100
                st.markdown(f"""
                <div class="opt-card">
                    <div class="opt-card-header">🎯 Demand Coverage</div>
                    <div class="opt-metric">
                        <div class="opt-metric-value">{avg_coverage:.1f}%</div>
                        <div class="opt-metric-label">Avg Daily Coverage</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Comparison table
            st.markdown("### 📋 Detailed Comparison")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.divider()

        # Cost Breakdown
        st.markdown("## 💵 Cost Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart of optimized costs
            cost_data = {
                "Category": ["Regular Labor", "Overtime", "Understaffing Penalty", "Overstaffing Penalty"],
                "Cost": [
                    result.regular_labor_cost,
                    result.overtime_cost,
                    result.understaffing_penalty,
                    result.overstaffing_penalty
                ]
            }
            cost_df = pd.DataFrame(cost_data)

            fig_pie = px.pie(
                cost_df,
                values="Cost",
                names="Category",
                title="Optimized Cost Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart comparison
            if historical is not None:
                compare_data = {
                    "Category": ["Regular Labor", "Overtime", "Penalties"],
                    "Before": [
                        historical.regular_labor_cost,
                        historical.overtime_cost,
                        historical.estimated_shortage_cost
                    ],
                    "After": [
                        result.regular_labor_cost,
                        result.overtime_cost,
                        result.understaffing_penalty + result.overstaffing_penalty
                    ]
                }
                compare_df = pd.DataFrame(compare_data)

                fig_bar = go.Figure(data=[
                    go.Bar(name='Before (Historical)', x=compare_df["Category"], y=compare_df["Before"], marker_color='#ef4444'),
                    go.Bar(name='After (Optimized)', x=compare_df["Category"], y=compare_df["After"], marker_color='#22c55e')
                ])
                fig_bar.update_layout(
                    barmode='group',
                    title="Cost Comparison by Category",
                    yaxis_title="Cost ($)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Staff Schedule
        st.markdown("## 👥 Optimized Staff Schedule")

        if result.staff_schedule is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Daily Staff Assignments**")
                st.dataframe(result.staff_schedule, use_container_width=True, hide_index=True)

            with col2:
                # Line chart of staff over time
                fig_staff = go.Figure()
                for col in ["Doctors", "Nurses", "Support"]:
                    if col in result.staff_schedule.columns:
                        fig_staff.add_trace(go.Scatter(
                            x=result.staff_schedule["Day"],
                            y=result.staff_schedule[col],
                            mode='lines+markers',
                            name=col
                        ))

                fig_staff.update_layout(
                    title="Staff Schedule Over Planning Horizon",
                    xaxis_title="Day",
                    yaxis_title="Staff Count",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_staff, use_container_width=True)

        # Overtime Schedule
        if result.overtime_schedule is not None:
            st.markdown("**Overtime Hours by Day**")
            st.dataframe(result.overtime_schedule, use_container_width=True, hide_index=True)

        st.divider()

        # Category Coverage
        st.markdown("## 🏥 Clinical Category Coverage")

        if result.category_coverage is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(result.category_coverage, use_container_width=True, hide_index=True)

            with col2:
                fig_coverage = px.bar(
                    result.category_coverage,
                    x="Category",
                    y="Coverage_%",
                    color="Coverage_%",
                    color_continuous_scale=["#ef4444", "#fbbf24", "#22c55e"],
                    title="Coverage by Clinical Category"
                )
                fig_coverage.add_hline(y=100, line_dash="dash", line_color="white", annotation_text="100% Target")
                fig_coverage.update_layout(template="plotly_dark")
                st.plotly_chart(fig_coverage, use_container_width=True)

        # Daily Summary
        if result.daily_summary is not None:
            st.markdown("### 📅 Daily Summary")
            st.dataframe(result.daily_summary, use_container_width=True, hide_index=True)

        # Download results
        st.divider()
        st.markdown("### 📥 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if result.staff_schedule is not None:
                csv = result.staff_schedule.to_csv(index=False)
                st.download_button(
                    "📥 Download Staff Schedule",
                    data=csv,
                    file_name="optimized_staff_schedule.csv",
                    mime="text/csv"
                )

        with col2:
            if result.daily_summary is not None:
                csv = result.daily_summary.to_csv(index=False)
                st.download_button(
                    "📥 Download Daily Summary",
                    data=csv,
                    file_name="optimization_daily_summary.csv",
                    mime="text/csv"
                )

        with col3:
            if result.category_coverage is not None:
                csv = result.category_coverage.to_csv(index=False)
                st.download_button(
                    "📥 Download Category Coverage",
                    data=csv,
                    file_name="category_coverage.csv",
                    mime="text/csv"
                )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Staff Scheduling Optimization powered by PuLP MILP Solver. "
    "Deterministic optimization with clinical category integration. "
    "All costs in USD."
)
