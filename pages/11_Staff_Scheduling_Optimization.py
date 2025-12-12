# =============================================================================
# 11_Staff_Scheduling_Optimization.py
# INTERACTIVE: Button-Driven Staff Optimization with Expert Manager View
# Design: Matches Modeling Hub (Page 08) styling
# DATA: Fetched from Supabase via button click (staff_scheduling, financial_data)
# =============================================================================
"""
Interactive Staff Scheduling Optimization

Workflow:
1. STEP 1: Click "Load Data" button to fetch real hospital data from Supabase
2. STEP 2: View current hospital situation (Expert Manager Dashboard)
3. STEP 3: Click "Optimize Staff" to apply forecast and see financial impact

Data Sources:
- Staff data: Supabase staff_scheduling table (via API button)
- Cost params: Supabase financial_data table (via API button)
- Forecasts: Session state from Modeling Hub (ml_mh_results, arima_mh_results, etc.)
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

# Import Supabase services for real data
from app_core.data.staff_scheduling_service import StaffSchedulingService, fetch_staff_scheduling_data
from app_core.data.financial_service import FinancialService, get_financial_service

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Staff Scheduling - HealthForecast AI",
    page_icon="👥",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# CONFIGURATION: STAFFING RATIOS BY CLINICAL CATEGORY
# =============================================================================
STAFFING_RATIOS = {
    "CARDIAC": {"doctors": 5, "nurses": 3, "support": 8},
    "TRAUMA": {"doctors": 6, "nurses": 2, "support": 5},
    "RESPIRATORY": {"doctors": 8, "nurses": 4, "support": 10},
    "GASTROINTESTINAL": {"doctors": 10, "nurses": 5, "support": 12},
    "INFECTIOUS": {"doctors": 12, "nurses": 6, "support": 15},
    "NEUROLOGICAL": {"doctors": 8, "nurses": 4, "support": 10},
    "OTHER": {"doctors": 15, "nurses": 8, "support": 20},
}

# Default cost parameters (USD)
DEFAULT_COST_PARAMS = {
    "doctor_hourly": 150,
    "nurse_hourly": 50,
    "support_hourly": 25,
    "shift_hours": 8,
    "overtime_multiplier": 1.5,
    "understaffing_penalty_per_patient": 200,
}


# =============================================================================
# FLUORESCENT EFFECTS + CUSTOM STYLING
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
    animation-delay: 5s;
}}

/* Custom Tab Styling */
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

/* Step Cards */
.step-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
}}

.step-card.active {{
    border-color: rgba(34, 197, 94, 0.6);
    box-shadow: 0 0 30px rgba(34, 197, 94, 0.2);
}}

.step-card.completed {{
    border-color: rgba(34, 197, 94, 0.8);
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.98));
}}

.step-number {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border-radius: 50%;
    font-size: 1.25rem;
    font-weight: 700;
    margin-right: 1rem;
}}

.step-number.completed {{
    background: linear-gradient(135deg, #22c55e, #16a34a);
}}

.step-title {{
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
}}

/* KPI Cards */
.kpi-card {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}}

.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}}

.kpi-card.warning {{
    border-color: rgba(234, 179, 8, 0.5);
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.95));
}}

.kpi-card.danger {{
    border-color: rgba(239, 68, 68, 0.5);
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
}}

.kpi-card.success {{
    border-color: rgba(34, 197, 94, 0.5);
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.95));
}}

.kpi-value {{
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}

.kpi-value.red {{ color: #ef4444; }}
.kpi-value.green {{ color: #22c55e; }}
.kpi-value.blue {{ color: #3b82f6; }}
.kpi-value.yellow {{ color: #eab308; }}

.kpi-label {{
    font-size: 0.9rem;
    color: #94a3b8;
}}

.kpi-delta {{
    font-size: 0.85rem;
    margin-top: 0.5rem;
}}

.kpi-delta.positive {{ color: #22c55e; }}
.kpi-delta.negative {{ color: #ef4444; }}

/* Action Button */
.action-button {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border: none;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 600;
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
}}

.action-button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}}

.action-button.success {{
    background: linear-gradient(135deg, #22c55e, #16a34a);
}}

.action-button.success:hover {{
    box-shadow: 0 8px 24px rgba(34, 197, 94, 0.4);
}}

/* Section Header */
.section-header {{
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(34, 197, 94, 0.3);
}}

/* Alert Box */
.alert-box {{
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}}

.alert-box.info {{
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #60a5fa;
}}

.alert-box.warning {{
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.4);
    color: #eab308;
}}

.alert-box.success {{
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22c55e;
}}

.alert-box.danger {{
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #ef4444;
}}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1.5rem;'>
      <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>👥</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Staff Scheduling Optimization</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Expert Manager Dashboard - Load hospital data, analyze current situation, and optimize with forecasts
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "staff_db_loaded" not in st.session_state:
    st.session_state.staff_db_loaded = False
if "staff_data" not in st.session_state:
    st.session_state.staff_data = None
if "financial_data" not in st.session_state:
    st.session_state.financial_data = None
if "cost_params" not in st.session_state:
    st.session_state.cost_params = DEFAULT_COST_PARAMS.copy()
if "optimization_applied" not in st.session_state:
    st.session_state.optimization_applied = False
if "optimized_schedule" not in st.session_state:
    st.session_state.optimized_schedule = None


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def fetch_staff_data_from_api() -> Dict[str, Any]:
    """Fetch staff scheduling data from Supabase API."""
    result = {
        "success": False,
        "data": None,
        "stats": {},
        "error": None,
    }

    try:
        staff_service = StaffSchedulingService()
        if not staff_service.is_connected():
            result["error"] = "Supabase not connected. Check your credentials."
            return result

        # Fetch all staff data
        staff_df = staff_service.fetch_staff_data()

        if staff_df.empty:
            result["error"] = "No staff scheduling data found in database."
            return result

        result["success"] = True
        result["data"] = staff_df

        # Calculate statistics
        result["stats"] = {
            "total_records": len(staff_df),
            "avg_doctors": int(staff_df["Doctors_on_Duty"].mean()) if "Doctors_on_Duty" in staff_df.columns else 0,
            "avg_nurses": int(staff_df["Nurses_on_Duty"].mean()) if "Nurses_on_Duty" in staff_df.columns else 0,
            "avg_support": int(staff_df["Support_Staff_on_Duty"].mean()) if "Support_Staff_on_Duty" in staff_df.columns else 0,
            "total_overtime": float(staff_df["Overtime_Hours"].sum()) if "Overtime_Hours" in staff_df.columns else 0,
            "avg_utilization": float(staff_df["Staff_Utilization_Rate"].mean()) if "Staff_Utilization_Rate" in staff_df.columns else 0,
            "shortage_days": int(staff_df["Staff_Shortage_Flag"].sum()) if "Staff_Shortage_Flag" in staff_df.columns else 0,
            "date_start": staff_df["Date"].min() if "Date" in staff_df.columns else None,
            "date_end": staff_df["Date"].max() if "Date" in staff_df.columns else None,
        }

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def fetch_financial_data_from_api() -> Dict[str, Any]:
    """Fetch financial data from Supabase API."""
    result = {
        "success": False,
        "data": None,
        "cost_params": DEFAULT_COST_PARAMS.copy(),
        "kpis": {},
        "error": None,
    }

    try:
        financial_service = get_financial_service()
        if not financial_service.is_connected():
            result["error"] = "Supabase not connected."
            return result

        # Fetch labor cost parameters
        labor_params = financial_service.get_labor_cost_params()
        if labor_params:
            result["cost_params"] = {
                "doctor_hourly": labor_params.get("doctor_hourly_rate", 150),
                "nurse_hourly": labor_params.get("nurse_hourly_rate", 50),
                "support_hourly": labor_params.get("support_hourly_rate", 25),
                "shift_hours": 8,
                "overtime_multiplier": labor_params.get("overtime_multiplier", 1.5),
                "understaffing_penalty_per_patient": labor_params.get("understaffing_penalty", 200),
                "overstaffing_penalty": labor_params.get("overstaffing_penalty", 0),
            }

        # Fetch comprehensive KPIs
        kpis = financial_service.get_comprehensive_financial_kpis()
        if kpis:
            result["kpis"] = kpis

        # Fetch raw data
        fin_df = financial_service.fetch_all()
        if not fin_df.empty:
            result["data"] = fin_df
            result["success"] = True
        else:
            result["success"] = True  # Cost params loaded even if no raw data

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def get_forecast_data() -> Dict[str, Any]:
    """Extract forecast data from session state."""
    result = {
        "has_forecast": False,
        "source": None,
        "total_forecast": [],
        "category_forecast": {},
        "dates": [],
        "horizon": 0,
    }

    # Try ML models
    ml_results = st.session_state.get("ml_mh_results")
    if ml_results and isinstance(ml_results, dict):
        for model_name in ["XGBoost", "LSTM", "ANN"]:
            if model_name in ml_results:
                per_h = ml_results[model_name].get("per_h", {})
                if per_h:
                    try:
                        horizons = sorted([int(h) for h in per_h.keys()])
                        forecasts = []
                        for h in horizons[:7]:
                            h_data = per_h.get(h, {})
                            pred = h_data.get("predictions") or h_data.get("forecast") or h_data.get("y_pred")
                            if pred is not None:
                                pred_arr = pred.values if hasattr(pred, 'values') else np.array(pred)
                                forecasts.append(float(pred_arr[-1]) if len(pred_arr) > 0 else 0)

                        if forecasts:
                            result["has_forecast"] = True
                            result["source"] = f"ML ({model_name})"
                            result["total_forecast"] = forecasts
                            result["horizon"] = len(forecasts)
                            break
                    except Exception:
                        continue

    # Try ARIMA
    if not result["has_forecast"]:
        arima_results = st.session_state.get("arima_mh_results")
        if arima_results and isinstance(arima_results, dict):
            per_h = arima_results.get("per_h", {})
            if per_h:
                try:
                    horizons = sorted([int(h) for h in per_h.keys()])
                    forecasts = []
                    for h in horizons[:7]:
                        fc = per_h.get(h, {}).get("forecast")
                        if fc is not None:
                            fc_arr = fc.values if hasattr(fc, 'values') else np.array(fc)
                            forecasts.append(float(fc_arr[-1]) if len(fc_arr) > 0 else 0)

                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = "ARIMA"
                        result["total_forecast"] = forecasts
                        result["horizon"] = len(forecasts)
                except Exception:
                    pass

    # Generate dates and category distribution
    if result["has_forecast"]:
        today = datetime.now().date()
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, result["horizon"] + 1)]

        distribution = {
            "CARDIAC": 0.15, "TRAUMA": 0.20, "RESPIRATORY": 0.25,
            "GASTROINTESTINAL": 0.15, "INFECTIOUS": 0.15, "NEUROLOGICAL": 0.05, "OTHER": 0.05,
        }
        for cat, pct in distribution.items():
            result["category_forecast"][cat] = [f * pct for f in result["total_forecast"]]

    return result


def calculate_optimized_schedule(forecast_data: Dict[str, Any], cost_params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimized staffing schedule based on forecast."""
    if not forecast_data["has_forecast"]:
        return {"success": False, "error": "No forecast data available"}

    daily_schedules = []
    total_cost = 0

    for day_idx in range(forecast_data["horizon"]):
        patients_by_category = {cat: forecasts[day_idx] for cat, forecasts in forecast_data["category_forecast"].items() if day_idx < len(forecasts)}

        # Calculate optimal staff
        doctors = sum(np.ceil(patients / STAFFING_RATIOS[cat]["doctors"]) for cat, patients in patients_by_category.items() if cat in STAFFING_RATIOS)
        nurses = sum(np.ceil(patients / STAFFING_RATIOS[cat]["nurses"]) for cat, patients in patients_by_category.items() if cat in STAFFING_RATIOS)
        support = sum(np.ceil(patients / STAFFING_RATIOS[cat]["support"]) for cat, patients in patients_by_category.items() if cat in STAFFING_RATIOS)

        # Calculate cost
        daily_cost = (
            doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
            nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
            support * cost_params["support_hourly"] * cost_params["shift_hours"]
        )

        daily_schedules.append({
            "date": forecast_data["dates"][day_idx],
            "expected_patients": sum(patients_by_category.values()),
            "doctors": int(doctors),
            "nurses": int(nurses),
            "support": int(support),
            "total_staff": int(doctors + nurses + support),
            "daily_cost": daily_cost,
        })

        total_cost += daily_cost

    return {
        "success": True,
        "schedules": daily_schedules,
        "total_cost": total_cost,
        "avg_daily_cost": total_cost / forecast_data["horizon"],
        "horizon": forecast_data["horizon"],
        "source": forecast_data["source"],
    }


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Step indicator
step1_complete = st.session_state.staff_db_loaded
step2_complete = step1_complete
step3_complete = st.session_state.optimization_applied

st.markdown("---")

# =============================================================================
# STEP 1: LOAD DATA FROM DATABASE
# =============================================================================
step1_class = "completed" if step1_complete else "active"
st.markdown(f"""
<div class="step-card {step1_class}">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number {'completed' if step1_complete else ''}">{'✓' if step1_complete else '1'}</span>
        <span class="step-title">Load Hospital Data from Database</span>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.button("🔄 Load Staff & Financial Data", type="primary", use_container_width=True):
        with st.spinner("Fetching data from Supabase..."):
            # Fetch staff data
            staff_result = fetch_staff_data_from_api()

            # Fetch financial data
            fin_result = fetch_financial_data_from_api()

            if staff_result["success"]:
                st.session_state.staff_data = staff_result
                st.session_state.staff_db_loaded = True

                if fin_result["success"]:
                    st.session_state.financial_data = fin_result
                    st.session_state.cost_params = fin_result["cost_params"]

                st.success(f"✅ Loaded {staff_result['stats']['total_records']} staff records!")
                st.rerun()
            else:
                st.error(f"❌ Failed to load data: {staff_result['error']}")

with col2:
    if step1_complete:
        st.success(f"✅ {st.session_state.staff_data['stats']['total_records']} days loaded")
    else:
        st.info("Click button to fetch from Supabase")

with col3:
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.staff_db_loaded = False
        st.session_state.staff_data = None
        st.session_state.financial_data = None
        st.session_state.optimization_applied = False
        st.session_state.optimized_schedule = None
        st.rerun()

st.markdown("---")

# =============================================================================
# STEP 2: CURRENT HOSPITAL SITUATION (Expert Manager View)
# =============================================================================
if st.session_state.staff_db_loaded and st.session_state.staff_data:

    step2_class = "completed" if step2_complete else "active"
    st.markdown(f"""
    <div class="step-card {step2_class}">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span class="step-number {'completed' if step2_complete else ''}">{'✓' if step2_complete else '2'}</span>
            <span class="step-title">Current Hospital Situation - Expert Manager Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = st.session_state.staff_data["stats"]
    cost_params = st.session_state.cost_params

    # Date range info
    st.markdown(f"""
    <div class="alert-box info">
        📅 Data Period: <strong>{stats['date_start'].strftime('%Y-%m-%d') if stats['date_start'] else 'N/A'}</strong>
        to <strong>{stats['date_end'].strftime('%Y-%m-%d') if stats['date_end'] else 'N/A'}</strong>
        ({stats['total_records']} days)
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards Row 1: Staff Levels
    st.markdown("### 👥 Current Staffing Levels (Daily Averages)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value blue">{stats['avg_doctors']}</div>
            <div class="kpi-label">Doctors/Day</div>
            <div class="kpi-delta">@ ${cost_params['doctor_hourly']}/hr</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value blue">{stats['avg_nurses']}</div>
            <div class="kpi-label">Nurses/Day</div>
            <div class="kpi-delta">@ ${cost_params['nurse_hourly']}/hr</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value blue">{stats['avg_support']}</div>
            <div class="kpi-label">Support Staff/Day</div>
            <div class="kpi-delta">@ ${cost_params['support_hourly']}/hr</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_staff = stats['avg_doctors'] + stats['avg_nurses'] + stats['avg_support']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value green">{total_staff}</div>
            <div class="kpi-label">Total Staff/Day</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI Cards Row 2: Financial & Efficiency
    st.markdown("### 💰 Financial & Operational Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Calculate daily cost
    daily_labor_cost = (
        stats['avg_doctors'] * cost_params['doctor_hourly'] * cost_params['shift_hours'] +
        stats['avg_nurses'] * cost_params['nurse_hourly'] * cost_params['shift_hours'] +
        stats['avg_support'] * cost_params['support_hourly'] * cost_params['shift_hours']
    )

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value yellow">${daily_labor_cost:,.0f}</div>
            <div class="kpi-label">Daily Labor Cost</div>
            <div class="kpi-delta">${daily_labor_cost * 30:,.0f}/month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        utilization = stats['avg_utilization']
        card_class = "success" if utilization >= 80 else "warning" if utilization >= 60 else "danger"
        st.markdown(f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-value {'green' if utilization >= 80 else 'yellow' if utilization >= 60 else 'red'}">{utilization:.1f}%</div>
            <div class="kpi-label">Staff Utilization</div>
            <div class="kpi-delta">{'Good' if utilization >= 80 else 'Needs Improvement' if utilization >= 60 else 'Critical'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        overtime = stats['total_overtime']
        overtime_cost = overtime * cost_params['nurse_hourly'] * cost_params['overtime_multiplier']
        card_class = "danger" if overtime > 100 else "warning" if overtime > 50 else ""
        st.markdown(f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-value {'red' if overtime > 100 else 'yellow' if overtime > 50 else 'blue'}">{overtime:.0f}h</div>
            <div class="kpi-label">Total Overtime Hours</div>
            <div class="kpi-delta negative">${overtime_cost:,.0f} extra cost</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        shortage = stats['shortage_days']
        shortage_pct = (shortage / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
        card_class = "danger" if shortage_pct > 20 else "warning" if shortage_pct > 10 else "success"
        st.markdown(f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-value {'red' if shortage_pct > 20 else 'yellow' if shortage_pct > 10 else 'green'}">{shortage}</div>
            <div class="kpi-label">Staff Shortage Days</div>
            <div class="kpi-delta {'negative' if shortage > 0 else ''}">{shortage_pct:.1f}% of days</div>
        </div>
        """, unsafe_allow_html=True)

    # Issues Summary
    st.markdown("### ⚠️ Current Issues & Alerts")

    issues = []
    if utilization < 70:
        issues.append(("danger", f"LOW UTILIZATION: Staff utilization at {utilization:.1f}% - potential overstaffing"))
    elif utilization < 80:
        issues.append(("warning", f"MODERATE UTILIZATION: Staff utilization at {utilization:.1f}% - room for improvement"))

    if overtime > 100:
        issues.append(("danger", f"HIGH OVERTIME: {overtime:.0f} hours of overtime costing ${overtime_cost:,.0f}"))
    elif overtime > 50:
        issues.append(("warning", f"MODERATE OVERTIME: {overtime:.0f} hours of overtime"))

    if shortage > stats['total_records'] * 0.2:
        issues.append(("danger", f"FREQUENT SHORTAGES: {shortage} days ({shortage_pct:.1f}%) with staff shortage"))
    elif shortage > stats['total_records'] * 0.1:
        issues.append(("warning", f"OCCASIONAL SHORTAGES: {shortage} days ({shortage_pct:.1f}%) with staff shortage"))

    if not issues:
        issues.append(("success", "All metrics within acceptable ranges. Hospital staffing is well-managed."))

    for level, msg in issues:
        st.markdown(f'<div class="alert-box {level}">{msg}</div>', unsafe_allow_html=True)

    # Data Table
    with st.expander("📋 View Raw Staff Data", expanded=False):
        if st.session_state.staff_data["data"] is not None:
            st.dataframe(st.session_state.staff_data["data"].tail(30), use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # STEP 3: OPTIMIZE WITH FORECAST
    # ==========================================================================
    forecast_data = get_forecast_data()

    step3_class = "completed" if step3_complete else "active" if forecast_data["has_forecast"] else ""
    st.markdown(f"""
    <div class="step-card {step3_class}">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span class="step-number {'completed' if step3_complete else ''}">{'✓' if step3_complete else '3'}</span>
            <span class="step-title">Optimize Staffing with Forecast</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not forecast_data["has_forecast"]:
        st.warning("""
        ⚠️ **No forecast data available!**

        Please run forecasting models in the **Modeling Hub (Page 08)** first, then return here to optimize staffing.
        """)
    else:
        st.markdown(f"""
        <div class="alert-box success">
            ✅ Forecast Available: <strong>{forecast_data['source']}</strong> - {forecast_data['horizon']} days ahead
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🚀 Apply Forecast Optimization", type="primary", use_container_width=True):
                with st.spinner("Calculating optimized schedule..."):
                    optimized = calculate_optimized_schedule(forecast_data, cost_params)

                    if optimized["success"]:
                        st.session_state.optimized_schedule = optimized
                        st.session_state.optimization_applied = True
                        st.success("✅ Optimization complete!")
                        st.rerun()
                    else:
                        st.error(f"❌ Optimization failed: {optimized.get('error', 'Unknown error')}")

        with col2:
            if st.session_state.optimization_applied:
                st.success("✅ Optimization applied - see results below")
            else:
                st.info("Click to generate optimized schedule based on patient forecast")

    # ==========================================================================
    # OPTIMIZATION RESULTS
    # ==========================================================================
    if st.session_state.optimization_applied and st.session_state.optimized_schedule:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Optimization Results: Before vs After</div>', unsafe_allow_html=True)

        optimized = st.session_state.optimized_schedule
        current_daily_cost = daily_labor_cost
        optimized_daily_cost = optimized["avg_daily_cost"]

        # Financial Impact Summary
        weekly_before = current_daily_cost * 7
        weekly_after = optimized["total_cost"]
        weekly_savings = weekly_before - weekly_after
        savings_pct = (weekly_savings / weekly_before * 100) if weekly_before > 0 else 0

        st.markdown("### 💰 Financial Impact (Weekly)")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="kpi-card danger">
                <div class="kpi-value red">${weekly_before:,.0f}</div>
                <div class="kpi-label">BEFORE (Current)</div>
                <div class="kpi-delta">Fixed staffing</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card success">
                <div class="kpi-value green">${weekly_after:,.0f}</div>
                <div class="kpi-label">AFTER (Optimized)</div>
                <div class="kpi-delta">Forecast-driven</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value yellow">${weekly_savings:,.0f}</div>
                <div class="kpi-label">Weekly Savings</div>
                <div class="kpi-delta positive">{savings_pct:.1f}% reduction</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            annual_savings = weekly_savings * 52
            st.markdown(f"""
            <div class="kpi-card success">
                <div class="kpi-value green">${annual_savings:,.0f}</div>
                <div class="kpi-label">Annual Savings</div>
                <div class="kpi-delta positive">Projected</div>
            </div>
            """, unsafe_allow_html=True)

        # Optimized Schedule Table
        st.markdown("### 📅 Optimized Staff Schedule")
        schedule_df = pd.DataFrame(optimized["schedules"])
        schedule_df["daily_cost"] = schedule_df["daily_cost"].apply(lambda x: f"${x:,.0f}")
        schedule_df["expected_patients"] = schedule_df["expected_patients"].apply(lambda x: f"{x:.0f}")
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        # Comparison Chart
        st.markdown("### 📈 Cost Comparison")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Current (Fixed)', 'Optimized (Forecast)'],
            y=[weekly_before, weekly_after],
            marker_color=['#ef4444', '#22c55e'],
            text=[f'${weekly_before:,.0f}', f'${weekly_after:,.0f}'],
            textposition='auto',
        ))

        fig.update_layout(
            title="Weekly Staffing Cost: Before vs After Optimization",
            yaxis_title="Cost ($)",
            template="plotly_dark",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.success(f"""
        ### 🎉 Optimization Summary

        By applying **forecast-driven staffing** using **{optimized['source']}** predictions:

        - **Weekly savings:** ${weekly_savings:,.0f} ({savings_pct:.1f}% reduction)
        - **Annual savings:** ${annual_savings:,.0f}
        - **Staff aligned to demand:** {optimized['horizon']} days optimized
        - **Zero overtime expected:** Staff matched to predicted patient volume
        """)

else:
    # No data loaded yet
    st.info("""
    ### 👆 Start Here

    Click **"Load Staff & Financial Data"** above to fetch real hospital data from Supabase.

    Once loaded, you'll see:
    1. **Current Hospital Situation** - Expert manager view of staffing and costs
    2. **Optimization Options** - Apply ML forecasts to optimize staffing
    """)


# =============================================================================
# SIDEBAR: CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    with st.expander("Staffing Ratios", expanded=False):
        st.caption("Patients per staff member by category")
        for cat, ratios in STAFFING_RATIOS.items():
            st.text(f"{cat}: Dr={ratios['doctors']}, RN={ratios['nurses']}")

    with st.expander("Cost Parameters", expanded=False):
        params = st.session_state.cost_params
        st.caption("Hourly rates (USD)")
        st.text(f"Doctor: ${params['doctor_hourly']}/hr")
        st.text(f"Nurse: ${params['nurse_hourly']}/hr")
        st.text(f"Support: ${params['support_hourly']}/hr")
        st.text(f"Overtime: {params['overtime_multiplier']}x")

    if st.session_state.staff_db_loaded:
        with st.expander("Data Status", expanded=False):
            stats = st.session_state.staff_data["stats"]
            st.text(f"Records: {stats['total_records']}")
            st.text(f"Avg Doctors: {stats['avg_doctors']}")
            st.text(f"Avg Nurses: {stats['avg_nurses']}")
            st.text(f"Avg Support: {stats['avg_support']}")
