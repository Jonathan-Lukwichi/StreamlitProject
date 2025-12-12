# =============================================================================
# 11_Staff_Scheduling_Optimization.py
# REDESIGNED: 4-Tab Structure with Data Upload, Situation Analysis, Optimization
# =============================================================================
"""
Tab Structure:
1. Data Upload & Preview - Load Staff + Financial data with KPIs & graphs
2. Current Hospital Situation - Summarize & link staff + financial metrics
3. After Optimization - Show forecast-adjusted parameters
4. Before vs After - Compare results

Data Sources:
- Staff data: Supabase staff_scheduling table
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
from app_core.data.staff_scheduling_service import StaffSchedulingService
from app_core.data.financial_service import FinancialService, get_financial_service

# ============================================================================
# AUTHENTICATION
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
# CONFIGURATION
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

DEFAULT_COST_PARAMS = {
    "doctor_hourly": 150,
    "nurse_hourly": 50,
    "support_hourly": 25,
    "shift_hours": 8,
    "overtime_multiplier": 1.5,
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

.data-box.staff {{
    border-color: rgba(34, 197, 94, 0.3);
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
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22c55e;
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
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>👥</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Staff Scheduling Optimization</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Load hospital data, analyze current situation, apply forecast optimization, and compare results
      </p>
    </div>
    """,
    unsafe_allow_html=True,
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

        # Calculate raw stats
        raw_avg_doctors = df["Doctors_on_Duty"].mean() if "Doctors_on_Duty" in df.columns else 0
        raw_avg_nurses = df["Nurses_on_Duty"].mean() if "Nurses_on_Duty" in df.columns else 0
        raw_avg_support = df["Support_Staff_on_Duty"].mean() if "Support_Staff_on_Duty" in df.columns else 0
        raw_total_overtime = df["Overtime_Hours"].sum() if "Overtime_Hours" in df.columns else 0
        raw_avg_utilization = df["Staff_Utilization_Rate"].mean() if "Staff_Utilization_Rate" in df.columns else 0
        raw_shortage_days = df["Staff_Shortage_Flag"].sum() if "Staff_Shortage_Flag" in df.columns else 0

        # Validate stats - ensure no negative values
        # Utilization should be 0-100%, others should be >= 0
        result["stats"] = {
            "total_records": len(df),
            "date_start": df["Date"].min() if "Date" in df.columns else None,
            "date_end": df["Date"].max() if "Date" in df.columns else None,
            "avg_doctors": max(0, raw_avg_doctors) if raw_avg_doctors else 0,
            "avg_nurses": max(0, raw_avg_nurses) if raw_avg_nurses else 0,
            "avg_support": max(0, raw_avg_support) if raw_avg_support else 0,
            "total_overtime": max(0, raw_total_overtime) if raw_total_overtime else 0,
            "avg_utilization": max(0, min(100, abs(raw_avg_utilization))) if raw_avg_utilization else 0,
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

        df = service.fetch_all()
        labor_params = service.get_labor_cost_params()
        kpis = service.get_comprehensive_financial_kpis()

        if labor_params:
            result["cost_params"] = {
                "doctor_hourly": labor_params.get("doctor_hourly_rate", 150),
                "nurse_hourly": labor_params.get("nurse_hourly_rate", 50),
                "support_hourly": labor_params.get("support_hourly_rate", 25),
                "shift_hours": 8,
                "overtime_multiplier": labor_params.get("overtime_multiplier", 1.5),
            }

        result["success"] = True
        result["data"] = df if not df.empty else None

        # Calculate stats
        result["stats"] = {
            "total_records": len(df) if not df.empty else 0,
            "doctor_hourly": result["cost_params"]["doctor_hourly"],
            "nurse_hourly": result["cost_params"]["nurse_hourly"],
            "support_hourly": result["cost_params"]["support_hourly"],
            "overtime_multiplier": result["cost_params"]["overtime_multiplier"],
            "daily_budget": kpis.get("avg_daily_labor_cost", 0) if kpis else 0,
            "monthly_budget": kpis.get("monthly_labor_budget", 0) if kpis else 0,
            "annual_budget": kpis.get("annual_labor_budget", 0) if kpis else 0,
        }

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def get_forecast_data() -> Dict[str, Any]:
    """Extract forecast data from session state."""
    result = {"has_forecast": False, "source": None, "forecasts": [], "horizon": 0, "dates": []}

    # Try ML models
    ml_results = st.session_state.get("ml_mh_results")
    if ml_results and isinstance(ml_results, dict):
        for model in ["XGBoost", "LSTM", "ANN"]:
            if model in ml_results:
                per_h = ml_results[model].get("per_h", {})
                if per_h:
                    try:
                        horizons = sorted([int(h) for h in per_h.keys()])
                        forecasts = []
                        for h in horizons[:7]:
                            pred = per_h.get(h, {}).get("predictions") or per_h.get(h, {}).get("forecast")
                            if pred is not None:
                                arr = pred.values if hasattr(pred, 'values') else np.array(pred)
                                forecasts.append(float(arr[-1]) if len(arr) > 0 else 0)
                        if forecasts:
                            result["has_forecast"] = True
                            result["source"] = f"ML ({model})"
                            result["forecasts"] = forecasts
                            result["horizon"] = len(forecasts)
                            break
                    except:
                        continue

    # Try ARIMA
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
                            forecasts.append(float(arr[-1]) if len(arr) > 0 else 0)
                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = "ARIMA"
                        result["forecasts"] = forecasts
                        result["horizon"] = len(forecasts)
                except:
                    pass

    if result["has_forecast"]:
        today = datetime.now().date()
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, result["horizon"] + 1)]

    return result


def calculate_optimization(forecast_data: Dict, staff_stats: Dict, cost_params: Dict) -> Dict[str, Any]:
    """Calculate optimized staffing based on forecast."""
    if not forecast_data["has_forecast"]:
        return {"success": False}

    schedules = []
    total_cost = 0

    # Patient distribution by category
    distribution = {
        "CARDIAC": 0.15, "TRAUMA": 0.20, "RESPIRATORY": 0.25,
        "GASTROINTESTINAL": 0.15, "INFECTIOUS": 0.15, "NEUROLOGICAL": 0.05, "OTHER": 0.05,
    }

    for i, patients in enumerate(forecast_data["forecasts"]):
        patients_by_cat = {cat: patients * pct for cat, pct in distribution.items()}

        doctors = sum(np.ceil(p / STAFFING_RATIOS[c]["doctors"]) for c, p in patients_by_cat.items())
        nurses = sum(np.ceil(p / STAFFING_RATIOS[c]["nurses"]) for c, p in patients_by_cat.items())
        support = sum(np.ceil(p / STAFFING_RATIOS[c]["support"]) for c, p in patients_by_cat.items())

        daily_cost = (
            doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
            nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
            support * cost_params["support_hourly"] * cost_params["shift_hours"]
        )

        schedules.append({
            "date": forecast_data["dates"][i],
            "patients": patients,
            "doctors": int(doctors),
            "nurses": int(nurses),
            "support": int(support),
            "total_staff": int(doctors + nurses + support),
            "cost": daily_cost,
        })
        total_cost += daily_cost

    # Calculate current costs for comparison
    current_daily_cost = (
        staff_stats.get("avg_doctors", 0) * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
        staff_stats.get("avg_nurses", 0) * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
        staff_stats.get("avg_support", 0) * cost_params["support_hourly"] * cost_params["shift_hours"]
    )

    horizon = forecast_data["horizon"]

    return {
        "success": True,
        "schedules": schedules,
        "total_cost": total_cost,
        "avg_daily_cost": total_cost / horizon,
        "current_daily_cost": current_daily_cost,
        "current_weekly_cost": current_daily_cost * 7,
        "optimized_weekly_cost": total_cost,
        "weekly_savings": (current_daily_cost * 7) - total_cost,
        "horizon": horizon,
        "source": forecast_data["source"],
        "avg_opt_doctors": np.mean([s["doctors"] for s in schedules]),
        "avg_opt_nurses": np.mean([s["nurses"] for s in schedules]),
        "avg_opt_support": np.mean([s["support"] for s in schedules]),
    }


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Upload & Preview",
    "🏥 Current Hospital Situation",
    "🚀 After Optimization",
    "📈 Before vs After Comparison"
])


# =============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">📊 Data Upload & Preview</div>', unsafe_allow_html=True)

    # Two columns for Staff and Financial
    col_staff, col_fin = st.columns(2)

    # =========================================================================
    # STAFF DATA SECTION
    # =========================================================================
    with col_staff:
        st.markdown('<div class="subsection-header">👥 Staff Scheduling Data</div>', unsafe_allow_html=True)

        # Load Button
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

        # Status
        if st.session_state.staff_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Staff Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.staff_stats

            # Description
            st.markdown(f"""
            <div class="data-box staff">
                <strong>Dataset Description:</strong><br>
                • <strong>{stats['total_records']}</strong> daily records<br>
                • Period: <strong>{stats['date_start'].strftime('%Y-%m-%d') if stats['date_start'] else 'N/A'}</strong> to <strong>{stats['date_end'].strftime('%Y-%m-%d') if stats['date_end'] else 'N/A'}</strong><br>
                • Contains: Doctors, Nurses, Support Staff, Overtime, Utilization
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            st.markdown("**Key Metrics:**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card staff">
                    <div class="kpi-value green">{stats['avg_doctors']:.0f}</div>
                    <div class="kpi-label">Avg Doctors/Day</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card staff">
                    <div class="kpi-value green">{stats['avg_nurses']:.0f}</div>
                    <div class="kpi-label">Avg Nurses/Day</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card staff">
                    <div class="kpi-value green">{stats['avg_support']:.0f}</div>
                    <div class="kpi-label">Avg Support/Day</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card {'warning' if stats['avg_utilization'] < 80 else 'success'}">
                    <div class="kpi-value {'yellow' if stats['avg_utilization'] < 80 else 'green'}">{stats['avg_utilization']:.1f}%</div>
                    <div class="kpi-label">Avg Utilization</div>
                </div>
                """, unsafe_allow_html=True)

            # Graph: Staffing Trends
            if st.session_state.staff_df is not None:
                df = st.session_state.staff_df.copy()
                if "Date" in df.columns:
                    df = df.sort_values("Date").tail(30)

                    fig = go.Figure()
                    if "Doctors_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Doctors_on_Duty"], name="Doctors", line=dict(color="#3b82f6")))
                    if "Nurses_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Nurses_on_Duty"], name="Nurses", line=dict(color="#22c55e")))
                    if "Support_Staff_on_Duty" in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df["Support_Staff_on_Duty"], name="Support", line=dict(color="#f97316")))

                    fig.update_layout(title="Staff Levels (Last 30 Days)", template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # Observations
            observations = []
            if stats['avg_utilization'] < 70:
                observations.append(("danger", f"⚠️ Low utilization ({stats['avg_utilization']:.1f}%) - potential overstaffing"))
            elif stats['avg_utilization'] < 80:
                observations.append(("warning", f"📊 Moderate utilization ({stats['avg_utilization']:.1f}%) - room for improvement"))

            if stats['total_overtime'] > 100:
                observations.append(("danger", f"⏰ High overtime: {stats['total_overtime']:.0f} hours total"))

            shortage_pct = (stats['shortage_days'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
            if shortage_pct > 10:
                observations.append(("warning", f"📉 {stats['shortage_days']} shortage days ({shortage_pct:.1f}%)"))

            if observations:
                st.markdown("**Observations:**")
                for obs_type, obs_text in observations:
                    st.markdown(f'<div class="observation-box {obs_type}">{obs_text}</div>', unsafe_allow_html=True)

            # Raw Data Preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(st.session_state.staff_df.tail(20), use_container_width=True, height=200)

        else:
            st.markdown('<span class="status-badge pending">⚠️ Not Loaded</span>', unsafe_allow_html=True)
            st.info("Click button above to load staff scheduling data from Supabase")

    # =========================================================================
    # FINANCIAL DATA SECTION
    # =========================================================================
    with col_fin:
        st.markdown('<div class="subsection-header">💰 Financial Data</div>', unsafe_allow_html=True)

        # Load Button
        if st.button("🔄 Load Financial Data", type="primary", use_container_width=True, key="load_fin"):
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

        # Status
        if st.session_state.financial_data_loaded:
            st.markdown('<span class="status-badge loaded">✅ Financial Data Loaded</span>', unsafe_allow_html=True)

            stats = st.session_state.financial_stats
            cost_params = st.session_state.cost_params

            # Description
            st.markdown(f"""
            <div class="data-box financial">
                <strong>Dataset Description:</strong><br>
                • Labor cost parameters from Supabase<br>
                • Contains: Hourly rates, budgets, cost multipliers<br>
                • Records: <strong>{stats['total_records']}</strong> entries
            </div>
            """, unsafe_allow_html=True)

            # KPI Cards
            st.markdown("**Cost Parameters:**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['doctor_hourly']:.0f}</div>
                    <div class="kpi-label">Doctor Hourly Rate</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['nurse_hourly']:.0f}</div>
                    <div class="kpi-label">Nurse Hourly Rate</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value blue">${cost_params['support_hourly']:.0f}</div>
                    <div class="kpi-label">Support Hourly Rate</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card financial">
                    <div class="kpi-value yellow">{cost_params['overtime_multiplier']:.2f}x</div>
                    <div class="kpi-label">Overtime Multiplier</div>
                </div>
                """, unsafe_allow_html=True)

            # Budget KPIs
            if stats.get("daily_budget", 0) > 0:
                st.markdown("**Budget Overview:**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value green">${stats['daily_budget']:,.0f}</div>
                        <div class="kpi-label">Daily Budget</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value green">${stats['monthly_budget']:,.0f}</div>
                        <div class="kpi-label">Monthly Budget</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value green">${stats['annual_budget']:,.0f}</div>
                        <div class="kpi-label">Annual Budget</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Cost Distribution Chart
            if st.session_state.staff_data_loaded:
                staff_stats = st.session_state.staff_stats
                doc_cost = staff_stats['avg_doctors'] * cost_params['doctor_hourly'] * cost_params['shift_hours']
                nurse_cost = staff_stats['avg_nurses'] * cost_params['nurse_hourly'] * cost_params['shift_hours']
                support_cost = staff_stats['avg_support'] * cost_params['support_hourly'] * cost_params['shift_hours']

                fig = go.Figure(data=[go.Pie(
                    labels=['Doctors', 'Nurses', 'Support'],
                    values=[doc_cost, nurse_cost, support_cost],
                    hole=0.4,
                    marker_colors=['#3b82f6', '#22c55e', '#f97316']
                )])
                fig.update_layout(title="Daily Cost Distribution", template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Observations
            st.markdown("**Observations:**")
            st.markdown(f'<div class="observation-box success">💵 Total daily labor cost: ${doc_cost + nurse_cost + support_cost:,.0f}</div>', unsafe_allow_html=True)

            # Raw Data Preview
            if st.session_state.financial_df is not None:
                with st.expander("📋 View Raw Data"):
                    st.dataframe(st.session_state.financial_df.tail(20), use_container_width=True, height=200)

        else:
            st.markdown('<span class="status-badge pending">⚠️ Not Loaded</span>', unsafe_allow_html=True)
            st.info("Click button above to load financial data from Supabase")


# =============================================================================
# TAB 2: CURRENT HOSPITAL SITUATION
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">🏥 Current Hospital Situation</div>', unsafe_allow_html=True)

    if st.session_state.staff_data_loaded and st.session_state.financial_data_loaded:
        staff_stats = st.session_state.staff_stats
        fin_stats = st.session_state.financial_stats
        cost_params = st.session_state.cost_params

        # Calculate linked variables
        daily_labor_cost = (
            staff_stats['avg_doctors'] * cost_params['doctor_hourly'] * cost_params['shift_hours'] +
            staff_stats['avg_nurses'] * cost_params['nurse_hourly'] * cost_params['shift_hours'] +
            staff_stats['avg_support'] * cost_params['support_hourly'] * cost_params['shift_hours']
        )
        weekly_labor_cost = daily_labor_cost * 7
        monthly_labor_cost = daily_labor_cost * 30
        overtime_cost = staff_stats['total_overtime'] * cost_params['nurse_hourly'] * cost_params['overtime_multiplier']
        total_staff = staff_stats['avg_doctors'] + staff_stats['avg_nurses'] + staff_stats['avg_support']
        cost_per_staff = daily_labor_cost / total_staff if total_staff > 0 else 0
        shortage_penalty = staff_stats['shortage_days'] * 200  # $200 penalty per shortage day

        # STAFF SITUATION SUMMARY
        st.markdown('<div class="subsection-header">👥 Staff Situation Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card staff">
                <div class="kpi-value blue">{staff_stats['avg_doctors']:.0f}</div>
                <div class="kpi-label">Avg Doctors/Day</div>
                <div class="kpi-delta">Range: {staff_stats['min_doctors']:.0f} - {staff_stats['max_doctors']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card staff">
                <div class="kpi-value blue">{staff_stats['avg_nurses']:.0f}</div>
                <div class="kpi-label">Avg Nurses/Day</div>
                <div class="kpi-delta">Range: {staff_stats['min_nurses']:.0f} - {staff_stats['max_nurses']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card staff">
                <div class="kpi-value blue">{staff_stats['avg_support']:.0f}</div>
                <div class="kpi-label">Avg Support/Day</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            util_class = "success" if staff_stats['avg_utilization'] >= 80 else "warning" if staff_stats['avg_utilization'] >= 60 else "danger"
            st.markdown(f"""
            <div class="kpi-card {util_class}">
                <div class="kpi-value {'green' if staff_stats['avg_utilization'] >= 80 else 'yellow' if staff_stats['avg_utilization'] >= 60 else 'red'}">{staff_stats['avg_utilization']:.1f}%</div>
                <div class="kpi-label">Staff Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        # FINANCIAL SITUATION SUMMARY
        st.markdown('<div class="subsection-header">💰 Financial Situation Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${daily_labor_cost:,.0f}</div>
                <div class="kpi-label">Daily Labor Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${weekly_labor_cost:,.0f}</div>
                <div class="kpi-label">Weekly Labor Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card financial">
                <div class="kpi-value blue">${monthly_labor_cost:,.0f}</div>
                <div class="kpi-label">Monthly Labor Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            ot_class = "danger" if overtime_cost > 5000 else "warning" if overtime_cost > 2000 else ""
            st.markdown(f"""
            <div class="kpi-card {ot_class}">
                <div class="kpi-value {'red' if overtime_cost > 5000 else 'yellow' if overtime_cost > 2000 else 'blue'}">${overtime_cost:,.0f}</div>
                <div class="kpi-label">Total Overtime Cost</div>
            </div>
            """, unsafe_allow_html=True)

        # LINKED VARIABLES (Staff × Financial)
        st.markdown('<div class="subsection-header">🔗 Linked Metrics (Staff × Financial)</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value purple">${cost_per_staff:,.0f}</div>
                <div class="kpi-label">Cost per Staff/Day</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            efficiency = (staff_stats['avg_utilization'] / 100) * (1 - staff_stats['shortage_days'] / staff_stats['total_records']) * 100
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value purple">{efficiency:.1f}%</div>
                <div class="kpi-label">Labor Efficiency Index</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value purple">{staff_stats['total_overtime']:.0f}h</div>
                <div class="kpi-label">Total Overtime Hours</div>
                <div class="kpi-delta negative">${overtime_cost:,.0f} extra</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card linked">
                <div class="kpi-value purple">{staff_stats['shortage_days']}</div>
                <div class="kpi-label">Shortage Days</div>
                <div class="kpi-delta negative">${shortage_penalty:,.0f} penalty</div>
            </div>
            """, unsafe_allow_html=True)

        # PARAMETERS TO OPTIMIZE
        st.markdown('<div class="subsection-header">🎯 Parameters to Optimize</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="kpi-card {'danger' if staff_stats['total_overtime'] > 100 else 'warning' if staff_stats['total_overtime'] > 50 else 'success'}">
                <div class="kpi-value {'red' if staff_stats['total_overtime'] > 100 else 'yellow' if staff_stats['total_overtime'] > 50 else 'green'}">{staff_stats['total_overtime']:.0f}h</div>
                <div class="kpi-label">🎯 Reduce Overtime</div>
                <div class="kpi-delta">Target: &lt;50h</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            shortage_pct = staff_stats['shortage_days'] / staff_stats['total_records'] * 100
            st.markdown(f"""
            <div class="kpi-card {'danger' if shortage_pct > 15 else 'warning' if shortage_pct > 5 else 'success'}">
                <div class="kpi-value {'red' if shortage_pct > 15 else 'yellow' if shortage_pct > 5 else 'green'}">{shortage_pct:.1f}%</div>
                <div class="kpi-label">🎯 Reduce Shortages</div>
                <div class="kpi-delta">Target: &lt;5%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card {'danger' if staff_stats['avg_utilization'] < 60 else 'warning' if staff_stats['avg_utilization'] < 80 else 'success'}">
                <div class="kpi-value {'red' if staff_stats['avg_utilization'] < 60 else 'yellow' if staff_stats['avg_utilization'] < 80 else 'green'}">{staff_stats['avg_utilization']:.1f}%</div>
                <div class="kpi-label">🎯 Improve Utilization</div>
                <div class="kpi-delta">Target: &gt;85%</div>
            </div>
            """, unsafe_allow_html=True)

        # VISUALIZATION
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Cost breakdown bar chart
            fig = go.Figure(data=[go.Bar(
                x=['Doctors', 'Nurses', 'Support', 'Overtime'],
                y=[
                    staff_stats['avg_doctors'] * cost_params['doctor_hourly'] * cost_params['shift_hours'],
                    staff_stats['avg_nurses'] * cost_params['nurse_hourly'] * cost_params['shift_hours'],
                    staff_stats['avg_support'] * cost_params['support_hourly'] * cost_params['shift_hours'],
                    overtime_cost / staff_stats['total_records']  # Daily average overtime cost
                ],
                marker_color=['#3b82f6', '#22c55e', '#f97316', '#ef4444'],
                text=[f'${v:,.0f}' for v in [
                    staff_stats['avg_doctors'] * cost_params['doctor_hourly'] * cost_params['shift_hours'],
                    staff_stats['avg_nurses'] * cost_params['nurse_hourly'] * cost_params['shift_hours'],
                    staff_stats['avg_support'] * cost_params['support_hourly'] * cost_params['shift_hours'],
                    overtime_cost / staff_stats['total_records']
                ]],
                textposition='auto',
            )])
            fig.update_layout(title="Daily Cost by Category", yaxis_title="Cost ($)", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Utilization gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=staff_stats['avg_utilization'],
                title={'text': "Staff Utilization"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#22c55e" if staff_stats['avg_utilization'] >= 80 else "#eab308" if staff_stats['avg_utilization'] >= 60 else "#ef4444"},
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(239,68,68,0.2)"},
                        {'range': [60, 80], 'color': "rgba(234,179,8,0.2)"},
                        {'range': [80, 100], 'color': "rgba(34,197,94,0.2)"},
                    ],
                    'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 85}
                }
            ))
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please load both **Staff Data** and **Financial Data** in the first tab to see the current situation analysis.")


# =============================================================================
# TAB 3: AFTER OPTIMIZATION
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">🚀 After Optimization (Forecast Applied)</div>', unsafe_allow_html=True)

    if st.session_state.staff_data_loaded and st.session_state.financial_data_loaded:
        forecast_data = get_forecast_data()

        if not forecast_data["has_forecast"]:
            st.warning("""
            ⚠️ **No forecast data available!**

            Please run forecasting models in the **Modeling Hub (Page 08)** first, then return here.
            """)
        else:
            st.success(f"✅ Forecast available: **{forecast_data['source']}** - {forecast_data['horizon']} days ahead")

            # Optimization Button
            if st.button("🚀 Apply Forecast Optimization", type="primary", use_container_width=True, key="optimize"):
                with st.spinner("Optimizing staffing based on forecast..."):
                    result = calculate_optimization(forecast_data, st.session_state.staff_stats, st.session_state.cost_params)
                    if result["success"]:
                        st.session_state.optimized_results = result
                        st.session_state.optimization_done = True
                        st.success("✅ Optimization complete!")
                        st.rerun()

            # Show Results
            if st.session_state.optimization_done and st.session_state.optimized_results:
                opt = st.session_state.optimized_results
                staff_stats = st.session_state.staff_stats
                cost_params = st.session_state.cost_params

                st.markdown("---")

                # ADJUSTED STAFF PARAMETERS
                st.markdown('<div class="subsection-header">👥 Adjusted Staff Parameters</div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    diff = opt['avg_opt_doctors'] - staff_stats['avg_doctors']
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">{opt['avg_opt_doctors']:.0f}</div>
                        <div class="kpi-label">Optimized Doctors/Day</div>
                        <div class="kpi-delta {'positive' if diff <= 0 else 'negative'}">{'+' if diff > 0 else ''}{diff:.0f} from current</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    diff = opt['avg_opt_nurses'] - staff_stats['avg_nurses']
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">{opt['avg_opt_nurses']:.0f}</div>
                        <div class="kpi-label">Optimized Nurses/Day</div>
                        <div class="kpi-delta {'positive' if diff <= 0 else 'negative'}">{'+' if diff > 0 else ''}{diff:.0f} from current</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    diff = opt['avg_opt_support'] - staff_stats['avg_support']
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">{opt['avg_opt_support']:.0f}</div>
                        <div class="kpi-label">Optimized Support/Day</div>
                        <div class="kpi-delta {'positive' if diff <= 0 else 'negative'}">{'+' if diff > 0 else ''}{diff:.0f} from current</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">95%</div>
                        <div class="kpi-label">Expected Utilization</div>
                        <div class="kpi-delta positive">+{95 - staff_stats['avg_utilization']:.1f}% improvement</div>
                    </div>
                    """, unsafe_allow_html=True)

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
                        <div class="kpi-value green">0h</div>
                        <div class="kpi-label">Expected Overtime</div>
                        <div class="kpi-delta positive">Eliminated</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="kpi-card success">
                        <div class="kpi-value green">0</div>
                        <div class="kpi-label">Expected Shortages</div>
                        <div class="kpi-delta positive">Eliminated</div>
                    </div>
                    """, unsafe_allow_html=True)

                # OPTIMIZED SCHEDULE TABLE
                st.markdown('<div class="subsection-header">📅 Optimized Schedule</div>', unsafe_allow_html=True)
                sched_df = pd.DataFrame(opt["schedules"])
                sched_df["cost"] = sched_df["cost"].apply(lambda x: f"${x:,.0f}")
                sched_df["patients"] = sched_df["patients"].apply(lambda x: f"{x:.0f}")
                st.dataframe(sched_df, use_container_width=True, hide_index=True)

                # GRAPHS
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    # Optimized staff allocation
                    sched = pd.DataFrame(opt["schedules"])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=sched["date"], y=sched["doctors"], name="Doctors", marker_color="#3b82f6"))
                    fig.add_trace(go.Bar(x=sched["date"], y=sched["nurses"], name="Nurses", marker_color="#22c55e"))
                    fig.add_trace(go.Bar(x=sched["date"], y=sched["support"], name="Support", marker_color="#f97316"))
                    fig.update_layout(barmode="stack", title="Optimized Staff Allocation", template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Daily cost trend
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sched["date"],
                        y=[s["cost"] for s in opt["schedules"]],
                        mode='lines+markers',
                        name='Daily Cost',
                        line=dict(color='#22c55e', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(34,197,94,0.2)'
                    ))
                    fig.add_hline(y=opt['current_daily_cost'], line_dash="dash", line_color="#ef4444", annotation_text="Current Avg")
                    fig.update_layout(title="Optimized Daily Cost", yaxis_title="Cost ($)", template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please load both **Staff Data** and **Financial Data** in the first tab first.")


# =============================================================================
# TAB 4: BEFORE VS AFTER COMPARISON
# =============================================================================
with tab4:
    st.markdown('<div class="section-header">📈 Before vs After Comparison</div>', unsafe_allow_html=True)

    if st.session_state.optimization_done and st.session_state.optimized_results:
        opt = st.session_state.optimized_results
        staff_stats = st.session_state.staff_stats
        cost_params = st.session_state.cost_params

        # Calculate values
        overtime_cost = staff_stats['total_overtime'] * cost_params['nurse_hourly'] * cost_params['overtime_multiplier']
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
                marker_color=['#ef4444', '#22c55e'],
                text=[f"${opt['current_weekly_cost']:,.0f}", f"${opt['optimized_weekly_cost']:,.0f}"],
                textposition='auto',
            ))
            fig.update_layout(title="Weekly Cost Comparison", yaxis_title="Cost ($)", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Staff comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(name='BEFORE', x=['Doctors', 'Nurses', 'Support'],
                                  y=[staff_stats['avg_doctors'], staff_stats['avg_nurses'], staff_stats['avg_support']],
                                  marker_color='#ef4444'))
            fig.add_trace(go.Bar(name='AFTER', x=['Doctors', 'Nurses', 'Support'],
                                  y=[opt['avg_opt_doctors'], opt['avg_opt_nurses'], opt['avg_opt_support']],
                                  marker_color='#22c55e'))
            fig.update_layout(barmode='group', title="Staff Levels Comparison", yaxis_title="Count", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Efficiency gauges
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=staff_stats['avg_utilization'],
                title={'text': "BEFORE: Utilization"},
                number={'suffix': '%'},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ef4444"},
                       'steps': [{'range': [0, 60], 'color': "rgba(239,68,68,0.2)"},
                                {'range': [60, 80], 'color': "rgba(234,179,8,0.2)"},
                                {'range': [80, 100], 'color': "rgba(34,197,94,0.2)"}]}
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=95,
                title={'text': "AFTER: Utilization"},
                number={'suffix': '%'},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#22c55e"},
                       'steps': [{'range': [0, 60], 'color': "rgba(239,68,68,0.2)"},
                                {'range': [60, 80], 'color': "rgba(234,179,8,0.2)"},
                                {'range': [80, 100], 'color': "rgba(34,197,94,0.2)"}]}
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # SUMMARY
        st.success(f"""
        ### 🎉 Optimization Summary

        By applying **{opt['source']}** forecast-driven staffing:

        | Metric | Before | After | Change |
        |--------|--------|-------|--------|
        | Weekly Cost | ${opt['current_weekly_cost']:,.0f} | ${opt['optimized_weekly_cost']:,.0f} | **-${opt['weekly_savings']:,.0f}** ({savings_pct:.1f}%) |
        | Annual Cost | ${opt['current_weekly_cost'] * 52:,.0f} | ${opt['optimized_weekly_cost'] * 52:,.0f} | **-${annual_savings:,.0f}** |
        | Utilization | {staff_stats['avg_utilization']:.1f}% | 95% | **+{95 - staff_stats['avg_utilization']:.1f}%** |
        | Overtime | {staff_stats['total_overtime']:.0f}h | 0h | **Eliminated** |
        | Shortages | {staff_stats['shortage_days']} days | 0 days | **Eliminated** |
        """)

    elif st.session_state.staff_data_loaded and st.session_state.financial_data_loaded:
        st.info("📊 Go to **'After Optimization'** tab and click **'Apply Forecast Optimization'** to see the comparison.")
    else:
        st.warning("⚠️ Please load data in the first tab and run optimization to see comparison.")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Data Status")

    st.markdown(f"**Staff Data:** {'✅ Loaded' if st.session_state.staff_data_loaded else '❌ Not loaded'}")
    st.markdown(f"**Financial Data:** {'✅ Loaded' if st.session_state.financial_data_loaded else '❌ Not loaded'}")
    st.markdown(f"**Optimization:** {'✅ Done' if st.session_state.optimization_done else '❌ Not done'}")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        for key in ["staff_data_loaded", "staff_df", "staff_stats", "financial_data_loaded",
                    "financial_df", "financial_stats", "optimization_done", "optimized_results"]:
            if key in st.session_state:
                st.session_state[key] = None if "df" in key or "stats" in key or "results" in key else False
        st.session_state.cost_params = DEFAULT_COST_PARAMS.copy()
        st.rerun()
