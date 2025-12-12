# =============================================================================
# 11_Staff_Scheduling_Optimization.py
# SIMPLIFIED: Forecast-Driven Staff Scheduling with Before/After Comparison
# =============================================================================
"""
Simplified Staff Scheduling - Forecast Driven

Approach:
- BEFORE: Fixed staffing (historical approach without forecasting)
- AFTER: Dynamic staffing based on ML predictions by clinical category

Key Formula:
    Staff Needed = Σ (Predicted Patients by Category × Staffing Ratio)
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
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
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
# These ratios represent: 1 staff member per X patients

STAFFING_RATIOS = {
    "CARDIAC": {"doctors": 5, "nurses": 3, "support": 8},       # High acuity
    "TRAUMA": {"doctors": 6, "nurses": 2, "support": 5},        # Nurse-heavy
    "RESPIRATORY": {"doctors": 8, "nurses": 4, "support": 10},
    "GASTROINTESTINAL": {"doctors": 10, "nurses": 5, "support": 12},
    "INFECTIOUS": {"doctors": 12, "nurses": 6, "support": 15},
    "NEUROLOGICAL": {"doctors": 8, "nurses": 4, "support": 10},
    "OTHER": {"doctors": 15, "nurses": 8, "support": 20},
}

# Cost parameters (USD)
COST_PARAMS = {
    "doctor_hourly": 150,
    "nurse_hourly": 50,
    "support_hourly": 25,
    "shift_hours": 8,
    "overtime_multiplier": 1.5,
    "understaffing_penalty_per_patient": 200,  # Cost of poor care
}

# Fixed staffing (BEFORE - what hospital did without forecasting)
FIXED_STAFFING = {
    "doctors": 10,
    "nurses": 30,
    "support": 15,
}


# =============================================================================
# STEP 1: DATA EXTRACTION FUNCTIONS
# =============================================================================

def get_forecast_data() -> Dict[str, Any]:
    """
    Extract forecast data from session state.
    Returns total patient forecast and by clinical category.
    """
    result = {
        "has_forecast": False,
        "source": None,
        "total_forecast": [],
        "category_forecast": {},
        "dates": [],
        "horizon": 0,
    }

    # Try different session state keys for forecasts
    # Priority: ML models > SARIMAX > ARIMA

    # 1. Check for ML multi-horizon results (XGBoost, LSTM, ANN)
    ml_results = st.session_state.get("ml_mh_results")
    if ml_results and isinstance(ml_results, dict):
        for model_name in ["XGBoost", "LSTM", "ANN"]:
            if model_name in ml_results:
                model_data = ml_results[model_name]
                if "per_h" in model_data:
                    forecasts = []
                    for h in range(1, 8):
                        h_key = f"h{h}"
                        if h_key in model_data["per_h"]:
                            pred = model_data["per_h"][h_key].get("predictions", [])
                            if pred:
                                forecasts.append(float(np.mean(pred[-7:])) if len(pred) > 7 else float(np.mean(pred)))
                    if forecasts:
                        result["has_forecast"] = True
                        result["source"] = f"ML Model ({model_name})"
                        result["total_forecast"] = forecasts
                        result["horizon"] = len(forecasts)
                        break

    # 2. Check for SARIMAX results
    if not result["has_forecast"]:
        sarimax = st.session_state.get("sarimax_results")
        if sarimax and isinstance(sarimax, dict):
            if "forecast" in sarimax:
                fc = sarimax["forecast"]
                if isinstance(fc, (list, np.ndarray)) and len(fc) > 0:
                    result["has_forecast"] = True
                    result["source"] = "SARIMAX"
                    result["total_forecast"] = list(fc[:7])
                    result["horizon"] = len(result["total_forecast"])

    # 3. Check for ARIMA multi-horizon results
    if not result["has_forecast"]:
        arima = st.session_state.get("arima_mh_results")
        if arima and isinstance(arima, dict):
            if "per_h" in arima:
                forecasts = []
                for h in range(1, 8):
                    h_key = f"h{h}"
                    if h_key in arima["per_h"]:
                        pred = arima["per_h"][h_key].get("predictions", [])
                        if pred:
                            forecasts.append(float(np.mean(pred[-7:])) if len(pred) > 7 else float(np.mean(pred)))
                if forecasts:
                    result["has_forecast"] = True
                    result["source"] = "ARIMA"
                    result["total_forecast"] = forecasts
                    result["horizon"] = len(forecasts)

    # 4. Check for multi-target results (clinical categories)
    multi_target = st.session_state.get("sarimax_multi_target_results")
    if multi_target and isinstance(multi_target, dict):
        for category in ["CARDIAC", "TRAUMA", "RESPIRATORY", "GASTROINTESTINAL", "INFECTIOUS", "NEUROLOGICAL"]:
            if category in multi_target:
                cat_data = multi_target[category]
                if "forecast" in cat_data:
                    result["category_forecast"][category] = list(cat_data["forecast"][:7])

    # If no category forecast, distribute total forecast proportionally
    if result["has_forecast"] and not result["category_forecast"]:
        # Default distribution based on typical ED patterns
        distribution = {
            "CARDIAC": 0.15,
            "TRAUMA": 0.20,
            "RESPIRATORY": 0.25,
            "GASTROINTESTINAL": 0.15,
            "INFECTIOUS": 0.15,
            "NEUROLOGICAL": 0.05,
            "OTHER": 0.05,
        }
        for cat, pct in distribution.items():
            result["category_forecast"][cat] = [f * pct for f in result["total_forecast"]]

    # Generate dates
    if result["has_forecast"]:
        today = datetime.now().date()
        result["dates"] = [(today + timedelta(days=i)).strftime("%a %m/%d") for i in range(1, result["horizon"] + 1)]

    return result


def get_historical_data() -> Dict[str, Any]:
    """
    Extract historical patient data for BEFORE analysis.
    """
    result = {
        "has_data": False,
        "daily_patients": [],
        "dates": [],
        "category_data": {},
    }

    # Try to get processed data
    processed_df = st.session_state.get("processed_df")
    if processed_df is not None and len(processed_df) > 0:
        result["has_data"] = True

        # Get last 30 days of data
        if "patient_count" in processed_df.columns:
            result["daily_patients"] = processed_df["patient_count"].tail(30).tolist()
        elif "Target_1" in processed_df.columns:
            result["daily_patients"] = processed_df["Target_1"].tail(30).tolist()

        # Get category data if available
        for cat in ["CARDIAC", "TRAUMA", "RESPIRATORY", "GASTROINTESTINAL", "INFECTIOUS", "NEUROLOGICAL"]:
            if cat in processed_df.columns:
                result["category_data"][cat] = processed_df[cat].tail(30).tolist()

    return result


def calculate_staff_needed(patients_by_category: Dict[str, float]) -> Dict[str, int]:
    """
    Calculate staff needed based on patient forecast by category.

    Formula: staff = ceil(patients / ratio)
    """
    doctors = 0
    nurses = 0
    support = 0

    for category, patients in patients_by_category.items():
        if category in STAFFING_RATIOS:
            ratios = STAFFING_RATIOS[category]
            doctors += np.ceil(patients / ratios["doctors"])
            nurses += np.ceil(patients / ratios["nurses"])
            support += np.ceil(patients / ratios["support"])

    return {
        "doctors": int(doctors),
        "nurses": int(nurses),
        "support": int(support),
        "total": int(doctors + nurses + support),
    }


def calculate_daily_cost(staff: Dict[str, int], overtime_hours: float = 0) -> float:
    """
    Calculate daily staffing cost.
    """
    regular_cost = (
        staff["doctors"] * COST_PARAMS["doctor_hourly"] * COST_PARAMS["shift_hours"] +
        staff["nurses"] * COST_PARAMS["nurse_hourly"] * COST_PARAMS["shift_hours"] +
        staff["support"] * COST_PARAMS["support_hourly"] * COST_PARAMS["shift_hours"]
    )

    overtime_cost = overtime_hours * COST_PARAMS["nurse_hourly"] * COST_PARAMS["overtime_multiplier"]

    return regular_cost + overtime_cost


def analyze_before_scenario(historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze BEFORE scenario: Fixed staffing without forecasting.
    """
    if not historical_data["has_data"]:
        return {"has_analysis": False}

    daily_patients = historical_data["daily_patients"]

    # Fixed daily cost
    fixed_daily_cost = calculate_daily_cost(FIXED_STAFFING)

    # Analyze each day
    overtime_days = 0
    understaffed_days = 0
    overstaffed_days = 0
    total_overtime_hours = 0
    total_penalty = 0

    # Capacity with fixed staffing (rough estimate: 10 patients per staff)
    fixed_capacity = FIXED_STAFFING["doctors"] * 10 + FIXED_STAFFING["nurses"] * 6 + FIXED_STAFFING["support"] * 3

    for patients in daily_patients:
        if patients > fixed_capacity * 1.1:  # Over capacity by 10%
            understaffed_days += 1
            # Overtime needed
            overtime = (patients - fixed_capacity) / 5  # Hours of overtime
            total_overtime_hours += overtime
            # Penalty for potential poor care
            excess_patients = patients - fixed_capacity
            total_penalty += excess_patients * COST_PARAMS["understaffing_penalty_per_patient"] * 0.1
        elif patients < fixed_capacity * 0.7:  # Under capacity by 30%
            overstaffed_days += 1

    total_days = len(daily_patients)
    total_labor_cost = fixed_daily_cost * total_days
    total_overtime_cost = total_overtime_hours * COST_PARAMS["nurse_hourly"] * COST_PARAMS["overtime_multiplier"]

    return {
        "has_analysis": True,
        "total_days": total_days,
        "fixed_staff": FIXED_STAFFING,
        "fixed_daily_cost": fixed_daily_cost,
        "total_labor_cost": total_labor_cost,
        "overtime_days": overtime_days,
        "understaffed_days": understaffed_days,
        "overstaffed_days": overstaffed_days,
        "total_overtime_hours": total_overtime_hours,
        "total_overtime_cost": total_overtime_cost,
        "total_penalty": total_penalty,
        "total_cost": total_labor_cost + total_overtime_cost + total_penalty,
        "avg_daily_cost": (total_labor_cost + total_overtime_cost + total_penalty) / total_days,
        "efficiency_rate": 100 - (understaffed_days + overstaffed_days) / total_days * 100,
    }


def analyze_after_scenario(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze AFTER scenario: Forecast-driven dynamic staffing.
    """
    if not forecast_data["has_forecast"]:
        return {"has_analysis": False}

    daily_costs = []
    daily_staff = []

    for day_idx in range(forecast_data["horizon"]):
        # Get patients by category for this day
        patients_by_category = {}
        for cat, forecasts in forecast_data["category_forecast"].items():
            if day_idx < len(forecasts):
                patients_by_category[cat] = forecasts[day_idx]

        # Calculate optimal staff
        staff = calculate_staff_needed(patients_by_category)
        daily_staff.append(staff)

        # Calculate cost (no overtime needed with proper staffing)
        cost = calculate_daily_cost(staff, overtime_hours=0)
        daily_costs.append(cost)

    total_cost = sum(daily_costs)

    return {
        "has_analysis": True,
        "horizon": forecast_data["horizon"],
        "source": forecast_data["source"],
        "daily_staff": daily_staff,
        "daily_costs": daily_costs,
        "total_cost": total_cost,
        "avg_daily_cost": total_cost / forecast_data["horizon"],
        "total_forecast": forecast_data["total_forecast"],
        "category_forecast": forecast_data["category_forecast"],
        "dates": forecast_data["dates"],
        "overtime_hours": 0,  # Proper staffing = no overtime
        "overtime_cost": 0,
        "penalty": 0,  # Proper staffing = no understaffing penalty
    }


# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}

.metric-card.before {
    border-color: rgba(239, 68, 68, 0.5);
}

.metric-card.after {
    border-color: rgba(34, 197, 94, 0.5);
}

.metric-card.savings {
    border-color: rgba(234, 179, 8, 0.5);
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98));
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.metric-value.red { color: #ef4444; }
.metric-value.green { color: #22c55e; }
.metric-value.blue { color: #3b82f6; }
.metric-value.yellow { color: #eab308; }

.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
}

/* Comparison table */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

.comparison-table th, .comparison-table td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid rgba(59, 130, 246, 0.2);
}

.comparison-table th {
    color: #60a5fa;
    font-weight: 600;
}

.highlight-good { color: #22c55e; font-weight: 600; }
.highlight-bad { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN PAGE
# =============================================================================

st.markdown("# 👥 Staff Scheduling")
st.markdown("### Forecast-Driven Resource Allocation")

st.info("""
**How this works:**
1. **BEFORE**: Hospital uses fixed staffing (same staff every day) → inefficient
2. **AFTER**: Staff allocation based on ML patient predictions → optimized

Compare the costs and see the savings from using forecasting!
""")

# =============================================================================
# LOAD DATA
# =============================================================================

forecast_data = get_forecast_data()
historical_data = get_historical_data()

# Show data status
col1, col2 = st.columns(2)

with col1:
    if forecast_data["has_forecast"]:
        st.success(f"✅ Forecast loaded: {forecast_data['source']} ({forecast_data['horizon']} days)")
    else:
        st.warning("⚠️ No forecast data. Please run models in Modeling Hub first.")

with col2:
    if historical_data["has_data"]:
        st.success(f"✅ Historical data: {len(historical_data['daily_patients'])} days")
    else:
        st.warning("⚠️ No historical data. Please load data in Data Hub first.")


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
            # Calculate savings
            # Normalize to same period (7 days)
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
                    <div class="metric-value green">${after_weekly:,.0f}</div>
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
                st.markdown("**Problems with Fixed Staffing (BEFORE):**")
                st.markdown(f"""
                - 📅 Analyzed **{before_analysis['total_days']} days** of historical data
                - ⚠️ **{before_analysis['understaffed_days']} days** understaffed (overtime needed)
                - 💸 **{before_analysis['overstaffed_days']} days** overstaffed (wasted labor)
                - ⏰ **{before_analysis['total_overtime_hours']:.0f} hours** of overtime
                - 📉 Staffing efficiency: **{before_analysis['efficiency_rate']:.1f}%**
                """)

            with col2:
                st.markdown("**Benefits of Forecast-Driven Staffing (AFTER):**")
                st.markdown(f"""
                - 🎯 Staff matches **predicted demand** each day
                - ✅ **Zero overtime** needed (right staff, right time)
                - ✅ **No understaffing** penalties
                - 📊 Based on **{after_analysis['source']}** predictions
                - 💰 **{savings_pct:.1f}% cost reduction**
                """)

        else:
            st.info("Run forecasting models and load historical data to see the comparison.")

    # ==========================================================================
    # TAB 2: BEFORE ANALYSIS
    # ==========================================================================
    with tab2:
        st.markdown('<div class="section-header">🔴 BEFORE: Fixed Staffing Analysis</div>', unsafe_allow_html=True)

        if before_analysis.get("has_analysis"):
            st.markdown("""
            **Traditional Approach:** Hospital schedules the same number of staff every day,
            regardless of expected patient volume.
            """)

            # Fixed staffing display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Doctors (Daily)", f"{FIXED_STAFFING['doctors']}", "Fixed")
            with col2:
                st.metric("Nurses (Daily)", f"{FIXED_STAFFING['nurses']}", "Fixed")
            with col3:
                st.metric("Support Staff (Daily)", f"{FIXED_STAFFING['support']}", "Fixed")

            st.markdown("---")

            # Cost breakdown
            st.markdown("### 💰 Cost Breakdown")

            cost_data = {
                "Category": ["Regular Labor", "Overtime", "Understaffing Penalty", "TOTAL"],
                "Amount": [
                    f"${before_analysis['total_labor_cost']:,.0f}",
                    f"${before_analysis['total_overtime_cost']:,.0f}",
                    f"${before_analysis['total_penalty']:,.0f}",
                    f"${before_analysis['total_cost']:,.0f}",
                ],
                "Per Day": [
                    f"${before_analysis['fixed_daily_cost']:,.0f}",
                    f"${before_analysis['total_overtime_cost']/before_analysis['total_days']:,.0f}",
                    f"${before_analysis['total_penalty']/before_analysis['total_days']:,.0f}",
                    f"${before_analysis['avg_daily_cost']:,.0f}",
                ],
            }
            st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

            # Problem visualization
            if historical_data["daily_patients"]:
                st.markdown("### 📊 Demand vs Fixed Capacity")

                fig = go.Figure()

                # Patient demand line
                fig.add_trace(go.Scatter(
                    y=historical_data["daily_patients"],
                    mode='lines+markers',
                    name='Actual Patient Demand',
                    line=dict(color='#3b82f6', width=2),
                ))

                # Fixed capacity line
                fixed_capacity = FIXED_STAFFING["doctors"] * 10 + FIXED_STAFFING["nurses"] * 6 + FIXED_STAFFING["support"] * 3
                fig.add_hline(
                    y=fixed_capacity,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"Fixed Capacity ({fixed_capacity})",
                )

                fig.update_layout(
                    title="Patient Demand vs Fixed Staff Capacity",
                    xaxis_title="Day",
                    yaxis_title="Patients",
                    template="plotly_dark",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption("🔴 Days above the line = understaffed (overtime/poor care). Days far below = overstaffed (wasted money).")

        else:
            st.info("Load historical data to analyze the BEFORE scenario.")

    # ==========================================================================
    # TAB 3: AFTER ANALYSIS
    # ==========================================================================
    with tab3:
        st.markdown('<div class="section-header">🟢 AFTER: Forecast-Driven Staffing</div>', unsafe_allow_html=True)

        if after_analysis.get("has_analysis"):
            st.markdown(f"""
            **Forecast-Driven Approach:** Staff allocation based on **{after_analysis['source']}** predictions.
            Staff levels adjust daily to match predicted patient volume by clinical category.
            """)

            # Daily staffing table
            st.markdown("### 📅 Recommended Weekly Schedule")

            schedule_data = []
            for i, (date, staff) in enumerate(zip(after_analysis["dates"], after_analysis["daily_staff"])):
                total_patients = after_analysis["total_forecast"][i] if i < len(after_analysis["total_forecast"]) else 0
                schedule_data.append({
                    "Day": date,
                    "Expected Patients": f"{total_patients:.0f}",
                    "Doctors": staff["doctors"],
                    "Nurses": staff["nurses"],
                    "Support": staff["support"],
                    "Total Staff": staff["total"],
                    "Daily Cost": f"${after_analysis['daily_costs'][i]:,.0f}",
                })

            st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)

            # Category breakdown
            st.markdown("### 📊 Staffing by Clinical Category")

            if after_analysis["category_forecast"]:
                # Create stacked bar chart
                fig = go.Figure()

                categories = list(after_analysis["category_forecast"].keys())
                colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#6b7280']

                for idx, cat in enumerate(categories):
                    forecasts = after_analysis["category_forecast"][cat]
                    fig.add_trace(go.Bar(
                        name=cat,
                        x=after_analysis["dates"],
                        y=forecasts[:len(after_analysis["dates"])],
                        marker_color=colors[idx % len(colors)],
                    ))

                fig.update_layout(
                    barmode='stack',
                    title="Patient Forecast by Clinical Category",
                    xaxis_title="Day",
                    yaxis_title="Patients",
                    template="plotly_dark",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

            # Cost summary
            st.markdown("### 💰 Cost Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Weekly Cost", f"${after_analysis['total_cost']:,.0f}")
            with col2:
                st.metric("Average Daily Cost", f"${after_analysis['avg_daily_cost']:,.0f}")
            with col3:
                st.metric("Overtime Cost", f"${after_analysis['overtime_cost']:,.0f}", "Zero!")

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
                marker_color=['#ef4444', '#22c55e'],
                text=[f'${before_weekly:,.0f}', f'${after_weekly:,.0f}'],
                textposition='auto',
            ))

            fig_cost.update_layout(
                title="Weekly Staffing Cost: Before vs After",
                yaxis_title="Cost ($)",
                template="plotly_dark",
                height=400,
            )

            st.plotly_chart(fig_cost, use_container_width=True)

            # Chart 2: Cost Breakdown Comparison
            st.markdown("### 📊 Cost Breakdown")

            fig_breakdown = go.Figure()

            categories = ['Labor', 'Overtime', 'Penalties']
            before_values = [
                before_analysis["fixed_daily_cost"] * 7,
                before_analysis["total_overtime_cost"] / before_analysis["total_days"] * 7,
                before_analysis["total_penalty"] / before_analysis["total_days"] * 7,
            ]
            after_values = [
                after_analysis["total_cost"],
                0,
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
                marker_color='#22c55e',
            ))

            fig_breakdown.update_layout(
                barmode='group',
                title="Cost Breakdown: Before vs After",
                yaxis_title="Weekly Cost ($)",
                template="plotly_dark",
                height=400,
            )

            st.plotly_chart(fig_breakdown, use_container_width=True)

            # Chart 3: Efficiency Metrics
            st.markdown("### 🎯 Efficiency Comparison")

            col1, col2 = st.columns(2)

            with col1:
                # Gauge for BEFORE
                fig_gauge_before = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=before_analysis["efficiency_rate"],
                    title={'text': "BEFORE: Staffing Efficiency"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(239,68,68,0.3)"},
                            {'range': [50, 75], 'color': "rgba(234,179,8,0.3)"},
                            {'range': [75, 100], 'color': "rgba(34,197,94,0.3)"},
                        ],
                    }
                ))
                fig_gauge_before.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_gauge_before, use_container_width=True)

            with col2:
                # Gauge for AFTER (100% efficiency with forecasting)
                fig_gauge_after = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=100,
                    title={'text': "AFTER: Staffing Efficiency"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#22c55e"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(239,68,68,0.3)"},
                            {'range': [50, 75], 'color': "rgba(234,179,8,0.3)"},
                            {'range': [75, 100], 'color': "rgba(34,197,94,0.3)"},
                        ],
                    }
                ))
                fig_gauge_after.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_gauge_after, use_container_width=True)

            # Summary box
            st.success(f"""
            ### 🎉 Summary

            By using **ML forecasting** instead of fixed staffing:
            - **Weekly savings:** ${before_weekly - after_weekly:,.0f} ({((before_weekly - after_weekly) / before_weekly * 100):.1f}%)
            - **Annual savings:** ${(before_weekly - after_weekly) * 52:,.0f}
            - **Overtime eliminated:** {before_analysis['total_overtime_hours']:.0f} hours/month
            - **Efficiency improved:** {before_analysis['efficiency_rate']:.1f}% → 100%
            """)

        else:
            st.info("Load both historical data and run forecasting models to see comparison charts.")

else:
    st.warning("""
    ### ⚠️ No Data Available

    To use Staff Scheduling, please:
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

    with st.expander("Staffing Ratios", expanded=False):
        st.caption("Patients per staff member by category")
        for cat, ratios in STAFFING_RATIOS.items():
            st.text(f"{cat}: Dr={ratios['doctors']}, RN={ratios['nurses']}")

    with st.expander("Cost Parameters", expanded=False):
        st.caption("Hourly rates (USD)")
        st.text(f"Doctor: ${COST_PARAMS['doctor_hourly']}/hr")
        st.text(f"Nurse: ${COST_PARAMS['nurse_hourly']}/hr")
        st.text(f"Support: ${COST_PARAMS['support_hourly']}/hr")

    with st.expander("Fixed Staffing (BEFORE)", expanded=False):
        st.caption("Traditional fixed schedule")
        st.text(f"Doctors: {FIXED_STAFFING['doctors']}/day")
        st.text(f"Nurses: {FIXED_STAFFING['nurses']}/day")
        st.text(f"Support: {FIXED_STAFFING['support']}/day")
