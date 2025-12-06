# =============================================================================
# 11_Staff_Scheduling_Optimization.py
# Staff Scheduling Optimization with Supabase Integration
# Fetches staff data from Supabase and integrates with forecasting outputs
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional

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

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
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

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR STAFF SCHEDULING
   ======================================== */

@keyframes float-orb {
    0%, 100% {
        transform: translate(0, 0) scale(1);
        opacity: 0.25;
    }
    50% {
        transform: translate(30px, -30px) scale(1.05);
        opacity: 0.35;
    }
}

.fluorescent-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}

.orb-1 {
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}

.orb-2 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}

@keyframes sparkle {
    0%, 100% {
        opacity: 0;
        transform: scale(0);
    }
    50% {
        opacity: 0.6;
        transform: scale(1);
    }
}

.sparkle {
    position: fixed;
    width: 3px;
    height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

.sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
.sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
.sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

@media (max-width: 768px) {
    .fluorescent-orb {
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }
    .sparkle {
        display: none;
    }
}

/* Staff metrics cards */
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
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
""", unsafe_allow_html=True)

# Premium Hero Header
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>👥</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Staff Scheduling Optimization</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Optimize hospital staff schedules based on demand forecasts, availability, and skill requirements with advanced scheduling algorithms
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# SUPABASE CONNECTION STATUS
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# DATA SOURCE SELECTION
# -------------------------------------------------------------
st.markdown("### Data Source")

data_source = st.radio(
    "Select data source:",
    ["🔗 Supabase (Cloud Database)", "📂 Upload File (CSV/Excel)"],
    horizontal=True,
    index=0 if is_connected else 1
)

staff_df = None

if "Supabase" in data_source:
    # ===== SUPABASE DATA FETCH =====
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
                else:
                    st.empty()

            if st.button("🔄 Fetch Data from Supabase", type="primary"):
                with st.spinner("Fetching data from Supabase..."):
                    service = StaffSchedulingService()

                    if fetch_mode == "All Data":
                        staff_df = service.fetch_staff_data()
                    elif fetch_mode == "Date Range":
                        staff_df = service.fetch_staff_data(start_date, end_date)
                    else:  # Latest N Days
                        staff_df = pd.DataFrame(service.get_latest_records(n_days))
                        if not staff_df.empty:
                            # Rename columns after fetch
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
                        st.session_state["staff_roster_df"] = staff_df
                        st.success(f"✅ Loaded {len(staff_df)} records from Supabase")
                    else:
                        st.warning("No data found in Supabase. The table might be empty.")

        # Load from session state if available
        if "staff_roster_df" in st.session_state:
            staff_df = st.session_state["staff_roster_df"]
    else:
        st.warning(
            "⚠️ Supabase is not connected. Please configure your credentials in `.streamlit/secrets.toml`:\n\n"
            "```toml\n"
            "[supabase]\n"
            'url = "https://your-project.supabase.co"\n'
            'key = "your-anon-key"\n'
            "```"
        )

else:
    # ===== FILE UPLOAD FALLBACK =====
    uploaded_file = st.file_uploader(
        "📂 Upload staff roster (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                staff_df = pd.read_csv(uploaded_file)
            else:
                staff_df = pd.read_excel(uploaded_file)

            st.session_state["staff_roster_df"] = staff_df
            st.success(f"✅ Loaded {len(staff_df)} records from file")

            # Option to upload to Supabase
            if is_connected:
                st.markdown("---")
                st.markdown("**💾 Save to Supabase?**")
                col1, col2 = st.columns(2)
                with col1:
                    replace_existing = st.checkbox("Replace existing data", value=False)
                with col2:
                    if st.button("Upload to Supabase"):
                        service = StaffSchedulingService()
                        if service.upload_dataframe(staff_df, replace_existing):
                            st.success("✅ Data uploaded to Supabase successfully!")
                        else:
                            st.error("Failed to upload data to Supabase")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif "staff_roster_df" in st.session_state:
        staff_df = st.session_state["staff_roster_df"]

# -------------------------------------------------------------
# DISPLAY DATA & STATISTICS
# -------------------------------------------------------------
if staff_df is not None and not staff_df.empty:
    st.divider()
    st.markdown("### 📊 Staff Data Overview")

    # Key metrics - using MIN-MAX ranges for staff (more meaningful than decimals)
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

    # Data preview - only first 10 rows
    with st.expander("📋 Data Preview (First 10 Rows)", expanded=False):
        st.dataframe(staff_df.head(10), use_container_width=True)
        st.caption(f"Showing 10 of {len(staff_df):,} total records")

        # Download option for full dataset
        csv = staff_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset as CSV",
            data=csv,
            file_name="staff_scheduling_data.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# INTEGRATION WITH FORECASTING
# -------------------------------------------------------------
st.divider()
st.markdown("### 🔗 Forecasting Integration")

# Check for forecasting results
has_forecast = "forecast_results" in st.session_state or "multi_target_results" in st.session_state

if has_forecast:
    st.success("✅ Forecasting results detected! Staff requirements can be calculated based on predicted patient arrivals.")

    with st.expander("🎯 Staff-to-Patient Ratio Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            patients_per_doctor = st.number_input(
                "Patients per Doctor",
                min_value=1, max_value=50, value=10,
                help="Maximum patients one doctor can handle per shift"
            )

        with col2:
            patients_per_nurse = st.number_input(
                "Patients per Nurse",
                min_value=1, max_value=20, value=5,
                help="Maximum patients one nurse can handle per shift"
            )

        with col3:
            patients_per_support = st.number_input(
                "Patients per Support Staff",
                min_value=1, max_value=30, value=15,
                help="Maximum patients one support staff can handle"
            )

        if st.button("📊 Calculate Staff Requirements", type="primary"):
            # Get forecast data
            forecast_data = st.session_state.get("forecast_results") or st.session_state.get("multi_target_results", {})

            if isinstance(forecast_data, dict) and "Target_1" in forecast_data:
                # Multi-target results
                target_1 = forecast_data["Target_1"]
                if "forecast" in target_1:
                    predicted_patients = target_1["forecast"]

                    # Calculate required staff
                    required_doctors = np.ceil(predicted_patients / patients_per_doctor)
                    required_nurses = np.ceil(predicted_patients / patients_per_nurse)
                    required_support = np.ceil(predicted_patients / patients_per_support)

                    # Create requirements DataFrame
                    req_df = pd.DataFrame({
                        "Day": [f"Day {i+1}" for i in range(len(predicted_patients))],
                        "Predicted_Patients": predicted_patients,
                        "Required_Doctors": required_doctors.astype(int),
                        "Required_Nurses": required_nurses.astype(int),
                        "Required_Support": required_support.astype(int),
                        "Total_Staff": (required_doctors + required_nurses + required_support).astype(int)
                    })

                    st.session_state["staff_requirements"] = req_df

                    st.markdown("#### 📋 Staff Requirements Forecast")
                    st.dataframe(req_df, use_container_width=True)

                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Doctors Needed", f"{req_df['Required_Doctors'].sum()}")
                    with col2:
                        st.metric("Total Nurses Needed", f"{req_df['Required_Nurses'].sum()}")
                    with col3:
                        st.metric("Total Support Needed", f"{req_df['Required_Support'].sum()}")
                else:
                    st.warning("Forecast data format not recognized")
            else:
                st.warning("Unable to extract forecast values. Please run forecasting first.")
else:
    st.info(
        "💡 **Tip:** Run patient arrival forecasting first (Dashboard or Modeling Hub) to enable "
        "automatic staff requirement calculations based on predicted demand."
    )

# -------------------------------------------------------------
# OPTIMIZATION SETTINGS
# -------------------------------------------------------------
st.divider()
st.markdown("### ⚙️ Optimization Settings")

c1, c2 = st.columns(2)
with c1:
    st.date_input("Scheduling start date", date.today())
    st.number_input("Planning horizon (days)", min_value=1, max_value=60, value=7)

with c2:
    st.multiselect(
        "Optimization objectives",
        ["Minimize overtime", "Balance shifts", "Match patient demand", "Maximize fairness"],
        default=["Minimize overtime", "Match patient demand"],
    )

# Constraint configuration
st.subheader("Constraints")
col1, col2, col3 = st.columns(3)
with col1:
    st.checkbox("Respect staff availability", value=True)
with col2:
    st.checkbox("Enforce maximum weekly hours", value=True)
with col3:
    st.checkbox("Include skill matching", value=False)

# Optimization mode
st.subheader("Solver Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    st.radio("Mode", ["Automatic (solver-based)", "Manual (user-guided)"], index=0)
with c2:
    st.selectbox("Solver type", ["Linear Programming (PuLP)", "Genetic Algorithm", "Constraint Programming"])
with c3:
    st.number_input("Max iterations", min_value=10, max_value=5000, value=200)

# Execution buttons
st.divider()
left, right = st.columns([1, 2])
with left:
    run_btn = st.button("🚀 Run Optimization", type="primary", disabled=staff_df is None)
    save_btn = st.button("💾 Save Schedule", disabled=True)

    if run_btn:
        st.info("🔧 Optimization engine coming soon! This will integrate with PuLP/OR-Tools.")

with right:
    st.markdown("### Schedule Preview")
    if "schedule_fig" in st.session_state:
        st.plotly_chart(st.session_state["schedule_fig"], use_container_width=True)
    elif "schedule_solution_df" in st.session_state:
        st.dataframe(st.session_state["schedule_solution_df"], use_container_width=True)
    elif "staff_requirements" in st.session_state:
        st.dataframe(st.session_state["staff_requirements"], use_container_width=True)
    else:
        st.info("Run optimization to see the schedule preview here.")

st.divider()
st.caption("Staff Scheduling Optimization powered by Supabase. Optimization engine (PuLP/OR-Tools) integration coming soon.")
