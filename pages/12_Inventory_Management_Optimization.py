# =============================================================================
# 12_Inventory_Management_Optimization.py
# Deterministic EOQ/MILP Inventory Optimization for Healthcare
# Connected to Forecast Hub (10_Forecast.py) for demand-driven optimization
# =============================================================================
"""
Healthcare Inventory Management Optimization

Mathematical Approach (Deterministic - for RPIW < 15%):
-------------------------------------------------------
1. EOQ (Economic Order Quantity):
   Q* = sqrt(2 * K * D / h)

2. MILP Multi-Item Formulation:
   min Z = sum_t sum_i [K_i * y_{i,t} + c_i * Q_{i,t} + h_i * I_{i,t} + p_i * B_{i,t}]

3. Reorder Point with Safety Stock:
   s_i = d_i * L_i + SS_i
   SS_i = z_CSL * sigma_i * sqrt(L_i)

Integration:
- Uses patient forecasts from Forecast Hub (10_Forecast.py)
- Converts patient arrivals to inventory demand: d_{i,t} = alpha_i * D_t
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Inventory Management Optimization - HealthForecast AI",
    page_icon="📦",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# IMPORT OPTIMIZATION MODULE
# =============================================================================
from app_core.optimization.inventory_optimizer import (
    HealthcareInventoryItem,
    ItemCriticality,
    DEFAULT_HEALTHCARE_ITEMS,
    EOQResult,
    MILPResult,
    calculate_eoq,
    solve_milp_inventory,
    convert_patient_forecast_to_inventory_demand,
    calculate_rpiw,
    get_optimization_recommendation,
    generate_reorder_alerts,
)

# =============================================================================
# SUPABASE INVENTORY SERVICE
# =============================================================================
try:
    from app_core.data.inventory_service import (
        InventoryService,
        get_inventory_service,
        fetch_inventory_data,
        check_inventory_connection
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* Inventory cards */
.inv-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    text-align: center;
}

.inv-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}

.inv-label {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.inv-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

/* Alert badges */
.alert-critical {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border-left: 4px solid #ef4444;
}

.alert-high {
    background: rgba(251, 146, 60, 0.2);
    color: #fb923c;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border-left: 4px solid #fb923c;
}

.alert-medium {
    background: rgba(250, 204, 21, 0.2);
    color: #facc15;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border-left: 4px solid #facc15;
}

/* RPIW indicator */
.rpiw-low {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
}

.rpiw-medium {
    background: rgba(250, 204, 21, 0.2);
    color: #facc15;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
}

.rpiw-high {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
}

/* Item criticality badges */
.crit-low { background: #22c55e20; color: #22c55e; }
.crit-medium { background: #3b82f620; color: #3b82f6; }
.crit-high { background: #f9731620; color: #f97316; }
.crit-critical { background: #ef444420; color: #ef4444; }

.criticality-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>📦</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Inventory Management Optimization</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Deterministic EOQ & MILP optimization for healthcare inventory<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Demand-driven by ML patient forecasts</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FORECAST DETECTION (same pattern as Staff Scheduling)
# =============================================================================
def detect_forecast_sources() -> List[Dict[str, Any]]:
    """Detect available forecast sources from session state."""
    sources = []
    found_keys = set()

    # FORECAST HUB (10_Forecast.py) - PRIORITY
    if "forecast_hub_demand" in st.session_state:
        data = st.session_state.get("forecast_hub_demand")
        if data is not None:
            model_name = data.get("model", "Unknown")
            sources.append({
                "name": f"Forecast Hub: {model_name}",
                "key": "forecast_hub_demand",
                "data": data,
                "type": "forecast_hub"
            })
            found_keys.add("forecast_hub_demand")

    # Full forecast hub results
    if "forecast_hub_results" in st.session_state and "forecast_hub_demand" not in found_keys:
        data = st.session_state.get("forecast_hub_results")
        if data is not None:
            model_name = data.get("model_name", "Unknown")
            sources.append({
                "name": f"Forecast Hub: {model_name} (Full)",
                "key": "forecast_hub_results",
                "data": data,
                "type": "forecast_hub_full"
            })
            found_keys.add("forecast_hub_results")

    # ML models from Modeling Hub
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

    # Statistical models from Benchmarks
    stat_keys = [
        ("sarimax_results", "SARIMAX"),
        ("arima_mh_results", "ARIMA"),
        ("sarimax_multi_target_results", "SARIMAX Multi-Target"),
        ("arima_multi_target_results", "ARIMA Multi-Target"),
    ]
    for key, name in stat_keys:
        if key in st.session_state and key not in found_keys:
            data = st.session_state.get(key)
            if data is not None:
                sources.append({
                    "name": name,
                    "key": key,
                    "data": data,
                    "type": "stat"
                })
                found_keys.add(key)

    return sources


def extract_patient_forecast(source: Dict[str, Any], horizon: int = 30) -> List[float]:
    """Extract patient forecast from a source."""
    data = source["data"]
    source_type = source["type"]

    forecast = None

    try:
        if source_type == "forecast_hub":
            if "forecast" in data:
                forecast = list(data["forecast"])
        elif source_type == "forecast_hub_full":
            if "forecast_values" in data:
                forecast = list(data["forecast_values"])
        elif source_type in ["ml", "stat"]:
            # Try common keys
            for key in ["forecast", "predictions", "forecast_values"]:
                if key in data:
                    f = data[key]
                    if hasattr(f, "values"):
                        forecast = list(f.values)
                    else:
                        forecast = list(f)
                    break

            # Try Target_1
            if forecast is None and "Target_1" in data:
                target_data = data["Target_1"]
                if isinstance(target_data, dict):
                    for key in ["forecast", "predictions"]:
                        if key in target_data:
                            f = target_data[key]
                            if hasattr(f, "values"):
                                forecast = list(f.values)
                            else:
                                forecast = list(f)
                            break
    except Exception:
        pass

    if forecast is None:
        return []

    # Extend or trim to horizon
    if len(forecast) < horizon:
        last_val = forecast[-1] if forecast else 100
        forecast = forecast + [last_val] * (horizon - len(forecast))

    return forecast[:horizon]


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "inventory_items" not in st.session_state:
    st.session_state["inventory_items"] = DEFAULT_HEALTHCARE_ITEMS.copy()
if "current_inventory" not in st.session_state:
    st.session_state["current_inventory"] = {}
if "eoq_results" not in st.session_state:
    st.session_state["eoq_results"] = None
if "milp_results" not in st.session_state:
    st.session_state["milp_results"] = None
if "patient_forecast" not in st.session_state:
    st.session_state["patient_forecast"] = None
if "inventory_data_df" not in st.session_state:
    st.session_state["inventory_data_df"] = None
if "inventory_data_source" not in st.session_state:
    st.session_state["inventory_data_source"] = None


# =============================================================================
# TABS
# =============================================================================
tab_data, tab_forecast, tab_items, tab_eoq, tab_milp, tab_results = st.tabs([
    "📁 Data Source",
    "📊 Demand Forecast",
    "📋 Inventory Items",
    "📐 EOQ Analysis",
    "🔧 MILP Optimization",
    "📈 Results & Alerts"
])


# =============================================================================
# TAB 0: DATA SOURCE (Supabase or File Upload)
# =============================================================================
with tab_data:
    st.markdown("### 📁 Inventory Data Source")
    st.info("""
    Load historical inventory data to analyze usage patterns and current stock levels.
    This data integrates with the optimization models to provide realistic inventory parameters.
    """)

    # Data source selection
    data_source_options = ["📤 Upload CSV File"]
    if SUPABASE_AVAILABLE:
        data_source_options.insert(0, "☁️ Supabase Database")

    selected_source = st.radio(
        "Select data source:",
        options=data_source_options,
        horizontal=True,
        key="inv_data_source_radio"
    )

    st.markdown("---")

    if "Supabase" in selected_source and SUPABASE_AVAILABLE:
        # Supabase data source
        st.markdown("#### ☁️ Supabase Connection")

        # Check connection
        is_connected = check_inventory_connection()

        if is_connected:
            st.success("✅ Connected to Supabase")

            col1, col2 = st.columns([2, 1])

            with col1:
                fetch_mode = st.radio(
                    "Fetch mode:",
                    options=["All Records", "Latest N Days", "Date Range"],
                    horizontal=True,
                    key="inv_fetch_mode"
                )

                if fetch_mode == "Latest N Days":
                    n_days = st.slider("Number of days:", 7, 365, 90, key="inv_n_days")
                elif fetch_mode == "Date Range":
                    date_cols = st.columns(2)
                    with date_cols[0]:
                        start_date = st.date_input("Start Date", key="inv_start_date")
                    with date_cols[1]:
                        end_date = st.date_input("End Date", key="inv_end_date")

            with col2:
                st.markdown("#### Connection Status")
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Status</div>
                    <div class="inv-value" style="color: #22c55e;">Connected</div>
                    <div class="inv-label">inventory_management</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("📥 Fetch Inventory Data", type="primary", key="fetch_inv_btn"):
                with st.spinner("Fetching data from Supabase..."):
                    service = get_inventory_service()

                    if fetch_mode == "All Records":
                        df = service.fetch_all()
                    elif fetch_mode == "Latest N Days":
                        df = service.fetch_latest(n_days)
                    else:
                        df = service.fetch_by_date_range(start_date, end_date)

                    if not df.empty:
                        st.session_state["inventory_data_df"] = df
                        st.session_state["inventory_data_source"] = "Supabase"
                        st.success(f"✅ Loaded {len(df)} records from Supabase!")

                        # Auto-populate current inventory from latest record
                        latest = df.sort_values("Date").iloc[-1] if "Date" in df.columns else df.iloc[-1]
                        current_inv = {}
                        if "Inventory_Level_Gloves" in df.columns:
                            current_inv["GLV001"] = int(latest.get("Inventory_Level_Gloves", 0))
                        if "Inventory_Level_PPE" in df.columns:
                            current_inv["PPE001"] = int(latest.get("Inventory_Level_PPE", 0))
                        if "Inventory_Level_Medications" in df.columns:
                            current_inv["MED001"] = int(latest.get("Inventory_Level_Medications", 0))
                        st.session_state["current_inventory"] = current_inv

                        st.rerun()
                    else:
                        st.warning("No data found in Supabase table.")

        else:
            st.warning("⚠️ Not connected to Supabase")
            st.info("""
            **To enable Supabase:**
            1. Set `SUPABASE_URL` and `SUPABASE_KEY` environment variables
            2. Run `python scripts/setup_inventory_table.py` to upload data
            """)

    else:
        # File upload source
        st.markdown("#### 📤 Upload Inventory CSV")

        st.info("""
        **Expected CSV format:**
        - Date
        - Inventory_Used_Gloves, Inventory_Used_PPE_Sets, Inventory_Used_Medications
        - Inventory_Level_Gloves, Inventory_Level_PPE, Inventory_Level_Medications
        - Restock_Event, Stockout_Risk_Score
        """)

        uploaded_file = st.file_uploader(
            "Upload inventory data CSV",
            type=["csv"],
            key="inv_csv_upload"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Parse date
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])

                st.session_state["inventory_data_df"] = df
                st.session_state["inventory_data_source"] = "CSV Upload"
                st.success(f"✅ Loaded {len(df)} records from CSV!")

                # Auto-populate current inventory from latest record
                latest = df.sort_values("Date").iloc[-1] if "Date" in df.columns else df.iloc[-1]
                current_inv = {}
                if "Inventory_Level_Gloves" in df.columns:
                    current_inv["GLV001"] = int(latest.get("Inventory_Level_Gloves", 0))
                if "Inventory_Level_PPE" in df.columns:
                    current_inv["PPE001"] = int(latest.get("Inventory_Level_PPE", 0))
                if "Inventory_Level_Medications" in df.columns:
                    current_inv["MED001"] = int(latest.get("Inventory_Level_Medications", 0))
                st.session_state["current_inventory"] = current_inv

            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    # Display loaded data
    st.markdown("---")
    if st.session_state.get("inventory_data_df") is not None:
        df = st.session_state["inventory_data_df"]
        source = st.session_state.get("inventory_data_source", "Unknown")

        st.markdown(f"#### Loaded Data ({source})")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Total Records</div>
                <div class="inv-value">{len(df):,}</div>
                <div class="inv-label">inventory entries</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if "Date" in df.columns:
                date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                days = (df['Date'].max() - df['Date'].min()).days
            else:
                date_range = "N/A"
                days = 0

            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Date Range</div>
                <div class="inv-value">{days}</div>
                <div class="inv-label">days</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if "Restock_Event" in df.columns:
                restock_count = int(df["Restock_Event"].sum())
            else:
                restock_count = 0

            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Restock Events</div>
                <div class="inv-value">{restock_count}</div>
                <div class="inv-label">total</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            if "Stockout_Risk_Score" in df.columns:
                avg_risk = df["Stockout_Risk_Score"].mean()
                risk_color = "#22c55e" if avg_risk < 0.3 else ("#facc15" if avg_risk < 0.6 else "#ef4444")
            else:
                avg_risk = 0
                risk_color = "#22c55e"

            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Avg Stockout Risk</div>
                <div class="inv-value" style="color: {risk_color};">{avg_risk:.2f}</div>
                <div class="inv-label">risk score</div>
            </div>
            """, unsafe_allow_html=True)

        # Data preview
        st.markdown("#### Data Preview (First 10 Rows)")
        display_df = df.head(10).copy()

        # Add simple index
        display_df.insert(0, "ID", range(1, len(display_df) + 1))

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing 10 of {len(df)} records")

        # Usage statistics
        st.markdown("#### Daily Usage Statistics")
        usage_cols = [c for c in df.columns if "Inventory_Used" in c]
        if usage_cols:
            usage_stats = df[usage_cols].describe().round(1)
            st.dataframe(usage_stats, use_container_width=True)

            # Usage chart
            if "Date" in df.columns:
                st.markdown("#### Usage Trends Over Time")
                usage_df = df[["Date"] + usage_cols].copy()
                usage_melted = usage_df.melt(id_vars=["Date"], var_name="Item", value_name="Used")

                fig = px.line(
                    usage_melted,
                    x="Date",
                    y="Used",
                    color="Item",
                    template="plotly_dark",
                    title="Daily Inventory Usage"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No inventory data loaded. Select a data source above to load historical inventory data.")


# =============================================================================
# TAB 1: DEMAND FORECAST (Connection to Forecast Hub)
# =============================================================================
with tab_forecast:
    st.markdown("### 📊 Patient Demand Forecast")
    st.info("""
    **Inventory demand is derived from patient forecasts:**
    - Formula: `d_{i,t} = α_i × D̂_t`
    - Where α_i = usage rate per patient for item i
    - D̂_t = forecasted patient arrivals on day t
    """)

    # Detect forecast sources
    forecast_sources = detect_forecast_sources()

    col1, col2 = st.columns([2, 1])

    with col1:
        if forecast_sources:
            st.success(f"✅ {len(forecast_sources)} forecast source(s) detected!")

            # Select source
            source_names = [s["name"] for s in forecast_sources]
            selected_source_name = st.selectbox(
                "Select forecast source:",
                options=source_names,
                key="inv_forecast_source"
            )

            selected_source = next(s for s in forecast_sources if s["name"] == selected_source_name)

            # Planning horizon
            planning_horizon = st.slider(
                "Planning Horizon (days)",
                min_value=7,
                max_value=90,
                value=30,
                step=7,
                key="inv_planning_horizon"
            )

            # Extract forecast
            patient_forecast = extract_patient_forecast(selected_source, planning_horizon)

            if patient_forecast:
                st.session_state["patient_forecast"] = patient_forecast

                # Calculate RPIW
                rpiw = calculate_rpiw(patient_forecast)
                recommendation = get_optimization_recommendation(rpiw)

                # Display RPIW
                st.markdown("---")
                st.markdown("#### Forecast Quality Assessment (RPIW)")

                rpiw_class = "rpiw-low" if rpiw < 15 else ("rpiw-medium" if rpiw <= 40 else "rpiw-high")

                st.markdown(f"""
                <div class="{rpiw_class}" style="margin-bottom: 1rem;">
                    <strong>RPIW: {rpiw:.1f}%</strong> — {recommendation['uncertainty_level']} Uncertainty
                </div>
                """, unsafe_allow_html=True)

                if rpiw < 15:
                    st.success(f"✅ **{recommendation['method']}** is appropriate. {recommendation['description']}")
                elif rpiw <= 40:
                    st.warning(f"⚠️ **{recommendation['method']}** recommended. {recommendation['description']}")
                else:
                    st.error(f"❌ **{recommendation['method']}** recommended. Current deterministic approach may be insufficient.")

                # Forecast preview
                st.markdown("#### Forecast Preview")
                forecast_df = pd.DataFrame({
                    "Day": list(range(1, len(patient_forecast) + 1)),
                    "Patient Arrivals": [round(p, 1) for p in patient_forecast]
                })

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_df["Day"],
                    y=forecast_df["Patient Arrivals"],
                    mode='lines+markers',
                    name='Patient Forecast',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title="Patient Arrival Forecast",
                    xaxis_title="Day",
                    yaxis_title="Patients",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Could not extract forecast from selected source.")

        else:
            st.warning("⚠️ No forecast sources found.")
            st.info("""
            Please generate a forecast first in:
            - **Forecast Hub** (10_Forecast.py) - Recommended
            - **Benchmarks** (05_Benchmarks.py) - ARIMA/SARIMAX
            - **Modeling Hub** (08_Modeling_Hub.py) - ML models
            """)

            # Manual input option
            st.markdown("---")
            st.markdown("#### Manual Demand Input")
            manual_demand = st.number_input(
                "Average daily patient arrivals",
                min_value=10,
                max_value=1000,
                value=150
            )
            manual_horizon = st.slider("Planning horizon (days)", 7, 90, 30)

            if st.button("Use Manual Demand"):
                st.session_state["patient_forecast"] = [float(manual_demand)] * manual_horizon
                st.success(f"✅ Using manual demand: {manual_demand} patients/day for {manual_horizon} days")

    with col2:
        st.markdown("#### Summary")
        if st.session_state.get("patient_forecast"):
            pf = st.session_state["patient_forecast"]
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Average Daily Demand</div>
                <div class="inv-value">{np.mean(pf):.0f}</div>
                <div class="inv-label">patients/day</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Total Period Demand</div>
                <div class="inv-value">{sum(pf):,.0f}</div>
                <div class="inv-label">patients over {len(pf)} days</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Demand Variability</div>
                <div class="inv-value">{np.std(pf):.1f}</div>
                <div class="inv-label">std deviation</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: INVENTORY ITEMS
# =============================================================================
with tab_items:
    st.markdown("### 📋 Healthcare Inventory Items")
    st.info("Configure inventory items with usage rates, costs, and criticality levels.")

    items = st.session_state["inventory_items"]

    # Item overview table
    item_data = []
    for item in items:
        crit_class = {
            ItemCriticality.LOW: "crit-low",
            ItemCriticality.MEDIUM: "crit-medium",
            ItemCriticality.HIGH: "crit-high",
            ItemCriticality.CRITICAL: "crit-critical",
        }[item.criticality]

        item_data.append({
            "ID": item.item_id,
            "Name": item.name,
            "Category": item.category,
            "Unit Cost": f"${item.unit_cost:.2f}",
            "Usage/Patient": item.usage_rate,
            "Lead Time": f"{item.lead_time} days",
            "Criticality": item.criticality.name,
            "Stockout Penalty": f"${item.effective_stockout_penalty:.2f}",
        })

    item_df = pd.DataFrame(item_data)
    st.dataframe(item_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Edit Item Parameters")

    # Select item to edit
    item_names = [item.name for item in items]
    selected_item_name = st.selectbox("Select item to edit:", item_names)
    selected_item_idx = item_names.index(selected_item_name)
    selected_item = items[selected_item_idx]

    col1, col2, col3 = st.columns(3)

    with col1:
        new_unit_cost = st.number_input(
            "Unit Cost ($)",
            min_value=0.01,
            value=float(selected_item.unit_cost),
            step=0.5,
            key="edit_unit_cost"
        )
        new_usage_rate = st.number_input(
            "Usage Rate (per patient)",
            min_value=0.01,
            value=float(selected_item.usage_rate),
            step=0.1,
            key="edit_usage_rate"
        )

    with col2:
        new_lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=30,
            value=int(selected_item.lead_time),
            key="edit_lead_time"
        )
        new_ordering_cost = st.number_input(
            "Ordering Cost ($)",
            min_value=1.0,
            value=float(selected_item.ordering_cost),
            step=5.0,
            key="edit_ordering_cost"
        )

    with col3:
        new_stockout_penalty = st.number_input(
            "Stockout Penalty ($)",
            min_value=1.0,
            value=float(selected_item.stockout_penalty),
            step=10.0,
            key="edit_stockout_penalty"
        )
        criticality_options = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        new_criticality = st.selectbox(
            "Criticality",
            options=criticality_options,
            index=criticality_options.index(selected_item.criticality.name),
            key="edit_criticality"
        )

    if st.button("Update Item"):
        # Update item
        items[selected_item_idx] = HealthcareInventoryItem(
            item_id=selected_item.item_id,
            name=selected_item.name,
            category=selected_item.category,
            unit_cost=new_unit_cost,
            holding_cost_rate=selected_item.holding_cost_rate,
            ordering_cost=new_ordering_cost,
            stockout_penalty=new_stockout_penalty,
            lead_time=new_lead_time,
            usage_rate=new_usage_rate,
            min_order_qty=selected_item.min_order_qty,
            max_order_qty=selected_item.max_order_qty,
            shelf_life=selected_item.shelf_life,
            criticality=ItemCriticality[new_criticality],
            volume_per_unit=selected_item.volume_per_unit,
        )
        st.session_state["inventory_items"] = items
        st.success(f"✅ Updated {selected_item.name}")
        st.rerun()

    # Current inventory levels
    st.markdown("---")
    st.markdown("#### Current Inventory Levels")

    current_inv = st.session_state.get("current_inventory", {})

    inv_cols = st.columns(4)
    for idx, item in enumerate(items):
        with inv_cols[idx % 4]:
            current_level = st.number_input(
                f"{item.name[:25]}...",
                min_value=0,
                value=int(current_inv.get(item.item_id, 0)),
                key=f"curr_inv_{item.item_id}"
            )
            current_inv[item.item_id] = current_level

    st.session_state["current_inventory"] = current_inv


# =============================================================================
# TAB 3: EOQ ANALYSIS
# =============================================================================
with tab_eoq:
    st.markdown("### 📐 Economic Order Quantity (EOQ) Analysis")
    st.info("""
    **EOQ Formula:** Q* = √(2 × K × D / h)

    Where:
    - K = Fixed ordering cost per order
    - D = Annual demand
    - h = Annual holding cost per unit
    """)

    patient_forecast = st.session_state.get("patient_forecast")
    items = st.session_state["inventory_items"]

    if not patient_forecast:
        st.warning("⚠️ No patient forecast available. Please configure in Tab 1.")
        st.stop()

    # Service level
    service_level = st.slider(
        "Service Level (CSL)",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        key="eoq_service_level"
    )

    if st.button("Calculate EOQ for All Items", type="primary"):
        with st.spinner("Calculating EOQ..."):
            eoq_results = []

            # Average daily patient demand
            avg_daily_patients = np.mean(patient_forecast)
            std_daily_patients = np.std(patient_forecast)

            for item in items:
                # Convert patient demand to item demand
                daily_demand = item.usage_rate * avg_daily_patients
                demand_std = item.usage_rate * std_daily_patients

                # Calculate EOQ
                result = calculate_eoq(
                    item=item,
                    daily_demand=daily_demand,
                    demand_std=demand_std,
                    service_level=service_level
                )
                eoq_results.append(result)

            st.session_state["eoq_results"] = eoq_results
            st.success("✅ EOQ calculation complete!")

    # Display results
    if st.session_state.get("eoq_results"):
        results = st.session_state["eoq_results"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_annual_cost = sum(r.total_annual_cost for r in results)
        total_orders = sum(r.orders_per_year for r in results)
        avg_cycle = np.mean([r.cycle_time_days for r in results])

        with col1:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Total Annual Cost</div>
                <div class="inv-value">${total_annual_cost:,.0f}</div>
                <div class="inv-label">all items</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Total Orders/Year</div>
                <div class="inv-value">{total_orders:.0f}</div>
                <div class="inv-label">across all items</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Avg Cycle Time</div>
                <div class="inv-value">{avg_cycle:.0f}</div>
                <div class="inv-label">days</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Items Analyzed</div>
                <div class="inv-value">{len(results)}</div>
                <div class="inv-label">items</div>
            </div>
            """, unsafe_allow_html=True)

        # Results table
        st.markdown("---")
        st.markdown("#### EOQ Results by Item")

        eoq_df = pd.DataFrame([r.to_dict() for r in results])
        st.dataframe(eoq_df, use_container_width=True, hide_index=True)

        # Cost breakdown chart
        st.markdown("#### Cost Breakdown")
        cost_data = pd.DataFrame({
            "Item": [r.item_name[:20] for r in results],
            "Ordering Cost": [r.annual_ordering_cost for r in results],
            "Holding Cost": [r.annual_holding_cost for r in results],
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Ordering Cost',
            x=cost_data["Item"],
            y=cost_data["Ordering Cost"],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            name='Holding Cost',
            x=cost_data["Item"],
            y=cost_data["Holding Cost"],
            marker_color='#22c55e'
        ))
        fig.update_layout(
            barmode='stack',
            title="Annual Inventory Cost by Item",
            xaxis_title="Item",
            yaxis_title="Cost ($)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 4: MILP OPTIMIZATION
# =============================================================================
with tab_milp:
    st.markdown("### 🔧 Multi-Item MILP Optimization")
    st.info("""
    **Objective:** Minimize total cost = Ordering + Purchase + Holding + Stockout

    **Constraints:**
    - Inventory balance equation
    - Storage capacity limits
    - Minimum/maximum order quantities
    - Budget constraints (optional)
    """)

    patient_forecast = st.session_state.get("patient_forecast")
    items = st.session_state["inventory_items"]
    current_inventory = st.session_state.get("current_inventory", {})

    if not patient_forecast:
        st.warning("⚠️ No patient forecast available. Please configure in Tab 1.")
        st.stop()

    # MILP parameters
    st.markdown("#### Optimization Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        planning_horizon = st.number_input(
            "Planning Horizon (days)",
            min_value=7,
            max_value=90,
            value=min(30, len(patient_forecast)),
            key="milp_horizon"
        )
        storage_capacity = st.number_input(
            "Storage Capacity (cubic meters)",
            min_value=10.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            key="milp_storage"
        )

    with col2:
        service_level = st.slider(
            "Service Level",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            key="milp_service_level"
        )
        use_budget = st.checkbox("Enable Budget Constraint", value=False)

    with col3:
        if use_budget:
            daily_budget = st.number_input(
                "Daily Budget ($)",
                min_value=100.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0,
                key="milp_budget"
            )
        else:
            daily_budget = None

    st.markdown("---")

    if st.button("Run MILP Optimization", type="primary"):
        with st.spinner("Solving MILP... This may take a moment."):
            try:
                # Convert patient forecast to inventory demand
                demand_forecast = convert_patient_forecast_to_inventory_demand(
                    patient_forecast[:planning_horizon],
                    items
                )

                # Initialize current inventory
                init_inv = {}
                for item in items:
                    init_inv[item.item_id] = current_inventory.get(item.item_id, 0)

                # Solve MILP
                result = solve_milp_inventory(
                    items=items,
                    demand_forecast=demand_forecast,
                    planning_horizon=planning_horizon,
                    initial_inventory=init_inv,
                    storage_capacity=storage_capacity,
                    daily_budget=daily_budget,
                    service_level=service_level,
                )

                st.session_state["milp_results"] = result

                if result.status == "Optimal":
                    st.success(f"✅ Optimization complete! Status: {result.status}")
                    st.balloons()
                else:
                    st.warning(f"⚠️ Optimization finished with status: {result.status}")

            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")

    # Display MILP results
    if st.session_state.get("milp_results"):
        result = st.session_state["milp_results"]

        if result.status == "Optimal":
            # Summary metrics
            st.markdown("---")
            st.markdown("#### Optimization Results")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Total Cost</div>
                    <div class="inv-value">${result.total_cost:,.0f}</div>
                    <div class="inv-label">over horizon</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Purchase Cost</div>
                    <div class="inv-value">${result.purchase_cost:,.0f}</div>
                    <div class="inv-label">{result.purchase_cost/result.total_cost*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Ordering Cost</div>
                    <div class="inv-value">${result.ordering_cost:,.0f}</div>
                    <div class="inv-label">{result.ordering_cost/result.total_cost*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Holding Cost</div>
                    <div class="inv-value">${result.holding_cost:,.0f}</div>
                    <div class="inv-label">{result.holding_cost/result.total_cost*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                st.markdown(f"""
                <div class="inv-card">
                    <div class="inv-title">Stockout Cost</div>
                    <div class="inv-value">${result.stockout_cost:,.0f}</div>
                    <div class="inv-label">{result.stockout_cost/result.total_cost*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Cost breakdown pie chart
            st.markdown("#### Cost Breakdown")
            cost_breakdown = {
                "Purchase": result.purchase_cost,
                "Ordering": result.ordering_cost,
                "Holding": result.holding_cost,
                "Stockout": result.stockout_cost,
            }

            fig = go.Figure(data=[go.Pie(
                labels=list(cost_breakdown.keys()),
                values=list(cost_breakdown.values()),
                hole=0.4,
                marker_colors=['#3b82f6', '#22c55e', '#f97316', '#ef4444']
            )])
            fig.update_layout(
                title="Cost Distribution",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # Order schedule
            st.markdown("#### Order Schedule")
            if not result.order_schedule.empty:
                st.dataframe(result.order_schedule, use_container_width=True, hide_index=True)

            # Inventory levels chart
            st.markdown("#### Inventory Levels Over Time")
            if not result.inventory_levels.empty:
                inv_df = result.inventory_levels.melt(
                    id_vars=["Day"],
                    var_name="Item",
                    value_name="Inventory"
                )

                fig = px.line(
                    inv_df,
                    x="Day",
                    y="Inventory",
                    color="Item",
                    template="plotly_dark",
                    title="Inventory Levels by Item"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Reorder points and safety stocks
            st.markdown("#### Reorder Points & Safety Stocks")

            rop_data = []
            for item in items:
                rop_data.append({
                    "Item": item.name,
                    "Reorder Point": round(result.reorder_points.get(item.item_id, 0)),
                    "Safety Stock": round(result.safety_stocks.get(item.item_id, 0)),
                    "Lead Time": f"{item.lead_time} days",
                })

            rop_df = pd.DataFrame(rop_data)
            st.dataframe(rop_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5: RESULTS & ALERTS
# =============================================================================
with tab_results:
    st.markdown("### 📈 Results & Reorder Alerts")

    items = st.session_state["inventory_items"]
    current_inventory = st.session_state.get("current_inventory", {})
    milp_results = st.session_state.get("milp_results")
    eoq_results = st.session_state.get("eoq_results")

    # Use reorder points from MILP or EOQ
    reorder_points = {}
    if milp_results and milp_results.status == "Optimal":
        reorder_points = milp_results.reorder_points
    elif eoq_results:
        for r in eoq_results:
            reorder_points[r.item_id] = r.reorder_point

    if reorder_points:
        # Generate alerts
        alerts = generate_reorder_alerts(
            current_inventory=current_inventory,
            reorder_points=reorder_points,
            items=items
        )

        if alerts:
            st.markdown("#### Reorder Alerts")

            for alert in alerts:
                alert_class = f"alert-{alert['urgency'].lower()}"
                st.markdown(f"""
                <div class="{alert_class}" style="margin-bottom: 0.5rem;">
                    <strong>{alert['urgency']}</strong>: {alert['item_name']}<br>
                    Current: {alert['current_inventory']:.0f} | Reorder Point: {alert['reorder_point']:.0f}<br>
                    <em>Recommended Order: {alert['recommended_order']:.0f} units (Lead time: {alert['lead_time']} days)</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ All inventory levels are above reorder points!")

        st.markdown("---")

        # Inventory status chart
        st.markdown("#### Inventory Status Overview")

        status_data = []
        for item in items:
            current = current_inventory.get(item.item_id, 0)
            rop = reorder_points.get(item.item_id, 0)

            status = "OK"
            color = "#22c55e"
            if current <= rop * 0.5:
                status = "CRITICAL"
                color = "#ef4444"
            elif current <= rop * 0.75:
                status = "LOW"
                color = "#f97316"
            elif current <= rop:
                status = "REORDER"
                color = "#facc15"

            status_data.append({
                "item": item.name[:20],
                "current": current,
                "reorder_point": rop,
                "status": status,
                "color": color
            })

        status_df = pd.DataFrame(status_data)

        fig = go.Figure()

        # Current inventory bars
        fig.add_trace(go.Bar(
            name='Current Inventory',
            x=status_df["item"],
            y=status_df["current"],
            marker_color=[d["color"] for d in status_data],
        ))

        # Reorder point line
        fig.add_trace(go.Scatter(
            name='Reorder Point',
            x=status_df["item"],
            y=status_df["reorder_point"],
            mode='markers+lines',
            marker=dict(symbol='line-ew', size=15, color='#ef4444'),
            line=dict(color='#ef4444', dash='dash')
        ))

        fig.update_layout(
            title="Current Inventory vs Reorder Points",
            xaxis_title="Item",
            yaxis_title="Units",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run EOQ or MILP optimization first to generate reorder alerts.")

    # Export options
    st.markdown("---")
    st.markdown("#### Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if eoq_results:
            eoq_df = pd.DataFrame([r.to_dict() for r in eoq_results])
            csv = eoq_df.to_csv(index=False)
            st.download_button(
                "📥 Download EOQ Results",
                data=csv,
                file_name="eoq_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        if milp_results and milp_results.status == "Optimal":
            order_csv = milp_results.order_schedule.to_csv(index=False)
            st.download_button(
                "📥 Download Order Schedule",
                data=order_csv,
                file_name="order_schedule.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col3:
        if reorder_points and current_inventory:
            alerts = generate_reorder_alerts(current_inventory, reorder_points, items)
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_csv = alerts_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Alerts",
                    data=alerts_csv,
                    file_name="reorder_alerts.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Inventory Management Optimization</strong><br>
    Deterministic EOQ & MILP for Healthcare | Demand-driven by ML Patient Forecasts<br>
    <em>Use when RPIW < 15% (narrow prediction intervals)</em>
</div>
""", unsafe_allow_html=True)
