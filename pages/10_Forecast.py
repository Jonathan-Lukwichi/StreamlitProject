# =============================================================================
# 10_Forecast.py — Forecast Hub (Foundation Only)
# Configure forecast horizon/date range, choose model, generate & preview
# =============================================================================
from __future__ import annotations
import streamlit as st
from datetime import date, timedelta

# Optional theme hook
try:
    from app_core.ui.theme import apply_css
    apply_css()
except Exception:
    pass

st.set_page_config(page_title="Forecast", layout="wide")
st.title("🔮 Forecast")

# -------------------------------------------------------------
# Expected session keys later (filled by training/runtime):
# st.session_state["trained_models"]     -> list[str] of model names available to forecast
# st.session_state["default_model_name"] -> str recommended default
# st.session_state["forecast_df"]        -> last generated forecast dataframe (any schema you choose)
# st.session_state["forecast_fig"]       -> plotly.Figure for latest forecast
# st.session_state["feature_catalog"]    -> list[str] of exogenous columns to use at forecast time
# -------------------------------------------------------------

with st.expander("📂 Data & Session (placeholder)", expanded=False):
    st.info("This page consumes trained models and feature data once available.")
    st.write("Available session keys:", list(st.session_state.keys()))

# Model picker
st.subheader("Model selection")
available_models = st.session_state.get("trained_models", [
    "ARIMA(placeholder)", "SARIMAX(placeholder)", "XGBoost(placeholder)",
    "LSTM(placeholder)", "ANN(placeholder)", "LSTM-XGBoost(placeholder)"
])
default_model = st.session_state.get("default_model_name", "XGBoost(placeholder)")
model_name = st.selectbox(
    "Choose the model to forecast with",
    options=available_models,
    index=available_models.index(default_model) if default_model in available_models else 0
)

# Forecast configuration
st.subheader("Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    mode = st.radio("Mode", ["Multi-horizon", "Date range"], index=0, horizontal=True)
with c2:
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=7)
with c3:
    include_intervals = st.checkbox("Prediction intervals (placeholder)", value=True)

# Date controls (used when Date range mode is active)
today = date.today()
start_date = st.date_input("Start date", today + timedelta(days=1))
end_date = st.date_input("End date", today + timedelta(days=7))

# Exogenous at forecast time (placeholder UI)
st.subheader("Exogenous inputs (optional)")
feature_catalog = st.session_state.get("feature_catalog", [])
use_exog = st.checkbox("Provide exogenous features at forecast time", value=False)
if use_exog:
    st.info("In the real app, you can upload a CSV or map constant values here.")
    exog_cols = st.multiselect("Select exogenous columns", options=feature_catalog)
    st.text_area("Enter constant values JSON (placeholder)", value='{"Average_Temp": 20, "Holiday": 0}')

# Run + preview (placeholders)
st.divider()
left, right = st.columns([1, 2])
with left:
    st.button("🚀 Generate Forecast (placeholder)", help="This will call the selected model later.")
    st.button("💾 Save Forecast (placeholder)")
with right:
    st.markdown("#### Preview")
    if st.session_state.get("forecast_fig") is not None:
        st.plotly_chart(st.session_state["forecast_fig"], use_container_width=True)
    else:
        st.info("No forecast plotted yet. Run a forecast to preview here.")
    if st.session_state.get("forecast_df") is not None:
        st.dataframe(st.session_state["forecast_df"], use_container_width=True)
    else:
        st.caption("When available, the forecast dataframe will appear here.")

st.divider()
st.caption("This is a foundation layout. Wire the buttons to your pipelines and set the session keys shown above.")
