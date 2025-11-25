# =============================================================================
# 10_Forecast.py — Forecast Hub (Foundation Only)
# Configure forecast horizon/date range, choose model, generate & preview
# =============================================================================
from __future__ import annotations
import streamlit as st
from datetime import date, timedelta

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
    page_title="Forecast - HealthForecast AI",
    page_icon="🔮",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR FORECAST
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
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🔮</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Forecast</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Generate and visualize future predictions using your trained models with configurable horizons and confidence intervals
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
