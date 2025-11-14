# =============================================================================
# 09_Results.py — Results Hub (Foundation Only)
# View metrics, plots, and saved artifacts from any model
# =============================================================================
from __future__ import annotations
import streamlit as st

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

st.set_page_config(
    page_title="Results - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR RESULTS
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
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>📊</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Results</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        View comprehensive metrics, diagnostic plots, and saved artifacts from your trained forecasting models
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Expected session keys (to be filled by your training pipelines later):
# st.session_state["results_summary_df"]   -> pd.DataFrame of overall metrics by model
# st.session_state["results_by_horizon"]   -> dict[str,pd.DataFrame] per horizon or a single df
# st.session_state["residual_plots"]       -> dict[str, plotly.Figure] e.g., {"ARIMA": fig, ...}
# st.session_state["forecast_plots"]       -> dict[str, plotly.Figure]
# st.session_state["artifacts"]            -> dict with paths/metadata (model pickles, params, etc.)
# -------------------------------------------------------------

with st.expander("📂 Data & Session (placeholder)", expanded=False):
    st.info("These placeholders will populate once training is implemented.")
    st.write("Available session keys:", list(st.session_state.keys()))

# Top summary section
st.subheader("Overview")
left, right = st.columns([2, 1])
with left:
    st.caption("High-level summary of best model(s) and metrics (placeholder).")
    best_model = st.session_state.get("best_model_name", "(not set)")
    st.metric("Best Model (placeholder)", best_model)
with right:
    st.selectbox(
        "Primary score to display", 
        ["RMSE", "MAE", "MAPE", "R²"], 
        key="results_primary_score"
    )

# Tabs for different result views
tab_summary, tab_horizons, tab_plots, tab_artifacts = st.tabs(
    ["📋 Summary Table", "🗓️ Per-Horizon", "📈 Plots", "📦 Artifacts"]
)

with tab_summary:
    st.markdown("#### Comparison table (by model)")
    if "results_summary_df" in st.session_state:
        st.dataframe(st.session_state["results_summary_df"], use_container_width=True)
    else:
        st.info("No summary table yet. Train a model to populate this.")

with tab_horizons:
    st.markdown("#### Metrics by forecast horizon")
    horizon_opt = st.selectbox("Select horizon (placeholder)", ["1","2","3","4","5","6","7"], index=0)
    if "results_by_horizon" in st.session_state:
        horizon_map = st.session_state["results_by_horizon"]
        df = horizon_map.get(str(horizon_opt)) if isinstance(horizon_map, dict) else None
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No per-horizon metrics yet for this selection.")
    else:
        st.info("No per-horizon metrics yet.")

with tab_plots:
    st.markdown("#### Diagnostics & Forecast plots (placeholders)")
    plot_type = st.radio(
        "Plot type", ["Residual diagnostics", "Forecast vs Actual"], horizontal=True
    )
    container = st.container()
    figs_key = "residual_plots" if "Residual" in plot_type else "forecast_plots"
    figs = st.session_state.get(figs_key, {})
    if isinstance(figs, dict) and figs:
        model_name = st.selectbox("Select model", options=list(figs.keys()))
        fig = figs.get(model_name)
        if fig is not None:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No figure available for this model yet.")
    else:
        st.info("No figures stored yet. Once training runs, plots will appear here.")

with tab_artifacts:
    st.markdown("#### Saved artifacts (placeholders)")
    art = st.session_state.get("artifacts", {})
    if art:
        for name, meta in art.items():
            with st.expander(f"{name}"):
                st.write(meta)
                st.button(f"Download {name}", key=f"dl_{name}")  # wire later
    else:
        st.info("No artifacts registered. After training, model files/metadata will list here.")

st.divider()
st.caption("This page will read from `st.session_state` as your pipelines produce outputs.")
