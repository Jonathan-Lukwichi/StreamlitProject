# =============================================================================
# 12_Inventory_Management_Optimization.py — Foundation Only
# Configure inventory parameters, constraints, and preview optimization results
# =============================================================================
from __future__ import annotations
import streamlit as st

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

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR INVENTORY MANAGEMENT
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
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>📦</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Inventory Management Optimization</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Optimize inventory levels, reorder points, and procurement decisions to balance costs and service levels with advanced algorithms
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Expected future session keys:
# st.session_state["inventory_df"]
# st.session_state["inventory_solution_df"]
# st.session_state["inventory_fig"]
# -------------------------------------------------------------

st.markdown("### Objective")
st.info(
    "This module will optimize inventory levels, reorder points, and procurement decisions "
    "to balance holding costs, shortages, and service-level targets."
)

# Input configuration
st.subheader("Input Configuration")
c1, c2 = st.columns(2)
with c1:
    st.file_uploader("📂 Upload inventory dataset (CSV or Excel)", type=["csv", "xlsx"])
    st.text_input("Product category filter (optional)")
with c2:
    st.selectbox("Forecast source", ["Use Forecast Module Output", "Manual Input"])
    st.number_input("Planning horizon (days)", min_value=1, max_value=180, value=30)

# Constraint configuration
st.subheader("Constraints (placeholder)")
st.checkbox("Enforce safety stock limits", value=True)
st.checkbox("Respect storage capacity", value=True)
st.checkbox("Include lead time variability", value=False)
st.checkbox("Account for supplier minimum order quantity", value=False)

# Optimization parameters
st.subheader("Optimization Settings")
c1, c2, c3 = st.columns(3)
with c1:
    st.radio("Mode", ["Automatic (solver-based)", "Manual tuning"], index=0)
with c2:
    st.selectbox("Solver type (placeholder)", ["Linear Programming", "Dynamic Programming", "Reinforcement Learning"])
with c3:
    st.number_input("Max iterations", min_value=10, max_value=5000, value=500)

# Run + preview (placeholders)
st.divider()
left, right = st.columns([1, 2])
with left:
    st.button("🚀 Optimize Inventory (placeholder)")
    st.button("💾 Save Results (placeholder)")
with right:
    st.markdown("### Optimization Preview")
    st.info("Placeholder for optimized reorder table or chart.")
    if "inventory_fig" in st.session_state:
        st.plotly_chart(st.session_state["inventory_fig"], use_container_width=True)
    elif "inventory_solution_df" in st.session_state:
        st.dataframe(st.session_state["inventory_solution_df"], use_container_width=True)
    else:
        st.caption("No results yet. Run optimization first.")

st.divider()
st.caption("This page sets up the foundation for algorithms like EOQ, MILP, or stochastic inventory models.")
