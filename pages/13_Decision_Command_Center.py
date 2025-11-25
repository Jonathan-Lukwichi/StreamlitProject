# =============================================================================
# 13_Decision_Command_Center.py — AI-Enhanced Foundation
# A unified command center for decisions + explainability + AI guidance
# =============================================================================
from __future__ import annotations
import streamlit as st
from datetime import date

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
    page_title="Decision Command Center - HealthForecast AI",
    page_icon="🧭",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR DECISION COMMAND CENTER
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
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧭</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Decision Command Center</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        A unified command center for AI-enhanced decisions, explainability, and operational guidance with intelligent recommendations
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SESSION EXPECTATIONS (future integrations)
# -----------------------------------------------------------------------------
# st.session_state["results_summary_df"]       -> model metrics
# st.session_state["forecast_fig"]             -> forecast plots
# st.session_state["schedule_solution_df"]     -> staff optimization results
# st.session_state["inventory_solution_df"]    -> inventory optimization results
# st.session_state["shap_fig"] / ["lime_fig"]  -> explainability visuals
# st.session_state["recommendations"]          -> list of text actions
# st.session_state["ai_agent_history"]         -> list of chat turns (for persistence)
# -----------------------------------------------------------------------------

cfg = st.session_state.setdefault("dcc_config", {
    "primary_kpi": "RMSE",
    "decision_horizon_days": 7,
    "confidence_level": 0.9,
    "ai_agent_active": True
})

# ==============================
# Global configuration section
# ==============================
with st.expander("⚙️ Global Decision Settings (placeholders)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["decision_horizon_days"] = st.number_input(
            "Decision horizon (days)", 1, 60, int(cfg.get("decision_horizon_days", 7))
        )
    with c2:
        cfg["primary_kpi"] = st.selectbox(
            "Primary KPI", ["RMSE", "MAE", "MAPE", "R²"],
            index=["RMSE","MAE","MAPE","R²"].index(cfg.get("primary_kpi","RMSE"))
        )
    with c3:
        cfg["confidence_level"] = st.slider("Confidence level (placeholder)", 0.5, 0.99, 0.9)

# ==============================
# KPI summary deck
# ==============================
st.subheader("📊 Situation at a Glance")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Best Model", st.session_state.get("best_model_name", "—"))
with k2: st.metric("Primary KPI", cfg["primary_kpi"])
with k3: st.metric("Decision Horizon (days)", cfg["decision_horizon_days"])
with k4: st.metric("Confidence Level", f"{int(cfg['confidence_level']*100)}%")

# ==============================
# Tabs for multi-domain insights
# ==============================
tab_overview, tab_explain, tab_staff, tab_inventory, tab_ai = st.tabs(
    ["🏁 Overview", "🧩 Explainable AI", "👥 Staffing", "📦 Inventory", "🤖 AI Assistant"]
)

# ------------------- Overview -------------------
with tab_overview:
    st.markdown("### Consolidated Overview (placeholder)")
    if "results_summary_df" in st.session_state:
        st.dataframe(st.session_state["results_summary_df"], use_container_width=True)
    else:
        st.info("No summary yet. Train models to populate results.")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Forecast vs Actual")
        if st.session_state.get("forecast_fig") is not None:
            st.plotly_chart(st.session_state["forecast_fig"], use_container_width=True)
        else:
            st.info("Forecast plot will appear here.")
    with c2:
        st.caption("Residual Diagnostics (placeholder)")
        st.info("Residual plots or metric charts can appear here later.")

# ------------------- Explainable AI -------------------
with tab_explain:
    st.markdown("### Model Explainability & Feature Importance (placeholders)")
    st.info(
        "This section will visualize model interpretability using SHAP (SHapley Additive Explanations) "
        "and LIME (Local Interpretable Model-agnostic Explanations)."
    )

    mode = st.radio(
        "Choose interpretability method",
        ["SHAP (Global feature importance)", "LIME (Local instance explanation)"],
        horizontal=True
    )

    st.button("🚀 Generate Explanation (placeholder)")
    if "shap_fig" in st.session_state and mode.startswith("SHAP"):
        st.plotly_chart(st.session_state["shap_fig"], use_container_width=True)
    elif "lime_fig" in st.session_state and mode.startswith("LIME"):
        st.plotly_chart(st.session_state["lime_fig"], use_container_width=True)
    else:
        st.caption("Explanation visual will appear here after analysis.")

    st.markdown("**Interpretation Panel (placeholder)**")
    st.text_area(
        "Insights (automated interpretation will appear here)",
        value="Feature impact summary will be generated by SHAP/LIME later.",
        height=120
    )

# ------------------- Staffing -------------------
with tab_staff:
    st.markdown("### Staffing Overview (placeholder)")
    c1, c2 = st.columns(2)
    with c1:
        if "schedule_solution_df" in st.session_state:
            st.dataframe(st.session_state["schedule_solution_df"], use_container_width=True)
        else:
            st.info("No schedule available yet.")
    with c2:
        if "schedule_fig" in st.session_state:
            st.plotly_chart(st.session_state["schedule_fig"], use_container_width=True)
        else:
            st.info("Staff schedule Gantt or summary chart will be shown here.")

# ------------------- Inventory -------------------
with tab_inventory:
    st.markdown("### Inventory Overview (placeholder)")
    c1, c2 = st.columns(2)
    with c1:
        if "inventory_solution_df" in st.session_state:
            st.dataframe(st.session_state["inventory_solution_df"], use_container_width=True)
        else:
            st.info("No inventory results yet.")
    with c2:
        if "inventory_fig" in st.session_state:
            st.plotly_chart(st.session_state["inventory_fig"], use_container_width=True)
        else:
            st.info("Inventory trend or reorder plots will be shown here.")

# ------------------- AI Assistant -------------------
with tab_ai:
    st.markdown("### 🤖 Decision Assistant (foundation)")
    st.caption(
        "This AI agent will later analyze model results, forecasts, and optimization outputs "
        "to generate recommendations or answer questions about the hospital’s operational decisions."
    )

    # Simulated chat interface
    if "ai_agent_history" not in st.session_state:
        st.session_state["ai_agent_history"] = []

    for turn in st.session_state["ai_agent_history"]:
        role = "🧠 Assistant" if turn["role"] == "assistant" else "👤 You"
        st.markdown(f"**{role}:** {turn['content']}")

    user_input = st.text_input("Ask the Decision Assistant (placeholder):", "")
    if st.button("Send", key="ai_agent_send"):
        if user_input.strip():
            st.session_state["ai_agent_history"].append({"role": "user", "content": user_input})
            # Placeholder AI reply
            reply = (
                "This is a foundation. In the full version, the agent will read your model outputs "
                "and suggest operational actions based on forecast trends, staffing gaps, and "
                "inventory projections."
            )
            st.session_state["ai_agent_history"].append({"role": "assistant", "content": reply})
            st.experimental_rerun()

st.divider()
st.caption("This page unifies results, explainability, and AI-based guidance — foundation only.")
