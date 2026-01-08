# =============================================================================
# 01_Dashboard.py - HORIZON Control (Unified Command Center)
# Consolidates: Dashboard, Patient Forecast, Staff Planner, Supply Planner, Action Center
# =============================================================================
"""
HORIZON Control - Advanced Predictive Intelligence Hub

This page consolidates 5 operational pages into a single, stunning, interactive dashboard:
- Executive Summary (Command Center tab)
- Patient Forecast (Forecast tab)
- Staff Optimization (Staff tab)
- Inventory Optimization (Supply tab)
- Prioritized Actions (Actions tab)

Design Principles:
1. Progressive Disclosure - Simple default view, detailed on demand
2. North Star Header - Hero KPIs always visible
3. Tab-Based Navigation - Focused workflows
4. Premium Aesthetics - Glassmorphism, animations, gradients
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# AUTHENTICATION & CORE IMPORTS
# =============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="HORIZON Control - HealthForecast AI",
    page_icon="🛸",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# Unified Dashboard Components
from app_core.ui.unified_dashboard import (
    get_unified_dashboard_styles,
    render_northstar_header,
    render_premium_card,
    render_sync_status,
    render_action_card,
    render_action_list,
    render_comparison_panel,
    render_empty_state,
    render_skeleton_loader,
    render_system_health_gauge,
    render_forecast_day_cards,
    render_section_header,
    render_glass_divider,
    create_forecast_area_chart,
    create_utilization_gauge,
    create_cost_comparison_bar,
    create_category_breakdown_chart,
    CHART_COLORS,
)

# State Management
from app_core.state.unified_dashboard_state import (
    initialize_dashboard_state,
    get_dashboard_state,
    refresh_dashboard_state,
    UnifiedDashboardState,
    ForecastState,
    StaffState,
    InventoryState,
    ActionItem,
)

# =============================================================================
# APPLY STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# Inject unified dashboard premium styles
st.markdown(get_unified_dashboard_styles(), unsafe_allow_html=True)

# Add fluorescent background effects
st.markdown("""
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
<div class="fluorescent-orb orb-3"></div>
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
<div class="sparkle sparkle-4"></div>

<style>
@keyframes float-orb {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.25; }
    50% { transform: translate(30px, -30px) scale(1.05); opacity: 0.35; }
}

.fluorescent-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}

.orb-1 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 10%; right: 15%;
    animation: float-orb 25s ease-in-out infinite;
}

.orb-2 {
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.2), transparent 70%);
    bottom: 15%; left: 10%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}

.orb-3 {
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.18), transparent 70%);
    top: 50%; left: 50%;
    animation: float-orb 35s ease-in-out infinite;
    animation-delay: 10s;
}

@keyframes sparkle {
    0%, 100% { opacity: 0; transform: scale(0); }
    50% { opacity: 0.6; transform: scale(1); }
}

.sparkle {
    position: fixed;
    width: 3px; height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

.sparkle-1 { top: 20%; left: 30%; animation-delay: 0s; }
.sparkle-2 { top: 60%; left: 75%; animation-delay: 1s; }
.sparkle-3 { top: 40%; left: 20%; animation-delay: 2s; }
.sparkle-4 { top: 75%; left: 45%; animation-delay: 1.5s; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SCI-FI EFFECTS - HORIZON CONTROL
# =============================================================================
st.markdown("""
<!-- Holographic Grid -->
<div class="holo-grid"></div>

<!-- Data Streams -->
<div class="data-stream data-stream-1"></div>
<div class="data-stream data-stream-2"></div>
<div class="data-stream data-stream-3"></div>

<!-- Scan Lines Overlay -->
<div class="scan-lines"></div>
""", unsafe_allow_html=True)


# =============================================================================
# STUNNING HERO HEADER - HORIZON CONTROL
# =============================================================================
st.markdown("""
<div class="dashboard-hero corner-brackets">
    <div class="hero-content">
        <div class="hero-eyebrow">
            <span class="status-online">SYSTEM ONLINE</span> · HealthForecast AI
        </div>
        <h1 class="hero-title hero-title-glitch">HORIZON Control</h1>
        <p class="hero-subtitle">
            Advanced predictive intelligence hub. Real-time monitoring,
            autonomous optimization, and mission-critical insights.
        </p>
    </div>
    <div class="hero-glow-line"></div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE AND REFRESH STATE
# =============================================================================
initialize_dashboard_state()
state = refresh_dashboard_state()


# =============================================================================
# NORTH STAR HEADER
# =============================================================================
render_northstar_header(state)


# =============================================================================
# REFRESH BUTTON
# =============================================================================
col_refresh, col_spacer = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Data", key="refresh_dashboard", use_container_width=True):
        state = refresh_dashboard_state()
        st.rerun()


# =============================================================================
# MAIN TAB STRUCTURE
# =============================================================================
tab_command, tab_forecast, tab_staff, tab_supply, tab_actions = st.tabs([
    "📊 Command Center",
    "🔮 Forecast",
    "👥 Staff",
    "📦 Supply",
    "✅ Actions"
])


# =============================================================================
# TAB 1: COMMAND CENTER (Executive Summary)
# =============================================================================
with tab_command:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # System Health Score
    col_health, col_spacer = st.columns([1, 2])
    with col_health:
        render_system_health_gauge(state.system_health_score)

    render_glass_divider()

    # Main KPI Grid
    render_section_header("Operational Overview", "📊")

    cols = st.columns(4)

    with cols[0]:
        render_premium_card(
            title="7-Day Forecast",
            value=f"{int(state.forecast.total_week):,}" if state.forecast.has_forecast else "—",
            icon="🔮",
            subtitle=f"Peak: {int(state.forecast.peak_patients)}" if state.forecast.has_forecast else "No data",
            accent_color="blue"
        )

    with cols[1]:
        render_premium_card(
            title="Staff Utilization",
            value=f"{state.staff.avg_utilization:.0f}%" if state.staff.data_loaded else "—",
            icon="👥",
            subtitle="Doctors, Nurses, Support",
            accent_color="purple" if state.staff.avg_utilization >= 80 else "orange"
        )

    with cols[2]:
        render_premium_card(
            title="Service Level",
            value=f"{state.inventory.service_level:.0f}%" if state.inventory.data_loaded else "—",
            icon="📦",
            subtitle=f"{state.inventory.items_critical} critical items" if state.inventory.data_loaded else "No data",
            accent_color="green" if state.inventory.service_level >= 95 else "red"
        )

    with cols[3]:
        render_premium_card(
            title="Monthly Savings",
            value=f"${state.total_monthly_savings:,.0f}" if state.total_monthly_savings > 0 else "$0",
            icon="💰",
            subtitle="From optimization",
            accent_color="green",
            delta=f"{(state.total_monthly_savings / max(1, state.staff.monthly_labor_cost + state.inventory.weekly_inventory_cost * 4.33) * 100):.1f}%" if state.total_monthly_savings > 0 else None,
            delta_positive=True
        )

    render_glass_divider()

    # Financial Impact Section
    if state.total_monthly_savings > 0:
        render_section_header("Financial Impact", "💰")

        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Weekly Savings</div>
                <div style="font-size: 2rem; font-weight: 800; color: #22c55e;">${state.total_weekly_savings:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Monthly Savings</div>
                <div style="font-size: 2rem; font-weight: 800; color: #22c55e;">${state.total_monthly_savings:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Annual Projection</div>
                <div style="font-size: 2rem; font-weight: 800; color: #22c55e;">${state.annual_savings:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        render_glass_divider()

    # Urgent Actions Summary
    urgent_actions = [a for a in state.actions if a.priority == "urgent"]
    important_actions = [a for a in state.actions if a.priority == "important"]

    if urgent_actions or important_actions:
        render_section_header(f"Attention Required ({len(urgent_actions)} Urgent, {len(important_actions)} Important)", "🚨")

        cols = st.columns(2)
        for i, action in enumerate((urgent_actions + important_actions)[:4]):
            with cols[i % 2]:
                render_action_card(action)
    else:
        st.success("✅ All systems operating normally. No urgent actions required.")

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: FORECAST
# =============================================================================
with tab_forecast:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if state.forecast.has_forecast:
        # Model info bar
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Active Model:</span>
                    <span style="color: #f1f5f9; font-weight: 700; margin-left: 0.5rem;">{state.forecast.source}</span>
                    <span class="status-badge status-good" style="margin-left: 1rem;">
                        {state.forecast.model_type}
                    </span>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Accuracy:</span>
                    <span style="color: #22c55e; font-weight: 700; margin-left: 0.5rem;">{state.forecast.avg_accuracy:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 7-Day Forecast Cards
        render_section_header("7-Day Patient Forecast", "📅")
        render_forecast_day_cards(state.forecast)

        render_glass_divider()

        # Forecast Chart
        render_section_header("Forecast Visualization", "📈")

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dates = state.forecast.dates if state.forecast.dates else day_names[:len(state.forecast.forecasts)]

        fig = create_forecast_area_chart(
            dates=dates,
            forecasts=state.forecast.forecasts,
            ci_lower=state.forecast.lower_bounds if state.forecast.lower_bounds else None,
            ci_upper=state.forecast.upper_bounds if state.forecast.upper_bounds else None,
            peak_index=state.forecast.peak_day,
            title=""
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Summary metrics
        render_glass_divider()
        render_section_header("Forecast Summary", "📊")

        cols = st.columns(4)
        with cols[0]:
            render_premium_card("Total Week", f"{int(state.forecast.total_week):,}", "📊", accent_color="blue")
        with cols[1]:
            render_premium_card("Daily Average", f"{int(state.forecast.avg_daily)}", "📈", accent_color="purple")
        with cols[2]:
            render_premium_card("Peak Day", f"{int(state.forecast.peak_patients)}", "⬆️", accent_color="orange")
        with cols[3]:
            render_premium_card("Trend", state.forecast.trend.capitalize(), "📉" if state.forecast.trend == "down" else "📈", accent_color="green" if state.forecast.trend == "up" else "blue")

    else:
        render_empty_state(
            icon="🔮",
            title="No Forecast Available",
            description="Train a model in the Train Models page to generate patient forecasts. The forecast will automatically sync here.",
            action_label="Go to Train Models",
            action_key="goto_train_models"
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: STAFF
# =============================================================================
with tab_staff:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if state.staff.data_loaded:
        # Current vs Optimized comparison
        render_section_header("Staff Overview", "👥")

        cols = st.columns(4)
        with cols[0]:
            render_premium_card("Doctors", f"{state.staff.avg_doctors:.0f}", "👨‍⚕️", accent_color="blue")
        with cols[1]:
            render_premium_card("Nurses", f"{state.staff.avg_nurses:.0f}", "👩‍⚕️", accent_color="purple")
        with cols[2]:
            render_premium_card("Support", f"{state.staff.avg_support:.0f}", "👷", accent_color="green")
        with cols[3]:
            render_premium_card("Utilization", f"{state.staff.avg_utilization:.0f}%", "📊", accent_color="orange" if state.staff.avg_utilization < 80 else "green")

        render_glass_divider()

        # Utilization Gauge
        col_gauge, col_costs = st.columns(2)

        with col_gauge:
            render_section_header("Utilization Rate", "📊")
            fig = create_utilization_gauge(
                value=state.staff.avg_utilization,
                title="Staff Utilization",
                target=85
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_costs:
            render_section_header("Labor Costs", "💰")
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Daily</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">${state.staff.daily_labor_cost:,.0f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Weekly</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">${state.staff.weekly_labor_cost:,.0f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Monthly</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">${state.staff.monthly_labor_cost:,.0f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Savings</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #22c55e;">${state.staff.weekly_savings:,.0f}/wk</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Optimization status
        if state.staff.optimization_done:
            render_glass_divider()
            render_section_header("Optimization Results", "🚀")
            render_comparison_panel(
                before_title="Current Weekly Cost",
                before_value=f"${state.staff.weekly_labor_cost:,.0f}",
                after_title="Optimized Weekly Cost",
                after_value=f"${state.staff.opt_weekly_cost:,.0f}",
                savings_value=f"${state.staff.weekly_savings:,.0f}",
                savings_label="Weekly Savings"
            )
        else:
            render_glass_divider()
            st.info("💡 Run Staff Optimization in the Staff Planner page to see potential savings.")

        # Recommendations
        if state.staff.recommendations:
            render_glass_divider()
            render_section_header("Recommendations", "💡")
            for rec in state.staff.recommendations[:3]:
                st.markdown(f"""
                <div class="action-card action-normal">
                    <div class="action-description">{rec}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        if render_empty_state(
            icon="👥",
            title="No Staff Data Loaded",
            description="Load staff data from the database to view utilization metrics and optimization opportunities.",
            action_label="🔄 Load Staff Data",
            action_key="load_staff_data"
        ):
            # Trigger data load
            st.info("Navigate to Staff Planner page (Tab 1) to load staff data from Supabase.")

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: SUPPLY
# =============================================================================
with tab_supply:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if state.inventory.data_loaded:
        # Inventory Overview
        render_section_header("Inventory Overview", "📦")

        cols = st.columns(4)
        with cols[0]:
            status_color = "green" if state.inventory.avg_stockout_risk < 5 else ("orange" if state.inventory.avg_stockout_risk < 15 else "red")
            render_premium_card("Stockout Risk", f"{state.inventory.avg_stockout_risk:.1f}%", "⚠️", accent_color=status_color)
        with cols[1]:
            render_premium_card("Service Level", f"{state.inventory.service_level:.1f}%", "✅", accent_color="green" if state.inventory.service_level >= 95 else "orange")
        with cols[2]:
            render_premium_card("Items OK", f"{state.inventory.items_ok}", "🟢", accent_color="green")
        with cols[3]:
            critical_color = "red" if state.inventory.items_critical > 0 else "green"
            render_premium_card("Items Critical", f"{state.inventory.items_critical}", "🔴", accent_color=critical_color)

        render_glass_divider()

        # Service Level Gauge
        col_gauge, col_costs = st.columns(2)

        with col_gauge:
            render_section_header("Service Level", "📊")
            fig = create_utilization_gauge(
                value=state.inventory.service_level,
                title="Service Level",
                target=98,
                max_value=100
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_costs:
            render_section_header("Inventory Costs", "💰")
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Daily</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">${state.inventory.daily_inventory_cost:,.0f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Weekly</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">${state.inventory.weekly_inventory_cost:,.0f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Restock Events</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{state.inventory.restock_events}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Savings</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #22c55e;">${state.inventory.weekly_savings:,.0f}/wk</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Optimization status
        if state.inventory.optimization_done:
            render_glass_divider()
            render_section_header("Optimization Results", "🚀")
            render_comparison_panel(
                before_title="Current Weekly Cost",
                before_value=f"${state.inventory.weekly_inventory_cost:,.0f}",
                after_title="Optimized Weekly Cost",
                after_value=f"${state.inventory.opt_weekly_cost:,.0f}",
                savings_value=f"${state.inventory.weekly_savings:,.0f}",
                savings_label="Weekly Savings"
            )
        else:
            render_glass_divider()
            st.info("💡 Run Supply Optimization in the Supply Planner page to see potential savings.")

    else:
        if render_empty_state(
            icon="📦",
            title="No Inventory Data Loaded",
            description="Load inventory data from the database to view stock levels and optimization opportunities.",
            action_label="🔄 Load Inventory Data",
            action_key="load_inventory_data"
        ):
            st.info("Navigate to Supply Planner page (Tab 1) to load inventory data from Supabase.")

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5: ACTIONS
# =============================================================================
with tab_actions:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Action counts
    urgent_count = len([a for a in state.actions if a.priority == "urgent"])
    important_count = len([a for a in state.actions if a.priority == "important"])
    normal_count = len([a for a in state.actions if a.priority == "normal"])

    # Filter buttons
    render_section_header(f"Action Items ({len(state.actions)} Total)", "✅")

    st.markdown(f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
        <div class="status-badge status-danger">
            <span>🔴</span>
            <span>Urgent: {urgent_count}</span>
        </div>
        <div class="status-badge status-warning">
            <span>🟡</span>
            <span>Important: {important_count}</span>
        </div>
        <div class="status-badge" style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #3b82f6;">
            <span>🔵</span>
            <span>Normal: {normal_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action list
    if state.actions:
        # Two-column layout for actions
        cols = st.columns(2)

        for i, action in enumerate(state.actions):
            with cols[i % 2]:
                render_action_card(action, show_category=True)
    else:
        render_empty_state(
            icon="✅",
            title="No Actions Required",
            description="All systems are operating normally. Complete the data workflow to generate actionable insights.",
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - Data Status
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### Data Status")

    # Forecast status
    if state.forecast.has_forecast:
        st.success(f"🔮 Forecast: {state.forecast.source}")
    else:
        st.warning("🔮 Forecast: Not Available")

    # Staff status
    if state.staff.data_loaded:
        st.success(f"👥 Staff: Loaded ({state.staff.avg_utilization:.0f}% util)")
    else:
        st.warning("👥 Staff: Not Loaded")

    # Inventory status
    if state.inventory.data_loaded:
        st.success(f"📦 Inventory: Loaded ({state.inventory.service_level:.0f}% SL)")
    else:
        st.warning("📦 Inventory: Not Loaded")

    # Last refresh
    if state.last_refresh:
        st.caption(f"Last refresh: {state.last_refresh.strftime('%H:%M:%S')}")


# =============================================================================
# HORIZON CONTROL - MISSION NAVIGATION
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="mission-nav-container">
    <div class="mission-title">▸ MISSION NAVIGATION ◂</div>
</div>
""", unsafe_allow_html=True)

# Navigation grid - 3 columns of page links
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="mission-category">📥 Data Acquisition</div>', unsafe_allow_html=True)
    if st.button("📁 Upload Data", key="nav_upload", use_container_width=True):
        st.switch_page("pages/02_Upload_Data.py")
    if st.button("🔧 Prepare Data", key="nav_prepare", use_container_width=True):
        st.switch_page("pages/03_Prepare_Data.py")
    if st.button("🔍 Explore Data", key="nav_explore", use_container_width=True):
        st.switch_page("pages/04_Explore_Data.py")

with col2:
    st.markdown('<div class="mission-category">🧠 Model Training</div>', unsafe_allow_html=True)
    if st.button("📈 Baseline Models", key="nav_baseline", use_container_width=True):
        st.switch_page("pages/05_Baseline_Models.py")
    if st.button("🛠️ Feature Studio", key="nav_features", use_container_width=True):
        st.switch_page("pages/06_Feature_Studio.py")
    if st.button("🎯 Feature Selection", key="nav_selection", use_container_width=True):
        st.switch_page("pages/07_Feature_Selection.py")

with col3:
    st.markdown('<div class="mission-category">🚀 Results & Analysis</div>', unsafe_allow_html=True)
    if st.button("🧠 Train Models", key="nav_train", use_container_width=True):
        st.switch_page("pages/08_Train_Models.py")
    if st.button("📊 Model Results", key="nav_results", use_container_width=True):
        st.switch_page("pages/09_Model_Results.py")
    if st.button("🏠 Home Base", key="nav_home", use_container_width=True):
        st.switch_page("Welcome.py")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem;">
    <div class="cyber-text" style="font-size: 0.7rem; opacity: 0.6;">
        HORIZON CONTROL v2.0 · PREDICTIVE INTELLIGENCE SYSTEM
    </div>
</div>
""", unsafe_allow_html=True)
