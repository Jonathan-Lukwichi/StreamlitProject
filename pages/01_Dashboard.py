# =============================================================================
# 01_Dashboard.py - HOSPITAL SCHEDULING KPI DASHBOARD
# Real-time operational performance metrics
# =============================================================================
"""
Hospital Scheduling KPI Dashboard

Features:
- Gauge-style KPI cards (Bed Occupancy, Staff Utilization)
- Stat cards with trend indicators
- Interactive Plotly charts (Monthly trends, Shift distribution, etc.)
- 4 tabs: Overview, Scheduling, Capacity, Staffing
- Dark theme with glassmorphic cards
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="KPI Dashboard - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import (
    apply_css, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, DANGER_COLOR, TEXT_COLOR, CARD_BG
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.kpi_components import (
    inject_kpi_styles, render_dashboard_header,
    render_gauge_kpi, render_stat_card,
    render_department_load_bars, render_shift_pie_chart,
    render_weekly_schedule_chart, render_monthly_trends_chart,
    COLORS
)
from app_core.services.kpi_service import calculate_hospital_kpis, HospitalKPIs

# =============================================================================
# APPLY BASE STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()
inject_kpi_styles()

# =============================================================================
# CALCULATE KPIs
# =============================================================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_kpis() -> HospitalKPIs:
    """Get or calculate hospital KPIs."""
    return calculate_hospital_kpis()

kpis = get_kpis()

# =============================================================================
# DASHBOARD HEADER
# =============================================================================
render_dashboard_header(
    title="Hospital Scheduling KPI",
    subtitle="Real-time operational performance metrics"
)

# =============================================================================
# TAB NAVIGATION
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📅 Scheduling",
    "🏥 Capacity",
    "👥 Staffing"
])

# =============================================================================
# OVERVIEW TAB
# =============================================================================
with tab1:
    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_gauge_kpi(
            value=kpis.bed_occupancy,
            max_val=100,
            label="Bed Occupancy",
            unit="%",
            icon="🛏️",
            color=COLORS['accent'],
            dim_color=COLORS['accent_dim']
        )

    with col2:
        render_gauge_kpi(
            value=kpis.staff_utilization,
            max_val=100,
            label="Staff Utilization",
            unit="%",
            icon="👤",
            color=COLORS['info'],
            dim_color=COLORS['info_dim']
        )

    with col3:
        render_stat_card(
            label="Avg Wait Time",
            value=kpis.avg_wait_time,
            unit="min",
            icon="⏱️",
            color=COLORS['warning'],
            trend="12%",
            trend_dir="down"
        )

    with col4:
        render_stat_card(
            label="Patient Satisfaction",
            value=kpis.patient_satisfaction,
            unit="/5.0",
            icon="⭐",
            color=COLORS['accent'],
            trend="0.3",
            trend_dir="up"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
            <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']}; letter-spacing: 0.5px;">
                📈 Monthly Trends — Occupancy & Wait Time
            </h3>
        </div>
        """, unsafe_allow_html=True)

        render_monthly_trends_chart(
            trends_data=kpis.monthly_trends,
            metrics=[
                {"key": "occupancy", "name": "Occupancy %", "color": COLORS['accent']},
                {"key": "waitTime", "name": "Wait Time (min)", "color": COLORS['warning']}
            ],
            height=240,
            chart_key="overview_monthly_trends"
        )

    with chart_col2:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
            <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']}; letter-spacing: 0.5px;">
                🕐 Shift Distribution
            </h3>
        </div>
        """, unsafe_allow_html=True)

        render_shift_pie_chart(kpis.shift_distribution, height=200, chart_key="overview_shift_pie")

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Stats Row
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        render_stat_card(
            label="Scheduled Surgeries",
            value=kpis.scheduled_surgeries,
            unit="this month",
            icon="🔬",
            color=COLORS['info']
        )

    with s_col2:
        render_stat_card(
            label="Cancelled Surgeries",
            value=kpis.cancelled_surgeries,
            unit="this month",
            icon="❌",
            color=COLORS['danger'],
            trend="3",
            trend_dir="down"
        )

    with s_col3:
        render_stat_card(
            label="Avg Length of Stay",
            value=kpis.avg_length_of_stay,
            unit="days",
            icon="📋",
            color=COLORS['accent'],
            trend="0.2",
            trend_dir="down"
        )

    with s_col4:
        render_stat_card(
            label="Readmission Rate",
            value=kpis.readmission_rate,
            unit="%",
            icon="🔄",
            color=COLORS['purple'],
            trend="1.1%",
            trend_dir="down"
        )


# =============================================================================
# SCHEDULING TAB
# =============================================================================
with tab2:
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_stat_card(
            label="No-Show Rate",
            value=kpis.no_show_rate,
            unit="%",
            icon="🚫",
            color=COLORS['warning'],
            trend="2.1%",
            trend_dir="down"
        )

    with col2:
        render_stat_card(
            label="ER Throughput",
            value=kpis.er_throughput,
            unit="patients/week",
            icon="🚑",
            color=COLORS['accent'],
            trend="8%",
            trend_dir="up"
        )

    with col3:
        render_stat_card(
            label="Avg Wait Time",
            value=kpis.avg_wait_time,
            unit="minutes",
            icon="⏳",
            color=COLORS['info']
        )

    with col4:
        render_stat_card(
            label="Surgery Cancellations",
            value=kpis.cancelled_surgeries,
            unit="this month",
            icon="⚠️",
            color=COLORS['danger']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Weekly Schedule Chart
    st.markdown(f"""
    <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
        <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']};">
            📅 Weekly Schedule — Planned vs Actual Staff
        </h3>
    </div>
    """, unsafe_allow_html=True)

    render_weekly_schedule_chart(kpis.weekly_schedule, height=280)


# =============================================================================
# CAPACITY TAB
# =============================================================================
with tab3:
    # Top Row
    cap_col1, cap_col2 = st.columns(2)

    with cap_col1:
        render_gauge_kpi(
            value=kpis.bed_occupancy,
            max_val=100,
            label="Overall Bed Occupancy",
            unit="%",
            icon="🛏️",
            color=COLORS['accent'],
            dim_color=COLORS['accent_dim']
        )

    with cap_col2:
        render_stat_card(
            label="ER Throughput",
            value=kpis.er_throughput,
            unit="patients/week",
            icon="🚑",
            color=COLORS['info'],
            trend="5%",
            trend_dir="up"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Department Load
    st.markdown(f"""
    <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
        <h3 style="margin: 0 0 20px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']};">
            🏥 Department Load (%)
        </h3>
    """, unsafe_allow_html=True)

    render_department_load_bars(kpis.department_load)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Throughput Trend
    st.markdown(f"""
    <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
        <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']};">
            📈 Monthly Throughput Trend
        </h3>
    </div>
    """, unsafe_allow_html=True)

    render_monthly_trends_chart(
        trends_data=kpis.monthly_trends,
        metrics=[
            {"key": "throughput", "name": "Throughput", "color": COLORS['accent']}
        ],
        height=220,
        chart_key="capacity_throughput_trends"
    )


# =============================================================================
# STAFFING TAB
# =============================================================================
with tab4:
    # Top KPI Row
    staff_col1, staff_col2, staff_col3 = st.columns(3)

    with staff_col1:
        render_gauge_kpi(
            value=kpis.staff_utilization,
            max_val=100,
            label="Staff Utilization",
            unit="%",
            icon="👤",
            color=COLORS['info'],
            dim_color=COLORS['info_dim']
        )

    with staff_col2:
        render_stat_card(
            label="Overtime Hours",
            value=kpis.overtime_hours,
            unit="hrs/month",
            icon="⏰",
            color=COLORS['warning'],
            trend="14%",
            trend_dir="down"
        )

    with staff_col3:
        render_stat_card(
            label="Staff Turnover",
            value=kpis.staff_turnover,
            unit="%",
            icon="🔄",
            color=COLORS['danger'],
            trend="1.8%",
            trend_dir="down"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    staff_chart1, staff_chart2 = st.columns(2)

    with staff_chart1:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
            <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']};">
                📊 Weekly Overtime Distribution
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Create overtime-only chart
        import plotly.graph_objects as go
        overtime_fig = go.Figure()
        overtime_fig.add_trace(go.Bar(
            x=[s['day'] for s in kpis.weekly_schedule],
            y=[s['overtime'] for s in kpis.weekly_schedule],
            name='Overtime Hrs',
            marker_color=COLORS['warning'],
            marker_cornerradius=8
        ))
        overtime_fig.update_layout(
            height=240,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_dim'], size=11),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
        )
        st.plotly_chart(overtime_fig, use_container_width=True, key="overtime_chart")

    with staff_chart2:
        st.markdown(f"""
        <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 20px;">
            <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 600; color: {COLORS['text_muted']};">
                🕐 Shift Coverage
            </h3>
        </div>
        """, unsafe_allow_html=True)

        render_shift_pie_chart(kpis.shift_distribution, height=200)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("HealthForecast AI — Hospital Scheduling KPI Dashboard")
