# =============================================================================
# 01_Dashboard.py - HEALTHFORECAST AI KPI DASHBOARD
# Real-time forecast and operational performance metrics
# =============================================================================
"""
HealthForecast AI Dashboard

Features:
- Forecast KPIs (Today's prediction, 7-day total, peak day)
- Model accuracy metrics (MAPE, RMSE, model comparison)
- Clinical category distribution
- Staff planning metrics
- Interactive Plotly charts
- 4 tabs: Overview, Models, Categories, Staff
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

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
    page_title="Dashboard - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.kpi_components import (
    inject_kpi_styles, render_dashboard_header,
    render_gauge_kpi, render_stat_card,
    render_fluorescent_chart_title,
    COLORS
)
from app_core.ui.effects import inject_fluorescent_effects, inject_tab_styling
from app_core.services.kpi_service import calculate_forecast_kpis, ForecastKPIs

# =============================================================================
# APPLY BASE STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()
inject_kpi_styles()
inject_fluorescent_effects()
inject_tab_styling()

# =============================================================================
# CALCULATE KPIs FROM REAL DATA
# =============================================================================
def _get_session_state_hash() -> str:
    """Generate a hash based on presence of key session state data.

    This allows caching to invalidate when relevant data changes.
    """
    keys_to_check = [
        'prepared_data',
        'active_forecast',
        # Model results
        'arima_results', 'sarimax_results', 'arima_mh_results',
        'ml_mh_results_xgboost', 'ml_mh_results_lstm', 'ml_mh_results_ann',
        'ml_mh_results_XGBoost', 'ml_mh_results_LSTM', 'ml_mh_results_ANN',
        'ml_mh_results', 'ml_results',
        'lstm_xgb_results', 'lstm_sarimax_results', 'lstm_ann_results',
        'opt_results_xgboost', 'opt_results_lstm', 'opt_results_ann',
        # Optimization results
        'optimized_results', 'staff_optimization_results', 'optimization_done',
        'inv_optimized_results', 'inv_optimization_done',
        # Category distribution
        'forecast_hub_demand', 'seasonal_proportions_results',
    ]
    # Create hash from which keys exist and have data
    state_sig = []
    for key in keys_to_check:
        val = st.session_state.get(key)
        if val is not None:
            if isinstance(val, pd.DataFrame):
                state_sig.append(f"{key}:{len(val)}")
            elif isinstance(val, dict):
                state_sig.append(f"{key}:{len(val)}")
            else:
                state_sig.append(f"{key}:1")
    return "|".join(state_sig)


# Calculate KPIs directly (no caching - fast calculation from session state)
kpis = calculate_forecast_kpis()

# =============================================================================
# DASHBOARD HEADER
# =============================================================================
render_dashboard_header(
    title="HealthForecast AI Dashboard",
    subtitle="Hospital demand forecasting and resource optimization metrics"
)

# Data status indicators
status_cols = st.columns(5)
with status_cols[0]:
    status = "✅" if kpis.has_historical else "⚪"
    st.caption(f"{status} Historical: {kpis.total_records} days")
with status_cols[1]:
    status = "✅" if kpis.has_forecast else "⚪"
    st.caption(f"{status} Forecast")
with status_cols[2]:
    status = "✅" if kpis.has_models else "⚪"
    st.caption(f"{status} {kpis.models_trained} Models")
with status_cols[3]:
    status = "✅" if kpis.has_staff_plan else "⚪"
    st.caption(f"{status} Staff Plan")
with status_cols[4]:
    status = "✅" if kpis.has_supply_plan else "⚪"
    st.caption(f"{status} Supply Plan")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TAB NAVIGATION
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Model Accuracy",
    "🏥 Clinical Categories",
    "👥 Staff Planning",
    "📦 Supply Planning"
])

# =============================================================================
# OVERVIEW TAB - Forecast Summary
# =============================================================================
with tab1:
    # Top KPI Row - Forecast Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if kpis.has_forecast:
            render_gauge_kpi(
                value=kpis.today_forecast,
                max_val=max(kpis.historical_max_ed, kpis.today_forecast) if kpis.has_historical else 200,
                label="Today's Forecast",
                unit=" patients",
                icon="📈",
                color=COLORS['accent'],
                dim_color=COLORS['accent_dim']
            )
        else:
            render_stat_card(
                label="Today's Forecast",
                value="—",
                unit="Run forecast first",
                icon="📈",
                color=COLORS['text_dim']
            )

    with col2:
        render_stat_card(
            label="7-Day Total",
            value=int(kpis.week_total_forecast) if kpis.has_forecast else "—",
            unit="patients" if kpis.has_forecast else "No forecast",
            icon="📅",
            color=COLORS['info']
        )

    with col3:
        render_stat_card(
            label="Peak Day",
            value=int(kpis.peak_day_forecast) if kpis.has_forecast else "—",
            unit=kpis.peak_day_name if kpis.has_forecast else "N/A",
            icon="🔺",
            color=COLORS['warning']
        )

    with col4:
        render_stat_card(
            label="Historical Avg",
            value=kpis.historical_avg_ed if kpis.has_historical else "—",
            unit="patients/day" if kpis.has_historical else "Upload data",
            icon="📊",
            color=COLORS['purple']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        render_fluorescent_chart_title("Patient Arrivals — Historical & Forecast", "📈")

        if kpis.forecast_trend:
            # Build chart from trend data
            fig = go.Figure()

            # Separate historical and forecast
            historical = [d for d in kpis.forecast_trend if d.get('type') == 'historical']
            forecast = [d for d in kpis.forecast_trend if d.get('type') == 'forecast']

            if historical:
                fig.add_trace(go.Scatter(
                    x=[d['date'] for d in historical],
                    y=[d.get('actual', 0) for d in historical],
                    name='Historical',
                    line=dict(color=COLORS['text_muted'], width=2),
                    mode='lines+markers'
                ))

            if forecast:
                fig.add_trace(go.Scatter(
                    x=[d['date'] for d in forecast],
                    y=[d.get('forecast', 0) for d in forecast],
                    name='Forecast',
                    line=dict(color=COLORS['accent'], width=3),
                    mode='lines+markers',
                    marker=dict(size=8)
                ))

            fig.update_layout(
                height=260,
                margin=dict(l=40, r=20, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_dim'], size=11),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, title='Patient Arrivals'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key="overview_ed_trend")
        else:
            st.info("📊 Upload data and run forecasts to see patient arrival trends")

    with chart_col2:
        render_fluorescent_chart_title("Day-of-Week Pattern", "📅")

        if kpis.daily_ed_pattern:
            fig = go.Figure(data=[go.Bar(
                x=[d['day'] for d in kpis.daily_ed_pattern],
                y=[d['avg_ed'] for d in kpis.daily_ed_pattern],
                marker_color=COLORS['info'],
                marker_cornerradius=6
            )])
            fig.update_layout(
                height=200,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_dim'], size=11),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True, key="overview_dow_pattern")
        else:
            st.info("📅 Historical data needed for patterns")

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Stats Row - Historical Stats
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        render_stat_card(
            label="Historical Max",
            value=int(kpis.historical_max_ed) if kpis.has_historical else "—",
            unit="patients/day",
            icon="📈",
            color=COLORS['danger']
        )

    with s_col2:
        render_stat_card(
            label="Historical Min",
            value=int(kpis.historical_min_ed) if kpis.has_historical else "—",
            unit="patients/day",
            icon="📉",
            color=COLORS['accent']
        )

    with s_col3:
        render_stat_card(
            label="Data Range",
            value=kpis.total_records if kpis.has_historical else "—",
            unit="days",
            icon="📋",
            color=COLORS['info']
        )

    with s_col4:
        render_stat_card(
            label="Best Model",
            value=kpis.best_model_name[:10] if kpis.has_models else "—",
            unit=f"MAPE: {kpis.best_model_mape}%" if kpis.has_models else "Train models",
            icon="🏆",
            color=COLORS['warning']
        )


# =============================================================================
# MODEL ACCURACY TAB
# =============================================================================
with tab2:
    if kpis.has_models:
        # Top KPI Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_stat_card(
                label="Best Model",
                value=kpis.best_model_name[:12],
                unit="top performer",
                icon="🏆",
                color=COLORS['accent']
            )

        with col2:
            render_gauge_kpi(
                value=100 - kpis.best_model_mape,  # Accuracy = 100 - MAPE
                max_val=100,
                label="Best Accuracy",
                unit="%",
                icon="🎯",
                color=COLORS['accent'],
                dim_color=COLORS['accent_dim']
            )

        with col3:
            render_stat_card(
                label="Best MAPE",
                value=kpis.best_model_mape,
                unit="%",
                icon="📊",
                color=COLORS['info']
            )

        with col4:
            render_stat_card(
                label="Models Trained",
                value=kpis.models_trained,
                unit="models",
                icon="🔧",
                color=COLORS['purple']
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Model Comparison Chart
        render_fluorescent_chart_title("Model Performance Comparison (MAPE %)", "📊")

        if kpis.model_comparison:
            fig = go.Figure()

            # Sort by MAPE (lower is better)
            models_sorted = sorted(kpis.model_comparison, key=lambda x: x['mape'])

            # Color gradient - best (green) to worst (red)
            n_models = len(models_sorted)
            colors = [f'rgba(0, 212, 170, {1 - i*0.15})' for i in range(n_models)]

            fig.add_trace(go.Bar(
                x=[m['name'] for m in models_sorted],
                y=[m['mape'] for m in models_sorted],
                marker_color=colors,
                marker_cornerradius=8,
                text=[f"{m['mape']}%" for m in models_sorted],
                textposition='outside',
                textfont=dict(color=COLORS['text'])
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=20, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_dim'], size=11),
                xaxis=dict(showgrid=False, zeroline=False, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                          zeroline=False, title='MAPE %')
            )
            st.plotly_chart(fig, use_container_width=True, key="model_comparison_chart")

        # Model Details Table
        if kpis.model_comparison:
            st.markdown("<br>", unsafe_allow_html=True)
            render_fluorescent_chart_title("Model Metrics Detail", "📋")

            df_models = pd.DataFrame(kpis.model_comparison)
            df_models.columns = ['Model', 'MAPE %', 'RMSE', 'MAE']
            st.dataframe(df_models, use_container_width=True, hide_index=True)

    else:
        st.info("🎯 **No models trained yet**\n\nGo to **Baseline Models** or **Train Models** page to train forecasting models.")


# =============================================================================
# CLINICAL CATEGORIES TAB
# =============================================================================
with tab3:
    render_fluorescent_chart_title("Patient Arrivals by Clinical Category", "🏥")

    if kpis.category_distribution:
        cat_col1, cat_col2 = st.columns([1, 1])

        with cat_col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=[c['name'] for c in kpis.category_distribution],
                values=[c['value'] for c in kpis.category_distribution],
                hole=0.55,
                marker=dict(colors=[c['color'] for c in kpis.category_distribution]),
                textinfo='percent',
                textfont=dict(size=11, color='white'),
                hovertemplate='%{label}<br>%{value:.1f}%<extra></extra>'
            )])

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_dim']),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="category_pie")

        with cat_col2:
            # Progress bars for each category with fluorescent styling
            for cat in kpis.category_distribution:
                st.markdown(f"""
                <div class="fluorescent-dept-item">
                    <div class="dept-header">
                        <span class="dept-name">{cat['name']}</span>
                        <span class="dept-value" style="color: {cat['color']};">{cat['value']}%</span>
                    </div>
                    <div class="dept-bar">
                        <div class="dept-fill" style="width: {cat['value']}%; background: linear-gradient(90deg, {cat['color']}, {cat['color']}88); color: {cat['color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Category Stats
        st.markdown("<br>", unsafe_allow_html=True)
        render_fluorescent_chart_title("Category Summary", "📊")

        cat_cols = st.columns(len(kpis.category_distribution[:4]))
        for i, cat in enumerate(kpis.category_distribution[:4]):
            with cat_cols[i]:
                render_stat_card(
                    label=cat['name'][:12],
                    value=f"{cat['value']}",
                    unit="%",
                    icon="🏥",
                    color=cat['color']
                )
    else:
        st.info("🏥 **Category data not available**\n\nRun seasonal proportions in **Baseline Models** or generate forecasts in **Patient Forecast** page.")


# =============================================================================
# STAFF PLANNING TAB
# =============================================================================
with tab4:
    # Top KPI Row
    staff_col1, staff_col2, staff_col3, staff_col4 = st.columns(4)

    with staff_col1:
        if kpis.has_staff_plan:
            render_gauge_kpi(
                value=kpis.staff_coverage_pct,
                max_val=100,
                label="Staff Coverage",
                unit="%",
                icon="✅",
                color=COLORS['accent'],
                dim_color=COLORS['accent_dim']
            )
        else:
            render_stat_card(
                label="Staff Coverage",
                value="—",
                unit="Run planner",
                icon="✅",
                color=COLORS['text_dim']
            )

    with staff_col2:
        render_stat_card(
            label="Total Staff Needed",
            value=kpis.total_staff_needed if kpis.has_staff_plan else "—",
            unit="per week" if kpis.has_staff_plan else "N/A",
            icon="👥",
            color=COLORS['info']
        )

    with staff_col3:
        render_stat_card(
            label="Overtime Hours",
            value=kpis.overtime_hours if kpis.has_staff_plan else "—",
            unit="hrs/week" if kpis.has_staff_plan else "N/A",
            icon="⏰",
            color=COLORS['warning']
        )

    with staff_col4:
        render_stat_card(
            label="Daily Cost",
            value=f"${kpis.daily_staff_cost:,.0f}" if kpis.has_staff_plan and kpis.daily_staff_cost > 0 else "—",
            unit="avg/day" if kpis.has_staff_plan else "N/A",
            icon="💰",
            color=COLORS['purple']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Staff Schedule Chart
    render_fluorescent_chart_title("Weekly Staff Schedule — Demand vs Allocated", "📅")

    if kpis.staff_schedule:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Demand',
            x=[s['day'] for s in kpis.staff_schedule],
            y=[s['demand'] for s in kpis.staff_schedule],
            marker_color=COLORS['warning'],
            marker_cornerradius=6,
            opacity=0.7
        ))

        fig.add_trace(go.Bar(
            name='Allocated',
            x=[s['day'] for s in kpis.staff_schedule],
            y=[s['staff'] for s in kpis.staff_schedule],
            marker_color=COLORS['accent'],
            marker_cornerradius=6
        ))

        fig.update_layout(
            barmode='group',
            bargap=0.15,
            height=280,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_dim'], size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, title='Staff Count')
        )
        st.plotly_chart(fig, use_container_width=True, key="staff_schedule_chart")
    else:
        st.info("📅 Run **Staff Planner** to generate staffing schedule")

    if not kpis.has_staff_plan:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👥 **No staff plan generated**\n\nGo to **Staff Planner** page to optimize staffing based on your forecasts.")


# =============================================================================
# SUPPLY PLANNING TAB
# =============================================================================
with tab5:
    # Top KPI Row
    supply_col1, supply_col2, supply_col3, supply_col4 = st.columns(4)

    with supply_col1:
        if kpis.has_supply_plan:
            render_gauge_kpi(
                value=kpis.supply_service_level,
                max_val=100,
                label="Service Level",
                unit="%",
                icon="🎯",
                color=COLORS['accent'],
                dim_color=COLORS['accent_dim']
            )
        else:
            render_stat_card(
                label="Service Level",
                value="—",
                unit="Run optimizer",
                icon="🎯",
                color=COLORS['text_dim']
            )

    with supply_col2:
        render_stat_card(
            label="Total Cost",
            value=f"${kpis.supply_total_cost:,.0f}" if kpis.has_supply_plan else "—",
            unit="for horizon" if kpis.has_supply_plan else "N/A",
            icon="💰",
            color=COLORS['info']
        )

    with supply_col3:
        if kpis.has_supply_plan:
            # Calculate demand coverage from supply results
            coverage = 100  # Default
            if "inv_optimized_results" in st.session_state:
                inv_results = st.session_state.inv_optimized_results
                horizon = inv_results.get('horizon', 7)
                total_demand = inv_results.get('total_demand', 0)
                total_inventory = inv_results.get('total_order_qty', 0) + inv_results.get('total_safety_stock', 0)
                if total_demand > 0:
                    coverage = min((total_inventory / total_demand) * 100, 150)

            render_stat_card(
                label="Demand Coverage",
                value=f"{coverage:.0f}%",
                unit="of forecast",
                icon="📊",
                color=COLORS['accent'] if coverage >= 95 else COLORS['warning']
            )
        else:
            render_stat_card(
                label="Demand Coverage",
                value="—",
                unit="N/A",
                icon="📊",
                color=COLORS['text_dim']
            )

    with supply_col4:
        render_stat_card(
            label="Items Optimized",
            value=kpis.supply_items_count if kpis.has_supply_plan else "—",
            unit="inventory items" if kpis.has_supply_plan else "N/A",
            icon="📦",
            color=COLORS['purple']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if kpis.has_supply_plan and kpis.supply_item_breakdown:
        # Item Breakdown Chart
        render_fluorescent_chart_title("Inventory Order Quantities by Item", "📦")

        # Create bar chart for order quantities
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Order Qty',
            x=[item['name'] for item in kpis.supply_item_breakdown],
            y=[item['order_qty'] for item in kpis.supply_item_breakdown],
            marker_color=COLORS['info'],
            marker_cornerradius=6
        ))

        fig.add_trace(go.Bar(
            name='Safety Stock',
            x=[item['name'] for item in kpis.supply_item_breakdown],
            y=[item['safety_stock'] for item in kpis.supply_item_breakdown],
            marker_color=COLORS['warning'],
            marker_cornerradius=6
        ))

        fig.update_layout(
            barmode='group',
            bargap=0.15,
            height=280,
            margin=dict(l=40, r=20, t=20, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_dim'], size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(showgrid=False, zeroline=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, title='Units')
        )
        st.plotly_chart(fig, use_container_width=True, key="supply_order_chart")

        # Item table
        st.markdown("<br>", unsafe_allow_html=True)
        render_fluorescent_chart_title("Item Details", "📋")

        supply_df = pd.DataFrame(kpis.supply_item_breakdown)
        supply_df.columns = ['Item', 'Forecast Demand', 'Order Qty', 'Safety Stock', 'Cost ($)']
        st.dataframe(supply_df, use_container_width=True, hide_index=True)

    else:
        st.info("📦 **No supply plan generated**\n\nGo to **Supply Planner** page to optimize inventory based on your forecasts.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("HealthForecast AI — Hospital Forecasting & Resource Planning Dashboard")
