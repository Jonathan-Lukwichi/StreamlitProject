"""
KPI Dashboard Components.

Reusable UI components for the Hospital Scheduling KPI Dashboard.
Includes gauge-style KPIs, stat cards, and chart renderers.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import pandas as pd

from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, DANGER_COLOR, TEXT_COLOR, CARD_BG, SUBTLE_TEXT
)
from app_core.ui.components import add_grid

# =============================================================================
# CONSTANTS
# =============================================================================

# Color palette matching the React reference
COLORS = {
    "bg": "#0a1628",
    "card": "#111d33",
    "card_hover": "#162440",
    "border": "#1e3354",
    "accent": "#00d4aa",  # Teal/cyan
    "accent_dim": "rgba(0,212,170,0.15)",
    "warning": "#f59e0b",
    "warning_dim": "rgba(245,158,11,0.15)",
    "danger": "#ef4444",
    "danger_dim": "rgba(239,68,68,0.15)",
    "info": "#3b82f6",
    "info_dim": "rgba(59,130,246,0.15)",
    "purple": "#a78bfa",
    "purple_dim": "rgba(167,139,250,0.15)",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_dim": "#64748b",
}


# =============================================================================
# CSS INJECTION
# =============================================================================

def inject_kpi_styles():
    """Inject CSS styles for KPI dashboard components."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* Gauge KPI Card */
    .gauge-kpi {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    .gauge-kpi:hover {{
        background: {COLORS['card_hover']};
        transform: translateY(-2px);
    }}
    .gauge-kpi .corner-accent {{
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        border-radius: 0 16px 0 80px;
        opacity: 0.5;
    }}
    .gauge-kpi .kpi-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }}
    .gauge-kpi .kpi-icon {{
        font-size: 18px;
    }}
    .gauge-kpi .kpi-label {{
        color: {COLORS['text_muted']};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .gauge-kpi .kpi-value-row {{
        display: flex;
        align-items: baseline;
        gap: 4px;
    }}
    .gauge-kpi .kpi-value {{
        font-size: 36px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .gauge-kpi .kpi-unit {{
        color: {COLORS['text_dim']};
        font-size: 14px;
    }}
    .gauge-kpi .progress-bar {{
        margin-top: 12px;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }}
    .gauge-kpi .progress-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 1s ease;
    }}

    /* Stat Card */
    .stat-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }}
    .stat-card:hover {{
        background: {COLORS['card_hover']};
        transform: translateY(-2px);
    }}
    .stat-card .card-content {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .stat-card .card-main {{}}
    .stat-card .card-label {{
        color: {COLORS['text_muted']};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .stat-card .card-value-row {{
        display: flex;
        align-items: baseline;
        gap: 4px;
    }}
    .stat-card .card-value {{
        font-size: 32px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .stat-card .card-unit {{
        color: {COLORS['text_dim']};
        font-size: 13px;
    }}
    .stat-card .trend-badge {{
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }}
    .stat-card .trend-up {{
        background: {COLORS['accent_dim']};
        color: {COLORS['accent']};
    }}
    .stat-card .trend-down {{
        background: {COLORS['danger_dim']};
        color: {COLORS['danger']};
    }}

    /* Department Load Bar */
    .dept-load-item {{
        margin-bottom: 16px;
    }}
    .dept-load-header {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }}
    .dept-load-name {{
        font-size: 13px;
        font-weight: 500;
        color: {COLORS['text']};
    }}
    .dept-load-value {{
        font-size: 13px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .dept-load-bar {{
        height: 10px;
        background: rgba(255,255,255,0.06);
        border-radius: 5px;
        overflow: hidden;
    }}
    .dept-load-fill {{
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease;
    }}

    /* Chart Container */
    .chart-container {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 20px;
    }}
    .chart-title {{
        margin: 0 0 16px;
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text_muted']};
        letter-spacing: 0.5px;
    }}

    /* Legend Item */
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        margin-top: 6px;
    }}
    .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 3px;
    }}
    .legend-label {{
        color: {COLORS['text_muted']};
    }}
    .legend-value {{
        margin-left: auto;
        font-weight: 600;
    }}

    /* Live Badge */
    .live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {COLORS['card']};
        border-radius: 12px;
        padding: 6px 14px;
        border: 1px solid {COLORS['border']};
    }}
    .live-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {COLORS['accent']};
        animation: pulse 2s infinite;
    }}
    .live-text {{
        color: {COLORS['text_muted']};
        font-size: 12px;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    /* Dashboard Header */
    .dashboard-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 28px;
        flex-wrap: wrap;
        gap: 16px;
    }}
    .dashboard-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .dashboard-logo {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['info']});
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 700;
        color: #fff;
    }}
    .dashboard-title {{
        margin: 0;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: {COLORS['text']};
    }}
    .dashboard-subtitle {{
        margin: 0;
        color: {COLORS['text_muted']};
        font-size: 13px;
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

def render_dashboard_header(title: str = "Hospital Scheduling KPI", subtitle: str = "Real-time operational performance metrics"):
    """Render the dashboard header with logo and live badge."""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-brand">
            <div class="dashboard-logo">H</div>
            <div>
                <h1 class="dashboard-title">{title}</h1>
                <p class="dashboard-subtitle">{subtitle}</p>
            </div>
        </div>
        <div class="live-badge">
            <div class="live-dot"></div>
            <span class="live-text">Live Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_gauge_kpi(
    value: float,
    max_val: float,
    label: str,
    unit: str,
    icon: str,
    color: str = None,
    dim_color: str = None
):
    """
    Render a gauge-style KPI card with progress bar.

    Args:
        value: Current value
        max_val: Maximum value (for percentage calculation)
        label: KPI label text
        unit: Unit suffix (e.g., "%", "min")
        icon: Emoji icon
        color: Main color (auto-calculated based on thresholds if None)
        dim_color: Dim version of color for accent
    """
    pct = (value / max_val) * 100 if max_val > 0 else 0

    # Auto-determine color based on threshold
    if color is None:
        if pct > 85:
            color = COLORS['danger']
            dim_color = COLORS['danger_dim']
        elif pct > 70:
            color = COLORS['warning']
            dim_color = COLORS['warning_dim']
        else:
            color = COLORS['accent']
            dim_color = COLORS['accent_dim']

    if dim_color is None:
        dim_color = f"{color}22"

    st.markdown(f"""
    <div class="gauge-kpi">
        <div class="corner-accent" style="background: {dim_color};"></div>
        <div class="kpi-header">
            <span class="kpi-icon">{icon}</span>
            <span class="kpi-label">{label}</span>
        </div>
        <div class="kpi-value-row">
            <span class="kpi-value" style="color: {color};">{value:.1f}</span>
            <span class="kpi-unit">{unit}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(pct, 100):.1f}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(
    label: str,
    value: Any,
    unit: str,
    icon: str,
    color: str,
    trend: str = None,
    trend_dir: str = None
):
    """
    Render a stat card with optional trend indicator.

    Args:
        label: KPI label text
        value: Value to display
        unit: Unit suffix
        icon: Emoji icon
        color: Value color
        trend: Trend text (e.g., "12%")
        trend_dir: "up" or "down"
    """
    trend_html = ""
    if trend and trend_dir:
        arrow = "↑" if trend_dir == "up" else "↓"
        trend_class = "trend-up" if trend_dir == "up" else "trend-down"
        trend_html = f'<div class="trend-badge {trend_class}">{arrow} {trend}</div>'

    # Format value
    if isinstance(value, float):
        value_str = f"{value:.1f}"
    else:
        value_str = str(value)

    st.markdown(f"""
    <div class="stat-card">
        <div class="card-content">
            <div class="card-main">
                <div class="card-label">{icon} {label}</div>
                <div class="card-value-row">
                    <span class="card-value" style="color: {color};">{value_str}</span>
                    <span class="card-unit">{unit}</span>
                </div>
            </div>
            {trend_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_department_load_bars(departments: List[Dict]):
    """
    Render horizontal progress bars for department capacity.

    Args:
        departments: List of dicts with 'name', 'load', 'capacity' keys
    """
    for dept in departments:
        load = dept.get('load', 0)
        capacity = dept.get('capacity', 100)
        pct = (load / capacity) * 100 if capacity > 0 else 0

        # Color based on load
        if pct > 85:
            color = COLORS['danger']
        elif pct > 70:
            color = COLORS['warning']
        else:
            color = COLORS['accent']

        st.markdown(f"""
        <div class="dept-load-item">
            <div class="dept-load-header">
                <span class="dept-load-name">{dept['name']}</span>
                <span class="dept-load-value" style="color: {color};">{load}%</span>
            </div>
            <div class="dept-load-bar">
                <div class="dept-load-fill" style="width: {pct}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_shift_pie_chart(shift_data: List[Dict], height: int = 200):
    """
    Render a donut chart for shift distribution.

    Args:
        shift_data: List of dicts with 'name', 'value', 'color' keys
        height: Chart height in pixels
    """
    fig = go.Figure(data=[go.Pie(
        labels=[s['name'] for s in shift_data],
        values=[s['value'] for s in shift_data],
        hole=0.65,
        marker=dict(colors=[s['color'] for s in shift_data]),
        textinfo='none',
        hovertemplate='%{label}<br>%{value}%<extra></extra>'
    )])

    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height
    )

    st.plotly_chart(fig, use_container_width=True, key=f"shift_pie_{id(shift_data)}")

    # Render legend
    for s in shift_data:
        st.markdown(f"""
        <div class="legend-item">
            <div class="legend-dot" style="background: {s['color']};"></div>
            <span class="legend-label">{s['name']}</span>
            <span class="legend-value" style="color: {s['color']};">{s['value']}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_weekly_schedule_chart(schedule_data: List[Dict], height: int = 280):
    """
    Render a grouped bar chart for weekly schedule.

    Args:
        schedule_data: List of dicts with 'day', 'scheduled', 'actual', 'overtime' keys
        height: Chart height in pixels
    """
    df = pd.DataFrame(schedule_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Scheduled',
        x=df['day'],
        y=df['scheduled'],
        marker_color=COLORS['info'],
        marker_cornerradius=6
    ))

    fig.add_trace(go.Bar(
        name='Actual',
        x=df['day'],
        y=df['actual'],
        marker_color=COLORS['accent'],
        marker_cornerradius=6
    ))

    fig.add_trace(go.Bar(
        name='Overtime',
        x=df['day'],
        y=df['overtime'],
        marker_color=COLORS['warning'],
        marker_cornerradius=6
    ))

    fig.update_layout(
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=height,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_dim'], size=11),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"weekly_schedule_{id(schedule_data)}")


def render_monthly_trends_chart(
    trends_data: List[Dict],
    metrics: List[Dict],
    height: int = 240,
    chart_key: str = None
):
    """
    Render a multi-line chart for monthly KPI trends.

    Args:
        trends_data: List of dicts with 'month' and metric values
        metrics: List of dicts with 'key', 'name', 'color' for each metric to plot
        height: Chart height in pixels
    """
    df = pd.DataFrame(trends_data)

    fig = go.Figure()

    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df[metric['key']],
            name=metric['name'],
            mode='lines',
            line=dict(color=metric['color'], width=2.5),
            hovertemplate=f"{metric['name']}: %{{y:.1f}}<extra></extra>"
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_dim'], size=11),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        hovermode='x unified'
    )

    key = chart_key or f"monthly_trends_{id(trends_data)}"
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_chart_container(title: str, icon: str = "📈"):
    """Context manager style wrapper for chart containers (returns None, use with st.container)."""
    st.markdown(f"""
    <div class="chart-container">
        <h3 class="chart-title">{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)
