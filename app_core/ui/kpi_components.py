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

# Color palette matching Welcome.py fluorescent design
COLORS = {
    "bg": "#0a1628",
    "card": "rgba(15, 23, 42, 0.9)",  # Matching Welcome.py gradient mid
    "card_hover": "#162440",
    "border": "rgba(6, 78, 145, 0.4)",  # Matching Welcome.py border
    "accent": "#22d3ee",  # Cyan from Welcome.py
    "accent_dim": "rgba(34, 211, 238, 0.15)",
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
    """Inject CSS styles for KPI dashboard components with fluorescent effects."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* ========================================
       FLUORESCENT ANIMATIONS (from Welcome.py)
       ======================================== */

    @keyframes card-glow {{
        0%, 100% {{
            box-shadow:
                0 0 20px rgba(6, 78, 145, 0.3),
                0 0 40px rgba(6, 78, 145, 0.2),
                0 0 60px rgba(34, 211, 238, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }}
        50% {{
            box-shadow:
                0 0 30px rgba(6, 78, 145, 0.4),
                0 0 60px rgba(6, 78, 145, 0.3),
                0 0 80px rgba(34, 211, 238, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }}
    }}

    @keyframes text-glow-cyan {{
        0%, 100% {{
            text-shadow: 0 0 10px rgba(34, 211, 238, 0.5), 0 0 20px rgba(34, 211, 238, 0.3);
        }}
        50% {{
            text-shadow: 0 0 20px rgba(34, 211, 238, 0.7), 0 0 40px rgba(34, 211, 238, 0.4);
        }}
    }}

    @keyframes shimmer-sweep {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}

    @keyframes pulse-glow {{
        0%, 100% {{ opacity: 1; box-shadow: 0 0 10px currentColor; }}
        50% {{ opacity: 0.7; box-shadow: 0 0 20px currentColor; }}
    }}

    /* ========================================
       FLUORESCENT KPI CARD
       ======================================== */

    .fluorescent-kpi-card {{
        background: linear-gradient(145deg, rgba(6, 78, 145, 0.15) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(6, 78, 145, 0.1) 100%);
        border: 1px solid rgba(6, 78, 145, 0.4);
        border-radius: 20px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        animation: card-glow 3s ease-in-out infinite;
        transition: all 0.3s ease;
    }}

    .fluorescent-kpi-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(34, 211, 238, 0.5);
    }}

    .fluorescent-kpi-card .kpi-shimmer {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.1), transparent);
        animation: shimmer-sweep 4s infinite;
        pointer-events: none;
    }}

    .fluorescent-kpi-card .kpi-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }}

    .fluorescent-kpi-card .kpi-icon {{
        font-size: 1.25rem;
    }}

    .fluorescent-kpi-card .kpi-label {{
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .fluorescent-kpi-card .kpi-value {{
        font-size: 2.25rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        animation: text-glow-cyan 3s ease-in-out infinite;
        margin: 0.5rem 0;
    }}

    .fluorescent-kpi-card .kpi-unit {{
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }}

    .fluorescent-kpi-card .kpi-trend {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }}

    .fluorescent-kpi-card .trend-up {{
        background: rgba(34, 211, 238, 0.15);
        color: #22d3ee;
    }}

    .fluorescent-kpi-card .trend-down {{
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
    }}

    /* ========================================
       FLUORESCENT GAUGE CARD
       ======================================== */

    .fluorescent-gauge-card {{
        background: linear-gradient(145deg, rgba(6, 78, 145, 0.15) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(6, 78, 145, 0.1) 100%);
        border: 1px solid rgba(6, 78, 145, 0.4);
        border-radius: 20px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        animation: card-glow 3s ease-in-out infinite;
        transition: all 0.3s ease;
    }}

    .fluorescent-gauge-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(34, 211, 238, 0.5);
    }}

    .fluorescent-gauge-card .kpi-shimmer {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.1), transparent);
        animation: shimmer-sweep 4s infinite;
        pointer-events: none;
    }}

    .fluorescent-gauge-card .gauge-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }}

    .fluorescent-gauge-card .gauge-icon {{
        font-size: 1.25rem;
    }}

    .fluorescent-gauge-card .gauge-label {{
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .fluorescent-gauge-card .gauge-value {{
        font-size: 2.25rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        animation: text-glow-cyan 3s ease-in-out infinite;
        margin: 0.5rem 0;
    }}

    .fluorescent-gauge-card .gauge-bar-container {{
        margin-top: 1rem;
        height: 8px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 4px;
        overflow: hidden;
    }}

    .fluorescent-gauge-card .gauge-bar-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
        box-shadow: 0 0 10px currentColor;
    }}

    /* ========================================
       FLUORESCENT CHART CONTAINER
       ======================================== */

    .fluorescent-chart-container {{
        background: linear-gradient(145deg, rgba(6, 78, 145, 0.12) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(6, 78, 145, 0.08) 100%);
        border: 1px solid rgba(6, 78, 145, 0.35);
        border-radius: 20px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        animation: card-glow 4s ease-in-out infinite;
    }}

    .fluorescent-chart-container .chart-shimmer {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.08), transparent);
        animation: shimmer-sweep 5s infinite;
        pointer-events: none;
    }}

    .fluorescent-chart-container .chart-title {{
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        animation: text-glow-cyan 4s ease-in-out infinite;
    }}

    /* ========================================
       FLUORESCENT DASHBOARD HEADER
       ======================================== */

    .fluorescent-dashboard-header {{
        background: linear-gradient(145deg, rgba(6, 78, 145, 0.15) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(6, 78, 145, 0.1) 100%);
        border: 1px solid rgba(6, 78, 145, 0.4);
        border-radius: 24px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
        animation: card-glow 3s ease-in-out infinite;
        margin-bottom: 2rem;
    }}

    .fluorescent-dashboard-header .header-shimmer {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.1), transparent);
        animation: shimmer-sweep 4s infinite;
        pointer-events: none;
    }}

    .fluorescent-dashboard-header .header-content {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }}

    .fluorescent-dashboard-header .header-brand {{
        display: flex;
        align-items: center;
        gap: 1rem;
    }}

    .fluorescent-dashboard-header .header-logo {{
        width: 50px;
        height: 50px;
        border-radius: 14px;
        background: linear-gradient(135deg, #064e91 0%, #22d3ee 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 800;
        color: #fff;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.4);
    }}

    .fluorescent-dashboard-header .header-title {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        animation: text-glow-cyan 3s ease-in-out infinite;
        letter-spacing: -0.5px;
    }}

    .fluorescent-dashboard-header .header-subtitle {{
        margin: 0.25rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }}

    .fluorescent-dashboard-header .live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(6, 78, 145, 0.3);
        border: 1px solid rgba(34, 211, 238, 0.3);
        border-radius: 20px;
        padding: 8px 16px;
    }}

    .fluorescent-dashboard-header .live-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22d3ee;
        animation: pulse-glow 2s infinite;
        box-shadow: 0 0 10px #22d3ee;
    }}

    .fluorescent-dashboard-header .live-text {{
        color: #22d3ee;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    /* ========================================
       FLUORESCENT DEPARTMENT LOAD BAR
       ======================================== */

    .fluorescent-dept-item {{
        margin-bottom: 1rem;
    }}

    .fluorescent-dept-item .dept-header {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }}

    .fluorescent-dept-item .dept-name {{
        font-size: 0.875rem;
        font-weight: 500;
        color: #e2e8f0;
    }}

    .fluorescent-dept-item .dept-value {{
        font-size: 0.875rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}

    .fluorescent-dept-item .dept-bar {{
        height: 10px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 5px;
        overflow: hidden;
    }}

    .fluorescent-dept-item .dept-fill {{
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease;
        box-shadow: 0 0 8px currentColor;
    }}

    /* ========================================
       LEGACY SUPPORT (keep for compatibility)
       ======================================== */

    /* Style native Streamlit metrics */
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(6, 78, 145, 0.15) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(6, 78, 145, 0.1) 100%);
        border: 1px solid rgba(6, 78, 145, 0.4);
        border-radius: 20px;
        padding: 16px 20px;
        animation: card-glow 3s ease-in-out infinite;
    }}
    [data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #22d3ee !important;
        font-family: 'JetBrains Mono', monospace !important;
        animation: text-glow-cyan 3s ease-in-out infinite !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 14px !important;
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
        color: #94a3b8;
    }}
    .legend-value {{
        margin-left: auto;
        font-weight: 600;
    }}

    /* ========================================
       MOBILE RESPONSIVE STYLES
       ======================================== */

    /* Tablet (768px and below) */
    @media (max-width: 768px) {{
        /* KPI Cards */
        .fluorescent-kpi-card,
        .fluorescent-gauge-card {{
            padding: 1rem;
            border-radius: 16px;
        }}

        .fluorescent-kpi-card .kpi-value,
        .fluorescent-gauge-card .gauge-value {{
            font-size: 1.75rem;
        }}

        .fluorescent-kpi-card .kpi-label,
        .fluorescent-gauge-card .gauge-label {{
            font-size: 0.7rem;
        }}

        .fluorescent-kpi-card .kpi-icon,
        .fluorescent-gauge-card .gauge-icon {{
            font-size: 1.1rem;
        }}

        /* Dashboard Header */
        .fluorescent-dashboard-header {{
            padding: 1.25rem;
            border-radius: 16px;
        }}

        .fluorescent-dashboard-header .header-title {{
            font-size: 1.25rem;
        }}

        .fluorescent-dashboard-header .header-subtitle {{
            font-size: 0.85rem;
        }}

        .fluorescent-dashboard-header .header-logo {{
            width: 40px;
            height: 40px;
            font-size: 1.2rem;
        }}

        .fluorescent-dashboard-header .header-content {{
            flex-direction: column;
            align-items: flex-start;
        }}

        .fluorescent-dashboard-header .live-badge {{
            padding: 6px 12px;
            font-size: 0.7rem;
        }}

        /* Chart Container */
        .fluorescent-chart-container {{
            padding: 1rem;
            border-radius: 16px;
        }}

        .fluorescent-chart-container .chart-title {{
            font-size: 0.9rem;
        }}

        /* Native Streamlit Metrics */
        [data-testid="stMetric"] {{
            padding: 12px 14px;
            border-radius: 16px;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.5rem !important;
        }}
    }}

    /* Mobile (480px and below) */
    @media (max-width: 480px) {{
        /* KPI Cards - Compact mode */
        .fluorescent-kpi-card,
        .fluorescent-gauge-card {{
            padding: 0.875rem;
            border-radius: 14px;
        }}

        .fluorescent-kpi-card .kpi-value,
        .fluorescent-gauge-card .gauge-value {{
            font-size: 1.5rem;
        }}

        .fluorescent-kpi-card .kpi-label,
        .fluorescent-gauge-card .gauge-label {{
            font-size: 0.65rem;
            letter-spacing: 0.5px;
        }}

        .fluorescent-kpi-card .kpi-unit {{
            font-size: 0.75rem;
        }}

        .fluorescent-kpi-card .kpi-trend {{
            padding: 3px 8px;
            font-size: 0.7rem;
        }}

        /* Dashboard Header - Minimal */
        .fluorescent-dashboard-header {{
            padding: 1rem;
            margin-bottom: 1rem;
        }}

        .fluorescent-dashboard-header .header-title {{
            font-size: 1.1rem;
        }}

        .fluorescent-dashboard-header .header-subtitle {{
            font-size: 0.75rem;
        }}

        .fluorescent-dashboard-header .header-logo {{
            width: 36px;
            height: 36px;
            font-size: 1rem;
            border-radius: 10px;
        }}

        .fluorescent-dashboard-header .header-brand {{
            gap: 0.75rem;
        }}

        .fluorescent-dashboard-header .live-badge {{
            padding: 4px 10px;
        }}

        .fluorescent-dashboard-header .live-dot {{
            width: 8px;
            height: 8px;
        }}

        .fluorescent-dashboard-header .live-text {{
            font-size: 0.65rem;
        }}

        /* Chart Container - Compact */
        .fluorescent-chart-container {{
            padding: 0.75rem;
            border-radius: 14px;
        }}

        /* Department bars */
        .fluorescent-dept-item .dept-name,
        .fluorescent-dept-item .dept-value {{
            font-size: 0.8rem;
        }}

        /* Native Streamlit Metrics - Compact */
        [data-testid="stMetric"] {{
            padding: 10px 12px;
            border-radius: 14px;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.25rem !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: 0.75rem !important;
        }}
    }}

    /* Very Small Mobile (360px and below) */
    @media (max-width: 360px) {{
        .fluorescent-kpi-card .kpi-value,
        .fluorescent-gauge-card .gauge-value {{
            font-size: 1.25rem;
        }}

        .fluorescent-dashboard-header .header-title {{
            font-size: 1rem;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.1rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

def render_dashboard_header(title: str = "Hospital Scheduling KPI", subtitle: str = "Real-time operational performance metrics"):
    """Render the fluorescent dashboard header with logo and live badge."""
    st.markdown(f"""
    <div class="fluorescent-dashboard-header">
        <div class="header-shimmer"></div>
        <div class="header-content">
            <div class="header-brand">
                <div class="header-logo">H</div>
                <div>
                    <h1 class="header-title">{title}</h1>
                    <p class="header-subtitle">{subtitle}</p>
                </div>
            </div>
            <div class="live-badge">
                <div class="live-dot"></div>
                <span class="live-text">Live Dashboard</span>
            </div>
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
    Render a fluorescent gauge-style KPI card with progress bar.

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
            color = "#ef4444"  # Red
        elif pct > 70:
            color = "#f59e0b"  # Amber
        else:
            color = "#22d3ee"  # Cyan

    # Format value
    if isinstance(value, float):
        value_str = f"{value:.1f}"
    else:
        value_str = str(value)

    st.markdown(f"""
    <div class="fluorescent-gauge-card">
        <div class="kpi-shimmer"></div>
        <div class="gauge-header">
            <span class="gauge-icon">{icon}</span>
            <span class="gauge-label">{label}</span>
        </div>
        <div class="gauge-value" style="color: {color};">{value_str}{unit}</div>
        <div class="gauge-bar-container">
            <div class="gauge-bar-fill" style="width: {min(pct, 100):.1f}%; background: linear-gradient(90deg, {color}, {color}88); color: {color};"></div>
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
    Render a fluorescent stat card with optional trend indicator.

    Args:
        label: KPI label text
        value: Value to display
        unit: Unit suffix
        icon: Emoji icon
        color: Value color (used for styling)
        trend: Trend text (e.g., "12%")
        trend_dir: "up" or "down"
    """
    # Format value
    if isinstance(value, float):
        value_str = f"{value:.1f}"
    else:
        value_str = str(value)

    # Build trend HTML if provided
    trend_html = ""
    if trend and trend_dir:
        trend_class = "trend-up" if trend_dir == "up" else "trend-down"
        trend_arrow = "↑" if trend_dir == "up" else "↓"
        trend_html = f'<div class="kpi-trend {trend_class}">{trend_arrow} {trend}</div>'

    st.markdown(f"""
    <div class="fluorescent-kpi-card">
        <div class="kpi-shimmer"></div>
        <div class="kpi-header">
            <span class="kpi-icon">{icon}</span>
            <span class="kpi-label">{label}</span>
        </div>
        <div class="kpi-value" style="color: {color};">{value_str}</div>
        <div class="kpi-unit">{unit}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_department_load_bars(departments: List[Dict]):
    """
    Render fluorescent horizontal progress bars for department capacity.

    Args:
        departments: List of dicts with 'name', 'load', 'capacity' keys
    """
    for dept in departments:
        load = dept.get('load', 0)
        capacity = dept.get('capacity', 100)
        pct = (load / capacity) * 100 if capacity > 0 else 0

        # Color based on load
        if pct > 85:
            color = "#ef4444"  # Red
        elif pct > 70:
            color = "#f59e0b"  # Amber
        else:
            color = "#22d3ee"  # Cyan

        st.markdown(f"""
        <div class="fluorescent-dept-item">
            <div class="dept-header">
                <span class="dept-name">{dept['name']}</span>
                <span class="dept-value" style="color: {color};">{load}%</span>
            </div>
            <div class="dept-bar">
                <div class="dept-fill" style="width: {pct}%; background: linear-gradient(90deg, {color}, {color}88); color: {color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_fluorescent_chart_title(title: str, icon: str = "📈"):
    """Render a fluorescent styled chart title header."""
    st.markdown(f"""
    <div class="fluorescent-chart-container" style="padding: 1rem 1.5rem; margin-bottom: 0.5rem;">
        <div class="chart-shimmer"></div>
        <h3 class="chart-title" style="margin: 0;">{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)


def render_shift_pie_chart(shift_data: List[Dict], height: int = 200, chart_key: str = None):
    """
    Render a donut chart for shift distribution.

    Args:
        shift_data: List of dicts with 'name', 'value', 'color' keys
        height: Chart height in pixels
        chart_key: Unique key for the Plotly chart
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

    key = chart_key or f"shift_pie_{id(shift_data)}"
    st.plotly_chart(fig, use_container_width=True, key=key)

    # Render legend
    for s in shift_data:
        st.markdown(f"""
        <div class="legend-item">
            <div class="legend-dot" style="background: {s['color']};"></div>
            <span class="legend-label">{s['name']}</span>
            <span class="legend-value" style="color: {s['color']};">{s['value']}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_weekly_schedule_chart(schedule_data: List[Dict], height: int = 280, chart_key: str = None):
    """
    Render a grouped bar chart for weekly schedule.

    Args:
        schedule_data: List of dicts with 'day', 'scheduled', 'actual', 'overtime' keys
        height: Chart height in pixels
        chart_key: Unique key for the Plotly chart
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

    key = chart_key or f"weekly_schedule_{id(schedule_data)}"
    st.plotly_chart(fig, use_container_width=True, key=key)


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
