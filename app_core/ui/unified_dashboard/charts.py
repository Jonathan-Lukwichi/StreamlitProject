# =============================================================================
# app_core/ui/unified_dashboard/charts.py
# Premium Chart Templates for the Unified Dashboard
# =============================================================================
"""
Plotly chart templates optimized for the dark theme Command Center.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any
import numpy as np


# =============================================================================
# COLOR PALETTE
# =============================================================================

CHART_COLORS = {
    "primary": "#3b82f6",       # Blue
    "secondary": "#22d3ee",     # Cyan
    "success": "#22c55e",       # Green
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "purple": "#a855f7",        # Purple
    "pink": "#ec4899",          # Pink
    "text": "#f1f5f9",          # Light text
    "muted": "#94a3b8",         # Muted text
    "grid": "rgba(255,255,255,0.05)",
    "line": "rgba(255,255,255,0.1)",
    "fill_primary": "rgba(59, 130, 246, 0.2)",
    "fill_secondary": "rgba(34, 211, 238, 0.2)",
    "fill_success": "rgba(34, 197, 94, 0.2)",
}

# Clinical category colors
CATEGORY_COLORS = {
    "RESPIRATORY": "#3b82f6",      # Blue
    "CARDIAC": "#ef4444",          # Red
    "TRAUMA": "#f59e0b",           # Amber
    "GASTROINTESTINAL": "#22c55e", # Green
    "INFECTIOUS": "#a855f7",       # Purple
    "NEUROLOGICAL": "#ec4899",     # Pink
    "OTHER": "#6b7280",            # Gray
}


# =============================================================================
# CHART LAYOUT TEMPLATE
# =============================================================================

def get_premium_chart_layout(
    title: str = "",
    height: int = 350,
    show_legend: bool = True,
    margin: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Return standard Plotly layout for premium dark-themed charts.

    Args:
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend
        margin: Optional custom margin dict

    Returns:
        Dict with Plotly layout configuration
    """
    default_margin = {"l": 40, "r": 20, "t": 60 if title else 30, "b": 40}

    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": height,
        "title": {
            "text": title,
            "font": {"size": 16, "color": CHART_COLORS["text"], "family": "Inter, sans-serif"},
            "x": 0,
            "xanchor": "left",
        } if title else None,
        "font": {"family": "Inter, sans-serif", "color": CHART_COLORS["muted"]},
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
        "margin": margin or default_margin,
        "xaxis": {
            "gridcolor": CHART_COLORS["grid"],
            "linecolor": CHART_COLORS["line"],
            "tickfont": {"size": 11},
            "showgrid": True,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": CHART_COLORS["grid"],
            "linecolor": CHART_COLORS["line"],
            "tickfont": {"size": 11},
            "showgrid": True,
            "zeroline": False,
        },
        "hoverlabel": {
            "bgcolor": "rgba(15, 23, 42, 0.95)",
            "bordercolor": CHART_COLORS["primary"],
            "font": {"size": 12, "color": CHART_COLORS["text"]},
        },
        "hovermode": "x unified",
    }


# =============================================================================
# FORECAST AREA CHART
# =============================================================================

def create_forecast_area_chart(
    dates: List[str],
    forecasts: List[float],
    ci_lower: Optional[List[float]] = None,
    ci_upper: Optional[List[float]] = None,
    actuals: Optional[List[float]] = None,
    peak_index: Optional[int] = None,
    title: str = "7-Day Patient Forecast"
) -> go.Figure:
    """
    Create a premium forecast area chart with confidence intervals.

    Args:
        dates: List of date labels
        forecasts: List of forecast values
        ci_lower: Optional lower confidence bound
        ci_upper: Optional upper confidence bound
        actuals: Optional actual values for comparison
        peak_index: Index of peak day to highlight
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Confidence interval band
    if ci_lower and ci_upper and len(ci_lower) == len(ci_upper):
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill="toself",
            fillcolor="rgba(168, 85, 247, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Actuals line (if provided)
    if actuals and len(actuals) > 0:
        fig.add_trace(go.Scatter(
            x=dates[:len(actuals)],
            y=actuals,
            mode="lines+markers",
            line=dict(color=CHART_COLORS["muted"], width=2, dash="dot"),
            marker=dict(size=6, color=CHART_COLORS["muted"]),
            name="Actual",
        ))

    # Forecast line with peak highlight
    marker_colors = [
        CHART_COLORS["warning"] if i == peak_index else CHART_COLORS["primary"]
        for i in range(len(forecasts))
    ]
    marker_sizes = [
        12 if i == peak_index else 8
        for i in range(len(forecasts))
    ]

    fig.add_trace(go.Scatter(
        x=dates,
        y=forecasts,
        mode="lines+markers",
        line=dict(color=CHART_COLORS["primary"], width=3),
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color="white", width=2),
        ),
        name="Forecast",
        fill="tozeroy",
        fillcolor=CHART_COLORS["fill_primary"],
    ))

    # Add average line
    avg_value = np.mean(forecasts)
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color=CHART_COLORS["secondary"],
        annotation_text=f"Avg: {avg_value:.0f}",
        annotation_position="right",
        annotation_font_color=CHART_COLORS["secondary"],
    )

    # Apply premium layout
    layout = get_premium_chart_layout(title=title, height=350)
    layout["yaxis"]["title"] = "Patients"
    fig.update_layout(**layout)

    return fig


# =============================================================================
# UTILIZATION GAUGE
# =============================================================================

def create_utilization_gauge(
    value: float,
    title: str = "Utilization",
    target: float = 85,
    max_value: float = 100
) -> go.Figure:
    """
    Create a utilization gauge chart.

    Args:
        value: Current utilization value (0-100)
        title: Gauge title
        target: Target value for reference
        max_value: Maximum gauge value

    Returns:
        Plotly Figure
    """
    # Determine color based on value
    if 80 <= value <= 95:
        bar_color = CHART_COLORS["success"]
    elif value < 70 or value > 100:
        bar_color = CHART_COLORS["danger"]
    else:
        bar_color = CHART_COLORS["warning"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "suffix": "%",
            "font": {"size": 36, "color": CHART_COLORS["text"]},
        },
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["muted"]},
        },
        gauge={
            "axis": {
                "range": [0, max_value],
                "tickwidth": 1,
                "tickcolor": CHART_COLORS["muted"],
                "tickfont": {"size": 10, "color": CHART_COLORS["muted"]},
            },
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "rgba(100, 116, 139, 0.2)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70], "color": "rgba(239, 68, 68, 0.1)"},
                {"range": [70, 85], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [85, 95], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [95, 100], "color": "rgba(245, 158, 11, 0.1)"},
            ],
            "threshold": {
                "line": {"color": CHART_COLORS["secondary"], "width": 3},
                "thickness": 0.8,
                "value": target,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
        height=250,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )

    return fig


# =============================================================================
# COST COMPARISON BAR
# =============================================================================

def create_cost_comparison_bar(
    before_value: float,
    after_value: float,
    before_label: str = "Before",
    after_label: str = "After",
    title: str = "Cost Comparison",
    value_prefix: str = "$"
) -> go.Figure:
    """
    Create a before/after cost comparison bar chart.

    Args:
        before_value: Value before optimization
        after_value: Value after optimization
        before_label: Label for before bar
        after_label: Label for after bar
        title: Chart title
        value_prefix: Prefix for values (e.g., "$")

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Before bar
    fig.add_trace(go.Bar(
        x=[before_label],
        y=[before_value],
        name=before_label,
        marker_color=CHART_COLORS["danger"],
        text=[f"{value_prefix}{before_value:,.0f}"],
        textposition="outside",
        textfont={"color": CHART_COLORS["danger"], "size": 14, "family": "Inter, sans-serif"},
    ))

    # After bar
    fig.add_trace(go.Bar(
        x=[after_label],
        y=[after_value],
        name=after_label,
        marker_color=CHART_COLORS["success"],
        text=[f"{value_prefix}{after_value:,.0f}"],
        textposition="outside",
        textfont={"color": CHART_COLORS["success"], "size": 14, "family": "Inter, sans-serif"},
    ))

    # Calculate savings
    savings = before_value - after_value
    savings_pct = (savings / before_value * 100) if before_value > 0 else 0

    # Apply layout
    layout = get_premium_chart_layout(title=title, height=300, show_legend=False)
    layout["yaxis"]["title"] = "Cost"

    # Add annotation for savings
    layout["annotations"] = [{
        "x": 0.5,
        "y": 1.1,
        "xref": "paper",
        "yref": "paper",
        "text": f"Savings: {value_prefix}{savings:,.0f} ({savings_pct:.1f}%)",
        "showarrow": False,
        "font": {"size": 14, "color": CHART_COLORS["success"]},
    }]

    fig.update_layout(**layout)

    return fig


# =============================================================================
# CATEGORY BREAKDOWN CHART
# =============================================================================

def create_category_breakdown_chart(
    categories: List[str],
    values: List[float],
    title: str = "Patient Distribution by Category"
) -> go.Figure:
    """
    Create a category breakdown pie/donut chart.

    Args:
        categories: List of category names
        values: List of values per category
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Get colors for categories
    colors = [CATEGORY_COLORS.get(cat.upper(), CHART_COLORS["muted"]) for cat in categories]

    fig = go.Figure(go.Pie(
        labels=categories,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(15, 23, 42, 1)", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=11, color=CHART_COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>%{value:.0f} patients<br>%{percent}<extra></extra>",
    ))

    # Add center annotation
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:.0f}</b><br><span style='font-size:11px'>Total</span>",
        x=0.5,
        y=0.5,
        font=dict(size=20, color=CHART_COLORS["text"]),
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": CHART_COLORS["muted"]},
        height=350,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        title={
            "text": title,
            "font": {"size": 16, "color": CHART_COLORS["text"]},
            "x": 0,
            "xanchor": "left",
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font={"size": 10},
        ),
    )

    return fig


# =============================================================================
# STAFF ALLOCATION CHART
# =============================================================================

def create_staff_allocation_chart(
    dates: List[str],
    doctors: List[float],
    nurses: List[float],
    support: List[float],
    title: str = "Staff Allocation by Day"
) -> go.Figure:
    """
    Create a stacked bar chart for staff allocation.

    Args:
        dates: List of date labels
        doctors: Doctor counts per day
        nurses: Nurse counts per day
        support: Support staff counts per day
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dates,
        y=doctors,
        name="Doctors",
        marker_color=CHART_COLORS["primary"],
    ))

    fig.add_trace(go.Bar(
        x=dates,
        y=nurses,
        name="Nurses",
        marker_color=CHART_COLORS["secondary"],
    ))

    fig.add_trace(go.Bar(
        x=dates,
        y=support,
        name="Support",
        marker_color=CHART_COLORS["purple"],
    ))

    layout = get_premium_chart_layout(title=title, height=350)
    layout["barmode"] = "stack"
    layout["yaxis"]["title"] = "Staff Count"

    fig.update_layout(**layout)

    return fig


# =============================================================================
# INVENTORY STATUS CHART
# =============================================================================

def create_inventory_status_chart(
    items: List[str],
    usage: List[float],
    levels: List[float],
    title: str = "Inventory Usage vs Levels"
) -> go.Figure:
    """
    Create a grouped bar chart for inventory status.

    Args:
        items: List of inventory item names
        usage: Daily usage per item
        levels: Current level per item
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=items,
        y=usage,
        name="Daily Usage",
        marker_color=CHART_COLORS["warning"],
    ))

    fig.add_trace(go.Bar(
        x=items,
        y=levels,
        name="Current Level",
        marker_color=CHART_COLORS["success"],
    ))

    layout = get_premium_chart_layout(title=title, height=300)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = "Units"

    fig.update_layout(**layout)

    return fig


# =============================================================================
# TREND LINE CHART
# =============================================================================

def create_trend_chart(
    dates: List[str],
    values: List[float],
    title: str = "Trend",
    value_label: str = "Value",
    color: str = None,
    show_fill: bool = True
) -> go.Figure:
    """
    Create a simple trend line chart.

    Args:
        dates: List of date labels
        values: List of values
        title: Chart title
        value_label: Label for y-axis
        color: Line color (defaults to primary)
        show_fill: Whether to show area fill

    Returns:
        Plotly Figure
    """
    line_color = color or CHART_COLORS["primary"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines+markers",
        line=dict(color=line_color, width=2),
        marker=dict(size=6, color=line_color),
        fill="tozeroy" if show_fill else None,
        fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)" if show_fill else None,
        name=value_label,
    ))

    layout = get_premium_chart_layout(title=title, height=300, show_legend=False)
    layout["yaxis"]["title"] = value_label

    fig.update_layout(**layout)

    return fig
