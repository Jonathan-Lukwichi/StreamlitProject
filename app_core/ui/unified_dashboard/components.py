# =============================================================================
# app_core/ui/unified_dashboard/components.py
# Reusable UI Components for the Unified Dashboard
# =============================================================================
"""
Premium reusable UI components for the Command Center dashboard.
"""
from __future__ import annotations

import streamlit as st
from typing import List, Optional, Callable, Any
from datetime import datetime

# Import state types
import sys
sys.path.append(str(__file__).rsplit("app_core", 1)[0])

from app_core.state.unified_dashboard_state import (
    UnifiedDashboardState,
    ForecastState,
    StaffState,
    InventoryState,
    ActionItem,
)


# =============================================================================
# NORTH STAR HEADER
# =============================================================================

def render_northstar_header(state: UnifiedDashboardState) -> None:
    """
    Render the sticky North Star header with 5 hero KPIs.

    Args:
        state: Unified dashboard state
    """
    # Calculate values
    total_patients = int(state.forecast.total_week) if state.forecast.has_forecast else 0
    avg_daily = int(state.forecast.avg_daily) if state.forecast.has_forecast else 0
    staff_util = f"{state.staff.avg_utilization:.0f}%" if state.staff.data_loaded else "—"
    service_level = f"{state.inventory.service_level:.1f}%" if state.inventory.data_loaded else "—"
    monthly_savings = f"${state.total_monthly_savings:,.0f}" if state.total_monthly_savings > 0 else "$0"

    # Sync status
    sync_items = [
        state.forecast.has_forecast,
        state.staff.data_loaded,
        state.inventory.data_loaded
    ]
    synced_count = sum(sync_items)

    if synced_count == 3:
        sync_class = "sync-complete"
        sync_text = "All Synced"
    elif synced_count > 0:
        sync_class = "sync-partial"
        sync_text = f"{synced_count}/3 Loaded"
    else:
        sync_class = "sync-none"
        sync_text = "No Data"

    # Render header
    header_html = f"""
    <div class="northstar-strip">
        <div class="northstar-content">
            <div class="northstar-kpi-grid">
                <div class="northstar-kpi">
                    <div class="northstar-value">{total_patients:,}</div>
                    <div class="northstar-label">Patients This Week</div>
                </div>
                <div class="northstar-kpi">
                    <div class="northstar-value">{avg_daily}</div>
                    <div class="northstar-label">Avg Daily</div>
                </div>
                <div class="northstar-kpi">
                    <div class="northstar-value">{staff_util}</div>
                    <div class="northstar-label">Staff Utilization</div>
                </div>
                <div class="northstar-kpi">
                    <div class="northstar-value">{service_level}</div>
                    <div class="northstar-label">Service Level</div>
                </div>
                <div class="northstar-kpi">
                    <div class="northstar-value">{monthly_savings}</div>
                    <div class="northstar-label">Monthly Savings</div>
                </div>
            </div>
            <div class="northstar-actions">
                <div class="sync-indicator {sync_class}">
                    <span class="sync-dot"></span>
                    <span>{sync_text}</span>
                </div>
            </div>
        </div>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)


# =============================================================================
# SYNC STATUS INDICATOR
# =============================================================================

def render_sync_status(
    forecast: ForecastState,
    staff: StaffState,
    inventory: InventoryState
) -> None:
    """
    Render live sync status indicator showing data freshness.

    Args:
        forecast: Forecast state
        staff: Staff state
        inventory: Inventory state
    """
    items = [
        ("🔮 Forecast", forecast.has_forecast, forecast.source if forecast.has_forecast else "Not Available"),
        ("👥 Staff", staff.data_loaded, "Loaded" if staff.data_loaded else "Not Loaded"),
        ("📦 Inventory", inventory.data_loaded, "Loaded" if inventory.data_loaded else "Not Loaded"),
    ]

    cols = st.columns(3)

    for i, (label, is_synced, detail) in enumerate(items):
        with cols[i]:
            status_class = "status-good" if is_synced else "status-warning"
            icon = "✓" if is_synced else "○"

            st.markdown(f"""
            <div class="status-badge {status_class}">
                <span>{icon}</span>
                <span>{label}</span>
                <span style="opacity: 0.7; font-size: 0.75rem;">({detail})</span>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PREMIUM KPI CARD
# =============================================================================

def render_premium_card(
    title: str,
    value: str,
    icon: str = "",
    subtitle: str = "",
    status: Optional[str] = None,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    accent_color: str = "blue"
) -> None:
    """
    Render a premium glassmorphic KPI card.

    Args:
        title: Card title
        value: Main value to display
        icon: Optional emoji icon
        subtitle: Optional subtitle text
        status: Optional status ("good", "warning", "danger")
        delta: Optional delta indicator (e.g., "+5%")
        delta_positive: Whether delta is positive (green) or negative (red)
        accent_color: Accent color variant ("blue", "green", "purple", "orange", "red")
    """
    status_class = f"glass-{accent_color}" if accent_color else ""

    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        delta_icon = "↑" if delta_positive else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_icon} {delta}</div>'

    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    subtitle_html = f'<div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""

    card_html = f"""
    <div class="kpi-card glass-card-accent {status_class}">
        {icon_html}
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{title}</div>
        {subtitle_html}
        {delta_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


# =============================================================================
# ACTION CARD
# =============================================================================

def render_action_card(item: ActionItem, show_category: bool = True) -> None:
    """
    Render a priority-styled action card.

    Args:
        item: ActionItem to render
        show_category: Whether to show category badge
    """
    priority_class = f"action-{item.priority}"
    pulse_class = "pulse" if item.priority == "urgent" else ""

    category_html = f'<span class="action-category">{item.category}</span>' if show_category else ""

    card_html = f"""
    <div class="action-card {priority_class} {pulse_class}">
        <div class="action-header">
            <span class="action-icon">{item.icon}</span>
            <span class="action-title">{item.title}</span>
        </div>
        <div class="action-description">{item.description}</div>
        {category_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_action_list(actions: List[ActionItem], max_items: int = 10) -> None:
    """
    Render a list of action cards.

    Args:
        actions: List of ActionItem objects
        max_items: Maximum items to display
    """
    if not actions:
        render_empty_state(
            icon="✅",
            title="No Actions Required",
            description="All systems are operating normally. Complete the workflow to generate actionable insights."
        )
        return

    st.markdown('<div class="stagger-list">', unsafe_allow_html=True)

    for item in actions[:max_items]:
        render_action_card(item)

    st.markdown('</div>', unsafe_allow_html=True)

    if len(actions) > max_items:
        st.info(f"Showing {max_items} of {len(actions)} actions")


# =============================================================================
# COMPARISON PANEL
# =============================================================================

def render_comparison_panel(
    before_title: str,
    before_value: str,
    after_title: str,
    after_value: str,
    savings_value: str,
    savings_label: str = "Weekly Savings"
) -> None:
    """
    Render a before/after comparison panel with savings.

    Args:
        before_title: Title for before section
        before_value: Value before optimization
        after_title: Title for after section
        after_value: Value after optimization
        savings_value: Calculated savings
        savings_label: Label for savings
    """
    comparison_html = f"""
    <div class="comparison-panel">
        <div class="comparison-before">
            <div class="comparison-title">Before</div>
            <div class="kpi-value" style="color: #ef4444; font-size: 2rem;">{before_value}</div>
            <div class="kpi-label">{before_title}</div>
        </div>

        <div class="comparison-arrow">→</div>

        <div class="comparison-after">
            <div class="comparison-title">After</div>
            <div class="kpi-value" style="color: #22c55e; font-size: 2rem;">{after_value}</div>
            <div class="kpi-label">{after_title}</div>
        </div>
    </div>

    <div class="comparison-savings">
        <div class="savings-value">{savings_value}</div>
        <div class="savings-label">{savings_label}</div>
    </div>
    """

    st.markdown(comparison_html, unsafe_allow_html=True)


# =============================================================================
# EMPTY STATE
# =============================================================================

def render_empty_state(
    icon: str,
    title: str,
    description: str,
    action_label: Optional[str] = None,
    action_key: Optional[str] = None
) -> bool:
    """
    Render a premium empty state with optional action button.

    Args:
        icon: Large emoji icon
        title: Title text
        description: Description text
        action_label: Optional button label
        action_key: Optional unique key for button

    Returns:
        True if action button was clicked, False otherwise
    """
    empty_html = f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-description">{description}</div>
    </div>
    """

    st.markdown(empty_html, unsafe_allow_html=True)

    if action_label:
        # Use Streamlit button for interactivity
        return st.button(
            action_label,
            key=action_key or f"empty_action_{hash(title)}",
            type="primary",
            use_container_width=True
        )

    return False


# =============================================================================
# SKELETON LOADER
# =============================================================================

def render_skeleton_loader(count: int = 4, columns: int = 4) -> None:
    """
    Render skeleton loading cards.

    Args:
        count: Number of skeleton cards
        columns: Number of columns
    """
    cols = st.columns(columns)

    for i in range(count):
        with cols[i % columns]:
            st.markdown("""
            <div class="skeleton-card">
                <div class="skeleton-icon loading-shimmer"></div>
                <div class="skeleton-value loading-shimmer"></div>
                <div class="skeleton-label loading-shimmer"></div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# SYSTEM HEALTH GAUGE
# =============================================================================

def render_system_health_gauge(score: float) -> None:
    """
    Render a system health score gauge.

    Args:
        score: Health score (0-100)
    """
    # Determine color and label
    if score >= 80:
        color = "#22c55e"
        label = "Excellent"
    elif score >= 60:
        color = "#22d3ee"
        label = "Good"
    elif score >= 40:
        color = "#f59e0b"
        label = "Fair"
    else:
        color = "#ef4444"
        label = "Needs Attention"

    # Calculate progress bar width
    progress_pct = min(100, max(0, score))

    gauge_html = f"""
    <div class="glass-card" style="padding: 1.5rem; text-align: center;">
        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            System Health Score
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
            {score:.0f}%
        </div>
        <div style="font-size: 0.9rem; color: {color}; font-weight: 600; margin-bottom: 1rem;">
            {label}
        </div>
        <div style="background: rgba(100, 116, 139, 0.2); border-radius: 10px; height: 10px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, {color}, {color}dd); width: {progress_pct}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """

    st.markdown(gauge_html, unsafe_allow_html=True)


# =============================================================================
# FORECAST DAY CARD
# =============================================================================

def render_forecast_day_card(
    day_name: str,
    date_label: str,
    forecast_value: float,
    accuracy: Optional[float] = None,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    is_peak: bool = False,
    actual_value: Optional[float] = None
) -> None:
    """
    Render a single forecast day card.

    Args:
        day_name: Day name (e.g., "Mon", "Tue")
        date_label: Date label (e.g., "Jan 15")
        forecast_value: Predicted patient count
        accuracy: Optional accuracy percentage
        ci_lower: Optional lower confidence interval
        ci_upper: Optional upper confidence interval
        is_peak: Whether this is the peak day
        actual_value: Optional actual value for comparison
    """
    peak_class = "peak" if is_peak else ""

    # Accuracy badge
    if accuracy is not None:
        if accuracy >= 90:
            acc_class = "accuracy-high"
        elif accuracy >= 75:
            acc_class = "accuracy-medium"
        else:
            acc_class = "accuracy-low"
        accuracy_html = f'<div class="forecast-accuracy {acc_class}">{accuracy:.0f}%</div>'
    else:
        accuracy_html = ""

    # Confidence interval
    if ci_lower is not None and ci_upper is not None:
        ci_html = f'<div class="forecast-ci">{ci_lower:.0f} - {ci_upper:.0f}</div>'
    else:
        ci_html = ""

    # Actual comparison
    if actual_value is not None:
        error = forecast_value - actual_value
        error_sign = "+" if error >= 0 else ""
        actual_html = f'''
        <div style="font-size: 0.7rem; color: #22d3ee; margin-top: 0.25rem;">
            Actual: {actual_value:.0f}
        </div>
        <div style="font-size: 0.65rem; color: #94a3b8;">
            ({error_sign}{error:.1f})
        </div>
        '''
    else:
        actual_html = ""

    card_html = f"""
    <div class="forecast-day-card {peak_class}" style="position: relative;">
        <div class="forecast-day">{day_name}</div>
        <div class="forecast-date">{date_label}</div>
        <div class="forecast-value">{int(forecast_value)}</div>
        {accuracy_html}
        {ci_html}
        {actual_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_forecast_day_cards(forecast: ForecastState) -> None:
    """
    Render all 7 forecast day cards.

    Args:
        forecast: ForecastState with forecast data
    """
    if not forecast.has_forecast or not forecast.forecasts:
        render_empty_state(
            icon="🔮",
            title="No Forecast Available",
            description="Train a model in the Train Models page to generate patient forecasts.",
            action_label="Go to Train Models"
        )
        return

    # Day names
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Create columns for cards
    cols = st.columns(min(7, len(forecast.forecasts)))

    st.markdown('<div class="stagger-list" style="display: flex; gap: 0.75rem; flex-wrap: wrap;">', unsafe_allow_html=True)

    for i, value in enumerate(forecast.forecasts[:7]):
        with cols[i]:
            # Get date label if available
            date_label = forecast.dates[i] if i < len(forecast.dates) else ""

            # Get CI if available
            ci_lower = forecast.lower_bounds[i] if i < len(forecast.lower_bounds) else None
            ci_upper = forecast.upper_bounds[i] if i < len(forecast.upper_bounds) else None

            render_forecast_day_card(
                day_name=day_names[i % 7],
                date_label=date_label,
                forecast_value=value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                is_peak=(i == forecast.peak_day)
            )

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SECTION HEADER
# =============================================================================

def render_section_header(title: str, icon: str = "") -> None:
    """
    Render a styled section header.

    Args:
        title: Section title
        icon: Optional emoji icon
    """
    icon_html = f'<span class="section-header-icon">{icon}</span>' if icon else ""

    st.markdown(f"""
    <div class="section-header">
        {icon_html}
        <span>{title}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# GLASS DIVIDER
# =============================================================================

def render_glass_divider() -> None:
    """Render a glassmorphic horizontal divider."""
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
