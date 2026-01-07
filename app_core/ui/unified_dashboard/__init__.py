# =============================================================================
# app_core/ui/unified_dashboard/__init__.py
# Premium Unified Dashboard Component Library
# =============================================================================
"""
Unified Dashboard component library for HealthForecast AI Command Center.

This package provides:
- Premium CSS styles with glassmorphism and animations
- Reusable UI components (cards, headers, empty states)
- Chart templates optimized for the dark theme
- State management integration
"""

from .styles import (
    get_unified_dashboard_styles,
    ANIMATION_KEYFRAMES,
    GLASSMORPHIC_CARD_CSS,
    NORTHSTAR_HEADER_CSS,
)

from .components import (
    render_northstar_header,
    render_premium_card,
    render_sync_status,
    render_action_card,
    render_action_list,
    render_comparison_panel,
    render_empty_state,
    render_skeleton_loader,
    render_system_health_gauge,
    render_forecast_day_card,
    render_forecast_day_cards,
    render_section_header,
    render_glass_divider,
)

from .charts import (
    create_forecast_area_chart,
    create_utilization_gauge,
    create_cost_comparison_bar,
    create_category_breakdown_chart,
    get_premium_chart_layout,
    CHART_COLORS,
)

__all__ = [
    # Styles
    "get_unified_dashboard_styles",
    "ANIMATION_KEYFRAMES",
    "GLASSMORPHIC_CARD_CSS",
    "NORTHSTAR_HEADER_CSS",
    # Components
    "render_northstar_header",
    "render_premium_card",
    "render_sync_status",
    "render_action_card",
    "render_action_list",
    "render_comparison_panel",
    "render_empty_state",
    "render_skeleton_loader",
    "render_system_health_gauge",
    "render_forecast_day_card",
    "render_forecast_day_cards",
    "render_section_header",
    "render_glass_divider",
    # Charts
    "create_forecast_area_chart",
    "create_utilization_gauge",
    "create_cost_comparison_bar",
    "create_category_breakdown_chart",
    "get_premium_chart_layout",
    "CHART_COLORS",
]
