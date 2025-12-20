# =============================================================================
# app_core/ui/insight_components.py
# UI Components for AI-Powered Insights Display
# =============================================================================
"""
Streamlit UI components for displaying AI-generated insights.

These components render:
- Smart KPI cards with contextual styling
- Key insight boxes with explanations
- Recommendation tables with priorities
- Executive summary sections
- Mode selector (Rule-based vs LLM)
"""

from __future__ import annotations
import streamlit as st
from typing import Dict, List, Any, Optional

from app_core.ai.recommendation_engine import (
    InsightResult, InsightMode, KPICard, Insight, Recommendation, Priority
)


# =============================================================================
# STYLING
# =============================================================================

INSIGHT_STYLES = """
<style>
/* AI Insight Container */
.ai-insight-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Smart KPI Card */
.smart-kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.smart-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.smart-kpi-card.investment {
    border-color: rgba(234, 179, 8, 0.5);
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1), rgba(15, 23, 42, 0.98));
}
.smart-kpi-card.savings {
    border-color: rgba(34, 197, 94, 0.5);
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(15, 23, 42, 0.98));
}
.smart-kpi-card.warning {
    border-color: rgba(239, 68, 68, 0.5);
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.98));
}
.smart-kpi-card.info {
    border-color: rgba(59, 130, 246, 0.5);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(15, 23, 42, 0.98));
}

.smart-kpi-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.smart-kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.smart-kpi-value.green { color: #22c55e; }
.smart-kpi-value.yellow { color: #eab308; }
.smart-kpi-value.red { color: #ef4444; }
.smart-kpi-value.blue { color: #3b82f6; }
.smart-kpi-value.purple { color: #a855f7; }

.smart-kpi-label {
    font-size: 0.85rem;
    color: #94a3b8;
}

.smart-kpi-delta {
    font-size: 0.8rem;
    margin-top: 0.25rem;
}
.smart-kpi-delta.positive { color: #22c55e; }
.smart-kpi-delta.negative { color: #ef4444; }
.smart-kpi-delta.neutral { color: #94a3b8; }

/* Key Insight Box */
.key-insight-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
}

.key-insight-box.investment {
    border-left: 4px solid #eab308;
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.08), rgba(15, 23, 42, 0.95));
}

.key-insight-box.savings {
    border-left: 4px solid #22c55e;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(15, 23, 42, 0.95));
}

.key-insight-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.key-insight-message {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.key-insight-details {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
}

.key-insight-details ul {
    margin: 0;
    padding-left: 1.25rem;
}

.key-insight-details li {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

/* Recommendation Card */
.recommendation-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}

.recommendation-card.critical {
    border-left: 3px solid #ef4444;
}
.recommendation-card.high {
    border-left: 3px solid #f97316;
}
.recommendation-card.medium {
    border-left: 3px solid #eab308;
}
.recommendation-card.low {
    border-left: 3px solid #22c55e;
}

.recommendation-priority {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    white-space: nowrap;
}

.recommendation-priority.critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.recommendation-priority.high { background: rgba(249, 115, 22, 0.2); color: #f97316; }
.recommendation-priority.medium { background: rgba(234, 179, 8, 0.2); color: #eab308; }
.recommendation-priority.low { background: rgba(34, 197, 94, 0.2); color: #22c55e; }

.recommendation-content {
    flex: 1;
}

.recommendation-action {
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}

.recommendation-impact {
    font-size: 0.85rem;
    color: #94a3b8;
}

/* Mode Selector */
.mode-selector {
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    margin-bottom: 1rem;
}

.mode-option {
    flex: 1;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.mode-option.active {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(59, 130, 246, 0.1));
    border-color: rgba(168, 85, 247, 0.4);
}

.mode-option:hover:not(.active) {
    background: rgba(148, 163, 184, 0.1);
}

/* Executive Summary */
.executive-summary {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.executive-summary h4 {
    color: #f1f5f9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
</style>
"""


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

def inject_insight_styles():
    """Inject CSS styles for insight components."""
    st.markdown(INSIGHT_STYLES, unsafe_allow_html=True)


def render_mode_selector(current_mode: str = "rule_based", key: str = "insight_mode") -> str:
    """
    Render a mode selector for choosing between Rule-based and LLM insights.

    Args:
        current_mode: Current selected mode
        key: Unique key for the component

    Returns:
        Selected mode string ("rule_based" or "llm")
    """
    col1, col2 = st.columns(2)

    with col1:
        rule_based = st.button(
            "Option A: Rule-Based (Fast, Free)",
            key=f"{key}_rule",
            type="primary" if current_mode == "rule_based" else "secondary",
            use_container_width=True
        )

    with col2:
        llm = st.button(
            "Option B: LLM-Powered (Natural Language)",
            key=f"{key}_llm",
            type="primary" if current_mode == "llm" else "secondary",
            use_container_width=True
        )

    if rule_based:
        return "rule_based"
    elif llm:
        return "llm"

    return current_mode


def render_smart_kpi_cards(kpi_cards: List[KPICard], columns: int = 4):
    """
    Render smart KPI cards with contextual styling.

    Args:
        kpi_cards: List of KPICard objects
        columns: Number of columns for layout
    """
    cols = st.columns(columns)

    for i, kpi in enumerate(kpi_cards):
        with cols[i % columns]:
            # Determine card class
            if "investment" in kpi.label.lower() or "needed" in kpi.label.lower():
                card_class = "investment"
            elif "saving" in kpi.label.lower() or "eliminated" in kpi.label.lower():
                card_class = "savings"
            elif kpi.delta_type == "negative":
                card_class = "warning"
            else:
                card_class = "info"

            # Determine value color
            color_class = kpi.color

            st.markdown(f"""
            <div class="smart-kpi-card {card_class}">
                <div class="smart-kpi-icon">{kpi.icon}</div>
                <div class="smart-kpi-value {color_class}">{kpi.value}</div>
                <div class="smart-kpi-label">{kpi.label}</div>
                <div class="smart-kpi-delta {kpi.delta_type}">{kpi.delta}</div>
            </div>
            """, unsafe_allow_html=True)


def render_key_insight(insight: Insight):
    """
    Render the key insight box with explanation.

    Args:
        insight: Insight object containing the main finding
    """
    # Determine box class
    if insight.color in ["yellow", "orange"]:
        box_class = "investment"
    else:
        box_class = "savings"

    # Build details section if present
    details_section = ""
    if insight.details and len(insight.details) > 0:
        details_items = "".join([f"<li>{d}</li>" for d in insight.details])
        details_section = f'<div class="key-insight-details"><ul>{details_items}</ul></div>'

    # Build the complete HTML (no leading whitespace to avoid rendering issues)
    html_content = f'''<div class="key-insight-box {box_class}">
<div class="key-insight-title">{insight.icon} {insight.title}</div>
<div class="key-insight-message">{insight.message}</div>
{details_section}
</div>'''

    st.markdown(html_content, unsafe_allow_html=True)


def render_recommendations(recommendations: List[Recommendation]):
    """
    Render the recommendations as styled cards.

    Args:
        recommendations: List of Recommendation objects
    """
    if not recommendations:
        return

    st.markdown("### Recommended Actions")

    for rec in recommendations:
        priority_class = rec.priority.value

        st.markdown(f"""
        <div class="recommendation-card {priority_class}">
            <span class="recommendation-priority {priority_class}">{rec.priority.value.upper()}</span>
            <div class="recommendation-content">
                <div class="recommendation-action">{rec.action}</div>
                <div class="recommendation-impact">{rec.impact}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_summary_table(summary_table: Dict[str, Dict[str, Any]]):
    """
    Render the before/after comparison table.

    Args:
        summary_table: Dictionary with metrics and their before/after values
    """
    # Build markdown table
    table_md = "| Metric | Before | After | Change |\n|--------|--------|-------|--------|\n"

    for metric, values in summary_table.items():
        before = values.get("before", "-")
        after = values.get("after", "-")
        change = values.get("change", "-")

        # Bold the change column
        table_md += f"| {metric} | {before} | {after} | **{change}** |\n"

    st.markdown(table_md)


def render_executive_summary(summary: str, mode: InsightMode):
    """
    Render the executive summary section.

    Args:
        summary: Executive summary text (supports markdown)
        mode: The mode used to generate the summary
    """
    mode_badge = "" if mode == InsightMode.RULE_BASED else ""
    mode_text = "Rule-Based Analysis" if mode == InsightMode.RULE_BASED else "AI-Powered Analysis"

    st.markdown(f"""
    <div class="executive-summary">
        <h4>{mode_badge} Executive Summary ({mode_text})</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(summary)


def render_full_insight_panel(
    result: InsightResult,
    show_mode_info: bool = True,
    show_recommendations: bool = True,
    show_table: bool = True
):
    """
    Render the complete insight panel with all components.

    Args:
        result: InsightResult from the recommendation engine
        show_mode_info: Whether to show the mode indicator
        show_recommendations: Whether to show recommendations
        show_table: Whether to show the comparison table
    """
    # Inject styles
    inject_insight_styles()

    # Mode indicator
    if show_mode_info:
        mode_text = "Rule-Based" if result.mode == InsightMode.RULE_BASED else "AI-Powered"
        mode_icon = "" if result.mode == InsightMode.RULE_BASED else ""
        st.caption(f"{mode_icon} Insight Mode: {mode_text}")

    # KPI Cards
    st.markdown("### Key Metrics")
    render_smart_kpi_cards(result.kpi_cards, columns=min(len(result.kpi_cards), 4))

    # Key Insight
    st.markdown("---")
    st.markdown("### Key Finding")
    render_key_insight(result.key_insight)

    # Recommendations
    if show_recommendations and result.recommendations:
        st.markdown("---")
        render_recommendations(result.recommendations)

    # Summary Table
    if show_table:
        st.markdown("---")
        st.markdown("### Before vs After Comparison")
        render_summary_table(result.summary_table)

    # Executive Summary
    st.markdown("---")
    render_executive_summary(result.executive_summary, result.mode)


# =============================================================================
# SIMPLIFIED INTEGRATION FUNCTIONS
# =============================================================================

def render_staff_optimization_insights(
    before_stats: Dict[str, Any],
    optimization_results: Dict[str, Any],
    cost_params: Optional[Dict[str, Any]] = None,
    mode: str = "rule_based",
    api_key: Optional[str] = None
) -> InsightResult:
    """
    Simplified function to generate and render staff optimization insights.

    Args:
        before_stats: Historical staff statistics
        optimization_results: Results from calculate_optimization()
        cost_params: Cost parameters
        mode: "rule_based" or "llm"
        api_key: Optional API key for LLM mode

    Returns:
        InsightResult for further processing if needed
    """
    from app_core.ai.recommendation_engine import StaffRecommendationEngine

    engine = StaffRecommendationEngine(mode=mode, api_key=api_key)
    result = engine.generate_insights(
        before_stats=before_stats,
        after_stats={},  # Not used separately
        optimization_results=optimization_results,
        cost_params=cost_params
    )

    render_full_insight_panel(result)
    return result


def render_supply_optimization_insights(
    before_stats: Dict[str, Any],
    optimization_results: Dict[str, Any],
    cost_params: Optional[Dict[str, Any]] = None,
    mode: str = "rule_based",
    api_key: Optional[str] = None
) -> InsightResult:
    """
    Simplified function to generate and render supply optimization insights.

    Args:
        before_stats: Historical inventory statistics
        optimization_results: Results from calculate_optimization()
        cost_params: Cost parameters
        mode: "rule_based" or "llm"
        api_key: Optional API key for LLM mode

    Returns:
        InsightResult for further processing if needed
    """
    from app_core.ai.recommendation_engine import SupplyRecommendationEngine

    engine = SupplyRecommendationEngine(mode=mode, api_key=api_key)
    result = engine.generate_insights(
        before_stats=before_stats,
        after_stats={},  # Not used separately
        optimization_results=optimization_results,
        cost_params=cost_params
    )

    render_full_insight_panel(result)
    return result
