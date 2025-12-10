# =============================================================================
# app_core/ui/effects.py
# Centralized Visual Effects for HealthForecast AI
# Eliminates duplicate CSS across pages
# =============================================================================

import streamlit as st
from typing import Optional, Tuple

from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)


# =============================================================================
# FLUORESCENT ORB EFFECTS
# =============================================================================

FLUORESCENT_ORBS_CSS = """
<style>
/* ========================================
   FLUORESCENT ORB EFFECTS (Shared)
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

/* Responsive adjustments */
@media (max-width: 768px) {
    .fluorescent-orb {
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }
    .sparkle { display: none; }
}
</style>

<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
"""


# =============================================================================
# CUSTOM TAB STYLING
# =============================================================================

def get_tab_styling_css() -> str:
    """Generate CSS for custom tab styling"""
    return f"""
<style>
/* ========================================
   CUSTOM TAB STYLING (Shared)
   ======================================== */

/* Tab container */
.stTabs {{
    background: transparent;
    margin-top: 1rem;
}}

/* Tab list (container for all tab buttons) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}

/* Individual tab buttons */
.stTabs [data-baseweb="tab"] {{
    height: 50px;
    background: transparent;
    border-radius: 12px;
    padding: 0 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: {BODY_TEXT};
    border: 1px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* Tab button hover effect */
.stTabs [data-baseweb="tab"]:hover {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
    border-color: rgba(59, 130, 246, 0.3);
    color: {TEXT_COLOR};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}}

/* Active/selected tab */
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(34, 211, 238, 0.2)) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    color: {TEXT_COLOR} !important;
    box-shadow:
        0 0 20px rgba(59, 130, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}

/* Active tab indicator (underline) - hide it */
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent;
}}

/* Tab panels (content area) */
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
}}

@media (max-width: 768px) {{
    /* Make tabs stack vertically on mobile */
    .stTabs [data-baseweb="tab-list"] {{
        flex-direction: column;
    }}
    .stTabs [data-baseweb="tab"] {{
        width: 100%;
    }}
}}
</style>
"""


# =============================================================================
# PAGE HEADER COMPONENT
# =============================================================================

def render_page_header(
    title: str,
    subtitle: str,
    icon: str,
    gradient_colors: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Render a standardized page header with gradient background.

    Args:
        title: Main page title
        subtitle: Description text
        icon: Emoji icon
        gradient_colors: Tuple of (start_color, end_color) for gradient
    """
    if gradient_colors is None:
        gradient_colors = (PRIMARY_COLOR, SECONDARY_COLOR)

    start_color, end_color = gradient_colors

    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>{icon}</div>
      <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;
          background: linear-gradient(135deg, {start_color}, {end_color});
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text;'>{title}</h1>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 auto;'>
        {subtitle}
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(
    title: str,
    icon: str,
    color: str = PRIMARY_COLOR,
    description: Optional[str] = None,
) -> None:
    """
    Render a section header within a page.

    Args:
        title: Section title
        icon: Emoji icon
        color: Accent color
        description: Optional description text
    """
    html = f"""
    <div style='background: linear-gradient(135deg, rgba({_hex_to_rgb(color)}, 0.15), rgba({_hex_to_rgb(color)}, 0.05));
                border: 1px solid rgba({_hex_to_rgb(color)}, 0.3);
                border-radius: 12px;
                padding: 1.25rem;
                margin: 1.5rem 0 1rem 0;'>
        <div style='display: flex; align-items: center; gap: 0.75rem;'>
            <span style='font-size: 1.5rem;'>{icon}</span>
            <div>
                <h3 style='color: {TEXT_COLOR}; margin: 0; font-size: 1.2rem; font-weight: 700;'>
                    {title}
                </h3>
    """

    if description:
        html += f"""
                <p style='color: {SUBTLE_TEXT}; margin: 0.25rem 0 0 0; font-size: 0.85rem;'>
                    {description}
                </p>
        """

    html += """
            </div>
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# STATUS CARDS
# =============================================================================

def render_status_card(
    title: str,
    status: str,
    is_ready: bool,
    description: Optional[str] = None,
) -> None:
    """
    Render a status card showing ready/pending state.

    Args:
        title: Card title
        status: Status text (e.g., "Ready", "Pending")
        is_ready: Whether the status is positive
        description: Optional description
    """
    status_icon = "✅" if is_ready else "⏳"
    rgb = "16,185,129" if is_ready else "245,158,11"
    color = SUCCESS_COLOR if is_ready else WARNING_COLOR

    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
      <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>
        {title}
      </div>
      <div style='margin: 1rem 0;'>
        <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3);
                                     font-size: 0.875rem; padding: 0.5rem 1rem;'>
          {status_icon} {status}
        </span>
      </div>
      {"<div class='hf-feature-description' style='font-size: 0.75rem;'>" + description + "</div>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# KPI INDICATOR
# =============================================================================

def create_kpi_indicator(
    title: str,
    value: float,
    unit: str = "",
    color: str = PRIMARY_COLOR,
) -> "go.Figure":
    """
    Create a Plotly figure for a KPI indicator.

    Args:
        title: KPI title
        value: Numeric value
        unit: Unit suffix (e.g., "%", "days")
        color: Accent color

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": TEXT_COLOR}},
        number={
            "font": {"size": 32, "color": color},
            "suffix": unit,
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        height=120,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def inject_fluorescent_effects() -> None:
    """Inject the fluorescent orb effects CSS and HTML"""
    st.markdown(FLUORESCENT_ORBS_CSS, unsafe_allow_html=True)


def inject_tab_styling() -> None:
    """Inject custom tab styling CSS"""
    st.markdown(get_tab_styling_css(), unsafe_allow_html=True)


def inject_all_effects() -> None:
    """Inject all visual effects (orbs + tab styling)"""
    inject_fluorescent_effects()
    inject_tab_styling()


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string for CSS"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
