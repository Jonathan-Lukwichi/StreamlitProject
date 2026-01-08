import streamlit as st
from .theme import (PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
                    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, CARD_BG)

def header(title: str, subtitle: str, icon: str = "🩺"):
    """Paste your current header() function here (no design changes)."""
    st.markdown(f"""
        <div class="main-header">
            <div style="display:flex;gap:1.2rem;align-items:center;">
                <div style="font-size:3rem;filter:drop-shadow(0 0 15px rgba(255,255,255,.5));">{icon}</div>
                <div>
                    <h1 style="margin:0; font-size:2.4rem; color:white;">{title}</h1>
                    <p style="margin:.35rem 0 0 0;color:rgba(255,255,255,.85);font-size:1.05rem">{subtitle}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"## {icon} {title}")
    st.caption(subtitle)

def add_grid(fig):
    """Paste your existing add_grid(fig) implementation here to keep plot styling consistent."""
    grid_color = 'rgba(255, 255, 255, 0.2)'
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False,
                     showline=True, linecolor=grid_color,
                     tickfont=dict(color=SUBTLE_TEXT), title_font=dict(color=TEXT_COLOR))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False,
                     showline=True, linecolor=grid_color,
                     tickfont=dict(color=SUBTLE_TEXT), title_font=dict(color=TEXT_COLOR))
    fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                      font=dict(family="Segoe UI, sans-serif", size=12, color=TEXT_COLOR),
                      title_font=dict(color=TEXT_COLOR))
    return fig


def render_scifi_hero_header(title: str, subtitle: str, status: str = "SYSTEM ONLINE"):
    """
    Render sci-fi themed hero header matching HORIZON Control style.

    Args:
        title: The main title text (e.g., "DATA Uplink", "NEURAL Core")
        subtitle: Descriptive subtitle text
        status: Status indicator text (default: "SYSTEM ONLINE")
    """
    st.markdown(f"""
    <div class="dashboard-hero corner-brackets">
        <div class="hero-content">
            <div class="hero-eyebrow">
                <span class="status-online">{status}</span> · HealthForecast AI
            </div>
            <h1 class="hero-title hero-title-glitch">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        <div class="hero-glow-line"></div>
    </div>
    """, unsafe_allow_html=True)

