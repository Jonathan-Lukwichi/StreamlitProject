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
    Self-contained with all CSS included.

    Args:
        title: The main title text (e.g., "Upload Data", "Train Models")
        subtitle: Descriptive subtitle text
        status: Status indicator text (default: "SYSTEM ONLINE")
    """
    # Inject CSS styles for the hero header
    st.markdown("""
        <style>
        /* ========================================
           PREMIUM HERO HEADER - SELF-CONTAINED CSS
           ======================================== */

        /* Status Indicator Blinking */
        @keyframes status-blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.5; }
        }

        /* Glitch Effect for Hero Title (subtle) */
        @keyframes glitch {
            0%, 90%, 100% { transform: translate(0); }
            92% { transform: translate(-2px, 1px); }
            94% { transform: translate(2px, -1px); }
            96% { transform: translate(-1px, 2px); }
            98% { transform: translate(1px, -2px); }
        }

        /* Glow Line Animation */
        @keyframes glow-pulse {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; }
        }

        .dashboard-hero {
            background: linear-gradient(135deg,
                rgba(6, 78, 145, 0.3) 0%,
                rgba(15, 23, 42, 0.95) 50%,
                rgba(6, 78, 145, 0.2) 100%);
            border: 1px solid rgba(34, 211, 238, 0.3);
            border-radius: 20px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .dashboard-hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(ellipse at top right, rgba(34, 211, 238, 0.1), transparent 50%),
                        radial-gradient(ellipse at bottom left, rgba(59, 130, 246, 0.1), transparent 50%);
            pointer-events: none;
        }

        .hero-content {
            position: relative;
            z-index: 2;
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: #22d3ee;
            margin-bottom: 0.5rem;
        }

        .hero-eyebrow::before {
            content: '';
            width: 24px;
            height: 2px;
            background: linear-gradient(90deg, #22d3ee, transparent);
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 900;
            color: #22d3ee;
            background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            margin-bottom: 0.75rem;
            margin-top: 0.5rem;
            filter: drop-shadow(0 0 30px rgba(34, 211, 238, 0.5));
        }

        .hero-title-glitch {
            animation: glitch 10s ease-in-out infinite;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            max-width: 600px;
            line-height: 1.5;
        }

        .hero-glow-line {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg,
                transparent,
                rgba(34, 211, 238, 0.8),
                rgba(59, 130, 246, 0.6),
                rgba(34, 211, 238, 0.8),
                transparent);
            animation: glow-pulse 3s ease-in-out infinite;
        }

        .status-online {
            color: #22c55e;
            animation: status-blink 2s ease-in-out infinite;
        }

        .status-online::before {
            content: '●';
            margin-right: 0.5rem;
        }

        /* Corner Brackets Effect */
        .corner-brackets {
            position: relative;
        }

        .corner-brackets::before,
        .corner-brackets::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border-color: rgba(34, 211, 238, 0.5);
            border-style: solid;
        }

        .corner-brackets::before {
            top: 0;
            left: 0;
            border-width: 2px 0 0 2px;
        }

        .corner-brackets::after {
            bottom: 0;
            right: 0;
            border-width: 0 2px 2px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Render the hero header HTML
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

