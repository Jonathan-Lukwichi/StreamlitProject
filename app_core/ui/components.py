import streamlit as st
from .theme import (PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
                    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, GRID_COLOR, CARD_BG_LIGHT)

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
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=GRID_COLOR,
                     tickfont=dict(color=SUBTLE_TEXT), title_font=dict(color=TEXT_COLOR))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=GRID_COLOR,
                     tickfont=dict(color=SUBTLE_TEXT), title_font=dict(color=TEXT_COLOR))
    fig.update_layout(plot_bgcolor=CARD_BG_LIGHT, paper_bgcolor=CARD_BG_LIGHT,
                      font=dict(family="Segoe UI, sans-serif", size=12, color=TEXT_COLOR),
                      title_font=dict(color=TEXT_COLOR))
    return fig
    
