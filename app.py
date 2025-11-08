# ============================================================================
# pages/00_Welcome_minimal_linked.py — Enhanced Professional Welcome Page
# Generic summary replacing feature cards for a broader prototype message.
# ============================================================================
from __future__ import annotations

import streamlit as st
import os

try:
    from app_core.state.session import init_state as _init_state
except Exception:
    def _init_state():
        for k, v in {
            "patient_loaded": False,
            "weather_loaded": False,
            "calendar_loaded": False,
            "merged_data": None,
            "_nav_intent": None,
        }.items():
            st.session_state.setdefault(k, v)

try:
    from app_core.ui.theme import apply_css as _apply_css
    from app_core.ui.theme import (
        PRIMARY_COLOR, TEXT_COLOR, SUBTLE_TEXT,
    )
except Exception:
    _apply_css = None
    PRIMARY_COLOR = "#2563eb"
    TEXT_COLOR = "#0f172a"
    SUBTLE_TEXT = "#64748b"

st.set_page_config(
    page_title="Healthcare Forecast Pro | Prototype",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _local_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*='css'] {{ font-family: 'Inter', system-ui, sans-serif; }}
        .hero {{
            border-radius: 20px; padding: 40px; color: #0b1220;
            background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(99,102,241,.08), rgba(20,184,166,.12));
            border: 1px solid rgba(37,99,235,.25);
            box-shadow: 0 20px 25px -5px rgba(0,0,0,.08), 0 10px 10px -5px rgba(0,0,0,.04);
            position: relative; overflow: hidden;
        }}
        .hero::before {{
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 80% 20%, rgba(99,102,241,.15), transparent 50%),
                        radial-gradient(circle at 20% 80%, rgba(20,184,166,.15), transparent 50%);
            pointer-events: none;
        }}
        .hero > * {{ position: relative; z-index: 1; }}
        .hero-title {{ font-size: 2.5rem; font-weight: 800; margin-bottom: 12px; color: {TEXT_COLOR}; 
                       letter-spacing: -0.03em; 
                       background: linear-gradient(135deg, {TEXT_COLOR}, {PRIMARY_COLOR});
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; }}
        .hero-sub {{ color: {SUBTLE_TEXT}; margin-bottom: 12px; font-size: 1.1rem; line-height: 1.5; }}
        .hero-note {{ color: {SUBTLE_TEXT}; font-size: .95rem; line-height: 1.6; }}
        .pill {{ display:inline-block; padding:6px 14px; border-radius:999px; font-size:.82rem; font-weight:600;
                 color:#fff; background: linear-gradient(135deg, {PRIMARY_COLOR}, #6366f1); 
                 border:1px solid rgba(255,255,255,.2); 
                 margin-right:8px; margin-bottom:10px; transition: all 0.3s ease;
                 box-shadow: 0 4px 6px -1px rgba(37,99,235,.3); }}
        .pill:hover {{ transform: translateY(-2px); box-shadow: 0 6px 12px -2px rgba(37,99,235,.4); }}
        .generic-card {{ border:1px solid #e5e7eb; border-radius:16px; padding:28px; 
                         background: linear-gradient(to bottom, #fff, rgba(249,250,251,1)); 
                         box-shadow: 0 4px 6px -1px rgba(0,0,0,.06), 0 2px 4px -1px rgba(0,0,0,.03);
                         border: 1px solid rgba(37,99,235,.08); }}
        .generic-title {{ font-weight:700; font-size:1.4rem; color:{TEXT_COLOR}; margin-bottom:14px; 
                          letter-spacing: -0.02em; }}
        .generic-text {{ color:{SUBTLE_TEXT}; font-size:1.05rem; line-height:1.8; }}
        .stats-bar {{ display: flex; gap: 24px; margin-top: 20px; flex-wrap: wrap; }}
        .stat-item {{ flex: 1; min-width: 140px; text-align: center; padding: 12px; 
                      background: rgba(255,255,255,.6); border-radius: 10px; 
                      border: 1px solid rgba(37,99,235,.15); }}
        .stat-number {{ font-size: 1.8rem; font-weight: 800; color: {PRIMARY_COLOR}; 
                        background: linear-gradient(135deg, {PRIMARY_COLOR}, #14b8a6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text; }}
        .stat-label {{ font-size: 0.85rem; color: {SUBTLE_TEXT}; font-weight: 500; text-transform: uppercase;
                       letter-spacing: 0.5px; margin-top: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    if _apply_css:
        try:
            _apply_css()
        except Exception:
            pass
    _local_css()
    _init_state()

    # HERO
    st.markdown(
        f"""
        <div class='hero'>
          <div>
            <span class='pill'>🤖 AI‑Driven Forecasting</span>
            <span class='pill'>📊 Data Integration</span>
            <span class='pill'>⚡ Real-Time Analytics</span>
          </div>
          <h1 class='hero-title'>Healthcare Forecast Pro</h1>
          <p class='hero-sub'>Next-generation platform for intelligent healthcare analytics and predictive insights</p>
          <p class='hero-note'>Navigate through the <strong>sidebar modules</strong> to upload and process data, merge multiple sources, and generate patient‑arrival forecasts powered by advanced machine learning models.</p>
          
          <div class='stats-bar'>
            <div class='stat-item'>
              <div class='stat-number'>∞</div>
              <div class='stat-label'>Data Points</div>
            </div>
            <div class='stat-item'>
              <div class='stat-number'>AI</div>
              <div class='stat-label'>Powered</div>
            </div>
            <div class='stat-item'>
              <div class='stat-number'>24/7</div>
              <div class='stat-label'>Monitoring</div>
            </div>
            <div class='stat-item'>
              <div class='stat-number'>∞</div>
              <div class='stat-label'>Scalability</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # GENERIC OVERVIEW SECTION
    st.markdown(
        f"""
        <div class='generic-card'>
            <div class='generic-title'>🚀 Empowering Healthcare Through Intelligence</div>
            <div class='generic-text'>
                This platform harnesses the power of artificial intelligence and data science to transform healthcare operations.
                By seamlessly integrating patient records, weather patterns, and calendar events, our advanced machine learning 
                algorithms predict patient arrival patterns with unprecedented accuracy. Healthcare professionals gain the insights 
                needed to optimize resource allocation, reduce wait times, and deliver exceptional patient care through 
                intelligent, data-driven decision-making.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()