from __future__ import annotations
import streamlit as st
import base64
from pathlib import Path
from app_core.ui.theme import apply_css
from app_core.auth.authentication import initialize_session_state
from app_core.auth.navigation import configure_sidebar_navigation
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.results_storage_ui import render_pipeline_status_dashboard

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="HealthForecast AI - Welcome",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
initialize_session_state()

# Configure sidebar navigation
configure_sidebar_navigation()

# Apply Premium CSS Theme
apply_css()
inject_sidebar_style()

# Render sidebar brand
with st.sidebar:
    render_sidebar_brand()


# ============================================================================
# IMAGE HELPER FUNCTIONS
# ============================================================================
def get_image_base64(image_path: str) -> str:
    """Convert image to base64 for CSS background."""
    try:
        img_path = Path(image_path)
        if img_path.exists():
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return ""


def get_image_css_url(image_name: str) -> str:
    """Get base64 data URL for image."""
    base_path = Path(__file__).parent / "images" / image_name
    b64 = get_image_base64(str(base_path))
    if b64:
        return f"data:image/jpeg;base64,{b64}"
    return ""


# ============================================================================
# LOAD IMAGES
# ============================================================================
hero_bg = get_image_css_url("hero-bg1.jpg")
login_bg = get_image_css_url("login-bg1.jpg")
carousel_1 = get_image_css_url("carousel-1.jpg")
carousel_2 = get_image_css_url("carousel-2.jpg")
carousel_3 = get_image_css_url("carousel-3.jpg")
team_bg1 = get_image_css_url("team-bg1.jpg")
team_bg2 = get_image_css_url("team-bg2.jpg")
dashboard_bg = get_image_css_url("dashboard-bg.jpg")
staff_bg = get_image_css_url("staff-bg.jpg")


# ============================================================================
# ENHANCED CSS WITH ALL IMPROVEMENTS
# ============================================================================
st.markdown(f"""
<style>
/* ========================================
   ANIMATIONS
   ======================================== */

@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
}}

@keyframes pulse-glow {{
    0%, 100% {{
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4),
                    0 0 40px rgba(59, 130, 246, 0.2),
                    0 0 60px rgba(59, 130, 246, 0.1);
    }}
    50% {{
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.6),
                    0 0 60px rgba(59, 130, 246, 0.4),
                    0 0 90px rgba(59, 130, 246, 0.2);
    }}
}}

@keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 100%; }}
}}

@keyframes text-glow {{
    0%, 100% {{
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.5),
                     0 0 40px rgba(34, 211, 238, 0.3),
                     0 4px 20px rgba(0, 0, 0, 0.8);
    }}
    50% {{
        text-shadow: 0 0 30px rgba(34, 211, 238, 0.7),
                     0 0 60px rgba(34, 211, 238, 0.5),
                     0 4px 20px rgba(0, 0, 0, 0.8);
    }}
}}

@keyframes border-glow {{
    0%, 100% {{ border-color: rgba(34, 211, 238, 0.3); }}
    50% {{ border-color: rgba(34, 211, 238, 0.6); }}
}}

/* ========================================
   HERO SECTION - ENHANCED
   ======================================== */

.hero-section {{
    position: relative;
    width: 100%;
    min-height: 520px;
    background: linear-gradient(135deg,
        rgba(2, 6, 23, 0.92) 0%,
        rgba(6, 78, 145, 0.75) 50%,
        rgba(2, 6, 23, 0.95) 100%),
        url('{hero_bg}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 28px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 4rem 2rem;
    margin-bottom: 2rem;
    overflow: hidden;
    border: 1px solid rgba(34, 211, 238, 0.25);
    box-shadow: 0 0 80px rgba(6, 78, 145, 0.4),
                0 25px 50px rgba(0, 0, 0, 0.5),
                inset 0 0 150px rgba(0, 0, 0, 0.6);
    animation: border-glow 4s ease-in-out infinite;
}}

.hero-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at 30% 20%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(34, 211, 238, 0.1) 0%, transparent 50%);
    pointer-events: none;
}}

.hero-section::after {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.03), transparent);
    animation: shimmer 8s infinite;
    pointer-events: none;
}}

.hero-badge {{
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.25) 0%, rgba(59, 130, 246, 0.25) 100%);
    border: 1px solid rgba(34, 211, 238, 0.5);
    border-radius: 50px;
    padding: 0.6rem 1.75rem;
    font-size: 0.8rem;
    color: #67e8f9;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 700;
    margin-bottom: 1.75rem;
    backdrop-filter: blur(15px);
    z-index: 1;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
    animation: float 6s ease-in-out infinite;
}}

.hero-title {{
    font-size: 3.75rem;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    margin: 0 0 0.5rem 0;
    z-index: 1;
    line-height: 1.15;
    letter-spacing: -1px;
    /* Enhanced text shadow for better readability */
    text-shadow:
        0 0 30px rgba(0, 0, 0, 0.9),
        0 0 60px rgba(0, 0, 0, 0.7),
        0 4px 30px rgba(0, 0, 0, 0.9),
        0 8px 40px rgba(0, 0, 0, 0.5);
}}

.hero-title-accent {{
    display: block;
    font-size: 4rem;
    background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 40%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(34, 211, 238, 0.5));
    animation: text-glow 3s ease-in-out infinite;
}}

.hero-subtitle {{
    font-size: 1.2rem;
    color: #cbd5e1;
    text-align: center;
    max-width: 720px;
    margin: 1.25rem auto 2.5rem auto;
    line-height: 1.8;
    z-index: 1;
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.8);
}}

.hero-stats {{
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 2rem;
    z-index: 1;
}}

.hero-stat {{
    text-align: center;
    padding: 1.25rem 2rem;
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
    border-radius: 18px;
    border: 1px solid rgba(34, 211, 238, 0.25);
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
    min-width: 140px;
}}

.hero-stat:hover {{
    transform: translateY(-5px) scale(1.02);
    border-color: rgba(34, 211, 238, 0.5);
    box-shadow: 0 15px 40px rgba(6, 78, 145, 0.4);
}}

.hero-stat-value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: #22d3ee;
    text-shadow: 0 0 25px rgba(34, 211, 238, 0.6);
}}

.hero-stat-label {{
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}}

/* ========================================
   BUTTON STYLING - ENHANCED GLOW
   ======================================== */

div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%) !important;
    border: 2px solid rgba(96, 165, 250, 0.6) !important;
    border-radius: 16px !important;
    padding: 1.25rem 3.5rem !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: white !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    animation: pulse-glow 2.5s ease-in-out infinite !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    min-height: 65px !important;
    position: relative !important;
    overflow: hidden !important;
}}

div[data-testid="stButton"] > button[kind="primary"]::before {{
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
    transition: left 0.5s ease !important;
}}

div[data-testid="stButton"] > button[kind="primary"]:hover {{
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 0 0 40px rgba(59, 130, 246, 0.7),
                0 0 80px rgba(59, 130, 246, 0.4),
                0 20px 40px rgba(0, 0, 0, 0.4) !important;
    border-color: rgba(147, 197, 253, 0.8) !important;
}}

div[data-testid="stButton"] > button[kind="primary"]:active {{
    transform: translateY(-2px) scale(1.01) !important;
}}

/* ========================================
   TEAM IMAGE STRIP - ENHANCED HOVER
   ======================================== */

.team-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    margin: 3rem 0;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid rgba(34, 211, 238, 0.2);
    box-shadow: 0 0 50px rgba(6, 78, 145, 0.25),
                0 25px 50px rgba(0, 0, 0, 0.3);
}}

.team-strip-item {{
    position: relative;
    height: 200px;
    background-size: cover;
    background-position: center;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}}

.team-strip-item::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg,
        rgba(2, 6, 23, 0.2) 0%,
        rgba(2, 6, 23, 0.5) 50%,
        rgba(2, 6, 23, 0.85) 100%);
    transition: all 0.5s ease;
}}

.team-strip-item:hover {{
    transform: scale(1.03);
    z-index: 10;
}}

.team-strip-item:hover::before {{
    background: linear-gradient(180deg,
        rgba(59, 130, 246, 0.3) 0%,
        rgba(34, 211, 238, 0.2) 50%,
        rgba(2, 6, 23, 0.7) 100%);
}}

.team-strip-item::after {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    pointer-events: none;
}}

.team-strip-item:hover::after {{
    border-color: rgba(34, 211, 238, 0.5);
    box-shadow: inset 0 0 30px rgba(34, 211, 238, 0.2);
}}

.team-strip-label {{
    position: absolute;
    bottom: 1.25rem;
    left: 1.25rem;
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    z-index: 1;
    text-shadow: 0 2px 15px rgba(0, 0, 0, 0.8);
    transition: all 0.3s ease;
}}

.team-strip-item:hover .team-strip-label {{
    color: #22d3ee;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.5),
                 0 2px 15px rgba(0, 0, 0, 0.8);
    transform: translateX(5px);
}}

/* ========================================
   WORKFLOW STEPS SECTION
   ======================================== */

.workflow-section {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.6) 0%, rgba(2, 6, 23, 0.8) 100%);
    border: 1px solid rgba(34, 211, 238, 0.15);
    border-radius: 24px;
    padding: 3rem 2rem;
    margin: 3rem 0;
}}

.workflow-title {{
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 2.5rem;
}}

.workflow-steps {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
}}

.workflow-step {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem 2rem;
    transition: all 0.3s ease;
}}

.workflow-step:hover {{
    transform: translateY(-5px);
}}

.workflow-icon {{
    width: 70px;
    height: 70px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}}

.workflow-step:hover .workflow-icon {{
    transform: scale(1.1);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
}}

.workflow-icon-upload {{ background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%); }}
.workflow-icon-explore {{ background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(34, 211, 238, 0.1) 100%); }}
.workflow-icon-train {{ background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%); }}
.workflow-icon-forecast {{ background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%); }}
.workflow-icon-optimize {{ background: linear-gradient(135deg, rgba(251, 146, 60, 0.2) 0%, rgba(251, 146, 60, 0.1) 100%); }}

.workflow-label {{
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
    text-align: center;
}}

.workflow-sublabel {{
    color: #64748b;
    font-size: 0.75rem;
    text-align: center;
    margin-top: 0.25rem;
    max-width: 120px;
}}

.workflow-arrow {{
    color: rgba(34, 211, 238, 0.4);
    font-size: 1.5rem;
    margin: 0 0.5rem;
}}

/* ========================================
   PLATFORM CAPABILITIES SECTION
   ======================================== */

.capabilities-header {{
    text-align: center;
    margin: 4rem 0 2.5rem 0;
}}

.capabilities-badge {{
    color: #22d3ee;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-bottom: 1rem;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
}}

.capabilities-title {{
    font-size: 2.75rem;
    font-weight: 800;
    color: white;
    margin: 0.75rem 0;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}

.capabilities-subtitle {{
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.7;
}}

/* ========================================
   FEATURE CARDS - ENHANCED
   ======================================== */

.feature-card-enhanced {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(6, 78, 145, 0.15) 100%);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 22px;
    padding: 2rem;
    height: 100%;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.feature-card-enhanced:hover {{
    transform: translateY(-8px);
    border-color: rgba(34, 211, 238, 0.5);
    box-shadow: 0 25px 50px rgba(6, 78, 145, 0.35),
                0 0 40px rgba(34, 211, 238, 0.1);
}}

.feature-card-enhanced::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.08), transparent);
    transition: left 0.6s ease;
}}

.feature-card-enhanced:hover::before {{
    left: 100%;
}}

.feature-icon-box {{
    width: 60px;
    height: 60px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}}

.feature-card-enhanced:hover .feature-icon-box {{
    transform: scale(1.1) rotate(5deg);
}}

.feature-icon-cyan {{ background: linear-gradient(135deg, rgba(34, 211, 238, 0.25) 0%, rgba(34, 211, 238, 0.1) 100%); border: 1px solid rgba(34, 211, 238, 0.35); }}
.feature-icon-blue {{ background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0.1) 100%); border: 1px solid rgba(59, 130, 246, 0.35); }}
.feature-icon-purple {{ background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(139, 92, 246, 0.1) 100%); border: 1px solid rgba(139, 92, 246, 0.35); }}
.feature-icon-green {{ background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(34, 197, 94, 0.1) 100%); border: 1px solid rgba(34, 197, 94, 0.35); }}
.feature-icon-orange {{ background: linear-gradient(135deg, rgba(251, 146, 60, 0.25) 0%, rgba(251, 146, 60, 0.1) 100%); border: 1px solid rgba(251, 146, 60, 0.35); }}
.feature-icon-pink {{ background: linear-gradient(135deg, rgba(244, 63, 94, 0.25) 0%, rgba(244, 63, 94, 0.1) 100%); border: 1px solid rgba(244, 63, 94, 0.35); }}

.feature-badge {{
    position: absolute;
    top: 1.25rem;
    right: 1.25rem;
    background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 100%);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    box-shadow: 0 4px 15px rgba(34, 211, 238, 0.3);
}}

.feature-title-enhanced {{
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.85rem;
}}

.feature-description-enhanced {{
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.7;
}}

/* ========================================
   CTA SECTION - ENHANCED
   ======================================== */

.cta-section {{
    position: relative;
    background: linear-gradient(135deg,
        rgba(2, 6, 23, 0.9) 0%,
        rgba(6, 78, 145, 0.5) 50%,
        rgba(2, 6, 23, 0.95) 100%),
        url('{login_bg}');
    background-size: cover;
    background-position: center;
    border-radius: 28px;
    padding: 5rem 2rem;
    text-align: center;
    margin: 4rem 0;
    border: 1px solid rgba(34, 211, 238, 0.2);
    overflow: hidden;
    box-shadow: 0 0 60px rgba(6, 78, 145, 0.3),
                0 25px 50px rgba(0, 0, 0, 0.4);
}}

.cta-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.15) 0%, transparent 60%);
    pointer-events: none;
}}

.cta-title {{
    font-size: 2.75rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1.25rem;
    position: relative;
    z-index: 1;
    text-shadow: 0 4px 30px rgba(0, 0, 0, 0.8);
}}

.cta-subtitle {{
    color: #cbd5e1;
    font-size: 1.15rem;
    max-width: 600px;
    margin: 0 auto 2.5rem auto;
    position: relative;
    z-index: 1;
    line-height: 1.7;
    text-align: center;
    display: block;
    width: 100%;
}}

/* ========================================
   PERFORMANCE METRICS SECTION
   ======================================== */

.performance-section {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(2, 6, 23, 0.9) 100%);
    border: 1px solid rgba(34, 211, 238, 0.15);
    border-radius: 24px;
    padding: 3rem;
    margin: 3rem 0;
    display: flex;
    gap: 3rem;
    flex-wrap: wrap;
}}

.performance-content {{
    flex: 1;
    min-width: 300px;
}}

.performance-badge {{
    color: #22d3ee;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
}}

.performance-title {{
    font-size: 2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1.5rem;
    line-height: 1.3;
}}

.performance-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}

.performance-list li {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 0;
    color: #cbd5e1;
    font-size: 1rem;
}}

.performance-list-icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}}

.performance-list-icon.red {{ background: rgba(244, 63, 94, 0.2); }}
.performance-list-icon.blue {{ background: rgba(59, 130, 246, 0.2); }}
.performance-list-icon.green {{ background: rgba(34, 197, 94, 0.2); }}

.performance-metrics {{
    flex: 1;
    min-width: 300px;
}}

.metrics-card {{
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(34, 211, 238, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
}}

.metrics-title {{
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.5rem;
}}

.metric-row {{
    margin-bottom: 1.25rem;
}}

.metric-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}}

.metric-name {{
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 500;
}}

.metric-value {{
    color: #22d3ee;
    font-size: 0.9rem;
    font-weight: 700;
}}

.metric-bar {{
    height: 8px;
    background: rgba(30, 41, 59, 0.8);
    border-radius: 4px;
    overflow: hidden;
}}

.metric-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease;
}}

.metric-fill.cyan {{ background: linear-gradient(90deg, #22d3ee, #06b6d4); }}
.metric-fill.blue {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
.metric-fill.green {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
.metric-fill.red {{ background: linear-gradient(90deg, #f43f5e, #dc2626); }}

/* ========================================
   MOBILE RESPONSIVE
   ======================================== */

@media (max-width: 768px) {{
    .hero-section {{
        min-height: 420px;
        padding: 2.5rem 1.5rem;
        border-radius: 20px;
    }}
    .hero-title {{
        font-size: 2.25rem;
    }}
    .hero-title-accent {{
        font-size: 2.5rem;
    }}
    .hero-subtitle {{
        font-size: 1rem;
    }}
    .hero-stats {{
        gap: 1rem;
    }}
    .hero-stat {{
        padding: 1rem 1.25rem;
        min-width: 120px;
    }}
    .hero-stat-value {{
        font-size: 1.75rem;
    }}
    .team-strip {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .team-strip-item {{
        height: 140px;
    }}
    .workflow-steps {{
        gap: 0.5rem;
    }}
    .workflow-step {{
        padding: 0.75rem 1rem;
    }}
    .workflow-icon {{
        width: 55px;
        height: 55px;
        font-size: 1.4rem;
    }}
    .workflow-arrow {{
        display: none;
    }}
    .capabilities-title {{
        font-size: 2rem;
    }}
    .cta-title {{
        font-size: 2rem;
    }}
    .performance-section {{
        flex-direction: column;
    }}
}}

@media (max-width: 480px) {{
    .hero-section {{
        min-height: 380px;
        padding: 2rem 1rem;
        border-radius: 16px;
    }}
    .hero-title {{
        font-size: 1.75rem;
    }}
    .hero-title-accent {{
        font-size: 2rem;
    }}
    .hero-badge {{
        font-size: 0.65rem;
        padding: 0.4rem 1rem;
        letter-spacing: 2px;
    }}
    .hero-stats {{
        gap: 0.75rem;
    }}
    .hero-stat {{
        padding: 0.75rem 1rem;
        min-width: 100px;
    }}
    .hero-stat-value {{
        font-size: 1.4rem;
    }}
    .hero-stat-label {{
        font-size: 0.7rem;
    }}
    .team-strip {{
        grid-template-columns: 1fr 1fr;
        border-radius: 16px;
    }}
    .team-strip-item {{
        height: 110px;
    }}
    .workflow-section {{
        padding: 2rem 1rem;
    }}
    .workflow-steps {{
        flex-direction: column;
        gap: 0;
    }}
    .workflow-step {{
        padding: 1rem;
    }}
    .capabilities-title {{
        font-size: 1.5rem;
    }}
    .cta-section {{
        padding: 3rem 1.5rem;
        border-radius: 20px;
    }}
    .cta-title {{
        font-size: 1.5rem;
    }}
}}
</style>

<!-- HERO SECTION -->
<div class="hero-section">
    <div class="hero-badge">Intelligent Hospital Resource Planning</div>
    <h1 class="hero-title">
        Smarter Hospitals,
        <span class="hero-title-accent">Powered by AI</span>
    </h1>
    <p class="hero-subtitle">
        Forecast patient demand, optimize staff schedules, and manage supplies with confidence.
        HealthForecast AI transforms hospital data into actionable insights that save time,
        reduce costs, and improve patient care.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">7-Day</div>
            <div class="hero-stat-label">Forecast Horizon</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">24/7</div>
            <div class="hero-stat-label">Real-time Monitoring</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">&lt; 5%</div>
            <div class="hero-stat-label">Prediction Error</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">30%</div>
            <div class="hero-stat-label">Cost Reduction</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# START HERE BUTTON
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("START HERE", type="primary", use_container_width=True, key="start_here_btn"):
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
        st.switch_page("pages/01_Dashboard.py")

# ============================================================================
# WORKFLOW STEPS SECTION
# ============================================================================
st.markdown(f"""
<div class="workflow-section">
    <div class="workflow-title">How It Works - Generate Forecasts in Minutes</div>
    <div class="workflow-steps">
        <div class="workflow-step">
            <div class="workflow-icon workflow-icon-upload">📤</div>
            <div class="workflow-label">Upload</div>
            <div class="workflow-sublabel">Import historical patient, weather, and calendar data</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon workflow-icon-explore">📊</div>
            <div class="workflow-label">Explore</div>
            <div class="workflow-sublabel">Visualize patterns, seasonality, and correlations</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon workflow-icon-train">🧠</div>
            <div class="workflow-label">Train</div>
            <div class="workflow-sublabel">Build and compare forecasting models automatically</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon workflow-icon-forecast">📈</div>
            <div class="workflow-label">Forecast</div>
            <div class="workflow-sublabel">Generate 7-day predictions with confidence intervals</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon workflow-icon-optimize">⚡</div>
            <div class="workflow-label">Optimize</div>
            <div class="workflow-sublabel">Create optimal staff schedules and supply orders</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TEAM IMAGE STRIP
# ============================================================================
st.markdown(f"""
<div class="team-strip">
    <div class="team-strip-item" style="background-image: url('{staff_bg}');">
        <div class="team-strip-label">Our Team</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{carousel_1}');">
        <div class="team-strip-label">Analytics</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{dashboard_bg}');">
        <div class="team-strip-label">Supply Chain</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{team_bg1}');">
        <div class="team-strip-label">Collaboration</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PLATFORM CAPABILITIES SECTION
# ============================================================================
st.markdown("""
<div class="capabilities-header">
    <div class="capabilities-badge">Platform Capabilities</div>
    <h2 class="capabilities-title">Everything You Need to Manage Hospital Demand</h2>
    <p class="capabilities-subtitle">
        From data ingestion to actionable recommendations - a complete end-to-end platform for hospital resource planning.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature Cards Row 1
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-badge">Core</div>
        <div class="feature-icon-box feature-icon-cyan">📈</div>
        <h3 class="feature-title-enhanced">Demand Forecasting</h3>
        <p class="feature-description-enhanced">
            Predict patient arrivals up to 7 days ahead. Understand trends, seasonality,
            and patterns in emergency department visits.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-badge">Core</div>
        <div class="feature-icon-box feature-icon-blue">👥</div>
        <h3 class="feature-title-enhanced">Staff Optimization</h3>
        <p class="feature-description-enhanced">
            Generate optimal staff schedules that balance coverage, costs, and preferences.
            Minimize overtime while ensuring patient safety.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-badge">Core</div>
        <div class="feature-icon-box feature-icon-purple">📦</div>
        <h3 class="feature-title-enhanced">Supply Management</h3>
        <p class="feature-description-enhanced">
            Optimize inventory levels, reduce waste, and prevent stockouts.
            Smart reorder alerts keep you prepared.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Feature Cards Row 2
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-icon-box feature-icon-green">📊</div>
        <h3 class="feature-title-enhanced">Data Exploration</h3>
        <p class="feature-description-enhanced">
            Visualize distributions, correlations, and temporal patterns.
            Understand what drives demand before making decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-icon-box feature-icon-orange">💡</div>
        <h3 class="feature-title-enhanced">AI-Powered Insights</h3>
        <p class="feature-description-enhanced">
            Receive intelligent recommendations tailored to your hospital.
            Actionable advice prioritized by impact.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card-enhanced">
        <div class="feature-icon-box feature-icon-pink">📱</div>
        <h3 class="feature-title-enhanced">Real-time Dashboard</h3>
        <p class="feature-description-enhanced">
            Monitor KPIs at a glance. Track forecasts, model performance,
            staff coverage, and supply levels.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PERFORMANCE METRICS SECTION
# ============================================================================
st.markdown("""
<div class="performance-section">
    <div class="performance-content">
        <div class="performance-badge">Why HealthForecast AI</div>
        <h3 class="performance-title">Built for Hospital Decision Makers</h3>
        <ul class="performance-list">
            <li>
                <span class="performance-list-icon red">❤️</span>
                <span><strong>Improve Patient Care</strong> - Ensure the right staff and supplies are available when patients need them most.</span>
            </li>
            <li>
                <span class="performance-list-icon blue">💰</span>
                <span><strong>Reduce Operational Costs</strong> - Minimize overtime, prevent overstocking, and eliminate inefficiencies in resource allocation.</span>
            </li>
            <li>
                <span class="performance-list-icon green">📊</span>
                <span><strong>Data-Driven Confidence</strong> - Every forecast includes prediction intervals so you can plan for best and worst case scenarios.</span>
            </li>
        </ul>
    </div>
    <div class="performance-metrics">
        <div class="metrics-card">
            <div class="metrics-title">Platform Performance</div>
            <div class="metric-row">
                <div class="metric-header">
                    <span class="metric-name">Forecast Accuracy</span>
                    <span class="metric-value">96%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill cyan" style="width: 96%;"></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-header">
                    <span class="metric-name">Staff Coverage</span>
                    <span class="metric-value">94%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill blue" style="width: 94%;"></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-header">
                    <span class="metric-name">Supply Service Level</span>
                    <span class="metric-value">98%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill green" style="width: 98%;"></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-header">
                    <span class="metric-name">Cost Efficiency</span>
                    <span class="metric-value">87%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill red" style="width: 87%;"></div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CTA SECTION WITH BACKGROUND
# ============================================================================
st.markdown(f"""
<div class="cta-section">
    <h2 class="cta-title">Ready to Transform Your Hospital Operations?</h2>
    <p class="cta-subtitle">
        Start forecasting patient demand, optimizing schedules, and making
        data-driven decisions today.
    </p>
</div>
""", unsafe_allow_html=True)

# Second CTA Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("START HERE", type="primary", use_container_width=True, key="start_here_btn_2"):
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
        st.switch_page("pages/01_Dashboard.py")

# ============================================================================
# PIPELINE STATUS DASHBOARD (for authenticated users)
# ============================================================================
if st.session_state.get("authenticated", False):
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    render_pipeline_status_dashboard()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid rgba(6, 78, 145, 0.3);'>
        <p style='margin: 0; font-size: 0.9rem;'>
            <strong style='color: #22d3ee; text-shadow: 0 0 10px rgba(34, 211, 238, 0.3);'>HealthForecast AI</strong>
            <span style='color: #475569;'> · </span>
            <span style='color: #94a3b8;'>Enterprise Healthcare Intelligence Platform</span>
        </p>
        <p style='margin: 0.75rem 0 0 0; font-size: 0.8rem; color: #64748b;'>
            Powered by Advanced Machine Learning · Built for Healthcare Excellence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
