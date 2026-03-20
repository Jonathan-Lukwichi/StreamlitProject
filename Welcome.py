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

# Apply Healthcare CSS Theme
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
dashboard_bg = get_image_css_url("dashboard-bg.jpg")
staff_bg = get_image_css_url("staff-bg.jpg")
team_bg1 = get_image_css_url("team-bg1.jpg")


# ============================================================================
# HEALTHCARE LIGHT THEME CSS
# ============================================================================
st.markdown(f"""
<style>
/* ========================================
   HEALTHCARE COLOR VARIABLES
   ======================================== */

:root {{
    --hc-blue-primary: #0284c7;
    --hc-blue-secondary: #0ea5e9;
    --hc-blue-light: #38bdf8;
    --hc-blue-pale: #e0f2fe;
    --hc-red-primary: #dc2626;
    --hc-red-hover: #b91c1c;
    --hc-green: #16a34a;
    --hc-text-dark: #0f172a;
    --hc-text-body: #334155;
    --hc-text-muted: #64748b;
    --hc-bg-white: #ffffff;
    --hc-bg-light: #f8fafc;
    --hc-bg-subtle: #f1f5f9;
    --hc-border: #e2e8f0;
}}

/* ========================================
   HERO SECTION - LIGHT WITH IMAGE
   ======================================== */

.hero-section {{
    position: relative;
    width: 100%;
    min-height: 520px;
    background: linear-gradient(135deg,
        rgba(255, 255, 255, 0.92) 0%,
        rgba(224, 242, 254, 0.85) 50%,
        rgba(255, 255, 255, 0.9) 100%),
        url('{hero_bg}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 4rem 2rem;
    margin-bottom: 2rem;
    overflow: hidden;
    border: 1px solid rgba(14, 165, 233, 0.2);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                0 2px 4px -2px rgba(0, 0, 0, 0.1),
                0 0 0 1px rgba(14, 165, 233, 0.05);
}}

.hero-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #0284c7, #0ea5e9, #38bdf8, #dc2626);
}}

.hero-badge {{
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    border: none;
    border-radius: 50px;
    padding: 0.6rem 1.75rem;
    font-size: 0.8rem;
    color: white;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-bottom: 1.75rem;
    z-index: 1;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
}}

.hero-title {{
    font-size: 3.5rem;
    font-weight: 900;
    color: #0f172a;
    text-align: center;
    margin: 0 0 0.5rem 0;
    z-index: 1;
    line-height: 1.15;
    letter-spacing: -1px;
}}

.hero-title-accent {{
    display: block;
    font-size: 3.75rem;
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-subtitle {{
    font-size: 1.2rem;
    color: #334155;
    text-align: center;
    max-width: 720px;
    margin: 1.25rem auto 2.5rem auto;
    line-height: 1.8;
    z-index: 1;
}}

.hero-stats {{
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 2rem;
    z-index: 1;
}}

.hero-stat {{
    text-align: center;
    padding: 1.25rem 2rem;
    background: white;
    border-radius: 16px;
    border: 1px solid rgba(14, 165, 233, 0.2);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    min-width: 140px;
}}

.hero-stat:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.15);
    border-color: rgba(14, 165, 233, 0.4);
}}

.hero-stat-value {{
    font-size: 2rem;
    font-weight: 800;
    color: #0284c7;
}}

.hero-stat-label {{
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}

/* ========================================
   CTA BUTTON - MEDICAL RED
   ======================================== */

div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.25rem 3.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    color: white !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4) !important;
    transition: all 0.3s ease !important;
    min-height: 60px !important;
}}

div[data-testid="stButton"] > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(220, 38, 38, 0.5) !important;
}}

/* ========================================
   WORKFLOW STEPS - LIGHT THEME
   ======================================== */

.workflow-section {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin: 2rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}}

.workflow-title {{
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 2rem;
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
    transform: translateY(-4px);
}}

.workflow-icon {{
    width: 64px;
    height: 64px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
    border: 1px solid;
}}

.workflow-step:hover .workflow-icon {{
    transform: scale(1.08);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}}

.workflow-icon-upload {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08));
    border-color: rgba(59, 130, 246, 0.3);
}}
.workflow-icon-explore {{
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(14, 165, 233, 0.08));
    border-color: rgba(14, 165, 233, 0.3);
}}
.workflow-icon-train {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.08));
    border-color: rgba(139, 92, 246, 0.3);
}}
.workflow-icon-forecast {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.08));
    border-color: rgba(34, 197, 94, 0.3);
}}
.workflow-icon-optimize {{
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(220, 38, 38, 0.08));
    border-color: rgba(220, 38, 38, 0.3);
}}

.workflow-label {{
    color: #0f172a;
    font-size: 0.9rem;
    font-weight: 700;
    text-align: center;
}}

.workflow-sublabel {{
    color: #64748b;
    font-size: 0.75rem;
    text-align: center;
    margin-top: 0.25rem;
    max-width: 110px;
    line-height: 1.4;
}}

.workflow-arrow {{
    color: #cbd5e1;
    font-size: 1.5rem;
    margin: 0 0.5rem;
}}

/* ========================================
   TEAM IMAGE STRIP - LIGHT OVERLAY
   ======================================== */

.team-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    margin: 2rem 0;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
}}

.team-strip-item {{
    position: relative;
    height: 180px;
    background-size: cover;
    background-position: center;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.4s ease;
}}

.team-strip-item::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg,
        rgba(255, 255, 255, 0.1) 0%,
        rgba(2, 132, 199, 0.4) 100%);
    transition: all 0.4s ease;
}}

.team-strip-item:hover {{
    transform: scale(1.02);
    z-index: 10;
}}

.team-strip-item:hover::before {{
    background: linear-gradient(180deg,
        rgba(14, 165, 233, 0.3) 0%,
        rgba(2, 132, 199, 0.6) 100%);
}}

.team-strip-label {{
    position: absolute;
    bottom: 1rem;
    left: 1rem;
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    z-index: 1;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}}

.team-strip-item:hover .team-strip-label {{
    transform: translateX(5px);
}}

/* ========================================
   CAPABILITIES SECTION
   ======================================== */

.capabilities-header {{
    text-align: center;
    margin: 3rem 0 2rem 0;
}}

.capabilities-badge {{
    color: #0284c7;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-bottom: 0.75rem;
}}

.capabilities-title {{
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.5rem 0;
}}

.capabilities-subtitle {{
    color: #64748b;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.7;
}}

/* ========================================
   FEATURE CARDS - WHITE
   ======================================== */

.feature-card-enhanced {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.75rem;
    height: 100%;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}}

.feature-card-enhanced:hover {{
    transform: translateY(-6px);
    border-color: rgba(14, 165, 233, 0.4);
    box-shadow: 0 12px 30px rgba(14, 165, 233, 0.12);
}}

.feature-card-enhanced::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0284c7, #0ea5e9);
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.feature-card-enhanced:hover::before {{
    opacity: 1;
}}

.feature-icon-box {{
    width: 52px;
    height: 52px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1.25rem;
    transition: all 0.3s ease;
}}

.feature-card-enhanced:hover .feature-icon-box {{
    transform: scale(1.08);
}}

.feature-icon-cyan {{ background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(14, 165, 233, 0.08)); border: 1px solid rgba(14, 165, 233, 0.25); }}
.feature-icon-blue {{ background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08)); border: 1px solid rgba(59, 130, 246, 0.25); }}
.feature-icon-purple {{ background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.08)); border: 1px solid rgba(139, 92, 246, 0.25); }}
.feature-icon-green {{ background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.08)); border: 1px solid rgba(34, 197, 94, 0.25); }}
.feature-icon-orange {{ background: linear-gradient(135deg, rgba(251, 146, 60, 0.15), rgba(251, 146, 60, 0.08)); border: 1px solid rgba(251, 146, 60, 0.25); }}
.feature-icon-pink {{ background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(220, 38, 38, 0.08)); border: 1px solid rgba(220, 38, 38, 0.25); }}

.feature-badge {{
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: white;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.feature-title-enhanced {{
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.65rem;
}}

.feature-description-enhanced {{
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.65;
}}

/* ========================================
   PERFORMANCE SECTION - WHITE CARDS
   ======================================== */

.performance-section {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    display: flex;
    gap: 3rem;
    flex-wrap: wrap;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
}}

.performance-content {{
    flex: 1;
    min-width: 280px;
}}

.performance-badge {{
    color: #0284c7;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.75rem;
}}

.performance-title {{
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
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
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 0;
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.5;
}}

.performance-list-icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}}

.performance-list-icon.red {{ background: rgba(220, 38, 38, 0.12); }}
.performance-list-icon.blue {{ background: rgba(14, 165, 233, 0.12); }}
.performance-list-icon.green {{ background: rgba(34, 197, 94, 0.12); }}

.performance-metrics {{
    flex: 1;
    min-width: 280px;
}}

.metrics-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
}}

.metrics-title {{
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.25rem;
}}

.metric-row {{
    margin-bottom: 1rem;
}}

.metric-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}}

.metric-name {{
    color: #334155;
    font-size: 0.9rem;
    font-weight: 500;
}}

.metric-value {{
    color: #0284c7;
    font-size: 0.9rem;
    font-weight: 700;
}}

.metric-bar {{
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}}

.metric-fill {{
    height: 100%;
    border-radius: 4px;
}}

.metric-fill.cyan {{ background: linear-gradient(90deg, #0ea5e9, #38bdf8); }}
.metric-fill.blue {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
.metric-fill.green {{ background: linear-gradient(90deg, #22c55e, #4ade80); }}
.metric-fill.red {{ background: linear-gradient(90deg, #dc2626, #f87171); }}

/* ========================================
   CTA SECTION - LIGHT WITH IMAGE
   ======================================== */

.cta-section {{
    position: relative;
    background: linear-gradient(135deg,
        rgba(255, 255, 255, 0.9) 0%,
        rgba(224, 242, 254, 0.85) 50%,
        rgba(255, 255, 255, 0.88) 100%),
        url('{login_bg}');
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 4rem 2rem;
    text-align: center;
    margin: 3rem 0;
    border: 1px solid rgba(14, 165, 233, 0.2);
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}}

.cta-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #dc2626, #ef4444, #0284c7, #0ea5e9);
}}

.cta-title {{
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}}

.cta-subtitle {{
    color: #334155;
    font-size: 1.1rem;
    max-width: 550px;
    margin: 0 auto 2rem auto;
    position: relative;
    z-index: 1;
    line-height: 1.7;
    text-align: center !important;
    display: block;
    width: 100%;
    left: 0;
    right: 0;
}}

/* ========================================
   MOBILE RESPONSIVE
   ======================================== */

@media (max-width: 768px) {{
    .hero-section {{
        min-height: 400px;
        padding: 2.5rem 1.5rem;
        border-radius: 18px;
    }}
    .hero-title {{
        font-size: 2rem;
    }}
    .hero-title-accent {{
        font-size: 2.25rem;
    }}
    .hero-stats {{
        gap: 1rem;
    }}
    .hero-stat {{
        padding: 1rem 1.25rem;
        min-width: 110px;
    }}
    .hero-stat-value {{
        font-size: 1.6rem;
    }}
    .team-strip {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .team-strip-item {{
        height: 130px;
    }}
    .workflow-arrow {{
        display: none;
    }}
    .workflow-steps {{
        gap: 0.5rem;
    }}
    .capabilities-title {{
        font-size: 1.75rem;
    }}
    .cta-title {{
        font-size: 1.75rem;
    }}
    .performance-section {{
        flex-direction: column;
    }}
}}

@media (max-width: 480px) {{
    .hero-section {{
        min-height: 360px;
        padding: 2rem 1rem;
        border-radius: 14px;
    }}
    .hero-title {{
        font-size: 1.6rem;
    }}
    .hero-title-accent {{
        font-size: 1.8rem;
    }}
    .hero-badge {{
        font-size: 0.65rem;
        padding: 0.4rem 1rem;
    }}
    .hero-stats {{
        gap: 0.75rem;
    }}
    .hero-stat {{
        padding: 0.85rem 1rem;
        min-width: 95px;
    }}
    .hero-stat-value {{
        font-size: 1.3rem;
    }}
    .hero-stat-label {{
        font-size: 0.65rem;
    }}
    .team-strip-item {{
        height: 100px;
    }}
    .workflow-section {{
        padding: 1.5rem 1rem;
    }}
    .workflow-steps {{
        flex-direction: column;
    }}
    .capabilities-title {{
        font-size: 1.4rem;
    }}
    .cta-section {{
        padding: 2.5rem 1.25rem;
    }}
    .cta-title {{
        font-size: 1.4rem;
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
    <div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid #e2e8f0;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            <strong style='color: #0284c7;'>HealthForecast AI</strong>
            <span style='color: #cbd5e1;'> · </span>
            <span style='color: #64748b;'>Enterprise Healthcare Intelligence Platform</span>
        </p>
        <p style='margin: 0.75rem 0 0 0; font-size: 0.8rem; color: #94a3b8;'>
            Powered by Advanced Machine Learning · Built for Healthcare Excellence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
