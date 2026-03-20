from __future__ import annotations
import streamlit as st
import base64
from pathlib import Path
from app_core.ui.theme import apply_css, hero_card, feature_card
from app_core.ui.effects import inject_fluorescent_effects
from app_core.ui.components import render_scifi_hero_header
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

# Render fluorescent effects
inject_fluorescent_effects()


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
IMAGES_DIR = Path(__file__).parent / "images"
hero_bg = get_image_css_url("hero-bg1.jpg")
login_bg = get_image_css_url("login-bg1.jpg")
carousel_1 = get_image_css_url("carousel-1.jpg")
carousel_2 = get_image_css_url("carousel-2.jpg")
carousel_3 = get_image_css_url("carousel-3.jpg")
team_bg1 = get_image_css_url("team-bg1.jpg")
team_bg2 = get_image_css_url("team-bg2.jpg")
team_bg3 = get_image_css_url("team-bg3.jpg")
team_bg4 = get_image_css_url("team-bg4.jpg")


# ============================================================================
# HERO SECTION WITH BACKGROUND IMAGE
# ============================================================================
st.markdown(f"""
<style>
/* ========================================
   HERO SECTION WITH BACKGROUND IMAGE
   ======================================== */

.hero-section {{
    position: relative;
    width: 100%;
    min-height: 500px;
    background: linear-gradient(135deg, rgba(10, 14, 39, 0.85) 0%, rgba(6, 78, 145, 0.6) 50%, rgba(10, 14, 39, 0.9) 100%),
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
    border: 1px solid rgba(34, 211, 238, 0.2);
    box-shadow: 0 0 60px rgba(6, 78, 145, 0.3),
                inset 0 0 120px rgba(0, 0, 0, 0.5);
}}

.hero-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg, transparent 0%, rgba(10, 14, 39, 0.4) 100%);
    pointer-events: none;
}}

.hero-badge {{
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    border: 1px solid rgba(34, 211, 238, 0.4);
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-size: 0.85rem;
    color: #22d3ee;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    z-index: 1;
}}

.hero-title {{
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 0 40px rgba(34, 211, 238, 0.5),
                 0 4px 20px rgba(0, 0, 0, 0.5);
    z-index: 1;
    line-height: 1.2;
}}

.hero-title-accent {{
    background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-subtitle {{
    font-size: 1.25rem;
    color: #94a3b8;
    text-align: center;
    max-width: 700px;
    margin: 1rem auto 2rem auto;
    line-height: 1.7;
    z-index: 1;
}}

.hero-stats {{
    display: flex;
    gap: 3rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 2rem;
    z-index: 1;
}}

.hero-stat {{
    text-align: center;
    padding: 1rem 1.5rem;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    border: 1px solid rgba(34, 211, 238, 0.2);
    backdrop-filter: blur(10px);
}}

.hero-stat-value {{
    font-size: 2rem;
    font-weight: 800;
    color: #22d3ee;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
}}

.hero-stat-label {{
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ========================================
   CTA BUTTONS
   ======================================== */

.hero-buttons {{
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    z-index: 1;
    flex-wrap: wrap;
    justify-content: center;
}}

.hero-btn {{
    padding: 1rem 2.5rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
    text-decoration: none;
    transition: all 0.3s ease;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}}

.hero-btn-primary {{
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: 2px solid rgba(59, 130, 246, 0.5);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.4);
}}

.hero-btn-primary:hover {{
    transform: translateY(-3px);
    box-shadow: 0 0 50px rgba(59, 130, 246, 0.6);
}}

.hero-btn-secondary {{
    background: rgba(15, 23, 42, 0.8);
    color: #22d3ee;
    border: 2px solid rgba(34, 211, 238, 0.4);
}}

.hero-btn-secondary:hover {{
    background: rgba(34, 211, 238, 0.1);
    border-color: rgba(34, 211, 238, 0.8);
}}

/* ========================================
   TEAM IMAGE STRIP
   ======================================== */

.team-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    margin: 3rem 0;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(34, 211, 238, 0.2);
    box-shadow: 0 0 40px rgba(6, 78, 145, 0.2);
}}

.team-strip-item {{
    position: relative;
    height: 180px;
    background-size: cover;
    background-position: center;
    overflow: hidden;
}}

.team-strip-item::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg, rgba(10, 14, 39, 0.3) 0%, rgba(10, 14, 39, 0.7) 100%);
    transition: all 0.3s ease;
}}

.team-strip-item:hover::before {{
    background: linear-gradient(180deg, rgba(59, 130, 246, 0.2) 0%, rgba(10, 14, 39, 0.5) 100%);
}}

.team-strip-label {{
    position: absolute;
    bottom: 1rem;
    left: 1rem;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    z-index: 1;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}}

/* ========================================
   PLATFORM CAPABILITIES SECTION
   ======================================== */

.capabilities-header {{
    text-align: center;
    margin: 4rem 0 2rem 0;
}}

.capabilities-badge {{
    color: #22d3ee;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
}}

.capabilities-title {{
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin: 0.5rem 0;
}}

.capabilities-subtitle {{
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}}

/* ========================================
   FEATURE CARDS WITH IMAGES
   ======================================== */

.feature-card-enhanced {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(6, 78, 145, 0.1) 100%);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 20px;
    padding: 2rem;
    height: 100%;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.feature-card-enhanced:hover {{
    transform: translateY(-5px);
    border-color: rgba(34, 211, 238, 0.5);
    box-shadow: 0 20px 40px rgba(6, 78, 145, 0.3);
}}

.feature-card-enhanced::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.05), transparent);
    transition: left 0.5s ease;
}}

.feature-card-enhanced:hover::before {{
    left: 100%;
}}

.feature-icon-box {{
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1.25rem;
}}

.feature-icon-cyan {{
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(34, 211, 238, 0.1) 100%);
    border: 1px solid rgba(34, 211, 238, 0.3);
}}

.feature-icon-blue {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
}}

.feature-icon-purple {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
}}

.feature-icon-green {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
    border: 1px solid rgba(34, 197, 94, 0.3);
}}

.feature-icon-orange {{
    background: linear-gradient(135deg, rgba(251, 146, 60, 0.2) 0%, rgba(251, 146, 60, 0.1) 100%);
    border: 1px solid rgba(251, 146, 60, 0.3);
}}

.feature-icon-pink {{
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(244, 63, 94, 0.1) 100%);
    border: 1px solid rgba(244, 63, 94, 0.3);
}}

.feature-badge {{
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 100%);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.feature-title-enhanced {{
    font-size: 1.25rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.75rem;
}}

.feature-description-enhanced {{
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
}}

/* ========================================
   CTA SECTION WITH BACKGROUND
   ======================================== */

.cta-section {{
    position: relative;
    background: linear-gradient(135deg, rgba(6, 78, 145, 0.3) 0%, rgba(10, 14, 39, 0.95) 100%),
                url('{login_bg}');
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 4rem 2rem;
    text-align: center;
    margin: 4rem 0;
    border: 1px solid rgba(34, 211, 238, 0.2);
    overflow: hidden;
}}

.cta-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
    pointer-events: none;
}}

.cta-title {{
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}}

.cta-subtitle {{
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto 2rem auto;
    position: relative;
    z-index: 1;
}}

/* ========================================
   PERFORMANCE METRICS
   ======================================== */

.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 3rem 0;
}}

.metric-card {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(6, 78, 145, 0.1) 100%);
    border: 1px solid rgba(34, 211, 238, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}}

.metric-value {{
    font-size: 2.5rem;
    font-weight: 800;
    color: #22d3ee;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.4);
}}

.metric-label {{
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ========================================
   MOBILE RESPONSIVE
   ======================================== */

@media (max-width: 768px) {{
    .hero-section {{
        min-height: 400px;
        padding: 2rem 1rem;
    }}
    .hero-title {{
        font-size: 2rem;
    }}
    .hero-subtitle {{
        font-size: 1rem;
    }}
    .hero-stats {{
        gap: 1rem;
    }}
    .hero-stat {{
        padding: 0.75rem 1rem;
    }}
    .hero-stat-value {{
        font-size: 1.5rem;
    }}
    .team-strip {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .team-strip-item {{
        height: 120px;
    }}
    .capabilities-title {{
        font-size: 1.75rem;
    }}
    .metrics-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .cta-title {{
        font-size: 1.75rem;
    }}
}}

@media (max-width: 480px) {{
    .hero-section {{
        min-height: 350px;
        padding: 1.5rem 1rem;
        border-radius: 16px;
    }}
    .hero-title {{
        font-size: 1.5rem;
    }}
    .hero-badge {{
        font-size: 0.7rem;
        padding: 0.35rem 1rem;
    }}
    .team-strip {{
        grid-template-columns: 1fr 1fr;
    }}
    .team-strip-item {{
        height: 100px;
    }}
    .metrics-grid {{
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
    }}
    .metric-card {{
        padding: 1rem;
    }}
    .metric-value {{
        font-size: 1.75rem;
    }}
}}

/* ========================================
   LOGIN BUTTON STYLING (existing)
   ======================================== */

div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: 2px solid rgba(59, 130, 246, 0.5) !important;
    border-radius: 14px !important;
    padding: 1.25rem 3rem !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: white !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.4) !important;
    transition: all 0.3s ease !important;
    min-height: 60px !important;
}}

div[data-testid="stButton"] > button[kind="primary"]:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 0 50px rgba(59, 130, 246, 0.6) !important;
    border-color: rgba(59, 130, 246, 0.8) !important;
}}

/* Feature cards hover */
.hf-feature-card:hover {{
    filter: brightness(1.1);
}}
</style>

<!-- HERO SECTION -->
<div class="hero-section">
    <div class="hero-badge">Intelligent Hospital Resource Planning</div>
    <h1 class="hero-title">Smarter Hospitals,<br><span class="hero-title-accent">Powered by AI</span></h1>
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
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("START HERE", type="primary", use_container_width=True, key="start_here_btn"):
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
        st.switch_page("pages/01_Dashboard.py")

# ============================================================================
# TEAM IMAGE STRIP
# ============================================================================
st.markdown(f"""
<div class="team-strip">
    <div class="team-strip-item" style="background-image: url('{team_bg1}');">
        <div class="team-strip-label">Our Team</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{carousel_1}');">
        <div class="team-strip-label">Analytics</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{team_bg2}');">
        <div class="team-strip-label">Supply Chain</div>
    </div>
    <div class="team-strip-item" style="background-image: url('{carousel_2}');">
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
        <div class="feature-icon-box feature-icon-cyan">
            <span>📈</span>
        </div>
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
        <div class="feature-icon-box feature-icon-blue">
            <span>👥</span>
        </div>
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
        <div class="feature-icon-box feature-icon-purple">
            <span>📦</span>
        </div>
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
        <div class="feature-icon-box feature-icon-green">
            <span>📊</span>
        </div>
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
        <div class="feature-icon-box feature-icon-orange">
            <span>💡</span>
        </div>
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
        <div class="feature-icon-box feature-icon-pink">
            <span>📱</span>
        </div>
        <h3 class="feature-title-enhanced">Real-time Dashboard</h3>
        <p class="feature-description-enhanced">
            Monitor KPIs at a glance. Track forecasts, model performance,
            staff coverage, and supply levels.
        </p>
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
            <span style='color: #475569;'>·</span>
            <span style='color: #94a3b8;'>Enterprise Healthcare Intelligence Platform</span>
        </p>
        <p style='margin: 0.75rem 0 0 0; font-size: 0.8rem; color: #64748b;'>
            Powered by Advanced Machine Learning · Built for Healthcare Excellence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
