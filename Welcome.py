from __future__ import annotations
import streamlit as st
from app_core.ui.theme import apply_css, hero_card, feature_card
from app_core.auth.authentication import initialize_session_state
from app_core.auth.navigation import configure_sidebar_navigation

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="HealthForecast AI - Login",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",  # Hide sidebar until login
)

# Initialize session state
initialize_session_state()

# Configure sidebar navigation
configure_sidebar_navigation()

# Apply Premium CSS Theme
apply_css()

# Warning banner removed per user request

# ============================================================================
# FLUORESCENT EFFECTS - Subtle Premium Visual Enhancements
# ============================================================================
st.markdown("""
<style>
/* ========================================
   ANIMATED FLUORESCENT ORBS (REDUCED)
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

/* Floating Orbs - Subtle */
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
    background: radial-gradient(circle, rgba(251, 191, 36, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}

.orb-2 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.25), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}

.orb-3 {
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.22), transparent 70%);
    top: 50%;
    right: 10%;
    animation: float-orb 28s ease-in-out infinite;
    animation-delay: 10s;
}

.orb-4 {
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.2), transparent 70%);
    bottom: 30%;
    right: 40%;
    animation: float-orb 32s ease-in-out infinite;
    animation-delay: 15s;
}

/* ========================================
   SUBTLE GLOW ANIMATIONS
   ======================================== */

@keyframes subtle-text-glow {
    0%, 100% {
        text-shadow:
            0 0 15px rgba(59, 130, 246, 0.3),
            0 0 30px rgba(34, 211, 238, 0.2),
            0 0 25px rgba(251, 191, 36, 0.15);
    }
    50% {
        text-shadow:
            0 0 20px rgba(59, 130, 246, 0.4),
            0 0 40px rgba(34, 211, 238, 0.3),
            0 0 35px rgba(251, 191, 36, 0.25);
    }
}

@keyframes subtle-border-glow {
    0%, 100% {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15), 0 0 10px rgba(34, 211, 238, 0.1);
    }
    33% {
        border-color: rgba(34, 211, 238, 0.35);
        box-shadow: 0 0 18px rgba(34, 211, 238, 0.2), 0 0 12px rgba(251, 191, 36, 0.1);
    }
    66% {
        border-color: rgba(251, 191, 36, 0.3);
        box-shadow: 0 0 16px rgba(251, 191, 36, 0.18), 0 0 10px rgba(34, 197, 94, 0.1);
    }
}

@keyframes gentle-shimmer {
    0% {
        background-position: -1000px 0;
    }
    100% {
        background-position: 1000px 0;
    }
}

/* Apply to Hero Title - Subtle */
.hf-hero-title {
    animation: subtle-text-glow 4s ease-in-out infinite !important;
}

/* Apply to Pills - Subtle */
.hf-pill {
    animation: subtle-border-glow 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

.hf-pill::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(59, 130, 246, 0.15),
        rgba(34, 211, 238, 0.15),
        rgba(251, 191, 36, 0.12),
        rgba(34, 197, 94, 0.12),
        transparent
    );
    animation: gentle-shimmer 5s infinite;
}

/* Feature Cards - Subtle Hover */
.hf-feature-card:hover {
    filter: brightness(1.1);
}

.hf-feature-card:hover .hf-feature-icon {
    transform: scale(1.05);
}

/* ========================================
   SPARKLES (REDUCED)
   ======================================== */

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
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
}

.sparkle-1 {
    top: 25%;
    left: 35%;
    animation-delay: 0s;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}
.sparkle-2 {
    top: 65%;
    left: 70%;
    animation-delay: 1s;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(251, 191, 36, 0.4));
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.6);
}
.sparkle-3 {
    top: 45%;
    left: 15%;
    animation-delay: 2s;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(34, 197, 94, 0.3));
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
}
.sparkle-4 {
    top: 35%;
    left: 80%;
    animation-delay: 1.5s;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(168, 85, 247, 0.4));
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.6);
}
.sparkle-5 {
    top: 75%;
    left: 45%;
    animation-delay: 2.5s;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(34, 211, 238, 0.4));
    box-shadow: 0 0 8px rgba(34, 211, 238, 0.6);
}

/* ========================================
   LOGIN BUTTON STYLING
   ======================================== */

.login-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.login-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    border-color: rgba(255, 255, 255, 0.4);
}

.login-button h3 {
    margin: 0;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
}

.login-button p {
    margin: 0.5rem 0 0 0;
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.95rem;
}

/* Admin button variant */
.login-button-admin {
    background: linear-gradient(135deg, #f59e0b 0%, #dc2626 100%);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}

.login-button-admin:hover {
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.4);
}

/* ========================================
   RESPONSIVE ADJUSTMENTS
   ======================================== */

@media (max-width: 768px) {
    .fluorescent-orb {
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }

    .sparkle {
        display: none;
    }
}
</style>

<!-- Fluorescent Floating Orbs (Reduced) -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>
<div class="fluorescent-orb orb-3"></div>
<div class="fluorescent-orb orb-4"></div>

<!-- Sparkle Particles (Reduced) -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
<div class="sparkle sparkle-4"></div>
<div class="sparkle sparkle-5"></div>

""", unsafe_allow_html=True)

# ============================================================================
# HERO SECTION - Premium Enterprise Header
# ============================================================================
hero_html = hero_card(
    title="Enterprise Healthcare Intelligence Platform",
    subtitle="Transform hospital operations with AI-powered forecasting. Predict patient arrivals, optimize resources, and deliver superior care through advanced predictive analytics.",
    pills=[
        "🎯 Production-Ready AI",
        "⚡ Real-Time Forecasting",
        "🔬 Deep Learning Models",
        "📊 Multi-Source Intelligence",
        "🏥 Healthcare-Optimized",
        "🚀 Enterprise Scale",
    ],
)
st.markdown(hero_html, unsafe_allow_html=True)

# ============================================================================
# START HERE SECTION - Beautiful Entry Point
# ============================================================================

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# Stunning "Start Here" button with custom CSS
st.markdown("""
<style>
/* ========================================
   START HERE BUTTON - Premium Styling
   ======================================== */

@keyframes pulse-glow {
    0%, 100% {
        box-shadow:
            0 0 20px rgba(59, 130, 246, 0.4),
            0 0 40px rgba(34, 211, 238, 0.3),
            0 0 60px rgba(168, 85, 247, 0.2),
            0 8px 32px rgba(0, 0, 0, 0.3);
    }
    50% {
        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.6),
            0 0 60px rgba(34, 211, 238, 0.4),
            0 0 80px rgba(168, 85, 247, 0.3),
            0 12px 40px rgba(0, 0, 0, 0.4);
    }
}

@keyframes shimmer {
    0% {
        background-position: -200% center;
    }
    100% {
        background-position: 200% center;
    }
}

.start-here-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 3rem 0;
}

.start-here-wrapper {
    text-align: center;
    padding: 2rem;
}

.start-here-title {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
}

.start-here-subtitle {
    color: #94a3b8;
    font-size: 1.15rem;
    margin-bottom: 2rem;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

/* The actual Streamlit button styling */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%) !important;
    background-size: 200% 200% !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.5rem 4rem !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: white !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    animation: pulse-glow 2s ease-in-out infinite !important;
    transition: all 0.3s ease !important;
    min-height: 80px !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-4px) scale(1.02) !important;
    background-position: right center !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(-2px) scale(1.01) !important;
}
</style>

<div class='start-here-wrapper'>
    <h2 class='start-here-title'>Ready to Transform Healthcare?</h2>
    <p class='start-here-subtitle'>
        Experience AI-powered patient forecasting that helps hospitals
        optimize operations and improve patient care.
    </p>
</div>
""", unsafe_allow_html=True)

# Centered Start Here button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("START HERE", type="primary", use_container_width=True, key="start_here_btn"):
        # Set authentication state for demo mode
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
        # Redirect to Dashboard
        st.switch_page("pages/01_Dashboard.py")

# ============================================================================
# FEATURE GRID - Showcase Core Capabilities
# ============================================================================
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='hf-feature-grid'>", unsafe_allow_html=True)

# Feature columns
col1, col2, col3 = st.columns(3)

with col1:
    card1 = feature_card(
        icon="🤖",
        title="Advanced AI Models",
        description="State-of-the-art forecasting architecture combining statistical rigor with deep learning innovation.",
        features=[
            "LSTM Neural Networks for complex temporal patterns",
            "XGBoost gradient boosting with feature importance",
            "SARIMAX statistical modeling with exogenous variables",
            "Hybrid ensemble methods for superior accuracy",
            "Multi-horizon forecasting (1-7 days ahead)",
            "Automated hyperparameter optimization"
        ]
    )
    st.markdown(card1, unsafe_allow_html=True)

with col2:
    card2 = feature_card(
        icon="📡",
        title="Intelligent Data Fusion",
        description="Seamlessly integrate multi-source data streams for comprehensive operational intelligence.",
        features=[
            "Patient arrival data with automated datetime parsing",
            "Weather impact analysis (temperature, precipitation, wind)",
            "Calendar integration (holidays, seasonality, events)",
            "Smart feature engineering with lag detection",
            "Automated missing value imputation",
            "Real-time data validation and quality checks"
        ]
    )
    st.markdown(card2, unsafe_allow_html=True)

with col3:
    card3 = feature_card(
        icon="⚙️",
        title="Operational Excellence",
        description="Deploy enterprise-grade forecasting that integrates seamlessly with hospital workflows.",
        features=[
            "Staff scheduling optimization based on demand",
            "Inventory management with predictive restocking",
            "Decision command center with AI recommendations",
            "Explainable AI with SHAP feature importance",
            "Model performance monitoring and diagnostics",
            "Export-ready predictions with confidence intervals"
        ]
    )
    st.markdown(card3, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# SECONDARY FEATURE CARDS - Additional Capabilities
# ============================================================================
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='hf-feature-grid'>", unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    card4 = feature_card(
        icon="🔍",
        title="Comprehensive Analytics Suite",
        description="Deep exploratory analysis and visualization tools for data-driven insights.",
        features=[
            "Automated EDA with statistical summaries",
            "Time series decomposition and stationarity tests",
            "ACF/PACF analysis for pattern identification",
            "Interactive Plotly visualizations",
            "Day-of-week and seasonal pattern detection",
            "Distribution analysis and outlier detection"
        ]
    )
    st.markdown(card4, unsafe_allow_html=True)

with col5:
    card5 = feature_card(
        icon="🎓",
        title="Research-Grade Framework",
        description="Built on proven scientific methods with full reproducibility and transparency.",
        features=[
            "Time-series aware cross-validation",
            "Expanding and rolling window evaluation",
            "Comprehensive metrics (MAE, RMSE, MAPE, R²)",
            "Residual diagnostics and model validation",
            "Bayesian optimization with Optuna",
            "Artifact versioning and experiment tracking"
        ]
    )
    st.markdown(card5, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# VALUE PROPOSITION SECTION
# ============================================================================
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

st.markdown("""
<div class='hf-feature-card' style='text-align: center;'>
    <h2 class='hf-feature-title' style='font-size: 2rem; margin-bottom: 1.5rem;'>
        Built for Healthcare Leaders
    </h2>
    <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 900px; margin: 0 auto 2rem auto;'>
        HealthForecast AI empowers hospital executives, operations managers, and clinical teams
        to make data-driven decisions with confidence. Our platform combines cutting-edge AI
        with healthcare domain expertise to deliver actionable insights that improve patient
        outcomes and operational efficiency.
    </p>
    <div style='display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; margin-top: 2rem;'>
        <div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #fbbf24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.4);'>8+</div>
            <div style='color: #94a3b8; margin-top: 0.5rem;'>Forecasting Models</div>
        </div>
        <div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #22d3ee; text-shadow: 0 0 20px rgba(34, 211, 238, 0.4);'>3</div>
            <div style='color: #94a3b8; margin-top: 0.5rem;'>Data Sources</div>
        </div>
        <div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #a855f7; text-shadow: 0 0 20px rgba(168, 85, 247, 0.4);'>1-7</div>
            <div style='color: #94a3b8; margin-top: 0.5rem;'>Day Horizons</div>
        </div>
        <div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #22c55e; text-shadow: 0 0 20px rgba(34, 197, 94, 0.4);'>100%</div>
            <div style='color: #94a3b8; margin-top: 0.5rem;'>Explainable AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.06);'>
        <p style='margin: 0; font-size: 0.875rem;'>
            <strong style='color: #d1d5db;'>HealthForecast AI</strong> · Enterprise Healthcare Intelligence Platform
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem;'>
            Powered by Advanced Machine Learning · Built for Healthcare Excellence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
