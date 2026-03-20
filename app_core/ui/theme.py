from __future__ import annotations
import streamlit as st

# ============================================================================
# A. HEALTHCARE COLOR SYSTEM
# ============================================================================

# Primary Colors - Healthcare Palette
PRIMARY_COLOR    = "#0284c7"   # Healthcare blue (sky-600)
SECONDARY_COLOR  = "#0ea5e9"   # Light blue (sky-500)
ACCENT_BLUE      = "#38bdf8"   # Sky blue (sky-400)
ACCENT_LIGHT     = "#7dd3fc"   # Pale blue (sky-300)

# Action Colors
CTA_COLOR        = "#dc2626"   # Medical red for CTAs (red-600)
CTA_HOVER        = "#b91c1c"   # Darker red on hover (red-700)
SUCCESS_COLOR    = "#16a34a"   # Health green (green-600)
WARNING_COLOR    = "#f59e0b"   # Amber warning
DANGER_COLOR     = "#dc2626"   # Red for alerts

# Text Colors (Dark on Light)
TEXT_PRIMARY     = "#0f172a"   # Slate-900 - main headings
TEXT_SECONDARY   = "#334155"   # Slate-700 - body text
TEXT_MUTED       = "#64748b"   # Slate-500 - subtle text
TEXT_LIGHT       = "#94a3b8"   # Slate-400 - placeholders

# Background Colors (Light Theme)
BG_WHITE         = "#ffffff"   # Pure white
BG_LIGHT         = "#f8fafc"   # Slate-50 - light gray
BG_SUBTLE        = "#f1f5f9"   # Slate-100 - subtle sections
BG_CARD          = "#ffffff"   # White cards

# Border Colors
BORDER_LIGHT     = "#e2e8f0"   # Slate-200
BORDER_DEFAULT   = "#cbd5e1"   # Slate-300
BORDER_ACCENT    = "#0ea5e9"   # Blue accent border

# Legacy exports (for backwards compatibility)
TEXT_COLOR       = TEXT_PRIMARY
BODY_TEXT        = TEXT_SECONDARY
SUBTLE_TEXT      = TEXT_MUTED
CARD_BG          = BG_WHITE
BG_BLACK         = BG_LIGHT
BG_GRADIENT_START = BG_WHITE
BG_GRADIENT_MID   = BG_LIGHT
BG_GRADIENT_END   = BG_SUBTLE
ACCENT_PINK      = CTA_COLOR
ACCENT_ROSE      = "#fb7185"


# ============================================================================
# B. GLOBAL CSS - HEALTHCARE LIGHT THEME
# ============================================================================
def apply_css() -> None:
    css_string = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ========================================
       GLOBAL RESET & TYPOGRAPHY
       ======================================== */

    html, body, [class*='css'] {{
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {TEXT_SECONDARY};
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* ========================================
       MAIN BACKGROUND - LIGHT HEALTHCARE
       ======================================== */

    .main {{
        background: linear-gradient(180deg, {BG_WHITE} 0%, {BG_LIGHT} 50%, {BG_SUBTLE} 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }}

    /* Subtle blue accent gradient overlay */
    .main::before {{
        content: '';
        position: fixed;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: radial-gradient(ellipse at top right, rgba(14, 165, 233, 0.05) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }}

    .main::after {{
        content: '';
        position: fixed;
        bottom: 0;
        left: 0;
        width: 50%;
        height: 50%;
        background: radial-gradient(ellipse at bottom left, rgba(2, 132, 199, 0.03) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }}

    /* ========================================
       SIDEBAR - DARK FOR CONTRAST
       ======================================== */

    .stSidebar {{
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(14, 165, 233, 0.2);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.1);
    }}

    .stSidebar [data-testid="stSidebarNav"] {{
        background-color: transparent;
        padding-top: 2rem;
    }}

    .stSidebar > div:first-child {{
        padding: 2rem 1.5rem 1rem 1.5rem;
    }}

    .stSidebar [data-testid="stSidebarNav"] ul {{
        padding: 0 1rem;
    }}

    .stSidebar [data-testid="stSidebarNav"] li {{
        margin-bottom: 0.35rem;
    }}

    .stSidebar [data-testid="stSidebarNav"] a {{
        display: flex;
        align-items: center;
        padding: 0.875rem 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        text-decoration: none;
        font-size: 0.9375rem;
        font-weight: 500;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid transparent;
    }}

    .stSidebar [data-testid="stSidebarNav"] a:hover {{
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(56, 189, 248, 0.15));
        border-color: rgba(14, 165, 233, 0.4);
        color: #ffffff;
        transform: translateX(4px);
    }}

    .stSidebar [data-testid="stSidebarNav"] a[aria-current="page"],
    .stSidebar [data-testid="stSidebarNav"] a.active {{
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.3), rgba(56, 189, 248, 0.2));
        border-color: rgba(14, 165, 233, 0.5);
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.2);
    }}

    .stSidebar [data-testid="stSidebarNav"] a[aria-current="page"]::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-radius: 0 3px 3px 0;
    }}

    .stSidebar hr {{
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 1rem;
    }}

    .stSidebar [data-testid="collapsedControl"] {{
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 0 12px 12px 0;
        color: {SECONDARY_COLOR};
        transition: all 0.3s ease;
    }}

    .stSidebar [data-testid="collapsedControl"]:hover {{
        background: rgba(14, 165, 233, 0.2);
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.3);
    }}

    .stSidebar .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(56, 189, 248, 0.15));
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #ffffff;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stSidebar .stButton > button:hover {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-color: transparent;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4);
        transform: translateY(-2px);
    }}

    .stSidebar .stSelectbox > div > div,
    .stSidebar .stTextInput > div > div > input {{
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }}

    .stSidebar .stSelectbox > div > div:hover,
    .stSidebar .stTextInput > div > div > input:hover {{
        border-color: rgba(14, 165, 233, 0.4);
    }}

    .stSidebar .stSelectbox > div > div:focus-within,
    .stSidebar .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
    }}

    /* ========================================
       MAIN CONTENT CONTAINER
       ======================================== */

    .block-container {{
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
        position: relative;
        z-index: 1;
    }}

    /* ========================================
       HEADINGS - DARK TEXT
       ======================================== */

    h1 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 800;
        letter-spacing: -0.03em;
    }}

    h2 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    h3 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600;
    }}

    h4, h5, h6 {{
        color: {TEXT_SECONDARY} !important;
        font-weight: 600;
    }}

    p {{
        color: {TEXT_SECONDARY};
    }}

    /* ========================================
       HEALTHCARE CARDS - WHITE WITH SHADOWS
       ======================================== */

    .hf-hero-container {{
        position: relative;
        border-radius: 24px;
        background: {BG_WHITE};
        border: 1px solid {BORDER_LIGHT};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 2px 4px -2px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(14, 165, 233, 0.05);
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        overflow: hidden;
    }}

    .hf-hero-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}, {ACCENT_BLUE});
    }}

    .hf-hero-content {{
        position: relative;
        z-index: 1;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    .hf-hero-title {{
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: {TEXT_PRIMARY};
        margin-bottom: 1.5rem;
        text-align: center !important;
        width: 100% !important;
    }}

    .hf-hero-subtitle {{
        font-size: clamp(1.125rem, 2vw, 1.375rem);
        font-weight: 400;
        color: {TEXT_SECONDARY};
        line-height: 1.7;
        max-width: 700px;
        margin: 0 auto 2rem auto !important;
        text-align: center !important;
        width: 100% !important;
        display: block !important;
    }}

    /* Pills/Tags - Healthcare Style */
    .hf-pill-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        justify-content: center;
        margin-top: 2rem;
    }}

    .hf-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(56, 189, 248, 0.08));
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: {PRIMARY_COLOR};
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.02em;
        cursor: default;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .hf-pill:hover {{
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(56, 189, 248, 0.15));
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
        transform: translateY(-2px);
    }}

    /* Feature Cards - Healthcare White */
    .hf-feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }}

    .hf-feature-card {{
        position: relative;
        border-radius: 20px;
        background: {BG_WHITE};
        border: 1px solid {BORDER_LIGHT};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 2px 4px -2px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }}

    .hf-feature-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .hf-feature-card:hover {{
        transform: translateY(-6px);
        border-color: rgba(14, 165, 233, 0.3);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1),
                    0 8px 10px -6px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(14, 165, 233, 0.1);
    }}

    .hf-feature-card:hover::before {{
        opacity: 1;
    }}

    .hf-feature-icon {{
        width: 56px;
        height: 56px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(56, 189, 248, 0.1));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(14, 165, 233, 0.2);
    }}

    .hf-feature-title {{
        font-size: 1.375rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }}

    .hf-feature-description {{
        font-size: 1rem;
        color: {TEXT_SECONDARY};
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }}

    .hf-feature-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}

    .hf-feature-list li {{
        position: relative;
        padding-left: 1.75rem;
        margin-bottom: 0.75rem;
        color: {TEXT_SECONDARY};
        font-size: 0.9375rem;
        line-height: 1.6;
    }}

    .hf-feature-list li::before {{
        content: '✓';
        position: absolute;
        left: 0;
        color: {SUCCESS_COLOR};
        font-weight: 700;
    }}

    /* ========================================
       BUTTONS - HEALTHCARE STYLE
       ======================================== */

    /* Primary Button - Healthcare Blue */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
    }}

    .stButton > button:hover {{
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4);
        transform: translateY(-2px);
    }}

    /* CTA Button - Medical Red */
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, {CTA_COLOR} 0%, #ef4444 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35) !important;
        transition: all 0.3s ease !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {CTA_HOVER} 0%, #dc2626 100%) !important;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.45) !important;
        transform: translateY(-3px) !important;
    }}

    /* ========================================
       INPUT FIELDS - LIGHT THEME
       ======================================== */

    .stTextInput > div > div > input {{
        background: {BG_WHITE};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 10px;
        color: {TEXT_PRIMARY};
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {TEXT_LIGHT};
    }}

    .stSelectbox > div > div {{
        background: {BG_WHITE};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 10px;
        color: {TEXT_PRIMARY};
    }}

    .stSelectbox > div > div:hover {{
        border-color: {PRIMARY_COLOR};
    }}

    .stNumberInput > div > div > input {{
        background: {BG_WHITE};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 10px;
        color: {TEXT_PRIMARY};
    }}

    /* ========================================
       METRICS - HEALTHCARE STYLE
       ======================================== */

    [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
    }}

    [data-testid="stMetricDelta"] svg {{
        stroke: {SUCCESS_COLOR};
    }}

    /* ========================================
       TABS - LIGHT THEME
       ======================================== */

    .stTabs [data-baseweb="tab-list"] {{
        background: {BG_SUBTLE};
        border-radius: 12px;
        padding: 0.25rem;
        gap: 0.25rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 10px;
        color: {TEXT_SECONDARY};
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: {BG_WHITE};
        color: {TEXT_PRIMARY};
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {BG_WHITE};
        color: {PRIMARY_COLOR};
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    /* ========================================
       EXPANDERS - LIGHT THEME
       ======================================== */

    .streamlit-expanderHeader {{
        background: {BG_WHITE};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 12px;
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {PRIMARY_COLOR};
    }}

    .streamlit-expanderContent {{
        background: {BG_WHITE};
        border: 1px solid {BORDER_LIGHT};
        border-top: none;
        border-radius: 0 0 12px 12px;
    }}

    /* ========================================
       ALERTS/INFO BOXES
       ======================================== */

    .stAlert {{
        border-radius: 12px;
        border: 1px solid;
    }}

    [data-testid="stAlert"][data-baseweb="notification"] {{
        background: rgba(14, 165, 233, 0.08);
        border-color: rgba(14, 165, 233, 0.3);
    }}

    /* ========================================
       DATAFRAMES/TABLES
       ======================================== */

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_LIGHT};
        border-radius: 12px;
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] table {{
        background: {BG_WHITE};
    }}

    [data-testid="stDataFrame"] th {{
        background: {BG_SUBTLE} !important;
        color: {TEXT_PRIMARY} !important;
        font-weight: 600;
    }}

    [data-testid="stDataFrame"] td {{
        color: {TEXT_SECONDARY};
    }}

    /* ========================================
       SECTION CARDS - LEGACY SUPPORT
       ======================================== */

    .hf-section-card {{
        border-radius: 20px;
        background: {BG_WHITE};
        border: 1px solid {BORDER_LIGHT};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 2px 4px -2px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin-bottom: 1.5rem;
    }}

    .hf-section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 1rem;
    }}

    .hf-section-body {{
        color: {TEXT_SECONDARY};
        font-size: 1rem;
        line-height: 1.7;
    }}

    /* ========================================
       RESPONSIVE STYLES
       ======================================== */

    @media (max-width: 1023px) {{
        .block-container {{
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }}
        .hf-hero-container {{
            padding: 3rem 2rem;
        }}
    }}

    @media (max-width: 767px) {{
        .block-container {{
            padding: 1.5rem 1rem;
        }}
        .hf-hero-container {{
            padding: 2rem 1.5rem;
            border-radius: 16px;
        }}
        .hf-feature-card {{
            padding: 1.5rem;
        }}

        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }}

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            display: none;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0.75rem;
            font-size: 0.85rem;
            white-space: nowrap;
        }}

        .stSidebar {{
            width: 280px !important;
        }}

        h1 {{ font-size: 1.5rem !important; }}
        h2 {{ font-size: 1.25rem !important; }}
        h3 {{ font-size: 1.1rem !important; }}

        .stButton > button {{
            padding: 0.625rem 1.5rem;
            font-size: 0.875rem;
        }}
    }}

    @media (max-width: 480px) {{
        .block-container {{
            padding: 1rem 0.75rem;
        }}
        .hf-hero-container {{
            padding: 1.5rem 1rem;
        }}
        .hf-feature-card {{
            padding: 1.25rem;
            border-radius: 14px;
        }}

        h1 {{ font-size: 1.25rem !important; }}
        h2 {{ font-size: 1.1rem !important; }}

        .stButton > button {{
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            width: 100%;
        }}
    }}

    </style>
    """
    st.markdown(css_string, unsafe_allow_html=True)


# ============================================================================
# C. HERO HELPER - HEALTHCARE STYLE
# ============================================================================
def hero_card(title: str, subtitle: str, pills: list[str]) -> str:
    """
    Returns an HTML string for a healthcare-styled hero section
    """
    pills_html = "".join([f"<span class='hf-pill'>{pill}</span>" for pill in pills])
    return f"""
    <div class='hf-hero-container'>
        <div class='hf-hero-content'>
            <h1 class='hf-hero-title'>{title}</h1>
            <p class='hf-hero-subtitle'>{subtitle}</p>
            <div class='hf-pill-container'>
                {pills_html}
            </div>
        </div>
    </div>
    """


# ============================================================================
# D. FEATURE CARD HELPER
# ============================================================================
def feature_card(icon: str, title: str, description: str, features: list[str]) -> str:
    """
    Returns an HTML string for a healthcare-styled feature card
    """
    features_html = "".join([f"<li>{feature}</li>" for feature in features])
    return f"""
    <div class='hf-feature-card'>
        <div class='hf-feature-icon'>{icon}</div>
        <h3 class='hf-feature-title'>{title}</h3>
        <p class='hf-feature-description'>{description}</p>
        <ul class='hf-feature-list'>
            {features_html}
        </ul>
    </div>
    """


# ============================================================================
# E. EXPORTS
# ============================================================================
__all__ = [
    "apply_css",
    "hero_card",
    "feature_card",
    # New healthcare colors
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "ACCENT_BLUE",
    "ACCENT_LIGHT",
    "CTA_COLOR",
    "CTA_HOVER",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DANGER_COLOR",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "TEXT_LIGHT",
    "BG_WHITE",
    "BG_LIGHT",
    "BG_SUBTLE",
    "BG_CARD",
    "BORDER_LIGHT",
    "BORDER_DEFAULT",
    "BORDER_ACCENT",
    # Legacy exports
    "TEXT_COLOR",
    "BODY_TEXT",
    "SUBTLE_TEXT",
    "CARD_BG",
    "BG_BLACK",
    "BG_GRADIENT_START",
    "BG_GRADIENT_MID",
    "BG_GRADIENT_END",
    "ACCENT_PINK",
    "ACCENT_ROSE",
]
