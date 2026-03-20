from __future__ import annotations
import streamlit as st

# A. COLOR SYSTEM (export publicly)
PRIMARY_COLOR   = "#3b82f6"   # electric blue
SECONDARY_COLOR = "#22d3ee"   # neon cyan
ACCENT_PINK     = "#f43f5e"   # neon pink
ACCENT_ROSE     = "#fb7185"   # soft rose
SUCCESS_COLOR   = "#22c55e"
WARNING_COLOR   = "#facc15"
DANGER_COLOR    = "#f97373"
TEXT_COLOR      = "#ffffff"   # pure white for headings
BODY_TEXT       = "#d1d5db"   # soft gray for body
SUBTLE_TEXT     = "#94a3b8"   # muted labels
CARD_BG         = "#0b1120"   # charcoal background
BG_BLACK        = "#0b1120"   # black background (same as CARD_BG)
BG_GRADIENT_START = "#020617" # deep cinematic gradient start
BG_GRADIENT_MID   = "#050816" # gradient middle
BG_GRADIENT_END   = "#02010f" # gradient end

# B. GLOBAL CSS (in apply_css())
def apply_css() -> None:
    css_string = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Reset & Typography */
    html, body, [class*='css'] {{
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {BODY_TEXT};
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* Cinematic Background Gradient */
    .main {{
        background: linear-gradient(135deg, {BG_GRADIENT_START} 0%, {BG_GRADIENT_MID} 50%, {BG_GRADIENT_END} 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }}

    /* Diagonal Neon Streaks (Raycast-style) */
    .main::before {{
        content: '';
        position: fixed;
        top: -50%;
        right: -20%;
        width: 100%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(59,130,246,0.15) 0%, transparent 70%);
        transform: rotate(-45deg);
        pointer-events: none;
        z-index: 0;
    }}
    .main::after {{
        content: '';
        position: fixed;
        bottom: -50%;
        left: -20%;
        width: 100%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(244,63,94,0.1) 0%, transparent 70%);
        transform: rotate(-45deg);
        pointer-events: none;
        z-index: 0;
    }}

    /* ========================================
       SIDEBAR - Premium Navigation Design
       ======================================== */

    /* Sidebar Container */
    .stSidebar {{
        background: linear-gradient(180deg, {CARD_BG} 0%, {BG_GRADIENT_START} 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.12);
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
    }}

    /* Sidebar Content */
    .stSidebar [data-testid="stSidebarNav"] {{
        background-color: transparent;
        padding-top: 2rem;
    }}

    /* Logo/Branding Area */
    .stSidebar > div:first-child {{
        padding: 2rem 1.5rem 1rem 1.5rem;
    }}

    /* Navigation Links Container */
    .stSidebar [data-testid="stSidebarNav"] ul {{
        padding: 0 1rem;
    }}

    /* Individual Navigation Items */
    .stSidebar [data-testid="stSidebarNav"] li {{
        margin-bottom: 0.35rem;
    }}

    /* Navigation Link Styling */
    .stSidebar [data-testid="stSidebarNav"] a {{
        display: flex;
        align-items: center;
        padding: 0.875rem 1.25rem;
        border-radius: 12px;
        color: {BODY_TEXT};
        text-decoration: none;
        font-size: 0.9375rem;
        font-weight: 500;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid transparent;
    }}

    /* Navigation Link Hover Effect */
    .stSidebar [data-testid="stSidebarNav"] a:hover {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
        border-color: rgba(59, 130, 246, 0.3);
        color: {TEXT_COLOR};
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }}

    /* Active/Selected Navigation Item */
    .stSidebar [data-testid="stSidebarNav"] a[aria-current="page"],
    .stSidebar [data-testid="stSidebarNav"] a.active {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(34, 211, 238, 0.15));
        border-color: rgba(59, 130, 246, 0.5);
        color: {TEXT_COLOR};
        font-weight: 600;
        box-shadow:
            0 0 20px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }}

    /* Active Item Left Accent Bar */
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

    /* Navigation Section Headers (if any) */
    .stSidebar [data-testid="stSidebarNav"] .css-1q8dd3e {{
        color: {SUBTLE_TEXT};
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 1.5rem 1.25rem 0.5rem 1.25rem;
        margin-top: 1rem;
    }}

    /* Sidebar Divider */
    .stSidebar hr {{
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1.5rem 1rem;
    }}

    /* Sidebar User Controls Area (collapse button, etc.) */
    .stSidebar [data-testid="collapsedControl"] {{
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 0 12px 12px 0;
        color: {SECONDARY_COLOR};
        transition: all 0.3s ease;
    }}

    .stSidebar [data-testid="collapsedControl"]:hover {{
        background: rgba(59, 130, 246, 0.2);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }}

    /* Sidebar Widgets Styling */
    .stSidebar .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(34, 211, 238, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: {TEXT_COLOR};
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stSidebar .stButton > button:hover {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-color: transparent;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }}

    /* Sidebar Select/Input Fields */
    .stSidebar .stSelectbox > div > div,
    .stSidebar .stTextInput > div > div > input {{
        background: rgba(11, 17, 32, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        color: {TEXT_COLOR};
    }}

    .stSidebar .stSelectbox > div > div:hover,
    .stSidebar .stTextInput > div > div > input:hover {{
        border-color: rgba(59, 130, 246, 0.3);
    }}

    .stSidebar .stSelectbox > div > div:focus-within,
    .stSidebar .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }}

    /* Main Content Container */
    .block-container {{
        max-width: 1400px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
        position: relative;
        z-index: 1;
    }}

    /* Hero Section (Bolt + Raycast Fusion) */
    .hf-hero-container {{
        position: relative;
        border-radius: 32px;
        background: linear-gradient(135deg, rgba(11,17,32,0.95), rgba(5,8,22,0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow:
            0 0 60px rgba(59,130,246,0.2),
            0 0 30px rgba(244,63,94,0.15),
            inset 0 1px 0 rgba(255,255,255,0.05);
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }}

    /* Hero Diagonal Glow Effect */
    .hf-hero-container::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 80%;
        height: 200%;
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(34,211,238,0.1));
        filter: blur(80px);
        transform: rotate(-25deg);
        pointer-events: none;
        z-index: 0;
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

    /* Hero Title */
    .hf-hero-title {{
        font-size: clamp(2.5rem, 5vw, 4.5rem);
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, {TEXT_COLOR} 0%, {SECONDARY_COLOR} 50%, {PRIMARY_COLOR} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 40px rgba(59,130,246,0.3);
        text-align: center !important;
        width: 100% !important;
    }}

    /* Hero Subtitle */
    .hf-hero-subtitle {{
        font-size: clamp(1.125rem, 2vw, 1.5rem);
        font-weight: 400;
        color: {BODY_TEXT};
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto 2rem auto !important;
        opacity: 0.9;
        text-align: center !important;
        width: 100% !important;
        display: block !important;
    }}

    /* Pills (Tags) */
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
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(34,211,238,0.1));
        border: 1px solid rgba(59,130,246,0.3);
        color: {SECONDARY_COLOR};
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.02em;
        cursor: default;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }}

    .hf-pill:hover {{
        background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(34,211,238,0.2));
        border-color: rgba(59,130,246,0.5);
        box-shadow: 0 0 20px rgba(59,130,246,0.4);
        transform: translateY(-2px);
    }}

    /* Feature Cards */
    .hf-feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }}

    .hf-feature-card {{
        position: relative;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(11,17,32,0.9), rgba(5,8,22,0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow:
            0 4px 24px rgba(0,0,0,0.3),
            0 0 0 1px rgba(255,255,255,0.05) inset;
        padding: 2.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        backdrop-filter: blur(10px);
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
        transform: translateY(-8px);
        border-color: rgba(59,130,246,0.3);
        box-shadow:
            0 8px 40px rgba(59,130,246,0.25),
            0 0 0 1px rgba(59,130,246,0.2) inset;
    }}

    .hf-feature-card:hover::before {{
        opacity: 1;
    }}

    .hf-feature-icon {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(34,211,238,0.15));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(59,130,246,0.3);
    }}

    .hf-feature-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }}

    .hf-feature-description {{
        font-size: 1rem;
        color: {BODY_TEXT};
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
        margin-bottom: 0.875rem;
        color: {BODY_TEXT};
        font-size: 0.9375rem;
        line-height: 1.6;
    }}

    .hf-feature-list li::before {{
        content: '→';
        position: absolute;
        left: 0;
        color: {PRIMARY_COLOR};
        font-weight: 700;
    }}

    /* Responsive Typography */
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
            padding: 2rem 1rem;
        }}
        .hf-hero-container {{
            padding: 2.5rem 1.5rem;
            border-radius: 20px;
        }}
        .hf-feature-card {{
            padding: 2rem;
        }}
        .hf-pill-container {{
            justify-content: center;
        }}

        /* Mobile: Stack columns vertically */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}

        /* Tabs - scrollable on mobile */
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            gap: 0.25rem;
        }}

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            display: none;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
            white-space: nowrap;
        }}

        /* Sidebar - collapse on mobile */
        .stSidebar {{
            width: 280px !important;
        }}

        .stSidebar [data-testid="stSidebarNav"] a {{
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
        }}

        /* Headers */
        h1 {{
            font-size: 1.5rem !important;
        }}
        h2 {{
            font-size: 1.25rem !important;
        }}
        h3 {{
            font-size: 1.1rem !important;
        }}

        /* Buttons */
        .stButton > button {{
            padding: 0.625rem 1.5rem;
            font-size: 0.875rem;
        }}

        /* Data frames - horizontal scroll */
        [data-testid="stDataFrame"] {{
            overflow-x: auto;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            font-size: 0.9rem;
        }}

        /* Section headers */
        .section-header, .subsection-header {{
            font-size: 1rem !important;
        }}
    }}

    /* Very small mobile (480px and below) */
    @media (max-width: 480px) {{
        .block-container {{
            padding: 1rem 0.75rem;
        }}

        .hf-hero-container {{
            padding: 1.5rem 1rem;
            border-radius: 16px;
        }}

        .hf-feature-card {{
            padding: 1.25rem;
            border-radius: 16px;
        }}

        h1 {{
            font-size: 1.25rem !important;
        }}
        h2 {{
            font-size: 1.1rem !important;
        }}
        h3 {{
            font-size: 1rem !important;
        }}

        .stButton > button {{
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            width: 100%;
        }}

        /* Tabs - even more compact */
        .stTabs [data-baseweb="tab"] {{
            padding: 0.4rem 0.6rem;
            font-size: 0.75rem;
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            font-size: 0.875rem;
            padding: 0.5rem 0.75rem;
        }}

        /* Sliders */
        .stSlider {{
            padding: 0.5rem 0;
        }}

        /* Info/Warning/Error boxes */
        .stAlert {{
            padding: 0.75rem;
            font-size: 0.85rem;
        }}
    }}

    /* Ultra small (360px and below) */
    @media (max-width: 360px) {{
        .block-container {{
            padding: 0.75rem 0.5rem;
        }}

        h1 {{
            font-size: 1.1rem !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0.35rem 0.5rem;
            font-size: 0.7rem;
        }}
    }}

    /* Headings */
    h1 {{
        color: {TEXT_COLOR};
        font-weight: 800;
        letter-spacing: -0.03em;
    }}
    h2 {{
        color: {TEXT_COLOR};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    h3 {{
        color: {TEXT_COLOR};
        font-weight: 600;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {TEXT_COLOR};
        font-weight: 700;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59,130,246,0.3);
    }}

    .stButton > button:hover {{
        box-shadow: 0 6px 30px rgba(59,130,246,0.5);
        transform: translateY(-2px);
    }}

    /* Input Fields */
    .stTextInput > div > div > input {{
        background: rgba(11,17,32,0.8);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        color: {TEXT_COLOR};
        padding: 0.75rem 1rem;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
    }}

    /* Legacy card classes (for backwards compatibility) */
    .hf-section-card {{
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(11,17,32,0.9), rgba(5,8,22,0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        padding: 2rem;
        margin-bottom: 1.5rem;
    }}

    .hf-section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-bottom: 1rem;
    }}

    .hf-section-body {{
        color: {BODY_TEXT};
        font-size: 1rem;
        line-height: 1.7;
    }}

    </style>
    """
    st.markdown(css_string, unsafe_allow_html=True)

# C. HERO HELPER (Modern Bolt/Raycast Style)
def hero_card(title: str, subtitle: str, pills: list[str]) -> str:
    """
    Returns an HTML string for a premium hero section inspired by bolt.new and raycast.com
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

# D. FEATURE CARD HELPER
def feature_card(icon: str, title: str, description: str, features: list[str]) -> str:
    """
    Returns an HTML string for a feature card
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

# E. EXPORTS
__all__ = [
    "apply_css",
    "hero_card",
    "feature_card",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "ACCENT_PINK",
    "ACCENT_ROSE",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DANGER_COLOR",
    "TEXT_COLOR",
    "BODY_TEXT",
    "SUBTLE_TEXT",
    "CARD_BG",
    "BG_BLACK",
    "BG_GRADIENT_START",
    "BG_GRADIENT_MID",
    "BG_GRADIENT_END",
]