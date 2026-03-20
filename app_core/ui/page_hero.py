# =============================================================================
# page_hero.py - Shared Hero Header & Background Image Utilities
# Provides consistent hero sections with background images across all pages
# =============================================================================
from __future__ import annotations

import streamlit as st
import base64
from pathlib import Path
from typing import Optional, Dict

# =============================================================================
# IMAGE PATHS CONFIGURATION
# Maps page identifiers to their background images
# =============================================================================
PAGE_IMAGES: Dict[str, Dict[str, str]] = {
    "dashboard": {
        "hero": "dashboard-bg.jpg",
        "section": "dashboard-bg2.jpg",
    },
    "upload": {
        "hero": "carousel-1.jpg",
        "section": "carousel-2.jpg",
    },
    "prepare": {
        "hero": "prepare-bg.jpg",
        "section": "carousel-3.jpg",
    },
    "explore": {
        "hero": "explore-bg.jpg",
        "section": "replace.jpg",
    },
    "baseline": {
        "hero": "train-bg.jpg",
        "section": "forecast-bg2.jpg",
    },
    "feature_studio": {
        "hero": "replace (2).jpg",
        "section": "replace (4).jpg",
    },
    "feature_selection": {
        "hero": "replace (4).jpg",
        "section": "replace (2).jpg",
    },
    "train": {
        "hero": "train-bg.jpg",
        "section": "forecast-bg.jpg",
    },
    "results": {
        "hero": "results-bg.jpg",
        "section": "dashboard-bg2.jpg",
    },
    "forecast": {
        "hero": "forecast-bg.jpg",
        "section": "forecast-bg2.jpg",
    },
    "staff": {
        "hero": "staff-bg.jpg",
        "section": "staff-bg2.jpg",
        "extra": "staff-bg3.jpg",
    },
    "supply": {
        "hero": "supply-bg.jpg",
        "section": "supply-bg2.jpg",
    },
    "actions": {
        "hero": "actions-bg.jpg",
        "section": "actions-bg2.jpg",
    },
}


def get_image_base64(image_name: str) -> str:
    """
    Load an image from the images folder and convert to base64.

    Args:
        image_name: Name of the image file (e.g., 'dashboard-bg.jpg')

    Returns:
        Base64 encoded string for use in CSS url()
    """
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent.parent / "images" / image_name,
        Path("images") / image_name,
        Path(__file__).resolve().parent.parent.parent / "images" / image_name,
    ]

    for img_path in possible_paths:
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                return f"data:image/jpeg;base64,{data}"
            except Exception:
                continue

    # Return empty string if image not found
    return ""


def get_page_images(page_id: str) -> Dict[str, str]:
    """
    Get all images for a specific page as base64 encoded strings.

    Args:
        page_id: Page identifier (e.g., 'dashboard', 'forecast', 'staff')

    Returns:
        Dictionary with image keys ('hero', 'section', etc.) and base64 values
    """
    images = PAGE_IMAGES.get(page_id, {})
    return {key: get_image_base64(filename) for key, filename in images.items()}


def inject_page_hero_styles(page_id: str) -> None:
    """
    Inject CSS styles for page hero section with background image.

    Args:
        page_id: Page identifier to load appropriate images
    """
    images = get_page_images(page_id)
    hero_bg = images.get("hero", "")
    section_bg = images.get("section", "")

    st.markdown(f"""
    <style>
    /* ========================================
       PAGE HERO SECTION
       ======================================== */

    .page-hero {{
        position: relative;
        background: linear-gradient(135deg,
            rgba(2, 6, 23, 0.85) 0%,
            rgba(6, 78, 145, 0.6) 50%,
            rgba(2, 6, 23, 0.9) 100%),
            url('{hero_bg}');
        background-size: cover;
        background-position: center;
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(34, 211, 238, 0.2);
        overflow: hidden;
        box-shadow:
            0 0 60px rgba(6, 78, 145, 0.3),
            0 20px 40px rgba(0, 0, 0, 0.4);
    }}

    .page-hero::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            180deg,
            transparent 0%,
            rgba(2, 6, 23, 0.3) 100%
        );
        pointer-events: none;
    }}

    .page-hero-content {{
        position: relative;
        z-index: 1;
        text-align: center;
    }}

    .page-hero-badge {{
        display: inline-block;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(59, 130, 246, 0.15));
        border: 1px solid rgba(34, 211, 238, 0.4);
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-size: 0.85rem;
        font-weight: 700;
        color: #22d3ee;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
    }}

    .page-hero-title {{
        font-size: 2.5rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0.5rem 0;
        text-shadow:
            0 0 30px rgba(0, 0, 0, 0.9),
            0 4px 20px rgba(0, 0, 0, 0.7);
    }}

    .page-hero-subtitle {{
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 600px;
        margin: 1rem auto 0 auto;
        line-height: 1.6;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
    }}

    .page-hero-stats {{
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }}

    .page-hero-stat {{
        text-align: center;
        padding: 1rem 1.5rem;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border-radius: 16px;
        border: 1px solid rgba(34, 211, 238, 0.2);
        backdrop-filter: blur(10px);
        min-width: 120px;
    }}

    .page-hero-stat-value {{
        font-size: 1.75rem;
        font-weight: 800;
        color: #22d3ee;
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
    }}

    .page-hero-stat-label {{
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }}

    /* ========================================
       SECTION DIVIDER WITH IMAGE
       ======================================== */

    .section-divider {{
        position: relative;
        background: linear-gradient(135deg,
            rgba(2, 6, 23, 0.9) 0%,
            rgba(6, 78, 145, 0.4) 50%,
            rgba(2, 6, 23, 0.95) 100%),
            url('{section_bg}');
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(34, 211, 238, 0.15);
        text-align: center;
    }}

    .section-divider-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 15px rgba(0, 0, 0, 0.8);
    }}

    .section-divider-subtitle {{
        font-size: 0.95rem;
        color: #94a3b8;
    }}

    /* ========================================
       MOBILE RESPONSIVE
       ======================================== */

    @media (max-width: 768px) {{
        .page-hero {{
            padding: 2rem 1.5rem;
            border-radius: 16px;
        }}

        .page-hero-title {{
            font-size: 1.75rem;
        }}

        .page-hero-subtitle {{
            font-size: 0.95rem;
        }}

        .page-hero-badge {{
            font-size: 0.75rem;
            padding: 0.4rem 1rem;
            letter-spacing: 2px;
        }}

        .page-hero-stats {{
            gap: 0.75rem;
        }}

        .page-hero-stat {{
            padding: 0.75rem 1rem;
            min-width: 100px;
        }}

        .page-hero-stat-value {{
            font-size: 1.25rem;
        }}

        .section-divider {{
            padding: 1.5rem;
            border-radius: 14px;
        }}

        .section-divider-title {{
            font-size: 1.25rem;
        }}
    }}

    @media (max-width: 480px) {{
        .page-hero {{
            padding: 1.5rem 1rem;
        }}

        .page-hero-title {{
            font-size: 1.5rem;
        }}

        .page-hero-stat {{
            min-width: 80px;
            padding: 0.5rem 0.75rem;
        }}

        .page-hero-stat-value {{
            font-size: 1.1rem;
        }}

        .page-hero-stat-label {{
            font-size: 0.65rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


def render_page_hero(
    title: str,
    subtitle: str,
    badge: Optional[str] = None,
    stats: Optional[list] = None,
    icon: str = ""
) -> None:
    """
    Render a hero section for a page.

    Args:
        title: Main title text
        subtitle: Subtitle/description text
        badge: Optional badge text above title
        stats: Optional list of dicts with 'value' and 'label' keys
        icon: Optional icon emoji to display before title
    """
    stats_html = ""
    if stats:
        stats_items = "".join([
            f'''<div class="page-hero-stat">
                <div class="page-hero-stat-value">{stat['value']}</div>
                <div class="page-hero-stat-label">{stat['label']}</div>
            </div>'''
            for stat in stats
        ])
        stats_html = f'<div class="page-hero-stats">{stats_items}</div>'

    badge_html = f'<div class="page-hero-badge">{badge}</div>' if badge else ""
    icon_html = f"{icon} " if icon else ""

    st.markdown(f"""
    <div class="page-hero">
        <div class="page-hero-content">
            {badge_html}
            <h1 class="page-hero-title">{icon_html}{title}</h1>
            <p class="page-hero-subtitle">{subtitle}</p>
            {stats_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_divider(title: str, subtitle: Optional[str] = None, icon: str = "") -> None:
    """
    Render a section divider with background image.

    Args:
        title: Section title
        subtitle: Optional subtitle
        icon: Optional icon emoji
    """
    icon_html = f"{icon} " if icon else ""
    subtitle_html = f'<p class="section-divider-subtitle">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div class="section-divider">
        <h2 class="section-divider-title">{icon_html}{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)
