# =============================================================================
# app_core/ui/sidebar_brand.py — Reusable Sidebar Brand & Styling
# =============================================================================
"""
Provides idempotent sidebar branding and styling for HealthForecast AI.
Call these functions immediately after st.set_page_config() on every page.
"""
from __future__ import annotations
import streamlit as st
from pathlib import Path


def inject_sidebar_style():
    """
    Injects CSS for sidebar brand and page navigation styling.
    Idempotent - uses session state to ensure CSS is only injected once per session.

    Features:
    - Sticky brand positioning at the top (above page navigation)
    - Glassmorphism styling with backdrop blur
    - Gradient text for brand title
    - Sidebar page link styling with hover effects
    - Renames "app" to "Home" via CSS
    - Active page highlight with gradient and border
    """
    # Idempotency check
    if st.session_state.get("_sidebar_style_injected", False):
        return

    css = """
    <style>
    /* ========================================
       PREMIUM SIDEBAR BRANDING
       ======================================== */

    /* Force brand to appear above page navigation */
    section[data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
    }

    /* Premium Brand Container - Raycast/Bolt Style */
    section[data-testid="stSidebar"] .hf-brand-wrap {
        order: -1;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin: 0 0 1.5rem 0;
        padding: 1.5rem 1.25rem;
        background: linear-gradient(135deg, rgba(11,17,32,0.95), rgba(5,8,22,0.9));
        backdrop-filter: blur(20px) saturate(150%);
        border-bottom: 1px solid rgba(59, 130, 246, 0.15);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    /* Brand Link Container */
    .hf-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        text-decoration: none;
        padding: 0.75rem;
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: transparent;
        border: 1px solid transparent;
    }

    .hf-brand:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05));
        border-color: rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }

    /* Logo Icon */
    .hf-brand img {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        box-shadow:
            0 0 20px rgba(59, 130, 246, 0.3),
            0 4px 12px rgba(34, 211, 238, 0.2);
        flex-shrink: 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }

    .hf-brand:hover img {
        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.5),
            0 6px 16px rgba(34, 211, 238, 0.3);
        transform: scale(1.05);
    }

    /* Brand Text Container */
    .hf-brand-text {
        display: flex;
        flex-direction: column;
        gap: 3px;
        flex: 1;
    }

    /* Brand Title - Gradient Effect */
    .hf-brand-title {
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #22d3ee 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    /* Brand Tagline */
    .hf-brand-tag {
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.3;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* Status Badge (optional) */
    .hf-brand-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.3);
        font-size: 0.65rem;
        font-weight: 700;
        color: #22d3ee;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    /* Responsive - Hide text on narrow widths */
    @media (max-width: 380px) {
        .hf-brand-title,
        .hf-brand-tag,
        .hf-brand-badge {
            display: none;
        }
        .hf-brand {
            justify-content: center;
        }
    }

    /* Ensure page navigation appears after brand */
    [data-testid="stSidebarNav"] {
        order: 0;
        padding-top: 0;
    }

    /* ========================================
       SIDEBAR PAGE NAVIGATION - PREMIUM STYLE
       ======================================== */

    /* Rename "app" to "🏠 Home" */
    section[data-testid="stSidebar"] ul li:first-child a span:first-child::after {
        content: "🏠 Home";
        position: absolute;
        left: 0;
        color: #d1d5db;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    section[data-testid="stSidebar"] ul li:first-child a span:first-child {
        position: relative;
        color: transparent !important;
    }

    /* Sidebar page links base styling */
    section[data-testid="stSidebar"] ul li a span {
        color: #d1d5db !important;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Hover effect on page links */
    section[data-testid="stSidebar"] ul li a:hover span {
        color: #ffffff !important;
    }

    /* Active (selected) page text gradient */
    section[data-testid="stSidebar"] ul li a[aria-current="page"] span {
        background: linear-gradient(135deg, #ffffff 0%, #22d3ee 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }

    /* Remove default Streamlit link underlines */
    section[data-testid="stSidebar"] ul li a {
        text-decoration: none !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Mark as injected
    st.session_state["_sidebar_style_injected"] = True


def render_sidebar_brand():
    """
    Renders the HealthForecast AI brand logo at the top of the sidebar.
    Idempotent - uses session state to ensure brand is only rendered once per session.

    Features:
    - STATIC logo (non-clickable)
    - SVG logo with gradient and glow
    - Glassmorphism styling
    - Responsive (hides text on narrow widths)
    - Robust fallback chain: SVG → PNG → Programmatic SVG → Text only
    """
    # Idempotency check
    if st.session_state.get("_sidebar_brand_rendered", False):
        return

    logo_html = ""

    # Try to load SVG logo
    try:
        svg_path = Path(__file__).parent.parent / "assets" / "brand_logo.svg"
        if svg_path.exists():
            svg_data = svg_path.read_text(encoding="utf-8")
            # URL encode the SVG for data URI
            import urllib.parse
            svg_encoded = urllib.parse.quote(svg_data)
            logo_html = f'<img src="data:image/svg+xml;utf8,{svg_encoded}" alt="HealthForecast AI" />'
        else:
            # Try PNG fallback
            png_path = Path(__file__).parent.parent / "assets" / "brand_logo_128.png"
            if png_path.exists():
                import base64
                png_data = png_path.read_bytes()
                png_b64 = base64.b64encode(png_data).decode()
                logo_html = f'<img src="data:image/png;base64,{png_b64}" alt="HealthForecast AI" />'
            else:
                # Programmatic SVG fallback
                logo_html = '<img src="data:image/svg+xml;utf8,%3Csvg%20width%3D%2248%22%20height%3D%2248%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%2224%22%20r%3D%2222%22%20fill%3D%22%2360A5FA%22%2F%3E%3Ctext%20x%3D%2224%22%20y%3D%2232%22%20text-anchor%3D%22middle%22%20fill%3D%22white%22%20font-size%3D%2218%22%20font-weight%3D%22bold%22%3EHF%3C%2Ftext%3E%3C%2Fsvg%3E" alt="HealthForecast AI" />'
    except Exception:
        # Final fallback: programmatic SVG
        logo_html = '<img src="data:image/svg+xml;utf8,%3Csvg%20width%3D%2248%22%20height%3D%2248%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%2224%22%20r%3D%2222%22%20fill%3D%22%2360A5FA%22%2F%3E%3Ctext%20x%3D%2224%22%20y%3D%2232%22%20text-anchor%3D%22middle%22%20fill%3D%22white%22%20font-size%3D%2218%22%20font-weight%3D%22bold%22%3EHF%3C%2Ftext%3E%3C%2Fsvg%3E" alt="HealthForecast AI" />'

    # Render premium brand HTML at the very top of sidebar
    brand_html = f"""
    <div class="hf-brand-wrap">
        <div class="hf-brand">
            {logo_html}
            <div class="hf-brand-text">
                <div class="hf-brand-title">HealthForecast AI</div>
                <div class="hf-brand-tag">Enterprise Healthcare Intelligence</div>
            </div>
        </div>
    </div>
    """

    st.sidebar.markdown(brand_html, unsafe_allow_html=True)

    # Mark as rendered
    st.session_state["_sidebar_brand_rendered"] = True


def sidebar_section_divider(label: str = "") -> None:
    """
    Renders a premium section divider in the sidebar with optional label.

    Args:
        label: Optional section label (e.g., "ANALYTICS", "OPERATIONS")
    """
    if label:
        divider_html = f"""
        <div style='
            margin: 1.5rem 0 0.75rem 0;
            padding: 0.5rem 0;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        '>
            <div style='
                color: #94a3b8;
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-top: 0.75rem;
                padding: 0 1.25rem;
            '>{label}</div>
        </div>
        """
    else:
        divider_html = """
        <div style='
            margin: 1.5rem 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        '></div>
        """

    st.sidebar.markdown(divider_html, unsafe_allow_html=True)


def sidebar_info_card(title: str, value: str, icon: str = "ℹ️") -> None:
    """
    Renders a small info card in the sidebar.

    Args:
        title: Card title
        value: Card value/content
        icon: Optional emoji icon
    """
    card_html = f"""
    <div style='
        margin: 0.75rem 0;
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05));
        border: 1px solid rgba(59, 130, 246, 0.2);
    '>
        <div style='
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        '>
            <span style='font-size: 1.25rem;'>{icon}</span>
            <span style='
                color: #94a3b8;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            '>{title}</span>
        </div>
        <div style='
            color: #ffffff;
            font-size: 0.9375rem;
            font-weight: 600;
            line-height: 1.4;
        '>{value}</div>
    </div>
    """

    st.sidebar.markdown(card_html, unsafe_allow_html=True)


def render_cache_management() -> None:
    """
    Renders cache management UI in the sidebar.
    Shows cache status and provides clear cache button.
    """
    try:
        from app_core.cache import get_cache_info, has_cached_data, clear_cache
        from app_core.state.session import clear_session_and_cache

        # Section divider
        sidebar_section_divider("CACHE")

        cache_info = get_cache_info()
        has_cache = has_cached_data()

        if has_cache:
            # Show cache status
            st.sidebar.markdown(
                f"""
                <div style='
                    margin: 0.5rem 0;
                    padding: 0.75rem;
                    border-radius: 10px;
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05));
                    border: 1px solid rgba(34, 197, 94, 0.2);
                '>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <span style='color: #22c55e; font-size: 1rem;'>💾</span>
                        <span style='color: #22c55e; font-size: 0.8rem; font-weight: 600;'>Cache Active</span>
                    </div>
                    <div style='color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;'>
                        {cache_info['item_count']} items • {cache_info['total_size_mb']:.1f} MB
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Clear cache button
            if st.sidebar.button("🗑️ Clear Cache", use_container_width=True, key="clear_cache_btn"):
                clear_session_and_cache()
                st.sidebar.success("Cache cleared!")
                st.rerun()
        else:
            st.sidebar.markdown(
                """
                <div style='
                    margin: 0.5rem 0;
                    padding: 0.75rem;
                    border-radius: 10px;
                    background: rgba(100, 116, 139, 0.1);
                    border: 1px solid rgba(100, 116, 139, 0.2);
                '>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <span style='color: #94a3b8; font-size: 1rem;'>💾</span>
                        <span style='color: #94a3b8; font-size: 0.8rem; font-weight: 600;'>No Cached Data</span>
                    </div>
                    <div style='color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;'>
                        Run data preparation to save results
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    except ImportError:
        # Cache module not available
        pass
    except Exception as e:
        st.sidebar.warning(f"Cache error: {e}")


def render_user_preferences() -> None:
    """
    Renders user preference toggles in the sidebar.

    Features:
    - Quiet Mode: Disables background fluorescent animations
    - Analyst Mode: Shows advanced technical charts and details

    These preferences persist in session state and affect the entire app.
    """
    from app_core.ui.theme import is_quiet_mode, is_analyst_mode

    # Section divider
    sidebar_section_divider("PREFERENCES")

    # Create a styled container for preferences
    st.sidebar.markdown(
        """
        <div style='
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(34, 211, 238, 0.04));
            border: 1px solid rgba(59, 130, 246, 0.15);
        '>
            <div style='
                color: #94a3b8;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.75rem;
            '>Display Settings</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quiet Mode Toggle
    quiet_mode = st.sidebar.toggle(
        "Quiet Mode",
        value=is_quiet_mode(),
        key="quiet_mode_toggle",
        help="Disable background neon animations for a cleaner, less distracting interface"
    )

    # Update session state if changed (using callback pattern)
    if quiet_mode != st.session_state.get("quiet_mode", False):
        st.session_state["quiet_mode"] = quiet_mode
        st.rerun()

    # Analyst Mode Toggle
    analyst_mode = st.sidebar.toggle(
        "Analyst Mode",
        value=is_analyst_mode(),
        key="analyst_mode_toggle",
        help="Show advanced technical details like ACF/PACF plots, model parameters, and diagnostic charts"
    )

    # Update session state if changed
    if analyst_mode != st.session_state.get("analyst_mode", False):
        st.session_state["analyst_mode"] = analyst_mode
        st.rerun()

    # Mode description
    if analyst_mode:
        st.sidebar.markdown(
            """
            <div style='
                margin-top: 0.5rem;
                padding: 0.5rem 0.75rem;
                border-radius: 8px;
                background: rgba(34, 211, 238, 0.1);
                border-left: 3px solid #22d3ee;
                font-size: 0.75rem;
                color: #94a3b8;
            '>
                Advanced charts and technical details are now visible across all pages.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style='
                margin-top: 0.5rem;
                padding: 0.5rem 0.75rem;
                border-radius: 8px;
                background: rgba(100, 116, 139, 0.1);
                border-left: 3px solid #64748b;
                font-size: 0.75rem;
                color: #64748b;
            '>
                Showing simplified Manager View. Enable Analyst Mode for technical details.
            </div>
            """,
            unsafe_allow_html=True
        )
