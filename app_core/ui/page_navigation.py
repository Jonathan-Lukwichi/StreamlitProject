# =============================================================================
# app_core/ui/page_navigation.py
# Dark Blue Fluorescent Navigation Buttons for Page Navigation
# =============================================================================
from __future__ import annotations
import streamlit as st

# Page order configuration
# Note: Pages 10-13 consolidated into HORIZON Control (01_Dashboard.py)
PAGES = [
    {"file": "pages/01_Dashboard.py", "name": "HORIZON Control", "icon": "🛸"},
    {"file": "pages/02_Upload_Data.py", "name": "Upload Data", "icon": "📁"},
    {"file": "pages/03_Prepare_Data.py", "name": "Prepare Data", "icon": "🔧"},
    {"file": "pages/04_Explore_Data.py", "name": "Explore Data", "icon": "🔍"},
    {"file": "pages/05_Feature_Studio.py", "name": "Feature Studio", "icon": "🛠️"},
    {"file": "pages/06_Feature_Selection.py", "name": "Feature Selection", "icon": "🎯"},
    {"file": "pages/07_Baseline_Models.py", "name": "Baseline Models", "icon": "📈"},
    {"file": "pages/08_Train_Models.py", "name": "Train Models", "icon": "🧠"},
    {"file": "pages/09_Model_Results.py", "name": "Model Results", "icon": "📊"},
]


def get_navigation_css() -> str:
    """Return CSS for dark blue fluorescent navigation buttons."""
    return """
<style>
/* ========================================
   DARK BLUE FLUORESCENT NAVIGATION BUTTONS
   ======================================== */

@keyframes nav-btn-pulse {
    0%, 100% {
        box-shadow:
            0 0 10px rgba(6, 78, 145, 0.4),
            0 0 20px rgba(6, 78, 145, 0.2),
            0 2px 10px rgba(0, 0, 0, 0.3);
    }
    50% {
        box-shadow:
            0 0 15px rgba(6, 78, 145, 0.6),
            0 0 30px rgba(6, 78, 145, 0.3),
            0 4px 15px rgba(0, 0, 0, 0.4);
    }
}

/* Navigation container */
.nav-buttons-container {
    background: linear-gradient(135deg, rgba(6, 78, 145, 0.1) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(6, 78, 145, 0.05) 100%);
    border: 1px solid rgba(6, 78, 145, 0.3);
    border-radius: 16px;
    padding: 1.25rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.nav-buttons-title {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}

/* Override Streamlit button styling for navigation */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #064e91 0%, #0a3d6e 40%, #041e42 100%) !important;
    border: 1px solid rgba(34, 211, 238, 0.3) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #22d3ee !important;
    animation: nav-btn-pulse 3s ease-in-out infinite !important;
    transition: all 0.3s ease !important;
    min-height: 50px !important;
    text-shadow: 0 0 8px rgba(34, 211, 238, 0.4) !important;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(34, 211, 238, 0.6) !important;
    color: #67e8f9 !important;
    text-shadow: 0 0 15px rgba(34, 211, 238, 0.7) !important;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:active {
    transform: translateY(-1px) !important;
}

/* Secondary style for Home/Dashboard buttons */
.nav-home-btn button {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%) !important;
}
</style>
"""


def render_page_navigation(current_page_index: int):
    """
    Render navigation buttons for the current page.

    Args:
        current_page_index: 0-based index of the current page (0 = Dashboard, 8 = Model Results)
                           For legacy pages (index >= len(PAGES)), shows Home/HORIZON only.
    """
    # Inject CSS
    st.markdown(get_navigation_css(), unsafe_allow_html=True)

    # Navigation container
    st.markdown("""
    <div class='nav-buttons-container'>
        <div class='nav-buttons-title'>Quick Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    # Remove the container div and use columns directly
    st.markdown("<div style='margin-top: -1rem;'></div>", unsafe_allow_html=True)

    # Handle out-of-range indices (legacy _old_ pages)
    is_legacy_page = current_page_index >= len(PAGES)

    # Create button columns
    col1, col2, col3, col4 = st.columns(4)

    # Previous button
    with col1:
        if is_legacy_page:
            # Legacy pages: go to last main page
            last_page = PAGES[-1]
            if st.button(f"◀ {last_page['name']}", key="nav_prev", use_container_width=True):
                st.switch_page(last_page["file"])
        elif current_page_index > 0:
            prev_page = PAGES[current_page_index - 1]
            if st.button(f"◀ {prev_page['name']}", key="nav_prev", use_container_width=True):
                st.switch_page(prev_page["file"])
        else:
            st.button("◀ Previous", key="nav_prev_disabled", use_container_width=True, disabled=True)

    # Home/Landing button
    with col2:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.switch_page("Welcome.py")

    # HORIZON Control button
    with col3:
        if current_page_index != 0:
            if st.button("🛸 HORIZON", key="nav_dashboard", use_container_width=True):
                st.switch_page("pages/01_Dashboard.py")
        else:
            st.button("🛸 HORIZON", key="nav_dashboard_current", use_container_width=True, disabled=True)

    # Next button
    with col4:
        if is_legacy_page:
            # Legacy pages: no next, show disabled
            st.button("Next ▶", key="nav_next_disabled", use_container_width=True, disabled=True)
        elif current_page_index < len(PAGES) - 1:
            next_page = PAGES[current_page_index + 1]
            if st.button(f"{next_page['name']} ▶", key="nav_next", use_container_width=True):
                st.switch_page(next_page["file"])
        else:
            st.button("Next ▶", key="nav_next_disabled", use_container_width=True, disabled=True)


def render_compact_navigation(current_page_index: int):
    """
    Render a more compact navigation bar.

    Args:
        current_page_index: 0-based index of the current page
                           For legacy pages (index >= len(PAGES)), shows Home/HORIZON only.
    """
    st.markdown(get_navigation_css(), unsafe_allow_html=True)

    st.markdown("---")

    # Handle out-of-range indices (legacy _old_ pages)
    is_legacy_page = current_page_index >= len(PAGES)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        if is_legacy_page:
            # Legacy pages: go to last main page
            last_page = PAGES[-1]
            if st.button(f"◀ Prev", key="nav_prev_c", use_container_width=True, help=f"Go to {last_page['name']}"):
                st.switch_page(last_page["file"])
        elif current_page_index > 0:
            prev_page = PAGES[current_page_index - 1]
            if st.button(f"◀ Prev", key="nav_prev_c", use_container_width=True, help=f"Go to {prev_page['name']}"):
                st.switch_page(prev_page["file"])
        else:
            st.button("◀ Prev", key="nav_prev_c_dis", use_container_width=True, disabled=True)

    with col2:
        if st.button("🏠 Home", key="nav_home_c", use_container_width=True, help="Go to Welcome page"):
            st.switch_page("Welcome.py")

    with col3:
        # Show current page indicator
        if is_legacy_page:
            st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem; background: rgba(6, 78, 145, 0.2);
                        border-radius: 8px; border: 1px solid rgba(34, 211, 238, 0.3);'>
                <span style='color: #22d3ee; font-size: 0.8rem; font-weight: 600;'>
                    📄 Legacy
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            current = PAGES[current_page_index]
            st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem; background: rgba(6, 78, 145, 0.2);
                        border-radius: 8px; border: 1px solid rgba(34, 211, 238, 0.3);'>
                <span style='color: #22d3ee; font-size: 0.8rem; font-weight: 600;'>
                    {current['icon']} {current_page_index + 1}/{len(PAGES)}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if current_page_index != 0:
            if st.button("🛸 HORIZON", key="nav_dash_c", use_container_width=True, help="Go to HORIZON Control"):
                st.switch_page("pages/01_Dashboard.py")
        else:
            st.button("🛸 HORIZON", key="nav_dash_c_dis", use_container_width=True, disabled=True)

    with col5:
        if is_legacy_page:
            st.button("Next ▶", key="nav_next_c_dis", use_container_width=True, disabled=True)
        elif current_page_index < len(PAGES) - 1:
            next_page = PAGES[current_page_index + 1]
            if st.button(f"Next ▶", key="nav_next_c", use_container_width=True, help=f"Go to {next_page['name']}"):
                st.switch_page(next_page["file"])
        else:
            st.button("Next ▶", key="nav_next_c_dis", use_container_width=True, disabled=True)
