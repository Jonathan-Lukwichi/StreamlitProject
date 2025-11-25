"""
Navigation module for role-based sidebar configuration.
Provides logout functionality and user info display in sidebar.

Note: All pages remain visible in sidebar. Page-level authentication checks
(require_admin_access, require_authentication) handle access control.
"""

import streamlit as st


def configure_sidebar_navigation():
    """
    Configure sidebar navigation based on user authentication status and role.

    This is a simplified version that relies on page-level authentication checks.
    All pages show in the sidebar, but unauthorized access attempts are blocked
    by require_admin_access() and require_authentication() decorators.

    For a production system, implement proper server-side navigation control.
    """
    # This function is kept for consistency with the architecture,
    # but the actual navigation control is handled by page-level auth checks
    pass


def add_logout_button():
    """
    Add a logout button to the sidebar.
    This should be called on authenticated pages.
    """
    with st.sidebar:
        st.markdown("---")

        # Show current user info
        if st.session_state.get("authenticated", False):
            user_name = st.session_state.get("name", "User")
            user_role = st.session_state.get("role", "user").title()

            st.markdown(f"""
            <div style='padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 8px; margin-bottom: 1rem;'>
                <div style='font-size: 0.875rem; color: #94a3b8;'>Logged in as</div>
                <div style='font-size: 1rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;'>{user_name}</div>
                <div style='font-size: 0.75rem; color: #a855f7; margin-top: 0.25rem;'>🔑 {user_role}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.switch_page("Welcome.py")


def initialize_navigation():
    """
    Initialize navigation system.
    Call this at the start of every page.
    """
    configure_sidebar_navigation()
