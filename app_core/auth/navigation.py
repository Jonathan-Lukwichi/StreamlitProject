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

    NOTE: Currently disabled for demo mode - no logout UI shown.
    """
    # Disabled for demo mode - no logout button or user info displayed
    pass


def initialize_navigation():
    """
    Initialize navigation system.
    Call this at the start of every page.
    """
    configure_sidebar_navigation()
