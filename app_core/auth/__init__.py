"""
Authentication module for HealthForecast AI platform.
Provides prototype-level user authentication and role-based access control.

⚠️ PROTOTYPE ONLY - NOT FOR PRODUCTION USE
This is a UI-level authentication system for demonstration purposes.
For production deployment, implement proper server-side authentication
with Supabase Auth, Streamlit Enterprise, or an auth proxy.
"""

from .authentication import (
    get_authenticator,
    check_authentication,
    check_admin_access,
    logout_user,
    get_user_role,
)
from .navigation import (
    configure_sidebar_navigation,
    add_logout_button,
    initialize_navigation,
)

__all__ = [
    "get_authenticator",
    "check_authentication",
    "check_admin_access",
    "logout_user",
    "get_user_role",
    "configure_sidebar_navigation",
    "add_logout_button",
    "initialize_navigation",
]
