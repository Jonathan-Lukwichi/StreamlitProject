"""
Authentication module for HealthForecast AI platform.

⚠️ PROTOTYPE ONLY - NOT FOR PRODUCTION USE
This module provides basic authentication for demonstration purposes.
Passwords are hashed using bcrypt, but the entire authentication flow
is client-side only and provides NO real security.

For production deployment:
- Use Supabase Auth with Row Level Security (RLS)
- Deploy with Streamlit Enterprise authentication
- Implement an auth proxy (e.g., nginx with OAuth2)
"""

import streamlit as st
import streamlit_authenticator as stauth
from typing import Optional, Dict, Any


# ==================== USER CREDENTIALS ====================
# In production, store credentials in a secure database with proper hashing
# These are demo credentials for prototype purposes only

def get_user_credentials() -> Dict[str, Any]:
    """
    Returns user credentials configuration for streamlit-authenticator.

    Default credentials:
    - Admin: username='admin', password='admin123'
    - User 1: username='user1', password='user123'
    - User 2: username='user2', password='user123'

    ⚠️ Change these passwords immediately for any deployment!
    """
    credentials = {
        "usernames": {
            "admin": {
                "name": "Administrator",
                "password": "$2b$12$6eg9XhAKqngO..BwN0De0OPdl1UdKYGEKcIiVvoP84pfRpg7xGQx2",  # admin123
                "role": "admin",
                "email": "admin@healthforecast.ai"
            },
            "user1": {
                "name": "John Smith",
                "password": "$2b$12$qTWdPgBBzeIZVWS3PAPTEO.ZNZdwU.Tzq/bTG.NERFJ/FuiMBoXh.",  # user123
                "role": "user",
                "email": "john.smith@hospital.com"
            },
            "user2": {
                "name": "Jane Doe",
                "password": "$2b$12$qTWdPgBBzeIZVWS3PAPTEO.ZNZdwU.Tzq/bTG.NERFJ/FuiMBoXh.",  # user123
                "role": "user",
                "email": "jane.doe@hospital.com"
            }
        }
    }
    return credentials


# ==================== AUTHENTICATOR SETUP ====================

def get_authenticator():
    """
    Creates and returns a configured streamlit-authenticator instance.

    Returns:
        stauth.Authenticate: Configured authenticator instance
    """
    credentials = get_user_credentials()

    authenticator = stauth.Authenticate(
        credentials,
        cookie_name="healthforecast_auth",
        key="healthforecast_secret_key_2024",  # Change in production!
        cookie_expiry_days=1  # Short expiry for prototype
    )

    return authenticator


# ==================== HELPER FUNCTIONS ====================

def check_authentication() -> bool:
    """
    Check if the current user is authenticated.

    Returns:
        bool: True if user is authenticated, False otherwise
    """
    return st.session_state.get("authenticated", False)


def check_admin_access() -> bool:
    """
    Check if the current user has admin privileges.

    Returns:
        bool: True if user is admin, False otherwise
    """
    if not check_authentication():
        return False

    return st.session_state.get("role") == "admin"


def get_user_role() -> Optional[str]:
    """
    Get the role of the currently authenticated user.

    Returns:
        Optional[str]: User role ('admin' or 'user') or None if not authenticated
    """
    if not check_authentication():
        return None

    return st.session_state.get("role")


def get_username() -> Optional[str]:
    """
    Get the username of the currently authenticated user.

    Returns:
        Optional[str]: Username or None if not authenticated
    """
    if not check_authentication():
        return None

    return st.session_state.get("username")


def get_user_name() -> Optional[str]:
    """
    Get the display name of the currently authenticated user.

    Returns:
        Optional[str]: Display name or None if not authenticated
    """
    if not check_authentication():
        return None

    return st.session_state.get("name")


def logout_user():
    """
    Logout the current user and clear session state.
    """
    # Clear authentication session state
    keys_to_clear = [
        "authenticated",
        "username",
        "name",
        "role",
        "email",
        "authentication_status"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def initialize_session_state():
    """
    Initialize session state variables for authentication.
    Call this at the start of your main app.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "role" not in st.session_state:
        st.session_state.role = None

    if "username" not in st.session_state:
        st.session_state.username = None

    if "name" not in st.session_state:
        st.session_state.name = None


# ==================== PAGE PROTECTION DECORATORS ====================

def require_authentication(redirect_to_welcome: bool = True):
    """
    Decorator to protect pages that require authentication.

    NOTE: Authentication is currently BYPASSED for demo purposes.
    All users are automatically granted access.

    Args:
        redirect_to_welcome: If True, provide a button to go to Welcome page
    """
    # AUTO-AUTHENTICATE: Bypass login for demo mode
    if not check_authentication():
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
    # Authentication bypassed - all pages accessible


def require_admin_access(redirect_to_welcome: bool = True):
    """
    Decorator to protect pages that require admin access.

    NOTE: Admin access is currently BYPASSED for demo purposes.
    All users are automatically granted admin access.

    Args:
        redirect_to_welcome: If True, provide a button to go to Welcome page
    """
    # AUTO-AUTHENTICATE: Bypass login for demo mode
    if not check_authentication():
        st.session_state.authenticated = True
        st.session_state.role = "admin"
        st.session_state.username = "demo_user"
        st.session_state.name = "Demo User"
    # Admin access bypassed - all pages accessible


# ==================== PASSWORD HASHING UTILITY ====================
# Use this to generate new password hashes if needed

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    Use this utility function to generate password hashes for new users.

    Args:
        password: Plain text password to hash

    Returns:
        str: Hashed password

    Example:
        >>> hash_password("mypassword123")
        '$2b$12$...'
    """
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


if __name__ == "__main__":
    # Utility to generate password hashes
    print("Password Hash Generator")
    print("=" * 50)
    print("\n⚠️  Use this to generate password hashes for user credentials")
    print("\nDefault passwords:")
    print("- admin123 -> Hash for admin user")
    print("- user123  -> Hash for regular users")
    print("\n" + "=" * 50)

    # Generate hashes for default passwords
    print(f"\nadmin123: {hash_password('admin123')}")
    print(f"user123:  {hash_password('user123')}")
