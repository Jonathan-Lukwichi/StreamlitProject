import streamlit as st

# Central registry for session-state keys used across the app.
SESSION_DEFAULTS = {
    "patient_data": None,
    "weather_data": None,
    "calendar_data": None,
    "merged_data": None,
    "processed_df": None,
    "patient_loaded": False,
    "weather_loaded": False,
    "calendar_loaded": False,
    "fusion_log": [],
    "nav_page": "🏠 Dashboard",
    "_nav_intent": None,
    "_eda_dow_quick": None,
    "selected_model": None,
    "model_results": None,
    "arima_results": None,
    "arima_params": {"p": 1, "d": 1, "q": 1},
    # Cache status tracking
    "_cache_loaded": False,
}


def _load_cached_data():
    """
    Attempt to load cached session data from disk.
    Returns True if any data was loaded, False otherwise.
    """
    try:
        from app_core.cache import get_cache_manager, has_cached_data

        if not has_cached_data():
            return False

        cache_mgr = get_cache_manager()
        loaded = cache_mgr.load_session_state()

        if loaded:
            for key, value in loaded.items():
                st.session_state[key] = value
            return True
        return False

    except ImportError:
        # Cache module not available
        return False
    except Exception as e:
        print(f"Cache load error: {e}")
        return False


def init_state():
    """Initialize session state with defaults and optionally load from cache."""
    # First, set all defaults
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Try to load from cache if not already loaded
    if not st.session_state.get("_cache_loaded", False):
        if _load_cached_data():
            st.session_state["_cache_loaded"] = True


def clear_session_and_cache():
    """Clear both session state and disk cache."""
    try:
        from app_core.cache import clear_cache
        clear_cache()
    except ImportError:
        pass

    # Clear session state (except auth)
    auth_keys = ["authenticated", "username", "name", "role", "email", "authentication_status"]
    for key in list(st.session_state.keys()):
        if key not in auth_keys:
            del st.session_state[key]

    # Re-initialize defaults
    for k, v in SESSION_DEFAULTS.items():
        st.session_state[k] = v
