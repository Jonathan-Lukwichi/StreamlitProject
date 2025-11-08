import streamlit as st

# Central registry for session-state keys used across the app.
SESSION_DEFAULTS = {
    "patient_data": None,
    "weather_data": None,
    "calendar_data": None,
    "merged_data": None,
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
}

def init_state():
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
