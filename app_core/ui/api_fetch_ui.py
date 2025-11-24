"""
API Fetch UI Components
Reusable UI components for fetching data from APIs in Data Hub
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from app_core.api.config_manager import APIConfigManager


def render_api_fetch_option(dataset_type: str, key_prefix: str):
    """
    Render API fetch UI for a specific dataset type

    Args:
        dataset_type: 'patient', 'weather', 'calendar', or 'reason'
        key_prefix: Session state key prefix (same as dataset_type)
    """
    st.markdown("---")
    st.markdown("**Or Fetch from API**")

    # Initialize API config manager
    config_manager = APIConfigManager()

    # Get available providers
    providers = config_manager.get_available_providers(dataset_type)

    # Provider selection
    provider = st.selectbox(
        "API Provider",
        options=providers,
        key=f"{key_prefix}_api_provider",
        help=f"Select API source for {dataset_type} data"
    )

    # Date range selection
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            key=f"{key_prefix}_start_date"
        )

    with col_date2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key=f"{key_prefix}_end_date"
        )

    # Provider-specific configuration
    api_params = _render_provider_config(dataset_type, provider, key_prefix)

    # Fetch button
    fetch_btn = st.button(
        f"🔄 Fetch {dataset_type.capitalize()} Data",
        key=f"{key_prefix}_fetch_btn",
        type="primary",
        use_container_width=True
    )

    if fetch_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        with st.spinner(f"Fetching {dataset_type} data from {provider}..."):
            try:
                # Get connector
                if dataset_type == "patient":
                    connector = config_manager.get_patient_connector(provider, **api_params)
                elif dataset_type == "weather":
                    connector = config_manager.get_weather_connector(provider, **api_params)
                elif dataset_type == "calendar":
                    connector = config_manager.get_calendar_connector(provider, **api_params)
                elif dataset_type == "reason":
                    connector = config_manager.get_reason_connector(provider, **api_params)
                else:
                    st.error(f"Unknown dataset type: {dataset_type}")
                    return

                # Fetch data
                df = connector.fetch_data(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    **api_params
                )

                # Store in session state
                st.session_state[f"{key_prefix}_data"] = df
                st.session_state[f"{key_prefix}_loaded"] = True

                # Update specific state keys for compatibility
                if key_prefix == "patient":
                    st.session_state["patient_data"] = df
                elif key_prefix == "weather":
                    st.session_state["weather_data"] = df
                elif key_prefix == "calendar":
                    st.session_state["calendar_data"] = df
                elif key_prefix == "reason":
                    st.session_state["reason_data"] = df

                st.success(f"✅ Fetched {len(df):,} rows from {provider}!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to fetch data: {str(e)}")


def _render_provider_config(dataset_type: str, provider: str, key_prefix: str) -> dict:
    """
    Render provider-specific configuration inputs

    Returns:
        Dict of additional parameters for the API
    """
    params = {}

    # Skip config for mock providers
    if provider == "mock":
        return params

    # Weather API configs
    if dataset_type == "weather":
        if provider == "openweather":
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=40.4168, key=f"{key_prefix}_lat")
            with col2:
                lon = st.number_input("Longitude", value=-3.7038, key=f"{key_prefix}_lon")
            params = {"lat": lat, "lon": lon, "units": "metric"}

        elif provider in ["visualcrossing", "weatherapi"]:
            location = st.text_input(
                "Location",
                value="Madrid,Spain",
                key=f"{key_prefix}_location",
                help="City name or coordinates"
            )
            params = {"location": location}

    # Calendar API configs
    elif dataset_type == "calendar":
        if provider in ["abstractapi", "calendarific"]:
            country = st.text_input(
                "Country Code",
                value="ES",
                key=f"{key_prefix}_country",
                help="ISO 3166-1 alpha-2 country code (e.g., ES, US, FR)"
            )
            params = {"country": country}

    # Patient/Reason API configs (hospital-specific)
    elif dataset_type in ["patient", "reason"]:
        if provider in ["hospital_api", "fhir"]:
            facility_id = st.text_input(
                "Facility ID",
                value="",
                key=f"{key_prefix}_facility_id",
                help="Optional facility/department ID"
            )
            if facility_id:
                params["facility_id"] = facility_id

    return params


def render_api_test_section():
    """
    Render API connection testing section
    Shows status of all configured APIs
    """
    st.markdown("---")
    st.markdown("### 🔌 Test API Connections")

    if st.button("Test All APIs", key="test_all_apis_btn"):
        config_manager = APIConfigManager()

        with st.spinner("Testing API connections..."):
            results = config_manager.test_all_connections()

        # Display results
        for dataset_type, result in results.items():
            status = result.get("status")
            message = result.get("message")

            if status == "success":
                st.success(f"✅ {dataset_type.capitalize()}: {message}")
            else:
                st.error(f"❌ {dataset_type.capitalize()}: {message}")


def render_bulk_api_fetch():
    """
    Render bulk API fetch section for fetching all 4 datasets at once
    """
    st.markdown("---")
    st.markdown("### ⚡ Bulk API Fetch")
    st.caption("Fetch all 4 datasets from APIs in one click")

    # Common date range
    col1, col2 = st.columns(2)
    with col1:
        bulk_start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            key="bulk_start_date"
        )

    with col2:
        bulk_end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="bulk_end_date"
        )

    # Provider selection for each dataset
    st.markdown("**Select API Providers:**")
    col_p, col_w, col_c, col_r = st.columns(4)

    config_manager = APIConfigManager()

    with col_p:
        patient_provider = st.selectbox(
            "Patient",
            options=config_manager.get_available_providers("patient"),
            key="bulk_patient_provider"
        )

    with col_w:
        weather_provider = st.selectbox(
            "Weather",
            options=config_manager.get_available_providers("weather"),
            key="bulk_weather_provider"
        )

    with col_c:
        calendar_provider = st.selectbox(
            "Calendar",
            options=config_manager.get_available_providers("calendar"),
            key="bulk_calendar_provider"
        )

    with col_r:
        reason_provider = st.selectbox(
            "Reason",
            options=config_manager.get_available_providers("reason"),
            key="bulk_reason_provider"
        )

    # Bulk fetch button
    if st.button("🚀 Fetch All Datasets", key="bulk_fetch_btn", type="primary", use_container_width=True):
        if bulk_start_date >= bulk_end_date:
            st.error("Start date must be before end date")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Fetch Patient Data
            status_text.text("Fetching patient data...")
            patient_connector = config_manager.get_patient_connector(patient_provider)
            patient_df = patient_connector.fetch_data(
                start_date=datetime.combine(bulk_start_date, datetime.min.time()),
                end_date=datetime.combine(bulk_end_date, datetime.min.time())
            )
            st.session_state["patient_data"] = patient_df
            st.session_state["patient_loaded"] = True
            progress_bar.progress(25)

            # Fetch Weather Data
            status_text.text("Fetching weather data...")
            weather_connector = config_manager.get_weather_connector(weather_provider)
            weather_df = weather_connector.fetch_data(
                start_date=datetime.combine(bulk_start_date, datetime.min.time()),
                end_date=datetime.combine(bulk_end_date, datetime.min.time())
            )
            st.session_state["weather_data"] = weather_df
            st.session_state["weather_loaded"] = True
            progress_bar.progress(50)

            # Fetch Calendar Data
            status_text.text("Fetching calendar data...")
            calendar_connector = config_manager.get_calendar_connector(calendar_provider)
            calendar_df = calendar_connector.fetch_data(
                start_date=datetime.combine(bulk_start_date, datetime.min.time()),
                end_date=datetime.combine(bulk_end_date, datetime.min.time())
            )
            st.session_state["calendar_data"] = calendar_df
            st.session_state["calendar_loaded"] = True
            progress_bar.progress(75)

            # Fetch Reason Data
            status_text.text("Fetching reason data...")
            reason_connector = config_manager.get_reason_connector(reason_provider)
            reason_df = reason_connector.fetch_data(
                start_date=datetime.combine(bulk_start_date, datetime.min.time()),
                end_date=datetime.combine(bulk_end_date, datetime.min.time())
            )
            st.session_state["reason_data"] = reason_df
            st.session_state["reason_loaded"] = True
            progress_bar.progress(100)

            status_text.text("✅ All datasets fetched successfully!")
            st.success(f"""
            Fetched all 4 datasets:
            - Patient: {len(patient_df):,} rows
            - Weather: {len(weather_df):,} rows
            - Calendar: {len(calendar_df):,} rows
            - Reason: {len(reason_df):,} rows
            """)

            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Bulk fetch failed: {str(e)}")
