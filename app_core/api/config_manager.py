"""
API Configuration Manager
Centralized management of API configurations and connector instances
"""
from typing import Dict, Any, Optional, Type
from datetime import datetime
import streamlit as st

from .base_connector import BaseAPIConnector, APIConfig
from .patient_connector import (
    PatientAPIConnector,
    SupabasePatientConnector,
    MockPatientConnector
)
from .weather_connector import (
    OpenWeatherConnector,
    VisualCrossingConnector,
    WeatherAPIComConnector,
    MockWeatherConnector,
    SupabaseWeatherConnector
)
from .calendar_connector import (
    AbstractAPIHolidaysConnector,
    CalendarificConnector,
    MockCalendarConnector,
    SupabaseCalendarConnector
)
from .reason_connector import (
    ReasonAPIConnector,
    SupabaseReasonConnector,
    FHIRReasonConnector,
    MockReasonConnector
)


class APIConfigManager:
    """
    Manages API configurations and creates connector instances

    Usage:
        config_manager = APIConfigManager()
        patient_connector = config_manager.get_patient_connector("mock")
        df = patient_connector.fetch_data(start_date, end_date)
    """

    # Registry of available connectors
    PATIENT_CONNECTORS = {
        "mock": MockPatientConnector,
        "hospital_api": PatientAPIConnector,
        "supabase": SupabasePatientConnector
    }

    WEATHER_CONNECTORS = {
        "mock": MockWeatherConnector,
        "openweather": OpenWeatherConnector,
        "visualcrossing": VisualCrossingConnector,
        "weatherapi": WeatherAPIComConnector,
        "supabase": SupabaseWeatherConnector
    }

    CALENDAR_CONNECTORS = {
        "mock": MockCalendarConnector,
        "abstractapi": AbstractAPIHolidaysConnector,
        "calendarific": CalendarificConnector,
        "supabase": SupabaseCalendarConnector
    }

    REASON_CONNECTORS = {
        "mock": MockReasonConnector,
        "hospital_api": ReasonAPIConnector,
        "supabase": SupabaseReasonConnector,
        "fhir": FHIRReasonConnector
    }

    def __init__(self):
        """Initialize with configurations from Streamlit secrets or defaults"""
        self.configs = self._load_configs_from_secrets()

    def _load_configs_from_secrets(self) -> Dict[str, Dict[str, Any]]:
        """
        Load API configurations from Streamlit secrets

        Expected secrets.toml format:
        [api.patient]
        provider = "hospital_api"
        base_url = "https://hospital.example.com/api"
        api_key = "your_api_key"

        [api.weather]
        provider = "visualcrossing"
        base_url = "https://weather.visualcrossing.com"
        api_key = "your_api_key"
        location = "Madrid,Spain"

        [api.calendar]
        provider = "calendarific"
        base_url = "https://calendarific.com/api"
        api_key = "your_api_key"
        country = "ES"

        [api.reason]
        provider = "hospital_api"
        base_url = "https://hospital.example.com/api"
        api_key = "your_api_key"
        """
        configs = {}

        try:
            if hasattr(st, "secrets") and "api" in st.secrets:
                api_secrets = st.secrets["api"]

                for dataset_type in ["patient", "weather", "calendar", "reason"]:
                    if dataset_type in api_secrets:
                        configs[dataset_type] = dict(api_secrets[dataset_type])
            else:
                # Default to mock connectors if no secrets configured
                configs = self._get_default_configs()

        except Exception:
            # Fallback to default configs
            configs = self._get_default_configs()

        return configs

    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return default configurations (using mock connectors)"""
        return {
            "patient": {"provider": "mock"},
            "weather": {"provider": "mock"},
            "calendar": {"provider": "mock"},
            "reason": {"provider": "mock"}
        }

    def get_patient_connector(self, provider: Optional[str] = None, **kwargs) -> PatientAPIConnector:
        """
        Get patient data connector

        Args:
            provider: Connector type ('mock', 'hospital_api', 'supabase')
            **kwargs: Override configuration parameters

        Returns:
            Configured patient connector instance
        """
        provider = provider or self.configs.get("patient", {}).get("provider", "mock")
        config = self._build_config("patient", provider, kwargs)

        connector_class = self.PATIENT_CONNECTORS.get(provider)
        if not connector_class:
            raise ValueError(f"Unknown patient connector: {provider}")

        return connector_class(config)

    def get_weather_connector(self, provider: Optional[str] = None, **kwargs) -> BaseAPIConnector:
        """
        Get weather data connector

        Args:
            provider: Connector type ('mock', 'openweather', 'visualcrossing', 'weatherapi')
            **kwargs: Override configuration parameters

        Returns:
            Configured weather connector instance
        """
        provider = provider or self.configs.get("weather", {}).get("provider", "mock")
        config = self._build_config("weather", provider, kwargs)

        connector_class = self.WEATHER_CONNECTORS.get(provider)
        if not connector_class:
            raise ValueError(f"Unknown weather connector: {provider}")

        return connector_class(config)

    def get_calendar_connector(self, provider: Optional[str] = None, **kwargs) -> BaseAPIConnector:
        """
        Get calendar data connector

        Args:
            provider: Connector type ('mock', 'abstractapi', 'calendarific')
            **kwargs: Override configuration parameters

        Returns:
            Configured calendar connector instance
        """
        provider = provider or self.configs.get("calendar", {}).get("provider", "mock")
        config = self._build_config("calendar", provider, kwargs)

        connector_class = self.CALENDAR_CONNECTORS.get(provider)
        if not connector_class:
            raise ValueError(f"Unknown calendar connector: {provider}")

        return connector_class(config)

    def get_reason_connector(self, provider: Optional[str] = None, **kwargs) -> ReasonAPIConnector:
        """
        Get reason for visit data connector

        Args:
            provider: Connector type ('mock', 'hospital_api', 'supabase', 'fhir')
            **kwargs: Override configuration parameters

        Returns:
            Configured reason connector instance
        """
        provider = provider or self.configs.get("reason", {}).get("provider", "mock")
        config = self._build_config("reason", provider, kwargs)

        connector_class = self.REASON_CONNECTORS.get(provider)
        if not connector_class:
            raise ValueError(f"Unknown reason connector: {provider}")

        return connector_class(config)

    def _build_config(self, dataset_type: str, provider: str, overrides: Dict[str, Any]) -> APIConfig:
        """Build APIConfig from stored config and overrides"""
        base_config = self.configs.get(dataset_type, {})

        # Merge with overrides
        merged = {**base_config, **overrides}

        return APIConfig(
            api_name=f"{dataset_type}_{provider}",
            base_url=merged.get("base_url", ""),
            api_key=merged.get("api_key"),
            api_secret=merged.get("api_secret"),
            headers=merged.get("headers"),
            timeout=merged.get("timeout", 30),
            rate_limit_delay=merged.get("rate_limit_delay", 0.0),
            additional_params={
                k: v for k, v in merged.items()
                if k not in ["provider", "base_url", "api_key", "api_secret", "headers", "timeout", "rate_limit_delay"]
            }
        )

    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Test connections for all configured APIs

        Returns:
            Dict with test results for each dataset type
        """
        results = {}

        # Test patient API
        try:
            patient_connector = self.get_patient_connector()
            results["patient"] = patient_connector.test_connection()
        except Exception as e:
            results["patient"] = {"status": "error", "message": str(e)}

        # Test weather API
        try:
            weather_connector = self.get_weather_connector()
            results["weather"] = weather_connector.test_connection()
        except Exception as e:
            results["weather"] = {"status": "error", "message": str(e)}

        # Test calendar API
        try:
            calendar_connector = self.get_calendar_connector()
            results["calendar"] = calendar_connector.test_connection()
        except Exception as e:
            results["calendar"] = {"status": "error", "message": str(e)}

        # Test reason API
        try:
            reason_connector = self.get_reason_connector()
            results["reason"] = reason_connector.test_connection()
        except Exception as e:
            results["reason"] = {"status": "error", "message": str(e)}

        return results

    def get_available_providers(self, dataset_type: str) -> list[str]:
        """
        Get list of available providers for a dataset type

        Args:
            dataset_type: 'patient', 'weather', 'calendar', or 'reason'

        Returns:
            List of provider names
        """
        registries = {
            "patient": self.PATIENT_CONNECTORS,
            "weather": self.WEATHER_CONNECTORS,
            "calendar": self.CALENDAR_CONNECTORS,
            "reason": self.REASON_CONNECTORS
        }

        registry = registries.get(dataset_type)
        if not registry:
            return []

        return list(registry.keys())
