"""
API Data Ingestion Module
Provides connectors for fetching data from various external APIs
"""

from .base_connector import BaseAPIConnector, APIConfig
from .config_manager import APIConfigManager

from .patient_connector import (
    PatientAPIConnector,
    SupabasePatientConnector,
    MockPatientConnector
)

from .weather_connector import (
    WeatherAPIConnector,
    OpenWeatherConnector,
    VisualCrossingConnector,
    WeatherAPIComConnector,
    MockWeatherConnector
)

from .calendar_connector import (
    CalendarAPIConnector,
    AbstractAPIHolidaysConnector,
    CalendarificConnector,
    MockCalendarConnector
)

from .reason_connector import (
    ReasonAPIConnector,
    SupabaseReasonConnector,
    FHIRReasonConnector,
    MockReasonConnector
)

__all__ = [
    # Base classes
    "BaseAPIConnector",
    "APIConfig",
    "APIConfigManager",

    # Patient connectors
    "PatientAPIConnector",
    "SupabasePatientConnector",
    "MockPatientConnector",

    # Weather connectors
    "WeatherAPIConnector",
    "OpenWeatherConnector",
    "VisualCrossingConnector",
    "WeatherAPIComConnector",
    "MockWeatherConnector",

    # Calendar connectors
    "CalendarAPIConnector",
    "AbstractAPIHolidaysConnector",
    "CalendarificConnector",
    "MockCalendarConnector",

    # Reason connectors
    "ReasonAPIConnector",
    "SupabaseReasonConnector",
    "FHIRReasonConnector",
    "MockReasonConnector",
]
