# =============================================================================
# app_core/errors/__init__.py
# Centralized Error Handling for HealthForecast AI
# =============================================================================

from .exceptions import (
    HealthForecastError,
    DataValidationError,
    DataIngestionError,
    DataFusionError,
    FeatureEngineeringError,
    ModelTrainingError,
    ForecastError,
    OptimizationError,
    ConfigurationError,
)

from .handlers import (
    handle_error,
    safe_execute,
    ErrorContext,
)

__all__ = [
    # Exceptions
    "HealthForecastError",
    "DataValidationError",
    "DataIngestionError",
    "DataFusionError",
    "FeatureEngineeringError",
    "ModelTrainingError",
    "ForecastError",
    "OptimizationError",
    "ConfigurationError",
    # Handlers
    "handle_error",
    "safe_execute",
    "ErrorContext",
]
