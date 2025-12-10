# =============================================================================
# app_core/errors/exceptions.py
# Custom Exception Hierarchy for HealthForecast AI
# =============================================================================

from typing import Optional, Dict, Any


class HealthForecastError(Exception):
    """
    Base exception for all HealthForecast AI errors.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (e.g., "DATA_001")
        details: Additional context as a dictionary
        recoverable: Whether the error can be recovered from
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "HF_000"
        self.details = details or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


# =============================================================================
# DATA LAYER EXCEPTIONS
# =============================================================================

class DataValidationError(HealthForecastError):
    """Raised when data fails validation checks"""

    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if column:
            details["column"] = column
        if expected:
            details["expected"] = expected
        if actual:
            details["actual"] = actual

        super().__init__(
            message=message,
            code="DATA_001",
            details=details,
            **kwargs,
        )


class DataIngestionError(HealthForecastError):
    """Raised when data ingestion/upload fails"""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if source:
            details["source"] = source
        if file_type:
            details["file_type"] = file_type

        super().__init__(
            message=message,
            code="DATA_002",
            details=details,
            **kwargs,
        )


class DataFusionError(HealthForecastError):
    """Raised when data fusion/merging fails"""

    def __init__(
        self,
        message: str,
        datasets: Optional[list] = None,
        join_key: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if datasets:
            details["datasets"] = datasets
        if join_key:
            details["join_key"] = join_key

        super().__init__(
            message=message,
            code="DATA_003",
            details=details,
            **kwargs,
        )


# =============================================================================
# FEATURE ENGINEERING EXCEPTIONS
# =============================================================================

class FeatureEngineeringError(HealthForecastError):
    """Raised when feature engineering operations fail"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        feature: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        if feature:
            details["feature"] = feature

        super().__init__(
            message=message,
            code="FE_001",
            details=details,
            **kwargs,
        )


# =============================================================================
# MODEL LAYER EXCEPTIONS
# =============================================================================

class ModelTrainingError(HealthForecastError):
    """Raised when model training fails"""

    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if model_type:
            details["model_type"] = model_type
        if hyperparameters:
            details["hyperparameters"] = hyperparameters
        if epoch is not None:
            details["epoch"] = epoch

        super().__init__(
            message=message,
            code="MODEL_001",
            details=details,
            **kwargs,
        )


class ForecastError(HealthForecastError):
    """Raised when forecasting fails"""

    def __init__(
        self,
        message: str,
        horizon: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if horizon:
            details["horizon"] = horizon
        if model_name:
            details["model_name"] = model_name

        super().__init__(
            message=message,
            code="FORECAST_001",
            details=details,
            **kwargs,
        )


# =============================================================================
# OPTIMIZATION EXCEPTIONS
# =============================================================================

class OptimizationError(HealthForecastError):
    """Raised when optimization solver fails"""

    def __init__(
        self,
        message: str,
        solver: Optional[str] = None,
        status: Optional[str] = None,
        constraints_violated: Optional[list] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if solver:
            details["solver"] = solver
        if status:
            details["status"] = status
        if constraints_violated:
            details["constraints_violated"] = constraints_violated

        super().__init__(
            message=message,
            code="OPT_001",
            details=details,
            **kwargs,
        )


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(HealthForecastError):
    """Raised when configuration is invalid or missing"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type

        super().__init__(
            message=message,
            code="CONFIG_001",
            details=details,
            recoverable=False,
            **kwargs,
        )
