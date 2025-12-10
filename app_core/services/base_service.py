# =============================================================================
# app_core/services/base_service.py
# Base Service Class with Common Functionality
# =============================================================================

from __future__ import annotations
from abc import ABC
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from app_core.logging import get_logger, LogContext
from app_core.errors import handle_error, HealthForecastError


@dataclass
class ServiceResult:
    """
    Standard result container for service operations.

    Provides consistent structure for all service method returns.
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.success

    @classmethod
    def ok(cls, data: Any = None, metadata: Dict[str, Any] = None) -> ServiceResult:
        """Create a successful result"""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str = "UNKNOWN",
        metadata: Dict[str, Any] = None
    ) -> ServiceResult:
        """Create a failed result"""
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            metadata=metadata,
        )

    @classmethod
    def from_exception(cls, e: Exception) -> ServiceResult:
        """Create a failed result from an exception"""
        if isinstance(e, HealthForecastError):
            return cls(
                success=False,
                error=e.message,
                error_code=e.code,
                metadata=e.details,
            )
        return cls(
            success=False,
            error=str(e),
            error_code="EXCEPTION",
        )


class BaseService(ABC):
    """
    Abstract base class for all services.

    Provides common functionality:
    - Logging
    - Error handling
    - Progress callbacks
    - Result standardization

    Usage:
        class MyService(BaseService):
            def do_something(self) -> ServiceResult:
                with self.log_operation("Doing something"):
                    result = ...
                    return ServiceResult.ok(result)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[int, str], None]
    ) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function that takes (percentage: int, message: str)
        """
        self._progress_callback = callback

    def _update_progress(self, percentage: int, message: str = "") -> None:
        """Update progress via callback if set"""
        if self._progress_callback:
            self._progress_callback(percentage, message)

    def log_operation(self, operation: str) -> LogContext:
        """
        Create a logging context for an operation.

        Usage:
            with self.log_operation("Training model"):
                model.fit(X, y)
        """
        return LogContext(self.logger, operation)

    def safe_execute(
        self,
        operation: str,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ServiceResult:
        """
        Execute a function with error handling and logging.

        Args:
            operation: Description of the operation
            func: Function to execute
            *args, **kwargs: Arguments to pass to func

        Returns:
            ServiceResult with success/failure status
        """
        with self.log_operation(operation):
            try:
                result = func(*args, **kwargs)
                return ServiceResult.ok(result)
            except HealthForecastError as e:
                handle_error(e, show_user_message=False)
                return ServiceResult.from_exception(e)
            except Exception as e:
                self.logger.error(f"{operation} failed: {e}", exc_info=True)
                return ServiceResult.fail(str(e))
