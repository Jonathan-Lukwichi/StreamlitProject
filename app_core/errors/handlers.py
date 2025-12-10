# =============================================================================
# app_core/errors/handlers.py
# Error Handling Utilities for HealthForecast AI
# =============================================================================

from __future__ import annotations
import functools
import traceback
from typing import Optional, Callable, TypeVar, Any
import streamlit as st

from app_core.logging import get_logger
from .exceptions import HealthForecastError

logger = get_logger(__name__)

T = TypeVar("T")


def handle_error(
    error: Exception,
    show_user_message: bool = True,
    log_error: bool = True,
    user_message: Optional[str] = None,
) -> None:
    """
    Centralized error handling function.

    Args:
        error: The exception to handle
        show_user_message: Whether to display error to user via st.error
        log_error: Whether to log the error
        user_message: Custom message to show user (uses error message if None)
    """
    # Determine message and details
    if isinstance(error, HealthForecastError):
        message = user_message or error.message
        code = error.code
        details = error.details
        recoverable = error.recoverable
    else:
        message = user_message or str(error)
        code = "UNKNOWN"
        details = {"traceback": traceback.format_exc()}
        recoverable = True

    # Log the error
    if log_error:
        logger.error(
            f"[{code}] {message}",
            extra={"details": details},
            exc_info=True,
        )

    # Show to user
    if show_user_message:
        if recoverable:
            st.error(f"Error: {message}")
        else:
            st.error(f"Critical Error: {message}. Please contact support.")

        # Show details in expander for debugging
        if details and st.session_state.get("debug_mode", False):
            with st.expander("Error Details", expanded=False):
                st.json(details)


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    error_message: Optional[str] = None,
    reraise: bool = False,
    **kwargs,
) -> Optional[T]:
    """
    Execute a function with automatic error handling.

    Args:
        func: Function to execute
        *args: Positional arguments to pass to func
        default: Default value to return on error
        error_message: Custom error message to display
        reraise: Whether to reraise the exception after handling
        **kwargs: Keyword arguments to pass to func

    Returns:
        Function result or default value on error

    Usage:
        result = safe_execute(
            risky_function,
            arg1, arg2,
            default=[],
            error_message="Failed to process data"
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, user_message=error_message)
        if reraise:
            raise
        return default


class ErrorContext:
    """
    Context manager for error handling with automatic logging and user feedback.

    Usage:
        with ErrorContext("Training XGBoost model", recoverable=True):
            model.fit(X, y)

        # On error, logs and shows: "Error during: Training XGBoost model"
    """

    def __init__(
        self,
        operation: str,
        recoverable: bool = True,
        show_success: bool = False,
        success_message: Optional[str] = None,
    ):
        self.operation = operation
        self.recoverable = recoverable
        self.show_success = show_success
        self.success_message = success_message

    def __enter__(self) -> ErrorContext:
        logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Error occurred
            if isinstance(exc_val, HealthForecastError):
                handle_error(exc_val)
            else:
                handle_error(
                    exc_val,
                    user_message=f"Error during: {self.operation}",
                )

            # Suppress exception if recoverable
            return self.recoverable
        else:
            # Success
            logger.info(f"Completed: {self.operation}")
            if self.show_success:
                st.success(self.success_message or f"{self.operation} completed")

        return False


def error_boundary(
    default_return: Any = None,
    error_message: Optional[str] = None,
    log: bool = True,
):
    """
    Decorator to wrap functions with error handling.

    Args:
        default_return: Value to return if function fails
        error_message: Custom error message
        log: Whether to log errors

    Usage:
        @error_boundary(default_return=[], error_message="Data loading failed")
        def load_data(path: str) -> List[dict]:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True,
                    )
                if error_message:
                    st.error(error_message)
                return default_return

        return wrapper

    return decorator


def require_data(
    *required_keys: str,
    error_message: str = "Required data not available",
):
    """
    Decorator to check required session state keys before function execution.

    Usage:
        @require_data("merged_data", "processed_df")
        def train_model():
            df = st.session_state["merged_data"]
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = [
                key for key in required_keys
                if key not in st.session_state or st.session_state[key] is None
            ]

            if missing:
                st.warning(f"{error_message}. Missing: {', '.join(missing)}")
                return None

            return func(*args, **kwargs)

        return wrapper

    return decorator
