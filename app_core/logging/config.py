# =============================================================================
# app_core/logging/config.py
# Logging Configuration for HealthForecast AI
# =============================================================================

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Log format
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory
LOG_DIR = Path("logs")


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to a file
        log_filename: Custom log filename (default: app_YYYY-MM-DD.log)
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        if log_filename is None:
            log_filename = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(LOG_DIR / log_filename)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("supabase").setLevel(logging.WARNING)

    # Log initialization
    logger = logging.getLogger("app_core")
    logger.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance

    Usage:
        from app_core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.error("An error occurred", exc_info=True)
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for logging operation timing and status.

    Usage:
        with LogContext(logger, "Training XGBoost model"):
            model.fit(X, y)
        # Logs: "Training XGBoost model... started"
        # Logs: "Training XGBoost model... completed (2.34s)"
    """

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"{self.operation}... started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"{self.operation}... completed ({elapsed:.2f}s)")
        else:
            self.logger.error(
                f"{self.operation}... failed ({elapsed:.2f}s): {exc_val}",
                exc_info=True
            )

        return False  # Don't suppress exceptions
