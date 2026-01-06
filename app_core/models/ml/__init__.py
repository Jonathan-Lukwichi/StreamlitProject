# =============================================================================
# app_core/models/ml/__init__.py
# Machine Learning Models Package
# =============================================================================

from .base_ml_pipeline import BaseMLPipeline
from .model_registry import MODEL_REGISTRY

# Recursive/Direct Multi-Step Forecasting Strategies (Step 3.2)
from .recursive_strategy import (
    RecursiveForecaster,
    DirectForecaster,
    RectifyForecaster,
    RecursiveForecastResult,
    DirectForecastResult,
    compare_strategies,
)

__all__ = [
    "BaseMLPipeline",
    "MODEL_REGISTRY",
    # Forecasting strategies
    "RecursiveForecaster",
    "DirectForecaster",
    "RectifyForecaster",
    "RecursiveForecastResult",
    "DirectForecastResult",
    "compare_strategies",
]
