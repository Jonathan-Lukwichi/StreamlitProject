# =============================================================================
# app_core/pipelines/__init__.py
# Automated ML Forecasting Pipeline Module
# =============================================================================
"""
This module provides an automated, academically-rigorous ML forecasting pipeline.

Components:
-----------
1. temporal_split    - Date-based train/calibration/test splitting
2. transform_pipeline - Feature scaling with persistence (coming next)
3. uncertainty       - Prediction intervals with coverage guarantees (coming next)
4. optimization      - RPIW-based optimization method selection (coming next)

Academic References:
-------------------
- Bergmeir & Benitez (2012): Time series cross-validation
- Tashman (2000): Out-of-sample forecasting tests
- Romano et al. (2019): Conformalized Quantile Regression
- Bertsimas & Sim (2004): Robust optimization

Author: HealthForecast AI Research Team
"""

from .temporal_split import (
    compute_temporal_split,
    TemporalSplitResult,
    validate_temporal_split,
)

__all__ = [
    "compute_temporal_split",
    "TemporalSplitResult",
    "validate_temporal_split",
]
