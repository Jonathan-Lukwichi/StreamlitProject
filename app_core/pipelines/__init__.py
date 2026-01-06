# =============================================================================
# app_core/pipelines/__init__.py
# Automated ML Forecasting Pipeline Module
# =============================================================================
"""
This module provides an automated, academically-rigorous ML forecasting pipeline.

Components:
-----------
1. temporal_split        - Date-based train/calibration/test splitting
2. conformal_prediction  - CQR prediction intervals with coverage guarantees
3. transform_pipeline    - Feature scaling with persistence (in Feature Engineering)

Academic References:
-------------------
- Bergmeir & Benitez (2012): Time series cross-validation
- Tashman (2000): Out-of-sample forecasting tests
- Romano et al. (2019): Conformalized Quantile Regression
- Bertsimas & Sim (2004): Robust optimization

Author: HealthForecast AI Research Team
"""

from .temporal_split import (
    # Single train/cal/test split
    compute_temporal_split,
    TemporalSplitResult,
    validate_temporal_split,
    split_dataframe,
    # Time series cross-validation
    CVFoldResult,
    CVConfig,
    create_cv_folds,
    visualize_cv_folds,
    evaluate_cv_metrics,
    format_cv_metric,
)

from .conformal_prediction import (
    # Data structures
    ConformalResult,
    PredictionInterval,
    RPIWResult,
    ModelRanking,
    # Core CQR functions
    calibrate_conformal,
    apply_conformal_correction,
    compute_conformity_scores,
    simple_conformal_intervals,
    # RPIW functions
    calculate_rpiw,
    calculate_rpiw_from_arrays,
    rank_models_by_rpiw,
    select_best_model,
    # Pipeline class
    CQRPipeline,
)

__all__ = [
    # Temporal split (single)
    "compute_temporal_split",
    "TemporalSplitResult",
    "validate_temporal_split",
    "split_dataframe",
    # Time series cross-validation
    "CVFoldResult",
    "CVConfig",
    "create_cv_folds",
    "visualize_cv_folds",
    "evaluate_cv_metrics",
    "format_cv_metric",
    # Conformal prediction
    "ConformalResult",
    "PredictionInterval",
    "RPIWResult",
    "ModelRanking",
    "calibrate_conformal",
    "apply_conformal_correction",
    "compute_conformity_scores",
    "simple_conformal_intervals",
    "calculate_rpiw",
    "calculate_rpiw_from_arrays",
    "rank_models_by_rpiw",
    "select_best_model",
    "CQRPipeline",
]
