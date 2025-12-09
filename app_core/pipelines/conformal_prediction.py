# =============================================================================
# app_core/pipelines/conformal_prediction.py
# Conformalized Quantile Regression (CQR) for Prediction Intervals
# =============================================================================
"""
Implements Conformalized Quantile Regression (CQR) for generating prediction
intervals with guaranteed coverage.

Academic References:
-------------------
- Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression"
  NeurIPS 2019. https://arxiv.org/abs/1905.03222

- Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World"
  Springer. (Foundation of Conformal Prediction)

Key Concepts:
------------
1. CQR uses quantile regression to get initial interval estimates
2. Calibration set computes conformity scores (residuals)
3. Scores are used to adjust intervals for guaranteed coverage
4. Coverage guarantee: P(Y ∈ [lower, upper]) ≥ 1 - α

Author: HealthForecast AI Research Team
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConformalResult:
    """
    Results from Conformal Prediction calibration.

    Attributes
    ----------
    alpha : float
        Significance level (e.g., 0.1 for 90% coverage)
    coverage_target : float
        Target coverage (1 - alpha)
    q_lo : float
        Lower quantile used (e.g., 0.05 for 90% interval)
    q_hi : float
        Upper quantile used (e.g., 0.95 for 90% interval)
    conformity_scores : np.ndarray
        Conformity scores from calibration set
    q_hat : float
        Calibrated quantile adjustment (the key CQR output)
    cal_coverage : float
        Empirical coverage on calibration set
    """
    alpha: float
    coverage_target: float
    q_lo: float
    q_hi: float
    conformity_scores: np.ndarray
    q_hat: float
    cal_coverage: float
    n_cal: int

    @property
    def adjustment_factor(self) -> float:
        """The interval width adjustment from calibration."""
        return self.q_hat


@dataclass
class PredictionInterval:
    """
    Prediction interval results for a set of predictions.

    Attributes
    ----------
    point_forecast : np.ndarray
        Point predictions (mean or median)
    lower : np.ndarray
        Lower bound of prediction interval
    upper : np.ndarray
        Upper bound of prediction interval
    alpha : float
        Significance level
    coverage_target : float
        Target coverage (1 - alpha)
    """
    point_forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    coverage_target: float

    @property
    def width(self) -> np.ndarray:
        """Interval width at each point."""
        return self.upper - self.lower

    @property
    def mean_width(self) -> float:
        """Average interval width."""
        return float(np.mean(self.width))

    def empirical_coverage(self, y_true: np.ndarray) -> float:
        """
        Calculate empirical coverage.

        Parameters
        ----------
        y_true : np.ndarray
            True values

        Returns
        -------
        float
            Proportion of true values within intervals
        """
        covered = (y_true >= self.lower) & (y_true <= self.upper)
        return float(np.mean(covered))


@dataclass
class RPIWResult:
    """
    Relative Prediction Interval Width (RPIW) results.

    RPIW = mean(upper - lower) / range(y_true)

    Lower RPIW is better (tighter intervals relative to data range).

    Attributes
    ----------
    rpiw : float
        Relative Prediction Interval Width
    mean_interval_width : float
        Mean absolute interval width
    data_range : float
        Range of actual values (max - min)
    coverage : float
        Empirical coverage achieved
    coverage_target : float
        Target coverage
    is_valid : bool
        True if coverage >= coverage_target (interval is reliable)
    """
    rpiw: float
    mean_interval_width: float
    data_range: float
    coverage: float
    coverage_target: float

    @property
    def is_valid(self) -> bool:
        """Check if coverage meets target (with small tolerance)."""
        return self.coverage >= (self.coverage_target - 0.02)

    @property
    def coverage_gap(self) -> float:
        """Gap between achieved and target coverage."""
        return self.coverage_target - self.coverage


# =============================================================================
# CONFORMALIZED QUANTILE REGRESSION (CQR)
# =============================================================================

def compute_conformity_scores(
    y_cal: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray
) -> np.ndarray:
    """
    Compute CQR conformity scores on calibration set.

    The conformity score measures how "non-conforming" each observation is
    with respect to the predicted interval. Following Romano et al. (2019):

    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set
    pred_lower : np.ndarray
        Lower quantile predictions on calibration set
    pred_upper : np.ndarray
        Upper quantile predictions on calibration set

    Returns
    -------
    np.ndarray
        Conformity scores for each calibration point
    """
    # Score = max(how much below lower bound, how much above upper bound)
    scores = np.maximum(pred_lower - y_cal, y_cal - pred_upper)
    return scores


def calibrate_conformal(
    y_cal: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    alpha: float = 0.1
) -> ConformalResult:
    """
    Calibrate conformal prediction using the calibration set.

    This implements the key CQR calibration step from Romano et al. (2019).
    The calibrated quantile q_hat is used to adjust prediction intervals
    to achieve the desired coverage.

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set
    pred_lower : np.ndarray
        Lower quantile predictions (e.g., 5th percentile)
    pred_upper : np.ndarray
        Upper quantile predictions (e.g., 95th percentile)
    alpha : float, default=0.1
        Significance level (0.1 = 90% coverage target)

    Returns
    -------
    ConformalResult
        Calibration results including q_hat adjustment

    Notes
    -----
    The key insight of CQR is that by computing conformity scores on the
    calibration set and taking an appropriate quantile, we get a finite-sample
    coverage guarantee.
    """
    n_cal = len(y_cal)

    # Compute conformity scores
    scores = compute_conformity_scores(y_cal, pred_lower, pred_upper)

    # Compute the (1-alpha)(1 + 1/n) quantile of scores
    # This is the key CQR formula for finite-sample coverage
    quantile_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

    q_hat = float(np.quantile(scores, quantile_level))

    # Compute empirical coverage on calibration set (for diagnostics)
    adjusted_lower = pred_lower - q_hat
    adjusted_upper = pred_upper + q_hat
    covered = (y_cal >= adjusted_lower) & (y_cal <= adjusted_upper)
    cal_coverage = float(np.mean(covered))

    return ConformalResult(
        alpha=alpha,
        coverage_target=1 - alpha,
        q_lo=0.5 - (1 - alpha) / 2,  # e.g., 0.05 for 90%
        q_hi=0.5 + (1 - alpha) / 2,  # e.g., 0.95 for 90%
        conformity_scores=scores,
        q_hat=q_hat,
        cal_coverage=cal_coverage,
        n_cal=n_cal
    )


def apply_conformal_correction(
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    conformal_result: ConformalResult,
    point_forecast: Optional[np.ndarray] = None
) -> PredictionInterval:
    """
    Apply conformal correction to get calibrated prediction intervals.

    Parameters
    ----------
    pred_lower : np.ndarray
        Lower quantile predictions on test set
    pred_upper : np.ndarray
        Upper quantile predictions on test set
    conformal_result : ConformalResult
        Calibration results from calibrate_conformal()
    point_forecast : np.ndarray, optional
        Point predictions. If None, uses midpoint of interval.

    Returns
    -------
    PredictionInterval
        Calibrated prediction intervals with coverage guarantee
    """
    q_hat = conformal_result.q_hat

    # Apply the correction: widen intervals by q_hat
    adjusted_lower = pred_lower - q_hat
    adjusted_upper = pred_upper + q_hat

    # Point forecast (use midpoint if not provided)
    if point_forecast is None:
        point_forecast = (adjusted_lower + adjusted_upper) / 2

    return PredictionInterval(
        point_forecast=point_forecast,
        lower=adjusted_lower,
        upper=adjusted_upper,
        alpha=conformal_result.alpha,
        coverage_target=conformal_result.coverage_target
    )


# =============================================================================
# SIMPLE CONFORMAL PREDICTION (for point forecasts)
# =============================================================================

def simple_conformal_intervals(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.1
) -> PredictionInterval:
    """
    Simple conformal prediction for models that only produce point forecasts.

    This is a simpler alternative to CQR when quantile predictions aren't
    available. Uses absolute residuals as conformity scores.

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set
    pred_cal : np.ndarray
        Point predictions on calibration set
    pred_test : np.ndarray
        Point predictions on test set
    alpha : float, default=0.1
        Significance level

    Returns
    -------
    PredictionInterval
        Prediction intervals for test set
    """
    n_cal = len(y_cal)

    # Conformity scores = absolute residuals
    scores = np.abs(y_cal - pred_cal)

    # Compute quantile
    quantile_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    quantile_level = min(quantile_level, 1.0)

    q_hat = float(np.quantile(scores, quantile_level))

    # Create symmetric intervals around point forecast
    lower = pred_test - q_hat
    upper = pred_test + q_hat

    return PredictionInterval(
        point_forecast=pred_test,
        lower=lower,
        upper=upper,
        alpha=alpha,
        coverage_target=1 - alpha
    )


# =============================================================================
# RPIW CALCULATOR
# =============================================================================

def calculate_rpiw(
    y_true: np.ndarray,
    pred_interval: PredictionInterval
) -> RPIWResult:
    """
    Calculate Relative Prediction Interval Width (RPIW).

    RPIW = mean(interval_width) / range(y_true)

    This metric allows comparing interval quality across different datasets
    and models. Lower RPIW is better (tighter intervals).

    Parameters
    ----------
    y_true : np.ndarray
        True values
    pred_interval : PredictionInterval
        Prediction interval results

    Returns
    -------
    RPIWResult
        RPIW calculation results
    """
    # Calculate metrics
    mean_width = pred_interval.mean_width
    data_range = float(np.max(y_true) - np.min(y_true))
    coverage = pred_interval.empirical_coverage(y_true)

    # RPIW calculation (handle edge case of zero range)
    if data_range > 0:
        rpiw = mean_width / data_range
    else:
        rpiw = float('inf')

    return RPIWResult(
        rpiw=rpiw,
        mean_interval_width=mean_width,
        data_range=data_range,
        coverage=coverage,
        coverage_target=pred_interval.coverage_target
    )


def calculate_rpiw_from_arrays(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage_target: float = 0.9
) -> RPIWResult:
    """
    Calculate RPIW directly from arrays (convenience function).

    Parameters
    ----------
    y_true : np.ndarray
        True values
    lower : np.ndarray
        Lower bounds
    upper : np.ndarray
        Upper bounds
    coverage_target : float
        Target coverage for validity check

    Returns
    -------
    RPIWResult
        RPIW calculation results
    """
    # Create interval object
    interval = PredictionInterval(
        point_forecast=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        alpha=1 - coverage_target,
        coverage_target=coverage_target
    )

    return calculate_rpiw(y_true, interval)


# =============================================================================
# MODEL SELECTOR BASED ON RPIW
# =============================================================================

@dataclass
class ModelRanking:
    """
    Model ranking based on RPIW.

    Attributes
    ----------
    model_name : str
        Name of the model
    rpiw : float
        RPIW score (lower is better)
    coverage : float
        Empirical coverage achieved
    is_valid : bool
        Whether coverage meets target
    rank : int
        Rank among all models (1 = best)
    """
    model_name: str
    rpiw: float
    coverage: float
    is_valid: bool
    rank: int = 0


def rank_models_by_rpiw(
    model_results: Dict[str, RPIWResult]
) -> List[ModelRanking]:
    """
    Rank models by RPIW, prioritizing those with valid coverage.

    Ranking logic:
    1. Models with valid coverage (coverage >= target) are ranked first
    2. Among valid models, lower RPIW is better
    3. Invalid models are ranked after valid ones

    Parameters
    ----------
    model_results : Dict[str, RPIWResult]
        Dictionary mapping model names to their RPIW results

    Returns
    -------
    List[ModelRanking]
        Sorted list of model rankings (best first)
    """
    rankings = []

    for name, result in model_results.items():
        rankings.append(ModelRanking(
            model_name=name,
            rpiw=result.rpiw,
            coverage=result.coverage,
            is_valid=result.is_valid
        ))

    # Sort: valid models first, then by RPIW
    rankings.sort(key=lambda x: (not x.is_valid, x.rpiw))

    # Assign ranks
    for i, ranking in enumerate(rankings):
        ranking.rank = i + 1

    return rankings


def select_best_model(
    model_results: Dict[str, RPIWResult],
    require_valid_coverage: bool = True
) -> Tuple[str, RPIWResult]:
    """
    Select the best model based on RPIW.

    Parameters
    ----------
    model_results : Dict[str, RPIWResult]
        Dictionary mapping model names to their RPIW results
    require_valid_coverage : bool, default=True
        If True, only consider models with valid coverage

    Returns
    -------
    Tuple[str, RPIWResult]
        Best model name and its RPIW result

    Raises
    ------
    ValueError
        If no valid models found and require_valid_coverage=True
    """
    rankings = rank_models_by_rpiw(model_results)

    if require_valid_coverage:
        valid_rankings = [r for r in rankings if r.is_valid]
        if not valid_rankings:
            raise ValueError("No models with valid coverage found")
        best = valid_rankings[0]
    else:
        best = rankings[0]

    return best.model_name, model_results[best.model_name]


# =============================================================================
# HIGH-LEVEL CQR PIPELINE
# =============================================================================

class CQRPipeline:
    """
    Complete CQR pipeline for generating calibrated prediction intervals.

    This class provides a high-level interface for:
    1. Calibrating on the calibration set
    2. Generating intervals on the test set
    3. Evaluating interval quality

    Example
    -------
    >>> pipeline = CQRPipeline(alpha=0.1)  # 90% coverage
    >>> pipeline.calibrate(y_cal, pred_lower_cal, pred_upper_cal)
    >>> intervals = pipeline.predict(pred_lower_test, pred_upper_test)
    >>> rpiw = pipeline.evaluate(y_test, intervals)
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize CQR pipeline.

        Parameters
        ----------
        alpha : float, default=0.1
            Significance level (0.1 = 90% coverage)
        """
        self.alpha = alpha
        self.coverage_target = 1 - alpha
        self.conformal_result: Optional[ConformalResult] = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Check if pipeline has been calibrated."""
        return self._is_calibrated

    def calibrate(
        self,
        y_cal: np.ndarray,
        pred_lower: np.ndarray,
        pred_upper: np.ndarray
    ) -> ConformalResult:
        """
        Calibrate the CQR pipeline on calibration data.

        Parameters
        ----------
        y_cal : np.ndarray
            True values on calibration set
        pred_lower : np.ndarray
            Lower quantile predictions
        pred_upper : np.ndarray
            Upper quantile predictions

        Returns
        -------
        ConformalResult
            Calibration results
        """
        self.conformal_result = calibrate_conformal(
            y_cal, pred_lower, pred_upper, self.alpha
        )
        self._is_calibrated = True
        return self.conformal_result

    def calibrate_simple(
        self,
        y_cal: np.ndarray,
        pred_cal: np.ndarray
    ) -> None:
        """
        Calibrate using simple conformal prediction (point forecasts only).

        Parameters
        ----------
        y_cal : np.ndarray
            True values on calibration set
        pred_cal : np.ndarray
            Point predictions on calibration set
        """
        n_cal = len(y_cal)
        scores = np.abs(y_cal - pred_cal)

        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        q_hat = float(np.quantile(scores, quantile_level))

        # Store as conformal result
        self.conformal_result = ConformalResult(
            alpha=self.alpha,
            coverage_target=self.coverage_target,
            q_lo=0.5 - self.coverage_target / 2,
            q_hi=0.5 + self.coverage_target / 2,
            conformity_scores=scores,
            q_hat=q_hat,
            cal_coverage=1.0,  # Will be recalculated
            n_cal=n_cal
        )
        self._is_calibrated = True

    def predict(
        self,
        pred_lower: np.ndarray,
        pred_upper: np.ndarray,
        point_forecast: Optional[np.ndarray] = None
    ) -> PredictionInterval:
        """
        Generate calibrated prediction intervals.

        Parameters
        ----------
        pred_lower : np.ndarray
            Lower quantile predictions on test set
        pred_upper : np.ndarray
            Upper quantile predictions on test set
        point_forecast : np.ndarray, optional
            Point predictions

        Returns
        -------
        PredictionInterval
            Calibrated prediction intervals

        Raises
        ------
        ValueError
            If pipeline not calibrated
        """
        if not self._is_calibrated:
            raise ValueError("Pipeline must be calibrated first. Call calibrate()")

        return apply_conformal_correction(
            pred_lower, pred_upper, self.conformal_result, point_forecast
        )

    def predict_simple(
        self,
        pred_test: np.ndarray
    ) -> PredictionInterval:
        """
        Generate intervals from point forecasts (simple conformal).

        Parameters
        ----------
        pred_test : np.ndarray
            Point predictions on test set

        Returns
        -------
        PredictionInterval
            Prediction intervals
        """
        if not self._is_calibrated:
            raise ValueError("Pipeline must be calibrated first")

        q_hat = self.conformal_result.q_hat
        return PredictionInterval(
            point_forecast=pred_test,
            lower=pred_test - q_hat,
            upper=pred_test + q_hat,
            alpha=self.alpha,
            coverage_target=self.coverage_target
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        intervals: PredictionInterval
    ) -> RPIWResult:
        """
        Evaluate prediction interval quality.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        intervals : PredictionInterval
            Prediction intervals to evaluate

        Returns
        -------
        RPIWResult
            Evaluation metrics including RPIW
        """
        return calculate_rpiw(y_true, intervals)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline state.

        Returns
        -------
        Dict[str, Any]
            Pipeline summary
        """
        if not self._is_calibrated:
            return {
                "is_calibrated": False,
                "alpha": self.alpha,
                "coverage_target": self.coverage_target
            }

        return {
            "is_calibrated": True,
            "alpha": self.alpha,
            "coverage_target": self.coverage_target,
            "q_hat": self.conformal_result.q_hat,
            "n_calibration": self.conformal_result.n_cal,
            "calibration_coverage": self.conformal_result.cal_coverage
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_quantile_predictions(
    model,
    X: np.ndarray,
    quantiles: List[float] = [0.05, 0.5, 0.95]
) -> Dict[str, np.ndarray]:
    """
    Generate quantile predictions from a model.

    This is a utility function that works with sklearn-style quantile
    regression models.

    Parameters
    ----------
    model : object
        Fitted quantile regression model with predict() method
    X : np.ndarray
        Features
    quantiles : List[float]
        Quantiles to predict

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys 'lower', 'median', 'upper'
    """
    # This is a placeholder - actual implementation depends on model type
    # For GradientBoostingRegressor with loss='quantile', you'd need
    # separate models for each quantile

    predictions = {}
    if hasattr(model, 'predict'):
        pred = model.predict(X)
        predictions['median'] = pred
        # For point forecast models, we can't get true quantiles
        # Use placeholder (would need separate quantile models)
        predictions['lower'] = pred
        predictions['upper'] = pred

    return predictions


def create_bootstrap_intervals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.1,
    n_bootstrap: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction intervals using bootstrap residuals.

    An alternative to CQR when a calibration set isn't available.
    Uses bootstrap resampling of residuals.

    Parameters
    ----------
    y_pred : np.ndarray
        Point predictions
    residuals : np.ndarray
        Historical residuals (y_true - y_pred)
    alpha : float
        Significance level
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lower and upper bounds
    """
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Bootstrap residuals
    bootstrap_residuals = np.random.choice(
        residuals, size=(n_bootstrap, len(y_pred)), replace=True
    )

    # Compute quantiles of residuals
    lower_residual = np.percentile(residuals, lower_q * 100)
    upper_residual = np.percentile(residuals, upper_q * 100)

    lower = y_pred + lower_residual
    upper = y_pred + upper_residual

    return lower, upper


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" CONFORMALIZED QUANTILE REGRESSION (CQR) DEMONSTRATION")
    print("=" * 70)

    # Simulate data
    np.random.seed(42)
    n_total = 500
    n_train = 300
    n_cal = 100
    n_test = 100

    # Generate synthetic data
    X = np.linspace(0, 10, n_total)
    y = 2 * X + np.sin(X * 2) + np.random.normal(0, 1, n_total)

    # Split data
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    X_test, y_test = X[n_train+n_cal:], y[n_train+n_cal:]

    # Simulate quantile predictions (in practice, use a quantile regression model)
    # Lower quantile (5th percentile)
    pred_lower_cal = 2 * X_cal + np.sin(X_cal * 2) - 1.5
    pred_lower_test = 2 * X_test + np.sin(X_test * 2) - 1.5

    # Upper quantile (95th percentile)
    pred_upper_cal = 2 * X_cal + np.sin(X_cal * 2) + 1.5
    pred_upper_test = 2 * X_test + np.sin(X_test * 2) + 1.5

    # Point predictions
    pred_cal = (pred_lower_cal + pred_upper_cal) / 2
    pred_test = (pred_lower_test + pred_upper_test) / 2

    print("\n1. DATA SPLIT")
    print(f"   Training:    {n_train} samples")
    print(f"   Calibration: {n_cal} samples")
    print(f"   Test:        {n_test} samples")

    # Demonstrate CQR Pipeline
    print("\n2. CQR CALIBRATION")
    pipeline = CQRPipeline(alpha=0.1)  # 90% coverage
    cal_result = pipeline.calibrate(y_cal, pred_lower_cal, pred_upper_cal)

    print(f"   Coverage target: {cal_result.coverage_target:.0%}")
    print(f"   Calibration q_hat: {cal_result.q_hat:.4f}")
    print(f"   Calibration coverage: {cal_result.cal_coverage:.2%}")

    # Generate intervals
    print("\n3. GENERATE INTERVALS")
    intervals = pipeline.predict(pred_lower_test, pred_upper_test, pred_test)

    print(f"   Mean interval width: {intervals.mean_width:.4f}")
    print(f"   Min width: {np.min(intervals.width):.4f}")
    print(f"   Max width: {np.max(intervals.width):.4f}")

    # Evaluate
    print("\n4. EVALUATE ON TEST SET")
    rpiw_result = pipeline.evaluate(y_test, intervals)

    print(f"   Empirical coverage: {rpiw_result.coverage:.2%}")
    print(f"   RPIW: {rpiw_result.rpiw:.4f}")
    print(f"   Valid (coverage >= target): {rpiw_result.is_valid}")

    # Demonstrate model comparison
    print("\n5. MODEL COMPARISON DEMO")

    # Simulate different models
    model_results = {
        "Model_A": RPIWResult(
            rpiw=0.35, mean_interval_width=5.2, data_range=15.0,
            coverage=0.92, coverage_target=0.90
        ),
        "Model_B": RPIWResult(
            rpiw=0.28, mean_interval_width=4.2, data_range=15.0,
            coverage=0.88, coverage_target=0.90  # Below target
        ),
        "Model_C": RPIWResult(
            rpiw=0.32, mean_interval_width=4.8, data_range=15.0,
            coverage=0.91, coverage_target=0.90
        ),
    }

    rankings = rank_models_by_rpiw(model_results)
    print("   Rankings (lower RPIW = better, valid coverage required):")
    for r in rankings:
        valid_str = "[VALID]" if r.is_valid else "[INVALID]"
        print(f"   #{r.rank}: {r.model_name} - RPIW={r.rpiw:.3f}, "
              f"Coverage={r.coverage:.1%} {valid_str}")

    best_name, best_result = select_best_model(model_results)
    print(f"\n   Best model: {best_name}")

    print("\n" + "=" * 70)
    print(" DEMONSTRATION COMPLETE")
    print("=" * 70)
