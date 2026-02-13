# =============================================================================
# statistical_tests.py - PhD-Level Statistical Testing for Forecast Evaluation
# =============================================================================
"""
Statistical tests and metrics for rigorous forecast model comparison.

This module implements:
1. Diebold-Mariano test for pairwise forecast comparison
2. Naive baseline forecasts (persistence, seasonal naive, moving average)
3. Forecast skill scores (MASE, Theil's U, relative skill)
4. Bootstrap confidence intervals for metrics
5. Multiple comparison corrections (Bonferroni, Benjamini-Hochberg)

References:
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of
  forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, t as t_dist
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DieboldMarianoResult:
    """Result of Diebold-Mariano test."""
    statistic: float
    p_value: float
    conclusion: str
    significance_level: str  # "***", "**", "*", "ns"
    model_1_name: str
    model_2_name: str
    better_model: Optional[str]

    def __repr__(self) -> str:
        return f"DM Test: {self.model_1_name} vs {self.model_2_name} | stat={self.statistic:.3f}, p={self.p_value:.4f} {self.significance_level}"


@dataclass
class SkillScoreResult:
    """Result of skill score calculation."""
    model_name: str
    skill_score: float  # 1 - (model_error / baseline_error)
    mase: float  # Mean Absolute Scaled Error
    theils_u: float  # Theil's U statistic
    improvement_pct: float  # Percentage improvement over baseline
    interpretation: str  # Human-readable interpretation
    baseline_type: str  # "naive", "seasonal_naive", "moving_average"


@dataclass
class BootstrapCIResult:
    """Result of bootstrap confidence interval."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


# =============================================================================
# DIEBOLD-MARIANO TEST
# =============================================================================

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss_type: str = "squared",
    alternative: str = "two-sided"
) -> DieboldMarianoResult:
    """
    Diebold-Mariano test for comparing predictive accuracy of two forecasts.

    H0: E[d_t] = 0 (equal predictive accuracy)
    H1: E[d_t] != 0 (unequal predictive accuracy)

    Parameters:
    -----------
    e1 : np.ndarray
        Forecast errors from model 1 (actual - predicted)
    e2 : np.ndarray
        Forecast errors from model 2 (actual - predicted)
    h : int
        Forecast horizon (for autocorrelation adjustment)
    loss_type : str
        Loss function: "squared" (MSE) or "absolute" (MAE)
    alternative : str
        "two-sided", "greater" (e1 > e2), or "less" (e1 < e2)

    Returns:
    --------
    DieboldMarianoResult with test statistic, p-value, and conclusion

    Notes:
    ------
    Uses Newey-West HAC estimator for variance when h > 1 to account
    for autocorrelation in multi-step ahead forecasts.
    """
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()

    if len(e1) != len(e2):
        raise ValueError("Error arrays must have same length")

    T = len(e1)

    if T < 10:
        warnings.warn("Sample size < 10: DM test may be unreliable")

    # Compute loss differential
    if loss_type == "squared":
        d = e1**2 - e2**2
    elif loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    d_mean = np.mean(d)

    # Compute variance with Newey-West HAC estimator
    gamma_0 = np.var(d, ddof=1)

    # Autocorrelation adjustment for multi-step forecasts
    gamma_sum = 0.0
    if h > 1:
        for k in range(1, h):
            if k < T:
                # Autocovariance at lag k
                gamma_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
                gamma_sum += 2 * gamma_k

    variance = (gamma_0 + gamma_sum) / T

    # Handle edge case of zero variance
    if variance <= 0:
        return DieboldMarianoResult(
            statistic=0.0,
            p_value=1.0,
            conclusion="Cannot compute: zero variance in loss differential",
            significance_level="ns",
            model_1_name="Model 1",
            model_2_name="Model 2",
            better_model=None
        )

    # DM statistic (asymptotically N(0,1) under H0)
    dm_stat = d_mean / np.sqrt(variance)

    # Compute p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(dm_stat)
    elif alternative == "less":
        p_value = norm.cdf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Determine significance level
    if p_value < 0.001:
        sig_level = "***"
    elif p_value < 0.01:
        sig_level = "**"
    elif p_value < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"

    # Determine which model is better
    better_model = None
    if p_value < 0.05:
        if d_mean < 0:  # e1 has lower loss
            better_model = "Model 1"
            conclusion = "Model 1 significantly outperforms Model 2"
        else:
            better_model = "Model 2"
            conclusion = "Model 2 significantly outperforms Model 1"
    else:
        conclusion = "No significant difference in predictive accuracy"

    return DieboldMarianoResult(
        statistic=dm_stat,
        p_value=p_value,
        conclusion=conclusion,
        significance_level=sig_level,
        model_1_name="Model 1",
        model_2_name="Model 2",
        better_model=better_model
    )


def diebold_mariano_matrix(
    model_data: Dict[str, Dict], # Renamed from errors_dict for clarity, expects {'model_name': {'y_true': array, 'residuals': array}}
    h: int = 1,
    loss_type: str = "squared"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Diebold-Mariano test matrix for multiple models.

    Parameters:
    -----------
    model_data : Dict[str, Dict]
        Dictionary mapping model names to their prediction data:
        e.g., {'Model Name': {'y_true': np.array, 'residuals': np.array}}
    h : int
        Forecast horizon
    loss_type : str
        Loss function type

    Returns:
    --------
    Tuple of (DataFrame with p-values, DataFrame with significance levels)
    """
    model_names = list(model_data.keys())
    n_models = len(model_names)

    if n_models < 2:
        return pd.DataFrame(index=model_names, columns=model_names), pd.DataFrame(index=model_names, columns=model_names)

    # Find the common observation period for all models' y_true
    y_true_arrays = {name: data['y_true'] for name, data in model_data.items()}
    
    # Determine the minimum length among all y_true arrays
    min_common_length = min(len(arr) for arr in y_true_arrays.values()) if y_true_arrays else 0

    if min_common_length == 0:
        return pd.DataFrame(index=model_names, columns=model_names), pd.DataFrame(index=model_names, columns=model_names)

    # Now, trim all residuals to this minimum common length
    # This also re-aligns them implicitly because they correspond to the same y_true prefix
    aligned_residuals = {
        name: data['residuals'][:min_common_length]
        for name, data in model_data.items()
    }

    # Initialize matrix
    p_matrix = pd.DataFrame(
        index=model_names,
        columns=model_names,
        dtype=float
    )
    sig_matrix = pd.DataFrame(
        index=model_names,
        columns=model_names,
        dtype=str
    )

    for i, m1_name in enumerate(model_names):
        for j, m2_name in enumerate(model_names):
            if i == j:
                p_matrix.loc[m1_name, m2_name] = np.nan
                sig_matrix.loc[m1_name, m2_name] = "—"
            else:
                # Use the aligned residuals
                e1_aligned = aligned_residuals[m1_name]
                e2_aligned = aligned_residuals[m2_name]

                # If after alignment, any array is too short, skip comparison
                if len(e1_aligned) < h or len(e2_aligned) < h: # DM test needs at least h observations
                    p_matrix.loc[m1_name, m2_name] = np.nan
                    sig_matrix.loc[m1_name, m2_name] = "N/A (data too short)"
                    continue

                try:
                    # Pass default model names to DM test, it's just for internal conclusion string
                    result = diebold_mariano_test(
                        e1_aligned,
                        e2_aligned,
                        h=h,
                        loss_type=loss_type,
                        model_1_name=m1_name, # Pass model names
                        model_2_name=m2_name  # Pass model names
                    )
                    p_matrix.loc[m1_name, m2_name] = result.p_value
                    sig_matrix.loc[m1_name, m2_name] = result.significance_level
                except Exception as e:
                    # Catch any other exceptions from DM test, e.g., zero variance
                    p_matrix.loc[m1_name, m2_name] = np.nan
                    sig_matrix.loc[m1_name, m2_name] = f"Error: {str(e)[:15]}..." # Truncate error message

    return p_matrix, sig_matrix


# =============================================================================
# NAIVE BASELINE FORECASTS
# =============================================================================

def compute_naive_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    method: str = "persistence"
) -> Tuple[np.ndarray, str]:
    """
    Compute naive baseline forecast.

    Parameters:
    -----------
    y_train : np.ndarray
        Training series
    y_test : np.ndarray
        Test series (for computing forecast length)
    method : str
        "persistence" - Last value repeated
        "seasonal_naive" - Same day from previous week
        "moving_average" - Rolling mean of last 7 days
        "drift" - Linear extrapolation

    Returns:
    --------
    Tuple of (forecast array, method description)
    """
    y_train = np.asarray(y_train).flatten()
    n_test = len(y_test)

    if method == "persistence":
        # Naive: Use last observed value
        forecast = np.full(n_test, y_train[-1])
        description = "Naive (persistence): Uses last observed value"

    elif method == "seasonal_naive":
        # Seasonal naive: Use value from same day last week
        season_length = 7
        if len(y_train) >= season_length:
            # Repeat the last season_length values
            last_season = y_train[-season_length:]
            n_repeats = (n_test // season_length) + 1
            forecast = np.tile(last_season, n_repeats)[:n_test]
        else:
            forecast = np.full(n_test, y_train[-1])
        description = "Seasonal Naive: Uses value from same weekday last week"

    elif method == "moving_average":
        # Moving average of last 7 days
        window = min(7, len(y_train))
        ma_value = np.mean(y_train[-window:])
        forecast = np.full(n_test, ma_value)
        description = f"Moving Average ({window}-day): Uses average of last {window} days"

    elif method == "drift":
        # Drift method: linear extrapolation
        n_train = len(y_train)
        drift = (y_train[-1] - y_train[0]) / (n_train - 1) if n_train > 1 else 0
        forecast = y_train[-1] + drift * np.arange(1, n_test + 1)
        description = "Drift: Linear extrapolation from training trend"

    else:
        raise ValueError(f"Unknown baseline method: {method}")

    return forecast, description


def compute_all_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute all baseline forecasts and their metrics.

    Returns dictionary with forecast, metrics, and description for each method.
    """
    y_test = np.asarray(y_test).flatten()
    baselines = {}

    for method in ["persistence", "seasonal_naive", "moving_average", "drift"]:
        forecast, description = compute_naive_baseline(y_train, y_test, method)
        errors = y_test - forecast

        baselines[method] = {
            "forecast": forecast,
            "errors": errors,
            "description": description,
            "mae": np.mean(np.abs(errors)),
            "rmse": np.sqrt(np.mean(errors**2)),
            "mape": np.mean(np.abs(errors / y_test)) * 100 if np.all(y_test != 0) else np.nan,
        }

    return baselines


# =============================================================================
# FORECAST SKILL SCORES
# =============================================================================

def compute_skill_score(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    baseline_type: str = "naive"
) -> SkillScoreResult:
    """
    Compute forecast skill score and related metrics.

    Parameters:
    -----------
    model_errors : np.ndarray
        Errors from the model (actual - predicted)
    baseline_errors : np.ndarray
        Errors from the baseline forecast
    y_test : np.ndarray
        Actual test values
    model_name : str
        Name of the model
    baseline_type : str
        Type of baseline used

    Returns:
    --------
    SkillScoreResult with comprehensive skill metrics
    """
    model_errors = np.asarray(model_errors).flatten()
    baseline_errors = np.asarray(baseline_errors).flatten()
    y_test = np.asarray(y_test).flatten()

    # Model metrics
    model_mae = np.mean(np.abs(model_errors))
    model_rmse = np.sqrt(np.mean(model_errors**2))

    # Baseline metrics
    baseline_mae = np.mean(np.abs(baseline_errors))
    baseline_rmse = np.sqrt(np.mean(baseline_errors**2))

    # Skill Score: 1 - (model_error / baseline_error)
    # Positive = model is better, Negative = baseline is better
    skill_score = 1 - (model_rmse / baseline_rmse) if baseline_rmse > 0 else 0

    # MASE (Mean Absolute Scaled Error)
    # Uses in-sample naive errors for scaling
    mase = model_mae / baseline_mae if baseline_mae > 0 else np.inf

    # Theil's U statistic
    theils_u = model_rmse / baseline_rmse if baseline_rmse > 0 else np.inf

    # Percentage improvement
    improvement_pct = ((baseline_rmse - model_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0

    # Interpretation
    if skill_score > 0.3:
        interpretation = "Substantial improvement over baseline"
    elif skill_score > 0.1:
        interpretation = "Moderate improvement over baseline"
    elif skill_score > 0:
        interpretation = "Marginal improvement over baseline"
    elif skill_score > -0.1:
        interpretation = "Similar to baseline"
    else:
        interpretation = "Worse than baseline - consider using simpler model"

    return SkillScoreResult(
        model_name=model_name,
        skill_score=skill_score,
        mase=mase,
        theils_u=theils_u,
        improvement_pct=improvement_pct,
        interpretation=interpretation,
        baseline_type=baseline_type
    )


def compute_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute directional accuracy (percentage of correct direction predictions).

    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    float: Percentage of correct direction predictions (0-100)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) < 2:
        return np.nan

    # Compute actual and predicted changes
    actual_change = np.diff(y_true)
    pred_change = np.diff(y_pred)

    # Direction is correct if signs match or both are zero
    correct = np.sign(actual_change) == np.sign(pred_change)

    return np.mean(correct) * 100


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_interval(
    residuals: np.ndarray,
    metric: str = "rmse",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals (actual - predicted)
    metric : str
        "rmse", "mae", "mape" (requires y_true for MAPE)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    BootstrapCIResult with point estimate and confidence interval
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    if random_state is not None:
        np.random.seed(random_state)

    # Compute point estimate
    if metric == "rmse":
        point_estimate = np.sqrt(np.mean(residuals**2))
        metric_func = lambda r: np.sqrt(np.mean(r**2))
    elif metric == "mae":
        point_estimate = np.mean(np.abs(residuals))
        metric_func = lambda r: np.mean(np.abs(r))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Bootstrap
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        boot_sample = np.random.choice(residuals, size=n, replace=True)
        bootstrap_values.append(metric_func(boot_sample))

    bootstrap_values = np.array(bootstrap_values)

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha/2) * 100)

    return BootstrapCIResult(
        metric_name=metric.upper(),
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )


# =============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# =============================================================================

def apply_multiple_comparison_correction(
    p_values: List[float],
    method: str = "benjamini-hochberg"
) -> Tuple[List[float], List[bool]]:
    """
    Apply correction for multiple comparisons.

    Parameters:
    -----------
    p_values : List[float]
        Raw p-values from multiple tests
    method : str
        "bonferroni" - Conservative family-wise error rate control
        "benjamini-hochberg" - False discovery rate control (recommended)

    Returns:
    --------
    Tuple of (corrected p-values, rejection decisions at alpha=0.05)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    alpha = 0.05

    if method == "bonferroni":
        # Bonferroni: multiply all p-values by number of tests
        corrected = np.minimum(p_values * n, 1.0)
        rejected = corrected < alpha

    elif method == "benjamini-hochberg":
        # Benjamini-Hochberg procedure for FDR control
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Compute BH critical values
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_p):
            corrected[sorted_idx[i]] = min(p * n / (i + 1), 1.0)

        # Make corrections monotonic
        for i in range(n - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1] if i < n - 1 else corrected[i])

        rejected = corrected < alpha

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist(), rejected.tolist()


# =============================================================================
# BIAS TESTING
# =============================================================================

def test_forecast_bias(
    residuals: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, str, bool]]:
    """
    Test for systematic forecast bias (over/under-forecasting).

    H0: Mean residual = 0 (no bias)
    H1: Mean residual != 0 (systematic bias)

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals (actual - predicted)
    alpha : float
        Significance level

    Returns:
    --------
    Dictionary with test results
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    # One-sample t-test against mean = 0
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1)
    se = std_residual / np.sqrt(n)

    t_stat = mean_residual / se if se > 0 else 0
    p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n-1))

    # Interpret
    if p_value < alpha:
        if mean_residual > 0:
            bias_type = "under-forecasting"
            interpretation = f"Significant under-forecasting bias: model underestimates by {mean_residual:.2f} on average"
        else:
            bias_type = "over-forecasting"
            interpretation = f"Significant over-forecasting bias: model overestimates by {-mean_residual:.2f} on average"
        has_bias = True
    else:
        bias_type = "none"
        interpretation = "No significant forecast bias detected"
        has_bias = False

    return {
        "mean_error": mean_residual,
        "t_statistic": t_stat,
        "p_value": p_value,
        "has_bias": has_bias,
        "bias_type": bias_type,
        "interpretation": interpretation,
        "significant": p_value < alpha
    }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR STREAMLIT INTEGRATION
# =============================================================================

def evaluate_model_vs_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Comprehensive evaluation of a model against all baselines.

    Returns dictionary with all skill scores and statistical tests.
    """
    y_test = np.asarray(y_test).flatten()
    y_pred = np.asarray(y_pred).flatten()
    model_errors = y_test - y_pred

    # Get all baselines
    baselines = compute_all_baselines(y_train, y_test)

    # Evaluate against each baseline
    results = {
        "model_name": model_name,
        "model_mae": np.mean(np.abs(model_errors)),
        "model_rmse": np.sqrt(np.mean(model_errors**2)),
        "model_mape": np.mean(np.abs(model_errors / y_test)) * 100 if np.all(y_test != 0) else np.nan,
        "directional_accuracy": compute_directional_accuracy(y_test, y_pred),
        "baselines": {},
        "skill_scores": {},
        "dm_tests": {},
        "bias_test": test_forecast_bias(model_errors)
    }

    # Compare to each baseline
    for baseline_name, baseline_data in baselines.items():
        results["baselines"][baseline_name] = baseline_data

        # Skill score
        skill = compute_skill_score(
            model_errors,
            baseline_data["errors"],
            y_test,
            model_name=model_name,
            baseline_type=baseline_name
        )
        results["skill_scores"][baseline_name] = {
            "skill_score": skill.skill_score,
            "mase": skill.mase,
            "theils_u": skill.theils_u,
            "improvement_pct": skill.improvement_pct,
            "interpretation": skill.interpretation
        }

        # DM test
        dm_result = diebold_mariano_test(model_errors, baseline_data["errors"])
        results["dm_tests"][baseline_name] = {
            "statistic": dm_result.statistic,
            "p_value": dm_result.p_value,
            "significance": dm_result.significance_level,
            "conclusion": dm_result.conclusion
        }

    # Bootstrap CI for model RMSE
    ci = bootstrap_confidence_interval(model_errors, metric="rmse", n_bootstrap=1000)
    results["rmse_95ci"] = {
        "lower": ci.ci_lower,
        "upper": ci.ci_upper,
        "point": ci.point_estimate
    }

    return results


def format_p_value_display(p_value: float) -> str:
    """Format p-value for display."""
    if p_value < 0.001:
        return "<0.001***"
    elif p_value < 0.01:
        return f"{p_value:.3f}**"
    elif p_value < 0.05:
        return f"{p_value:.3f}*"
    else:
        return f"{p_value:.3f}"


def get_significance_badge(p_value: float) -> Tuple[str, str]:
    """Get significance badge and color for display."""
    if p_value < 0.001:
        return "***", "#22C55E"  # Green
    elif p_value < 0.01:
        return "**", "#22C55E"
    elif p_value < 0.05:
        return "*", "#F59E0B"  # Amber
    else:
        return "ns", "#6B7280"  # Gray
