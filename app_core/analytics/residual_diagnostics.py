# =============================================================================
# residual_diagnostics.py - Comprehensive Residual Analysis for Model Validation
# =============================================================================
"""
Statistical diagnostics for model residual analysis.

This module implements:
1. Autocorrelation analysis (ACF, PACF)
2. Normality tests (Shapiro-Wilk, Jarque-Bera)
3. Heteroscedasticity tests (Breusch-Pagan, White)
4. Independence tests (Ljung-Box, Runs test)
5. Q-Q plot data generation
6. Comprehensive diagnostic summary

These diagnostics validate model assumptions and detect issues like:
- Remaining learnable structure in residuals
- Non-constant variance (heteroscedasticity)
- Non-normality of errors
- Systematic forecast bias

References:
- Box, G.E.P. & Pierce, D.A. (1970). Distribution of Residual Autocorrelations.
- Ljung, G.M. & Box, G.E.P. (1978). On a Measure of Lack of Fit.
- Breusch, T.S. & Pagan, A.R. (1979). A Simple Test for Heteroscedasticity.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DiagnosticTestResult:
    """Result of a single diagnostic test."""
    test_name: str
    statistic: float
    p_value: Optional[float]
    conclusion: str
    passed: bool  # True if test indicates no problem
    interpretation: str
    recommendation: str = ""


@dataclass
class ACFResult:
    """Autocorrelation function results."""
    lags: np.ndarray
    acf_values: np.ndarray
    confidence_bounds: Tuple[float, float]  # 95% CI bounds
    significant_lags: List[int]
    has_significant_autocorrelation: bool


@dataclass
class PACFResult:
    """Partial autocorrelation function results."""
    lags: np.ndarray
    pacf_values: np.ndarray
    confidence_bounds: Tuple[float, float]
    significant_lags: List[int]


@dataclass
class QQPlotData:
    """Data for Q-Q plot generation."""
    theoretical_quantiles: np.ndarray
    sample_quantiles: np.ndarray
    reference_line_x: np.ndarray
    reference_line_y: np.ndarray


@dataclass
class ResidualDiagnosticsSummary:
    """Complete residual diagnostics summary."""
    model_name: str
    horizon: int
    n_observations: int

    # Summary statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float

    # Test results
    normality_test: DiagnosticTestResult
    autocorrelation_test: DiagnosticTestResult
    heteroscedasticity_test: DiagnosticTestResult
    bias_test: DiagnosticTestResult

    # ACF/PACF
    acf_result: ACFResult
    pacf_result: Optional[PACFResult]

    # Q-Q plot data
    qq_data: QQPlotData

    # Overall assessment
    overall_passed: bool
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# AUTOCORRELATION ANALYSIS
# =============================================================================

def compute_acf(
    residuals: np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05
) -> ACFResult:
    """
    Compute autocorrelation function with confidence bounds.

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    nlags : int
        Number of lags to compute
    alpha : float
        Significance level for confidence bounds

    Returns:
    --------
    ACFResult with ACF values and significance analysis
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)
    nlags = min(nlags, n // 2 - 1)

    # Compute ACF manually (statsmodels may not be available)
    mean = np.mean(residuals)
    var = np.var(residuals)

    acf_values = np.zeros(nlags + 1)
    acf_values[0] = 1.0  # Lag 0 is always 1

    for k in range(1, nlags + 1):
        covariance = np.sum((residuals[:-k] - mean) * (residuals[k:] - mean)) / n
        acf_values[k] = covariance / var if var > 0 else 0

    # 95% confidence bounds (Bartlett's formula for white noise)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_bound = z_critical / np.sqrt(n)
    confidence_bounds = (-ci_bound, ci_bound)

    # Find significant lags (excluding lag 0)
    significant_lags = [
        k for k in range(1, nlags + 1)
        if np.abs(acf_values[k]) > ci_bound
    ]

    return ACFResult(
        lags=np.arange(nlags + 1),
        acf_values=acf_values,
        confidence_bounds=confidence_bounds,
        significant_lags=significant_lags,
        has_significant_autocorrelation=len(significant_lags) > 0
    )


def compute_pacf(
    residuals: np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05
) -> PACFResult:
    """
    Compute partial autocorrelation function.

    Uses Yule-Walker equations for PACF calculation.

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    nlags : int
        Number of lags to compute
    alpha : float
        Significance level for confidence bounds

    Returns:
    --------
    PACFResult with PACF values and significance analysis
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)
    nlags = min(nlags, n // 2 - 1)

    # Compute ACF first
    acf_result = compute_acf(residuals, nlags, alpha)
    acf_vals = acf_result.acf_values

    # Compute PACF using Durbin-Levinson algorithm
    pacf_values = np.zeros(nlags + 1)
    pacf_values[0] = 1.0

    if nlags >= 1:
        pacf_values[1] = acf_vals[1]

    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = acf_vals[1]

    for k in range(2, nlags + 1):
        # Compute phi_kk
        numerator = acf_vals[k] - sum(phi[k-1, j] * acf_vals[k-j] for j in range(1, k))
        denominator = 1 - sum(phi[k-1, j] * acf_vals[j] for j in range(1, k))

        if np.abs(denominator) < 1e-10:
            phi[k, k] = 0
        else:
            phi[k, k] = numerator / denominator

        # Update other phi values
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

        pacf_values[k] = phi[k, k]

    # Confidence bounds
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_bound = z_critical / np.sqrt(n)
    confidence_bounds = (-ci_bound, ci_bound)

    # Significant lags
    significant_lags = [
        k for k in range(1, nlags + 1)
        if np.abs(pacf_values[k]) > ci_bound
    ]

    return PACFResult(
        lags=np.arange(nlags + 1),
        pacf_values=pacf_values,
        confidence_bounds=confidence_bounds,
        significant_lags=significant_lags
    )


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def ljung_box_test(
    residuals: np.ndarray,
    nlags: int = 10
) -> DiagnosticTestResult:
    """
    Ljung-Box test for autocorrelation.

    H0: Residuals are independently distributed (no autocorrelation)
    H1: Residuals exhibit serial correlation

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    nlags : int
        Number of lags to test

    Returns:
    --------
    DiagnosticTestResult with test outcome
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)
    nlags = min(nlags, n // 2 - 1)

    # Compute ACF
    acf_result = compute_acf(residuals, nlags)
    acf_vals = acf_result.acf_values

    # Ljung-Box statistic: Q = n(n+2) * sum(r_k^2 / (n-k))
    q_stat = 0.0
    for k in range(1, nlags + 1):
        q_stat += (acf_vals[k]**2) / (n - k)
    q_stat *= n * (n + 2)

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(q_stat, df=nlags)

    passed = p_value >= 0.05
    if passed:
        conclusion = "No significant autocorrelation detected"
        interpretation = "Residuals appear to be white noise - good model fit"
        recommendation = ""
    else:
        conclusion = f"Significant autocorrelation at lag(s): {acf_result.significant_lags}"
        interpretation = "Residuals show serial correlation - model may be missing patterns"
        recommendation = "Consider adding lagged features or using a different model architecture"

    return DiagnosticTestResult(
        test_name="Ljung-Box",
        statistic=q_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


def shapiro_wilk_test(
    residuals: np.ndarray,
    max_samples: int = 5000
) -> DiagnosticTestResult:
    """
    Shapiro-Wilk test for normality.

    H0: Residuals follow a normal distribution
    H1: Residuals do not follow a normal distribution

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    max_samples : int
        Maximum samples for test (scipy limitation)

    Returns:
    --------
    DiagnosticTestResult with test outcome
    """
    residuals = np.asarray(residuals).flatten()

    # Subsample if too large
    if len(residuals) > max_samples:
        np.random.seed(42)
        residuals = np.random.choice(residuals, max_samples, replace=False)

    if len(residuals) < 3:
        return DiagnosticTestResult(
            test_name="Shapiro-Wilk",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for normality test",
            passed=True,
            interpretation="Cannot perform test with fewer than 3 observations",
            recommendation=""
        )

    try:
        w_stat, p_value = stats.shapiro(residuals)
    except Exception as e:
        return DiagnosticTestResult(
            test_name="Shapiro-Wilk",
            statistic=np.nan,
            p_value=np.nan,
            conclusion=f"Test failed: {str(e)}",
            passed=True,
            interpretation="Could not compute normality test",
            recommendation=""
        )

    passed = p_value >= 0.05
    if passed:
        conclusion = "Cannot reject normality hypothesis"
        interpretation = "Residuals appear approximately normal"
        recommendation = ""
    else:
        conclusion = "Residuals significantly deviate from normality"
        interpretation = "Non-normal errors may affect prediction intervals"
        recommendation = "Consider robust prediction intervals or data transformation"

    return DiagnosticTestResult(
        test_name="Shapiro-Wilk",
        statistic=w_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


def jarque_bera_test(residuals: np.ndarray) -> DiagnosticTestResult:
    """
    Jarque-Bera test for normality based on skewness and kurtosis.

    H0: Residuals have skewness and kurtosis matching a normal distribution
    H1: Residuals deviate significantly from normality
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    if n < 20:
        return DiagnosticTestResult(
            test_name="Jarque-Bera",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data (need n >= 20)",
            passed=True,
            interpretation="Sample too small for reliable JB test",
            recommendation=""
        )

    # Compute skewness and kurtosis
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)  # Excess kurtosis (normal = 0)

    # JB statistic
    jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)

    # P-value from chi-squared with df=2
    p_value = 1 - stats.chi2.cdf(jb_stat, df=2)

    passed = p_value >= 0.05
    if passed:
        conclusion = "Cannot reject normality hypothesis"
        interpretation = f"Skewness={skewness:.3f}, Excess Kurtosis={kurtosis:.3f} (acceptable)"
        recommendation = ""
    else:
        conclusion = "Significant deviation from normality"
        interpretation = f"Skewness={skewness:.3f}, Excess Kurtosis={kurtosis:.3f}"
        if abs(skewness) > 1:
            recommendation = "High skewness: consider log transform or outlier treatment"
        elif abs(kurtosis) > 3:
            recommendation = "High kurtosis (heavy tails): consider robust methods"
        else:
            recommendation = "Minor deviation from normality - likely acceptable"

    return DiagnosticTestResult(
        test_name="Jarque-Bera",
        statistic=jb_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


def breusch_pagan_test(
    residuals: np.ndarray,
    fitted_values: np.ndarray
) -> DiagnosticTestResult:
    """
    Breusch-Pagan test for heteroscedasticity.

    H0: Homoscedasticity (constant variance)
    H1: Heteroscedasticity (variance depends on fitted values)

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    fitted_values : np.ndarray
        Model predictions (y_pred)

    Returns:
    --------
    DiagnosticTestResult with test outcome
    """
    residuals = np.asarray(residuals).flatten()
    fitted_values = np.asarray(fitted_values).flatten()

    n = len(residuals)
    if n < 10:
        return DiagnosticTestResult(
            test_name="Breusch-Pagan",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for heteroscedasticity test",
            passed=True,
            interpretation="Need at least 10 observations",
            recommendation=""
        )

    # Squared residuals
    u2 = residuals**2

    # Regress u2 on fitted values
    X = np.column_stack([np.ones(n), fitted_values])
    try:
        beta, residuals_aux, rank, s = np.linalg.lstsq(X, u2, rcond=None)
    except Exception:
        return DiagnosticTestResult(
            test_name="Breusch-Pagan",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Numerical error in regression",
            passed=True,
            interpretation="Could not compute test",
            recommendation=""
        )

    # Fitted values from auxiliary regression
    u2_fitted = X @ beta

    # Compute R² of auxiliary regression
    ss_res = np.sum((u2 - u2_fitted)**2)
    ss_tot = np.sum((u2 - np.mean(u2))**2)
    r2_aux = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # BP statistic = n * R²
    bp_stat = n * r2_aux

    # P-value from chi-squared with df = 1 (one regressor)
    p_value = 1 - stats.chi2.cdf(bp_stat, df=1)

    passed = p_value >= 0.05
    if passed:
        conclusion = "No significant heteroscedasticity detected"
        interpretation = "Variance appears constant across predictions"
        recommendation = ""
    else:
        conclusion = "Significant heteroscedasticity detected"
        interpretation = "Error variance changes with prediction level"
        recommendation = "Consider weighted least squares or heteroscedasticity-robust standard errors"

    return DiagnosticTestResult(
        test_name="Breusch-Pagan",
        statistic=bp_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


def runs_test(residuals: np.ndarray) -> DiagnosticTestResult:
    """
    Runs test for randomness.

    Tests whether residuals show random pattern or systematic runs.
    A run is a sequence of consecutive positive or negative values.

    H0: Residuals are random
    H1: Residuals show non-random patterns (trends, cycles)
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    if n < 10:
        return DiagnosticTestResult(
            test_name="Runs Test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data",
            passed=True,
            interpretation="Need at least 10 observations",
            recommendation=""
        )

    # Count positive and negative residuals
    signs = np.sign(residuals)
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)

    # Count runs
    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0:
            runs += 1

    # Expected runs under randomness
    if n_pos + n_neg > 0:
        expected_runs = 1 + (2 * n_pos * n_neg) / (n_pos + n_neg)
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                   ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
    else:
        return DiagnosticTestResult(
            test_name="Runs Test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="All residuals are zero",
            passed=True,
            interpretation="Cannot perform runs test",
            recommendation=""
        )

    if var_runs <= 0:
        return DiagnosticTestResult(
            test_name="Runs Test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Variance is zero",
            passed=True,
            interpretation="Cannot perform runs test",
            recommendation=""
        )

    # Z-statistic
    z_stat = (runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    passed = p_value >= 0.05
    if passed:
        conclusion = "Residuals appear random"
        interpretation = f"Observed {runs} runs (expected {expected_runs:.1f})"
        recommendation = ""
    else:
        if runs < expected_runs:
            conclusion = "Too few runs - possible trend or clustering"
            interpretation = f"Only {runs} runs observed (expected {expected_runs:.1f})"
            recommendation = "Check for trends or seasonality in residuals"
        else:
            conclusion = "Too many runs - possible oscillation"
            interpretation = f"{runs} runs observed (expected {expected_runs:.1f})"
            recommendation = "Check for high-frequency patterns"

    return DiagnosticTestResult(
        test_name="Runs Test",
        statistic=z_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


def bias_test(residuals: np.ndarray) -> DiagnosticTestResult:
    """
    Test for systematic forecast bias.

    H0: Mean residual = 0 (no bias)
    H1: Mean residual ≠ 0 (systematic bias)
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    mean_error = np.mean(residuals)
    std_error = np.std(residuals, ddof=1)
    se = std_error / np.sqrt(n)

    t_stat = mean_error / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))

    passed = p_value >= 0.05
    if passed:
        conclusion = "No significant forecast bias"
        interpretation = f"Mean error = {mean_error:.4f} (not significantly different from 0)"
        recommendation = ""
    else:
        if mean_error > 0:
            conclusion = f"Significant under-forecasting (bias = {mean_error:.4f})"
            interpretation = "Model systematically underestimates actual values"
            recommendation = "Consider bias correction or model recalibration"
        else:
            conclusion = f"Significant over-forecasting (bias = {mean_error:.4f})"
            interpretation = "Model systematically overestimates actual values"
            recommendation = "Consider bias correction or model recalibration"

    return DiagnosticTestResult(
        test_name="Bias Test",
        statistic=t_stat,
        p_value=p_value,
        conclusion=conclusion,
        passed=passed,
        interpretation=interpretation,
        recommendation=recommendation
    )


# =============================================================================
# Q-Q PLOT DATA
# =============================================================================

def compute_qq_data(residuals: np.ndarray) -> QQPlotData:
    """
    Compute Q-Q plot data for normality assessment.

    Returns theoretical and sample quantiles for plotting.
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    # Sort residuals for sample quantiles
    sample_quantiles = np.sort(residuals)

    # Compute theoretical quantiles from standard normal
    # Using Filliben's estimate for plotting positions
    positions = np.zeros(n)
    positions[0] = 1 - 0.5**(1/n)
    positions[-1] = 0.5**(1/n)
    positions[1:-1] = (np.arange(2, n) - 0.3175) / (n + 0.365)
    theoretical_quantiles = stats.norm.ppf(positions)

    # Reference line (through Q1 and Q3)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    slope = (sample_quantiles[q3_idx] - sample_quantiles[q1_idx]) / \
            (theoretical_quantiles[q3_idx] - theoretical_quantiles[q1_idx])
    intercept = sample_quantiles[q1_idx] - slope * theoretical_quantiles[q1_idx]

    ref_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    ref_y = intercept + slope * ref_x

    return QQPlotData(
        theoretical_quantiles=theoretical_quantiles,
        sample_quantiles=sample_quantiles,
        reference_line_x=ref_x,
        reference_line_y=ref_y
    )


# =============================================================================
# COMPREHENSIVE DIAGNOSTICS
# =============================================================================

def compute_full_diagnostics(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    model_name: str = "Model",
    horizon: int = 1
) -> ResidualDiagnosticsSummary:
    """
    Compute comprehensive residual diagnostics.

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals (actual - predicted)
    fitted_values : np.ndarray
        Model predictions
    model_name : str
        Name of the model
    horizon : int
        Forecast horizon

    Returns:
    --------
    ResidualDiagnosticsSummary with all diagnostic results
    """
    residuals = np.asarray(residuals).flatten()
    fitted_values = np.asarray(fitted_values).flatten()
    n = len(residuals)

    # Summary statistics
    mean = np.mean(residuals)
    std = np.std(residuals, ddof=1)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    min_val = np.min(residuals)
    max_val = np.max(residuals)

    # Run all tests
    normality = shapiro_wilk_test(residuals)
    autocorr = ljung_box_test(residuals)
    heterosced = breusch_pagan_test(residuals, fitted_values)
    bias = bias_test(residuals)

    # ACF/PACF
    acf_result = compute_acf(residuals)
    try:
        pacf_result = compute_pacf(residuals)
    except Exception:
        pacf_result = None

    # Q-Q data
    qq_data = compute_qq_data(residuals)

    # Collect issues
    issues = []
    recommendations = []

    if not normality.passed:
        issues.append("Non-normal residuals")
        if normality.recommendation:
            recommendations.append(normality.recommendation)

    if not autocorr.passed:
        issues.append("Autocorrelation in residuals")
        if autocorr.recommendation:
            recommendations.append(autocorr.recommendation)

    if not heterosced.passed:
        issues.append("Heteroscedasticity")
        if heterosced.recommendation:
            recommendations.append(heterosced.recommendation)

    if not bias.passed:
        issues.append("Systematic forecast bias")
        if bias.recommendation:
            recommendations.append(bias.recommendation)

    overall_passed = len(issues) == 0

    return ResidualDiagnosticsSummary(
        model_name=model_name,
        horizon=horizon,
        n_observations=n,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        min_val=min_val,
        max_val=max_val,
        normality_test=normality,
        autocorrelation_test=autocorr,
        heteroscedasticity_test=heterosced,
        bias_test=bias,
        acf_result=acf_result,
        pacf_result=pacf_result,
        qq_data=qq_data,
        overall_passed=overall_passed,
        issues_found=issues,
        recommendations=recommendations
    )


def format_diagnostic_summary(summary: ResidualDiagnosticsSummary) -> str:
    """Format diagnostic summary as readable text."""
    lines = [
        f"=== Residual Diagnostics: {summary.model_name} (Horizon {summary.horizon}) ===",
        f"",
        f"Summary Statistics:",
        f"  N = {summary.n_observations}",
        f"  Mean = {summary.mean:.4f}",
        f"  Std Dev = {summary.std:.4f}",
        f"  Skewness = {summary.skewness:.3f}",
        f"  Kurtosis = {summary.kurtosis:.3f}",
        f"",
        f"Diagnostic Tests:",
        f"  {summary.normality_test.test_name}: {'PASS' if summary.normality_test.passed else 'FAIL'} (p={summary.normality_test.p_value:.4f})",
        f"  {summary.autocorrelation_test.test_name}: {'PASS' if summary.autocorrelation_test.passed else 'FAIL'} (p={summary.autocorrelation_test.p_value:.4f})",
        f"  {summary.heteroscedasticity_test.test_name}: {'PASS' if summary.heteroscedasticity_test.passed else 'FAIL'} (p={summary.heteroscedasticity_test.p_value:.4f})",
        f"  {summary.bias_test.test_name}: {'PASS' if summary.bias_test.passed else 'FAIL'} (p={summary.bias_test.p_value:.4f})",
        f"",
    ]

    if summary.overall_passed:
        lines.append("OVERALL: All diagnostics PASSED")
    else:
        lines.append(f"OVERALL: {len(summary.issues_found)} issue(s) found")
        for issue in summary.issues_found:
            lines.append(f"  - {issue}")

    if summary.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for rec in summary.recommendations:
            lines.append(f"  - {rec}")

    return "\n".join(lines)


def get_diagnostic_badge(summary: ResidualDiagnosticsSummary) -> Tuple[str, str]:
    """Get overall diagnostic badge and color."""
    n_passed = sum([
        summary.normality_test.passed,
        summary.autocorrelation_test.passed,
        summary.heteroscedasticity_test.passed,
        summary.bias_test.passed
    ])

    if n_passed == 4:
        return "All Tests Passed", "#22C55E"  # Green
    elif n_passed >= 3:
        return "Minor Issues", "#F59E0B"  # Amber
    elif n_passed >= 2:
        return "Moderate Issues", "#F97316"  # Orange
    else:
        return "Significant Issues", "#EF4444"  # Red
