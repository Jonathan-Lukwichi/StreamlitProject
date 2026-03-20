# =============================================================================
# app_core/analytics/metrics_utils.py
# Centralized Forecast Performance Metrics Utility Module
# For HealthForecast AI - Master's Thesis Evaluation Framework
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# METRIC THRESHOLDS (Color Coding)
# =============================================================================

METRIC_THRESHOLDS = {
    "R2": {"good": 0.7, "warning": 0.4},  # >= good is green, >= warning is yellow, else red
    "sMAPE": {"good": 15.0, "warning": 30.0},  # <= good is green, <= warning is yellow, else red
    "DA": {"good": 70.0, "warning": 50.0},  # >= good is green, >= warning is yellow, else red
    "ME": {"good": 3.0, "warning": 8.0},  # |ME| <= good is green, |ME| <= warning is yellow, else red
    "MPE": {"good": 5.0, "warning": 15.0},  # |MPE| <= good is green, |MPE| <= warning is yellow, else red
    "MAPE": {"good": 10.0, "warning": 20.0},  # <= good is green, <= warning is yellow, else red
    "Accuracy": {"good": 90.0, "warning": 80.0},  # >= good is green, >= warning is yellow, else red
}

# Color codes matching app theme
COLORS = {
    "good": "#22c55e",      # Green
    "warning": "#facc15",   # Yellow
    "bad": "#f97373",       # Red
    "neutral": "#94a3b8",   # Gray
}


# =============================================================================
# CORE METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_mae(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        MAE value
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan
    return float(mean_absolute_error(y_actual, y_pred))


def calculate_rmse(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        RMSE value
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_actual, y_pred)))


def calculate_mape(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        MAPE as percentage (0-100 scale)
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan

    # Filter out zero actuals to avoid division by zero
    mask = y_actual != 0
    if not mask.any():
        return np.nan

    y_actual_filtered = y_actual[mask]
    y_pred_filtered = y_pred[mask]

    mape = np.mean(np.abs((y_actual_filtered - y_pred_filtered) / y_actual_filtered)) * 100
    return float(mape)


def calculate_r2(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination).

    Measures explained variance - how much of the variation in actual demand
    the model captures.

    Formula: R² = 1 - (SS_res / SS_tot)
        SS_res = Σ(y_actual - y_predicted)²
        SS_tot = Σ(y_actual - mean(y_actual))²

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        R² value between -inf and 1 (typically 0-1 for good models)
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) < 2:
        return np.nan

    # Check for constant series (would cause division by zero)
    if np.std(y_actual) == 0:
        return np.nan

    return float(r2_score(y_actual, y_pred))


def calculate_smape(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    Purpose: Fair percentage error that penalises over-forecasts and under-forecasts
    equally. Standard MAPE is biased - it penalises over-forecasts more, which is
    dangerous in healthcare where under-forecasting (missing demand) is costlier.

    Formula: sMAPE = (100% / n) * Σ( |y_actual - y_pred| / ((|y_actual| + |y_pred|) / 2) )

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        sMAPE as percentage (0-200 scale, typically 0-100 for reasonable forecasts)
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan

    numerator = np.abs(y_actual - y_pred)
    denominator = (np.abs(y_actual) + np.abs(y_pred)) / 2

    # Handle division by zero when both actual and predicted are 0
    # In this case, error should be 0 (perfect prediction of zero)
    mask = denominator != 0

    if not mask.any():
        # All values are zero - perfect prediction
        return 0.0

    smape_values = np.zeros_like(numerator, dtype=float)
    smape_values[mask] = numerator[mask] / denominator[mask]
    # Where both are zero, error is 0 (already initialized)

    smape = np.mean(smape_values) * 100
    return float(smape)


def calculate_directional_accuracy(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Directional Accuracy (DA).

    Purpose: Measures what percentage of the time the model correctly predicts
    whether demand goes UP or DOWN compared to the previous period. A model can
    have decent MAPE but consistently miss turning points, which is operationally
    dangerous for hospital supply chains.

    Formula: DA = (100% / (n-1)) * Σ 1[(y_actual[t] - y_actual[t-1]) * (y_pred[t] - y_pred[t-1]) > 0]

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Directional Accuracy as percentage (0-100)
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) < 2:
        return np.nan

    # Calculate direction changes
    actual_direction = np.diff(y_actual)
    pred_direction = np.diff(y_pred)

    # Count when directions agree (both positive, both negative, or both zero)
    # Agreement when product > 0, or both are exactly 0
    agreement = (actual_direction * pred_direction > 0) | \
                ((actual_direction == 0) & (pred_direction == 0))

    da = np.mean(agreement) * 100
    return float(da)


def calculate_me(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Error (ME) - Bias Detection.

    Purpose: Detects systematic bias.
        Positive ME = model over-forecasts on average
        Negative ME = model under-forecasts on average

    In hospital supply chains, consistent under-forecasting means stockouts
    and patient care failures.

    Formula: ME = (1/n) * Σ(y_pred - y_actual)

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Mean Error (signed value)
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan

    me = np.mean(y_pred - y_actual)
    return float(me)


def calculate_mpe(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Percentage Error (MPE) - Relative Bias Detection.

    Purpose: Same as ME but as a percentage, making bias comparable across
    models with different scales.

    Formula: MPE = (100% / n) * Σ((y_pred - y_actual) / y_actual)

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Mean Percentage Error as signed percentage
    """
    y_actual, y_pred = _validate_inputs(y_actual, y_pred)
    if len(y_actual) == 0:
        return np.nan

    # Filter out zero actuals to avoid division by zero
    mask = y_actual != 0
    if not mask.any():
        return np.nan

    y_actual_filtered = y_actual[mask]
    y_pred_filtered = y_pred[mask]

    mpe = np.mean((y_pred_filtered - y_actual_filtered) / y_actual_filtered) * 100
    return float(mpe)


def calculate_accuracy(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Forecast Accuracy (100 - MAPE).

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Accuracy as percentage (0-100)
    """
    mape = calculate_mape(y_actual, y_pred)
    if np.isnan(mape):
        return np.nan
    return max(0.0, 100.0 - mape)


# =============================================================================
# COMPREHENSIVE METRIC CALCULATION
# =============================================================================

def calculate_all_metrics(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    include_existing: bool = True
) -> Dict[str, float]:
    """
    Calculate all forecast performance metrics.

    Args:
        y_actual: Array of actual values
        y_pred: Array of predicted values
        include_existing: If True, include MAE, RMSE, MAPE, Accuracy

    Returns:
        Dictionary with all metric values
    """
    metrics = {}

    if include_existing:
        metrics["MAE"] = calculate_mae(y_actual, y_pred)
        metrics["RMSE"] = calculate_rmse(y_actual, y_pred)
        metrics["MAPE"] = calculate_mape(y_actual, y_pred)
        metrics["Accuracy"] = calculate_accuracy(y_actual, y_pred)

    # New metrics
    metrics["R2"] = calculate_r2(y_actual, y_pred)
    metrics["sMAPE"] = calculate_smape(y_actual, y_pred)
    metrics["DA"] = calculate_directional_accuracy(y_actual, y_pred)
    metrics["ME"] = calculate_me(y_actual, y_pred)
    metrics["MPE"] = calculate_mpe(y_actual, y_pred)

    return metrics


def calculate_metrics_per_horizon(
    per_horizon_data: Dict[int, Dict],
    horizon_key: str = "Horizon",
    actual_key: str = "y_test",
    pred_key: str = "forecast"
) -> pd.DataFrame:
    """
    Calculate all metrics for each forecast horizon.

    Args:
        per_horizon_data: Dict with horizon as key, containing y_test and forecast arrays
        horizon_key: Column name for horizon
        actual_key: Key for actual values in the dict
        pred_key: Key for predicted values in the dict

    Returns:
        DataFrame with metrics per horizon plus average row
    """
    rows = []

    for horizon, data in sorted(per_horizon_data.items()):
        if isinstance(horizon, (int, float)):
            y_actual = np.array(data.get(actual_key, []))
            y_pred = np.array(data.get(pred_key, data.get("y_pred", [])))

            if len(y_actual) > 0 and len(y_pred) > 0:
                metrics = calculate_all_metrics(y_actual, y_pred)
                metrics[horizon_key] = f"h={horizon}"
                rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Calculate average row
    avg_row = {horizon_key: "Average"}
    metric_cols = [c for c in df.columns if c != horizon_key]
    for col in metric_cols:
        avg_row[col] = df[col].mean()

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Reorder columns
    col_order = [horizon_key, "MAE", "RMSE", "MAPE", "sMAPE", "R2", "Accuracy", "DA", "ME", "MPE"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    return df


# =============================================================================
# COLOR CODING AND FORMATTING
# =============================================================================

def get_metric_status(metric_name: str, value: float) -> str:
    """
    Get status (good/warning/bad) for a metric value based on thresholds.

    Args:
        metric_name: Name of the metric (R2, sMAPE, DA, ME, MPE, MAPE, Accuracy)
        value: The metric value

    Returns:
        Status string: "good", "warning", or "bad"
    """
    if np.isnan(value):
        return "neutral"

    thresholds = METRIC_THRESHOLDS.get(metric_name)
    if not thresholds:
        return "neutral"

    good_threshold = thresholds["good"]
    warning_threshold = thresholds["warning"]

    # Metrics where higher is better: R2, DA, Accuracy
    if metric_name in ["R2", "DA", "Accuracy"]:
        if value >= good_threshold:
            return "good"
        elif value >= warning_threshold:
            return "warning"
        else:
            return "bad"

    # Metrics where lower is better: sMAPE, MAPE
    elif metric_name in ["sMAPE", "MAPE"]:
        if value <= good_threshold:
            return "good"
        elif value <= warning_threshold:
            return "warning"
        else:
            return "bad"

    # Metrics where absolute value matters: ME, MPE
    elif metric_name in ["ME", "MPE"]:
        abs_value = abs(value)
        if abs_value <= good_threshold:
            return "good"
        elif abs_value <= warning_threshold:
            return "warning"
        else:
            return "bad"

    return "neutral"


def get_metric_color(metric_name: str, value: float) -> str:
    """
    Get hex color code for a metric value based on thresholds.

    Args:
        metric_name: Name of the metric
        value: The metric value

    Returns:
        Hex color code string
    """
    status = get_metric_status(metric_name, value)
    return COLORS.get(status, COLORS["neutral"])


def format_metric_value(metric_name: str, value: float) -> str:
    """
    Format a metric value for display.

    Args:
        metric_name: Name of the metric
        value: The metric value

    Returns:
        Formatted string
    """
    if np.isnan(value):
        return "N/A"

    # Percentage metrics
    if metric_name in ["MAPE", "sMAPE", "DA", "Accuracy"]:
        return f"{value:.1f}%"

    # Signed percentage
    elif metric_name == "MPE":
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"

    # R² (0-1 scale)
    elif metric_name == "R2":
        return f"{value:.3f}"

    # Signed values (ME)
    elif metric_name == "ME":
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}"

    # Error metrics (MAE, RMSE)
    elif metric_name in ["MAE", "RMSE"]:
        return f"{value:.2f}"

    # Default
    return f"{value:.4f}"


def get_bias_interpretation(me: float, mpe: float) -> Tuple[str, str]:
    """
    Get text interpretation of forecast bias.

    Args:
        me: Mean Error value
        mpe: Mean Percentage Error value

    Returns:
        Tuple of (bias_type, interpretation_text)
    """
    if np.isnan(me) or np.isnan(mpe):
        return ("neutral", "Insufficient data for bias analysis")

    abs_me = abs(me)
    abs_mpe = abs(mpe)

    if abs_me < 1 and abs_mpe < 1:
        return ("neutral", "No significant bias detected")

    if me > 0:
        bias_type = "over"
        direction = "over-forecasts"
    else:
        bias_type = "under"
        direction = "under-forecasts"

    # Severity
    if abs_me > 8 or abs_mpe > 15:
        severity = "significantly"
    elif abs_me > 3 or abs_mpe > 5:
        severity = "moderately"
    else:
        severity = "slightly"

    interpretation = (
        f"Model {severity} {direction} by an average of {abs_me:.1f} patients "
        f"(MPE: {mpe:+.1f}%), indicating systematic demand "
        f"{'overestimation' if me > 0 else 'underestimation'}."
    )

    return (bias_type, interpretation)


# =============================================================================
# STYLED DATAFRAME HELPERS
# =============================================================================

def style_metrics_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply conditional color formatting to a metrics DataFrame.

    Args:
        df: DataFrame with metric columns

    Returns:
        Styled DataFrame
    """
    def color_cell(val, metric_name):
        if pd.isna(val):
            return ""
        color = get_metric_color(metric_name, val)
        return f"background-color: {color}20; color: {color}"

    styler = df.style

    metric_columns = ["MAE", "RMSE", "MAPE", "sMAPE", "R2", "Accuracy", "DA", "ME", "MPE"]

    for col in metric_columns:
        if col in df.columns:
            styler = styler.applymap(
                lambda x, c=col: color_cell(x, c),
                subset=[col]
            )

    # Format numbers
    format_dict = {}
    for col in df.columns:
        if col in ["MAE", "RMSE"]:
            format_dict[col] = "{:.2f}"
        elif col in ["MAPE", "sMAPE", "DA", "Accuracy"]:
            format_dict[col] = "{:.1f}%"
        elif col == "R2":
            format_dict[col] = "{:.3f}"
        elif col in ["ME", "MPE"]:
            format_dict[col] = "{:+.2f}" if col == "ME" else "{:+.1f}%"

    if format_dict:
        styler = styler.format(format_dict, na_rep="N/A")

    return styler


def create_metric_card_html(
    metric_name: str,
    value: float,
    label: Optional[str] = None,
    show_color: bool = True
) -> str:
    """
    Create HTML for a single metric card matching app theme.

    Args:
        metric_name: Name of the metric
        value: The metric value
        label: Optional display label (defaults to metric_name)
        show_color: Whether to apply color coding

    Returns:
        HTML string for the metric card
    """
    display_label = label or metric_name
    formatted_value = format_metric_value(metric_name, value)

    if show_color and not np.isnan(value):
        color = get_metric_color(metric_name, value)
    else:
        color = "#ffffff"

    html = f"""
    <div class="metric-card" style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid {color}50;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 15px {color}20;
    ">
        <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">
            {display_label}
        </div>
        <div style="color: {color}; font-size: 1.75rem; font-weight: 700;">
            {formatted_value}
        </div>
    </div>
    """
    return html


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _validate_inputs(
    y_actual: Union[np.ndarray, list, pd.Series],
    y_pred: Union[np.ndarray, list, pd.Series]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert inputs to numpy arrays.

    Args:
        y_actual: Actual values
        y_pred: Predicted values

    Returns:
        Tuple of validated numpy arrays
    """
    # Convert to numpy arrays
    y_actual = np.asarray(y_actual, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    # Remove NaN values
    mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
    y_actual = y_actual[mask]
    y_pred = y_pred[mask]

    # Ensure same length
    min_len = min(len(y_actual), len(y_pred))
    y_actual = y_actual[:min_len]
    y_pred = y_pred[:min_len]

    return y_actual, y_pred


# =============================================================================
# METRIC DISPLAY NAMES AND DESCRIPTIONS
# =============================================================================

METRIC_INFO = {
    "MAE": {
        "name": "Mean Absolute Error",
        "short_name": "MAE",
        "description": "Average absolute difference between predicted and actual values",
        "unit": "patients",
        "higher_is_better": False,
    },
    "RMSE": {
        "name": "Root Mean Squared Error",
        "short_name": "RMSE",
        "description": "Square root of average squared errors, penalizes large errors more",
        "unit": "patients",
        "higher_is_better": False,
    },
    "MAPE": {
        "name": "Mean Absolute Percentage Error",
        "short_name": "MAPE",
        "description": "Average percentage error (can be biased towards over-forecasts)",
        "unit": "%",
        "higher_is_better": False,
    },
    "sMAPE": {
        "name": "Symmetric MAPE",
        "short_name": "sMAPE",
        "description": "Symmetric percentage error that treats over/under-forecasts equally",
        "unit": "%",
        "higher_is_better": False,
    },
    "R2": {
        "name": "R² (Coefficient of Determination)",
        "short_name": "R²",
        "description": "Proportion of variance in actual values explained by the model",
        "unit": "",
        "higher_is_better": True,
    },
    "Accuracy": {
        "name": "Forecast Accuracy",
        "short_name": "Accuracy",
        "description": "100 minus MAPE, represents overall forecast accuracy",
        "unit": "%",
        "higher_is_better": True,
    },
    "DA": {
        "name": "Directional Accuracy",
        "short_name": "DA",
        "description": "Percentage of times the model correctly predicts direction of change",
        "unit": "%",
        "higher_is_better": True,
    },
    "ME": {
        "name": "Mean Error (Bias)",
        "short_name": "ME",
        "description": "Average signed error indicating systematic over/under-forecasting",
        "unit": "patients",
        "higher_is_better": None,  # Closer to 0 is better
    },
    "MPE": {
        "name": "Mean Percentage Error",
        "short_name": "MPE",
        "description": "Relative bias as percentage, indicates systematic forecast direction",
        "unit": "%",
        "higher_is_better": None,  # Closer to 0 is better
    },
}


def get_metric_info(metric_name: str) -> Dict:
    """Get information about a metric."""
    return METRIC_INFO.get(metric_name, {
        "name": metric_name,
        "short_name": metric_name,
        "description": "",
        "unit": "",
        "higher_is_better": None,
    })
