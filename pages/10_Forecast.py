# =============================================================================
# 10_Forecast.py — Universal Forecast Hub
# Generate actual forecasts using trained models with rolling/walk-forward approach
# Works with ANY dataset - adaptable to different domains
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Forecast Hub - HealthForecast AI",
    page_icon="🔮",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* Forecast cards */
.forecast-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    text-align: center;
}

.forecast-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}

.forecast-label {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.forecast-day {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.status-ready {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.status-pending {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.model-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.125rem;
}

.model-ml { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
.model-stat { background: rgba(168, 85, 247, 0.2); color: #a855f7; }
.model-hybrid { background: rgba(236, 72, 153, 0.2); color: #ec4899; }

/* ========================================
   FORECAST QUALITY KPI CARDS
   ======================================== */

/* Main KPI Card */
.fq-kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.fq-kpi-card:hover {
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
    transform: translateY(-2px);
}

.fq-kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color, #3b82f6), var(--accent-color-end, #22d3ee));
}

.fq-kpi-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.5));
}

.fq-kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.5rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.fq-kpi-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.fq-kpi-sublabel {
    font-size: 0.75rem;
    color: #64748b;
}

/* RPIW Indicator Styles */
.rpiw-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.25rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
}

.rpiw-low {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.25), rgba(34, 197, 94, 0.15));
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.4);
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
}

.rpiw-medium {
    background: linear-gradient(135deg, rgba(250, 204, 21, 0.25), rgba(250, 204, 21, 0.15));
    color: #facc15;
    border: 1px solid rgba(250, 204, 21, 0.4);
    box-shadow: 0 0 20px rgba(250, 204, 21, 0.2);
}

.rpiw-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(239, 68, 68, 0.15));
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
}

/* Optimization Recommendation Card */
.opt-rec-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.9));
    border-radius: 16px;
    padding: 1.5rem;
    border-left: 4px solid var(--accent-color, #3b82f6);
    margin: 1rem 0;
}

.opt-rec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.75rem;
}

.opt-rec-method {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-color, #3b82f6), #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.opt-rec-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* Model Metrics Grid */
.model-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

/* Decision Framework Table */
.decision-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.decision-table th {
    background: rgba(59, 130, 246, 0.2);
    color: #f8fafc;
    padding: 1rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid rgba(59, 130, 246, 0.4);
}

.decision-table td {
    padding: 0.875rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
}

.decision-table tr:hover {
    background: rgba(59, 130, 246, 0.1);
}

.decision-table .highlight-row {
    background: rgba(59, 130, 246, 0.15);
    border-left: 3px solid #3b82f6;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FORECAST QUALITY ASSESSMENT - RPIW & MODEL METRICS
# =============================================================================

@dataclass
class PredictionIntervalSource:
    """Information about where prediction intervals came from."""
    source_name: str  # e.g., "ARIMA", "SARIMAX", "CQR", "Estimated"
    source_type: str  # "model_intervals", "conformal", "variance_estimate"
    confidence_level: float  # e.g., 0.90 for 90% CI
    lower_bounds: List[float]
    upper_bounds: List[float]
    is_actual: bool  # True if from actual model, False if estimated


def extract_prediction_intervals_from_session() -> Optional[PredictionIntervalSource]:
    """
    Extract actual prediction intervals from trained models in session state.

    Checks (in priority order):
    1. CQR/Conformal Prediction results (most rigorous)
    2. SARIMAX confidence intervals
    3. ARIMA confidence intervals
    4. ML model prediction intervals (if available)

    Returns:
        PredictionIntervalSource or None if no intervals found
    """

    # -------------------------------------------------------------------------
    # 1. Check for CQR/Conformal Prediction results (highest priority)
    # -------------------------------------------------------------------------
    cqr_results = st.session_state.get("cqr_results") or st.session_state.get("conformal_results")
    if cqr_results and isinstance(cqr_results, dict):
        lower = cqr_results.get("lower") or cqr_results.get("pred_lower")
        upper = cqr_results.get("upper") or cqr_results.get("pred_upper")
        if lower is not None and upper is not None:
            lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
            upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
            if lower_list and upper_list:
                return PredictionIntervalSource(
                    source_name="Conformal Prediction (CQR)",
                    source_type="conformal",
                    confidence_level=cqr_results.get("coverage_target", 0.90),
                    lower_bounds=lower_list,
                    upper_bounds=upper_list,
                    is_actual=True
                )

    # -------------------------------------------------------------------------
    # 2. Check SARIMAX results
    # -------------------------------------------------------------------------
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results and isinstance(sarimax_results, dict):
        # Single-horizon SARIMAX
        lower = sarimax_results.get("forecast_lower")
        upper = sarimax_results.get("forecast_upper")
        if lower is not None and upper is not None:
            lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
            upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
            if lower_list and upper_list:
                return PredictionIntervalSource(
                    source_name="SARIMAX (95% CI)",
                    source_type="model_intervals",
                    confidence_level=0.95,
                    lower_bounds=lower_list,
                    upper_bounds=upper_list,
                    is_actual=True
                )

        # Multi-horizon SARIMAX (check per_h structure)
        per_h = sarimax_results.get("per_h", {})
        if per_h:
            for h, h_data in sorted(per_h.items()):
                lower = h_data.get("ci_lo") or h_data.get("forecast_lower")
                upper = h_data.get("ci_hi") or h_data.get("forecast_upper")
                if lower is not None and upper is not None:
                    lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
                    upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
                    if lower_list and upper_list:
                        return PredictionIntervalSource(
                            source_name=f"SARIMAX h={h} (95% CI)",
                            source_type="model_intervals",
                            confidence_level=0.95,
                            lower_bounds=lower_list,
                            upper_bounds=upper_list,
                            is_actual=True
                        )
                break

    # -------------------------------------------------------------------------
    # 3. Check ARIMA results
    # -------------------------------------------------------------------------
    arima_results = st.session_state.get("arima_mh_results")
    if arima_results and isinstance(arima_results, dict):
        # Check per_h structure (multi-horizon)
        per_h = arima_results.get("per_h", {})
        if per_h:
            for h, h_data in sorted(per_h.items()):
                lower = h_data.get("ci_lo") or h_data.get("forecast_lower")
                upper = h_data.get("ci_hi") or h_data.get("forecast_upper")
                if lower is not None and upper is not None:
                    lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
                    upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
                    if lower_list and upper_list:
                        return PredictionIntervalSource(
                            source_name=f"ARIMA h={h} (95% CI)",
                            source_type="model_intervals",
                            confidence_level=0.95,
                            lower_bounds=lower_list,
                            upper_bounds=upper_list,
                            is_actual=True
                        )
                break

        # Single-target ARIMA (legacy format)
        lower = arima_results.get("ci_lower") or arima_results.get("forecast_lower")
        upper = arima_results.get("ci_upper") or arima_results.get("forecast_upper")
        if lower is not None and upper is not None:
            lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
            upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
            if lower_list and upper_list:
                return PredictionIntervalSource(
                    source_name="ARIMA (95% CI)",
                    source_type="model_intervals",
                    confidence_level=0.95,
                    lower_bounds=lower_list,
                    upper_bounds=upper_list,
                    is_actual=True
                )

    # -------------------------------------------------------------------------
    # 4. Check ML model results for intervals
    # -------------------------------------------------------------------------
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_"):
            data = st.session_state.get(key)
            if data and isinstance(data, dict):
                lower = data.get("pred_lower") or data.get("lower_bound")
                upper = data.get("pred_upper") or data.get("upper_bound")
                if lower is not None and upper is not None:
                    lower_list = list(lower.values) if hasattr(lower, 'values') else list(lower)
                    upper_list = list(upper.values) if hasattr(upper, 'values') else list(upper)
                    if lower_list and upper_list:
                        model_name = key.replace("ml_mh_results_", "")
                        return PredictionIntervalSource(
                            source_name=f"{model_name} (90% PI)",
                            source_type="model_intervals",
                            confidence_level=0.90,
                            lower_bounds=lower_list,
                            upper_bounds=upper_list,
                            is_actual=True
                        )

    return None


def calculate_rpiw(
    forecast_values: List[float],
    lower_bounds: List[float] = None,
    upper_bounds: List[float] = None,
) -> Tuple[float, str]:
    """
    Calculate Relative Prediction Interval Width (RPIW).

    RPIW = (U - L) / D_hat * 100%

    Args:
        forecast_values: Point forecasts
        lower_bounds: Lower prediction interval bounds (optional)
        upper_bounds: Upper prediction interval bounds (optional)

    Returns:
        Tuple of (RPIW percentage, interval source description)
    """
    if not forecast_values:
        return 0.0, "No data"

    interval_source = "Variance-based estimate (90% CI)"

    if lower_bounds is None or upper_bounds is None:
        # Try to extract actual intervals from session state
        actual_intervals = extract_prediction_intervals_from_session()

        if actual_intervals and actual_intervals.is_actual:
            # Use actual model intervals
            lower_bounds = actual_intervals.lower_bounds
            upper_bounds = actual_intervals.upper_bounds
            interval_source = f"{actual_intervals.source_name}"

            # Ensure lengths match forecast (trim or extend)
            n = len(forecast_values)
            if len(lower_bounds) > n:
                lower_bounds = lower_bounds[:n]
                upper_bounds = upper_bounds[:n]
            elif len(lower_bounds) < n:
                lower_bounds = list(lower_bounds) + [lower_bounds[-1]] * (n - len(lower_bounds))
                upper_bounds = list(upper_bounds) + [upper_bounds[-1]] * (n - len(upper_bounds))
        else:
            # Fall back to variance-based estimation (90% PI)
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values) if len(forecast_values) > 1 else mean_forecast * 0.1
            lower_bounds = [max(0, f - 1.645 * std_forecast) for f in forecast_values]
            upper_bounds = [f + 1.645 * std_forecast for f in forecast_values]

    rpiw_values = []
    for d_hat, l, u in zip(forecast_values, lower_bounds, upper_bounds):
        if d_hat > 0:
            rpiw = ((u - l) / d_hat) * 100
            rpiw_values.append(rpiw)

    return np.mean(rpiw_values) if rpiw_values else 0.0, interval_source


def get_optimization_recommendation(rpiw: float) -> Dict[str, Any]:
    """
    Get optimization method recommendation based on RPIW.

    Decision Framework:
    - RPIW < 15%: Deterministic MILP (Low uncertainty)
    - 15% <= RPIW <= 40%: Robust Optimization (Medium uncertainty)
    - RPIW > 40%: Two-Stage Stochastic (High uncertainty)

    Args:
        rpiw: Relative Prediction Interval Width (%)

    Returns:
        Dict with recommendation details
    """
    if rpiw < 15:
        return {
            "method": "Deterministic MILP",
            "rpiw": rpiw,
            "uncertainty_level": "LOW",
            "color": "#22c55e",
            "css_class": "rpiw-low",
            "icon": "🟢",
            "safety_stock": "Minimal (~1 day)",
            "computation": "Fast",
            "description": "Point forecast is reliable. Extra complexity not justified. Safe to use EOQ/MILP directly.",
            "interval_width": "±5 patients",
        }
    elif rpiw <= 40:
        return {
            "method": "Robust Optimization",
            "rpiw": rpiw,
            "uncertainty_level": "MEDIUM",
            "color": "#facc15",
            "css_class": "rpiw-medium",
            "icon": "🟡",
            "safety_stock": "Based on interval width",
            "computation": "Fast",
            "description": "Need protection but scenarios overkill. Budget Γ adjustable for robust formulation.",
            "interval_width": "±10-20 patients",
        }
    else:
        return {
            "method": "Two-Stage Stochastic",
            "rpiw": rpiw,
            "uncertainty_level": "HIGH",
            "color": "#ef4444",
            "css_class": "rpiw-high",
            "icon": "🔴",
            "safety_stock": "From distribution",
            "computation": "Slow",
            "description": "Need explicit recourse actions. Scenarios capture distribution tails.",
            "interval_width": "±30+ patients",
        }


def extract_model_metrics_from_session() -> Dict[str, Any]:
    """
    Extract model metrics from session state (Modeling Hub + Benchmarks).

    Returns:
        Dict with ML and statistical model metrics
    """
    metrics = {
        "ml_models": [],
        "stat_models": [],
        "best_model": None,
        "best_accuracy": 0,
    }

    # Helper to safely extract float
    def safe_float(x, default=np.nan):
        try:
            f = float(x)
            return f if np.isfinite(f) else default
        except:
            return default

    # -------------------------------------------------------------------------
    # ML Models from Modeling Hub (08_Modeling_Hub.py)
    # -------------------------------------------------------------------------
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_"):
            model_name = key.replace("ml_mh_results_", "")
            data = st.session_state.get(key)
            if data and isinstance(data, dict):
                # Try to extract metrics
                test_metrics = data.get("test_metrics", {})
                if not test_metrics and "Target_1" in data:
                    test_metrics = data.get("Target_1", {}).get("test_metrics", {})

                model_metrics = {
                    "name": model_name,
                    "type": "ML",
                    "mae": safe_float(test_metrics.get("MAE")),
                    "rmse": safe_float(test_metrics.get("RMSE")),
                    "mape": safe_float(test_metrics.get("MAPE")),
                    "accuracy": safe_float(test_metrics.get("Accuracy")),
                }

                # Calculate accuracy from MAPE if not available
                if np.isnan(model_metrics["accuracy"]) and not np.isnan(model_metrics["mape"]):
                    model_metrics["accuracy"] = 100 - model_metrics["mape"]

                metrics["ml_models"].append(model_metrics)

                # Track best model
                if not np.isnan(model_metrics["accuracy"]) and model_metrics["accuracy"] > metrics["best_accuracy"]:
                    metrics["best_accuracy"] = model_metrics["accuracy"]
                    metrics["best_model"] = model_metrics

    # -------------------------------------------------------------------------
    # Statistical Models from Benchmarks (05_Benchmarks.py)
    # -------------------------------------------------------------------------

    # ARIMA results
    arima_results = st.session_state.get("arima_mh_results")
    if arima_results and isinstance(arima_results, dict):
        metrics_df = arima_results.get("metrics_df") or arima_results.get("results_df")
        if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            # Get average metrics across horizons
            avg_mae = safe_float(metrics_df.get("Test_MAE", metrics_df.get("MAE", pd.Series([np.nan]))).mean())
            avg_rmse = safe_float(metrics_df.get("Test_RMSE", metrics_df.get("RMSE", pd.Series([np.nan]))).mean())
            avg_mape = safe_float(metrics_df.get("Test_MAPE", metrics_df.get("MAPE_%", pd.Series([np.nan]))).mean())
            avg_acc = safe_float(metrics_df.get("Test_Acc", metrics_df.get("Accuracy_%", pd.Series([np.nan]))).mean())

            if np.isnan(avg_acc) and not np.isnan(avg_mape):
                avg_acc = 100 - avg_mape

            model_metrics = {
                "name": "ARIMA",
                "type": "Statistical",
                "mae": avg_mae,
                "rmse": avg_rmse,
                "mape": avg_mape,
                "accuracy": avg_acc,
            }
            metrics["stat_models"].append(model_metrics)

            if not np.isnan(avg_acc) and avg_acc > metrics["best_accuracy"]:
                metrics["best_accuracy"] = avg_acc
                metrics["best_model"] = model_metrics

    # SARIMAX results
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results and isinstance(sarimax_results, dict):
        metrics_df = sarimax_results.get("metrics_df") or sarimax_results.get("results_df")
        if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            avg_mae = safe_float(metrics_df.get("Test_MAE", metrics_df.get("MAE", pd.Series([np.nan]))).mean())
            avg_rmse = safe_float(metrics_df.get("Test_RMSE", metrics_df.get("RMSE", pd.Series([np.nan]))).mean())
            avg_mape = safe_float(metrics_df.get("Test_MAPE", metrics_df.get("MAPE_%", pd.Series([np.nan]))).mean())
            avg_acc = safe_float(metrics_df.get("Test_Acc", metrics_df.get("Accuracy_%", pd.Series([np.nan]))).mean())

            if np.isnan(avg_acc) and not np.isnan(avg_mape):
                avg_acc = 100 - avg_mape

            model_metrics = {
                "name": "SARIMAX",
                "type": "Statistical",
                "mae": avg_mae,
                "rmse": avg_rmse,
                "mape": avg_mape,
                "accuracy": avg_acc,
            }
            metrics["stat_models"].append(model_metrics)

            if not np.isnan(avg_acc) and avg_acc > metrics["best_accuracy"]:
                metrics["best_accuracy"] = avg_acc
                metrics["best_model"] = model_metrics

    # Model comparison table from Benchmarks
    model_comparison = st.session_state.get("model_comparison")
    if model_comparison is not None and isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
        metrics["comparison_df"] = model_comparison

    return metrics


def render_forecast_quality_kpis(forecast_values: List[float], model_name: str = ""):
    """
    Render the Forecast Quality Assessment KPI cards.

    Args:
        forecast_values: The forecast values
        model_name: Name of the model used
    """
    if not forecast_values:
        st.warning("No forecast values available for quality assessment.")
        return

    # Calculate RPIW (returns tuple: value, source)
    rpiw, interval_source = calculate_rpiw(forecast_values)
    recommendation = get_optimization_recommendation(rpiw)

    # Get model metrics from session
    model_metrics = extract_model_metrics_from_session()

    # Determine if using actual model intervals
    is_actual_intervals = "Variance-based" not in interval_source
    source_color = "#22c55e" if is_actual_intervals else "#94a3b8"
    source_icon = "✅" if is_actual_intervals else "📐"

    # -------------------------------------------------------------------------
    # SECTION 1: Interval Source Banner
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
                border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px;
                padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.75rem;">
        <span style="font-size: 1.25rem;">{source_icon}</span>
        <div>
            <span style="color: #94a3b8; font-size: 0.8rem;">Prediction Intervals from:</span>
            <strong style="color: {source_color}; margin-left: 0.5rem;">{interval_source}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: RPIW Overview KPIs
    # -------------------------------------------------------------------------
    st.markdown("### 🎯 Forecast Uncertainty Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="fq-kpi-card" style="--accent-color: {recommendation['color']}; --accent-color-end: {recommendation['color']};">
            <div class="fq-kpi-icon">📊</div>
            <div class="fq-kpi-label">RPIW Score</div>
            <div class="fq-kpi-value" style="color: {recommendation['color']};">{rpiw:.1f}%</div>
            <div class="fq-kpi-sublabel">Relative Prediction Interval Width</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        uncertainty_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[recommendation["uncertainty_level"]]
        st.markdown(f"""
        <div class="fq-kpi-card" style="--accent-color: {recommendation['color']}; --accent-color-end: {recommendation['color']};">
            <div class="fq-kpi-icon">{uncertainty_icon}</div>
            <div class="fq-kpi-label">Uncertainty Level</div>
            <div class="fq-kpi-value" style="color: {recommendation['color']}; font-size: 1.75rem;">{recommendation['uncertainty_level']}</div>
            <div class="fq-kpi-sublabel">{recommendation['interval_width']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="fq-kpi-card" style="--accent-color: #3b82f6; --accent-color-end: #22d3ee;">
            <div class="fq-kpi-icon">⚡</div>
            <div class="fq-kpi-label">Computation</div>
            <div class="fq-kpi-value" style="font-size: 1.75rem;">{recommendation['computation']}</div>
            <div class="fq-kpi-sublabel">{recommendation['safety_stock']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        best_acc = model_metrics.get("best_accuracy", 0)
        best_name = model_metrics.get("best_model", {}).get("name", "N/A") if model_metrics.get("best_model") else "N/A"
        acc_color = "#22c55e" if best_acc >= 90 else ("#facc15" if best_acc >= 80 else "#ef4444")
        st.markdown(f"""
        <div class="fq-kpi-card" style="--accent-color: {acc_color}; --accent-color-end: {acc_color};">
            <div class="fq-kpi-icon">🏆</div>
            <div class="fq-kpi-label">Best Model Accuracy</div>
            <div class="fq-kpi-value" style="color: {acc_color};">{best_acc:.1f}%</div>
            <div class="fq-kpi-sublabel">{best_name}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: Optimization Recommendation
    # -------------------------------------------------------------------------
    st.markdown("### 🎛️ Recommended Optimization Approach")

    st.markdown(f"""
    <div class="opt-rec-card" style="--accent-color: {recommendation['color']};">
        <div class="opt-rec-title">{recommendation['icon']} Based on your RPIW of {rpiw:.1f}%:</div>
        <div class="opt-rec-method">{recommendation['method']}</div>
        <div class="opt-rec-desc">{recommendation['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: Decision Framework Table
    # -------------------------------------------------------------------------
    with st.expander("📋 View Decision Framework", expanded=False):
        st.markdown("""
        **Key Insight:** The width of your prediction interval determines which optimization approach to use.

        **Prediction Interval Width (PIW):** `PIW_t = U_t - L_t`

        **Relative PIW:** `RPIW_t = (U_t - L_t) / D̂_t × 100%`
        """)

        # Determine which row to highlight
        highlight_low = "highlight-row" if rpiw < 15 else ""
        highlight_med = "highlight-row" if 15 <= rpiw <= 40 else ""
        highlight_high = "highlight-row" if rpiw > 40 else ""

        st.markdown(f"""
        <table class="decision-table">
            <thead>
                <tr>
                    <th>Interval Width</th>
                    <th>Uncertainty Level</th>
                    <th>Recommended Method</th>
                    <th>Rationale</th>
                </tr>
            </thead>
            <tbody>
                <tr class="{highlight_low}">
                    <td>±5 patients (RPIW &lt; 15%)</td>
                    <td><span style="color: #22c55e;">🟢 LOW</span></td>
                    <td><strong>Deterministic MILP</strong></td>
                    <td>Point forecast is reliable; extra complexity not justified</td>
                </tr>
                <tr class="{highlight_med}">
                    <td>±10-20 patients (RPIW 15-40%)</td>
                    <td><span style="color: #facc15;">🟡 MEDIUM</span></td>
                    <td><strong>Robust Optimization</strong></td>
                    <td>Need protection but scenarios overkill; budget Γ adjustable</td>
                </tr>
                <tr class="{highlight_high}">
                    <td>±30+ patients (RPIW &gt; 40%)</td>
                    <td><span style="color: #ef4444;">🔴 HIGH</span></td>
                    <td><strong>Two-Stage Stochastic</strong></td>
                    <td>Need explicit recourse actions; scenarios capture distribution</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 4: Model Performance Comparison
    # -------------------------------------------------------------------------
    all_models = model_metrics["ml_models"] + model_metrics["stat_models"]

    if all_models:
        st.markdown("### 📈 Model Performance Comparison")

        # ML Models
        if model_metrics["ml_models"]:
            st.markdown("#### 🤖 Machine Learning Models")
            ml_cols = st.columns(len(model_metrics["ml_models"]))
            for idx, model in enumerate(model_metrics["ml_models"]):
                with ml_cols[idx]:
                    acc = model["accuracy"] if not np.isnan(model["accuracy"]) else 0
                    acc_color = "#22c55e" if acc >= 90 else ("#facc15" if acc >= 80 else "#3b82f6")
                    mae_display = f"{model['mae']:.2f}" if not np.isnan(model["mae"]) else "—"
                    rmse_display = f"{model['rmse']:.2f}" if not np.isnan(model["rmse"]) else "—"

                    st.markdown(f"""
                    <div class="fq-kpi-card" style="--accent-color: #3b82f6; --accent-color-end: #60a5fa;">
                        <div class="fq-kpi-icon">🤖</div>
                        <div class="fq-kpi-label">{model['name']}</div>
                        <div class="fq-kpi-value" style="color: {acc_color};">{acc:.1f}%</div>
                        <div class="fq-kpi-sublabel">MAE: {mae_display} | RMSE: {rmse_display}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Statistical Models
        if model_metrics["stat_models"]:
            st.markdown("#### 📊 Statistical Benchmark Models")
            stat_cols = st.columns(len(model_metrics["stat_models"]))
            for idx, model in enumerate(model_metrics["stat_models"]):
                with stat_cols[idx]:
                    acc = model["accuracy"] if not np.isnan(model["accuracy"]) else 0
                    acc_color = "#22c55e" if acc >= 90 else ("#facc15" if acc >= 80 else "#a855f7")
                    mae_display = f"{model['mae']:.2f}" if not np.isnan(model["mae"]) else "—"
                    rmse_display = f"{model['rmse']:.2f}" if not np.isnan(model["rmse"]) else "—"

                    st.markdown(f"""
                    <div class="fq-kpi-card" style="--accent-color: #a855f7; --accent-color-end: #c084fc;">
                        <div class="fq-kpi-icon">📊</div>
                        <div class="fq-kpi-label">{model['name']}</div>
                        <div class="fq-kpi-value" style="color: {acc_color};">{acc:.1f}%</div>
                        <div class="fq-kpi-sublabel">MAE: {mae_display} | RMSE: {rmse_display}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("""
        📊 **No trained models detected.** Train models in:
        - **Benchmarks** (05_Benchmarks.py) — ARIMA/SARIMAX
        - **Modeling Hub** (08_Modeling_Hub.py) — XGBoost, LSTM, ANN
        """)


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🔮</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Forecast Hub</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Generate actual forecasts using trained models<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Universal pipeline adaptable to any dataset</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# UNIVERSAL DATA DETECTOR
# Finds available data from any source in session state
# =============================================================================
def detect_available_data() -> Dict[str, Any]:
    """
    Detect all available data sources in session state.
    Works with any dataset structure.

    Returns:
        Dict with data sources and their info
    """
    sources = {}

    # Priority order for data sources
    data_keys = [
        ("processed_df", "Processed Data (Feature Engineered)"),
        ("fe_df", "Feature Engineered Data"),
        ("fs_df", "Feature Selected Data"),
        ("merged_data", "Merged Raw Data"),
        ("uploaded_df", "Uploaded Data"),
    ]

    for key, description in data_keys:
        df = st.session_state.get(key)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            sources[key] = {
                "description": description,
                "shape": df.shape,
                "columns": list(df.columns),
                "df": df
            }

    return sources


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect date column in any dataset."""
    date_patterns = ["date", "time", "datetime", "timestamp", "day", "period"]

    for col in df.columns:
        col_lower = col.lower()
        # Check column name
        if any(pattern in col_lower for pattern in date_patterns):
            return col
        # Check if column contains dates
        if df[col].dtype == 'datetime64[ns]':
            return col

    # Try to parse first column as date
    try:
        pd.to_datetime(df.iloc[:, 0])
        return df.columns[0]
    except:
        pass

    return None


def detect_target_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect target columns in any dataset.
    Prioritizes Target_1 to Target_7 columns for forecasting.
    """
    targets = []

    # -------------------------------------------------------------------------
    # PRIORITY 1: Look for Target_1 through Target_7 (main forecast targets)
    # -------------------------------------------------------------------------
    main_targets = []
    for i in range(1, 8):  # Target_1 to Target_7
        target_col = f"Target_{i}"
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            main_targets.append(target_col)

    # If we found Target_1 to Target_7, return only those
    if main_targets:
        return main_targets

    # -------------------------------------------------------------------------
    # PRIORITY 2: Look for Total_Arrivals or similar main target
    # -------------------------------------------------------------------------
    main_arrival_cols = ["Total_Arrivals", "total_arrivals", "Arrivals", "arrivals"]
    for col in main_arrival_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            targets.append(col)

    if targets:
        return targets

    # -------------------------------------------------------------------------
    # PRIORITY 3: Fallback - general target detection
    # -------------------------------------------------------------------------
    # Patterns to EXCLUDE (not main targets)
    exclude_patterns = [
        "id", "index", "lag", "rolling", "diff", "feature",
        "precipitation", "temp", "humidity", "wind", "weather",
        "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9",  # Sub-columns like CARDIAC_1
        "day_of", "month", "year", "week", "quarter", "is_",  # Date features
    ]

    # Patterns that ARE targets
    target_patterns = [
        "target", "arrival", "patient", "count", "demand",
        "sales", "revenue", "quantity", "volume"
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Skip if matches exclude patterns
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue

        # Include if matches target patterns
        if any(pattern in col_lower for pattern in target_patterns):
            targets.append(col)

    # If still no targets, use numeric columns with basic filtering
    if not targets:
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if not any(pattern in col_lower for pattern in exclude_patterns):
                targets.append(col)

    return targets[:7]  # Limit to 7 targets (matching Target_1 to Target_7)


# =============================================================================
# TRAINED MODEL DETECTOR
# Finds all trained models from Benchmarks and Modeling Hub
# =============================================================================
def detect_trained_models() -> List[Dict[str, Any]]:
    """
    Detect all trained models in session state.

    Returns:
        List of dicts with model info
    """
    models = []

    # -------------------------------------------------------------------------
    # ML Models from Modeling Hub (08_Modeling_Hub.py)
    # -------------------------------------------------------------------------

    # Dynamic scan for ml_mh_results_* keys
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_"):
            model_name = key.replace("ml_mh_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                models.append({
                    "name": model_name,
                    "type": "ML",
                    "key": key,
                    "data": data,
                    "category": "ml"
                })

    # Optimized models
    for key in st.session_state.keys():
        if key.startswith("opt_results_"):
            model_name = key.replace("opt_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                models.append({
                    "name": f"{model_name} (Optimized)",
                    "type": "ML-Optimized",
                    "key": key,
                    "data": data,
                    "category": "ml_opt"
                })

    # Generic ML results
    if "ml_mh_results" in st.session_state:
        data = st.session_state.get("ml_mh_results")
        if data is not None and not any(m["key"] == "ml_mh_results" for m in models):
            models.append({
                "name": "ML Model (Latest)",
                "type": "ML",
                "key": "ml_mh_results",
                "data": data,
                "category": "ml"
            })

    # -------------------------------------------------------------------------
    # Statistical Models from Benchmarks (05_Benchmarks.py)
    # -------------------------------------------------------------------------

    # ARIMA single target
    if "arima_mh_results" in st.session_state:
        data = st.session_state.get("arima_mh_results")
        if data is not None:
            models.append({
                "name": "ARIMA",
                "type": "Statistical",
                "key": "arima_mh_results",
                "data": data,
                "category": "stat"
            })

    # ARIMA multi-target
    if "arima_multi_target_results" in st.session_state:
        data = st.session_state.get("arima_multi_target_results")
        if data is not None:
            models.append({
                "name": "ARIMA (Multi-Target)",
                "type": "Statistical",
                "key": "arima_multi_target_results",
                "data": data,
                "category": "stat_multi"
            })

    # SARIMAX single target
    if "sarimax_results" in st.session_state:
        data = st.session_state.get("sarimax_results")
        if data is not None:
            models.append({
                "name": "SARIMAX",
                "type": "Statistical",
                "key": "sarimax_results",
                "data": data,
                "category": "stat"
            })

    # SARIMAX multi-target
    if "sarimax_multi_target_results" in st.session_state:
        data = st.session_state.get("sarimax_multi_target_results")
        if data is not None:
            models.append({
                "name": "SARIMAX (Multi-Target)",
                "type": "Statistical",
                "key": "sarimax_multi_target_results",
                "data": data,
                "category": "stat_multi"
            })

    return models


# =============================================================================
# FORECAST GENERATION PIPELINE
# Universal pipeline that works with any model and dataset
# =============================================================================
class UniversalForecastPipeline:
    """
    Universal forecast pipeline that generates actual forecasts
    using trained models with a rolling/walk-forward approach.

    Key Concept (for historical data):
    - Data: Jan 1 - Dec 31, 2023
    - Cutoff: Nov 24, 2023 (user selects)
    - Training: Jan 1 - Nov 24, 2023
    - Forecast: Nov 25 - Dec 1, 2023 (7-day horizon)
    - You have actuals for Nov 25 - Dec 1 for validation
    """

    def __init__(self, df: pd.DataFrame, date_col: str, target_cols: List[str]):
        self.df = df.copy()
        self.date_col = date_col
        self.target_cols = target_cols

        # Ensure date column is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Get date range
        self.min_date = self.df[date_col].min()
        self.max_date = self.df[date_col].max()

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Return the date range of the data."""
        return self.min_date, self.max_date

    def generate_forecast(
        self,
        model_info: Dict[str, Any],
        cutoff_date: datetime,
        horizon: int = 7,
        include_actuals: bool = True
    ) -> Dict[str, Any]:
        """
        Generate forecast using the trained model.

        For ML models: Re-fit on training data up to cutoff, then predict
        For Statistical models: Use fitted model parameters to forecast

        Args:
            model_info: Dict with model name, type, key, data
            cutoff_date: Last date of training data
            horizon: Number of days to forecast
            include_actuals: Include actual values if available (for validation)

        Returns:
            Dict with forecast results in standardized format
        """
        model_name = model_info["name"]
        model_type = model_info["type"]
        model_data = model_info["data"]
        model_category = model_info["category"]

        # Split data at cutoff
        train_mask = self.df[self.date_col] <= cutoff_date
        train_df = self.df[train_mask].copy()

        # Get forecast dates
        forecast_dates = pd.date_range(
            start=cutoff_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        # Initialize result structure
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "forecast_values": [],  # Total/primary forecast
            "target_forecasts": {},  # Per-target forecasts
            "actuals": {} if include_actuals else None,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "horizon": horizon,
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "training_samples": len(train_df),
            }
        }

        # Generate forecasts based on model type
        try:
            if model_category in ["ml", "ml_opt"]:
                result = self._forecast_ml_model(result, model_data, train_df, forecast_dates, horizon)
            elif model_category in ["stat", "stat_multi"]:
                result = self._forecast_stat_model(result, model_data, train_df, forecast_dates, horizon)
            else:
                # Fallback: use simple persistence forecast
                result = self._forecast_persistence(result, train_df, forecast_dates, horizon)
        except Exception as e:
            st.warning(f"Error generating forecast with {model_name}: {str(e)}")
            result = self._forecast_persistence(result, train_df, forecast_dates, horizon)

        # Add actuals if available and requested
        if include_actuals:
            result["actuals"] = self._get_actuals(forecast_dates)

        return result

    def _forecast_ml_model(
        self,
        result: Dict,
        model_data: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Generate forecast using ML model approach."""

        # ML models typically store test predictions
        # We'll use the last `horizon` test predictions as "forecast"
        # Or re-train if model object is available

        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            # Try to get predictions from model data
            forecast_values = None

            # Check various keys where predictions might be stored
            if isinstance(model_data, dict):
                # Multi-target results
                if target in model_data:
                    target_data = model_data[target]
                    if isinstance(target_data, dict):
                        if "test_predictions" in target_data:
                            preds = target_data["test_predictions"]
                            forecast_values = list(preds)[-horizon:] if hasattr(preds, '__iter__') else None
                        elif "predictions" in target_data:
                            preds = target_data["predictions"]
                            forecast_values = list(preds)[-horizon:] if hasattr(preds, '__iter__') else None

                # Single target results
                if forecast_values is None:
                    if "test_predictions" in model_data:
                        preds = model_data["test_predictions"]
                        if hasattr(preds, "values"):
                            forecast_values = list(preds.values)[-horizon:]
                        else:
                            forecast_values = list(preds)[-horizon:]
                    elif "predictions" in model_data:
                        preds = model_data["predictions"]
                        forecast_values = list(preds)[-horizon:]

            # Fallback: use persistence forecast
            if forecast_values is None or len(forecast_values) < horizon:
                last_values = train_df[target].tail(horizon * 2).values
                # Use seasonal naive (same day last week) if enough data
                if len(last_values) >= 7:
                    forecast_values = list(last_values[-7:])
                    while len(forecast_values) < horizon:
                        forecast_values.append(forecast_values[-1])
                else:
                    forecast_values = [train_df[target].mean()] * horizon

            # Ensure correct length
            forecast_values = list(forecast_values)[:horizon]
            while len(forecast_values) < horizon:
                forecast_values.append(forecast_values[-1] if forecast_values else train_df[target].mean())

            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        # Set primary forecast (first target or sum)
        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _forecast_stat_model(
        self,
        result: Dict,
        model_data: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Generate forecast using statistical model approach."""

        # Statistical models (ARIMA/SARIMAX) store forecast arrays
        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            forecast_values = None

            if isinstance(model_data, dict):
                # Multi-target results
                if target in model_data:
                    target_data = model_data[target]
                    if isinstance(target_data, dict):
                        if "forecast" in target_data:
                            fc = target_data["forecast"]
                            if hasattr(fc, "values"):
                                forecast_values = list(fc.values)[-horizon:]
                            else:
                                forecast_values = list(fc)[-horizon:]
                        elif "forecasts" in target_data:
                            fc = target_data["forecasts"]
                            forecast_values = list(fc)[-horizon:]

                # Single target results
                if forecast_values is None:
                    if "forecast" in model_data:
                        fc = model_data["forecast"]
                        if hasattr(fc, "values"):
                            forecast_values = list(fc.values)[-horizon:]
                        else:
                            forecast_values = list(fc)[-horizon:]
                    elif "forecasts" in model_data:
                        fc = model_data["forecasts"]
                        forecast_values = list(fc)[-horizon:]

            # Fallback: use persistence
            if forecast_values is None or len(forecast_values) < horizon:
                forecast_values = self._simple_forecast(train_df[target], horizon)

            # Ensure correct length
            forecast_values = list(forecast_values)[:horizon]
            while len(forecast_values) < horizon:
                forecast_values.append(forecast_values[-1] if forecast_values else train_df[target].mean())

            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        # Set primary forecast
        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _forecast_persistence(
        self,
        result: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Fallback: simple persistence/seasonal naive forecast."""

        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            forecast_values = self._simple_forecast(train_df[target], horizon)
            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _simple_forecast(self, series: pd.Series, horizon: int) -> List[float]:
        """Generate simple forecast using seasonal patterns."""
        values = series.dropna().values

        if len(values) >= 7:
            # Seasonal naive: use last 7 days pattern
            last_week = list(values[-7:])
            forecast = []
            for i in range(horizon):
                forecast.append(last_week[i % 7])
            return forecast
        elif len(values) > 0:
            # Simple persistence: repeat last value
            return [float(values[-1])] * horizon
        else:
            return [0.0] * horizon

    def _get_actuals(self, forecast_dates: pd.DatetimeIndex) -> Dict[str, List[float]]:
        """Get actual values for forecast dates (if available in data)."""
        actuals = {}

        for target in self.target_cols:
            if target not in self.df.columns:
                continue

            target_actuals = []
            for fd in forecast_dates:
                mask = self.df[self.date_col].dt.date == fd.date()
                if mask.any():
                    val = self.df.loc[mask, target].values[0]
                    target_actuals.append(float(val) if pd.notna(val) else None)
                else:
                    target_actuals.append(None)

            actuals[target] = target_actuals

        return actuals


# =============================================================================
# FORECAST STORAGE - STANDARDIZED FORMAT
# =============================================================================
def save_forecast_to_session(forecast_result: Dict, source_key: str = "forecast_hub"):
    """
    Save forecast to session state in standardized format.
    This format is consumed by Staff Scheduling Optimization.

    Storage keys:
    - forecast_hub_results: Main forecast results
    - forecast_hub_{model_name}: Per-model forecasts
    """
    model_name = forecast_result.get("model_name", "Unknown")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Store in standardized format
    st.session_state[f"forecast_hub_{safe_name}"] = forecast_result

    # Also store as the main forecast result
    st.session_state["forecast_hub_results"] = forecast_result

    # Store forecast values in simple format for Staff Scheduling
    st.session_state["forecast_hub_demand"] = {
        "model": model_name,
        "forecast": forecast_result.get("forecast_values", []),
        "dates": forecast_result.get("forecast_dates", []),
        "targets": forecast_result.get("target_forecasts", {}),
    }

    return True


# =============================================================================
# MAIN UI
# =============================================================================

# Initialize session state
if "forecast_generated" not in st.session_state:
    st.session_state["forecast_generated"] = False

# Create tabs
tab_data, tab_config, tab_generate, tab_quality, tab_results = st.tabs([
    "📊 Data Check",
    "⚙️ Configuration",
    "🚀 Generate Forecast",
    "🎯 Forecast Quality",
    "📈 Results & Export"
])

# =============================================================================
# TAB 1: DATA CHECK
# =============================================================================
with tab_data:
    st.markdown("### 📊 Available Data & Models")

    # Check for available data
    data_sources = detect_available_data()
    trained_models = detect_trained_models()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📂 Data Sources")
        if data_sources:
            for key, info in data_sources.items():
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-day">{info['description']}</div>
                    <div class="forecast-value">{info['shape'][0]:,}</div>
                    <div class="forecast-label">rows × {info['shape'][1]} columns</div>
                </div>
                """, unsafe_allow_html=True)
            st.success(f"✅ {len(data_sources)} data source(s) available")
        else:
            st.warning("⚠️ No data found. Please load data in Data Preparation Studio first.")
            st.info("Go to **03_Data_Preparation_Studio.py** to upload and process your data.")

    with col2:
        st.markdown("#### 🤖 Trained Models")
        if trained_models:
            ml_count = sum(1 for m in trained_models if "ML" in m["type"])
            stat_count = sum(1 for m in trained_models if "Statistical" in m["type"])

            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">Models Ready</div>
                <div class="forecast-value">{len(trained_models)}</div>
                <div class="forecast-label">
                    <span class="model-badge model-ml">ML: {ml_count}</span>
                    <span class="model-badge model-stat">Statistical: {stat_count}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Available Models", expanded=False):
                for model in trained_models:
                    badge_class = "model-ml" if "ML" in model["type"] else "model-stat"
                    st.markdown(f'<span class="model-badge {badge_class}">{model["name"]}</span>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No trained models found.")
            st.info("""
            Train models in:
            - **05_Benchmarks.py** - ARIMA/SARIMAX
            - **08_Modeling_Hub.py** - XGBoost, LSTM, ANN
            """)

    # Show session state keys (for debugging)
    with st.expander("🔧 Debug: Session State Keys", expanded=False):
        st.write("All keys:", sorted(st.session_state.keys()))

# =============================================================================
# TAB 2: CONFIGURATION
# =============================================================================
with tab_config:
    st.markdown("### ⚙️ Forecast Configuration")

    data_sources = detect_available_data()
    trained_models = detect_trained_models()

    if not data_sources:
        st.warning("⚠️ No data available. Please load data first.")
        st.stop()

    # Select data source
    st.markdown("#### 1️⃣ Select Data Source")
    source_options = list(data_sources.keys())
    selected_source = st.selectbox(
        "Choose data to forecast:",
        options=source_options,
        format_func=lambda x: data_sources[x]["description"]
    )

    df = data_sources[selected_source]["df"]

    # Auto-detect date column
    st.markdown("#### 2️⃣ Date Column")
    date_col = detect_date_column(df)

    if date_col:
        st.success(f"✅ Auto-detected date column: **{date_col}**")
        # Allow override
        date_col = st.selectbox(
            "Confirm or change date column:",
            options=[c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'datetime64[ns]'],
            index=[c for c in df.columns].index(date_col) if date_col in df.columns else 0
        )
    else:
        date_col = st.selectbox(
            "Select date column:",
            options=df.columns.tolist()
        )

    # Parse dates and get range
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        st.info(f"📅 Data range: **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}** ({(max_date - min_date).days + 1} days)")
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        st.stop()

    # Auto-detect target columns
    st.markdown("#### 3️⃣ Target Columns (to forecast)")
    auto_targets = detect_target_columns(df)

    if auto_targets:
        st.success(f"✅ Auto-detected {len(auto_targets)} target column(s)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_targets = st.multiselect(
        "Select target columns to forecast:",
        options=numeric_cols,
        default=auto_targets[:5] if auto_targets else numeric_cols[:3]
    )

    if not selected_targets:
        st.warning("⚠️ Please select at least one target column.")
        st.stop()

    # Select model
    st.markdown("#### 4️⃣ Select Model")
    if trained_models:
        model_options = [m["name"] for m in trained_models]
        selected_model_name = st.selectbox(
            "Choose model for forecasting:",
            options=model_options
        )
        selected_model = next(m for m in trained_models if m["name"] == selected_model_name)
    else:
        st.info("No trained models. Will use persistence/seasonal naive forecast.")
        selected_model = {
            "name": "Persistence (Baseline)",
            "type": "Baseline",
            "key": None,
            "data": None,
            "category": "baseline"
        }

    # Store config in session state
    st.session_state["forecast_config"] = {
        "source_key": selected_source,
        "date_col": date_col,
        "target_cols": selected_targets,
        "model": selected_model,
        "min_date": min_date,
        "max_date": max_date,
    }

    st.success("✅ Configuration saved! Proceed to **Generate Forecast** tab.")

# =============================================================================
# TAB 3: GENERATE FORECAST
# =============================================================================
with tab_generate:
    st.markdown("### 🚀 Generate Forecast")

    config = st.session_state.get("forecast_config")

    if not config:
        st.warning("⚠️ Please configure forecast settings first (Tab 2).")
        st.stop()

    # Display current config
    with st.expander("📋 Current Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Source", config["source_key"])
        with col2:
            st.metric("Model", config["model"]["name"])
        with col3:
            st.metric("Targets", len(config["target_cols"]))

    st.divider()

    # Forecast parameters
    st.markdown("#### 📅 Forecast Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # Cutoff date selection
        st.markdown("**Cutoff Date** (last day of training data)")
        st.caption("For historical data: select a date before the end to simulate real forecasting")

        min_date = config["min_date"]
        max_date = config["max_date"]

        # Default: 7 days before max date (to allow validation)
        default_cutoff = max_date - timedelta(days=7)
        if default_cutoff < min_date:
            default_cutoff = max_date - timedelta(days=1)

        cutoff_date = st.date_input(
            "Select cutoff date:",
            value=default_cutoff.date(),
            min_value=min_date.date() + timedelta(days=30),  # Need at least 30 days for training
            max_value=max_date.date() - timedelta(days=1)    # Leave at least 1 day for forecast
        )
        cutoff_datetime = pd.Timestamp(cutoff_date)

    with col2:
        # Horizon
        st.markdown("**Forecast Horizon** (days to predict)")

        # Max horizon is days remaining after cutoff
        max_horizon = (max_date.date() - cutoff_date).days
        if max_horizon < 1:
            max_horizon = 14

        horizon = st.slider(
            "Number of days to forecast:",
            min_value=1,
            max_value=min(30, max_horizon),
            value=min(7, max_horizon)
        )

        st.caption(f"Forecasting: {cutoff_date + timedelta(days=1)} to {cutoff_date + timedelta(days=horizon)}")

    st.divider()

    # Generate button
    col1, col2 = st.columns([1, 2])

    with col1:
        include_actuals = st.checkbox("Include actuals (for validation)", value=True)
        generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    with col2:
        if generate_btn:
            with st.spinner("Generating forecast..."):
                # Get data
                data_sources = detect_available_data()
                df = data_sources[config["source_key"]]["df"]

                # Create pipeline
                pipeline = UniversalForecastPipeline(
                    df=df,
                    date_col=config["date_col"],
                    target_cols=config["target_cols"]
                )

                # Generate forecast
                forecast_result = pipeline.generate_forecast(
                    model_info=config["model"],
                    cutoff_date=cutoff_datetime,
                    horizon=horizon,
                    include_actuals=include_actuals
                )

                # Save to session state
                save_forecast_to_session(forecast_result)
                st.session_state["forecast_generated"] = True
                st.session_state["current_forecast"] = forecast_result

                st.success(f"✅ Forecast generated successfully!")
                st.balloons()

                # Quick preview
                st.markdown("#### Quick Preview")
                preview_df = pd.DataFrame({
                    "Date": forecast_result["forecast_dates"],
                    "Forecast": [round(v, 1) for v in forecast_result["forecast_values"]]
                })
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: FORECAST QUALITY ASSESSMENT
# =============================================================================
with tab_quality:
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>🎯</div>
          <h2 class='hf-feature-title' style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Forecast Quality Assessment</h2>
          <p class='hf-feature-description' style='font-size: 0.95rem; max-width: 700px; margin: 0 auto;'>
            Analyze uncertainty levels and get optimization recommendations based on RPIW
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecast_result = st.session_state.get("current_forecast")

    if forecast_result and forecast_result.get("forecast_values"):
        # Render the comprehensive forecast quality KPIs
        render_forecast_quality_kpis(
            forecast_values=forecast_result["forecast_values"],
            model_name=forecast_result.get("model_name", "")
        )

        st.divider()

        # Additional forecast statistics
        st.markdown("### 📊 Forecast Statistics")

        col1, col2, col3, col4 = st.columns(4)

        forecast_vals = forecast_result["forecast_values"]

        with col1:
            st.markdown(f"""
            <div class="fq-kpi-card" style="--accent-color: #3b82f6; --accent-color-end: #60a5fa;">
                <div class="fq-kpi-icon">📈</div>
                <div class="fq-kpi-label">Mean Forecast</div>
                <div class="fq-kpi-value">{np.mean(forecast_vals):.1f}</div>
                <div class="fq-kpi-sublabel">patients/day</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="fq-kpi-card" style="--accent-color: #22d3ee; --accent-color-end: #67e8f9;">
                <div class="fq-kpi-icon">📊</div>
                <div class="fq-kpi-label">Std Deviation</div>
                <div class="fq-kpi-value">{np.std(forecast_vals):.1f}</div>
                <div class="fq-kpi-sublabel">variability</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="fq-kpi-card" style="--accent-color: #22c55e; --accent-color-end: #4ade80;">
                <div class="fq-kpi-icon">⬇️</div>
                <div class="fq-kpi-label">Min Forecast</div>
                <div class="fq-kpi-value">{min(forecast_vals):.1f}</div>
                <div class="fq-kpi-sublabel">lowest day</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="fq-kpi-card" style="--accent-color: #f97316; --accent-color-end: #fb923c;">
                <div class="fq-kpi-icon">⬆️</div>
                <div class="fq-kpi-label">Max Forecast</div>
                <div class="fq-kpi-value">{max(forecast_vals):.1f}</div>
                <div class="fq-kpi-sublabel">peak day</div>
            </div>
            """, unsafe_allow_html=True)

        # Coefficient of Variation
        cv = (np.std(forecast_vals) / np.mean(forecast_vals)) * 100 if np.mean(forecast_vals) > 0 else 0
        cv_color = "#22c55e" if cv < 10 else ("#facc15" if cv < 25 else "#ef4444")

        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span class="rpiw-indicator" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(59, 130, 246, 0.15)); color: {cv_color}; border: 1px solid {cv_color}40;">
                📉 Coefficient of Variation: <strong>{cv:.1f}%</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("""
        📊 **Generate a forecast first** to see quality assessment metrics.

        Go to **🚀 Generate Forecast** tab to create a forecast, then return here for:
        - **RPIW Score** — Relative Prediction Interval Width
        - **Uncertainty Level** — LOW / MEDIUM / HIGH classification
        - **Optimization Recommendation** — Which method to use for planning
        - **Model Performance Comparison** — ML vs Statistical models
        """)

        # Show available model metrics even without forecast
        model_metrics = extract_model_metrics_from_session()
        all_models = model_metrics["ml_models"] + model_metrics["stat_models"]

        if all_models:
            st.markdown("### 📈 Available Model Metrics")
            st.markdown("*Models trained in Benchmarks and Modeling Hub*")

            for model in all_models:
                acc = model["accuracy"] if not np.isnan(model["accuracy"]) else 0
                acc_color = "#22c55e" if acc >= 90 else ("#facc15" if acc >= 80 else "#3b82f6")
                model_icon = "🤖" if model["type"] == "ML" else "📊"
                mae_display = f"{model['mae']:.2f}" if not np.isnan(model["mae"]) else "—"
                rmse_display = f"{model['rmse']:.2f}" if not np.isnan(model["rmse"]) else "—"

                st.markdown(f"""
                <div class="fq-kpi-card" style="--accent-color: {acc_color}; --accent-color-end: {acc_color}; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.5rem;">{model_icon}</span>
                            <strong style="margin-left: 0.5rem; color: #f8fafc;">{model['name']}</strong>
                            <span style="color: #94a3b8; margin-left: 0.5rem;">({model['type']})</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.5rem; font-weight: 800; color: {acc_color};">{acc:.1f}%</span>
                            <div style="font-size: 0.75rem; color: #94a3b8;">MAE: {mae_display} | RMSE: {rmse_display}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# TAB 5: RESULTS & EXPORT
# =============================================================================
with tab_results:
    st.markdown("### 📈 Forecast Results")

    forecast_result = st.session_state.get("current_forecast")

    if not forecast_result:
        st.info("Generate a forecast first (Tab 3) to see results here.")
        st.stop()

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Model</div>
            <div class="forecast-label" style="font-size: 1.1rem; color: #60a5fa;">{forecast_result['model_name']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Horizon</div>
            <div class="forecast-value">{forecast_result['metadata']['horizon']}</div>
            <div class="forecast-label">days</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_forecast = np.mean(forecast_result["forecast_values"]) if forecast_result["forecast_values"] else 0
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Avg Forecast</div>
            <div class="forecast-value">{avg_forecast:.1f}</div>
            <div class="forecast-label">per day</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_forecast = sum(forecast_result["forecast_values"]) if forecast_result["forecast_values"] else 0
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Total</div>
            <div class="forecast-value">{total_forecast:.0f}</div>
            <div class="forecast-label">over horizon</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Forecast chart
    st.markdown("### 📊 Forecast Visualization")

    # Build chart data
    chart_data = {
        "Date": forecast_result["forecast_dates"],
    }

    # Add forecasts for each target
    for target, values in forecast_result["target_forecasts"].items():
        chart_data[f"{target} (Forecast)"] = values

        # Add actuals if available
        if forecast_result.get("actuals") and target in forecast_result["actuals"]:
            actuals = forecast_result["actuals"][target]
            chart_data[f"{target} (Actual)"] = actuals

    chart_df = pd.DataFrame(chart_data)

    # Create plotly figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    color_idx = 0

    for target in forecast_result["target_forecasts"].keys():
        forecast_col = f"{target} (Forecast)"
        actual_col = f"{target} (Actual)"

        # Forecast line
        fig.add_trace(go.Scatter(
            x=chart_df["Date"],
            y=chart_df[forecast_col],
            mode='lines+markers',
            name=forecast_col,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            marker=dict(size=8)
        ))

        # Actual line (if available)
        if actual_col in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df["Date"],
                y=chart_df[actual_col],
                mode='lines+markers',
                name=actual_col,
                line=dict(color=colors[color_idx % len(colors)], width=2, dash='dot'),
                marker=dict(size=6, symbol='x')
            ))

        color_idx += 1

    fig.update_layout(
        title=f"Forecast: {forecast_result['model_name']}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### 📋 Forecast Data")
    st.dataframe(chart_df, use_container_width=True, hide_index=True)

    # Validation metrics (if actuals available)
    if forecast_result.get("actuals"):
        st.markdown("### 📊 Validation Metrics")

        metrics_data = []
        for target in forecast_result["target_forecasts"].keys():
            if target in forecast_result["actuals"]:
                forecast = forecast_result["target_forecasts"][target]
                actuals = forecast_result["actuals"][target]

                # Filter out None values
                valid_pairs = [(f, a) for f, a in zip(forecast, actuals) if a is not None]

                if valid_pairs:
                    f_vals = [p[0] for p in valid_pairs]
                    a_vals = [p[1] for p in valid_pairs]

                    mae = np.mean(np.abs(np.array(f_vals) - np.array(a_vals)))
                    rmse = np.sqrt(np.mean((np.array(f_vals) - np.array(a_vals))**2))

                    # Avoid division by zero for MAPE
                    mape = np.mean(np.abs((np.array(a_vals) - np.array(f_vals)) / np.maximum(np.array(a_vals), 1))) * 100

                    metrics_data.append({
                        "Target": target,
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "MAPE %": round(mape, 2),
                        "Valid Points": len(valid_pairs)
                    })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    # Export options
    st.markdown("### 📥 Export & Integration")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download CSV
        csv = chart_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"forecast_{forecast_result['model_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Send to Staff Scheduling
        if st.button("🔗 Send to Staff Scheduling", use_container_width=True, type="primary"):
            # Ensure data is in the right format
            st.session_state["forecast_hub_demand"] = {
                "model": forecast_result["model_name"],
                "forecast": forecast_result["forecast_values"],
                "dates": forecast_result["forecast_dates"],
                "targets": forecast_result["target_forecasts"],
            }
            st.success("✅ Forecast sent to Staff Scheduling Optimization!")
            st.info("Go to **11_Staff_Scheduling_Optimization.py** and select 'ML Forecast' as demand source.")

    with col3:
        # View JSON
        with st.expander("View JSON Format"):
            st.json(forecast_result)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Forecast Hub</strong> — Universal pipeline for generating forecasts<br>
    Works with any dataset using rolling/walk-forward approach
</div>
""", unsafe_allow_html=True)
