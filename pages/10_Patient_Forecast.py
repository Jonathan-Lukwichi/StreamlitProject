# =============================================================================
# 10_Forecast.py — Universal Forecast Hub
# View forecasts from trained models (Benchmarks & Modeling Hub)
# Reads existing test forecasts - no re-generation needed
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
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header
from app_core.ui.effects import inject_fluorescent_effects

# Seasonal Proportions for category distribution (DOW x Monthly)
from app_core.analytics.seasonal_proportions import (
    SeasonalProportionConfig,
    calculate_seasonal_proportions,
    combine_proportions_multiplicatively,
)

# Cloud Model Sync (Train Locally, Deploy to Cloud)
from app_core.ui.model_download_ui import render_cloud_models_info

# Preprocessing artifacts check (for cloud deployment)
def _ensure_preprocessing_on_startup():
    """
    Check and download preprocessing artifacts (transformers, feature selection)
    from Supabase Storage if not present locally.

    This enables the 'Train Locally, Deploy to Cloud' workflow where
    preprocessing must match training data scaling.
    """
    try:
        from app_core.data.model_storage_service import ensure_preprocessing_artifacts
        status = ensure_preprocessing_artifacts()
        # Only log, don't show warning to user (artifacts may not be needed)
        import logging
        logger = logging.getLogger(__name__)
        if status.get("transformers"):
            logger.info("Transformers downloaded from cloud storage")
        if status.get("feature_selection"):
            logger.info("Feature selection config downloaded from cloud storage")
    except ImportError:
        pass  # Model storage service not available
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Preprocessing check: {e}")

# Auto-check on page load
_ensure_preprocessing_on_startup()

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Patient Forecast - HealthForecast AI",
    page_icon="🔮",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
inject_fluorescent_effects()

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

/* Forecast Mini Cards */
.forecast-mini-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.2);
    text-align: center;
    transition: all 0.3s ease;
}

.forecast-mini-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.forecast-mini-card-primary {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(139, 92, 246, 0.15));
    border: 2px solid rgba(59, 130, 246, 0.4);
}

.forecast-number {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.forecast-accuracy {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #10b981;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS - Extract forecasts from session state
# =============================================================================

def _safe_float(x, default=np.nan):
    """Safely convert to float."""
    try:
        f = float(x)
        return f if np.isfinite(f) else default
    except:
        return default


def extract_arima_forecast_data() -> Optional[Dict[str, Any]]:
    """
    Extract forecast data from ARIMA results in session state.

    Returns dict with:
        - F: Forecast matrix (T x H)
        - L: Lower CI matrix (T x H)
        - U: Upper CI matrix (T x H)
        - test_eval: DataFrame with actual values
        - metrics_df: DataFrame with metrics per horizon
        - horizons: List of horizon numbers
    """
    arima_results = st.session_state.get("arima_mh_results")
    if not arima_results or not isinstance(arima_results, dict):
        return None

    per_h = arima_results.get("per_h", {})
    results_df = arima_results.get("results_df")
    successful = arima_results.get("successful", [])

    if not per_h or not successful:
        return None

    try:
        horizons = sorted(successful)
        H = len(horizons)

        # Get test set length from first horizon
        first_h = horizons[0]
        y_test_first = per_h[first_h].get("y_test")
        if y_test_first is None:
            return None
        T = len(y_test_first)

        if T == 0:
            return None

        # Initialize matrices
        F = np.full((T, H), np.nan)
        L = np.full((T, H), np.nan)
        U = np.full((T, H), np.nan)
        test_eval_dict = {}

        # Fill matrices from per_h data
        for idx, h in enumerate(horizons):
            h_data = per_h[h]

            fc = h_data.get("forecast")
            lo = h_data.get("ci_lo")
            hi = h_data.get("ci_hi")
            yt = h_data.get("y_test")

            if fc is not None:
                fc_vals = fc.values if hasattr(fc, 'values') else np.array(fc)
                if len(fc_vals) == T:
                    F[:, idx] = fc_vals

            if lo is not None:
                lo_vals = lo.values if hasattr(lo, 'values') else np.array(lo)
                if len(lo_vals) == T:
                    L[:, idx] = lo_vals

            if hi is not None:
                hi_vals = hi.values if hasattr(hi, 'values') else np.array(hi)
                if len(hi_vals) == T:
                    U[:, idx] = hi_vals

            if yt is not None:
                test_eval_dict[f"Target_{h}"] = yt.values if hasattr(yt, 'values') else yt

        # Create test_eval DataFrame with date index
        test_eval = pd.DataFrame(test_eval_dict, index=y_test_first.index)

        return {
            "model": "ARIMA",
            "F": F,
            "L": L,
            "U": U,
            "test_eval": test_eval,
            "metrics_df": results_df,
            "horizons": horizons,
        }
    except Exception as e:
        st.warning(f"Error extracting ARIMA data: {e}")
        return None


def extract_sarimax_forecast_data() -> Optional[Dict[str, Any]]:
    """
    Extract forecast data from SARIMAX results in session state.
    """
    sarimax_results = st.session_state.get("sarimax_results")
    if not sarimax_results or not isinstance(sarimax_results, dict):
        return None

    try:
        from app_core.models.sarimax_pipeline import to_multihorizon_artifacts
        metrics_df, F, L, U, test_eval, _, _, horizons = to_multihorizon_artifacts(sarimax_results)

        if F is None or test_eval is None or len(F) == 0:
            return None

        return {
            "model": "SARIMAX",
            "F": F,
            "L": L,
            "U": U,
            "test_eval": test_eval,
            "metrics_df": metrics_df,
            "horizons": horizons if horizons else list(range(1, F.shape[1] + 1)),
        }
    except Exception as e:
        st.warning(f"Error extracting SARIMAX data: {e}")
        return None


def extract_ml_forecast_data(model_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract forecast data from ML model results in session state.

    Args:
        model_key: Session state key (e.g., "ml_mh_results_xgboost")
    """
    ml_results = st.session_state.get(model_key)
    if not ml_results or not isinstance(ml_results, dict):
        return None

    per_h = ml_results.get("per_h", {})
    results_df = ml_results.get("results_df")
    successful = ml_results.get("successful", [])

    if not per_h:
        return None

    try:
        # Get horizons
        if successful:
            horizons = sorted(successful)
        else:
            horizons = sorted([int(h) for h in per_h.keys()])

        if not horizons:
            return None

        H = len(horizons)

        # Get test set length from first horizon
        first_h = horizons[0]
        y_test_first = per_h[first_h].get("y_test")
        if y_test_first is None:
            # Try alternate key
            y_test_first = per_h[first_h].get("actual")
        if y_test_first is None:
            return None

        T = len(y_test_first)
        if T == 0:
            return None

        # Initialize matrices
        F = np.full((T, H), np.nan)
        L = np.full((T, H), np.nan)
        U = np.full((T, H), np.nan)
        test_eval_dict = {}

        # Fill matrices from per_h data
        for idx, h in enumerate(horizons):
            h_data = per_h[h]

            # Forecast values - use explicit None checks to avoid array truth value error
            fc = h_data.get("forecast")
            if fc is None:
                fc = h_data.get("predictions")
            if fc is None:
                fc = h_data.get("y_pred")

            if fc is not None:
                fc_vals = fc.values if hasattr(fc, 'values') else np.array(fc)
                if len(fc_vals) == T:
                    F[:, idx] = fc_vals

            # Confidence intervals (if available) - explicit None checks
            lo = h_data.get("ci_lo")
            if lo is None:
                lo = h_data.get("lower")

            hi = h_data.get("ci_hi")
            if hi is None:
                hi = h_data.get("upper")

            if lo is not None:
                lo_vals = lo.values if hasattr(lo, 'values') else np.array(lo)
                if len(lo_vals) == T:
                    L[:, idx] = lo_vals

            if hi is not None:
                hi_vals = hi.values if hasattr(hi, 'values') else np.array(hi)
                if len(hi_vals) == T:
                    U[:, idx] = hi_vals

            # Actual values - explicit None check
            yt = h_data.get("y_test")
            if yt is None:
                yt = h_data.get("actual")
            if yt is not None:
                test_eval_dict[f"Target_{h}"] = yt.values if hasattr(yt, 'values') else yt

        # Create test_eval DataFrame - get datetime index
        # PRIORITY: Use 'datetime' field from per_h (same approach as Dashboard & Modeling Hub)
        test_index = None

        # 1. First try: Get datetime directly from per_h (Modeling Hub stores it here)
        first_h_data = per_h[first_h]
        dt_field = first_h_data.get("datetime")
        if dt_field is not None:
            try:
                dt_array = dt_field.values if hasattr(dt_field, 'values') else np.array(dt_field)
                if len(dt_array) == T:
                    test_index = pd.to_datetime(dt_array)
            except Exception:
                pass

        # 2. Second try: Check if y_test_first already has a datetime index
        if test_index is None and hasattr(y_test_first, 'index'):
            idx = y_test_first.index
            if isinstance(idx, pd.DatetimeIndex):
                test_index = idx
            elif hasattr(idx, 'dtype') and pd.api.types.is_datetime64_any_dtype(idx):
                test_index = idx

        # 3. Third try: Get dates from fusion_df or processed_df in session state
        if test_index is None:
            fusion_df = st.session_state.get("fusion_df")
            if fusion_df is not None and hasattr(fusion_df, 'index'):
                if isinstance(fusion_df.index, pd.DatetimeIndex):
                    if len(fusion_df) >= T:
                        test_index = fusion_df.index[-T:]
                elif 'Date' in fusion_df.columns or 'date' in fusion_df.columns:
                    date_col = 'Date' if 'Date' in fusion_df.columns else 'date'
                    try:
                        dates = pd.to_datetime(fusion_df[date_col])
                        if len(dates) >= T:
                            test_index = pd.DatetimeIndex(dates[-T:])
                    except:
                        pass

            if test_index is None:
                processed_df = st.session_state.get("processed_df")
                if processed_df is not None:
                    if isinstance(processed_df.index, pd.DatetimeIndex) and len(processed_df) >= T:
                        test_index = processed_df.index[-T:]

        # Create test_eval DataFrame with appropriate index
        if test_index is not None and len(test_index) == T:
            test_eval = pd.DataFrame(test_eval_dict, index=test_index)
        elif hasattr(y_test_first, 'index'):
            test_eval = pd.DataFrame(test_eval_dict, index=y_test_first.index)
        else:
            test_eval = pd.DataFrame(test_eval_dict)

        model_name = model_key.replace("ml_mh_results_", "")

        return {
            "model": model_name,
            "F": F,
            "L": L,
            "U": U,
            "test_eval": test_eval,
            "metrics_df": results_df,
            "horizons": horizons,
        }
    except Exception as e:
        st.warning(f"Error extracting ML data from {model_key}: {e}")
        return None


def extract_hybrid_forecast_data(hybrid_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract forecast data from two-stage hybrid model results.

    Args:
        hybrid_key: Key within hybrid_pipeline_results (e.g., "lstm_xgb")
    """
    hybrid_results = st.session_state.get("hybrid_pipeline_results", {})
    if not hybrid_results or hybrid_key not in hybrid_results:
        return None

    hybrid_data = hybrid_results[hybrid_key]
    if not hybrid_data or not isinstance(hybrid_data, dict) or not hybrid_data.get("success"):
        return None

    per_h = hybrid_data.get("per_h", {})
    if not per_h:
        return None

    try:
        horizons = sorted([int(h) for h in per_h.keys()])
        if not horizons:
            return None

        H = len(horizons)

        # Get test set length from first horizon
        first_h = horizons[0]
        first_h_data = per_h[first_h]

        # Hybrid results store actual values
        y_actual = first_h_data.get("actual")
        if y_actual is None:
            return None

        T = len(y_actual)
        if T == 0:
            return None

        # Initialize matrices
        F = np.full((T, H), np.nan)  # Forecasts (final_pred)
        L = np.full((T, H), np.nan)  # Lower bound (not available for hybrids)
        U = np.full((T, H), np.nan)  # Upper bound (not available for hybrids)
        test_eval_dict = {}

        # Fill matrices from per_h data
        for idx, h in enumerate(horizons):
            h_data = per_h[h]

            # Use final_pred as forecast (Stage1 + Stage2 correction)
            fc = h_data.get("final_pred")
            if fc is not None:
                fc_arr = np.array(fc).flatten()
                F[:len(fc_arr), idx] = fc_arr[:T]

            # Store evaluation metrics
            test_eval_dict[h] = {
                "MAE": h_data.get("mae"),
                "RMSE": h_data.get("rmse"),
            }

        test_eval = pd.DataFrame(test_eval_dict).T
        test_eval.index.name = "Horizon"
        test_eval = test_eval.reset_index()

        model_name = hybrid_data.get("model_name", hybrid_key)

        return {
            "horizons": horizons,
            "F": F,
            "L": L,
            "U": U,
            "test_eval": test_eval,
            "model_name": model_name,
            "type": "Hybrid",
        }

    except Exception as e:
        print(f"Hybrid forecast extraction failed for {hybrid_key}: {e}")
        return None


def get_available_forecast_models() -> List[Dict[str, Any]]:
    """
    Get list of all models with available forecast data.

    Returns list of dicts with model info.
    """
    models = []

    # Check ARIMA
    arima_data = extract_arima_forecast_data()
    if arima_data:
        models.append({
            "name": "ARIMA",
            "type": "Statistical",
            "extractor": extract_arima_forecast_data,
            "data": arima_data,
        })

    # Check SARIMAX
    sarimax_data = extract_sarimax_forecast_data()
    if sarimax_data:
        models.append({
            "name": "SARIMAX",
            "type": "Statistical",
            "extractor": extract_sarimax_forecast_data,
            "data": sarimax_data,
        })

    # Check ML models
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_"):
            ml_data = extract_ml_forecast_data(key)
            if ml_data:
                model_name = key.replace("ml_mh_results_", "")
                models.append({
                    "name": model_name,
                    "type": "ML",
                    "extractor": lambda k=key: extract_ml_forecast_data(k),
                    "data": ml_data,
                })

    # Check two-stage hybrid models
    hybrid_results = st.session_state.get("hybrid_pipeline_results", {})
    if hybrid_results and isinstance(hybrid_results, dict):
        hybrid_name_map = {
            "lstm_xgb": "LSTM→XGBoost",
            "sarimax_xgb": "SARIMAX→XGBoost",
            "lstm_sarimax": "LSTM→SARIMAX"
        }
        for hybrid_key, hybrid_data in hybrid_results.items():
            if hybrid_data and isinstance(hybrid_data, dict) and hybrid_data.get("success"):
                hybrid_forecast = extract_hybrid_forecast_data(hybrid_key)
                if hybrid_forecast:
                    display_name = hybrid_name_map.get(hybrid_key, hybrid_data.get("model_name", hybrid_key))
                    models.append({
                        "name": display_name,
                        "type": "Hybrid",
                        "extractor": lambda k=hybrid_key: extract_hybrid_forecast_data(k),
                        "data": hybrid_forecast,
                    })

    return models


def get_all_trained_models_from_session() -> Dict[str, Dict[str, Any]]:
    """
    Dynamically detect ALL models that have been trained in the Modeling Hub.

    This function scans session state for all model-related keys and returns
    a comprehensive dictionary of all trained models with their metadata.

    Returns:
        Dict mapping model_key to model info dict containing:
            - name: Display name
            - type: Model category (Statistical, ML, Optimized, Hybrid)
            - source: Session state key
            - has_ci: Whether confidence intervals are available
            - status: Training status
    """
    trained_models = {}

    # Statistical Models
    if "arima_mh_results" in st.session_state and st.session_state["arima_mh_results"]:
        arima_data = st.session_state["arima_mh_results"]
        has_ci = False
        if "per_h" in arima_data:
            first_h = list(arima_data["per_h"].keys())[0] if arima_data["per_h"] else None
            if first_h and arima_data["per_h"][first_h].get("ci_lo") is not None:
                has_ci = True
        trained_models["arima_mh_results"] = {
            "name": "ARIMA",
            "type": "Statistical",
            "source": "arima_mh_results",
            "has_ci": has_ci,
            "status": "✅ Trained",
            "horizons": arima_data.get("successful", [])
        }

    if "sarimax_results" in st.session_state and st.session_state["sarimax_results"]:
        sarimax_data = st.session_state["sarimax_results"]
        has_ci = False
        if "per_h" in sarimax_data:
            first_h = list(sarimax_data["per_h"].keys())[0] if sarimax_data["per_h"] else None
            if first_h and sarimax_data["per_h"][first_h].get("ci_lo") is not None:
                has_ci = True
        trained_models["sarimax_results"] = {
            "name": "SARIMAX",
            "type": "Statistical",
            "source": "sarimax_results",
            "has_ci": has_ci,
            "status": "✅ Trained",
            "horizons": sarimax_data.get("successful", [])
        }

    # ML Models - scan for all ml_mh_results_* keys
    ml_model_types = ["XGBoost", "LSTM", "ANN", "RandomForest", "GradientBoosting", "LightGBM"]
    for model_type in ml_model_types:
        key = f"ml_mh_results_{model_type}"
        if key in st.session_state and st.session_state[key]:
            ml_data = st.session_state[key]
            has_ci = False
            if "per_h" in ml_data:
                first_h = list(ml_data["per_h"].keys())[0] if ml_data["per_h"] else None
                if first_h:
                    h_data = ml_data["per_h"][first_h]
                    if h_data.get("ci_lo") is not None or h_data.get("lower") is not None:
                        has_ci = True
            trained_models[key] = {
                "name": model_type,
                "type": "ML",
                "source": key,
                "has_ci": has_ci,
                "status": "✅ Trained",
                "horizons": ml_data.get("successful", list(ml_data.get("per_h", {}).keys()))
            }

    # Also check for any other ml_mh_results_* keys dynamically
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_") and key not in trained_models:
            ml_data = st.session_state[key]
            if ml_data and isinstance(ml_data, dict):
                model_name = key.replace("ml_mh_results_", "")
                has_ci = False
                if "per_h" in ml_data:
                    first_h = list(ml_data["per_h"].keys())[0] if ml_data["per_h"] else None
                    if first_h:
                        h_data = ml_data["per_h"][first_h]
                        if h_data.get("ci_lo") is not None or h_data.get("lower") is not None:
                            has_ci = True
                trained_models[key] = {
                    "name": model_name,
                    "type": "ML",
                    "source": key,
                    "has_ci": has_ci,
                    "status": "✅ Trained",
                    "horizons": ml_data.get("successful", list(ml_data.get("per_h", {}).keys()))
                }

    # Optimized Models
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        opt_key = f"opt_results_{model_type}"
        if opt_key in st.session_state and st.session_state[opt_key]:
            opt_data = st.session_state[opt_key]
            trained_models[opt_key] = {
                "name": f"{model_type} (Optimized)",
                "type": "Optimized",
                "source": opt_key,
                "has_ci": False,  # Optimized models typically don't have CI
                "status": "⚡ Optimized",
                "best_params": opt_data.get("best_params", {}),
                "best_score": opt_data.get("best_score"),
                "method": st.session_state.get("opt_last_method", "Unknown")
            }

    # Hybrid Models
    if "ensemble_results" in st.session_state and st.session_state["ensemble_results"]:
        trained_models["ensemble_results"] = {
            "name": "Ensemble",
            "type": "Hybrid",
            "source": "ensemble_results",
            "has_ci": False,
            "status": "🔗 Hybrid"
        }

    if "stacking_results" in st.session_state and st.session_state["stacking_results"]:
        trained_models["stacking_results"] = {
            "name": "Stacking",
            "type": "Hybrid",
            "source": "stacking_results",
            "has_ci": False,
            "status": "🔗 Hybrid"
        }

    if "decomposition_results" in st.session_state and st.session_state["decomposition_results"]:
        trained_models["decomposition_results"] = {
            "name": "Decomposition",
            "type": "Hybrid",
            "source": "decomposition_results",
            "has_ci": False,
            "status": "🔗 Hybrid"
        }

    # LSTM Hybrid Models (legacy format)
    lstm_hybrid_keys = ["lstm_sarimax_results", "lstm_xgb_results", "lstm_ann_results"]
    lstm_hybrid_names = ["LSTM+SARIMAX", "LSTM+XGBoost", "LSTM+ANN"]

    for key, name in zip(lstm_hybrid_keys, lstm_hybrid_names):
        if key in st.session_state and st.session_state[key]:
            trained_models[key] = {
                "name": name,
                "type": "Hybrid",
                "source": key,
                "has_ci": False,
                "status": "🔗 Hybrid"
            }

    # Two-Stage Residual Correction Hybrid Models (new format from Train Models page)
    hybrid_results = st.session_state.get("hybrid_pipeline_results", {})
    if hybrid_results and isinstance(hybrid_results, dict):
        hybrid_name_map = {
            "lstm_xgb": "LSTM→XGBoost",
            "sarimax_xgb": "SARIMAX→XGBoost",
            "lstm_sarimax": "LSTM→SARIMAX"
        }
        for hybrid_key, hybrid_data in hybrid_results.items():
            if hybrid_data and isinstance(hybrid_data, dict) and hybrid_data.get("success"):
                display_name = hybrid_name_map.get(hybrid_key, hybrid_data.get("model_name", hybrid_key))
                model_key = f"hybrid_{hybrid_key}"
                trained_models[model_key] = {
                    "name": display_name,
                    "type": "Hybrid",
                    "source": "hybrid_pipeline_results",
                    "sub_key": hybrid_key,
                    "has_ci": False,
                    "status": "🧬 Two-Stage",
                    "horizons": list(hybrid_data.get("per_h", {}).keys()),
                    "avg_mae": hybrid_data.get("avg_mae"),
                    "avg_rmse": hybrid_data.get("avg_rmse")
                }

    return trained_models


def extract_model_evaluation_metrics(model_key: str, model_info: Dict) -> Dict[str, Any]:
    """
    Extract evaluation metrics (MAE, RMSE, MAPE, Accuracy) for a given model.

    Args:
        model_key: Session state key for the model
        model_info: Model info dict from get_all_trained_models_from_session()

    Returns:
        Dict with MAE, RMSE, MAPE, Accuracy values or None if not available
    """
    metrics = {"MAE": None, "RMSE": None, "MAPE": None, "Accuracy": None}

    # Handle two-stage hybrid models (stored in hybrid_pipeline_results)
    if model_info.get("source") == "hybrid_pipeline_results":
        sub_key = model_info.get("sub_key")
        hybrid_results = st.session_state.get("hybrid_pipeline_results", {})
        if sub_key and sub_key in hybrid_results:
            hybrid_data = hybrid_results[sub_key]
            if hybrid_data and isinstance(hybrid_data, dict):
                metrics["MAE"] = hybrid_data.get("avg_mae")
                metrics["RMSE"] = hybrid_data.get("avg_rmse")
                return metrics
        return metrics

    model_data = st.session_state.get(model_key)
    if not model_data or not isinstance(model_data, dict):
        return metrics

    # Try to get metrics from results_df
    results_df = model_data.get("results_df")
    if results_df is not None and not results_df.empty:
        # Calculate average metrics across all horizons
        mae_cols = [c for c in results_df.columns if "MAE" in c.upper() and "TEST" in c.upper()]
        rmse_cols = [c for c in results_df.columns if "RMSE" in c.upper() and "TEST" in c.upper()]
        mape_cols = [c for c in results_df.columns if "MAPE" in c.upper() and "TEST" in c.upper()]
        acc_cols = [c for c in results_df.columns if "ACC" in c.upper() and "TEST" in c.upper()]

        # Fallback to non-test columns
        if not mae_cols:
            mae_cols = [c for c in results_df.columns if "MAE" in c.upper()]
        if not rmse_cols:
            rmse_cols = [c for c in results_df.columns if "RMSE" in c.upper()]
        if not mape_cols:
            mape_cols = [c for c in results_df.columns if "MAPE" in c.upper()]
        if not acc_cols:
            acc_cols = [c for c in results_df.columns if "ACC" in c.upper() or "ACCURACY" in c.upper()]

        try:
            if mae_cols:
                metrics["MAE"] = results_df[mae_cols[0]].mean()
            if rmse_cols:
                metrics["RMSE"] = results_df[rmse_cols[0]].mean()
            if mape_cols:
                metrics["MAPE"] = results_df[mape_cols[0]].mean()
            if acc_cols:
                metrics["Accuracy"] = results_df[acc_cols[0]].mean()
        except Exception:
            pass

    # Try per_h data if results_df didn't have the metrics
    per_h = model_data.get("per_h", {})
    if per_h and metrics["MAE"] is None:
        mae_list, rmse_list, mape_list, acc_list = [], [], [], []
        for h, h_data in per_h.items():
            if isinstance(h_data, dict):
                # Check for metrics in h_data
                if "mae" in h_data:
                    mae_list.append(h_data["mae"])
                if "rmse" in h_data:
                    rmse_list.append(h_data["rmse"])
                if "mape" in h_data:
                    mape_list.append(h_data["mape"])
                if "accuracy" in h_data:
                    acc_list.append(h_data["accuracy"])

        if mae_list:
            metrics["MAE"] = np.mean(mae_list)
        if rmse_list:
            metrics["RMSE"] = np.mean(rmse_list)
        if mape_list:
            metrics["MAPE"] = np.mean(mape_list)
        if acc_list:
            metrics["Accuracy"] = np.mean(acc_list)

    # For optimized models, try to get metrics from best_score
    if model_info.get("type") == "Optimized":
        best_score = model_info.get("best_score")
        if best_score is not None:
            # Usually best_score is negative MSE or similar
            if best_score < 0:
                metrics["RMSE"] = np.sqrt(-best_score) if metrics["RMSE"] is None else metrics["RMSE"]

    return metrics


def get_metrics_from_df(metrics_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    """Extract metrics for a specific horizon from metrics DataFrame."""
    if metrics_df is None or metrics_df.empty:
        return {}

    # Sanitize Horizon column
    if "Horizon" in metrics_df.columns:
        df = metrics_df.copy()
        if df["Horizon"].dtype == 'object':
            df["Horizon"] = df["Horizon"].astype(str).str.replace("h=", "").str.extract(r"(\d+)", expand=False)
        df["Horizon"] = pd.to_numeric(df["Horizon"], errors='coerce')
        df = df.dropna(subset=["Horizon"])
        df["Horizon"] = df["Horizon"].astype(int)

        h_row = df[df["Horizon"] == horizon]
        if not h_row.empty:
            row = h_row.iloc[0]
            return {
                "MAE": _safe_float(row.get("Test_MAE") or row.get("MAE")),
                "RMSE": _safe_float(row.get("Test_RMSE") or row.get("RMSE")),
                "MAPE": _safe_float(row.get("Test_MAPE") or row.get("MAPE_%") or row.get("MAPE")),
                "Accuracy": _safe_float(row.get("Test_Acc") or row.get("Accuracy_%") or row.get("Accuracy")),
            }

    return {}


def save_forecast_to_session(
    forecast_data: Dict,
    model_name: str,
    horizons: List[int] = None,
    forecast_days: int = 7,
    category_forecasts: Dict[str, List[float]] = None,
    base_date=None
):
    """
    Save comprehensive forecast data to session state for use by Staff Scheduling,
    Inventory Management, and Decision Command Center.

    This is the SINGLE SOURCE OF TRUTH for forecast data across all pages.

    Args:
        forecast_data: Dictionary containing F, L, U matrices and test_eval
        model_name: Name of the model
        horizons: List of horizon numbers
        forecast_days: Number of forecast days
        category_forecasts: Dict mapping category -> list of daily forecasts
                           e.g., {"RESPIRATORY": [45, 48, 42, ...], "CARDIAC": [32, 35, 30, ...]}
        base_date: The base date from which forecast dates are calculated (from test set)
    """
    if not forecast_data:
        return

    from datetime import datetime, timedelta

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Get dates from test_eval index
    test_eval = forecast_data.get("test_eval")
    dates = []
    if test_eval is not None and hasattr(test_eval.index, '__iter__'):
        try:
            dates = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in test_eval.index]
        except:
            pass

    # Get forecast matrix
    F = forecast_data.get("F")
    L = forecast_data.get("L")
    U = forecast_data.get("U")

    # Get the last row forecasts for each horizon (future predictions)
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    if F is not None and len(F) > 0:
        num_horizons = min(forecast_days, F.shape[1]) if F.shape[1] > 0 else 0
        last_idx = len(F) - 1

        for h_idx in range(num_horizons):
            # Get forecast value
            fv = float(F[last_idx, h_idx]) if not np.isnan(F[last_idx, h_idx]) else 0
            forecast_values.append(max(0, fv))  # Ensure non-negative

            # Get confidence intervals
            if L is not None and not np.isnan(L[last_idx, h_idx]):
                lower_bounds.append(max(0, float(L[last_idx, h_idx])))
            else:
                lower_bounds.append(max(0, fv * 0.9))

            if U is not None and not np.isnan(U[last_idx, h_idx]):
                upper_bounds.append(max(0, float(U[last_idx, h_idx])))
            else:
                upper_bounds.append(max(0, fv * 1.1))

    # Generate future dates for the forecast horizon
    today = datetime.now().date()
    future_dates = [(today + timedelta(days=i+1)).strftime("%a %m/%d") for i in range(len(forecast_values))]

    # Determine model type
    model_type = "Statistical" if model_name in ["ARIMA", "SARIMAX"] else "ML"

    # Get metrics if available
    metrics_df = forecast_data.get("metrics_df")
    avg_accuracy = None
    avg_mape = None
    if metrics_df is not None and not metrics_df.empty:
        acc_col = next((c for c in metrics_df.columns if "Acc" in c or "Accuracy" in c), None)
        mape_col = next((c for c in metrics_df.columns if "MAPE" in c), None)
        if acc_col:
            avg_accuracy = float(metrics_df[acc_col].mean())
        if mape_col:
            avg_mape = float(metrics_df[mape_col].mean())

    # Store comprehensive forecast data - SINGLE SOURCE OF TRUTH
    st.session_state["forecast_hub_results"] = {
        "model_name": model_name,
        "forecast_data": forecast_data,
        "timestamp": datetime.now().isoformat(),
    }

    # Primary forecast data for Pages 11, 12, 13
    st.session_state["forecast_hub_demand"] = {
        "model": model_name,
        "model_type": model_type,
        "forecast": forecast_values,
        "forecasts": forecast_values,  # Alias for compatibility
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "dates": future_dates,
        "historical_dates": dates,
        "horizon": len(forecast_values),
        "avg_accuracy": avg_accuracy,
        "accuracy": avg_accuracy,  # Alias for compatibility
        "avg_mape": avg_mape,
        "timestamp": datetime.now().isoformat(),
        "source": "Forecast Hub (Page 10)",
        "model_name": model_name,  # Alias for compatibility
        # NEW: Category-level forecasts for Staff & Inventory optimization
        "category_forecasts": category_forecasts,
    }

    # Also store in a dedicated key for active forecast tracking
    st.session_state["active_forecast"] = {
        "model": model_name,
        "model_type": model_type,
        "model_name": model_name,  # Alias for compatibility
        "forecasts": forecast_values,
        "lower": lower_bounds,
        "upper": upper_bounds,
        "dates": future_dates,
        "horizon": len(forecast_values),
        "accuracy": avg_accuracy,
        "updated_at": datetime.now().isoformat(),
        # NEW: Category-level forecasts for Staff & Inventory optimization
        "category_forecasts": category_forecasts,
    }


# =============================================================================
# HEADER
# =============================================================================
render_scifi_hero_header(
    title="Patient Forecast",
    subtitle="View forecasts from trained models. Uses test set predictions from Benchmarks & Modeling Hub.",
    status="SYSTEM ONLINE"
)


# =============================================================================
# MAIN UI
# =============================================================================

# Get available models with forecast data
available_models = get_available_forecast_models()

if not available_models:
    st.warning("⚠️ No trained models with forecast data found.")
    st.info("""
    **Train models first to view forecasts:**

    - **Benchmarks** (05_Benchmarks.py) — ARIMA / SARIMAX
    - **Modeling Hub** (08_Modeling_Hub.py) — XGBoost, LSTM, ANN

    After training, the test set forecasts will be available here.
    """)

    # Cloud sync info - for Train Locally, Deploy to Cloud workflow
    render_cloud_models_info()

    # Show session state keys for debugging
    with st.expander("🔧 Debug: Session State Keys", expanded=False):
        model_keys = [k for k in st.session_state.keys() if 'result' in k.lower() or 'model' in k.lower()]
        st.write("Model-related keys:", model_keys)

    st.stop()

# Create tabs
tab_overview, tab_forecast, tab_comparison, tab_export = st.tabs([
    "📊 Overview",
    "🔮 Forecast View",
    "📈 Model Comparison",
    "📥 Export"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.markdown("### 📊 Available Models & Forecasts")

    # Summary cards
    col1, col2, col3 = st.columns(3)

    ml_models = [m for m in available_models if m["type"] == "ML"]
    stat_models = [m for m in available_models if m["type"] == "Statistical"]

    with col1:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Total Models</div>
            <div class="forecast-value">{len(available_models)}</div>
            <div class="forecast-label">with forecast data</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">ML Models</div>
            <div class="forecast-value">{len(ml_models)}</div>
            <div class="forecast-label">XGBoost, LSTM, etc.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Statistical Models</div>
            <div class="forecast-value">{len(stat_models)}</div>
            <div class="forecast-label">ARIMA, SARIMAX</div>
        </div>
        """, unsafe_allow_html=True)

    # Check for optimized models
    optimized_models = []
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        opt_key = f"opt_results_{model_type}"
        if opt_key in st.session_state and st.session_state[opt_key] is not None:
            optimized_models.append(model_type)

    if optimized_models:
        st.markdown("### 🔬 Hyperparameter Optimization Status")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1));
                    border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">⚡</span>
                <span style="font-weight: 600; color: #e2e8f0;">Optimized Models Available</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.875rem;">
                The following models have hyperparameter optimization results:
                <span style="color: #a78bfa; font-weight: 600;">{', '.join(optimized_models)}</span>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: #64748b;">
                💡 View detailed optimization results in <strong>09_Results.py → 🔬 Hyperparameter Tuning</strong> tab
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Model details
    st.markdown("### 🤖 Model Details")

    for model_info in available_models:
        data = model_info["data"]
        test_eval = data.get("test_eval")
        F = data.get("F")
        horizons = data.get("horizons", [])

        # Get date range
        date_range = "N/A"
        if test_eval is not None and hasattr(test_eval.index, 'min'):
            try:
                min_date = test_eval.index.min()
                max_date = test_eval.index.max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            except:
                pass

        badge_class = "model-ml" if model_info["type"] == "ML" else "model-stat"

        # Check if this model has optimization results
        is_optimized = model_info["name"] in optimized_models
        opt_badge = " ⚡ Optimized" if is_optimized else ""

        with st.expander(f"**{model_info['name']}** ({model_info['type']}){opt_badge}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Test Samples", F.shape[0] if F is not None else 0)
            with col2:
                st.metric("Horizons", len(horizons))
            with col3:
                st.metric("Max Horizon", max(horizons) if horizons else 0)
            with col4:
                st.metric("Date Range", date_range)

            # Show optimization info if available
            if is_optimized:
                opt_results = st.session_state.get(f"opt_results_{model_info['name']}")
                if opt_results:
                    st.markdown("**🔬 Hyperparameter Optimization Results:**")
                    opt_col1, opt_col2 = st.columns(2)

                    with opt_col1:
                        best_params = opt_results.get("best_params", {})
                        if best_params:
                            params_df = pd.DataFrame(
                                list(best_params.items())[:5],  # Show first 5 params
                                columns=["Parameter", "Value"]
                            )
                            st.dataframe(params_df, use_container_width=True, hide_index=True, height=150)

                    with opt_col2:
                        all_scores = opt_results.get("all_scores", {})
                        if all_scores:
                            st.metric("Optimized Accuracy", f"{all_scores.get('Accuracy', 0):.2f}%")
                            st.metric("Optimized RMSE", f"{all_scores.get('RMSE', 0):.4f}")

                    st.caption("💡 Full details in **09_Results.py → 🔬 Hyperparameter Tuning** tab")
                    st.divider()

            # Show metrics if available
            metrics_df = data.get("metrics_df")
            if metrics_df is not None and not metrics_df.empty:
                st.markdown("**Performance Metrics:**")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: FORECAST VIEW
# =============================================================================
with tab_forecast:
    st.markdown("### 🔮 View Forecasts")

    # Model selection
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        model_names = [m["name"] for m in available_models]
        selected_model_name = st.selectbox(
            "Select Model:",
            options=model_names,
            key="forecast_model_select"
        )

    # Get selected model data
    selected_model = next(m for m in available_models if m["name"] == selected_model_name)
    forecast_data = selected_model["data"]

    F = forecast_data["F"]
    L = forecast_data["L"]
    U = forecast_data["U"]
    test_eval = forecast_data["test_eval"]
    metrics_df = forecast_data.get("metrics_df")
    horizons = forecast_data["horizons"]

    # Date selection - with safe date handling
    with col2:
        selected_date = None
        try:
            if hasattr(test_eval.index, 'min') and len(test_eval) > 0:
                min_date = test_eval.index.min()
                max_date = test_eval.index.max()

                # Safe conversion to datetime
                if hasattr(min_date, 'to_pydatetime'):
                    min_dt = min_date.to_pydatetime()
                    max_dt = max_date.to_pydatetime()
                elif isinstance(min_date, (pd.Timestamp, datetime)):
                    min_dt = min_date
                    max_dt = max_date
                else:
                    # Fallback for integer index
                    min_dt = None
                    max_dt = None

                if min_dt is not None and max_dt is not None:
                    # Calculate date difference safely
                    try:
                        date_diff = (max_dt - min_dt).days if hasattr(max_dt - min_dt, 'days') else 7
                    except:
                        date_diff = 7

                    if date_diff > 7:
                        default_date = max_dt - timedelta(days=7)
                    else:
                        default_date = min_dt

                    selected_date = st.date_input(
                        "Forecast From:",
                        value=default_date.date() if hasattr(default_date, 'date') else default_date,
                        min_value=min_dt.date() if hasattr(min_dt, 'date') else min_dt,
                        max_value=max_dt.date() if hasattr(max_dt, 'date') else max_dt,
                        key="forecast_date_select"
                    )
        except Exception as e:
            st.caption(f"Date selection unavailable")

    with col3:
        max_h = len(horizons)
        forecast_days = st.selectbox(
            "Days to Show:",
            options=[h for h in [1, 3, 7] if h <= max_h] or [max_h],
            index=min(2, max_h - 1) if max_h > 0 else 0,
            key="forecast_days_select"
        )

    with col4:
        st.write("")  # Spacer
        generate_forecast = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    st.divider()

    # Only show forecast when button is clicked or if already generated
    if "forecast_generated" not in st.session_state:
        st.session_state.forecast_generated = False

    if generate_forecast:
        st.session_state.forecast_generated = True
        st.session_state.forecast_model = selected_model_name

    if not st.session_state.forecast_generated:
        st.info("👆 Select a model and click **Generate Forecast** to view predictions.")
    else:
        # Find the index for selected date
        forecast_idx = -1
        if selected_date is not None:
            try:
                selected_ts = pd.Timestamp(selected_date)
                if hasattr(test_eval.index, 'get_indexer'):
                    idx_loc = test_eval.index.get_indexer([selected_ts], method='nearest')[0]
                    forecast_idx = int(idx_loc)
                else:
                    forecast_idx = len(test_eval) - 1
            except:
                forecast_idx = len(test_eval) - 1
        else:
            forecast_idx = len(test_eval) - 1

        if forecast_idx >= 0 and forecast_idx < len(test_eval):
            base_date = test_eval.index[forecast_idx]

            # Format base_date safely
            if hasattr(base_date, 'strftime'):
                base_date_str = base_date.strftime("%b %d, %Y")
            else:
                base_date_str = str(base_date)

            # --- CATEGORY CONFIGURATION AND PROPORTIONS ---
            category_config = {
                "RESPIRATORY": {"icon": "🫁", "color": "#3b82f6"},
                "CARDIAC": {"icon": "❤️", "color": "#ef4444"},
                "TRAUMA": {"icon": "🩹", "color": "#f59e0b"},
                "GASTROINTESTINAL": {"icon": "🤢", "color": "#10b981"},
                "INFECTIOUS": {"icon": "🦠", "color": "#8b5cf6"},
                "NEUROLOGICAL": {"icon": "🧠", "color": "#ec4899"},
                "OTHER": {"icon": "📋", "color": "#6b7280"},
            }

            # Calculate SEASONAL proportions (DOW x Monthly) for category distribution
            seasonal_props = None
            merged_df = st.session_state.get("processed_df")
            if merged_df is None or merged_df.empty:
                merged_df = st.session_state.get("merged_data")

            if merged_df is not None and not merged_df.empty:
                # Find date column
                date_col = None
                for col in ["Date", "date", "DATE"]:
                    if col in merged_df.columns:
                        date_col = col
                        break

                if date_col is not None:
                    try:
                        config = SeasonalProportionConfig(
                            use_dow_seasonality=True,
                            use_monthly_seasonality=True,
                            normalization_method="multiplicative",
                        )
                        seasonal_props = calculate_seasonal_proportions(
                            df=merged_df,
                            config=config,
                            date_col=date_col,
                            category_cols=list(category_config.keys()),
                        )
                    except Exception as e:
                        st.warning(f"Could not calculate seasonal proportions: {e}")
                        seasonal_props = None

            # Helper function for smart rounding
            def distribute_with_smart_rounding(total: int, proportions: dict) -> dict:
                """Distribute total into categories using proportions with smart rounding."""
                if not proportions:
                    return {}
                result = {}
                remainders = {}
                allocated = 0
                for cat, prop in proportions.items():
                    exact_val = total * prop
                    floored_val = int(exact_val)
                    result[cat] = floored_val
                    remainders[cat] = exact_val - floored_val
                    allocated += floored_val
                remaining = total - allocated
                sorted_cats = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
                for i in range(min(remaining, len(sorted_cats))):
                    result[sorted_cats[i]] += 1
                return result

            # Pre-calculate category breakdown for each horizon
            # UNIFIED PIPELINE: Use seasonal proportions for ALL model types
            # (Statistical, ML, Hybrid, and Optimized models all use the same approach)
            category_forecasts_by_horizon = {}
            has_seasonal_props = seasonal_props is not None

            if has_seasonal_props:
                # Use SEASONAL PROPORTIONS (DOW x Monthly) for distribution
                for h_idx in range(min(forecast_days, len(horizons))):
                    horizon_num = horizons[h_idx]
                    total_forecast = int(round(float(F[forecast_idx, h_idx])))

                    # Calculate forecast date for this horizon
                    try:
                        forecast_date = pd.Timestamp(base_date) + pd.Timedelta(days=horizon_num)
                    except:
                        forecast_date = pd.Timestamp.now() + pd.Timedelta(days=horizon_num)

                    # Get seasonal proportions for this specific date (DOW x Monthly)
                    date_proportions = combine_proportions_multiplicatively(
                        dow_proportions=seasonal_props.dow_proportions,
                        monthly_proportions=seasonal_props.monthly_proportions,
                        target_date=forecast_date,
                        category_cols=list(category_config.keys())
                    )

                    # Distribute using seasonal proportions with smart rounding
                    category_forecasts_by_horizon[horizon_num] = distribute_with_smart_rounding(
                        total_forecast, date_proportions
                    )

            # --- FORECAST HEADER ---
            if has_seasonal_props:
                categories_text = " • <span style='color: #10b981;'>Categories (Seasonal DOW×Monthly)</span>"
            else:
                categories_text = ""
            st.markdown(f"""
            <div class='forecast-card' style='padding: 1.5rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='font-size: 1.25rem; font-weight: 700; color: #e2e8f0;'>
                            📅 {forecast_days}-Day Forecast
                        </div>
                        <div style='font-size: 0.875rem; color: #94a3b8; margin-top: 0.25rem;'>
                            Model: <span style='color: #22d3ee;'>{selected_model_name}</span> •
                            From: <span style='color: #a78bfa;'>{base_date_str}</span>{categories_text}
                        </div>
                    </div>
                    <div style='padding: 0.5rem 1rem; background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px;'>
                        <span style='color: #a78bfa; font-weight: 600;'>⚡ Test Set Forecast</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")  # Spacer

            # --- FORECAST CARDS ---
            num_cols = min(forecast_days, len(horizons), 7)
            forecast_cols = st.columns(num_cols, gap="small")

            for h_idx in range(min(forecast_days, len(horizons))):
                horizon_num = horizons[h_idx]

                # Safe date calculation
                forecast_dt_day = f"DAY {horizon_num}"
                forecast_dt_date = f"h={horizon_num}"
                if hasattr(base_date, '__add__'):
                    try:
                        forecast_dt = base_date + pd.Timedelta(days=horizon_num)
                        if hasattr(forecast_dt, 'strftime'):
                            forecast_dt_day = forecast_dt.strftime("%a").upper()
                            forecast_dt_date = forecast_dt.strftime("%b %d")
                    except:
                        pass

                # Get forecast and CI values
                forecast_val = float(F[forecast_idx, h_idx]) if not np.isnan(F[forecast_idx, h_idx]) else 0
                lower_val = float(L[forecast_idx, h_idx]) if L is not None and not np.isnan(L[forecast_idx, h_idx]) else forecast_val * 0.9
                upper_val = float(U[forecast_idx, h_idx]) if U is not None and not np.isnan(U[forecast_idx, h_idx]) else forecast_val * 1.1

                # Get actual value if available
                actual_col = f"Target_{horizon_num}"
                actual_val = None
                if actual_col in test_eval.columns:
                    actual_val = test_eval.iloc[forecast_idx][actual_col]
                    if pd.isna(actual_val):
                        actual_val = None

                # Get accuracy from metrics
                metrics = get_metrics_from_df(metrics_df, horizon_num)
                accuracy = metrics.get("Accuracy", np.nan)
                accuracy_text = f"{accuracy:.0f}%" if pd.notna(accuracy) else "N/A"

                # Accuracy color
                if pd.notna(accuracy):
                    if accuracy >= 90:
                        acc_color = "#10b981"
                        acc_bg = "rgba(16, 185, 129, 0.15)"
                    elif accuracy >= 80:
                        acc_color = "#22d3ee"
                        acc_bg = "rgba(34, 211, 238, 0.15)"
                    elif accuracy >= 70:
                        acc_color = "#f59e0b"
                        acc_bg = "rgba(245, 158, 11, 0.15)"
                    else:
                        acc_color = "#ef4444"
                        acc_bg = "rgba(239, 68, 68, 0.15)"
                else:
                    acc_color = "#64748b"
                    acc_bg = "rgba(100, 116, 139, 0.15)"

                col_idx = h_idx % num_cols
                with forecast_cols[col_idx]:
                    is_primary = (h_idx == 0)
                    border_style = "border: 2px solid rgba(59, 130, 246, 0.4);" if is_primary else "border: 1px solid rgba(59, 130, 246, 0.2);"
                    bg_style = "background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(139, 92, 246, 0.15));" if is_primary else "background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));"

                    # Build card HTML - no nested f-strings to avoid rendering issues
                    card_html = f"""<div style="{bg_style} {border_style} border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.5rem;">
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;">{forecast_dt_day}</div>
                        <div style="font-size: 0.7rem; color: #64748b; background: rgba(100, 116, 139, 0.15); padding: 0.2rem 0.5rem; border-radius: 4px; display: inline-block; margin: 0.25rem 0;">{forecast_dt_date}</div>
                        <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.5rem 0;">{int(forecast_val)}</div>
                        <div style="font-size: 0.6rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.5px;">Total Patients</div>
                        <div style="display: inline-block; padding: 0.25rem 0.5rem; background: {acc_bg}; border: 1px solid {acc_color}40; border-radius: 6px; font-size: 0.7rem; font-weight: 600; color: {acc_color}; margin: 0.5rem 0;">✓ {accuracy_text}</div>
                        <div style="font-size: 0.65rem; color: #64748b; background: rgba(30, 41, 59, 0.5); padding: 0.25rem 0.5rem; border-radius: 4px;">{int(lower_val)} - {int(upper_val)}</div>
                    </div>"""

                    st.markdown(card_html, unsafe_allow_html=True)

                    # Category breakdown expander
                    horizon_categories = category_forecasts_by_horizon.get(horizon_num, {})
                    if horizon_categories:
                        with st.expander("📊 Categories", expanded=False):
                            for cat, cfg in category_config.items():
                                cat_val = horizon_categories.get(cat, 0)
                                if pd.notna(cat_val) and cat_val > 0:
                                    st.markdown(f"""
                                    <div style='display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid rgba(100, 116, 139, 0.1);'>
                                        <span style='display: flex; align-items: center; gap: 6px;'>
                                            <span style='font-size: 0.875rem;'>{cfg['icon']}</span>
                                            <span style='color: #94a3b8; font-size: 0.75rem; font-weight: 600;'>{cat[:4]}</span>
                                        </span>
                                        <span style='color: {cfg["color"]}; font-weight: 700; font-size: 0.875rem;'>{int(round(cat_val))}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                    # Show actual vs predicted comparison separately (not as nested HTML)
                    if actual_val is not None:
                        error = abs(forecast_val - actual_val)
                        error_color = "#10b981" if error < 5 else "#f59e0b"
                        actual_html = f"""<div style="background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); border-radius: 8px; padding: 0.5rem; text-align: center; margin-top: -0.25rem;">
                            <div style="font-size: 0.65rem; color: #64748b;">Actual</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #22d3ee;">{int(actual_val)}</div>
                            <div style="font-size: 0.6rem; color: {error_color};">Error: ±{error:.1f}</div>
                        </div>"""
                        st.markdown(actual_html, unsafe_allow_html=True)

            # --- CATEGORY STATISTICS SECTION ---
            if has_seasonal_props:
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div class='forecast-card' style='padding: 1.25rem;'>
                    <div style='margin-bottom: 1rem;'>
                        <div style='font-size: 1rem; font-weight: 700; color: #e2e8f0;'>📊 Category Statistics</div>
                        <div style='font-size: 0.75rem; color: #64748b;'>Average daily arrivals and seasonal proportion (DOW × Monthly) per category</div>
                    </div>
                """, unsafe_allow_html=True)

                # Get seasonal proportions for base_date (for display)
                base_date_proportions = {}
                if seasonal_props is not None:
                    try:
                        base_date_proportions = combine_proportions_multiplicatively(
                            dow_proportions=seasonal_props.dow_proportions,
                            monthly_proportions=seasonal_props.monthly_proportions,
                            target_date=pd.Timestamp(base_date),
                            category_cols=list(category_config.keys())
                        )
                    except:
                        pass

                # Create summary cards for each category
                num_cats = len(category_config)
                cat_cols = st.columns(num_cats, gap="small")

                for idx, (cat, config) in enumerate(category_config.items()):
                    # Calculate average for category from forecasts
                    total_for_cat = 0
                    count = 0
                    for h_cats in category_forecasts_by_horizon.values():
                        if cat in h_cats and pd.notna(h_cats[cat]):
                            total_for_cat += h_cats[cat]
                            count += 1
                    avg_for_cat = total_for_cat / count if count > 0 else 0

                    # Get seasonal proportion for this category (from base_date)
                    proportion = base_date_proportions.get(cat, 0) * 100

                    # Color based on proportion
                    if proportion >= 15:
                        prop_color = "#10b981"
                        prop_bg = "rgba(16, 185, 129, 0.15)"
                    elif proportion >= 10:
                        prop_color = "#22d3ee"
                        prop_bg = "rgba(34, 211, 238, 0.15)"
                    else:
                        prop_color = "#a78bfa"
                        prop_bg = "rgba(167, 139, 250, 0.15)"

                    with cat_cols[idx]:
                        if avg_for_cat > 0 or proportion > 0:
                            st.markdown(f"""
                            <div style='
                                background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
                                border-radius: 12px;
                                padding: 0.875rem 0.5rem;
                                text-align: center;
                                border: 1px solid {config['color']}40;
                            '>
                                <div style='font-size: 1.25rem; margin-bottom: 0.125rem;'>{config['icon']}</div>
                                <div style='font-size: 0.5625rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.375rem;'>{cat[:4]}</div>
                                <div style='font-size: 1.25rem; font-weight: 800; color: {config['color']}; margin-bottom: 0.125rem;'>~{int(round(avg_for_cat))}</div>
                                <div style='font-size: 0.5rem; color: #64748b; margin-bottom: 0.375rem;'>avg/day</div>
                                <div style='
                                    display: inline-block;
                                    padding: 0.1875rem 0.375rem;
                                    background: {prop_bg};
                                    border: 1px solid {prop_color}40;
                                    border-radius: 4px;
                                    font-size: 0.5625rem;
                                    font-weight: 700;
                                    color: {prop_color};
                                '>{proportion:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='
                                background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.6));
                                border-radius: 12px;
                                padding: 0.875rem 0.5rem;
                                text-align: center;
                                border: 1px solid rgba(100, 116, 139, 0.2);
                                opacity: 0.5;
                            '>
                                <div style='font-size: 1.25rem; margin-bottom: 0.125rem;'>{config['icon']}</div>
                                <div style='font-size: 0.5625rem; color: #64748b; text-transform: uppercase;'>{cat[:4]}</div>
                                <div style='font-size: 0.875rem; color: #64748b; margin: 0.375rem 0;'>—</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # --- SEASONAL PROPORTIONS INFO EXPANDER ---
                if seasonal_props is not None:
                    with st.expander("📊 Seasonal Proportions Details (DOW × Monthly)", expanded=False):
                        st.markdown("""
                        <div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 1rem;'>
                            Seasonal proportions combine <strong>Day-of-Week</strong> and <strong>Monthly</strong> patterns
                            to distribute total forecasts across clinical categories. Each cell shows the proportion
                            of that category for that time period.
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Day-of-Week Proportions:**")
                            dow_display = seasonal_props.dow_proportions.copy()
                            dow_display = dow_display * 100  # Convert to percentage
                            st.dataframe(
                                dow_display.style.format("{:.1f}%").background_gradient(cmap='Blues', axis=0),
                                use_container_width=True,
                                height=300
                            )
                        with col2:
                            st.markdown("**Monthly Proportions:**")
                            monthly_display = seasonal_props.monthly_proportions.copy()
                            monthly_display = monthly_display * 100  # Convert to percentage
                            st.dataframe(
                                monthly_display.style.format("{:.1f}%").background_gradient(cmap='Greens', axis=0),
                                use_container_width=True,
                                height=300
                            )

                        st.info(f"""
                        **How it works:** For a forecast on {base_date_str}:
                        - Get DOW factor for that day (e.g., Monday)
                        - Get Monthly factor for that month (e.g., January)
                        - Multiply: Final = DOW × Monthly
                        - Normalize so all categories sum to 100%
                        """)

        # --- TIME SERIES CHART ---
        st.markdown("### 📊 Forecast vs Actual")

        # Create chart
        fig = go.Figure()

        # Determine window to show
        window_days = 30
        start_idx = max(0, forecast_idx - window_days)
        end_idx = min(len(test_eval), forecast_idx + forecast_days + 1)

        # Get dates
        dates_window = test_eval.index[start_idx:end_idx]

        # Get actual values (Target_1)
        if "Target_1" in test_eval.columns:
            actual_values = test_eval["Target_1"].iloc[start_idx:end_idx].values
            fig.add_trace(go.Scatter(
                x=dates_window,
                y=actual_values,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#22d3ee', width=2.5),
                marker=dict(size=5),
            ))

        # Add forecast points
        forecast_dates_plot = []
        forecast_values_plot = []
        lower_values_plot = []
        upper_values_plot = []

        for h_idx in range(min(forecast_days, len(horizons))):
            horizon_num = horizons[h_idx]
            try:
                fd = base_date + pd.Timedelta(days=horizon_num)
            except:
                continue

            try:
                is_in_window = fd in dates_window
            except:
                is_in_window = False

            if is_in_window:
                fv = F[forecast_idx, h_idx]
                if pd.notna(fv):
                    forecast_dates_plot.append(fd)
                    forecast_values_plot.append(fv)
                    lower_values_plot.append(L[forecast_idx, h_idx] if L is not None else fv * 0.9)
                    upper_values_plot.append(U[forecast_idx, h_idx] if U is not None else fv * 1.1)

        if forecast_dates_plot:
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_dates_plot + forecast_dates_plot[::-1],
                y=upper_values_plot + lower_values_plot[::-1],
                fill='toself',
                fillcolor='rgba(167, 139, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=True,
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_dates_plot,
                y=forecast_values_plot,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#a78bfa', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond', line=dict(width=2, color='white')),
            ))

        # Vertical line at forecast start
        try:
            fig.add_vline(x=base_date, line_dash="dot", line_color="#ef4444", line_width=2)
        except:
            pass  # Skip vline if date format incompatible

        fig.update_layout(
            title=dict(text=f"Forecast vs Actual - {selected_model_name}", x=0.5),
            xaxis_title="Date",
            yaxis_title="Patient Count",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

        # Convert category_forecasts_by_horizon from {horizon: {cat: val}} to {cat: [vals]}
        # This format is easier for downstream pages (Staff Planner, Supply Planner)
        category_forecasts_for_session = None
        if category_forecasts_by_horizon:
            category_forecasts_for_session = {}
            # Get all categories from the first horizon
            first_horizon = list(category_forecasts_by_horizon.keys())[0]
            all_categories = list(category_forecasts_by_horizon[first_horizon].keys())

            for cat in all_categories:
                category_forecasts_for_session[cat] = []
                for h in sorted(category_forecasts_by_horizon.keys()):
                    val = category_forecasts_by_horizon[h].get(cat, 0)
                    category_forecasts_for_session[cat].append(val)

        # Save to session state for staff scheduling, inventory, and command center
        save_forecast_to_session(
            forecast_data,
            selected_model_name,
            horizons=horizons,
            forecast_days=forecast_days,
            category_forecasts=category_forecasts_for_session
        )

# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================
with tab_comparison:
    st.markdown("### 📈 Model Comparison")

    if len(available_models) < 2:
        st.info("Train at least 2 models to compare performance.")
    else:
        # Collect metrics from all models
        comparison_data = []

        for model_info in available_models:
            data = model_info["data"]
            metrics_df = data.get("metrics_df")

            if metrics_df is not None and not metrics_df.empty:
                # Get average metrics across horizons
                df = metrics_df.copy()

                # Extract numeric metrics
                mae_col = "Test_MAE" if "Test_MAE" in df.columns else "MAE"
                rmse_col = "Test_RMSE" if "Test_RMSE" in df.columns else "RMSE"
                mape_col = "Test_MAPE" if "Test_MAPE" in df.columns else ("MAPE_%" if "MAPE_%" in df.columns else "MAPE")
                acc_col = "Test_Acc" if "Test_Acc" in df.columns else ("Accuracy_%" if "Accuracy_%" in df.columns else "Accuracy")

                avg_mae = _safe_float(df[mae_col].mean()) if mae_col in df.columns else np.nan
                avg_rmse = _safe_float(df[rmse_col].mean()) if rmse_col in df.columns else np.nan
                avg_mape = _safe_float(df[mape_col].mean()) if mape_col in df.columns else np.nan
                avg_acc = _safe_float(df[acc_col].mean()) if acc_col in df.columns else np.nan

                # Calculate accuracy from MAPE if not available
                if np.isnan(avg_acc) and not np.isnan(avg_mape):
                    avg_acc = 100 - avg_mape

                comparison_data.append({
                    "Model": model_info["name"],
                    "Type": model_info["type"],
                    "MAE": avg_mae,
                    "RMSE": avg_rmse,
                    "MAPE %": avg_mape,
                    "Accuracy %": avg_acc,
                })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)

            # Sort by accuracy (descending)
            if "Accuracy %" in comp_df.columns:
                comp_df = comp_df.sort_values("Accuracy %", ascending=False)

            # Display table
            st.dataframe(
                comp_df.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "MAPE %": "{:.2f}",
                    "Accuracy %": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )

            # Bar chart comparison
            st.markdown("#### Accuracy Comparison")

            fig = go.Figure()

            colors = [SUCCESS_COLOR if i == 0 else PRIMARY_COLOR for i in range(len(comp_df))]

            fig.add_trace(go.Bar(
                x=comp_df["Model"],
                y=comp_df["Accuracy %"],
                marker_color=colors,
                text=[f"{v:.1f}%" for v in comp_df["Accuracy %"]],
                textposition='outside',
            ))

            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy (%)",
                template="plotly_dark",
                height=400,
                yaxis=dict(range=[max(0, comp_df["Accuracy %"].min() - 10), 100]),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Best model recommendation
            best_model = comp_df.iloc[0]
            st.success(f"""
            🏆 **Best Model: {best_model['Model']}**
            - Accuracy: {best_model['Accuracy %']:.2f}%
            - MAE: {best_model['MAE']:.2f}
            - Type: {best_model['Type']}
            """)

# =============================================================================
# TAB 4: EXPORT
# =============================================================================
with tab_export:
    st.markdown("### 📥 Export Forecasts")

    # Select model to export
    model_names = [m["name"] for m in available_models]
    export_model = st.selectbox(
        "Select Model to Export:",
        options=model_names,
        key="export_model_select"
    )

    # Get model data
    export_model_info = next(m for m in available_models if m["name"] == export_model)
    export_data = export_model_info["data"]

    F = export_data["F"]
    L = export_data["L"]
    U = export_data["U"]
    test_eval = export_data["test_eval"]
    horizons = export_data["horizons"]

    # Create export DataFrame
    export_df = pd.DataFrame(index=test_eval.index)

    # Add forecasts for each horizon
    for h_idx, h in enumerate(horizons):
        export_df[f"Forecast_h{h}"] = F[:, h_idx]
        if L is not None:
            export_df[f"Lower_h{h}"] = L[:, h_idx]
        if U is not None:
            export_df[f"Upper_h{h}"] = U[:, h_idx]

    # Add actuals
    for col in test_eval.columns:
        export_df[f"Actual_{col}"] = test_eval[col]

    st.markdown("#### Preview")
    st.dataframe(export_df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download CSV
        csv = export_df.to_csv()
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"forecast_{export_model}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Send to Staff Scheduling
        if st.button("🔗 Send to Staff Scheduling", use_container_width=True, type="primary"):
            save_forecast_to_session(export_data, export_model)
            st.success("✅ Forecast sent to Staff Scheduling!")
            st.info("Go to **11_Staff_Scheduling_Optimization.py** to use the forecast.")

    with col3:
        # View JSON
        with st.expander("View JSON"):
            st.json({
                "model": export_model,
                "horizons": horizons,
                "num_samples": len(test_eval),
                "date_range": f"{test_eval.index.min()} to {test_eval.index.max()}" if hasattr(test_eval.index, 'min') else "N/A",
            })

# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(9)  # Patient Forecast is page index 9

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Patient Forecast</strong> — View test set forecasts from trained models<br>
    Uses predictions generated during model training in Benchmarks & Modeling Hub
</div>
""", unsafe_allow_html=True)
