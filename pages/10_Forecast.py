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
        model_key: Session state key (e.g., "ml_mh_results_XGBoost")
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

        # Create test_eval DataFrame
        if hasattr(y_test_first, 'index'):
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

    return models


# =============================================================================
# UNCERTAINTY WIDTH PIPELINE
# =============================================================================

@dataclass
class UncertaintyMetrics:
    """Container for uncertainty width metrics."""
    piw: np.ndarray          # Prediction Interval Width (T x H)
    rpiw: np.ndarray         # Relative PIW as percentage (T x H)
    piw_avg: np.ndarray      # Average PIW per horizon (H,)
    rpiw_avg: np.ndarray     # Average RPIW per horizon (H,)
    overall_rpiw: float      # Overall average RPIW
    uncertainty_level: str   # LOW, MEDIUM, HIGH
    recommended_method: str  # Deterministic MILP, Robust Optimization, Two-Stage Stochastic
    rationale: str           # Explanation for recommendation


def calculate_uncertainty_width(F: np.ndarray, L: np.ndarray, U: np.ndarray) -> Optional[UncertaintyMetrics]:
    """
    Calculate Prediction Interval Width (PIW) and Relative PIW (RPIW) metrics.

    Formulas:
        PIW_t = U_t - L_t                           (PIW-1)
        RPIW_t = (U_t - L_t) / D̂_t × 100%          (PIW-2)
        RPIW_avg = (1/T) × Σ_t RPIW_t              (PIW-3)

    Args:
        F: Forecast matrix (T x H) - point forecasts
        L: Lower bound matrix (T x H) - lower confidence interval
        U: Upper bound matrix (T x H) - upper confidence interval

    Returns:
        UncertaintyMetrics object or None if calculation fails
    """
    if F is None or L is None or U is None:
        return None

    # Check if we have valid CI data
    if np.all(np.isnan(L)) or np.all(np.isnan(U)):
        return None

    try:
        T, H = F.shape

        # PIW: Prediction Interval Width = Upper - Lower (PIW-1)
        piw = U - L  # Shape: (T, H)

        # Handle cases where forecast is zero or very small
        F_safe = np.where(np.abs(F) < 1e-6, np.nan, F)

        # RPIW: Relative Prediction Interval Width (PIW-2)
        rpiw = (piw / np.abs(F_safe)) * 100  # As percentage

        # Average PIW per horizon
        piw_avg = np.nanmean(piw, axis=0)  # Shape: (H,)

        # Average RPIW per horizon (PIW-3)
        rpiw_avg = np.nanmean(rpiw, axis=0)  # Shape: (H,)

        # Overall average RPIW across all horizons
        overall_rpiw = np.nanmean(rpiw)

        # Determine uncertainty level and recommendation
        if np.isnan(overall_rpiw):
            uncertainty_level = "UNKNOWN"
            recommended_method = "Manual Assessment Required"
            rationale = "Unable to calculate RPIW - check confidence interval data"
        elif overall_rpiw < 15:
            uncertainty_level = "LOW"
            recommended_method = "Deterministic MILP"
            rationale = "Point forecast is reliable; extra complexity not justified"
        elif overall_rpiw <= 40:
            uncertainty_level = "MEDIUM"
            recommended_method = "Robust Optimization"
            rationale = "Need protection but scenarios overkill; budget Γ adjustable"
        else:
            uncertainty_level = "HIGH"
            recommended_method = "Two-Stage Stochastic"
            rationale = "Need explicit recourse actions; scenarios capture distribution"

        return UncertaintyMetrics(
            piw=piw,
            rpiw=rpiw,
            piw_avg=piw_avg,
            rpiw_avg=rpiw_avg,
            overall_rpiw=overall_rpiw,
            uncertainty_level=uncertainty_level,
            recommended_method=recommended_method,
            rationale=rationale,
        )
    except Exception as e:
        st.warning(f"Error calculating uncertainty width: {e}")
        return None


def get_uncertainty_for_all_models(models: List[Dict[str, Any]]) -> Dict[str, UncertaintyMetrics]:
    """
    Calculate uncertainty metrics for all available models.

    Args:
        models: List of model info dicts from get_available_forecast_models()

    Returns:
        Dict mapping model name to UncertaintyMetrics
    """
    results = {}

    for model_info in models:
        model_name = model_info["name"]
        data = model_info.get("data", {})

        F = data.get("F")
        L = data.get("L")
        U = data.get("U")

        metrics = calculate_uncertainty_width(F, L, U)
        if metrics:
            results[model_name] = metrics

    return results


def create_uncertainty_summary_df(uncertainty_results: Dict[str, UncertaintyMetrics]) -> pd.DataFrame:
    """
    Create a summary DataFrame of uncertainty metrics for all models.
    """
    rows = []
    for model_name, metrics in uncertainty_results.items():
        rows.append({
            "Model": model_name,
            "Avg RPIW (%)": round(metrics.overall_rpiw, 2) if not np.isnan(metrics.overall_rpiw) else "N/A",
            "Uncertainty Level": metrics.uncertainty_level,
            "Recommended Method": metrics.recommended_method,
            "Rationale": metrics.rationale,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("Avg RPIW (%)", ascending=True)


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


def save_forecast_to_session(forecast_data: Dict, model_name: str):
    """Save forecast data to session state for use by Staff Scheduling."""
    if not forecast_data:
        return

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Get dates from test_eval index
    test_eval = forecast_data.get("test_eval")
    dates = []
    if test_eval is not None and hasattr(test_eval.index, '__iter__'):
        try:
            dates = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in test_eval.index]
        except:
            pass

    # Get forecast values (first horizon column)
    F = forecast_data.get("F")
    forecast_values = list(F[:, 0]) if F is not None and F.shape[1] > 0 else []

    st.session_state["forecast_hub_results"] = {
        "model_name": model_name,
        "forecast_data": forecast_data,
    }

    st.session_state["forecast_hub_demand"] = {
        "model": model_name,
        "forecast": forecast_values,
        "dates": dates,
    }


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🔮</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Forecast Hub</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        View forecasts from trained models<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Uses test set predictions from Benchmarks & Modeling Hub</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
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

    # Show session state keys for debugging
    with st.expander("🔧 Debug: Session State Keys", expanded=False):
        model_keys = [k for k in st.session_state.keys() if 'result' in k.lower() or 'model' in k.lower()]
        st.write("Model-related keys:", model_keys)

    st.stop()

# Create tabs
tab_overview, tab_forecast, tab_comparison, tab_uncertainty, tab_export = st.tabs([
    "📊 Overview",
    "🔮 Forecast View",
    "📈 Model Comparison",
    "📏 Uncertainty Analysis",
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

        with st.expander(f"**{model_info['name']}** ({model_info['type']})", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Test Samples", F.shape[0] if F is not None else 0)
            with col2:
                st.metric("Horizons", len(horizons))
            with col3:
                st.metric("Max Horizon", max(horizons) if horizons else 0)
            with col4:
                st.metric("Date Range", date_range)

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

            # --- FORECAST HEADER ---
            st.markdown(f"""
            <div class='forecast-card' style='padding: 1.5rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='font-size: 1.25rem; font-weight: 700; color: #e2e8f0;'>
                            📅 {forecast_days}-Day Forecast
                        </div>
                        <div style='font-size: 0.875rem; color: #94a3b8; margin-top: 0.25rem;'>
                            Model: <span style='color: #22d3ee;'>{selected_model_name}</span> •
                            From: <span style='color: #a78bfa;'>{base_date_str}</span>
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
                forecast_dt = None
                forecast_dt_day = f"Day {horizon_num}"
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
                    primary_class = " forecast-mini-card-primary" if is_primary else ""

                    actual_display = ""
                    if actual_val is not None:
                        error = abs(forecast_val - actual_val)
                        actual_display = f"""
                        <div style='margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(100, 116, 139, 0.2);'>
                            <div style='font-size: 0.7rem; color: #64748b;'>Actual</div>
                            <div style='font-size: 1rem; font-weight: 700; color: #22d3ee;'>{int(actual_val)}</div>
                            <div style='font-size: 0.65rem; color: {"#10b981" if error < 5 else "#f59e0b"};'>Error: ±{error:.1f}</div>
                        </div>
                        """

                    st.markdown(f"""
                    <div class='forecast-mini-card{primary_class}' style='margin-bottom: 0.5rem;'>
                        <div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;'>{forecast_dt_day}</div>
                        <div style='font-size: 0.7rem; color: #64748b; background: rgba(100, 116, 139, 0.15); padding: 0.2rem 0.5rem; border-radius: 4px; display: inline-block; margin: 0.25rem 0;'>{forecast_dt_date}</div>
                        <div class='forecast-number' style='margin: 0.5rem 0;'>{int(forecast_val)}</div>
                        <div style='font-size: 0.6rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.5px;'>Predicted</div>
                        <div style='display: inline-block; padding: 0.25rem 0.5rem; background: {acc_bg}; border: 1px solid {acc_color}40; border-radius: 6px; font-size: 0.7rem; font-weight: 600; color: {acc_color}; margin: 0.5rem 0;'>✓ {accuracy_text}</div>
                        <div style='font-size: 0.65rem; color: #64748b; background: rgba(30, 41, 59, 0.5); padding: 0.25rem 0.5rem; border-radius: 4px;'>{int(lower_val)} - {int(upper_val)}</div>
                        {actual_display}
                    </div>
                    """, unsafe_allow_html=True)

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

        # Save to session state for staff scheduling
        save_forecast_to_session(forecast_data, selected_model_name)

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
# TAB 4: UNCERTAINTY ANALYSIS
# =============================================================================
with tab_uncertainty:
    st.markdown("### 📏 Uncertainty Width Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
                padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;">
        <p style="margin: 0; color: #94a3b8;">
            <strong>Key Insight:</strong> The width of your prediction interval determines which optimization approach to use for staff scheduling.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate uncertainty for all models
    uncertainty_results = get_uncertainty_for_all_models(available_models)

    if not uncertainty_results:
        st.warning("No models with valid confidence intervals found. Uncertainty analysis requires Upper (U) and Lower (L) bounds.")
        st.info("Train models with confidence intervals enabled (ARIMA/SARIMAX) to see uncertainty analysis.")
    else:
        # Summary table
        st.markdown("#### Model Uncertainty Summary")
        summary_df = create_uncertainty_summary_df(uncertainty_results)

        # Color code the uncertainty levels
        def style_uncertainty(val):
            if val == "LOW":
                return "background-color: rgba(34, 197, 94, 0.2); color: #22c55e;"
            elif val == "MEDIUM":
                return "background-color: rgba(251, 191, 36, 0.2); color: #fbbf24;"
            elif val == "HIGH":
                return "background-color: rgba(239, 68, 68, 0.2); color: #ef4444;"
            return ""

        st.dataframe(
            summary_df.style.applymap(style_uncertainty, subset=["Uncertainty Level"]),
            use_container_width=True,
            hide_index=True
        )

        # Decision Framework Reference
        st.markdown("#### Decision Framework: Uncertainty Width → Optimization Method")

        decision_cols = st.columns(3)

        with decision_cols[0]:
            st.markdown("""
            <div class="fq-kpi-card" style="--accent-color: #22c55e; --accent-color-end: #10b981;">
                <div class="fq-kpi-icon">🟢</div>
                <div class="fq-kpi-label">LOW UNCERTAINTY</div>
                <div class="fq-kpi-value" style="font-size: 1.2rem;">RPIW &lt; 15%</div>
                <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px;">
                    <div style="font-weight: 600; color: #22c55e;">Deterministic MILP</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                        ±5 patients<br>Point forecast is reliable
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with decision_cols[1]:
            st.markdown("""
            <div class="fq-kpi-card" style="--accent-color: #fbbf24; --accent-color-end: #f59e0b;">
                <div class="fq-kpi-icon">🟡</div>
                <div class="fq-kpi-label">MEDIUM UNCERTAINTY</div>
                <div class="fq-kpi-value" style="font-size: 1.2rem;">RPIW 15-40%</div>
                <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(251, 191, 36, 0.1); border-radius: 8px;">
                    <div style="font-weight: 600; color: #fbbf24;">Robust Optimization</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                        ±10-20 patients<br>Budget Γ adjustable
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with decision_cols[2]:
            st.markdown("""
            <div class="fq-kpi-card" style="--accent-color: #ef4444; --accent-color-end: #dc2626;">
                <div class="fq-kpi-icon">🔴</div>
                <div class="fq-kpi-label">HIGH UNCERTAINTY</div>
                <div class="fq-kpi-value" style="font-size: 1.2rem;">RPIW &gt; 40%</div>
                <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                    <div style="font-weight: 600; color: #ef4444;">Two-Stage Stochastic</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                        ±30+ patients<br>Explicit recourse actions
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed per-model analysis
        st.markdown("#### Detailed Model Analysis")

        model_select = st.selectbox(
            "Select Model for Detailed Analysis:",
            options=list(uncertainty_results.keys()),
            key="uncertainty_model_select"
        )

        if model_select:
            metrics = uncertainty_results[model_select]
            model_data = next((m["data"] for m in available_models if m["name"] == model_select), None)

            if model_data:
                horizons = model_data.get("horizons", [])

                # Display overall metrics
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    level_color = {"LOW": "#22c55e", "MEDIUM": "#fbbf24", "HIGH": "#ef4444"}.get(metrics.uncertainty_level, "#94a3b8")
                    st.markdown(f"""
                    <div class="fq-kpi-card" style="--accent-color: {level_color}; --accent-color-end: {level_color};">
                        <div class="fq-kpi-label">OVERALL RPIW</div>
                        <div class="fq-kpi-value">{metrics.overall_rpiw:.1f}%</div>
                        <div class="fq-kpi-sublabel">Relative Prediction Interval Width</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[1]:
                    st.markdown(f"""
                    <div class="fq-kpi-card" style="--accent-color: {level_color}; --accent-color-end: {level_color};">
                        <div class="fq-kpi-label">UNCERTAINTY LEVEL</div>
                        <div class="fq-kpi-value" style="color: {level_color};">{metrics.uncertainty_level}</div>
                        <div class="fq-kpi-sublabel">{metrics.rationale[:40]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[2]:
                    avg_piw = np.nanmean(metrics.piw_avg)
                    st.markdown(f"""
                    <div class="fq-kpi-card">
                        <div class="fq-kpi-label">AVG INTERVAL WIDTH</div>
                        <div class="fq-kpi-value">{avg_piw:.1f}</div>
                        <div class="fq-kpi-sublabel">Patients (absolute)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[3]:
                    st.markdown(f"""
                    <div class="fq-kpi-card" style="--accent-color: #8b5cf6; --accent-color-end: #a855f7;">
                        <div class="fq-kpi-label">RECOMMENDED</div>
                        <div class="fq-kpi-value" style="font-size: 1rem;">{metrics.recommended_method}</div>
                        <div class="fq-kpi-sublabel">Optimization Method</div>
                    </div>
                    """, unsafe_allow_html=True)

                # RPIW per horizon chart
                st.markdown("##### RPIW by Forecast Horizon")

                if len(horizons) > 0 and len(metrics.rpiw_avg) > 0:
                    rpiw_df = pd.DataFrame({
                        "Horizon": [f"h={h}" for h in horizons[:len(metrics.rpiw_avg)]],
                        "RPIW (%)": metrics.rpiw_avg[:len(horizons)],
                        "PIW (abs)": metrics.piw_avg[:len(horizons)]
                    })

                    fig_rpiw = go.Figure()

                    # Add bar for RPIW
                    colors = []
                    for rpiw in rpiw_df["RPIW (%)"]:
                        if rpiw < 15:
                            colors.append("#22c55e")
                        elif rpiw <= 40:
                            colors.append("#fbbf24")
                        else:
                            colors.append("#ef4444")

                    fig_rpiw.add_trace(go.Bar(
                        x=rpiw_df["Horizon"],
                        y=rpiw_df["RPIW (%)"],
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in rpiw_df["RPIW (%)"]],
                        textposition="auto",
                        name="RPIW (%)"
                    ))

                    # Add threshold lines
                    fig_rpiw.add_hline(y=15, line_dash="dash", line_color="#22c55e",
                                       annotation_text="Low threshold (15%)", annotation_position="right")
                    fig_rpiw.add_hline(y=40, line_dash="dash", line_color="#fbbf24",
                                       annotation_text="Medium threshold (40%)", annotation_position="right")

                    fig_rpiw.update_layout(
                        title=f"Relative Prediction Interval Width - {model_select}",
                        xaxis_title="Forecast Horizon",
                        yaxis_title="RPIW (%)",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                    )

                    st.plotly_chart(fig_rpiw, use_container_width=True)

                    # PIW per horizon table
                    st.markdown("##### Metrics per Horizon")
                    st.dataframe(rpiw_df.round(2), use_container_width=True, hide_index=True)

        # Save recommendation to session state
        if uncertainty_results:
            # Find best model (lowest RPIW with valid data)
            best_model = min(uncertainty_results.items(), key=lambda x: x[1].overall_rpiw if not np.isnan(x[1].overall_rpiw) else float('inf'))
            st.session_state["recommended_optimization_method"] = best_model[1].recommended_method
            st.session_state["forecast_uncertainty_level"] = best_model[1].uncertainty_level
            st.session_state["forecast_rpiw"] = best_model[1].overall_rpiw

            st.markdown("---")
            st.info(f"""
            **Recommendation saved to session:**
            - Best Model: {best_model[0]}
            - Uncertainty Level: {best_model[1].uncertainty_level}
            - RPIW: {best_model[1].overall_rpiw:.1f}%
            - Recommended Method: {best_model[1].recommended_method}

            This will be used by the Staff Scheduling module to select the appropriate optimization approach.
            """)

# =============================================================================
# TAB 5: EXPORT
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
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Forecast Hub</strong> — View test set forecasts from trained models<br>
    Uses predictions generated during model training in Benchmarks & Modeling Hub
</div>
""", unsafe_allow_html=True)
