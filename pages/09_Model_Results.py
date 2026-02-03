# =============================================================================
# 09_Results.py — Comprehensive Results & Model Comparison Hub
# Thesis-Quality Analysis for Multi-Horizon Forecasting Models
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import base64

# =============================================================================
# SHAP AND LIME IMPORTS (Model Explainability)
# =============================================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# =============================================================================
# PhD-LEVEL STATISTICAL ANALYSIS IMPORTS
# =============================================================================
try:
    from app_core.analytics.statistical_tests import (
        diebold_mariano_test,
        diebold_mariano_matrix,
        compute_naive_baseline,
        compute_all_baselines,
        compute_skill_score,
        bootstrap_confidence_interval,
        evaluate_model_vs_baselines,
        format_p_value_display,
        get_significance_badge,
        test_forecast_bias,
    )
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False

try:
    from app_core.analytics.residual_diagnostics import (
        compute_full_diagnostics,
        compute_acf,
        compute_pacf,
        ljung_box_test,
        shapiro_wilk_test,
        breusch_pagan_test,
        get_diagnostic_badge,
    )
    RESIDUAL_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    RESIDUAL_DIAGNOSTICS_AVAILABLE = False

try:
    from app_core.ui.diagnostic_plots import (
        create_diagnostic_panel,
        create_actual_vs_predicted_plot,
        create_residuals_vs_predicted_plot,
        create_residual_histogram,
        create_qq_plot,
        create_acf_plot,
        create_skill_score_gauge,
        create_dm_heatmap,
        create_baseline_comparison_chart,
        create_training_loss_plot,
    )
    DIAGNOSTIC_PLOTS_AVAILABLE = True
except ImportError:
    DIAGNOSTIC_PLOTS_AVAILABLE = False

# Experiment tracking service
from app_core.data.experiment_results_service import (
    ExperimentResultsService,
    ExperimentRecord,
    get_experiment_service,
    get_model_category,
)

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Model Results - HealthForecast AI",
    page_icon="📊",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# Auto-load saved results from cloud storage
auto_load_if_available("Model Results")

# Cloud storage panel in sidebar
render_results_storage_panel(page_key="Model Results")

# =============================================================================
# CONSTANTS & STYLING
# =============================================================================

METRIC_FORMATTERS = {
    "MAE": "{:.4f}",
    "RMSE": "{:.4f}",
    "MAPE_%": "{:.2f}",
    "MAPE": "{:.2f}",
    "Accuracy_%": "{:.2f}",
    "Accuracy": "{:.2f}",
    "R2": "{:.4f}",
    "R²": "{:.4f}",
    "Runtime_s": "{:.2f}",
    "AIC": "{:.1f}",
    "BIC": "{:.1f}",
}

# Model category colors
MODEL_COLORS = {
    "ARIMA": "#3B82F6",      # Blue
    "SARIMAX": "#8B5CF6",    # Purple
    "XGBoost": "#22C55E",    # Green
    "LSTM": "#F59E0B",       # Amber
    "ANN": "#EF4444",        # Red
    "Ensemble": "#06B6D4",   # Cyan
    "Stacking": "#EC4899",   # Pink
    "Hybrid": "#84CC16",     # Lime
    "Optimized": "#14B8A6",  # Teal
    "Other": "#6B7280",      # Gray
}

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR RESULTS
   ======================================== */

@keyframes float-orb {
    0%, 100% {
        transform: translate(0, 0) scale(1);
        opacity: 0.25;
    }
    50% {
        transform: translate(30px, -30px) scale(1.05);
        opacity: 0.35;
    }
}

.fluorescent-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}

.orb-1 {
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}

.orb-2 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}

@keyframes sparkle {
    0%, 100% {
        opacity: 0;
        transform: scale(0);
    }
    50% {
        opacity: 0.6;
        transform: scale(1);
    }
}

.sparkle {
    position: fixed;
    width: 3px;
    height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

.sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
.sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
.sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

/* Best Model Highlight Card */
.best-model-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
    border: 2px solid #22C55E;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 40px rgba(34, 197, 94, 0.3);
}

/* Statistical Significance Badge */
.sig-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.sig-significant {
    background: rgba(34, 197, 94, 0.2);
    color: #22C55E;
    border: 1px solid #22C55E;
}

.sig-not-significant {
    background: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid #EF4444;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Ranking Table */
.rank-gold { color: #FFD700; font-weight: bold; }
.rank-silver { color: #C0C0C0; font-weight: bold; }
.rank-bronze { color: #CD7F32; font-weight: bold; }

@media (max-width: 768px) {
    .fluorescent-orb {
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }
    .sparkle {
        display: none;
    }
}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_float(val, default=np.nan) -> float:
    """Safely convert value to float."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _get_model_category(model_name: str) -> str:
    """Determine model category from name."""
    name_lower = model_name.lower()
    # Check hybrid/ensemble/stacking first (before individual model types)
    if "hybrid" in name_lower:
        return "Hybrid"
    elif "ensemble" in name_lower:
        return "Ensemble"
    elif "stack" in name_lower:
        return "Stacking"
    elif "arima" in name_lower and "sarimax" not in name_lower:
        return "ARIMA"
    elif "sarimax" in name_lower:
        return "SARIMAX"
    elif "xgboost" in name_lower or "xgb" in name_lower:
        return "XGBoost"
    elif "lstm" in name_lower:
        return "LSTM"
    elif "ann" in name_lower or "neural" in name_lower or "mlp" in name_lower:
        return "ANN"
    elif "optimized" in name_lower:
        return "Optimized"
    return "Other"


def _get_model_color(model_name: str) -> str:
    """Get color for a model based on its category."""
    category = _get_model_category(model_name)
    return MODEL_COLORS.get(category, "#6B7280")


# =============================================================================
# DATA AGGREGATION - Collect all model results from session state
# =============================================================================

def aggregate_all_model_results() -> pd.DataFrame:
    """
    Aggregate results from all trained models across Benchmarks and Modeling Hub.

    Returns:
        DataFrame with unified metrics for all models
    """
    all_results = []

    # 1. Main comparison table (from Benchmarks)
    comp_df = st.session_state.get("model_comparison")
    if comp_df is not None and isinstance(comp_df, pd.DataFrame) and not comp_df.empty:
        for _, row in comp_df.iterrows():
            all_results.append({
                "Model": row.get("Model", "Unknown"),
                "Category": _get_model_category(str(row.get("Model", ""))),
                "Source": "Benchmarks",
                "MAE": _safe_float(row.get("MAE")),
                "RMSE": _safe_float(row.get("RMSE")),
                "MAPE_%": _safe_float(row.get("MAPE_%")),
                "Accuracy_%": _safe_float(row.get("Accuracy_%")),
                "Runtime_s": _safe_float(row.get("Runtime_s")),
                "Timestamp": row.get("Timestamp", ""),
                "Parameters": row.get("ModelParams", ""),
            })

    # 2. ARIMA multi-horizon results
    arima_results = st.session_state.get("arima_mh_results")
    if arima_results:
        results_df = arima_results.get("results_df")
        if results_df is not None and not results_df.empty:
            # Aggregate across horizons
            avg_metrics = _aggregate_horizon_metrics(results_df, "ARIMA")
            if avg_metrics:
                all_results.append(avg_metrics)

    # 3. SARIMAX results
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results:
        results_df = sarimax_results.get("results_df")
        if results_df is not None and not results_df.empty:
            avg_metrics = _aggregate_horizon_metrics(results_df, "SARIMAX")
            if avg_metrics:
                all_results.append(avg_metrics)

    # 4. ML models (XGBoost, LSTM, ANN)
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        # Check both lowercase and titlecase keys (08_Train_Models.py uses lowercase)
        ml_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")
        if ml_results is None:
            ml_results = st.session_state.get(f"ml_mh_results_{model_name}")
        if ml_results:
            metrics = _extract_ml_metrics(ml_results, model_name)
            if metrics:
                all_results.append(metrics)

    # 5. Single ML model results (if only one model trained)
    ml_mh = st.session_state.get("ml_mh_results")
    # Check if any model-specific results exist (both lowercase and titlecase)
    has_specific_results = any(
        st.session_state.get(f"ml_mh_results_{m.lower()}") or st.session_state.get(f"ml_mh_results_{m}")
        for m in ["XGBoost", "LSTM", "ANN"]
    )
    if ml_mh and not has_specific_results:
        # Determine model type from session state config
        model_type = st.session_state.get("ml_choice", "ML Model")
        metrics = _extract_ml_metrics(ml_mh, model_type)
        if metrics:
            all_results.append(metrics)

    # 6. Ensemble results
    ensemble_results = st.session_state.get("ensemble_results")
    if ensemble_results:
        metrics = _extract_ensemble_metrics(ensemble_results)
        if metrics:
            all_results.append(metrics)

    # 7. Stacking results
    stacking_results = st.session_state.get("stacking_results")
    if stacking_results:
        metrics = _extract_stacking_metrics(stacking_results)
        if metrics:
            all_results.append(metrics)

    # 8. Hybrid LSTM-SARIMAX results
    hybrid_results = st.session_state.get("lstm_sarimax_results")
    if hybrid_results:
        metrics = _extract_hybrid_metrics(hybrid_results)
        if metrics:
            all_results.append(metrics)

    # 9. Hybrid LSTM-XGBoost results
    lstm_xgb_results = st.session_state.get("lstm_xgb_results")
    if lstm_xgb_results:
        metrics = _extract_lstm_xgb_metrics(lstm_xgb_results)
        if metrics:
            all_results.append(metrics)

    # 10. Hybrid LSTM-ANN results
    lstm_ann_results = st.session_state.get("lstm_ann_results")
    if lstm_ann_results:
        metrics = _extract_lstm_ann_metrics(lstm_ann_results)
        if metrics:
            all_results.append(metrics)

    # 11. Optimized model results
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        # Check both titlecase and lowercase keys
        opt_results = st.session_state.get(f"opt_results_{model_type}")
        if opt_results is None:
            opt_results = st.session_state.get(f"opt_results_{model_type.lower()}")
        if opt_results:
            metrics = _extract_optimized_metrics(opt_results, model_type)
            if metrics:
                all_results.append(metrics)

    # 12. Generic Hybrid Pipeline results (from diagnostics)
    hybrid_pipeline = st.session_state.get("hybrid_pipeline_results")
    if hybrid_pipeline and isinstance(hybrid_pipeline, dict):
        for hybrid_name, hybrid_data in hybrid_pipeline.items():
            if hybrid_data and isinstance(hybrid_data, dict) and hybrid_data.get("success"):
                metrics = _extract_generic_hybrid_metrics(hybrid_data, hybrid_name)
                if metrics:
                    all_results.append(metrics)

    # 13. Decomposition ensemble results
    decomp_results = st.session_state.get("decomposition_results")
    if decomp_results:
        metrics = _extract_decomposition_metrics(decomp_results)
        if metrics:
            all_results.append(metrics)

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        # Remove duplicates based on model name (keep most recent)
        df = df.drop_duplicates(subset=["Model"], keep="last")
        return df

    return pd.DataFrame()


def _aggregate_horizon_metrics(results_df: pd.DataFrame, model_name: str) -> Optional[Dict]:
    """Aggregate metrics across horizons for statistical models."""
    try:
        # Handle different column naming conventions
        mae_col = "Test_MAE" if "Test_MAE" in results_df.columns else "MAE"
        rmse_col = "Test_RMSE" if "Test_RMSE" in results_df.columns else "RMSE"
        mape_col = "Test_MAPE" if "Test_MAPE" in results_df.columns else "MAPE_%"
        acc_col = "Test_Acc" if "Test_Acc" in results_df.columns else "Accuracy_%"

        return {
            "Model": f"{model_name} (Avg)",
            "Category": model_name,
            "Source": "Benchmarks",
            "MAE": _safe_float(results_df[mae_col].mean()) if mae_col in results_df.columns else np.nan,
            "RMSE": _safe_float(results_df[rmse_col].mean()) if rmse_col in results_df.columns else np.nan,
            "MAPE_%": _safe_float(results_df[mape_col].mean()) if mape_col in results_df.columns else np.nan,
            "Accuracy_%": _safe_float(results_df[acc_col].mean()) if acc_col in results_df.columns else np.nan,
            "Runtime_s": np.nan,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Horizons: {len(results_df)}",
        }
    except Exception:
        return None


def _extract_ml_metrics(ml_results: Dict, model_name: str) -> Optional[Dict]:
    """Extract aggregated metrics from ML model results.

    ML results store metrics in results_df DataFrame with columns:
    - Test_MAE, Test_RMSE, Test_MAPE, Test_Acc (for test metrics)
    - Train_MAE, Train_RMSE, Train_MAPE, Train_Acc (for training metrics)
    """
    try:
        results_df = ml_results.get("results_df")
        successful = ml_results.get("successful", [])

        # Check if we have valid results DataFrame
        if results_df is None or results_df.empty:
            return None

        # Extract test metrics from results_df (average across all horizons)
        mae_avg = results_df["Test_MAE"].mean() if "Test_MAE" in results_df.columns else np.nan
        rmse_avg = results_df["Test_RMSE"].mean() if "Test_RMSE" in results_df.columns else np.nan
        mape_avg = results_df["Test_MAPE"].mean() if "Test_MAPE" in results_df.columns else np.nan
        acc_avg = results_df["Test_Acc"].mean() if "Test_Acc" in results_df.columns else np.nan

        return {
            "Model": f"{model_name} (Avg)",
            "Category": _get_model_category(model_name),
            "Source": "Modeling Hub",
            "MAE": _safe_float(mae_avg),
            "RMSE": _safe_float(rmse_avg),
            "MAPE_%": _safe_float(mape_avg),
            "Accuracy_%": _safe_float(acc_avg),
            "Runtime_s": _safe_float(ml_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Horizons: {len(successful) if successful else len(results_df)}",
        }
    except Exception as e:
        # Log error for debugging
        print(f"Error extracting ML metrics for {model_name}: {e}")
        return None


def _extract_ensemble_metrics(ensemble_results: Dict) -> Optional[Dict]:
    """Extract metrics from ensemble results."""
    try:
        metrics = ensemble_results.get("metrics", {})
        # Metrics stored with uppercase keys (MAE, RMSE, MAPE, Accuracy)
        return {
            "Model": "Ensemble (Weighted)",
            "Category": "Ensemble",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(ensemble_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Models: {', '.join(ensemble_results.get('models', []))}",
        }
    except Exception:
        return None


def _extract_stacking_metrics(stacking_results: Dict) -> Optional[Dict]:
    """Extract metrics from stacking results."""
    try:
        metrics = stacking_results.get("metrics", {})
        # Metrics stored with uppercase keys (MAE, RMSE, MAPE, Accuracy)
        return {
            "Model": "Stacking Meta-Learner",
            "Category": "Stacking",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(stacking_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Meta: {stacking_results.get('meta_learner', 'Ridge')}",
        }
    except Exception:
        return None


def _extract_hybrid_metrics(hybrid_results: Dict) -> Optional[Dict]:
    """Extract metrics from hybrid LSTM-SARIMAX results."""
    try:
        metrics = hybrid_results.get("metrics", {})
        # Metrics stored with uppercase keys (MAE, RMSE, MAPE, Accuracy)
        return {
            "Model": "Hybrid LSTM-SARIMAX",
            "Category": "Hybrid",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(hybrid_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": "LSTM + SARIMAX residual correction",
        }
    except Exception:
        return None


def _extract_lstm_xgb_metrics(hybrid_results: Dict) -> Optional[Dict]:
    """Extract metrics from hybrid LSTM-XGBoost results."""
    try:
        # Results structure: {"artifacts": artifacts, "config": config, "dataset_name": ...}
        # artifacts.metrics contains the metrics dict
        artifacts = hybrid_results.get("artifacts")
        if artifacts is None:
            return None

        # Get metrics from artifacts object
        metrics = getattr(artifacts, "metrics", {}) if hasattr(artifacts, "metrics") else {}
        if not metrics:
            metrics = hybrid_results.get("metrics", {})

        return {
            "Model": "Hybrid LSTM-XGBoost",
            "Category": "Hybrid",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(hybrid_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": "LSTM + XGBoost residual correction",
        }
    except Exception:
        return None


def _extract_lstm_ann_metrics(hybrid_results: Dict) -> Optional[Dict]:
    """Extract metrics from hybrid LSTM-ANN results."""
    try:
        # Results structure: {"artifacts": artifacts, "config": config, "dataset_name": ...}
        # artifacts.metrics contains the metrics dict
        artifacts = hybrid_results.get("artifacts")
        if artifacts is None:
            return None

        # Get metrics from artifacts object
        metrics = getattr(artifacts, "metrics", {}) if hasattr(artifacts, "metrics") else {}
        if not metrics:
            metrics = hybrid_results.get("metrics", {})

        return {
            "Model": "Hybrid LSTM-ANN",
            "Category": "Hybrid",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(hybrid_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": "LSTM + ANN residual correction",
        }
    except Exception:
        return None


def _extract_optimized_metrics(opt_results: Dict, model_type: str) -> Optional[Dict]:
    """Extract metrics from hyperparameter-optimized model results."""
    try:
        metrics = opt_results.get("best_metrics", opt_results.get("metrics", {}))
        return {
            "Model": f"{model_type} (Optimized)",
            "Category": _get_model_category(model_type),
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("mae", metrics.get("MAE"))),
            "RMSE": _safe_float(metrics.get("rmse", metrics.get("RMSE"))),
            "MAPE_%": _safe_float(metrics.get("mape", metrics.get("MAPE"))),
            "Accuracy_%": _safe_float(metrics.get("accuracy", metrics.get("Accuracy"))),
            "Runtime_s": _safe_float(opt_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Optimized via {opt_results.get('method', 'Grid Search')}",
        }
    except Exception:
        return None


def _extract_generic_hybrid_metrics(hybrid_data: Dict, hybrid_name: str) -> Optional[Dict]:
    """Extract metrics from generic hybrid pipeline results."""
    try:
        metrics = hybrid_data.get("metrics", {})
        return {
            "Model": f"Hybrid {hybrid_name}",
            "Category": "Hybrid",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(hybrid_data.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": f"Hybrid pipeline: {hybrid_name}",
        }
    except Exception:
        return None


def _extract_decomposition_metrics(decomp_results: Dict) -> Optional[Dict]:
    """Extract metrics from decomposition ensemble results."""
    try:
        metrics = decomp_results.get("metrics", {})
        return {
            "Model": "Decomposition Ensemble",
            "Category": "Ensemble",
            "Source": "Modeling Hub",
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae"))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse"))),
            "MAPE_%": _safe_float(metrics.get("MAPE", metrics.get("mape"))),
            "Accuracy_%": _safe_float(metrics.get("Accuracy", metrics.get("accuracy"))),
            "Runtime_s": _safe_float(decomp_results.get("runtime_s")),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Parameters": "STL Decomposition + Component Models",
        }
    except Exception:
        return None


# =============================================================================
# RESIDUAL COLLECTION FOR STATISTICAL TESTS
# =============================================================================

def collect_model_residuals(horizon: int = 1) -> Dict[str, Dict]:
    """
    Collect residuals from all trained models for a specific horizon.

    Returns dictionary mapping model names to their prediction data:
    {
        "Model Name": {
            "y_true": np.array,
            "y_pred": np.array,
            "residuals": np.array,
        }
    }
    """
    residuals_dict = {}

    # ARIMA
    arima_results = st.session_state.get("arima_mh_results")
    if arima_results:
        per_h = arima_results.get("per_h", {})
        h_data = per_h.get(horizon, per_h.get(str(horizon), {}))
        if h_data:
            y_test = h_data.get("y_test")
            forecast = h_data.get("forecast")
            if y_test is not None and forecast is not None:
                y_test = np.asarray(y_test).flatten()
                forecast = np.asarray(forecast).flatten()
                min_len = min(len(y_test), len(forecast))
                residuals_dict["ARIMA"] = {
                    "y_true": y_test[:min_len],
                    "y_pred": forecast[:min_len],
                    "residuals": y_test[:min_len] - forecast[:min_len],
                }

    # SARIMAX
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results:
        per_h = sarimax_results.get("per_h", {})
        h_data = per_h.get(horizon, per_h.get(str(horizon), {}))
        if h_data:
            y_test = h_data.get("y_test")
            forecast = h_data.get("forecast")
            if y_test is not None and forecast is not None:
                y_test = np.asarray(y_test).flatten()
                forecast = np.asarray(forecast).flatten()
                min_len = min(len(y_test), len(forecast))
                residuals_dict["SARIMAX"] = {
                    "y_true": y_test[:min_len],
                    "y_pred": forecast[:min_len],
                    "residuals": y_test[:min_len] - forecast[:min_len],
                }

    # ML Models (XGBoost, LSTM, ANN)
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        ml_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")
        if ml_results is None:
            ml_results = st.session_state.get(f"ml_mh_results_{model_name}")

        if ml_results:
            per_h = ml_results.get("per_h", {})
            h_data = per_h.get(horizon, per_h.get(str(horizon), {}))
            if h_data:
                y_test = h_data.get("y_test")
                y_pred = h_data.get("forecast", h_data.get("y_test_pred", h_data.get("y_pred")))
                if y_test is not None and y_pred is not None:
                    y_test = np.asarray(y_test).flatten()
                    y_pred = np.asarray(y_pred).flatten()
                    min_len = min(len(y_test), len(y_pred))
                    residuals_dict[model_name] = {
                        "y_true": y_test[:min_len],
                        "y_pred": y_pred[:min_len],
                        "residuals": y_test[:min_len] - y_pred[:min_len],
                    }

    # Hybrid models
    for hybrid_name, hybrid_key in [
        ("Hybrid LSTM-SARIMAX", "lstm_sarimax_results"),
        ("Hybrid LSTM-XGBoost", "lstm_xgb_results"),
        ("Hybrid LSTM-ANN", "lstm_ann_results"),
    ]:
        hybrid_results = st.session_state.get(hybrid_key)
        if hybrid_results:
            artifacts = hybrid_results.get("artifacts")
            if artifacts:
                y_test = getattr(artifacts, "y_test", None)
                y_pred = getattr(artifacts, "y_pred", None)
                if y_test is not None and y_pred is not None:
                    y_test = np.asarray(y_test).flatten()
                    y_pred = np.asarray(y_pred).flatten()
                    min_len = min(len(y_test), len(y_pred))
                    residuals_dict[hybrid_name] = {
                        "y_true": y_test[:min_len],
                        "y_pred": y_pred[:min_len],
                        "residuals": y_test[:min_len] - y_pred[:min_len],
                    }

    return residuals_dict


def get_training_data_for_baseline() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get training and test data from session state for baseline calculation.

    Returns:
        Tuple of (y_train, y_test) or (None, None) if not available
    """
    # Try to get from feature engineering state
    fe_state = st.session_state.get("feature_engineering", {})

    # Check for prepared data
    prepared_data = st.session_state.get("prepared_data")
    if prepared_data is not None and isinstance(prepared_data, pd.DataFrame):
        # Look for target column
        target_cols = [c for c in prepared_data.columns if c.startswith("Target_") or c == "ED_Total" or c == "Total_Patients"]
        if target_cols:
            y = prepared_data[target_cols[0]].values
            # Simple 80/20 split
            split_idx = int(len(y) * 0.8)
            return y[:split_idx], y[split_idx:]

    # Try to get from any model's per_h data
    for model_key in ["arima_mh_results", "sarimax_results", "ml_mh_results_xgboost"]:
        results = st.session_state.get(model_key)
        if results:
            per_h = results.get("per_h", {})
            for h_data in per_h.values():
                y_train = h_data.get("y_train")
                y_test = h_data.get("y_test")
                if y_train is not None and y_test is not None:
                    return np.asarray(y_train).flatten(), np.asarray(y_test).flatten()

    return None, None


def compute_naive_baselines_for_comparison() -> Dict[str, Dict]:
    """
    Compute naive baseline forecasts for skill score calculation.

    Returns dictionary with baseline forecasts and metrics.
    """
    y_train, y_test = get_training_data_for_baseline()

    if y_train is None or y_test is None:
        return {}

    if not STATISTICAL_TESTS_AVAILABLE:
        return {}

    return compute_all_baselines(y_train, y_test)


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_model_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive model rankings based on multiple metrics.

    Uses a weighted scoring system considering:
    - RMSE (weight: 0.35) - Primary accuracy measure
    - MAE (weight: 0.30) - Robust error measure
    - MAPE (weight: 0.20) - Relative error measure
    - Accuracy (weight: 0.15) - Overall accuracy

    Returns:
        DataFrame with rankings and composite score
    """
    if df.empty:
        return df

    ranking_df = df.copy()

    # Weights for composite score (must sum to 1.0)
    weights = {
        "RMSE": 0.35,
        "MAE": 0.30,
        "MAPE_%": 0.20,
        "Accuracy_%": 0.15,
    }

    # Compute individual metric ranks (lower is better for RMSE, MAE, MAPE)
    if "RMSE" in ranking_df.columns and ranking_df["RMSE"].notna().any():
        ranking_df["RMSE_Rank"] = ranking_df["RMSE"].rank(ascending=True, na_option="bottom")
    else:
        ranking_df["RMSE_Rank"] = np.nan

    if "MAE" in ranking_df.columns and ranking_df["MAE"].notna().any():
        ranking_df["MAE_Rank"] = ranking_df["MAE"].rank(ascending=True, na_option="bottom")
    else:
        ranking_df["MAE_Rank"] = np.nan

    if "MAPE_%" in ranking_df.columns and ranking_df["MAPE_%"].notna().any():
        ranking_df["MAPE_Rank"] = ranking_df["MAPE_%"].rank(ascending=True, na_option="bottom")
    else:
        ranking_df["MAPE_Rank"] = np.nan

    # Higher is better for accuracy
    if "Accuracy_%" in ranking_df.columns and ranking_df["Accuracy_%"].notna().any():
        ranking_df["Acc_Rank"] = ranking_df["Accuracy_%"].rank(ascending=False, na_option="bottom")
    else:
        ranking_df["Acc_Rank"] = np.nan

    # Compute weighted composite rank
    def compute_composite(row):
        ranks = []
        total_weight = 0

        if pd.notna(row.get("RMSE_Rank")):
            ranks.append(row["RMSE_Rank"] * weights["RMSE"])
            total_weight += weights["RMSE"]
        if pd.notna(row.get("MAE_Rank")):
            ranks.append(row["MAE_Rank"] * weights["MAE"])
            total_weight += weights["MAE"]
        if pd.notna(row.get("MAPE_Rank")):
            ranks.append(row["MAPE_Rank"] * weights["MAPE_%"])
            total_weight += weights["MAPE_%"]
        if pd.notna(row.get("Acc_Rank")):
            ranks.append(row["Acc_Rank"] * weights["Accuracy_%"])
            total_weight += weights["Accuracy_%"]

        if total_weight > 0 and ranks:
            return sum(ranks) / total_weight
        return np.nan

    ranking_df["Composite_Score"] = ranking_df.apply(compute_composite, axis=1)
    ranking_df["Overall_Rank"] = ranking_df["Composite_Score"].rank(ascending=True, na_option="bottom")

    return ranking_df.sort_values("Overall_Rank")


def compute_pairwise_improvements(df: pd.DataFrame, best_model: str) -> pd.DataFrame:
    """
    Compute percentage improvement of best model over all others.

    Returns:
        DataFrame with pairwise comparison metrics
    """
    if df.empty or best_model not in df["Model"].values:
        return pd.DataFrame()

    best_row = df[df["Model"] == best_model].iloc[0]
    comparisons = []

    for _, row in df.iterrows():
        if row["Model"] == best_model:
            continue

        comparison = {"Model": row["Model"]}

        for metric in ["MAE", "RMSE", "MAPE_%"]:
            best_val = _safe_float(best_row.get(metric))
            other_val = _safe_float(row.get(metric))

            if np.isfinite(best_val) and np.isfinite(other_val) and other_val != 0:
                improvement = ((other_val - best_val) / other_val) * 100
                comparison[f"{metric}_Improvement_%"] = improvement
            else:
                comparison[f"{metric}_Improvement_%"] = np.nan

        # Accuracy (higher is better)
        best_acc = _safe_float(best_row.get("Accuracy_%"))
        other_acc = _safe_float(row.get("Accuracy_%"))
        if np.isfinite(best_acc) and np.isfinite(other_acc) and other_acc != 0:
            comparison["Accuracy_Improvement_%"] = ((best_acc - other_acc) / other_acc) * 100
        else:
            comparison["Accuracy_Improvement_%"] = np.nan

        comparisons.append(comparison)

    return pd.DataFrame(comparisons)


def select_best_model(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Intelligently select the best model based on comprehensive analysis.

    Selection criteria:
    1. Composite ranking score
    2. Consistency across metrics
    3. Statistical significance of improvements

    Returns:
        Tuple of (best_model_name, analysis_details)
    """
    if df.empty:
        return "No models trained", {}

    ranked_df = compute_model_rankings(df)

    if ranked_df.empty:
        return "No models trained", {}

    best_model = ranked_df.iloc[0]["Model"]

    # Compute detailed analysis
    analysis = {
        "best_model": best_model,
        "category": ranked_df.iloc[0].get("Category", "Unknown"),
        "overall_rank": 1,
        "total_models": len(ranked_df),
        "metrics": {
            "MAE": _safe_float(ranked_df.iloc[0].get("MAE")),
            "RMSE": _safe_float(ranked_df.iloc[0].get("RMSE")),
            "MAPE_%": _safe_float(ranked_df.iloc[0].get("MAPE_%")),
            "Accuracy_%": _safe_float(ranked_df.iloc[0].get("Accuracy_%")),
        },
        "ranks": {
            "MAE_Rank": int(ranked_df.iloc[0].get("MAE_Rank", 0)) if pd.notna(ranked_df.iloc[0].get("MAE_Rank")) else None,
            "RMSE_Rank": int(ranked_df.iloc[0].get("RMSE_Rank", 0)) if pd.notna(ranked_df.iloc[0].get("RMSE_Rank")) else None,
            "MAPE_Rank": int(ranked_df.iloc[0].get("MAPE_Rank", 0)) if pd.notna(ranked_df.iloc[0].get("MAPE_Rank")) else None,
            "Acc_Rank": int(ranked_df.iloc[0].get("Acc_Rank", 0)) if pd.notna(ranked_df.iloc[0].get("Acc_Rank")) else None,
        },
        "composite_score": _safe_float(ranked_df.iloc[0].get("Composite_Score")),
    }

    # Determine consistency (how many metrics it ranks first)
    rank_1_count = sum(1 for v in analysis["ranks"].values() if v == 1)
    analysis["consistency"] = rank_1_count / 4 if len(analysis["ranks"]) == 4 else 0

    # Compute margin over second-best
    if len(ranked_df) > 1:
        second_best = ranked_df.iloc[1]
        analysis["runner_up"] = second_best["Model"]
        analysis["margin_over_second"] = {
            "MAE": ((_safe_float(second_best.get("MAE")) - analysis["metrics"]["MAE"]) /
                   _safe_float(second_best.get("MAE")) * 100) if _safe_float(second_best.get("MAE")) != 0 else 0,
            "RMSE": ((_safe_float(second_best.get("RMSE")) - analysis["metrics"]["RMSE"]) /
                    _safe_float(second_best.get("RMSE")) * 100) if _safe_float(second_best.get("RMSE")) != 0 else 0,
        }

    return best_model, analysis


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_model_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive model comparison radar chart."""
    if df.empty:
        return go.Figure()

    # Normalize metrics to 0-100 scale for radar chart
    metrics = ["MAE", "RMSE", "MAPE_%", "Accuracy_%"]
    categories = ["MAE", "RMSE", "MAPE", "Accuracy"]

    fig = go.Figure()

    for _, row in df.iterrows():
        model_name = row["Model"]
        values = []

        for metric in metrics:
            val = _safe_float(row.get(metric))
            if metric == "Accuracy_%":
                # Accuracy: higher is better, keep as-is (already 0-100)
                values.append(val if np.isfinite(val) else 0)
            else:
                # Error metrics: invert so higher = better on radar
                # Normalize based on max in dataset
                max_val = df[metric].max() if metric in df.columns else 1
                if np.isfinite(val) and max_val > 0:
                    values.append(100 * (1 - val / max_val))
                else:
                    values.append(0)

        # Close the radar chart
        values.append(values[0])
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=model_name,
            line=dict(color=_get_model_color(model_name)),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color=SUBTLE_TEXT),
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=TEXT_COLOR),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        title=dict(
            text="Model Performance Comparison (Normalized)",
            font=dict(size=18, color=TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        height=500,
    )

    return fig


def create_metric_bar_chart(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Create bar chart comparing models on a specific metric."""
    if df.empty or metric not in df.columns:
        return go.Figure()

    # Sort by metric (ascending for errors, descending for accuracy)
    ascending = metric != "Accuracy_%"
    sorted_df = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)

    colors = [_get_model_color(m) for m in sorted_df["Model"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["Model"],
        y=sorted_df[metric],
        marker_color=colors,
        text=sorted_df[metric].round(4),
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Model",
            tickangle=45,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title=metric.replace("_", " ").replace("%", " (%)"),
            tickfont=dict(color=TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(b=120),
    )

    return fig


def create_ranking_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap showing model rankings across metrics."""
    if df.empty:
        return go.Figure()

    ranked_df = compute_model_rankings(df)

    rank_cols = ["MAE_Rank", "RMSE_Rank", "MAPE_Rank", "Acc_Rank"]
    display_cols = ["MAE", "RMSE", "MAPE", "Accuracy"]

    # Filter to only columns that exist
    available_ranks = [c for c in rank_cols if c in ranked_df.columns]
    available_display = [display_cols[rank_cols.index(c)] for c in available_ranks]

    if not available_ranks:
        return go.Figure()

    z_data = ranked_df[available_ranks].values

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=available_display,
        y=ranked_df["Model"].tolist(),
        colorscale="RdYlGn_r",  # Red=bad (high rank), Green=good (low rank)
        text=z_data.astype(int),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Rank: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Model Rankings by Metric (1 = Best)",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Metric",
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title="Model",
            tickfont=dict(color=TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(ranked_df) * 40 + 100),
    )

    return fig


def create_improvement_chart(improvements_df: pd.DataFrame, best_model: str) -> go.Figure:
    """Create chart showing improvement of best model over others."""
    if improvements_df.empty:
        return go.Figure()

    metrics = ["MAE_Improvement_%", "RMSE_Improvement_%", "MAPE_%_Improvement_%"]
    display_names = ["MAE", "RMSE", "MAPE"]

    available = [m for m in metrics if m in improvements_df.columns]
    if not available:
        return go.Figure()

    fig = go.Figure()

    for metric, display in zip(available, display_names[:len(available)]):
        fig.add_trace(go.Bar(
            name=display,
            x=improvements_df["Model"],
            y=improvements_df[metric],
            text=improvements_df[metric].round(1).astype(str) + "%",
            textposition="auto",
        ))

    fig.update_layout(
        title=dict(
            text=f"Performance Improvement: {best_model} vs Other Models",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Compared Model",
            tickangle=45,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title="Improvement (%)",
            tickfont=dict(color=TEXT_COLOR),
        ),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def _generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table code for thesis inclusion."""
    if df.empty:
        return "% No data available"

    latex = r"""\begin{table}[htbp]
\centering
\caption{Forecasting Model Comparison Results}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE (\%)} & \textbf{Accuracy (\%)} \\
\midrule
"""

    ranked = compute_model_rankings(df)

    for _, row in ranked.iterrows():
        model = row["Model"].replace("_", r"\_")
        mae = f"{row['MAE']:.4f}" if pd.notna(row.get('MAE')) else "--"
        rmse = f"{row['RMSE']:.4f}" if pd.notna(row.get('RMSE')) else "--"
        mape = f"{row['MAPE_%']:.2f}" if pd.notna(row.get('MAPE_%')) else "--"
        acc = f"{row['Accuracy_%']:.2f}" if pd.notna(row.get('Accuracy_%')) else "--"

        # Bold the best model
        if row.get("Overall_Rank") == 1:
            latex += f"\\textbf{{{model}}} & \\textbf{{{mae}}} & \\textbf{{{rmse}}} & \\textbf{{{mape}}} & \\textbf{{{acc}}} \\\\\n"
        else:
            latex += f"{model} & {mae} & {rmse} & {mape} & {acc} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Bold values indicate the best performing model based on composite ranking.
\item MAE = Mean Absolute Error, RMSE = Root Mean Square Error, MAPE = Mean Absolute Percentage Error.
\end{tablenotes}
\end{table}
"""

    return latex


def create_horizon_performance_chart(source: str = "all") -> Optional[go.Figure]:
    """Create per-horizon performance comparison chart."""
    horizon_data = []

    # Collect horizon-wise data from all sources
    if source in ["all", "benchmarks"]:
        # ARIMA
        arima_results = st.session_state.get("arima_mh_results")
        if arima_results:
            results_df = arima_results.get("results_df")
            if results_df is not None:
                for _, row in results_df.iterrows():
                    horizon_data.append({
                        "Model": "ARIMA",
                        "Horizon": int(row.get("Horizon", 0)),
                        "MAE": _safe_float(row.get("Test_MAE", row.get("MAE"))),
                        "RMSE": _safe_float(row.get("Test_RMSE", row.get("RMSE"))),
                    })

        # SARIMAX
        sarimax_results = st.session_state.get("sarimax_results")
        if sarimax_results:
            results_df = sarimax_results.get("results_df")
            if results_df is not None:
                for _, row in results_df.iterrows():
                    horizon_data.append({
                        "Model": "SARIMAX",
                        "Horizon": int(row.get("Horizon", 0)),
                        "MAE": _safe_float(row.get("Test_MAE", row.get("MAE"))),
                        "RMSE": _safe_float(row.get("Test_RMSE", row.get("RMSE"))),
                    })

    if source in ["all", "ml"]:
        # ML Models - metrics are in results_df, not per_h
        for model_name in ["XGBoost", "LSTM", "ANN"]:
            # Check both lowercase and titlecase keys
            ml_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")
            if ml_results is None:
                ml_results = st.session_state.get(f"ml_mh_results_{model_name}")
            if ml_results:
                results_df = ml_results.get("results_df")
                if results_df is not None and not results_df.empty:
                    for _, row in results_df.iterrows():
                        horizon_data.append({
                            "Model": model_name,
                            "Horizon": int(row.get("Horizon", 0)),
                            "MAE": _safe_float(row.get("Test_MAE")),
                            "RMSE": _safe_float(row.get("Test_RMSE")),
                        })

    if not horizon_data:
        return None

    df = pd.DataFrame(horizon_data)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE by Horizon", "RMSE by Horizon"))

    for model in df["Model"].unique():
        model_df = df[df["Model"] == model].sort_values("Horizon")
        color = _get_model_color(model)

        fig.add_trace(
            go.Scatter(
                x=model_df["Horizon"],
                y=model_df["MAE"],
                mode="lines+markers",
                name=model,
                line=dict(color=color),
                showlegend=True,
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=model_df["Horizon"],
                y=model_df["RMSE"],
                mode="lines+markers",
                name=model,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=dict(
            text="Performance Degradation Across Forecast Horizons",
            font=dict(size=18, color=TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(title_text="Forecast Horizon (days)", tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(title_text="Error", tickfont=dict(color=TEXT_COLOR))

    return fig


# =============================================================================
# THESIS-QUALITY SUMMARY GENERATION
# =============================================================================

def generate_thesis_summary(df: pd.DataFrame, best_model: str, analysis: Dict) -> str:
    """
    Generate academic-quality summary text for thesis inclusion.

    Returns:
        Markdown-formatted summary text
    """
    if df.empty:
        return "No models have been trained yet."

    ranked_df = compute_model_rankings(df)
    n_models = len(ranked_df)

    # Model category breakdown
    categories = df["Category"].value_counts().to_dict()
    category_text = ", ".join([f"{v} {k}" for k, v in categories.items()])

    # Best model metrics
    metrics = analysis.get("metrics", {})

    summary = f"""
### Model Comparison Summary

A comprehensive evaluation was conducted across **{n_models} forecasting models** spanning {len(categories)} categories ({category_text}).

#### Best Performing Model: **{best_model}**

The **{best_model}** model demonstrated superior performance based on a weighted composite ranking system that considers multiple error metrics:

| Metric | Value | Rank |
|--------|-------|------|
| MAE (Mean Absolute Error) | {metrics.get('MAE', 'N/A'):.4f} | #{analysis.get('ranks', {}).get('MAE_Rank', 'N/A')} |
| RMSE (Root Mean Square Error) | {metrics.get('RMSE', 'N/A'):.4f} | #{analysis.get('ranks', {}).get('RMSE_Rank', 'N/A')} |
| MAPE (Mean Absolute Percentage Error) | {metrics.get('MAPE_%', 'N/A'):.2f}% | #{analysis.get('ranks', {}).get('MAPE_Rank', 'N/A')} |
| Accuracy | {metrics.get('Accuracy_%', 'N/A'):.2f}% | #{analysis.get('ranks', {}).get('Acc_Rank', 'N/A')} |

#### Ranking Methodology

The composite ranking was computed using a weighted scoring system:
- RMSE: 35% weight (primary accuracy measure for healthcare forecasting)
- MAE: 30% weight (robust to outliers, interpretable in original units)
- MAPE: 20% weight (relative error, useful for comparing across different scales)
- Accuracy: 15% weight (overall prediction accuracy)

"""

    # Add comparison with runner-up if available
    if "runner_up" in analysis:
        margin = analysis.get("margin_over_second", {})
        summary += f"""
#### Comparison with Runner-Up

The best model ({best_model}) outperformed the runner-up ({analysis['runner_up']}) with:
- **{margin.get('RMSE', 0):.2f}%** lower RMSE
- **{margin.get('MAE', 0):.2f}%** lower MAE

"""

    # Add consistency note
    consistency = analysis.get("consistency", 0)
    if consistency >= 0.75:
        summary += f"The {best_model} model showed **highly consistent** performance, ranking first in {int(consistency * 4)} out of 4 metrics.\n\n"
    elif consistency >= 0.5:
        summary += f"The {best_model} model showed **good consistency**, ranking first or second in most metrics.\n\n"
    else:
        summary += f"Note: While {best_model} achieved the best composite score, performance varied across individual metrics. Consider the specific use case when selecting a model.\n\n"

    return summary


# =============================================================================
# MAIN PAGE RENDERING
# =============================================================================

# Premium Hero Header
render_scifi_hero_header(
    title="Model Results",
    subtitle="Performance analysis and model comparison. Comprehensive metrics and thesis-quality reporting.",
    status="SYSTEM ONLINE"
)

# Aggregate all results
all_models_df = aggregate_all_model_results()

# Quick stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Models Trained", len(all_models_df) if not all_models_df.empty else 0)

with col2:
    if not all_models_df.empty:
        categories = all_models_df["Category"].nunique()
        st.metric("Model Categories", categories)
    else:
        st.metric("Model Categories", 0)

with col3:
    if not all_models_df.empty and "RMSE" in all_models_df.columns:
        best_rmse = all_models_df["RMSE"].min()
        st.metric("Best RMSE", f"{best_rmse:.4f}" if np.isfinite(best_rmse) else "—")
    else:
        st.metric("Best RMSE", "—")

with col4:
    if not all_models_df.empty and "Accuracy_%" in all_models_df.columns:
        best_acc = all_models_df["Accuracy_%"].max()
        st.metric("Best Accuracy", f"{best_acc:.2f}%" if np.isfinite(best_acc) else "—")
    else:
        st.metric("Best Accuracy", "—")

st.divider()

# Check if we have results
if all_models_df.empty:
    st.warning(
        "⚠️ **No trained models found**\n\n"
        "Please train models from:\n"
        "- **05_Benchmarks.py** - ARIMA, SARIMAX statistical models\n"
        "- **08_Modeling_Hub.py** - XGBoost, LSTM, ANN machine learning models\n\n"
        "After training, return here for comprehensive analysis."
    )

    # Show available session keys for debugging
    with st.expander("📂 Debug: Available Session Keys", expanded=False):
        st.write("Current session state keys:", list(st.session_state.keys()))

    st.stop()

# Best Model Selection
best_model, analysis = select_best_model(all_models_df)

# Display best model prominently
st.markdown(
    f"""
    <div class="best-model-card">
        <h2 style="color: #22C55E; margin: 0 0 1rem 0; font-size: 1.5rem;">
            🏆 Recommended Model: {best_model}
        </h2>
        <p style="color: {SUBTLE_TEXT}; margin: 0;">
            Category: <strong>{analysis.get('category', 'Unknown')}</strong> |
            Composite Score: <strong>{analysis.get('composite_score', 0):.3f}</strong> |
            Consistency: <strong>{analysis.get('consistency', 0)*100:.0f}%</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main tabs - PhD-level structure (6 tabs)
tab_summary, tab_performance, tab_validation, tab_diagnostics, tab_explainability, tab_export = st.tabs([
    "🎯 Executive Summary", "📊 Model Performance", "📐 Statistical Validation", "🔬 Diagnostics", "🔍 Explainability", "📤 Export"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY (PhD-Level)
# -----------------------------------------------------------------------------
with tab_summary:
    st.markdown("### 🎯 Executive Summary")
    st.caption("30-second answer to: Which model should I use?")

    # Main recommendation card
    metrics = analysis.get("metrics", {})
    ranks = analysis.get("ranks", {})

    # Compute skill score if baselines available
    skill_score_pct = None
    skill_interpretation = "N/A"
    if STATISTICAL_TESTS_AVAILABLE:
        baselines = compute_naive_baselines_for_comparison()
        if baselines and "persistence" in baselines:
            naive_rmse = baselines["persistence"]["rmse"]
            model_rmse = metrics.get("RMSE", np.nan)
            if np.isfinite(model_rmse) and naive_rmse > 0:
                skill_score_pct = (1 - model_rmse / naive_rmse) * 100
                if skill_score_pct > 30:
                    skill_interpretation = "Substantial improvement"
                elif skill_score_pct > 10:
                    skill_interpretation = "Moderate improvement"
                elif skill_score_pct > 0:
                    skill_interpretation = "Marginal improvement"
                else:
                    skill_interpretation = "No improvement over baseline"

    # Recommendation card with key metrics
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
                border: 2px solid #22C55E; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
                box-shadow: 0 0 40px rgba(34, 197, 94, 0.3);">
        <h2 style="color: #22C55E; margin: 0 0 1rem 0; font-size: 1.5rem;">
            🏆 RECOMMENDED: {best_model}
        </h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="color: {SUBTLE_TEXT}; font-size: 0.8rem;">RMSE</div>
                <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: bold;">
                    {metrics.get('RMSE', 0):.2f}
                </div>
                <div style="color: {SUBTLE_TEXT}; font-size: 0.7rem;">patients</div>
            </div>
            <div style="text-align: center;">
                <div style="color: {SUBTLE_TEXT}; font-size: 0.8rem;">MAE</div>
                <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: bold;">
                    {metrics.get('MAE', 0):.2f}
                </div>
                <div style="color: {SUBTLE_TEXT}; font-size: 0.7rem;">patients</div>
            </div>
            <div style="text-align: center;">
                <div style="color: {SUBTLE_TEXT}; font-size: 0.8rem;">Accuracy</div>
                <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: bold;">
                    {metrics.get('Accuracy_%', 0):.1f}%
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: {SUBTLE_TEXT}; font-size: 0.8rem;">Skill Score</div>
                <div style="color: {'#22C55E' if skill_score_pct and skill_score_pct > 10 else WARNING_COLOR}; font-size: 1.3rem; font-weight: bold;">
                    {f'+{skill_score_pct:.1f}%' if skill_score_pct else 'N/A'}
                </div>
                <div style="color: {SUBTLE_TEXT}; font-size: 0.7rem;">vs naive</div>
            </div>
        </div>
        <div style="color: {SUBTLE_TEXT}; font-size: 0.85rem;">
            Category: <strong style="color: {TEXT_COLOR}">{analysis.get('category', 'Unknown')}</strong> |
            Consistency: <strong style="color: {TEXT_COLOR}">{analysis.get('consistency', 0)*100:.0f}%</strong> |
            Skill: <strong style="color: {'#22C55E' if skill_score_pct and skill_score_pct > 10 else WARNING_COLOR}">{skill_interpretation}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Business Impact Section
    st.markdown("### 💼 Business Impact Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### What These Metrics Mean")
        mae_val = metrics.get("MAE", np.nan)
        rmse_val = metrics.get("RMSE", np.nan)

        if np.isfinite(mae_val) and np.isfinite(rmse_val):
            st.markdown(f"""
            - **Average forecast error:** ±{mae_val:.1f} patients per day
            - **95% of forecasts within:** ±{rmse_val * 1.96:.1f} patients
            - **For staffing:** Plan for ±{int(np.ceil(rmse_val))} staff buffer
            """)
        else:
            st.info("Train models to see business impact metrics")

    with col2:
        st.markdown("#### Model Selection Guidance")

        # Create guidance table based on available models
        if len(all_models_df) > 1:
            guidance_data = []
            ranked = compute_model_rankings(all_models_df)

            # Best for 1-day ahead (lowest RMSE)
            if "RMSE" in ranked.columns:
                best_1day = ranked.nsmallest(1, "RMSE")["Model"].iloc[0]
                guidance_data.append({"Use Case": "1-day ahead", "Model": best_1day, "Why": "Lowest RMSE"})

            # Best accuracy
            if "Accuracy_%" in ranked.columns:
                best_acc = ranked.nlargest(1, "Accuracy_%")["Model"].iloc[0]
                guidance_data.append({"Use Case": "Overall accuracy", "Model": best_acc, "Why": "Highest accuracy"})

            # Hybrid for complexity
            hybrids = ranked[ranked["Category"] == "Hybrid"]
            if not hybrids.empty:
                best_hybrid = hybrids.iloc[0]["Model"]
                guidance_data.append({"Use Case": "Complex patterns", "Model": best_hybrid, "Why": "Captures nonlinearity"})

            if guidance_data:
                guidance_df = pd.DataFrame(guidance_data)
                st.dataframe(guidance_df, use_container_width=True, hide_index=True)

    st.divider()

    # Quick Rankings
    st.markdown("### 🏅 Quick Rankings")

    ranked_df = compute_model_rankings(all_models_df)
    top_3 = ranked_df.head(3)

    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 2rem;">{medals[i]}</div>
                <div style="color: {TEXT_COLOR}; font-weight: bold;">{row['Model']}</div>
                <div style="color: {SUBTLE_TEXT}; font-size: 0.8rem;">
                    RMSE: {row.get('RMSE', 0):.3f} | Acc: {row.get('Accuracy_%', 0):.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE (Enhanced with Baseline Comparison)
# -----------------------------------------------------------------------------
with tab_performance:
    st.markdown("### 📊 Model Performance Analysis")

    # Section 1: All Models Table with Skill Scores
    st.markdown("#### All Trained Models")

    display_df = all_models_df.copy()

    # Add skill score column if baselines available
    if STATISTICAL_TESTS_AVAILABLE:
        baselines = compute_naive_baselines_for_comparison()
        if baselines and "persistence" in baselines:
            naive_rmse = baselines["persistence"]["rmse"]
            display_df["Skill_%"] = display_df["RMSE"].apply(
                lambda x: (1 - x / naive_rmse) * 100 if pd.notna(x) and naive_rmse > 0 else np.nan
            )

    # Select columns to display
    display_cols = ["Model", "Category", "MAE", "RMSE", "MAPE_%", "Accuracy_%"]
    if "Skill_%" in display_df.columns:
        display_cols.insert(4, "Skill_%")
    display_cols = [c for c in display_cols if c in display_df.columns]

    # Apply formatting
    format_dict = {col: METRIC_FORMATTERS.get(col, "{:.4f}") for col in display_cols if col in METRIC_FORMATTERS}
    if "Skill_%" in display_df.columns:
        format_dict["Skill_%"] = "{:+.1f}"

    st.dataframe(
        display_df[display_cols].style.format(format_dict, na_rep="—").background_gradient(
            subset=[c for c in ["MAE", "RMSE", "MAPE_%"] if c in display_cols],
            cmap="RdYlGn_r"
        ).background_gradient(
            subset=[c for c in ["Accuracy_%", "Skill_%"] if c in display_cols],
            cmap="RdYlGn"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Section 2: Baseline Comparison (NEW)
    st.markdown("#### 📏 Baseline Comparison")

    if STATISTICAL_TESTS_AVAILABLE and DIAGNOSTIC_PLOTS_AVAILABLE:
        baselines = compute_naive_baselines_for_comparison()
        if baselines:
            col1, col2 = st.columns(2)

            with col1:
                # Baseline metrics table
                baseline_data = []
                for name, data in baselines.items():
                    baseline_data.append({
                        "Baseline": name.replace("_", " ").title(),
                        "RMSE": data["rmse"],
                        "MAE": data["mae"],
                        "MAPE_%": data.get("mape", np.nan),
                    })
                baseline_df = pd.DataFrame(baseline_data)

                st.markdown("##### Naive Baseline Forecasts")
                st.dataframe(
                    baseline_df.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE_%": "{:.2f}"}, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                # Best model vs baselines bar chart
                if not all_models_df.empty:
                    best_rmse = all_models_df["RMSE"].min()
                    model_metrics = {"rmse": best_rmse, "mae": all_models_df["MAE"].min()}
                    baseline_metrics = {name: {"rmse": data["rmse"], "mae": data["mae"]}
                                       for name, data in baselines.items()}

                    fig_baseline = create_baseline_comparison_chart(model_metrics, baseline_metrics, "rmse")
                    st.plotly_chart(fig_baseline, use_container_width=True)
        else:
            st.info("💡 Train models to see baseline comparison. Baselines include: Naive (persistence), Seasonal Naive, Moving Average.")
    else:
        st.warning("⚠️ Statistical tests module not available. Install required dependencies.")

    st.divider()

    # Section 3: Visual Comparison (Radar + Bar)
    st.markdown("#### 📈 Visual Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart
        radar_fig = create_model_comparison_chart(all_models_df)
        st.plotly_chart(radar_fig, use_container_width=True)

    with col2:
        # Metric selection for bar chart
        metric_option = st.selectbox(
            "Select metric to compare",
            ["RMSE", "MAE", "MAPE_%", "Accuracy_%"],
            index=0,
            key="perf_metric_select"
        )

        bar_fig = create_metric_bar_chart(
            all_models_df,
            metric_option,
            f"Model Comparison: {metric_option.replace('_', ' ')}"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.divider()

    # Section 4: Per-Horizon Analysis
    st.markdown("#### 🗓️ Performance by Forecast Horizon")
    st.caption("How accuracy degrades as forecast horizon increases")

    horizon_fig = create_horizon_performance_chart()
    if horizon_fig:
        st.plotly_chart(horizon_fig, use_container_width=True)
    else:
        st.info("Train multi-horizon models to see per-horizon analysis.")

    st.divider()

    # Section 5: Rankings with Heatmap
    st.markdown("#### 🏅 Model Rankings")

    ranked_df = compute_model_rankings(all_models_df)
    heatmap_fig = create_ranking_heatmap(all_models_df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Ranking methodology
    with st.expander("📖 Ranking Methodology"):
        st.markdown("""
        **Composite Score Calculation:**

        The ranking uses weighted scoring designed for healthcare forecasting:

        | Metric | Weight | Rationale |
        |--------|--------|-----------|
        | RMSE | 35% | Penalizes large errors critical in healthcare |
        | MAE | 30% | Robust to outliers, interpretable |
        | MAPE | 20% | Relative error for scale comparison |
        | Accuracy | 15% | Overall prediction accuracy |
        """)

# -----------------------------------------------------------------------------
# TAB 3: STATISTICAL VALIDATION (PhD-Level)
# -----------------------------------------------------------------------------
with tab_validation:
    st.markdown("### 📐 Statistical Validation")
    st.caption("Prove your model is statistically significantly better - the PhD-level standard")

    if not STATISTICAL_TESTS_AVAILABLE:
        st.error("⚠️ Statistical tests module not available. Please check installation.")
    else:
        # Section 1: Diebold-Mariano Test Matrix
        st.markdown("#### Diebold-Mariano Test Matrix")
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <strong>What is this?</strong> The Diebold-Mariano test compares forecast accuracy between two models.
            <br><br>
            <strong>How to interpret:</strong> p < 0.05 means the models are <em>significantly</em> different.
            <br>
            *** = p<0.001 | ** = p<0.01 | * = p<0.05 | ns = not significant
        </div>
        """, unsafe_allow_html=True)

        # Select horizon for comparison
        horizon_select = st.selectbox(
            "Select forecast horizon for comparison:",
            options=[1, 2, 3, 4, 5, 6, 7],
            index=0,
            key="dm_horizon_select"
        )

        # Collect residuals for all models
        residuals_data = collect_model_residuals(horizon=horizon_select)

        if len(residuals_data) >= 2:
            # Compute DM test matrix
            errors_dict = {name: data["residuals"] for name, data in residuals_data.items()}
            p_matrix, sig_matrix = diebold_mariano_matrix(errors_dict, h=horizon_select)

            # Display heatmap
            if DIAGNOSTIC_PLOTS_AVAILABLE:
                dm_heatmap = create_dm_heatmap(p_matrix, sig_matrix)
                st.plotly_chart(dm_heatmap, use_container_width=True)
            else:
                st.dataframe(p_matrix.style.format("{:.4f}", na_rep="—"), use_container_width=True)

            # Text summary
            st.markdown("##### Key Findings")

            # Find which models are significantly different from best
            best_model_name = best_model.replace(" (Avg)", "")
            if best_model_name in p_matrix.index:
                sig_vs_best = []
                for other_model in p_matrix.columns:
                    if other_model != best_model_name:
                        p_val = p_matrix.loc[best_model_name, other_model]
                        if pd.notna(p_val) and p_val < 0.05:
                            sig_vs_best.append(f"**{best_model_name}** vs **{other_model}**: p={p_val:.4f} {sig_matrix.loc[best_model_name, other_model]}")

                if sig_vs_best:
                    st.success(f"✅ The best model ({best_model_name}) is **statistically significantly better** than:")
                    for finding in sig_vs_best:
                        st.markdown(f"  - {finding}")
                else:
                    st.warning(f"⚠️ No statistically significant differences found between {best_model_name} and other models at p < 0.05")
        else:
            st.warning(f"Need at least 2 models with residuals for comparison. Found: {list(residuals_data.keys())}")

        st.divider()

        # Section 2: Skill Scores
        st.markdown("#### 📊 Forecast Skill Scores")
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <strong>Skill Score</strong> = 1 - (Model Error / Baseline Error)
            <br><br>
            <strong>Interpretation:</strong>
            • Skill > 0.3 = Substantial improvement
            • Skill 0.1-0.3 = Moderate improvement
            • Skill < 0.1 = Marginal improvement
            • Skill ≤ 0 = No better than baseline
        </div>
        """, unsafe_allow_html=True)

        baselines = compute_naive_baselines_for_comparison()

        if baselines and residuals_data:
            # Compute skill scores for each model
            skill_data = []

            for model_name, model_data in residuals_data.items():
                for baseline_name, baseline_data in baselines.items():
                    y_test = model_data["y_true"]
                    model_errors = model_data["residuals"]
                    baseline_errors = baseline_data["errors"]

                    # Align lengths
                    min_len = min(len(model_errors), len(baseline_errors))
                    if min_len > 0:
                        skill = compute_skill_score(
                            model_errors[:min_len],
                            baseline_errors[:min_len],
                            y_test[:min_len],
                            model_name=model_name,
                            baseline_type=baseline_name
                        )

                        skill_data.append({
                            "Model": model_name,
                            "vs Baseline": baseline_name.replace("_", " ").title(),
                            "Skill Score": skill.skill_score,
                            "MASE": skill.mase,
                            "Improvement %": skill.improvement_pct,
                            "Interpretation": skill.interpretation,
                        })

            if skill_data:
                skill_df = pd.DataFrame(skill_data)

                # Show vs primary baseline (persistence/naive)
                primary_baseline = "Persistence"
                primary_df = skill_df[skill_df["vs Baseline"] == primary_baseline]

                if not primary_df.empty:
                    st.markdown(f"##### Skill Scores vs {primary_baseline} Baseline")

                    st.dataframe(
                        primary_df[["Model", "Skill Score", "MASE", "Improvement %", "Interpretation"]].style.format({
                            "Skill Score": "{:.3f}",
                            "MASE": "{:.3f}",
                            "Improvement %": "{:+.1f}%",
                        }, na_rep="—").background_gradient(
                            subset=["Skill Score", "Improvement %"],
                            cmap="RdYlGn"
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Skill score gauges for top 3 models
                if DIAGNOSTIC_PLOTS_AVAILABLE and len(primary_df) > 0:
                    top_skills = primary_df.nlargest(3, "Skill Score")
                    cols = st.columns(min(3, len(top_skills)))

                    for i, (_, row) in enumerate(top_skills.iterrows()):
                        with cols[i]:
                            gauge = create_skill_score_gauge(
                                row["Skill Score"],
                                row["Model"],
                                primary_baseline
                            )
                            st.plotly_chart(gauge, use_container_width=True)

                with st.expander("📋 Full Skill Scores vs All Baselines"):
                    st.dataframe(
                        skill_df.style.format({
                            "Skill Score": "{:.3f}",
                            "MASE": "{:.3f}",
                            "Improvement %": "{:+.1f}%",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("Train models to compute skill scores against naive baselines.")

        st.divider()

        # Section 3: Confidence Intervals
        st.markdown("#### 📉 Metric Confidence Intervals (Bootstrap)")

        if residuals_data:
            ci_data = []
            for model_name, model_data in residuals_data.items():
                residuals = model_data["residuals"]
                if len(residuals) > 10:
                    rmse_ci = bootstrap_confidence_interval(residuals, metric="rmse", n_bootstrap=500)
                    mae_ci = bootstrap_confidence_interval(residuals, metric="mae", n_bootstrap=500)

                    ci_data.append({
                        "Model": model_name,
                        "RMSE": f"{rmse_ci.point_estimate:.3f}",
                        "RMSE 95% CI": f"[{rmse_ci.ci_lower:.3f}, {rmse_ci.ci_upper:.3f}]",
                        "MAE": f"{mae_ci.point_estimate:.3f}",
                        "MAE 95% CI": f"[{mae_ci.ci_lower:.3f}, {mae_ci.ci_upper:.3f}]",
                    })

            if ci_data:
                st.dataframe(pd.DataFrame(ci_data), use_container_width=True, hide_index=True)

                st.caption("Confidence intervals computed using 500 bootstrap samples. Overlapping CIs suggest models may not be significantly different.")
        else:
            st.info("Train models to compute confidence intervals.")

# -----------------------------------------------------------------------------
# TAB 4: DIAGNOSTICS (PhD-Level Residual Analysis)
# -----------------------------------------------------------------------------
with tab_diagnostics:
    st.markdown("### 🔬 Residual Diagnostics")
    st.caption("Validate model assumptions and detect issues - essential for thesis-quality work")

    if not RESIDUAL_DIAGNOSTICS_AVAILABLE or not DIAGNOSTIC_PLOTS_AVAILABLE:
        st.error("⚠️ Diagnostics modules not available. Please check installation.")
    else:
        # Model and horizon selection
        col1, col2 = st.columns(2)

        # Get available models with residuals
        test_residuals = collect_model_residuals(horizon=1)
        available_for_diagnostics = list(test_residuals.keys()) if test_residuals else []

        with col1:
            diag_model = st.selectbox(
                "Select Model:",
                options=available_for_diagnostics if available_for_diagnostics else ["No models available"],
                key="diag_model_select"
            )

        with col2:
            diag_horizon = st.selectbox(
                "Select Horizon:",
                options=[1, 2, 3, 4, 5, 6, 7],
                index=0,
                key="diag_horizon_select"
            )

        if diag_model and diag_model != "No models available":
            # Get residuals for selected model and horizon
            residuals_data = collect_model_residuals(horizon=diag_horizon)

            if diag_model in residuals_data:
                model_data = residuals_data[diag_model]
                y_true = model_data["y_true"]
                y_pred = model_data["y_pred"]
                residuals = model_data["residuals"]

                # Section 1: Comprehensive Diagnostic Panel
                st.markdown(f"#### Diagnostic Panel: {diag_model} (Horizon {diag_horizon})")

                diag_panel = create_diagnostic_panel(y_true, y_pred, diag_model, diag_horizon)
                st.plotly_chart(diag_panel, use_container_width=True)

                st.divider()

                # Section 2: Statistical Tests Summary
                st.markdown("#### 📋 Statistical Tests Summary")

                # Compute full diagnostics
                full_diag = compute_full_diagnostics(residuals, y_pred, diag_model, diag_horizon)

                # Overall badge
                badge_text, badge_color = get_diagnostic_badge(full_diag)
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.8); border-left: 4px solid {badge_color};
                            padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong style="color: {badge_color};">{badge_text}</strong>
                    <span style="color: {SUBTLE_TEXT};">
                        ({4 - len(full_diag.issues_found)}/4 tests passed)
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Tests table
                tests_data = [
                    {
                        "Test": full_diag.normality_test.test_name,
                        "Statistic": f"{full_diag.normality_test.statistic:.4f}" if pd.notna(full_diag.normality_test.statistic) else "—",
                        "p-value": format_p_value_display(full_diag.normality_test.p_value) if pd.notna(full_diag.normality_test.p_value) else "—",
                        "Result": "✅ PASS" if full_diag.normality_test.passed else "❌ FAIL",
                        "Interpretation": full_diag.normality_test.interpretation,
                    },
                    {
                        "Test": full_diag.autocorrelation_test.test_name,
                        "Statistic": f"{full_diag.autocorrelation_test.statistic:.4f}" if pd.notna(full_diag.autocorrelation_test.statistic) else "—",
                        "p-value": format_p_value_display(full_diag.autocorrelation_test.p_value) if pd.notna(full_diag.autocorrelation_test.p_value) else "—",
                        "Result": "✅ PASS" if full_diag.autocorrelation_test.passed else "❌ FAIL",
                        "Interpretation": full_diag.autocorrelation_test.interpretation,
                    },
                    {
                        "Test": full_diag.heteroscedasticity_test.test_name,
                        "Statistic": f"{full_diag.heteroscedasticity_test.statistic:.4f}" if pd.notna(full_diag.heteroscedasticity_test.statistic) else "—",
                        "p-value": format_p_value_display(full_diag.heteroscedasticity_test.p_value) if pd.notna(full_diag.heteroscedasticity_test.p_value) else "—",
                        "Result": "✅ PASS" if full_diag.heteroscedasticity_test.passed else "❌ FAIL",
                        "Interpretation": full_diag.heteroscedasticity_test.interpretation,
                    },
                    {
                        "Test": full_diag.bias_test.test_name,
                        "Statistic": f"{full_diag.bias_test.statistic:.4f}" if pd.notna(full_diag.bias_test.statistic) else "—",
                        "p-value": format_p_value_display(full_diag.bias_test.p_value) if pd.notna(full_diag.bias_test.p_value) else "—",
                        "Result": "✅ PASS" if full_diag.bias_test.passed else "❌ FAIL",
                        "Interpretation": full_diag.bias_test.interpretation,
                    },
                ]

                tests_df = pd.DataFrame(tests_data)
                st.dataframe(tests_df, use_container_width=True, hide_index=True)

                # Issues and recommendations
                if full_diag.issues_found:
                    st.warning(f"⚠️ **Issues Found:** {', '.join(full_diag.issues_found)}")

                if full_diag.recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in full_diag.recommendations:
                        st.markdown(f"- {rec}")

                st.divider()

                # Section 3: Residual Statistics
                st.markdown("#### 📊 Residual Statistics")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{full_diag.mean:.4f}", help="Should be close to 0 for unbiased model")
                with col2:
                    st.metric("Std Dev", f"{full_diag.std:.4f}")
                with col3:
                    st.metric("Skewness", f"{full_diag.skewness:.3f}", help="Should be close to 0 for normal distribution")
                with col4:
                    st.metric("Kurtosis", f"{full_diag.kurtosis:.3f}", help="Excess kurtosis; 0 for normal distribution")

                # ACF significant lags
                if full_diag.acf_result.significant_lags:
                    st.warning(f"⚠️ Significant autocorrelation at lags: {full_diag.acf_result.significant_lags}")
                else:
                    st.success("✅ No significant autocorrelation in residuals")

            else:
                st.warning(f"No residual data available for {diag_model} at horizon {diag_horizon}. Please train this model first.")
        else:
            st.info("Train models to see residual diagnostics.")

# -----------------------------------------------------------------------------
# NOTE: Legacy tabs (Experiments, Hyperparameters) removed in PhD restructure
# These features are now consolidated in Tab 6: Export & Reports
# -----------------------------------------------------------------------------

# [REMOVED: ~650 lines of legacy tab_experiments and tab_hyperparams code]

    # --- Section 1: Current Dataset Info ---
    st.markdown("#### Current Dataset Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Get current feature engineering and selection info from session state
        fe_state = st.session_state.get("feature_engineering", {})
        fs_state = st.session_state.get("feature_selection", {})

        # Extract method name from results
        fs_results = fs_state.get("results", {})
        current_method = "Unknown"
        current_features = []

        # Try to get the active method index
        method_idx = st.session_state.get("fs_selected_method_idx", 0)

        # Get method names from results
        method_names = {
            0: "Baseline (All Features)",
            1: "Permutation Importance",
            2: "Lasso Regularization",
            3: "Gradient Boosting"
        }

        if fs_results:
            for method_key, method_data in fs_results.items():
                if isinstance(method_data, dict) and method_data.get("method_name"):
                    current_method = method_data.get("method_name")
                    break

        current_method = method_names.get(method_idx, current_method)

        # Get feature count from selected features
        selected_features = fs_state.get("selected_features", {})
        if selected_features:
            # Get features for the current method index
            for key, features in selected_features.items():
                if isinstance(features, list):
                    current_features = features
                    break

        feature_count = len(current_features) if current_features else 0

        # Get engineering variant
        fe_summary = fe_state.get("summary", {})
        current_variant = fe_summary.get("variant_used", "Unknown")

        st.info(f"""
        **Current Dataset Configuration:**
        - Feature Selection Method: **{current_method}**
        - Engineering Variant: **{current_variant}**
        - Feature Count: **{feature_count}**
        """)

    with col2:
        # Generate dataset ID
        if feature_count > 0:
            dataset_id = ExperimentResultsService.generate_dataset_id(
                method=current_method,
                date=datetime.now(),
                feature_count=feature_count
            )
            st.code(f"Dataset ID: {dataset_id}", language=None)
        else:
            dataset_id = None
            st.warning("No features selected. Please run Feature Selection first.")

    st.divider()

    # --- Section 2: Save Current Results ---
    st.markdown("#### Save Current Model Results to Experiment Database")

    # Get available models with results
    available_models_to_save = []
    for model_name in ["XGBoost", "LSTM", "ANN", "ARIMA", "SARIMAX"]:
        if model_name in ["ARIMA", "SARIMAX"]:
            results_key = f"{model_name.lower()}_mh_results" if model_name == "ARIMA" else f"{model_name.lower()}_results"
            has_results = st.session_state.get(results_key) is not None
        else:
            # Check both lowercase and titlecase keys for ML models
            has_results = (
                st.session_state.get(f"ml_mh_results_{model_name.lower()}") is not None or
                st.session_state.get(f"ml_mh_results_{model_name}") is not None
            )

        if has_results:
            available_models_to_save.append(model_name)

    if available_models_to_save and dataset_id:
        models_to_save = st.multiselect(
            "Select models to save:",
            options=available_models_to_save,
            default=available_models_to_save,
            key="exp_models_to_save"
        )

        notes = st.text_area(
            "Experiment Notes (optional):",
            placeholder="Add any notes about this experiment...",
            key="exp_notes"
        )

        col_save1, col_save2 = st.columns([1, 3])

        with col_save1:
            if st.button("💾 Save to Experiment Database", type="primary", use_container_width=True):
                saved_count = 0
                error_count = 0

                progress_bar = st.progress(0)

                for i, model_name in enumerate(models_to_save):
                    try:
                        # Get model results
                        if model_name in ["ARIMA", "SARIMAX"]:
                            results_key = "arima_mh_results" if model_name == "ARIMA" else "sarimax_results"
                            ml_results = st.session_state.get(results_key)
                        else:
                            # Check both lowercase and titlecase keys for ML models
                            ml_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")
                            if ml_results is None:
                                ml_results = st.session_state.get(f"ml_mh_results_{model_name}")

                        if ml_results:
                            results_df = ml_results.get("results_df")

                            if results_df is not None and not results_df.empty:
                                # Extract metrics per horizon
                                horizon_metrics = []
                                predictions = {}

                                for _, row in results_df.iterrows():
                                    horizon = int(row.get("Horizon", 1))
                                    horizon_metrics.append({
                                        "horizon": horizon,
                                        "mae": float(row.get("Test_MAE", row.get("MAE", 0))),
                                        "rmse": float(row.get("Test_RMSE", row.get("RMSE", 0))),
                                        "mape": float(row.get("Test_MAPE", row.get("MAPE_%", 0))),
                                        "accuracy": float(row.get("Test_Acc", row.get("Accuracy_%", 0))),
                                    })

                                # Get predictions from per_h data
                                per_h = ml_results.get("per_h", {})
                                for h, h_data in per_h.items():
                                    y_test = h_data.get("y_test")
                                    y_pred = h_data.get("y_test_pred", h_data.get("y_pred"))

                                    if y_test is not None and y_pred is not None:
                                        predictions[h] = {
                                            "actual": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                                            "predicted": y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                                        }

                                # Create experiment record
                                record = ExperimentRecord(
                                    dataset_id=dataset_id,
                                    feature_selection_method=current_method,
                                    feature_engineering_variant=current_variant,
                                    feature_count=feature_count,
                                    feature_names=current_features[:50] if current_features else [],  # Limit to 50 features
                                    model_name=model_name,
                                    model_category=get_model_category(model_name),
                                    model_params=ml_results.get("params", {}),
                                    horizon_metrics=horizon_metrics,
                                    avg_mae=float(results_df["Test_MAE"].mean()) if "Test_MAE" in results_df.columns else float(results_df.get("MAE", pd.Series([0])).mean()),
                                    avg_rmse=float(results_df["Test_RMSE"].mean()) if "Test_RMSE" in results_df.columns else float(results_df.get("RMSE", pd.Series([0])).mean()),
                                    avg_mape=float(results_df["Test_MAPE"].mean()) if "Test_MAPE" in results_df.columns else float(results_df.get("MAPE_%", pd.Series([0])).mean()),
                                    avg_accuracy=float(results_df["Test_Acc"].mean()) if "Test_Acc" in results_df.columns else float(results_df.get("Accuracy_%", pd.Series([0])).mean()),
                                    avg_r2=None,
                                    runtime_seconds=float(ml_results.get("runtime_s", 0)),
                                    predictions=predictions,
                                    trained_at=datetime.now(),
                                    horizons_trained=ml_results.get("successful", list(range(1, 8))),
                                    train_size=fe_summary.get("train_size", 0),
                                    test_size=fe_summary.get("test_size", 0),
                                    notes=notes if notes else None,
                                )

                                success, msg = exp_service.save_experiment(record)

                                if success:
                                    saved_count += 1
                                else:
                                    error_count += 1

                    except Exception as e:
                        error_count += 1
                        st.error(f"Error saving {model_name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(models_to_save))

                progress_bar.empty()

                if saved_count > 0:
                    st.success(f"✅ Successfully saved {saved_count} model(s) to experiment database!")
                if error_count > 0:
                    st.warning(f"⚠️ Failed to save {error_count} model(s)")

    elif not available_models_to_save:
        st.info("No trained models available to save. Please train models from the Modelling Studio first.")
    elif not dataset_id:
        st.info("Please configure Feature Engineering and Feature Selection before saving experiments.")

    st.divider()

    # --- Section 3: Experiment History Table ---
    st.markdown("#### Experiment History")

    if exp_service.is_connected():
        experiments_df = exp_service.list_experiments()

        if not experiments_df.empty:
            # Format the dataframe for display
            display_cols = ["dataset_id", "model_name", "feature_selection_method",
                           "feature_count", "avg_rmse", "avg_accuracy", "trained_at"]
            available_display_cols = [c for c in display_cols if c in experiments_df.columns]

            st.dataframe(
                experiments_df[available_display_cols].style.format({
                    "avg_rmse": "{:.4f}",
                    "avg_accuracy": "{:.2f}%",
                    "trained_at": lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "—"
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

            # Delete functionality
            with st.expander("🗑️ Delete Experiments"):
                unique_datasets = experiments_df["dataset_id"].unique().tolist()
                unique_models = experiments_df["model_name"].unique().tolist()

                del_col1, del_col2 = st.columns(2)
                with del_col1:
                    del_dataset = st.selectbox("Select Dataset:", [""] + unique_datasets, key="del_dataset")
                with del_col2:
                    del_model = st.selectbox("Select Model:", [""] + unique_models, key="del_model")

                if del_dataset and del_model:
                    if st.button("🗑️ Delete Selected Experiment", type="secondary"):
                        if exp_service.delete_experiment(del_dataset, del_model):
                            st.success(f"Deleted {del_model} for dataset {del_dataset}")
                            st.rerun()
                        else:
                            st.error("Failed to delete experiment")

        else:
            st.info("No experiments saved yet. Train some models and save them to the database.")
    else:
        st.warning("⚠️ Supabase not connected. Experiment tracking is unavailable.")

    st.divider()

    # --- Section 4: Dataset Comparison Charts ---
    st.markdown("#### Performance Variation Across Datasets")

    if exp_service.is_connected():
        experiments_df = exp_service.list_experiments()

        if not experiments_df.empty:
            unique_models = experiments_df["model_name"].unique().tolist()

            selected_model_compare = st.selectbox(
                "Select model to compare across datasets:",
                options=unique_models,
                key="exp_compare_model"
            )

            if selected_model_compare:
                comparison_df = exp_service.compare_across_datasets(selected_model_compare)

                if not comparison_df.empty:
                    # Create visualizations
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        # Histogram: RMSE distribution
                        if "avg_rmse" in comparison_df.columns:
                            fig_hist = px.histogram(
                                comparison_df,
                                x="avg_rmse",
                                color="feature_selection_method" if "feature_selection_method" in comparison_df.columns else None,
                                title=f"{selected_model_compare}: RMSE Distribution Across Datasets",
                                nbins=15,
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_hist.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=TEXT_COLOR),
                                height=350
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                    with viz_col2:
                        # Histogram: Accuracy distribution
                        if "avg_accuracy" in comparison_df.columns:
                            fig_acc_hist = px.histogram(
                                comparison_df,
                                x="avg_accuracy",
                                color="feature_selection_method" if "feature_selection_method" in comparison_df.columns else None,
                                title=f"{selected_model_compare}: Accuracy Distribution",
                                nbins=15,
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_acc_hist.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=TEXT_COLOR),
                                height=350
                            )
                            st.plotly_chart(fig_acc_hist, use_container_width=True)

                    # Bar Chart: Metrics by dataset
                    st.markdown("##### Metrics by Dataset")

                    if len(comparison_df) > 0:
                        metrics_to_plot = ["avg_mae", "avg_rmse", "avg_mape"]
                        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

                        if available_metrics:
                            # Melt for grouped bar chart
                            melted_df = comparison_df.melt(
                                id_vars=["dataset_id"],
                                value_vars=available_metrics,
                                var_name="Metric",
                                value_name="Value"
                            )

                            fig_bar = px.bar(
                                melted_df,
                                x="dataset_id",
                                y="Value",
                                color="Metric",
                                barmode="group",
                                title=f"{selected_model_compare}: Error Metrics by Dataset"
                            )
                            fig_bar.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=TEXT_COLOR),
                                xaxis_tickangle=-45,
                                height=400
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    # Line Chart: Performance trend over time
                    st.markdown("##### Performance Trend Over Time")

                    if "trained_at" in comparison_df.columns and "avg_accuracy" in comparison_df.columns:
                        sorted_df = comparison_df.sort_values("trained_at")

                        fig_trend = px.line(
                            sorted_df,
                            x="trained_at",
                            y="avg_accuracy",
                            markers=True,
                            title=f"{selected_model_compare}: Accuracy Trend Over Experiments"
                        )
                        fig_trend.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=TEXT_COLOR),
                            height=350
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                    # Feature selection method comparison
                    st.markdown("##### Performance by Feature Selection Method")

                    if "feature_selection_method" in comparison_df.columns:
                        method_stats = comparison_df.groupby("feature_selection_method").agg({
                            "avg_rmse": ["mean", "std", "min", "max"],
                            "avg_accuracy": ["mean", "std", "min", "max"],
                        }).round(4)

                        # Flatten column names
                        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]
                        method_stats = method_stats.reset_index()

                        st.dataframe(method_stats, use_container_width=True, hide_index=True)

                else:
                    st.info(f"No experiments found for {selected_model_compare}.")
        else:
            st.info("No experiments in database. Save some model results first.")
    else:
        st.warning("⚠️ Supabase not connected.")

# -----------------------------------------------------------------------------
# DISABLED: OLD TAB - HYPERPARAMETER TUNING RESULTS
# (Moved to a collapsible section in Export tab for cleaner 6-tab structure)
# -----------------------------------------------------------------------------
if False:  # Disabled - old tab_hyperparams
    st.markdown("### 🔬 Hyperparameter Optimization Results")
    st.caption("Results from Grid Search, Random Search, and Bayesian Optimization")

    # Collect all hyperparameter optimization results
    opt_models = []
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        opt_key = f"opt_results_{model_type}"
        if opt_key in st.session_state and st.session_state[opt_key] is not None:
            opt_models.append(model_type)

    if not opt_models:
        st.warning(
            "⚠️ **No hyperparameter optimization results found**\n\n"
            "Please run hyperparameter tuning from:\n"
            "- **08_Modeling_Hub.py** → 🔬 Hyperparameter Tuning tab\n\n"
            "After optimization, return here to view and compare results."
        )

        # Show available session keys for debugging
        with st.expander("📂 Debug: Available Session Keys", expanded=False):
            opt_keys = [k for k in st.session_state.keys() if 'opt' in k.lower()]
            st.write("Optimization-related keys:", opt_keys)
    else:
        # Summary metrics
        st.markdown("#### 📊 Optimization Summary")

        summary_data = []
        for model_type in opt_models:
            opt_results = st.session_state.get(f"opt_results_{model_type}")
            if opt_results:
                all_scores = opt_results.get("all_scores", {})
                summary_data.append({
                    "Model": model_type,
                    "Method": st.session_state.get("opt_last_method", "Unknown"),
                    "RMSE": _safe_float(all_scores.get("RMSE")),
                    "MAE": _safe_float(all_scores.get("MAE")),
                    "MAPE_%": _safe_float(all_scores.get("MAPE")),
                    "Accuracy_%": _safe_float(all_scores.get("Accuracy")),
                    "Refit Metric": opt_results.get("refit_metric", "N/A"),
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Display summary table with color coding
            st.dataframe(
                summary_df.style.format({
                    "RMSE": "{:.4f}",
                    "MAE": "{:.4f}",
                    "MAPE_%": "{:.2f}",
                    "Accuracy_%": "{:.2f}",
                }, na_rep="—").background_gradient(
                    subset=["RMSE", "MAE", "MAPE_%"],
                    cmap="RdYlGn_r"
                ).background_gradient(
                    subset=["Accuracy_%"],
                    cmap="RdYlGn"
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Find best optimized model
            if "Accuracy_%" in summary_df.columns:
                best_idx = summary_df["Accuracy_%"].idxmax()
                best_opt_model = summary_df.loc[best_idx, "Model"]
                best_acc = summary_df.loc[best_idx, "Accuracy_%"]
                st.success(f"🏆 **Best Optimized Model: {best_opt_model}** with {best_acc:.2f}% accuracy")

        st.divider()

        # Detailed results per model
        st.markdown("#### 🔍 Detailed Results by Model")

        selected_opt_model = st.selectbox(
            "Select Model:",
            options=opt_models,
            key="results_opt_model_select"
        )

        if selected_opt_model:
            opt_results = st.session_state.get(f"opt_results_{selected_opt_model}")

            if opt_results:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 🏆 Best Parameters")
                    best_params = opt_results.get("best_params", {})
                    if best_params:
                        params_df = pd.DataFrame(
                            list(best_params.items()),
                            columns=["Parameter", "Value"]
                        )
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No parameters available")

                with col2:
                    st.markdown("##### 📈 Best Scores")
                    all_scores = opt_results.get("all_scores", {})
                    if all_scores:
                        scores_df = pd.DataFrame([
                            {"Metric": "RMSE", "Value": f"{all_scores.get('RMSE', 'N/A'):.4f}" if isinstance(all_scores.get('RMSE'), (int, float)) else "N/A"},
                            {"Metric": "MAE", "Value": f"{all_scores.get('MAE', 'N/A'):.4f}" if isinstance(all_scores.get('MAE'), (int, float)) else "N/A"},
                            {"Metric": "MAPE (%)", "Value": f"{all_scores.get('MAPE', 'N/A'):.2f}" if isinstance(all_scores.get('MAPE'), (int, float)) else "N/A"},
                            {"Metric": "Accuracy (%)", "Value": f"{all_scores.get('Accuracy', 'N/A'):.2f}" if isinstance(all_scores.get('Accuracy'), (int, float)) else "N/A"},
                        ])
                        st.dataframe(scores_df, use_container_width=True, hide_index=True)

                # CV Results
                cv_results = opt_results.get("cv_results")
                if cv_results is not None and isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
                    with st.expander("📋 Cross-Validation Results (All Parameter Combinations)", expanded=False):
                        # Select relevant columns
                        display_cols = ['params', 'mean_test_RMSE', 'mean_test_MAE', 'mean_test_MAPE', 'mean_test_Accuracy']
                        available_cols = [col for col in display_cols if col in cv_results.columns]

                        if available_cols:
                            display_cv = cv_results[available_cols].copy()

                            # Invert negative scores for display
                            for col in ['mean_test_RMSE', 'mean_test_MAE', 'mean_test_MAPE']:
                                if col in display_cv.columns:
                                    display_cv[col] = -display_cv[col]

                            st.dataframe(display_cv, use_container_width=True, height=300)

                # Backtesting results
                backtest_key = f"backtest_results_{selected_opt_model}"
                backtest_results = st.session_state.get(backtest_key)

                if backtest_results:
                    st.markdown("##### 📅 7-Day Rolling-Origin Backtesting")

                    aggregate_metrics = backtest_results.get("aggregate_metrics", {})
                    n_windows = backtest_results.get("n_successful_windows", 0)

                    if aggregate_metrics.get("RMSE") is not None:
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("RMSE", f"{aggregate_metrics['RMSE']:.4f}")
                        with metric_cols[1]:
                            st.metric("MAE", f"{aggregate_metrics['MAE']:.4f}")
                        with metric_cols[2]:
                            st.metric("MAPE", f"{aggregate_metrics['MAPE']:.2f}%")
                        with metric_cols[3]:
                            st.metric("Accuracy", f"{aggregate_metrics['Accuracy']:.2f}%")

                        st.caption(f"Aggregated across {n_windows} rolling windows")

                        # Window-by-window results
                        window_results = backtest_results.get("window_results", [])
                        if window_results:
                            with st.expander(f"📋 Individual Window Results ({len(window_results)} windows)", expanded=False):
                                window_df = pd.DataFrame(window_results)
                                st.dataframe(window_df, use_container_width=True, hide_index=True)

        # Comparison chart if multiple models optimized
        if len(opt_models) > 1:
            st.divider()
            st.markdown("#### 📊 Optimized Models Comparison")

            # Create comparison bar chart
            comp_data = []
            for model_type in opt_models:
                opt_results = st.session_state.get(f"opt_results_{model_type}")
                if opt_results:
                    all_scores = opt_results.get("all_scores", {})
                    comp_data.append({
                        "Model": model_type,
                        "Accuracy": _safe_float(all_scores.get("Accuracy")),
                        "RMSE": _safe_float(all_scores.get("RMSE")),
                    })

            if comp_data:
                comp_df = pd.DataFrame(comp_data)

                fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy (%)", "RMSE"))

                # Accuracy bars
                fig_comp.add_trace(
                    go.Bar(
                        x=comp_df["Model"],
                        y=comp_df["Accuracy"],
                        marker_color=[MODEL_COLORS.get(m, "#6B7280") for m in comp_df["Model"]],
                        text=[f"{v:.1f}%" for v in comp_df["Accuracy"]],
                        textposition="auto",
                        showlegend=False,
                    ),
                    row=1, col=1
                )

                # RMSE bars
                fig_comp.add_trace(
                    go.Bar(
                        x=comp_df["Model"],
                        y=comp_df["RMSE"],
                        marker_color=[MODEL_COLORS.get(m, "#6B7280") for m in comp_df["Model"]],
                        text=[f"{v:.4f}" for v in comp_df["RMSE"]],
                        textposition="auto",
                        showlegend=False,
                    ),
                    row=1, col=2
                )

                fig_comp.update_layout(
                    title="Hyperparameter-Optimized Model Comparison",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                )

                st.plotly_chart(fig_comp, use_container_width=True)

        # Export optimized parameters
        st.divider()
        st.markdown("#### 📥 Export Best Parameters")

        export_model = st.selectbox(
            "Select Model to Export Parameters:",
            options=opt_models,
            key="export_opt_params_select"
        )

        if export_model:
            opt_results = st.session_state.get(f"opt_results_{export_model}")
            if opt_results:
                best_params = opt_results.get("best_params", {})
                all_scores = opt_results.get("all_scores", {})

                export_data = {
                    "model": export_model,
                    "method": st.session_state.get("opt_last_method", "Unknown"),
                    "best_params": best_params,
                    "best_scores": all_scores,
                }

                import json
                json_str = json.dumps(export_data, indent=2, default=str)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Parameters (JSON)",
                        json_str,
                        f"best_params_{export_model}.json",
                        "application/json",
                        use_container_width=True,
                    )

                with col2:
                    with st.expander("👁️ Preview JSON"):
                        st.json(export_data)

# -----------------------------------------------------------------------------
# TAB 6: MODEL EXPLAINABILITY (SHAP & LIME)
# -----------------------------------------------------------------------------
with tab_explainability:
    st.markdown("### 🔍 Model Explainability")
    st.caption("SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)")

    # Check if SHAP/LIME are available
    if not SHAP_AVAILABLE and not LIME_AVAILABLE:
        st.error(
            "⚠️ **SHAP and LIME libraries are not installed.**\n\n"
            "Please install them using:\n"
            "```\npip install shap lime\n```"
        )
        st.stop()

    # Check for available ML models with explainability data
    available_models = []
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        # Check both lowercase and titlecase keys
        ml_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")
        if ml_results is None:
            ml_results = st.session_state.get(f"ml_mh_results_{model_name}")
        if ml_results and ml_results.get("pipelines"):
            available_models.append(model_name)

    # Also check single model results
    ml_mh = st.session_state.get("ml_mh_results")
    if ml_mh and ml_mh.get("pipelines"):
        model_type = ml_mh.get("results_df", pd.DataFrame()).get("Model", pd.Series()).iloc[0] if not ml_mh.get("results_df", pd.DataFrame()).empty else "ML Model"
        if model_type not in available_models:
            available_models.append(model_type)

    if not available_models:
        st.warning(
            "⚠️ **No ML models with explainability data found.**\n\n"
            "Please train ML models (XGBoost, LSTM, or ANN) from the **Train Models** page first.\n\n"
            "The models need to be re-trained after installing SHAP/LIME to capture the required data."
        )
    else:
        # Model selection
        col1, col2 = st.columns(2)

        with col1:
            selected_model = st.selectbox(
                "Select Model:",
                options=available_models,
                key="explainability_model_select"
            )

        # Get the selected model's results
        if selected_model in ["XGBoost", "LSTM", "ANN"]:
            # Check both lowercase and titlecase keys
            ml_results = st.session_state.get(f"ml_mh_results_{selected_model.lower()}")
            if ml_results is None:
                ml_results = st.session_state.get(f"ml_mh_results_{selected_model}")
            if ml_results is None:
                ml_results = st.session_state.get("ml_mh_results")
        else:
            ml_results = st.session_state.get("ml_mh_results")

        if ml_results:
            pipelines = ml_results.get("pipelines", {})
            explainability_data = ml_results.get("explainability_data", {})
            successful = ml_results.get("successful", [])

            with col2:
                if successful:
                    selected_horizon = st.selectbox(
                        "Select Horizon:",
                        options=successful,
                        format_func=lambda x: f"Horizon {x} (Day {x})",
                        key="explainability_horizon_select"
                    )
                else:
                    st.warning("No successful horizons found.")
                    selected_horizon = None

            if selected_horizon and selected_horizon in pipelines:
                pipeline = pipelines[selected_horizon]
                exp_data = explainability_data.get(selected_horizon, {})
                X_train = exp_data.get("X_train")
                X_val = exp_data.get("X_val")
                feature_names = exp_data.get("feature_names", [])

                if X_val is None or len(feature_names) == 0:
                    st.error("Explainability data not available for this horizon. Please re-train the model.")
                else:
                    st.divider()

                    # SHAP Section
                    if SHAP_AVAILABLE:
                        st.markdown("#### 📊 SHAP Analysis")
                        st.caption("Global feature importance and individual prediction explanations")

                        with st.spinner("Computing SHAP values... (this may take a moment)"):
                            try:
                                # Get the underlying model for SHAP
                                if selected_model == "XGBoost":
                                    # XGBoost uses TreeExplainer (fast)
                                    if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'named_steps'):
                                        model_for_shap = pipeline.model.named_steps.get('model', pipeline.model)
                                    elif hasattr(pipeline, 'model'):
                                        model_for_shap = pipeline.model
                                    else:
                                        model_for_shap = pipeline

                                    explainer = shap.TreeExplainer(model_for_shap)
                                    shap_values = explainer.shap_values(X_val)
                                else:
                                    # LSTM/ANN use KernelExplainer (slower, use sampling)
                                    # Create a prediction function
                                    def predict_fn(X):
                                        return pipeline.predict(X)

                                    # Sample background data for efficiency
                                    background_samples = min(100, len(X_train)) if X_train is not None else 50
                                    background = shap.sample(X_train, background_samples) if X_train is not None else X_val[:50]

                                    explainer = shap.KernelExplainer(predict_fn, background)
                                    # Limit test samples for speed
                                    X_val_sample = X_val[:min(50, len(X_val))]
                                    shap_values = explainer.shap_values(X_val_sample)
                                    X_val = X_val_sample  # Use sampled data for plots

                                # Store SHAP values in session state for caching
                                shap_cache_key = f"shap_values_{selected_model}_{selected_horizon}"
                                st.session_state[shap_cache_key] = {
                                    "shap_values": shap_values,
                                    "X_val": X_val,
                                    "feature_names": feature_names,
                                }

                                # SHAP Summary Plot
                                st.markdown("##### Feature Importance (Summary Plot)")

                                fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(
                                    shap_values,
                                    X_val,
                                    feature_names=feature_names,
                                    show=False,
                                    plot_size=None
                                )
                                plt.tight_layout()

                                # Convert matplotlib to streamlit
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                                           facecolor='#0f172a', edgecolor='none')
                                buf.seek(0)
                                st.image(buf, use_container_width=True)
                                plt.close()

                                # SHAP Bar Plot (Mean absolute SHAP values)
                                st.markdown("##### Mean Feature Importance (Bar Plot)")

                                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(
                                    shap_values,
                                    X_val,
                                    feature_names=feature_names,
                                    plot_type="bar",
                                    show=False,
                                    plot_size=None
                                )
                                plt.tight_layout()

                                buf2 = io.BytesIO()
                                plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight',
                                           facecolor='#0f172a', edgecolor='none')
                                buf2.seek(0)
                                st.image(buf2, use_container_width=True)
                                plt.close()

                                # Individual prediction explanation
                                with st.expander("🔍 Explain Individual Prediction", expanded=False):
                                    pred_idx = st.slider(
                                        "Select prediction to explain:",
                                        min_value=0,
                                        max_value=len(X_val) - 1,
                                        value=0,
                                        key="shap_pred_idx"
                                    )

                                    st.markdown(f"**Explaining prediction #{pred_idx}**")

                                    # Force plot for single prediction
                                    if hasattr(explainer, 'expected_value'):
                                        expected_value = explainer.expected_value
                                        if isinstance(expected_value, np.ndarray):
                                            expected_value = expected_value[0]

                                        # Create force plot
                                        force_plot = shap.force_plot(
                                            expected_value,
                                            shap_values[pred_idx],
                                            X_val[pred_idx],
                                            feature_names=feature_names,
                                            matplotlib=True,
                                            show=False
                                        )
                                        plt.tight_layout()

                                        buf3 = io.BytesIO()
                                        plt.savefig(buf3, format='png', dpi=150, bbox_inches='tight',
                                                   facecolor='white', edgecolor='none')
                                        buf3.seek(0)
                                        st.image(buf3, use_container_width=True)
                                        plt.close()

                                    # Feature contributions table
                                    contrib_df = pd.DataFrame({
                                        "Feature": feature_names,
                                        "Value": X_val[pred_idx],
                                        "SHAP Value": shap_values[pred_idx]
                                    }).sort_values("SHAP Value", key=abs, ascending=False)

                                    st.dataframe(
                                        contrib_df.style.format({"Value": "{:.4f}", "SHAP Value": "{:.4f}"}).background_gradient(
                                            subset=["SHAP Value"],
                                            cmap="RdYlGn",
                                            vmin=-abs(shap_values[pred_idx]).max(),
                                            vmax=abs(shap_values[pred_idx]).max()
                                        ),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                            except Exception as e:
                                st.error(f"Error computing SHAP values: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                    else:
                        st.info("📊 **SHAP not available.** Install with: `pip install shap`")

                    st.divider()

                    # LIME Section
                    if LIME_AVAILABLE:
                        st.markdown("#### 🍋 LIME Analysis")
                        st.caption("Local Interpretable Model-agnostic Explanations for individual predictions")

                        with st.expander("🔍 Generate LIME Explanation", expanded=True):
                            lime_pred_idx = st.slider(
                                "Select prediction to explain with LIME:",
                                min_value=0,
                                max_value=len(X_val) - 1,
                                value=0,
                                key="lime_pred_idx"
                            )

                            if st.button("Generate LIME Explanation", key="lime_generate_btn"):
                                with st.spinner("Generating LIME explanation..."):
                                    try:
                                        # Create LIME explainer
                                        lime_explainer = LimeTabularExplainer(
                                            X_train if X_train is not None else X_val,
                                            feature_names=feature_names,
                                            mode='regression',
                                            verbose=False
                                        )

                                        # Create prediction function
                                        def predict_fn(X):
                                            return pipeline.predict(X)

                                        # Generate explanation
                                        explanation = lime_explainer.explain_instance(
                                            X_val[lime_pred_idx],
                                            predict_fn,
                                            num_features=min(10, len(feature_names))
                                        )

                                        # Display as bar chart using Plotly
                                        exp_list = explanation.as_list()
                                        exp_df = pd.DataFrame(exp_list, columns=["Feature", "Contribution"])
                                        exp_df = exp_df.sort_values("Contribution", key=abs, ascending=True)

                                        # Create horizontal bar chart
                                        colors = [SUCCESS_COLOR if c > 0 else DANGER_COLOR for c in exp_df["Contribution"]]

                                        fig_lime = go.Figure(go.Bar(
                                            x=exp_df["Contribution"],
                                            y=exp_df["Feature"],
                                            orientation='h',
                                            marker_color=colors,
                                            text=[f"{c:.4f}" for c in exp_df["Contribution"]],
                                            textposition="auto"
                                        ))

                                        fig_lime.update_layout(
                                            title=f"LIME Explanation for Prediction #{lime_pred_idx}",
                                            xaxis_title="Feature Contribution",
                                            yaxis_title="Feature",
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            height=400,
                                            font=dict(color=TEXT_COLOR)
                                        )

                                        st.plotly_chart(fig_lime, use_container_width=True)

                                        # Show feature values
                                        st.markdown("**Feature Values for this Prediction:**")
                                        feature_vals_df = pd.DataFrame({
                                            "Feature": feature_names,
                                            "Value": X_val[lime_pred_idx]
                                        })
                                        st.dataframe(
                                            feature_vals_df.style.format({"Value": "{:.4f}"}),
                                            use_container_width=True,
                                            hide_index=True
                                        )

                                    except Exception as e:
                                        st.error(f"Error generating LIME explanation: {str(e)}")
                                        import traceback
                                        with st.expander("Error Details"):
                                            st.code(traceback.format_exc())
                    else:
                        st.info("🍋 **LIME not available.** Install with: `pip install lime`")

            elif selected_horizon:
                st.warning(f"No pipeline data found for Horizon {selected_horizon}. Please re-train the model.")

# -----------------------------------------------------------------------------
# TAB 8: EXPORT (Enhanced)
# -----------------------------------------------------------------------------
with tab_export:
    st.markdown("### Export Results")
    st.caption("Export model results to CSV, Markdown, or LaTeX for thesis and reporting")

    # Initialize experiment service
    export_exp_service = get_experiment_service()

    # --- Section 1: Export Scope Selection ---
    st.markdown("#### Export Scope")

    export_scope = st.radio(
        "Select what to export:",
        ["Current Session Results", "All Experiment History", "Filtered Experiments"],
        horizontal=True,
        key="export_scope_radio"
    )

    # --- Section 2: Filters (if Filtered Experiments selected) ---
    if export_scope == "Filtered Experiments":
        st.markdown("##### Filter Options")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Get unique models from experiments
            if export_exp_service.is_connected():
                all_models_list = export_exp_service.get_unique_values("model_name")
            else:
                all_models_list = []
            filter_models = st.multiselect(
                "Filter by Model:",
                options=all_models_list,
                key="export_filter_models"
            )

        with filter_col2:
            if export_exp_service.is_connected():
                all_methods_list = export_exp_service.get_unique_values("feature_selection_method")
            else:
                all_methods_list = []
            filter_methods = st.multiselect(
                "Filter by Feature Selection Method:",
                options=all_methods_list,
                key="export_filter_methods"
            )

        with filter_col3:
            if export_exp_service.is_connected():
                all_datasets_list = export_exp_service.get_unique_values("dataset_id")
            else:
                all_datasets_list = []
            filter_datasets = st.multiselect(
                "Filter by Dataset ID:",
                options=all_datasets_list,
                key="export_filter_datasets"
            )

    st.divider()

    # --- Section 3: Generate Export ---
    st.markdown("#### Generate Export Files")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.markdown("##### CSV Exports")

        # Current Session Results
        if export_scope == "Current Session Results":
            if not all_models_df.empty:
                csv_data = all_models_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Current Results (CSV)",
                    csv_data,
                    f"model_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                )

                # Also offer rankings
                ranked_df = compute_model_rankings(all_models_df)
                ranked_csv = ranked_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Rankings (CSV)",
                    ranked_csv,
                    f"model_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No current session results to export.")

        # All Experiment History
        elif export_scope == "All Experiment History":
            if export_exp_service.is_connected():
                exp_csv = export_exp_service.export_to_csv()
                if exp_csv:
                    st.download_button(
                        "📥 Download All Experiments (CSV)",
                        exp_csv,
                        f"all_experiments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No experiments in database to export.")
            else:
                st.warning("Supabase not connected.")

        # Filtered Experiments
        else:
            if export_exp_service.is_connected():
                filters = {}
                if filter_models:
                    filters["model_name"] = filter_models
                if filter_methods:
                    filters["feature_selection_method"] = filter_methods

                filtered_csv = export_exp_service.export_to_csv(filters if filters else None)
                if filtered_csv:
                    st.download_button(
                        "📥 Download Filtered Experiments (CSV)",
                        filtered_csv,
                        f"filtered_experiments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No experiments match the selected filters.")
            else:
                st.warning("Supabase not connected.")

    with export_col2:
        st.markdown("##### Thesis Exports")

        # Thesis summary
        thesis_summary = generate_thesis_summary(all_models_df, best_model, analysis)

        st.download_button(
            "📥 Download Thesis Summary (Markdown)",
            thesis_summary,
            f"thesis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            "text/markdown",
            use_container_width=True,
        )

        # LaTeX table
        latex_table = _generate_latex_table(all_models_df)
        st.download_button(
            "📥 Download LaTeX Table",
            latex_table,
            f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.tex",
            "text/plain",
            use_container_width=True,
        )

    st.divider()

    # --- Section 4: Upload to Supabase ---
    st.markdown("#### Import Experiments from CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV file to import experiments:",
        type=["csv"],
        key="exp_csv_upload"
    )

    if uploaded_file:
        try:
            import_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(import_df)} rows")

            # Preview
            with st.expander("Preview Data"):
                st.dataframe(import_df.head(10), use_container_width=True)

            # Required columns check
            required_cols = ["dataset_id", "model_name"]
            missing_cols = [c for c in required_cols if c not in import_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("📤 Import to Supabase", type="primary"):
                    if export_exp_service.is_connected():
                        success_count = export_exp_service.bulk_import(import_df)
                        if success_count > 0:
                            st.success(f"✅ Successfully imported {success_count} experiments!")
                        else:
                            st.warning("No experiments were imported. Check the data format.")
                    else:
                        st.error("Supabase not connected. Cannot import.")

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    st.divider()

    # --- Section 5: Thesis Summary Preview ---
    st.markdown("#### Thesis Summary Preview")

    with st.expander("📄 View Thesis Summary", expanded=False):
        st.markdown(thesis_summary)

    # --- Section 6: LaTeX Table Preview ---
    st.markdown("#### LaTeX Table Preview")

    with st.expander("📋 View LaTeX Code", expanded=False):
        st.code(latex_table, language="latex")


# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(8)  # Model Results is page index 8

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "💡 **Tip:** Train models from Benchmarks (ARIMA, SARIMAX) and Modeling Hub (XGBoost, LSTM, ANN) "
    "pages, then return here for comprehensive comparison and thesis-quality exports."
)
