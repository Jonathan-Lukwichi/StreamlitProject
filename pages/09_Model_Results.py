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
    if "arima" in name_lower and "sarimax" not in name_lower:
        return "ARIMA"
    elif "sarimax" in name_lower:
        return "SARIMAX"
    elif "xgboost" in name_lower or "xgb" in name_lower:
        return "XGBoost"
    elif "lstm" in name_lower:
        return "LSTM"
    elif "ann" in name_lower or "neural" in name_lower or "mlp" in name_lower:
        return "ANN"
    elif "ensemble" in name_lower:
        return "Ensemble"
    elif "stack" in name_lower:
        return "Stacking"
    elif "hybrid" in name_lower:
        return "Hybrid"
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
        ml_results = st.session_state.get(f"ml_mh_results_{model_name}")
        if ml_results:
            metrics = _extract_ml_metrics(ml_results, model_name)
            if metrics:
                all_results.append(metrics)

    # 5. Single ML model results (if only one model trained)
    ml_mh = st.session_state.get("ml_mh_results")
    if ml_mh and not any(st.session_state.get(f"ml_mh_results_{m}") for m in ["XGBoost", "LSTM", "ANN"]):
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

    # 9. Optimized model results
    for model_type in ["XGBoost", "LSTM", "ANN"]:
        opt_results = st.session_state.get(f"opt_results_{model_type}")
        if opt_results:
            metrics = _extract_optimized_metrics(opt_results, model_type)
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

# Main tabs
tab_overview, tab_comparison, tab_horizons, tab_rankings, tab_experiments, tab_hyperparams, tab_explainability, tab_export = st.tabs([
    "📋 Overview", "📊 Comparison", "🗓️ Per-Horizon", "🏅 Rankings", "🧪 Dataset Experiments", "🔬 Hyperparameter Tuning", "🔍 Explainability", "📤 Export"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown("### All Trained Models")

    # Format and display the main comparison table
    display_df = all_models_df.copy()

    # Select columns to display
    display_cols = ["Model", "Category", "Source", "MAE", "RMSE", "MAPE_%", "Accuracy_%", "Runtime_s"]
    display_cols = [c for c in display_cols if c in display_df.columns]

    # Apply formatting
    format_dict = {col: METRIC_FORMATTERS.get(col, "{:.4f}") for col in display_cols if col in METRIC_FORMATTERS}

    st.dataframe(
        display_df[display_cols].style.format(format_dict, na_rep="—").background_gradient(
            subset=[c for c in ["MAE", "RMSE", "MAPE_%"] if c in display_cols],
            cmap="RdYlGn_r"
        ).background_gradient(
            subset=["Accuracy_%"] if "Accuracy_%" in display_cols else [],
            cmap="RdYlGn"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Best model details
    st.markdown("### Best Model Performance Metrics")

    metrics = analysis.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mae_val = metrics.get("MAE", np.nan)
        st.metric(
            "MAE",
            f"{mae_val:.4f}" if np.isfinite(mae_val) else "—",
            help="Mean Absolute Error - average magnitude of errors"
        )

    with col2:
        rmse_val = metrics.get("RMSE", np.nan)
        st.metric(
            "RMSE",
            f"{rmse_val:.4f}" if np.isfinite(rmse_val) else "—",
            help="Root Mean Square Error - penalizes larger errors more"
        )

    with col3:
        mape_val = metrics.get("MAPE_%", np.nan)
        st.metric(
            "MAPE",
            f"{mape_val:.2f}%" if np.isfinite(mape_val) else "—",
            help="Mean Absolute Percentage Error"
        )

    with col4:
        acc_val = metrics.get("Accuracy_%", np.nan)
        st.metric(
            "Accuracy",
            f"{acc_val:.2f}%" if np.isfinite(acc_val) else "—",
            help="Overall prediction accuracy"
        )

# -----------------------------------------------------------------------------
# TAB 2: COMPARISON
# -----------------------------------------------------------------------------
with tab_comparison:
    st.markdown("### Visual Model Comparison")

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
        )

        bar_fig = create_metric_bar_chart(
            all_models_df,
            metric_option,
            f"Model Comparison: {metric_option.replace('_', ' ')}"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Improvement chart
    st.markdown("### Best Model vs Others")
    improvements_df = compute_pairwise_improvements(all_models_df, best_model)

    if not improvements_df.empty:
        improvement_fig = create_improvement_chart(improvements_df, best_model)
        st.plotly_chart(improvement_fig, use_container_width=True)

        # Show improvement table
        with st.expander("📊 Detailed Improvement Percentages"):
            st.dataframe(
                improvements_df.style.format({
                    c: "{:.2f}%" for c in improvements_df.columns if "Improvement" in c
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Train multiple models to see comparative improvements.")

# -----------------------------------------------------------------------------
# TAB 3: PER-HORIZON ANALYSIS
# -----------------------------------------------------------------------------
with tab_horizons:
    st.markdown("### Forecast Horizon Analysis")
    st.caption("How model performance degrades as forecast horizon increases")

    horizon_fig = create_horizon_performance_chart()

    if horizon_fig:
        st.plotly_chart(horizon_fig, use_container_width=True)

        # Horizon-specific metrics table
        st.markdown("#### Detailed Per-Horizon Metrics")

        horizon_source = st.radio(
            "Data source",
            ["All Models", "Statistical (ARIMA/SARIMAX)", "ML (XGBoost/LSTM/ANN)"],
            horizontal=True,
        )

        # Collect per-horizon data based on selection
        horizon_df_data = []

        if horizon_source in ["All Models", "Statistical (ARIMA/SARIMAX)"]:
            for model_key, model_name in [("arima_mh_results", "ARIMA"), ("sarimax_results", "SARIMAX")]:
                results = st.session_state.get(model_key)
                if results:
                    results_df = results.get("results_df")
                    if results_df is not None:
                        for _, row in results_df.iterrows():
                            horizon_df_data.append({
                                "Model": model_name,
                                "Horizon": int(row.get("Horizon", 0)),
                                "MAE": _safe_float(row.get("Test_MAE", row.get("MAE"))),
                                "RMSE": _safe_float(row.get("Test_RMSE", row.get("RMSE"))),
                                "MAPE_%": _safe_float(row.get("Test_MAPE", row.get("MAPE_%"))),
                                "Accuracy_%": _safe_float(row.get("Test_Acc", row.get("Accuracy_%"))),
                            })

        if horizon_source in ["All Models", "ML (XGBoost/LSTM/ANN)"]:
            for model_name in ["XGBoost", "LSTM", "ANN"]:
                ml_results = st.session_state.get(f"ml_mh_results_{model_name}")
                if ml_results:
                    # ML metrics are stored in results_df, not per_h
                    results_df = ml_results.get("results_df")
                    if results_df is not None and not results_df.empty:
                        for _, row in results_df.iterrows():
                            horizon_df_data.append({
                                "Model": model_name,
                                "Horizon": int(row.get("Horizon", 0)),
                                "MAE": _safe_float(row.get("Test_MAE")),
                                "RMSE": _safe_float(row.get("Test_RMSE")),
                                "MAPE_%": _safe_float(row.get("Test_MAPE")),
                                "Accuracy_%": _safe_float(row.get("Test_Acc")),
                            })

        if horizon_df_data:
            horizon_df = pd.DataFrame(horizon_df_data)
            horizon_df = horizon_df.sort_values(["Model", "Horizon"])

            st.dataframe(
                horizon_df.style.format({
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE_%": "{:.2f}",
                    "Accuracy_%": "{:.2f}",
                }, na_rep="—").background_gradient(
                    subset=["MAE", "RMSE", "MAPE_%"],
                    cmap="RdYlGn_r"
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No per-horizon data available. Train models with multi-horizon configuration.")
    else:
        st.info("No per-horizon data available. Train models from Benchmarks or Modeling Hub pages first.")

# -----------------------------------------------------------------------------
# TAB 4: RANKINGS
# -----------------------------------------------------------------------------
with tab_rankings:
    st.markdown("### Comprehensive Model Rankings")

    ranked_df = compute_model_rankings(all_models_df)

    # Ranking heatmap
    heatmap_fig = create_ranking_heatmap(all_models_df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Detailed ranking table
    st.markdown("#### Ranking Details")

    rank_display_cols = ["Model", "Category", "Overall_Rank", "Composite_Score",
                         "MAE_Rank", "RMSE_Rank", "MAPE_Rank", "Acc_Rank"]
    rank_display_cols = [c for c in rank_display_cols if c in ranked_df.columns]

    # Apply medal styling
    def style_rank(val):
        if pd.isna(val):
            return ""
        if val == 1:
            return "background-color: rgba(255, 215, 0, 0.3); font-weight: bold;"
        elif val == 2:
            return "background-color: rgba(192, 192, 192, 0.3);"
        elif val == 3:
            return "background-color: rgba(205, 127, 50, 0.3);"
        return ""

    st.dataframe(
        ranked_df[rank_display_cols].style.format({
            "Composite_Score": "{:.4f}",
            "Overall_Rank": "{:.0f}",
            "MAE_Rank": "{:.0f}",
            "RMSE_Rank": "{:.0f}",
            "MAPE_Rank": "{:.0f}",
            "Acc_Rank": "{:.0f}",
        }, na_rep="—").applymap(
            style_rank,
            subset=[c for c in ["Overall_Rank", "MAE_Rank", "RMSE_Rank", "MAPE_Rank", "Acc_Rank"] if c in rank_display_cols]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Ranking methodology explanation
    with st.expander("📖 Ranking Methodology"):
        st.markdown("""
        **Composite Score Calculation:**

        The composite ranking uses a weighted scoring system designed for healthcare forecasting:

        | Metric | Weight | Rationale |
        |--------|--------|-----------|
        | RMSE | 35% | Primary accuracy measure, penalizes large errors critical in healthcare |
        | MAE | 30% | Robust to outliers, directly interpretable as average error |
        | MAPE | 20% | Relative error, enables comparison across different scales |
        | Accuracy | 15% | Overall prediction accuracy measure |

        **Interpretation:**
        - Lower composite score = better overall performance
        - A model ranking #1 in all metrics would have the lowest possible composite score
        - The ranking accounts for missing metrics by adjusting weights proportionally
        """)

# -----------------------------------------------------------------------------
# TAB 5: DATASET EXPERIMENTS
# -----------------------------------------------------------------------------
with tab_experiments:
    st.markdown("### Dataset-Based Experiment Tracking")
    st.caption("Track and compare model performance across different feature configurations")

    # Initialize experiment service
    exp_service = get_experiment_service()

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
        else:
            results_key = f"ml_mh_results_{model_name}"

        if st.session_state.get(results_key):
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
                        else:
                            results_key = f"ml_mh_results_{model_name}"

                        ml_results = st.session_state.get(results_key)

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
# TAB 5: HYPERPARAMETER TUNING RESULTS
# -----------------------------------------------------------------------------
with tab_hyperparams:
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
            ml_results = st.session_state.get(f"ml_mh_results_{selected_model}")
            if not ml_results:
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
# TAB 7: THESIS EXPORT
# -----------------------------------------------------------------------------
with tab_thesis:
    st.markdown("### Thesis-Quality Summary")
    st.caption("Academic-grade analysis ready for thesis inclusion")

    # Generate summary
    thesis_summary = generate_thesis_summary(all_models_df, best_model, analysis)

    # Display in markdown
    st.markdown(thesis_summary)

    # Export options
    st.markdown("---")
    st.markdown("### Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export summary as markdown
        st.download_button(
            "📥 Download Summary (Markdown)",
            thesis_summary,
            "model_comparison_summary.md",
            "text/markdown",
            use_container_width=True,
        )

    with col2:
        # Export full results as CSV
        csv_data = all_models_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv_data,
            "model_results.csv",
            "text/csv",
            use_container_width=True,
        )

    with col3:
        # Export rankings as CSV
        ranked_csv = compute_model_rankings(all_models_df).to_csv(index=False)
        st.download_button(
            "📥 Download Rankings (CSV)",
            ranked_csv,
            "model_rankings.csv",
            "text/csv",
            use_container_width=True,
        )

    # LaTeX table generation
    st.markdown("### LaTeX Table Export")

    with st.expander("📋 LaTeX Table Code"):
        latex_table = _generate_latex_table(all_models_df)
        st.code(latex_table, language="latex")

        st.download_button(
            "📥 Download LaTeX",
            latex_table,
            "model_comparison_table.tex",
            "text/plain",
        )


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
