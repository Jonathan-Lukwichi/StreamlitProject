# =============================================================================
# 08_Modeling_Hub.py
# Unified Modeling Hub (foundation only)
# - Benchmarks (ARIMA / SARIMAX): Model selection dropdown + Auto/Manual + Train/Test
# - ML (LSTM / ANN / XGBoost): Model selection dropdown ("Choose a ml model") with XGBoost default + Auto/Manual + Train/Test
# - Hybrids (LSTM-XGBoost / LSTM-ANN / LSTM-SARIMAX): Model selection dropdown + Auto/Manual + Train/Test
# Robust defaults (no KeyErrors even with old sessions)
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
from pathlib import Path

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand, render_cache_management
from app_core.cache import save_to_cache, get_cache_manager
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available
from app_core.plots import build_multihorizon_results_dashboard

# =============================================================================
# CONFORMAL PREDICTION IMPORTS (CQR for Prediction Intervals)
# =============================================================================
try:
    from app_core.pipelines import (
        CQRPipeline,
        simple_conformal_intervals,
        calculate_rpiw,
        PredictionInterval,
        RPIWResult,
    )
    CQR_AVAILABLE = True
except ImportError:
    CQR_AVAILABLE = False

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Modeling Hub - HealthForecast AI",
    page_icon="🧠",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()
render_cache_management()

# Results Storage Panel (Supabase persistence)
render_results_storage_panel(
    page_type="modeling_hub",
    page_title="Modeling Hub",
    custom_keys=["ml_model_results", "benchmark_results", "hybrid_results"],
)

# Auto-load saved results if available
auto_load_if_available("modeling_hub")

# Fluorescent effects + Custom Tab Styling
st.markdown(f"""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR MODELING HUB
   ======================================== */

@keyframes float-orb {{
    0%, 100% {{
        transform: translate(0, 0) scale(1);
        opacity: 0.25;
    }}
    50% {{
        transform: translate(30px, -30px) scale(1.05);
        opacity: 0.35;
    }}
}}

.fluorescent-orb {{
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}}

.orb-1 {{
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}}

.orb-2 {{
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}}

@keyframes sparkle {{
    0%, 100% {{
        opacity: 0;
        transform: scale(0);
    }}
    50% {{
        opacity: 0.6;
        transform: scale(1);
    }}
}}

.sparkle {{
    position: fixed;
    width: 3px;
    height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}}

.sparkle-1 {{ top: 25%; left: 35%; animation-delay: 0s; }}
.sparkle-2 {{ top: 65%; left: 70%; animation-delay: 1s; }}
.sparkle-3 {{ top: 45%; left: 15%; animation-delay: 2s; }}

/* ========================================
   CUSTOM TAB STYLING (Modern Design)
   ======================================== */

/* Tab container */
.stTabs {{
    background: transparent;
    margin-top: 1rem;
}}

/* Tab list (container for all tab buttons) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}

/* Individual tab buttons */
.stTabs [data-baseweb="tab"] {{
    height: 50px;
    background: transparent;
    border-radius: 12px;
    padding: 0 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: {BODY_TEXT};
    border: 1px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* Tab button hover effect */
.stTabs [data-baseweb="tab"]:hover {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
    border-color: rgba(59, 130, 246, 0.3);
    color: {TEXT_COLOR};
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}}

/* Active/selected tab */
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(34, 211, 238, 0.2)) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    color: {TEXT_COLOR} !important;
    box-shadow:
        0 0 20px rgba(59, 130, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}

/* Active tab indicator (underline) - hide it */
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent;
}}

/* Tab panels (content area) */
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
}}

@media (max-width: 768px) {{
    .fluorescent-orb {{
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }}
    .sparkle {{
        display: none;
    }}

    /* Make tabs stack vertically on mobile */
    .stTabs [data-baseweb="tab-list"] {{
        flex-direction: column;
    }}

    .stTabs [data-baseweb="tab"] {{
        width: 100%;
    }}
}}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1.5rem;'>
      <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
        <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>🧠</div>
        <h1 class='hf-feature-title' style='font-size: 1.75rem; margin: 0;'>Modeling Hub</h1>
      </div>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 0 0 4rem;'>
        Unified platform for configuring and training time series forecasting models
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Seed defaults robustly (works even if an old session dict exists)
# -----------------------------------------------------------------------------
DEFAULTS = {
    # Global-ish
    "target_col": None,
    "feature_cols": [],
    "split_ratio": 0.8,
    "cv_folds": 3,

    # ---------- Benchmarks (ARIMA / SARIMAX) ----------
    "benchmark_model": "ARIMA",              # default
    "benchmark_param_mode": "Automatic",     # 'Automatic' or 'Manual'
    "optimization_method": "Auto (AIC-based)",
    "seasonal_enabled": False,
    "exogenous_cols": [],
    "seasonal_period": 7,
    # Auto bounds
    "max_p": 5, "max_d": 2, "max_q": 5,
    "max_P": 2, "max_D": 1, "max_Q": 2,
    # Manual params
    "arima_order": (1, 0, 0),
    "sarimax_order": (1, 0, 0),
    "sarimax_seasonal_order": (0, 0, 0, 0),

    # ---------- ML (LSTM / ANN / XGBoost) ----------
    "ml_choice": "XGBoost",                  # << default as requested
    # LSTM
    "ml_lstm_param_mode": "Automatic",
    "lstm_layers": 2, "lstm_hidden_units": 64, "lstm_epochs": 100, "lstm_lr": 1e-3,
    "lookback_window": 14, "dropout": 0.2, "batch_size": 32,
    "lstm_search_method": "Grid (bounded)",
    "lstm_max_layers": 3, "lstm_max_units": 256, "lstm_max_epochs": 50,
    "lstm_lr_min": 1e-5, "lstm_lr_max": 1e-2,
    # ANN
    "ml_ann_param_mode": "Automatic",
    "ann_hidden_layers": 2, "ann_neurons": 64, "ann_epochs": 30, "ann_activation": "relu",
    "ann_search_method": "Grid (bounded)",
    "ann_max_layers": 4, "ann_max_neurons": 256, "ann_max_epochs": 60,
    # XGB
    "ml_xgb_param_mode": "Automatic",
    "xgb_n_estimators": 300, "xgb_max_depth": 6, "xgb_min_child_weight": 1, "xgb_eta": 0.1,
    "xgb_search_method": "Grid (bounded)",
    "xgb_n_estimators_min": 100, "xgb_n_estimators_max": 1000,
    "xgb_depth_min": 3, "xgb_depth_max": 12,
    "xgb_eta_min": 0.01, "xgb_eta_max": 0.3,

    # ---------- Hybrids (LSTM-XGB / LSTM-ANN / LSTM-SARIMAX) ----------
    "hybrid_choice": "LSTM-XGBoost",         # sensible default
    # LSTM-XGB
    "hyb_lstm_xgb_param_mode": "Automatic",
    "hyb_lstm_units": 64, "hyb_xgb_estimators": 300, "hyb_xgb_depth": 6, "hyb_xgb_eta": 0.1,
    "hyb_lstm_search_method": "Grid (bounded)",
    "hyb_lstm_max_units": 256,
    "hyb_xgb_estimators_min": 100, "hyb_xgb_estimators_max": 1000,
    "hyb_xgb_depth_min": 3, "hyb_xgb_depth_max": 12,
    "hyb_xgb_eta_min": 0.01, "hyb_xgb_eta_max": 0.3,
    # LSTM-ANN
    "hyb_lstm_ann_param_mode": "Automatic",
    "hyb2_lstm_units": 64, "hyb2_ann_neurons": 64, "hyb2_ann_layers": 2, "hyb2_ann_activation": "relu",
    "hyb2_lstm_search_method": "Grid (bounded)",
    "hyb2_lstm_max_units": 256, "hyb2_ann_max_layers": 4, "hyb2_ann_max_neurons": 256,
    # LSTM-SARIMAX
    "hyb_lstm_sarimax_param_mode": "Automatic",
    "hyb3_lstm_units": 64, "hyb3_sarimax_order": (1, 0, 0),
    "hyb3_sarimax_seasonal_order": (0, 0, 0, 0), "hyb3_seasonal_period": 7,
    "hyb3_exogenous_cols": [],
    "hyb3_lstm_search_method": "Grid (bounded)",
    "hyb3_lstm_max_units": 256,
    "hyb3_max_p": 5, "hyb3_max_d": 2, "hyb3_max_q": 5,
    "hyb3_max_P": 2, "hyb3_max_D": 1, "hyb3_max_Q": 2,
}

cfg = st.session_state.setdefault("modeling_config", {})
for k, v in DEFAULTS.items():
    cfg.setdefault(k, v)

with st.sidebar.expander("⚙️ Config"):
    if st.button("Reset modeling config"):
        st.session_state["modeling_config"] = DEFAULTS.copy()
        st.rerun()

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def data_source_placeholder():
    with st.expander("📂 Data Source (placeholder)", expanded=False):
        st.info(
            "In the complete version, you'll select your working DataFrame here "
            "(e.g., from st.session_state['fs_df'] / 'fe_df'), or upload a CSV."
        )
        st.write("Detected session keys:")
        st.code([k for k in st.session_state.keys()])

def get_feature_selection_datasets():
    """
    Extract filtered datasets from Feature Selection results.

    Returns:
        dict: Mapping of method names to DataFrames
        dict: Metadata about the feature selection run
    """
    fs_results = st.session_state.get("feature_selection")

    if fs_results is None:
        return {}, {}

    filtered_datasets = fs_results.get("filtered_datasets", {})

    # Map method indices to readable names
    method_names = {
        0: "0 — Baseline (No Selection)",
        1: "1 — Permutation Importance",
        2: "2 — Lasso (Linear L1)",
        3: "3 — Gradient Boosting"
    }

    # Create dataset mapping with method names
    datasets = {}
    for idx, df in filtered_datasets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            method_name = method_names.get(idx, f"Method {idx}")
            datasets[method_name] = df

    # Extract metadata
    metadata = {
        "variant": fs_results.get("variant", "Unknown"),
        "target": fs_results.get("target", "Target_1"),
        "selected_features": fs_results.get("selected_features", {}),
        "summary": fs_results.get("summary")
    }

    return datasets, metadata

def target_feature_placeholders(prefix="", df=None):
    """
    Target and feature selection UI.

    Args:
        prefix: Key prefix for uniqueness
        df: Optional DataFrame to extract columns from
    """
    st.subheader("🎯 Target & Features Selection")

    if df is not None and not df.empty:
        # Extract target and feature columns from the dataset
        all_columns = df.columns.tolist()

        # Target columns are Target_1 through Target_7
        target_columns = [col for col in all_columns if col.startswith("Target_")]

        # Feature columns are all non-target columns
        feature_columns = [col for col in all_columns if not col.startswith("Target_")]

        # Target selection
        if target_columns:
            default_target = target_columns[0]  # Default to Target_1
            current_target = cfg.get(f"{prefix}_target_col", default_target)

            # Ensure current_target is in target_columns
            if current_target not in target_columns:
                current_target = default_target

            selected_target = st.selectbox(
                "Target column",
                options=target_columns,
                index=target_columns.index(current_target),
                help="Select which target variable to predict (Target_1 through Target_7)",
                key=f"{prefix}_tf_target"
            )
            cfg[f"{prefix}_target_col"] = selected_target
        else:
            st.warning("⚠️ No target columns found in dataset")
            cfg[f"{prefix}_target_col"] = None

        # Feature selection
        if feature_columns:
            current_features = cfg.get(f"{prefix}_feature_cols", feature_columns)

            # Ensure current_features are valid
            current_features = [f for f in current_features if f in feature_columns]
            if not current_features:
                current_features = feature_columns

            selected_features = st.multiselect(
                "Feature columns",
                options=feature_columns,
                default=current_features,
                help="Select features to use for training (all selected by default)",
                key=f"{prefix}_tf_feats"
            )
            cfg[f"{prefix}_feature_cols"] = selected_features

            # Show feature count
            st.caption(f"✓ {len(selected_features)} features selected")
        else:
            st.warning("⚠️ No feature columns found in dataset")
            cfg[f"{prefix}_feature_cols"] = []
    else:
        # Fallback to placeholder when no dataset is available
        st.selectbox("Target column (placeholder)", ["(choose later)"], key=f"{prefix}_tf_target")
        st.multiselect("Feature columns (placeholder)", [], key=f"{prefix}_tf_feats")
        st.caption("Select a dataset above to configure target and features")

def debug_panel():
    st.divider()
    with st.expander("🔎 Debug — current modeling config", expanded=False):
        st.json(st.session_state["modeling_config"])

# -----------------------------------------------------------------------------
# Helper functions for extracting metrics from benchmark results
# -----------------------------------------------------------------------------
def _safe_float(x):
    """Convert to float safely, returning NaN for invalid values."""
    try:
        f = float(x)
        if np.isfinite(f):
            return f
        return np.nan
    except Exception:
        return np.nan

def _sanitize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure metrics DataFrame has numeric columns."""
    if df is None or df.empty:
        return df
    work = df.copy()
    # Known metric columns to coerce
    cols = [
        "Train_MAE", "Train_RMSE", "Train_MAPE", "Train_Acc",
        "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_Acc",
        "MAE", "RMSE", "MAPE_%", "Accuracy_%", "AIC", "BIC"
    ]
    for c in cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    return work

def _mean_from_any(df: pd.DataFrame, candidates: list[str]) -> float:
    """Return mean of first available candidate column (numeric), or NaN."""
    if df is None or df.empty:
        return np.nan
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s[np.isfinite(s)]
            if s.notna().any():
                return float(s.mean())
    return np.nan

def _create_kpi_indicator(title: str, value, suffix: str = "", color: str = PRIMARY_COLOR):
    """Create a KPI indicator card using Plotly with fluorescent glow effect."""
    try:
        is_num = isinstance(value, (int, float, np.floating)) and np.isfinite(value)
    except Exception:
        is_num = False

    display_val = float(value) if is_num else 0.0

    # Map colors to fluorescent glow colors
    glow_colors = {
        PRIMARY_COLOR: "rgba(59, 130, 246, 0.6)",      # Blue glow
        SECONDARY_COLOR: "rgba(34, 211, 238, 0.6)",    # Cyan glow
        SUCCESS_COLOR: "rgba(34, 197, 94, 0.6)",       # Green glow
        WARNING_COLOR: "rgba(245, 158, 11, 0.6)",      # Orange glow
    }
    glow_color = glow_colors.get(color, "rgba(59, 130, 246, 0.6)")

    number_color = color if is_num else "rgba(0,0,0,0)"

    fig = go.Figure(go.Indicator(
        mode="number",
        value=display_val,
        number={
            "suffix": suffix,
            "font": {
                "size": 26,
                "color": number_color,
                "family": "Arial Black"
            }
        },
        title={
            "text": f"<span style='color:{TEXT_COLOR};font-size:12px;font-weight:700;letter-spacing:0.5px'>{title}</span>"
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        margin=dict(l=5, r=5, t=30, b=5),
        paper_bgcolor="rgba(15, 23, 42, 0.8)",  # Dark semi-transparent background
        height=110
    )

    if not is_num:
        fig.add_annotation(
            x=0.5, y=0.45,
            text="—",
            showarrow=False,
            font=dict(size=22, color=color, family="Arial")
        )

    return fig

def _extract_arima_metrics():
    """Extract average metrics from ARIMA multi-horizon results."""
    arima_mh = st.session_state.get("arima_mh_results")
    if arima_mh is None:
        return None

    results_df = arima_mh.get("results_df")
    if results_df is None or results_df.empty:
        return None

    results_df = _sanitize_metrics_df(results_df)

    return {
        "MAE": _mean_from_any(results_df, ["Test_MAE", "MAE"]),
        "RMSE": _mean_from_any(results_df, ["Test_RMSE", "RMSE"]),
        "MAPE": _mean_from_any(results_df, ["Test_MAPE", "MAPE_%", "MAPE"]),
        "Accuracy": _mean_from_any(results_df, ["Test_Acc", "Accuracy_%", "Accuracy"]),
    }

def _extract_sarimax_metrics():
    """Extract average metrics from SARIMAX multi-horizon results."""
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results is None:
        return None

    results_df = sarimax_results.get("results_df")
    if results_df is None or results_df.empty:
        return None

    results_df = _sanitize_metrics_df(results_df)

    return {
        "MAE": _mean_from_any(results_df, ["Test_MAE", "MAE"]),
        "RMSE": _mean_from_any(results_df, ["Test_RMSE", "RMSE"]),
        "MAPE": _mean_from_any(results_df, ["Test_MAPE", "MAPE_%", "MAPE"]),
        "Accuracy": _mean_from_any(results_df, ["Test_Acc", "Accuracy_%", "Accuracy"]),
    }

def _extract_ml_metrics():
    """Extract average metrics from ML multi-horizon results."""
    ml_mh_results = st.session_state.get("ml_mh_results")
    if ml_mh_results is None:
        return None

    results_df = ml_mh_results.get("results_df")
    if results_df is None or results_df.empty:
        return None

    results_df = _sanitize_metrics_df(results_df)

    return {
        "MAE": _mean_from_any(results_df, ["Test_MAE", "MAE"]),
        "RMSE": _mean_from_any(results_df, ["Test_RMSE", "RMSE"]),
        "MAPE": _mean_from_any(results_df, ["Test_MAPE", "MAPE_%", "MAPE"]),
        "Accuracy": _mean_from_any(results_df, ["Test_Acc", "Accuracy_%", "Accuracy"]),
        "Model": results_df["Model"].iloc[0] if "Model" in results_df.columns else "ML Model",
    }

# -----------------------------------------------------------------------------
# PROBABILITY-BASED CATEGORY FORECASTING UTILITIES
# -----------------------------------------------------------------------------

# Category configuration (icons and colors for clinical categories)
CATEGORY_CONFIG = {
    "RESPIRATORY": {"icon": "🫁", "color": "#3b82f6"},
    "CARDIAC": {"icon": "❤️", "color": "#ef4444"},
    "TRAUMA": {"icon": "🩹", "color": "#f59e0b"},
    "GASTROINTESTINAL": {"icon": "🤢", "color": "#10b981"},
    "INFECTIOUS": {"icon": "🦠", "color": "#8b5cf6"},
    "NEUROLOGICAL": {"icon": "🧠", "color": "#ec4899"},
    "OTHER": {"icon": "📋", "color": "#6b7280"},
}

def _get_category_proportions():
    """
    Calculate historical proportions for clinical categories from processed_df.
    Returns a dict of {category: proportion} where proportions sum to 1.0
    """
    category_proportions = {}

    # Use processed_df which contains the aggregated clinical categories
    merged_df = st.session_state.get("processed_df")
    if merged_df is None or (hasattr(merged_df, 'empty') and merged_df.empty):
        # Fallback to merged_data if processed_df not available
        merged_df = st.session_state.get("merged_data")

    if merged_df is None or (hasattr(merged_df, 'empty') and merged_df.empty):
        return {}

    # Build uppercase column map for case-insensitive matching
    col_map = {col.upper(): col for col in merged_df.columns}

    # Sum up historical values for each category
    cat_totals = {}
    for cat in CATEGORY_CONFIG.keys():
        cat_upper = cat.upper()

        # Look for columns that match this category (case-insensitive)
        matching_cols = []
        for col_upper, original_col in col_map.items():
            # Match exact category name or category with suffix (e.g., RESPIRATORY_1)
            if col_upper == cat_upper or col_upper.startswith(cat_upper + "_"):
                matching_cols.append(original_col)

        # Sum all matching columns for this category
        if matching_cols:
            total_for_cat = 0
            for col in matching_cols:
                col_sum = merged_df[col].sum()
                if pd.notna(col_sum):
                    total_for_cat += col_sum
            if total_for_cat > 0:
                cat_totals[cat] = total_for_cat

    # Calculate proportions
    total_cat_sum = sum(cat_totals.values()) if cat_totals else 0
    if total_cat_sum > 0:
        for cat, cat_sum in cat_totals.items():
            category_proportions[cat] = cat_sum / total_cat_sum

    return category_proportions

def _distribute_with_smart_rounding(total: int, proportions: dict) -> dict:
    """
    Distribute total into categories using proportions with smart rounding.
    Ensures the sum of distributed values equals the total exactly.
    """
    if not proportions:
        return {}

    # Calculate initial values (floored)
    result = {}
    remainders = {}
    allocated = 0

    for cat, prop in proportions.items():
        exact_val = total * prop
        floored_val = int(exact_val)
        result[cat] = floored_val
        remainders[cat] = exact_val - floored_val
        allocated += floored_val

    # Distribute remaining units based on largest remainders
    remaining = total - allocated
    sorted_cats = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)

    for i in range(min(remaining, len(sorted_cats))):
        result[sorted_cats[i]] += 1

    return result

def _render_category_forecast_section(predictions: list, title: str = "Category Breakdown"):
    """
    Render the probability-based category forecast section for ML model predictions.

    Args:
        predictions: List of prediction values (one per horizon/day)
        title: Section title
    """
    proportions = _get_category_proportions()

    if not proportions:
        return  # No category data available

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 211, 238, 0.05));
                border-left: 4px solid #10b981;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;'>
        <h4 style='color: #10b981; margin: 0 0 0.5rem 0;'>📊 {title}</h4>
        <p style='color: {TEXT_COLOR}; margin: 0; font-size: 0.875rem;'>
            Category distribution based on historical proportions (probability-based approach)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate category breakdown for each prediction
    num_predictions = min(len(predictions), 7)

    if num_predictions > 0:
        # Create columns for forecast cards
        cols = st.columns(num_predictions, gap="small")

        for idx, pred_val in enumerate(predictions[:num_predictions]):
            total_forecast = int(round(float(pred_val)))
            category_breakdown = _distribute_with_smart_rounding(total_forecast, proportions)

            with cols[idx]:
                # Day card
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    border: 1px solid rgba(34, 211, 238, 0.3);
                    margin-bottom: 0.5rem;
                '>
                    <div style='font-size: 0.75rem; color: #64748b; text-transform: uppercase;'>Day {idx + 1}</div>
                    <div style='font-size: 1.5rem; font-weight: 800; color: #22d3ee;'>{total_forecast}</div>
                    <div style='font-size: 0.625rem; color: #94a3b8;'>Total Patients</div>
                </div>
                """, unsafe_allow_html=True)

                # Category breakdown expander
                with st.expander("📊 Categories", expanded=False):
                    for cat, cfg in CATEGORY_CONFIG.items():
                        cat_val = category_breakdown.get(cat, 0)
                        if cat_val > 0:
                            st.markdown(f"""
                            <div style='display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid rgba(100, 116, 139, 0.1);'>
                                <span style='display: flex; align-items: center; gap: 6px;'>
                                    <span style='font-size: 0.875rem;'>{cfg['icon']}</span>
                                    <span style='color: #94a3b8; font-size: 0.75rem; font-weight: 600;'>{cat[:4]}</span>
                                </span>
                                <span style='color: {cfg["color"]}; font-weight: 700; font-size: 0.875rem;'>{cat_val}</span>
                            </div>
                            """, unsafe_allow_html=True)

    # Category Statistics Section
    st.markdown("""
    <div style='margin-top: 1rem;'>
        <div style='font-size: 0.875rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.75rem;'>
            📈 Category Proportions (Historical)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display proportions in compact cards
    prop_cols = st.columns(len(CATEGORY_CONFIG), gap="small")

    for idx, (cat, cfg) in enumerate(CATEGORY_CONFIG.items()):
        proportion = proportions.get(cat, 0) * 100

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

        with prop_cols[idx]:
            if proportion > 0:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.6));
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    border: 1px solid {cfg['color']}30;
                '>
                    <div style='font-size: 1rem;'>{cfg['icon']}</div>
                    <div style='font-size: 0.5rem; color: #94a3b8; text-transform: uppercase;'>{cat[:4]}</div>
                    <div style='
                        display: inline-block;
                        padding: 0.125rem 0.25rem;
                        background: {prop_bg};
                        border: 1px solid {prop_color}40;
                        border-radius: 4px;
                        font-size: 0.5rem;
                        font-weight: 700;
                        color: {prop_color};
                        margin-top: 0.25rem;
                    '>{proportion:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

def _render_category_statistics_compact():
    """Render a compact category statistics section showing proportions only."""
    proportions = _get_category_proportions()

    if not proportions:
        return

    st.markdown("""
    <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px; padding: 0.75rem; margin: 1rem 0;'>
        <div style='font-size: 0.75rem; font-weight: 600; color: #10b981; margin-bottom: 0.5rem;'>
            📊 Category Distribution (Probability-Based)
        </div>
    """, unsafe_allow_html=True)

    # Display as inline badges
    badges_html = ""
    for cat, cfg in CATEGORY_CONFIG.items():
        proportion = proportions.get(cat, 0) * 100
        if proportion > 0:
            badges_html += f"""
            <span style='
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 0.25rem 0.5rem;
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid {cfg["color"]}40;
                border-radius: 6px;
                margin: 0.125rem;
                font-size: 0.6875rem;
            '>
                <span>{cfg['icon']}</span>
                <span style='color: #94a3b8;'>{cat[:4]}</span>
                <span style='color: {cfg["color"]}; font-weight: 700;'>{proportion:.1f}%</span>
            </span>
            """

    st.markdown(f"<div style='display: flex; flex-wrap: wrap; gap: 0.25rem;'>{badges_html}</div></div>", unsafe_allow_html=True)


# =============================================================================
# CONFORMAL PREDICTION (CQR) HELPER FUNCTIONS
# =============================================================================

def _get_calibration_data():
    """
    Get calibration indices and data from Feature Engineering session state.

    Returns:
        tuple: (cal_idx, split_result, available) - calibration indices, split result, and availability flag
    """
    fe = st.session_state.get("feature_engineering", {})

    cal_idx = fe.get("cal_idx")
    split_result = fe.get("split_result")

    # Check if calibration data is available
    available = (
        cal_idx is not None and
        isinstance(cal_idx, np.ndarray) and
        len(cal_idx) > 0
    )

    return cal_idx, split_result, available


def _compute_cqr_intervals(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.1
) -> dict:
    """
    Compute CQR prediction intervals for ML model predictions.

    Uses simple conformal prediction since ML models produce point forecasts only.

    Args:
        y_cal: Actual values on calibration set
        pred_cal: Predictions on calibration set
        pred_test: Predictions on test set
        y_test: Actual values on test set
        alpha: Significance level (0.1 = 90% coverage)

    Returns:
        dict with keys: intervals (PredictionInterval), rpiw (RPIWResult), success (bool)
    """
    if not CQR_AVAILABLE:
        return {"success": False, "error": "CQR module not available"}

    try:
        # Use simple conformal prediction (symmetric intervals around point forecast)
        intervals = simple_conformal_intervals(
            y_cal=np.asarray(y_cal),
            pred_cal=np.asarray(pred_cal),
            pred_test=np.asarray(pred_test),
            alpha=alpha
        )

        # Calculate RPIW
        rpiw_result = calculate_rpiw(np.asarray(y_test), intervals)

        return {
            "success": True,
            "intervals": intervals,
            "rpiw": rpiw_result,
            "coverage": rpiw_result.coverage,
            "mean_width": intervals.mean_width,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _render_cqr_metrics(rpiw_result: "RPIWResult", model_name: str = "Model"):
    """
    Render CQR/RPIW metrics as compact KPI cards.

    Args:
        rpiw_result: RPIWResult from CQR calculation
        model_name: Name of the model for display
    """
    if rpiw_result is None:
        return

    # Coverage color based on validity
    if rpiw_result.is_valid:
        cov_color = SUCCESS_COLOR
        cov_status = "Valid"
    else:
        cov_color = WARNING_COLOR
        cov_status = f"Below target ({rpiw_result.coverage_gap:.1%} gap)"

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.05));
                border: 1px solid rgba(139, 92, 246, 0.3);
                border-radius: 12px;
                padding: 1rem;
                margin: 1rem 0;'>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;'>
            <span style='font-size: 1.25rem;'>📊</span>
            <span style='font-size: 1rem; font-weight: 700; color: {TEXT_COLOR};'>
                Prediction Interval Quality ({model_name})
            </span>
            <span style='font-size: 0.75rem; color: {SUBTLE_TEXT}; margin-left: auto;'>
                Conformalized Quantile Regression (CQR)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(
            _create_kpi_indicator("RPIW", rpiw_result.rpiw, "", "#8b5cf6"),
            use_container_width=True
        )
        st.caption("Lower = Tighter intervals")

    with col2:
        st.plotly_chart(
            _create_kpi_indicator("Coverage", rpiw_result.coverage * 100, "%", cov_color),
            use_container_width=True
        )
        st.caption(f"Target: {rpiw_result.coverage_target:.0%}")

    with col3:
        st.plotly_chart(
            _create_kpi_indicator("Mean Width", rpiw_result.mean_interval_width, "", SECONDARY_COLOR),
            use_container_width=True
        )
        st.caption("Average interval size")

    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Data Range", rpiw_result.data_range, "", PRIMARY_COLOR),
            use_container_width=True
        )
        st.caption("max(y) - min(y)")


def _add_intervals_to_plot(fig: go.Figure, datetime_vals, lower: np.ndarray, upper: np.ndarray,
                           alpha: float = 0.1, color: str = "rgba(139, 92, 246, 0.2)"):
    """
    Add prediction interval band to an existing Plotly figure.

    Args:
        fig: Existing Plotly figure
        datetime_vals: X-axis values (datetime)
        lower: Lower bound of intervals
        upper: Upper bound of intervals
        alpha: Significance level for legend
        color: Fill color for the band
    """
    coverage_pct = int((1 - alpha) * 100)

    # Add upper bound (invisible line)
    fig.add_trace(go.Scatter(
        x=datetime_vals,
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Add lower bound with fill to upper
    fig.add_trace(go.Scatter(
        x=datetime_vals,
        y=lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=color,
        name=f'{coverage_pct}% Prediction Interval',
        hovertemplate=f'<b>{coverage_pct}% Interval</b><br>Lower: %{{y:.2f}}<extra></extra>',
    ))

    return fig


def compute_cqr_for_multihorizon(ml_results: dict, alpha: float = 0.1) -> dict:
    """
    Compute CQR prediction intervals for all horizons in multi-horizon ML results.

    Args:
        ml_results: Multi-horizon results from run_ml_multihorizon()
        alpha: Significance level (0.1 = 90% coverage)

    Returns:
        dict with keys:
            - success: bool
            - intervals: Dict[horizon, PredictionInterval]
            - rpiw_results: Dict[horizon, RPIWResult]
            - aggregated_rpiw: float (average RPIW across horizons)
            - aggregated_coverage: float (average coverage across horizons)
    """
    if not CQR_AVAILABLE:
        return {"success": False, "error": "CQR module not available"}

    per_h = ml_results.get("per_h", {})
    successful = ml_results.get("successful", [])
    has_cqr_data = ml_results.get("has_cqr_data", False)

    if not has_cqr_data:
        return {"success": False, "error": "No calibration data available. Run Feature Engineering first."}

    intervals = {}
    rpiw_results = {}
    all_rpiwsv = []
    all_coverages = []

    for h in successful:
        h_data = per_h.get(h, {})

        y_cal = h_data.get("y_cal")
        y_cal_pred = h_data.get("y_cal_pred")
        y_test = h_data.get("y_test")
        forecast = h_data.get("forecast")

        if y_cal is None or y_cal_pred is None or y_test is None or forecast is None:
            continue

        try:
            # Compute CQR intervals for this horizon
            cqr_result = _compute_cqr_intervals(
                y_cal=y_cal,
                pred_cal=y_cal_pred,
                pred_test=forecast,
                y_test=y_test,
                alpha=alpha
            )

            if cqr_result.get("success"):
                intervals[h] = cqr_result["intervals"]
                rpiw_results[h] = cqr_result["rpiw"]
                all_rpiwsv.append(cqr_result["rpiw"].rpiw)
                all_coverages.append(cqr_result["rpiw"].coverage)

                # Update per_h with interval bounds
                per_h[h]["ci_lo"] = cqr_result["intervals"].lower
                per_h[h]["ci_hi"] = cqr_result["intervals"].upper

        except Exception as e:
            continue

    if not intervals:
        return {"success": False, "error": "Failed to compute CQR for any horizon"}

    return {
        "success": True,
        "intervals": intervals,
        "rpiw_results": rpiw_results,
        "aggregated_rpiw": float(np.mean(all_rpiwsv)) if all_rpiwsv else None,
        "aggregated_coverage": float(np.mean(all_coverages)) if all_coverages else None,
        "n_horizons": len(intervals),
    }


def render_cqr_section(ml_results: dict, model_name: str, alpha: float = 0.1):
    """
    Render CQR prediction intervals section for multi-horizon ML results.

    Args:
        ml_results: Multi-horizon results from run_ml_multihorizon()
        model_name: Name of the model for display
        alpha: Significance level (0.1 = 90% coverage)
    """
    if not CQR_AVAILABLE:
        return

    has_cqr_data = ml_results.get("has_cqr_data", False)

    if not has_cqr_data:
        st.info("📊 **Prediction Intervals**: Run Feature Engineering with a calibration set to enable CQR prediction intervals.")
        return

    # Compute CQR intervals
    cqr_output = compute_cqr_for_multihorizon(ml_results, alpha)

    if not cqr_output.get("success"):
        st.warning(f"⚠️ Could not compute prediction intervals: {cqr_output.get('error', 'Unknown error')}")
        return

    # Header
    coverage_pct = int((1 - alpha) * 100)
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1));
                border: 1px solid rgba(139, 92, 246, 0.4);
                border-radius: 12px;
                padding: 1.25rem;
                margin: 2rem 0 1rem 0;'>
        <div style='display: flex; align-items: center; gap: 0.75rem;'>
            <span style='font-size: 1.5rem;'>📊</span>
            <div>
                <h3 style='color: {TEXT_COLOR}; margin: 0; font-size: 1.2rem; font-weight: 700;'>
                    {coverage_pct}% Prediction Intervals ({model_name})
                </h3>
                <p style='color: {SUBTLE_TEXT}; margin: 0.25rem 0 0 0; font-size: 0.85rem;'>
                    Conformalized Quantile Regression (CQR) - Romano et al., 2019
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Aggregated metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(
            _create_kpi_indicator("Avg RPIW", cqr_output.get("aggregated_rpiw", 0), "", "#8b5cf6"),
            use_container_width=True,
            key=f"cqr_rpiw_{model_name}"
        )
        st.caption("Lower = Tighter intervals")

    with col2:
        coverage = cqr_output.get("aggregated_coverage", 0)
        cov_color = SUCCESS_COLOR if coverage >= (1 - alpha - 0.02) else WARNING_COLOR
        st.plotly_chart(
            _create_kpi_indicator("Avg Coverage", coverage * 100 if coverage else 0, "%", cov_color),
            use_container_width=True,
            key=f"cqr_coverage_{model_name}"
        )
        st.caption(f"Target: {coverage_pct}%")

    with col3:
        st.plotly_chart(
            _create_kpi_indicator("Horizons", cqr_output.get("n_horizons", 0), "", PRIMARY_COLOR),
            use_container_width=True,
            key=f"cqr_horizons_{model_name}"
        )
        st.caption("With valid intervals")

    with col4:
        # Validity indicator
        is_valid = (cqr_output.get("aggregated_coverage", 0) or 0) >= (1 - alpha - 0.02)
        validity_color = SUCCESS_COLOR if is_valid else WARNING_COLOR
        validity_text = "Valid" if is_valid else "Under-coverage"
        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.8);
                    border-radius: 12px;
                    padding: 1.25rem;
                    text-align: center;
                    box-shadow: 0 0 25px {validity_color}40;
                    border: 1px solid {validity_color}40;
                    height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;'>
            <div style='color: {TEXT_COLOR}; font-size: 12px; font-weight: 700;
                        letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                STATUS
            </div>
            <div style='color: {validity_color}; font-size: 20px; font-weight: 800;
                        text-shadow: 0 0 15px {validity_color}80;'>
                {validity_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Per-horizon details in expander
    with st.expander("📈 Per-Horizon Interval Details", expanded=False):
        rpiw_results = cqr_output.get("rpiw_results", {})
        if rpiw_results:
            rows = []
            for h, rpiw in sorted(rpiw_results.items()):
                rows.append({
                    "Horizon": h,
                    "RPIW": round(rpiw.rpiw, 4),
                    "Coverage": f"{rpiw.coverage:.1%}",
                    "Mean Width": round(rpiw.mean_interval_width, 2),
                    "Data Range": round(rpiw.data_range, 2),
                    "Valid": "Yes" if rpiw.is_valid else "No"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# BENCHMARKS (ARIMA / SARIMAX)
# -----------------------------------------------------------------------------
def page_benchmarks():
    # Add custom CSS for fluorescent KPI cards
    st.markdown("""
    <style>
    /* Fluorescent KPI Card Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.js-plotly-plot) {
        border-radius: 16px;
        overflow: hidden;
    }

    /* Individual KPI card glow effects */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }

    /* Add pulsing glow animation to charts */
    @keyframes kpi-glow {
        0%, 100% {
            box-shadow:
                0 0 20px rgba(59, 130, 246, 0.3),
                0 0 40px rgba(34, 211, 238, 0.2),
                0 0 60px rgba(34, 197, 94, 0.1),
                inset 0 0 20px rgba(59, 130, 246, 0.1);
        }
        50% {
            box-shadow:
                0 0 30px rgba(59, 130, 246, 0.5),
                0 0 60px rgba(34, 211, 238, 0.3),
                0 0 80px rgba(34, 197, 94, 0.2),
                inset 0 0 30px rgba(59, 130, 246, 0.15);
        }
    }

    /* Apply to plotly charts */
    div.js-plotly-plot {
        border-radius: 16px;
        animation: kpi-glow 4s ease-in-out infinite;
        border: none;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8)) !important;
        backdrop-filter: blur(10px);
    }

    /* Enhanced hover effect */
    div.js-plotly-plot:hover {
        animation: kpi-glow 2s ease-in-out infinite;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: left; margin-bottom: 1rem; padding: 1rem;'>
          <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 1.8rem;'>📏</div>
            <h1 class='hf-feature-title' style='font-size: 1.5rem; margin: 0;'>Benchmarks Dashboard</h1>
          </div>
          <p class='hf-feature-description' style='font-size: 0.9rem; max-width: 600px; margin: 0 0 0 3.5rem;'>
            Performance metrics for ARIMA and SARIMAX models trained in the Benchmarks page
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Extract metrics from session state
    arima_metrics = _extract_arima_metrics()
    sarimax_metrics = _extract_sarimax_metrics()
    ml_metrics = _extract_ml_metrics()

    # Check if any results are available
    has_results = arima_metrics is not None or sarimax_metrics is not None or ml_metrics is not None

    if not has_results:
        st.info(
            "📊 **No benchmark results available yet.**\n\n"
            "- Train **ARIMA/SARIMAX** models in the **Benchmarks** page (Page 05)\n"
            "- Train **ML models** in the **Machine Learning** tab above\n\n"
            "Results will appear here automatically."
        )

        # Show example of what will appear
        with st.expander("📖 What you'll see here", expanded=False):
            st.markdown("""
            After training models, this dashboard will display:

            - 📈 **ARIMA Performance** - Average metrics across all forecast horizons
            - 📊 **SARIMAX Performance** - Average metrics across all forecast horizons
            - 🧮 **ML Model Performance** - Average metrics across all forecast horizons (XGBoost, LSTM, ANN)
            - 📉 **Key Metrics** - MAE, RMSE, MAPE, and Accuracy displayed as interactive KPI cards

            **To get started:**
            - For ARIMA/SARIMAX: Navigate to **Benchmarks** (Page 05) → Configure and train
            - For ML models: **Machine Learning** tab → Configure and train
            """)

        return

    # Display results
    # ARIMA Results
    if arima_metrics is not None:
        st.markdown(
            f"""
            <div style='text-align: center; margin: 0.75rem 0 0.5rem 0;'>
              <h3 style='font-size: 1.1rem; font-weight: 700; color: {PRIMARY_COLOR}; margin: 0;'>
                📈 ARIMA Performance (Avg Across All Horizons)
              </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", arima_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", arima_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", arima_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", arima_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True
            )

    # SARIMAX Results
    if sarimax_metrics is not None:
        st.markdown(
            f"""
            <div style='text-align: center; margin: 1rem 0 0.5rem 0;'>
              <h3 style='font-size: 1.1rem; font-weight: 700; color: {SECONDARY_COLOR}; margin: 0;'>
                📊 SARIMAX Performance (Avg Across All Horizons)
              </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", sarimax_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", sarimax_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", sarimax_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", sarimax_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True
            )

    # ML Model Results
    if ml_metrics is not None:
        model_name = ml_metrics.get("Model", "ML Model")
        st.markdown(
            f"""
            <div style='text-align: center; margin: 1rem 0 0.5rem 0;'>
              <h3 style='font-size: 1.1rem; font-weight: 700; color: {SUCCESS_COLOR}; margin: 0;'>
                🧮 {model_name} Performance (Avg Across All Horizons)
              </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", ml_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", ml_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", ml_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", ml_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True
            )

    # Additional info
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05));
                    border-left: 3px solid {PRIMARY_COLOR};
                    padding: 0.75rem;
                    border-radius: 6px;
                    margin-top: 1rem;'>
          <p style='margin: 0; color: {TEXT_COLOR}; font-size: 0.85rem;'>
            💡 <strong>Note:</strong> Metrics averaged across all forecast horizons. For per-horizon analysis, visit:
            <ul style='margin: 0.5rem 0 0 1.5rem;'>
              <li><strong>Benchmarks</strong> (Page 05) for ARIMA/SARIMAX</li>
              <li><strong>Machine Learning</strong> tab for ML models</li>
            </ul>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# ML Multi-Horizon Training Pipeline
# -----------------------------------------------------------------------------
def run_ml_multihorizon(
    model_type: str,
    config: dict,
    df: pd.DataFrame,
    feature_cols: list,
    split_ratio: float,
    horizons: int = 7,
    date_col: str = None,
) -> dict:
    """
    Train independent ML models for Target_1..Target_H (multi-horizon forecasting).

    Args:
        model_type: Model name ("LSTM", "ANN", or "XGBoost")
        config: Model configuration dictionary
        df: Training dataframe with Target_1...Target_H columns
        feature_cols: List of feature column names
        split_ratio: Train/validation split ratio
        horizons: Number of forecast horizons (default 7)
        date_col: Optional datetime column for visualization

    Returns:
        dict: Multi-horizon results with structure:
            - results_df: DataFrame with metrics per horizon
            - per_h: Dict mapping horizon to predictions/actuals/metrics
            - successful: List of successful horizons
    """
    import time
    from datetime import datetime

    rows = []
    per_h = {}
    successful = []
    errors = {}  # Store errors for each horizon

    # Parameter sharing: find optimal parameters once for h=1, reuse for h>1 (if using auto mode)
    shared_params = None

    for h in range(1, int(horizons) + 1):
        target_col = f"Target_{h}"

        if target_col not in df.columns:
            continue

        try:
            # Train model for this horizon
            result = run_ml_model(
                model_type=model_type,
                config=config,
                df=df,
                target_col=target_col,
                feature_cols=feature_cols,
                split_ratio=split_ratio,
            )

            if not result.get("success", False):
                # Store the error message from result
                error_msg = result.get("error", "Unknown error")
                errors[h] = error_msg
                print(f"❌ Horizon {h} failed: {error_msg}")
                continue

            # Extract metrics
            train_metrics = result.get("train_metrics", {})
            val_metrics = result.get("val_metrics", {})
            predictions = result.get("predictions")

            # Store per-horizon data
            rows.append({
                "Horizon": h,
                "Model": model_type,
                "Train_MAE": train_metrics.get("MAE", np.nan),
                "Train_RMSE": train_metrics.get("RMSE", np.nan),
                "Train_MAPE": train_metrics.get("MAPE", np.nan),
                "Train_Acc": train_metrics.get("Accuracy", np.nan),
                "Test_MAE": val_metrics.get("MAE", np.nan),
                "Test_RMSE": val_metrics.get("RMSE", np.nan),
                "Test_MAPE": val_metrics.get("MAPE", np.nan),
                "Test_Acc": val_metrics.get("Accuracy", np.nan),
                "Train_N": result.get("n_train", 0),
                "Test_N": result.get("n_val", 0),
            })

            per_h[h] = {
                "y_train": None,  # Not stored currently
                "y_test": predictions["actual"].values if predictions is not None else None,
                "forecast": predictions["predicted"].values if predictions is not None else None,
                "datetime": predictions["datetime"].values if predictions is not None else None,
                "ci_lo": None,  # Will be filled by CQR if available
                "ci_hi": None,  # Will be filled by CQR if available
                "params": result.get("params", {}),
                # CQR calibration data
                "has_calibration": result.get("has_calibration", False),
                "y_cal": result.get("y_cal"),
                "y_cal_pred": result.get("y_cal_pred"),
            }

            successful.append(h)

        except Exception as e:
            # Skip failed horizons and store error
            error_msg = str(e)
            errors[h] = error_msg
            print(f"❌ Horizon {h} failed: {error_msg}")
            import traceback
            traceback.print_exc()  # Print full traceback to console
            continue

    # Create results DataFrame (handle empty case)
    if rows:
        results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)
    else:
        # No successful horizons - create empty DataFrame with correct schema
        results_df = pd.DataFrame(columns=[
            "Horizon", "Model", "Train_MAE", "Train_RMSE", "Train_MAPE", "Train_Acc",
            "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_Acc", "Train_N", "Test_N"
        ])

    # Check if CQR calibration data is available (from first successful horizon)
    has_cqr_data = False
    if successful and per_h:
        first_h = successful[0]
        has_cqr_data = per_h[first_h].get("has_calibration", False)

    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": successful,
        "errors": errors,
        "has_cqr_data": has_cqr_data,  # Flag for CQR availability
    }

def ml_to_multihorizon_artifacts(ml_out: dict):
    """
    Convert ML multi-horizon results to unified artifact format (compatible with ARIMA/SARIMAX).

    Args:
        ml_out: Multi-horizon results from run_ml_multihorizon()

    Returns:
        tuple: (metrics_df, F, L, U, test_eval_df, train, res, horizons)
            - metrics_df: DataFrame with formatted metrics
            - F: Forecast matrix (T x H)
            - L: Lower CI matrix (T x H) - None for ML
            - U: Upper CI matrix (T x H) - None for ML
            - test_eval_df: DataFrame with actual values per horizon
            - train: Training data (None for ML)
            - res: Model results (None for ML)
            - horizons: List of horizon indices
    """
    per_h = ml_out.get("per_h", {})
    results_df = ml_out.get("results_df")
    successful = ml_out.get("successful", [])

    if not per_h or not successful:
        # Return empty structure
        empty_df = pd.DataFrame()
        empty_array = np.array([])
        return (results_df if results_df is not None else empty_df,
                empty_array, empty_array, empty_array, empty_df,
                None, None, [])

    # Get dimensions
    horizons = sorted(successful)
    H = len(horizons)

    # Get test set length from first horizon
    first_h = horizons[0]
    y_test_first = per_h[first_h].get("y_test")
    T = len(y_test_first) if y_test_first is not None else 0

    if T == 0:
        # No test data
        return (results_df, np.array([]), np.array([]), np.array([]),
                pd.DataFrame(), None, None, horizons)

    # Initialize matrices
    F = np.full((T, H), np.nan)
    L = np.full((T, H), np.nan)  # ML models don't have confidence intervals, fill with NaN
    U = np.full((T, H), np.nan)
    test_eval = {}
    datetime_index = None

    # Fill matrices from per_h data
    for idx, h in enumerate(horizons):
        h_data = per_h[h]

        # Get forecast and actuals
        fc = h_data.get("forecast")
        yt = h_data.get("y_test")
        dt = h_data.get("datetime")

        if fc is not None and len(fc) == T:
            F[:, idx] = fc

        if yt is not None and len(yt) == T:
            test_eval[f"Target_{h}"] = yt

        # Get datetime index from first horizon
        if datetime_index is None and dt is not None and len(dt) == T:
            datetime_index = pd.to_datetime(dt)

    # Create test_eval DataFrame with datetime index
    test_eval_df = pd.DataFrame(test_eval)
    if datetime_index is not None:
        test_eval_df.index = datetime_index

    # Create dummy train series for visualization compatibility
    # The dashboard expects train to have an index for plotting temporal split
    if datetime_index is not None and len(datetime_index) > 0:
        # Create empty train Series with datetime index
        # Use a dummy index that represents the training period (before test period)
        train_end = datetime_index[0] - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=T)  # Same length as test
        train_index = pd.date_range(start=train_start, end=train_end, periods=T)
        train_series = pd.Series(np.nan, index=train_index, name="train")
    else:
        train_series = pd.Series([], name="train")

    # Format metrics_df for compatibility
    metrics_df = results_df.copy() if results_df is not None else pd.DataFrame()
    if not metrics_df.empty and "Horizon" in metrics_df.columns:
        # Rename columns to match expected format
        rename_map = {
            "Test_MAE": "MAE",
            "Test_RMSE": "RMSE",
            "Test_MAPE": "MAPE_%",
            "Test_Acc": "Accuracy_%"
        }
        for old_col, new_col in rename_map.items():
            if old_col in metrics_df.columns and new_col not in metrics_df.columns:
                metrics_df[new_col] = metrics_df[old_col]

        # Keep Horizon as integer for dashboard compatibility
        # Dashboard visualization expects numeric Horizon values, not strings like "h=1"
        if metrics_df["Horizon"].dtype not in ['int64', 'int32']:
            metrics_df["Horizon"] = pd.to_numeric(metrics_df["Horizon"], errors='coerce').astype('int64')

    return (metrics_df, F, L, U, test_eval_df, train_series, None, horizons)

# -----------------------------------------------------------------------------
# ML Training Pipeline (Single Horizon)
# -----------------------------------------------------------------------------
def run_ml_model(model_type: str, config: dict, df: pd.DataFrame,
                 target_col: str, feature_cols: list, split_ratio: float):
    """
    Unified ML training pipeline for all models (LSTM, ANN, XGBoost).

    Supports both 2-way split (train/test) and 3-way split (train/cal/test) when
    Feature Engineering calibration indices are available.

    Args:
        model_type: Model name ("LSTM", "ANN", or "XGBoost")
        config: Model configuration dictionary
        df: Training dataframe
        target_col: Target column name
        feature_cols: List of feature column names
        split_ratio: Train/validation split ratio

    Returns:
        dict: Results containing metrics, predictions, and metadata
              Includes calibration predictions when cal_idx is available.
    """
    import time
    from datetime import datetime

    try:
        # Find datetime column in dataframe
        datetime_col = None
        datetime_cols_to_exclude = []

        for col in df.columns:
            # Check if column is datetime type or has datetime-like name
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if datetime_col is None:
                    datetime_col = col
                datetime_cols_to_exclude.append(col)
            # Check common datetime column names
            elif col.lower() in ['date', 'datetime', 'timestamp', 'time', 'ds']:
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col])
                    if datetime_col is None:
                        datetime_col = col
                    datetime_cols_to_exclude.append(col)
                except:
                    continue

        # Filter out datetime columns from features (ML models need numeric data only)
        numeric_feature_cols = [col for col in feature_cols if col not in datetime_cols_to_exclude]

        if not numeric_feature_cols:
            return {
                "success": False,
                "error": "No numeric features available after excluding datetime columns. Please select numeric features only."
            }

        # Extract features and target
        X = df[numeric_feature_cols].values
        y = df[target_col].values

        # =====================================================================
        # Check for Feature Engineering split indices (3-way split for CQR)
        # =====================================================================
        fe = st.session_state.get("feature_engineering", {})
        train_idx = fe.get("train_idx")
        cal_idx = fe.get("cal_idx")
        test_idx = fe.get("test_idx")

        # Check if 3-way split is available and indices are valid for this dataframe
        use_fe_split = (
            train_idx is not None and
            cal_idx is not None and
            test_idx is not None and
            isinstance(train_idx, np.ndarray) and
            isinstance(cal_idx, np.ndarray) and
            isinstance(test_idx, np.ndarray) and
            len(cal_idx) > 0 and
            max(train_idx.max() if len(train_idx) > 0 else 0,
                cal_idx.max() if len(cal_idx) > 0 else 0,
                test_idx.max() if len(test_idx) > 0 else 0) < len(df)
        )

        if use_fe_split:
            # Use Feature Engineering 3-way split (train/calibration/test)
            X_train, y_train = X[train_idx], y[train_idx]
            X_cal, y_cal = X[cal_idx], y[cal_idx]
            X_val, y_val = X[test_idx], y[test_idx]

            # Extract datetime values
            if datetime_col:
                val_datetime = df[datetime_col].iloc[test_idx].values
                cal_datetime = df[datetime_col].iloc[cal_idx].values
            else:
                val_datetime = test_idx.tolist()
                cal_datetime = cal_idx.tolist()

            has_calibration = True
        else:
            # Fallback to 2-way split (train/validation)
            split_idx = int(len(df) * split_ratio)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # No calibration set
            X_cal, y_cal = None, None
            cal_datetime = None
            has_calibration = False

            # Extract datetime values for validation set
            if datetime_col:
                val_datetime = df[datetime_col].iloc[split_idx:].values
            else:
                val_datetime = df.index[split_idx:].tolist() if hasattr(df, 'index') else list(range(split_idx, len(df)))

        # Initialize model based on type
        if model_type == "XGBoost":
            from app_core.models.ml.xgboost_pipeline import XGBoostPipeline

            # Extract XGBoost params from config
            model_config = {
                "n_estimators": config.get("xgb_n_estimators", 300),
                "max_depth": config.get("xgb_max_depth", 6),
                "eta": config.get("xgb_eta", 0.1),
                "min_child_weight": config.get("xgb_min_child_weight", 1),
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
            pipeline = XGBoostPipeline(model_config)

        elif model_type == "LSTM":
            from app_core.models.ml.lstm_pipeline import LSTMPipeline

            # Extract LSTM params from config
            model_config = {
                "lookback_window": config.get("lookback_window", 14),
                "lstm_hidden_units": config.get("lstm_hidden_units", 64),
                "lstm_layers": config.get("lstm_layers", 2),
                "dropout": config.get("dropout", 0.2),
                "lstm_epochs": config.get("lstm_epochs", 100),
                "lstm_lr": config.get("lstm_lr", 0.001),
                "batch_size": config.get("batch_size", 32),
            }
            pipeline = LSTMPipeline(model_config)

        elif model_type == "ANN":
            from app_core.models.ml.ann_pipeline import ANNPipeline

            # Extract ANN params from config
            model_config = {
                "ann_hidden_layers": config.get("ann_hidden_layers", 2),
                "ann_neurons": config.get("ann_neurons", 64),
                "ann_activation": config.get("ann_activation", "relu"),
                "dropout": config.get("dropout", 0.2),
                "ann_epochs": config.get("ann_epochs", 30),
            }
            pipeline = ANNPipeline(model_config)
        else:
            return {
                "success": False,
                "error": f"Unknown model type: {model_type}"
            }

        # Train model
        start_time = time.time()
        n_features = X_train.shape[1]  # Get number of features
        pipeline.build_model(n_features)  # Pass n_features to build_model
        pipeline.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        # Generate calibration predictions if calibration set is available (for CQR)
        y_cal_pred = None
        calibration_df = None
        if has_calibration and X_cal is not None:
            y_cal_pred = pipeline.predict(X_cal)
            calibration_df = pd.DataFrame({
                "datetime": cal_datetime,
                "actual": y_cal,
                "predicted": y_cal_pred
            })

        # Calculate metrics
        train_metrics = pipeline.evaluate(y_train, y_train_pred)
        val_metrics = pipeline.evaluate(y_val, y_val_pred)

        # Create predictions dataframe with datetime values
        predictions_df = pd.DataFrame({
            "datetime": val_datetime,
            "actual": y_val,
            "predicted": y_val_pred
        })

        # Build result dictionary
        result = {
            "success": True,
            "model_type": model_type,
            "params": model_config if model_type == "XGBoost" else {},
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "predictions": predictions_df,
            "training_time": training_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_ratio": split_ratio,
            "n_features": len(numeric_feature_cols),
            "n_train": len(y_train),
            "n_val": len(y_val),
            # CQR-related data
            "has_calibration": has_calibration,
        }

        # Add calibration data if available
        if has_calibration:
            result["calibration"] = calibration_df
            result["y_cal"] = y_cal
            result["y_cal_pred"] = y_cal_pred
            result["n_cal"] = len(y_cal)

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def render_ml_results(results: dict):
    """
    Render ML training results with KPI cards and plots.

    Args:
        results: Results dictionary from run_ml_model()
    """
    if not results.get("success", False):
        st.error(f"❌ **Training Failed**\n\n{results.get('error', 'Unknown error')}")
        return

    # Compact success message with key stats
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.success(f"✅ **{results['model_type']} Training Completed!**")
    with col2:
        st.metric("Time", f"{results['training_time']:.1f}s", label_visibility="visible")
    with col3:
        st.metric("Train", results['n_train'], label_visibility="visible")
    with col4:
        st.metric("Val", results['n_val'], label_visibility="visible")

    # Validation Metrics (KPI Cards)
    st.markdown("### 📊 Validation Metrics")
    val_metrics = results['val_metrics']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(
            _create_kpi_indicator("MAE", val_metrics["MAE"], "", PRIMARY_COLOR),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            _create_kpi_indicator("RMSE", val_metrics["RMSE"], "", SECONDARY_COLOR),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            _create_kpi_indicator("MAPE", val_metrics["MAPE"], "%", WARNING_COLOR),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Accuracy", val_metrics["Accuracy"], "%", SUCCESS_COLOR),
            use_container_width=True
        )

    # Actuals vs Forecast Plot
    st.markdown("### 📈 Actuals vs Forecast")

    predictions = results['predictions']

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=predictions['datetime'],
        y=predictions['actual'],
        mode='lines+markers',
        name='Actual',
        line=dict(color=PRIMARY_COLOR, width=2.5),
        marker=dict(size=5, symbol='circle'),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=predictions['datetime'],
        y=predictions['predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=WARNING_COLOR, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='diamond'),
        hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Date/Time",
        yaxis_title="Target Value",
        hovermode='x unified',
        plot_bgcolor='rgba(15, 23, 42, 0.95)',
        paper_bgcolor='rgba(15, 23, 42, 0.8)',
        font=dict(color=TEXT_COLOR, size=11),
        height=450,
        margin=dict(l=50, r=30, t=20, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor=PRIMARY_COLOR,
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            gridwidth=1,
            griddash='dot',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            gridwidth=1,
            griddash='dot',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.2)'
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="ml_forecast_plot")

    # Training metrics in expander
    with st.expander("📋 Training Metrics", expanded=False):
        train_metrics = results['train_metrics']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train MAE", f"{train_metrics['MAE']:.4f}")
        with col2:
            st.metric("Train RMSE", f"{train_metrics['RMSE']:.4f}")
        with col3:
            st.metric("Train MAPE", f"{train_metrics['MAPE']:.2f}%")
        with col4:
            st.metric("Train Accuracy", f"{train_metrics['Accuracy']:.2f}%")

def render_ml_metrics_only(ml_mh_results: dict, model_name: str):
    """
    Render only performance metrics (KPIs and table) without detailed graphs.
    Used for "All Models" comparison mode.

    Args:
        ml_mh_results: Multi-horizon results from run_ml_multihorizon()
        model_name: Name of the model (e.g., "XGBoost", "LSTM", "ANN")
    """
    if not ml_mh_results:
        st.warning(f"No results available for {model_name}.")
        return

    results_df = ml_mh_results.get("results_df")
    successful = ml_mh_results.get("successful", [])

    if results_df is None or results_df.empty or not successful:
        st.error(f"❌ **{model_name} Training Failed**")
        return

    # Average Performance Metrics (KPI Cards)
    col1, col2, col3, col4 = st.columns(4)

    avg_test_mae = results_df["Test_MAE"].mean()
    avg_test_rmse = results_df["Test_RMSE"].mean()
    avg_test_mape = results_df["Test_MAPE"].mean()
    avg_test_acc = results_df["Test_Acc"].mean()

    with col1:
        st.plotly_chart(
            _create_kpi_indicator("MAE", avg_test_mae, "", PRIMARY_COLOR),
            use_container_width=True,
            key=f"all_models_mae_{model_name}"
        )
    with col2:
        st.plotly_chart(
            _create_kpi_indicator("RMSE", avg_test_rmse, "", SECONDARY_COLOR),
            use_container_width=True,
            key=f"all_models_rmse_{model_name}"
        )
    with col3:
        st.plotly_chart(
            _create_kpi_indicator("MAPE", avg_test_mape, "%", WARNING_COLOR),
            use_container_width=True,
            key=f"all_models_mape_{model_name}"
        )
    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Accuracy", avg_test_acc, "%", SUCCESS_COLOR),
            use_container_width=True,
            key=f"all_models_acc_{model_name}"
        )

    # Compact metrics table
    with st.expander("📊 Per-Horizon Performance", expanded=False):
        compact_metrics = results_df[['Horizon', 'Test_MAE', 'Test_RMSE', 'Test_MAPE', 'Test_Acc']].copy()
        compact_metrics.columns = ['Horizon', 'MAE', 'RMSE', 'MAPE (%)', 'Accuracy (%)']

        st.dataframe(
            compact_metrics.style.format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAPE (%)': '{:.2f}',
                'Accuracy (%)': '{:.2f}'
            }),
            use_container_width=True,
            height=300
        )

def render_ml_multihorizon_results(ml_mh_results: dict, model_name: str):
    """
    Render multi-horizon ML training results with optimized fluorescent design.
    Shows full detailed graphs and visualizations.

    Args:
        ml_mh_results: Multi-horizon results from run_ml_multihorizon()
        model_name: Name of the model (e.g., "XGBoost", "LSTM", "ANN")
    """
    if not ml_mh_results:
        st.warning("No multi-horizon results available.")
        return

    results_df = ml_mh_results.get("results_df")
    successful = ml_mh_results.get("successful", [])
    errors = ml_mh_results.get("errors", {})
    per_h = ml_mh_results.get("per_h", {})  # For fallback error plots

    if results_df is None or results_df.empty or not successful:
        st.error(
            f"❌ **{model_name} Training Failed - No Successful Horizons**\n\n"
            "**Possible causes:**\n"
            "- Data format issues (check feature types)\n"
            "- Insufficient data (LSTM needs at least 15+ rows)\n"
            "- Missing or invalid features\n\n"
            "**Detailed errors below:**"
        )

        # Display errors for each horizon
        if errors:
            st.markdown("### 🔍 Error Details by Horizon")
            for horizon, error_msg in sorted(errors.items()):
                with st.expander(f"❌ Horizon {horizon} - Click to view error"):
                    st.code(error_msg, language="python")
        else:
            st.info("No detailed error information available. Check the terminal/console output.")

        return

    # Premium success banner - Left aligned and compact
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
                    border: 2px solid {SUCCESS_COLOR};
                    padding: 1.25rem 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 30px rgba(34, 197, 94, 0.3), inset 0 0 20px rgba(34, 197, 94, 0.1);'>
            <div style='display: flex; align-items: center;'>
                <div style='font-size: 2.5rem; margin-right: 1rem;'>✅</div>
                <div>
                    <h2 style='margin: 0 0 0.25rem 0; color: {SUCCESS_COLOR}; font-size: 1.3rem; font-weight: 800; text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);'>
                        Multi-Horizon {model_name} Training Complete!
                    </h2>
                    <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9; font-size: 0.95rem;'>
                        🎯 All {len(successful)} forecast horizons successfully trained and validated
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fluorescent KPI Cards - Left aligned and compact
    st.markdown(
        f"""
        <div style='text-align: left; margin: 1.5rem 0 1rem 0;'>
            <h3 style='font-size: 1.2rem; font-weight: 700; color: {PRIMARY_COLOR};
                       text-shadow: 0 0 20px rgba(59, 130, 246, 0.6); margin: 0 0 0.25rem 0;'>
                🏆 Average Performance Metrics
            </h3>
            <p style='color: {SUBTLE_TEXT}; margin: 0; font-size: 0.9rem;'>
                Averaged across all {len(successful)} forecast horizons
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    avg_test_mae = results_df["Test_MAE"].mean()
    avg_test_rmse = results_df["Test_RMSE"].mean()
    avg_test_mape = results_df["Test_MAPE"].mean()
    avg_test_acc = results_df["Test_Acc"].mean()

    with col1:
        st.plotly_chart(
            _create_kpi_indicator("MAE", avg_test_mae, "", PRIMARY_COLOR),
            use_container_width=True,
            key="ml_kpi_mae"
        )
    with col2:
        st.plotly_chart(
            _create_kpi_indicator("RMSE", avg_test_rmse, "", SECONDARY_COLOR),
            use_container_width=True,
            key="ml_kpi_rmse"
        )
    with col3:
        st.plotly_chart(
            _create_kpi_indicator("MAPE", avg_test_mape, "%", WARNING_COLOR),
            use_container_width=True,
            key="ml_kpi_mape"
        )
    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Accuracy", avg_test_acc, "%", SUCCESS_COLOR),
            use_container_width=True,
            key="ml_kpi_accuracy"
        )

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # Convert to artifacts for visualization
    metrics_df, F, L, U, test_eval_df, train, res, horizons = ml_to_multihorizon_artifacts(ml_mh_results)

    # Use the comprehensive multi-horizon dashboard from app_core.plots
    try:
        figs = build_multihorizon_results_dashboard(
            metrics_df=metrics_df,
            F=F,
            L=L,
            U=U,
            test_eval=test_eval_df,
            train=train,
            res=res,
            horizons=horizons,
        )

        # Optimized vertical layout with fluorescent sections
        if figs:
            # ===== SECTION 1: Performance Metrics Table (Compact) =====
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
                            border-left: 4px solid {PRIMARY_COLOR};
                            padding: 1rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
                    <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 0.75rem 0; font-size: 1.2rem; font-weight: 700;
                               text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                        📊 Per-Horizon Performance Breakdown
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "metrics_styler" in figs:
                # Create compact view - only show key columns
                compact_metrics = metrics_df[['Horizon', 'MAE', 'RMSE', 'MAPE_%', 'Accuracy_%']].copy()
                compact_metrics.columns = ['Horizon', 'MAE', 'RMSE', 'MAPE (%)', 'Accuracy (%)']

                st.dataframe(
                    compact_metrics.style.format({
                        'MAE': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'MAPE (%)': '{:.2f}',
                        'Accuracy (%)': '{:.2f}'
                    }).background_gradient(cmap='viridis', subset=['MAE', 'RMSE'])
                     .background_gradient(cmap='plasma', subset=['MAPE (%)', 'Accuracy (%)']),
                    use_container_width=True,
                    height=350
                )

            # Metrics Dashboard Visualization (Full Width)
            if "fig_metrics_dashboard" in figs and hasattr(figs["fig_metrics_dashboard"], 'savefig'):
                st.pyplot(figs["fig_metrics_dashboard"])

            st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

            # ===== SECTION 2: Forecast Visualizations (Vertical Layout) =====
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, rgba(167, 139, 250, 0.15), rgba(139, 92, 246, 0.1));
                            border-left: 4px solid {WARNING_COLOR};
                            padding: 1rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            box-shadow: 0 0 20px rgba(167, 139, 250, 0.2);'>
                    <h3 style='color: {WARNING_COLOR}; margin: 0 0 0.75rem 0; font-size: 1.2rem; font-weight: 700;
                               text-shadow: 0 0 15px rgba(167, 139, 250, 0.5);'>
                        📈 Forecast Analysis & Visualization
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Multi-Horizon Forecasts (Full Width)
            if "fig_forecast_wall" in figs and hasattr(figs["fig_forecast_wall"], 'savefig'):
                st.markdown("**🌊 Multi-Horizon Forecasts Comparison**")
                st.pyplot(figs["fig_forecast_wall"])
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            # Forecast Fan Chart (Full Width)
            if "fig_fanchart" in figs and hasattr(figs["fig_fanchart"], 'savefig'):
                st.markdown("**📊 Forecast Fan Chart - All Horizons**")
                st.pyplot(figs["fig_fanchart"])
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            # Temporal Split (Full Width)
            if "fig_temporal" in figs and hasattr(figs["fig_temporal"], 'savefig'):
                st.markdown("**⏳ Training/Test Split Visualization**")
                st.pyplot(figs["fig_temporal"])

            st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

            # ===== SECTION 3: Prediction Quality =====
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, rgba(34, 211, 238, 0.15), rgba(6, 182, 212, 0.1));
                            border-left: 4px solid {SECONDARY_COLOR};
                            padding: 1rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);'>
                    <h3 style='color: {SECONDARY_COLOR}; margin: 0 0 0.75rem 0; font-size: 1.2rem; font-weight: 700;
                               text-shadow: 0 0 15px rgba(34, 211, 238, 0.5);'>
                        🎯 Prediction Quality Assessment
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "fig_pred_vs_actual" in figs and hasattr(figs["fig_pred_vs_actual"], 'savefig'):
                st.pyplot(figs["fig_pred_vs_actual"])

            st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

            # ===== SECTION 4: Error Analysis (Vertical Layout) =====
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(251, 146, 60, 0.1));
                            border-left: 4px solid {DANGER_COLOR};
                            padding: 1rem;
                            border-radius: 12px;
                            margin-bottom: 1.5rem;
                            box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);'>
                    <h3 style='color: {DANGER_COLOR}; margin: 0 0 0.75rem 0; font-size: 1.2rem; font-weight: 700;
                               text-shadow: 0 0 15px rgba(245, 158, 11, 0.5);'>
                        📉 Error Distribution Analysis
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Error Box Plot (Full Width)
            if "fig_err_box" in figs and hasattr(figs.get("fig_err_box"), 'savefig'):
                st.markdown("**📦 Error Distribution (Box Plot)**")
                st.pyplot(figs["fig_err_box"])
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            else:
                # Fallback: Create error plot manually if not available
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    # Calculate errors for each horizon
                    errors_by_horizon = {}
                    for h in successful:
                        if h in per_h:
                            y_test = per_h[h].get("y_test")
                            forecast = per_h[h].get("forecast")
                            if y_test is not None and forecast is not None:
                                errors_by_horizon[f"H{h}"] = y_test - forecast

                    if errors_by_horizon:
                        st.markdown("**📦 Error Distribution (Box Plot)**")
                        fig, ax = plt.subplots(figsize=(12, 6))

                        # Create box plot
                        data_to_plot = [errors for errors in errors_by_horizon.values()]
                        labels = list(errors_by_horizon.keys())

                        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                        # Style the plot
                        for patch in bp['boxes']:
                            patch.set_facecolor('#3b82f6')
                            patch.set_alpha(0.6)

                        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
                        ax.set_xlabel('Horizon', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Prediction Error', fontsize=12, fontweight='bold')
                        ax.set_title('Error Distribution by Horizon', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                except Exception as plot_error:
                    st.info(f"Error distribution plot not available: {str(plot_error)}")

            # Error KDE (Full Width)
            if "fig_err_kde" in figs and hasattr(figs.get("fig_err_kde"), 'savefig'):
                st.markdown("**🌊 Error Density Distribution (KDE)**")
                st.pyplot(figs["fig_err_kde"])
            else:
                # Fallback: Create KDE plot manually if not available
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    # Calculate errors for each horizon
                    errors_by_horizon = {}
                    for h in successful:
                        if h in per_h:
                            y_test = per_h[h].get("y_test")
                            forecast = per_h[h].get("forecast")
                            if y_test is not None and forecast is not None:
                                errors_by_horizon[f"Horizon {h}"] = y_test - forecast

                    if errors_by_horizon:
                        st.markdown("**🌊 Error Density Distribution (KDE)**")
                        fig, ax = plt.subplots(figsize=(12, 6))

                        # Create KDE plot for each horizon
                        colors = plt.cm.viridis(np.linspace(0, 1, len(errors_by_horizon)))

                        for (label, errors), color in zip(errors_by_horizon.items(), colors):
                            sns.kdeplot(data=errors, label=label, ax=ax, color=color, linewidth=2, alpha=0.7)

                        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
                        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
                        ax.set_title('Error Density Distribution (KDE)', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='best')

                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as plot_error:
                    st.info(f"Error KDE plot not available: {str(plot_error)}")

    except Exception as e:
        # Fallback to basic metrics table if dashboard fails
        st.warning(f"⚠️ Could not generate full dashboard visualization: {str(e)}")

        st.markdown("### 📊 Performance Metrics by Horizon")
        st.dataframe(
            results_df.style.format({
                "Train_MAE": "{:.4f}",
                "Train_RMSE": "{:.4f}",
                "Train_MAPE": "{:.2f}",
                "Train_Acc": "{:.2f}",
                "Test_MAE": "{:.4f}",
                "Test_RMSE": "{:.4f}",
                "Test_MAPE": "{:.2f}",
                "Test_Acc": "{:.2f}",
            }),
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# MACHINE LEARNING (LSTM / ANN / XGBoost)
# -----------------------------------------------------------------------------
def page_ml():
    # Import ML UI components
    from app_core.ui.ml_components import MLUIComponents

    # Header (left-aligned like Dashboard)
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: left; margin-bottom: 2rem;'>
          <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>🧮</div>
            <h1 class='hf-feature-title' style='font-size: 2.5rem; margin: 0;'>Machine Learning</h1>
          </div>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 0 0 4.5rem;'>
            Configure and train advanced machine learning models including LSTM, ANN, and XGBoost for sophisticated forecasting
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model Selection (Generic - replaces 15 lines with 3)
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
                    border-left: 4px solid rgba(168, 85, 247, 0.8);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 2rem 0 1.5rem 0;
                    box-shadow: 0 0 20px rgba(168, 85, 247, 0.2);'>
            <h3 style='color: rgba(168, 85, 247, 1); margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(168, 85, 247, 0.5);'>
                🤖 Model Selection
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Training mode selection
    col1, col2 = st.columns([2, 1])
    with col1:
        training_mode = st.radio(
            "Training Mode",
            options=["Single Model", "All Models (XGBoost + LSTM + ANN)"],
            index=0 if cfg.get("ml_training_mode", "Single Model") == "Single Model" else 1,
            horizontal=True,
            help="Choose to train a single model or all three models at once"
        )
        cfg["ml_training_mode"] = training_mode

    # Only show model selector if training single model
    if training_mode == "Single Model":
        selected_model = MLUIComponents.render_model_selector(cfg.get("ml_choice", "XGBoost"))
        cfg["ml_choice"] = selected_model
    else:
        st.info("🚀 **All Models Mode**: XGBoost, LSTM, and ANN will be trained sequentially with default configurations")
        selected_model = None  # Not needed for all models mode

    # 2. Dataset Selection (from Feature Selection results)
    datasets, metadata = get_feature_selection_datasets()
    selected_df = None  # Initialize for later use in target/feature selection

    if datasets:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(6, 182, 212, 0.1));
                        border-left: 4px solid rgba(34, 211, 238, 0.8);
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 2rem 0 1.5rem 0;
                        box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);'>
                <h3 style='color: rgba(34, 211, 238, 1); margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                           text-shadow: 0 0 15px rgba(34, 211, 238, 0.5);'>
                    📊 Dataset Selection
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        dataset_options = list(datasets.keys())
        selected_dataset_name = st.selectbox(
            "Select feature-selected dataset",
            options=dataset_options,
            help="Choose a dataset from Automated Feature Selection results"
        )
        selected_df = datasets[selected_dataset_name]

        # Show dataset info in fluorescent KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("Features", len(selected_df.columns) - 7, "", PRIMARY_COLOR),
                use_container_width=True,
                key="dataset_kpi_features"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("Samples", len(selected_df), "", SUCCESS_COLOR),
                use_container_width=True,
                key="dataset_kpi_samples"
            )
        with col3:
            # For variant, we'll create a special styled card for text value
            variant_name = metadata.get("variant", "Unknown")
            st.markdown(
                f"""
                <div style='background: rgba(15, 23, 42, 0.8);
                            border-radius: 12px;
                            padding: 1.25rem;
                            text-align: center;
                            box-shadow: 0 0 25px rgba(245, 158, 11, 0.3);
                            border: 1px solid rgba(245, 158, 11, 0.2);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;'>
                    <div style='color: {TEXT_COLOR}; font-size: 12px; font-weight: 700;
                                letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                        VARIANT
                    </div>
                    <div style='color: {WARNING_COLOR}; font-size: 20px; font-weight: 800;
                                text-shadow: 0 0 15px rgba(245, 158, 11, 0.6);'>
                        {variant_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Preview
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(selected_df.head(10), use_container_width=True)

        # Store selected dataset for training
        cfg["selected_dataset"] = selected_dataset_name
        cfg["training_data"] = selected_df
    else:
        st.warning("⚠️ No feature selection results found. Please run **Automated Feature Selection** (Page 07) first.")
        data_source_placeholder()

    # 3. Manual Configuration - Only show for Single Model mode
    if training_mode == "Single Model":
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
                        border-left: 4px solid {PRIMARY_COLOR};
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 2rem 0 1.5rem 0;
                        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
                <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                           text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                    ⚙️ Model Configuration
                </h3>
                <p style='margin: 0; color: {SUBTLE_TEXT}; font-size: 0.9rem;'>
                    Configure hyperparameters manually. For automated optimization, use the <strong>Hyperparameter Tuning</strong> tab.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Manual params (model-specific, but rendered generically)
        manual_params = MLUIComponents.render_manual_params(selected_model, cfg)
        cfg.update(manual_params)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Train/test split ratio - Full width with better styling
    st.markdown(
        f"""
        <div style='background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
                    border: 1px solid rgba(59, 130, 246, 0.2);'>
            <p style='color: {TEXT_COLOR}; font-weight: 600; margin: 0 0 0.5rem 0; font-size: 0.95rem;'>
                📊 Train/Validation Split Ratio
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    cfg["split_ratio"] = st.slider(
        "Split ratio",
        0.50, 0.95,
        float(cfg.get("split_ratio", 0.80)),
        0.01,
        key="ml_split_ratio",
        help="Proportion of data to use for training",
        label_visibility="collapsed"
    )
    st.caption(f"🎯 Training: {int(cfg['split_ratio']*100)}% | Validation: {int((1-cfg['split_ratio'])*100)}%")

    # Auto-configure horizons (default to 7)
    cfg["ml_horizons"] = 7

    # Auto-select all feature columns
    if selected_df is not None and not selected_df.empty:
        all_columns = selected_df.columns.tolist()
        feature_columns = [col for col in all_columns if not col.startswith("Target_")]
        cfg["ml_feature_cols"] = feature_columns
    else:
        cfg["ml_feature_cols"] = []

    # 7. Run Model Button
    st.divider()

    # Check if all required data is available
    can_run = all([
        selected_df is not None,
        cfg.get("ml_feature_cols") and len(cfg.get("ml_feature_cols", [])) > 0
    ])

    if can_run:
        # Different button based on training mode
        if training_mode == "Single Model":
            # Single Model Training
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_button = st.button(
                    f"🚀 Train Multi-Horizon {cfg['ml_choice']}",
                    type="primary",
                    use_container_width=True,
                    help=f"Train {cfg.get('ml_horizons', 7)} models for horizons 1-{cfg.get('ml_horizons', 7)}"
                )

            if run_button:
                with st.spinner(f"Training {cfg['ml_choice']} for {cfg.get('ml_horizons', 7)} horizons..."):
                    try:
                        results = run_ml_multihorizon(
                            model_type=cfg['ml_choice'],
                            config=cfg,
                            df=selected_df,
                            feature_cols=cfg['ml_feature_cols'],
                            split_ratio=cfg.get('split_ratio', 0.8),
                            horizons=cfg.get('ml_horizons', 7),
                        )

                        # Store results in session state
                        st.session_state["ml_mh_results"] = results

                        # Auto-save to cache for persistence
                        try:
                            cache_mgr = get_cache_manager()
                            proc_df = st.session_state.get("processed_df")
                            data_hash = cache_mgr._compute_data_hash(proc_df) if proc_df is not None else None
                            save_to_cache("ml_mh_results", results, data_hash)
                        except Exception:
                            pass  # Cache save is optional
                    except Exception as e:
                        st.error(f"❌ **Training failed with error:**\n\n```\n{str(e)}\n```")
                        import traceback
                        st.code(traceback.format_exc())

            # Display results if available
            if "ml_mh_results" in st.session_state:
                st.divider()
                render_ml_multihorizon_results(st.session_state["ml_mh_results"], cfg['ml_choice'])

                # Add CQR prediction intervals section
                ml_results = st.session_state["ml_mh_results"]
                render_cqr_section(ml_results, cfg['ml_choice'])

                # Add probability-based category forecast section
                per_h = ml_results.get("per_h", {})
                successful = ml_results.get("successful", [])

                if per_h and successful:
                    # Extract average forecast value for each horizon
                    horizon_predictions = []
                    for h in sorted(successful):
                        h_data = per_h.get(h, {})
                        forecast = h_data.get("forecast")
                        if forecast is not None and len(forecast) > 0:
                            # Use the last forecast value (most recent prediction for this horizon)
                            horizon_predictions.append(forecast[-1])

                    if horizon_predictions:
                        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                        _render_category_forecast_section(
                            horizon_predictions,
                            f"{cfg['ml_choice']} - Clinical Category Forecast (Probability-Based)"
                        )

        else:
            # All Models Training
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_all_button = st.button(
                    "🚀 Train All Models (XGBoost + LSTM + ANN)",
                    type="primary",
                    use_container_width=True,
                    help=f"Train all three models for {cfg.get('ml_horizons', 7)} horizons each"
                )

            if run_all_button:
                all_models = ["XGBoost", "LSTM", "ANN"]
                all_results = {}

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, model_name in enumerate(all_models):
                    status_text.markdown(f"**Training {idx+1}/3: {model_name}...**")
                    progress_bar.progress((idx) / len(all_models))

                    with st.spinner(f"Training {model_name} for {cfg.get('ml_horizons', 7)} horizons..."):
                        try:
                            results = run_ml_multihorizon(
                                model_type=model_name,
                                config=cfg,
                                df=selected_df,
                                feature_cols=cfg['ml_feature_cols'],
                                split_ratio=cfg.get('split_ratio', 0.8),
                                horizons=cfg.get('ml_horizons', 7),
                            )

                            # Store results for each model
                            all_results[model_name] = results
                            st.session_state[f"ml_mh_results_{model_name}"] = results

                        except Exception as e:
                            st.error(f"❌ **{model_name} training failed:**\n\n```\n{str(e)}\n```")
                            import traceback
                            st.code(traceback.format_exc())
                            all_results[model_name] = None

                # Complete progress
                progress_bar.progress(1.0)
                status_text.markdown("✅ **All models training completed!**")

                # Store flag that all models were trained
                st.session_state["ml_all_models_trained"] = True

            # Display results for all models if available
            if st.session_state.get("ml_all_models_trained"):
                st.divider()
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
                                border: 2px solid {SUCCESS_COLOR};
                                padding: 1.25rem;
                                border-radius: 12px;
                                margin-bottom: 2rem;
                                text-align: center;
                                box-shadow: 0 0 30px rgba(34, 197, 94, 0.3);'>
                        <h2 style='margin: 0; color: {SUCCESS_COLOR}; font-size: 1.5rem; font-weight: 800;'>
                            📊 All Models Comparison
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Display each model's results (metrics only, no detailed graphs)
                for model_name in ["XGBoost", "LSTM", "ANN"]:
                    if f"ml_mh_results_{model_name}" in st.session_state:
                        results = st.session_state[f"ml_mh_results_{model_name}"]
                        if results:
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05));
                                            border-left: 4px solid {PRIMARY_COLOR};
                                            padding: 1rem;
                                            border-radius: 10px;
                                            margin-bottom: 1rem;'>
                                    <h3 style='color: {PRIMARY_COLOR}; margin: 0; font-size: 1.2rem;'>
                                        🤖 {model_name} Performance
                                    </h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            render_ml_metrics_only(results, model_name)

                            # Add CQR prediction intervals section (compact for comparison)
                            if results.get("has_cqr_data"):
                                with st.expander(f"📊 {model_name} Prediction Intervals (CQR)", expanded=False):
                                    render_cqr_section(results, model_name)

                            # Add probability-based category forecast for each model
                            per_h = results.get("per_h", {})
                            successful = results.get("successful", [])

                            if per_h and successful:
                                horizon_predictions = []
                                for h in sorted(successful):
                                    h_data = per_h.get(h, {})
                                    forecast = h_data.get("forecast")
                                    if forecast is not None and len(forecast) > 0:
                                        horizon_predictions.append(forecast[-1])

                                if horizon_predictions:
                                    with st.expander(f"📊 {model_name} Category Forecast (Probability-Based)", expanded=False):
                                        _render_category_forecast_section(
                                            horizon_predictions,
                                            f"Clinical Category Distribution"
                                        )

                            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    else:
        st.warning(
            "⚠️ **Cannot run model**\n\n"
            "Please ensure:\n"
            f"- {'✅' if selected_df is not None else '❌'} Dataset is selected\n"
            f"- {'✅' if cfg.get('ml_feature_cols') and len(cfg.get('ml_feature_cols', [])) > 0 else '❌'} Features are selected"
        )

    # 7. Debug Panel (Preserved)
    debug_panel()

# -----------------------------------------------------------------------------
# HYPERPARAMETER OPTIMIZATION HELPERS
# -----------------------------------------------------------------------------
def run_grid_search_optimization(model_type: str, X_train, y_train, n_splits: int,
                                 primary_metric: str, param_grid: dict, progress_callback=None):
    """
    Run Grid Search optimization with TimeSeriesSplit CV.

    Args:
        model_type: "XGBoost", "LSTM", or "ANN"
        X_train: Training features
        y_train: Training target
        n_splits: Number of CV folds
        primary_metric: Metric to optimize ("RMSE", "MAE", "MAPE", "Accuracy")
        param_grid: Parameter grid to search
        progress_callback: Optional callback for progress updates

    Returns:
        dict with best_params, cv_results, and all metrics
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
    import numpy as np

    # Define custom scorers
    def mape_scorer(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def accuracy_scorer(y_true, y_pred):
        """Accuracy = 100 - MAPE"""
        mape = mape_scorer(y_true, y_pred)
        return 100 - mape

    def rmse_scorer(y_true, y_pred):
        """Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Create scoring dictionary
    scoring = {
        'RMSE': make_scorer(rmse_scorer, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MAPE': make_scorer(mape_scorer, greater_is_better=False),
        'Accuracy': make_scorer(accuracy_scorer, greater_is_better=True),
    }

    # Map primary metric to refit parameter
    refit_map = {
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'MAPE': 'MAPE',
        'Accuracy': 'Accuracy'
    }
    refit_metric = refit_map.get(primary_metric, 'RMSE')

    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Build model based on type
    if model_type == "XGBoost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(random_state=42, n_jobs=-1)

    elif model_type == "LSTM":
        from app_core.models.ml.lstm_pipeline import LSTMPipeline
        # LSTM requires special handling - wrap in sklearn-compatible estimator
        base_model = create_sklearn_lstm_wrapper()

    elif model_type == "ANN":
        from app_core.models.ml.ann_pipeline import ANNPipeline
        base_model = create_sklearn_ann_wrapper()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # Extract results
    cv_results = pd.DataFrame(grid_search.cv_results_)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': cv_results,
        'refit_metric': refit_metric,
        'all_scores': {
            'RMSE': -cv_results[f'mean_test_RMSE'].iloc[grid_search.best_index_],
            'MAE': -cv_results[f'mean_test_MAE'].iloc[grid_search.best_index_],
            'MAPE': -cv_results[f'mean_test_MAPE'].iloc[grid_search.best_index_],
            'Accuracy': cv_results[f'mean_test_Accuracy'].iloc[grid_search.best_index_],
        }
    }

def create_sklearn_lstm_wrapper():
    """Create sklearn-compatible wrapper for LSTM pipeline"""
    from sklearn.base import BaseEstimator, RegressorMixin

    class LSTMWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, lookback_window=14, lstm_hidden_units=64, lstm_layers=2,
                     dropout=0.2, lstm_epochs=50, lstm_lr=0.001, batch_size=32):
            self.lookback_window = lookback_window
            self.lstm_hidden_units = lstm_hidden_units
            self.lstm_layers = lstm_layers
            self.dropout = dropout
            self.lstm_epochs = lstm_epochs
            self.lstm_lr = lstm_lr
            self.batch_size = batch_size
            self.pipeline = None

        def fit(self, X, y):
            from app_core.models.ml.lstm_pipeline import LSTMPipeline
            config = {
                'lookback_window': self.lookback_window,
                'lstm_hidden_units': self.lstm_hidden_units,
                'lstm_layers': self.lstm_layers,
                'dropout': self.dropout,
                'lstm_epochs': self.lstm_epochs,
                'lstm_lr': self.lstm_lr,
                'batch_size': self.batch_size
            }
            self.pipeline = LSTMPipeline(config)
            self.pipeline.build_model(X.shape[1])
            self.pipeline.train(X, y, X, y)  # No validation split for CV
            return self

        def predict(self, X):
            if self.pipeline is None:
                raise ValueError("Model not fitted yet")
            return self.pipeline.predict(X)

    return LSTMWrapper()

def create_sklearn_ann_wrapper():
    """Create sklearn-compatible wrapper for ANN pipeline"""
    from sklearn.base import BaseEstimator, RegressorMixin

    class ANNWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, ann_hidden_layers=2, ann_neurons=64, ann_activation='relu',
                     dropout=0.2, ann_epochs=30):
            self.ann_hidden_layers = ann_hidden_layers
            self.ann_neurons = ann_neurons
            self.ann_activation = ann_activation
            self.dropout = dropout
            self.ann_epochs = ann_epochs
            self.pipeline = None

        def fit(self, X, y):
            from app_core.models.ml.ann_pipeline import ANNPipeline
            config = {
                'ann_hidden_layers': self.ann_hidden_layers,
                'ann_neurons': self.ann_neurons,
                'ann_activation': self.ann_activation,
                'dropout': self.dropout,
                'ann_epochs': self.ann_epochs
            }
            self.pipeline = ANNPipeline(config)
            self.pipeline.build_model(X.shape[1])
            self.pipeline.train(X, y, X, y)  # No validation split for CV
            return self

        def predict(self, X):
            if self.pipeline is None:
                raise ValueError("Model not fitted yet")
            return self.pipeline.predict(X)

    return ANNWrapper()

def get_param_grid(model_type: str, cfg: dict):
    """Get parameter grid for Grid Search based on model type.

    Optimized grids for faster training:
    - XGBoost: 8 combinations (was 108)
    - LSTM: 8 combinations (was 72)
    - ANN: 8 combinations (was 72)
    """

    if model_type == "XGBoost":
        return {
            'n_estimators': [200, 400],
            'max_depth': [4, 8],
            'learning_rate': [0.05, 0.1],
        }

    elif model_type == "LSTM":
        return {
            'lookback_window': [7, 14],
            'lstm_hidden_units': [64, 128],
            'lstm_layers': [2],
            'dropout': [0.2],
            'lstm_epochs': [30],
        }

    elif model_type == "ANN":
        return {
            'ann_hidden_layers': [2, 3],
            'ann_neurons': [64, 128],
            'ann_activation': ['relu'],
            'dropout': [0.2],
            'ann_epochs': [25],
        }

    return {}

def get_param_distributions(model_type: str, cfg: dict):
    """Get parameter distributions for Random Search based on model type"""
    from scipy.stats import randint, uniform

    if model_type == "XGBoost":
        return {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.30
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'min_child_weight': randint(1, 10),
        }

    elif model_type == "LSTM":
        return {
            'lookback_window': randint(5, 30),
            'lstm_hidden_units': [32, 64, 128, 256],
            'lstm_layers': randint(1, 4),
            'dropout': uniform(0.1, 0.4),  # 0.1 to 0.5
            'lstm_epochs': randint(20, 100),
        }

    elif model_type == "ANN":
        return {
            'ann_hidden_layers': randint(2, 6),
            'ann_neurons': [32, 64, 128, 256],
            'ann_activation': ['relu', 'tanh', 'sigmoid'],
            'dropout': uniform(0.1, 0.4),
            'ann_epochs': randint(20, 60),
        }

    return {}

def run_random_search_optimization(model_type: str, X_train, y_train, n_splits: int,
                                   primary_metric: str, param_distributions: dict,
                                   n_iter: int = 20, progress_callback=None):
    """
    Run Random Search optimization with TimeSeriesSplit CV.

    Args:
        model_type: "XGBoost", "LSTM", or "ANN"
        X_train: Training features
        y_train: Training target
        n_splits: Number of CV folds
        primary_metric: Metric to optimize
        param_distributions: Parameter distributions to sample
        n_iter: Number of iterations
        progress_callback: Optional callback for progress updates

    Returns:
        dict with best_params, cv_results, and all metrics
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
    import numpy as np

    # Define custom scorers (same as Grid Search)
    def mape_scorer(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def accuracy_scorer(y_true, y_pred):
        mape = mape_scorer(y_true, y_pred)
        return 100 - mape

    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    scoring = {
        'RMSE': make_scorer(rmse_scorer, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MAPE': make_scorer(mape_scorer, greater_is_better=False),
        'Accuracy': make_scorer(accuracy_scorer, greater_is_better=True),
    }

    refit_map = {'RMSE': 'RMSE', 'MAE': 'MAE', 'MAPE': 'MAPE', 'Accuracy': 'Accuracy'}
    refit_metric = refit_map.get(primary_metric, 'RMSE')

    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Build model
    if model_type == "XGBoost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
    elif model_type == "LSTM":
        base_model = create_sklearn_lstm_wrapper()
    elif model_type == "ANN":
        base_model = create_sklearn_ann_wrapper()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run Random Search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    # Extract results
    cv_results = pd.DataFrame(random_search.cv_results_)

    return {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'best_estimator': random_search.best_estimator_,
        'cv_results': cv_results,
        'refit_metric': refit_metric,
        'all_scores': {
            'RMSE': -cv_results[f'mean_test_RMSE'].iloc[random_search.best_index_],
            'MAE': -cv_results[f'mean_test_MAE'].iloc[random_search.best_index_],
            'MAPE': -cv_results[f'mean_test_MAPE'].iloc[random_search.best_index_],
            'Accuracy': cv_results[f'mean_test_Accuracy'].iloc[random_search.best_index_],
        }
    }

def run_bayesian_optimization(model_type: str, X_train, y_train, n_splits: int,
                              primary_metric: str, n_trials: int = 30,
                              progress_callback=None):
    """
    Run Bayesian Optimization using Optuna with TimeSeriesSplit CV.

    Args:
        model_type: "XGBoost", "LSTM", or "ANN"
        X_train: Training features
        y_train: Training target
        n_splits: Number of CV folds
        primary_metric: Metric to optimize
        n_trials: Number of Optuna trials
        progress_callback: Optional callback for progress updates

    Returns:
        dict with best_params, trial_results, and all metrics
    """
    import optuna
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
    import numpy as np

    # Define custom scorers
    def mape_scorer(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def accuracy_scorer(y_true, y_pred):
        mape = mape_scorer(y_true, y_pred)
        return 100 - mape

    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Map primary metric to scorer
    if primary_metric == "RMSE":
        scorer = make_scorer(rmse_scorer, greater_is_better=False)
        direction = "minimize"
    elif primary_metric == "MAE":
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        direction = "minimize"
    elif primary_metric == "MAPE":
        scorer = make_scorer(mape_scorer, greater_is_better=False)
        direction = "minimize"
    elif primary_metric == "Accuracy":
        scorer = make_scorer(accuracy_scorer, greater_is_better=True)
        direction = "maximize"
    else:
        scorer = make_scorer(rmse_scorer, greater_is_better=False)
        direction = "minimize"

    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define objective function
    def objective(trial):
        # Suggest hyperparameters based on model type
        if model_type == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=42, n_jobs=-1, **params)

        elif model_type == "LSTM":
            params = {
                'lookback_window': trial.suggest_int('lookback_window', 5, 30),
                'lstm_hidden_units': trial.suggest_categorical('lstm_hidden_units', [32, 64, 128, 256]),
                'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'lstm_epochs': trial.suggest_int('lstm_epochs', 20, 100),
            }
            model = create_sklearn_lstm_wrapper()
            model.set_params(**params)

        elif model_type == "ANN":
            params = {
                'ann_hidden_layers': trial.suggest_int('ann_hidden_layers', 2, 5),
                'ann_neurons': trial.suggest_categorical('ann_neurons', [32, 64, 128, 256]),
                'ann_activation': trial.suggest_categorical('ann_activation', ['relu', 'tanh']),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'ann_epochs': trial.suggest_int('ann_epochs', 20, 60),
            }
            model = create_sklearn_ann_wrapper()
            model.set_params(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scorer, n_jobs=-1)
        return scores.mean()

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get best parameters
    best_params = study.best_params

    # Train final model with best params and get all metrics
    if model_type == "XGBoost":
        from xgboost import XGBRegressor
        final_model = XGBRegressor(random_state=42, n_jobs=-1, **best_params)
    elif model_type == "LSTM":
        final_model = create_sklearn_lstm_wrapper()
        final_model.set_params(**best_params)
    elif model_type == "ANN":
        final_model = create_sklearn_ann_wrapper()
        final_model.set_params(**best_params)

    final_model.fit(X_train, y_train)

    # Compute all metrics on cross-validation
    from sklearn.model_selection import cross_validate

    all_scorers = {
        'RMSE': make_scorer(rmse_scorer, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MAPE': make_scorer(mape_scorer, greater_is_better=False),
        'Accuracy': make_scorer(accuracy_scorer, greater_is_better=True),
    }

    cv_results = cross_validate(final_model, X_train, y_train, cv=tscv, scoring=all_scorers, return_train_score=False)

    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'best_estimator': final_model,
        'study': study,
        'refit_metric': primary_metric,
        'all_scores': {
            'RMSE': -cv_results['test_RMSE'].mean(),
            'MAE': -cv_results['test_MAE'].mean(),
            'MAPE': -cv_results['test_MAPE'].mean(),
            'Accuracy': cv_results['test_Accuracy'].mean(),
        },
        'trial_results': pd.DataFrame({
            'trial': range(len(study.trials)),
            'value': [t.value for t in study.trials],
            'params': [t.params for t in study.trials],
        })
    }

def run_7day_backtesting(model_type: str, best_params: dict, X, y, dates,
                         n_windows: int = 5, horizon: int = 7):
    """
    Perform 7-day rolling-origin backtesting.

    Args:
        model_type: Type of model ('XGBoost', 'LSTM', 'ANN')
        best_params: Best hyperparameters from optimization
        X: Full feature array
        y: Full target array
        dates: Date array corresponding to data points
        n_windows: Number of backtesting windows
        horizon: Forecast horizon (days)

    Returns:
        dict with aggregate metrics and window results
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    # Validate inputs
    if len(X) < 20:  # Minimum data requirement
        return {
            'aggregate_metrics': {
                'RMSE': None,
                'MAE': None,
                'MAPE': None,
                'Accuracy': None
            },
            'window_results': [],
            'n_successful_windows': 0,
            'error': f'Insufficient data: only {len(X)} samples available, need at least 20'
        }

    # Calculate window starting points (evenly spaced in the latter half of data)
    data_len = len(X)
    min_train_size = max(int(data_len * 0.5), 10)  # Use at least 50% for training, minimum 10 samples
    max_train_size = data_len - horizon  # Leave room for horizon

    # Check if we have enough data for even one window
    if max_train_size < min_train_size:
        return {
            'aggregate_metrics': {
                'RMSE': None,
                'MAE': None,
                'MAPE': None,
                'Accuracy': None
            },
            'window_results': [],
            'n_successful_windows': 0,
            'error': f'Insufficient data: need at least {min_train_size + horizon} samples, have {data_len}'
        }

    # Generate cutoff indices for rolling windows
    if n_windows == 1:
        cutoff_indices = [max_train_size]
    else:
        step = max((max_train_size - min_train_size) // (n_windows - 1), 1)
        cutoff_indices = list(range(min_train_size, max_train_size + 1, step))[:n_windows]

    window_results = []
    all_y_true = []
    all_y_pred = []
    failed_windows = []

    for window_idx, cutoff_idx in enumerate(cutoff_indices):
        # Split data at cutoff
        X_train = X[:cutoff_idx]
        y_train = y[:cutoff_idx]
        X_test = X[cutoff_idx:cutoff_idx + horizon]
        y_test = y[cutoff_idx:cutoff_idx + horizon]

        # Skip if not enough test data
        if len(X_test) < horizon:
            failed_windows.append({
                'window': window_idx + 1,
                'reason': f'Not enough test data: {len(X_test)} < {horizon}'
            })
            continue

        try:
            # Build and train model with best params
            if model_type == "XGBoost":
                from app_core.models.ml.xgboost_pipeline import XGBoostPipeline

                # Create config dict for XGBoostPipeline
                config = {
                    'n_estimators': best_params.get('n_estimators', 300),
                    'max_depth': best_params.get('max_depth', 6),
                    'eta': best_params.get('learning_rate', 0.1),  # XGBoost uses 'eta' not 'learning_rate'
                    'subsample': best_params.get('subsample', 0.8),
                    'colsample_bytree': best_params.get('colsample_bytree', 0.8),
                    'min_child_weight': best_params.get('min_child_weight', 1),
                    'gamma': best_params.get('gamma', 0),
                }

                pipeline = XGBoostPipeline(config)
                pipeline.build_model()
                pipeline.train(X_train, y_train)
                y_pred = pipeline.predict(X_test)

            elif model_type == "LSTM":
                # Use the LSTM wrapper
                wrapper = create_sklearn_lstm_wrapper()
                wrapper.set_params(**best_params)
                wrapper.fit(X_train, y_train)
                y_pred = wrapper.predict(X_test)

            elif model_type == "ANN":
                # Use the ANN wrapper
                wrapper = create_sklearn_ann_wrapper()
                wrapper.set_params(**best_params)
                wrapper.fit(X_train, y_train)
                y_pred = wrapper.predict(X_test)

            # Calculate metrics for this window
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            accuracy = 100 - mape

            window_results.append({
                'window': window_idx + 1,
                'cutoff_idx': cutoff_idx,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Accuracy': accuracy,
                'y_true': y_test,
                'y_pred': y_pred,
                'dates': dates[cutoff_idx:cutoff_idx + horizon] if dates is not None else None
            })

            # Collect all predictions for aggregate metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        except Exception as e:
            # Capture error details for debugging
            import traceback
            failed_windows.append({
                'window': window_idx + 1,
                'reason': f'{type(e).__name__}: {str(e)}',
                'traceback': traceback.format_exc()
            })
            continue

    # Calculate aggregate metrics across all windows
    if len(all_y_true) > 0:
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        aggregate_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        aggregate_mae = mean_absolute_error(all_y_true, all_y_pred)
        aggregate_mape = np.mean(np.abs((all_y_true - all_y_pred) / (all_y_true + 1e-10))) * 100
        aggregate_accuracy = 100 - aggregate_mape
    else:
        aggregate_rmse = aggregate_mae = aggregate_mape = aggregate_accuracy = None

    return {
        'aggregate_metrics': {
            'RMSE': aggregate_rmse,
            'MAE': aggregate_mae,
            'MAPE': aggregate_mape,
            'Accuracy': aggregate_accuracy
        },
        'window_results': window_results,
        'n_successful_windows': len(window_results),
        'failed_windows': failed_windows
    }

# -----------------------------------------------------------------------------
# HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------
def page_hyperparameter_tuning():
    """Dedicated hyperparameter optimization workspace."""
    from app_core.ui.ml_components import MLUIComponents

    # Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: left; margin-bottom: 2rem;'>
          <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>🔬</div>
            <h1 class='hf-feature-title' style='font-size: 2.5rem; margin: 0;'>Hyperparameter Tuning</h1>
          </div>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 0 0 4.5rem;'>
            Automated hyperparameter optimization using Grid Search, Random Search, or Bayesian Optimization
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 1: MODEL PERFORMANCE COMPARISON (Always visible)
    # =========================================================================
    st.subheader("📊 Model Performance Comparison")
    st.caption("Automatically detecting all trained models and their performance metrics")

    # Auto-detect all trained models
    arima_metrics = _extract_arima_metrics()
    sarimax_metrics = _extract_sarimax_metrics()
    ml_metrics = _extract_ml_metrics()

    # Collect all available models
    available_models = []

    if arima_metrics is not None:
        available_models.append(("ARIMA", arima_metrics))

    if sarimax_metrics is not None:
        available_models.append(("SARIMAX", sarimax_metrics))

    # Check for single ML model result
    if ml_metrics is not None:
        model_name = ml_metrics.get("Model", "ML Model")
        available_models.append((model_name, ml_metrics))

    # Check for individual model results (when "All Models" was trained)
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        model_results = st.session_state.get(f"ml_mh_results_{model_name}")
        if model_results is not None:
            results_df = model_results.get("results_df")
            if results_df is not None and not results_df.empty:
                results_df = _sanitize_metrics_df(results_df)
                metrics = {
                    "MAE": _mean_from_any(results_df, ["Test_MAE", "MAE"]),
                    "RMSE": _mean_from_any(results_df, ["Test_RMSE", "RMSE"]),
                    "MAPE": _mean_from_any(results_df, ["Test_MAPE", "MAPE_%", "MAPE"]),
                    "Accuracy": _mean_from_any(results_df, ["Test_Acc", "Accuracy_%", "Accuracy"]),
                    "Model": model_name,
                }
                # Only add if not already in the list (avoid duplicates)
                if not any(m[0] == model_name for m in available_models):
                    available_models.append((model_name, metrics))

    # Display results
    if len(available_models) == 0:
        st.info(
            "📈 **No models trained yet**\n\n"
            "Train models in the **Benchmarks** or **Machine Learning** tabs to see their performance comparison here.\n\n"
            "Available models:\n"
            "- 📊 ARIMA (Statistical)\n"
            "- 📊 SARIMAX (Statistical with seasonality)\n"
            "- 🤖 XGBoost (Gradient Boosting)\n"
            "- 🧠 LSTM (Recurrent Neural Network)\n"
            "- 🔷 ANN (Feedforward Neural Network)"
        )
    else:
        st.success(f"✅ **{len(available_models)} model(s) detected**")

        # Create comparison table
        st.markdown("### 📋 Performance Metrics Comparison")

        # Display metrics in a structured table
        import pandas as pd

        comparison_data = []
        for model_name, metrics in available_models:
            comparison_data.append({
                "Model": model_name,
                "MAE": f"{metrics['MAE']:.4f}" if metrics.get('MAE') is not None else "N/A",
                "RMSE": f"{metrics['RMSE']:.4f}" if metrics.get('RMSE') is not None else "N/A",
                "MAPE": f"{metrics['MAPE']:.2f}%" if metrics.get('MAPE') is not None else "N/A",
                "Accuracy": f"{metrics['Accuracy']:.2f}%" if metrics.get('Accuracy') is not None else "N/A",
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Display table with highlighting
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        # Display KPI cards for each model
        st.markdown("### 🎯 Detailed Metrics by Model")

        for model_name, metrics in available_models:
            with st.expander(f"**{model_name}** - Detailed Metrics", expanded=False):
                cols = st.columns(4)

                with cols[0]:
                    mae_val = metrics.get('MAE')
                    if mae_val is not None:
                        st.plotly_chart(
                            _create_kpi_indicator("MAE", mae_val, "", PRIMARY_COLOR),
                            use_container_width=True,
                            key=f"tuning_mae_{model_name}",
                        )
                    else:
                        st.metric("MAE", "N/A")

                with cols[1]:
                    rmse_val = metrics.get('RMSE')
                    if rmse_val is not None:
                        st.plotly_chart(
                            _create_kpi_indicator("RMSE", rmse_val, "", SECONDARY_COLOR),
                            use_container_width=True,
                            key=f"tuning_rmse_{model_name}",
                        )
                    else:
                        st.metric("RMSE", "N/A")

                with cols[2]:
                    mape_val = metrics.get('MAPE')
                    if mape_val is not None:
                        st.plotly_chart(
                            _create_kpi_indicator("MAPE", mape_val, "%", WARNING_COLOR),
                            use_container_width=True,
                            key=f"tuning_mape_{model_name}",
                        )
                    else:
                        st.metric("MAPE", "N/A")

                with cols[3]:
                    acc_val = metrics.get('Accuracy')
                    if acc_val is not None:
                        st.plotly_chart(
                            _create_kpi_indicator("Accuracy", acc_val, "%", SUCCESS_COLOR),
                            use_container_width=True,
                            key=f"tuning_acc_{model_name}",
                        )
                    else:
                        st.metric("Accuracy", "N/A")

        # Best model recommendation
        st.markdown("### 🏆 Best Model Recommendation")

        # Find best model by Accuracy (higher is better)
        best_by_accuracy = max(available_models, key=lambda x: x[1].get('Accuracy', 0) if x[1].get('Accuracy') is not None else 0)

        # Find best model by RMSE (lower is better)
        valid_rmse_models = [(name, metrics) for name, metrics in available_models if metrics.get('RMSE') is not None]
        if valid_rmse_models:
            best_by_rmse = min(valid_rmse_models, key=lambda x: x[1]['RMSE'])
        else:
            best_by_rmse = best_by_accuracy

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class='hf-feature-card' style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05)); border-left: 4px solid {SUCCESS_COLOR};'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🏆 Best by Accuracy</div>
                    <div style='font-size: 2rem; font-weight: bold; color: {SUCCESS_COLOR};'>{best_by_accuracy[0]}</div>
                    <div style='font-size: 1.2rem; margin-top: 0.5rem;'>{best_by_accuracy[1].get('Accuracy', 0):.2f}% Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class='hf-feature-card' style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); border-left: 4px solid {PRIMARY_COLOR};'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📉 Best by RMSE</div>
                    <div style='font-size: 2rem; font-weight: bold; color: {PRIMARY_COLOR};'>{best_by_rmse[0]}</div>
                    <div style='font-size: 1.2rem; margin-top: 0.5rem;'>RMSE: {best_by_rmse[1].get('RMSE', 0):.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # SECTION 2: HYPERPARAMETER OPTIMIZATION (Requires configuration)
    # =========================================================================
    st.divider()
    st.subheader("🎯 Hyperparameter Optimization")
    st.caption("Automated search for optimal model parameters using Grid Search, Random Search, or Bayesian Optimization")

    # Check if we have trained data from ML tab
    dataset_available = cfg.get("training_data") is not None
    features_selected = cfg.get("ml_feature_cols") and len(cfg.get("ml_feature_cols", [])) > 0

    if not all([dataset_available, features_selected]):
        st.warning(
            "⚠️ **Configuration Required**\n\n"
            "Please configure the following in the **Machine Learning** tab first:\n"
            f"- {'✅' if dataset_available else '❌'} Dataset selection\n"
            f"- {'✅' if features_selected else '❌'} Feature selection"
        )
        st.info("👈 Navigate to the **🧮 Machine Learning** tab to complete the configuration.")
        return

    # 1. Optimization Mode Selection (Single Model / All Models)
    st.markdown("### 🎯 Optimization Mode")

    opt_mode = st.radio(
        "Training Mode",
        options=["Single Model", "All Models (XGBoost + LSTM + ANN)"],
        index=0 if cfg.get("opt_training_mode", "Single Model") == "Single Model" else 1,
        horizontal=True,
        help="Single Model: Optimize one model with full visualization | All Models: Optimize all three with metrics comparison only",
        key="opt_mode_radio"
    )
    cfg["opt_training_mode"] = opt_mode

    # Display current configuration summary
    with st.expander("📋 Current Configuration", expanded=False):
        if opt_mode == "Single Model":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mode", "Single Model")
            with col2:
                st.metric("Dataset", cfg.get("selected_dataset", "N/A"))
            with col3:
                st.metric("Features", len(cfg.get('ml_feature_cols', [])))
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mode", "All Models")
            with col2:
                st.metric("Models", "XGBoost + LSTM + ANN")
            with col3:
                st.metric("Features", len(cfg.get('ml_feature_cols', [])))

    # Model Selection for Single Model mode
    if opt_mode == "Single Model":
        st.markdown("### 🤖 Model Selection")
        from app_core.ui.ml_components import MLUIComponents
        selected_model_opt = st.selectbox(
            "Select model to optimize",
            options=["XGBoost", "LSTM", "ANN"],
            index=["XGBoost", "LSTM", "ANN"].index(cfg.get("opt_model_choice", "XGBoost")),
            help="Choose which ML model to optimize",
            key="opt_model_select"
        )
        cfg["opt_model_choice"] = selected_model_opt
    else:
        st.info("🚀 **All Models Mode**: XGBoost, LSTM, and ANN will be optimized sequentially")
        selected_model_opt = None

    # 2. Optimization Settings
    st.markdown("### ⚙️ Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Search method selection
        search_method = st.selectbox(
            "Search Method",
            options=["Grid Search", "Random Search", "Bayesian Optimization"],
            help="Grid: Exhaustive search | Random: Random sampling | Bayesian: Smart optimization"
        )
        cfg["tuning_search_method"] = search_method

    with col2:
        # Optimization metric selection
        optimization_metric = st.selectbox(
            "Optimization Metric",
            options=["RMSE", "MAE", "MAPE", "Accuracy"],
            help="Metric to optimize. Lower is better for RMSE/MAE/MAPE, higher is better for Accuracy."
        )
        cfg["tuning_optimization_metric"] = optimization_metric

    # 2. Cross-Validation Strategy
    st.subheader("🔄 Cross-Validation Strategy")

    col1, col2 = st.columns(2)

    with col1:
        n_splits = st.slider(
            "Number of CV Folds",
            min_value=3,
            max_value=10,
            value=cfg.get("tuning_cv_splits", 5),
            help="TimeSeriesSplit folds - more folds = more robust but slower"
        )
        cfg["tuning_cv_splits"] = n_splits

    with col2:
        if search_method == "Random Search":
            n_iter = st.slider(
                "Number of Iterations",
                min_value=10,
                max_value=100,
                value=cfg.get("tuning_n_iter", 20),
                help="Number of random parameter combinations to try"
            )
            cfg["tuning_n_iter"] = n_iter
        elif search_method == "Bayesian Optimization":
            n_iter = st.slider(
                "Number of Trials",
                min_value=10,
                max_value=100,
                value=cfg.get("tuning_n_iter", 30),
                help="Number of Bayesian optimization trials"
            )
            cfg["tuning_n_iter"] = n_iter

    # 3. 7-Day Rolling-Origin Backtesting
    st.subheader("📅 7-Day Rolling-Origin Backtesting (Optional)")

    col1, col2 = st.columns(2)

    with col1:
        enable_backtesting = st.checkbox(
            "Enable 7-Day Backtesting",
            value=cfg.get("enable_backtesting", False),
            help="Perform rolling-origin backtesting with 7-day horizon after optimization"
        )
        cfg["enable_backtesting"] = enable_backtesting

    if enable_backtesting:
        with col2:
            n_windows = st.slider(
                "Number of Backtesting Windows",
                min_value=3,
                max_value=10,
                value=cfg.get("backtest_n_windows", 5),
                help="Number of different cutoff dates for backtesting"
            )
            cfg["backtest_n_windows"] = n_windows

        st.info(
            "🔄 **Backtesting Process:**\n\n"
            f"After optimization, the best model will be trained on {n_windows} different historical cutoffs. "
            f"Each training uses data up to the cutoff date to forecast the next 7 days. "
            f"Metrics are aggregated across all windows to provide robust performance estimates."
        )

    # 4. Run Optimization
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if opt_mode == "Single Model":
            button_text = f"🚀 Run {search_method} - {selected_model_opt}"
        else:
            button_text = f"🚀 Run {search_method} - All Models"

        run_optimization = st.button(
            button_text,
            type="primary",
            use_container_width=True,
            help="Start hyperparameter optimization with TimeSeriesSplit CV"
        )

    # Execute optimization
    if run_optimization:
        if opt_mode == "Single Model" and search_method == "Grid Search":
            # PHASE 1: Grid Search for Single Model
            st.markdown(f"### 🔄 Running {search_method} for {selected_model_opt}")

            # Get dataset
            df = cfg.get("training_data")
            feature_cols = cfg.get("ml_feature_cols")
            split_ratio = cfg.get("split_ratio", 0.8)

            # Prepare data for first target (Target_1)
            target_col = "Target_1"

            if target_col not in df.columns:
                st.error(f"❌ Target column '{target_col}' not found in dataset")
            else:
                # Filter to only numeric features (exclude datetime, object, string, etc.)
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        # Only keep numeric dtypes (int, float) - exclude datetime, object, etc.
                        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_datetime64_any_dtype(dtype):
                            numeric_feature_cols.append(col)

                if len(numeric_feature_cols) == 0:
                    st.error("❌ No numeric features found in the dataset. Please check your feature selection.")
                else:
                    # Extract features and target
                    X = df[numeric_feature_cols].values
                    y = df[target_col].values

                    # Train/test split
                    split_idx = int(len(df) * split_ratio)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # Get parameter grid
                    param_grid = get_param_grid(selected_model_opt, cfg)

                    # Show optimization info
                    st.info(
                        f"**Optimization Configuration:**\n\n"
                        f"- Model: {selected_model_opt}\n"
                        f"- Search Method: {search_method}\n"
                        f"- Optimization Metric: {optimization_metric}\n"
                        f"- CV Folds: {n_splits}\n"
                        f"- Training Samples: {len(X_train)}\n"
                        f"- Test Samples: {len(X_test)}\n"
                        f"- Features: {len(numeric_feature_cols)}\n"
                        f"- Parameter Combinations: {np.prod([len(v) for v in param_grid.values()])}"
                    )

                    # Run optimization with progress
                    with st.spinner(f"⏳ Running Grid Search... This may take several minutes."):
                        try:
                            results = run_grid_search_optimization(
                                model_type=selected_model_opt,
                                X_train=X_train,
                                y_train=y_train,
                                n_splits=n_splits,
                                primary_metric=optimization_metric,
                                param_grid=param_grid
                            )

                            # Store results in session state
                            st.session_state[f"opt_results_{selected_model_opt}"] = results
                            st.session_state["opt_last_model"] = selected_model_opt
                            st.session_state["opt_last_method"] = search_method

                            st.success(f"✅ Optimization completed successfully!")

                            # Run 7-day backtesting if enabled
                            if cfg.get("enable_backtesting", False):
                                n_backtest_windows = cfg.get("backtest_n_windows", 5)

                                with st.spinner(f"⏳ Running 7-day backtesting ({n_backtest_windows} windows)..."):
                                    try:
                                        # Get dates if available
                                        date_col = None
                                        for col in df.columns:
                                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                                date_col = df[col].values
                                                break

                                        backtest_results = run_7day_backtesting(
                                            model_type=selected_model_opt,
                                            best_params=results['best_params'],
                                            X=X,
                                            y=y,
                                            dates=date_col,
                                            n_windows=n_backtest_windows,
                                            horizon=7
                                        )

                                        # Store backtesting results
                                        st.session_state[f"backtest_results_{selected_model_opt}"] = backtest_results

                                        st.success(f"✅ Backtesting completed! {backtest_results['n_successful_windows']}/{n_backtest_windows} windows successful.")

                                    except Exception as e:
                                        st.warning(f"⚠️ Backtesting failed: {str(e)}")
                                        import traceback
                                        with st.expander("🔍 Backtesting Error Details"):
                                            st.code(traceback.format_exc())

                        except Exception as e:
                            st.error(f"❌ **Optimization failed:**\n\n```\n{str(e)}\n```")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())

        elif opt_mode == "Single Model" and search_method == "Random Search":
            # PHASE 2: Random Search for Single Model
            st.markdown(f"### 🎲 Running {search_method} for {selected_model_opt}")

            # Get dataset
            df = cfg.get("training_data")
            feature_cols = cfg.get("ml_feature_cols")
            split_ratio = cfg.get("split_ratio", 0.8)
            target_col = "Target_1"

            if target_col not in df.columns:
                st.error(f"❌ Target column '{target_col}' not found in dataset")
            else:
                # Filter to only numeric features (exclude datetime, object, string, etc.)
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        # Only keep numeric dtypes (int, float) - exclude datetime, object, etc.
                        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_datetime64_any_dtype(dtype):
                            numeric_feature_cols.append(col)

                if len(numeric_feature_cols) == 0:
                    st.error("❌ No numeric features found in the dataset. Please check your feature selection.")
                else:
                    # Extract features and target
                    X = df[numeric_feature_cols].values
                    y = df[target_col].values

                    # Train/test split
                    split_idx = int(len(df) * split_ratio)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # Get parameter distributions
                    param_distributions = get_param_distributions(selected_model_opt, cfg)

                    # Show optimization info
                    st.info(
                        f"**Optimization Configuration:**\n\n"
                        f"- Model: {selected_model_opt}\n"
                        f"- Search Method: {search_method}\n"
                        f"- Optimization Metric: {optimization_metric}\n"
                        f"- CV Folds: {n_splits}\n"
                        f"- Iterations: {n_iter}\n"
                        f"- Training Samples: {len(X_train)}\n"
                        f"- Test Samples: {len(X_test)}\n"
                        f"- Features: {len(numeric_feature_cols)}"
                    )

                # Run optimization with progress
                with st.spinner(f"⏳ Running Random Search... This may take several minutes."):
                    try:
                        results = run_random_search_optimization(
                            model_type=selected_model_opt,
                            X_train=X_train,
                            y_train=y_train,
                            n_splits=n_splits,
                            primary_metric=optimization_metric,
                            param_distributions=param_distributions,
                            n_iter=n_iter
                        )

                        # Store results in session state
                        st.session_state[f"opt_results_{selected_model_opt}"] = results
                        st.session_state["opt_last_model"] = selected_model_opt
                        st.session_state["opt_last_method"] = search_method

                        st.success(f"✅ Optimization completed successfully!")

                        # Run 7-day backtesting if enabled
                        if cfg.get("enable_backtesting", False):
                            n_backtest_windows = cfg.get("backtest_n_windows", 5)

                            with st.spinner(f"⏳ Running 7-day backtesting ({n_backtest_windows} windows)..."):
                                try:
                                    # Get dates if available
                                    date_col = None
                                    for col in df.columns:
                                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                                            date_col = df[col].values
                                            break

                                    backtest_results = run_7day_backtesting(
                                        model_type=selected_model_opt,
                                        best_params=results['best_params'],
                                        X=X,
                                        y=y,
                                        dates=date_col,
                                        n_windows=n_backtest_windows,
                                        horizon=7
                                    )

                                    # Store backtesting results
                                    st.session_state[f"backtest_results_{selected_model_opt}"] = backtest_results

                                    st.success(f"✅ Backtesting completed! {backtest_results['n_successful_windows']}/{n_backtest_windows} windows successful.")

                                except Exception as e:
                                    st.warning(f"⚠️ Backtesting failed: {str(e)}")
                                    import traceback
                                    with st.expander("🔍 Backtesting Error Details"):
                                        st.code(traceback.format_exc())

                    except Exception as e:
                        st.error(f"❌ **Optimization failed:**\n\n```\n{str(e)}\n```")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

        elif opt_mode == "Single Model" and search_method == "Bayesian Optimization":
            # PHASE 2: Bayesian Optimization for Single Model
            st.markdown(f"### 🧠 Running {search_method} for {selected_model_opt}")

            # Get dataset
            df = cfg.get("training_data")
            feature_cols = cfg.get("ml_feature_cols")
            split_ratio = cfg.get("split_ratio", 0.8)
            target_col = "Target_1"

            if target_col not in df.columns:
                st.error(f"❌ Target column '{target_col}' not found in dataset")
            else:
                # Filter to only numeric features (exclude datetime, object, string, etc.)
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        # Only keep numeric dtypes (int, float) - exclude datetime, object, etc.
                        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_datetime64_any_dtype(dtype):
                            numeric_feature_cols.append(col)

                if len(numeric_feature_cols) == 0:
                    st.error("❌ No numeric features found in the dataset. Please check your feature selection.")
                else:
                    # Extract features and target
                    X = df[numeric_feature_cols].values
                    y = df[target_col].values

                    # Train/test split
                    split_idx = int(len(df) * split_ratio)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # Show optimization info
                    st.info(
                        f"**Optimization Configuration:**\n\n"
                        f"- Model: {selected_model_opt}\n"
                        f"- Search Method: {search_method}\n"
                        f"- Optimization Metric: {optimization_metric}\n"
                        f"- CV Folds: {n_splits}\n"
                        f"- Trials: {n_iter}\n"
                        f"- Training Samples: {len(X_train)}\n"
                        f"- Test Samples: {len(X_test)}\n"
                        f"- Features: {len(numeric_feature_cols)}"
                    )

                    # Run optimization with progress
                    with st.spinner(f"⏳ Running Bayesian Optimization... This may take several minutes."):
                        try:
                            results = run_bayesian_optimization(
                                model_type=selected_model_opt,
                                X_train=X_train,
                                y_train=y_train,
                                n_splits=n_splits,
                                primary_metric=optimization_metric,
                                n_trials=n_iter
                            )

                            # Store results in session state
                            st.session_state[f"opt_results_{selected_model_opt}"] = results
                            st.session_state["opt_last_model"] = selected_model_opt
                            st.session_state["opt_last_method"] = search_method

                            st.success(f"✅ Optimization completed successfully!")

                            # Show Optuna optimization history
                            if 'study' in results:
                                with st.expander("📈 Optimization History", expanded=False):
                                    trial_results = results['trial_results']
                                    st.line_chart(trial_results.set_index('trial')['value'])

                            # Run 7-day backtesting if enabled
                            if cfg.get("enable_backtesting", False):
                                n_backtest_windows = cfg.get("backtest_n_windows", 5)

                                with st.spinner(f"⏳ Running 7-day backtesting ({n_backtest_windows} windows)..."):
                                    try:
                                        # Get dates if available
                                        date_col = None
                                        for col in df.columns:
                                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                                date_col = df[col].values
                                                break

                                        backtest_results = run_7day_backtesting(
                                            model_type=selected_model_opt,
                                            best_params=results['best_params'],
                                            X=X,
                                            y=y,
                                            dates=date_col,
                                            n_windows=n_backtest_windows,
                                            horizon=7
                                        )

                                        # Store backtesting results
                                        st.session_state[f"backtest_results_{selected_model_opt}"] = backtest_results

                                        st.success(f"✅ Backtesting completed! {backtest_results['n_successful_windows']}/{n_backtest_windows} windows successful.")

                                    except Exception as e:
                                        st.warning(f"⚠️ Backtesting failed: {str(e)}")
                                        import traceback
                                        with st.expander("🔍 Backtesting Error Details"):
                                            st.code(traceback.format_exc())

                        except Exception as e:
                            st.error(f"❌ **Optimization failed:**\n\n```\n{str(e)}\n```")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())

        elif opt_mode == "All Models (XGBoost + LSTM + ANN)":
            # PHASE 3: Optimize all three models sequentially
            st.markdown(f"### 🚀 Running {search_method} for All Models")

            # Get dataset (shared for all models)
            df = cfg.get("training_data")
            feature_cols = cfg.get("ml_feature_cols")
            split_ratio = cfg.get("split_ratio", 0.8)
            target_col = "Target_1"

            if target_col not in df.columns:
                st.error(f"❌ Target column '{target_col}' not found in dataset")
            else:
                # Filter to only numeric features (exclude datetime, object, string, etc.)
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        # Only keep numeric dtypes (int, float) - exclude datetime, object, etc.
                        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_datetime64_any_dtype(dtype):
                            numeric_feature_cols.append(col)

                if len(numeric_feature_cols) == 0:
                    st.error("❌ No numeric features found in the dataset. Please check your feature selection.")
                else:
                    # Extract features and target
                    X = df[numeric_feature_cols].values
                    y = df[target_col].values

                    # Train/test split
                    split_idx = int(len(df) * split_ratio)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # Show optimization info
                    st.info(
                        f"**Optimization Configuration:**\n\n"
                        f"- Models: XGBoost, LSTM, ANN\n"
                        f"- Search Method: {search_method}\n"
                        f"- Optimization Metric: {optimization_metric}\n"
                        f"- CV Folds: {n_splits}\n"
                        f"- Training Samples: {len(X_train)}\n"
                        f"- Test Samples: {len(X_test)}\n"
                        f"- Features: {len(numeric_feature_cols)}"
                    )

                    # Models to optimize
                    all_models = ["XGBoost", "LSTM", "ANN"]

                    # Progress tracking
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    # Store all results
                    all_model_results = {}
                    failed_models = []

                    # Optimize each model
                    for idx, model_type in enumerate(all_models):
                        try:
                            # Update progress
                            progress = (idx) / len(all_models)
                            progress_bar.progress(progress)
                            status_text.markdown(f"**🔄 Optimizing Model {idx+1}/3: {model_type}**")

                            # Select optimization function based on search method
                            if search_method == "Grid Search":
                                param_grid = get_param_grid(model_type, cfg)

                                with st.spinner(f"⏳ Optimizing {model_type}..."):
                                    results = run_grid_search_optimization(
                                        model_type=model_type,
                                        X_train=X_train,
                                        y_train=y_train,
                                        n_splits=n_splits,
                                        primary_metric=optimization_metric,
                                        param_grid=param_grid
                                    )

                            elif search_method == "Random Search":
                                param_distributions = get_param_distributions(model_type, cfg)
                                # Use cfg fallback for n_iter in case variable not in scope
                                _n_iter = cfg.get("tuning_n_iter", 20)

                                with st.spinner(f"⏳ Optimizing {model_type}..."):
                                    results = run_random_search_optimization(
                                        model_type=model_type,
                                        X_train=X_train,
                                        y_train=y_train,
                                        n_splits=n_splits,
                                        primary_metric=optimization_metric,
                                        param_distributions=param_distributions,
                                        n_iter=_n_iter
                                    )

                            elif search_method == "Bayesian Optimization":
                                # Use cfg fallback for n_iter in case variable not in scope
                                _n_trials = cfg.get("tuning_n_iter", 30)

                                with st.spinner(f"⏳ Optimizing {model_type}..."):
                                    results = run_bayesian_optimization(
                                        model_type=model_type,
                                        X_train=X_train,
                                        y_train=y_train,
                                        n_splits=n_splits,
                                        primary_metric=optimization_metric,
                                        n_trials=_n_trials
                                    )

                            # Store results
                            all_model_results[model_type] = results
                            st.session_state[f"opt_results_{model_type}"] = results

                            status_text.markdown(f"**✅ {model_type} optimization completed!**")

                        except Exception as e:
                            failed_models.append(model_type)
                            st.error(f"❌ **{model_type} optimization failed:**\n\n```\n{str(e)}\n```")
                            import traceback
                            with st.expander(f"🔍 {model_type} Error Details"):
                                st.code(traceback.format_exc())

                    # Final progress update
                    progress_bar.progress(1.0)

                    # Store metadata
                    st.session_state["opt_all_models_results"] = all_model_results
                    st.session_state["opt_last_mode"] = "All Models"
                    st.session_state["opt_last_method"] = search_method

                    # Summary
                    if len(all_model_results) == len(all_models):
                        status_text.markdown("**✅ All models optimized successfully!**")
                        st.success(f"🎉 Successfully optimized {len(all_model_results)}/{len(all_models)} models!")
                    elif len(all_model_results) > 0:
                        status_text.markdown(f"**⚠️ Partial success: {len(all_model_results)}/{len(all_models)} models optimized**")
                        st.warning(f"⚠️ {len(failed_models)} model(s) failed: {', '.join(failed_models)}")
                    else:
                        status_text.markdown("**❌ All models failed to optimize**")
                        st.error("❌ All optimizations failed. Please check the error details above.")

    # Display optimization results
    if opt_mode == "Single Model" and selected_model_opt:
        opt_results = st.session_state.get(f"opt_results_{selected_model_opt}")

        if opt_results is not None:
            st.divider()
            st.markdown("### 📊 Optimization Results")

            # Best parameters and scores
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏆 Best Parameters")
                best_params_df = pd.DataFrame(list(opt_results['best_params'].items()), columns=['Parameter', 'Value'])
                st.dataframe(best_params_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### 📈 Performance Metrics")
                all_scores = opt_results['all_scores']

                # Display as KPI cards
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.plotly_chart(
                        _create_kpi_indicator("RMSE", all_scores['RMSE'], "", PRIMARY_COLOR),
                        use_container_width=True,
                        key=f"opt_rmse_{selected_model_opt}"
                    )
                with metric_cols[1]:
                    st.plotly_chart(
                        _create_kpi_indicator("MAE", all_scores['MAE'], "", SECONDARY_COLOR),
                        use_container_width=True,
                        key=f"opt_mae_{selected_model_opt}"
                    )
                with metric_cols[2]:
                    st.plotly_chart(
                        _create_kpi_indicator("MAPE", all_scores['MAPE'], "%", WARNING_COLOR),
                        use_container_width=True,
                        key=f"opt_mape_{selected_model_opt}"
                    )
                with metric_cols[3]:
                    st.plotly_chart(
                        _create_kpi_indicator("Accuracy", all_scores['Accuracy'], "%", SUCCESS_COLOR),
                        use_container_width=True,
                        key=f"opt_acc_{selected_model_opt}"
                    )

            # CV Results Table
            with st.expander("📋 All Parameter Combinations (CV Results)", expanded=False):
                cv_results = opt_results['cv_results']

                # Select relevant columns
                display_cols = ['params', 'mean_test_RMSE', 'mean_test_MAE', 'mean_test_MAPE', 'mean_test_Accuracy', 'rank_test_' + opt_results['refit_metric']]
                available_cols = [col for col in display_cols if col in cv_results.columns]

                if available_cols:
                    display_df = cv_results[available_cols].copy()

                    # Invert negative scores for display
                    if 'mean_test_RMSE' in display_df.columns:
                        display_df['mean_test_RMSE'] = -display_df['mean_test_RMSE']
                    if 'mean_test_MAE' in display_df.columns:
                        display_df['mean_test_MAE'] = -display_df['mean_test_MAE']
                    if 'mean_test_MAPE' in display_df.columns:
                        display_df['mean_test_MAPE'] = -display_df['mean_test_MAPE']

                    st.dataframe(display_df, use_container_width=True, height=400)

            # Display 7-day backtesting results if available
            backtest_results = st.session_state.get(f"backtest_results_{selected_model_opt}")

            if backtest_results is not None:
                st.divider()
                st.markdown("### 📅 7-Day Rolling-Origin Backtesting Results")

                aggregate_metrics = backtest_results['aggregate_metrics']
                window_results = backtest_results['window_results']
                n_successful = backtest_results['n_successful_windows']

                if aggregate_metrics['RMSE'] is not None:
                    # Aggregate metrics KPI cards
                    st.markdown("#### 🎯 Aggregate Metrics (All Windows)")

                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.plotly_chart(
                            _create_kpi_indicator("RMSE", aggregate_metrics['RMSE'], "", PRIMARY_COLOR),
                            use_container_width=True,
                            key=f"backtest_agg_rmse_{selected_model_opt}"
                        )
                    with metric_cols[1]:
                        st.plotly_chart(
                            _create_kpi_indicator("MAE", aggregate_metrics['MAE'], "", SECONDARY_COLOR),
                            use_container_width=True,
                            key=f"backtest_agg_mae_{selected_model_opt}"
                        )
                    with metric_cols[2]:
                        st.plotly_chart(
                            _create_kpi_indicator("MAPE", aggregate_metrics['MAPE'], "%", WARNING_COLOR),
                            use_container_width=True,
                            key=f"backtest_agg_mape_{selected_model_opt}"
                        )
                    with metric_cols[3]:
                        st.plotly_chart(
                            _create_kpi_indicator("Accuracy", aggregate_metrics['Accuracy'], "%", SUCCESS_COLOR),
                            use_container_width=True,
                            key=f"backtest_agg_acc_{selected_model_opt}"
                        )

                    # Window-by-window metrics table
                    with st.expander(f"📋 Individual Window Results ({n_successful} windows)", expanded=False):
                        window_data = []
                        for w in window_results:
                            window_data.append({
                                'Window': w['window'],
                                'Train Size': w['train_size'],
                                'Test Size': w['test_size'],
                                'RMSE': w['RMSE'],
                                'MAE': w['MAE'],
                                'MAPE': w['MAPE'],
                                'Accuracy': w['Accuracy']
                            })

                        window_df = pd.DataFrame(window_data)
                        st.dataframe(window_df.style.format({
                            'RMSE': '{:.4f}',
                            'MAE': '{:.4f}',
                            'MAPE': '{:.2f}',
                            'Accuracy': '{:.2f}'
                        }), use_container_width=True, hide_index=True)

                    # Representative forecast plot (first window)
                    if len(window_results) > 0:
                        with st.expander("📈 Sample Forecast (First Window)", expanded=False):
                            first_window = window_results[0]
                            y_true = first_window['y_true']
                            y_pred = first_window['y_pred']

                            import plotly.graph_objects as go

                            fig = go.Figure()

                            # Actual values
                            fig.add_trace(go.Scatter(
                                x=list(range(len(y_true))),
                                y=y_true,
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color=PRIMARY_COLOR, width=2),
                                marker=dict(size=8)
                            ))

                            # Predicted values
                            fig.add_trace(go.Scatter(
                                x=list(range(len(y_pred))),
                                y=y_pred,
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color=SECONDARY_COLOR, width=2, dash='dash'),
                                marker=dict(size=8)
                            ))

                            fig.update_layout(
                                title=f"7-Day Forecast - Window 1 (Train Size: {first_window['train_size']})",
                                xaxis_title="Day",
                                yaxis_title="Target Value",
                                height=400,
                                hovermode='x unified',
                                template='plotly_white'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                else:
                    # No successful windows - show error details
                    st.warning("⚠️ Backtesting completed but no successful windows were processed.")

                    # Show error information if available
                    if 'error' in backtest_results:
                        st.error(f"**Error:** {backtest_results['error']}")

                    # Show failed window details
                    failed_windows = backtest_results.get('failed_windows', [])
                    if failed_windows:
                        with st.expander(f"🔍 Failed Windows Details ({len(failed_windows)} windows)", expanded=True):
                            for failed in failed_windows:
                                st.markdown(f"**Window {failed['window']}:** {failed['reason']}")
                                if 'traceback' in failed:
                                    with st.expander(f"Traceback for Window {failed['window']}"):
                                        st.code(failed['traceback'])

    # Display "All Models" optimization results
    elif opt_mode == "All Models (XGBoost + LSTM + ANN)":
        all_model_results = st.session_state.get("opt_all_models_results", {})

        if all_model_results:
            st.divider()

            # =====================================================================
            # PROFESSIONAL MODEL COMPARISON DASHBOARD
            # =====================================================================

            try:
                # Create comparison data with validation
                comparison_data = []
                for model_name, results in all_model_results.items():
                    if 'all_scores' not in results:
                        st.warning(f"⚠️ Missing scores for {model_name}")
                        continue
                    all_scores = results['all_scores']
                    # Validate scores are numeric and not NaN/Inf
                    rmse = all_scores.get('RMSE', float('nan'))
                    mae = all_scores.get('MAE', float('nan'))
                    mape = all_scores.get('MAPE', float('nan'))
                    accuracy = all_scores.get('Accuracy', float('nan'))

                    comparison_data.append({
                        "Model": model_name,
                        "RMSE": rmse if pd.notna(rmse) and np.isfinite(rmse) else 999999,
                        "MAE": mae if pd.notna(mae) and np.isfinite(mae) else 999999,
                        "MAPE (%)": mape if pd.notna(mape) and np.isfinite(mape) else 999999,
                        "Accuracy (%)": accuracy if pd.notna(accuracy) and np.isfinite(accuracy) else 0
                    })

                if len(comparison_data) == 0:
                    st.error("❌ No valid model results to compare.")
                    return

                comparison_df = pd.DataFrame(comparison_data)

                # Determine overall best model (based on number of wins)
                best_rmse_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
                best_mae_model = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
                best_mape_model = comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model']
                best_accuracy_model = comparison_df.loc[comparison_df['Accuracy (%)'].idxmax(), 'Model']

                # Count wins per model
                wins = {}
                for model in comparison_df['Model']:
                    wins[model] = 0
                wins[best_rmse_model] += 1
                wins[best_mae_model] += 1
                wins[best_mape_model] += 1
                wins[best_accuracy_model] += 1

                overall_best = max(wins, key=wins.get)

                # =====================================================================
                # HEADER: OVERALL WINNER
                # =====================================================================
                st.markdown(
                    f"""
                    <div style='
                        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
                        border: 2px solid rgba(34, 197, 94, 0.5);
                        border-radius: 16px;
                        padding: 1.5rem 2rem;
                        margin-bottom: 2rem;
                        text-align: center;
                    '>
                        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🏆</div>
                        <div style='font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem;'>Overall Best Model</div>
                        <div style='font-size: 2rem; font-weight: 800; color: #22c55e;'>{overall_best}</div>
                        <div style='font-size: 0.875rem; color: #64748b; margin-top: 0.5rem;'>Winner in {wins[overall_best]} of 4 metrics</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =====================================================================
                # SECTION 1: PERFORMANCE METRICS COMPARISON TABLE
                # =====================================================================
                st.markdown("### 📊 Performance Metrics Comparison")

                # Styled comparison table with highlighting
                def highlight_best_values(row):
                    styles = [''] * len(row)
                    model_col_idx = comparison_df.columns.get_loc('Model')

                    # Highlight best RMSE (lowest)
                    if row['Model'] == best_rmse_model:
                        styles[comparison_df.columns.get_loc('RMSE')] = 'background-color: rgba(34, 197, 94, 0.3); font-weight: bold; color: #16a34a;'
                    # Highlight best MAE (lowest)
                    if row['Model'] == best_mae_model:
                        styles[comparison_df.columns.get_loc('MAE')] = 'background-color: rgba(34, 197, 94, 0.3); font-weight: bold; color: #16a34a;'
                    # Highlight best MAPE (lowest)
                    if row['Model'] == best_mape_model:
                        styles[comparison_df.columns.get_loc('MAPE (%)')] = 'background-color: rgba(34, 197, 94, 0.3); font-weight: bold; color: #16a34a;'
                    # Highlight best Accuracy (highest)
                    if row['Model'] == best_accuracy_model:
                        styles[comparison_df.columns.get_loc('Accuracy (%)')] = 'background-color: rgba(34, 197, 94, 0.3); font-weight: bold; color: #16a34a;'
                    # Highlight overall winner row
                    if row['Model'] == overall_best:
                        styles[model_col_idx] = 'background-color: rgba(234, 179, 8, 0.2); font-weight: bold;'
                    return styles

                styled_df = comparison_df.style.apply(highlight_best_values, axis=1).format({
                    'RMSE': '{:.4f}',
                    'MAE': '{:.4f}',
                    'MAPE (%)': '{:.2f}',
                    'Accuracy (%)': '{:.2f}'
                })

                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=150)

                st.caption("🟢 Green = Best value for that metric | 🟡 Yellow = Overall winner")

                # =====================================================================
                # SECTION 2: METRIC WINNERS SUMMARY
                # =====================================================================
                st.markdown("### 🥇 Best Model by Metric")

                metric_cols = st.columns(4)

                with metric_cols[0]:
                    rmse_val = comparison_df[comparison_df['Model'] == best_rmse_model]['RMSE'].values[0]
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.1));
                                    border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(59, 130, 246, 0.3);'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Best RMSE</div>
                            <div style='font-size: 1.25rem; font-weight: 700; color: #3b82f6; margin: 0.25rem 0;'>{best_rmse_model}</div>
                            <div style='font-size: 1rem; color: #64748b;'>{rmse_val:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with metric_cols[1]:
                    mae_val = comparison_df[comparison_df['Model'] == best_mae_model]['MAE'].values[0]
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(6, 182, 212, 0.1));
                                    border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(34, 211, 238, 0.3);'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Best MAE</div>
                            <div style='font-size: 1.25rem; font-weight: 700; color: #22d3ee; margin: 0.25rem 0;'>{best_mae_model}</div>
                            <div style='font-size: 1rem; color: #64748b;'>{mae_val:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with metric_cols[2]:
                    mape_val = comparison_df[comparison_df['Model'] == best_mape_model]['MAPE (%)'].values[0]
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
                                    border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(251, 191, 36, 0.3);'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Best MAPE</div>
                            <div style='font-size: 1.25rem; font-weight: 700; color: #fbbf24; margin: 0.25rem 0;'>{best_mape_model}</div>
                            <div style='font-size: 1rem; color: #64748b;'>{mape_val:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with metric_cols[3]:
                    acc_val = comparison_df[comparison_df['Model'] == best_accuracy_model]['Accuracy (%)'].values[0]
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
                                    border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(34, 197, 94, 0.3);'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Best Accuracy</div>
                            <div style='font-size: 1.25rem; font-weight: 700; color: #22c55e; margin: 0.25rem 0;'>{best_accuracy_model}</div>
                            <div style='font-size: 1rem; color: #64748b;'>{acc_val:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =====================================================================
                # SECTION 3: BEST PARAMETERS PER MODEL
                # =====================================================================
                st.markdown("### ⚙️ Best Hyperparameters")

                param_cols = st.columns(len(all_model_results))

                for idx, (model_name, results) in enumerate(all_model_results.items()):
                    with param_cols[idx]:
                        is_winner = model_name == overall_best
                        border_color = "rgba(34, 197, 94, 0.5)" if is_winner else "rgba(100, 116, 139, 0.3)"
                        header_bg = "rgba(34, 197, 94, 0.1)" if is_winner else "rgba(100, 116, 139, 0.1)"

                        # Model header
                        winner_badge = " 🏆" if is_winner else ""
                        st.markdown(
                            f"""
                            <div style='background: {header_bg}; border: 1px solid {border_color};
                                        border-radius: 12px 12px 0 0; padding: 0.75rem; text-align: center;'>
                                <div style='font-weight: 700; font-size: 1rem;'>{model_name}{winner_badge}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Parameters table
                        params_df = pd.DataFrame(
                            list(results['best_params'].items()),
                            columns=['Parameter', 'Value']
                        )
                        st.dataframe(params_df, use_container_width=True, hide_index=True, height=200)

            except Exception as e:
                st.error(f"❌ Error displaying comparison results: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Debug panel
    debug_panel()

# -----------------------------------------------------------------------------
# HYBRID MODELS - Ensemble Forecasting
# -----------------------------------------------------------------------------
def page_hybrid():
    """Hybrid Models: Weighted Ensemble and Decomposition + Ensemble."""

    # Main Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: left; margin-bottom: 2rem;'>
          <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>🧬</div>
            <h1 class='hf-feature-title' style='font-size: 2.5rem; margin: 0;'>Hybrid Models</h1>
          </div>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 750px; margin: 0 0 0 4.5rem;'>
            Advanced ensemble strategies combining multiple models or decomposing time series components for superior forecasting
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ensemble Strategy Selection
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
                    border-left: 4px solid #A855F7;
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1.5rem 0;
                    box-shadow: 0 0 20px rgba(168, 85, 247, 0.2);'>
            <h3 style='color: #A855F7; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(168, 85, 247, 0.5);'>
                🎯 Select Ensemble Strategy
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    ensemble_type = st.radio(
        "Choose your hybrid modeling approach",
        options=[
            "⚖️ Weighted Ensemble (Combine Trained Models)",
            "🔬 Decomposition + Ensemble (Component-Based Modeling)",
            "🔗 Stacking Ensemble (Meta-Learning)",
            "🧠 LSTM-SARIMAX Hybrid (Neural + Statistical)",
            "🚀 LSTM-XGBoost Hybrid (Neural + Gradient Boosting)",
            "🎯 LSTM-ANN Hybrid (Neural + Neural Residuals)"
        ],
        index=0,
        help="Weighted: Combine trained models | Decomposition: Component-based | Stacking: Meta-learning | LSTM-SARIMAX: LSTM + ARIMA residuals | LSTM-XGBoost: LSTM + XGBoost residuals | LSTM-ANN: LSTM + ANN residual network"
    )

    # Dataset Selection (Feature Engineering Integration)
    st.markdown("### 🎨 Dataset Selection")

    # Check if feature engineering has been run
    feature_eng_data = st.session_state.get("feature_engineering", {})
    has_variant_a = "A" in feature_eng_data and isinstance(feature_eng_data["A"], pd.DataFrame) and not feature_eng_data["A"].empty
    has_variant_b = "B" in feature_eng_data and isinstance(feature_eng_data["B"], pd.DataFrame) and not feature_eng_data["B"].empty

    dataset_options = ["📊 Original Dataset (Basic Features)"]
    if has_variant_a:
        dataset_options.append("🔢 Variant A (OHE + Calendar Features)")
    if has_variant_b:
        dataset_options.append("🌀 Variant B (Cyclical + Harmonic Features)")

    if len(dataset_options) > 1:
        st.info("✨ **Feature-engineered datasets detected!** Select a dataset with advanced features to improve base model performance.")

    selected_dataset = st.selectbox(
        "Choose dataset for hybrid models",
        options=dataset_options,
        index=0,
        help="Original: Basic time series data | Variant A: One-hot encoded calendar features | Variant B: Cyclical sin/cos encoded features"
    )

    # Store selected dataset in session state for hybrid functions to access
    st.session_state["hybrid_selected_dataset"] = selected_dataset

    st.divider()

    if "Weighted Ensemble" in ensemble_type:
        _page_hybrid_weighted()
    elif "Decomposition" in ensemble_type:
        _page_hybrid_decomposition()
    elif "LSTM-SARIMAX" in ensemble_type:
        _page_hybrid_lstm_sarimax()
    elif "LSTM-XGBoost" in ensemble_type:
        _page_hybrid_lstm_xgb()
    elif "LSTM-ANN" in ensemble_type:
        _page_hybrid_lstm_ann()
    else:
        _page_hybrid_stacking()


def _get_hybrid_dataset():
    """
    Get the selected dataset for hybrid models based on user choice.
    Returns the appropriate dataset (original, Variant A, or Variant B).
    """
    selected = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset (Basic Features)")

    # Try Variant A (OHE + Calendar Features)
    if "Variant A" in selected:
        feature_eng_data = st.session_state.get("feature_engineering", {})
        variant_a = feature_eng_data.get("A")
        if isinstance(variant_a, pd.DataFrame) and not variant_a.empty:
            return variant_a.copy()

    # Try Variant B (Cyclical + Harmonic Features)
    if "Variant B" in selected:
        feature_eng_data = st.session_state.get("feature_engineering", {})
        variant_b = feature_eng_data.get("B")
        if isinstance(variant_b, pd.DataFrame) and not variant_b.empty:
            return variant_b.copy()

    # Fallback to original dataset
    fused_data = st.session_state.get("processed_df")
    if fused_data is None or (isinstance(fused_data, pd.DataFrame) and fused_data.empty):
        fused_data = st.session_state.get("merged_data")

    return fused_data


def _page_hybrid_weighted():
    """Weighted Ensemble combining SARIMAX, XGBoost, LSTM, and ANN predictions."""

    # Weighted Ensemble Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
                    border-left: 4px solid {SUCCESS_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);'>
            <h2 style='color: {SUCCESS_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(34, 197, 94, 0.5);'>
                ⚖️ Weighted Ensemble
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                Combine predictions from multiple trained models with optimized weights
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check available models in session state
    available_models = {}
    model_info = []

    # Check SARIMAX
    sarimax_results = st.session_state.get("sarimax_results")
    if sarimax_results and sarimax_results.get("per_h"):
        available_models["SARIMAX"] = sarimax_results
        sarimax_metrics = _extract_sarimax_metrics()
        if sarimax_metrics:
            model_info.append({
                "Model": "SARIMAX",
                "MAE": sarimax_metrics.get("MAE", np.nan),
                "RMSE": sarimax_metrics.get("RMSE", np.nan),
                "MAPE": sarimax_metrics.get("MAPE", np.nan),
                "Accuracy": sarimax_metrics.get("Accuracy", np.nan),
            })

    # Check ML models
    ml_mh_results = st.session_state.get("ml_mh_results")
    if ml_mh_results and ml_mh_results.get("per_h"):
        model_type = ml_mh_results.get("results_df", {}).get("Model", pd.Series(["Unknown"])).iloc[0] if not ml_mh_results.get("results_df", pd.DataFrame()).empty else "ML Model"
        available_models[model_type] = ml_mh_results
        ml_metrics = _extract_ml_metrics()
        if ml_metrics:
            model_info.append({
                "Model": model_type,
                "MAE": ml_metrics.get("MAE", np.nan),
                "RMSE": ml_metrics.get("RMSE", np.nan),
                "MAPE": ml_metrics.get("MAPE", np.nan),
                "Accuracy": ml_metrics.get("Accuracy", np.nan),
            })

    # Check for individual XGBoost/LSTM/ANN results (from All Models mode)
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        key = f"ml_mh_results_{model_name.lower()}"
        model_results = st.session_state.get(key)
        if model_results and model_results.get("per_h") and model_name not in available_models:
            available_models[model_name] = model_results
            results_df = model_results.get("results_df")
            if results_df is not None and not results_df.empty:
                model_info.append({
                    "Model": model_name,
                    "MAE": results_df["Test_MAE"].mean(),
                    "RMSE": results_df["Test_RMSE"].mean(),
                    "MAPE": results_df["Test_MAPE"].mean(),
                    "Accuracy": results_df["Test_Acc"].mean(),
                })

    # Display available models
    if len(available_models) < 2:
        st.warning(
            f"""
            ⚠️ **Insufficient Models for Ensemble**

            Found **{len(available_models)} model(s)**. Need at least **2 models** to create an ensemble.

            **Available models:** {', '.join(available_models.keys()) if available_models else 'None'}

            **To train models:**
            - **SARIMAX**: Go to Benchmarks (Page 05)
            - **XGBoost/LSTM/ANN**: Go to Machine Learning tab above

            Train at least 2 different models, then return here to create your ensemble!
            """
        )
        return

    # Success - show available models
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
                    border-left: 4px solid {SUCCESS_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);'>
            <h3 style='color: {SUCCESS_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(34, 197, 94, 0.5);'>
                ✅ {len(available_models)} Models Ready for Ensemble
            </h3>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                {', '.join(available_models.keys())}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show model comparison table
    if model_info:
        st.markdown("### 📊 Individual Model Performance")
        comparison_df = pd.DataFrame(model_info)
        comparison_df = comparison_df.sort_values("RMSE", ascending=True).reset_index(drop=True)

        st.dataframe(
            comparison_df.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAPE": "{:.2f}",
                "Accuracy": "{:.2f}"
            }).background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE'])
             .background_gradient(cmap='RdYlGn', subset=['Accuracy']),
            use_container_width=True,
            height=200
        )

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # Weighting strategy selection
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
                    border-left: 4px solid {PRIMARY_COLOR};
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 2rem 0 1.5rem 0;
                    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
            <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                ⚖️ Weighting Strategy
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    weighting_strategy = st.radio(
        "Choose how to combine model predictions",
        options=[
            "Simple Average (Equal Weights)",
            "Inverse Error Weighting (Performance-Based)",
            "Manual Weights"
        ],
        index=0,
        help="Simple Average: All models weighted equally | Inverse Error: Better models get higher weights | Manual: Set your own weights"
    )

    # Configure weights based on strategy
    weights = {}

    if "Simple Average" in weighting_strategy:
        # Equal weights
        equal_weight = 1.0 / len(available_models)
        for model_name in available_models.keys():
            weights[model_name] = equal_weight

        st.info(f"ℹ️ Each model weighted equally at {equal_weight:.3f} ({equal_weight*100:.1f}%)")

    elif "Inverse Error" in weighting_strategy:
        # Weight by inverse RMSE
        rmse_values = {}
        for info in model_info:
            rmse_values[info["Model"]] = info["RMSE"]

        # Calculate inverse error weights
        inverse_rmse = {k: 1.0 / v if v > 0 else 0 for k, v in rmse_values.items()}
        total_inv = sum(inverse_rmse.values())

        if total_inv > 0:
            weights = {k: v / total_inv for k, v in inverse_rmse.items()}
        else:
            # Fallback to equal weights
            equal_weight = 1.0 / len(available_models)
            weights = {k: equal_weight for k in available_models.keys()}

        # Show weights
        st.markdown("**📊 Calculated Weights (Lower RMSE = Higher Weight)**")
        weight_df = pd.DataFrame([
            {"Model": k, "RMSE": rmse_values.get(k, np.nan), "Weight": v, "Weight %": v * 100}
            for k, v in weights.items()
        ]).sort_values("Weight", ascending=False)

        st.dataframe(
            weight_df.style.format({
                "RMSE": "{:.4f}",
                "Weight": "{:.4f}",
                "Weight %": "{:.2f}%"
            }).background_gradient(cmap='RdYlGn_r', subset=['RMSE'])
             .background_gradient(cmap='YlGn', subset=['Weight %']),
            use_container_width=True,
            height=200
        )

    else:  # Manual weights
        st.markdown("**🎛️ Set Custom Weights**")
        st.caption("Weights will be automatically normalized to sum to 1.0")

        cols = st.columns(len(available_models))
        manual_weights = {}

        for idx, model_name in enumerate(available_models.keys()):
            with cols[idx]:
                manual_weights[model_name] = st.slider(
                    f"{model_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(available_models),
                    step=0.05,
                    key=f"weight_{model_name}"
                )

        # Normalize weights
        total_weight = sum(manual_weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in manual_weights.items()}
        else:
            equal_weight = 1.0 / len(available_models)
            weights = {k: equal_weight for k in available_models.keys()}

        # Show normalized weights
        st.info(f"ℹ️ Normalized weights: {' | '.join([f'{k}: {v:.3f} ({v*100:.1f}%)' for k, v in weights.items()])}")

    # Create Ensemble Button
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        create_ensemble = st.button(
            "🚀 Create Weighted Ensemble",
            type="primary",
            use_container_width=True,
            help="Combine models with selected weights"
        )

    if create_ensemble:
        with st.spinner("Creating weighted ensemble..."):
            try:
                # Extract predictions from each model (for first target horizon as example)
                ensemble_predictions = []
                ensemble_actuals = []
                model_predictions_dict = {}

                # Get horizon 1 data from each model
                horizon = 1

                for model_name, model_results in available_models.items():
                    per_h = model_results.get("per_h", {})

                    if horizon in per_h:
                        h_data = per_h[horizon]
                        forecast = h_data.get("forecast")
                        actual = h_data.get("y_test")

                        if forecast is not None and actual is not None:
                            model_predictions_dict[model_name] = {
                                "forecast": forecast,
                                "actual": actual,
                                "weight": weights.get(model_name, 0)
                            }

                            if len(ensemble_actuals) == 0:
                                ensemble_actuals = actual

                # Combine predictions with weights
                if model_predictions_dict:
                    # Ensure all forecasts have same length
                    min_length = min(len(d["forecast"]) for d in model_predictions_dict.values())

                    ensemble_forecast = np.zeros(min_length)
                    for model_name, data in model_predictions_dict.items():
                        forecast_trimmed = data["forecast"][:min_length]
                        ensemble_forecast += forecast_trimmed * data["weight"]

                    # Trim actual to match
                    ensemble_actuals = ensemble_actuals[:min_length]

                    # Calculate ensemble metrics directly
                    from sklearn.metrics import mean_absolute_error, mean_squared_error

                    mae = mean_absolute_error(ensemble_actuals, ensemble_forecast)
                    rmse = np.sqrt(mean_squared_error(ensemble_actuals, ensemble_forecast))

                    # Calculate MAPE
                    mape = np.mean(np.abs((ensemble_actuals - ensemble_forecast) / ensemble_actuals)) * 100

                    # Calculate Accuracy (1 - MAPE/100)
                    accuracy = max(0, 100 - mape)

                    ensemble_metrics = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": mape,
                        "Accuracy": accuracy
                    }

                    # Store ensemble results in session state
                    st.session_state["ensemble_results"] = {
                        "predictions": ensemble_forecast,
                        "actuals": ensemble_actuals,
                        "metrics": ensemble_metrics,
                        "weights": weights,
                        "models": list(available_models.keys()),
                        "model_predictions": model_predictions_dict
                    }

                    st.success("✅ **Ensemble Created Successfully!**")

                    # Display ensemble metrics
                    st.markdown("### 🏆 Ensemble Performance")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.plotly_chart(
                            _create_kpi_indicator("MAE", ensemble_metrics["MAE"], "", PRIMARY_COLOR),
                            use_container_width=True,
                            key="ensemble_mae"
                        )
                    with col2:
                        st.plotly_chart(
                            _create_kpi_indicator("RMSE", ensemble_metrics["RMSE"], "", SECONDARY_COLOR),
                            use_container_width=True,
                            key="ensemble_rmse"
                        )
                    with col3:
                        st.plotly_chart(
                            _create_kpi_indicator("MAPE", ensemble_metrics["MAPE"], "%", WARNING_COLOR),
                            use_container_width=True,
                            key="ensemble_mape"
                        )
                    with col4:
                        st.plotly_chart(
                            _create_kpi_indicator("Accuracy", ensemble_metrics["Accuracy"], "%", SUCCESS_COLOR),
                            use_container_width=True,
                            key="ensemble_accuracy"
                        )

                    # Comparison with individual models
                    st.markdown("### 📊 Ensemble vs Individual Models")

                    comparison_data = []
                    for info in model_info:
                        comparison_data.append({
                            "Model": info["Model"],
                            "Type": "Individual",
                            "MAE": info["MAE"],
                            "RMSE": info["RMSE"],
                            "MAPE": info["MAPE"],
                            "Accuracy": info["Accuracy"],
                        })

                    # Add ensemble
                    comparison_data.append({
                        "Model": "Weighted Ensemble",
                        "Type": "Ensemble",
                        "MAE": ensemble_metrics["MAE"],
                        "RMSE": ensemble_metrics["RMSE"],
                        "MAPE": ensemble_metrics["MAPE"],
                        "Accuracy": ensemble_metrics["Accuracy"],
                    })

                    comparison_df = pd.DataFrame(comparison_data)

                    st.dataframe(
                        comparison_df.style.format({
                            "MAE": "{:.4f}",
                            "RMSE": "{:.4f}",
                            "MAPE": "{:.2f}",
                            "Accuracy": "{:.2f}"
                        }).apply(lambda x: ['background-color: rgba(34, 197, 94, 0.2)' if v == "Ensemble" else '' for v in x], subset=['Type'], axis=0)
                         .background_gradient(cmap='RdYlGn_r', subset=['RMSE']),
                        use_container_width=True,
                        height=300
                    )

                    # Forecast visualization
                    st.markdown("### 📈 Ensemble Forecast vs Actuals (Horizon 1)")

                    fig = go.Figure()

                    # Actual values
                    fig.add_trace(go.Scatter(
                        y=ensemble_actuals,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color=PRIMARY_COLOR, width=2.5),
                        marker=dict(size=5, symbol='circle'),
                    ))

                    # Ensemble forecast
                    fig.add_trace(go.Scatter(
                        y=ensemble_forecast,
                        mode='lines+markers',
                        name='Ensemble Forecast',
                        line=dict(color=SUCCESS_COLOR, width=2.5, dash='dash'),
                        marker=dict(size=5, symbol='diamond'),
                    ))

                    # Individual model forecasts
                    colors = [SECONDARY_COLOR, WARNING_COLOR, DANGER_COLOR, "#9333EA"]
                    for idx, (model_name, data) in enumerate(model_predictions_dict.items()):
                        fig.add_trace(go.Scatter(
                            y=data["forecast"][:min_length],
                            mode='lines',
                            name=f'{model_name} (w={data["weight"]:.2f})',
                            line=dict(color=colors[idx % len(colors)], width=1.5, dash='dot'),
                            opacity=0.6
                        ))

                    fig.update_layout(
                        xaxis_title="Time Step",
                        yaxis_title="Target Value",
                        hovermode='x unified',
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(15, 23, 42, 0.8)',
                        font=dict(color=TEXT_COLOR, size=11),
                        height=500,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="right",
                            x=1.15,
                            bgcolor='rgba(15, 23, 42, 0.8)',
                            bordercolor=PRIMARY_COLOR,
                            borderwidth=1
                        ),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )

                    st.plotly_chart(fig, use_container_width=True, key="ensemble_forecast_plot")

                else:
                    st.error("❌ Could not extract predictions from models. Please ensure models are properly trained.")

            except Exception as e:
                st.error(f"❌ **Ensemble Creation Failed**\n\n{str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Show existing ensemble if available
    elif "ensemble_results" in st.session_state:
        st.info("ℹ️ **Previous ensemble results available.** Click 'Create Weighted Ensemble' to generate new results with current settings.")

        ensemble_data = st.session_state["ensemble_results"]
        ensemble_metrics = ensemble_data["metrics"]

        st.markdown("### 🏆 Previous Ensemble Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", ensemble_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_ensemble_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", ensemble_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_ensemble_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", ensemble_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_ensemble_mape"
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", ensemble_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_ensemble_accuracy"
            )


# -----------------------------------------------------------------------------
# FFT Decomposition Helper Functions
# -----------------------------------------------------------------------------
def _extract_fft_cycles(y: pd.Series, top_k: int=5, detrend_mean: bool=True) -> pd.DataFrame:
    """
    Extract dominant frequency components using FFT.
    Returns DataFrame with cycle information: period_days, amplitude, frequency.
    """
    # Ensure daily frequency and interpolate
    y = y.asfreq("D").interpolate(method="time", limit_direction="both")
    sig = y.values.astype(float)

    if detrend_mean:
        sig = sig - np.nanmean(sig)

    # Perform FFT
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)

    # Filter out zero frequency
    valid = freqs > 0
    freqs = freqs[valid]
    amps = np.abs(fft_vals[valid])

    if len(freqs) == 0:
        return pd.DataFrame(columns=["period_days", "amplitude", "frequency"])

    # Get top K frequencies
    k = max(1, min(top_k, len(freqs)))
    idx = np.argpartition(amps, -k)[-k:]
    idx = idx[np.argsort(amps[idx])[::-1]]

    periods = 1.0 / freqs[idx]
    amplitudes = amps[idx]
    frequencies = freqs[idx]

    out = pd.DataFrame({
        "period_days": periods,
        "amplitude": amplitudes,
        "frequency": frequencies
    })

    return out.sort_values("amplitude", ascending=False).reset_index(drop=True)


def _fft_decompose(ts: pd.Series, top_k: int=3) -> dict:
    """
    Decompose time series using FFT into:
    - Trend: Low-frequency component (moving average)
    - Seasonal components: Top K frequency components
    - Residual: What's left after removing trend and seasonal

    Returns dict with 'trend', 'seasonal' (dict of components), 'residual', 'cycles_info'
    """
    # Extract dominant cycles
    cycles_df = _extract_fft_cycles(ts, top_k=top_k, detrend_mean=True)

    if cycles_df.empty:
        # Fallback: just return mean as trend
        trend = pd.Series(np.full(len(ts), ts.mean()), index=ts.index)
        residual = ts - trend
        return {
            "trend": trend.values,
            "seasonal": {},
            "residual": residual.values,
            "cycles_info": cycles_df,
            "seasonal_combined": np.zeros(len(ts))
        }

    # Calculate trend as low-pass filter (moving average)
    window = max(7, int(cycles_df["period_days"].max() // 2))
    if window % 2 == 0:
        window += 1
    trend = ts.rolling(window=window, center=True, min_periods=1).mean()

    # Detrend signal
    detrended = ts - trend

    # Extract seasonal components using inverse FFT
    sig = detrended.values.astype(float)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)

    # Create mask for seasonal components (keep only top K frequencies)
    seasonal_mask = np.zeros(len(fft_vals), dtype=bool)
    seasonal_components = {}

    for idx, row in cycles_df.iterrows():
        freq = row["frequency"]
        period = row["period_days"]

        # Find closest frequency in FFT
        freq_idx = np.argmin(np.abs(freqs - freq))
        seasonal_mask[freq_idx] = True

        # Extract this component
        component_fft = np.zeros_like(fft_vals)
        component_fft[freq_idx] = fft_vals[freq_idx]
        component_signal = np.fft.irfft(component_fft, n=len(sig))

        seasonal_components[f"seasonal_{int(period)}d"] = component_signal

    # Combined seasonal component
    seasonal_fft = fft_vals * seasonal_mask
    seasonal_combined = np.fft.irfft(seasonal_fft, n=len(sig))

    # Residual
    residual = sig - seasonal_combined

    return {
        "trend": trend.values,
        "seasonal": seasonal_components,
        "seasonal_combined": seasonal_combined,
        "residual": residual,
        "cycles_info": cycles_df
    }


def _fft_multiscale_decompose(ts: pd.Series, low_cutoff: int=30, high_cutoff: int=7) -> dict:
    """
    4-Component FFT Multi-Scale Decomposition.

    Separates time series into 4 frequency bands:
    1. Trend: Very low frequency (DC + smooth moving average)
    2. Low-Frequency Seasonal: Periods >= low_cutoff (monthly, quarterly, yearly)
    3. High-Frequency Seasonal: Periods <= high_cutoff (daily, weekly)
    4. Residual: Mid-frequency + noise

    Args:
        ts: Time series to decompose
        low_cutoff: Minimum period (days) for low-frequency seasonal
        high_cutoff: Maximum period (days) for high-frequency seasonal

    Returns:
        dict with 'trend', 'low_freq_seasonal', 'high_freq_seasonal', 'residual', 'cycles_info'
    """
    # Extract all dominant cycles
    cycles_df = _extract_fft_cycles(ts, top_k=20, detrend_mean=True)

    if cycles_df.empty:
        # Fallback
        trend = pd.Series(np.full(len(ts), ts.mean()), index=ts.index)
        return {
            "trend": trend.values,
            "low_freq_seasonal": np.zeros(len(ts)),
            "high_freq_seasonal": np.zeros(len(ts)),
            "residual": (ts - trend).values,
            "cycles_info": cycles_df,
            "low_freq_cycles": pd.DataFrame(),
            "high_freq_cycles": pd.DataFrame()
        }

    # Calculate trend as moving average
    window = max(7, int(low_cutoff * 2))
    if window % 2 == 0:
        window += 1
    trend = ts.rolling(window=window, center=True, min_periods=1).mean()

    # Detrend signal
    detrended = ts - trend

    # Perform FFT on detrended signal
    sig = detrended.values.astype(float)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)

    # Separate cycles into low-freq and high-freq
    low_freq_mask = np.zeros(len(fft_vals), dtype=bool)
    high_freq_mask = np.zeros(len(fft_vals), dtype=bool)

    low_freq_cycles = []
    high_freq_cycles = []

    for idx, row in cycles_df.iterrows():
        freq = row["frequency"]
        period = row["period_days"]

        # Find closest frequency in FFT
        freq_idx = np.argmin(np.abs(freqs - freq))

        if period >= low_cutoff:
            # Low-frequency seasonal (monthly, quarterly, yearly)
            low_freq_mask[freq_idx] = True
            low_freq_cycles.append(row)
        elif period <= high_cutoff:
            # High-frequency seasonal (daily, weekly)
            high_freq_mask[freq_idx] = True
            high_freq_cycles.append(row)

    # Extract low-frequency seasonal component
    low_freq_fft = fft_vals * low_freq_mask
    low_freq_seasonal = np.fft.irfft(low_freq_fft, n=len(sig))

    # Extract high-frequency seasonal component
    high_freq_fft = fft_vals * high_freq_mask
    high_freq_seasonal = np.fft.irfft(high_freq_fft, n=len(sig))

    # Residual: everything else (mid-frequency + noise)
    residual = sig - low_freq_seasonal - high_freq_seasonal

    # Convert cycles lists to DataFrames
    low_freq_cycles_df = pd.DataFrame(low_freq_cycles) if low_freq_cycles else pd.DataFrame(columns=cycles_df.columns)
    high_freq_cycles_df = pd.DataFrame(high_freq_cycles) if high_freq_cycles else pd.DataFrame(columns=cycles_df.columns)

    return {
        "trend": trend.values,
        "low_freq_seasonal": low_freq_seasonal,
        "high_freq_seasonal": high_freq_seasonal,
        "residual": residual,
        "cycles_info": cycles_df,
        "low_freq_cycles": low_freq_cycles_df,
        "high_freq_cycles": high_freq_cycles_df
    }


def _page_hybrid_decomposition():
    """Decomposition + Ensemble: Break time series into Trend, Seasonal, and Residual components."""

    # Decomposition Ensemble Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
                    border-left: 4px solid {PRIMARY_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
            <h2 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                🔬 Decomposition + Ensemble
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                Decompose time series into Trend, Seasonal, and Residual components, model each separately, then combine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get selected dataset (original, Variant A, or Variant B)
    fused_data = _get_hybrid_dataset()
    selected_dataset_name = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset")

    # Show dataset info
    st.info(f"🎨 **Using dataset**: {selected_dataset_name} | **Features**: {len(fused_data.columns)} columns | **Samples**: {len(fused_data)} rows")

    if fused_data is None or (isinstance(fused_data, pd.DataFrame) and fused_data.empty):
        st.warning(
            """
            ⚠️ **No Fused Data Available**

            You need to load and fuse data before using Decomposition + Ensemble.

            **Steps:**
            1. Go to **Data Hub** (Page 02) to load Patient, Weather, and Calendar data
            2. Go to **Data Preparation Studio** (Page 03) to fuse the datasets
            3. Return here to decompose and model your time series
            """
        )
        return

    # Info box explaining the approach
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 146, 60, 0.1));
                    border-left: 4px solid {WARNING_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);'>
            <h3 style='color: {WARNING_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 700;'>
                💡 How It Works
            </h3>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9; line-height: 1.6;'>
                <strong>1. STL Decomposition:</strong> Breaks your time series into 3 components<br>
                <strong>2. Component Modeling:</strong> Trains specialized models on each component<br>
                &nbsp;&nbsp;&nbsp;• <strong>Trend</strong> → XGBoost (smooth patterns)<br>
                &nbsp;&nbsp;&nbsp;• <strong>Seasonal</strong> → SARIMAX (cyclic patterns)<br>
                &nbsp;&nbsp;&nbsp;• <strong>Residual</strong> → LSTM (complex non-linear patterns)<br>
                <strong>3. Reconstruction:</strong> Combines all forecasts (Trend + Seasonal + Residual = Final)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Configuration section
    st.markdown("### ⚙️ Configuration")

    # Initialize default values
    seasonal_period = 7  # Default for classical mode
    top_k_freq = 3  # Default for FFT mode

    # Decomposition method selection
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(168, 85, 247, 0.1));
                    border-left: 4px solid #8B5CF6;
                    padding: 1rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;'>
            <h4 style='color: #8B5CF6; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>
                🔬 Decomposition Method
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    decomp_method = st.radio(
        "Choose decomposition approach",
        options=[
            "📊 Classical (Seasonal Decomposition)",
            "🌊 FFT Basic (Frequency-Based)",
            "🎯 FFT Multi-Scale (4-Model Architecture)"
        ],
        index=0,
        help="Classical: Trend+Seasonal+Residual (3 models) | FFT Basic: Auto-detect frequencies (3 models) | FFT Multi-Scale: Separate frequency bands (4 models with ANN)"
    )

    use_fft = "FFT" in decomp_method
    use_multiscale = "Multi-Scale" in decomp_method

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    if use_multiscale:
        # FFT Multi-Scale specific configuration
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            target_col = st.selectbox(
                "Target Column",
                options=[col for col in fused_data.columns if col not in ["Date", "datetime"]],
                index=0 if "Target_1" not in fused_data.columns else list(fused_data.columns).index("Target_1") if "Target_1" in fused_data.columns else 0,
                help="The time series column to decompose and forecast"
            )

        with col2:
            low_freq_cutoff = st.slider(
                "Low-Freq Cutoff (days)",
                min_value=15,
                max_value=90,
                value=30,
                help="Periods >= this are Low-Frequency (SARIMAX). E.g., monthly, quarterly"
            )

        with col3:
            high_freq_cutoff = st.slider(
                "High-Freq Cutoff (days)",
                min_value=2,
                max_value=14,
                value=7,
                help="Periods <= this are High-Frequency (ANN). E.g., daily, weekly"
            )

        with col4:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            )

        st.info("ℹ️ **Multi-Scale Mode**: 4 components → Trend (XGBoost) + Low-freq Seasonal (SARIMAX) + High-freq Seasonal (ANN) + Residual (LSTM)")

    elif use_fft:
        # FFT-specific configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            target_col = st.selectbox(
                "Target Column",
                options=[col for col in fused_data.columns if col not in ["Date", "datetime"]],
                index=0 if "Target_1" not in fused_data.columns else list(fused_data.columns).index("Target_1") if "Target_1" in fused_data.columns else 0,
                help="The time series column to decompose and forecast"
            )

        with col2:
            top_k_freq = st.slider(
                "Top K Frequencies",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of dominant frequency components to extract"
            )

        with col3:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            )

        st.info("ℹ️ **FFT Mode**: Will automatically detect and extract the strongest frequency patterns from your data")

    else:
        # Classical decomposition configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            target_col = st.selectbox(
                "Target Column",
                options=[col for col in fused_data.columns if col not in ["Date", "datetime"]],
                index=0 if "Target_1" not in fused_data.columns else list(fused_data.columns).index("Target_1") if "Target_1" in fused_data.columns else 0,
                help="The time series column to decompose and forecast"
            )

        with col2:
            seasonal_period = st.number_input(
                "Seasonal Period (must be odd)",
                min_value=3,
                max_value=365,
                value=7,
                step=2,
                help="Period of seasonality - must be odd number (e.g., 7 for weekly, 31 for monthly)"
            )

            # Ensure odd number for seasonal_decompose
            if seasonal_period % 2 == 0:
                seasonal_period += 1
                st.caption(f"⚠️ Adjusted to {seasonal_period} (requires odd numbers)")

        with col3:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            )

    st.divider()

    # Train button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        train_decomp = st.button(
            "🚀 Train Decomposition Ensemble",
            type="primary",
            use_container_width=True,
            help="Decompose, train specialized models, and combine forecasts"
        )

    if train_decomp:
        with st.spinner("🔬 Decomposing time series and training component models..."):
            try:
                # Import required libraries
                from statsmodels.tsa.seasonal import STL
                from xgboost import XGBRegressor
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras import layers

                # Prepare data
                df = fused_data.copy()
                if "Date" not in df.columns and "datetime" in df.columns:
                    df["Date"] = pd.to_datetime(df["datetime"])
                elif "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])

                df = df.sort_values("Date").reset_index(drop=True)

                # Create proper time series with DatetimeIndex
                df = df.set_index("Date")
                ts = df[target_col]

                # Ensure daily frequency and interpolate any gaps (following EDA approach)
                ts = ts.asfreq("D").interpolate(method="time", limit_direction="both")

                # Train/test split
                split_idx = int(len(ts) * (1 - test_size / 100))
                ts_train = ts.iloc[:split_idx]
                ts_test = ts.iloc[split_idx:]

                st.info(f"📊 Data split: {len(ts_train)} train samples, {len(ts_test)} test samples")

                # Branch based on decomposition method
                if use_multiscale:
                    # ========== FFT MULTI-SCALE DECOMPOSITION PATH (4 components) ==========
                    st.markdown("### 🎯 Step 1: FFT Multi-Scale Decomposition")
                    progress_bar = st.progress(0, text="Analyzing multi-scale frequency components...")

                    # Apply FFT multi-scale decomposition
                    decomp_result = _fft_multiscale_decompose(ts_train, low_cutoff=low_freq_cutoff, high_cutoff=high_freq_cutoff)

                    trend_train = decomp_result["trend"]
                    low_freq_seasonal_train = decomp_result["low_freq_seasonal"]
                    high_freq_seasonal_train = decomp_result["high_freq_seasonal"]
                    residual_train = decomp_result["residual"]
                    cycles_info = decomp_result["cycles_info"]
                    low_freq_cycles = decomp_result["low_freq_cycles"]
                    high_freq_cycles = decomp_result["high_freq_cycles"]

                    # Store the index for later use
                    train_index = ts_train.index

                    progress_bar.progress(20, text="Multi-scale FFT decomposition complete ✓")

                    # Show detected cycles in both bands
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"#### 🌊 Low-Frequency Components (≥{low_freq_cutoff} days)")
                        if not low_freq_cycles.empty:
                            low_display = low_freq_cycles.copy()
                            low_display["period_days"] = low_display["period_days"].round(2)
                            low_display["amplitude"] = low_display["amplitude"].round(4)
                            st.dataframe(low_display, use_container_width=True, hide_index=True, height=150)

                            interpretations = []
                            for _, row in low_freq_cycles.iterrows():
                                period = row["period_days"]
                                if 28 <= period <= 31:
                                    interpretations.append(f"📆 Monthly (~{period:.1f}d)")
                                elif 89 <= period <= 93:
                                    interpretations.append(f"📈 Quarterly (~{period:.1f}d)")
                                elif 360 <= period <= 370:
                                    interpretations.append(f"🌍 Yearly (~{period:.1f}d)")
                                else:
                                    interpretations.append(f"🔄 {period:.1f}d cycle")
                            st.success("**Patterns:** " + " | ".join(interpretations))
                        else:
                            st.info("No low-frequency patterns detected")

                    with col2:
                        st.markdown(f"#### 📅 High-Frequency Components (≤{high_freq_cutoff} days)")
                        if not high_freq_cycles.empty:
                            high_display = high_freq_cycles.copy()
                            high_display["period_days"] = high_display["period_days"].round(2)
                            high_display["amplitude"] = high_display["amplitude"].round(4)
                            st.dataframe(high_display, use_container_width=True, hide_index=True, height=150)

                            interpretations = []
                            for _, row in high_freq_cycles.iterrows():
                                period = row["period_days"]
                                if 6 <= period <= 8:
                                    interpretations.append(f"📅 Weekly (~{period:.1f}d)")
                                else:
                                    interpretations.append(f"🔄 {period:.1f}d cycle")
                            st.success("**Patterns:** " + " | ".join(interpretations))
                        else:
                            st.info("No high-frequency patterns detected")

                    # Visualize multi-scale decomposition
                    fig_decomp = go.Figure()

                    ts_train_clean = ts_train.dropna()

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=ts_train_clean.values,
                        mode='lines',
                        name='Original',
                        line=dict(color=PRIMARY_COLOR, width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=trend_train,
                        mode='lines',
                        name='Trend',
                        line=dict(color=SUCCESS_COLOR, width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=low_freq_seasonal_train,
                        mode='lines',
                        name=f'Low-Freq Seasonal (≥{low_freq_cutoff}d)',
                        line=dict(color='#F59E0B', width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=high_freq_seasonal_train,
                        mode='lines',
                        name=f'High-Freq Seasonal (≤{high_freq_cutoff}d)',
                        line=dict(color='#8B5CF6', width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=residual_train,
                        mode='lines',
                        name='Residual',
                        line=dict(color=SECONDARY_COLOR, width=1.5)
                    ))

                    fig_decomp.update_layout(
                        title="FFT Multi-Scale Decomposition (4 Components)",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(15, 23, 42, 0.8)',
                        font=dict(color=TEXT_COLOR, size=11),
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig_decomp, use_container_width=True, key="multiscale_decomposition_plot")

                    # Mark that we're using 4-component mode
                    use_4_components = True

                elif use_fft:
                    # ========== FFT BASIC DECOMPOSITION PATH (3 components) ==========
                    st.markdown("### 🌊 Step 1: FFT Basic Decomposition")
                    progress_bar = st.progress(0, text="Analyzing frequency components...")

                    # Apply FFT decomposition
                    decomp_result = _fft_decompose(ts_train, top_k=top_k_freq)

                    trend_train = decomp_result["trend"]
                    seasonal_train = decomp_result["seasonal_combined"]
                    residual_train = decomp_result["residual"]
                    cycles_info = decomp_result["cycles_info"]

                    # Store the index for later use
                    train_index = ts_train.index

                    progress_bar.progress(20, text="FFT decomposition complete ✓")

                    # Mark that we're using 3-component mode
                    use_4_components = False

                    # Show detected cycles
                    if not cycles_info.empty:
                        st.markdown("#### 🔍 Detected Frequency Components")
                        cycles_display = cycles_info.copy()
                        cycles_display["period_days"] = cycles_display["period_days"].round(2)
                        cycles_display["amplitude"] = cycles_display["amplitude"].round(4)
                        st.dataframe(cycles_display, use_container_width=True, hide_index=True)

                        # Interpret cycles
                        interpretations = []
                        for _, row in cycles_info.iterrows():
                            period = row["period_days"]
                            if 6 <= period <= 8:
                                interpretations.append(f"📅 **Weekly pattern** (~{period:.1f} days)")
                            elif 28 <= period <= 31:
                                interpretations.append(f"📆 **Monthly pattern** (~{period:.1f} days)")
                            elif 89 <= period <= 93:
                                interpretations.append(f"📈 **Quarterly pattern** (~{period:.1f} days)")
                            elif 360 <= period <= 370:
                                interpretations.append(f"🌍 **Yearly pattern** (~{period:.1f} days)")
                            else:
                                interpretations.append(f"🔄 **{period:.1f}-day cycle**")

                        st.success("**Identified patterns:** " + " | ".join(interpretations))

                    # Visualize FFT decomposition
                    fig_decomp = go.Figure()

                    ts_train_clean = ts_train.dropna()

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=ts_train_clean.values,
                        mode='lines',
                        name='Original',
                        line=dict(color=PRIMARY_COLOR, width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=trend_train,
                        mode='lines',
                        name='Trend (Low-freq)',
                        line=dict(color=SUCCESS_COLOR, width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=seasonal_train,
                        mode='lines',
                        name=f'Seasonal (Top {top_k_freq} freqs)',
                        line=dict(color=WARNING_COLOR, width=2)
                    ))

                    fig_decomp.add_trace(go.Scatter(
                        x=train_index,
                        y=residual_train,
                        mode='lines',
                        name='Residual',
                        line=dict(color=SECONDARY_COLOR, width=1.5)
                    ))

                    fig_decomp.update_layout(
                        title="FFT-Based Time Series Decomposition (Training Data)",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(15, 23, 42, 0.8)',
                        font=dict(color=TEXT_COLOR, size=11),
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig_decomp, use_container_width=True, key="fft_decomposition_plot")

                else:
                    # ========== CLASSICAL DECOMPOSITION PATH ==========
                    # Ensure seasonal_period is odd
                    if seasonal_period % 2 == 0:
                        seasonal_period += 1
                        st.info(f"ℹ️ Adjusted seasonal period to {seasonal_period} (requires odd numbers)")

                    # Validate data for decomposition
                    min_required_points = 2 * seasonal_period + 1
                    if len(ts_train.dropna()) < min_required_points:
                        st.error(
                            f"""
                            ❌ **Insufficient Data for Decomposition**

                            - **Training samples (non-null):** {len(ts_train.dropna())}
                            - **Minimum required:** {min_required_points} (2 × seasonal period + 1)
                        - **Seasonal period:** {seasonal_period}

                        **Solutions:**
                        1. Reduce test set size to get more training data
                        2. Use a smaller seasonal period
                        3. Load more historical data
                        """
                    )
                    return

                # ========== STEP 1: STL Decomposition ==========
                st.markdown("### 📐 Step 1: STL Decomposition")

                progress_bar = st.progress(0, text="Decomposing time series...")

                # Apply STL decomposition on training data (use dropna like in EDA)
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose

                    # Use seasonal_decompose instead of STL (more robust, same as EDA page)
                    result = seasonal_decompose(
                        ts_train.dropna(),
                        model='additive',
                        period=int(seasonal_period),
                        extrapolate_trend='freq'
                    )

                    trend_train = result.trend.values
                    seasonal_train = result.seasonal.values
                    residual_train = result.resid.values

                    # Store the index for later use
                    train_index = ts_train.dropna().index

                except Exception as e:
                    st.error(f"❌ Decomposition failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

                progress_bar.progress(20, text="Decomposition complete ✓")

                # Visualize decomposition
                fig_decomp = go.Figure()

                ts_train_clean = ts_train.dropna()

                fig_decomp.add_trace(go.Scatter(
                    x=train_index,
                    y=ts_train_clean.values,
                    mode='lines',
                    name='Original',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))

                fig_decomp.add_trace(go.Scatter(
                    x=train_index,
                    y=trend_train,
                    mode='lines',
                    name='Trend',
                    line=dict(color=SUCCESS_COLOR, width=2)
                ))

                fig_decomp.add_trace(go.Scatter(
                    x=train_index,
                    y=seasonal_train,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=WARNING_COLOR, width=2)
                ))

                fig_decomp.add_trace(go.Scatter(
                    x=train_index,
                    y=residual_train,
                    mode='lines',
                    name='Residual',
                    line=dict(color=SECONDARY_COLOR, width=1.5)
                ))

                fig_decomp.update_layout(
                    title="Time Series Decomposition (Training Data)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    plot_bgcolor='rgba(15, 23, 42, 0.95)',
                    paper_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(color=TEXT_COLOR, size=11),
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_decomp, use_container_width=True, key="decomposition_plot")

                # ========== STEP 2: Train XGBoost on Trend ==========
                st.markdown("### 🌳 Step 2: Train XGBoost on Trend")

                progress_bar.progress(30, text="Training XGBoost on Trend component...")

                # Create features for trend (time index, lags)
                def create_trend_features(series, lookback=14):
                    X_list = []
                    y_list = []
                    for i in range(lookback, len(series)):
                        X_list.append([
                            i,  # time index
                            series[i-1], series[i-7], series[i-14] if i >= 14 else series[0]
                        ])
                        y_list.append(series[i])
                    return np.array(X_list), np.array(y_list)

                X_trend, y_trend = create_trend_features(trend_train)

                # Train XGBoost
                xgb_trend = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_trend.fit(X_trend, y_trend)

                # Forecast trend for test period
                trend_forecast = []
                last_trend_vals = list(trend_train[-14:])

                for i in range(len(ts_test)):
                    time_idx = len(trend_train) + i
                    features = np.array([[
                        time_idx,
                        last_trend_vals[-1],
                        last_trend_vals[-7] if len(last_trend_vals) >= 7 else last_trend_vals[0],
                        last_trend_vals[-14] if len(last_trend_vals) >= 14 else last_trend_vals[0]
                    ]])
                    pred = xgb_trend.predict(features)[0]
                    trend_forecast.append(pred)
                    last_trend_vals.append(pred)

                trend_forecast = np.array(trend_forecast)

                progress_bar.progress(50, text="XGBoost training complete ✓")

                # ========== STEP 3: Train SARIMAX on Seasonal ==========
                st.markdown("### 📊 Step 3: Train SARIMAX on Seasonal")

                progress_bar.progress(60, text="Training SARIMAX on Seasonal component...")

                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    # Train SARIMAX on seasonal component
                    sarimax_seasonal = SARIMAX(
                        seasonal_train,
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    sarimax_fit = sarimax_seasonal.fit(disp=False)

                    # Forecast seasonal component
                    seasonal_forecast = sarimax_fit.forecast(steps=len(ts_test))

                    progress_bar.progress(70, text="SARIMAX training complete ✓")

                except Exception as e:
                    st.warning(f"⚠️ SARIMAX failed, using mean seasonal pattern: {str(e)}")
                    # Fallback: repeat seasonal pattern
                    seasonal_forecast = np.tile(seasonal_train[-seasonal_period:], int(np.ceil(len(ts_test) / seasonal_period)))[:len(ts_test)]
                    progress_bar.progress(70, text="Using fallback seasonal forecast ✓")

                # ========== STEP 3b (Multi-Scale Only): Train ANN on High-Frequency Seasonal ==========
                if use_4_components:
                    st.markdown("### 🎨 Step 3b: Train ANN on High-Frequency Seasonal")

                    progress_bar.progress(75, text="Training ANN on High-frequency component...")

                    # Create features for high-freq seasonal (time index, lags)
                    def create_seasonal_features(series, lookback=7):
                        X_list = []
                        y_list = []
                        for i in range(lookback, len(series)):
                            X_list.append([
                                i % high_freq_cutoff,  # cyclic time feature
                                series[i-1], series[i-3], series[i-7] if i >= 7 else series[0]
                            ])
                            y_list.append(series[i])
                        return np.array(X_list), np.array(y_list)

                    X_high_freq, y_high_freq = create_seasonal_features(high_freq_seasonal_train)

                    # Build ANN model
                    ann_high_freq = keras.Sequential([
                        layers.Dense(64, activation='relu', input_shape=(X_high_freq.shape[1],)),
                        layers.Dropout(0.2),
                        layers.Dense(32, activation='relu'),
                        layers.Dropout(0.1),
                        layers.Dense(1)
                    ])
                    ann_high_freq.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')

                    # Train ANN
                    ann_high_freq.fit(X_high_freq, y_high_freq, epochs=30, batch_size=16, verbose=0)

                    # Forecast high-frequency seasonal component
                    high_freq_forecast = []
                    last_high_freq_vals = list(high_freq_seasonal_train[-7:])

                    for i in range(len(ts_test)):
                        time_idx = len(high_freq_seasonal_train) + i
                        features = np.array([[
                            time_idx % high_freq_cutoff,
                            last_high_freq_vals[-1],
                            last_high_freq_vals[-3] if len(last_high_freq_vals) >= 3 else last_high_freq_vals[0],
                            last_high_freq_vals[-7] if len(last_high_freq_vals) >= 7 else last_high_freq_vals[0]
                        ]])
                        pred = ann_high_freq.predict(features, verbose=0)[0, 0]
                        high_freq_forecast.append(pred)
                        last_high_freq_vals.append(pred)

                    high_freq_forecast = np.array(high_freq_forecast)

                    progress_bar.progress(80, text="ANN training complete ✓")

                    # Update step numbers for 4-component mode
                    lstm_step_num = "4"
                    combine_step_num = "5"
                    lstm_progress = 85
                    combine_progress = 95
                else:
                    # 3-component mode: no ANN, use seasonal_train directly
                    high_freq_forecast = None  # Not used in 3-component mode
                    lstm_step_num = "4"
                    combine_step_num = "5"
                    lstm_progress = 80
                    combine_progress = 95

                # ========== STEP 4/5: Train LSTM on Residual ==========
                st.markdown(f"### 🧠 Step {lstm_step_num}: Train LSTM on Residual")

                progress_bar.progress(lstm_progress, text="Training LSTM on Residual component...")

                # Create sequences for LSTM
                def create_lstm_sequences(series, lookback=14):
                    X_list = []
                    y_list = []
                    for i in range(lookback, len(series)):
                        X_list.append(series[i-lookback:i])
                        y_list.append(series[i])
                    return np.array(X_list), np.array(y_list)

                lookback = min(14, len(residual_train) // 3)
                X_residual, y_residual = create_lstm_sequences(residual_train, lookback)
                X_residual = X_residual.reshape((X_residual.shape[0], X_residual.shape[1], 1))

                # Build LSTM model
                lstm_residual = keras.Sequential([
                    layers.LSTM(50, activation='relu', input_shape=(lookback, 1)),
                    layers.Dropout(0.1),
                    layers.Dense(1)
                ])
                lstm_residual.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')

                # Train LSTM
                lstm_residual.fit(X_residual, y_residual, epochs=20, batch_size=16, verbose=0)

                # Forecast residual
                residual_forecast = []
                last_residual_vals = list(residual_train[-lookback:])

                for i in range(len(ts_test)):
                    X_input = np.array(last_residual_vals[-lookback:]).reshape((1, lookback, 1))
                    pred = lstm_residual.predict(X_input, verbose=0)[0, 0]
                    residual_forecast.append(pred)
                    last_residual_vals.append(pred)

                residual_forecast = np.array(residual_forecast)

                progress_bar.progress(combine_progress - 5, text="LSTM training complete ✓")

                # ========== STEP 5: Combine Forecasts ==========
                st.markdown("### 🎯 Step 5: Combine Component Forecasts")

                progress_bar.progress(combine_progress, text="Combining forecasts...")

                # Final ensemble forecast
                if use_4_components:
                    # 4-component architecture: Trend + Low-Freq Seasonal + High-Freq Seasonal + Residual
                    final_forecast = trend_forecast + seasonal_forecast + high_freq_forecast + residual_forecast
                    st.info(f"🎯 **4-Component Ensemble**: Combining Trend (XGBoost) + Low-Freq Seasonal (SARIMAX) + High-Freq Seasonal (ANN) + Residual (LSTM)")
                else:
                    # 3-component architecture: Trend + Seasonal + Residual
                    final_forecast = trend_forecast + seasonal_forecast + residual_forecast
                    st.info(f"🎯 **3-Component Ensemble**: Combining Trend (XGBoost) + Seasonal (SARIMAX) + Residual (LSTM)")

                # Calculate metrics (using ts_test values)
                y_test_values = ts_test.values
                mae = mean_absolute_error(y_test_values, final_forecast)
                rmse = np.sqrt(mean_squared_error(y_test_values, final_forecast))
                mape = np.mean(np.abs((y_test_values - final_forecast) / y_test_values)) * 100
                accuracy = max(0, 100 - mape)

                decomp_metrics = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Accuracy": accuracy
                }

                # Store results in session state
                results_dict = {
                    "forecast": final_forecast,
                    "actuals": y_test_values,
                    "trend_forecast": trend_forecast,
                    "seasonal_forecast": seasonal_forecast,
                    "residual_forecast": residual_forecast,
                    "metrics": decomp_metrics,
                    "target_col": target_col,
                    "dates_test": ts_test.index,
                    "use_4_components": use_4_components
                }

                # Add high-frequency forecast if in 4-component mode
                if use_4_components:
                    results_dict["high_freq_forecast"] = high_freq_forecast

                st.session_state["decomposition_results"] = results_dict

                progress_bar.progress(100, text="Training complete! ✅")

                st.success("✅ **Decomposition Ensemble Trained Successfully!**")

                # Display metrics
                st.markdown("### 🏆 Ensemble Performance")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.plotly_chart(
                        _create_kpi_indicator("MAE", decomp_metrics["MAE"], "", PRIMARY_COLOR),
                        use_container_width=True,
                        key="decomp_mae"
                    )
                with col2:
                    st.plotly_chart(
                        _create_kpi_indicator("RMSE", decomp_metrics["RMSE"], "", SECONDARY_COLOR),
                        use_container_width=True,
                        key="decomp_rmse"
                    )
                with col3:
                    st.plotly_chart(
                        _create_kpi_indicator("MAPE", decomp_metrics["MAPE"], "%", WARNING_COLOR),
                        use_container_width=True,
                        key="decomp_mape"
                    )
                with col4:
                    st.plotly_chart(
                        _create_kpi_indicator("Accuracy", decomp_metrics["Accuracy"], "%", SUCCESS_COLOR),
                        use_container_width=True,
                        key="decomp_accuracy"
                    )

                # Visualize component contributions
                st.markdown("### 📊 Component Contributions to Final Forecast")

                fig_components = go.Figure()

                # Use test index for x-axis (dates)
                test_dates = ts_test.index

                fig_components.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_test_values,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color=PRIMARY_COLOR, width=3),
                    marker=dict(size=5)
                ))

                fig_components.add_trace(go.Scatter(
                    x=test_dates,
                    y=final_forecast,
                    mode='lines+markers',
                    name='Final Forecast',
                    line=dict(color=SUCCESS_COLOR, width=3, dash='dash'),
                    marker=dict(size=5, symbol='diamond')
                ))

                fig_components.add_trace(go.Scatter(
                    x=test_dates,
                    y=trend_forecast,
                    mode='lines',
                    name='Trend Component',
                    line=dict(color='#10B981', width=2, dash='dot'),
                    opacity=0.7
                ))

                # Add seasonal component (low-freq in 4-component mode)
                seasonal_label = 'Low-Freq Seasonal' if use_4_components else 'Seasonal Component'
                fig_components.add_trace(go.Scatter(
                    x=test_dates,
                    y=seasonal_forecast,
                    mode='lines',
                    name=seasonal_label,
                    line=dict(color=WARNING_COLOR, width=2, dash='dot'),
                    opacity=0.7
                ))

                # Add high-frequency component if in 4-component mode
                if use_4_components:
                    fig_components.add_trace(go.Scatter(
                        x=test_dates,
                        y=high_freq_forecast,
                        mode='lines',
                        name='High-Freq Seasonal',
                        line=dict(color='#A78BFA', width=2, dash='dot'),  # Purple color for high-freq
                        opacity=0.7
                    ))

                fig_components.add_trace(go.Scatter(
                    x=test_dates,
                    y=residual_forecast,
                    mode='lines',
                    name='Residual Component',
                    line=dict(color=SECONDARY_COLOR, width=2, dash='dot'),
                    opacity=0.7
                ))

                fig_components.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    plot_bgcolor='rgba(15, 23, 42, 0.95)',
                    paper_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(color=TEXT_COLOR, size=11),
                    height=500,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="right",
                        x=1.15,
                        bgcolor='rgba(15, 23, 42, 0.8)',
                        bordercolor=PRIMARY_COLOR,
                        borderwidth=1
                    ),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )

                st.plotly_chart(fig_components, use_container_width=True, key="component_forecast_plot")

                # Component statistics
                st.markdown("### 📈 Component Statistics")

                if use_4_components:
                    # 4-component statistics
                    component_stats = pd.DataFrame({
                        "Component": ["Trend", "Low-Freq Seasonal", "High-Freq Seasonal", "Residual", "Combined"],
                        "Mean": [
                            np.mean(trend_forecast),
                            np.mean(seasonal_forecast),
                            np.mean(high_freq_forecast),
                            np.mean(residual_forecast),
                            np.mean(final_forecast)
                        ],
                        "Std Dev": [
                            np.std(trend_forecast),
                            np.std(seasonal_forecast),
                            np.std(high_freq_forecast),
                            np.std(residual_forecast),
                            np.std(final_forecast)
                        ],
                        "Min": [
                            np.min(trend_forecast),
                            np.min(seasonal_forecast),
                            np.min(high_freq_forecast),
                            np.min(residual_forecast),
                            np.min(final_forecast)
                        ],
                        "Max": [
                            np.max(trend_forecast),
                            np.max(seasonal_forecast),
                            np.max(high_freq_forecast),
                            np.max(residual_forecast),
                            np.max(final_forecast)
                        ]
                    })
                else:
                    # 3-component statistics
                    component_stats = pd.DataFrame({
                        "Component": ["Trend", "Seasonal", "Residual", "Combined"],
                        "Mean": [
                            np.mean(trend_forecast),
                            np.mean(seasonal_forecast),
                            np.mean(residual_forecast),
                            np.mean(final_forecast)
                        ],
                        "Std Dev": [
                            np.std(trend_forecast),
                            np.std(seasonal_forecast),
                            np.std(residual_forecast),
                            np.std(final_forecast)
                        ],
                        "Min": [
                            np.min(trend_forecast),
                            np.min(seasonal_forecast),
                            np.min(residual_forecast),
                            np.min(final_forecast)
                        ],
                        "Max": [
                            np.max(trend_forecast),
                            np.max(seasonal_forecast),
                            np.max(residual_forecast),
                            np.max(final_forecast)
                        ]
                    })

                st.dataframe(
                    component_stats.style.format({
                        "Mean": "{:.4f}",
                        "Std Dev": "{:.4f}",
                        "Min": "{:.4f}",
                        "Max": "{:.4f}"
                    }).background_gradient(cmap='Blues', subset=['Mean', 'Std Dev']),
                    use_container_width=True,
                    height=200
                )

            except Exception as e:
                st.error(f"❌ **Decomposition Ensemble Training Failed**\n\n{str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Show existing results if available
    elif "decomposition_results" in st.session_state:
        st.info("ℹ️ **Previous decomposition results available.** Click 'Train Decomposition Ensemble' to generate new results.")

        decomp_data = st.session_state["decomposition_results"]
        decomp_metrics = decomp_data["metrics"]

        st.markdown("### 🏆 Previous Decomposition Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", decomp_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_decomp_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", decomp_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_decomp_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", decomp_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_decomp_mape"
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", decomp_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_decomp_accuracy"
            )


def _page_hybrid_stacking():
    """Stacking Ensemble with meta-learning to optimally combine base models."""

    # Stacking Ensemble Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.1));
                    border-left: 4px solid {PRIMARY_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
            <h2 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                🔗 Stacking Ensemble
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                Meta-model learns optimal combination of base models through two-level architecture
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation box
    st.markdown(
        f"""
        <div style='background: rgba(59, 130, 246, 0.1);
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1.5rem;'>
            <h4 style='color: {PRIMARY_COLOR}; margin-top: 0;'>📚 How Stacking Works</h4>
            <ul style='color: {TEXT_COLOR}; margin-bottom: 0;'>
                <li><strong>Level 0 (Base Models):</strong> SARIMAX, XGBoost, LSTM, and ANN make independent predictions</li>
                <li><strong>Out-of-Fold Training:</strong> Base models generate predictions on validation folds (prevents data leakage)</li>
                <li><strong>Level 1 (Meta-Model):</strong> Learns optimal combination strategy from base model predictions</li>
                <li><strong>Final Prediction:</strong> Meta-model combines base predictions adaptively</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Get selected dataset (original, Variant A, or Variant B)
    fused_data = _get_hybrid_dataset()
    selected_dataset_name = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset")

    # Show dataset info
    st.info(f"🎨 **Using dataset**: {selected_dataset_name} | **Features**: {len(fused_data.columns)} columns | **Samples**: {len(fused_data)} rows")

    if fused_data is None or (isinstance(fused_data, pd.DataFrame) and fused_data.empty):
        st.warning("⚠️ **No data available.** Please complete the Data Ingestion and Preprocessing steps first.")
        return

    # Configuration Section
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        target_col = st.selectbox(
            "Target Column",
            options=[col for col in fused_data.columns if col not in ["Date", "datetime"]],
            index=0 if "Target_1" not in fused_data.columns else list(fused_data.columns).index("Target_1")
        )

    with col2:
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data for final testing"
        )

    with col3:
        n_cv_splits = st.slider(
            "CV Splits",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of time series cross-validation splits for out-of-fold predictions"
        )

    with col4:
        meta_model_type = st.selectbox(
            "Meta-Model",
            options=["Ridge Regression", "Lasso Regression", "XGBoost"],
            index=0,
            help="Ridge: Simple linear (recommended) | Lasso: Feature selection | XGBoost: Non-linear"
        )

    # Train button
    if st.button("🚀 Train Stacking Ensemble", type="primary", use_container_width=True):
        try:
            # Import required libraries
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from xgboost import XGBRegressor
            from tensorflow import keras
            from tensorflow.keras import layers
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.linear_model import Ridge, Lasso

            # Prepare data
            if "Date" not in fused_data.columns:
                st.error("❌ **Error**: 'Date' column not found in data.")
                return

            df = fused_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = pd.to_datetime(df["Date"])

            df = df.set_index("Date")
            ts = df[target_col]

            # Ensure daily frequency and interpolate
            ts = ts.asfreq("D").interpolate(method="time", limit_direction="both")

            # Train-test split
            test_size_samples = int(len(ts) * (test_size / 100))
            ts_train = ts.iloc[:-test_size_samples]
            ts_test = ts.iloc[-test_size_samples:]

            st.markdown(f"**Data Split**: {len(ts_train)} training samples, {len(ts_test)} test samples")

            # Progress tracking
            progress_bar = st.progress(0, text="Initializing stacking ensemble...")

            # ========== STEP 1: Generate Out-of-Fold Predictions ==========
            st.markdown("### 🔄 Step 1: Generate Out-of-Fold (OOF) Predictions")
            st.info("Training base models on cross-validation folds to generate OOF predictions (prevents data leakage)")

            progress_bar.progress(5, text="Setting up time series cross-validation...")

            tscv = TimeSeriesSplit(n_splits=n_cv_splits)

            # Initialize OOF prediction arrays
            oof_sarimax = np.full(len(ts_train), np.nan)
            oof_xgb = np.full(len(ts_train), np.nan)
            oof_lstm = np.full(len(ts_train), np.nan)
            oof_ann = np.full(len(ts_train), np.nan)

            fold_num = 0
            for train_idx, val_idx in tscv.split(ts_train):
                fold_num += 1
                progress_pct = 5 + (fold_num / n_cv_splits) * 60  # 5% to 65%
                progress_bar.progress(int(progress_pct), text=f"Training base models on fold {fold_num}/{n_cv_splits}...")

                # Split data for this fold
                train_fold = ts_train.iloc[train_idx]
                val_fold = ts_train.iloc[val_idx]

                # === SARIMAX ===
                try:
                    sarimax_model = SARIMAX(
                        train_fold,
                        order=(1, 1, 1),
                        seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    oof_sarimax[val_idx] = sarimax_model.forecast(steps=len(val_idx)).values
                except Exception as e:
                    st.warning(f"⚠️ SARIMAX failed on fold {fold_num}: {str(e)}")
                    oof_sarimax[val_idx] = np.mean(train_fold)

                # === XGBoost ===
                try:
                    # Create lagged features
                    def create_lag_features(series, n_lags=7):
                        X_list, y_list = [], []
                        for i in range(n_lags, len(series)):
                            X_list.append([series[i-j] for j in range(1, n_lags+1)])
                            y_list.append(series[i])
                        return np.array(X_list), np.array(y_list)

                    X_train_xgb, y_train_xgb = create_lag_features(train_fold.values, n_lags=7)
                    xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                    xgb_model.fit(X_train_xgb, y_train_xgb, verbose=0)

                    # Predict on validation fold
                    for i, idx in enumerate(val_idx):
                        if idx >= 7:  # Need at least 7 previous values
                            X_pred = ts_train.iloc[idx-7:idx].values.reshape(1, -1)
                            oof_xgb[idx] = xgb_model.predict(X_pred)[0]
                        else:
                            oof_xgb[idx] = np.mean(train_fold)
                except Exception as e:
                    st.warning(f"⚠️ XGBoost failed on fold {fold_num}: {str(e)}")
                    oof_xgb[val_idx] = np.mean(train_fold)

                # === LSTM ===
                try:
                    def create_lstm_sequences(series, lookback=14):
                        X_list, y_list = [], []
                        for i in range(lookback, len(series)):
                            X_list.append(series[i-lookback:i])
                            y_list.append(series[i])
                        return np.array(X_list), np.array(y_list)

                    lookback = min(14, len(train_fold) // 3)
                    X_train_lstm, y_train_lstm = create_lstm_sequences(train_fold.values, lookback)
                    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

                    lstm_model = keras.Sequential([
                        layers.LSTM(32, activation='relu', input_shape=(lookback, 1)),
                        layers.Dense(1)
                    ])
                    lstm_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
                    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=15, batch_size=16, verbose=0)

                    # Predict on validation fold
                    for i, idx in enumerate(val_idx):
                        if idx >= lookback:
                            X_pred = ts_train.iloc[idx-lookback:idx].values.reshape((1, lookback, 1))
                            oof_lstm[idx] = lstm_model.predict(X_pred, verbose=0)[0, 0]
                        else:
                            oof_lstm[idx] = np.mean(train_fold)
                except Exception as e:
                    st.warning(f"⚠️ LSTM failed on fold {fold_num}: {str(e)}")
                    oof_lstm[val_idx] = np.mean(train_fold)

                # === ANN ===
                try:
                    X_train_ann, y_train_ann = create_lag_features(train_fold.values, n_lags=7)

                    ann_model = keras.Sequential([
                        layers.Dense(32, activation='relu', input_shape=(7,)),
                        layers.Dropout(0.2),
                        layers.Dense(16, activation='relu'),
                        layers.Dense(1)
                    ])
                    ann_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
                    ann_model.fit(X_train_ann, y_train_ann, epochs=15, batch_size=16, verbose=0)

                    # Predict on validation fold
                    for i, idx in enumerate(val_idx):
                        if idx >= 7:
                            X_pred = ts_train.iloc[idx-7:idx].values.reshape(1, -1)
                            oof_ann[idx] = ann_model.predict(X_pred, verbose=0)[0, 0]
                        else:
                            oof_ann[idx] = np.mean(train_fold)
                except Exception as e:
                    st.warning(f"⚠️ ANN failed on fold {fold_num}: {str(e)}")
                    oof_ann[val_idx] = np.mean(train_fold)

            progress_bar.progress(65, text="OOF predictions generated ✓")

            # Display OOF statistics
            st.markdown("#### 📊 Out-of-Fold Performance")

            # Remove NaN values for metrics
            valid_mask = ~(np.isnan(oof_sarimax) | np.isnan(oof_xgb) | np.isnan(oof_lstm) | np.isnan(oof_ann))
            y_train_valid = ts_train.values[valid_mask]
            oof_sarimax_valid = oof_sarimax[valid_mask]
            oof_xgb_valid = oof_xgb[valid_mask]
            oof_lstm_valid = oof_lstm[valid_mask]
            oof_ann_valid = oof_ann[valid_mask]

            oof_metrics = []
            for name, preds in [("SARIMAX", oof_sarimax_valid), ("XGBoost", oof_xgb_valid),
                               ("LSTM", oof_lstm_valid), ("ANN", oof_ann_valid)]:
                if len(preds) > 0:
                    mae = mean_absolute_error(y_train_valid, preds)
                    rmse = np.sqrt(mean_squared_error(y_train_valid, preds))
                    mape = np.mean(np.abs((y_train_valid - preds) / y_train_valid)) * 100
                    oof_metrics.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

            oof_df = pd.DataFrame(oof_metrics)
            st.dataframe(
                oof_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%"})
                           .background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE', 'MAPE']),
                use_container_width=True,
                height=200
            )

            # ========== STEP 2: Train Meta-Model ==========
            st.markdown("### 🧠 Step 2: Train Meta-Model")
            st.info(f"Training {meta_model_type} to learn optimal combination of base model predictions")

            progress_bar.progress(70, text="Training meta-model...")

            # Prepare meta-model features (OOF predictions as features)
            X_meta = np.column_stack([oof_sarimax_valid, oof_xgb_valid, oof_lstm_valid, oof_ann_valid])
            y_meta = y_train_valid

            # Train meta-model
            if meta_model_type == "Ridge Regression":
                meta_model = Ridge(alpha=1.0)
            elif meta_model_type == "Lasso Regression":
                meta_model = Lasso(alpha=0.01, max_iter=5000)
            else:  # XGBoost
                meta_model = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)

            meta_model.fit(X_meta, y_meta)

            # Display meta-model weights/importances
            if hasattr(meta_model, 'coef_'):
                weights = meta_model.coef_
                intercept = meta_model.intercept_
                st.markdown("#### 🎯 Learned Meta-Model Weights")

                weights_df = pd.DataFrame({
                    "Base Model": ["SARIMAX", "XGBoost", "LSTM", "ANN"],
                    "Weight": weights,
                    "Abs Weight": np.abs(weights)
                })

                # Normalize weights for percentage
                total_abs_weight = np.sum(np.abs(weights))
                weights_df["Contribution %"] = (weights_df["Abs Weight"] / total_abs_weight * 100) if total_abs_weight > 0 else 0

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(
                        weights_df.style.format({"Weight": "{:.4f}", "Abs Weight": "{:.4f}", "Contribution %": "{:.2f}%"})
                                     .background_gradient(cmap='Blues', subset=['Abs Weight']),
                        use_container_width=True,
                        height=200
                    )
                with col2:
                    st.metric("Intercept", f"{intercept:.4f}")
                    st.caption("Meta-model equation: y = w₁×SARIMAX + w₂×XGB + w₃×LSTM + w₄×ANN + b")
            elif hasattr(meta_model, 'feature_importances_'):
                importances = meta_model.feature_importances_
                st.markdown("#### 🎯 Meta-Model Feature Importances")

                imp_df = pd.DataFrame({
                    "Base Model": ["SARIMAX", "XGBoost", "LSTM", "ANN"],
                    "Importance": importances,
                    "Contribution %": importances * 100
                })

                st.dataframe(
                    imp_df.style.format({"Importance": "{:.4f}", "Contribution %": "{:.2f}%"})
                                 .background_gradient(cmap='Greens', subset=['Importance']),
                    use_container_width=True,
                    height=200
                )

            progress_bar.progress(80, text="Meta-model trained ✓")

            # ========== STEP 3: Train Final Base Models & Generate Predictions ==========
            st.markdown("### 🎯 Step 3: Generate Final Stacked Predictions")
            st.info("Training base models on full training set and generating test predictions")

            progress_bar.progress(85, text="Training final base models...")

            # Train final SARIMAX
            sarimax_final = SARIMAX(
                ts_train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            sarimax_forecast = sarimax_final.forecast(steps=len(ts_test)).values

            # Train final XGBoost
            X_train_xgb_final, y_train_xgb_final = create_lag_features(ts_train.values, n_lags=7)
            xgb_final = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            xgb_final.fit(X_train_xgb_final, y_train_xgb_final, verbose=0)

            xgb_forecast = []
            last_vals = list(ts_train.values[-7:])
            for i in range(len(ts_test)):
                X_pred = np.array(last_vals[-7:]).reshape(1, -1)
                pred = xgb_final.predict(X_pred)[0]
                xgb_forecast.append(pred)
                last_vals.append(pred)
            xgb_forecast = np.array(xgb_forecast)

            progress_bar.progress(90, text="Training LSTM and ANN...")

            # Train final LSTM
            lookback = min(14, len(ts_train) // 3)
            X_train_lstm_final, y_train_lstm_final = create_lstm_sequences(ts_train.values, lookback)
            X_train_lstm_final = X_train_lstm_final.reshape((X_train_lstm_final.shape[0], X_train_lstm_final.shape[1], 1))

            lstm_final = keras.Sequential([
                layers.LSTM(32, activation='relu', input_shape=(lookback, 1)),
                layers.Dense(1)
            ])
            lstm_final.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            lstm_final.fit(X_train_lstm_final, y_train_lstm_final, epochs=20, batch_size=16, verbose=0)

            lstm_forecast = []
            last_lstm_vals = list(ts_train.values[-lookback:])
            for i in range(len(ts_test)):
                X_pred = np.array(last_lstm_vals[-lookback:]).reshape((1, lookback, 1))
                pred = lstm_final.predict(X_pred, verbose=0)[0, 0]
                lstm_forecast.append(pred)
                last_lstm_vals.append(pred)
            lstm_forecast = np.array(lstm_forecast)

            # Train final ANN
            X_train_ann_final, y_train_ann_final = create_lag_features(ts_train.values, n_lags=7)

            ann_final = keras.Sequential([
                layers.Dense(32, activation='relu', input_shape=(7,)),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            ann_final.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            ann_final.fit(X_train_ann_final, y_train_ann_final, epochs=20, batch_size=16, verbose=0)

            ann_forecast = []
            last_ann_vals = list(ts_train.values[-7:])
            for i in range(len(ts_test)):
                X_pred = np.array(last_ann_vals[-7:]).reshape(1, -1)
                pred = ann_final.predict(X_pred, verbose=0)[0, 0]
                ann_forecast.append(pred)
                last_ann_vals.append(pred)
            ann_forecast = np.array(ann_forecast)

            progress_bar.progress(95, text="Generating stacked predictions...")

            # Meta-model combines base predictions
            X_test_meta = np.column_stack([sarimax_forecast, xgb_forecast, lstm_forecast, ann_forecast])
            final_stacked_forecast = meta_model.predict(X_test_meta)

            # Calculate metrics
            y_test_values = ts_test.values
            mae = mean_absolute_error(y_test_values, final_stacked_forecast)
            rmse = np.sqrt(mean_squared_error(y_test_values, final_stacked_forecast))
            mape = np.mean(np.abs((y_test_values - final_stacked_forecast) / y_test_values)) * 100
            accuracy = max(0, 100 - mape)

            stacking_metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Accuracy": accuracy
            }

            # Store results in session state
            st.session_state["stacking_results"] = {
                "forecast": final_stacked_forecast,
                "actuals": y_test_values,
                "sarimax_forecast": sarimax_forecast,
                "xgb_forecast": xgb_forecast,
                "lstm_forecast": lstm_forecast,
                "ann_forecast": ann_forecast,
                "metrics": stacking_metrics,
                "meta_model_type": meta_model_type,
                "target_col": target_col,
                "dates_test": ts_test.index
            }

            progress_bar.progress(100, text="Training complete! ✅")

            st.success("✅ **Stacking Ensemble Trained Successfully!**")

            # Display metrics
            st.markdown("### 🏆 Stacking Ensemble Performance")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.plotly_chart(
                    _create_kpi_indicator("MAE", stacking_metrics["MAE"], "", PRIMARY_COLOR),
                    use_container_width=True,
                    key="stacking_mae"
                )
            with col2:
                st.plotly_chart(
                    _create_kpi_indicator("RMSE", stacking_metrics["RMSE"], "", SECONDARY_COLOR),
                    use_container_width=True,
                    key="stacking_rmse"
                )
            with col3:
                st.plotly_chart(
                    _create_kpi_indicator("MAPE", stacking_metrics["MAPE"], "%", WARNING_COLOR),
                    use_container_width=True,
                    key="stacking_mape"
                )
            with col4:
                st.plotly_chart(
                    _create_kpi_indicator("Accuracy", stacking_metrics["Accuracy"], "%", SUCCESS_COLOR),
                    use_container_width=True,
                    key="stacking_accuracy"
                )

            # Visualize predictions
            st.markdown("### 📊 Stacked Forecast vs Actuals")

            fig = go.Figure()

            test_dates = ts_test.index

            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_test_values,
                mode='lines+markers',
                name='Actual',
                line=dict(color=PRIMARY_COLOR, width=3),
                marker=dict(size=6)
            ))

            fig.add_trace(go.Scatter(
                x=test_dates,
                y=final_stacked_forecast,
                mode='lines+markers',
                name='Stacked Forecast (Meta-Model)',
                line=dict(color=SUCCESS_COLOR, width=3, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))

            # Add base model predictions (with lower opacity)
            for name, forecast, color in [
                ("SARIMAX", sarimax_forecast, '#F59E0B'),
                ("XGBoost", xgb_forecast, '#10B981'),
                ("LSTM", lstm_forecast, '#8B5CF6'),
                ("ANN", ann_forecast, '#EC4899')
            ]:
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=forecast,
                    mode='lines',
                    name=f'{name} (Base)',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.5
                ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color=TEXT_COLOR, size=11),
                height=500,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1.15,
                    bgcolor='rgba(15, 23, 42, 0.8)',
                    bordercolor=PRIMARY_COLOR,
                    borderwidth=1
                ),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )

            st.plotly_chart(fig, use_container_width=True, key="stacking_forecast_plot")

            # Base model comparison
            st.markdown("### 📈 Base Models vs Stacked Performance")

            base_metrics = []
            for name, forecast in [
                ("SARIMAX", sarimax_forecast),
                ("XGBoost", xgb_forecast),
                ("LSTM", lstm_forecast),
                ("ANN", ann_forecast),
                ("**Stacked**", final_stacked_forecast)
            ]:
                mae_base = mean_absolute_error(y_test_values, forecast)
                rmse_base = np.sqrt(mean_squared_error(y_test_values, forecast))
                mape_base = np.mean(np.abs((y_test_values - forecast) / y_test_values)) * 100
                base_metrics.append({
                    "Model": name,
                    "MAE": mae_base,
                    "RMSE": rmse_base,
                    "MAPE": mape_base,
                    "Accuracy": max(0, 100 - mape_base)
                })

            comparison_df = pd.DataFrame(base_metrics)
            st.dataframe(
                comparison_df.style.format({
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE": "{:.2f}%",
                    "Accuracy": "{:.2f}%"
                }).background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE', 'MAPE'])
                  .background_gradient(cmap='Greens', subset=['Accuracy']),
                use_container_width=True,
                height=250
            )

        except Exception as e:
            st.error(f"❌ **Stacking Ensemble Training Failed**\n\n{str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

    # Show existing results if available
    elif "stacking_results" in st.session_state:
        st.info("ℹ️ **Previous stacking results available.** Click 'Train Stacking Ensemble' to generate new results.")

        stacking_data = st.session_state["stacking_results"]
        stacking_metrics = stacking_data["metrics"]

        st.markdown("### 🏆 Previous Stacking Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", stacking_metrics["MAE"], "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_stacking_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", stacking_metrics["RMSE"], "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_stacking_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", stacking_metrics["MAPE"], "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_stacking_mape"
            )
        with col4:
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", stacking_metrics["Accuracy"], "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_stacking_accuracy"
            )


def _page_hybrid_lstm_sarimax():
    """LSTM-SARIMAX Hybrid: Two-stage model with LSTM + SARIMAX residual correction."""

    # LSTM-SARIMAX Hybrid Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(79, 70, 229, 0.1));
                    border-left: 4px solid {PRIMARY_COLOR};
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);'>
            <h2 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(99, 102, 241, 0.5);'>
                🧠 LSTM-SARIMAX Hybrid
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                <strong>Stage 1:</strong> LSTM learns primary time-series signal | <strong>Stage 2:</strong> SARIMAX corrects residuals
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get selected dataset
    fused_data = _get_hybrid_dataset()
    selected_dataset_name = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset")

    if fused_data is None or fused_data.empty:
        st.warning("⚠️ No data available. Please load and process data first.")
        return

    # Show dataset info
    st.info(f"🎨 **Using dataset**: {selected_dataset_name} | **Features**: {len(fused_data.columns)} columns | **Samples**: {len(fused_data)} rows")

    # Configuration Section
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Stage 1: LSTM")
        lookback = st.slider("Lookback Window", min_value=7, max_value=60, value=14, step=1, help="Number of past time steps for LSTM sequences")
        lstm_units = st.slider("LSTM Units", min_value=16, max_value=256, value=50, step=16, help="Number of LSTM neurons")
        lstm_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, help="Dropout for regularization")
        lstm_epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20, step=5, help="Number of training iterations")
        lstm_batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1, help="Samples per training batch")

    with col2:
        st.markdown("#### 📊 Stage 2: SARIMAX")
        st.markdown("**ARIMA Order (p, d, q)**")
        sarimax_p = st.slider("p (AR order)", min_value=0, max_value=5, value=1, step=1)
        sarimax_d = st.slider("d (Differencing)", min_value=0, max_value=2, value=1, step=1)
        sarimax_q = st.slider("q (MA order)", min_value=0, max_value=5, value=1, step=1)

        st.markdown("**Seasonal Order (P, D, Q, s)**")
        seasonal_p = st.slider("P (Seasonal AR)", min_value=0, max_value=2, value=1, step=1)
        seasonal_d = st.slider("D (Seasonal Diff)", min_value=0, max_value=1, value=0, step=1)
        seasonal_q = st.slider("Q (Seasonal MA)", min_value=0, max_value=2, value=1, step=1)
        seasonal_s = st.slider("s (Seasonal Period)", min_value=4, max_value=52, value=7, step=1, help="7 for weekly, 12 for monthly")

        use_exog = st.checkbox("Use Exogenous Features in SARIMAX", value=True, help="Include tabular features in SARIMAX stage")

    col3, col4 = st.columns(2)
    with col3:
        # Filter for numeric columns only (exclude datetime columns)
        numeric_cols = [col for col in fused_data.select_dtypes(include=[np.number]).columns if col != "Date"]
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset. Please check your data.")
            return
        target_col = st.selectbox("Target Variable", options=numeric_cols, index=0)
    with col4:
        eval_tail_size = st.slider("Evaluation Set Size", min_value=0.10, max_value=0.30, value=0.15, step=0.05, help="Fraction of data for final evaluation")

    oof_splits = st.slider("Out-of-Fold Splits", min_value=2, max_value=10, value=4, step=1, help="Number of time-series cross-validation splits")

    # Build Configuration Dictionary
    config = {
        "random_state": 42,
        "features": {
            "lags": [7, 14, 21],
            "rolls": [7, 14, 21],
        },
        "eval": {"tail_size": eval_tail_size},
        "oof_splits": oof_splits,
        "stage1": {
            "lookback": lookback,
            "units": lstm_units,
            "dropout": lstm_dropout,
            "epochs": lstm_epochs,
            "batch_size": lstm_batch_size,
            "lr": 0.001,
        },
        "stage2": {
            "order": [sarimax_p, sarimax_d, sarimax_q],
            "seasonal_order": [seasonal_p, seasonal_d, seasonal_q, seasonal_s],
            "use_exog": use_exog,
        },
    }

    # Training Button
    if st.button("🚀 Train LSTM-SARIMAX Hybrid", type="primary", use_container_width=True):
        with st.spinner("🔄 Training LSTM-SARIMAX hybrid model... This may take several minutes."):
            try:
                # Import the hybrid model
                from app_core.models.ml.lstm_sarimax import LSTMSARIMAXHybrid

                # Initialize and train
                hybrid_model = LSTMSARIMAXHybrid()
                artifacts = hybrid_model.fit(fused_data, target=target_col, config=config)

                # Store results in session state
                st.session_state["lstm_sarimax_results"] = {
                    "artifacts": artifacts,
                    "config": config,
                    "dataset_name": selected_dataset_name,
                }

                st.success("✅ **LSTM-SARIMAX Hybrid trained successfully!**")

                # Display Metrics
                st.markdown("### 🏆 Hybrid Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.plotly_chart(
                        _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                        use_container_width=True,
                        key="lstm_sarimax_mae"
                    )
                with col2:
                    st.plotly_chart(
                        _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                        use_container_width=True,
                        key="lstm_sarimax_rmse"
                    )
                with col3:
                    st.plotly_chart(
                        _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                        use_container_width=True,
                        key="lstm_sarimax_mape"
                    )
                with col4:
                    accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
                    st.plotly_chart(
                        _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                        use_container_width=True,
                        key="lstm_sarimax_accuracy"
                    )

                # Show artifacts paths
                with st.expander("📁 Model Artifacts"):
                    st.markdown(f"**Stage 1 (LSTM):** `{artifacts.stage1_path}`")
                    st.markdown(f"**Stage 2 (SARIMAX):** `{artifacts.stage2_path}`")
                    st.markdown(f"**Scaler:** `{artifacts.scaler_path}`")

            except ImportError as ie:
                st.error(f"❌ **Import Error**: {str(ie)}\n\nPlease ensure statsmodels and TensorFlow are installed.")
            except Exception as e:
                st.error(f"❌ **Training Failed**\n\n{str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Show existing results if available
    elif "lstm_sarimax_results" in st.session_state:
        st.info("ℹ️ **Previous LSTM-SARIMAX results available.** Click 'Train LSTM-SARIMAX Hybrid' to generate new results.")

        results = st.session_state["lstm_sarimax_results"]
        artifacts = results["artifacts"]

        st.markdown("### 🏆 Previous Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_lstm_sarimax_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_lstm_sarimax_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_lstm_sarimax_mape"
            )
        with col4:
            accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_lstm_sarimax_accuracy"
            )


def _page_hybrid_lstm_xgb():
    """LSTM-XGBoost Hybrid: Two-stage model with LSTM + XGBoost residual correction."""

    # LSTM-XGBoost Hybrid Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(219, 39, 119, 0.1));
                    border-left: 4px solid #ec4899;
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(236, 72, 153, 0.2);'>
            <h2 style='color: #ec4899; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(236, 72, 153, 0.5);'>
                🚀 LSTM-XGBoost Hybrid
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                <strong>Stage 1:</strong> LSTM learns primary time-series signal | <strong>Stage 2:</strong> XGBoost corrects residuals
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get selected dataset
    fused_data = _get_hybrid_dataset()
    selected_dataset_name = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset")

    if fused_data is None or fused_data.empty:
        st.warning("⚠️ No data available. Please load and process data first.")
        return

    # Show dataset info
    st.info(f"🎨 **Using dataset**: {selected_dataset_name} | **Features**: {len(fused_data.columns)} columns | **Samples**: {len(fused_data)} rows")

    # Configuration Section
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Stage 1: LSTM")
        lookback = st.slider("Lookback Window", min_value=7, max_value=60, value=14, step=1, help="Number of past time steps for LSTM sequences", key="xgb_lookback")
        lstm_units = st.slider("LSTM Units", min_value=16, max_value=256, value=50, step=16, help="Number of LSTM neurons", key="xgb_units")
        lstm_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, help="Dropout for regularization", key="xgb_dropout")
        lstm_epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20, step=5, help="Number of training iterations", key="xgb_epochs")
        lstm_batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1, help="Samples per training batch", key="xgb_batch")

    with col2:
        st.markdown("#### 🌳 Stage 2: XGBoost")
        n_estimators = st.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50, help="Number of boosting rounds")
        max_depth = st.slider("Max Tree Depth", min_value=3, max_value=10, value=5, step=1, help="Maximum depth of each tree")
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01, help="Step size for boosting")
        early_stopping = st.slider("Early Stopping Rounds", min_value=5, max_value=50, value=10, step=5, help="Stop if no improvement")

    col3, col4 = st.columns(2)
    with col3:
        # Filter for numeric columns only (exclude datetime columns)
        numeric_cols = [col for col in fused_data.select_dtypes(include=[np.number]).columns if col != "Date"]
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset. Please check your data.")
            return
        target_col = st.selectbox("Target Variable", options=numeric_cols, index=0, key="xgb_target")
    with col4:
        eval_tail_size = st.slider("Evaluation Set Size", min_value=0.10, max_value=0.30, value=0.15, step=0.05, help="Fraction of data for final evaluation", key="xgb_eval")

    oof_splits = st.slider("Out-of-Fold Splits", min_value=2, max_value=10, value=4, step=1, help="Number of time-series cross-validation splits", key="xgb_oof")

    # Build Configuration Dictionary
    config = {
        "random_state": 42,
        "features": {
            "lags": [7, 14, 21],
            "rolls": [7, 14, 21],
        },
        "eval": {"tail_size": eval_tail_size},
        "oof_splits": oof_splits,
        "stage1": {
            "lookback": lookback,
            "units": lstm_units,
            "dropout": lstm_dropout,
            "epochs": lstm_epochs,
            "batch_size": lstm_batch_size,
            "lr": 0.001,
        },
        "stage2": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "early_stopping_rounds": early_stopping,
        },
    }

    # Training Button
    if st.button("🚀 Train LSTM-XGBoost Hybrid", type="primary", use_container_width=True):
        with st.spinner("🔄 Training LSTM-XGBoost hybrid model... This may take several minutes."):
            try:
                # Import the hybrid model
                from app_core.models.ml.lstm_xgb import LSTMXGBHybrid

                # Initialize and train
                hybrid_model = LSTMXGBHybrid()
                artifacts = hybrid_model.fit(fused_data, target=target_col, config=config)

                # Store results in session state
                st.session_state["lstm_xgb_results"] = {
                    "artifacts": artifacts,
                    "config": config,
                    "dataset_name": selected_dataset_name,
                }

                st.success("✅ **LSTM-XGBoost Hybrid trained successfully!**")

                # Display Metrics
                st.markdown("### 🏆 Hybrid Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.plotly_chart(
                        _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                        use_container_width=True,
                        key="lstm_xgb_mae"
                    )
                with col2:
                    st.plotly_chart(
                        _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                        use_container_width=True,
                        key="lstm_xgb_rmse"
                    )
                with col3:
                    st.plotly_chart(
                        _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                        use_container_width=True,
                        key="lstm_xgb_mape"
                    )
                with col4:
                    accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
                    st.plotly_chart(
                        _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                        use_container_width=True,
                        key="lstm_xgb_accuracy"
                    )

                # Show artifacts paths
                with st.expander("📁 Model Artifacts"):
                    st.markdown(f"**Stage 1 (LSTM):** `{artifacts.stage1_path}`")
                    st.markdown(f"**Stage 2 (XGBoost):** `{artifacts.stage2_path}`")
                    st.markdown(f"**Scaler:** `{artifacts.scaler_path}`")

            except ImportError as ie:
                st.error(f"❌ **Import Error**: {str(ie)}\n\nPlease ensure XGBoost and TensorFlow are installed.")
            except Exception as e:
                st.error(f"❌ **Training Failed**\n\n{str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Show existing results if available
    elif "lstm_xgb_results" in st.session_state:
        st.info("ℹ️ **Previous LSTM-XGBoost results available.** Click 'Train LSTM-XGBoost Hybrid' to generate new results.")

        results = st.session_state["lstm_xgb_results"]
        artifacts = results["artifacts"]

        st.markdown("### 🏆 Previous Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_lstm_xgb_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_lstm_xgb_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_lstm_xgb_mape"
            )
        with col4:
            accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_lstm_xgb_accuracy"
            )


def _page_hybrid_lstm_ann():
    """LSTM-ANN Hybrid: Two-stage model with LSTM + ANN residual correction."""

    # LSTM-ANN Hybrid Header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(147, 51, 234, 0.1));
                    border-left: 4px solid #a855f7;
                    padding: 1.25rem;
                    border-radius: 12px;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(168, 85, 247, 0.2);'>
            <h2 style='color: #a855f7; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(168, 85, 247, 0.5);'>
                🎯 LSTM-ANN Hybrid
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                <strong>Stage 1:</strong> LSTM learns primary time-series signal | <strong>Stage 2:</strong> ANN neural network corrects residuals
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get selected dataset
    fused_data = _get_hybrid_dataset()
    selected_dataset_name = st.session_state.get("hybrid_selected_dataset", "📊 Original Dataset")

    if fused_data is None or fused_data.empty:
        st.warning("⚠️ No data available. Please load and process data first.")
        return

    # Show dataset info
    st.info(f"🎨 **Using dataset**: {selected_dataset_name} | **Features**: {len(fused_data.columns)} columns | **Samples**: {len(fused_data)} rows")

    # Configuration Section
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Stage 1: LSTM")
        lookback = st.slider("Lookback Window", min_value=7, max_value=60, value=14, step=1, help="Number of past time steps for LSTM sequences", key="ann_lookback")
        lstm_units = st.slider("LSTM Units", min_value=16, max_value=256, value=50, step=16, help="Number of LSTM neurons", key="ann_units")
        lstm_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, help="Dropout for regularization", key="ann_dropout")
        lstm_epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20, step=5, help="Number of training iterations", key="ann_epochs")
        lstm_batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1, help="Samples per training batch", key="ann_batch")

    with col2:
        st.markdown("#### 🧬 Stage 2: ANN (Feedforward Neural Network)")
        st.markdown("**Network Architecture**")

        # ANN Layer Configuration
        layer1_units = st.slider("Layer 1 Units", min_value=16, max_value=256, value=64, step=16, help="Neurons in first hidden layer")
        layer2_units = st.slider("Layer 2 Units", min_value=16, max_value=128, value=32, step=16, help="Neurons in second hidden layer")

        ann_dropout = st.slider("ANN Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05, help="Dropout for ANN regularization", key="ann_dropout2")
        ann_epochs = st.slider("ANN Training Epochs", min_value=10, max_value=200, value=50, step=10, help="Number of ANN training iterations")
        ann_batch_size = st.selectbox("ANN Batch Size", options=[8, 16, 32, 64], index=1, help="Samples per ANN training batch", key="ann_batch2")
        ann_patience = st.slider("Early Stopping Patience", min_value=3, max_value=20, value=5, step=1, help="Stop if no improvement for N epochs")
        ann_lr = st.slider("ANN Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f", help="Step size for ANN optimization")

    col3, col4 = st.columns(2)
    with col3:
        # Filter for numeric columns only (exclude datetime columns)
        numeric_cols = [col for col in fused_data.select_dtypes(include=[np.number]).columns if col != "Date"]
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset. Please check your data.")
            return
        target_col = st.selectbox("Target Variable", options=numeric_cols, index=0, key="ann_target")
    with col4:
        eval_tail_size = st.slider("Evaluation Set Size", min_value=0.10, max_value=0.30, value=0.15, step=0.05, help="Fraction of data for final evaluation", key="ann_eval")

    oof_splits = st.slider("Out-of-Fold Splits", min_value=2, max_value=10, value=4, step=1, help="Number of time-series cross-validation splits", key="ann_oof")

    # Build Configuration Dictionary
    config = {
        "random_state": 42,
        "features": {
            "lags": [7, 14, 21],
            "rolls": [7, 14, 21],
        },
        "eval": {"tail_size": eval_tail_size},
        "oof_splits": oof_splits,
        "stage1": {
            "lookback": lookback,
            "units": lstm_units,
            "dropout": lstm_dropout,
            "epochs": lstm_epochs,
            "batch_size": lstm_batch_size,
            "lr": 0.001,
        },
        "stage2": {
            "layers": [layer1_units, layer2_units],
            "dropout": ann_dropout,
            "lr": ann_lr,
            "epochs": ann_epochs,
            "batch_size": ann_batch_size,
            "patience": ann_patience,
        },
    }

    # Training Button
    if st.button("🚀 Train LSTM-ANN Hybrid", type="primary", use_container_width=True):
        with st.spinner("🔄 Training LSTM-ANN hybrid model... This may take several minutes."):
            try:
                # Import the hybrid model
                from app_core.models.ml.lstm_ann import LSTMANNHybrid

                # Initialize and train
                hybrid_model = LSTMANNHybrid()
                artifacts = hybrid_model.fit(fused_data, target=target_col, config=config)

                # Store results in session state
                st.session_state["lstm_ann_results"] = {
                    "artifacts": artifacts,
                    "config": config,
                    "dataset_name": selected_dataset_name,
                }

                st.success("✅ **LSTM-ANN Hybrid trained successfully!**")

                # Display Metrics
                st.markdown("### 🏆 Hybrid Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.plotly_chart(
                        _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                        use_container_width=True,
                        key="lstm_ann_mae"
                    )
                with col2:
                    st.plotly_chart(
                        _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                        use_container_width=True,
                        key="lstm_ann_rmse"
                    )
                with col3:
                    st.plotly_chart(
                        _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                        use_container_width=True,
                        key="lstm_ann_mape"
                    )
                with col4:
                    accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
                    st.plotly_chart(
                        _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                        use_container_width=True,
                        key="lstm_ann_accuracy"
                    )

                # Show artifacts paths
                with st.expander("📁 Model Artifacts"):
                    st.markdown(f"**Stage 1 (LSTM):** `{artifacts.stage1_path}`")
                    st.markdown(f"**Stage 2 (ANN):** `{artifacts.stage2_path}`")
                    st.markdown(f"**Scaler:** `{artifacts.scaler_path}`")

            except ImportError as ie:
                st.error(f"❌ **Import Error**: {str(ie)}\n\nPlease ensure TensorFlow/Keras is installed.")
            except Exception as e:
                st.error(f"❌ **Training Failed**\n\n{str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Show existing results if available
    elif "lstm_ann_results" in st.session_state:
        st.info("ℹ️ **Previous LSTM-ANN results available.** Click 'Train LSTM-ANN Hybrid' to generate new results.")

        results = st.session_state["lstm_ann_results"]
        artifacts = results["artifacts"]

        st.markdown("### 🏆 Previous Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(
                _create_kpi_indicator("MAE", artifacts.metrics.get("MAE", 0), "", PRIMARY_COLOR),
                use_container_width=True,
                key="prev_lstm_ann_mae"
            )
        with col2:
            st.plotly_chart(
                _create_kpi_indicator("RMSE", artifacts.metrics.get("RMSE", 0), "", SECONDARY_COLOR),
                use_container_width=True,
                key="prev_lstm_ann_rmse"
            )
        with col3:
            st.plotly_chart(
                _create_kpi_indicator("MAPE", artifacts.metrics.get("MAPE", 0), "%", WARNING_COLOR),
                use_container_width=True,
                key="prev_lstm_ann_mape"
            )
        with col4:
            accuracy = artifacts.metrics.get("Accuracy", max(0, 100 - artifacts.metrics.get("MAPE", 100)))
            st.plotly_chart(
                _create_kpi_indicator("Accuracy", accuracy, "%", SUCCESS_COLOR),
                use_container_width=True,
                key="prev_lstm_ann_accuracy"
            )


def page_classification():
    """Classification page for predicting patient arrival reasons."""

    # Import sklearn modules (lazy import to avoid issues if not installed)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
    except ImportError:
        st.error("❌ **sklearn not installed.** Please install scikit-learn to use classification features.")
        return

    # Initialize session state for classification
    if "classification_data" not in st.session_state:
        st.session_state.classification_data = None
    if "classification_model" not in st.session_state:
        st.session_state.classification_model = None
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None
    if "feature_scaler" not in st.session_state:
        st.session_state.feature_scaler = None

    # Classification header
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
                    border-left: 4px solid #A855F7;
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1.5rem 0;
                    box-shadow: 0 0 20px rgba(168, 85, 247, 0.2);'>
            <h2 style='color: #A855F7; margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(168, 85, 247, 0.5);'>
                🎯 Patient Arrival Reason Classification
            </h2>
            <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.9;'>
                Predict and analyze reasons for patient arrivals using advanced machine learning classification models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs for classification
    class_tab1, class_tab2, class_tab3, class_tab4 = st.tabs([
        "📊 Data Upload",
        "🤖 Model Training",
        "📈 Results & Metrics",
        "🔮 Prediction"
    ])

    # ========== TAB 1: Data Upload ==========
    with class_tab1:
        st.markdown("### 📊 Data Upload")
        st.info("Upload dataset with patient arrival reasons for classification")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=["csv"],
                key="classification_upload_mh"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.classification_data = df

                    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

                    st.markdown("#### Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    st.markdown("#### Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.values,
                        'Non-Null': df.count().values,
                        'Null': df.isnull().sum().values
                    })
                    st.dataframe(col_info, use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

        with col2:
            if st.session_state.classification_data is not None:
                df = st.session_state.classification_data

                st.markdown("### Dataset Statistics")
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Features", len(df.columns))
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

                # Show unique values for categorical columns
                st.markdown("### Categorical Columns")
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols[:5]:  # Show first 5
                    unique_count = df[col].nunique()
                    st.metric(col, unique_count, delta="unique values")

    # ========== TAB 2: Model Training ==========
    with class_tab2:
        st.markdown("### 🤖 Model Training")

        if st.session_state.classification_data is None:
            st.warning("⚠️ Please upload data in the Data Upload tab first.")
        else:
            df = st.session_state.classification_data.copy()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Configuration")

                # Select target column
                target_col = st.selectbox(
                    "Select Target Column (Reason)",
                    options=df.columns.tolist(),
                    help="Column containing the reason categories"
                )

                # Show class distribution for selected target
                if target_col:
                    st.markdown("##### Class Distribution")
                    class_dist = df[target_col].value_counts()

                    # Highlight rare classes (< 2 samples)
                    rare_count = (class_dist < 2).sum()
                    if rare_count > 0:
                        st.error(f"⚠️ **{rare_count} class(es) with < 2 samples** - May cause training issues!")

                    # Display distribution
                    dist_df = pd.DataFrame({
                        'Class': class_dist.index,
                        'Count': class_dist.values,
                        'Percentage': (class_dist.values / len(df) * 100).round(2)
                    }).reset_index(drop=True)

                    # Color code rare classes
                    def highlight_rare(row):
                        if row['Count'] < 2:
                            return ['background-color: #FEE2E2'] * len(row)  # Light red
                        elif row['Count'] < 5:
                            return ['background-color: #FEF3C7'] * len(row)  # Light yellow
                        return [''] * len(row)

                    styled_df = dist_df.style.apply(highlight_rare, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=200)

                # Select feature columns
                available_features = [col for col in df.columns if col != target_col]

                # Initialize default selection in session state
                if "selected_features" not in st.session_state:
                    st.session_state.selected_features = available_features[:min(10, len(available_features))]

                # Add Select All button
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    feature_cols = st.multiselect(
                        "Select Feature Columns",
                        options=available_features,
                        default=st.session_state.selected_features,
                        help="Columns to use as predictors"
                    )
                with col_b:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    if st.button("✅ Select All", use_container_width=True):
                        st.session_state.selected_features = available_features
                        st.rerun()

                # Model selection
                model_type = st.selectbox(
                    "Select Classification Model",
                    options=[
                        "Random Forest",
                        "Gradient Boosting",
                        "Logistic Regression"
                    ]
                )

                # Train/test split
                test_size = st.slider(
                    "Test Set Size (%)",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5
                ) / 100

                random_state = st.number_input(
                    "Random State (for reproducibility)",
                    min_value=0,
                    max_value=1000,
                    value=42
                )

            with col2:
                st.markdown("#### Model Hyperparameters")

                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                    max_depth = st.slider("Max Depth", 3, 50, 10, 1)
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)

                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", 10, 500, 100, 10)
                    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
                    max_depth = st.slider("Max Depth", 3, 20, 3, 1)

                elif model_type == "Logistic Regression":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
                    max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

            st.markdown("---")

            # Train button
            if st.button("🚀 Train Classification Model", type="primary", use_container_width=True):
                if not feature_cols:
                    st.error("Please select at least one feature column!")
                else:
                    with st.spinner("Training model..."):
                        try:
                            # Prepare data
                            X = df[feature_cols].copy()
                            y = df[target_col].copy()

                            # Handle missing values
                            X = X.fillna(X.mean(numeric_only=True))
                            for col in X.select_dtypes(include=['object']).columns:
                                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")

                            # Encode categorical features
                            for col in X.select_dtypes(include=['object']).columns:
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))

                            # Encode target
                            label_encoder = LabelEncoder()
                            y_encoded = label_encoder.fit_transform(y)
                            st.session_state.label_encoder = label_encoder

                            # Check class distribution for stratification
                            class_counts = pd.Series(y_encoded).value_counts()
                            min_class_count = class_counts.min()
                            can_stratify = min_class_count >= 2

                            # Display warning if classes are too small
                            if not can_stratify:
                                rare_classes = class_counts[class_counts < 2].index
                                rare_class_names = [label_encoder.classes_[i] for i in rare_classes]
                                st.warning(
                                    f"⚠️ **Stratification disabled**: Some classes have only 1 sample ({', '.join(rare_class_names)}). "
                                    f"Consider collecting more data or removing rare classes for better model performance."
                                )

                            # Split data with smart stratification
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded,
                                test_size=test_size,
                                random_state=int(random_state),
                                stratify=y_encoded if can_stratify else None  # Only stratify if possible
                            )

                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            st.session_state.feature_scaler = scaler

                            # Train model
                            if model_type == "Random Forest":
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    random_state=int(random_state),
                                    n_jobs=-1
                                )
                            elif model_type == "Gradient Boosting":
                                model = GradientBoostingClassifier(
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=int(random_state)
                                )
                            else:  # Logistic Regression
                                model = LogisticRegression(
                                    C=C,
                                    max_iter=max_iter,
                                    random_state=int(random_state),
                                    multi_class='multinomial'
                                )

                            model.fit(X_train_scaled, y_train)

                            # Make predictions
                            y_train_pred = model.predict(X_train_scaled)
                            y_test_pred = model.predict(X_test_scaled)

                            # Calculate metrics
                            train_accuracy = accuracy_score(y_train, y_train_pred)
                            test_accuracy = accuracy_score(y_test, y_test_pred)

                            # Multi-class metrics (weighted average)
                            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                            # Confusion matrix
                            cm = confusion_matrix(y_test, y_test_pred)

                            # Classification report
                            report = classification_report(
                                y_test, y_test_pred,
                                target_names=label_encoder.classes_,
                                output_dict=True,
                                zero_division=0
                            )

                            # Feature importance
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                            else:
                                feature_importance = None

                            # Store results
                            st.session_state.classification_model = model
                            st.session_state.classification_results = {
                                'model_type': model_type,
                                'train_accuracy': train_accuracy,
                                'test_accuracy': test_accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'confusion_matrix': cm,
                                'classification_report': report,
                                'feature_importance': feature_importance,
                                'feature_cols': feature_cols,
                                'target_col': target_col,
                                'classes': label_encoder.classes_.tolist(),
                                'X_test': X_test,
                                'y_test': y_test,
                                'y_test_pred': y_test_pred,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            st.success(f"✅ Model trained successfully! Test Accuracy: {test_accuracy:.2%}")
                            st.balloons()

                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    # ========== TAB 3: Results & Metrics ==========
    with class_tab3:
        st.markdown("### 📈 Results & Metrics")

        if st.session_state.classification_results is None:
            st.warning("⚠️ Please train a model first in the Model Training tab.")
        else:
            results = st.session_state.classification_results

            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{results['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{results['recall']:.2%}")
            with col4:
                st.metric("F1 Score", f"{results['f1_score']:.2%}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Confusion Matrix
                st.markdown("#### Confusion Matrix")
                cm = results['confusion_matrix']
                classes = results['classes']

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=classes,
                    y=classes,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    hoverongaps=False
                ))

                fig_cm.update_layout(
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=TEXT_COLOR)
                )

                st.plotly_chart(fig_cm, use_container_width=True)

            with col2:
                # Classification Report
                st.markdown("#### Classification Report by Class")

                report_df = pd.DataFrame(results['classification_report']).T
                report_df = report_df[report_df.index.isin(classes)]
                report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
                report_df = report_df.round(3)

                st.dataframe(report_df, use_container_width=True)

                # Bar chart of F1 scores by class
                fig_f1 = px.bar(
                    x=report_df.index,
                    y=report_df['f1-score'],
                    labels={'x': 'Class', 'y': 'F1 Score'},
                    title='F1 Score by Class',
                    color=report_df['f1-score'],
                    color_continuous_scale='Viridis'
                )

                fig_f1.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=TEXT_COLOR),
                    showlegend=False
                )

                st.plotly_chart(fig_f1, use_container_width=True)

            # Feature Importance
            if results['feature_importance'] is not None:
                st.markdown("---")
                st.markdown("#### Feature Importance")

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_imp = px.bar(
                        results['feature_importance'].head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features',
                        color='Importance',
                        color_continuous_scale='Plasma'
                    )

                    fig_imp.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=TEXT_COLOR),
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig_imp, use_container_width=True)

                with col2:
                    st.markdown("**Top 10 Features**")
                    st.dataframe(
                        results['feature_importance'].head(10),
                        use_container_width=True,
                        hide_index=True
                    )

            # Model info
            st.markdown("---")
            st.markdown("#### Model Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info(f"**Model Type:** {results['model_type']}")
            with col2:
                st.info(f"**Number of Classes:** {len(results['classes'])}")
            with col3:
                st.info(f"**Trained:** {results['timestamp']}")

            # Download model
            st.markdown("---")
            if st.button("💾 Save Classification Model", use_container_width=True):
                try:
                    # Create artifacts directory
                    artifacts_dir = Path("pipeline_artifacts/classification")
                    artifacts_dir.mkdir(parents=True, exist_ok=True)

                    # Save model
                    model_path = artifacts_dir / f"model_{results['model_type'].replace(' ', '_')}.pkl"
                    joblib.dump(st.session_state.classification_model, model_path)

                    # Save label encoder
                    encoder_path = artifacts_dir / "label_encoder.pkl"
                    joblib.dump(st.session_state.label_encoder, encoder_path)

                    # Save scaler
                    scaler_path = artifacts_dir / "scaler.pkl"
                    joblib.dump(st.session_state.feature_scaler, scaler_path)

                    st.success(f"✅ Model saved to {artifacts_dir}")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")

    # ========== TAB 4: Prediction ==========
    with class_tab4:
        st.markdown("### 🔮 Prediction")

        if st.session_state.classification_model is None:
            st.warning("⚠️ Please train a model first in the Model Training tab.")
        else:
            results = st.session_state.classification_results
            model = st.session_state.classification_model

            prediction_mode = st.radio(
                "Prediction Mode",
                options=["Single Prediction", "Batch Prediction"],
                horizontal=True
            )

            if prediction_mode == "Single Prediction":
                st.markdown("#### Enter Feature Values")

                # Create input fields for each feature
                feature_values = {}
                cols = st.columns(3)

                for idx, feature in enumerate(results['feature_cols']):
                    with cols[idx % 3]:
                        feature_values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.2f",
                            key=f"class_feat_{feature}"
                        )

                if st.button("🎯 Predict Reason", type="primary", use_container_width=True):
                    try:
                        # Prepare input
                        input_df = pd.DataFrame([feature_values])
                        input_scaled = st.session_state.feature_scaler.transform(input_df)

                        # Make prediction
                        prediction = model.predict(input_scaled)[0]
                        prediction_proba = model.predict_proba(input_scaled)[0]

                        # Decode prediction
                        predicted_class = st.session_state.label_encoder.inverse_transform([prediction])[0]

                        # Display result
                        st.markdown("---")
                        st.success(f"### 🎯 Predicted Reason: **{predicted_class}**")

                        # Probability distribution
                        st.markdown("#### Prediction Probabilities")
                        proba_df = pd.DataFrame({
                            'Reason': results['classes'],
                            'Probability': prediction_proba
                        }).sort_values('Probability', ascending=False)

                        fig_proba = px.bar(
                            proba_df,
                            x='Reason',
                            y='Probability',
                            title='Prediction Confidence by Class',
                            color='Probability',
                            color_continuous_scale='Turbo'
                        )

                        fig_proba.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=TEXT_COLOR)
                        )

                        st.plotly_chart(fig_proba, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

            else:  # Batch Prediction
                st.markdown("#### Upload Data for Batch Prediction")

                batch_file = st.file_uploader(
                    "Choose CSV file with same features",
                    type=["csv"],
                    key="batch_prediction_class"
                )

                if batch_file is not None:
                    try:
                        batch_df = pd.read_csv(batch_file)

                        # Validate features
                        missing_features = set(results['feature_cols']) - set(batch_df.columns)
                        if missing_features:
                            st.error(f"Missing features: {missing_features}")
                        else:
                            X_batch = batch_df[results['feature_cols']].copy()

                            # Handle missing values
                            X_batch = X_batch.fillna(X_batch.mean(numeric_only=True))
                            for col in X_batch.select_dtypes(include=['object']).columns:
                                X_batch[col] = X_batch[col].fillna(X_batch[col].mode()[0] if not X_batch[col].mode().empty else "Unknown")

                            # Encode categorical features
                            for col in X_batch.select_dtypes(include=['object']).columns:
                                le = LabelEncoder()
                                X_batch[col] = le.fit_transform(X_batch[col].astype(str))

                            # Scale and predict
                            X_batch_scaled = st.session_state.feature_scaler.transform(X_batch)
                            predictions = model.predict(X_batch_scaled)
                            predictions_proba = model.predict_proba(X_batch_scaled)

                            # Decode predictions
                            predicted_classes = st.session_state.label_encoder.inverse_transform(predictions)

                            # Add predictions to dataframe
                            batch_df['Predicted_Reason'] = predicted_classes
                            batch_df['Prediction_Confidence'] = predictions_proba.max(axis=1)

                            st.success(f"✅ Predictions made for {len(batch_df)} records!")

                            # Display results
                            st.dataframe(batch_df, use_container_width=True)

                            # Distribution of predictions
                            st.markdown("#### Prediction Distribution")
                            pred_dist = pd.DataFrame(predicted_classes, columns=['Predicted_Reason'])
                            fig_dist = px.pie(
                                pred_dist,
                                names='Predicted_Reason',
                                title='Distribution of Predicted Reasons'
                            )

                            fig_dist.update_layout(
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=TEXT_COLOR)
                            )

                            st.plotly_chart(fig_dist, use_container_width=True)

                            # Download predictions
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Error processing batch predictions: {str(e)}")


# -----------------------------------------------------------------------------
# Tab Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📏 Benchmarks",
    "🧮 Machine Learning",
    "🔬 Hyperparameter Tuning",
    "🧬 Hybrid Models",
    "🎯 Classification"
])

with tab1:
    page_benchmarks()

with tab2:
    page_ml()

with tab3:
    page_hyperparameter_tuning()

with tab4:
    page_hybrid()

with tab5:
    page_classification()
