# =============================================================================
# 08_Train_Models.py - Modelling Studio
# Unified Modelling Studio (foundation only)
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
import gc
from pathlib import Path

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand, render_cache_management
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.components import render_scifi_hero_header
from app_core.cache import save_to_cache, get_cache_manager
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available
from app_core.plots import build_multihorizon_results_dashboard
from app_core.ui.model_upload_ui import render_cloud_sync_tab

# =============================================================================
# OPTUNA IMPORT CHECK (For Bayesian Optimization)
# =============================================================================
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_contour,
    )
    OPTUNA_AVAILABLE = True
    OPTUNA_VIS_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OPTUNA_VIS_AVAILABLE = False

# =============================================================================
# TIME SERIES CROSS-VALIDATION IMPORT
# =============================================================================
try:
    from app_core.pipelines import (
        CVFoldResult,
        CVConfig,
        create_cv_folds,
        evaluate_cv_metrics,
        format_cv_metric,
    )
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Modelling Studio - HealthForecast AI",
    page_icon="🧠",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()
render_cache_management()

# Results Storage Panel (Supabase persistence)
render_results_storage_panel(page_key="Modelling Studio")

# Auto-load saved results if available
auto_load_if_available("Modelling Studio")

# Fluorescent effects + Custom Tab Styling
st.markdown(f"""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR MODELLING STUDIO
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
# Premium Hero Header
# -----------------------------------------------------------------------------
render_scifi_hero_header(
    title="Modelling Studio",
    subtitle="Configure and train machine learning models. XGBoost, LSTM, ANN, and hybrid architectures.",
    status="SYSTEM ONLINE"
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
    "lstm_layers": 2, "lstm_hidden_units": 64, "lstm_epochs": 15, "lstm_lr": 1e-3,
    "lookback_window": 14, "dropout": 0.2, "batch_size": 32,
    "lstm_search_method": "Grid (bounded)",
    "lstm_max_layers": 3, "lstm_max_units": 256, "lstm_max_epochs": 50,
    "lstm_lr_min": 1e-5, "lstm_lr_max": 1e-2,
    # ANN
    "ml_ann_param_mode": "Automatic",
    "ann_hidden_layers": 2, "ann_neurons": 64, "ann_epochs": 10, "ann_activation": "relu",
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

# =============================================================================
# CROSS-VALIDATION HELPERS
# =============================================================================

def render_cv_status_banner():
    """Render a banner showing cross-validation status from Feature Studio configuration."""
    cv_enabled = st.session_state.get("cv_enabled", False)
    cv_config = st.session_state.get("cv_config")
    cv_folds = st.session_state.get("cv_folds")

    if cv_enabled and cv_folds:
        n_folds = len(cv_folds)
        gap = cv_config.gap if cv_config else 0
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
                        border-left: 4px solid rgba(34, 197, 94, 0.8);
                        padding: 1rem 1.5rem;
                        border-radius: 12px;
                        margin: 1rem 0;
                        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
                        display: flex;
                        align-items: center;
                        gap: 1rem;'>
                <span style='font-size: 1.5rem;'>🔄</span>
                <div>
                    <div style='color: rgba(34, 197, 94, 1); font-weight: 700; font-size: 1rem;
                                text-shadow: 0 0 10px rgba(34, 197, 94, 0.5);'>
                        Cross-Validation Enabled
                    </div>
                    <div style='color: {SUBTLE_TEXT}; font-size: 0.875rem; margin-top: 0.25rem;'>
                        {n_folds}-fold time series CV • Gap: {gap} days • Metrics will show mean ± std
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    else:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(107, 114, 128, 0.15), rgba(75, 85, 99, 0.1));
                        border-left: 4px solid rgba(107, 114, 128, 0.5);
                        padding: 0.75rem 1.5rem;
                        border-radius: 12px;
                        margin: 1rem 0;
                        display: flex;
                        align-items: center;
                        gap: 1rem;'>
                <span style='font-size: 1.2rem;'>📊</span>
                <div style='color: {SUBTLE_TEXT}; font-size: 0.875rem;'>
                    Cross-Validation: <strong>Disabled</strong> • Configure in Feature Studio → Cross-Validation tab
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return False


def _create_cv_kpi_indicator(title: str, mean_val: float, std_val: float = None,
                              suffix: str = "", color: str = PRIMARY_COLOR):
    """
    Create a KPI indicator card showing mean ± std for CV metrics.

    Parameters
    ----------
    title : str
        The metric name (e.g., "MAE", "RMSE")
    mean_val : float
        Mean value across CV folds
    std_val : float, optional
        Standard deviation across CV folds. If None, shows just the mean.
    suffix : str
        Suffix to append (e.g., "%")
    color : str
        Color for the indicator
    """
    try:
        is_num = isinstance(mean_val, (int, float, np.floating)) and np.isfinite(mean_val)
    except Exception:
        is_num = False

    display_val = float(mean_val) if is_num else 0.0

    # Map colors to fluorescent glow colors
    glow_colors = {
        PRIMARY_COLOR: "rgba(59, 130, 246, 0.6)",
        SECONDARY_COLOR: "rgba(34, 211, 238, 0.6)",
        SUCCESS_COLOR: "rgba(34, 197, 94, 0.6)",
        WARNING_COLOR: "rgba(245, 158, 11, 0.6)",
    }
    glow_color = glow_colors.get(color, "rgba(59, 130, 246, 0.6)")
    number_color = color if is_num else "rgba(0,0,0,0)"

    # Format the display text
    if is_num and std_val is not None and np.isfinite(std_val):
        # Show mean ± std format
        value_text = f"{display_val:.2f} ± {std_val:.2f}{suffix}"
    elif is_num:
        value_text = f"{display_val:.2f}{suffix}"
    else:
        value_text = "—"

    fig = go.Figure()

    # Create a simple indicator with custom annotation for ± format
    fig.add_annotation(
        x=0.5, y=0.35,
        text=value_text,
        showarrow=False,
        font=dict(size=22 if std_val is None else 18, color=number_color, family="Arial Black"),
        xanchor="center",
        yanchor="middle"
    )

    fig.add_annotation(
        x=0.5, y=0.8,
        text=f"<b style='letter-spacing:0.5px'>{title}</b>",
        showarrow=False,
        font=dict(size=12, color=TEXT_COLOR),
        xanchor="center",
        yanchor="middle"
    )

    fig.update_layout(
        margin=dict(l=5, r=5, t=10, b=5),
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        height=110,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
    )

    return fig


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
            "valueformat": ".2f",  # Prevent SI suffix auto-formatting (K, M, B)
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
# SEASONAL-PROPORTION-BASED CATEGORY FORECASTING
# -----------------------------------------------------------------------------
def _render_seasonal_category_forecast(predictions: list, dates: list, title: str):
    """
    Calculates and renders the clinical category breakdown using the seasonal proportions method.
    This is the same advanced statistical distribution method used in the Baseline Models page.
    """
    if not st.session_state.get("enable_seasonal_proportions", False):
        return

    # Lazy import for performance, assuming these components exist from the other page.
    from app_core.analytics.seasonal_proportions import (
        SeasonalProportionConfig,
        calculate_seasonal_proportions,
        render_seasonal_forecast_breakdown,
    )

    config = st.session_state.get("seasonal_proportions_config", SeasonalProportionConfig())
    # Use 'training_data' which is the selected dataset in this page
    data = st.session_state.get("cfg", {}).get("training_data")

    if data is None or data.empty:
        return  # Silently skip if no dataset selected

    try:
        # This function should return the proportion data needed for rendering
        proportions_data = calculate_seasonal_proportions(data.copy(), config)

        if not proportions_data:
            st.info("Could not calculate seasonal proportions (insufficient data or no categories found).")
            return

        # This function should handle all the UI rendering for the breakdown
        render_seasonal_forecast_breakdown(
            predictions=predictions,
            forecast_dates=pd.to_datetime(dates),
            seasonal_proportions=proportions_data,
            title=title,
        )

    except Exception as e:
        st.error(f"Failed to render seasonal category forecast: {e}")
        import traceback
        st.code(traceback.format_exc())



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
    pipelines = {}  # Store trained pipelines for SHAP/LIME
    explainability_data = {}  # Store X_train, X_val, feature_names for SHAP/LIME

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
                "params": result.get("params", {}),
            }

            # Store pipeline and explainability data for SHAP/LIME
            if result.get("pipeline") is not None:
                pipelines[h] = result["pipeline"]
                explainability_data[h] = {
                    "X_train": result.get("X_train"),
                    "X_val": result.get("X_val"),
                    "feature_names": result.get("feature_names", []),
                }

            successful.append(h)

            # Clean up TensorFlow/Keras memory after each horizon (CRITICAL for LSTM/ANN)
            if model_type in ["LSTM", "ANN"]:
                try:
                    import keras.backend as K
                    K.clear_session()
                except Exception:
                    pass
                gc.collect()

        except Exception as e:
            # Skip failed horizons and store error
            error_msg = str(e)
            errors[h] = error_msg
            print(f"❌ Horizon {h} failed: {error_msg}")
            import traceback
            traceback.print_exc()  # Print full traceback to console

            # Clean up even on failure
            if model_type in ["LSTM", "ANN"]:
                try:
                    import keras.backend as K
                    K.clear_session()
                except Exception:
                    pass
                gc.collect()

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

    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": successful,
        "errors": errors,
        # SHAP/LIME explainability data
        "pipelines": pipelines,
        "explainability_data": explainability_data,
    }


# =============================================================================
# MODEL SAVING FOR CLOUD SYNC
# =============================================================================

def save_ml_models_to_disk(
    model_type: str,
    ml_results: dict,
    artifact_dir: str = "pipeline_artifacts"
) -> dict:
    """
    Save trained ML models to disk for Cloud Sync (Train Locally, Deploy to Cloud).

    Args:
        model_type: Model name ("XGBoost", "LSTM", "ANN")
        ml_results: Results from run_ml_multihorizon()
        artifact_dir: Base directory for saving artifacts

    Returns:
        dict: Mapping of horizon to saved file paths
    """
    import os

    saved_paths = {}
    pipelines = ml_results.get("pipelines", {})

    if not pipelines:
        return saved_paths

    # Create model-specific directory
    model_dir = os.path.join(artifact_dir, model_type.lower())
    os.makedirs(model_dir, exist_ok=True)

    for h, pipeline in pipelines.items():
        try:
            if pipeline is None:
                continue

            # Determine file extension based on model type
            if model_type.upper() in ["LSTM", "ANN"]:
                # Keras models use .keras or .h5
                filepath = os.path.join(model_dir, f"horizon_{h}.keras")
                if hasattr(pipeline, 'model') and pipeline.model is not None:
                    pipeline.model.save(filepath)
                    saved_paths[h] = filepath
            else:
                # XGBoost and other sklearn-compatible models use joblib
                filepath = os.path.join(model_dir, f"horizon_{h}.pkl")
                pipeline.save(filepath)
                saved_paths[h] = filepath

        except Exception as e:
            print(f"Warning: Could not save {model_type} horizon {h}: {e}")
            continue

    return saved_paths


def save_baseline_model_to_disk(
    model_type: str,
    results: dict,
    artifact_dir: str = "pipeline_artifacts"
) -> dict:
    """
    Save ARIMA/SARIMAX model results to disk.

    Note: ARIMA/SARIMAX models from statsmodels can be large and
    sometimes don't pickle well. We save what we can.

    Args:
        model_type: "ARIMA" or "SARIMAX"
        results: Results from the baseline model training
        artifact_dir: Base directory for saving artifacts

    Returns:
        dict: Mapping of horizon to saved file paths
    """
    import os

    saved_paths = {}
    per_h = results.get("per_h", {})

    if not per_h:
        return saved_paths

    # Create model-specific directory
    model_dir = os.path.join(artifact_dir, model_type.lower())
    os.makedirs(model_dir, exist_ok=True)

    for h, h_data in per_h.items():
        try:
            res = h_data.get("res")  # The fitted model object
            if res is None:
                continue

            filepath = os.path.join(model_dir, f"horizon_{h}.pkl")

            # Try to save the model
            joblib.dump({
                "model": res,
                "order": h_data.get("order"),
                "sorder": h_data.get("sorder"),
                "horizon": h,
            }, filepath)

            saved_paths[h] = filepath

        except Exception as e:
            print(f"Warning: Could not save {model_type} horizon {h}: {e}")
            continue

    return saved_paths


def save_optimized_model_to_disk(
    model_name: str,
    opt_results: dict,
    artifact_dir: str = "pipeline_artifacts/optimized"
) -> dict:
    """
    Save Optuna/Grid/Random search optimized results to disk.

    This saves:
    1. best_params as JSON (always saved if available)
    2. Trained pipelines if available

    Args:
        model_name: Model name (e.g., "XGBoost_Optimized")
        opt_results: Results from optimization (must have best_params)
        artifact_dir: Base directory for saving artifacts

    Returns:
        dict: Mapping of saved file types to paths
    """
    import os
    import json

    saved_paths = {}

    # Create optimized model directory
    model_dir = os.path.join(artifact_dir, model_name.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    # 1. Always save best_params as JSON
    best_params = opt_results.get("best_params", {})
    if best_params:
        params_path = os.path.join(model_dir, "best_params.json")
        try:
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=2, default=str)
            saved_paths["best_params"] = params_path
            logger.info(f"Saved best_params to {params_path}")
        except Exception as e:
            logger.warning(f"Could not save best_params: {e}")

    # 2. Save optimization metrics summary
    metrics_to_save = {
        "best_cv_score": opt_results.get("best_cv_score"),
        "best_rmse": opt_results.get("best_rmse"),
        "best_mae": opt_results.get("best_mae"),
        "best_r2": opt_results.get("best_r2"),
        "search_method": opt_results.get("search_method", "unknown"),
    }
    metrics_path = os.path.join(model_dir, "optimization_metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        saved_paths["metrics"] = metrics_path
    except Exception as e:
        logger.warning(f"Could not save optimization metrics: {e}")

    # 3. Save trained pipelines if available
    pipelines = opt_results.get("pipelines", {})

    if not pipelines:
        # Try to get from per_h structure
        per_h = opt_results.get("per_h", {})
        for h, h_data in per_h.items():
            if "pipeline" in h_data:
                pipelines[h] = h_data["pipeline"]

    for h, pipeline in pipelines.items():
        try:
            if pipeline is None:
                continue

            filepath = os.path.join(model_dir, f"horizon_{h}.pkl")

            if hasattr(pipeline, 'save'):
                pipeline.save(filepath)
            else:
                joblib.dump(pipeline, filepath)

            saved_paths[f"horizon_{h}"] = filepath

        except Exception as e:
            logger.warning(f"Could not save optimized {model_name} horizon {h}: {e}")
            continue

    return saved_paths


# =============================================================================
# CLINICAL CATEGORY DETECTION & RESIDUAL CAPTURE
# =============================================================================
CLINICAL_CATEGORIES = [
    "RESPIRATORY", "CARDIAC", "TRAUMA", "GASTROINTESTINAL",
    "INFECTIOUS", "NEUROLOGICAL", "OTHER"
]


def detect_category_targets(df: pd.DataFrame, horizons: int = 7) -> dict:
    """
    Detect which clinical category target columns exist in the dataframe.

    Args:
        df: DataFrame with potential category columns (e.g., RESPIRATORY_1, CARDIAC_2)
        horizons: Number of horizons to check (1-7)

    Returns:
        Dict mapping category name to list of existing target columns.
        Example: {"RESPIRATORY": ["RESPIRATORY_1", "RESPIRATORY_2", ...]}
    """
    category_targets = {}

    for category in CLINICAL_CATEGORIES:
        cols = []
        for h in range(1, horizons + 1):
            col_name = f"{category}_{h}"
            if col_name in df.columns:
                cols.append(col_name)
        if cols:
            category_targets[category] = cols

    return category_targets


def capture_residuals_from_results(results: dict, df: pd.DataFrame, feature_cols: list,
                                   split_ratio: float) -> dict:
    """
    Extract and structure residuals from ML training results for hybrid models.

    Args:
        results: Results from run_ml_multihorizon()
        df: Original training dataframe
        feature_cols: Feature column names
        split_ratio: Train/validation split ratio

    Returns:
        Dict with train and validation residuals per horizon.
    """
    per_h = results.get("per_h", {})
    successful = results.get("successful", [])

    if not per_h or not successful:
        return {}

    # Get split index
    fe = st.session_state.get("feature_engineering", {})
    train_idx = fe.get("train_idx")
    test_idx = fe.get("test_idx")

    use_fe_split = (
        train_idx is not None and test_idx is not None and
        isinstance(train_idx, np.ndarray) and isinstance(test_idx, np.ndarray) and
        len(train_idx) > 0 and len(test_idx) > 0
    )

    residuals = {
        "train_residuals": {},
        "val_residuals": {},
        "train_predictions": {},
        "val_predictions": {},
        "feature_data": {},
    }

    for h in successful:
        h_data = per_h.get(h, {})
        y_test = h_data.get("y_test")
        forecast = h_data.get("forecast")

        if y_test is not None and forecast is not None:
            # Validation residuals (actual - predicted)
            val_resid = np.array(y_test) - np.array(forecast)
            residuals["val_residuals"][f"Target_{h}"] = val_resid
            residuals["val_predictions"][f"Target_{h}"] = np.array(forecast)

            # Get training predictions if pipeline is available
            # Note: Training residuals would require re-predicting on train set
            # For now, we'll compute them during hybrid training if needed

    # Store feature data for Stage 2 training
    # IMPORTANT: Filter out datetime columns - XGBoost can't handle Timestamps
    numeric_feature_cols = [
        col for col in feature_cols
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col])
        and col.lower() not in ['date', 'datetime', 'timestamp', 'time', 'ds']
    ]

    if use_fe_split:
        X = df[numeric_feature_cols].values.astype(np.float64)
        residuals["feature_data"]["X_train"] = X[train_idx]
        residuals["feature_data"]["X_val"] = X[test_idx]
        residuals["feature_data"]["train_idx"] = train_idx
        residuals["feature_data"]["test_idx"] = test_idx
    else:
        split_idx = int(len(df) * split_ratio)
        X = df[numeric_feature_cols].values.astype(np.float64)
        residuals["feature_data"]["X_train"] = X[:split_idx]
        residuals["feature_data"]["X_val"] = X[split_idx:]

    residuals["feature_data"]["feature_cols"] = numeric_feature_cols
    residuals["trained_at"] = datetime.now().isoformat()

    return residuals


def run_ml_with_categories(
    model_type: str,
    config: dict,
    df: pd.DataFrame,
    feature_cols: list,
    split_ratio: float,
    horizons: int = 7,
) -> dict:
    """
    Train ML model for total patient count AND clinical categories.

    This function:
    1. Trains on Target_1...Target_7 (total patients)
    2. Trains on CATEGORY_1...CATEGORY_7 for each detected category
    3. Captures residuals for hybrid model use
    4. Stores all results in a structured format

    Args:
        model_type: "LSTM", "ANN", or "XGBoost"
        config: Model configuration
        df: Training dataframe
        feature_cols: Feature column names
        split_ratio: Train/validation split
        horizons: Number of horizons (default 7)

    Returns:
        dict with total_results, category_results, and residuals
    """
    from datetime import datetime

    # 1. Train on total patient count (Target_1...Target_7)
    total_results = run_ml_multihorizon(
        model_type=model_type,
        config=config,
        df=df,
        feature_cols=feature_cols,
        split_ratio=split_ratio,
        horizons=horizons,
    )

    # 2. Capture residuals from total model
    total_residuals = capture_residuals_from_results(
        total_results, df, feature_cols, split_ratio
    )

    # 3. Detect and train on category targets
    category_targets = detect_category_targets(df, horizons)
    category_results = {}
    category_residuals = {}

    for category, cols in category_targets.items():
        # Train model for this category's horizons
        cat_per_h = {}
        cat_successful = []

        for h in range(1, horizons + 1):
            target_col = f"{category}_{h}"
            if target_col not in df.columns:
                continue

            try:
                result = run_ml_model(
                    model_type=model_type,
                    config=config,
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    split_ratio=split_ratio,
                )

                if result.get("success", False):
                    predictions = result.get("predictions")
                    cat_per_h[h] = {
                        "y_test": predictions["actual"].values if predictions is not None else None,
                        "forecast": predictions["predicted"].values if predictions is not None else None,
                        "datetime": predictions["datetime"].values if predictions is not None else None,
                        "mae": result.get("val_metrics", {}).get("MAE"),
                        "rmse": result.get("val_metrics", {}).get("RMSE"),
                    }
                    cat_successful.append(h)
            except Exception as e:
                print(f"Category {category} horizon {h} failed: {e}")
                continue

        if cat_successful:
            category_results[category] = {
                "per_h": cat_per_h,
                "successful": cat_successful,
            }

            # Capture category residuals
            cat_resid = {}
            for h in cat_successful:
                h_data = cat_per_h.get(h, {})
                y_test = h_data.get("y_test")
                forecast = h_data.get("forecast")
                if y_test is not None and forecast is not None:
                    cat_resid[f"{category}_{h}"] = np.array(y_test) - np.array(forecast)
            category_residuals[category] = cat_resid

    return {
        "model_type": model_type,
        "total_results": total_results,
        "category_results": category_results,
        "total_residuals": total_residuals,
        "category_residuals": category_residuals,
        "trained_at": datetime.now().isoformat(),
        "categories_detected": list(category_targets.keys()),
    }


def store_stage1_residuals(model_name: str, residuals: dict):
    """Store Stage 1 residuals in session state for hybrid model use."""
    key = f"stage1_residuals_{model_name.lower()}"
    st.session_state[key] = residuals


def get_stage1_residuals(model_name: str) -> dict:
    """Retrieve Stage 1 residuals from session state."""
    key = f"stage1_residuals_{model_name.lower()}"
    return st.session_state.get(key, {})


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

    Args:
        model_type: Model name ("LSTM", "ANN", or "XGBoost")
        config: Model configuration dictionary
        df: Training dataframe
        target_col: Target column name
        feature_cols: List of feature column names
        split_ratio: Train/validation split ratio

    Returns:
        dict: Results containing metrics, predictions, and metadata.
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
        # Check for Feature Engineering split indices (80/20 split from Feature Studio)
        # =====================================================================
        fe = st.session_state.get("feature_engineering", {})
        train_idx = fe.get("train_idx")
        test_idx = fe.get("test_idx")

        # Check if Feature Studio split is available and valid for this dataframe
        use_fe_split = (
            train_idx is not None and
            test_idx is not None and
            isinstance(train_idx, np.ndarray) and
            isinstance(test_idx, np.ndarray) and
            len(train_idx) > 0 and
            len(test_idx) > 0 and
            max(train_idx.max() if len(train_idx) > 0 else 0,
                test_idx.max() if len(test_idx) > 0 else 0) < len(df)
        )

        if use_fe_split:
            # Use Feature Studio split (80/20 by default)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[test_idx], y[test_idx]

            # Extract datetime values
            if datetime_col:
                val_datetime = df[datetime_col].iloc[test_idx].values
            else:
                val_datetime = test_idx.tolist()

        else:
            # Fallback to slider-based split (if Feature Studio not run)
            split_idx = int(len(df) * split_ratio)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

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
            # NOTE: Default epochs reduced from 100 to 30 for faster training on cloud
            model_config = {
                "lookback_window": config.get("lookback_window", 14),
                "lstm_hidden_units": config.get("lstm_hidden_units", 64),
                "lstm_layers": config.get("lstm_layers", 2),
                "dropout": config.get("dropout", 0.2),
                "lstm_epochs": config.get("lstm_epochs", 15),  # Reduced for speed + early stopping
                "lstm_lr": config.get("lstm_lr", 0.001),
                "batch_size": config.get("batch_size", 32),
            }
            pipeline = LSTMPipeline(model_config)

        elif model_type == "ANN":
            from app_core.models.ml.ann_pipeline import ANNPipeline

            # Extract ANN params from config
            # NOTE: Default epochs reduced from 30 to 20 for faster training on cloud
            model_config = {
                "ann_hidden_layers": config.get("ann_hidden_layers", 2),
                "ann_neurons": config.get("ann_neurons", 64),
                "ann_activation": config.get("ann_activation", "relu"),
                "dropout": config.get("dropout", 0.2),
                "ann_epochs": config.get("ann_epochs", 10),  # Reduced for speed + early stopping
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
            # Store for SHAP/LIME explainability
            "pipeline": pipeline,
            "X_train": X_train,
            "X_val": X_val,
            "feature_names": numeric_feature_cols,
        }

        return result

    except MemoryError as e:
        gc.collect()  # Try to free memory
        return {
            "success": False,
            "error": f"Out of memory: {e}. Try reducing epochs, batch size, or data size."
        }
    except ValueError as e:
        return {
            "success": False,
            "error": f"Data shape/value error: {e}. Check feature consistency across horizons."
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ML Training Error:\n{error_traceback}")  # Log to console
        return {
            "success": False,
            "error": str(e),
            "traceback": error_traceback
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
    from app_core.analytics.seasonal_proportions import SeasonalProportionConfig

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

    # Cross-Validation Status Banner
    cv_is_enabled = render_cv_status_banner()

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
        st.info("🚀 **All Models Mode**: XGBoost, LSTM, and ANN will be trained sequentially using optimized defaults (XGBoost: 200 trees, LSTM: 30 epochs, ANN: 20 epochs with early stopping).")
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

    # 3. Seasonal Proportions Configuration (NEW)
    st.markdown("---")
    st.markdown("#### 📊 Seasonal Category Distribution")
    st.caption("Distribute total patient forecast into clinical categories using seasonal patterns")

    enable_seasonal = st.checkbox(
        "Enable Seasonal Proportions",
        value=True,
        key="enable_seasonal_proportions",
        help="Apply day-of-week and monthly patterns to distribute total forecasts across clinical categories"
    )
    # Note: Widget key already manages session state - no manual assignment needed

    if enable_seasonal:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            use_dow = st.checkbox("Day-of-Week Seasonality", value=True, key="seasonal_use_dow")
            use_monthly = st.checkbox("Monthly Seasonality", value=True, key="seasonal_use_monthly")
        with col_s2:
            stl_period = st.selectbox("STL Period", [7, 30, 365], index=0, key="seasonal_stl_period")
            stl_model = st.selectbox("STL Model", ["additive", "multiplicative"], index=0, key="seasonal_stl_model")

        st.session_state["seasonal_proportions_config"] = SeasonalProportionConfig(
            use_dow_seasonality=use_dow,
            use_monthly_seasonality=use_monthly,
            stl_period=stl_period,
            stl_model=stl_model
        )
        st.success("✅ Seasonal proportions will be applied to forecasts")
    else:
        st.info("ℹ️ Seasonal proportions disabled.")
    st.markdown("---")


    # 4. Manual Configuration - Only show for Single Model mode
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

    # 5. Train/test split ratio - Full width with better styling
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

    # 6. Run Model Button
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
                # Clear previous "All Models" results to avoid confusion
                if "ml_all_models_trained" in st.session_state:
                    del st.session_state["ml_all_models_trained"]
                # Clear old individual model results from "All Models" mode
                for old_model in ["XGBoost", "LSTM", "ANN"]:
                    old_key = f"ml_mh_results_{old_model}"
                    if old_key in st.session_state:
                        del st.session_state[old_key]

                # Pass a copy of the config to prevent any potential state pollution between runs
                current_run_config = cfg.copy()
                model_name = current_run_config['ml_choice']
                with st.spinner(f"Training {model_name} for {current_run_config.get('ml_horizons', 7)} horizons (including clinical categories)..."):
                    try:
                        # Use new function that trains on total + categories and captures residuals
                        full_results = run_ml_with_categories(
                            model_type=model_name,
                            config=current_run_config,
                            df=selected_df,
                            feature_cols=current_run_config['ml_feature_cols'],
                            split_ratio=current_run_config.get('split_ratio', 0.8),
                            horizons=current_run_config.get('ml_horizons', 7),
                        )

                        # Extract total results for backward compatibility
                        results = full_results.get("total_results", {})

                        # Store total results in session state (backward compatible)
                        st.session_state["ml_mh_results"] = results

                        # Also save to individual model key for hybrid model detection
                        model_key = f"ml_mh_results_{model_name.lower()}"
                        st.session_state[model_key] = results

                        # ===== SAVE MODELS TO DISK FOR CLOUD SYNC =====
                        try:
                            saved_paths = save_ml_models_to_disk(model_name, results)
                            if saved_paths:
                                st.success(f"💾 Saved {len(saved_paths)} model file(s) to `pipeline_artifacts/{model_name.lower()}/`")
                        except Exception as save_err:
                            st.warning(f"⚠️ Could not save model files: {save_err}")
                        # ================================================

                        # Store residuals for hybrid models (Stage 1 → Stage 2)
                        total_residuals = full_results.get("total_residuals", {})
                        if total_residuals:
                            store_stage1_residuals(model_name, total_residuals)
                            st.success(f"✅ Residuals captured for {model_name} (ready for hybrid models)")

                        # Store category results for Patient Forecast page
                        category_results = full_results.get("category_results", {})
                        if category_results:
                            st.session_state["ml_category_results"] = {
                                "model_name": model_name,
                                "categories": category_results,
                                "categories_detected": full_results.get("categories_detected", []),
                                "trained_at": full_results.get("trained_at"),
                            }
                            detected = full_results.get("categories_detected", [])
                            if detected:
                                st.info(f"📊 Category predictions captured: {', '.join(detected)}")

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

                # Add seasonal category forecast section
                ml_results = st.session_state["ml_mh_results"]
                per_h = ml_results.get("per_h", {})
                successful = ml_results.get("successful", [])

                if per_h and successful:
                    # Construct a proper 7-day forecast from the multi-horizon results
                    future_forecast_values = []
                    future_forecast_dates = []
                    for h in range(1, 8):  # We want a 7-day forecast
                        if h in per_h:
                            h_data = per_h.get(h, {})
                            forecast_series = h_data.get("forecast")
                            datetime_series = h_data.get("datetime")
                            if forecast_series is not None and len(forecast_series) > 0:
                                future_forecast_values.append(forecast_series[0])
                                future_forecast_dates.append(datetime_series[0])
                        else:
                            future_forecast_values.append(np.nan)
                            if future_forecast_dates:
                                future_forecast_dates.append(future_forecast_dates[-1] + pd.Timedelta(days=1))
                            else:
                                future_forecast_dates.append(pd.NaT)

                    if future_forecast_values:
                        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                        _render_seasonal_category_forecast(
                            predictions=future_forecast_values,
                            dates=future_forecast_dates,
                            title=f"{cfg['ml_choice']} - Clinical Category Forecast (Seasonal Distribution)"
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
                # Clear previous single model results to avoid confusion
                if "ml_mh_results" in st.session_state:
                    del st.session_state["ml_mh_results"]
                # Clear old "All Models" flag and results
                if "ml_all_models_trained" in st.session_state:
                    del st.session_state["ml_all_models_trained"]
                for old_model in ["XGBoost", "LSTM", "ANN"]:
                    old_key = f"ml_mh_results_{old_model}"
                    if old_key in st.session_state:
                        del st.session_state[old_key]

                # CRITICAL: Clear TensorFlow session and free memory before training
                try:
                    import keras.backend as K
                    K.clear_session()
                except Exception:
                    pass
                gc.collect()

                all_models = ["XGBoost", "LSTM", "ANN"]
                all_results = {}

                # Pre-populate config with optimized defaults for cloud training
                # These are used since manual config UI is not shown in "All Models" mode
                cfg.setdefault("xgb_n_estimators", 200)  # Fast XGBoost
                cfg.setdefault("xgb_max_depth", 5)
                cfg.setdefault("xgb_eta", 0.1)
                cfg.setdefault("xgb_min_child_weight", 1)
                cfg.setdefault("lstm_epochs", 15)  # Reduced for speed + early stopping
                cfg.setdefault("lstm_hidden_units", 64)
                cfg.setdefault("lstm_layers", 2)
                cfg.setdefault("lookback_window", 14)
                cfg.setdefault("lstm_lr", 0.001)
                cfg.setdefault("batch_size", 32)
                cfg.setdefault("ann_epochs", 10)  # Reduced for speed + early stopping
                cfg.setdefault("ann_hidden_layers", 2)
                cfg.setdefault("ann_neurons", 64)
                cfg.setdefault("ann_activation", "relu")
                cfg.setdefault("dropout", 0.2)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_category_results = {}  # Track category results for all models

                for idx, model_name in enumerate(all_models):
                    status_text.markdown(f"**Training {idx+1}/3: {model_name} (including clinical categories)...**")
                    progress_bar.progress((idx) / len(all_models))

                    with st.spinner(f"Training {model_name} for {cfg.get('ml_horizons', 7)} horizons..."):
                        try:
                            # Pass a copy of the config to prevent any potential state pollution between runs
                            current_run_config = cfg.copy()

                            # Use new function that trains on total + categories and captures residuals
                            full_results = run_ml_with_categories(
                                model_type=model_name,
                                config=current_run_config,
                                df=selected_df,
                                feature_cols=current_run_config['ml_feature_cols'],
                                split_ratio=current_run_config.get('split_ratio', 0.8),
                                horizons=current_run_config.get('ml_horizons', 7),
                            )

                            # Extract total results for backward compatibility
                            results = full_results.get("total_results", {})

                            # Store results for each model
                            all_results[model_name] = results
                            st.session_state[f"ml_mh_results_{model_name}"] = results

                            # ===== SAVE MODELS TO DISK FOR CLOUD SYNC =====
                            try:
                                saved_paths = save_ml_models_to_disk(model_name, results)
                                if saved_paths:
                                    logger.info(f"Saved {len(saved_paths)} {model_name} model files to disk")
                            except Exception as save_err:
                                logger.warning(f"Could not save {model_name} model files: {save_err}")

                            # Store residuals for hybrid models (Stage 1 → Stage 2)
                            total_residuals = full_results.get("total_residuals", {})
                            if total_residuals:
                                store_stage1_residuals(model_name, total_residuals)

                            # Track category results
                            category_results = full_results.get("category_results", {})
                            if category_results:
                                all_category_results[model_name] = {
                                    "categories": category_results,
                                    "categories_detected": full_results.get("categories_detected", []),
                                }

                        except Exception as e:
                            st.error(f"❌ **{model_name} training failed:**\n\n```\n{str(e)}\n```")
                            import traceback
                            st.code(traceback.format_exc())
                            all_results[model_name] = None

                # Complete progress
                progress_bar.progress(1.0)
                status_text.markdown("✅ **All models training completed!**")

                # Store category results from all models for Patient Forecast page
                if all_category_results:
                    st.session_state["ml_all_category_results"] = all_category_results
                    # Count categories detected
                    all_cats = set()
                    for model_data in all_category_results.values():
                        all_cats.update(model_data.get("categories_detected", []))
                    if all_cats:
                        st.info(f"📊 Category predictions captured for all models: {', '.join(sorted(all_cats))}")

                # Show residuals status
                residuals_ready = []
                for model_name in all_models:
                    if get_stage1_residuals(model_name):
                        residuals_ready.append(model_name)
                if residuals_ready:
                    st.success(f"✅ Residuals captured for: {', '.join(residuals_ready)} (ready for hybrid models)")

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

                            # Add seasonal category forecast for each model
                            per_h = results.get("per_h", {})
                            successful = results.get("successful", [])

                            if per_h and successful:
                                future_forecast_values = []
                                future_forecast_dates = []
                                for h in range(1, 8):
                                    if h in per_h:
                                        h_data = per_h.get(h, {})
                                        forecast_series = h_data.get("forecast")
                                        datetime_series = h_data.get("datetime")
                                        if forecast_series is not None and len(forecast_series) > 0:
                                            future_forecast_values.append(forecast_series[0])
                                            future_forecast_dates.append(datetime_series[0])
                                    else:
                                        future_forecast_values.append(np.nan)
                                        if future_forecast_dates:
                                            future_forecast_dates.append(future_forecast_dates[-1] + pd.Timedelta(days=1))
                                        else:
                                            future_forecast_dates.append(pd.NaT)

                                if future_forecast_values:
                                    with st.expander(f"📊 {model_name} Category Forecast (Seasonal Distribution)", expanded=False):
                                        _render_seasonal_category_forecast(
                                            predictions=future_forecast_values,
                                            dates=future_forecast_dates,
                                            title="Clinical Category Distribution"
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
                              gap: int = 0, progress_callback=None):
    """
    Run Bayesian Optimization using Optuna with TimeSeriesSplit CV.

    Academic Reference:
        Bergmeir & Benítez (2012) - Time series cross-validation ensures
        temporal ordering is respected to prevent data leakage.

    Args:
        model_type: "XGBoost", "LSTM", or "ANN"
        X_train: Training features
        y_train: Training target
        n_splits: Number of CV folds
        primary_metric: Metric to optimize
        n_trials: Number of Optuna trials
        gap: Number of samples between train/val sets (prevents autocorrelation leakage)
        progress_callback: Optional callback for progress updates

    Returns:
        dict with best_params, trial_results, study object, and all metrics
    """
    # Check if optuna is available
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. To use Bayesian Optimization, please install it:\n\n"
            "pip install optuna\n\n"
            "Alternatively, use 'Grid Search' or 'Random Search' which don't require optuna."
        )

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

    # Create TimeSeriesSplit with gap to prevent autocorrelation leakage
    # Academic Reference: Bergmeir & Benítez (2012) - Time series predictor evaluation
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

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


# =============================================================================
# OPTUNA VISUALIZATIONS (Step 2.3: HPO with Time Series CV)
# Academic Reference: Akiba et al. (2019) - Optuna: A Next-generation HPO Framework
# =============================================================================
def _render_optuna_visualizations(study: "optuna.Study", model_name: str) -> None:
    """
    Render Optuna optimization visualizations in an expandable section.

    Visualizations include:
    1. Optimization History - Shows objective value improvement over trials
    2. Parameter Importance - Shows which hyperparameters most affect performance
    3. Slice Plot - Shows individual parameter effects on objective
    4. Contour Plot - Shows parameter interactions (for top 2 params)

    Academic Reference:
        Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
        Optuna: A next-generation hyperparameter optimization framework.
        KDD '19.

    Args:
        study: Completed Optuna study object
        model_name: Name of the model for display purposes
    """
    if not OPTUNA_VIS_AVAILABLE:
        st.warning("Optuna visualization not available. Install with: `pip install optuna`")
        return

    if study is None or len(study.trials) < 2:
        st.info("Need at least 2 completed trials for visualizations.")
        return

    st.markdown("### 📊 Optuna Optimization Analysis")
    st.caption(f"Model: {model_name} | Trials: {len(study.trials)} | Best Value: {study.best_value:.4f}")

    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "📈 History",
        "🎯 Importance",
        "📉 Slice Plot",
        "🗺️ Contour"
    ])

    with viz_tabs[0]:
        st.markdown("#### Optimization History")
        st.info(
            "Shows how the objective value improves over trials. "
            "A decreasing trend (for minimization) indicates successful optimization."
        )
        try:
            fig_history = plot_optimization_history(study)
            fig_history.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400
            )
            st.plotly_chart(fig_history, use_container_width=True, key=f"optuna_history_{model_name}")
        except Exception as e:
            st.warning(f"Could not render history plot: {str(e)}")

    with viz_tabs[1]:
        st.markdown("#### Hyperparameter Importance")
        st.info(
            "Shows which hyperparameters have the most impact on the objective. "
            "Higher importance means the parameter significantly affects model performance."
        )
        try:
            # Need at least a few trials to compute importance
            if len(study.trials) >= 5:
                fig_importance = plot_param_importances(study)
                fig_importance.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_importance, use_container_width=True, key=f"optuna_importance_{model_name}")
            else:
                st.info("Need at least 5 trials to compute parameter importance.")
        except Exception as e:
            st.warning(f"Could not render importance plot: {str(e)}")

    with viz_tabs[2]:
        st.markdown("#### Slice Plot")
        st.info(
            "Shows how individual parameters affect the objective value. "
            "Helps identify optimal ranges for each hyperparameter."
        )
        try:
            fig_slice = plot_slice(study)
            fig_slice.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500
            )
            st.plotly_chart(fig_slice, use_container_width=True, key=f"optuna_slice_{model_name}")
        except Exception as e:
            st.warning(f"Could not render slice plot: {str(e)}")

    with viz_tabs[3]:
        st.markdown("#### Contour Plot (Top 2 Parameters)")
        st.info(
            "Shows the interaction between the two most important parameters. "
            "Helps identify optimal parameter combinations."
        )
        try:
            # Get parameter names for contour plot
            param_names = list(study.best_params.keys())
            if len(param_names) >= 2:
                fig_contour = plot_contour(study, params=param_names[:2])
                fig_contour.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                st.plotly_chart(fig_contour, use_container_width=True, key=f"optuna_contour_{model_name}")
            else:
                st.info("Need at least 2 parameters for contour plot.")
        except Exception as e:
            st.warning(f"Could not render contour plot: {str(e)}")

    # Summary statistics
    with st.expander("📋 Optimization Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trials", len(study.trials))
        with col2:
            st.metric("Best Trial", study.best_trial.number + 1)
        with col3:
            st.metric("Best Value", f"{study.best_value:.4f}")

        # Trial value statistics
        values = [t.value for t in study.trials if t.value is not None]
        if values:
            st.markdown("**Trial Value Statistics:**")
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Std", "Min", "Max", "Best"],
                "Value": [
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                    f"{study.best_value:.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


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

    # Check for individual model results
    # Check BOTH capital and lowercase keys (for compatibility with single/all modes)
    for model_name in ["XGBoost", "LSTM", "ANN"]:
        # Try capital key first (All Models mode stores with capitals)
        model_results = st.session_state.get(f"ml_mh_results_{model_name}")

        # If not found, try lowercase key (Single Model mode stores with lowercase)
        if model_results is None:
            model_results = st.session_state.get(f"ml_mh_results_{model_name.lower()}")

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
        # Search method selection - show warning if Bayesian selected but optuna not available
        search_options = ["Grid Search", "Random Search", "Bayesian Optimization"]
        search_method = st.selectbox(
            "Search Method",
            options=search_options,
            help="Grid: Exhaustive search | Random: Random sampling | Bayesian: Smart optimization (requires optuna)"
        )
        cfg["tuning_search_method"] = search_method

        # Show warning if Bayesian Optimization selected but optuna not installed
        if search_method == "Bayesian Optimization" and not OPTUNA_AVAILABLE:
            st.warning(
                "⚠️ **Optuna not installed**\n\n"
                "Bayesian Optimization requires the `optuna` package. "
                "Please install it with:\n\n"
                "```bash\npip install optuna\n```\n\n"
                "Or select **Grid Search** or **Random Search** instead."
            )

    with col2:
        # Optimization metric selection
        optimization_metric = st.selectbox(
            "Optimization Metric",
            options=["RMSE", "MAE", "MAPE", "Accuracy"],
            help="Metric to optimize. Lower is better for RMSE/MAE/MAPE, higher is better for Accuracy."
        )
        cfg["tuning_optimization_metric"] = optimization_metric

    # 2. Cross-Validation Strategy (with gap to prevent autocorrelation leakage)
    # Academic Reference: Bergmeir & Benítez (2012) - Time series CV
    st.subheader("🔄 Cross-Validation Strategy")

    # Check for CV config from Feature Studio
    # Handle both CVConfig dataclass and dict formats
    cv_config_from_studio = st.session_state.get("cv_config")
    cv_enabled = st.session_state.get("cv_enabled", False)

    # Extract CV parameters safely (works for dataclass or dict)
    if cv_config_from_studio is not None:
        if hasattr(cv_config_from_studio, 'n_splits'):
            # It's a CVConfig dataclass
            cv_n_splits = getattr(cv_config_from_studio, 'n_splits', 5)
            cv_gap_val = getattr(cv_config_from_studio, 'gap', 0)
        elif isinstance(cv_config_from_studio, dict):
            # It's a dict
            cv_n_splits = cv_config_from_studio.get('n_splits', cv_config_from_studio.get('n_folds', 5))
            cv_gap_val = cv_config_from_studio.get('gap', 0)
        else:
            cv_n_splits = 5
            cv_gap_val = 0
    else:
        cv_n_splits = 5
        cv_gap_val = 0

    if cv_enabled and cv_config_from_studio is not None:
        st.info(
            f"💡 **CV Config Detected from Feature Studio:**\n"
            f"- Folds: {cv_n_splits}\n"
            f"- Gap: {cv_gap_val} days"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        n_splits = st.slider(
            "Number of CV Folds",
            min_value=3,
            max_value=10,
            value=cv_n_splits if cv_enabled else cfg.get("tuning_cv_splits", 5),
            help="TimeSeriesSplit folds - more folds = more robust but slower"
        )
        cfg["tuning_cv_splits"] = n_splits

    with col2:
        cv_gap = st.slider(
            "Gap Between Train/Val (days)",
            min_value=0,
            max_value=14,
            value=cv_gap_val if cv_enabled else cfg.get("tuning_cv_gap", 0),
            help="Number of samples to skip between train and validation sets. "
                 "Helps prevent autocorrelation leakage. Recommended: 0-7 for daily data."
        )
        cfg["tuning_cv_gap"] = cv_gap

    with col3:
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
        else:
            st.caption("Grid Search uses all parameter combinations")

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

        # Disable button if Bayesian selected but optuna not available
        button_disabled = (search_method == "Bayesian Optimization" and not OPTUNA_AVAILABLE)

        run_optimization = st.button(
            button_text,
            type="primary",
            use_container_width=True,
            help="Start hyperparameter optimization with TimeSeriesSplit CV",
            disabled=button_disabled
        )

        if button_disabled:
            st.caption("⚠️ Install optuna to enable Bayesian Optimization")

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

                            # ===== SAVE OPTIMIZED PARAMS TO DISK FOR CLOUD SYNC =====
                            try:
                                results["search_method"] = "grid_search"
                                saved_opt = save_optimized_model_to_disk(
                                    f"{selected_model_opt}_GridSearch",
                                    results
                                )
                                if saved_opt:
                                    st.success(f"💾 Saved optimized params to `pipeline_artifacts/optimized/`")
                            except Exception as save_err:
                                st.warning(f"⚠️ Could not save optimized params: {save_err}")

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

                            # ===== SAVE OPTIMIZED PARAMS TO DISK FOR CLOUD SYNC =====
                            try:
                                results["search_method"] = "random_search"
                                saved_opt = save_optimized_model_to_disk(
                                    f"{selected_model_opt}_RandomSearch",
                                    results
                                )
                                if saved_opt:
                                    st.success(f"💾 Saved optimized params to `pipeline_artifacts/optimized/`")
                            except Exception as save_err:
                                st.warning(f"⚠️ Could not save optimized params: {save_err}")

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
                                n_trials=n_iter,
                                gap=cv_gap
                            )

                            # Store results in session state
                            st.session_state[f"opt_results_{selected_model_opt}"] = results
                            st.session_state["opt_last_model"] = selected_model_opt
                            st.session_state["opt_last_method"] = search_method

                            st.success(f"✅ Optimization completed successfully!")

                            # ===== SAVE OPTIMIZED PARAMS TO DISK FOR CLOUD SYNC =====
                            try:
                                results["search_method"] = "bayesian_optuna"
                                saved_opt = save_optimized_model_to_disk(
                                    f"{selected_model_opt}_Bayesian",
                                    results
                                )
                                if saved_opt:
                                    st.success(f"💾 Saved optimized params to `pipeline_artifacts/optimized/`")
                            except Exception as save_err:
                                st.warning(f"⚠️ Could not save optimized params: {save_err}")

                            # Show Optuna visualizations
                            if 'study' in results and OPTUNA_VIS_AVAILABLE:
                                _render_optuna_visualizations(results['study'], selected_model_opt)

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
                                # Use cfg fallback for n_iter and gap in case variable not in scope
                                _n_trials = cfg.get("tuning_n_iter", 30)
                                _cv_gap = cfg.get("tuning_cv_gap", 0)

                                with st.spinner(f"⏳ Optimizing {model_type}..."):
                                    results = run_bayesian_optimization(
                                        model_type=model_type,
                                        X_train=X_train,
                                        y_train=y_train,
                                        n_splits=n_splits,
                                        primary_metric=optimization_metric,
                                        n_trials=_n_trials,
                                        gap=_cv_gap
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

            # CV Results Table (handle both Grid/Random Search and Bayesian Optimization)
            with st.expander("📋 All Parameter Combinations (CV Results)", expanded=False):
                # Check which type of results we have
                if 'cv_results' in opt_results:
                    # Grid Search or Random Search (sklearn format)
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
                    else:
                        st.info("No CV results available to display.")

                elif 'trial_results' in opt_results:
                    # Bayesian Optimization (Optuna format)
                    trial_results = opt_results['trial_results']

                    st.markdown("**Bayesian Optimization Trial History:**")

                    # Format params column for better display
                    display_df = trial_results.copy()
                    display_df['params'] = display_df['params'].apply(lambda x: str(x) if isinstance(x, dict) else x)

                    # Rename columns for clarity
                    display_df = display_df.rename(columns={
                        'trial': 'Trial #',
                        'value': f'Score ({opt_results.get("refit_metric", "Metric")})',
                        'params': 'Parameters'
                    })

                    st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.info("No detailed CV results available for this optimization method.")

            # Display Optuna visualizations if study is available
            if 'study' in opt_results and OPTUNA_VIS_AVAILABLE:
                st.divider()
                _render_optuna_visualizations(opt_results['study'], selected_model_opt)

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
        else:
            # No results yet - provide guidance
            st.info(
                "📋 **No optimization results yet**\n\n"
                "Click the **Run** button above to optimize all three models (XGBoost, LSTM, ANN) sequentially.\n\n"
                "Each model will be optimized using the selected search method and the best hyperparameters "
                "will be displayed in a comparison dashboard."
            )

    # Debug panel
    debug_panel()

# -----------------------------------------------------------------------------
# HYBRID MODELS - Ensemble Forecasting
# -----------------------------------------------------------------------------
def page_hybrid():
    """
    Hybrid Models: Two-Stage Residual Correction Approach.

    This tab provides a clean interface for building hybrid models using
    residuals from Stage 1 models (LSTM, XGBoost, ANN, SARIMAX) to train
    Stage 2 correctors.

    Three hybrid pipelines available:
    1. LSTM → XGBoost: XGBoost learns to correct LSTM residuals
    2. SARIMAX → XGBoost: XGBoost learns to correct SARIMAX residuals
    3. LSTM → SARIMAX: SARIMAX learns to correct LSTM residuals
    """

    # Main Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: left; margin-bottom: 2rem;'>
          <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <div class='hf-feature-icon' style='margin: 0 1rem 0 0; font-size: 2.5rem;'>🧬</div>
            <h1 class='hf-feature-title' style='font-size: 2.5rem; margin: 0;'>Hybrid Models</h1>
          </div>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 750px; margin: 0 0 0 4.5rem;'>
            Two-stage residual correction: Stage 1 model predictions + Stage 2 residual corrections = Superior forecasts
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # SECTION 1: STAGE 1 READINESS CHECK
    # ==========================================================================
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.1));
                    border-left: 4px solid {PRIMARY_COLOR};
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1.5rem 0;
                    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);'>
            <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>
                📊 Stage 1: Check Model Readiness
            </h3>
            <p style='color: {TEXT_COLOR}; margin: 0; font-size: 0.95rem;'>
                Verify which Stage 1 models are trained and have residuals captured for hybrid use
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔍 Check Stage 1 Models & Residuals", type="secondary", use_container_width=True):
        _check_stage1_readiness()

    # Always show current status
    _display_stage1_status()

    st.divider()

    # ==========================================================================
    # SECTION 2: BUILD HYBRID MODELS
    # ==========================================================================
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
                    border-left: 4px solid {SUCCESS_COLOR};
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1.5rem 0;
                    box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);'>
            <h3 style='color: {SUCCESS_COLOR}; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(34, 197, 94, 0.5);'>
                🔧 Stage 2: Build Hybrid Models
            </h3>
            <p style='color: {TEXT_COLOR}; margin: 0; font-size: 0.95rem;'>
                Train Stage 2 models on residuals from Stage 1 to create hybrid forecasters
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check eligibility for each hybrid
    can_build = _get_hybrid_eligibility()

    # Hybrid selection checkboxes
    st.markdown("**Select Hybrid Models to Build:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        lstm_xgb_disabled = not can_build["LSTM → XGBoost"]
        build_lstm_xgb = st.checkbox(
            "🧠→🚀 LSTM → XGBoost",
            value=False,
            disabled=lstm_xgb_disabled,
            help="XGBoost learns to correct LSTM residuals" if not lstm_xgb_disabled else "Requires LSTM residuals"
        )
        if lstm_xgb_disabled:
            st.caption("❌ Missing LSTM residuals")

    with col2:
        sarimax_xgb_disabled = not can_build["SARIMAX → XGBoost"]
        build_sarimax_xgb = st.checkbox(
            "📈→🚀 SARIMAX → XGBoost",
            value=False,
            disabled=sarimax_xgb_disabled,
            help="XGBoost learns to correct SARIMAX residuals" if not sarimax_xgb_disabled else "Requires SARIMAX residuals"
        )
        if sarimax_xgb_disabled:
            st.caption("❌ Missing SARIMAX residuals")

    with col3:
        lstm_sarimax_disabled = not can_build["LSTM → SARIMAX"]
        build_lstm_sarimax = st.checkbox(
            "🧠→📈 LSTM → SARIMAX",
            value=False,
            disabled=lstm_sarimax_disabled,
            help="SARIMAX learns to correct LSTM residuals" if not lstm_sarimax_disabled else "Requires LSTM residuals"
        )
        if lstm_sarimax_disabled:
            st.caption("❌ Missing LSTM residuals")

    # Build button
    any_selected = build_lstm_xgb or build_sarimax_xgb or build_lstm_sarimax

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button(
            "🚀 Build Hybrid Models",
            type="primary",
            use_container_width=True,
            disabled=not any_selected
        ):
            _build_all_hybrids(build_lstm_xgb, build_sarimax_xgb, build_lstm_sarimax)

    if not any_selected:
        st.info("ℹ️ Select at least one hybrid model to build. Ensure Stage 1 models are trained first.")

    st.divider()

    # ==========================================================================
    # SECTION 3: RESULTS DISPLAY
    # ==========================================================================
    _display_hybrid_results()

def _check_stage1_readiness():
    """Display detailed status of Stage 1 models and their residuals."""

    st.markdown("### 🔍 Stage 1 Model Analysis")

    models_status = {
        "LSTM": {
            "trained": "ml_mh_results_lstm" in st.session_state and st.session_state.get("ml_mh_results_lstm"),
            "residuals": bool(get_stage1_residuals("LSTM")),
        },
        "XGBoost": {
            "trained": "ml_mh_results_xgboost" in st.session_state and st.session_state.get("ml_mh_results_xgboost"),
            "residuals": bool(get_stage1_residuals("XGBoost")),
        },
        "ANN": {
            "trained": "ml_mh_results_ann" in st.session_state and st.session_state.get("ml_mh_results_ann"),
            "residuals": bool(get_stage1_residuals("ANN")),
        },
        "SARIMAX": {
            "trained": "sarimax_results" in st.session_state and st.session_state.get("sarimax_results"),
            "residuals": bool(get_stage1_residuals("SARIMAX")),
        }
    }

    # Display detailed status
    for model, status in models_status.items():
        if status["trained"] and status["residuals"]:
            st.success(f"✅ **{model}**: Model trained + Residuals captured → Ready for hybrid!")
        elif status["trained"]:
            st.warning(f"⚠️ **{model}**: Model trained but residuals NOT captured. Re-train in ML tab to capture residuals.")
        else:
            st.error(f"❌ **{model}**: Not trained. Train in ML tab (or Baseline Models for SARIMAX).")

    # Show residual statistics if available
    st.markdown("---")
    st.markdown("### 📈 Residual Statistics")

    for model in ["LSTM", "XGBoost", "ANN", "SARIMAX"]:
        residuals = get_stage1_residuals(model)
        if residuals:
            val_resid = residuals.get("val_residuals", {})
            if val_resid:
                with st.expander(f"📊 {model} Residual Stats", expanded=False):
                    stats_data = []
                    for target, resid in val_resid.items():
                        if resid is not None and len(resid) > 0:
                            stats_data.append({
                                "Target": target,
                                "Mean": np.mean(resid),
                                "Std": np.std(resid),
                                "Min": np.min(resid),
                                "Max": np.max(resid),
                            })
                    if stats_data:
                        st.dataframe(pd.DataFrame(stats_data).style.format({
                            "Mean": "{:.2f}",
                            "Std": "{:.2f}",
                            "Min": "{:.2f}",
                            "Max": "{:.2f}",
                        }), use_container_width=True, hide_index=True)


def _display_stage1_status():
    """Display compact Stage 1 status cards."""

    models_status = {
        "LSTM": {
            "trained": "ml_mh_results_lstm" in st.session_state and st.session_state.get("ml_mh_results_lstm"),
            "residuals": bool(get_stage1_residuals("LSTM")),
        },
        "XGBoost": {
            "trained": "ml_mh_results_xgboost" in st.session_state and st.session_state.get("ml_mh_results_xgboost"),
            "residuals": bool(get_stage1_residuals("XGBoost")),
        },
        "ANN": {
            "trained": "ml_mh_results_ann" in st.session_state and st.session_state.get("ml_mh_results_ann"),
            "residuals": bool(get_stage1_residuals("ANN")),
        },
        "SARIMAX": {
            "trained": "sarimax_results" in st.session_state and st.session_state.get("sarimax_results"),
            "residuals": bool(get_stage1_residuals("SARIMAX")),
        }
    }

    cols = st.columns(4)
    model_icons = {"LSTM": "🧠", "XGBoost": "🚀", "ANN": "🎯", "SARIMAX": "📈"}

    for i, (model, status) in enumerate(models_status.items()):
        icon = model_icons.get(model, "📦")
        with cols[i]:
            if status["trained"] and status["residuals"]:
                st.success(f"{icon} **{model}**\n\n✅ Ready")
            elif status["trained"]:
                st.warning(f"{icon} **{model}**\n\n⚠️ No residuals")
            else:
                st.error(f"{icon} **{model}**\n\n❌ Not trained")


def _get_hybrid_eligibility() -> dict:
    """Check which hybrid models can be built based on available residuals."""

    lstm_residuals = bool(get_stage1_residuals("LSTM"))
    sarimax_residuals = bool(get_stage1_residuals("SARIMAX"))

    # XGBoost as Stage 2 doesn't need prior training - it will be trained fresh on residuals
    # SARIMAX as Stage 2 doesn't need prior training - it will be trained fresh on residuals

    return {
        "LSTM → XGBoost": lstm_residuals,
        "SARIMAX → XGBoost": sarimax_residuals,
        "LSTM → SARIMAX": lstm_residuals,
    }


def _build_all_hybrids(build_lstm_xgb: bool, build_sarimax_xgb: bool, build_lstm_sarimax: bool):
    """Build selected hybrid models using residual correction approach."""

    results = {}

    # Get feature data
    fe = st.session_state.get("feature_engineering", {})
    # Note: Cannot use `or` with DataFrames - pandas raises ValueError
    df_a = fe.get("A")
    df = df_a if df_a is not None else fe.get("B")

    if df is None or df.empty:
        st.error("❌ No feature-engineered data found. Run Feature Studio first.")
        return

    progress = st.progress(0)
    status = st.empty()

    total_hybrids = sum([build_lstm_xgb, build_sarimax_xgb, build_lstm_sarimax])
    completed = 0

    with st.spinner("Building hybrid models..."):
        try:
            # HYBRID 1: LSTM → XGBoost
            if build_lstm_xgb:
                status.info("🔧 Building LSTM → XGBoost hybrid...")
                results["lstm_xgb"] = _train_hybrid_lstm_xgboost(df)
                completed += 1
                progress.progress(completed / total_hybrids)

            # HYBRID 2: SARIMAX → XGBoost
            if build_sarimax_xgb:
                status.info("🔧 Building SARIMAX → XGBoost hybrid...")
                results["sarimax_xgb"] = _train_hybrid_sarimax_xgboost(df)
                completed += 1
                progress.progress(completed / total_hybrids)

            # HYBRID 3: LSTM → SARIMAX
            if build_lstm_sarimax:
                status.info("🔧 Building LSTM → SARIMAX hybrid...")
                results["lstm_sarimax"] = _train_hybrid_lstm_sarimax(df)
                completed += 1
                progress.progress(completed / total_hybrids)

            progress.progress(1.0)
            status.empty()

            # Store results
            st.session_state["hybrid_pipeline_results"] = results
            st.success(f"✅ Built {len(results)} hybrid model(s) successfully!")

            # Show brief summary
            for name, res in results.items():
                if res and res.get("success"):
                    avg_mae = res.get("avg_mae", 0)
                    st.info(f"📊 **{name}**: Avg MAE = {avg_mae:.2f}")

        except Exception as e:
            st.error(f"❌ Hybrid model building failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def _train_hybrid_lstm_xgboost(df: pd.DataFrame) -> dict:
    """
    LSTM → XGBoost Hybrid using two-stage residual correction.

    Stage 1: LSTM predictions (already trained, residuals captured)
    Stage 2: XGBoost learns to correct LSTM residuals

    Final prediction = Stage1_LSTM_pred + Stage2_XGBoost_correction
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    residuals = get_stage1_residuals("LSTM")
    if not residuals:
        return {"success": False, "error": "LSTM residuals not found"}

    val_residuals = residuals.get("val_residuals", {})
    val_predictions = residuals.get("val_predictions", {})
    feature_data = residuals.get("feature_data", {})

    X_val = feature_data.get("X_val")
    feature_cols = feature_data.get("feature_cols", [])

    if X_val is None:
        return {"success": False, "error": "Validation features not found"}

    # Ensure X_val is numeric (handle legacy data with timestamps)
    try:
        X_val = np.asarray(X_val, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return {"success": False, "error": f"Features contain non-numeric data: {e}. Re-train the ML model."}

    results = {"per_h": {}, "success": True, "model_name": "LSTM → XGBoost"}

    # Train XGBoost on residuals for each horizon
    for h in range(1, 8):
        target_key = f"Target_{h}"

        if target_key not in val_residuals or target_key not in val_predictions:
            continue

        resid = val_residuals[target_key]
        stage1_pred = val_predictions[target_key]

        # Get actual values from the target column
        target_col = f"Target_{h}"
        if target_col not in df.columns:
            continue

        # Align data lengths
        n_samples = min(len(X_val), len(resid))
        X_train_resid = X_val[:n_samples].copy()
        y_train_resid = np.asarray(resid[:n_samples], dtype=np.float64)
        stage1_pred_aligned = np.asarray(stage1_pred[:n_samples], dtype=np.float64)

        # Clean NaN/inf values - XGBoost cannot handle them
        valid_mask = np.isfinite(y_train_resid) & np.isfinite(stage1_pred_aligned)
        if not valid_mask.all():
            X_train_resid = X_train_resid[valid_mask]
            y_train_resid = y_train_resid[valid_mask]
            stage1_pred_aligned = stage1_pred_aligned[valid_mask]

        # =====================================================
        # SPLIT VALIDATION DATA INTO STAGE2_TRAIN / STAGE2_TEST
        # This prevents data leakage (training and testing on same data)
        # =====================================================
        n_total = len(X_train_resid)
        split_idx = int(n_total * 0.8)  # 80% train, 20% test (same as single models)

        # Stage 2 Training data
        X_s2_train = X_train_resid[:split_idx]
        y_s2_train = y_train_resid[:split_idx]
        stage1_pred_train = stage1_pred_aligned[:split_idx]

        # Stage 2 Test data (held out for honest evaluation)
        X_s2_test = X_train_resid[split_idx:]
        y_s2_test = y_train_resid[split_idx:]
        stage1_pred_test = stage1_pred_aligned[split_idx:]

        # Skip if not enough samples for train/test
        if len(X_s2_train) < 10 or len(X_s2_test) < 5:
            continue

        # =====================================================
        # TRAIN XGBOOST ON STAGE2_TRAIN ONLY
        # =====================================================
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_s2_train, y_s2_train)

        # =====================================================
        # PREDICT ON HELD-OUT STAGE2_TEST (HONEST EVALUATION)
        # =====================================================
        stage2_correction_test = xgb_model.predict(X_s2_test)
        final_pred_test = stage1_pred_test + stage2_correction_test
        y_actual_test = stage1_pred_test + y_s2_test

        # =====================================================
        # CALCULATE METRICS ON TEST SET ONLY
        # =====================================================
        mae = mean_absolute_error(y_actual_test, final_pred_test)
        rmse = np.sqrt(mean_squared_error(y_actual_test, final_pred_test))

        # MAPE with safe division (avoid near-zero values)
        non_zero_mask = np.abs(y_actual_test) > 1e-8
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs(
                (y_actual_test[non_zero_mask] - final_pred_test[non_zero_mask])
                / y_actual_test[non_zero_mask]
            )) * 100
        else:
            mape = 0.0

        # Accuracy (capped at 0-100)
        accuracy = max(0.0, min(100.0, 100.0 - mape))

        # Store results (use TEST data for charts)
        results["per_h"][h] = {
            "stage1_pred": stage1_pred_test,
            "stage2_correction": stage2_correction_test,
            "final_pred": final_pred_test,
            "actual": y_actual_test,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "accuracy": accuracy,
            "n_train": len(X_s2_train),
            "n_test": len(X_s2_test),
        }

    # Calculate average metrics
    if results["per_h"]:
        results["avg_mae"] = np.mean([r["mae"] for r in results["per_h"].values()])
        results["avg_rmse"] = np.mean([r["rmse"] for r in results["per_h"].values()])
        results["avg_mape"] = np.mean([r["mape"] for r in results["per_h"].values()])
        results["avg_accuracy"] = np.mean([r["accuracy"] for r in results["per_h"].values()])

    return results


def _train_hybrid_sarimax_xgboost(df: pd.DataFrame) -> dict:
    """
    SARIMAX → XGBoost Hybrid using two-stage residual correction.

    Stage 1: SARIMAX predictions (already trained, residuals captured)
    Stage 2: XGBoost learns to correct SARIMAX residuals

    Final prediction = Stage1_SARIMAX_pred + Stage2_XGBoost_correction
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    residuals = get_stage1_residuals("SARIMAX")
    if not residuals:
        return {"success": False, "error": "SARIMAX residuals not found"}

    val_residuals = residuals.get("val_residuals", {})
    val_predictions = residuals.get("val_predictions", {})
    feature_data = residuals.get("feature_data", {})

    X_val = feature_data.get("X_val")

    if X_val is None:
        return {"success": False, "error": "Validation features not found"}

    # Ensure X_val is numeric (handle legacy data with timestamps)
    try:
        X_val = np.asarray(X_val, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return {"success": False, "error": f"Features contain non-numeric data: {e}. Re-train the SARIMAX model."}

    results = {"per_h": {}, "success": True, "model_name": "SARIMAX → XGBoost"}

    # Train XGBoost on residuals for each horizon
    for h in range(1, 8):
        target_key = f"Target_{h}"

        if target_key not in val_residuals or target_key not in val_predictions:
            continue

        resid = val_residuals[target_key]
        stage1_pred = val_predictions[target_key]

        # Align data lengths
        n_samples = min(len(X_val), len(resid))
        X_train_resid = X_val[:n_samples].copy()
        y_train_resid = np.asarray(resid[:n_samples], dtype=np.float64)
        stage1_pred_aligned = np.asarray(stage1_pred[:n_samples], dtype=np.float64)

        # Clean NaN/inf values - XGBoost cannot handle them
        valid_mask = np.isfinite(y_train_resid) & np.isfinite(stage1_pred_aligned)
        if not valid_mask.all():
            X_train_resid = X_train_resid[valid_mask]
            y_train_resid = y_train_resid[valid_mask]
            stage1_pred_aligned = stage1_pred_aligned[valid_mask]

        # =====================================================
        # SPLIT VALIDATION DATA INTO STAGE2_TRAIN / STAGE2_TEST
        # This prevents data leakage (training and testing on same data)
        # =====================================================
        n_total = len(X_train_resid)
        split_idx = int(n_total * 0.8)  # 80% train, 20% test (same as single models)

        # Stage 2 Training data
        X_s2_train = X_train_resid[:split_idx]
        y_s2_train = y_train_resid[:split_idx]
        stage1_pred_train = stage1_pred_aligned[:split_idx]

        # Stage 2 Test data (held out for honest evaluation)
        X_s2_test = X_train_resid[split_idx:]
        y_s2_test = y_train_resid[split_idx:]
        stage1_pred_test = stage1_pred_aligned[split_idx:]

        # Skip if not enough samples for train/test
        if len(X_s2_train) < 10 or len(X_s2_test) < 5:
            continue

        # =====================================================
        # TRAIN XGBOOST ON STAGE2_TRAIN ONLY
        # =====================================================
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_s2_train, y_s2_train)

        # =====================================================
        # PREDICT ON HELD-OUT STAGE2_TEST (HONEST EVALUATION)
        # =====================================================
        stage2_correction_test = xgb_model.predict(X_s2_test)
        final_pred_test = stage1_pred_test + stage2_correction_test
        y_actual_test = stage1_pred_test + y_s2_test

        # =====================================================
        # CALCULATE METRICS ON TEST SET ONLY
        # =====================================================
        mae = mean_absolute_error(y_actual_test, final_pred_test)
        rmse = np.sqrt(mean_squared_error(y_actual_test, final_pred_test))

        # MAPE with safe division (avoid near-zero values)
        non_zero_mask = np.abs(y_actual_test) > 1e-8
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs(
                (y_actual_test[non_zero_mask] - final_pred_test[non_zero_mask])
                / y_actual_test[non_zero_mask]
            )) * 100
        else:
            mape = 0.0

        # Accuracy (capped at 0-100)
        accuracy = max(0.0, min(100.0, 100.0 - mape))

        # Store results (use TEST data for charts)
        results["per_h"][h] = {
            "stage1_pred": stage1_pred_test,
            "stage2_correction": stage2_correction_test,
            "final_pred": final_pred_test,
            "actual": y_actual_test,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "accuracy": accuracy,
            "n_train": len(X_s2_train),
            "n_test": len(X_s2_test),
        }

    # Calculate average metrics
    if results["per_h"]:
        results["avg_mae"] = np.mean([r["mae"] for r in results["per_h"].values()])
        results["avg_rmse"] = np.mean([r["rmse"] for r in results["per_h"].values()])
        results["avg_mape"] = np.mean([r["mape"] for r in results["per_h"].values()])
        results["avg_accuracy"] = np.mean([r["accuracy"] for r in results["per_h"].values()])

    return results


def _train_hybrid_lstm_sarimax(df: pd.DataFrame) -> dict:
    """
    LSTM → SARIMAX Hybrid using two-stage residual correction.

    Stage 1: LSTM predictions (already trained, residuals captured)
    Stage 2: SARIMAX learns to model LSTM residual time series

    Final prediction = Stage1_LSTM_pred + Stage2_SARIMAX_correction
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')

    residuals = get_stage1_residuals("LSTM")
    if not residuals:
        return {"success": False, "error": "LSTM residuals not found"}

    val_residuals = residuals.get("val_residuals", {})
    val_predictions = residuals.get("val_predictions", {})

    results = {"per_h": {}, "success": True, "model_name": "LSTM → SARIMAX"}

    # Train SARIMAX on residual time series for each horizon
    for h in range(1, 8):
        target_key = f"Target_{h}"

        if target_key not in val_residuals or target_key not in val_predictions:
            continue

        resid = np.asarray(val_residuals[target_key], dtype=np.float64)
        stage1_pred = np.asarray(val_predictions[target_key], dtype=np.float64)

        # Clean NaN/inf values - SARIMAX cannot handle them
        valid_mask = np.isfinite(resid) & np.isfinite(stage1_pred)
        if not valid_mask.all():
            resid = resid[valid_mask]
            stage1_pred = stage1_pred[valid_mask]

        # =====================================================
        # SPLIT RESIDUALS INTO STAGE2_TRAIN / STAGE2_TEST
        # This prevents data leakage (using fittedvalues on same data)
        # =====================================================
        n_total = len(resid)
        split_idx = int(n_total * 0.8)  # 80% train, 20% test (same as single models)

        resid_train = resid[:split_idx]
        resid_test = resid[split_idx:]
        stage1_pred_train = stage1_pred[:split_idx]
        stage1_pred_test = stage1_pred[split_idx:]

        # Skip if not enough samples for train/test
        if len(resid_train) < 10 or len(resid_test) < 5:
            continue

        try:
            # =====================================================
            # TRAIN SARIMAX ON RESIDUAL TRAIN SET ONLY
            # =====================================================
            sarimax_model = SARIMAX(
                resid_train,
                order=(1, 0, 1),  # Simple ARMA on residuals
                seasonal_order=(0, 0, 0, 0),  # No seasonality on residuals
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_fit = sarimax_model.fit(disp=False, maxiter=100)

            # =====================================================
            # FORECAST ON TEST SET (NOT fittedvalues!)
            # =====================================================
            n_test = len(resid_test)
            stage2_correction_test = sarimax_fit.forecast(steps=n_test)
            stage2_correction_test = np.asarray(stage2_correction_test, dtype=np.float64)

            # Final prediction on test set
            final_pred_test = stage1_pred_test + stage2_correction_test
            y_actual_test = stage1_pred_test + resid_test

            # =====================================================
            # CALCULATE METRICS ON TEST SET ONLY
            # =====================================================
            mae = mean_absolute_error(y_actual_test, final_pred_test)
            rmse = np.sqrt(mean_squared_error(y_actual_test, final_pred_test))

            # MAPE with safe division (avoid near-zero values)
            non_zero_mask = np.abs(y_actual_test) > 1e-8
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs(
                    (y_actual_test[non_zero_mask] - final_pred_test[non_zero_mask])
                    / y_actual_test[non_zero_mask]
                )) * 100
            else:
                mape = 0.0

            # Accuracy (capped at 0-100)
            accuracy = max(0.0, min(100.0, 100.0 - mape))

            # Store results (use TEST data for charts)
            results["per_h"][h] = {
                "stage1_pred": stage1_pred_test,
                "stage2_correction": stage2_correction_test,
                "final_pred": final_pred_test,
                "actual": y_actual_test,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "accuracy": accuracy,
                "n_train": len(resid_train),
                "n_test": len(resid_test),
            }

        except Exception as e:
            print(f"LSTM→SARIMAX horizon {h} failed: {e}")
            continue

    # Calculate average metrics
    if results["per_h"]:
        results["avg_mae"] = np.mean([r["mae"] for r in results["per_h"].values()])
        results["avg_rmse"] = np.mean([r["rmse"] for r in results["per_h"].values()])
        results["avg_mape"] = np.mean([r["mape"] for r in results["per_h"].values()])
        results["avg_accuracy"] = np.mean([r["accuracy"] for r in results["per_h"].values()])

    return results


def _display_hybrid_results():
    """Display hybrid model results with professional KPI cards and comparison table."""

    if "hybrid_pipeline_results" not in st.session_state:
        st.info("ℹ️ No hybrid models built yet. Train Stage 1 models in the **Machine Learning** tab, then click **Build Hybrid Models** above.")
        return

    results = st.session_state["hybrid_pipeline_results"]

    if not results:
        st.warning("⚠️ Hybrid results are empty.")
        return

    # Count successful models
    successful_models = [r for r in results.values() if r and r.get("success")]
    num_successful = len(successful_models)

    if num_successful == 0:
        st.warning("⚠️ No successful hybrid models to display.")
        return

    # =========================================================================
    # SUCCESS BANNER
    # =========================================================================
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.15));
                    border-left: 4px solid #22C55E;
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1rem 0 1.5rem 0;
                    box-shadow: 0 0 25px rgba(34, 197, 94, 0.2);'>
            <h3 style='color: #22C55E; margin: 0 0 0.5rem 0; font-size: 1.4rem; font-weight: 700;
                       text-shadow: 0 0 15px rgba(34, 197, 94, 0.5);'>
                ✅ Hybrid Models Trained Successfully!
            </h3>
            <p style='margin: 0; color: #E5E7EB; font-size: 1rem;'>
                {num_successful} hybrid model(s) built using two-stage residual correction
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================================
    # AGGREGATE KPI CARDS (Best Model's Metrics)
    # =========================================================================
    # Find best model by lowest MAE
    best_model = min(successful_models, key=lambda x: x.get("avg_mae", float('inf')))
    best_name = best_model.get("model_name", "Hybrid")

    st.markdown(f"#### 🏆 Best Hybrid Model: **{best_name}**")

    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(
            _create_kpi_indicator("MAE", best_model.get("avg_mae", 0), "", PRIMARY_COLOR),
            use_container_width=True,
            key="hybrid_kpi_mae"
        )
    with col2:
        st.plotly_chart(
            _create_kpi_indicator("RMSE", best_model.get("avg_rmse", 0), "", SECONDARY_COLOR),
            use_container_width=True,
            key="hybrid_kpi_rmse"
        )
    with col3:
        st.plotly_chart(
            _create_kpi_indicator("MAPE", best_model.get("avg_mape", 0), "%", WARNING_COLOR),
            use_container_width=True,
            key="hybrid_kpi_mape"
        )
    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Accuracy", best_model.get("avg_accuracy", 0), "%", SUCCESS_COLOR),
            use_container_width=True,
            key="hybrid_kpi_acc"
        )

    # =========================================================================
    # MODEL COMPARISON TABLE (Styled)
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")

    comparison_data = []
    for hybrid_name, hybrid_results in results.items():
        if hybrid_results and hybrid_results.get("success"):
            comparison_data.append({
                "Model": hybrid_results.get("model_name", hybrid_name),
                "MAE": hybrid_results.get("avg_mae", 0),
                "RMSE": hybrid_results.get("avg_rmse", 0),
                "MAPE (%)": hybrid_results.get("avg_mape", 0),
                "Accuracy (%)": hybrid_results.get("avg_accuracy", 0),
                "Horizons": len(hybrid_results.get("per_h", {})),
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Apply styling with color gradients
        styled_df = comparison_df.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "MAPE (%)": "{:.2f}",
            "Accuracy (%)": "{:.2f}",
        }).background_gradient(
            subset=["MAE", "RMSE", "MAPE (%)"],
            cmap="RdYlGn_r",  # Red for high (bad), Green for low (good)
            vmin=0
        ).background_gradient(
            subset=["Accuracy (%)"],
            cmap="RdYlGn",  # Green for high (good), Red for low (bad)
            vmin=0,
            vmax=100
        ).highlight_min(
            subset=["MAE", "RMSE", "MAPE (%)"],
            props="font-weight: bold; border: 2px solid #22C55E;"
        ).highlight_max(
            subset=["Accuracy (%)"],
            props="font-weight: bold; border: 2px solid #22C55E;"
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # DETAILED TABS FOR EACH HYBRID MODEL
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 📈 Detailed Model Analysis")

    tab_names = [res.get("model_name", name) for name, res in results.items() if res and res.get("success")]
    tabs = st.tabs(tab_names)

    for tab, (name, res) in zip(tabs, [(n, r) for n, r in results.items() if r and r.get("success")]):
        with tab:
            _plot_hybrid_results(res.get("model_name", name), res)


def _plot_hybrid_results(name: str, res: dict):
    """Plot detailed results for a single hybrid model with comprehensive charts."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    per_h = res.get("per_h", {})

    if not per_h:
        st.info("No horizon data available.")
        return

    # =========================================================================
    # PER-HORIZON METRICS TABLE (Enhanced with MAPE and Accuracy)
    # =========================================================================
    metrics_data = []
    for h, h_data in sorted(per_h.items()):
        metrics_data.append({
            "Horizon": f"Day {h}",
            "MAE": h_data.get("mae", 0),
            "RMSE": h_data.get("rmse", 0),
            "MAPE (%)": h_data.get("mape", 0),
            "Accuracy (%)": h_data.get("accuracy", 0),
        })

    if metrics_data:
        st.markdown("##### 📋 Per-Horizon Metrics")
        metrics_df = pd.DataFrame(metrics_data)
        styled_metrics = metrics_df.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "MAPE (%)": "{:.2f}",
            "Accuracy (%)": "{:.2f}",
        }).background_gradient(
            subset=["MAE", "RMSE", "MAPE (%)"],
            cmap="RdYlGn_r",
            vmin=0
        ).background_gradient(
            subset=["Accuracy (%)"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100
        )
        st.dataframe(styled_metrics, use_container_width=True, hide_index=True)

    # =========================================================================
    # CHART 1: FORECAST COMPARISON (Actual vs Stage1 vs Hybrid)
    # =========================================================================
    first_h = list(per_h.keys())[0]
    h_data = per_h[first_h]

    actual = h_data.get("actual")
    final_pred = h_data.get("final_pred")
    stage1_pred = h_data.get("stage1_pred")

    st.markdown("##### 📈 Forecast Comparison")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        if actual is not None and final_pred is not None:
            fig_forecast = go.Figure()

            # Actual values
            fig_forecast.add_trace(go.Scatter(
                y=actual,
                mode='lines+markers',
                name='Actual',
                line=dict(color=PRIMARY_COLOR, width=2),
                marker=dict(size=4)
            ))

            # Stage 1 predictions
            if stage1_pred is not None:
                fig_forecast.add_trace(go.Scatter(
                    y=stage1_pred,
                    mode='lines',
                    name='Stage 1 (Base)',
                    line=dict(color=WARNING_COLOR, width=2, dash='dot'),
                ))

            # Final hybrid predictions
            fig_forecast.add_trace(go.Scatter(
                y=final_pred,
                mode='lines+markers',
                name='Hybrid Final',
                line=dict(color=SUCCESS_COLOR, width=2),
                marker=dict(size=4)
            ))

            fig_forecast.update_layout(
                title=f"Horizon {first_h}: Actual vs Predictions",
                xaxis_title="Sample Index",
                yaxis_title="ED Arrivals",
                height=350,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40)
            )

            st.plotly_chart(fig_forecast, use_container_width=True)

    # =========================================================================
    # CHART 2: RESIDUAL ANALYSIS (Stage2 Corrections)
    # =========================================================================
    with col_chart2:
        stage2_correction = h_data.get("stage2_correction")
        if stage2_correction is not None:
            fig_residual = go.Figure()

            # Residual distribution as histogram
            fig_residual.add_trace(go.Histogram(
                x=stage2_correction,
                name='Stage 2 Corrections',
                marker_color=SECONDARY_COLOR,
                opacity=0.7,
                nbinsx=30
            ))

            # Add vertical line at zero
            fig_residual.add_vline(
                x=0,
                line_dash="dash",
                line_color=WARNING_COLOR,
                annotation_text="Zero",
                annotation_position="top"
            )

            fig_residual.update_layout(
                title="Stage 2 Correction Distribution",
                xaxis_title="Correction Value",
                yaxis_title="Frequency",
                height=350,
                template='plotly_white',
                showlegend=False,
                margin=dict(l=40, r=20, t=60, b=40)
            )

            st.plotly_chart(fig_residual, use_container_width=True)

    # =========================================================================
    # CHART 3: ERROR METRICS BY HORIZON (Bar Chart)
    # =========================================================================
    st.markdown("##### 📊 Error Metrics by Horizon")

    horizons = list(sorted(per_h.keys()))
    mae_values = [per_h[h].get("mae", 0) for h in horizons]
    rmse_values = [per_h[h].get("rmse", 0) for h in horizons]
    accuracy_values = [per_h[h].get("accuracy", 0) for h in horizons]

    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        fig_errors = go.Figure()

        fig_errors.add_trace(go.Bar(
            name='MAE',
            x=[f"Day {h}" for h in horizons],
            y=mae_values,
            marker_color=PRIMARY_COLOR,
        ))

        fig_errors.add_trace(go.Bar(
            name='RMSE',
            x=[f"Day {h}" for h in horizons],
            y=rmse_values,
            marker_color=SECONDARY_COLOR,
        ))

        fig_errors.update_layout(
            title="MAE & RMSE by Horizon",
            xaxis_title="Forecast Horizon",
            yaxis_title="Error Value",
            barmode='group',
            height=300,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(fig_errors, use_container_width=True)

    with col_bar2:
        fig_accuracy = go.Figure()

        fig_accuracy.add_trace(go.Bar(
            name='Accuracy',
            x=[f"Day {h}" for h in horizons],
            y=accuracy_values,
            marker_color=SUCCESS_COLOR,
            text=[f"{v:.1f}%" for v in accuracy_values],
            textposition='outside'
        ))

        fig_accuracy.update_layout(
            title="Accuracy by Horizon",
            xaxis_title="Forecast Horizon",
            yaxis_title="Accuracy (%)",
            height=300,
            template='plotly_white',
            yaxis=dict(range=[0, max(100, max(accuracy_values) * 1.1)]),
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(fig_accuracy, use_container_width=True)

    # =========================================================================
    # CHART 4: SCATTER PLOT (Actual vs Predicted)
    # =========================================================================
    st.markdown("##### 🎯 Prediction Accuracy Scatter")

    if actual is not None and final_pred is not None:
        fig_scatter = go.Figure()

        # Scatter plot of actual vs predicted
        fig_scatter.add_trace(go.Scatter(
            x=actual,
            y=final_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=PRIMARY_COLOR,
                size=8,
                opacity=0.6
            )
        ))

        # Perfect prediction line
        min_val = min(min(actual), min(final_pred))
        max_val = max(max(actual), max(final_pred))
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color=SUCCESS_COLOR, width=2, dash='dash')
        ))

        fig_scatter.update_layout(
            title=f"Actual vs Predicted (Horizon {first_h})",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=350,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(fig_scatter, use_container_width=True)


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


def _render_hybrid_diagnostic_pipeline():
    """
    Render the enhanced hybrid model diagnostic pipeline with RAG-powered recommendations.

    This pipeline:
    1. Analyzes ALL trained models (SARIMAX, LSTM, ANN, XGBoost) individually
    2. Compares residual patterns across models
    3. Uses RAG approach to provide intelligent recommendations
    4. Suggests best hybrid strategies and model combinations
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
                border-left: 4px solid #6366F1;
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                box-shadow: 0 0 25px rgba(99, 102, 241, 0.2);'>
        <h3 style='color: #6366F1; margin: 0 0 0.5rem 0; font-size: 1.4rem;
                   text-shadow: 0 0 15px rgba(99, 102, 241, 0.5);'>
            🔬 Hybrid Model Diagnostic Pipeline
        </h3>
        <p style='margin: 0; color: #E5E7EB; font-size: 1rem;'>
            Analyzes ALL trained models, compares residual patterns, and recommends the best hybrid strategy
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 **Run Diagnostic Pipeline** - Analyze all models and get recommendations", expanded=True):
        # =========================================================================
        # STEP 1: CHECK FOR TRAINED MODELS
        # =========================================================================
        st.markdown("### Step 1: Detect Trained Models")

        trained_models = {}
        model_predictions = {}

        # Check SARIMAX
        sarimax_results = st.session_state.get("sarimax_results")
        if sarimax_results and sarimax_results.get("per_h"):
            trained_models["SARIMAX"] = sarimax_results
            y_true, y_pred, dates = _extract_predictions_for_diagnostic("SARIMAX")
            if y_true is not None and y_pred is not None:
                model_predictions["SARIMAX"] = y_pred
                if "y_true" not in model_predictions:
                    model_predictions["y_true"] = y_true
                    model_predictions["dates"] = dates

        # Check individual ML models
        for model_name in ["XGBoost", "LSTM", "ANN"]:
            key = f"ml_mh_results_{model_name.lower()}"
            results = st.session_state.get(key, {})
            if results and results.get("per_h"):
                trained_models[model_name] = results
                y_true, y_pred, dates = _extract_predictions_for_diagnostic(model_name)
                if y_true is not None and y_pred is not None:
                    model_predictions[model_name] = y_pred
                    if "y_true" not in model_predictions:
                        model_predictions["y_true"] = y_true
                        model_predictions["dates"] = dates

        # Also check general ML results (for single model training mode)
        ml_results = st.session_state.get("ml_mh_results")
        if ml_results and ml_results.get("per_h"):
            # Extract actual model name from results_df (e.g., "XGBoost", "LSTM", "ANN")
            results_df = ml_results.get("results_df")
            if results_df is not None and not results_df.empty and "Model" in results_df.columns:
                model_type = results_df["Model"].iloc[0]  # Get model name from first row
            else:
                model_type = "ML"  # Fallback

            # Only add if not already detected via individual keys
            if model_type not in trained_models:
                trained_models[model_type] = ml_results
                # Use specific model name for extraction
                source_model = model_type if model_type in ["XGBoost", "LSTM", "ANN"] else "ML Model (XGBoost/LSTM/ANN)"
                y_true, y_pred, dates = _extract_predictions_for_diagnostic(source_model)
                if y_true is not None and y_pred is not None:
                    model_predictions[model_type] = y_pred
                    if "y_true" not in model_predictions:
                        model_predictions["y_true"] = y_true
                        model_predictions["dates"] = dates

        if not trained_models:
            st.warning("""
            ⚠️ **No trained models found!**

            To run the diagnostic pipeline, you need to first train models on this page:

            **Steps:**
            1. Go to **Statistical Models** → Train SARIMAX
            2. Go to **Machine Learning** → Train XGBoost, LSTM, or ANN
            3. Return here to run the diagnostic pipeline

            The pipeline will analyze ALL trained models and recommend the best hybrid approach.
            """)
            return

        # Display found models with optional selection checkboxes
        model_icons = {
            "SARIMAX": "📈",
            "LSTM": "🧠",
            "XGBoost": "🚀",
            "ANN": "🎯",
            "ML": "🤖"
        }

        st.info(f"✅ Found **{len(trained_models)}** trained model(s): {', '.join(trained_models.keys())}")

        # =========================================================================
        # OPTIONAL MODEL SELECTION (Multiselect Dropdown)
        # =========================================================================
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1);
                    border-left: 3px solid #10B981;
                    padding: 0.8rem;
                    border-radius: 8px;
                    margin: 0.5rem 0;'>
            <span style='color: #10B981; font-weight: 600;'>📋 Model Selection</span>
            <span style='color: #9CA3AF; font-size: 0.9rem;'> — Select models to include in the diagnostic analysis</span>
        </div>
        """, unsafe_allow_html=True)

        # Create multiselect dropdown with all trained models as default
        available_model_options = []
        for model_name in trained_models.keys():
            icon = model_icons.get(model_name, "✅")
            available_model_options.append(f"{icon} {model_name}")

        # Multiselect dropdown - default is all models selected
        selected_model_labels = st.multiselect(
            "Select Models for Analysis",
            options=available_model_options,
            default=available_model_options,  # All selected by default
            help="Choose which trained models to include in the diagnostic analysis",
            key="diag_model_multiselect"
        )

        # Extract model names from selected labels (remove emoji prefix)
        selected_names = [label.split(" ", 1)[1] if " " in label else label for label in selected_model_labels]
        selected_models = {name: (name in selected_names) for name in trained_models.keys()}
        num_selected = len(selected_names)

        if num_selected == 0:
            st.warning("⚠️ Please select at least one model to analyze.")
            return
        elif num_selected < len(trained_models):
            st.caption(f"🔍 Will analyze **{num_selected}** of {len(trained_models)} models: {', '.join(selected_names)}")

        # =========================================================================
        # STEP 2: LLM CONFIGURATION (OPTIONAL)
        # =========================================================================
        st.markdown("### Step 2: Configure Analysis Mode")

        use_llm = st.checkbox(
            "🧠 Enable AI-Powered Recommendations (LLM)",
            value=False,
            help="Uses Claude or GPT to provide enhanced natural language recommendations"
        )

        api_key = None
        api_provider = "anthropic"

        if use_llm:
            with st.container():
                llm_col1, llm_col2 = st.columns(2)

                with llm_col1:
                    api_provider = st.selectbox(
                        "Select AI Provider",
                        options=["anthropic", "openai"],
                        format_func=lambda x: "🟣 Anthropic (Claude)" if x == "anthropic" else "🟢 OpenAI (GPT)",
                        help="Choose your preferred LLM provider"
                    )

                with llm_col2:
                    api_key = st.text_input(
                        f"{'Anthropic' if api_provider == 'anthropic' else 'OpenAI'} API Key",
                        type="password",
                        help="Enter your API key for enhanced AI recommendations"
                    )

                if not api_key:
                    st.caption("💡 Without API key, rule-based recommendations will be used (still very useful!)")

        # =========================================================================
        # STEP 3: RUN PIPELINE
        # =========================================================================
        st.markdown("### Step 3: Run Diagnostic Pipeline")

        run_pipeline = st.button(
            "🚀 **Run Full Diagnostic Pipeline**",
            type="primary",
            use_container_width=True,
            help="Analyzes all trained models, compares residuals, and generates recommendations"
        )

        if run_pipeline:
            if "y_true" not in model_predictions or len(model_predictions) <= 1:
                st.error("Could not extract predictions from trained models. Please ensure models have prediction results.")
                return

            with st.spinner(f"🔬 Running diagnostic pipeline... Analyzing {num_selected} model(s)..."):
                try:
                    from app_core.models.ml.hybrid_diagnostics import HybridDiagnosticPipeline

                    # Create pipeline
                    pipeline = HybridDiagnosticPipeline(
                        api_key=api_key if use_llm else None,
                        api_provider=api_provider
                    )

                    # Prepare predictions (exclude y_true, dates, and unselected models)
                    preds_only = {
                        k: v for k, v in model_predictions.items()
                        if k not in ["y_true", "dates"] and selected_models.get(k, False)
                    }

                    # Run pipeline
                    results = pipeline.run_full_pipeline(
                        y_true=model_predictions["y_true"],
                        model_predictions=preds_only,
                        dates=model_predictions.get("dates"),
                        use_llm=use_llm and bool(api_key)
                    )

                    # Store in session state
                    st.session_state["hybrid_pipeline_results"] = results

                    st.success("✅ Diagnostic pipeline complete!")

                except Exception as e:
                    st.error(f"Error running pipeline: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    return

        # =========================================================================
        # STEP 4: DISPLAY RESULTS
        # =========================================================================
        if "hybrid_pipeline_results" in st.session_state:
            results = st.session_state["hybrid_pipeline_results"]

            st.markdown("---")
            st.markdown("## 📊 Pipeline Results")

            # Tab layout for results
            result_tabs = st.tabs([
                "🎯 Recommendations",
                "📈 Model Comparison",
                "📋 Individual Analysis",
                "🔬 RAG Context"
            ])

            # ---- TAB 1: RECOMMENDATIONS ----
            with result_tabs[0]:
                st.markdown("### 🎯 Hybrid Strategy Recommendations")

                recommendations = results.get("recommendations", [])

                if not recommendations:
                    st.info("No recommendations generated. This may indicate insufficient data or model results.")
                else:
                    for i, rec in enumerate(recommendations):
                        # Determine color based on confidence
                        if rec.confidence >= 0.75:
                            color = "#22C55E"  # Green
                            badge = "HIGH CONFIDENCE"
                        elif rec.confidence >= 0.5:
                            color = "#EAB308"  # Yellow
                            badge = "MODERATE"
                        else:
                            color = "#EF4444"  # Red
                            badge = "LOW CONFIDENCE"

                        risk_color = {"low": "#22C55E", "medium": "#EAB308", "high": "#EF4444"}.get(rec.risk_level, "#6366F1")

                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color}15, {color}08);
                                    border-left: 4px solid {color};
                                    padding: 1.25rem;
                                    border-radius: 12px;
                                    margin: 1rem 0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                <h4 style='color: {color}; margin: 0; font-size: 1.2rem;'>
                                    #{i+1} {rec.strategy}
                                </h4>
                                <span style='background: {color}; color: white; padding: 0.25rem 0.75rem;
                                             border-radius: 20px; font-size: 0.75rem; font-weight: 600;'>
                                    {badge} ({rec.confidence*100:.0f}%)
                                </span>
                            </div>
                            <div style='margin: 0.75rem 0;'>
                                <p style='margin: 0.25rem 0; color: #E5E7EB;'>
                                    <strong>Stage 1:</strong> {rec.stage1_model}
                                </p>
                                <p style='margin: 0.25rem 0; color: #E5E7EB;'>
                                    <strong>Stage 2:</strong> {rec.stage2_model}
                                </p>
                            </div>
                            <p style='margin: 0.75rem 0; color: #D1D5DB; font-style: italic;'>
                                {rec.rationale}
                            </p>
                            <div style='display: flex; gap: 2rem; margin-top: 0.75rem;'>
                                <span style='color: #9CA3AF;'>
                                    📈 Expected: <strong style='color: #22C55E;'>{rec.expected_improvement}</strong>
                                </span>
                                <span style='color: #9CA3AF;'>
                                    ⚠️ Risk: <strong style='color: {risk_color};'>{rec.risk_level.upper()}</strong>
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # ---- TAB 2: MODEL COMPARISON ----
            with result_tabs[1]:
                st.markdown("### 📈 Multi-Model Comparison")

                comparison_plot = results.get("comparison_plot")
                if comparison_plot:
                    st.plotly_chart(comparison_plot, use_container_width=True)

                st.caption("""
                **How to read this:**
                - **Top-left (RMSE)**: Lower is better. Best model for Stage 1.
                - **Top-right (Distributions)**: Overlapping = similar errors; Different = diverse models.
                - **Bottom-left (ACF)**: Bars exceeding red lines = learnable structure.
                - **Bottom-right (Entropy)**: Below 0.7 = concentrated frequencies (good for hybrid).
                """)

            # ---- TAB 3: INDIVIDUAL ANALYSIS ----
            with result_tabs[2]:
                st.markdown("### 📋 Individual Model Analysis")

                model_results = results.get("model_results", {})

                if not model_results:
                    st.info("No model results available.")
                else:
                    for model_name, result in model_results.items():
                        icon = "✅" if result.has_learnable_structure else "❌"
                        status = "Learnable Structure" if result.has_learnable_structure else "Random Residuals"

                        with st.expander(f"{icon} **{model_name}** - {status}", expanded=False):
                            # Metrics row
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("MAE", f"{result.mae:.4f}")
                            with m2:
                                st.metric("RMSE", f"{result.rmse:.4f}")
                            with m3:
                                st.metric("MAPE", f"{result.mape:.2f}%")
                            with m4:
                                st.metric("Significant ACF Lags", result.autocorr_significant_lags)

                            # Statistical tests
                            st.markdown("**Statistical Tests:**")
                            test_cols = st.columns(3)
                            with test_cols[0]:
                                ljung_icon = "✅" if result.ljung_box_pvalue < 0.05 else "❌"
                                st.write(f"{ljung_icon} Ljung-Box p-value: `{result.ljung_box_pvalue:.4f}`")
                            with test_cols[1]:
                                runs_icon = "✅" if result.runs_test_pvalue < 0.05 else "❌"
                                st.write(f"{runs_icon} Runs Test p-value: `{result.runs_test_pvalue:.4f}`")
                            with test_cols[2]:
                                entropy_icon = "✅" if result.spectral_entropy < 0.7 else "❌"
                                st.write(f"{entropy_icon} Spectral Entropy: `{result.spectral_entropy:.3f}`")

                            # Recommendation
                            st.info(f"💡 **Analysis:** {result.recommendation}")

            # ---- TAB 4: RAG CONTEXT ----
            with result_tabs[3]:
                st.markdown("### 🔬 RAG Analysis Context")

                context = results.get("context", "No context available.")

                st.markdown("""
                <div style='background: rgba(30, 30, 40, 0.5); padding: 1rem; border-radius: 8px;
                            border: 1px solid rgba(99, 102, 241, 0.3);'>
                    <p style='color: #9CA3AF; margin-bottom: 0.5rem;'>
                        This is the context used for generating recommendations. If LLM mode is enabled,
                        this context is sent to the AI for enhanced analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.code(context, language="text")

                best_model = results.get("best_single_model")
                if best_model:
                    st.success(f"🏆 **Best Single Model:** {best_model}")


def _extract_predictions_for_diagnostic(source_model: str):
    """
    Extract y_true, y_pred, and dates from a trained model's results.

    Uses the per_h structure which contains:
    - SARIMAX: y_test, forecast, fitted, ci_lo, ci_hi
    - ML models: y_test, forecast, datetime

    Returns:
        Tuple of (y_true, y_pred, dates) or (None, None, None) if extraction fails.
    """
    import pandas as pd

    try:
        if source_model == "SARIMAX":
            results = st.session_state.get("sarimax_results", {})
            per_h = results.get("per_h", {})
            if per_h:
                # Get first horizon's results (or aggregate all horizons)
                first_horizon = list(per_h.values())[0]

                # Extract from per_h structure: y_test and forecast keys
                y_test = first_horizon.get("y_test")
                forecast = first_horizon.get("forecast")

                if y_test is not None and forecast is not None:
                    # Convert to numpy arrays if needed
                    if hasattr(y_test, 'values'):
                        y_true = y_test.values
                    else:
                        y_true = np.array(y_test)

                    if hasattr(forecast, 'values'):
                        y_pred = forecast.values
                    else:
                        y_pred = np.array(forecast)

                    # Try to get dates from y_test index
                    dates = None
                    if hasattr(y_test, 'index') and isinstance(y_test.index, pd.DatetimeIndex):
                        dates = y_test.index

                    # Ensure same length (forecast might be shorter due to horizon)
                    min_len = min(len(y_true), len(y_pred))
                    y_true = y_true[:min_len]
                    y_pred = y_pred[:min_len]
                    if dates is not None:
                        dates = dates[:min_len]

                    return y_true, y_pred, dates

        elif source_model in ["ML Model (XGBoost/LSTM/ANN)", "XGBoost", "LSTM", "ANN"]:
            # Try individual model key first, then fall back to general ml_mh_results
            results = {}
            if source_model in ["XGBoost", "LSTM", "ANN"]:
                # First try individual key
                key = f"ml_mh_results_{source_model.lower()}"
                results = st.session_state.get(key, {})
                # Fall back to general ml_mh_results if individual key not found
                if not results or not results.get("per_h"):
                    results = st.session_state.get("ml_mh_results", {})
            else:
                results = st.session_state.get("ml_mh_results", {})

            per_h = results.get("per_h", {})
            if per_h:
                first_horizon = list(per_h.values())[0]

                # Extract from per_h structure: y_test and forecast keys
                y_test = first_horizon.get("y_test")
                forecast = first_horizon.get("forecast")
                datetime_arr = first_horizon.get("datetime")

                if y_test is not None and forecast is not None:
                    # Convert to numpy arrays if needed
                    if hasattr(y_test, 'values'):
                        y_true = y_test.values
                    else:
                        y_true = np.array(y_test) if y_test is not None else None

                    if hasattr(forecast, 'values'):
                        y_pred = forecast.values
                    else:
                        y_pred = np.array(forecast) if forecast is not None else None

                    # Get dates from datetime array or index
                    dates = None
                    if datetime_arr is not None:
                        if hasattr(datetime_arr, 'values'):
                            dates = pd.to_datetime(datetime_arr.values)
                        else:
                            dates = pd.to_datetime(datetime_arr)

                    if y_true is not None and y_pred is not None:
                        # Ensure same length
                        min_len = min(len(y_true), len(y_pred))
                        y_true = y_true[:min_len]
                        y_pred = y_pred[:min_len]
                        if dates is not None:
                            dates = dates[:min_len]

                        return y_true, y_pred, dates

        # If we couldn't find predictions from model results, try to compute from raw data
        # This is a fallback that uses the model's test set predictions
        fused_data = _get_hybrid_dataset()
        if fused_data is not None and not fused_data.empty:
            # Get target column (assume first numeric column)
            numeric_cols = [col for col in fused_data.select_dtypes(include=[np.number]).columns if col != "Date"]
            if numeric_cols:
                target_col = numeric_cols[0]
                y_full = fused_data[target_col].values

                # Simple train/test split for demo
                split_idx = int(len(y_full) * 0.85)
                y_true = y_full[split_idx:]

                # Use naive forecast as "Stage 1" prediction for diagnostic
                y_pred = y_full[split_idx - 1:-1]  # Shifted by 1 (naive forecast)

                dates = None
                if "Date" in fused_data.columns:
                    dates = pd.to_datetime(fused_data["Date"].iloc[split_idx:])

                return y_true, y_pred, dates

    except Exception as e:
        st.warning(f"Could not extract predictions: {str(e)}")

    return None, None, None


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


# -----------------------------------------------------------------------------
# Tab Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📏 Benchmarks",
    "🧮 Machine Learning",
    "🔬 Hyperparameter Tuning",
    "🧬 Hybrid Models",
    "☁️ Cloud Sync"
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
    render_cloud_sync_tab()

# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(7)  # Train Models is page index 7
