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

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.plots import build_multihorizon_results_dashboard

st.set_page_config(
    page_title="Modeling Hub - HealthForecast AI",
    page_icon="🧠",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

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
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>🧠</div>
      <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Modeling Hub</h1>
      <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 auto;'>
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
    "lstm_layers": 1, "lstm_hidden_units": 64, "lstm_epochs": 20, "lstm_lr": 1e-3,
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
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.5rem auto; font-size: 1.8rem;'>📏</div>
          <h1 class='hf-feature-title' style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Benchmarks Dashboard</h1>
          <p class='hf-feature-description' style='font-size: 0.9rem; max-width: 600px; margin: 0 auto;'>
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
                "ci_lo": None,  # ML models don't have confidence intervals by default
                "ci_hi": None,
                "params": result.get("params", {}),
            }

            successful.append(h)

        except Exception as e:
            # Skip failed horizons
            continue

    results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)

    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": successful,
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

    Args:
        model_type: Model name ("LSTM", "ANN", or "XGBoost")
        config: Model configuration dictionary
        df: Training dataframe
        target_col: Target column name
        feature_cols: List of feature column names
        split_ratio: Train/validation split ratio

    Returns:
        dict: Results containing metrics, predictions, and metadata
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

        # Split train/validation (time series - no shuffle)
        split_idx = int(len(df) * split_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Extract datetime values for validation set
        if datetime_col:
            val_datetime = df[datetime_col].iloc[split_idx:].values
        else:
            # Fallback to dataframe index
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
            # TODO: Implement LSTM pipeline
            return {
                "success": False,
                "error": "LSTM pipeline not yet implemented. Coming soon!"
            }

        elif model_type == "ANN":
            # TODO: Implement ANN pipeline
            return {
                "success": False,
                "error": "ANN pipeline not yet implemented. Coming soon!"
            }
        else:
            return {
                "success": False,
                "error": f"Unknown model type: {model_type}"
            }

        # Train model
        start_time = time.time()
        pipeline.build_model()
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

        # Return results
        return {
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
        }

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

def render_ml_multihorizon_results(ml_mh_results: dict, model_name: str):
    """
    Render multi-horizon ML training results with optimized fluorescent design.

    Args:
        ml_mh_results: Multi-horizon results from run_ml_multihorizon()
        model_name: Name of the model (e.g., "XGBoost", "LSTM", "ANN")
    """
    if not ml_mh_results:
        st.warning("No multi-horizon results available.")
        return

    results_df = ml_mh_results.get("results_df")
    successful = ml_mh_results.get("successful", [])

    if results_df is None or results_df.empty or not successful:
        st.error("❌ **Training Failed** - No successful horizons")
        return

    # Premium success banner
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
                    border: 2px solid {SUCCESS_COLOR};
                    padding: 2rem;
                    border-radius: 16px;
                    margin-bottom: 2rem;
                    box-shadow: 0 0 30px rgba(34, 197, 94, 0.3), inset 0 0 20px rgba(34, 197, 94, 0.1);'>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>✅</div>
                <h2 style='margin: 0 0 0.5rem 0; color: {SUCCESS_COLOR}; font-size: 1.5rem; font-weight: 800; text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);'>
                    Multi-Horizon {model_name} Training Complete!
                </h2>
                <p style='margin: 0; color: {TEXT_COLOR}; opacity: 0.95; font-size: 1.1rem;'>
                    🎯 All {len(successful)} forecast horizons successfully trained and validated
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fluorescent KPI Cards - Similar to benchmarks page
    st.markdown(
        f"""
        <div style='text-align: center; margin: 2rem 0 1rem 0;'>
            <h3 style='font-size: 1.3rem; font-weight: 700; color: {PRIMARY_COLOR};
                       text-shadow: 0 0 20px rgba(59, 130, 246, 0.6); margin: 0;'>
                🏆 Average Performance Metrics
            </h3>
            <p style='color: {SUBTLE_TEXT}; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
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
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            _create_kpi_indicator("RMSE", avg_test_rmse, "", SECONDARY_COLOR),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            _create_kpi_indicator("MAPE", avg_test_mape, "%", WARNING_COLOR),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            _create_kpi_indicator("Accuracy", avg_test_acc, "%", SUCCESS_COLOR),
            use_container_width=True
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
            if "fig_err_box" in figs and hasattr(figs["fig_err_box"], 'savefig'):
                st.markdown("**📦 Error Distribution (Box Plot)**")
                st.pyplot(figs["fig_err_box"])
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            # Error KDE (Full Width)
            if "fig_err_kde" in figs and hasattr(figs["fig_err_kde"], 'savefig'):
                st.markdown("**🌊 Error Density Distribution (KDE)**")
                st.pyplot(figs["fig_err_kde"])

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

    # Header (preserved design - exact same styling)
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧮</div>
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Machine Learning</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Configure and train advanced machine learning models including LSTM, ANN, and XGBoost for sophisticated forecasting
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model Selection (Generic - replaces 15 lines with 3)
    st.subheader("Model selection")
    selected_model = MLUIComponents.render_model_selector(cfg.get("ml_choice", "XGBoost"))
    cfg["ml_choice"] = selected_model

    # 2. Dataset Selection (from Feature Selection results)
    datasets, metadata = get_feature_selection_datasets()
    selected_df = None  # Initialize for later use in target/feature selection

    if datasets:
        st.subheader("📊 Dataset Selection")
        dataset_options = list(datasets.keys())
        selected_dataset_name = st.selectbox(
            "Select feature-selected dataset",
            options=dataset_options,
            help="Choose a dataset from Automated Feature Selection results"
        )
        selected_df = datasets[selected_dataset_name]

        # Show dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features", len(selected_df.columns) - 7)  # Exclude Target_1..7
        with col2:
            st.metric("Samples", len(selected_df))
        with col3:
            st.metric("Variant", metadata.get("variant", "Unknown"))

        # Preview
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(selected_df.head(10), use_container_width=True)

        # Store selected dataset for training
        cfg["selected_dataset"] = selected_dataset_name
        cfg["training_data"] = selected_df
    else:
        st.warning("⚠️ No feature selection results found. Please run **Automated Feature Selection** (Page 07) first.")
        data_source_placeholder()

    # 3. Manual Configuration
    st.subheader("Model Configuration")
    st.caption("Configure hyperparameters manually. For automated optimization, use the **Hyperparameter Tuning** tab.")

    # Manual params (model-specific, but rendered generically)
    manual_params = MLUIComponents.render_manual_params(selected_model, cfg)
    cfg.update(manual_params)

    # Train/test split ratio
    cfg["split_ratio"] = st.slider(
        "Train/Validation split ratio",
        0.50, 0.95,
        float(cfg.get("split_ratio", 0.80)),
        0.01,
        key="ml_split_ratio",
        help="Proportion of data to use for training"
    )

    # 5. Forecast Horizons Selection
    st.subheader("⏰ Forecast Horizons")
    cfg["ml_horizons"] = st.slider(
        "Number of forecast horizons",
        min_value=1,
        max_value=7,
        value=cfg.get("ml_horizons", 7),
        help="Train separate models for Target_1 through Target_N (multi-horizon forecasting)"
    )

    # 6. Features Selection (based on selected dataset)
    # Note: No target selection needed for multi-horizon - we train for all Target_1...Target_H
    st.subheader("🎯 Features Selection")

    if selected_df is not None and not selected_df.empty:
        # Feature columns are all non-target columns
        all_columns = selected_df.columns.tolist()
        feature_columns = [col for col in all_columns if not col.startswith("Target_")]

        if feature_columns:
            current_features = cfg.get("ml_feature_cols", feature_columns)
            current_features = [f for f in current_features if f in feature_columns]
            if not current_features:
                current_features = feature_columns

            selected_features = st.multiselect(
                "Feature columns",
                options=feature_columns,
                default=current_features,
                help="Select features to use for training (all selected by default)",
                key="ml_tf_feats"
            )
            cfg["ml_feature_cols"] = selected_features

            st.caption(f"✓ {len(selected_features)} features selected | Training {cfg.get('ml_horizons', 7)} horizons (Target_1...Target_{cfg.get('ml_horizons', 7)})")
        else:
            st.warning("⚠️ No feature columns found in dataset")
            cfg["ml_feature_cols"] = []
    else:
        st.caption("Select a dataset above to configure features")

    # 7. Run Model Button
    st.divider()

    # Check if all required data is available
    can_run = all([
        selected_df is not None,
        cfg.get("ml_feature_cols") and len(cfg.get("ml_feature_cols", [])) > 0
    ])

    if can_run:
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

        # Display results if available
        if "ml_mh_results" in st.session_state:
            st.divider()
            render_ml_multihorizon_results(st.session_state["ml_mh_results"], cfg['ml_choice'])
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
# HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------
def page_hyperparameter_tuning():
    """Dedicated hyperparameter optimization workspace."""
    from app_core.ui.ml_components import MLUIComponents

    # Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🔬</div>
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Hyperparameter Tuning</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Automated hyperparameter optimization using Grid Search, Random Search, or Bayesian Optimization
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if model and dataset are configured
    model_selected = cfg.get("ml_choice")
    dataset_available = cfg.get("training_data") is not None
    target_selected = cfg.get("ml_target_col") is not None
    features_selected = cfg.get("ml_feature_cols") and len(cfg.get("ml_feature_cols", [])) > 0

    if not all([model_selected, dataset_available, target_selected, features_selected]):
        st.warning(
            "⚠️ **Configuration Required**\n\n"
            "Please configure the following in the **Machine Learning** tab first:\n"
            f"- {'✅' if model_selected else '❌'} Model selection\n"
            f"- {'✅' if dataset_available else '❌'} Dataset selection\n"
            f"- {'✅' if target_selected else '❌'} Target variable\n"
            f"- {'✅' if features_selected else '❌'} Feature selection"
        )
        st.info("👈 Navigate to the **🧮 Machine Learning** tab to complete the configuration.")
        return

    # Display current configuration summary
    with st.expander("📋 Current Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", cfg.get("ml_choice", "N/A"))
        with col2:
            st.metric("Dataset", cfg.get("selected_dataset", "N/A"))
        with col3:
            st.metric("Target", cfg.get("ml_target_col", "N/A"))

        st.caption(f"Features: {', '.join(cfg.get('ml_feature_cols', [])[:5])}{'...' if len(cfg.get('ml_feature_cols', [])) > 5 else ''}")

    # 1. Optimization Settings
    st.subheader("🎯 Optimization Settings")

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

    # 3. Search Space / Bounds
    st.subheader("🔍 Search Space Configuration")
    st.caption("Define the hyperparameter ranges to explore")

    selected_model = cfg.get("ml_choice")
    auto_params = MLUIComponents.render_automatic_params(selected_model, cfg)
    cfg.update(auto_params)

    # 4. Run Optimization
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        run_optimization = st.button(
            f"🚀 Run {search_method}",
            type="primary",
            use_container_width=True,
            help="Start hyperparameter optimization (placeholder)"
        )

    if run_optimization:
        st.info(
            f"🔄 **Optimization Started**\n\n"
            f"- Method: {search_method}\n"
            f"- Metric: {optimization_metric}\n"
            f"- CV Folds: {n_splits}\n"
            f"- Model: {selected_model}\n\n"
            "This is a placeholder. Full implementation will include:\n"
            "- TimeSeriesSplit CV\n"
            "- Parallel training\n"
            "- Progress tracking\n"
            "- Results storage in session state"
        )

    # 5. Results Comparison (placeholder for after optimization)
    st.divider()
    st.subheader("📊 Results Comparison")

    # Check if optimization results exist
    tuning_results = st.session_state.get("tuning_results")

    if tuning_results is None:
        st.info(
            "📈 **Results will appear here after optimization**\n\n"
            "After running optimization, you'll see:\n"
            "- ✅ Baseline vs Optimized metrics comparison\n"
            "- 📋 Parameter changes table\n"
            "- 📊 Cross-validation scores distribution\n"
            "- ⏱️ Optimization time and trial count\n"
            "- 🎯 Best parameters found"
        )
    else:
        # Display results (will be implemented when training logic is added)
        st.success("✅ Optimization completed!")

        # Metrics comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline Model**")
            # Placeholder metrics
        with col2:
            st.markdown("**Optimized Model**")
            # Placeholder metrics

    # Debug panel
    debug_panel()

# -----------------------------------------------------------------------------
# HYBRID MODELS (LSTM-XGB / LSTM-ANN / LSTM-SARIMAX)
# -----------------------------------------------------------------------------
def page_hybrid():
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧬</div>
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Hybrid Models</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Explore and configure powerful hybrid forecasting architectures that combine multiple modeling approaches
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model selection header + dropdown
    st.subheader("Model selection")
    hyb_options = ["LSTM-XGBoost", "LSTM-ANN", "LSTM-SARIMAX"]
    cfg["hybrid_choice"] = st.selectbox(
        "Choose a hybrid model",
        options=hyb_options,
        index=hyb_options.index(cfg.get("hybrid_choice","LSTM-XGBoost")),
        help="Pick a hybrid architecture to configure."
    )

    data_source_placeholder()

    # Parameter mode
    st.subheader("Parameter mode")
    mode = st.radio(
        "How do you want to set parameters?",
        options=["Automatic (optimize)", "Manual"], horizontal=True,
        index=0 if (
            (cfg["hybrid_choice"] == "LSTM-XGBoost"  and cfg.get("hyb_lstm_xgb_param_mode","Automatic")     == "Automatic") or
            (cfg["hybrid_choice"] == "LSTM-ANN"      and cfg.get("hyb_lstm_ann_param_mode","Automatic")     == "Automatic") or
            (cfg["hybrid_choice"] == "LSTM-SARIMAX"  and cfg.get("hyb_lstm_sarimax_param_mode","Automatic") == "Automatic")
        ) else 1,
        key="hyb_mode_radio"
    )

    # ---------------- LSTM-XGBoost ----------------
    if cfg["hybrid_choice"] == "LSTM-XGBoost":
        cfg["hyb_lstm_xgb_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• Phase 1: LSTM → predictions/residuals")
            st.write("• Phase 2: XGBoost on residuals or stacked features")

        if cfg["hyb_lstm_xgb_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb_lstm_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb_lstm_search_method","Grid (bounded)")),
                    key="hyb_lstm_xgb_search_method_select"
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01, key="hyb_lstm_xgb_split_ratio")

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb_lstm_max_units",256)), step=8))
            with c2:
                cfg["hyb_xgb_estimators_min"] = int(st.number_input("XGB min n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators_min",100)), step=50))
                cfg["hyb_xgb_estimators_max"] = int(st.number_input("XGB max n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators_max",1000)), step=50))
            with c3:
                cfg["hyb_xgb_depth_min"] = int(st.number_input("XGB min max_depth", 1, 30, int(cfg.get("hyb_xgb_depth_min",3))))
                cfg["hyb_xgb_depth_max"] = int(st.number_input("XGB max max_depth", 1, 30, int(cfg.get("hyb_xgb_depth_max",12))))
            st.slider("XGB eta range (placeholder)",
                      float(cfg.get("hyb_xgb_eta_min",0.01)),
                      float(cfg.get("hyb_xgb_eta_max",0.3)),
                      float(cfg.get("hyb_xgb_eta",0.1)),
                      key="hyb_xgb_eta_range")
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb_lstm_units",64)), step=8))
            with c2: cfg["hyb_xgb_estimators"] = int(st.number_input("XGB n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators",300)), step=50))
            with c3: cfg["hyb_xgb_depth"] = int(st.number_input("XGB max_depth", 1, 30, int(cfg.get("hyb_xgb_depth",6))))
            cfg["hyb_xgb_eta"] = float(st.number_input("XGB learning rate (eta)", 0.001, 1.0, float(cfg.get("hyb_xgb_eta",0.1))))

    # ---------------- LSTM-ANN ----------------
    elif cfg["hybrid_choice"] == "LSTM-ANN":
        cfg["hyb_lstm_ann_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• LSTM sequence embedding → ANN dense head")

        if cfg["hyb_lstm_ann_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb2_lstm_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb2_lstm_search_method","Grid (bounded)")),
                    key="hyb_lstm_ann_search_method_select"
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01, key="hyb_lstm_ann_split_ratio")

            st.markdown("**Bounds (placeholders)**")
            c1, c2 = st.columns(2)
            with c1: cfg["hyb2_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb2_lstm_max_units",256)), step=8))
            with c2:
                cfg["hyb2_ann_max_layers"]  = int(st.number_input("ANN max layers", 1, 8, int(cfg.get("hyb2_ann_max_layers",4))))
                cfg["hyb2_ann_max_neurons"] = int(st.number_input("ANN max neurons", 8, 1024, int(cfg.get("hyb2_ann_max_neurons",256)), step=8))
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb2_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb2_lstm_units",64)), step=8))
            with c2: cfg["hyb2_ann_layers"] = int(st.number_input("ANN hidden layers", 1, 8, int(cfg.get("hyb2_ann_layers",2))))
            with c3: cfg["hyb2_ann_neurons"] = int(st.number_input("ANN neurons per layer", 4, 1024, int(cfg.get("hyb2_ann_neurons",64)), step=4))
            cfg["hyb2_ann_activation"] = st.selectbox(
                "ANN activation", ["relu", "tanh", "sigmoid"],
                index=["relu","tanh","sigmoid"].index(cfg.get("hyb2_ann_activation","relu"))
            )

    # ---------------- LSTM-SARIMAX ----------------
    else:
        cfg["hyb_lstm_sarimax_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• SARIMAX: seasonality & exogenous; LSTM: nonlinear patterns")
            st.write("• Combine via residuals or stacked features")

        if cfg["hyb_lstm_sarimax_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb3_lstm_search_method"] = st.selectbox(
                    "LSTM search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb3_lstm_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01, key="hyb_lstm_sarimax_split_ratio")

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["hyb3_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb3_lstm_max_units",256)), step=8))
            with c2: cfg["hyb3_max_p"] = int(st.number_input("SARIMAX max p", 0, 10, int(cfg.get("hyb3_max_p",5))))
            with c3: cfg["hyb3_max_d"] = int(st.number_input("SARIMAX max d", 0, 3, int(cfg.get("hyb3_max_d",2))))
            with c4: cfg["hyb3_max_q"] = int(st.number_input("SARIMAX max q", 0, 10, int(cfg.get("hyb3_max_q",5))))

            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["hyb3_max_P"] = int(st.number_input("SARIMAX max P", 0, 5, int(cfg.get("hyb3_max_P",2))))
            with c2: cfg["hyb3_max_D"] = int(st.number_input("SARIMAX max D", 0, 2, int(cfg.get("hyb3_max_D",1))))
            with c3: cfg["hyb3_max_Q"] = int(st.number_input("SARIMAX max Q", 0, 5, int(cfg.get("hyb3_max_Q",2))))
            with c4: cfg["hyb3_seasonal_period"] = int(st.number_input("Seasonal period (s)", 1, 365, int(cfg.get("hyb3_seasonal_period",7))))

            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="hyb3_exog_auto")
            cfg["hyb3_exogenous_cols"] = st.session_state.get("hyb3_exog_auto", [])
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2 = st.columns(2)
            with c1: cfg["hyb3_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb3_lstm_units",64)), step=8))
            with c2:
                st.markdown("**SARIMAX orders**")
                r1, r2, r3 = st.columns(3)
                with r1:
                    p = st.number_input("p", 0, 10, int(cfg.get("hyb3_sarimax_order",(1,0,0))[0]), key="hyb3_p")
                with r2:
                    d = st.number_input("d", 0, 3, int(cfg.get("hyb3_sarimax_order",(1,0,0))[1]), key="hyb3_d")
                with r3:
                    q = st.number_input("q", 0, 10, int(cfg.get("hyb3_sarimax_order",(1,0,0))[2]), key="hyb3_q")
                cfg["hyb3_sarimax_order"] = (int(p), int(d), int(q))

            s1, s2, s3, s4 = st.columns(4)
            so = cfg.get("hyb3_sarimax_seasonal_order", (0,0,0,0))
            with s1: P = st.number_input("P", 0, 5, int(so[0]), key="hyb3_P")
            with s2: D = st.number_input("D", 0, 2, int(so[1]), key="hyb3_D")
            with s3: Q = st.number_input("Q", 0, 5, int(so[2]), key="hyb3_Q")
            with s4:
                s = st.number_input("s (period)", 1, 365, int(cfg.get("hyb3_seasonal_period",7)), key="hyb3_seasonal_period")
            cfg["hyb3_sarimax_seasonal_order"] = (int(P), int(D), int(Q), int(s))
            cfg["hyb3_seasonal_period"] = int(s)

            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="hyb3_exog_manual")
            cfg["hyb3_exogenous_cols"] = st.session_state.get("hyb3_exog_manual", [])

    target_feature_placeholders(prefix="hybrid")
    c1, c2 = st.columns(2)
    with c1: st.button(f"🚀 Train {cfg['hybrid_choice']} (placeholder)", key="hybrid_train_button")
    with c2: st.button(f"🧪 Test {cfg['hybrid_choice']} (placeholder)", key="hybrid_test_button")
    debug_panel()

# -----------------------------------------------------------------------------
# Tab Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📏 Benchmarks",
    "🧮 Machine Learning",
    "🔬 Hyperparameter Tuning",
    "🧬 Hybrid Models"
])

with tab1:
    page_benchmarks()

with tab2:
    page_ml()

with tab3:
    page_hyperparameter_tuning()

with tab4:
    page_hybrid()
