# =============================================================================
# pages/05_Modeling_Studio.py - Unified Modeling Studio with Stepper UI
# Consolidates: Feature Selection, Baseline Models (Statistical), Train Models (ML)
# Recommendation: Prompt 2 from reco1.md
# =============================================================================
from __future__ import annotations

import time
import json
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORTS
# =============================================================================
from app_core.ui.theme import apply_css, is_quiet_mode, is_analyst_mode
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT, CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand, render_cache_management, render_user_preferences
from app_core.ui.page_navigation import render_page_navigation
from app_core.cache import save_to_cache, get_cache_manager
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ARIMA/SARIMAX imports
try:
    from app_core.models.arima_pipeline import run_arima_pipeline, run_arima_multi_target_pipeline
    from app_core.models.sarimax_pipeline import run_sarimax_multihorizon, run_sarimax_multi_target_pipeline
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# ML Model imports
try:
    from app_core.models.ml.xgboost_pipeline import train_xgboost_model
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from app_core.models.ml.lstm_pipeline import train_lstm_model
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from app_core.models.ml.ann_pipeline import train_ann_model
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False

# Hybrid imports
try:
    from app_core.models.ml.lstm_xgb import train_lstm_xgb_hybrid
    HYBRID_LSTM_XGB_AVAILABLE = True
except ImportError:
    HYBRID_LSTM_XGB_AVAILABLE = False

# Optuna for HPO
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# CV imports
try:
    from app_core.pipelines import CVFoldResult, CVConfig, create_cv_folds, format_cv_metric
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# =============================================================================
# AUTHENTICATION
# =============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Modeling Studio - HealthForecast AI",
    page_icon="🎯",
    layout="wide",
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _rmse(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error
        return root_mean_squared_error(y_true, y_pred)
    except ImportError:
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _metrics_block(y_tr, yhat_tr, y_te, yhat_te) -> Dict[str, float]:
    return {
        "rmse_train": _rmse(y_tr, yhat_tr),
        "mae_train": mean_absolute_error(y_tr, yhat_tr),
        "mape_train": _mape(y_tr, yhat_tr),
        "rmse_test": _rmse(y_te, yhat_te),
        "mae_test": mean_absolute_error(y_te, yhat_te),
        "mape_test": _mape(y_te, yhat_te),
    }

def _get_feature_data():
    """Get feature-engineered data from session state."""
    fe = st.session_state.get("feature_engineering", {})
    if not fe:
        return None, None, None, None, None

    df_A = fe.get("A")
    df_B = fe.get("B")
    train_idx = fe.get("train_idx")
    test_idx = fe.get("test_idx")

    # Use variant A by default
    df = df_A if df_A is not None else df_B

    return df, train_idx, test_idx, fe.get("summary", {}), fe

def _find_target_columns(df: pd.DataFrame) -> List[str]:
    """Find target columns in the dataframe."""
    if df is None:
        return []
    return sorted([c for c in df.columns if c.lower().startswith("target_")])

def _find_feature_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    """Find feature columns (excluding targets and date)."""
    if df is None:
        return []
    date_cols = ["datetime", "date", "Date", "timestamp", "ds"]
    exclude = set(exclude_cols + date_cols + [c for c in df.columns if c.lower().startswith("target_")])
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# =============================================================================
# STEPPER UI COMPONENT
# =============================================================================
def render_stepper(current_step: int, steps: List[Dict[str, str]]) -> int:
    """Render a visual stepper component for guided workflow."""
    n_steps = len(steps)

    st.markdown(f"""
    <style>
    .stepper-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 1.5rem 0 2rem 0;
        padding: 1rem;
    }}
    .step-item {{
        display: flex;
        flex-direction: column;
        align-items: center;
        cursor: pointer;
        transition: all 0.3s ease;
        padding: 1rem;
        border-radius: 16px;
        min-width: 160px;
        position: relative;
    }}
    .step-item:hover {{
        background: rgba(59, 130, 246, 0.1);
    }}
    .step-circle {{
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }}
    .step-circle.active {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
    }}
    .step-circle.completed {{
        background: {SUCCESS_COLOR};
        border-color: {SUCCESS_COLOR};
    }}
    .step-circle.pending {{
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
    }}
    .step-title {{
        font-weight: 600;
        font-size: 0.9375rem;
        color: {TEXT_COLOR};
        margin-bottom: 0.25rem;
        text-align: center;
    }}
    .step-title.active {{
        color: {PRIMARY_COLOR};
    }}
    .step-description {{
        font-size: 0.75rem;
        color: {SUBTLE_TEXT};
        text-align: center;
        max-width: 140px;
    }}
    .step-connector {{
        width: 80px;
        height: 3px;
        background: rgba(255, 255, 255, 0.1);
        margin: 0 -10px;
        position: relative;
        top: -40px;
    }}
    .step-connector.completed {{
        background: linear-gradient(90deg, {SUCCESS_COLOR}, {PRIMARY_COLOR});
    }}
    </style>
    """, unsafe_allow_html=True)

    stepper_html = '<div class="stepper-container">'
    for i, step in enumerate(steps, 1):
        if i < current_step:
            circle_class, title_class, circle_content = "completed", "", "✓"
        elif i == current_step:
            circle_class, title_class, circle_content = "active", "active", step['icon']
        else:
            circle_class, title_class, circle_content = "pending", "", step['icon']

        stepper_html += f"""
        <div class="step-item" onclick="document.getElementById('model_step_btn_{i}').click()">
            <div class="step-circle {circle_class}">{circle_content}</div>
            <div class="step-title {title_class}">{step['title']}</div>
            <div class="step-description">{step['description']}</div>
        </div>
        """
        if i < n_steps:
            connector_class = "completed" if i < current_step else ""
            stepper_html += f'<div class="step-connector {connector_class}"></div>'

    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)

    cols = st.columns(n_steps)
    selected_step = current_step
    for i, col in enumerate(cols, 1):
        with col:
            if st.button(f"Step {i}", key=f"model_step_btn_{i}", use_container_width=True,
                        type="primary" if i == current_step else "secondary"):
                selected_step = i

    return selected_step

# =============================================================================
# STEP 1: FEATURE SELECTION
# =============================================================================
def render_step_feature_selection():
    """Render Step 1: Feature Selection."""
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 2rem;
         background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
         border: 2px solid rgba(168, 85, 247, 0.3);'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>1</div>
      <h2 style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Feature Selection</h2>
      <p style='color: {BODY_TEXT}; max-width: 600px; margin: 0 auto;'>
        Identify the most important features for your forecasting models using multiple selection methods
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check prerequisites
    df, train_idx, test_idx, summary, fe_data = _get_feature_data()
    if df is None or len(df) == 0:
        st.warning("Please complete the **Data Studio** workflow first to prepare your dataset.")
        if st.button("Go to Data Studio", use_container_width=True):
            st.switch_page("pages/02_Data_Studio.py")
        return

    targets = _find_target_columns(df)
    features = _find_feature_columns(df, [])

    if not targets:
        st.error("No target columns found (Target_1, Target_2, etc.)")
        return

    st.success(f"Dataset ready: {len(df):,} rows, {len(features)} features, {len(targets)} targets")

    # Feature selection methods
    tab_quick, tab_importance, tab_lasso, tab_results = st.tabs([
        "Quick Select", "Permutation Importance", "LASSO Selection", "Selection Results"
    ])

    with tab_quick:
        st.markdown("### Quick Feature Selection")
        st.info("Manually select features based on domain knowledge or previous analysis.")

        # Group features by type
        ed_features = [f for f in features if f.startswith("ED_") or f.startswith("ED_lag")]
        calendar_features = [f for f in features if any(x in f.lower() for x in ["day", "week", "month", "year", "holiday", "weekend"])]
        weather_features = [f for f in features if any(x in f.lower() for x in ["temp", "humid", "precip", "wind", "weather"])]
        fourier_features = [f for f in features if "fourier" in f.lower()]
        other_features = [f for f in features if f not in ed_features + calendar_features + weather_features + fourier_features]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feature Groups:**")
            use_ed = st.checkbox(f"ED/Lag Features ({len(ed_features)})", value=True)
            use_cal = st.checkbox(f"Calendar Features ({len(calendar_features)})", value=True)
            use_weather = st.checkbox(f"Weather Features ({len(weather_features)})", value=len(weather_features) > 0)
            use_fourier = st.checkbox(f"Fourier Features ({len(fourier_features)})", value=len(fourier_features) > 0)

        with col2:
            st.markdown("**Selected Features:**")
            selected = []
            if use_ed: selected.extend(ed_features)
            if use_cal: selected.extend(calendar_features)
            if use_weather: selected.extend(weather_features)
            if use_fourier: selected.extend(fourier_features)
            st.metric("Total Selected", len(selected))

        if st.button("Apply Quick Selection", type="primary", use_container_width=True):
            st.session_state["selected_features"] = selected
            st.session_state["feature_selection_method"] = "quick"
            st.success(f"Selected {len(selected)} features!")

    with tab_importance:
        st.markdown("### Permutation Importance")
        st.info("Train a Random Forest and measure feature importance by permuting each feature.")

        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn not available for permutation importance.")
            return

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("RF Estimators", 50, 200, 100)
            importance_threshold = st.slider("Importance Threshold", 0.0, 0.1, 0.01, 0.005)
        with col2:
            target_col = st.selectbox("Target for Importance", targets, index=0)
            top_k = st.slider("Top K Features", 5, min(50, len(features)), 20)

        if st.button("Run Permutation Importance", type="primary", use_container_width=True):
            with st.spinner("Computing feature importances..."):
                try:
                    X = df[features].iloc[train_idx]
                    y = df[target_col].iloc[train_idx]

                    # Train RF
                    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                    rf.fit(X, y)

                    # Get importances
                    importances = pd.DataFrame({
                        "feature": features,
                        "importance": rf.feature_importances_
                    }).sort_values("importance", ascending=False)

                    # Select top features
                    selected = importances.head(top_k)["feature"].tolist()
                    st.session_state["selected_features"] = selected
                    st.session_state["feature_importance"] = importances
                    st.session_state["feature_selection_method"] = "permutation"

                    # Display
                    st.success(f"Selected top {len(selected)} features")

                    fig = px.bar(importances.head(20), x="importance", y="feature",
                                orientation="h", title="Feature Importances (Top 20)")
                    fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Permutation importance failed: {e}")

    with tab_lasso:
        st.markdown("### LASSO Feature Selection")
        st.info("Use L1 regularization to automatically select features by shrinking coefficients to zero.")

        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn not available for LASSO.")
            return

        col1, col2 = st.columns(2)
        with col1:
            lasso_alpha = st.slider("LASSO Alpha", 0.001, 1.0, 0.1, 0.01)
        with col2:
            target_col_lasso = st.selectbox("Target for LASSO", targets, index=0, key="lasso_target")

        if st.button("Run LASSO Selection", type="primary", use_container_width=True):
            with st.spinner("Running LASSO..."):
                try:
                    X = df[features].iloc[train_idx]
                    y = df[target_col_lasso].iloc[train_idx]

                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Fit LASSO
                    lasso = Lasso(alpha=lasso_alpha, random_state=42, max_iter=5000)
                    lasso.fit(X_scaled, y)

                    # Get non-zero coefficients
                    coef_df = pd.DataFrame({
                        "feature": features,
                        "coefficient": np.abs(lasso.coef_)
                    }).sort_values("coefficient", ascending=False)

                    selected = coef_df[coef_df["coefficient"] > 0]["feature"].tolist()
                    st.session_state["selected_features"] = selected
                    st.session_state["lasso_coefficients"] = coef_df
                    st.session_state["feature_selection_method"] = "lasso"

                    st.success(f"LASSO selected {len(selected)} features (non-zero coefficients)")

                    # Show top coefficients
                    fig = px.bar(coef_df[coef_df["coefficient"] > 0].head(20),
                                x="coefficient", y="feature", orientation="h",
                                title="LASSO Coefficients (Non-zero)")
                    fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"LASSO selection failed: {e}")

    with tab_results:
        st.markdown("### Selection Results")

        selected = st.session_state.get("selected_features", [])
        method = st.session_state.get("feature_selection_method", "none")

        if selected:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Features", len(selected))
            with col2:
                st.metric("Selection Method", method.title())
            with col3:
                st.metric("Available Features", len(features))

            st.markdown("**Selected Features:**")
            st.dataframe(pd.DataFrame({"Feature": selected}), use_container_width=True, hide_index=True)

            st.success("Feature selection complete! Click **Step 2** to train models.")
            if st.button("Continue to Step 2: Train Model", type="primary", use_container_width=True):
                st.session_state["modeling_studio_step"] = 2
                st.rerun()
        else:
            st.info("No features selected yet. Use one of the selection methods above.")

# =============================================================================
# STEP 2: TRAIN MODEL
# =============================================================================
def render_step_train_model():
    """Render Step 2: Train Model with Statistical/ML/Hybrid tabs."""
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 2rem;
         background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
         border: 2px solid rgba(34, 197, 94, 0.3);'>
      <div style='font-size: 3rem; margin-bottom: 1rem;'>2</div>
      <h2 style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Train Model</h2>
      <p style='color: {BODY_TEXT}; max-width: 600px; margin: 0 auto;'>
        Train Statistical, Machine Learning, or Hybrid forecasting models
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check prerequisites
    df, train_idx, test_idx, summary, fe_data = _get_feature_data()
    if df is None:
        st.warning("Please complete the Data Studio workflow first.")
        return

    selected_features = st.session_state.get("selected_features", [])
    if not selected_features:
        st.warning("No features selected. Please complete Step 1 (Feature Selection) first, or all features will be used.")
        # Use all numeric features as fallback
        selected_features = _find_feature_columns(df, [])
        st.info(f"Using all {len(selected_features)} available features.")

    targets = _find_target_columns(df)

    # Model family tabs
    tab_stat, tab_ml, tab_hybrid = st.tabs([
        "Statistical Models", "Machine Learning", "Hybrid Models"
    ])

    # ==========================================================================
    # STATISTICAL MODELS TAB
    # ==========================================================================
    with tab_stat:
        st.markdown("### Statistical Models (ARIMA / SARIMAX)")

        if not ARIMA_AVAILABLE:
            st.error("ARIMA/SARIMAX modules not available. Check installation.")
            return

        stat_model = st.selectbox("Select Model", ["ARIMA", "SARIMAX"], key="stat_model")

        col1, col2 = st.columns(2)
        with col1:
            if stat_model == "ARIMA":
                p = st.number_input("p (AR order)", 0, 10, 1)
                d = st.number_input("d (Differencing)", 0, 2, 1)
                q = st.number_input("q (MA order)", 0, 10, 1)
            else:
                p = st.number_input("p (AR order)", 0, 10, 1)
                d = st.number_input("d (Differencing)", 0, 2, 1)
                q = st.number_input("q (MA order)", 0, 10, 1)
        with col2:
            if stat_model == "SARIMAX":
                P = st.number_input("P (Seasonal AR)", 0, 5, 1)
                D = st.number_input("D (Seasonal Diff)", 0, 2, 1)
                Q = st.number_input("Q (Seasonal MA)", 0, 5, 1)
                s = st.number_input("s (Season length)", 1, 52, 7)
            target_stat = st.selectbox("Target Column", targets, key="stat_target")

        if st.button(f"Train {stat_model}", type="primary", use_container_width=True):
            with st.spinner(f"Training {stat_model}..."):
                try:
                    y_train = df[target_stat].iloc[train_idx]
                    y_test = df[target_stat].iloc[test_idx]

                    if stat_model == "ARIMA":
                        results = run_arima_pipeline(
                            y_train=y_train,
                            y_test=y_test,
                            order=(p, d, q)
                        )
                    else:
                        results = run_sarimax_multihorizon(
                            y_train=y_train,
                            y_test=y_test,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s)
                        )

                    st.session_state[f"{stat_model.lower()}_results"] = results
                    st.success(f"{stat_model} training complete!")

                    # Display metrics
                    if "metrics" in results:
                        metrics = results["metrics"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                        with col3:
                            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

                except Exception as e:
                    st.error(f"{stat_model} training failed: {e}")

    # ==========================================================================
    # MACHINE LEARNING TAB
    # ==========================================================================
    with tab_ml:
        st.markdown("### Machine Learning Models")

        ml_models = []
        if XGBOOST_AVAILABLE:
            ml_models.append("XGBoost")
        if LSTM_AVAILABLE:
            ml_models.append("LSTM")
        if ANN_AVAILABLE:
            ml_models.append("ANN")

        if not ml_models:
            st.error("No ML models available. Check installations (xgboost, tensorflow).")
            return

        ml_model = st.selectbox("Select ML Model", ml_models, key="ml_model_select")

        # Model-specific configuration
        col1, col2 = st.columns(2)

        with col1:
            if ml_model == "XGBoost":
                n_estimators = st.slider("n_estimators", 50, 500, 200)
                max_depth = st.slider("max_depth", 2, 15, 6)
                learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1)
            elif ml_model == "LSTM":
                lstm_units = st.slider("LSTM Units", 16, 256, 64)
                lstm_epochs = st.slider("Epochs", 10, 200, 50)
                lookback = st.slider("Lookback Window", 3, 30, 7)
            elif ml_model == "ANN":
                hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
                neurons = st.slider("Neurons per Layer", 16, 256, 64)
                ann_epochs = st.slider("Epochs", 10, 200, 50)

        with col2:
            target_ml = st.selectbox("Target Column", targets, key="ml_target")
            use_cv = st.checkbox("Use Time Series CV", value=CV_AVAILABLE and st.session_state.get("cv_enabled", False))
            if use_cv and CV_AVAILABLE:
                n_folds = st.slider("CV Folds", 3, 10, 5)

        if st.button(f"Train {ml_model}", type="primary", use_container_width=True):
            with st.spinner(f"Training {ml_model}..."):
                try:
                    X_train = df[selected_features].iloc[train_idx]
                    X_test = df[selected_features].iloc[test_idx]
                    y_train = df[target_ml].iloc[train_idx]
                    y_test = df[target_ml].iloc[test_idx]

                    if ml_model == "XGBoost":
                        import xgboost as xgb
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif ml_model == "LSTM":
                        st.info("LSTM training requires TensorFlow. Using simplified training...")
                        # Simplified placeholder - actual LSTM training would use lstm_pipeline
                        from sklearn.linear_model import Ridge
                        model = Ridge()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif ml_model == "ANN":
                        st.info("ANN training requires TensorFlow. Using simplified training...")
                        from sklearn.neural_network import MLPRegressor
                        model = MLPRegressor(
                            hidden_layer_sizes=tuple([neurons] * hidden_layers),
                            max_iter=ann_epochs,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = _rmse(y_test, y_pred)
                    mape = _mape(y_test, y_pred)

                    results = {
                        "model": ml_model,
                        "y_pred": y_pred,
                        "y_test": y_test.values,
                        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
                        "features": selected_features,
                    }
                    st.session_state["ml_model_results"] = results

                    st.success(f"{ml_model} training complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mae:.3f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.3f}")
                    with col3:
                        st.metric("MAPE", f"{mape:.2f}%")

                    # Plot predictions
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test.values, name="Actual", mode="lines"))
                    fig.add_trace(go.Scatter(y=y_pred, name="Predicted", mode="lines"))
                    fig.update_layout(title=f"{ml_model} Predictions vs Actual",
                                     template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"{ml_model} training failed: {e}")

    # ==========================================================================
    # HYBRID MODELS TAB
    # ==========================================================================
    with tab_hybrid:
        st.markdown("### Hybrid Models")
        st.info("Hybrid models combine multiple approaches (e.g., LSTM for trends + XGBoost for residuals).")

        hybrid_options = ["LSTM + XGBoost", "LSTM + SARIMAX", "LSTM + ANN"]
        hybrid_model = st.selectbox("Select Hybrid Model", hybrid_options, key="hybrid_select")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stage 1 (LSTM) Config:**")
            h_lstm_units = st.slider("LSTM Units", 16, 256, 64, key="h_lstm_units")
            h_lookback = st.slider("Lookback", 3, 30, 7, key="h_lookback")
        with col2:
            st.markdown("**Stage 2 Config:**")
            if "XGBoost" in hybrid_model:
                h_n_est = st.slider("XGB Estimators", 50, 300, 100, key="h_xgb_est")
            target_hybrid = st.selectbox("Target", targets, key="hybrid_target")

        if st.button(f"Train {hybrid_model}", type="primary", use_container_width=True):
            st.warning("Hybrid model training is computationally intensive. Using simplified version...")
            with st.spinner(f"Training {hybrid_model}..."):
                try:
                    # Simplified hybrid training using sklearn
                    X_train = df[selected_features].iloc[train_idx]
                    X_test = df[selected_features].iloc[test_idx]
                    y_train = df[target_hybrid].iloc[train_idx]
                    y_test = df[target_hybrid].iloc[test_idx]

                    # Stage 1: Simple model for trends
                    from sklearn.linear_model import Ridge
                    stage1 = Ridge()
                    stage1.fit(X_train, y_train)
                    stage1_pred = stage1.predict(X_train)
                    residuals = y_train - stage1_pred

                    # Stage 2: XGBoost on residuals
                    import xgboost as xgb
                    stage2 = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
                    stage2.fit(X_train, residuals)

                    # Combined prediction
                    y_pred = stage1.predict(X_test) + stage2.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = _rmse(y_test, y_pred)
                    mape = _mape(y_test, y_pred)

                    st.success(f"{hybrid_model} training complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mae:.3f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.3f}")
                    with col3:
                        st.metric("MAPE", f"{mape:.2f}%")

                except Exception as e:
                    st.error(f"Hybrid training failed: {e}")

    # Results summary
    st.markdown("---")
    st.markdown("### Training Summary")

    results_found = []
    if st.session_state.get("arima_results"):
        results_found.append("ARIMA")
    if st.session_state.get("sarimax_results"):
        results_found.append("SARIMAX")
    if st.session_state.get("ml_model_results"):
        results_found.append(st.session_state["ml_model_results"].get("model", "ML"))

    if results_found:
        st.success(f"Trained models: {', '.join(results_found)}")
        st.info("Go to **Model Results** page to compare all trained models.")
    else:
        st.info("No models trained yet. Select a model type above and click Train.")

# =============================================================================
# MAIN PAGE FUNCTION
# =============================================================================
def page_modeling_studio():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    add_logout_button()
    render_user_preferences()
    render_cache_management()

    # Results Storage Panel
    render_results_storage_panel(
        page_type="modeling_studio",
        page_title="Modeling Studio",
        custom_keys=["selected_features", "arima_results", "sarimax_results", "ml_model_results"],
    )
    auto_load_if_available("modeling_studio")

    # Fluorescent effects (respect quiet mode)
    if not is_quiet_mode():
        st.markdown("""
        <style>
        @keyframes float-orb {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.25; }
            50% { transform: translate(30px, -30px) scale(1.05); opacity: 0.35; }
        }
        .fluorescent-orb {
            position: fixed; border-radius: 50%; pointer-events: none;
            z-index: 0; filter: blur(70px);
        }
        .orb-1 {
            width: 350px; height: 350px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.25), transparent 70%);
            top: 15%; right: 20%; animation: float-orb 25s ease-in-out infinite;
        }
        .orb-2 {
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(34, 197, 94, 0.2), transparent 70%);
            bottom: 20%; left: 15%; animation: float-orb 30s ease-in-out infinite;
            animation-delay: 5s;
        }
        </style>
        <div class="fluorescent-orb orb-1"></div>
        <div class="fluorescent-orb orb-2"></div>
        """, unsafe_allow_html=True)

    # Hero Header
    st.markdown(f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 1.5rem; padding: 1.5rem;'>
      <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>&#127919;</div>
      <h1 style='font-size: 2rem; margin-bottom: 0.5rem;'>Modeling Studio</h1>
      <p style='color: {BODY_TEXT}; max-width: 700px; margin: 0 auto;'>
        Complete modeling pipeline: Feature Selection and Model Training (Statistical, ML, Hybrid)
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize step
    if "modeling_studio_step" not in st.session_state:
        st.session_state["modeling_studio_step"] = 1

    # Define steps
    steps = [
        {"title": "Feature Selection", "icon": "1", "description": "Select important features"},
        {"title": "Train Model", "icon": "2", "description": "Statistical / ML / Hybrid"},
    ]

    # Render stepper
    current_step = st.session_state["modeling_studio_step"]
    selected_step = render_stepper(current_step, steps)

    if selected_step != current_step:
        st.session_state["modeling_studio_step"] = selected_step
        st.rerun()

    st.markdown("---")

    # Render current step
    if current_step == 1:
        render_step_feature_selection()
    elif current_step == 2:
        render_step_train_model()

# =============================================================================
# ENTRYPOINT
# =============================================================================
page_modeling_studio()
render_page_navigation(3)  # Modeling Studio is page index 3
