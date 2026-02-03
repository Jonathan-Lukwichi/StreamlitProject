# =============================================================================
# 00_Auto_Pipeline.py - One-Click Automated Forecasting Pipeline
# =============================================================================
"""
Auto Pipeline - Automated HealthForecast AI Workflow

This page automates the full forecasting workflow in one click:
1. Data Loading (CSV/Excel/Parquet)
2. Data Preparation (process_dataset)
3. Feature Engineering (temporal split)
4. Model Training (ARIMA, SARIMAX, XGBoost, LSTM, ANN, Hybrids)
5. Results Storage (compatible with 09_Model_Results.py)

All results are stored in session state using the same keys as existing pages,
ensuring full compatibility with the Model Results visualization page.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# =============================================================================
# AUTHENTICATION & CORE IMPORTS
# =============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button

require_authentication()
configure_sidebar_navigation()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Auto Pipeline - HealthForecast AI",
    page_icon="🚀",
    layout="wide",
)

# =============================================================================
# UI IMPORTS
# =============================================================================
from app_core.ui.theme import (
    apply_css,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    DANGER_COLOR,
    TEXT_COLOR,
    SUBTLE_TEXT,
    BODY_TEXT,
    CARD_BG,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# =============================================================================
# APPLY STYLES
# =============================================================================
apply_css()
inject_sidebar_style()
render_sidebar_brand()
add_logout_button()

# =============================================================================
# DATA PROCESSING IMPORTS
# =============================================================================
from app_core.data.data_processing import process_dataset
from app_core.data.fusion import fuse_data


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # General
    "train_ratio": 0.8,
    "n_lags": 7,
    "horizons": 7,

    # ARIMA
    "arima_order": (1, 1, 1),
    "arima_auto_select": True,

    # SARIMAX
    "sarimax_order": (1, 1, 1),
    "sarimax_seasonal_order": (1, 1, 1, 7),

    # XGBoost
    "xgb_n_estimators": 300,
    "xgb_max_depth": 6,
    "xgb_eta": 0.1,
    "xgb_min_child_weight": 1,

    # LSTM
    "lookback_window": 14,
    "lstm_hidden_units": 64,
    "lstm_layers": 2,
    "dropout": 0.2,
    "lstm_epochs": 50,
    "lstm_lr": 0.001,
    "batch_size": 32,

    # ANN
    "ann_hidden_layers": 2,
    "ann_neurons": 64,
    "ann_activation": "relu",
    "ann_epochs": 30,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load CSV/Excel/Parquet file into DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load file: {str(e)}")


def load_and_fuse_uploaded_data() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data from session state (uploaded via Upload Data page) and fuse them.
    Returns fused DataFrame and log messages.
    """
    patient_df = st.session_state.get("patient_data")
    weather_df = st.session_state.get("weather_data")
    calendar_df = st.session_state.get("calendar_data")
    reason_df = st.session_state.get("reason_data")

    # Check minimum requirements
    if patient_df is None or patient_df.empty:
        raise Exception("Patient data is required. Please upload data via Upload Data page.")

    if weather_df is None or weather_df.empty:
        raise Exception("Weather data is required. Please upload data via Upload Data page.")

    if calendar_df is None or calendar_df.empty:
        raise Exception("Calendar data is required. Please upload data via Upload Data page.")

    # Fuse datasets using existing fusion function
    fused_df, fusion_log = fuse_data(
        patient_df=patient_df,
        weather_df=weather_df,
        calendar_df=calendar_df,
        reason_df=reason_df if reason_df is not None and not reason_df.empty else None,
    )

    if fused_df is None or fused_df.empty:
        raise Exception(f"Data fusion failed. Check the fusion log: {fusion_log}")

    return fused_df, fusion_log


def prepare_pipeline_data(df_raw: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Process raw data using existing process_dataset function."""
    processed_df, report = process_dataset(
        df_raw,
        n_lags=config["n_lags"],
        use_aggregated_categories=True,
    )
    st.session_state["prepared_data"] = processed_df
    return processed_df, report


def build_features(df: pd.DataFrame, config: Dict) -> Dict:
    """Build feature DataFrame with train/test split."""
    horizons = config["horizons"]
    train_ratio = config["train_ratio"]

    # Identify target and feature columns
    target_cols = [f"Target_{h}" for h in range(1, horizons + 1)]
    date_cols = []

    # Find date column
    for col in df.columns:
        if col.lower() in ['date', 'datetime', 'timestamp', 'ds']:
            date_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)

    # Feature columns = all columns except targets and date
    exclude_cols = set(target_cols + date_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Temporal split (80/20)
    n = len(df)
    n_train = int(n * train_ratio)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)

    fe_result = {
        "A": df.copy(),
        "B": df.copy(),  # Alternative variant (same for simplicity)
        "train_idx": train_idx,
        "test_idx": test_idx,
        "feature_cols": feature_cols,
        "summary": {
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_ratio": train_ratio,
            "n_features": len(feature_cols),
            "variant_used": "Auto Pipeline",
        }
    }
    st.session_state["feature_engineering"] = fe_result
    return fe_result


def run_ml_model_internal(
    model_type: str,
    config: Dict,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """
    Internal ML training for a single target column.
    Simplified version of run_ml_model from 08_Train_Models.py
    """
    try:
        # Find datetime column
        datetime_col = None
        datetime_cols_to_exclude = []

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if datetime_col is None:
                    datetime_col = col
                datetime_cols_to_exclude.append(col)
            elif col.lower() in ['date', 'datetime', 'timestamp', 'time', 'ds']:
                try:
                    pd.to_datetime(df[col])
                    if datetime_col is None:
                        datetime_col = col
                    datetime_cols_to_exclude.append(col)
                except:
                    continue

        # Filter out datetime columns from features
        numeric_feature_cols = [col for col in feature_cols if col not in datetime_cols_to_exclude]

        if not numeric_feature_cols:
            return {"success": False, "error": "No numeric features available"}

        # Extract features and target
        X = df[numeric_feature_cols].values
        y = df[target_col].values

        # Split data using provided indices
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[test_idx], y[test_idx]

        # Extract datetime values for validation set
        if datetime_col:
            val_datetime = df[datetime_col].iloc[test_idx].values
        else:
            val_datetime = test_idx.tolist()

        # Initialize model
        if model_type == "XGBoost":
            from app_core.models.ml.xgboost_pipeline import XGBoostPipeline
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
            model_config = {
                "lookback_window": config.get("lookback_window", 14),
                "lstm_hidden_units": config.get("lstm_hidden_units", 64),
                "lstm_layers": config.get("lstm_layers", 2),
                "dropout": config.get("dropout", 0.2),
                "lstm_epochs": config.get("lstm_epochs", 50),
                "lstm_lr": config.get("lstm_lr", 0.001),
                "batch_size": config.get("batch_size", 32),
            }
            pipeline = LSTMPipeline(model_config)

        elif model_type == "ANN":
            from app_core.models.ml.ann_pipeline import ANNPipeline
            model_config = {
                "ann_hidden_layers": config.get("ann_hidden_layers", 2),
                "ann_neurons": config.get("ann_neurons", 64),
                "ann_activation": config.get("ann_activation", "relu"),
                "dropout": config.get("dropout", 0.2),
                "ann_epochs": config.get("ann_epochs", 30),
            }
            pipeline = ANNPipeline(model_config)

        else:
            return {"success": False, "error": f"Unknown model type: {model_type}"}

        # Build and train model
        n_features = X_train.shape[1]
        pipeline.build_model(n_features)
        pipeline.train(X_train, y_train, X_val, y_val)

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        # Metrics
        train_metrics = pipeline.evaluate(y_train, y_train_pred)
        val_metrics = pipeline.evaluate(y_val, y_val_pred)

        # Predictions DataFrame
        predictions_df = pd.DataFrame({
            "datetime": val_datetime,
            "actual": y_val,
            "predicted": y_val_pred
        })

        return {
            "success": True,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "predictions": predictions_df,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "params": model_config,
            "pipeline": pipeline,
            "X_train": X_train,
            "X_val": X_val,
            "feature_names": numeric_feature_cols,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def train_ml_multihorizon(
    model_type: str,
    config: Dict,
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizons: int = 7,
) -> Dict:
    """
    Train ML model for multi-horizon forecasting (Target_1...Target_7).
    Replicates logic from 08_Train_Models.py run_ml_multihorizon()
    """
    rows = []
    per_h = {}
    successful = []
    errors = {}
    pipelines = {}
    explainability_data = {}

    for h in range(1, horizons + 1):
        target_col = f"Target_{h}"

        if target_col not in df.columns:
            continue

        try:
            result = run_ml_model_internal(
                model_type=model_type,
                config=config,
                df=df,
                target_col=target_col,
                feature_cols=feature_cols,
                train_idx=train_idx,
                test_idx=test_idx,
            )

            if not result.get("success", False):
                errors[h] = result.get("error", "Unknown error")
                continue

            train_metrics = result.get("train_metrics", {})
            val_metrics = result.get("val_metrics", {})
            predictions = result.get("predictions")

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
                "y_train": None,
                "y_test": predictions["actual"].values if predictions is not None else None,
                "forecast": predictions["predicted"].values if predictions is not None else None,
                "datetime": predictions["datetime"].values if predictions is not None else None,
                "params": result.get("params", {}),
            }

            if result.get("pipeline") is not None:
                pipelines[h] = result["pipeline"]
                explainability_data[h] = {
                    "X_train": result.get("X_train"),
                    "X_val": result.get("X_val"),
                    "feature_names": result.get("feature_names", []),
                }

            successful.append(h)

        except Exception as e:
            errors[h] = str(e)
            continue

    # Create results DataFrame
    if rows:
        results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)
    else:
        results_df = pd.DataFrame(columns=[
            "Horizon", "Model", "Train_MAE", "Train_RMSE", "Train_MAPE", "Train_Acc",
            "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_Acc", "Train_N", "Test_N"
        ])

    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": successful,
        "errors": errors,
        "pipelines": pipelines,
        "explainability_data": explainability_data,
    }


def train_arima(df: pd.DataFrame, config: Dict) -> Dict:
    """Train ARIMA multi-horizon model."""
    try:
        from app_core.models.arima_pipeline import run_arima_multi_horizon_pipeline

        result = run_arima_multi_horizon_pipeline(
            df=df,
            order=config.get("arima_order"),
            train_ratio=config["train_ratio"],
            max_horizon=config["horizons"],
            auto_select=config.get("arima_auto_select", True),
            target_col="Target_1",
        )
        st.session_state["arima_mh_results"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_sarimax(df: pd.DataFrame, config: Dict) -> Dict:
    """Train SARIMAX multi-horizon model."""
    try:
        from app_core.models.sarimax_pipeline import run_sarimax_multihorizon

        result = run_sarimax_multihorizon(
            df=df,
            train_ratio=config["train_ratio"],
            max_horizon=config["horizons"],
            order=config.get("sarimax_order", (1, 1, 1)),
            seasonal_order=config.get("sarimax_seasonal_order", (1, 1, 1, 7)),
        )
        st.session_state["sarimax_results"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_xgboost(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train XGBoost multi-horizon model."""
    try:
        result = train_ml_multihorizon(
            model_type="XGBoost",
            config=config,
            df=df,
            feature_cols=fe_result["feature_cols"],
            train_idx=fe_result["train_idx"],
            test_idx=fe_result["test_idx"],
            horizons=config["horizons"],
        )
        st.session_state["ml_mh_results_xgboost"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_lstm(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train LSTM multi-horizon model."""
    try:
        result = train_ml_multihorizon(
            model_type="LSTM",
            config=config,
            df=df,
            feature_cols=fe_result["feature_cols"],
            train_idx=fe_result["train_idx"],
            test_idx=fe_result["test_idx"],
            horizons=config["horizons"],
        )
        st.session_state["ml_mh_results_lstm"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_ann(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train ANN multi-horizon model."""
    try:
        result = train_ml_multihorizon(
            model_type="ANN",
            config=config,
            df=df,
            feature_cols=fe_result["feature_cols"],
            train_idx=fe_result["train_idx"],
            test_idx=fe_result["test_idx"],
            horizons=config["horizons"],
        )
        st.session_state["ml_mh_results_ann"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_lstm_xgb_hybrid(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train LSTM-XGBoost hybrid model."""
    try:
        from app_core.models.ml.lstm_xgb import LSTMXGBHybrid

        hybrid = LSTMXGBHybrid(artifact_dir="pipeline_artifacts/hybrids/lstm_xgb")
        artifacts = hybrid.fit(df, target="Target_1", config=config)

        result = {
            "results_df": pd.DataFrame(),  # Placeholder
            "per_h": {},
            "successful": [1],
            "artifacts": artifacts,
        }
        st.session_state["lstm_xgb_results"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_lstm_sarimax_hybrid(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train LSTM-SARIMAX hybrid model."""
    try:
        from app_core.models.ml.lstm_sarimax import LSTMSARIMAXHybrid

        hybrid = LSTMSARIMAXHybrid(artifact_dir="pipeline_artifacts/hybrids/lstm_sarimax")
        artifacts = hybrid.fit(df, target="Target_1", config=config)

        result = {
            "results_df": pd.DataFrame(),
            "per_h": {},
            "successful": [1],
            "artifacts": artifacts,
        }
        st.session_state["lstm_sarimax_results"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def train_lstm_ann_hybrid(df: pd.DataFrame, fe_result: Dict, config: Dict) -> Dict:
    """Train LSTM-ANN hybrid model."""
    try:
        from app_core.models.ml.lstm_ann import LSTMANNHybrid

        hybrid = LSTMANNHybrid(artifact_dir="pipeline_artifacts/hybrids/lstm_ann")
        artifacts = hybrid.fit(df, target="Target_1", config=config)

        result = {
            "results_df": pd.DataFrame(),
            "per_h": {},
            "successful": [1],
            "artifacts": artifacts,
        }
        st.session_state["lstm_ann_results"] = result
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_summary_metrics(result: Dict) -> Dict:
    """Extract summary metrics from model result."""
    try:
        results_df = result.get("result", {}).get("results_df")
        if results_df is not None and not results_df.empty:
            return {
                "MAE": round(results_df["Test_MAE"].mean(), 2),
                "RMSE": round(results_df["Test_RMSE"].mean(), 2),
                "MAPE": round(results_df["Test_MAPE"].mean(), 2),
                "Accuracy": round(results_df["Test_Acc"].mean(), 2),
            }
    except:
        pass
    return {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A", "Accuracy": "N/A"}


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_hero_header():
    """Render the hero header for the page."""
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
                    border: 1px solid rgba(102, 126, 234, 0.3);
                    border-radius: 16px;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    text-align: center;'>
            <h1 style='font-size: 2.5rem; font-weight: 800; margin: 0;
                       background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       background-clip: text;'>
                AUTO PIPELINE
            </h1>
            <p style='font-size: 1.1rem; color: {SUBTLE_TEXT}; margin: 0.5rem 0 0 0;'>
                One-Click Automated Forecasting Workflow
            </p>
            <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;'>
                <span style='color: {SUCCESS_COLOR}; font-size: 0.9rem;'>
                    Upload Data &rarr; Fuse &rarr; Process &rarr; Train Models &rarr; View Results
                </span>
            </div>
            <p style='font-size: 0.85rem; color: {SUBTLE_TEXT}; margin: 1rem 0 0 0;'>
                Uses data from <strong>Upload Data</strong> page
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_uploaded_data_available() -> Dict:
    """Check if data from Upload Data page is available in session state."""
    patient_df = st.session_state.get("patient_data")
    weather_df = st.session_state.get("weather_data")
    calendar_df = st.session_state.get("calendar_data")
    reason_df = st.session_state.get("reason_data")

    has_patient = patient_df is not None and not getattr(patient_df, "empty", True)
    has_weather = weather_df is not None and not getattr(weather_df, "empty", True)
    has_calendar = calendar_df is not None and not getattr(calendar_df, "empty", True)
    has_reason = reason_df is not None and not getattr(reason_df, "empty", True)

    # Minimum required: patient data
    # Recommended: patient + weather + calendar
    return {
        "patient": has_patient,
        "weather": has_weather,
        "calendar": has_calendar,
        "reason": has_reason,
        "has_minimum": has_patient,
        "has_recommended": has_patient and has_weather and has_calendar,
        "has_all": has_patient and has_weather and has_calendar and has_reason,
        "patient_rows": len(patient_df) if has_patient else 0,
        "weather_rows": len(weather_df) if has_weather else 0,
        "calendar_rows": len(calendar_df) if has_calendar else 0,
        "reason_rows": len(reason_df) if has_reason else 0,
    }


def render_configuration_panel() -> Dict:
    """Render the configuration panel and return selected options."""
    st.markdown(
        f"""
        <div style='background: {CARD_BG};
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;'>
            <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 1rem 0; font-size: 1.2rem;'>
                Configuration
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check for existing data from Upload Data page
    uploaded_status = check_uploaded_data_available()

    # Data Source Selection
    st.markdown(f"**Data Source**", unsafe_allow_html=True)

    # Show status of uploaded data
    if uploaded_status["has_minimum"]:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05));
                        border: 1px solid rgba(34, 197, 94, 0.3);
                        border-radius: 8px;
                        padding: 0.75rem;
                        margin-bottom: 1rem;'>
                <p style='margin: 0; color: {SUCCESS_COLOR}; font-size: 0.9rem;'>
                    Data detected from Upload Data page
                </p>
                <div style='display: flex; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap;'>
                    <span style='font-size: 0.8rem; color: {TEXT_COLOR};'>
                        {"Patient: " + str(uploaded_status['patient_rows']) + " rows" if uploaded_status['patient'] else "Patient: ---"}
                    </span>
                    <span style='font-size: 0.8rem; color: {TEXT_COLOR};'>
                        {"Weather: " + str(uploaded_status['weather_rows']) + " rows" if uploaded_status['weather'] else "Weather: ---"}
                    </span>
                    <span style='font-size: 0.8rem; color: {TEXT_COLOR};'>
                        {"Calendar: " + str(uploaded_status['calendar_rows']) + " rows" if uploaded_status['calendar'] else "Calendar: ---"}
                    </span>
                    <span style='font-size: 0.8rem; color: {TEXT_COLOR};'>
                        {"Reason: " + str(uploaded_status['reason_rows']) + " rows" if uploaded_status['reason'] else "Reason: ---"}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        use_uploaded_data = True
    else:
        # No data detected - warning will be shown in execution panel only
        use_uploaded_data = False

    # No file upload option - data must come from Upload Data page
    uploaded_file = None

    # Parameters
    with st.expander("Pipeline Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider("Train Ratio", 0.6, 0.9, 0.8, 0.05)
            n_lags = st.number_input("Number of Lags", 1, 14, 7)
        with col2:
            horizons = st.number_input("Forecast Horizons", 1, 14, 7)

    # Model Selection
    st.markdown("---")
    st.markdown(f"**Model Selection**", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<small style='color: {SUBTLE_TEXT};'>Baseline</small>", unsafe_allow_html=True)
        train_arima_cb = st.checkbox("ARIMA", value=True, key="cb_arima")
        train_sarimax_cb = st.checkbox("SARIMAX", value=True, key="cb_sarimax")

    with col2:
        st.markdown(f"<small style='color: {SUBTLE_TEXT};'>Machine Learning</small>", unsafe_allow_html=True)
        train_xgboost_cb = st.checkbox("XGBoost", value=True, key="cb_xgboost")
        train_lstm_cb = st.checkbox("LSTM", value=False, key="cb_lstm", help="Slower - Neural network")
        train_ann_cb = st.checkbox("ANN", value=False, key="cb_ann", help="Neural network")

    with col3:
        st.markdown(f"<small style='color: {SUBTLE_TEXT};'>Hybrid Models</small>", unsafe_allow_html=True)
        train_lstm_xgb_cb = st.checkbox("LSTM-XGBoost", value=False, key="cb_lstm_xgb")
        train_lstm_sarimax_cb = st.checkbox("LSTM-SARIMAX", value=False, key="cb_lstm_sarimax")
        train_lstm_ann_cb = st.checkbox("LSTM-ANN", value=False, key="cb_lstm_ann")

    # Build config
    config = {
        **DEFAULT_CONFIG,
        "train_ratio": train_ratio,
        "n_lags": n_lags,
        "horizons": horizons,
    }

    models = {
        "arima": train_arima_cb,
        "sarimax": train_sarimax_cb,
        "xgboost": train_xgboost_cb,
        "lstm": train_lstm_cb,
        "ann": train_ann_cb,
        "lstm_xgb": train_lstm_xgb_cb,
        "lstm_sarimax": train_lstm_sarimax_cb,
        "lstm_ann": train_lstm_ann_cb,
    }

    return {
        "uploaded_file": uploaded_file,
        "use_uploaded_data": use_uploaded_data,
        "uploaded_status": uploaded_status,
        "config": config,
        "models": models,
    }


def render_completion_summary(results: List[Dict]):
    """Display summary of pipeline execution results."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05));
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;'>
            <h2 style='color: {SUCCESS_COLOR}; margin: 0 0 1rem 0;'>
                Pipeline Results Summary
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]

    # Metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Trained", len(successes), delta=None)
    with col2:
        st.metric("Models Failed", len(failures), delta=None if len(failures) == 0 else f"-{len(failures)}")
    with col3:
        st.metric("Total Attempted", len(results))

    # Success table
    if successes:
        st.markdown(f"### Successfully Trained Models")
        success_data = []
        for r in successes:
            metrics = r.get("metrics", {})
            success_data.append({
                "Model": r["model"].upper(),
                "MAE": metrics.get("MAE", "N/A"),
                "RMSE": metrics.get("RMSE", "N/A"),
                "MAPE (%)": metrics.get("MAPE", "N/A"),
                "Accuracy (%)": metrics.get("Accuracy", "N/A"),
            })
        st.dataframe(pd.DataFrame(success_data), use_container_width=True)

    # Failure details
    if failures:
        st.markdown(f"### Failed Models")
        for f in failures:
            with st.expander(f"{f['model'].upper()} - Error"):
                st.error(f["error"])

    # Navigation
    st.markdown("---")
    st.info("View detailed results in the **Model Results** page (09_Model_Results.py)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Model Results", type="primary", use_container_width=True):
            st.switch_page("pages/09_Model_Results.py")
    with col2:
        if st.button("Run Another Pipeline", use_container_width=True):
            # Clear pipeline state to allow re-run
            if "pipeline_complete" in st.session_state:
                del st.session_state["pipeline_complete"]
            st.rerun()


# =============================================================================
# MAIN PIPELINE EXECUTOR
# =============================================================================

def run_full_pipeline(pipeline_config: Dict):
    """Execute the full automated pipeline with progress tracking."""
    config = pipeline_config["config"]
    models = pipeline_config["models"]
    uploaded_file = pipeline_config["uploaded_file"]
    use_uploaded_data = pipeline_config.get("use_uploaded_data", False)

    # Count selected models
    selected_models = [name for name, selected in models.items() if selected]
    total_steps = 3 + len(selected_models)  # Load + Prepare + Features + Models

    # Initialize progress
    progress_bar = st.progress(0)
    status = st.status("Initializing pipeline...", expanded=True)

    results_summary = []
    current_step = 0

    try:
        # =====================================================================
        # Step 1: Load Data
        # =====================================================================
        current_step += 1

        if use_uploaded_data:
            # Use data from Upload Data page (fuse patient, weather, calendar, reason)
            with status:
                st.write("**Step 1**: Loading and fusing data from Upload Data page...")
            df_raw, fusion_log = load_and_fuse_uploaded_data()
            st.write(f"Fused data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
            # Show fusion summary
            success_count = sum(1 for log in fusion_log if log.startswith("✅"))
            st.write(f"Fusion complete: {success_count} datasets merged successfully")
        else:
            # Use uploaded file
            with status:
                st.write("**Step 1**: Loading data file...")
            df_raw = load_uploaded_data(uploaded_file)
            st.write(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

        progress_bar.progress(current_step / total_steps)
        time.sleep(0.3)

        # =====================================================================
        # Step 2: Prepare Data
        # =====================================================================
        current_step += 1
        with status:
            st.write("**Step 2**: Processing and preparing data...")
        df_processed, report = prepare_pipeline_data(df_raw, config)
        st.write(f"Processed: {len(df_processed)} rows, created {len(report.get('created_lags', []))} lag features")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.3)

        # =====================================================================
        # Step 3: Feature Engineering
        # =====================================================================
        current_step += 1
        with status:
            st.write("**Step 3**: Building feature DataFrame and train/test split...")
        fe_result = build_features(df_processed, config)
        st.write(f"Train: {fe_result['summary']['train_size']} samples, Test: {fe_result['summary']['test_size']} samples")
        st.write(f"Features: {fe_result['summary']['n_features']} numeric columns")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.3)

        # =====================================================================
        # Step 4+: Train Selected Models
        # =====================================================================
        model_trainers = {
            "arima": lambda: train_arima(df_processed, config),
            "sarimax": lambda: train_sarimax(df_processed, config),
            "xgboost": lambda: train_xgboost(df_processed, fe_result, config),
            "lstm": lambda: train_lstm(df_processed, fe_result, config),
            "ann": lambda: train_ann(df_processed, fe_result, config),
            "lstm_xgb": lambda: train_lstm_xgb_hybrid(df_processed, fe_result, config),
            "lstm_sarimax": lambda: train_lstm_sarimax_hybrid(df_processed, fe_result, config),
            "lstm_ann": lambda: train_lstm_ann_hybrid(df_processed, fe_result, config),
        }

        for model_name in selected_models:
            current_step += 1
            with status:
                st.write(f"**Step {current_step}**: Training {model_name.upper()}...")

            try:
                start_time = time.time()
                result = model_trainers[model_name]()
                elapsed = time.time() - start_time

                if result.get("success"):
                    metrics = get_summary_metrics(result)
                    st.write(f"{model_name.upper()} trained in {elapsed:.1f}s - MAE: {metrics.get('MAE', 'N/A')}")
                    results_summary.append({
                        "model": model_name,
                        "status": "success",
                        "metrics": metrics,
                        "elapsed": elapsed,
                    })
                else:
                    error_msg = result.get("error", "Unknown error")
                    st.write(f"{model_name.upper()} failed: {error_msg}")
                    results_summary.append({
                        "model": model_name,
                        "status": "failed",
                        "error": error_msg,
                    })

            except Exception as e:
                st.write(f"{model_name.upper()} failed: {str(e)}")
                results_summary.append({
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                })

            progress_bar.progress(current_step / total_steps)

        # =====================================================================
        # Pipeline Complete
        # =====================================================================
        status.update(label="Pipeline Complete!", state="complete")
        st.session_state["pipeline_complete"] = True
        st.session_state["pipeline_results"] = results_summary

    except Exception as e:
        status.update(label="Pipeline Failed", state="error")
        st.error(f"Pipeline failed: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    """Main page function."""
    render_hero_header()

    # Two-column layout
    col_config, col_main = st.columns([1, 2])

    with col_config:
        pipeline_config = render_configuration_panel()

    with col_main:
        st.markdown(
            f"""
            <div style='background: {CARD_BG};
                        border: 1px solid rgba(102, 126, 234, 0.2);
                        border-radius: 12px;
                        padding: 1.5rem;
                        margin-bottom: 1rem;'>
                <h3 style='color: {PRIMARY_COLOR}; margin: 0 0 1rem 0; font-size: 1.2rem;'>
                    Pipeline Execution
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Check if pipeline already completed
        if st.session_state.get("pipeline_complete"):
            results = st.session_state.get("pipeline_results", [])
            render_completion_summary(results)
            return

        # Validation
        use_uploaded_data = pipeline_config.get("use_uploaded_data", False)
        uploaded_status = pipeline_config.get("uploaded_status", {})
        selected_models = [name for name, selected in pipeline_config["models"].items() if selected]

        # Check data availability - data must come from Upload Data page
        if not use_uploaded_data or not uploaded_status.get("has_recommended", False):
            st.warning("Please upload data via **Upload Data** page first. Need Patient, Weather, and Calendar data.")
            if st.button("Go to Upload Data", use_container_width=True):
                st.switch_page("pages/02_Upload_Data.py")
            return

        if not selected_models:
            st.warning("Select at least one model to train.")
            return

        # Show selected configuration
        st.markdown("Using data from **Upload Data** page")
        st.markdown(f"**Datasets:** Patient ({uploaded_status['patient_rows']} rows), "
                   f"Weather ({uploaded_status['weather_rows']} rows), "
                   f"Calendar ({uploaded_status['calendar_rows']} rows)"
                   f"{', Reason (' + str(uploaded_status['reason_rows']) + ' rows)' if uploaded_status.get('reason') else ''}")
        st.markdown(f"**Models to train:** {', '.join([m.upper() for m in selected_models])}")
        st.markdown(f"**Train/Test split:** {int(pipeline_config['config']['train_ratio']*100)}% / {int((1-pipeline_config['config']['train_ratio'])*100)}%")
        st.markdown(f"**Forecast horizons:** {pipeline_config['config']['horizons']} days")

        st.markdown("---")

        # Run button
        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            run_full_pipeline(pipeline_config)
            st.rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
else:
    main()
