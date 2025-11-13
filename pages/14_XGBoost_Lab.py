from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import traceback
import random

try:
    import optuna
    _optuna_available = True
except ImportError:
    _optuna_available = False

from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# --- Core App Imports ---
try:
    from app_core.ui.theme import (
        apply_css,
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
        DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
    )
    from app_core.ui.components import header
    from app_core.ui.model_config_panel import GlobalMLConfig, render_global_ml_config
    from app_core.models.ml import TrainConfig, train_xgb
    _is_core_available = True
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")
    # Fallbacks if run as a standalone script
    def apply_css(): pass
    def header(title, subtitle, icon=""): st.title(f"{icon} {title}")
    def render_global_ml_config(d): return None
    def train_xgb(df, cfg, params): raise NotImplementedError("train_xgb not found")
    TrainConfig = dict
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT = ("#000",)*7
    _is_core_available = False

# --- Helper Functions ---

def _collect_available_datasets():
    """Collect all available datasets from session state."""
    datasets = {}
    if "df_clean" in st.session_state and isinstance(st.session_state.df_clean, pd.DataFrame):
        datasets["Original: df_clean"] = st.session_state.df_clean
    if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
        datasets["Original: df"] = st.session_state.df
    if "processed_df" in st.session_state and isinstance(st.session_state.processed_df, pd.DataFrame):
        datasets["Original: processed_df"] = st.session_state.processed_df

    fe = st.session_state.get("feature_engineering", {})
    if isinstance(fe, dict):
        if "A" in fe and isinstance(fe["A"], pd.DataFrame):
            datasets["Feature Engineering: Variant A (Decomposition)"] = fe["A"]
        if "B" in fe and isinstance(fe["B"], pd.DataFrame):
            datasets["Feature Engineering: Variant B (Cyclical)"] = fe["B"]

    fs = st.session_state.get("feature_selection", {})
    if isinstance(fs, dict) and "filtered_datasets" in fs and isinstance(fs["filtered_datasets"], dict):
        filtered = fs["filtered_datasets"]
        method_names = {0: "Baseline (No Selection)", 1: "Permutation Importance", 2: "Lasso (Linear L1)", 3: "Gradient Boosting"}
        for idx, df_data in filtered.items():
            if isinstance(df_data, pd.DataFrame):
                method_name = method_names.get(idx, f"Method {idx}")
                datasets[f"Feature Selection: {method_name}"] = df_data
    return datasets

def _detect_date_col(df: pd.DataFrame) -> str | None:
    """Detects a date-like column in the dataframe."""
    common_names = ["Date", "date", "DATE", "ds", "DS", "timestamp", "Timestamp", "time", "datetime"]
    for name in common_names:
        if name in df.columns:
            return name
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    return None

def create_kpi_card_html(title: str, value, suffix: str = "", color: str = "#FFFFFF"):
    """Creates a styled HTML string for a KPI card."""
    try:
        is_num = isinstance(value, (int, float, np.floating)) and np.isfinite(value)
    except Exception:
        is_num = False

    if is_num:
        if suffix == "%":
            display_val = f"{value:.2f}{suffix}"
        else:
            display_val = f"{value:.3f}"
    else:
        display_val = "—"

    return f"""
    <div class="kpi-card" style="border-top: 4px solid {color};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {color};">{display_val}</div>
    </div>
    """

def _rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def _mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0))

def _run_xgb_grid_search(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    date_col: str | None,
    n_splits: int,
    param_grid: dict[str, list],
    metric_to_optimize: str,
):
    """
    Simple time-series-aware grid search for XGBoost.
    """
    if df is None or df.empty:
        return None, pd.DataFrame()

    work = df.copy()
    if date_col is not None and date_col in work.columns:
        work = work.sort_values(by=date_col)
    elif isinstance(work.index, pd.DatetimeIndex):
        work = work.sort_index()

    X = work[feature_cols].values
    y = work[target_col].values

    tscv = TimeSeriesSplit(n_splits=max(2, int(n_splits)))

    keys = list(param_grid.keys())
    combos = [dict(zip(keys, values)) for values in product(*[param_grid[k] for k in keys])]

    records = []
    best_params = None
    best_score = None

    for combo in combos:
        model = XGBRegressor(**combo, objective="reg:squarederror", n_jobs=0, random_state=42)
        fold_metrics = {"rmse": [], "mae": [], "mape": [], "acc": []}

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            rmse = _rmse(y_val, y_pred)
            mae = _mae(y_val, y_pred)
            mape = _mape(y_val, y_pred)
            acc = 100.0 - mape if np.isfinite(mape) else np.nan

            fold_metrics["rmse"].append(rmse)
            fold_metrics["mae"].append(mae)
            fold_metrics["mape"].append(mape)
            fold_metrics["acc"].append(acc)

        mean_rmse = float(np.nanmean(fold_metrics["rmse"]))
        mean_mae = float(np.nanmean(fold_metrics["mae"]))
        mean_mape = float(np.nanmean(fold_metrics["mape"]))
        mean_acc = float(np.nanmean(fold_metrics["acc"]))

        record = {**combo, "RMSE": mean_rmse, "MAE": mean_mae, "MAPE": mean_mape, "Accuracy": mean_acc}
        records.append(record)

        metric = metric_to_optimize.lower()
        score = record.get(metric.upper())
        
        is_better = False
        if score is not None:
            if best_score is None:
                is_better = True
            elif metric == "accuracy" and score > best_score:
                is_better = True
            elif metric != "accuracy" and score < best_score:
                is_better = True
        
        if is_better:
            best_score = score
            best_params = combo

    results_df = pd.DataFrame(records)
    return best_params, results_df

def _run_xgb_random_search(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    date_col: str | None,
    n_splits: int,
    param_grid: dict[str, list],
    metric_to_optimize: str,
    n_iter: int = 20,
):
    """
    Random Search for XGBoost using TimeSeriesSplit.
    """
    if df is None or df.empty:
        return None, pd.DataFrame()

    keys = list(param_grid.keys())
    all_combos = [dict(zip(keys, values)) for values in product(*[param_grid[k] for k in keys])]

    if not all_combos:
        return None, pd.DataFrame()

    n_iter = min(n_iter, len(all_combos))
    sampled_combos = random.sample(all_combos, n_iter)

    return _run_xgb_grid_search(df, target_col, feature_cols, date_col, n_splits, {"combos": sampled_combos}, metric_to_optimize)

def _run_xgb_bayesian_opt(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    date_col: str | None,
    n_splits: int,
    bounds: dict[str, tuple],
    metric_to_optimize: str,
    n_trials: int = 20,
):
    """
    Bayesian optimization for XGBoost using Optuna.
    """
    if not _optuna_available:
        return None, pd.DataFrame()
    if df is None or df.empty:
        return None, pd.DataFrame()

    work = df.copy()
    if date_col is not None and date_col in work.columns:
        work = work.sort_values(by=date_col)
    elif isinstance(work.index, pd.DatetimeIndex):
        work = work.sort_index()

    X = work[feature_cols].values
    y = work[target_col].values

    tscv = TimeSeriesSplit(n_splits=max(2, int(n_splits)))
    metric = metric_to_optimize.lower()

    def objective(trial: "optuna.trial.Trial"):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *bounds["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *bounds["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *bounds["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *bounds["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *bounds["colsample_bytree"]),
        }
        model = XGBRegressor(**params, objective="reg:squarederror", n_jobs=0, random_state=42)
        
        fold_metrics = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            if metric == "accuracy":
                mape = _mape(y_val, y_pred)
                fold_metrics.append(100.0 - mape if np.isfinite(mape) else np.nan)
            elif metric == "mae":
                fold_metrics.append(_mae(y_val, y_pred))
            elif metric == "mape":
                fold_metrics.append(_mape(y_val, y_pred))
            else: # rmse
                fold_metrics.append(_rmse(y_val, y_pred))
        
        return float(np.nanmean(fold_metrics))

    direction = "maximize" if metric == "accuracy" else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    
    # Format results to be consistent
    params_df = results_df['params'].apply(pd.Series)
    results_df = pd.concat([results_df.drop('params', axis=1), params_df], axis=1)
    results_df = results_df.rename(columns={"value": metric_to_optimize.upper()})

    return best_params, results_df

# --- Page Logic ---

st.set_page_config(page_title="XGBoost Lab | HealthForecast AI", page_icon="🧠", layout="wide")
apply_css()

header("XGBoost Lab", "Baseline XGBoost training playground", icon="🧠")

if not _is_core_available:
    st.error("Core application modules could not be imported. This page requires `app_core`.")
    st.stop()

available_datasets = _collect_available_datasets()
if not available_datasets:
    st.warning("⚠️ No datasets found!")
    st.info("""
    **To get started:**
    1. Upload data in the **Data Hub**.
    2. Prepare features in the **Data Preparation Studio**.
    3. Optionally, run **Advanced Feature Engineering** or **Automated Feature Selection**.
    Then return here to train models.
    """)
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Step 1: Global Configuration",
    "Step 2: Baseline XGBoost Training",
    "Step 3: Hyperparameter Tuning"
])

cfg_global = None
with tab1:
    st.markdown("### ⚙️ Step 1 — Global Configuration")
    st.write("Select the dataset, target, features, and cross-validation strategy. This configuration will apply to both baseline training and hyperparameter tuning.")
    cfg_global = render_global_ml_config(available_datasets)

    if cfg_global:
        if len(cfg_global.datasets_to_train) == 1:
            with st.expander("📋 Dataset Preview", expanded=True):
                st.dataframe(cfg_global.reference_df.head(10), use_container_width=True)
                st.caption(f"**Shape:** {cfg_global.reference_df.shape[0]} rows × {cfg_global.reference_df.shape[1]} columns | **Source:** {cfg_global.reference_name}")
        else:
            with st.expander(f"📋 Datasets to be Trained ({len(cfg_global.datasets_to_train)})", expanded=False):
                for name, data in cfg_global.datasets_to_train.items():
                    if data is not None:
                        st.markdown(f"**{name}**: {data.shape[0]} rows × {data.shape[1]} columns")

with tab2:
    st.markdown("### ⚙️ Step 2 — Baseline XGBoost Training")
    st.write("Manually configure and train a single XGBoost model. Results and plots will be displayed below.")

    if cfg_global is None:
        st.warning("Please complete Step 1 (Global Configuration) first.")
    else:
        datasets_to_train = cfg_global.datasets_to_train
        target = cfg_global.target
        feature_cols = cfg_global.feature_cols
        split_ratio = cfg_global.split_ratio
        n_splits = cfg_global.n_splits

        c1, c2 = st.columns(2)
        with c1:
            n_estimators = st.slider("n_estimators", 100, 1200, 400, key="xgb_n_estimators")
            max_depth = st.slider("max_depth", 2, 16, 6, key="xgb_max_depth")
            learning_rate = st.number_input("learning_rate", 0.001, 1.0, 0.1, format="%.3f", key="xgb_lr")
        with c2:
            subsample = st.number_input("subsample", 0.1, 1.0, 0.9, step=0.05, key="xgb_subsample")
            colsample_bytree = st.number_input("colsample_bytree", 0.1, 1.0, 0.9, step=0.05, key="xgb_colsample")

        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
        }

        if st.button("🚀 Train Baseline XGBoost", type="primary", use_container_width=True):
            last_artifacts = None
            last_dataset_name = None

            for dataset_name, df_current in datasets_to_train.items():
                if df_current is None or df_current.empty:
                    st.error(f"Skipping '{dataset_name}': Dataset is empty.")
                    continue

                with st.spinner(f"Training on '{dataset_name}'..."):
                    try:
                        # Detect date column
                        date_col_detected = _detect_date_col(df_current)
                        date_col_for_cfg = date_col_detected if date_col_detected != "__index__" else None

                        # Validate target
                        if target not in df_current.columns:
                            st.error(f"Skipping '{dataset_name}': Target column '{target}' not found.")
                            continue
                        if not pd.api.types.is_numeric_dtype(df_current[target]):
                            st.error(f"Skipping '{dataset_name}': Target column '{target}' is not numeric.")
                            continue

                        # Determine feature columns
                        if feature_cols == "ALL_DYNAMIC":
                            datetime_keywords = ['date', 'time', 'datetime', 'timestamp', 'ds']
                            excluded = [target] + [c for c in df_current.columns if any(k in c.lower() for k in datetime_keywords)]
                            final_feature_cols = [c for c in df_current.columns if c not in excluded]
                        else:
                            final_feature_cols = [c for c in df_current.columns if c in feature_cols and pd.api.types.is_numeric_dtype(df_current[c])]

                        if not final_feature_cols:
                            st.error(f"Skipping '{dataset_name}': No valid numeric feature columns found.")
                            continue

                        # Build TrainConfig
                        cfg = TrainConfig(
                            target_col=target,
                            feature_cols=final_feature_cols,
                            split_ratio=float(split_ratio),
                            n_splits=int(n_splits),
                            date_col=date_col_for_cfg,
                        )

                        # Prepare data for training
                        df_for_training = df_current.copy()
                        for col in final_feature_cols + [target]:
                            df_for_training[col] = pd.to_numeric(df_for_training[col], errors='coerce')
                        df_for_training = df_for_training.dropna(subset=final_feature_cols + [target])

                        # Train model
                        art = train_xgb(df_for_training, cfg, params)
                        last_artifacts = art
                        last_dataset_name = dataset_name
                        st.session_state['xgboost_lab_artifacts'] = art
                        st.session_state['xgboost_lab_cfg'] = cfg
                        st.session_state['xgboost_lab_df'] = df_for_training

                    except Exception as e:
                        st.error(f"Failed to train on '{dataset_name}': {e}")
                        st.code(traceback.format_exc())

            # --- Results Display ---
            if 'xgboost_lab_artifacts' in st.session_state:
                st.markdown("---")
                st.markdown("### 📈 Results")
                
                art = st.session_state['xgboost_lab_artifacts']
                cfg = st.session_state['xgboost_lab_cfg']
                df_for_training = st.session_state['xgboost_lab_df']
                
                st.success(f"✅ Successfully trained XGBoost on the last dataset.")

                # A) KPI Metrics
                st.markdown("#### Key Performance Indicators")
                
                st.markdown("""
                <style>
                .kpi-card {
                    background-color: #0f1116;
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 0 15px 5px rgba(128, 0, 128, 0.5);
                    text-align: center;
                    color: white;
                    height: 160px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .kpi-title {
                    font-size: 16px;
                    font-weight: 600;
                    color: #a0aec0;
                    margin-bottom: 10px;
                }
                .kpi-value {
                    font-size: 36px;
                    font-weight: 700;
                }
                </style>
                """, unsafe_allow_html=True)

                st.write("Train Metrics")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(create_kpi_card_html("Train MAE", art.metrics.get('train_MAE'), color=PRIMARY_COLOR), unsafe_allow_html=True)
                with c2:
                    st.markdown(create_kpi_card_html("Train RMSE", art.metrics.get('train_RMSE'), color=SECONDARY_COLOR), unsafe_allow_html=True)
                with c3:
                    st.markdown(create_kpi_card_html("Train MAPE", art.metrics.get('train_MAPE'), "%", color=WARNING_COLOR), unsafe_allow_html=True)
                with c4:
                    st.markdown(create_kpi_card_html("Train Accuracy", art.metrics.get('train_Accuracy'), "%", color=SUCCESS_COLOR), unsafe_allow_html=True)
                
                st.write("Test Metrics")
                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    st.markdown(create_kpi_card_html("Test MAE", art.metrics.get('test_MAE'), color=PRIMARY_COLOR), unsafe_allow_html=True)
                with c6:
                    st.markdown(create_kpi_card_html("Test RMSE", art.metrics.get('test_RMSE'), color=SECONDARY_COLOR), unsafe_allow_html=True)
                with c7:
                    st.markdown(create_kpi_card_html("Test MAPE", art.metrics.get('test_MAPE'), "%", color=WARNING_COLOR), unsafe_allow_html=True)
                with c8:
                    st.markdown(create_kpi_card_html("Test Accuracy", art.metrics.get('test_Accuracy'), "%", color=SUCCESS_COLOR), unsafe_allow_html=True)

                st.markdown("""
                <style>
                .chart-card {
                    background-color: #0f1116;
                    border-radius: 15px;
                    padding: 20px;
                    border: 1px solid #2D3748;
                    margin-bottom: 20px;
                }
                </style>
                """, unsafe_allow_html=True)

                # B) Forecast vs Actual Plot
                st.markdown("#### Forecast vs. Actual")
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                
                fig = go.Figure()
                if art.y_train_actual is not None:
                    fig.add_trace(go.Scatter(x=art.y_train_actual.index, y=art.y_train_actual, name="Training Actual", mode='lines', line=dict(color=SUBTLE_TEXT)))
                if art.y_train_pred is not None:
                    fig.add_trace(go.Scatter(x=art.y_train_pred.index, y=art.y_train_pred, name="Training Predicted", mode='lines', line=dict(color=SECONDARY_COLOR, dash='dot')))
                if art.y_test_actual is not None:
                    fig.add_trace(go.Scatter(x=art.y_test_actual.index, y=art.y_test_actual, name="Testing Actual", mode='lines', line=dict(color=PRIMARY_COLOR, width=2.5)))
                if art.y_test_pred is not None:
                    fig.add_trace(go.Scatter(x=art.y_test_pred.index, y=art.y_test_pred, name="Testing Predicted", mode='lines', line=dict(color=SUCCESS_COLOR, width=2.5, dash='dash')))
                
                fig.update_layout(
                    title="Forecast vs. Actual Values",
                    xaxis_title="Date",
                    yaxis_title=target,
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1),
                    yaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # C) Feature Importance
                if art.importances:
                    st.markdown("#### Feature Importance")
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    imp_df = pd.DataFrame({"Feature": list(art.importances.keys()), "Importance": list(art.importances.values())})
                    imp_df = imp_df.sort_values("Importance", ascending=False).head(20)
                    
                    fig_imp = px.bar(imp_df.sort_values("Importance", ascending=True), x="Importance", y="Feature", orientation='h', title="Top 20 Feature Importances")
                    fig_imp.update_layout(
                        xaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1),
                        yaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 🔍 Step 3 — Hyperparameter Tuning")
    st.write("Use TimeSeriesSplit to tune XGBoost hyperparameters on the reference dataset.")

    if cfg_global is None:
        st.warning("Please complete Step 1 (Global Configuration) first.")
    else:
        df_ref = cfg_global.reference_df
        ref_name = cfg_global.reference_name
        target = cfg_global.target
        feature_cols = cfg_global.feature_cols
        n_splits = cfg_global.n_splits
        metric_to_optimize = cfg_global.metric_to_optimize or "RMSE"

        if df_ref is None:
            st.warning("No reference dataset available for tuning (selected in Step 1).")
        else:
            st.info(f"Using **{ref_name}** as the tuning dataset.")
            
            optimization_method = st.radio(
                "Choose optimization method",
                options=["Grid Search", "Random Search", "Bayesian Optimization"],
                index=0,
                horizontal=True,
            )

            with st.expander("🎛️ Hyperparameter search space", expanded=True):
                ne_text = st.text_input("n_estimators candidates", value="200,400,600")
                md_text = st.text_input("max_depth candidates", value="3,6,9")
                lr_text = st.text_input("learning_rate candidates", value="0.05,0.1,0.2")
                ss_text = st.text_input("subsample candidates", value="0.8,0.9,1.0")
                cs_text = st.text_input("colsample_bytree candidates", value="0.8,0.9,1.0")

                n_iter, n_trials = 0, 0
                if optimization_method == "Random Search":
                    n_iter = st.slider("Number of random samples", 5, 50, 20, 1)
                elif optimization_method == "Bayesian Optimization":
                    n_trials = st.slider("Number of Bayesian trials", 5, 50, 20, 1)

            st.markdown(f"**Metric to optimize:** `{metric_to_optimize}`")

            if st.button("🔁 Run Hyperparameter Optimization", type="secondary", use_container_width=True):
                def _parse_list(text, dtype=int):
                    vals = [dtype(t.strip()) for t in text.split(",") if t.strip()]
                    return sorted(list(set(vals)))

                ne_list = _parse_list(ne_text, int)
                md_list = _parse_list(md_text, int)
                lr_list = _parse_list(lr_text, float)
                ss_list = _parse_list(ss_text, float)
                cs_list = _parse_list(cs_text, float)

                if not all([ne_list, md_list, lr_list, ss_list, cs_list]):
                    st.error("Please provide at least one candidate value for each hyperparameter.")
                else:
                    if feature_cols == "ALL_DYNAMIC":
                        excluded = [target] + [c for c in df_ref.columns if pd.api.types.is_datetime64_any_dtype(df_ref[c])]
                        ref_feature_cols = [c for c in df_ref.columns if c not in excluded]
                    else:
                        ref_feature_cols = [c for c in feature_cols if c in df_ref.columns]

                    date_col_for_tuning = _detect_date_col(df_ref)
                    if date_col_for_tuning == "__index__":
                        date_col_for_tuning = None

                    best_params, results_df = None, pd.DataFrame()
                    key_name = None

                    if optimization_method == "Grid Search":
                        param_grid = {"n_estimators": ne_list, "max_depth": md_list, "learning_rate": lr_list, "subsample": ss_list, "colsample_bytree": cs_list}
                        with st.spinner("Running Grid Search..."):
                            best_params, results_df = _run_xgb_grid_search(df_ref, target, ref_feature_cols, date_col_for_tuning, n_splits, param_grid, metric_to_optimize)
                        key_name = "xgb_best_params_grid"
                    
                    elif optimization_method == "Random Search":
                        param_grid = {"n_estimators": ne_list, "max_depth": md_list, "learning_rate": lr_list, "subsample": ss_list, "colsample_bytree": cs_list}
                        with st.spinner("Running Random Search..."):
                            best_params, results_df = _run_xgb_random_search(df_ref, target, ref_feature_cols, date_col_for_tuning, n_splits, param_grid, metric_to_optimize, n_iter)
                        key_name = "xgb_best_params_random"

                    elif optimization_method == "Bayesian Optimization":
                        if not _optuna_available:
                            st.error("Optuna is not installed. Bayesian Optimization is not available.")
                        else:
                            def _bounds(lst, default): return (min(lst), max(lst)) if lst else default
                            bounds = {
                                "n_estimators": _bounds(ne_list, (200, 800)),
                                "max_depth": _bounds(md_list, (3, 12)),
                                "learning_rate": _bounds(lr_list, (0.01, 0.3)),
                                "subsample": _bounds(ss_list, (0.6, 1.0)),
                                "colsample_bytree": _bounds(cs_list, (0.6, 1.0)),
                            }
                            with st.spinner("Running Bayesian Optimization with Optuna..."):
                                best_params, results_df = _run_xgb_bayesian_opt(df_ref, target, ref_feature_cols, date_col_for_tuning, n_splits, bounds, metric_to_optimize, n_trials)
                            key_name = "xgb_best_params_bayes"

                    if results_df is None or results_df.empty:
                        st.error("Hyperparameter optimization did not produce any results.")
                    else:
                        st.success("✅ Hyperparameter optimization completed.")
                        
                        metric_col = metric_to_optimize.upper()
                        ascending = metric_col != "ACCURACY"
                        if metric_col in results_df.columns:
                            results_df = results_df.sort_values(by=metric_col, ascending=ascending)

                        st.markdown("#### 📊 Optimization Results")
                        st.dataframe(results_df.round(4), use_container_width=True)

                        if best_params:
                            st.markdown("#### 🏆 Best Parameters")
                            st.json(best_params)
                            if key_name:
                                st.session_state[key_name] = best_params
                            st.info("You can now reuse these best parameters in Step 2 (baseline training).")