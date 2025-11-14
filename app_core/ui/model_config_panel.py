from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional, Union

@dataclass
class GlobalMLConfig:
    """Dataclass to hold global ML configuration settings."""
    datasets_to_train: Dict[str, pd.DataFrame]
    reference_df: Optional[pd.DataFrame]
    reference_name: Optional[str]
    target: str
    feature_cols: Union[List[str], str]
    split_ratio: float
    n_splits: int
    metric_to_optimize: str

def render_global_ml_config(available_datasets: Dict[str, pd.DataFrame]) -> Optional[GlobalMLConfig]:
    """
    Renders a reusable panel for global ML configuration and returns the settings.

    Args:
        available_datasets: A dictionary of available datasets for training.

    Returns:
        A GlobalMLConfig instance if configuration is successful, otherwise None.
    """
    st.markdown("### 📊 Global ML Configuration")

    if not available_datasets:
        st.warning("⚠️ No datasets available for configuration. Please prepare datasets first.")
        return None

    # A) Dataset mode
    dataset_mode = st.radio(
        "Choose dataset source mode",
        options=["Manual Selection"],
        index=0,
        horizontal=True,
        help=(
            "- **Manual Selection:** Choose one dataset from the list."
        )
    )

    # B) Dataset selection
    datasets_to_train = {}
    reference_df = None
    reference_name = None

    # Since only "Manual Selection" is available
    dataset_name = st.selectbox("Choose dataset source", options=list(available_datasets.keys()))
    if dataset_name:
        reference_df = available_datasets[dataset_name]
        reference_name = dataset_name
        datasets_to_train[dataset_name] = reference_df
    else:
        st.error("No dataset selected.")
        return None

    if reference_df is None:
        st.error("Could not obtain a reference dataset.")
        return None

    # C) Target column
    st.markdown("--- ")
    st.markdown("#### 🎯 Target & Features")
    cols = list(reference_df.columns)
    if "target_1" in cols:
        target = "target_1"
        st.selectbox("Target column", [target], index=0, disabled=True, help="`target_1` is the fixed target for this model.")
    else:
        default_target_idx = len(cols) - 1
        target = st.selectbox("Target column", cols, index=default_target_idx)


    # D) Feature selection
    datetime_keywords = ['date', 'time', 'datetime', 'timestamp', 'ds']
    feature_cols: Union[List[str], str]

    excluded_cols = [target]
    for col in reference_df.columns:
        if col == target:
            continue
        if pd.api.types.is_datetime64_any_dtype(reference_df[col]) or any(k in col.lower() for k in datetime_keywords):
            excluded_cols.append(col)
    
    available_features = [c for c in cols if c not in excluded_cols]
    if len(excluded_cols) > 1:
        st.warning(f"⚠️ Excluded {len(excluded_cols)-1} datetime/date column(s) from feature selection: {', '.join([c for c in excluded_cols if c != target])}")
    
    col_select1, col_select2 = st.columns([1, 4])
    with col_select1:
        select_all = st.checkbox("Select All Features", value=True)
    
    if select_all:
        feature_cols = available_features
        with col_select2:
            st.info(f"✅ All {len(available_features)} features selected")
    else:
        feature_cols = st.multiselect("Feature columns", available_features, key="feature_multiselect")

    # E) Time-based splitting + CV
    st.markdown("--- ")
    st.markdown("#### ⚙️ Splitting & Validation")
    c1, c2 = st.columns(2)
    with c1:
        split_ratio = st.slider("Train size (time split)", 0.60, 0.95, 0.80, 0.01)
    with c2:
        n_splits = st.slider("CV folds (TimeSeriesSplit)", 2, 10, 3, 1)

    # F) Metric to optimize
    st.markdown("--- ")
    st.markdown("#### 🏆 Optimization Metric")
    metric_to_optimize = st.selectbox(
        "Metric to optimize",
        options=["RMSE", "MAE", "MAPE", "Accuracy"],
        index=0,
        help="This metric will be used by tuning functions to find the best model."
    )

    # Summary expander
    with st.expander("📋 Configuration Summary", expanded=False):
        st.json({
            "Dataset Mode": dataset_mode,
            "Reference Dataset": reference_name,
            "Target Column": target,
            "Number of Features": len(feature_cols) if isinstance(feature_cols, list) else feature_cols,
            "Train/Test Split": f"{int(split_ratio*100)}% / {int((1-split_ratio)*100)}%",
            "CV Folds": n_splits,
            "Optimization Metric": metric_to_optimize,
        })

    return GlobalMLConfig(
        datasets_to_train=datasets_to_train,
        reference_df=reference_df,
        reference_name=reference_name,
        target=target,
        feature_cols=feature_cols,
        split_ratio=split_ratio,
        n_splits=n_splits,
        metric_to_optimize=metric_to_optimize,
    )
