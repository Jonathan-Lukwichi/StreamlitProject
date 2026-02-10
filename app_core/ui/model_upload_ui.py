"""
Model Upload UI Component.

Provides UI for uploading trained models to Supabase Storage
for the "Train Locally, Deploy to Cloud" workflow.
"""

import os
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, DANGER_COLOR, TEXT_COLOR
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Model types and their expected file extensions
MODEL_FILE_PATTERNS = {
    # === ML Models ===
    "lstm": {
        "extensions": [".h5", ".keras"],
        "folder": "lstm",
        "description": "LSTM Neural Network"
    },
    "xgboost": {
        "extensions": [".pkl", ".joblib", ".json"],
        "folder": "xgboost",
        "description": "XGBoost Regressor"
    },
    "ann": {
        "extensions": [".h5", ".keras"],
        "folder": "ann",
        "description": "Artificial Neural Network"
    },
    # === Statistical Models ===
    "arima": {
        "extensions": [".pkl"],
        "folder": "arima",
        "description": "ARIMA Time Series"
    },
    "sarimax": {
        "extensions": [".pkl"],
        "folder": "sarimax",
        "description": "SARIMAX Time Series"
    },
    # === Hybrid Models ===
    "hybrid": {
        "extensions": [".h5", ".pkl", ".keras"],
        "folder": "hybrids",
        "description": "Hybrid Model"
    },
    "hybrid_lstm_xgb": {
        "extensions": [".h5", ".pkl", ".keras"],
        "folder": "hybrids/lstm_xgb",
        "description": "Hybrid: LSTM → XGBoost"
    },
    "hybrid_lstm_sarimax": {
        "extensions": [".h5", ".pkl", ".keras"],
        "folder": "hybrids/lstm_sarimax",
        "description": "Hybrid: LSTM → SARIMAX"
    },
    "hybrid_lstm_ann": {
        "extensions": [".h5", ".keras"],
        "folder": "hybrids/lstm_ann",
        "description": "Hybrid: LSTM → ANN"
    },
    # === Optimized Models ===
    "optimized": {
        "extensions": [".pkl", ".json"],
        "folder": "optimized",
        "description": "Optimized Model"
    },
    "xgboost_optimized": {
        "extensions": [".pkl", ".json"],
        "folder": "optimized/xgboost",
        "description": "XGBoost (Tuned)"
    },
    "lstm_optimized": {
        "extensions": [".h5", ".keras", ".json"],
        "folder": "optimized/lstm",
        "description": "LSTM (Tuned)"
    },
    "ann_optimized": {
        "extensions": [".h5", ".keras", ".json"],
        "folder": "optimized/ann",
        "description": "ANN (Tuned)"
    },
    # === Preprocessors ===
    "preprocessor": {
        "extensions": [".pkl", ".joblib"],
        "folder": "transformers",
        "description": "Scalers & Preprocessors"
    },
    # === Feature Selection ===
    "feature_selection": {
        "extensions": [".json", ".csv"],
        "folder": "feature_selection",
        "description": "Feature Selection Config"
    }
}

# Local artifact directories to scan
ARTIFACT_DIRECTORIES = [
    "pipeline_artifacts",
    # ML models
    "pipeline_artifacts/xgboost",
    "pipeline_artifacts/lstm",
    "pipeline_artifacts/ann",
    # Baseline models
    "pipeline_artifacts/arima",
    "pipeline_artifacts/sarimax",
    # Hybrid models
    "pipeline_artifacts/hybrids",
    "pipeline_artifacts/hybrids/lstm_xgb",
    "pipeline_artifacts/hybrids/lstm_sarimax",
    "pipeline_artifacts/hybrids/lstm_ann",
    # Optimized models (from hyperparameter tuning)
    "pipeline_artifacts/optimized",
    "pipeline_artifacts/optimized/xgboost_gridsearch",
    "pipeline_artifacts/optimized/xgboost_randomsearch",
    "pipeline_artifacts/optimized/xgboost_bayesian",
    "pipeline_artifacts/optimized/lstm_gridsearch",
    "pipeline_artifacts/optimized/lstm_randomsearch",
    "pipeline_artifacts/optimized/lstm_bayesian",
    "pipeline_artifacts/optimized/ann_gridsearch",
    "pipeline_artifacts/optimized/ann_randomsearch",
    "pipeline_artifacts/optimized/ann_bayesian",
    # Legacy/other
    "models",
    # Feature Studio transformers (scalers, encoders)
    "app_core/pipelines/saved_transformers",
    # Feature Selection results
    "pipeline_artifacts/feature_selection",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_local_models() -> List[Dict]:
    """
    Scan local directories for trained model files.

    Returns:
        List of dicts with model file information
    """
    models = []

    for dir_path in ARTIFACT_DIRECTORIES:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if os.path.isfile(filepath):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in [".h5", ".keras", ".pkl", ".joblib", ".json", ".npy", ".npz", ".csv"]:
                        # Pass full path to detect model type from parent folder
                        model_type = _detect_model_type_from_path(filepath)
                        models.append({
                            "filename": filename,
                            "local_path": filepath,
                            "extension": ext.lower(),
                            "size_mb": os.path.getsize(filepath) / (1024 * 1024),
                            "model_type": model_type
                        })

    return models


def _detect_model_type_from_path(filepath: str) -> str:
    """
    Detect model type from the full file path.
    Uses parent folder name as primary detection method.

    Args:
        filepath: Full path to the model file

    Returns:
        Model type string (xgboost, lstm, ann, hybrid, etc.)
    """
    # Normalize path separators
    filepath_normalized = filepath.replace("\\", "/").lower()
    filename = os.path.basename(filepath).lower()
    parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
    grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath))).lower()

    # === Priority 1: Check parent folder name (most reliable) ===

    # Hybrid models - check for hybrid folder patterns
    if parent_folder in ["lstm_xgb", "lstm-xgb", "lstm_xgboost"]:
        return "hybrid_lstm_xgb"
    elif parent_folder in ["lstm_sarimax", "lstm-sarimax"]:
        return "hybrid_lstm_sarimax"
    elif parent_folder in ["lstm_ann", "lstm-ann"]:
        return "hybrid_lstm_ann"
    elif parent_folder == "hybrids" or grandparent_folder == "hybrids":
        # Check filename for specific hybrid type
        if "lstm_xgb" in filename or "lstm-xgb" in filename:
            return "hybrid_lstm_xgb"
        elif "lstm_sarimax" in filename or "lstm-sarimax" in filename:
            return "hybrid_lstm_sarimax"
        elif "lstm_ann" in filename or "lstm-ann" in filename:
            return "hybrid_lstm_ann"
        return "hybrid"

    # ML models - check folder name
    elif parent_folder == "xgboost":
        return "xgboost"
    elif parent_folder == "lstm":
        return "lstm"
    elif parent_folder == "ann":
        return "ann"
    elif parent_folder == "arima":
        return "arima"
    elif parent_folder == "sarimax":
        return "sarimax"

    # Optimized models - check grandparent folder
    elif parent_folder.startswith("xgboost_") or "xgboost" in parent_folder:
        return "xgboost_optimized"
    elif parent_folder.startswith("lstm_") and grandparent_folder == "optimized":
        return "lstm_optimized"
    elif parent_folder.startswith("ann_") and grandparent_folder == "optimized":
        return "ann_optimized"
    elif grandparent_folder == "optimized":
        return "optimized"

    # Feature selection
    elif parent_folder == "feature_selection":
        return "feature_selection"

    # Transformers/preprocessors
    elif parent_folder == "saved_transformers" or "transformer" in parent_folder:
        return "preprocessor"

    # === Priority 2: Check filename patterns (fallback) ===

    if "lstm_xgb" in filename or "lstm-xgb" in filename:
        return "hybrid_lstm_xgb"
    elif "lstm_sarimax" in filename or "lstm-sarimax" in filename:
        return "hybrid_lstm_sarimax"
    elif "lstm_ann" in filename or "lstm-ann" in filename:
        return "hybrid_lstm_ann"
    elif "xgboost" in filename or "xgb" in filename:
        return "xgboost"
    elif "lstm" in filename:
        return "lstm"
    elif "ann" in filename:
        return "ann"
    elif "arima" in filename:
        return "arima"
    elif "sarimax" in filename:
        return "sarimax"
    elif any(x in filename for x in ["scaler", "preprocessor", "ohe_", "scale_cols_", "encoder_"]):
        return "preprocessor"
    elif "selected_features" in filename or "selection_config" in filename:
        return "feature_selection"

    return "unknown"


def get_remote_path(local_path: str, model_type: str) -> str:
    """
    Generate remote storage path from local path.

    Args:
        local_path: Local file path
        model_type: Type of model

    Returns:
        Remote path for Supabase Storage
    """
    filename = os.path.basename(local_path)

    # === Preprocessors / Transformers ===
    if model_type == "preprocessor" or any(
        x in filename.lower() for x in ["ohe_", "scaler_", "scale_cols_", "encoder_"]
    ):
        return f"transformers/{filename}"

    # === Feature Selection ===
    if model_type == "feature_selection":
        return f"feature_selection/{filename}"

    # === Hybrid Models ===
    if model_type == "hybrid_lstm_xgb":
        return f"hybrids/lstm_xgb/{filename}"
    elif model_type == "hybrid_lstm_sarimax":
        return f"hybrids/lstm_sarimax/{filename}"
    elif model_type == "hybrid_lstm_ann":
        return f"hybrids/lstm_ann/{filename}"
    elif model_type == "hybrid":
        return f"hybrids/{filename}"

    # === Optimized Models ===
    if model_type == "xgboost_optimized":
        return f"optimized/xgboost/{filename}"
    elif model_type == "lstm_optimized":
        return f"optimized/lstm/{filename}"
    elif model_type == "ann_optimized":
        return f"optimized/ann/{filename}"
    elif model_type == "optimized":
        return f"optimized/{filename}"

    # === Standard ML Models ===
    if model_type == "xgboost":
        return f"xgboost/{filename}"
    elif model_type == "lstm":
        return f"lstm/{filename}"
    elif model_type == "ann":
        return f"ann/{filename}"
    elif model_type == "arima":
        return f"arima/{filename}"
    elif model_type == "sarimax":
        return f"sarimax/{filename}"

    # === Fallback to pattern matching ===
    folder = MODEL_FILE_PATTERNS.get(model_type, {}).get("folder", "other")
    return f"{folder}/{filename}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_supabase_status() -> bool:
    """
    Render Supabase connection status indicator.

    Returns:
        True if connected, False otherwise
    """
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()

        if service.is_connected():
            st.success("✅ Connected to Supabase Storage")
            return True
        else:
            st.error("❌ Not connected to Supabase")
            st.info("Check your `.streamlit/secrets.toml` configuration")
            return False
    except Exception as e:
        st.error(f"❌ Supabase connection error: {e}")
        return False


def render_local_models_table():
    """Render table of locally available model files."""
    models = find_local_models()

    if not models:
        st.info("""
        **No trained models found locally.**

        Train models first using the other tabs:
        - **Benchmarks Tab**: Quick model comparison
        - **Machine Learning Tab**: XGBoost, LSTM, ANN
        - **Hyperparameter Tab**: Optuna optimization
        - **Hybrid Models Tab**: LSTM+XGBoost, LSTM+SARIMAX

        After training, model files will appear here for upload.
        """)
        return

    st.markdown(f"**Found {len(models)} model file(s):**")

    # Create a table
    import pandas as pd
    df = pd.DataFrame(models)
    df["size_mb"] = df["size_mb"].round(2)
    df = df.rename(columns={
        "filename": "Filename",
        "model_type": "Type",
        "size_mb": "Size (MB)",
        "local_path": "Path"
    })

    st.dataframe(
        df[["Filename", "Type", "Size (MB)", "Path"]],
        use_container_width=True,
        hide_index=True
    )


def render_upload_section():
    """Render the upload to Supabase section."""
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()
    except ImportError as e:
        st.error(f"ModelStorageService not available: {e}")
        return

    if not service.is_connected():
        st.warning("Connect to Supabase first to upload models.")
        return

    models = find_local_models()

    if not models:
        return

    st.markdown("---")
    st.markdown("### 📤 Upload Models to Supabase")

    # Select models to upload
    model_options = {m["filename"]: m for m in models}

    selected = st.multiselect(
        "Select models to upload:",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        help="Select which model files to upload to Supabase Storage"
    )

    if selected:
        # Show what will be uploaded
        st.markdown("**Upload Preview:**")
        for filename in selected:
            model = model_options[filename]
            remote = get_remote_path(model["local_path"], model["model_type"])
            st.text(f"  {filename} → {remote} ({model['size_mb']:.2f} MB)")

        # Upload button
        col1, col2 = st.columns([1, 3])
        with col1:
            upload_btn = st.button(
                "☁️ Upload Selected",
                type="primary",
                use_container_width=True
            )

        if upload_btn:
            progress = st.progress(0)
            status = st.empty()

            results = []
            for i, filename in enumerate(selected):
                model = model_options[filename]
                remote_path = get_remote_path(model["local_path"], model["model_type"])

                status.text(f"Uploading {filename}...")
                progress.progress((i + 1) / len(selected))

                success, msg = service.upload_model(
                    local_path=model["local_path"],
                    remote_path=remote_path
                )
                results.append((filename, success, msg))

            progress.progress(100)
            status.empty()

            # Show results
            success_count = sum(1 for _, s, _ in results if s)
            if success_count == len(results):
                st.success(f"✅ All {success_count} models uploaded successfully!")
            else:
                st.warning(f"⚠️ {success_count}/{len(results)} models uploaded")
                for filename, success, msg in results:
                    if not success:
                        st.error(f"Failed: {filename} - {msg}")


def render_remote_models_table():
    """Render table of models in Supabase Storage."""
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()
    except ImportError:
        return

    if not service.is_connected():
        return

    st.markdown("---")
    st.markdown("### ☁️ Models in Supabase Storage")

    with st.spinner("Loading remote models..."):
        models = service.list_models()

    if not models:
        st.info("No models found in Supabase Storage.")
        return

    # Create table
    import pandas as pd
    data = []
    for m in models:
        data.append({
            "Name": m.name,
            "Type": m.model_type,
            "Path": m.storage_path,
            "Size (KB)": m.file_size_bytes / 1024 if m.file_size_bytes else 0,
            "Horizon": m.horizon or "-"
        })

    df = pd.DataFrame(data)
    df["Size (KB)"] = df["Size (KB)"].round(1)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Storage stats
    stats = service.get_storage_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", stats["total_files"])
    with col2:
        st.metric("Total Size", f"{stats['total_size_mb']:.2f} MB")
    with col3:
        st.metric("Max File Size", f"{stats['max_file_size_mb']} MB")


def render_delete_section():
    """Render section to delete models from Supabase."""
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()
    except ImportError:
        return

    if not service.is_connected():
        return

    models = service.list_models()
    if not models:
        return

    with st.expander("🗑️ Delete Models from Supabase"):
        st.warning("⚠️ This will permanently delete models from cloud storage.")

        model_paths = [m.storage_path for m in models]
        to_delete = st.multiselect(
            "Select models to delete:",
            options=model_paths
        )

        if to_delete:
            if st.button("🗑️ Delete Selected", type="secondary"):
                for path in to_delete:
                    success, msg = service.delete_model(path)
                    if success:
                        st.success(f"Deleted: {path}")
                    else:
                        st.error(f"Failed to delete {path}: {msg}")
                st.rerun()


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_cloud_sync_tab():
    """
    Render the complete Cloud Sync tab for model management.

    This tab allows users to:
    1. See locally trained models
    2. Upload models to Supabase Storage
    3. View models in cloud storage
    4. Delete models from cloud
    """
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    '>
        <h3 style='margin: 0 0 0.5rem 0; color: #667eea;'>☁️ Cloud Model Sync</h3>
        <p style='margin: 0; opacity: 0.8; font-size: 0.95rem;'>
            Upload your locally trained models to Supabase Storage so they can be used
            on Streamlit Cloud for predictions without memory-intensive training.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Workflow explanation
    with st.expander("📖 How This Works", expanded=False):
        st.markdown("""
        **Train Locally, Predict in Cloud** workflow:

        1. **Train Models Locally** (your computer)
           - Use the ML tabs to train XGBoost, LSTM, etc.
           - Models are saved to `pipeline_artifacts/`
           - Your computer has 8-16 GB RAM for training

        2. **Upload to Supabase** (this tab)
           - Click "Upload Selected" to push models to cloud storage
           - Models are stored in your Supabase project
           - 50 MB max file size on free tier

        3. **Predict on Streamlit Cloud**
           - App downloads models from Supabase on startup
           - Only ~100 MB RAM needed for predictions
           - No training = No crashes!
        """)

    # Connection status
    connected = render_supabase_status()

    st.markdown("---")
    st.markdown("### 💾 Local Model Files")
    render_local_models_table()

    if connected:
        render_upload_section()
        render_remote_models_table()
        render_delete_section()
