"""
Model Download UI Component.

Provides UI for downloading trained models from Supabase Storage
for the "Train Locally, Deploy to Cloud" workflow.

This is used on Streamlit Cloud to fetch models that were trained locally.
"""

import os
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, ERROR_COLOR, TEXT_COLOR
)

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL DOWNLOAD STATUS
# =============================================================================

def check_local_models() -> Dict[str, bool]:
    """
    Check which model types exist locally.

    Returns:
        Dict mapping model type to exists (True/False)
    """
    model_dirs = {
        "lstm": ["pipeline_artifacts/hybrids/lstm_xgb", "pipeline_artifacts/hybrids/lstm_sarimax"],
        "xgboost": ["pipeline_artifacts/hybrids/lstm_xgb"],
        "ann": ["pipeline_artifacts/hybrids/lstm_ann"],
        "arima": ["pipeline_artifacts"],
        "sarimax": ["pipeline_artifacts"],
    }

    status = {}
    for model_type, dirs in model_dirs.items():
        found = False
        for dir_path in dirs:
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path) if os.path.isdir(dir_path) else []:
                    if model_type.lower() in f.lower():
                        found = True
                        break
        status[model_type] = found

    return status


def check_remote_models() -> Dict[str, int]:
    """
    Check which models exist in Supabase Storage.

    Returns:
        Dict mapping model type to count of files
    """
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()

        if not service.is_connected():
            return {}

        models = service.list_models()

        counts = {}
        for m in models:
            model_type = m.model_type.lower()
            counts[model_type] = counts.get(model_type, 0) + 1

        return counts
    except Exception as e:
        logger.error(f"Error checking remote models: {e}")
        return {}


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_model_status_badge():
    """
    Render a compact status badge showing model availability.
    Use in sidebar or page header.
    """
    local = check_local_models()
    local_count = sum(1 for v in local.values() if v)

    if local_count > 0:
        st.markdown(f"""
        <div style='
            display: inline-flex;
            align-items: center;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
        '>
            <span style='color: #22c55e;'>✓ {local_count} model types available</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            display: inline-flex;
            align-items: center;
            background: rgba(251, 191, 36, 0.15);
            border: 1px solid rgba(251, 191, 36, 0.3);
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
        '>
            <span style='color: #fbbf24;'>⚠ No local models</span>
        </div>
        """, unsafe_allow_html=True)


def render_model_download_panel():
    """
    Render panel for downloading models from Supabase.
    Shows remote models and allows downloading.
    """
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()
    except ImportError as e:
        st.warning(f"Model storage service not available: {e}")
        return

    if not service.is_connected():
        st.info("""
        **Connect to Supabase** to download pre-trained models.

        Models trained locally can be uploaded via the Training page's "Cloud Sync" tab.
        """)
        return

    # Check remote models
    with st.spinner("Checking cloud storage..."):
        remote_models = service.list_models()

    if not remote_models:
        st.info("""
        **No models in cloud storage.**

        Train models locally and upload them using the "Cloud Sync" tab
        in the Training page.
        """)
        return

    st.markdown(f"**☁️ {len(remote_models)} model(s) available in cloud storage:**")

    # Group by type
    by_type = {}
    for m in remote_models:
        t = m.model_type
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(m)

    # Show grouped models
    for model_type, models in by_type.items():
        with st.expander(f"{model_type} ({len(models)} files)", expanded=False):
            for m in models:
                size_kb = m.file_size_bytes / 1024 if m.file_size_bytes else 0
                st.text(f"  {m.storage_path} ({size_kb:.1f} KB)")

    # Download all button
    st.markdown("---")
    if st.button("📥 Download All Models", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        success_count = 0
        for i, m in enumerate(remote_models):
            status.text(f"Downloading {m.storage_path}...")
            progress.progress((i + 1) / len(remote_models))

            # Create local path
            local_path = f"pipeline_artifacts/{m.storage_path}"
            success, msg = service.download_model(m.storage_path, local_path)

            if success:
                success_count += 1

        progress.progress(100)
        status.empty()

        if success_count == len(remote_models):
            st.success(f"✅ Downloaded all {success_count} models!")
        else:
            st.warning(f"⚠️ Downloaded {success_count}/{len(remote_models)} models")


def render_cloud_models_info():
    """
    Render informational section about cloud model sync.
    For use on forecast/prediction pages.
    """
    with st.expander("☁️ Cloud Model Sync", expanded=False):
        st.markdown("""
        **Train Locally, Predict in Cloud**

        This app supports training models on your computer (with more RAM)
        and uploading them to Supabase Storage for use on Streamlit Cloud.

        **Current Status:**
        """)

        # Show status
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Local Models:**")
            local = check_local_models()
            for model_type, exists in local.items():
                icon = "✅" if exists else "❌"
                st.text(f"  {icon} {model_type}")

        with col2:
            st.markdown("**Cloud Models:**")
            remote = check_remote_models()
            if remote:
                for model_type, count in remote.items():
                    st.text(f"  ☁️ {model_type}: {count} files")
            else:
                st.text("  (not connected)")

        st.markdown("---")
        st.markdown("""
        **To use cloud models:**
        1. Train models locally using the Training page
        2. Upload via "Cloud Sync" tab
        3. Models will be available here

        **Note:** Results (predictions) are saved separately to the database
        and load automatically.
        """)


def ensure_models_available() -> bool:
    """
    Check if models are available (local or download from cloud).

    Returns:
        True if at least some models are available
    """
    local = check_local_models()

    if any(local.values()):
        return True

    # Try to download from cloud
    try:
        from app_core.data.model_storage_service import get_model_storage_service
        service = get_model_storage_service()

        if service.is_connected():
            remote = service.list_models()
            if remote:
                return True  # Models available for download
    except Exception:
        pass

    return False
