"""
Model Storage Service - Upload/Download trained models to/from Supabase Storage.

This service handles:
- Uploading trained model files (.h5, .pkl) to Supabase Storage
- Downloading models from Supabase Storage for predictions
- Listing available models in storage
- Tracking model metadata in the database

Usage:
    from app_core.data.model_storage_service import get_model_storage_service

    service = get_model_storage_service()

    # Upload a model
    service.upload_model("pipeline_artifacts/lstm_h1.h5", "lstm/horizon_1.h5")

    # Download a model
    service.download_model("lstm/horizon_1.h5", "pipeline_artifacts/lstm_h1.h5")

    # List all models
    models = service.list_models()
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json

import streamlit as st

from app_core.data.supabase_client import get_cached_supabase_client

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

MODELS_BUCKET = "models"  # Supabase Storage bucket name

# Supported model file extensions
SUPPORTED_EXTENSIONS = {
    ".h5": "Keras/TensorFlow model",
    ".keras": "Keras model",
    ".pkl": "Pickle (scikit-learn, XGBoost)",
    ".joblib": "Joblib (scikit-learn)",
    ".json": "JSON configuration",
    ".npy": "NumPy array",
    ".npz": "NumPy compressed array",
}

# Maximum file size (50 MB for Supabase free tier)
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a stored model."""
    name: str                      # e.g., "lstm_horizon_1"
    model_type: str                # e.g., "LSTM", "XGBoost"
    storage_path: str              # e.g., "lstm/horizon_1.h5"
    file_size_bytes: int
    horizon: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None  # rmse, mae, accuracy, etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.uploaded_at:
            result["uploaded_at"] = self.uploaded_at.isoformat()
        return result


# =============================================================================
# MODEL STORAGE SERVICE
# =============================================================================

class ModelStorageService:
    """
    Service for storing and retrieving trained ML models from Supabase Storage.

    This enables the "Train Locally, Deploy to Cloud" architecture:
    1. Train models on your computer (full RAM)
    2. Upload trained models to Supabase Storage
    3. Streamlit Cloud downloads and uses models for predictions
    """

    def __init__(self):
        """Initialize the service."""
        self.client = None
        self.bucket_name = MODELS_BUCKET
        self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize Supabase client."""
        try:
            self.client = get_cached_supabase_client()
            return self.client is not None
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to Supabase."""
        return self.client is not None

    # =========================================================================
    # UPLOAD METHODS
    # =========================================================================

    def upload_model(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = True,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Upload a trained model file to Supabase Storage.

        Args:
            local_path: Path to the local model file
            remote_path: Path in Supabase Storage (e.g., "lstm/horizon_1.h5")
            overwrite: Whether to overwrite if file exists
            metadata: Optional metadata to store with the model

        Returns:
            Tuple of (success: bool, message: str)

        Example:
            success, msg = service.upload_model(
                local_path="pipeline_artifacts/models/lstm_h1.h5",
                remote_path="lstm/horizon_1.h5"
            )
        """
        if not self.is_connected():
            return False, "Not connected to Supabase"

        # Check if local file exists
        if not os.path.exists(local_path):
            return False, f"Local file not found: {local_path}"

        # Check file size
        file_size = os.path.getsize(local_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            return False, f"File too large: {file_size / (1024*1024):.1f} MB (max {MAX_FILE_SIZE_MB} MB)"

        # Check file extension
        _, ext = os.path.splitext(local_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unusual file extension: {ext}")

        try:
            # Read file
            with open(local_path, "rb") as f:
                file_data = f.read()

            # Delete existing file if overwrite is True
            if overwrite:
                try:
                    self.client.storage.from_(self.bucket_name).remove([remote_path])
                except Exception:
                    pass  # File might not exist, that's OK

            # Upload to Supabase Storage
            result = self.client.storage.from_(self.bucket_name).upload(
                path=remote_path,
                file=file_data,
                file_options={"content-type": "application/octet-stream"}
            )

            logger.info(f"Uploaded model: {local_path} -> {remote_path} ({file_size / 1024:.1f} KB)")
            return True, f"Successfully uploaded to {remote_path}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Upload failed: {error_msg}")
            return False, f"Upload failed: {error_msg}"

    def upload_model_directory(
        self,
        local_dir: str,
        remote_prefix: str = "",
        extensions: Optional[List[str]] = None
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Upload all model files from a local directory.

        Args:
            local_dir: Local directory containing model files
            remote_prefix: Prefix for remote paths (e.g., "lstm/")
            extensions: List of extensions to upload (default: all supported)

        Returns:
            Dict mapping filename to (success, message)

        Example:
            results = service.upload_model_directory(
                local_dir="pipeline_artifacts/models/lstm",
                remote_prefix="lstm/"
            )
        """
        if extensions is None:
            extensions = list(SUPPORTED_EXTENSIONS.keys())

        results = {}

        if not os.path.isdir(local_dir):
            return {"error": (False, f"Directory not found: {local_dir}")}

        for filename in os.listdir(local_dir):
            _, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                local_path = os.path.join(local_dir, filename)
                remote_path = f"{remote_prefix}{filename}"
                results[filename] = self.upload_model(local_path, remote_path)

        return results

    # =========================================================================
    # DOWNLOAD METHODS
    # =========================================================================

    def download_model(
        self,
        remote_path: str,
        local_path: str,
        create_dirs: bool = True
    ) -> Tuple[bool, str]:
        """
        Download a model file from Supabase Storage.

        Args:
            remote_path: Path in Supabase Storage (e.g., "lstm/horizon_1.h5")
            local_path: Local path to save the file
            create_dirs: Whether to create parent directories if they don't exist

        Returns:
            Tuple of (success: bool, message: str)

        Example:
            success, msg = service.download_model(
                remote_path="lstm/horizon_1.h5",
                local_path="pipeline_artifacts/models/lstm_h1.h5"
            )
        """
        if not self.is_connected():
            return False, "Not connected to Supabase"

        try:
            # Create directories if needed
            if create_dirs:
                local_dir = os.path.dirname(local_path)
                if local_dir and not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)

            # Download from Supabase Storage
            response = self.client.storage.from_(self.bucket_name).download(remote_path)

            # Save to local file
            with open(local_path, "wb") as f:
                f.write(response)

            file_size = os.path.getsize(local_path)
            logger.info(f"Downloaded model: {remote_path} -> {local_path} ({file_size / 1024:.1f} KB)")
            return True, f"Successfully downloaded to {local_path}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download failed: {error_msg}")
            return False, f"Download failed: {error_msg}"

    def download_if_not_exists(
        self,
        remote_path: str,
        local_path: str
    ) -> Tuple[bool, str]:
        """
        Download a model only if it doesn't exist locally.

        Useful for Streamlit Cloud to download models on first run.

        Args:
            remote_path: Path in Supabase Storage
            local_path: Local path to check/save

        Returns:
            Tuple of (success: bool, message: str)
        """
        if os.path.exists(local_path):
            return True, f"Model already exists locally: {local_path}"

        return self.download_model(remote_path, local_path)

    def download_all_models(
        self,
        local_dir: str,
        folder: Optional[str] = None
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Download all models from storage to a local directory.

        Args:
            local_dir: Local directory to save models
            folder: Optional folder in storage to download from (e.g., "lstm")

        Returns:
            Dict mapping filename to (success, message)
        """
        results = {}

        # List files in storage
        files = self.list_files(folder=folder)

        for file_info in files:
            remote_path = file_info.get("name", "")
            if remote_path:
                filename = os.path.basename(remote_path)
                local_path = os.path.join(local_dir, filename)
                results[filename] = self.download_model(remote_path, local_path)

        return results

    # =========================================================================
    # LIST/QUERY METHODS
    # =========================================================================

    def list_files(self, folder: Optional[str] = None) -> List[Dict]:
        """
        List all files in the models bucket.

        Args:
            folder: Optional subfolder to list (e.g., "lstm")

        Returns:
            List of file info dictionaries
        """
        if not self.is_connected():
            return []

        try:
            path = folder if folder else ""
            response = self.client.storage.from_(self.bucket_name).list(path=path)
            return response if response else []
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def list_models(self) -> List[ModelInfo]:
        """
        List all models with parsed information.

        Returns:
            List of ModelInfo objects
        """
        models = []

        # Get files from each model type folder
        model_folders = ["lstm", "xgboost", "arima", "ann", "hybrid", "preprocessors", "transformers", "feature_selection"]

        for folder in model_folders:
            files = self.list_files(folder=folder)

            for file_info in files:
                name = file_info.get("name", "")
                if name and not name.endswith("/"):  # Skip folders
                    model_info = ModelInfo(
                        name=os.path.splitext(name)[0],
                        model_type=folder.upper(),
                        storage_path=f"{folder}/{name}",
                        file_size_bytes=file_info.get("metadata", {}).get("size", 0),
                        uploaded_at=self._parse_datetime(
                            file_info.get("created_at") or file_info.get("updated_at")
                        )
                    )

                    # Try to extract horizon from filename
                    if "horizon" in name.lower() or name.startswith("h"):
                        try:
                            # Extract number from filename like "horizon_1.h5" or "h1.pkl"
                            import re
                            match = re.search(r'(\d+)', name)
                            if match:
                                model_info.horizon = int(match.group(1))
                        except:
                            pass

                    models.append(model_info)

        return models

    def model_exists(self, remote_path: str) -> bool:
        """
        Check if a model exists in storage.

        Args:
            remote_path: Path in storage (e.g., "lstm/horizon_1.h5")

        Returns:
            True if model exists
        """
        folder = os.path.dirname(remote_path)
        filename = os.path.basename(remote_path)

        files = self.list_files(folder=folder)
        return any(f.get("name") == filename for f in files)

    # =========================================================================
    # DELETE METHODS
    # =========================================================================

    def delete_model(self, remote_path: str) -> Tuple[bool, str]:
        """
        Delete a model from storage.

        Args:
            remote_path: Path in storage to delete

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_connected():
            return False, "Not connected to Supabase"

        try:
            self.client.storage.from_(self.bucket_name).remove([remote_path])
            logger.info(f"Deleted model: {remote_path}")
            return True, f"Successfully deleted {remote_path}"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Delete failed: {error_msg}")
            return False, f"Delete failed: {error_msg}"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_datetime(self, dt_string: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from Supabase."""
        if not dt_string:
            return None
        try:
            # Handle various formats
            for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(dt_string, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict with storage stats (total files, total size, by type)
        """
        models = self.list_models()

        total_size = sum(m.file_size_bytes for m in models)
        by_type = {}

        for model in models:
            if model.model_type not in by_type:
                by_type[model.model_type] = {"count": 0, "size_bytes": 0}
            by_type[model.model_type]["count"] += 1
            by_type[model.model_type]["size_bytes"] += model.file_size_bytes

        return {
            "total_files": len(models),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "max_file_size_mb": MAX_FILE_SIZE_MB
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_model_storage_service: Optional[ModelStorageService] = None

def get_model_storage_service() -> ModelStorageService:
    """
    Get the singleton ModelStorageService instance.

    Returns:
        ModelStorageService instance
    """
    global _model_storage_service

    if _model_storage_service is None:
        _model_storage_service = ModelStorageService()

    return _model_storage_service


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def upload_trained_model(
    local_path: str,
    model_type: str,
    horizon: int,
    file_extension: str = ".h5"
) -> Tuple[bool, str]:
    """
    Convenience function to upload a trained model with standard naming.

    Args:
        local_path: Path to local model file
        model_type: Type of model (lstm, xgboost, arima, ann, hybrid)
        horizon: Forecast horizon (1-7)
        file_extension: File extension (.h5, .pkl, etc.)

    Returns:
        Tuple of (success, message)

    Example:
        success, msg = upload_trained_model(
            "pipeline_artifacts/lstm_h1.h5",
            model_type="lstm",
            horizon=1
        )
    """
    service = get_model_storage_service()
    remote_path = f"{model_type.lower()}/horizon_{horizon}{file_extension}"
    return service.upload_model(local_path, remote_path)


def download_model_for_prediction(
    model_type: str,
    horizon: int,
    local_dir: str = "pipeline_artifacts/models",
    file_extension: str = ".h5"
) -> Tuple[bool, str, Optional[str]]:
    """
    Convenience function to download a model for prediction.

    Args:
        model_type: Type of model (lstm, xgboost, arima, ann, hybrid)
        horizon: Forecast horizon (1-7)
        local_dir: Local directory to save model
        file_extension: Expected file extension

    Returns:
        Tuple of (success, message, local_path or None)

    Example:
        success, msg, path = download_model_for_prediction(
            model_type="lstm",
            horizon=1
        )
        if success:
            model = load_model(path)
    """
    service = get_model_storage_service()

    remote_path = f"{model_type.lower()}/horizon_{horizon}{file_extension}"
    local_path = os.path.join(local_dir, f"{model_type.lower()}_horizon_{horizon}{file_extension}")

    # Download only if not exists locally
    success, msg = service.download_if_not_exists(remote_path, local_path)

    if success:
        return True, msg, local_path
    else:
        return False, msg, None


def download_transformers(
    local_dir: str = "app_core/pipelines/saved_transformers"
) -> Dict[str, Tuple[bool, str]]:
    """
    Download all transformer files (scalers, encoders) from Supabase Storage.

    Used on Streamlit Cloud startup to ensure transformers match
    the trained models for correct feature scaling.

    Args:
        local_dir: Local directory to save transformers

    Returns:
        Dict mapping filename to (success, message)

    Example:
        results = download_transformers()
        # Downloads ohe_A.joblib, scaler_A.joblib, etc.
    """
    service = get_model_storage_service()

    if not service.is_connected():
        return {"error": (False, "Not connected to Supabase")}

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    return service.download_all_models(
        local_dir=local_dir,
        folder="transformers"
    )


def download_feature_selection_config(
    local_dir: str = "pipeline_artifacts/feature_selection"
) -> Dict[str, Tuple[bool, str]]:
    """
    Download feature selection configuration from Supabase Storage.

    Args:
        local_dir: Local directory to save config files

    Returns:
        Dict mapping filename to (success, message)
    """
    service = get_model_storage_service()

    if not service.is_connected():
        return {"error": (False, "Not connected to Supabase")}

    os.makedirs(local_dir, exist_ok=True)

    return service.download_all_models(
        local_dir=local_dir,
        folder="feature_selection"
    )


def ensure_preprocessing_artifacts(
    transformer_dir: str = "app_core/pipelines/saved_transformers",
    selection_dir: str = "pipeline_artifacts/feature_selection"
) -> Dict[str, bool]:
    """
    Ensure all preprocessing artifacts are available locally.
    Downloads from cloud if missing.

    This is the main entry point for cloud deployments to fetch
    all preprocessing components needed for predictions.

    Args:
        transformer_dir: Local directory for transformers
        selection_dir: Local directory for feature selection config

    Returns:
        Dict with status of each artifact type:
        {"transformers": bool, "feature_selection": bool}

    Example:
        # Call on app startup
        status = ensure_preprocessing_artifacts()
        if not status["transformers"]:
            st.warning("Transformers not available")
    """
    service = get_model_storage_service()
    status = {"transformers": False, "feature_selection": False}

    if not service.is_connected():
        logger.warning("Cannot ensure preprocessing artifacts: not connected to Supabase")
        return status

    # Download transformers
    try:
        result = download_transformers(transformer_dir)
        # Check if any downloads succeeded (excluding error key)
        status["transformers"] = any(
            success for key, (success, _) in result.items()
            if key != "error" and success
        )
        if status["transformers"]:
            logger.info(f"Transformers downloaded to {transformer_dir}")
    except Exception as e:
        logger.error(f"Failed to download transformers: {e}")

    # Download feature selection config
    try:
        result = download_feature_selection_config(selection_dir)
        status["feature_selection"] = any(
            success for key, (success, _) in result.items()
            if key != "error" and success
        )
        if status["feature_selection"]:
            logger.info(f"Feature selection config downloaded to {selection_dir}")
    except Exception as e:
        logger.error(f"Failed to download feature selection config: {e}")

    return status
