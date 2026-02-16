# =============================================================================
# app_core/data/cloud_model_sync.py
# Helper module for syncing model results between session state and Supabase.
# Enables cloud deployment support for HealthForecast AI.
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from app_core.data.results_storage_service import (
    ResultsStorageService,
    get_storage_service,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Page key used for all model results in Supabase
PAGE_KEY = "model_results"

# Session state keys that contain model results (used by pages 09-13)
MODEL_SESSION_KEYS = [
    # ML Models
    "ml_mh_results_xgboost",
    "ml_mh_results_lstm",
    "ml_mh_results_ann",
    "ml_mh_results",  # Generic fallback
    # Statistical Models
    "arima_mh_results",
    "sarimax_results",
    # Hybrid Models
    "lstm_xgb_results",
    "lstm_sarimax_results",
    "lstm_ann_results",
    # Optimized Models
    "opt_results_XGBoost",
    "opt_results_LSTM",
    "opt_results_ANN",
    # Ensemble/Stacking
    "ensemble_results",
    "stacking_results",
    # Forecast Hub (for pages 10-13)
    "forecast_hub_demand",
    "active_forecast",
    # Category results
    "ml_category_results",
]

# Human-readable names for display
MODEL_DISPLAY_NAMES = {
    "ml_mh_results_xgboost": "XGBoost",
    "ml_mh_results_lstm": "LSTM",
    "ml_mh_results_ann": "ANN",
    "ml_mh_results": "ML Model (Generic)",
    "arima_mh_results": "ARIMA",
    "sarimax_results": "SARIMAX",
    "lstm_xgb_results": "Hybrid LSTM+XGBoost",
    "lstm_sarimax_results": "Hybrid LSTM+SARIMAX",
    "lstm_ann_results": "Hybrid LSTM+ANN",
    "opt_results_XGBoost": "XGBoost (Optimized)",
    "opt_results_LSTM": "LSTM (Optimized)",
    "opt_results_ANN": "ANN (Optimized)",
    "ensemble_results": "Ensemble",
    "stacking_results": "Stacking",
    "forecast_hub_demand": "Forecast Hub Demand",
    "active_forecast": "Active Forecast",
    "ml_category_results": "Category Results",
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_service() -> Optional[ResultsStorageService]:
    """Get the ResultsStorageService instance."""
    try:
        return get_storage_service()
    except Exception as e:
        logger.warning(f"Could not get storage service: {e}")
        return None


def save_model_to_supabase(session_key: str) -> Tuple[bool, str]:
    """
    Save a specific model result from session state to Supabase.

    Args:
        session_key: The session state key containing the model results

    Returns:
        Tuple of (success: bool, message: str)
    """
    service = get_service()
    if service is None:
        return False, "Storage service not available"

    if not service.is_connected():
        return False, "Supabase not connected"

    data = st.session_state.get(session_key)
    if data is None:
        return False, f"No data in session state for {session_key}"

    try:
        # Add metadata for the model
        metadata = {
            "model_key": session_key,
            "display_name": MODEL_DISPLAY_NAMES.get(session_key, session_key),
            "saved_from": "08_Train_Models",
        }

        # Check if data has model_type attribute
        if isinstance(data, dict):
            if "model_type" in data:
                metadata["model_type"] = data["model_type"]
            if "successful" in data:
                metadata["horizons"] = data["successful"]

        success, msg = service.save_results(PAGE_KEY, session_key, data, metadata)

        if success:
            logger.info(f"Saved {session_key} to Supabase: {msg}")
        else:
            logger.warning(f"Failed to save {session_key}: {msg}")

        return success, msg

    except Exception as e:
        error_msg = f"Error saving {session_key}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def load_model_from_supabase(session_key: str) -> Tuple[bool, str]:
    """
    Load a specific model result from Supabase into session state.

    Args:
        session_key: The session state key to load into

    Returns:
        Tuple of (success: bool, message: str)
    """
    service = get_service()
    if service is None:
        return False, "Storage service not available"

    if not service.is_connected():
        return False, "Supabase not connected"

    try:
        data = service.load_results(PAGE_KEY, session_key)

        if data is None:
            return False, f"No data found in Supabase for {session_key}"

        # Store in session state
        st.session_state[session_key] = data

        logger.info(f"Loaded {session_key} from Supabase")
        return True, f"Loaded {MODEL_DISPLAY_NAMES.get(session_key, session_key)}"

    except Exception as e:
        error_msg = f"Error loading {session_key}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def load_all_models_from_supabase() -> Tuple[int, List[str]]:
    """
    Load all model results from Supabase into session state.

    Returns:
        Tuple of (loaded_count, list_of_loaded_keys)
    """
    service = get_service()
    if service is None:
        return 0, []

    if not service.is_connected():
        return 0, []

    loaded_keys = []

    try:
        # Load all results for the model_results page
        all_results = service.load_all_for_page(PAGE_KEY)

        for key, data in all_results.items():
            if data is not None:
                st.session_state[key] = data
                loaded_keys.append(key)
                logger.info(f"Loaded {key} from Supabase")

        return len(loaded_keys), loaded_keys

    except Exception as e:
        logger.error(f"Error loading all models: {e}")
        return 0, []


def save_all_models_to_supabase() -> Tuple[int, int, List[str]]:
    """
    Save all model results from session state to Supabase.

    Returns:
        Tuple of (success_count, total_count, error_messages)
    """
    service = get_service()
    if service is None:
        return 0, 0, ["Storage service not available"]

    if not service.is_connected():
        return 0, 0, ["Supabase not connected"]

    success_count = 0
    total_count = 0
    errors = []

    for key in MODEL_SESSION_KEYS:
        data = st.session_state.get(key)
        if data is not None:
            total_count += 1
            success, msg = save_model_to_supabase(key)
            if success:
                success_count += 1
            else:
                errors.append(f"{key}: {msg}")

    return success_count, total_count, errors


def list_cloud_models() -> List[Dict]:
    """
    List all model results available in Supabase.

    Returns:
        List of dicts with model metadata
    """
    service = get_service()
    if service is None:
        return []

    if not service.is_connected():
        return []

    try:
        results = service.list_saved_results(PAGE_KEY)

        # Enhance with display names
        for result in results:
            key = result.get("result_key", "")
            result["display_name"] = MODEL_DISPLAY_NAMES.get(key, key)

        return results

    except Exception as e:
        logger.error(f"Error listing cloud models: {e}")
        return []


def get_cloud_model_status() -> Dict[str, Dict]:
    """
    Get status showing which models are in cloud vs session.

    Returns:
        Dict mapping session_key to status info:
        {
            "session_key": {
                "in_cloud": bool,
                "in_session": bool,
                "display_name": str,
                "cloud_saved_at": str or None,
                "cloud_size_kb": float or None
            }
        }
    """
    status = {}

    # Get cloud models
    cloud_models = list_cloud_models()
    cloud_keys = {}
    for model in cloud_models:
        key = model.get("result_key", "")
        cloud_keys[key] = {
            "saved_at": model.get("metadata", {}).get("saved_at"),
            "size_bytes": model.get("data_size_bytes", 0),
        }

    # Check all known model keys
    for key in MODEL_SESSION_KEYS:
        in_cloud = key in cloud_keys
        in_session = st.session_state.get(key) is not None

        status[key] = {
            "in_cloud": in_cloud,
            "in_session": in_session,
            "display_name": MODEL_DISPLAY_NAMES.get(key, key),
            "cloud_saved_at": cloud_keys.get(key, {}).get("saved_at"),
            "cloud_size_kb": (cloud_keys.get(key, {}).get("size_bytes", 0) or 0) / 1024,
        }

    return status


def delete_model_from_supabase(session_key: str) -> Tuple[bool, str]:
    """
    Delete a specific model result from Supabase.

    Args:
        session_key: The session state key to delete

    Returns:
        Tuple of (success: bool, message: str)
    """
    service = get_service()
    if service is None:
        return False, "Storage service not available"

    if not service.is_connected():
        return False, "Supabase not connected"

    try:
        success = service.delete_results(PAGE_KEY, session_key)

        if success:
            logger.info(f"Deleted {session_key} from Supabase")
            return True, f"Deleted {MODEL_DISPLAY_NAMES.get(session_key, session_key)}"
        else:
            return False, f"Failed to delete {session_key}"

    except Exception as e:
        error_msg = f"Error deleting {session_key}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
