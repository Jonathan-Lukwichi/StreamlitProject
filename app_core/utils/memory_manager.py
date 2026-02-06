"""
Memory Management Utilities for Streamlit Cloud Deployment.

Streamlit Cloud has ~1 GB memory limit. This module provides utilities
to prevent memory exhaustion during ML training operations.
"""

import gc
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# =============================================================================
# MEMORY MONITORING
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with memory metrics (percent, used_gb, available_gb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "percent": mem.percent,
            "used_gb": mem.used / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "total_gb": mem.total / (1024 ** 3),
        }
    except ImportError:
        return {"percent": 0, "used_gb": 0, "available_gb": 0, "total_gb": 0}


def check_memory_threshold(threshold_percent: float = 80.0) -> bool:
    """
    Check if memory usage exceeds threshold.

    Args:
        threshold_percent: Memory usage threshold (default 80%)

    Returns:
        True if memory is OK, False if exceeds threshold
    """
    mem = get_memory_usage()
    if mem["percent"] > threshold_percent:
        logger.warning(f"Memory usage high: {mem['percent']:.1f}%")
        return False
    return True


def memory_guard(threshold_percent: float = 80.0):
    """
    Decorator to check memory before executing function.

    Usage:
        @memory_guard(threshold_percent=75)
        def train_model():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not check_memory_threshold(threshold_percent):
                force_garbage_collection()
                if not check_memory_threshold(threshold_percent):
                    raise MemoryError(
                        f"Memory usage exceeds {threshold_percent}%. "
                        "Reduce model complexity or clear session state."
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# GARBAGE COLLECTION
# =============================================================================

def force_garbage_collection():
    """
    Force aggressive garbage collection.
    Run multiple times to catch circular references.
    """
    gc.collect()
    gc.collect()
    gc.collect()


def cleanup_tensorflow():
    """
    Clean up TensorFlow/Keras memory.
    Call after each model training to prevent memory leaks.
    """
    try:
        import tensorflow as tf
        from keras import backend as K

        # Clear Keras session
        K.clear_session()

        # Clear TensorFlow session
        tf.keras.backend.clear_session()

        # Reset default graph (TF1 compatibility)
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass

        # Force garbage collection
        force_garbage_collection()

        logger.debug("TensorFlow memory cleanup completed")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"TensorFlow cleanup error: {e}")


def cleanup_after_training(model=None):
    """
    Comprehensive cleanup after model training.

    Args:
        model: Optional model object to explicitly delete
    """
    # Delete model if provided
    if model is not None:
        try:
            del model
        except Exception:
            pass

    # TensorFlow-specific cleanup
    cleanup_tensorflow()

    # General garbage collection
    force_garbage_collection()

    # Log memory status
    mem = get_memory_usage()
    logger.debug(f"Post-cleanup memory: {mem['percent']:.1f}%")


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def get_session_state_size() -> Dict[str, float]:
    """
    Estimate memory usage of session state objects.

    Returns:
        Dict mapping key names to approximate size in MB
    """
    import sys
    try:
        import streamlit as st

        sizes = {}
        for key, value in st.session_state.items():
            try:
                size_bytes = sys.getsizeof(value)
                # For containers, estimate nested size
                if hasattr(value, '__len__') and not isinstance(value, str):
                    size_bytes *= 10  # Rough multiplier for nested objects
                sizes[key] = size_bytes / (1024 * 1024)  # Convert to MB
            except Exception:
                sizes[key] = 0

        return dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}


def clear_large_session_objects(threshold_mb: float = 50.0):
    """
    Clear session state objects larger than threshold.

    Args:
        threshold_mb: Size threshold in MB
    """
    try:
        import streamlit as st

        sizes = get_session_state_size()
        cleared = []

        for key, size_mb in sizes.items():
            if size_mb > threshold_mb:
                try:
                    del st.session_state[key]
                    cleared.append(key)
                    logger.info(f"Cleared session state: {key} ({size_mb:.1f} MB)")
                except Exception:
                    pass

        if cleared:
            force_garbage_collection()

        return cleared
    except Exception:
        return []


def lightweight_model_storage(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a lightweight version of model results for session state.
    Removes large arrays that can be recomputed.

    Args:
        results: Full model results dictionary

    Returns:
        Lightweight version without large arrays
    """
    import numpy as np

    lightweight = {}

    for key, value in results.items():
        # Skip large numpy arrays
        if isinstance(value, np.ndarray):
            if value.size > 10000:  # Skip arrays with >10k elements
                lightweight[f"{key}_shape"] = value.shape
                continue

        # For nested dicts, recurse
        if isinstance(value, dict):
            lightweight[key] = lightweight_model_storage(value)
            continue

        # Keep everything else
        lightweight[key] = value

    return lightweight


# =============================================================================
# STREAMLIT CLOUD CONFIGURATION
# =============================================================================

def configure_for_cloud():
    """
    Apply configurations optimized for Streamlit Cloud.
    Call this at app startup.
    """
    import os

    # Limit TensorFlow threads
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Disable TensorFlow GPU (not available on Cloud anyway)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Limit memory growth for TensorFlow
    try:
        import tensorflow as tf

        # CPU only
        tf.config.set_visible_devices([], 'GPU')

        # Enable memory growth (prevents TF from grabbing all memory)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except Exception:
        pass

    # Set XGBoost to single-threaded
    os.environ["OMP_NUM_THREADS"] = "1"

    logger.info("Configured for Streamlit Cloud deployment")


def is_cloud_environment() -> bool:
    """
    Detect if running on Streamlit Cloud.

    Returns:
        True if running on Streamlit Cloud
    """
    import os

    # Streamlit Cloud sets these environment variables
    cloud_indicators = [
        "STREAMLIT_SHARING_MODE",
        "IS_STREAMLIT_CLOUD",
        "HOSTNAME",  # Often contains 'streamlit' on cloud
    ]

    for var in cloud_indicators:
        if var in os.environ:
            if "streamlit" in os.environ.get(var, "").lower():
                return True

    # Check for limited memory (< 2GB suggests cloud)
    mem = get_memory_usage()
    if mem["total_gb"] < 2.0:
        return True

    return False


def get_cloud_safe_config() -> Dict[str, Any]:
    """
    Get training configuration safe for Streamlit Cloud.

    Returns:
        Dict with cloud-safe hyperparameters
    """
    return {
        "n_jobs": 1,  # Single-threaded
        "n_trials": 10,  # Reduced Optuna trials
        "timeout": 300,  # 5 minute timeout
        "batch_size": 32,  # Smaller batches
        "epochs": 50,  # Reduced epochs
        "early_stopping_patience": 5,
        "max_horizons": 3,  # Reduce horizons if needed
        "store_training_data": False,  # Don't store X_train/X_val
        "mc_samples": 10,  # Reduced Monte Carlo samples
    }


# =============================================================================
# TRAINING CONTEXT MANAGER
# =============================================================================

class CloudSafeTraining:
    """
    Context manager for cloud-safe model training.

    Usage:
        with CloudSafeTraining() as config:
            model.fit(X, y, **config)
    """

    def __init__(self, memory_threshold: float = 75.0):
        self.memory_threshold = memory_threshold
        self.initial_memory = None

    def __enter__(self):
        # Check memory before training
        self.initial_memory = get_memory_usage()

        if self.initial_memory["percent"] > self.memory_threshold:
            force_garbage_collection()
            cleanup_tensorflow()

        # Return cloud-safe config
        if is_cloud_environment():
            return get_cloud_safe_config()
        return {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup after training
        cleanup_tensorflow()
        force_garbage_collection()

        # Log memory change
        final_memory = get_memory_usage()
        delta = final_memory["percent"] - self.initial_memory["percent"]

        if delta > 10:
            logger.warning(f"Training increased memory by {delta:.1f}%")

        return False  # Don't suppress exceptions
