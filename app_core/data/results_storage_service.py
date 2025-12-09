# =============================================================================
# app_core/data/results_storage_service.py
# Results Storage Service for HealthForecast AI
# Stores and retrieves page results from Supabase for session persistence
# =============================================================================

from __future__ import annotations
import streamlit as st
from typing import Optional, Dict, Any, List
import pandas as pd
import json
import hashlib
from datetime import datetime
import pickle
import base64

from app_core.data.supabase_client import SupabaseService, get_cached_supabase_client


# Table name for storing results
RESULTS_TABLE = "pipeline_results"


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash to uniquely identify a dataset.
    Uses shape, columns, and sample data for fast hashing.
    """
    if df is None or df.empty:
        return ""

    hash_parts = [
        str(df.shape),
        str(sorted(df.columns.tolist())),
        str(df.head(3).values.tobytes()) if len(df) > 0 else "",
        str(df.tail(3).values.tobytes()) if len(df) > 3 else "",
    ]
    return hashlib.md5("".join(hash_parts).encode()).hexdigest()[:16]


def _serialize_value(value: Any) -> str:
    """Serialize a Python object to base64-encoded pickle string."""
    try:
        pickled = pickle.dumps(value)
        return base64.b64encode(pickled).decode('utf-8')
    except Exception as e:
        st.warning(f"Could not serialize value: {e}")
        return ""


def _deserialize_value(encoded: str) -> Any:
    """Deserialize a base64-encoded pickle string back to Python object."""
    try:
        if not encoded:
            return None
        pickled = base64.b64decode(encoded.encode('utf-8'))
        return pickle.loads(pickled)
    except Exception as e:
        st.warning(f"Could not deserialize value: {e}")
        return None


class ResultsStorageService:
    """
    Service for storing and retrieving pipeline results from Supabase.

    Each page can save its results with a dataset hash to allow:
    - Loading previous results when returning to a page
    - Clearing results when starting with a new dataset
    - Persisting work across browser sessions
    """

    # Page types and their associated session state keys
    PAGE_KEYS = {
        "data_preparation": [
            "merged_data",
            "processed_df",
            "data_summary",
            "target_columns",
            "date_column",
            "aggregated_categories_enabled",
        ],
        "feature_engineering": [
            "fe_cfg",
            "X_engineered",
            "y_engineered",
            "feature_names",
            "train_idx",
            "cal_idx",
            "test_idx",
            "temporal_split",
            "split_result",
        ],
        "feature_selection": [
            "fs_results",
            "fs_selected_features",
            "feature_selection",
        ],
        "modeling_hub": [
            "ml_mh_results",
            "multi_target_results",
            "forecast_results",
            "cqr_results",
        ],
        "benchmarks": [
            "arima_results",
            "sarimax_results",
            "backtest_results",
        ],
    }

    def __init__(self):
        """Initialize the results storage service."""
        self.client = get_cached_supabase_client()
        self._service = SupabaseService(RESULTS_TABLE) if self.client else None

    def is_connected(self) -> bool:
        """Check if Supabase client is available."""
        return self.client is not None and self._service is not None

    def _get_user_id(self) -> str:
        """
        Get a unique user identifier.
        Uses browser session ID or generates one.
        """
        if "user_session_id" not in st.session_state:
            # Generate a simple session ID
            import uuid
            st.session_state["user_session_id"] = str(uuid.uuid4())[:8]
        return st.session_state["user_session_id"]

    def save_page_results(
        self,
        page_type: str,
        dataset_hash: Optional[str] = None,
        custom_keys: Optional[List[str]] = None,
    ) -> bool:
        """
        Save current session state results for a page to Supabase.

        Args:
            page_type: One of 'data_preparation', 'feature_engineering',
                       'feature_selection', 'modeling_hub', 'benchmarks'
            dataset_hash: Hash to identify the dataset (computed from merged_data if not provided)
            custom_keys: Additional session state keys to save

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            st.warning("⚠️ Supabase not connected. Results not saved.")
            return False

        if page_type not in self.PAGE_KEYS:
            st.error(f"Unknown page type: {page_type}")
            return False

        # Compute dataset hash if not provided
        if dataset_hash is None:
            merged = st.session_state.get("merged_data")
            if merged is not None and isinstance(merged, pd.DataFrame):
                dataset_hash = _compute_dataset_hash(merged)
            else:
                dataset_hash = "no_data"

        # Collect values to save
        keys_to_save = self.PAGE_KEYS[page_type].copy()
        if custom_keys:
            keys_to_save.extend(custom_keys)

        results_data = {}
        for key in keys_to_save:
            if key in st.session_state and st.session_state[key] is not None:
                value = st.session_state[key]
                # Serialize the value
                serialized = _serialize_value(value)
                if serialized:
                    results_data[key] = serialized

        if not results_data:
            st.info("No results to save for this page.")
            return False

        # Prepare record
        user_id = self._get_user_id()
        record = {
            "user_id": user_id,
            "page_type": page_type,
            "dataset_hash": dataset_hash,
            "results_json": json.dumps(results_data),
            "saved_at": datetime.now().isoformat(),
            "keys_saved": list(results_data.keys()),
        }

        try:
            # Delete existing record for this user/page/dataset combination
            self.client.table(RESULTS_TABLE).delete().eq(
                "user_id", user_id
            ).eq(
                "page_type", page_type
            ).eq(
                "dataset_hash", dataset_hash
            ).execute()

            # Insert new record
            self.client.table(RESULTS_TABLE).insert(record).execute()
            return True

        except Exception as e:
            st.error(f"Error saving results: {e}")
            return False

    def load_page_results(
        self,
        page_type: str,
        dataset_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load saved results for a page from Supabase.

        Args:
            page_type: The page type to load results for
            dataset_hash: Hash to identify the dataset

        Returns:
            Dictionary of loaded session state values
        """
        if not self.is_connected():
            return {}

        # Compute dataset hash if not provided
        if dataset_hash is None:
            merged = st.session_state.get("merged_data")
            if merged is not None and isinstance(merged, pd.DataFrame):
                dataset_hash = _compute_dataset_hash(merged)
            else:
                dataset_hash = "no_data"

        user_id = self._get_user_id()

        try:
            response = (
                self.client.table(RESULTS_TABLE)
                .select("*")
                .eq("user_id", user_id)
                .eq("page_type", page_type)
                .eq("dataset_hash", dataset_hash)
                .order("saved_at", desc=True)
                .limit(1)
                .execute()
            )

            if not response.data:
                return {}

            record = response.data[0]
            results_json = json.loads(record["results_json"])

            # Deserialize values
            loaded = {}
            for key, serialized in results_json.items():
                value = _deserialize_value(serialized)
                if value is not None:
                    loaded[key] = value

            return loaded

        except Exception as e:
            st.error(f"Error loading results: {e}")
            return {}

    def apply_loaded_results(self, loaded: Dict[str, Any]) -> int:
        """
        Apply loaded results to session state.

        Args:
            loaded: Dictionary of key-value pairs to apply

        Returns:
            Number of keys applied
        """
        count = 0
        for key, value in loaded.items():
            st.session_state[key] = value
            count += 1
        return count

    def has_saved_results(
        self,
        page_type: str,
        dataset_hash: Optional[str] = None,
    ) -> bool:
        """
        Check if there are saved results for a page.

        Args:
            page_type: The page type to check
            dataset_hash: Hash to identify the dataset

        Returns:
            True if saved results exist
        """
        if not self.is_connected():
            return False

        # Compute dataset hash if not provided
        if dataset_hash is None:
            merged = st.session_state.get("merged_data")
            if merged is not None and isinstance(merged, pd.DataFrame):
                dataset_hash = _compute_dataset_hash(merged)
            else:
                dataset_hash = "no_data"

        user_id = self._get_user_id()

        try:
            response = (
                self.client.table(RESULTS_TABLE)
                .select("id")
                .eq("user_id", user_id)
                .eq("page_type", page_type)
                .eq("dataset_hash", dataset_hash)
                .limit(1)
                .execute()
            )
            return len(response.data) > 0

        except Exception:
            return False

    def get_saved_info(
        self,
        page_type: str,
        dataset_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about saved results.

        Returns:
            Dict with saved_at, keys_saved, etc. or None if not found
        """
        if not self.is_connected():
            return None

        if dataset_hash is None:
            merged = st.session_state.get("merged_data")
            if merged is not None and isinstance(merged, pd.DataFrame):
                dataset_hash = _compute_dataset_hash(merged)
            else:
                dataset_hash = "no_data"

        user_id = self._get_user_id()

        try:
            response = (
                self.client.table(RESULTS_TABLE)
                .select("saved_at, keys_saved")
                .eq("user_id", user_id)
                .eq("page_type", page_type)
                .eq("dataset_hash", dataset_hash)
                .order("saved_at", desc=True)
                .limit(1)
                .execute()
            )

            if response.data:
                return response.data[0]
            return None

        except Exception:
            return None

    def clear_page_results(
        self,
        page_type: str,
        dataset_hash: Optional[str] = None,
    ) -> bool:
        """
        Clear saved results for a specific page.

        Args:
            page_type: The page type to clear
            dataset_hash: Hash to identify the dataset (None = clear all for this page)

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            query = (
                self.client.table(RESULTS_TABLE)
                .delete()
                .eq("user_id", user_id)
                .eq("page_type", page_type)
            )

            if dataset_hash:
                query = query.eq("dataset_hash", dataset_hash)

            query.execute()
            return True

        except Exception as e:
            st.error(f"Error clearing results: {e}")
            return False

    def clear_all_results(self, dataset_hash: Optional[str] = None) -> bool:
        """
        Clear all saved results for this user.

        Args:
            dataset_hash: Optional - only clear results for this dataset

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            query = (
                self.client.table(RESULTS_TABLE)
                .delete()
                .eq("user_id", user_id)
            )

            if dataset_hash:
                query = query.eq("dataset_hash", dataset_hash)

            query.execute()
            return True

        except Exception as e:
            st.error(f"Error clearing all results: {e}")
            return False

    def get_all_saved_pages(self, dataset_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of all pages with saved results.

        Args:
            dataset_hash: Optional - filter by dataset

        Returns:
            List of dicts with page_type, saved_at info
        """
        if not self.is_connected():
            return []

        user_id = self._get_user_id()

        try:
            query = (
                self.client.table(RESULTS_TABLE)
                .select("page_type, saved_at, dataset_hash, keys_saved")
                .eq("user_id", user_id)
            )

            if dataset_hash:
                query = query.eq("dataset_hash", dataset_hash)

            response = query.order("saved_at", desc=True).execute()
            return response.data if response.data else []

        except Exception:
            return []


# Singleton instance
_results_storage: Optional[ResultsStorageService] = None


def get_results_storage() -> ResultsStorageService:
    """Get the singleton results storage service instance."""
    global _results_storage
    if _results_storage is None:
        _results_storage = ResultsStorageService()
    return _results_storage


def compute_current_dataset_hash() -> str:
    """Compute hash for the current dataset in session state."""
    merged = st.session_state.get("merged_data")
    if merged is not None and isinstance(merged, pd.DataFrame):
        return _compute_dataset_hash(merged)
    return "no_data"
