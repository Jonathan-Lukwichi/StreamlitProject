# =============================================================================
# app_core/data/results_storage_service.py
# Persistent storage service for HealthForecast AI pipeline results.
# Handles serialization/deserialization of complex Python objects (DataFrames,
# NumPy arrays, etc.) and saves them to Supabase for cross-session persistence.
# Based on memory1.md implementation guide.
# =============================================================================

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app_core.data.supabase_client import get_cached_supabase_client


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy, Pandas, and other non-standard Python types.
    Encodes objects with type markers for proper reconstruction on load.
    """

    def default(self, obj: Any) -> Any:
        # === NumPy Types ===
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape)
            }

        # === Pandas Types ===
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "dataframe",
                "data": obj.to_json(orient='split', date_format='iso'),
                "shape": list(obj.shape),
                "columns": list(obj.columns),
                "dtypes": {col: str(dtype) for col, dtype in obj.dtypes.items()}
            }
        if isinstance(obj, pd.Series):
            return {
                "__type__": "series",
                "data": obj.to_json(orient='split', date_format='iso'),
                "name": obj.name,
                "dtype": str(obj.dtype)
            }
        if isinstance(obj, pd.Timestamp):
            return {"__type__": "pd_timestamp", "data": obj.isoformat()}
        if isinstance(obj, pd.Timedelta):
            return {"__type__": "pd_timedelta", "data": str(obj)}
        if isinstance(obj, pd.Interval):
            return {"__type__": "pd_interval", "data": str(obj)}
        if isinstance(obj, (pd.Index, pd.RangeIndex)):
            return {"__type__": "pd_index", "data": obj.tolist()}

        # === Handle NaN/None ===
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass

        # === Datetime Types ===
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}

        # === Sets and Tuples ===
        if isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": list(obj)}

        # === Bytes ===
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "data": obj.decode('utf-8', errors='replace')}

        # === Sklearn Models (non-serializable) ===
        # Check for sklearn-like objects by module name
        obj_module = getattr(type(obj), '__module__', '')
        if 'sklearn' in obj_module:
            return {
                "__type__": "sklearn_model",
                "class": type(obj).__name__,
                "params": getattr(obj, 'get_params', lambda: {})()
            }

        # === Dataclasses ===
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                "__type__": "dataclass",
                "class": type(obj).__name__,
                "data": dataclasses.asdict(obj)
            }

        # === Objects with __dict__ (custom classes) ===
        if hasattr(obj, '__dict__') and not isinstance(obj, type):
            try:
                return {
                    "__type__": "object",
                    "class": type(obj).__name__,
                    "data": obj.__dict__
                }
            except Exception:
                pass

        # === Default ===
        try:
            return super().default(obj)
        except TypeError:
            # Last resort: convert to string
            return {"__type__": "str_fallback", "data": str(obj)}


def decode_special_types(obj: Any) -> Any:
    """
    Recursively decode special types from JSON back to Python objects.
    Handles all types encoded by NpEncoder.
    """
    if isinstance(obj, dict):
        type_marker = obj.get("__type__")

        if type_marker == "dataframe":
            try:
                df = pd.read_json(obj["data"], orient='split')
                return df
            except Exception:
                return obj

        elif type_marker == "series":
            try:
                series = pd.read_json(obj["data"], orient='split', typ='series')
                series.name = obj.get("name")
                return series
            except Exception:
                return obj

        elif type_marker == "ndarray":
            try:
                arr = np.array(obj["data"])
                dtype = obj.get("dtype")
                if dtype:
                    try:
                        arr = arr.astype(dtype)
                    except (TypeError, ValueError):
                        pass
                return arr
            except Exception:
                return obj

        elif type_marker == "pd_timestamp":
            try:
                return pd.Timestamp(obj["data"])
            except Exception:
                return obj

        elif type_marker == "pd_timedelta":
            try:
                return pd.Timedelta(obj["data"])
            except Exception:
                return obj

        elif type_marker == "datetime":
            try:
                return datetime.fromisoformat(obj["data"])
            except Exception:
                return obj

        elif type_marker == "set":
            return set(obj["data"])

        elif type_marker == "tuple":
            return tuple(obj["data"])

        elif type_marker == "pd_index":
            return pd.Index(obj["data"])

        elif type_marker == "bytes":
            return obj["data"].encode('utf-8')

        elif type_marker == "str_fallback":
            return obj["data"]

        else:
            # Regular dict - recurse into values
            return {k: decode_special_types(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [decode_special_types(item) for item in obj]

    return obj


class ResultsStorageService:
    """
    Service for persisting HealthForecast AI pipeline results to Supabase.

    Supports:
    - Saving/loading any Python object (DataFrames, dicts, lists, numpy arrays)
    - Upsert functionality (update existing or insert new)
    - Listing saved results with metadata
    - Deleting results by page or specific key
    - User isolation via session-based IDs
    """

    TABLE_NAME = "app_results"

    def __init__(self, client=None):
        """Initialize with Supabase client."""
        self.client = client or get_cached_supabase_client()

    def is_connected(self) -> bool:
        """Check if Supabase client is available."""
        return self.client is not None

    def _get_user_id(self) -> str:
        """
        Get current user ID.
        Uses authenticated username for persistence across sessions.
        Falls back to session-based ID if not authenticated.
        """
        # First try to get the authenticated username
        username = st.session_state.get("username")
        if username:
            return f"user_{username}"

        # Fallback to session-based ID (not persistent across sessions)
        if "user_session_id" not in st.session_state:
            import uuid
            st.session_state["user_session_id"] = str(uuid.uuid4())[:12]
        return st.session_state["user_session_id"]

    def save_results(
        self,
        page_key: str,
        result_key: str,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Save results to Supabase using upsert (update or insert).

        Args:
            page_key: Page identifier (e.g., 'Upload Data', 'Train Models')
            result_key: Result identifier (e.g., 'prepared_data', 'ml_results')
            data: Any Python object to save
            metadata: Optional additional metadata

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_connected():
            return False, "Supabase not connected"

        user_id = self._get_user_id()

        try:
            # Serialize data to JSON
            json_data = json.loads(json.dumps(data, cls=NpEncoder))
            json_string = json.dumps(json_data)
            data_size = len(json_string.encode('utf-8'))

            # Build comprehensive metadata
            full_metadata = {
                "saved_at": datetime.now().isoformat(),
                "data_type": type(data).__name__,
                "size_bytes": data_size,
            }

            # Add DataFrame-specific metadata
            if isinstance(data, pd.DataFrame):
                full_metadata["shape"] = list(data.shape)
                full_metadata["columns"] = list(data.columns)[:20]  # First 20 columns
                full_metadata["memory_mb"] = data.memory_usage(deep=True).sum() / 1024 / 1024

            # Add dict/list length
            if isinstance(data, (dict, list)):
                full_metadata["length"] = len(data)

            # Merge with user-provided metadata
            if metadata:
                full_metadata.update(metadata)

            # Build record
            record = {
                "user_id": user_id,
                "page_key": page_key,
                "result_key": result_key,
                "result_data": json_data,
                "metadata": full_metadata,
                "data_size_bytes": data_size
            }

            # Delete existing record first (workaround for upsert on non-primary key unique constraints)
            self.client.table(self.TABLE_NAME).delete().eq(
                "user_id", user_id
            ).eq(
                "page_key", page_key
            ).eq(
                "result_key", result_key
            ).execute()

            # Insert new record
            self.client.table(self.TABLE_NAME).insert(record).execute()

            return True, f"Saved {result_key} ({data_size / 1024:.1f} KB)"

        except Exception as e:
            return False, f"Save failed: {str(e)}"

    def load_results(self, page_key: str, result_key: str) -> Optional[Any]:
        """
        Load results from Supabase.

        Args:
            page_key: Page identifier
            result_key: Result identifier

        Returns:
            Deserialized Python object or None if not found
        """
        if not self.is_connected():
            return None

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("result_data") \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .eq("result_key", result_key) \
                .limit(1) \
                .execute()

            if not response.data:
                return None

            # Parse and decode special types
            raw_data = response.data[0]['result_data']
            return decode_special_types(raw_data)

        except Exception as e:
            print(f"Load error for {page_key}/{result_key}: {e}")
            return None

    def load_all_for_page(self, page_key: str) -> Dict[str, Any]:
        """
        Load all saved results for a specific page.

        Returns:
            Dict mapping result_key to deserialized data
        """
        if not self.is_connected():
            return {}

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("result_key, result_data") \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .execute()

            results = {}
            for record in response.data or []:
                try:
                    raw_data = record['result_data']
                    results[record['result_key']] = decode_special_types(raw_data)
                except Exception:
                    continue

            return results

        except Exception as e:
            print(f"Load all error for {page_key}: {e}")
            return {}

    def list_saved_results(self, page_key: Optional[str] = None) -> List[Dict]:
        """
        List all saved results with metadata.

        Args:
            page_key: Optional filter by page

        Returns:
            List of result metadata dicts
        """
        if not self.is_connected():
            return []

        user_id = self._get_user_id()

        try:
            query = self.client.table(self.TABLE_NAME) \
                .select("page_key, result_key, metadata, created_at, updated_at, data_size_bytes") \
                .eq("user_id", user_id)

            if page_key:
                query = query.eq("page_key", page_key)

            response = query.order("updated_at", desc=True).execute()
            return response.data or []

        except Exception:
            return []

    def delete_results(self, page_key: str, result_key: str) -> bool:
        """Delete a specific result."""
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .eq("result_key", result_key) \
                .execute()
            return True
        except Exception:
            return False

    def delete_all_for_page(self, page_key: str) -> bool:
        """Delete all results for a page."""
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .execute()
            return True
        except Exception:
            return False

    def delete_all_user_results(self) -> bool:
        """Delete ALL results for current user (use with caution)."""
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .execute()
            return True
        except Exception:
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for current user."""
        if not self.is_connected():
            return {}

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("page_key, data_size_bytes") \
                .eq("user_id", user_id) \
                .execute()

            total_bytes = sum(r.get('data_size_bytes', 0) or 0 for r in response.data or [])
            by_page = {}
            for r in response.data or []:
                page = r['page_key']
                by_page[page] = by_page.get(page, 0) + (r.get('data_size_bytes', 0) or 0)

            return {
                "total_bytes": total_bytes,
                "total_mb": total_bytes / 1024 / 1024,
                "total_items": len(response.data or []),
                "by_page": by_page
            }
        except Exception:
            return {}


# === Singleton Pattern ===
_storage_service_instance: Optional[ResultsStorageService] = None


def get_storage_service() -> ResultsStorageService:
    """Get singleton ResultsStorageService instance."""
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = ResultsStorageService()
    return _storage_service_instance


def reset_storage_service():
    """Reset singleton (useful for testing or reconnection)."""
    global _storage_service_instance
    _storage_service_instance = None
