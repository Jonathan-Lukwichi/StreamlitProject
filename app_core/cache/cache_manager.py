# app_core/cache/cache_manager.py
"""
Cache Manager for persisting processed data, feature engineering results, and model outputs.
Uses pickle for complex objects and parquet for DataFrames.
"""
import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Check if pyarrow is available for parquet support
try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


class CacheManager:
    """Manages caching of session data to disk."""

    # Keys that should be cached
    CACHEABLE_KEYS = [
        # Data Preparation
        "merged_data",
        "processed_df",
        "uploaded_files",  # Just metadata, not actual files
        "data_summary",

        # Feature Engineering
        "fe_cfg",
        "X_engineered",
        "y_engineered",
        "feature_names",
        "train_idx",
        "test_idx",

        # Feature Selection
        "fs_results",
        "fs_selected_features",

        # Model Results
        "arima_results",
        "sarimax_results",
        "ml_mh_results",
        "multi_target_results",

        # Settings
        "target_columns",
        "date_column",
        "aggregated_categories_enabled",
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager with optional custom directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame to detect data changes."""
        if df is None or df.empty:
            return ""
        # Use shape + column names + first/last few rows for faster hashing
        hash_parts = [
            str(df.shape),
            str(list(df.columns)),
            str(df.head(5).values.tobytes()) if len(df) > 0 else "",
            str(df.tail(5).values.tobytes()) if len(df) > 5 else "",
        ]
        return hashlib.md5("".join(hash_parts).encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cached item."""
        return self.cache_dir / f"{key}.pkl"

    def _get_parquet_path(self, key: str) -> Path:
        """Get the parquet file path for DataFrames."""
        return self.cache_dir / f"{key}.parquet"

    def save(self, key: str, value: Any, data_hash: Optional[str] = None) -> bool:
        """
        Save a value to cache.

        Args:
            key: Cache key
            value: Value to cache
            data_hash: Optional hash of source data for invalidation

        Returns:
            True if saved successfully
        """
        try:
            if value is None:
                return False

            # Use parquet for DataFrames if pyarrow is available (faster, smaller)
            # Fall back to pickle if pyarrow is not installed
            if isinstance(value, pd.DataFrame) and HAS_PYARROW:
                path = self._get_parquet_path(key)
                value.to_parquet(path, index=True)
                file_type = "parquet"
            else:
                path = self._get_cache_path(key)
                with open(path, "wb") as f:
                    pickle.dump(value, f)
                file_type = "pickle"

            # Update metadata
            self._metadata[key] = {
                "path": str(path),
                "type": file_type,
                "saved_at": datetime.now().isoformat(),
                "data_hash": data_hash,
            }
            self._save_metadata()
            return True

        except Exception as e:
            print(f"Cache save error for {key}: {e}")
            return False

    def load(self, key: str) -> Optional[Any]:
        """
        Load a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            if key not in self._metadata:
                return None

            meta = self._metadata[key]
            path = Path(meta["path"])

            if not path.exists():
                return None

            if meta["type"] == "parquet" and HAS_PYARROW:
                return pd.read_parquet(path)
            else:
                with open(path, "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            print(f"Cache load error for {key}: {e}")
            return None

    def has(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if key not in self._metadata:
            return False
        path = Path(self._metadata[key]["path"])
        return path.exists()

    def get_data_hash(self, key: str) -> Optional[str]:
        """Get the data hash associated with a cached item."""
        if key in self._metadata:
            return self._metadata[key].get("data_hash")
        return None

    def is_valid(self, key: str, current_hash: str) -> bool:
        """Check if cached data is still valid (hash matches)."""
        cached_hash = self.get_data_hash(key)
        return cached_hash is not None and cached_hash == current_hash

    def delete(self, key: str) -> bool:
        """Delete a cached item."""
        try:
            if key in self._metadata:
                path = Path(self._metadata[key]["path"])
                if path.exists():
                    path.unlink()
                del self._metadata[key]
                self._save_metadata()
            return True
        except Exception as e:
            print(f"Cache delete error for {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cached data."""
        try:
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            self._metadata = {}
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        total_size = 0
        items = []

        for key, meta in self._metadata.items():
            path = Path(meta["path"])
            if path.exists():
                size = path.stat().st_size
                total_size += size
                items.append({
                    "key": key,
                    "type": meta["type"],
                    "size_mb": round(size / (1024 * 1024), 2),
                    "saved_at": meta["saved_at"],
                })

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "item_count": len(items),
            "items": items,
            "cache_dir": str(self.cache_dir),
        }

    def save_session_state(self, session_state: Dict[str, Any], source_df: Optional[pd.DataFrame] = None) -> int:
        """
        Save relevant session state items to cache.

        Args:
            session_state: Streamlit session state dict
            source_df: Source DataFrame for computing hash

        Returns:
            Number of items saved
        """
        data_hash = self._compute_data_hash(source_df) if source_df is not None else None
        saved_count = 0

        for key in self.CACHEABLE_KEYS:
            if key in session_state and session_state[key] is not None:
                if self.save(key, session_state[key], data_hash):
                    saved_count += 1

        return saved_count

    def load_session_state(self) -> Dict[str, Any]:
        """
        Load all cached session state items.

        Returns:
            Dict of key -> value for all cached items
        """
        loaded = {}
        for key in self.CACHEABLE_KEYS:
            value = self.load(key)
            if value is not None:
                loaded[key] = value
        return loaded

    def has_cached_session(self) -> bool:
        """Check if there's any cached session data."""
        return any(self.has(key) for key in self.CACHEABLE_KEYS)


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def save_to_cache(key: str, value: Any, data_hash: Optional[str] = None) -> bool:
    """Convenience function to save to cache."""
    return get_cache_manager().save(key, value, data_hash)


def load_from_cache(key: str) -> Optional[Any]:
    """Convenience function to load from cache."""
    return get_cache_manager().load(key)


def clear_cache() -> bool:
    """Convenience function to clear cache."""
    return get_cache_manager().clear()


def get_cache_info() -> Dict[str, Any]:
    """Convenience function to get cache info."""
    return get_cache_manager().get_info()


def has_cached_data() -> bool:
    """Convenience function to check for cached data."""
    return get_cache_manager().has_cached_session()
