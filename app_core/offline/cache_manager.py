# =============================================================================
# app_core/offline/cache_manager.py
# Cache Manager for Files and ML Models
# =============================================================================
"""
CacheManager - Handles local caching of files, datasets, and trained ML models.

Features:
- File-based caching with metadata tracking
- ML model serialization (joblib, pickle)
- Automatic cache cleanup
- Cache integrity verification
"""

from __future__ import annotations
import os
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import joblib for model serialization
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle
    JOBLIB_AVAILABLE = False


class CacheManager:
    """
    Manages local file and model caching for offline operation.

    Directory Structure:
    -------------------
    local_data/
    ├── cache/
    │   ├── datasets/          # Cached CSV/Parquet files
    │   ├── models/            # Trained ML models
    │   ├── results/           # Model results and metrics
    │   └── temp/              # Temporary files
    └── cache_index.json       # Metadata about cached items
    """

    DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "local_data" / "cache"
    CACHE_INDEX_FILE = "cache_index.json"

    # Cache expiry settings (days)
    DEFAULT_EXPIRY = {
        "datasets": 30,
        "models": 90,
        "results": 30,
        "temp": 1,
    }

    _instance: Optional[CacheManager] = None

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self._index: Dict[str, Dict] = {}
        self._ensure_directories()
        self._load_index()

    @classmethod
    def get_instance(cls, cache_dir: Optional[Path] = None) -> CacheManager:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = CacheManager(cache_dir)
        return cls._instance

    def _ensure_directories(self) -> None:
        """Create cache directory structure."""
        subdirs = ["datasets", "models", "results", "temp", "uploaded"]
        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> None:
        """Load cache index from file."""
        index_path = self.cache_dir / self.CACHE_INDEX_FILE
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to file."""
        index_path = self.cache_dir / self.CACHE_INDEX_FILE
        try:
            with open(index_path, "w") as f:
                json.dump(self._index, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Error saving cache index: {e}")

    def _generate_key(self, name: str, category: str) -> str:
        """Generate a unique cache key."""
        return f"{category}:{name}"

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for integrity checking."""
        if not file_path.exists():
            return ""

        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # =========================================================================
    # DATASET CACHING
    # =========================================================================

    def cache_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        metadata: Optional[Dict] = None,
        format: str = "parquet"
    ) -> str:
        """
        Cache a pandas DataFrame.

        Args:
            df: DataFrame to cache
            name: Unique name for the dataset
            metadata: Optional metadata to store
            format: Storage format ('parquet' or 'csv')

        Returns:
            Cache key for retrieval
        """
        key = self._generate_key(name, "datasets")

        # Determine file path
        ext = ".parquet" if format == "parquet" else ".csv"
        file_path = self.cache_dir / "datasets" / f"{name}{ext}"

        # Save DataFrame
        try:
            if format == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"Error caching DataFrame: {e}")
            raise

        # Update index
        self._index[key] = {
            "name": name,
            "category": "datasets",
            "file_path": str(file_path),
            "format": format,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": datetime.now().isoformat(),
            "accessed_at": datetime.now().isoformat(),
            "file_hash": self._get_file_hash(file_path),
            "metadata": metadata or {},
        }
        self._save_index()

        logger.info(f"Cached DataFrame '{name}' ({len(df)} rows)")
        return key

    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached DataFrame.

        Args:
            name: Name of the cached dataset

        Returns:
            DataFrame if found, None otherwise
        """
        key = self._generate_key(name, "datasets")

        if key not in self._index:
            return None

        info = self._index[key]
        file_path = Path(info["file_path"])

        if not file_path.exists():
            # Remove stale index entry
            del self._index[key]
            self._save_index()
            return None

        try:
            if info.get("format") == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            # Update access time
            self._index[key]["accessed_at"] = datetime.now().isoformat()
            self._save_index()

            return df
        except Exception as e:
            logger.error(f"Error loading cached DataFrame: {e}")
            return None

    def has_dataframe(self, name: str) -> bool:
        """Check if a DataFrame is cached."""
        key = self._generate_key(name, "datasets")
        return key in self._index and Path(self._index[key]["file_path"]).exists()

    # =========================================================================
    # ML MODEL CACHING
    # =========================================================================

    def cache_model(
        self,
        model: Any,
        name: str,
        model_type: str,
        metrics: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> str:
        """
        Cache a trained ML model.

        Args:
            model: Trained model object (sklearn, XGBoost, etc.)
            name: Unique name for the model
            model_type: Type of model (e.g., 'XGBoost', 'ARIMA')
            metrics: Model performance metrics
            parameters: Model hyperparameters
            feature_columns: List of feature column names
            target_column: Target column name

        Returns:
            Cache key for retrieval
        """
        key = self._generate_key(name, "models")
        file_path = self.cache_dir / "models" / f"{name}.joblib"

        # Save model
        try:
            if JOBLIB_AVAILABLE:
                joblib.dump(model, file_path)
            else:
                import pickle
                with open(file_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Error caching model: {e}")
            raise

        # Update index
        self._index[key] = {
            "name": name,
            "category": "models",
            "file_path": str(file_path),
            "model_type": model_type,
            "metrics": metrics or {},
            "parameters": parameters or {},
            "feature_columns": feature_columns or [],
            "target_column": target_column,
            "created_at": datetime.now().isoformat(),
            "accessed_at": datetime.now().isoformat(),
            "file_hash": self._get_file_hash(file_path),
        }
        self._save_index()

        logger.info(f"Cached model '{name}' (type: {model_type})")
        return key

    def get_model(self, name: str) -> Optional[Any]:
        """
        Retrieve a cached model.

        Args:
            name: Name of the cached model

        Returns:
            Model object if found, None otherwise
        """
        key = self._generate_key(name, "models")

        if key not in self._index:
            return None

        info = self._index[key]
        file_path = Path(info["file_path"])

        if not file_path.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            if JOBLIB_AVAILABLE:
                model = joblib.load(file_path)
            else:
                import pickle
                with open(file_path, "rb") as f:
                    model = pickle.load(f)

            self._index[key]["accessed_at"] = datetime.now().isoformat()
            self._save_index()

            return model
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            return None

    def get_model_info(self, name: str) -> Optional[Dict]:
        """Get metadata about a cached model."""
        key = self._generate_key(name, "models")
        return self._index.get(key)

    def list_models(self) -> List[Dict]:
        """List all cached models."""
        return [
            info for key, info in self._index.items()
            if info.get("category") == "models"
        ]

    def has_model(self, name: str) -> bool:
        """Check if a model is cached."""
        key = self._generate_key(name, "models")
        return key in self._index and Path(self._index[key]["file_path"]).exists()

    # =========================================================================
    # RESULTS CACHING
    # =========================================================================

    def cache_results(
        self,
        results: Dict[str, Any],
        name: str,
        result_type: str = "forecast"
    ) -> str:
        """
        Cache model results or forecasts.

        Args:
            results: Results dictionary
            name: Unique name for the results
            result_type: Type of results

        Returns:
            Cache key for retrieval
        """
        key = self._generate_key(name, "results")
        file_path = self.cache_dir / "results" / f"{name}.json"

        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif pd.isna(obj):
                return None
            return obj

        # Clean results for JSON
        clean_results = json.loads(
            json.dumps(results, default=convert)
        )

        try:
            with open(file_path, "w") as f:
                json.dump(clean_results, f, indent=2)
        except Exception as e:
            logger.error(f"Error caching results: {e}")
            raise

        self._index[key] = {
            "name": name,
            "category": "results",
            "file_path": str(file_path),
            "result_type": result_type,
            "created_at": datetime.now().isoformat(),
            "accessed_at": datetime.now().isoformat(),
        }
        self._save_index()

        return key

    def get_results(self, name: str) -> Optional[Dict]:
        """Retrieve cached results."""
        key = self._generate_key(name, "results")

        if key not in self._index:
            return None

        info = self._index[key]
        file_path = Path(info["file_path"])

        if not file_path.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cached results: {e}")
            return None

    # =========================================================================
    # UPLOADED FILES CACHING
    # =========================================================================

    def cache_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str
    ) -> str:
        """
        Cache an uploaded file.

        Args:
            file_content: Raw file content
            filename: Original filename
            file_type: File type/extension

        Returns:
            Path to cached file
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
        cached_filename = f"{timestamp}_{safe_name}"

        file_path = self.cache_dir / "uploaded" / cached_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        key = self._generate_key(cached_filename, "uploaded")
        self._index[key] = {
            "name": cached_filename,
            "original_name": filename,
            "category": "uploaded",
            "file_path": str(file_path),
            "file_type": file_type,
            "size_bytes": len(file_content),
            "created_at": datetime.now().isoformat(),
            "file_hash": self._get_file_hash(file_path),
        }
        self._save_index()

        return str(file_path)

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_items": len(self._index),
            "by_category": {},
            "total_size_bytes": 0,
        }

        for key, info in self._index.items():
            category = info.get("category", "unknown")
            if category not in stats["by_category"]:
                stats["by_category"][category] = {"count": 0, "size_bytes": 0}

            stats["by_category"][category]["count"] += 1

            file_path = Path(info.get("file_path", ""))
            if file_path.exists():
                size = file_path.stat().st_size
                stats["by_category"][category]["size_bytes"] += size
                stats["total_size_bytes"] += size

        return stats

    def cleanup_expired(self) -> int:
        """
        Remove expired cache items.

        Returns:
            Number of items removed
        """
        removed = 0
        now = datetime.now()

        for key in list(self._index.keys()):
            info = self._index[key]
            category = info.get("category", "temp")
            expiry_days = self.DEFAULT_EXPIRY.get(category, 30)

            created_str = info.get("created_at")
            if created_str:
                created = datetime.fromisoformat(created_str)
                if now - created > timedelta(days=expiry_days):
                    self.delete(info.get("name", ""), category)
                    removed += 1

        return removed

    def delete(self, name: str, category: str) -> bool:
        """Delete a cached item."""
        key = self._generate_key(name, category)

        if key not in self._index:
            return False

        info = self._index[key]
        file_path = Path(info.get("file_path", ""))

        if file_path.exists():
            try:
                file_path.unlink()
            except IOError as e:
                logger.error(f"Error deleting cache file: {e}")
                return False

        del self._index[key]
        self._save_index()
        return True

    def clear_category(self, category: str) -> int:
        """Clear all items in a category."""
        removed = 0
        for key in list(self._index.keys()):
            if self._index[key].get("category") == category:
                name = self._index[key].get("name", "")
                if self.delete(name, category):
                    removed += 1
        return removed

    def clear_all(self) -> None:
        """Clear entire cache."""
        for subdir in ["datasets", "models", "results", "temp", "uploaded"]:
            dir_path = self.cache_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True)

        self._index = {}
        self._save_index()
        logger.info("Cache cleared")


# Singleton accessor
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global CacheManager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager.get_instance()
    return _cache_manager
