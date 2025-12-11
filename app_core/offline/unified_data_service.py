# =============================================================================
# app_core/offline/unified_data_service.py
# Unified Data Service - Single API for Online/Offline Operations
# =============================================================================
"""
UnifiedDataService - The primary API for all data operations.

This service provides a unified interface that automatically handles:
- Online mode: Direct Supabase operations with local caching
- Offline mode: SQLite operations with sync queue
- Seamless fallback when connection changes
- Automatic sync when coming back online

Usage:
------
from app_core.offline import get_data_service

service = get_data_service()

# Fetch data (auto-selects source)
df = service.fetch_patient_data()

# Save data (auto-queues for sync if offline)
service.save_patient_data(df)

# Check status
print(f"Online: {service.is_online}")
print(f"Pending sync: {service.pending_sync_count}")
"""

from __future__ import annotations
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class UnifiedDataService:
    """
    Unified data service providing a single API for online/offline operations.

    This is the main entry point for all data operations in the application.
    It automatically handles:
    - Connection state detection
    - Data source selection (Supabase vs SQLite)
    - Caching for performance
    - Sync queue management
    """

    _instance: Optional[UnifiedDataService] = None
    _lock = threading.Lock()

    # Table mappings between local and remote
    TABLE_MAPPING = {
        "patient_arrivals": "patient_arrivals",
        "inventory_data": "inventory_management",
        "financial_data": "financial_data",
        "uploaded_datasets": "uploaded_datasets",
        "trained_models": "trained_models",
    }

    def __init__(self):
        """Initialize the unified data service."""
        self._connection_manager = None
        self._local_db = None
        self._cache_manager = None
        self._sync_engine = None
        self._supabase_service = None
        self._initialized = False
        self._callbacks: List[Callable[[bool], None]] = []

    @classmethod
    def get_instance(cls) -> UnifiedDataService:
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = UnifiedDataService()
        return cls._instance

    # =========================================================================
    # LAZY LOADING OF DEPENDENCIES
    # =========================================================================

    def _get_connection_manager(self):
        """Lazy load connection manager."""
        if self._connection_manager is None:
            from app_core.offline.connection_manager import get_connection_manager
            self._connection_manager = get_connection_manager()
        return self._connection_manager

    def _get_local_db(self):
        """Lazy load local database."""
        if self._local_db is None:
            from app_core.offline.local_database import get_local_database
            self._local_db = get_local_database()
        return self._local_db

    def _get_cache_manager(self):
        """Lazy load cache manager."""
        if self._cache_manager is None:
            from app_core.offline.cache_manager import get_cache_manager
            self._cache_manager = get_cache_manager()
        return self._cache_manager

    def _get_sync_engine(self):
        """Lazy load sync engine."""
        if self._sync_engine is None:
            from app_core.offline.sync_engine import get_sync_engine
            self._sync_engine = get_sync_engine()
        return self._sync_engine

    def _get_supabase_service(self, table: str):
        """Get Supabase service for a table."""
        try:
            from app_core.data.supabase_client import SupabaseService
            remote_table = self.TABLE_MAPPING.get(table, table)
            return SupabaseService(remote_table)
        except Exception as e:
            logger.debug(f"Supabase service not available: {e}")
            return None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_online(self) -> bool:
        """Check if currently online."""
        return self._get_connection_manager().is_online

    @property
    def is_offline(self) -> bool:
        """Check if currently offline."""
        return self._get_connection_manager().is_offline

    @property
    def connection_status(self) -> str:
        """Get connection status string."""
        return self._get_connection_manager().status.value

    @property
    def pending_sync_count(self) -> int:
        """Get number of pending sync operations."""
        return self._get_local_db().get_pending_count()

    @property
    def last_sync(self) -> Optional[datetime]:
        """Get last successful sync time."""
        return self._get_sync_engine().state.last_sync_success

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self, start_sync: bool = True) -> None:
        """
        Initialize the unified data service.

        Args:
            start_sync: Whether to start background sync
        """
        if self._initialized:
            return

        # Initialize all components
        self._get_connection_manager()
        self._get_local_db()
        self._get_cache_manager()

        # Register for connection changes
        self._get_connection_manager().register_callback(self._on_connection_change)

        # Start sync engine
        if start_sync:
            self._get_sync_engine().start()

        self._initialized = True
        logger.info(f"UnifiedDataService initialized. Online: {self.is_online}")

    def _on_connection_change(self, state) -> None:
        """Handle connection status changes."""
        is_online = state.status.value == "online"
        logger.info(f"Connection changed: online={is_online}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(is_online)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    def register_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Register a callback for online/offline status changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # =========================================================================
    # GENERIC DATA OPERATIONS
    # =========================================================================

    def fetch_data(
        self,
        table: str,
        use_cache: bool = True,
        cache_ttl_minutes: int = 30,
        force_online: bool = False,
        force_offline: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch data from the appropriate source.

        Args:
            table: Table name to fetch from
            use_cache: Whether to use cached data
            cache_ttl_minutes: Cache time-to-live in minutes
            force_online: Force online fetch (if available)
            force_offline: Force offline fetch from local DB

        Returns:
            DataFrame with table data
        """
        cache_key = f"table_{table}"

        # Check cache first
        if use_cache:
            cached_df = self._get_cache_manager().get_dataframe(cache_key)
            if cached_df is not None:
                logger.debug(f"Cache hit for {table}")
                return cached_df

        # Determine source
        use_online = (self.is_online or force_online) and not force_offline

        df = pd.DataFrame()

        if use_online:
            # Try online first
            try:
                service = self._get_supabase_service(table)
                if service and service.is_connected():
                    df = service.fetch_all()
                    if not df.empty:
                        # Cache the result
                        self._get_cache_manager().cache_dataframe(df, cache_key)
                        # Also save to local DB for offline access
                        self._save_to_local(table, df, sync_status="synced")
                        logger.debug(f"Fetched {len(df)} rows from Supabase: {table}")
                        return df
            except Exception as e:
                logger.warning(f"Online fetch failed for {table}: {e}")

        # Fall back to local database
        try:
            df = self._get_local_db().to_dataframe(table)
            if not df.empty:
                # Cache the result
                if use_cache:
                    self._get_cache_manager().cache_dataframe(df, cache_key)
                logger.debug(f"Fetched {len(df)} rows from local DB: {table}")
        except Exception as e:
            logger.error(f"Local fetch failed for {table}: {e}")

        return df

    def save_data(
        self,
        table: str,
        df: pd.DataFrame,
        replace: bool = True,
    ) -> bool:
        """
        Save data to the appropriate destination.

        Args:
            table: Table name to save to
            df: DataFrame to save
            replace: Whether to replace existing data

        Returns:
            True if saved successfully
        """
        if df.empty:
            return False

        # Always save locally first
        try:
            self._save_to_local(
                table, df,
                sync_status="pending" if self.is_offline else "synced",
                replace=replace
            )
        except Exception as e:
            logger.error(f"Failed to save locally: {e}")
            return False

        # If online, also save to Supabase
        if self.is_online:
            try:
                service = self._get_supabase_service(table)
                if service and service.is_connected():
                    records = df.to_dict(orient="records")
                    if replace:
                        # For replace, we upsert each record
                        for record in records:
                            service.upsert(record)
                    else:
                        service.insert_many(records)

                    # Update local sync status
                    self._get_local_db().execute(
                        f"UPDATE {table} SET sync_status = 'synced'"
                    )
            except Exception as e:
                logger.warning(f"Failed to save to Supabase: {e}")
                # Data is saved locally and queued for sync

        # Invalidate cache
        cache_key = f"table_{table}"
        self._get_cache_manager().delete(cache_key, "datasets")

        return True

    def _save_to_local(
        self,
        table: str,
        df: pd.DataFrame,
        sync_status: str = "pending",
        replace: bool = True,
    ) -> int:
        """Save DataFrame to local database."""
        local_db = self._get_local_db()
        if_exists = "replace" if replace else "append"
        return local_db.from_dataframe(df, table, if_exists=if_exists, sync_status=sync_status)

    # =========================================================================
    # PATIENT DATA OPERATIONS
    # =========================================================================

    def fetch_patient_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch patient arrivals data.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with patient arrivals
        """
        df = self.fetch_data("patient_arrivals", use_cache=use_cache)

        # Apply date filters if provided
        if not df.empty and (start_date or end_date):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if start_date:
                    df = df[df["date"] >= start_date]
                if end_date:
                    df = df[df["date"] <= end_date]

        return df

    def save_patient_data(self, df: pd.DataFrame) -> bool:
        """Save patient arrivals data."""
        return self.save_data("patient_arrivals", df)

    # =========================================================================
    # INVENTORY DATA OPERATIONS
    # =========================================================================

    def fetch_inventory_data(
        self,
        category: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch inventory data.

        Args:
            category: Optional category filter
            use_cache: Whether to use cached data

        Returns:
            DataFrame with inventory data
        """
        df = self.fetch_data("inventory_data", use_cache=use_cache)

        if not df.empty and category:
            if "category" in df.columns:
                df = df[df["category"] == category]

        return df

    def save_inventory_data(self, df: pd.DataFrame) -> bool:
        """Save inventory data."""
        return self.save_data("inventory_data", df)

    # =========================================================================
    # FINANCIAL DATA OPERATIONS
    # =========================================================================

    def fetch_financial_data(
        self,
        department: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch financial data.

        Args:
            department: Optional department filter
            use_cache: Whether to use cached data

        Returns:
            DataFrame with financial data
        """
        df = self.fetch_data("financial_data", use_cache=use_cache)

        if not df.empty and department:
            if "department" in df.columns:
                df = df[df["department"] == department]

        return df

    def save_financial_data(self, df: pd.DataFrame) -> bool:
        """Save financial data."""
        return self.save_data("financial_data", df)

    # =========================================================================
    # MODEL OPERATIONS
    # =========================================================================

    def save_model(
        self,
        model: Any,
        name: str,
        model_type: str,
        metrics: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> str:
        """
        Save a trained model.

        Args:
            model: Trained model object
            name: Unique name for the model
            model_type: Type of model
            metrics: Performance metrics
            parameters: Model hyperparameters
            feature_columns: Feature column names
            target_column: Target column name

        Returns:
            Cache key for the model
        """
        # Save to cache
        cache_key = self._get_cache_manager().cache_model(
            model=model,
            name=name,
            model_type=model_type,
            metrics=metrics,
            parameters=parameters,
            feature_columns=feature_columns,
            target_column=target_column,
        )

        # Save metadata to local DB
        self._get_local_db().insert("trained_models", {
            "name": name,
            "model_type": model_type,
            "target_column": target_column,
            "feature_columns": ",".join(feature_columns or []),
            "metrics_json": str(metrics) if metrics else None,
            "parameters_json": str(parameters) if parameters else None,
            "file_path": self._get_cache_manager()._index.get(cache_key, {}).get("file_path"),
            "accuracy": metrics.get("accuracy") if metrics else None,
            "mae": metrics.get("mae") if metrics else None,
            "rmse": metrics.get("rmse") if metrics else None,
        })

        return cache_key

    def load_model(self, name: str) -> Optional[Any]:
        """
        Load a trained model.

        Args:
            name: Name of the model to load

        Returns:
            Model object if found, None otherwise
        """
        return self._get_cache_manager().get_model(name)

    def list_models(self) -> List[Dict]:
        """List all saved models."""
        return self._get_cache_manager().list_models()

    # =========================================================================
    # RESULTS OPERATIONS
    # =========================================================================

    def save_results(
        self,
        results: Dict[str, Any],
        name: str,
        result_type: str = "forecast",
    ) -> str:
        """
        Save model results or forecasts.

        Args:
            results: Results dictionary
            name: Unique name for the results
            result_type: Type of results

        Returns:
            Cache key for the results
        """
        return self._get_cache_manager().cache_results(results, name, result_type)

    def load_results(self, name: str) -> Optional[Dict]:
        """
        Load cached results.

        Args:
            name: Name of the results to load

        Returns:
            Results dictionary if found, None otherwise
        """
        return self._get_cache_manager().get_results(name)

    # =========================================================================
    # UPLOADED FILES
    # =========================================================================

    def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
    ) -> str:
        """
        Save an uploaded file.

        Args:
            file_content: Raw file content
            filename: Original filename
            file_type: File type/extension

        Returns:
            Path to saved file
        """
        return self._get_cache_manager().cache_uploaded_file(
            file_content, filename, file_type
        )

    def save_uploaded_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Save an uploaded DataFrame with metadata.

        Args:
            df: DataFrame to save
            name: Name for the dataset
            description: Optional description

        Returns:
            Cache key for the dataset
        """
        # Cache the DataFrame
        cache_key = self._get_cache_manager().cache_dataframe(df, name, metadata={
            "description": description,
            "upload_source": "user_upload",
        })

        # Save metadata to local DB
        self._get_local_db().insert("uploaded_datasets", {
            "name": name,
            "description": description,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_json": ",".join(df.columns.tolist()),
        })

        return cache_key

    def load_uploaded_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load an uploaded DataFrame.

        Args:
            name: Name of the dataset

        Returns:
            DataFrame if found, None otherwise
        """
        return self._get_cache_manager().get_dataframe(name)

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def sync_now(self) -> bool:
        """
        Trigger immediate sync.

        Returns:
            True if sync completed successfully
        """
        if not self.is_online:
            logger.warning("Cannot sync: offline")
            return False

        return self._get_sync_engine().sync_now()

    def full_sync(self) -> Dict[str, int]:
        """
        Perform full bidirectional sync.

        Returns:
            Dict with sync statistics
        """
        return self._get_sync_engine().full_sync()

    def pull_latest(self, tables: Optional[List[str]] = None) -> int:
        """
        Pull latest data from cloud.

        Args:
            tables: Tables to pull (None for all)

        Returns:
            Total records pulled
        """
        if not self.is_online:
            return 0

        tables = tables or list(self.TABLE_MAPPING.keys())
        total = 0

        sync_engine = self._get_sync_engine()
        for table in tables:
            try:
                pulled = sync_engine.pull_from_cloud(table)
                total += pulled
                # Invalidate cache
                self._get_cache_manager().delete(f"table_{table}", "datasets")
            except Exception as e:
                logger.error(f"Error pulling {table}: {e}")

        return total

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get an app setting."""
        return self._get_local_db().get_setting(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set an app setting."""
        self._get_local_db().set_setting(key, value)

    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.

        Returns:
            Dict with status information for UI display
        """
        return {
            "connection": self._get_connection_manager().get_status_display(),
            "sync": self._get_sync_engine().get_status_display(),
            "cache": self._get_cache_manager().get_cache_stats(),
            "is_online": self.is_online,
            "pending_sync": self.pending_sync_count,
        }

    def force_offline(self) -> None:
        """Force offline mode (for testing)."""
        self._get_connection_manager().force_offline()

    def force_check_connection(self) -> None:
        """Force an immediate connection check."""
        self._get_connection_manager().force_check()

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._get_sync_engine().stop()
            self._get_connection_manager().stop_monitoring()
            self._get_local_db().close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Singleton accessor
_data_service: Optional[UnifiedDataService] = None


def get_data_service() -> UnifiedDataService:
    """
    Get the global UnifiedDataService instance.

    Returns:
        UnifiedDataService singleton

    Usage:
        from app_core.offline import get_data_service

        service = get_data_service()
        df = service.fetch_patient_data()
    """
    global _data_service
    if _data_service is None:
        _data_service = UnifiedDataService.get_instance()
        _data_service.initialize()
    return _data_service


# Convenience functions for common operations
def fetch_patient_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to fetch patient data."""
    return get_data_service().fetch_patient_data(start_date, end_date)


def fetch_inventory_data(category: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to fetch inventory data."""
    return get_data_service().fetch_inventory_data(category)


def fetch_financial_data(department: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to fetch financial data."""
    return get_data_service().fetch_financial_data(department)


def is_online() -> bool:
    """Convenience function to check online status."""
    return get_data_service().is_online


def sync_now() -> bool:
    """Convenience function to trigger sync."""
    return get_data_service().sync_now()
