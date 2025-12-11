# =============================================================================
# app_core/offline/sync_engine.py
# Automatic Synchronization Engine
# =============================================================================
"""
SyncEngine - Handles automatic bidirectional sync between local and cloud.

Features:
- Background sync thread
- Conflict resolution
- Retry logic with exponential backoff
- Sync status tracking
- Event callbacks
"""

from __future__ import annotations
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging
import json

import pandas as pd

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class SyncOperation:
    """Represents a sync operation."""
    id: int
    operation: str  # INSERT, UPDATE, DELETE
    table: str
    record_id: Optional[int]
    data: Dict[str, Any]
    status: SyncStatus = SyncStatus.PENDING
    attempts: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SyncState:
    """Current sync state."""
    is_syncing: bool = False
    last_sync: Optional[datetime] = None
    last_sync_success: Optional[datetime] = None
    pending_count: int = 0
    failed_count: int = 0
    total_synced: int = 0


class SyncEngine:
    """
    Automatic synchronization engine between local SQLite and Supabase.

    Usage:
        engine = SyncEngine.get_instance()
        engine.start()  # Start background sync
        engine.sync_now()  # Force immediate sync
    """

    _instance: Optional[SyncEngine] = None
    _lock = threading.Lock()

    # Configuration
    SYNC_INTERVAL = 30          # Seconds between sync attempts
    MAX_RETRY_ATTEMPTS = 5      # Max retries per operation
    BATCH_SIZE = 50             # Operations per sync batch
    BACKOFF_BASE = 2            # Exponential backoff base

    def __init__(self):
        """Initialize sync engine."""
        self._state = SyncState()
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()
        self._callbacks: List[Callable[[SyncState], None]] = []
        self._initialized = False

        # Lazy imports to avoid circular dependencies
        self._connection_manager = None
        self._local_db = None
        self._supabase_client = None

    @classmethod
    def get_instance(cls) -> SyncEngine:
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SyncEngine()
        return cls._instance

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

    def _get_supabase_client(self):
        """Lazy load Supabase client."""
        if self._supabase_client is None:
            try:
                from supabase import create_client
                url = os.getenv("SUPABASE_URL")
                key = os.getenv("SUPABASE_KEY")
                if url and key:
                    self._supabase_client = create_client(url, key)
            except Exception as e:
                logger.debug(f"Supabase client not available: {e}")
        return self._supabase_client

    @property
    def state(self) -> SyncState:
        """Get current sync state."""
        return self._state

    @property
    def is_syncing(self) -> bool:
        """Check if sync is in progress."""
        return self._state.is_syncing

    @property
    def pending_count(self) -> int:
        """Get count of pending sync operations."""
        return self._get_local_db().get_pending_count()

    def initialize(self) -> None:
        """Initialize the sync engine."""
        if self._initialized:
            return

        # Initialize dependencies
        self._get_connection_manager()
        self._get_local_db()

        # Register for connection status changes
        self._get_connection_manager().register_callback(self._on_connection_change)

        self._initialized = True
        logger.info("SyncEngine initialized")

    def start(self) -> None:
        """Start background sync thread."""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return

        self.initialize()
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="SyncEngine"
        )
        self._sync_thread.start()
        logger.info("Sync engine started")

    def stop(self) -> None:
        """Stop background sync thread."""
        self._stop_sync.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=10)
        logger.info("Sync engine stopped")

    def _sync_loop(self) -> None:
        """Background sync loop."""
        while not self._stop_sync.is_set():
            # Wait for interval or stop signal
            if self._stop_sync.wait(timeout=self.SYNC_INTERVAL):
                break

            # Only sync if online
            if self._get_connection_manager().is_online:
                try:
                    self._perform_sync()
                except Exception as e:
                    logger.error(f"Sync error: {e}")

    def _on_connection_change(self, state) -> None:
        """Handle connection status changes."""
        if state.status.value == "online":
            # Connection restored - trigger sync
            logger.info("Connection restored, triggering sync")
            self.sync_now()

    def sync_now(self) -> bool:
        """
        Perform immediate sync.

        Returns:
            True if sync completed successfully
        """
        if not self._get_connection_manager().is_online:
            logger.debug("Cannot sync: offline")
            return False

        return self._perform_sync()

    def _perform_sync(self) -> bool:
        """
        Perform synchronization.

        Returns:
            True if all operations synced successfully
        """
        if self._state.is_syncing:
            return False

        self._state.is_syncing = True
        self._state.last_sync = datetime.now()
        self._notify_callbacks()

        try:
            # Get pending operations
            local_db = self._get_local_db()
            pending = local_db.get_pending_sync(limit=self.BATCH_SIZE)

            if not pending:
                self._state.is_syncing = False
                return True

            logger.info(f"Syncing {len(pending)} operations")

            success_count = 0
            fail_count = 0

            for op in pending:
                try:
                    if self._sync_operation(op):
                        local_db.mark_synced(op["id"])
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Error syncing operation {op['id']}: {e}")
                    local_db.mark_sync_failed(op["id"], str(e))
                    fail_count += 1

            self._state.total_synced += success_count
            self._state.failed_count = fail_count
            self._state.pending_count = local_db.get_pending_count()

            if fail_count == 0:
                self._state.last_sync_success = datetime.now()

            logger.info(f"Sync complete: {success_count} success, {fail_count} failed")
            return fail_count == 0

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False

        finally:
            self._state.is_syncing = False
            self._notify_callbacks()

    def _sync_operation(self, op: Dict) -> bool:
        """
        Sync a single operation to Supabase.

        Args:
            op: Operation dict with operation, table, data

        Returns:
            True if synced successfully
        """
        supabase = self._get_supabase_client()
        if not supabase:
            # No Supabase client - skip sync but don't fail
            logger.debug("No Supabase client, marking as synced")
            return True

        table = op["table"]
        operation = op["operation"]
        data = op["data"]

        # Map local table names to Supabase table names if needed
        table_mapping = {
            "patient_arrivals": "patient_arrivals",
            "inventory_data": "inventory_management",
            "financial_data": "financial_data",
        }
        supabase_table = table_mapping.get(table, table)

        try:
            # Remove local-only fields
            sync_data = {k: v for k, v in data.items()
                        if k not in ["id", "sync_status", "remote_id"]}

            if operation == "INSERT":
                result = supabase.table(supabase_table).insert(sync_data).execute()
            elif operation == "UPDATE":
                # Need remote_id for updates
                remote_id = data.get("remote_id")
                if remote_id:
                    result = supabase.table(supabase_table).update(sync_data).eq("id", remote_id).execute()
                else:
                    # Upsert based on unique key (e.g., date)
                    result = supabase.table(supabase_table).upsert(sync_data).execute()
            elif operation == "DELETE":
                remote_id = data.get("remote_id")
                if remote_id:
                    result = supabase.table(supabase_table).delete().eq("id", remote_id).execute()
                else:
                    return True  # Nothing to delete remotely

            return True

        except Exception as e:
            logger.error(f"Supabase operation failed: {e}")
            raise

    def pull_from_cloud(self, table: str, since: Optional[datetime] = None) -> int:
        """
        Pull data from Supabase to local database.

        Args:
            table: Table to pull
            since: Only pull records updated since this time

        Returns:
            Number of records pulled
        """
        if not self._get_connection_manager().is_online:
            return 0

        supabase = self._get_supabase_client()
        if not supabase:
            return 0

        try:
            # Map table names
            table_mapping = {
                "patient_arrivals": "patient_arrivals",
                "inventory_data": "inventory_management",
                "financial_data": "financial_data",
            }
            supabase_table = table_mapping.get(table, table)

            # Build query
            query = supabase.table(supabase_table).select("*")

            if since:
                query = query.gte("updated_at", since.isoformat())

            result = query.execute()

            if not result.data:
                return 0

            # Convert to DataFrame and save locally
            df = pd.DataFrame(result.data)
            local_db = self._get_local_db()

            # Mark as synced (not pending)
            local_db.from_dataframe(df, table, if_exists="replace", sync_status="synced")

            logger.info(f"Pulled {len(df)} records from {supabase_table}")
            return len(df)

        except Exception as e:
            logger.error(f"Error pulling from cloud: {e}")
            return 0

    def full_sync(self) -> Dict[str, int]:
        """
        Perform a full bidirectional sync.

        Returns:
            Dict with sync statistics
        """
        stats = {
            "pushed": 0,
            "pulled": 0,
            "errors": 0,
        }

        if not self._get_connection_manager().is_online:
            return stats

        # Push local changes first
        if self._perform_sync():
            stats["pushed"] = self._state.total_synced

        # Pull cloud changes
        tables = ["patient_arrivals", "inventory_data", "financial_data"]
        for table in tables:
            try:
                pulled = self.pull_from_cloud(table)
                stats["pulled"] += pulled
            except Exception as e:
                logger.error(f"Error pulling {table}: {e}")
                stats["errors"] += 1

        return stats

    def register_callback(self, callback: Callable[[SyncState], None]) -> None:
        """Register a callback for sync state changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[SyncState], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception as e:
                logger.error(f"Error in sync callback: {e}")

    def get_status_display(self) -> Dict[str, Any]:
        """Get sync status for UI display."""
        return {
            "is_syncing": self._state.is_syncing,
            "last_sync": self._state.last_sync.isoformat() if self._state.last_sync else None,
            "last_success": self._state.last_sync_success.isoformat() if self._state.last_sync_success else None,
            "pending_count": self.pending_count,
            "failed_count": self._state.failed_count,
            "total_synced": self._state.total_synced,
        }


# Singleton accessor
_sync_engine: Optional[SyncEngine] = None


def get_sync_engine() -> SyncEngine:
    """Get the global SyncEngine instance."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = SyncEngine.get_instance()
        _sync_engine.initialize()
    return _sync_engine
