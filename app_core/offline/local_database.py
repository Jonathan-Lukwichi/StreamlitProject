# =============================================================================
# app_core/offline/local_database.py
# Local SQLite Database for Offline Operations
# =============================================================================
"""
LocalDatabase - SQLite-based local storage that mirrors Supabase schema.

Features:
- Automatic schema creation
- Full CRUD operations
- DataFrame integration (pandas)
- Transaction support
- Thread-safe operations
"""

from __future__ import annotations
import os
import sqlite3
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LocalDatabase:
    """
    Local SQLite database for offline data storage.

    Mirrors the Supabase schema for seamless online/offline switching.
    """

    # Default database location
    DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "local_data" / "healthforecast.db"

    # Schema definitions matching Supabase tables
    SCHEMA = {
        "patient_arrivals": """
            CREATE TABLE IF NOT EXISTS patient_arrivals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_arrivals INTEGER,
                respiratory INTEGER,
                cardiac INTEGER,
                trauma INTEGER,
                gastrointestinal INTEGER,
                neurological INTEGER,
                infectious INTEGER,
                other INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                sync_status TEXT DEFAULT 'pending',
                remote_id TEXT,
                UNIQUE(date)
            )
        """,
        "inventory_data": """
            CREATE TABLE IF NOT EXISTS inventory_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                item_id TEXT,
                item_name TEXT,
                quantity INTEGER,
                reorder_level INTEGER,
                unit_cost REAL,
                category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                sync_status TEXT DEFAULT 'pending',
                remote_id TEXT
            )
        """,
        "financial_data": """
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                revenue REAL,
                expenses REAL,
                labor_cost REAL,
                supply_cost REAL,
                overhead REAL,
                department TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                sync_status TEXT DEFAULT 'pending',
                remote_id TEXT
            )
        """,
        "uploaded_datasets": """
            CREATE TABLE IF NOT EXISTS uploaded_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                file_path TEXT,
                file_type TEXT,
                row_count INTEGER,
                column_count INTEGER,
                columns_json TEXT,
                upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT,
                sync_status TEXT DEFAULT 'pending',
                remote_id TEXT
            )
        """,
        "trained_models": """
            CREATE TABLE IF NOT EXISTS trained_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_type TEXT,
                target_column TEXT,
                feature_columns TEXT,
                metrics_json TEXT,
                parameters_json TEXT,
                file_path TEXT,
                training_date TEXT DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                mae REAL,
                rmse REAL,
                sync_status TEXT DEFAULT 'pending',
                remote_id TEXT
            )
        """,
        "sync_queue": """
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                table_name TEXT NOT NULL,
                record_id INTEGER,
                data_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER DEFAULT 0,
                last_attempt TEXT,
                status TEXT DEFAULT 'pending',
                error_message TEXT
            )
        """,
        "app_settings": """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
    }

    _instance: Optional[LocalDatabase] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize local database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._ensure_directory()
        self._local = threading.local()
        self._initialized = False

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> LocalDatabase:
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LocalDatabase(db_path)
        return cls._instance

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        for table_name, schema in self.SCHEMA.items():
            try:
                cursor.execute(schema)
                logger.debug(f"Created/verified table: {table_name}")
            except sqlite3.Error as e:
                logger.error(f"Error creating table {table_name}: {e}")

        conn.commit()
        self._initialized = True
        logger.info(f"Local database initialized at: {self.db_path}")

    # =========================================================================
    # GENERIC CRUD OPERATIONS
    # =========================================================================

    def insert(
        self,
        table: str,
        data: Dict[str, Any],
        sync_status: str = "pending"
    ) -> int:
        """
        Insert a record into a table.

        Args:
            table: Table name
            data: Dictionary of column:value pairs
            sync_status: Sync status for the record

        Returns:
            Inserted row ID
        """
        data = data.copy()
        data["sync_status"] = sync_status
        data["created_at"] = datetime.now().isoformat()
        data["updated_at"] = datetime.now().isoformat()

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())

        with self.transaction() as conn:
            cursor = conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                values
            )
            row_id = cursor.lastrowid

        # Queue for sync
        self._queue_sync("INSERT", table, row_id, data)

        return row_id

    def insert_many(
        self,
        table: str,
        records: List[Dict[str, Any]],
        sync_status: str = "pending"
    ) -> List[int]:
        """Insert multiple records."""
        if not records:
            return []

        ids = []
        with self.transaction() as conn:
            for data in records:
                data = data.copy()
                data["sync_status"] = sync_status
                data["created_at"] = datetime.now().isoformat()
                data["updated_at"] = datetime.now().isoformat()

                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?" for _ in data])
                values = list(data.values())

                cursor = conn.execute(
                    f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                    values
                )
                ids.append(cursor.lastrowid)

        return ids

    def update(
        self,
        table: str,
        record_id: int,
        data: Dict[str, Any]
    ) -> bool:
        """
        Update a record.

        Args:
            table: Table name
            record_id: ID of record to update
            data: Dictionary of column:value pairs to update

        Returns:
            True if updated successfully
        """
        data = data.copy()
        data["updated_at"] = datetime.now().isoformat()
        data["sync_status"] = "pending"

        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        values = list(data.values()) + [record_id]

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE id = ?",
                values
            )
            success = cursor.rowcount > 0

        if success:
            self._queue_sync("UPDATE", table, record_id, data)

        return success

    def delete(self, table: str, record_id: int) -> bool:
        """Delete a record."""
        # Get record data before deleting (for sync)
        record = self.get_by_id(table, record_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE id = ?",
                [record_id]
            )
            success = cursor.rowcount > 0

        if success and record:
            self._queue_sync("DELETE", table, record_id, dict(record))

        return success

    def get_by_id(self, table: str, record_id: int) -> Optional[sqlite3.Row]:
        """Get a record by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?",
            [record_id]
        )
        return cursor.fetchone()

    def get_all(
        self,
        table: str,
        where: Optional[str] = None,
        params: Optional[List] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[sqlite3.Row]:
        """Get all records from a table with optional filtering."""
        query = f"SELECT * FROM {table}"

        if where:
            query += f" WHERE {where}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        conn = self._get_connection()
        cursor = conn.execute(query, params or [])
        return cursor.fetchall()

    def query(self, sql: str, params: Optional[List] = None) -> List[sqlite3.Row]:
        """Execute a raw SQL query."""
        conn = self._get_connection()
        cursor = conn.execute(sql, params or [])
        return cursor.fetchall()

    def execute(self, sql: str, params: Optional[List] = None) -> int:
        """Execute a raw SQL statement."""
        with self.transaction() as conn:
            cursor = conn.execute(sql, params or [])
            return cursor.rowcount

    # =========================================================================
    # PANDAS INTEGRATION
    # =========================================================================

    def to_dataframe(
        self,
        table: str,
        where: Optional[str] = None,
        params: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Load a table into a pandas DataFrame.

        Args:
            table: Table name
            where: Optional WHERE clause
            params: Parameters for WHERE clause

        Returns:
            DataFrame with table data
        """
        query = f"SELECT * FROM {table}"
        if where:
            query += f" WHERE {where}"

        conn = self._get_connection()
        return pd.read_sql_query(query, conn, params=params)

    def from_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append",
        sync_status: str = "pending"
    ) -> int:
        """
        Save a DataFrame to a table.

        Args:
            df: DataFrame to save
            table: Target table name
            if_exists: 'append', 'replace', or 'fail'
            sync_status: Sync status for records

        Returns:
            Number of rows inserted
        """
        df = df.copy()

        # Add metadata columns
        df["sync_status"] = sync_status
        df["created_at"] = datetime.now().isoformat()
        df["updated_at"] = datetime.now().isoformat()

        # Handle NaN values
        df = df.replace({np.nan: None})

        conn = self._get_connection()
        rows = df.to_sql(table, conn, if_exists=if_exists, index=False)

        # Queue all rows for sync
        if sync_status == "pending":
            for _, row in df.iterrows():
                self._queue_sync("INSERT", table, None, row.to_dict())

        return rows or len(df)

    # =========================================================================
    # SYNC QUEUE MANAGEMENT
    # =========================================================================

    def _queue_sync(
        self,
        operation: str,
        table: str,
        record_id: Optional[int],
        data: Dict[str, Any]
    ) -> None:
        """Add an operation to the sync queue."""
        # Clean data for JSON serialization
        clean_data = {}
        for k, v in data.items():
            if isinstance(v, (datetime,)):
                clean_data[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                clean_data[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_data[k] = float(v)
            elif pd.isna(v):
                clean_data[k] = None
            else:
                clean_data[k] = v

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO sync_queue (operation, table_name, record_id, data_json)
            VALUES (?, ?, ?, ?)
            """,
            [operation, table, record_id, json.dumps(clean_data)]
        )
        conn.commit()

    def get_pending_sync(self, limit: int = 100) -> List[Dict]:
        """Get pending sync operations."""
        rows = self.query(
            """
            SELECT * FROM sync_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            [limit]
        )
        return [
            {
                "id": row["id"],
                "operation": row["operation"],
                "table": row["table_name"],
                "record_id": row["record_id"],
                "data": json.loads(row["data_json"]) if row["data_json"] else {},
                "created_at": row["created_at"],
                "attempts": row["attempts"],
            }
            for row in rows
        ]

    def mark_synced(self, sync_id: int) -> None:
        """Mark a sync operation as completed."""
        self.execute(
            "UPDATE sync_queue SET status = 'synced' WHERE id = ?",
            [sync_id]
        )

    def mark_sync_failed(self, sync_id: int, error: str) -> None:
        """Mark a sync operation as failed."""
        self.execute(
            """
            UPDATE sync_queue
            SET status = 'failed', attempts = attempts + 1,
                last_attempt = ?, error_message = ?
            WHERE id = ?
            """,
            [datetime.now().isoformat(), error, sync_id]
        )

    def get_pending_count(self) -> int:
        """Get count of pending sync operations."""
        result = self.query("SELECT COUNT(*) as count FROM sync_queue WHERE status = 'pending'")
        return result[0]["count"] if result else 0

    # =========================================================================
    # CONVENIENCE METHODS FOR SPECIFIC TABLES
    # =========================================================================

    def save_patient_arrivals(self, df: pd.DataFrame) -> int:
        """Save patient arrivals data."""
        # Ensure date column is string
        if "date" in df.columns or "Date" in df.columns:
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            if date_col == "Date":
                df = df.rename(columns={"Date": "date"})

        return self.from_dataframe(df, "patient_arrivals", if_exists="replace")

    def get_patient_arrivals(self) -> pd.DataFrame:
        """Get patient arrivals data."""
        return self.to_dataframe("patient_arrivals")

    def save_inventory_data(self, df: pd.DataFrame) -> int:
        """Save inventory data."""
        if "date" in df.columns or "Date" in df.columns:
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            if date_col == "Date":
                df = df.rename(columns={"Date": "date"})

        return self.from_dataframe(df, "inventory_data", if_exists="replace")

    def get_inventory_data(self) -> pd.DataFrame:
        """Get inventory data."""
        return self.to_dataframe("inventory_data")

    def save_financial_data(self, df: pd.DataFrame) -> int:
        """Save financial data."""
        if "date" in df.columns or "Date" in df.columns:
            date_col = "date" if "date" in df.columns else "Date"
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            if date_col == "Date":
                df = df.rename(columns={"Date": "date"})

        return self.from_dataframe(df, "financial_data", if_exists="replace")

    def get_financial_data(self) -> pd.DataFrame:
        """Get financial data."""
        return self.to_dataframe("financial_data")

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get an app setting."""
        result = self.query(
            "SELECT value FROM app_settings WHERE key = ?",
            [key]
        )
        if result:
            try:
                return json.loads(result[0]["value"])
            except json.JSONDecodeError:
                return result[0]["value"]
        return default

    def set_setting(self, key: str, value: Any) -> None:
        """Set an app setting."""
        value_str = json.dumps(value) if not isinstance(value, str) else value
        self.execute(
            """
            INSERT OR REPLACE INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            [key, value_str, datetime.now().isoformat()]
        )

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Singleton accessor
_local_database: Optional[LocalDatabase] = None


def get_local_database() -> LocalDatabase:
    """Get the global LocalDatabase instance."""
    global _local_database
    if _local_database is None:
        _local_database = LocalDatabase.get_instance()
        _local_database.initialize()
    return _local_database
