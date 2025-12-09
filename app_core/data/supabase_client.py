# =============================================================================
# app_core/data/supabase_client.py
# Supabase Client Configuration for HealthForecast AI
# Handles database connections and CRUD operations
# =============================================================================

from __future__ import annotations
import streamlit as st
from typing import Optional, Dict, Any, List
import pandas as pd


def get_supabase_client():
    """
    Initialize and return Supabase client using Streamlit secrets.

    Expects secrets in .streamlit/secrets.toml:
        [supabase]
        url = "https://your-project.supabase.co"
        key = "your-anon-key"

    Returns:
        Supabase client instance or None if not configured
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        st.error("Supabase not installed. Run: `pip install supabase`")
        return None

    # Check if secrets are configured
    if "supabase" not in st.secrets:
        st.warning(
            "⚠️ Supabase credentials not found. Please configure `.streamlit/secrets.toml`:\n\n"
            "```toml\n"
            "[supabase]\n"
            'url = "https://your-project.supabase.co"\n'
            'key = "your-anon-key"\n'
            "```"
        )
        return None

    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]

        # Create Supabase client
        client: Client = create_client(url, key)
        return client

    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        return None


# Global client reference for cleanup
_supabase_client = None


@st.cache_resource(ttl=3600)  # Cache for 1 hour, then refresh
def get_cached_supabase_client():
    """
    Get cached Supabase client (reused across sessions).

    Uses TTL to periodically refresh the connection and prevent stale connections.

    Returns:
        Cached Supabase client instance
    """
    global _supabase_client
    _supabase_client = get_supabase_client()
    return _supabase_client


def cleanup_supabase_connections():
    """
    Explicitly close Supabase client connections.
    Call this when you're done with database operations.
    """
    global _supabase_client
    try:
        if _supabase_client is not None:
            # The Supabase client uses httpx internally
            # Try to close the underlying HTTP client
            if hasattr(_supabase_client, 'postgrest'):
                if hasattr(_supabase_client.postgrest, '_session'):
                    _supabase_client.postgrest._session.aclose()
    except Exception:
        pass  # Ignore cleanup errors


class SupabaseService:
    """
    Generic Supabase service for CRUD operations.
    """

    def __init__(self, table_name: str):
        """
        Initialize service for a specific table.

        Args:
            table_name: Name of the Supabase table
        """
        self.table_name = table_name
        self.client = get_cached_supabase_client()

    def is_connected(self) -> bool:
        """Check if Supabase client is available."""
        return self.client is not None

    def fetch_all(self, order_by: Optional[str] = None, ascending: bool = True) -> pd.DataFrame:
        """
        Fetch ALL records from the table (handles Supabase 1000 row limit).

        Uses pagination to fetch all records when table has more than 1000 rows.

        Args:
            order_by: Column to order by (optional)
            ascending: Sort order (default: ascending)

        Returns:
            DataFrame with all records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_data = []
            batch_size = 1000
            offset = 0

            while True:
                query = self.client.table(self.table_name).select("*")

                if order_by:
                    query = query.order(order_by, desc=not ascending)

                # Fetch batch with range
                query = query.range(offset, offset + batch_size - 1)
                response = query.execute()

                if response.data:
                    all_data.extend(response.data)
                    # If we got fewer than batch_size, we've reached the end
                    if len(response.data) < batch_size:
                        break
                    offset += batch_size
                else:
                    break

            if all_data:
                return pd.DataFrame(all_data)
            return pd.DataFrame()

        except Exception as e:
            st.error(f"Error fetching data from {self.table_name}: {e}")
            return pd.DataFrame()

    def fetch_by_date_range(
        self,
        date_column: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch ALL records within a date range (handles Supabase 1000 row limit).

        Args:
            date_column: Name of the date column
            start_date: Start date (ISO format: YYYY-MM-DD)
            end_date: End date (ISO format: YYYY-MM-DD)

        Returns:
            DataFrame with filtered records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_data = []
            batch_size = 1000
            offset = 0

            while True:
                response = (
                    self.client.table(self.table_name)
                    .select("*")
                    .gte(date_column, start_date)
                    .lte(date_column, end_date)
                    .order(date_column)
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if response.data:
                    all_data.extend(response.data)
                    if len(response.data) < batch_size:
                        break
                    offset += batch_size
                else:
                    break

            if all_data:
                return pd.DataFrame(all_data)
            return pd.DataFrame()

        except Exception as e:
            st.error(f"Error fetching data by date range: {e}")
            return pd.DataFrame()

    def insert(self, data: Dict[str, Any]) -> bool:
        """
        Insert a single record.

        Args:
            data: Dictionary of column:value pairs

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            self.client.table(self.table_name).insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error inserting data: {e}")
            return False

    def insert_many(self, records: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple records (bulk insert).

        Args:
            records: List of dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            self.client.table(self.table_name).insert(records).execute()
            return True
        except Exception as e:
            st.error(f"Error inserting records: {e}")
            return False

    def update(self, filters: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update records matching filters.

        Args:
            filters: Dictionary of filter conditions
            data: Dictionary of values to update

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            query = self.client.table(self.table_name).update(data)

            for col, val in filters.items():
                query = query.eq(col, val)

            query.execute()
            return True
        except Exception as e:
            st.error(f"Error updating data: {e}")
            return False

    def delete(self, filters: Dict[str, Any]) -> bool:
        """
        Delete records matching filters.

        Args:
            filters: Dictionary of filter conditions

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            query = self.client.table(self.table_name).delete()

            for col, val in filters.items():
                query = query.eq(col, val)

            query.execute()
            return True
        except Exception as e:
            st.error(f"Error deleting data: {e}")
            return False

    def upsert(self, data: Dict[str, Any]) -> bool:
        """
        Insert or update a record (upsert).

        Args:
            data: Dictionary with column:value pairs (must include primary key)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            self.client.table(self.table_name).upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error upserting data: {e}")
            return False
