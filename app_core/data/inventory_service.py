# =============================================================================
# app_core/data/inventory_service.py
# Supabase Service for Inventory Management Data
# =============================================================================
"""
Inventory Management Service for Supabase Integration

Handles CRUD operations for inventory data stored in Supabase.
Table: inventory_management

Schema:
    - id: UUID (auto-generated)
    - Date: date
    - Inventory_Used_Gloves: int
    - Inventory_Used_PPE_Sets: int
    - Inventory_Used_Medications: int
    - Inventory_Level_Gloves: int
    - Inventory_Level_PPE: int
    - Inventory_Level_Medications: int
    - Restock_Event: int (0/1)
    - Stockout_Risk_Score: float
    - created_at: timestamp
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import streamlit as st

from .supabase_client import SupabaseClient


# Table name in Supabase
INVENTORY_TABLE = "inventory_management"


class InventoryService:
    """
    Service class for inventory management data operations.
    """

    def __init__(self):
        """Initialize with Supabase client."""
        self.client = SupabaseClient()
        self.table = INVENTORY_TABLE

    def is_connected(self) -> bool:
        """Check if Supabase connection is active."""
        return self.client.is_connected()

    def fetch_all(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch all inventory records with pagination to bypass 1000 row limit.

        Args:
            limit: Optional limit on records (None = all)

        Returns:
            DataFrame with all inventory data
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_records = []
            batch_size = 1000
            offset = 0

            while True:
                # Fetch batch
                response = (
                    self.client.supabase
                    .table(self.table)
                    .select("*")
                    .order("Date")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                all_records.extend(response.data)

                # Check if we got a full batch (more data might exist)
                if len(response.data) < batch_size:
                    break

                offset += batch_size

                # Apply limit if specified
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records)
            return self._clean_dataframe(df)

        except Exception as e:
            st.error(f"Error fetching inventory data: {str(e)}")
            return pd.DataFrame()

    def fetch_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch inventory records within a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with filtered records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_records = []
            batch_size = 1000
            offset = 0

            start_str = start_date.isoformat()
            end_str = end_date.isoformat()

            while True:
                response = (
                    self.client.supabase
                    .table(self.table)
                    .select("*")
                    .gte("Date", start_str)
                    .lte("Date", end_str)
                    .order("Date")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                all_records.extend(response.data)

                if len(response.data) < batch_size:
                    break

                offset += batch_size

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records)
            return self._clean_dataframe(df)

        except Exception as e:
            st.error(f"Error fetching inventory data: {str(e)}")
            return pd.DataFrame()

    def fetch_latest(self, n_days: int = 30) -> pd.DataFrame:
        """
        Fetch the most recent n days of inventory data.

        Args:
            n_days: Number of recent days to fetch

        Returns:
            DataFrame with recent records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            response = (
                self.client.supabase
                .table(self.table)
                .select("*")
                .order("Date", desc=True)
                .limit(n_days)
                .execute()
            )

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df = self._clean_dataframe(df)

            # Sort by date ascending
            if "Date" in df.columns:
                df = df.sort_values("Date").reset_index(drop=True)

            return df

        except Exception as e:
            st.error(f"Error fetching latest inventory data: {str(e)}")
            return pd.DataFrame()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from inventory data.

        Returns:
            Dict with summary statistics
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        stats = {
            "total_records": len(df),
            "date_range": {
                "start": df["Date"].min() if "Date" in df.columns else None,
                "end": df["Date"].max() if "Date" in df.columns else None,
            },
            "avg_usage": {
                "gloves": df["Inventory_Used_Gloves"].mean() if "Inventory_Used_Gloves" in df.columns else 0,
                "ppe": df["Inventory_Used_PPE_Sets"].mean() if "Inventory_Used_PPE_Sets" in df.columns else 0,
                "medications": df["Inventory_Used_Medications"].mean() if "Inventory_Used_Medications" in df.columns else 0,
            },
            "avg_levels": {
                "gloves": df["Inventory_Level_Gloves"].mean() if "Inventory_Level_Gloves" in df.columns else 0,
                "ppe": df["Inventory_Level_PPE"].mean() if "Inventory_Level_PPE" in df.columns else 0,
                "medications": df["Inventory_Level_Medications"].mean() if "Inventory_Level_Medications" in df.columns else 0,
            },
            "restock_events": df["Restock_Event"].sum() if "Restock_Event" in df.columns else 0,
            "avg_stockout_risk": df["Stockout_Risk_Score"].mean() if "Stockout_Risk_Score" in df.columns else 0,
        }

        return stats

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format the dataframe.

        Args:
            df: Raw dataframe from Supabase

        Returns:
            Cleaned dataframe
        """
        if df.empty:
            return df

        # Drop Supabase internal columns
        drop_cols = ["id", "created_at", "id_numeric"]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Convert Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # Ensure numeric columns are correct type
        numeric_cols = [
            "Inventory_Used_Gloves", "Inventory_Used_PPE_Sets", "Inventory_Used_Medications",
            "Inventory_Level_Gloves", "Inventory_Level_PPE", "Inventory_Level_Medications",
            "Restock_Event"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Stockout risk as float
        if "Stockout_Risk_Score" in df.columns:
            df["Stockout_Risk_Score"] = pd.to_numeric(df["Stockout_Risk_Score"], errors="coerce")

        return df

    def insert_records(self, df: pd.DataFrame) -> bool:
        """
        Insert inventory records into Supabase.

        Args:
            df: DataFrame with inventory data

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        try:
            # Prepare records
            records = []
            for _, row in df.iterrows():
                record = {
                    "Date": row["Date"].isoformat() if hasattr(row["Date"], "isoformat") else str(row["Date"]),
                    "Inventory_Used_Gloves": int(row.get("Inventory_Used_Gloves", 0)),
                    "Inventory_Used_PPE_Sets": int(row.get("Inventory_Used_PPE_Sets", 0)),
                    "Inventory_Used_Medications": int(row.get("Inventory_Used_Medications", 0)),
                    "Inventory_Level_Gloves": int(row.get("Inventory_Level_Gloves", 0)),
                    "Inventory_Level_PPE": int(row.get("Inventory_Level_PPE", 0)),
                    "Inventory_Level_Medications": int(row.get("Inventory_Level_Medications", 0)),
                    "Restock_Event": int(row.get("Restock_Event", 0)),
                    "Stockout_Risk_Score": float(row.get("Stockout_Risk_Score", 0)),
                }
                records.append(record)

            # Insert in batches
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.client.supabase.table(self.table).insert(batch).execute()

            return True

        except Exception as e:
            st.error(f"Error inserting inventory data: {str(e)}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_inventory_service() -> InventoryService:
    """Get or create InventoryService instance."""
    if "inventory_service" not in st.session_state:
        st.session_state["inventory_service"] = InventoryService()
    return st.session_state["inventory_service"]


def fetch_inventory_data(limit: int = None) -> pd.DataFrame:
    """
    Convenience function to fetch inventory data.

    Args:
        limit: Optional limit on records

    Returns:
        DataFrame with inventory data
    """
    service = get_inventory_service()
    return service.fetch_all(limit)


def check_inventory_connection() -> bool:
    """Check if Supabase connection is available."""
    service = get_inventory_service()
    return service.is_connected()
