# =============================================================================
# app_core/data/financial_service.py
# Supabase Service for Hospital Financial Data
# =============================================================================
"""
Financial Data Service for Supabase Integration

Handles CRUD operations for financial data stored in Supabase.
Table: financial_data

Schema includes:
    - Date, Labor costs, Inventory costs, Revenue, Operating costs
    - Optimization metrics (overstaffing, understaffing penalties)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import streamlit as st

from .supabase_client import get_cached_supabase_client


# Table name in Supabase
FINANCIAL_TABLE = "financial_data"


class FinancialService:
    """
    Service class for financial data operations.
    """

    def __init__(self):
        """Initialize with Supabase client."""
        self.supabase = get_cached_supabase_client()
        self.table = FINANCIAL_TABLE

    def is_connected(self) -> bool:
        """Check if Supabase connection is active."""
        return self.supabase is not None

    def fetch_all(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch all financial records with pagination to bypass 1000 row limit.

        Args:
            limit: Optional limit on records (None = all)

        Returns:
            DataFrame with all financial data
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
                    self.supabase
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
            st.error(f"Error fetching financial data: {str(e)}")
            return pd.DataFrame()

    def fetch_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch financial records within a date range.

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
                    self.supabase
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
            st.error(f"Error fetching financial data: {str(e)}")
            return pd.DataFrame()

    def fetch_latest(self, n_days: int = 30) -> pd.DataFrame:
        """
        Fetch the most recent n days of financial data.

        Args:
            n_days: Number of recent days to fetch

        Returns:
            DataFrame with recent records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            response = (
                self.supabase
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
            st.error(f"Error fetching latest financial data: {str(e)}")
            return pd.DataFrame()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from financial data.

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
            "avg_labor_cost": {
                "doctor_rate": df["Doctor_Hourly_Rate"].mean() if "Doctor_Hourly_Rate" in df.columns else 0,
                "nurse_rate": df["Nurse_Hourly_Rate"].mean() if "Nurse_Hourly_Rate" in df.columns else 0,
                "support_rate": df["Support_Staff_Hourly_Rate"].mean() if "Support_Staff_Hourly_Rate" in df.columns else 0,
                "total_labor": df["Total_Labor_Cost"].mean() if "Total_Labor_Cost" in df.columns else 0,
            },
            "avg_inventory_cost": {
                "gloves": df["Gloves_Cost"].mean() if "Gloves_Cost" in df.columns else 0,
                "ppe": df["PPE_Cost"].mean() if "PPE_Cost" in df.columns else 0,
                "medication": df["Medication_Cost"].mean() if "Medication_Cost" in df.columns else 0,
                "total": df["Total_Inventory_Cost"].mean() if "Total_Inventory_Cost" in df.columns else 0,
            },
            "avg_revenue": df["Total_Revenue"].mean() if "Total_Revenue" in df.columns else 0,
            "avg_profit": df["Daily_Profit"].mean() if "Daily_Profit" in df.columns else 0,
            "avg_profit_margin": df["Profit_Margin_Percent"].mean() if "Profit_Margin_Percent" in df.columns else 0,
        }

        return stats

    def get_labor_cost_params(self) -> Dict[str, float]:
        """
        Get average labor cost parameters for optimization.

        Returns:
            Dict with hourly rates and multipliers
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        params = {}

        if "Doctor_Hourly_Rate" in df.columns:
            params["doctor_hourly_rate"] = float(df["Doctor_Hourly_Rate"].mean())

        if "Nurse_Hourly_Rate" in df.columns:
            params["nurse_hourly_rate"] = float(df["Nurse_Hourly_Rate"].mean())

        if "Support_Staff_Hourly_Rate" in df.columns:
            params["support_hourly_rate"] = float(df["Support_Staff_Hourly_Rate"].mean())

        if "Overtime_Premium_Rate" in df.columns:
            params["overtime_multiplier"] = float(df["Overtime_Premium_Rate"].mean())

        if "Understaffing_Penalty" in df.columns:
            params["understaffing_penalty"] = float(df["Understaffing_Penalty"].mean())

        if "Overstaffing_Cost" in df.columns:
            params["overstaffing_penalty"] = float(df["Overstaffing_Cost"].mean())

        return params

    def get_inventory_cost_params(self) -> Dict[str, float]:
        """
        Get average inventory cost parameters for optimization.

        Returns:
            Dict with unit costs and penalties
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        params = {}

        if "Cost_Per_Glove_Pair" in df.columns:
            params["gloves_unit_cost"] = float(df["Cost_Per_Glove_Pair"].mean())

        if "Cost_Per_PPE_Set" in df.columns:
            params["ppe_unit_cost"] = float(df["Cost_Per_PPE_Set"].mean())

        if "Cost_Per_Medication_Unit" in df.columns:
            params["medication_unit_cost"] = float(df["Cost_Per_Medication_Unit"].mean())

        if "Inventory_Holding_Cost" in df.columns:
            params["holding_cost"] = float(df["Inventory_Holding_Cost"].mean())

        if "Restock_Order_Cost" in df.columns:
            params["ordering_cost"] = float(df["Restock_Order_Cost"].mean())

        if "Stockout_Penalty" in df.columns:
            params["stockout_penalty"] = float(df["Stockout_Penalty"].mean())

        if "Overstock_Penalty" in df.columns:
            params["overstock_penalty"] = float(df["Overstock_Penalty"].mean())

        return params

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
            "Doctor_Hourly_Rate", "Nurse_Hourly_Rate", "Support_Staff_Hourly_Rate",
            "Overtime_Premium_Rate", "Doctor_Daily_Cost", "Nurse_Daily_Cost",
            "Support_Staff_Daily_Cost", "Overtime_Cost", "Agency_Staff_Cost",
            "Total_Labor_Cost", "Cost_Per_Glove_Pair", "Cost_Per_PPE_Set",
            "Cost_Per_Medication_Unit", "Gloves_Cost", "PPE_Cost", "Medication_Cost",
            "Total_Inventory_Usage_Cost", "Inventory_Holding_Cost", "Restock_Order_Cost",
            "Emergency_Procurement_Cost", "Inventory_Wastage_Cost", "Total_Inventory_Cost",
            "Revenue_Per_Patient_Avg", "Total_Daily_Revenue", "Government_Subsidy",
            "Insurance_Reimbursement", "Total_Revenue", "Utility_Cost",
            "Equipment_Maintenance_Cost", "Administrative_Cost", "Facility_Cost",
            "Total_Operating_Cost", "Daily_Profit", "Profit_Margin_Percent",
            "Cost_Per_Patient", "Revenue_Per_Patient", "Labor_Cost_Ratio",
            "Inventory_Cost_Ratio", "Budget_Variance", "Budget_Variance_Percent",
            "Labor_Cost_Per_Patient", "Inventory_Cost_Per_Patient", "Overstaffing_Cost",
            "Understaffing_Penalty", "Stockout_Penalty", "Overstock_Penalty",
            "Total_Optimization_Cost"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def insert_records(self, df: pd.DataFrame) -> bool:
        """
        Insert financial records into Supabase.

        Args:
            df: DataFrame with financial data

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
                }

                # Add all numeric columns
                numeric_cols = [col for col in df.columns if col != "Date"]
                for col in numeric_cols:
                    if col in row:
                        val = row[col]
                        if pd.notna(val):
                            record[col] = float(val)
                        else:
                            record[col] = 0.0

                records.append(record)

            # Insert in batches
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.supabase.table(self.table).insert(batch).execute()

            return True

        except Exception as e:
            st.error(f"Error inserting financial data: {str(e)}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_financial_service() -> FinancialService:
    """Get or create FinancialService instance."""
    if "financial_service" not in st.session_state:
        st.session_state["financial_service"] = FinancialService()
    return st.session_state["financial_service"]


def fetch_financial_data(limit: int = None) -> pd.DataFrame:
    """
    Convenience function to fetch financial data.

    Args:
        limit: Optional limit on records

    Returns:
        DataFrame with financial data
    """
    service = get_financial_service()
    return service.fetch_all(limit)


def check_financial_connection() -> bool:
    """Check if Supabase connection is available."""
    service = get_financial_service()
    return service.is_connected()
