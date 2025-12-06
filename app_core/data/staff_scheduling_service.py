# =============================================================================
# app_core/data/staff_scheduling_service.py
# Staff Scheduling Data Service for HealthForecast AI
# Fetches staff scheduling data from Supabase
# =============================================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from .supabase_client import SupabaseService


# Supabase table name for staff scheduling data
STAFF_SCHEDULING_TABLE = "staff_scheduling"


class StaffSchedulingService(SupabaseService):
    """
    Service for Staff Scheduling data operations.

    Table Schema (staff_scheduling):
        - id: UUID (primary key, auto-generated)
        - date: DATE (scheduling date)
        - doctors_on_duty: INTEGER (number of doctors)
        - nurses_on_duty: INTEGER (number of nurses)
        - support_staff_on_duty: INTEGER (number of support staff)
        - overtime_hours: FLOAT (total overtime hours)
        - average_shift_length_hours: FLOAT (average shift duration)
        - staff_shortage_flag: BOOLEAN (0 or 1)
        - staff_utilization_rate: FLOAT (utilization percentage)
        - created_at: TIMESTAMP (auto-generated)
    """

    def __init__(self):
        """Initialize Staff Scheduling service."""
        super().__init__(STAFF_SCHEDULING_TABLE)

    def fetch_staff_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch staff scheduling data from Supabase.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with staff scheduling records
        """
        if not self.is_connected():
            st.warning("Supabase not connected. Please check your configuration.")
            return pd.DataFrame()

        try:
            if start_date and end_date:
                df = self.fetch_by_date_range(
                    date_column="date",
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )
            else:
                df = self.fetch_all(order_by="date", ascending=True)

            if df.empty:
                return df

            # Standardize column names to match expected format
            column_mapping = {
                "date": "Date",
                "doctors_on_duty": "Doctors_on_Duty",
                "nurses_on_duty": "Nurses_on_Duty",
                "support_staff_on_duty": "Support_Staff_on_Duty",
                "overtime_hours": "Overtime_Hours",
                "average_shift_length_hours": "Average_Shift_Length_Hours",
                "staff_shortage_flag": "Staff_Shortage_Flag",
                "staff_utilization_rate": "Staff_Utilization_Rate"
            }

            # Rename columns if they exist
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Ensure Date is datetime type
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])

            # Drop Supabase-specific columns if present
            cols_to_drop = ["id", "created_at", "updated_at"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

            return df

        except Exception as e:
            st.error(f"Error fetching staff scheduling data: {e}")
            return pd.DataFrame()

    def get_latest_records(self, n_days: int = 30) -> pd.DataFrame:
        """
        Fetch the most recent N days of staff data.

        Args:
            n_days: Number of days to fetch

        Returns:
            DataFrame with recent records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            response = (
                self.client.table(self.table_name)
                .select("*")
                .order("date", desc=True)
                .limit(n_days)
                .execute()
            )

            if response.data:
                df = pd.DataFrame(response.data)
                # Sort by date ascending after fetching
                if "date" in df.columns:
                    df = df.sort_values("date").reset_index(drop=True)
                return df
            return pd.DataFrame()

        except Exception as e:
            st.error(f"Error fetching latest records: {e}")
            return pd.DataFrame()

    def upload_dataframe(self, df: pd.DataFrame, replace_existing: bool = False) -> bool:
        """
        Upload a DataFrame to Supabase (bulk insert).

        Args:
            df: DataFrame to upload
            replace_existing: If True, delete all existing records first

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            # Standardize column names for Supabase (snake_case)
            column_mapping = {
                "Date": "date",
                "Doctors_on_Duty": "doctors_on_duty",
                "Nurses_on_Duty": "nurses_on_duty",
                "Support_Staff_on_Duty": "support_staff_on_duty",
                "Overtime_Hours": "overtime_hours",
                "Average_Shift_Length_Hours": "average_shift_length_hours",
                "Staff_Shortage_Flag": "staff_shortage_flag",
                "Staff_Utilization_Rate": "staff_utilization_rate"
            }

            df_upload = df.copy()
            df_upload = df_upload.rename(columns={k: v for k, v in column_mapping.items() if k in df_upload.columns})

            # Convert date to string format
            if "date" in df_upload.columns:
                df_upload["date"] = pd.to_datetime(df_upload["date"]).dt.strftime("%Y-%m-%d")

            # Remove any columns not in schema
            valid_columns = list(column_mapping.values())
            df_upload = df_upload[[c for c in df_upload.columns if c in valid_columns]]

            # Convert to records
            records = df_upload.to_dict(orient="records")

            if replace_existing:
                # Delete all existing records (use with caution!)
                self.client.table(self.table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

            # Insert new records
            return self.insert_many(records)

        except Exception as e:
            st.error(f"Error uploading data to Supabase: {e}")
            return False

    def get_staff_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for staff data.

        Returns:
            Dictionary with statistics
        """
        df = self.fetch_staff_data()

        if df.empty:
            return {}

        stats = {
            "total_records": len(df),
            "date_range": {
                "start": df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else None,
                "end": df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else None,
            },
            "averages": {},
            "totals": {}
        }

        numeric_cols = ["Doctors_on_Duty", "Nurses_on_Duty", "Support_Staff_on_Duty",
                       "Overtime_Hours", "Average_Shift_Length_Hours", "Staff_Utilization_Rate"]

        for col in numeric_cols:
            if col in df.columns:
                stats["averages"][col] = float(df[col].mean())
                if col in ["Doctors_on_Duty", "Nurses_on_Duty", "Support_Staff_on_Duty"]:
                    stats["totals"][col] = int(df[col].sum())

        if "Staff_Shortage_Flag" in df.columns:
            stats["shortage_days"] = int(df["Staff_Shortage_Flag"].sum())
            stats["shortage_percentage"] = float(df["Staff_Shortage_Flag"].mean() * 100)

        return stats


# Convenience function for quick access
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_staff_scheduling_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Cached function to fetch staff scheduling data.

    Args:
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        DataFrame with staff scheduling data
    """
    service = StaffSchedulingService()
    return service.fetch_staff_data(start_date, end_date)


def check_supabase_connection() -> bool:
    """
    Check if Supabase is properly configured and connected.

    Returns:
        True if connected, False otherwise
    """
    service = StaffSchedulingService()
    return service.is_connected()
