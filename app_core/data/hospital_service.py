# =============================================================================
# app_core/data/hospital_service.py
# Hospital Data Service - Fetch all datasets filtered by hospital name
# =============================================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Optional, List
from datetime import datetime

from .supabase_client import get_cached_supabase_client, SupabaseService


class HospitalDataService:
    """
    Service to fetch all hospital datasets (patient, weather, calendar, reason)
    filtered by hospital name.
    """

    # Table names in Supabase (must match your actual table names)
    TABLES = {
        "patient": "patient_arrivals",
        "weather": "weather_data",
        "calendar": "calendar_data",
        "reason": "clinical_visits",
    }

    def __init__(self):
        """Initialize the hospital data service."""
        self.client = get_cached_supabase_client()

    def is_connected(self) -> bool:
        """Check if Supabase client is available."""
        return self.client is not None

    def get_available_hospitals(self) -> List[str]:
        """
        Fetch list of all available hospitals from Supabase.

        Returns:
            List of hospital names
        """
        if not self.is_connected():
            return ["Pamplona Spain Hospital"]  # Default fallback

        try:
            # Try to fetch from the view first
            response = self.client.table("available_hospitals").select("hospital_name").execute()

            if response.data:
                hospitals = [row["hospital_name"] for row in response.data if row.get("hospital_name")]
                if hospitals:
                    return sorted(hospitals)

        except Exception:
            pass

        # Fallback: query patient_data table directly
        try:
            response = (
                self.client.table("patient_data")
                .select("hospital_name")
                .limit(1000)
                .execute()
            )

            if response.data:
                hospitals = list(set(row.get("hospital_name") for row in response.data if row.get("hospital_name")))
                if hospitals:
                    return sorted(hospitals)

        except Exception:
            pass

        # Default if nothing found
        return ["Pamplona Spain Hospital"]

    def fetch_dataset_by_hospital(
        self,
        dataset_type: str,
        hospital_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch a specific dataset filtered by hospital name.

        Args:
            dataset_type: One of 'patient', 'weather', 'calendar', 'reason'
            hospital_name: Name of the hospital to filter by
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with the filtered data
        """
        if not self.is_connected():
            return pd.DataFrame()

        table_name = self.TABLES.get(dataset_type)
        if not table_name:
            st.error(f"Unknown dataset type: {dataset_type}")
            return pd.DataFrame()

        try:
            all_data = []
            batch_size = 1000
            offset = 0

            while True:
                query = (
                    self.client.table(table_name)
                    .select("*")
                    .eq("hospital_name", hospital_name)
                )

                # Add date filters if provided
                date_column = "datetime" if dataset_type != "calendar" else "datetime"

                if start_date:
                    query = query.gte(date_column, start_date.strftime("%Y-%m-%d"))
                if end_date:
                    query = query.lte(date_column, end_date.strftime("%Y-%m-%d"))

                # Order and paginate
                query = query.order(date_column).range(offset, offset + batch_size - 1)
                response = query.execute()

                if response.data:
                    all_data.extend(response.data)
                    if len(response.data) < batch_size:
                        break
                    offset += batch_size
                else:
                    break

            if all_data:
                df = pd.DataFrame(all_data)
                # Remove hospital_name column from the result (not needed in app)
                if "hospital_name" in df.columns:
                    df = df.drop(columns=["hospital_name"])
                return df

            return pd.DataFrame()

        except Exception as e:
            st.error(f"Error fetching {dataset_type} data: {e}")
            return pd.DataFrame()

    def fetch_all_datasets_by_hospital(
        self,
        hospital_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback=None,
    ) -> dict:
        """
        Fetch all 4 datasets for a specific hospital.

        Args:
            hospital_name: Name of the hospital
            start_date: Optional start date filter
            end_date: Optional end date filter
            progress_callback: Optional callback function(dataset_type, status)

        Returns:
            Dictionary with keys: patient, weather, calendar, reason
            Each value is a DataFrame (may be empty if fetch failed)
        """
        results = {}

        for dataset_type in ["patient", "weather", "calendar", "reason"]:
            if progress_callback:
                progress_callback(dataset_type, "fetching")

            df = self.fetch_dataset_by_hospital(
                dataset_type=dataset_type,
                hospital_name=hospital_name,
                start_date=start_date,
                end_date=end_date,
            )

            results[dataset_type] = df

            if progress_callback:
                status = "success" if not df.empty else "empty"
                progress_callback(dataset_type, status)

        return results


def get_hospital_service() -> HospitalDataService:
    """Get or create the hospital data service."""
    return HospitalDataService()
