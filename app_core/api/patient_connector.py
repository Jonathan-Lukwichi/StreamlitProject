"""
Patient Data API Connector
Fetches ED arrival/patient count data from hospital systems or EHR APIs
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import requests

from .base_connector import BaseAPIConnector, APIConfig


class PatientAPIConnector(BaseAPIConnector):
    """
    Connector for fetching patient arrival data from hospital APIs

    Expected API Response Format:
    [
        {"date": "2018-01-08", "patient_count": 58, "timestamp": "2018-01-08T00:00:00"},
        {"date": "2018-01-09", "patient_count": 46, "timestamp": "2018-01-09T00:00:00"},
        ...
    ]

    Or CSV format with columns: date, patient_count
    """

    def _set_auth_header(self):
        """Set authentication header for hospital API"""
        if self.config.api_key:
            # Common patterns for hospital APIs
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            })

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch patient arrival data from hospital API

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            **kwargs: Additional parameters (e.g., department_id, facility_id)

        Returns:
            DataFrame with columns: datetime, Target_1 (patient count)
        """
        # Build query parameters
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "format": "json"
        }

        # Add optional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)

        params.update(kwargs)

        # Make API request
        response = self._make_request(
            endpoint="patient_arrivals",  # Common endpoint name
            method="GET",
            params=params
        )

        # Validate response
        if not self.validate_response(response):
            raise ValueError(f"Invalid response from {self.config.api_name}")

        # Parse response to DataFrame
        raw_data = self._parse_response(response)

        # Map to standard schema
        standardized_data = self.map_to_standard_schema(raw_data)

        return standardized_data

    def validate_response(self, response: requests.Response) -> bool:
        """Validate API response contains required data"""
        try:
            data = response.json()

            # Check if response is a list of records
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if not isinstance(data, list) or len(data) == 0:
                return False

            # Check first record has required fields
            first_record = data[0]
            required_fields = ["date", "patient_count"]  # or ["timestamp", "count"]

            # Flexible field checking
            has_date = any(field in first_record for field in ["date", "timestamp", "datetime"])
            has_count = any(field in first_record for field in ["patient_count", "count", "arrivals", "visits"])

            return has_date and has_count

        except Exception:
            return False

    def _parse_response(self, response: requests.Response) -> pd.DataFrame:
        """Parse API response into DataFrame"""
        data = response.json()

        # Handle nested response structure
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        return df

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map API response to application's standard schema

        Input (API format):
            - Various date fields: date, timestamp, datetime
            - Various count fields: patient_count, count, arrivals, visits, ed_arrivals

        Output (App format):
            - datetime: normalized datetime column
            - Target_1: patient count
        """
        df = raw_data.copy()

        # Step 1: Find and standardize date column
        date_candidates = ["timestamp", "datetime", "date", "date_time", "ds"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is None:
            raise ValueError("No date column found in patient API response")

        # Convert to datetime
        df["datetime"] = pd.to_datetime(df[date_col])

        # Step 2: Find and standardize patient count column
        count_candidates = ["patient_count", "count", "arrivals", "visits", "ed_arrivals", "patients", "total_patients"]
        count_col = None
        for candidate in count_candidates:
            if candidate in df.columns:
                count_col = candidate
                break

        if count_col is None:
            raise ValueError("No patient count column found in API response")

        # Rename to Target_1 (app standard)
        df["Target_1"] = pd.to_numeric(df[count_col], errors="coerce")

        # Step 3: Keep only required columns
        result = df[["datetime", "Target_1"]].copy()

        # Sort by date
        result = result.sort_values("datetime").reset_index(drop=True)

        return result


class SupabasePatientConnector(PatientAPIConnector):
    """
    Specialized connector for fetching patient data from Supabase

    Uses Supabase REST API to query patient_arrivals table
    """

    def _set_auth_header(self):
        """Set Supabase authentication header"""
        if self.config.api_key:
            self.session.headers.update({
                "apikey": self.config.api_key,
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            })

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch from Supabase table with proper date range filtering"""
        # Supabase uses PostgREST API with special filter syntax
        params = {
            "date": f"gte.{start_date.strftime('%Y-%m-%d')}",
            "date": f"lte.{end_date.strftime('%Y-%m-%d')}",
            "order": "date.asc",
            "select": "*"
        }

        response = self._make_request(
            endpoint="patient_arrivals",  # Table name
            method="GET",
            params=params
        )

        if not self.validate_response(response):
            raise ValueError("Invalid Supabase response")

        raw_data = self._parse_response(response)

        if len(raw_data) == 0:
            raise ValueError(f"No data found in Supabase for date range {start_date.date()} to {end_date.date()}")

        return self.map_to_standard_schema(raw_data)


class MockPatientConnector(PatientAPIConnector):
    """
    Mock connector for testing - generates synthetic patient data
    Useful for demos and development
    """

    def _set_auth_header(self):
        """No auth needed for mock"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Generate mock patient data"""
        import numpy as np

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate realistic patient counts with weekly seasonality
        np.random.seed(42)
        base_count = 50
        trend = np.linspace(0, 10, len(date_range))
        weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)
        noise = np.random.normal(0, 5, len(date_range))

        patient_counts = base_count + trend + weekly_pattern + noise
        patient_counts = np.maximum(patient_counts, 20).astype(int)  # Min 20 patients

        df = pd.DataFrame({
            "datetime": date_range,
            "Target_1": patient_counts
        })

        return df

    def validate_response(self, response: requests.Response) -> bool:
        """Always valid for mock"""
        return True
