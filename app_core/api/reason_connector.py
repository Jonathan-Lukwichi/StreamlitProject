"""
Reason for Visit API Connector
Fetches detailed medical condition data from hospital clinical systems
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import requests

from .base_connector import BaseAPIConnector, APIConfig


class ReasonAPIConnector(BaseAPIConnector):
    """
    Connector for fetching reason for visit / medical condition data from hospital APIs

    Expected API Response Format:
    [
        {
            "date": "2018-01-08",
            "Asthma": 5,
            "Pneumonia": 2,
            "Fracture": 3,
            ...
        },
        ...
    ]

    Or returns granular ICD-10 codes that need to be mapped to categories
    """

    def _set_auth_header(self):
        """Set authentication header for clinical API"""
        if self.config.api_key:
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
        Fetch reason for visit data from clinical API

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            **kwargs: Additional parameters (e.g., department_id, facility_id)

        Returns:
            DataFrame with columns: Date + granular medical conditions
        """
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "format": "json",
            "aggregation": "daily"
        }

        if self.config.additional_params:
            params.update(self.config.additional_params)

        params.update(kwargs)

        # Make API request
        response = self._make_request(
            endpoint="clinical/reason_for_visit",
            method="GET",
            params=params
        )

        if not self.validate_response(response):
            raise ValueError(f"Invalid response from {self.config.api_name}")

        raw_data = self._parse_response(response)
        standardized_data = self.map_to_standard_schema(raw_data)

        return standardized_data

    def validate_response(self, response: requests.Response) -> bool:
        """Validate API response contains required data"""
        try:
            data = response.json()

            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if not isinstance(data, list) or len(data) == 0:
                return False

            first_record = data[0]
            has_date = any(field in first_record for field in ["date", "timestamp", "datetime"])

            return has_date

        except Exception:
            return False

    def _parse_response(self, response: requests.Response) -> pd.DataFrame:
        """Parse API response into DataFrame"""
        data = response.json()

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        df = pd.DataFrame(data)
        return df

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map API response to application's standard schema

        Input (API format):
            - Date/timestamp field
            - Medical condition counts (granular)

        Output (App format):
            - Date: normalized date column
            - Granular columns: Asthma, Pneumonia, Fracture, etc.
            OR
            - Aggregated categories (if API returns aggregated data)
        """
        df = raw_data.copy()

        # Find and standardize date column
        date_candidates = ["date", "timestamp", "datetime", "visit_date"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is None:
            raise ValueError("No date column found in reason API response")

        df["Date"] = pd.to_datetime(df[date_col]).dt.date

        # Expected granular medical condition columns
        expected_conditions = [
            "Asthma", "Pneumonia", "Shortness_of_Breath",
            "Chest_Pain", "Arrhythmia", "Hypertensive_Emergency",
            "Fracture", "Laceration", "Burn", "Fall_Injury",
            "Abdominal_Pain", "Vomiting", "Diarrhea",
            "Flu_Symptoms", "Fever", "Viral_Infection",
            "Headache", "Dizziness", "Allergic_Reaction", "Mental_Health",
            "Total_Arrivals"
        ]

        # Keep all numeric columns that match expected conditions or are unknown
        result_cols = ["Date"]
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                result_cols.append(col)

        result = df[result_cols].copy()
        result = result.sort_values("Date").reset_index(drop=True)

        return result


class SupabaseReasonConnector(ReasonAPIConnector):
    """
    Specialized connector for fetching reason data from Supabase

    Uses Supabase REST API to query clinical_visits table
    """

    def _set_auth_header(self):
        """Set Supabase authentication header"""
        if self.config.api_key:
            self.session.headers.update({
                "apikey": self.config.api_key,
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            })

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch from Supabase table"""
        params = {
            "date": f"gte.{start_date.strftime('%Y-%m-%d')}",
            "date": f"lte.{end_date.strftime('%Y-%m-%d')}",
            "order": "date.asc"
        }

        response = self._make_request(
            endpoint="clinical_visits",  # Table name
            method="GET",
            params=params
        )

        if not self.validate_response(response):
            raise ValueError("Invalid Supabase response")

        raw_data = self._parse_response(response)
        return self.map_to_standard_schema(raw_data)


class FHIRReasonConnector(ReasonAPIConnector):
    """
    Connector for FHIR (Fast Healthcare Interoperability Resources) APIs
    Standard used by many hospital EHR systems (Epic, Cerner, etc.)
    """

    def _set_auth_header(self):
        """Set FHIR authentication header (OAuth 2.0 typically)"""
        if self.config.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/fhir+json"
            })

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch encounter data from FHIR API

        FHIR uses Encounter resources with reasonCode for diagnosis
        """
        params = {
            "date": f"ge{start_date.strftime('%Y-%m-%d')}",
            "date": f"le{end_date.strftime('%Y-%m-%d')}",
            "_count": 1000  # Max results per page
        }

        response = self._make_request(
            endpoint="Encounter",  # FHIR resource type
            method="GET",
            params=params
        )

        if not self.validate_response(response):
            raise ValueError("Invalid FHIR response")

        # Parse FHIR bundle
        fhir_data = response.json()
        encounters = fhir_data.get("entry", [])

        # Extract and aggregate by date and condition
        aggregated_data = self._aggregate_fhir_encounters(encounters, start_date, end_date)

        df = pd.DataFrame(aggregated_data)
        return self.map_to_standard_schema(df)

    def _aggregate_fhir_encounters(self, encounters: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Aggregate FHIR encounters by date and reason code"""
        # Initialize date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Count conditions by date
        condition_counts = {date.date(): {} for date in date_range}

        for entry in encounters:
            resource = entry.get("resource", {})
            period = resource.get("period", {})
            start = period.get("start")

            if start:
                encounter_date = pd.to_datetime(start).date()

                # Extract reason codes (diagnosis)
                reason_codes = resource.get("reasonCode", [])
                for reason in reason_codes:
                    codings = reason.get("coding", [])
                    for coding in codings:
                        code = coding.get("code")
                        display = coding.get("display", code)

                        # Map ICD-10/SNOMED codes to condition names
                        condition_name = self._map_code_to_condition(code, display)

                        if condition_name:
                            if condition_name not in condition_counts[encounter_date]:
                                condition_counts[encounter_date][condition_name] = 0
                            condition_counts[encounter_date][condition_name] += 1

        # Convert to list of dicts
        result = []
        for date, conditions in condition_counts.items():
            record = {"date": date}
            record.update(conditions)
            result.append(record)

        return result

    def _map_code_to_condition(self, code: str, display: str) -> Optional[str]:
        """Map ICD-10 or SNOMED code to condition name"""
        # Simplified mapping - expand based on your needs
        icd10_mapping = {
            "J45": "Asthma",
            "J18": "Pneumonia",
            "R06": "Shortness_of_Breath",
            "R07": "Chest_Pain",
            "I49": "Arrhythmia",
            "S52": "Fracture",
            "R51": "Headache",
            # Add more mappings as needed
        }

        # Check if code starts with any mapping key
        for key, condition in icd10_mapping.items():
            if code.startswith(key):
                return condition

        return None

    def validate_response(self, response: requests.Response) -> bool:
        """Validate FHIR response"""
        try:
            data = response.json()
            return data.get("resourceType") == "Bundle"
        except Exception:
            return False


class MockReasonConnector(ReasonAPIConnector):
    """Mock reason connector for testing"""

    def _set_auth_header(self):
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Generate mock reason for visit data"""
        import numpy as np

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        n_days = len(date_range)

        # Generate realistic medical condition counts
        df = pd.DataFrame({
            "Date": date_range.date,
            "Asthma": np.random.poisson(5, n_days),
            "Pneumonia": np.random.poisson(2, n_days),
            "Shortness_of_Breath": np.random.poisson(3, n_days),
            "Chest_Pain": np.random.poisson(4, n_days),
            "Arrhythmia": np.random.poisson(1, n_days),
            "Hypertensive_Emergency": np.random.poisson(1, n_days),
            "Fracture": np.random.poisson(3, n_days),
            "Laceration": np.random.poisson(2, n_days),
            "Burn": np.random.poisson(1, n_days),
            "Fall_Injury": np.random.poisson(2, n_days),
            "Abdominal_Pain": np.random.poisson(5, n_days),
            "Vomiting": np.random.poisson(2, n_days),
            "Diarrhea": np.random.poisson(3, n_days),
            "Flu_Symptoms": np.random.poisson(4, n_days),
            "Fever": np.random.poisson(3, n_days),
            "Viral_Infection": np.random.poisson(2, n_days),
            "Headache": np.random.poisson(3, n_days),
            "Dizziness": np.random.poisson(1, n_days),
            "Allergic_Reaction": np.random.poisson(2, n_days),
            "Mental_Health": np.random.poisson(2, n_days)
        })

        # Total arrivals = sum of all conditions
        condition_cols = [c for c in df.columns if c != "Date"]
        df["Total_Arrivals"] = df[condition_cols].sum(axis=1)

        return df

    def validate_response(self, response: requests.Response) -> bool:
        return True

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return raw_data
