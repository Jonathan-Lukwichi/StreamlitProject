"""
Base API Connector Class for Data Ingestion
Provides abstract interface for fetching data from various APIs
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for API connection"""
    api_name: str
    base_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    rate_limit_delay: float = 0.0  # seconds between requests
    additional_params: Optional[Dict[str, Any]] = None


class BaseAPIConnector(ABC):
    """Abstract base class for all API connectors"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()

        # Set default headers
        if config.headers:
            self.session.headers.update(config.headers)

        # Add API key to headers if provided
        if config.api_key:
            self._set_auth_header()

    @abstractmethod
    def _set_auth_header(self):
        """Set authentication header based on API requirements"""
        pass

    @abstractmethod
    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from API for specified date range

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            **kwargs: Additional API-specific parameters

        Returns:
            DataFrame with standardized columns
        """
        pass

    @abstractmethod
    def validate_response(self, response: requests.Response) -> bool:
        """Validate API response"""
        pass

    @abstractmethod
    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map API response to application's standard schema

        Args:
            raw_data: DataFrame with API-specific column names

        Returns:
            DataFrame with standardized column names expected by the app
        """
        pass

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> requests.Response:
        """
        Make HTTP request with error handling

        Args:
            endpoint: API endpoint (appended to base_url)
            method: HTTP method (GET, POST, etc.)
            params: Query parameters (can be dict or list of tuples)
            data: Request body data

        Returns:
            Response object
        """
        url = f"{self.config.base_url}/{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed for {self.config.api_name}: {str(e)}")

    def get_available_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the available date range from the API

        Returns:
            Tuple of (earliest_date, latest_date)
        """
        # Default implementation - override in subclasses
        return datetime(2018, 1, 1), datetime.now()

    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and return status

        Returns:
            Dict with status, message, and metadata
        """
        try:
            # Try a minimal request to check connectivity
            start_date = datetime.now() - timedelta(days=1)
            end_date = datetime.now()
            df = self.fetch_data(start_date, end_date)

            return {
                "status": "success",
                "message": f"Successfully connected to {self.config.api_name}",
                "rows_fetched": len(df),
                "columns": list(df.columns)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}"
            }
