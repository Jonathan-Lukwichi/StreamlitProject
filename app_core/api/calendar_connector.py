"""
Calendar Data API Connectors
Fetches holiday and calendar information from various sources
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import requests

from .base_connector import BaseAPIConnector, APIConfig


class CalendarAPIConnector(BaseAPIConnector):
    """Base connector for calendar/holiday APIs"""

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map calendar API response to application's standard schema

        Output (App format):
            - Date: date column
            - Day_of_week: day number (0=Monday, 6=Sunday)
            - Holiday: binary flag (0 or 1)
            - Holiday_prev: if previous day was holiday
            - Moon_Phase: optional moon phase data
        """
        pass


class AbstractAPIHolidaysConnector(CalendarAPIConnector):
    """
    Connector for AbstractAPI Holidays
    Free tier: 1000 requests/month
    https://www.abstractapi.com/holidays-api
    """

    def _set_auth_header(self):
        """AbstractAPI uses API key in URL params"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch holiday data from AbstractAPI

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: country (required, e.g., 'ES' for Spain)

        Returns:
            DataFrame with calendar features
        """
        country = kwargs.get("country") or self.config.additional_params.get("country")

        if not country:
            raise ValueError("Country code required for AbstractAPI (e.g., 'ES', 'US', 'FR')")

        all_holidays = []

        # AbstractAPI requires year-by-year requests
        for year in range(start_date.year, end_date.year + 1):
            params = {
                "api_key": self.config.api_key,
                "country": country,
                "year": year
            }

            response = self._make_request(
                endpoint="v1",
                method="GET",
                params=params
            )

            if self.validate_response(response):
                holidays = response.json()
                all_holidays.extend(holidays)

        # Create full date range with holiday flags
        df = self._create_calendar_dataframe(start_date, end_date, all_holidays)
        return self.map_to_standard_schema(df)

    def _create_calendar_dataframe(self, start_date: datetime, end_date: datetime, holidays: List[Dict]) -> pd.DataFrame:
        """Create calendar DataFrame with holiday flags"""
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Extract holiday dates
        holiday_dates = set()
        for holiday in holidays:
            holiday_date = pd.to_datetime(holiday.get("date"))
            if start_date <= holiday_date <= end_date:
                holiday_dates.add(holiday_date.date())

        # Build DataFrame
        df = pd.DataFrame({
            "date": date_range,
            "day_of_week": date_range.dayofweek,
            "is_holiday": [d.date() in holiday_dates for d in date_range]
        })

        return df

    def validate_response(self, response: requests.Response) -> bool:
        """Validate AbstractAPI response"""
        try:
            data = response.json()
            return isinstance(data, list)
        except Exception:
            return False

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map AbstractAPI format to app standard"""
        df = raw_data.copy()

        df["Date"] = pd.to_datetime(df["date"]).dt.date
        df["Day_of_week"] = df["day_of_week"]
        df["Holiday"] = df["is_holiday"].astype(int)

        # Holiday_prev: was previous day a holiday?
        df["Holiday_prev"] = df["Holiday"].shift(1).fillna(0).astype(int)

        # Moon_Phase: placeholder (0-7, 0=new moon, 4=full moon)
        # Would need separate moon phase API
        df["Moon_Phase"] = 0

        result = df[["Date", "Day_of_week", "Holiday", "Holiday_prev", "Moon_Phase"]].copy()
        return result.sort_values("Date").reset_index(drop=True)


class CalendarificConnector(CalendarAPIConnector):
    """
    Connector for Calendarific API
    Free tier: 1000 requests/month
    https://calendarific.com/
    """

    def _set_auth_header(self):
        """Calendarific uses API key in URL params"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch holiday data from Calendarific"""
        country = kwargs.get("country") or self.config.additional_params.get("country")

        if not country:
            raise ValueError("Country code required for Calendarific")

        all_holidays = []

        for year in range(start_date.year, end_date.year + 1):
            params = {
                "api_key": self.config.api_key,
                "country": country,
                "year": year
            }

            response = self._make_request(
                endpoint="v2/holidays",
                method="GET",
                params=params
            )

            if self.validate_response(response):
                data = response.json()
                all_holidays.extend(data["response"]["holidays"])

        df = self._create_calendar_dataframe(start_date, end_date, all_holidays)
        return self.map_to_standard_schema(df)

    def _create_calendar_dataframe(self, start_date: datetime, end_date: datetime, holidays: List[Dict]) -> pd.DataFrame:
        """Create calendar DataFrame from Calendarific holidays"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Extract holiday dates
        holiday_dates = set()
        for holiday in holidays:
            date_obj = holiday["date"]["iso"]
            holiday_date = pd.to_datetime(date_obj)
            if start_date <= holiday_date <= end_date:
                holiday_dates.add(holiday_date.date())

        df = pd.DataFrame({
            "date": date_range,
            "day_of_week": date_range.dayofweek,
            "is_holiday": [d.date() in holiday_dates for d in date_range]
        })

        return df

    def validate_response(self, response: requests.Response) -> bool:
        """Validate Calendarific response"""
        try:
            data = response.json()
            return "response" in data and "holidays" in data["response"]
        except Exception:
            return False

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map Calendarific format to app standard"""
        df = raw_data.copy()

        df["Date"] = pd.to_datetime(df["date"]).dt.date
        df["Day_of_week"] = df["day_of_week"]
        df["Holiday"] = df["is_holiday"].astype(int)
        df["Holiday_prev"] = df["Holiday"].shift(1).fillna(0).astype(int)
        df["Moon_Phase"] = 0  # Placeholder

        result = df[["Date", "Day_of_week", "Holiday", "Holiday_prev", "Moon_Phase"]].copy()
        return result.sort_values("Date").reset_index(drop=True)


class MockCalendarConnector(CalendarAPIConnector):
    """Mock calendar connector for testing"""

    def _set_auth_header(self):
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Generate mock calendar data with realistic holiday patterns"""
        import numpy as np

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Simulate holidays: ~10 per year, typically on weekends or specific dates
        np.random.seed(42)
        n_days = len(date_range)

        # Base holiday probability
        holiday_prob = 0.03  # ~10-11 holidays per year

        # Higher probability on weekends
        is_weekend = date_range.dayofweek >= 5
        holiday_flags = np.random.random(n_days) < holiday_prob
        holiday_flags = holiday_flags | (is_weekend & (np.random.random(n_days) < 0.1))

        df = pd.DataFrame({
            "Date": date_range.date,
            "Day_of_week": date_range.dayofweek,
            "Holiday": holiday_flags.astype(int),
            "Moon_Phase": np.random.randint(0, 8, n_days)
        })

        # Holiday_prev
        df["Holiday_prev"] = df["Holiday"].shift(1).fillna(0).astype(int)

        return df[["Date", "Day_of_week", "Holiday", "Holiday_prev", "Moon_Phase"]]

    def validate_response(self, response: requests.Response) -> bool:
        return True

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return raw_data


class SupabaseCalendarConnector(CalendarAPIConnector):
    """Connector for Supabase calendar_data table"""

    def _set_auth_header(self):
        """Set Supabase authentication headers"""
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
        """
        Fetch calendar data from Supabase with pagination
        Supabase has 1000 row per request limit, so we fetch in batches
        """
        # Format dates for PostgREST query
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        all_data = []
        offset = 0
        batch_size = 1000  # Supabase/PostgREST default limit

        while True:
            # PostgREST filter syntax for date range
            # Use list of tuples to allow duplicate "date" parameter keys
            params = [
                ("date", f"gte.{start_str}"),  # Greater than or equal
                ("date", f"lte.{end_str}"),    # Less than or equal
                ("order", "date.asc"),
                ("select", "*"),
                ("limit", str(batch_size)),
                ("offset", str(offset))
            ]

            # Make request to calendar_data table
            response = self._make_request(
                endpoint="calendar_data",
                method="GET",
                params=params
            )

            # Parse JSON response
            raw_data = response.json()

            if len(raw_data) == 0:
                # No more data to fetch
                break

            all_data.extend(raw_data)
            offset += batch_size

            # If we got less than batch_size rows, we've reached the end
            if len(raw_data) < batch_size:
                break

        if len(all_data) == 0:
            raise ValueError(
                f"No data found in Supabase for date range {start_str} to {end_str}. "
                f"Please check that data exists in the calendar_data table."
            )

        df = pd.DataFrame(all_data)

        # Map to standard schema
        return self.map_to_standard_schema(df)

    def validate_response(self, response: requests.Response) -> bool:
        """Validate Supabase response"""
        if response.status_code == 200:
            return True

        error_msg = f"Supabase API error (status {response.status_code})"
        try:
            error_detail = response.json()
            error_msg += f": {error_detail}"
        except:
            error_msg += f": {response.text}"

        raise ValueError(error_msg)

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map Supabase columns to standard calendar schema"""
        result = pd.DataFrame()

        # Parse date
        if "date" in raw_data.columns:
            result["Date"] = pd.to_datetime(raw_data["date"]).dt.date

        # Map calendar columns (handle both upper and lowercase)
        if "day_of_week" in raw_data.columns:
            result["Day_of_week"] = pd.to_numeric(raw_data["day_of_week"], errors='coerce').astype(int)

        if "holiday" in raw_data.columns:
            result["Holiday"] = pd.to_numeric(raw_data["holiday"], errors='coerce').astype(int)

        if "holiday_prev" in raw_data.columns:
            result["Holiday_prev"] = pd.to_numeric(raw_data["holiday_prev"], errors='coerce').astype(int)

        if "moon_phase" in raw_data.columns:
            result["Moon_Phase"] = pd.to_numeric(raw_data["moon_phase"], errors='coerce').astype(int)

        return result.sort_values("Date").reset_index(drop=True)
