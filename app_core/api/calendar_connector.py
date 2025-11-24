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
