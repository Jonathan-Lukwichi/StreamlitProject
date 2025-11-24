"""
Weather Data API Connectors
Supports multiple weather APIs: OpenWeatherMap, WeatherAPI.com, Visual Crossing, NOAA
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests

from .base_connector import BaseAPIConnector, APIConfig


class WeatherAPIConnector(BaseAPIConnector):
    """Base connector for weather APIs"""

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map weather API response to application's standard schema

        Output (App format):
            - datetime: normalized datetime column
            - Date: date only
            - Average_Temp, Max_temp, Min_temp
            - Total_precipitation
            - Average_wind, Max_wind
            - Average_mslp (pressure)
        """
        # To be implemented by specific weather API connectors
        pass


class OpenWeatherConnector(WeatherAPIConnector):
    """
    Connector for OpenWeatherMap API (Historical Weather Data)
    Requires paid subscription for historical data
    """

    def _set_auth_header(self):
        """OpenWeather uses API key in URL params, not headers"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical weather data from OpenWeatherMap

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: lat, lon (required), units='metric'

        Returns:
            DataFrame with weather features
        """
        lat = kwargs.get("lat") or self.config.additional_params.get("lat")
        lon = kwargs.get("lon") or self.config.additional_params.get("lon")

        if not lat or not lon:
            raise ValueError("Latitude and longitude required for OpenWeatherMap")

        all_data = []

        # OpenWeather historical API requires date-by-date requests
        current_date = start_date
        while current_date <= end_date:
            timestamp = int(current_date.timestamp())

            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": self.config.api_key,
                "units": kwargs.get("units", "metric")
            }

            response = self._make_request(
                endpoint="data/2.5/onecall/timemachine",
                method="GET",
                params=params
            )

            if self.validate_response(response):
                data = response.json()
                all_data.append(self._parse_daily_data(data, current_date))

            current_date += timedelta(days=1)

        # Combine all daily data
        df = pd.DataFrame(all_data)
        return self.map_to_standard_schema(df)

    def _parse_daily_data(self, data: Dict, date: datetime) -> Dict:
        """Parse OpenWeather response for a single day"""
        current = data.get("current", {})

        return {
            "date": date,
            "temp": current.get("temp"),
            "temp_min": current.get("temp"),  # Daily data has single temp
            "temp_max": current.get("temp"),
            "pressure": current.get("pressure"),
            "humidity": current.get("humidity"),
            "wind_speed": current.get("wind_speed"),
            "precipitation": data.get("hourly", [{}])[0].get("rain", {}).get("1h", 0)
        }

    def validate_response(self, response: requests.Response) -> bool:
        """Validate OpenWeather response"""
        try:
            data = response.json()
            return "current" in data
        except Exception:
            return False

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map OpenWeather format to app standard"""
        df = raw_data.copy()

        df["datetime"] = pd.to_datetime(df["date"])
        df["Date"] = df["datetime"].dt.date
        df["Average_Temp"] = df["temp"]
        df["Max_temp"] = df["temp_max"]
        df["Min_temp"] = df["temp_min"]
        df["Average_mslp"] = df["pressure"]
        df["Average_wind"] = df["wind_speed"]
        df["Max_wind"] = df["wind_speed"] * 1.2  # Estimate
        df["Total_precipitation"] = df["precipitation"]

        result = df[[
            "datetime", "Date", "Average_Temp", "Max_temp", "Min_temp",
            "Total_precipitation", "Average_wind", "Max_wind", "Average_mslp"
        ]].copy()

        return result.sort_values("datetime").reset_index(drop=True)


class VisualCrossingConnector(WeatherAPIConnector):
    """
    Connector for Visual Crossing Weather API
    Supports historical weather data with generous free tier
    """

    def _set_auth_header(self):
        """Visual Crossing uses API key in URL"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical weather from Visual Crossing

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: location (required), unitGroup='metric'

        Returns:
            DataFrame with weather features
        """
        location = kwargs.get("location") or self.config.additional_params.get("location")

        if not location:
            raise ValueError("Location required for Visual Crossing (e.g., 'Madrid,Spain')")

        params = {
            "key": self.config.api_key,
            "unitGroup": kwargs.get("unitGroup", "metric"),
            "include": "days",
            "contentType": "json"
        }

        # Format: location/start_date/end_date
        date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        endpoint = f"VisualCrossingWebServices/rest/services/timeline/{location}/{date_range}"

        response = self._make_request(
            endpoint=endpoint,
            method="GET",
            params=params
        )

        if not self.validate_response(response):
            raise ValueError("Invalid Visual Crossing response")

        raw_data = self._parse_response(response)
        return self.map_to_standard_schema(raw_data)

    def _parse_response(self, response: requests.Response) -> pd.DataFrame:
        """Parse Visual Crossing response"""
        data = response.json()
        days = data.get("days", [])

        df = pd.DataFrame(days)
        return df

    def validate_response(self, response: requests.Response) -> bool:
        """Validate Visual Crossing response"""
        try:
            data = response.json()
            return "days" in data and len(data["days"]) > 0
        except Exception:
            return False

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map Visual Crossing format to app standard"""
        df = raw_data.copy()

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["Date"] = df["datetime"].dt.date
        df["Average_Temp"] = df["temp"]
        df["Max_temp"] = df["tempmax"]
        df["Min_temp"] = df["tempmin"]
        df["Average_mslp"] = df["pressure"]
        df["Average_wind"] = df["windspeed"]
        df["Max_wind"] = df["windgust"].fillna(df["windspeed"] * 1.5)
        df["Total_precipitation"] = df["precip"].fillna(0)

        result = df[[
            "datetime", "Date", "Average_Temp", "Max_temp", "Min_temp",
            "Total_precipitation", "Average_wind", "Max_wind", "Average_mslp"
        ]].copy()

        return result.sort_values("datetime").reset_index(drop=True)


class WeatherAPIComConnector(WeatherAPIConnector):
    """
    Connector for WeatherAPI.com
    Good free tier for historical data (7 days back on free plan)
    """

    def _set_auth_header(self):
        """WeatherAPI.com uses key in URL params"""
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical weather from WeatherAPI.com

        Args:
            start_date: Start date
            end_date: End date
            **kwargs: q (location, required)

        Returns:
            DataFrame with weather features
        """
        location = kwargs.get("q") or self.config.additional_params.get("location")

        if not location:
            raise ValueError("Location required for WeatherAPI.com (e.g., 'Madrid')")

        all_data = []

        # WeatherAPI.com requires date-by-date requests
        current_date = start_date
        while current_date <= end_date:
            params = {
                "key": self.config.api_key,
                "q": location,
                "dt": current_date.strftime("%Y-%m-%d")
            }

            response = self._make_request(
                endpoint="v1/history.json",
                method="GET",
                params=params
            )

            if self.validate_response(response):
                data = response.json()
                day_data = data["forecast"]["forecastday"][0]["day"]
                day_data["date"] = current_date
                all_data.append(day_data)

            current_date += timedelta(days=1)

        df = pd.DataFrame(all_data)
        return self.map_to_standard_schema(df)

    def validate_response(self, response: requests.Response) -> bool:
        """Validate WeatherAPI.com response"""
        try:
            data = response.json()
            return "forecast" in data
        except Exception:
            return False

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map WeatherAPI.com format to app standard"""
        df = raw_data.copy()

        df["datetime"] = pd.to_datetime(df["date"])
        df["Date"] = df["datetime"].dt.date
        df["Average_Temp"] = df["avgtemp_c"]
        df["Max_temp"] = df["maxtemp_c"]
        df["Min_temp"] = df["mintemp_c"]
        df["Average_mslp"] = df.get("avgpressure_mb", 1013)  # Default if missing
        df["Average_wind"] = df["avgwind_kph"] / 3.6  # Convert to m/s
        df["Max_wind"] = df["maxwind_kph"] / 3.6
        df["Total_precipitation"] = df["totalprecip_mm"]

        result = df[[
            "datetime", "Date", "Average_Temp", "Max_temp", "Min_temp",
            "Total_precipitation", "Average_wind", "Max_wind", "Average_mslp"
        ]].copy()

        return result.sort_values("datetime").reset_index(drop=True)


class MockWeatherConnector(WeatherAPIConnector):
    """Mock weather connector for testing"""

    def _set_auth_header(self):
        pass

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Generate mock weather data"""
        import numpy as np

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        np.random.seed(42)
        n_days = len(date_range)

        # Realistic weather patterns
        base_temp = 15
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        temp_noise = np.random.normal(0, 3, n_days)
        avg_temp = base_temp + seasonal + temp_noise

        df = pd.DataFrame({
            "datetime": date_range,
            "Date": date_range.date,
            "Average_Temp": avg_temp,
            "Max_temp": avg_temp + np.random.uniform(2, 5, n_days),
            "Min_temp": avg_temp - np.random.uniform(2, 5, n_days),
            "Total_precipitation": np.random.exponential(2, n_days),
            "Average_wind": np.random.uniform(1, 5, n_days),
            "Max_wind": np.random.uniform(5, 10, n_days),
            "Average_mslp": np.random.normal(1013, 10, n_days)
        })

        return df

    def validate_response(self, response: requests.Response) -> bool:
        return True

    def map_to_standard_schema(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return raw_data
