"""
ONE-TIME HOSPITAL DATABASE SETUP SCRIPT

This script simulates the "Hospital IT Team" uploading historical data to the hospital database.
Run this ONCE to populate Supabase with your CSV data.

Usage:
    python setup_hospital_database.py --all
    python setup_hospital_database.py --patient "path/to/patient.csv"
    python setup_hospital_database.py --clear --all

Requirements:
    pip install supabase pandas python-dotenv
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Error: 'supabase' package not installed")
    print("Run: pip install supabase")
    sys.exit(1)


class HospitalDatabaseSetup:
    """Upload CSV data to Supabase (simulates hospital IT team)"""

    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.stats = {
            "patient": 0,
            "weather": 0,
            "calendar": 0,
            "clinical": 0
        }

    def clear_tables(self):
        """Clear all data from tables (for testing new datasets)"""
        print("\n🗑️  Clearing all tables...")

        tables = [
            "patient_arrivals",
            "weather_data",
            "calendar_data",
            "clinical_visits"
        ]

        for table in tables:
            try:
                # Delete all rows
                self.client.table(table).delete().neq("id", 0).execute()
                print(f"   ✅ Cleared {table}")
            except Exception as e:
                print(f"   ⚠️  {table}: {str(e)}")

        print("✅ All tables cleared!\n")

    def upload_patient_data(self, csv_path: str) -> int:
        """Upload patient arrivals data"""
        print(f"\n📊 Uploading Patient Data from {Path(csv_path).name}...")

        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows")

        # Map columns to Supabase schema
        df_mapped = pd.DataFrame()

        # Find date column
        date_col = self._find_date_column(df, ["datetime", "Date", "date", "timestamp"])
        if not date_col:
            raise ValueError("No date column found in patient CSV")

        df_mapped["datetime"] = pd.to_datetime(df[date_col])
        df_mapped["date"] = df_mapped["datetime"].dt.date

        # Find patient count column
        count_col = self._find_column(df, ["Target_1", "patient_count", "count", "patients", "arrivals"])
        if not count_col:
            raise ValueError("No patient count column found")

        df_mapped["patient_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)

        # Upload in batches
        return self._upload_batch(df_mapped, "patient_arrivals", "patient")

    def upload_weather_data(self, csv_path: str) -> int:
        """Upload weather observations data"""
        print(f"\n🌤️  Uploading Weather Data from {Path(csv_path).name}...")

        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows")

        df_mapped = pd.DataFrame()

        # Find date column
        date_col = self._find_date_column(df, ["datetime", "Date", "date", "timestamp"])
        if not date_col:
            raise ValueError("No date column found in weather CSV")

        df_mapped["datetime"] = pd.to_datetime(df[date_col])
        df_mapped["date"] = df_mapped["datetime"].dt.date

        # Map weather columns
        df_mapped["average_temp"] = self._get_numeric_col(df, ["Average_Temp", "avg_temp", "temperature"])
        df_mapped["max_temp"] = self._get_numeric_col(df, ["Max_temp", "max_temperature"])
        df_mapped["min_temp"] = self._get_numeric_col(df, ["Min_temp", "min_temperature"])
        df_mapped["total_precipitation"] = self._get_numeric_col(df, ["Total_precipitation", "precipitation", "precip"])
        df_mapped["average_wind"] = self._get_numeric_col(df, ["Average_wind", "avg_wind", "wind_speed"])
        df_mapped["max_wind"] = self._get_numeric_col(df, ["Max_wind", "max_wind_speed"])
        df_mapped["average_mslp"] = self._get_numeric_col(df, ["Average_mslp", "avg_mslp", "pressure"])
        df_mapped["moon_phase"] = self._get_numeric_col(df, ["Moon_Phase", "moon_phase"], default=0, as_int=True)

        return self._upload_batch(df_mapped, "weather_data", "weather")

    def upload_calendar_data(self, csv_path: str) -> int:
        """Upload calendar/holiday data"""
        print(f"\n📅 Uploading Calendar Data from {Path(csv_path).name}...")

        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows")

        df_mapped = pd.DataFrame()

        # Find date column
        date_col = self._find_date_column(df, ["Date", "date", "datetime"])
        if not date_col:
            raise ValueError("No date column found in calendar CSV")

        df_mapped["date"] = pd.to_datetime(df[date_col]).dt.date

        # Map calendar columns
        df_mapped["day_of_week"] = self._get_numeric_col(df, ["Day_of_week", "day_of_week", "weekday"], default=0, as_int=True)
        df_mapped["holiday"] = self._get_numeric_col(df, ["Holiday", "holiday", "is_holiday"], default=0, as_int=True)
        df_mapped["holiday_prev"] = self._get_numeric_col(df, ["Holiday_prev", "holiday_prev"], default=0, as_int=True)
        df_mapped["moon_phase"] = self._get_numeric_col(df, ["Moon_Phase", "moon_phase"], default=0, as_int=True)

        return self._upload_batch(df_mapped, "calendar_data", "calendar")

    def upload_clinical_data(self, csv_path: str) -> int:
        """Upload clinical visit conditions data"""
        print(f"\n🏥 Uploading Clinical Visit Data from {Path(csv_path).name}...")

        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows")

        df_mapped = pd.DataFrame()

        # Find date column
        date_col = self._find_date_column(df, ["datetime", "Date", "date", "timestamp"])
        if not date_col:
            raise ValueError("No date column found in clinical CSV")

        df_mapped["datetime"] = pd.to_datetime(df[date_col])
        df_mapped["date"] = df_mapped["datetime"].dt.date

        # Map all medical condition columns (case-insensitive)
        conditions = [
            "asthma", "pneumonia", "shortness_of_breath",
            "chest_pain", "arrhythmia", "hypertensive_emergency",
            "fracture", "laceration", "burn", "fall_injury",
            "abdominal_pain", "vomiting", "diarrhea",
            "flu_symptoms", "fever", "viral_infection",
            "headache", "dizziness", "allergic_reaction", "mental_health",
            "total_arrivals"
        ]

        for condition in conditions:
            df_mapped[condition] = self._get_numeric_col(df, [condition, condition.replace("_", " ").title()], default=0, as_int=True)

        return self._upload_batch(df_mapped, "clinical_visits", "clinical")

    def _upload_batch(self, df: pd.DataFrame, table_name: str, stats_key: str, batch_size: int = 500) -> int:
        """Upload dataframe in batches with progress"""
        total_rows = len(df)
        uploaded = 0

        # Convert DataFrame to list of dicts
        records = df.to_dict('records')

        # Convert dates to strings for JSON serialization
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    record[key] = value.isoformat()
                elif hasattr(value, 'date'):  # date object
                    record[key] = str(value)

        print(f"   Uploading {total_rows} records in batches of {batch_size}...")

        for i in range(0, total_rows, batch_size):
            batch = records[i:i + batch_size]

            try:
                # Upsert (insert or update)
                self.client.table(table_name).upsert(batch).execute()
                uploaded += len(batch)
                print(f"   Progress: {uploaded}/{total_rows} ({int(uploaded/total_rows*100)}%)")

            except Exception as e:
                print(f"   ⚠️  Error uploading batch {i}-{i+batch_size}: {str(e)}")

        self.stats[stats_key] = uploaded
        print(f"   ✅ Uploaded {uploaded} records to {table_name}")
        return uploaded

    def _find_date_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find date column (case-insensitive)"""
        df_lower = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_lower:
                return df_lower[candidate.lower()]
        return None

    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find column by name (case-insensitive)"""
        df_lower = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_lower:
                return df_lower[candidate.lower()]
        return None

    def _get_numeric_col(self, df: pd.DataFrame, candidates: list, default=None, as_int=False) -> pd.Series:
        """Get numeric column with fallback"""
        col_name = self._find_column(df, candidates)

        if col_name:
            series = pd.to_numeric(df[col_name], errors="coerce")
        else:
            series = pd.Series([default] * len(df))

        if default is not None:
            series = series.fillna(default)

        if as_int:
            series = series.astype(int)

        return series

    def print_summary(self):
        """Print upload summary"""
        print("\n" + "="*60)
        print("✅ HOSPITAL DATABASE SETUP COMPLETE!")
        print("="*60)
        print(f"\n📊 Patient Arrivals:     {self.stats['patient']:,} records")
        print(f"🌤️  Weather Data:         {self.stats['weather']:,} records")
        print(f"📅 Calendar Data:        {self.stats['calendar']:,} records")
        print(f"🏥 Clinical Visits:      {self.stats['clinical']:,} records")
        print(f"\n📈 TOTAL:                {sum(self.stats.values()):,} records")
        print("\n🔒 Database is now READ-ONLY for API access")
        print("✅ Your Streamlit app can now fetch data via API!")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Upload CSV data to Supabase (simulates hospital IT setup)"
    )

    parser.add_argument("--supabase-url", help="Supabase project URL")
    parser.add_argument("--supabase-key", help="Supabase API key")
    parser.add_argument("--patient", help="Path to patient CSV file")
    parser.add_argument("--weather", help="Path to weather CSV file")
    parser.add_argument("--calendar", help="Path to calendar CSV file")
    parser.add_argument("--reason", help="Path to reason/clinical CSV file")
    parser.add_argument("--all", action="store_true", help="Upload all datasets from default paths")
    parser.add_argument("--clear", action="store_true", help="Clear tables before uploading")

    args = parser.parse_args()

    # Get Supabase credentials from args or environment
    supabase_url = args.supabase_url
    supabase_key = args.supabase_key

    # Try loading from .streamlit/secrets.toml if not provided
    if not supabase_url or not supabase_key:
        try:
            import toml
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                secrets = toml.load(secrets_path)
                supabase_url = supabase_url or secrets.get("supabase", {}).get("url")
                supabase_key = supabase_key or secrets.get("supabase", {}).get("key")
        except Exception:
            pass

    if not supabase_url or not supabase_key:
        print("❌ Error: Supabase credentials not found")
        print("\nProvide via arguments:")
        print("  --supabase-url YOUR_URL --supabase-key YOUR_KEY")
        print("\nOr create .streamlit/secrets.toml:")
        print("  [supabase]")
        print("  url = \"https://xxx.supabase.co\"")
        print("  key = \"your_anon_key\"")
        sys.exit(1)

    # Initialize setup
    setup = HospitalDatabaseSetup(supabase_url, supabase_key)

    # Clear tables if requested
    if args.clear:
        setup.clear_tables()

    # Upload datasets
    uploaded_any = False

    if args.all:
        # Default paths (adjust to your actual paths)
        base_path = Path(r"c:\officie_master\dataset test\Espane dataset hospital\pamplona_complete_dataset")

        default_paths = {
            "patient": base_path / "Patient_Data.csv",
            "weather": base_path / "Weather_Data.csv",
            "calendar": base_path / "Calendar_Data.csv",
            "reason": base_path / "Reason_for_Visit_Detailed.csv"
        }

        for dataset_type, path in default_paths.items():
            if path.exists():
                if dataset_type == "patient":
                    setup.upload_patient_data(str(path))
                elif dataset_type == "weather":
                    setup.upload_weather_data(str(path))
                elif dataset_type == "calendar":
                    setup.upload_calendar_data(str(path))
                elif dataset_type == "reason":
                    setup.upload_clinical_data(str(path))
                uploaded_any = True
            else:
                print(f"⚠️  File not found: {path}")

    else:
        # Upload individual datasets
        if args.patient:
            setup.upload_patient_data(args.patient)
            uploaded_any = True

        if args.weather:
            setup.upload_weather_data(args.weather)
            uploaded_any = True

        if args.calendar:
            setup.upload_calendar_data(args.calendar)
            uploaded_any = True

        if args.reason:
            setup.upload_clinical_data(args.reason)
            uploaded_any = True

    if uploaded_any:
        setup.print_summary()
    else:
        print("❌ No files uploaded. Use --all or specify individual files.")
        parser.print_help()


if __name__ == "__main__":
    main()
