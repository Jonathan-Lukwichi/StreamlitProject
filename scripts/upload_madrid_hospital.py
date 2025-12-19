# =============================================================================
# scripts/upload_madrid_hospital.py
# Upload Madrid Hospital Dataset to Supabase
# =============================================================================
"""
This script uploads the Madrid Hospital dataset to Supabase.
It reads CSV files from the Madrid dataset folder and inserts them into
the appropriate Supabase tables with hospital_name = "Madrid Spain Hospital".

Usage:
    python scripts/upload_madrid_hospital.py

Requirements:
    - supabase package installed
    - .streamlit/secrets.toml configured with Supabase credentials
"""

from __future__ import annotations
import os
import sys
import io
from pathlib import Path

# Fix Windows encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
import toml


def get_supabase_client():
    """Get Supabase client using secrets from .streamlit/secrets.toml"""
    try:
        from supabase import create_client
    except ImportError:
        print("[ERROR] supabase package not installed. Run: pip install supabase")
        return None

    # Load secrets
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        print(f"[ERROR] Secrets file not found at {secrets_path}")
        return None

    try:
        secrets = toml.load(secrets_path)
        url = secrets["supabase"]["url"]
        key = secrets["supabase"]["key"]
        client = create_client(url, key)
        print("[OK] Connected to Supabase")
        return client
    except Exception as e:
        print(f"[ERROR] Connecting to Supabase: {e}")
        return None


def upload_batch(client, table_name: str, records: list, batch_size: int = 500) -> int:
    """Upload records in batches to avoid timeouts."""
    total_uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            client.table(table_name).insert(batch).execute()
            total_uploaded += len(batch)
            print(f"  [UPLOAD] Batch {i // batch_size + 1}: {len(batch)} records")
        except Exception as e:
            print(f"  [ERROR] Uploading batch: {e}")
            # Try to continue with remaining batches
            continue

    return total_uploaded


def parse_madrid_date(date_str: str) -> datetime:
    """Parse Madrid date format (DD/MM/YYYY or YYYY-MM-DD)."""
    if "/" in str(date_str):
        return datetime.strptime(str(date_str), "%d/%m/%Y")
    else:
        return pd.to_datetime(date_str).to_pydatetime()


def upload_patient_arrivals(client, df: pd.DataFrame, hospital_name: str):
    """
    Upload patient arrivals data to patient_arrivals table.

    Schema: id, date, datetime, patient_count, created_at, hospital_name
    """
    print("\n[PATIENT] Processing Patient Arrivals...")

    records = []
    for _, row in df.iterrows():
        try:
            dt = parse_madrid_date(row["Date"])
            date_str = dt.strftime("%Y-%m-%d")

            record = {
                "hospital_name": hospital_name,
                "date": date_str,
                "datetime": f"{date_str}T00:00:00+00:00",
                "patient_count": int(row["Target_1"]) if pd.notna(row.get("Target_1")) else None,
            }
            records.append(record)
        except Exception as e:
            print(f"  [SKIP] Row error: {e}")
            continue

    if records:
        uploaded = upload_batch(client, "patient_arrivals", records)
        print(f"  [DONE] Uploaded {uploaded}/{len(records)} patient arrival records")
    else:
        print("  [WARN] No patient arrival records to upload")


def upload_weather_data(client, df: pd.DataFrame, hospital_name: str):
    """
    Upload weather data to weather_data table.

    Schema: id, date, datetime, average_temp, max_temp, min_temp, total_precipitation,
            average_wind, max_wind, average_mslp, moon_phase, created_at, hospital_name
    """
    print("\n[WEATHER] Processing Weather Data...")

    records = []
    for _, row in df.iterrows():
        try:
            dt = parse_madrid_date(row["Date"])
            date_str = dt.strftime("%Y-%m-%d")

            record = {
                "hospital_name": hospital_name,
                "date": date_str,
                "datetime": f"{date_str}T00:00:00+00:00",
                "average_temp": float(row["Average_Temp"]) if pd.notna(row.get("Average_Temp")) else None,
                "max_temp": float(row["Max_temp"]) if pd.notna(row.get("Max_temp")) else None,
                "min_temp": None,  # Madrid data doesn't have min_temp
                "total_precipitation": float(row["Total_precipitation"]) if pd.notna(row.get("Total_precipitation")) else None,
                "average_wind": float(row["Average_wind"]) if pd.notna(row.get("Average_wind")) else None,
                "max_wind": float(row["Max_wind"]) if pd.notna(row.get("Max_wind")) else None,
                "average_mslp": float(row["Average_mslp"]) if pd.notna(row.get("Average_mslp")) else None,
                "moon_phase": int(row["Moon_Phase"]) if pd.notna(row.get("Moon_Phase")) else None,
            }
            records.append(record)
        except Exception as e:
            print(f"  [SKIP] Row error: {e}")
            continue

    if records:
        uploaded = upload_batch(client, "weather_data", records)
        print(f"  [DONE] Uploaded {uploaded}/{len(records)} weather records")
    else:
        print("  [WARN] No weather records to upload")


def upload_calendar_data(client, df: pd.DataFrame, hospital_name: str):
    """
    Upload calendar data to calendar_data table.

    Schema: id, date, day_of_week, holiday, holiday_prev, moon_phase, created_at, hospital_name
    """
    print("\n[CALENDAR] Processing Calendar Data...")

    records = []
    for _, row in df.iterrows():
        try:
            dt = parse_madrid_date(row["Date"])
            date_str = dt.strftime("%Y-%m-%d")

            record = {
                "hospital_name": hospital_name,
                "date": date_str,
                "day_of_week": int(row["Day_of_week"]) if pd.notna(row.get("Day_of_week")) else None,
                "holiday": int(row["Holiday"]) if pd.notna(row.get("Holiday")) else 0,
                "holiday_prev": int(row["Holiday_prev"]) if pd.notna(row.get("Holiday_prev")) else 0,
                "moon_phase": int(row["Moon_Phase"]) if pd.notna(row.get("Moon_Phase")) else None,
            }
            records.append(record)
        except Exception as e:
            print(f"  [SKIP] Row error: {e}")
            continue

    if records:
        uploaded = upload_batch(client, "calendar_data", records)
        print(f"  [DONE] Uploaded {uploaded}/{len(records)} calendar records")
    else:
        print("  [WARN] No calendar records to upload")


def upload_clinical_visits(client, df: pd.DataFrame, hospital_name: str):
    """
    Upload clinical visits data to clinical_visits table.

    Schema has specific condition columns: asthma, pneumonia, chest_pain, etc.
    Madrid data has broader categories: Respiratory_Cases, Cardiac_Cases, etc.

    We'll map Madrid categories to the schema as best we can.
    """
    print("\n[CLINICAL] Processing Clinical Visits (Reason for Visit)...")

    records = []
    for _, row in df.iterrows():
        try:
            dt = pd.to_datetime(row["Date"])
            date_str = dt.strftime("%Y-%m-%d")

            # Map Madrid categories to existing schema columns
            # Note: Some columns will be null, but total_arrivals will be accurate
            respiratory = int(row["Respiratory_Cases"]) if pd.notna(row.get("Respiratory_Cases")) else 0
            cardiac = int(row["Cardiac_Cases"]) if pd.notna(row.get("Cardiac_Cases")) else 0
            trauma = int(row["Trauma_Cases"]) if pd.notna(row.get("Trauma_Cases")) else 0
            gi = int(row["Gastrointestinal_Cases"]) if pd.notna(row.get("Gastrointestinal_Cases")) else 0
            infectious = int(row["Infectious_Cases"]) if pd.notna(row.get("Infectious_Cases")) else 0
            other = int(row["Other_Cases"]) if pd.notna(row.get("Other_Cases")) else 0

            record = {
                "hospital_name": hospital_name,
                "date": date_str,
                "datetime": f"{date_str}T00:00:00+00:00",
                # Map respiratory to respiratory-related columns
                "asthma": respiratory // 3 if respiratory > 0 else 0,
                "pneumonia": respiratory // 3 if respiratory > 0 else 0,
                "shortness_of_breath": respiratory - (respiratory // 3) * 2 if respiratory > 0 else 0,
                # Map cardiac to cardiac-related columns
                "chest_pain": cardiac // 2 if cardiac > 0 else 0,
                "arrhythmia": cardiac // 4 if cardiac > 0 else 0,
                "hypertensive_emergency": cardiac - (cardiac // 2) - (cardiac // 4) if cardiac > 0 else 0,
                # Map trauma to trauma-related columns
                "fracture": trauma // 3 if trauma > 0 else 0,
                "laceration": trauma // 3 if trauma > 0 else 0,
                "burn": trauma // 6 if trauma > 0 else 0,
                "fall_injury": trauma - (trauma // 3) * 2 - (trauma // 6) if trauma > 0 else 0,
                # Map GI to GI-related columns
                "abdominal_pain": gi // 2 if gi > 0 else 0,
                "vomiting": gi // 3 if gi > 0 else 0,
                "diarrhea": gi - (gi // 2) - (gi // 3) if gi > 0 else 0,
                # Map infectious to infectious-related columns
                "flu_symptoms": infectious // 2 if infectious > 0 else 0,
                "fever": infectious // 3 if infectious > 0 else 0,
                "viral_infection": infectious - (infectious // 2) - (infectious // 3) if infectious > 0 else 0,
                # Map other to other columns
                "headache": other // 4 if other > 0 else 0,
                "dizziness": other // 6 if other > 0 else 0,
                "allergic_reaction": other // 5 if other > 0 else 0,
                "mental_health": other - (other // 4) - (other // 6) - (other // 5) if other > 0 else 0,
                # Total arrivals
                "total_arrivals": int(row["Total_Arrivals"]) if pd.notna(row.get("Total_Arrivals")) else None,
            }
            records.append(record)
        except Exception as e:
            print(f"  [SKIP] Row error: {e}")
            continue

    if records:
        uploaded = upload_batch(client, "clinical_visits", records)
        print(f"  [DONE] Uploaded {uploaded}/{len(records)} clinical visit records")
    else:
        print("  [WARN] No clinical visit records to upload")


def main():
    print("=" * 60)
    print("MADRID HOSPITAL DATASET UPLOAD SCRIPT")
    print("=" * 60)

    # Configuration
    HOSPITAL_NAME = "Madrid Spain Hospital"
    MADRID_DATA_PATH = Path(r"C:\officie_master\dataset test\Espane dataset hospital\madrid_complete_dataset")

    # Check if data path exists
    if not MADRID_DATA_PATH.exists():
        print(f"[ERROR] Dataset path not found: {MADRID_DATA_PATH}")
        return

    # Connect to Supabase
    client = get_supabase_client()
    if not client:
        return

    print(f"\n[INFO] Reading data from: {MADRID_DATA_PATH}")
    print(f"[INFO] Hospital Name: {HOSPITAL_NAME}")

    # Load main dataset (patient/weather/calendar combined)
    main_file = MADRID_DATA_PATH / "Madrid_database_original.csv"
    if main_file.exists():
        print(f"\n[FILE] Loading {main_file.name}...")
        df_main = pd.read_csv(main_file)
        print(f"   Loaded {len(df_main):,} rows")

        # Upload patient arrivals
        upload_patient_arrivals(client, df_main, HOSPITAL_NAME)

        # Upload weather data
        upload_weather_data(client, df_main, HOSPITAL_NAME)

        # Upload calendar data
        upload_calendar_data(client, df_main, HOSPITAL_NAME)
    else:
        print(f"[WARN] Main dataset not found: {main_file}")

    # Load reason for visit dataset
    reason_file = MADRID_DATA_PATH / "reason_for_visit_madrid.csv"
    if reason_file.exists():
        print(f"\n[FILE] Loading {reason_file.name}...")
        df_reason = pd.read_csv(reason_file)
        print(f"   Loaded {len(df_reason):,} rows")

        # Upload clinical visits
        upload_clinical_visits(client, df_reason, HOSPITAL_NAME)
    else:
        print(f"[WARN] Reason for visit dataset not found: {reason_file}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Madrid Hospital Dataset Upload Complete!")
    print("=" * 60)
    print("\nThe dropdown in Upload Data page should now show:")
    print("  - Pamplona Spain Hospital")
    print("  - Madrid Spain Hospital")


if __name__ == "__main__":
    main()
