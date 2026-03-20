# =============================================================================
# scripts/upload_steve_biko_data.py
# Upload Steve Biko Hospital data to Supabase
# =============================================================================

from __future__ import annotations
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import toml
from datetime import datetime

# =============================================================================
# DATA FILE PATHS (Steve Biko Hospital)
# =============================================================================
STEVE_BIKO_DATA = {
    "patient": r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\data transformation pipline\healthforecast_pipeline\healthforecast_pipeline\intermediate\raw_extracted\all_daily_extracted.csv",
    "weather": r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\data steve biko\external factor data\pretoria_weather_2019_2026.csv",
    "calendar": r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\data steve biko\external factor data\calendar_features_2019_2026.csv",
    "clinical": r"C:\Users\BIBINBUSINESS\OneDrive\Desktop\data transformation pipline\healthforecast_pipeline\healthforecast_pipeline\intermediate\clinical\clinical_reasons_daily.csv",
}

HOSPITAL_NAME = "Steve Biko Hospital"

# =============================================================================
# COLUMN MAPPINGS (Steve Biko format -> App expected format)
# =============================================================================

def map_patient_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Steve Biko patient data columns to app format.

    Target schema (patient_arrivals):
    - id, date, datetime, patient_count, created_at, hospital_name
    """
    df = df.copy()

    # Ensure datetime is proper format
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")

    # patient_count already exists in source
    if "patient_count" not in df.columns:
        # Try to calculate from other columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df["patient_count"] = df[numeric_cols[0]]

    # Add hospital name
    df["hospital_name"] = HOSPITAL_NAME

    # Select only columns that match the target schema
    output_columns = ["date", "datetime", "patient_count", "hospital_name"]
    existing_cols = [c for c in output_columns if c in df.columns]

    return df[existing_cols]


def map_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Steve Biko weather data columns to app format.

    Target schema (weather_data):
    - id, date, datetime, average_temp, max_temp, min_temp,
      total_precipitation, average_wind, max_wind, average_mslp,
      moon_phase, created_at, hospital_name
    """
    df = df.copy()

    # Ensure datetime is proper format
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")

    # Map weather columns (Steve Biko uses Celsius, Pamplona used Kelvin)
    # Convert to match existing format or keep as-is
    if "temp_mean_C" in df.columns:
        # Convert Celsius to Kelvin to match existing Pamplona data
        df["average_temp"] = df["temp_mean_C"] + 273.15
    if "temp_max_C" in df.columns:
        df["max_temp"] = df["temp_max_C"] + 273.15
    if "temp_min_C" in df.columns:
        df["min_temp"] = df["temp_min_C"] + 273.15

    # Wind (convert km/h to m/s to match Pamplona format: divide by 3.6)
    if "wind_max_kmh" in df.columns:
        df["max_wind"] = df["wind_max_kmh"] / 3.6
    if "wind_gust_max_kmh" in df.columns:
        df["average_wind"] = df["wind_gust_max_kmh"] / 3.6
    elif "wind_max_kmh" in df.columns:
        df["average_wind"] = df["wind_max_kmh"] / 3.6

    # Precipitation
    if "precipitation_mm" in df.columns:
        df["total_precipitation"] = df["precipitation_mm"]

    # Moon phase - Steve Biko doesn't have this, set to 0
    df["moon_phase"] = 0

    # MSLP - not available
    df["average_mslp"] = None

    # Add hospital name
    df["hospital_name"] = HOSPITAL_NAME

    # Select only columns that match the target schema
    output_columns = ["date", "datetime", "average_temp", "max_temp", "min_temp",
                      "total_precipitation", "average_wind", "max_wind",
                      "average_mslp", "moon_phase", "hospital_name"]
    existing_cols = [c for c in output_columns if c in df.columns]

    return df[existing_cols]


def map_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Steve Biko calendar data columns to app format.

    Target schema (calendar_data):
    - id, date, day_of_week, holiday, holiday_prev, moon_phase,
      created_at, hospital_name
    """
    df = df.copy()

    # Ensure date is proper format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # day_of_week: Steve Biko uses 0-6 (Mon=0), convert to 1-7 (Mon=1)
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"] + 1

    # Holiday mapping
    if "is_public_holiday" in df.columns:
        df["holiday"] = df["is_public_holiday"].astype(int)
    else:
        df["holiday"] = 0

    # Holiday_prev
    if "is_day_before_holiday" in df.columns:
        df["holiday_prev"] = df["is_day_before_holiday"].astype(int)
    else:
        df["holiday_prev"] = 0

    # Moon phase - not available in Steve Biko data, use default
    df["moon_phase"] = 0

    # Add hospital name
    df["hospital_name"] = HOSPITAL_NAME

    # Select only columns that match the target schema
    output_columns = ["date", "day_of_week", "holiday", "holiday_prev",
                      "moon_phase", "hospital_name"]
    existing_cols = [c for c in output_columns if c in df.columns]

    return df[existing_cols]


def map_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Steve Biko clinical/reason data columns to app format."""
    df = df.copy()

    # Rename date column
    df = df.rename(columns={"date": "datetime"})

    # Ensure datetime is proper format
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Create aggregated clinical categories from Steve Biko's detailed data
    # Map to app's expected categories: RESPIRATORY, CARDIAC, TRAUMA, etc.

    # TRAUMA = accidents + violence + falls
    if "total_trauma" in df.columns:
        df["TRAUMA"] = df["total_trauma"]
    else:
        trauma_cols = ["total_accidents", "total_violence", "total_falls"]
        existing_cols = [c for c in trauma_cols if c in df.columns]
        if existing_cols:
            df["TRAUMA"] = df[existing_cols].fillna(0).sum(axis=1)

    # INFECTIOUS = tb, malaria, measles, meningitis
    infectious_cols = ["tb", "malaria", "measles", "meningitis", "total_infectious_disease"]
    existing_inf = [c for c in infectious_cols if c in df.columns]
    if "total_infectious_disease" in df.columns:
        df["INFECTIOUS"] = df["total_infectious_disease"]
    elif existing_inf:
        df["INFECTIOUS"] = df[existing_inf].fillna(0).sum(axis=1)

    # GASTROINTESTINAL - use poisoning as proxy
    if "total_poisoning" in df.columns:
        df["GASTROINTESTINAL"] = df["total_poisoning"]

    # NEUROLOGICAL - use substance abuse as proxy (affects CNS)
    if "total_substance_abuse" in df.columns:
        df["NEUROLOGICAL"] = df["total_substance_abuse"]

    # OTHER - animal bites + gynae + suicide attempts
    other_cols = ["total_animal_bites", "total_gynae", "total_suicide_attempts"]
    existing_other = [c for c in other_cols if c in df.columns]
    if existing_other:
        df["OTHER"] = df[existing_other].fillna(0).sum(axis=1)

    # RESPIRATORY and CARDIAC - not directly available, set to 0 or estimate
    # Could use spec_medicine as a proxy
    if "spec_medicine" in df.columns:
        # Split medicine into respiratory/cardiac estimates
        df["RESPIRATORY"] = (df["spec_medicine"] * 0.4).fillna(0).astype(int)
        df["CARDIAC"] = (df["spec_medicine"] * 0.3).fillna(0).astype(int)
    else:
        df["RESPIRATORY"] = 0
        df["CARDIAC"] = 0

    # Add hospital name
    df["hospital_name"] = HOSPITAL_NAME

    return df


def upload_to_supabase(client, table_name: str, df: pd.DataFrame, batch_size: int = 500):
    """Upload dataframe to Supabase table in batches."""
    total_rows = len(df)
    uploaded = 0
    errors = []

    # Convert datetime columns to ISO format strings for JSON serialization
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Replace NaN with None for JSON compatibility
    df = df.where(pd.notnull(df), None)

    # Upload in batches
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size].to_dict(orient="records")

        try:
            response = client.table(table_name).upsert(batch).execute()
            uploaded += len(batch)
            print(f"  Uploaded {uploaded}/{total_rows} rows...")
        except Exception as e:
            errors.append(f"Batch {i}-{i+batch_size}: {e}")
            print(f"  Error in batch {i}-{i+batch_size}: {e}")

    return uploaded, errors


def main():
    print("=" * 70)
    print("STEVE BIKO HOSPITAL DATA UPLOAD TO SUPABASE")
    print("=" * 70)

    # Load Supabase credentials
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        print(f"ERROR: Secrets file not found at {secrets_path}")
        return

    secrets = toml.load(secrets_path)

    from supabase import create_client
    client = create_client(secrets["supabase"]["url"], secrets["supabase"]["key"])

    # Table mappings
    table_mapping = {
        "patient": "patient_arrivals",
        "weather": "weather_data",
        "calendar": "calendar_data",
        "clinical": "clinical_visits",
    }

    mapper_functions = {
        "patient": map_patient_columns,
        "weather": map_weather_columns,
        "calendar": map_calendar_columns,
        "clinical": map_clinical_columns,
    }

    results = {}

    for dataset_type, file_path in STEVE_BIKO_DATA.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_type.upper()}")
        print(f"{'='*60}")

        # Check if file exists
        if not Path(file_path).exists():
            print(f"  ERROR: File not found: {file_path}")
            results[dataset_type] = {"status": "error", "message": "File not found"}
            continue

        try:
            # Read CSV
            print(f"  Reading: {file_path}")
            df = pd.read_csv(file_path)
            print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

            # Apply column mapping
            mapper = mapper_functions.get(dataset_type)
            if mapper:
                df = mapper(df)
                print(f"  Mapped columns, now {len(df.columns)} columns")

            # Upload to Supabase
            table_name = table_mapping[dataset_type]
            print(f"  Uploading to table: {table_name}")

            uploaded, errors = upload_to_supabase(client, table_name, df)

            results[dataset_type] = {
                "status": "success" if not errors else "partial",
                "uploaded": uploaded,
                "total": len(df),
                "errors": errors,
            }

            print(f"  DONE: {uploaded}/{len(df)} rows uploaded")
            if errors:
                print(f"  WARNINGS: {len(errors)} batch errors")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[dataset_type] = {"status": "error", "message": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    for dataset_type, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            print(f"  {dataset_type}: SUCCESS - {result['uploaded']} rows")
        elif status == "partial":
            print(f"  {dataset_type}: PARTIAL - {result['uploaded']}/{result['total']} rows")
        else:
            print(f"  {dataset_type}: ERROR - {result.get('message', 'Unknown error')}")

    print("\n" + "=" * 70)
    print(f"Hospital '{HOSPITAL_NAME}' should now appear in the Upload Data page dropdown!")
    print("=" * 70)


if __name__ == "__main__":
    main()
