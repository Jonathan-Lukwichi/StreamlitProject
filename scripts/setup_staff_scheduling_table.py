"""
setup_staff_scheduling_table.py
================================
Script to create staff_scheduling table in Supabase and upload data.

Usage:
    python scripts/setup_staff_scheduling_table.py --csv path/to/staff_data.csv

Or run without arguments to just create the table schema.
"""

import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_supabase_client():
    """Get Supabase client from secrets."""
    try:
        import streamlit as st
        from supabase import create_client

        # Load secrets from .streamlit/secrets.toml
        import toml
        secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"

        if not secrets_path.exists():
            print("ERROR: .streamlit/secrets.toml not found!")
            print("Please configure your Supabase credentials first.")
            return None

        secrets = toml.load(secrets_path)

        url = secrets["supabase"]["url"]
        key = secrets["supabase"]["key"]

        return create_client(url, key)

    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        print("Run: pip install supabase toml")
        return None
    except Exception as e:
        print(f"ERROR: Failed to create Supabase client - {e}")
        return None


def print_sql_schema():
    """Print SQL schema for staff_scheduling table."""
    sql = """
-- ============================================================================
-- STAFF SCHEDULING TABLE SCHEMA FOR SUPABASE
-- ============================================================================
-- Run this SQL in your Supabase SQL Editor to create the table

CREATE TABLE IF NOT EXISTS staff_scheduling (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL,
    doctors_on_duty INTEGER NOT NULL DEFAULT 0,
    nurses_on_duty INTEGER NOT NULL DEFAULT 0,
    support_staff_on_duty INTEGER NOT NULL DEFAULT 0,
    overtime_hours FLOAT DEFAULT 0.0,
    average_shift_length_hours FLOAT DEFAULT 8.0,
    staff_shortage_flag BOOLEAN DEFAULT FALSE,
    staff_utilization_rate FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),

    -- Ensure unique date entries
    CONSTRAINT unique_date UNIQUE (date)
);

-- Create index for faster date queries
CREATE INDEX IF NOT EXISTS idx_staff_scheduling_date ON staff_scheduling(date);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE staff_scheduling ENABLE ROW LEVEL SECURITY;

-- Create policy for anonymous read access
CREATE POLICY "Allow anonymous read access" ON staff_scheduling
    FOR SELECT USING (true);

-- Create policy for authenticated insert/update (optional)
CREATE POLICY "Allow authenticated write access" ON staff_scheduling
    FOR ALL USING (auth.role() = 'authenticated');

-- Grant necessary permissions
GRANT SELECT ON staff_scheduling TO anon;
GRANT ALL ON staff_scheduling TO authenticated;
"""
    print(sql)
    return sql


def upload_csv_to_supabase(csv_path: str):
    """
    Upload CSV data to Supabase staff_scheduling table.

    Args:
        csv_path: Path to CSV file with staff scheduling data
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Standardize column names
    column_mapping = {
        "Date": "date",
        "Doctors_on_Duty": "doctors_on_duty",
        "Nurses_on_Duty": "nurses_on_duty",
        "Support_Staff_on_Duty": "support_staff_on_duty",
        "Overtime_Hours": "overtime_hours",
        "Average_Shift_Length_Hours": "average_shift_length_hours",
        "Staff_Shortage_Flag": "staff_shortage_flag",
        "Staff_Utilization_Rate": "staff_utilization_rate"
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Ensure date is string format for Supabase
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Convert boolean
    if "staff_shortage_flag" in df.columns:
        df["staff_shortage_flag"] = df["staff_shortage_flag"].astype(bool)

    # Get client
    client = get_supabase_client()
    if client is None:
        return False

    # Convert to records
    records = df.to_dict(orient="records")

    print(f"Uploading {len(records)} records to Supabase...")

    try:
        # Upsert to handle duplicates
        response = client.table("staff_scheduling").upsert(records).execute()
        print(f"SUCCESS: Uploaded {len(records)} records to staff_scheduling table")
        return True
    except Exception as e:
        print(f"ERROR: Failed to upload data - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup staff_scheduling table in Supabase"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file to upload"
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only print SQL schema, don't upload data"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STAFF SCHEDULING TABLE SETUP FOR SUPABASE")
    print("=" * 70)

    if args.schema_only or not args.csv:
        print("\nSQL Schema (copy and run in Supabase SQL Editor):\n")
        print_sql_schema()

        if not args.csv:
            print("\n" + "=" * 70)
            print("To upload data, run:")
            print("  python scripts/setup_staff_scheduling_table.py --csv path/to/data.csv")
            print("=" * 70)

    if args.csv:
        print(f"\nUploading data from: {args.csv}")
        upload_csv_to_supabase(args.csv)

    print("\nDone!")


if __name__ == "__main__":
    main()
