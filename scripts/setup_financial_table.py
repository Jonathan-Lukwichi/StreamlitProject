# =============================================================================
# scripts/setup_financial_table.py
# Setup script to upload Hospital Financial Data to Supabase
# =============================================================================
"""
Run this script to:
1. Create the financial_data table in Supabase (if not exists)
2. Upload the Financial_Data.csv data

Usage:
    python scripts/setup_financial_table.py

Prerequisites:
    - Configure .streamlit/secrets.toml with Supabase credentials
    - Run the SQL in create_financial_table.sql first in Supabase SQL Editor
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime


def load_secrets_toml():
    """Load credentials from .streamlit/secrets.toml"""
    secrets_path = project_root / ".streamlit" / "secrets.toml"

    if not secrets_path.exists():
        return None, None

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Manual parsing for simple TOML
            url = None
            key = None
            with open(secrets_path, "r", encoding="utf-8") as f:
                content = f.read()
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("url"):
                        url = line.split("=")[1].strip().strip('"').strip("'")
                    elif line.startswith("key"):
                        key = line.split("=")[1].strip().strip('"').strip("'")
            return url, key

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)

    supabase = secrets.get("supabase", {})
    return supabase.get("url"), supabase.get("key")


def get_supabase_client():
    """Get Supabase client with credentials from secrets.toml or environment."""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase package not installed.")
        print("Run: pip install supabase")
        sys.exit(1)

    # Try secrets.toml first
    url, key = load_secrets_toml()

    # Fallback to environment variables
    if not url:
        url = os.getenv("SUPABASE_URL")
    if not key:
        key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("ERROR: Missing Supabase credentials.")
        print()
        print("Option 1: Configure .streamlit/secrets.toml:")
        print('  [supabase]')
        print('  url = "https://your-project.supabase.co"')
        print('  key = "your-anon-key"')
        print()
        print("Option 2: Set environment variables:")
        print("  SUPABASE_URL and SUPABASE_KEY")
        sys.exit(1)

    print(f"Using Supabase URL: {url[:40]}...")
    return create_client(url, key)


def load_financial_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate financial CSV."""
    print(f"\n[1/4] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"      Columns: {list(df.columns)[:10]}... (showing first 10)")

    # Validate required columns
    required_cols = [
        "Date",
        "Doctor_Hourly_Rate",
        "Nurse_Hourly_Rate",
        "Total_Labor_Cost",
        "Total_Inventory_Cost",
        "Total_Revenue"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    print("      All required columns present.")
    return df


def prepare_records(df: pd.DataFrame) -> list:
    """Prepare records for Supabase insertion."""
    print("\n[2/4] Preparing records for upload...")

    records = []
    for _, row in df.iterrows():
        # Parse date
        date_val = row["Date"]
        if isinstance(date_val, str):
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    date_val = datetime.strptime(date_val, fmt).date().isoformat()
                    break
                except ValueError:
                    continue
        elif hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()

        record = {"Date": str(date_val)}

        # Add all numeric columns
        for col in df.columns:
            if col != "Date":
                val = row[col]
                if pd.notna(val):
                    record[col] = float(val)
                else:
                    record[col] = 0.0

        records.append(record)

    print(f"      Prepared {len(records)} records.")
    return records


def upload_to_supabase(client, records: list, table_name: str = "financial_data"):
    """Upload records to Supabase in batches."""
    print(f"\n[3/4] Uploading to Supabase table: {table_name}")

    batch_size = 500
    total = len(records)
    uploaded = 0

    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            client.table(table_name).insert(batch).execute()
            uploaded += len(batch)
            print(f"      Uploaded {uploaded}/{total} records...")
        except Exception as e:
            print(f"ERROR uploading batch {i//batch_size + 1}: {e}")
            raise

    print(f"      Successfully uploaded {uploaded} records!")


def verify_upload(client, table_name: str = "financial_data"):
    """Verify the upload by counting records."""
    print(f"\n[4/4] Verifying upload...")

    try:
        # Get count
        response = client.table(table_name).select("*", count="exact").limit(1).execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"      Table {table_name} now has {count} records.")

        # Get sample
        sample = client.table(table_name).select("*").order("Date").limit(3).execute()
        if sample.data:
            print("      Sample records:")
            for rec in sample.data[:3]:
                print(f"        - Date: {rec.get('Date')}, "
                      f"Labor Cost: ${rec.get('Total_Labor_Cost', 0):,.2f}, "
                      f"Revenue: ${rec.get('Total_Revenue', 0):,.2f}")

        return True
    except Exception as e:
        print(f"ERROR verifying upload: {e}")
        return False


def main():
    """Main execution."""
    print("=" * 70)
    print(" FINANCIAL DATA - SUPABASE SETUP SCRIPT")
    print("=" * 70)

    # Default CSV path
    csv_path = r"C:\officie_master\dataset test\Espane dataset hospital\pamplona data 2\Financial data\Financial_Data.csv"

    # Allow override via command line
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    # Get Supabase client
    client = get_supabase_client()
    print("      Supabase client connected.")

    # Load CSV
    df = load_financial_csv(csv_path)

    # Prepare records
    records = prepare_records(df)

    # Upload
    upload_to_supabase(client, records)

    # Verify
    success = verify_upload(client)

    print("\n" + "=" * 70)
    if success:
        print(" SETUP COMPLETE! Financial data is now in Supabase.")
    else:
        print(" SETUP COMPLETED WITH WARNINGS. Please verify manually.")
    print("=" * 70)


if __name__ == "__main__":
    main()
