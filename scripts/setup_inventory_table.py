# =============================================================================
# scripts/setup_inventory_table.py
# Setup script to upload Inventory Management data to Supabase
# =============================================================================
"""
Run this script to:
1. Create the inventory_management table in Supabase (if not exists)
2. Upload the Inventory_Management___Pamplona.csv data

Usage:
    python scripts/setup_inventory_table.py

Prerequisites:
    - Configure .streamlit/secrets.toml with Supabase credentials
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

# Try to load environment variables as fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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


def load_inventory_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate inventory CSV."""
    print(f"\n[1/4] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} records with columns: {list(df.columns)}")

    # Validate required columns
    required_cols = [
        "Date",
        "Inventory_Used_Gloves",
        "Inventory_Used_PPE_Sets",
        "Inventory_Used_Medications",
        "Inventory_Level_Gloves",
        "Inventory_Level_PPE",
        "Inventory_Level_Medications",
        "Restock_Event",
        "Stockout_Risk_Score"
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

        record = {
            "Date": str(date_val),
            "Inventory_Used_Gloves": int(row.get("Inventory_Used_Gloves", 0)),
            "Inventory_Used_PPE_Sets": int(row.get("Inventory_Used_PPE_Sets", 0)),
            "Inventory_Used_Medications": int(row.get("Inventory_Used_Medications", 0)),
            "Inventory_Level_Gloves": int(row.get("Inventory_Level_Gloves", 0)),
            "Inventory_Level_PPE": int(row.get("Inventory_Level_PPE", 0)),
            "Inventory_Level_Medications": int(row.get("Inventory_Level_Medications", 0)),
            "Restock_Event": int(row.get("Restock_Event", 0)),
            "Stockout_Risk_Score": float(row.get("Stockout_Risk_Score", 0)),
        }
        records.append(record)

    print(f"      Prepared {len(records)} records.")
    return records


def upload_to_supabase(client, records: list, table_name: str = "inventory_management"):
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


def verify_upload(client, table_name: str = "inventory_management"):
    """Verify the upload by counting records."""
    print(f"\n[4/4] Verifying upload...")

    try:
        # Get count
        response = client.table(table_name).select("*", count="exact").limit(1).execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"      Table {table_name} now has {count} records.")

        # Get sample
        sample = client.table(table_name).select("*").limit(3).execute()
        if sample.data:
            print("      Sample records:")
            for rec in sample.data[:3]:
                print(f"        - Date: {rec.get('Date')}, "
                      f"Gloves Used: {rec.get('Inventory_Used_Gloves')}, "
                      f"Risk Score: {rec.get('Stockout_Risk_Score')}")

        return True
    except Exception as e:
        print(f"ERROR verifying upload: {e}")
        return False


def main():
    """Main execution."""
    print("=" * 70)
    print(" INVENTORY MANAGEMENT - SUPABASE SETUP SCRIPT")
    print("=" * 70)

    # Default CSV path
    csv_path = r"C:\officie_master\dataset test\Espane dataset hospital\pamplona data 2\Inventory management_data\Inventory_Management___Pamplona.csv"

    # Allow override via command line
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    # Get Supabase client
    client = get_supabase_client()
    print("      Supabase client connected.")

    # Load CSV
    df = load_inventory_csv(csv_path)

    # Prepare records
    records = prepare_records(df)

    # Upload
    upload_to_supabase(client, records)

    # Verify
    success = verify_upload(client)

    print("\n" + "=" * 70)
    if success:
        print(" SETUP COMPLETE! Inventory data is now in Supabase.")
    else:
        print(" SETUP COMPLETED WITH WARNINGS. Please verify manually.")
    print("=" * 70)


if __name__ == "__main__":
    main()
