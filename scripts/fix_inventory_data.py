# =============================================================================
# scripts/fix_inventory_data.py
# Clears and re-uploads correct inventory data to Supabase
# =============================================================================
"""
This script:
1. Deletes ALL existing records from inventory_management table
2. Re-uploads the correct data from Inventory_Management___Pamplona.csv

Usage:
    python scripts/fix_inventory_data.py
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
    """Get Supabase client with credentials from secrets.toml"""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase package not installed.")
        print("Run: pip install supabase")
        sys.exit(1)

    url, key = load_secrets_toml()

    if not url or not key:
        print("ERROR: Missing Supabase credentials in .streamlit/secrets.toml")
        sys.exit(1)

    print(f"Connecting to Supabase: {url[:40]}...")
    return create_client(url, key)


def clear_table(client, table_name: str):
    """Delete ALL records from the table."""
    print(f"\n[1/4] Clearing existing data from '{table_name}'...")

    try:
        # Delete all records (using a filter that matches everything)
        # We use gte on Date to get all records, then delete
        response = client.table(table_name).delete().gte("Date", "1900-01-01").execute()
        print(f"      Deleted existing records.")
        return True
    except Exception as e:
        print(f"      Error clearing table: {e}")
        # Try alternative: delete by fetching all IDs first
        try:
            print("      Trying alternative method...")
            # Fetch all IDs
            all_ids = []
            offset = 0
            batch_size = 1000

            while True:
                response = client.table(table_name).select("id").range(offset, offset + batch_size - 1).execute()
                if not response.data:
                    break
                all_ids.extend([r["id"] for r in response.data])
                if len(response.data) < batch_size:
                    break
                offset += batch_size

            print(f"      Found {len(all_ids)} records to delete...")

            # Delete in batches by ID
            for i in range(0, len(all_ids), 100):
                batch_ids = all_ids[i:i+100]
                for record_id in batch_ids:
                    client.table(table_name).delete().eq("id", record_id).execute()
                print(f"      Deleted {min(i+100, len(all_ids))}/{len(all_ids)} records...")

            return True
        except Exception as e2:
            print(f"      Alternative method also failed: {e2}")
            return False


def load_and_prepare_csv(csv_path: str):
    """Load CSV and prepare records for upload."""
    print(f"\n[2/4] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} records")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Prepare records
    print("\n[3/4] Preparing records for upload...")
    records = []
    for _, row in df.iterrows():
        # Parse date
        date_val = row["Date"]
        if isinstance(date_val, str):
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

    print(f"      Prepared {len(records)} records")
    print(f"      First record: {records[0]}")
    return records


def upload_records(client, records: list, table_name: str):
    """Upload records to Supabase in batches."""
    print(f"\n[4/4] Uploading to Supabase...")

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
            print(f"ERROR uploading batch: {e}")
            raise

    print(f"      Successfully uploaded {uploaded} records!")
    return uploaded


def verify_upload(client, table_name: str):
    """Verify the upload by checking a few records."""
    print("\n[VERIFY] Checking uploaded data...")

    try:
        # Get first 5 records ordered by date
        response = client.table(table_name).select("*").order("Date").limit(5).execute()

        if response.data:
            print("      First 5 records (ordered by Date):")
            for rec in response.data:
                print(f"        - Date: {rec.get('Date')}, "
                      f"Gloves Used: {rec.get('Inventory_Used_Gloves')}, "
                      f"PPE Used: {rec.get('Inventory_Used_PPE_Sets')}")

        # Get count
        count_response = client.table(table_name).select("*", count="exact").limit(1).execute()
        count = count_response.count if hasattr(count_response, 'count') else "Unknown"
        print(f"\n      Total records in table: {count}")

        return True
    except Exception as e:
        print(f"      Error verifying: {e}")
        return False


def main():
    print("=" * 70)
    print(" FIX INVENTORY DATA - Clear and Re-upload")
    print("=" * 70)

    table_name = "inventory_management"
    csv_path = r"C:\officie_master\dataset test\Espane dataset hospital\pamplona data 2\Inventory management_data\Inventory_Management___Pamplona.csv"

    # Connect to Supabase
    client = get_supabase_client()
    print("      Connected!")

    # Step 1: Clear existing data
    clear_table(client, table_name)

    # Step 2 & 3: Load and prepare CSV
    records = load_and_prepare_csv(csv_path)

    # Step 4: Upload new data
    upload_records(client, records, table_name)

    # Verify
    verify_upload(client, table_name)

    print("\n" + "=" * 70)
    print(" DONE! Inventory data has been fixed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
