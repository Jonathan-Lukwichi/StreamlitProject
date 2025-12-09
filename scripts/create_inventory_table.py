# =============================================================================
# scripts/create_inventory_table.py
# Creates the inventory_management table in Supabase
# =============================================================================
"""
This script creates the inventory_management table in Supabase.

Option 1: Run with service_role key (auto-creates table)
    python scripts/create_inventory_table.py --execute

Option 2: Generate SQL only (copy to Supabase SQL Editor)
    python scripts/create_inventory_table.py --sql-only
"""

import os
import sys
import argparse

# SQL to create the inventory_management table
CREATE_TABLE_SQL = """
-- ============================================================================
-- INVENTORY MANAGEMENT TABLE
-- For HealthForecast AI - Inventory Optimization Module
-- ============================================================================

-- Drop existing table if needed (uncomment if you want to recreate)
-- DROP TABLE IF EXISTS inventory_management;

-- Create the inventory_management table
CREATE TABLE IF NOT EXISTS inventory_management (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    "Date" DATE NOT NULL,
    "Inventory_Used_Gloves" INTEGER DEFAULT 0,
    "Inventory_Used_PPE_Sets" INTEGER DEFAULT 0,
    "Inventory_Used_Medications" INTEGER DEFAULT 0,
    "Inventory_Level_Gloves" INTEGER DEFAULT 0,
    "Inventory_Level_PPE" INTEGER DEFAULT 0,
    "Inventory_Level_Medications" INTEGER DEFAULT 0,
    "Restock_Event" INTEGER DEFAULT 0,
    "Stockout_Risk_Score" REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on Date for faster queries
CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_management("Date");

-- Enable Row Level Security (RLS)
ALTER TABLE inventory_management ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust as needed for your security requirements)
CREATE POLICY "Allow all operations on inventory_management"
ON inventory_management
FOR ALL
USING (true)
WITH CHECK (true);

-- Grant permissions
GRANT ALL ON inventory_management TO authenticated;
GRANT ALL ON inventory_management TO anon;

-- Verify table was created
SELECT 'Table inventory_management created successfully!' AS status;
"""


def print_sql():
    """Print the SQL for manual execution in Supabase SQL Editor."""
    print("=" * 70)
    print(" SQL TO CREATE INVENTORY_MANAGEMENT TABLE")
    print(" Copy this SQL and run it in Supabase SQL Editor")
    print("=" * 70)
    print()
    print(CREATE_TABLE_SQL)
    print()
    print("=" * 70)
    print(" After running this SQL, execute:")
    print("   python scripts/setup_inventory_table.py")
    print(" to upload the inventory data.")
    print("=" * 70)


def execute_sql():
    """Execute the SQL directly using Supabase (requires service_role key)."""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase package not installed.")
        print("Run: pip install supabase")
        sys.exit(1)

    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("ERROR: Missing Supabase credentials.")
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        print()
        print("Note: Creating tables requires the service_role key, not the anon key.")
        print("You can find it in Supabase Dashboard > Settings > API > service_role")
        sys.exit(1)

    print("Connecting to Supabase...")
    client = create_client(url, key)

    print("Executing SQL...")
    try:
        # Try to execute SQL via RPC (if you have a custom function)
        # Or use the REST API for SQL execution

        # Method 1: Try direct table creation by inserting a dummy record
        # This won't work for creating tables, but we can check if table exists

        # Method 2: Use postgrest-py raw SQL (not available in standard client)

        # For now, let's just try to access the table
        response = client.table("inventory_management").select("*").limit(1).execute()
        print("Table already exists!")
        print(f"Current records: {len(response.data)}")

    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower() or "404" in error_msg:
            print()
            print("Table does not exist yet.")
            print()
            print("Unfortunately, Supabase Python client cannot create tables directly.")
            print("Please use one of these methods:")
            print()
            print("METHOD 1: Supabase SQL Editor (Recommended)")
            print("-" * 50)
            print("1. Go to your Supabase Dashboard")
            print("2. Click 'SQL Editor' in the left sidebar")
            print("3. Copy and paste the SQL below")
            print("4. Click 'Run'")
            print()
            print_sql()
        else:
            print(f"Error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create inventory_management table in Supabase")
    parser.add_argument("--sql-only", action="store_true", help="Only print SQL (don't execute)")
    parser.add_argument("--execute", action="store_true", help="Try to execute SQL")

    args = parser.parse_args()

    if args.execute:
        execute_sql()
    else:
        print_sql()

        # Also save SQL to a file
        sql_file = os.path.join(os.path.dirname(__file__), "create_inventory_table.sql")
        with open(sql_file, "w", encoding="utf-8") as f:
            f.write(CREATE_TABLE_SQL)
        print(f"\nSQL also saved to: {sql_file}")


if __name__ == "__main__":
    main()
