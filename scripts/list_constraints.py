# List all constraints in Supabase tables
from __future__ import annotations
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import toml

def main():
    from supabase import create_client

    secrets_path = project_root / ".streamlit" / "secrets.toml"
    secrets = toml.load(secrets_path)
    client = create_client(secrets["supabase"]["url"], secrets["supabase"]["key"])

    # Query to list all constraints
    sql = """
    SELECT
        tc.table_name,
        tc.constraint_name,
        tc.constraint_type,
        string_agg(kcu.column_name, ', ') as columns
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = 'public'
        AND tc.table_name IN ('patient_arrivals', 'weather_data', 'calendar_data', 'clinical_visits')
        AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
    GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
    ORDER BY tc.table_name, tc.constraint_type;
    """

    print("Querying constraints...")
    try:
        response = client.rpc('exec_sql', {'query': sql}).execute()
        print(response.data)
    except Exception as e:
        print(f"RPC failed: {e}")
        print("\nTrying direct query approach...")

        # Alternative: just list what we know
        tables = ["patient_arrivals", "weather_data", "calendar_data", "clinical_visits"]
        for table in tables:
            print(f"\n{table}:")
            try:
                # Try to insert a duplicate to see the constraint name
                response = client.table(table).select("*").limit(1).execute()
                if response.data:
                    print(f"  Sample row keys: {list(response.data[0].keys())}")
            except Exception as e2:
                print(f"  Error: {e2}")

if __name__ == "__main__":
    main()
