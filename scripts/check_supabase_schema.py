# Check Supabase table schemas
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

    tables = ["patient_arrivals", "weather_data", "calendar_data", "clinical_visits"]

    for table in tables:
        print(f"\n{'='*60}")
        print(f"Table: {table}")
        print(f"{'='*60}")
        try:
            # Fetch one row to see column structure
            response = client.table(table).select("*").limit(1).execute()
            if response.data:
                row = response.data[0]
                print("Columns:")
                for key, value in row.items():
                    print(f"  - {key}: {type(value).__name__} = {repr(value)[:50]}")
            else:
                print("  (no data found)")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
