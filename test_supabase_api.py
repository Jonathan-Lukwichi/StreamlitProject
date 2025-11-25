"""
TEST SUPABASE API CONNECTIONS

This script tests that your Streamlit app can fetch data from Supabase.
Run this AFTER uploading data to Supabase to verify everything works.

Usage:
    python test_supabase_api.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add app_core to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app_core.api.config_manager import APIConfigManager
except ImportError:
    print("❌ Error: Could not import APIConfigManager")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_connection(dataset_type: str, provider: str):
    """Test connection for a specific dataset type"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_type.upper()} API ({provider})")
    print('='*60)

    config_manager = APIConfigManager()

    try:
        # Get connector
        if dataset_type == "patient":
            connector = config_manager.get_patient_connector(provider)
        elif dataset_type == "weather":
            connector = config_manager.get_weather_connector(provider)
        elif dataset_type == "calendar":
            connector = config_manager.get_calendar_connector(provider)
        elif dataset_type == "reason":
            connector = config_manager.get_reason_connector(provider)
        else:
            print(f"[ERROR] Unknown dataset type: {dataset_type}")
            return False

        print(f"[OK] Connector created: {connector.__class__.__name__}")

        # Test fetch for last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        print(f"[FETCH] Fetching data from {start_date.date()} to {end_date.date()}...")

        df = connector.fetch_data(start_date, end_date)

        print(f"[OK] SUCCESS! Fetched {len(df)} rows")
        print(f"[DATA] Columns: {list(df.columns)}")
        print(f"\n[SAMPLE] Sample data (first 3 rows):")
        print(df.head(3).to_string())

        return True

    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


def main():
    print("\n" + "="*60)
    print("[TEST] SUPABASE API CONNECTION TEST")
    print("="*60)

    config_manager = APIConfigManager()

    # Check configuration
    print("\n[CONFIG] Checking configuration...")
    print(f"   Patient provider: {config_manager.configs.get('patient', {}).get('provider', 'NOT SET')}")
    print(f"   Weather provider: {config_manager.configs.get('weather', {}).get('provider', 'NOT SET')}")
    print(f"   Calendar provider: {config_manager.configs.get('calendar', {}).get('provider', 'NOT SET')}")
    print(f"   Reason provider: {config_manager.configs.get('reason', {}).get('provider', 'NOT SET')}")

    # Test each dataset
    results = {}
    results["patient"] = test_connection("patient", "supabase")
    results["weather"] = test_connection("weather", "supabase")
    results["calendar"] = test_connection("calendar", "supabase")
    results["reason"] = test_connection("reason", "supabase")

    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] TEST SUMMARY")
    print("="*60)

    all_passed = True
    for dataset_type, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}  {dataset_type.capitalize()} API")
        if not success:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("[OK] Your Streamlit app can now fetch data from Supabase")
        print("[OK] Ready for production use!")
    else:
        print("\n[WARN] SOME TESTS FAILED")
        print("\n[INFO] Troubleshooting:")
        print("1. Verify Supabase URL and API key in .streamlit/secrets.toml")
        print("2. Ensure data is uploaded to Supabase (run setup_hospital_database.py)")
        print("3. Check Supabase RLS policies (run supabase_readonly_security.sql)")
        print("4. Verify tables exist: patient_arrivals, weather_data, calendar_data, clinical_visits")

    print()


if __name__ == "__main__":
    main()
