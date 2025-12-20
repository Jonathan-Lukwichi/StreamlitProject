# =============================================================================
# scripts/diagnose_staff_data.py
# Diagnostic Script - Analyze Staff Data vs Forecast to Understand Cost Changes
# Run: python scripts/diagnose_staff_data.py
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

import toml
import pandas as pd
import numpy as np
from datetime import datetime


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def format_currency(value: float) -> str:
    """Format number as currency."""
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    """Format number as percentage."""
    return f"{value:.1f}%"


def main():
    print_header("STAFF DATA DIAGNOSTIC REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # CONNECT TO SUPABASE
    # =========================================================================
    print_subheader("1. Connecting to Supabase")

    try:
        from supabase import create_client
        secrets_path = project_root / ".streamlit" / "secrets.toml"
        secrets = toml.load(secrets_path)
        client = create_client(secrets["supabase"]["url"], secrets["supabase"]["key"])
        print("[OK] Connected to Supabase")
    except Exception as e:
        print(f"[ERROR] Could not connect to Supabase: {e}")
        return

    # =========================================================================
    # FETCH STAFF SCHEDULING DATA
    # =========================================================================
    print_subheader("2. Fetching Staff Scheduling Data")

    try:
        # Fetch all staff data with pagination
        all_data = []
        offset = 0
        batch_size = 1000

        while True:
            response = (
                client.table("staff_scheduling")
                .select("*")
                .order("date")
                .range(offset, offset + batch_size - 1)
                .execute()
            )

            if response.data:
                all_data.extend(response.data)
                if len(response.data) < batch_size:
                    break
                offset += batch_size
            else:
                break

        if not all_data:
            print("[ERROR] No staff scheduling data found!")
            return

        staff_df = pd.DataFrame(all_data)
        print(f"[OK] Fetched {len(staff_df)} records")

    except Exception as e:
        print(f"[ERROR] Could not fetch staff data: {e}")
        return

    # =========================================================================
    # ANALYZE STAFF DATA
    # =========================================================================
    print_subheader("3. Staff Data Analysis")

    # Show column names
    print(f"\nColumns in staff_scheduling table:")
    for col in staff_df.columns:
        print(f"  - {col}")

    # Show sample data
    print(f"\nSample data (first 3 rows):")
    print(staff_df.head(3).to_string())

    # =========================================================================
    # UTILIZATION ANALYSIS
    # =========================================================================
    print_subheader("4. Staff Utilization Analysis")

    util_col = None
    for col in ["staff_utilization_rate", "Staff_Utilization_Rate"]:
        if col in staff_df.columns:
            util_col = col
            break

    if util_col:
        utilization_values = pd.to_numeric(staff_df[util_col], errors='coerce')

        print(f"\nUtilization column: '{util_col}'")
        print(f"  Min value:    {utilization_values.min():.4f}")
        print(f"  Max value:    {utilization_values.max():.4f}")
        print(f"  Mean value:   {utilization_values.mean():.4f}")
        print(f"  Median value: {utilization_values.median():.4f}")

        # Check if values are decimals (0-1) or percentages (0-100)
        if utilization_values.max() <= 1.0:
            print(f"\n  [WARNING] Values appear to be DECIMALS (0-1), not percentages!")
            print(f"  [INFO] The app expects percentages (0-100)")
            print(f"  [INFO] Mean as percentage: {utilization_values.mean() * 100:.1f}%")
        elif utilization_values.max() <= 100:
            print(f"\n  [OK] Values appear to be percentages (0-100)")
        else:
            print(f"\n  [WARNING] Values exceed 100 - unexpected range!")

        # Show distribution
        print(f"\n  Value distribution:")
        if utilization_values.max() <= 1.0:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        else:
            bins = [0, 20, 40, 60, 80, 100]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

        counts = pd.cut(utilization_values, bins=bins, labels=labels).value_counts().sort_index()
        for label, count in counts.items():
            pct = count / len(utilization_values) * 100
            bar = "#" * int(pct / 2)
            print(f"    {label}: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print("[WARNING] No utilization column found!")

    # =========================================================================
    # STAFF COUNTS ANALYSIS
    # =========================================================================
    print_subheader("5. Staff Counts Analysis")

    staff_cols = {
        "doctors": ["doctors_on_duty", "Doctors_on_Duty"],
        "nurses": ["nurses_on_duty", "Nurses_on_Duty"],
        "support": ["support_staff_on_duty", "Support_Staff_on_Duty"]
    }

    staff_stats = {}

    for staff_type, possible_cols in staff_cols.items():
        for col in possible_cols:
            if col in staff_df.columns:
                values = pd.to_numeric(staff_df[col], errors='coerce')
                staff_stats[staff_type] = {
                    "col": col,
                    "min": values.min(),
                    "max": values.max(),
                    "mean": values.mean(),
                    "median": values.median()
                }
                print(f"\n{staff_type.upper()} (column: '{col}'):")
                print(f"  Min:    {values.min():.0f}")
                print(f"  Max:    {values.max():.0f}")
                print(f"  Mean:   {values.mean():.1f}")
                print(f"  Median: {values.median():.0f}")
                break

    # =========================================================================
    # OVERTIME ANALYSIS
    # =========================================================================
    print_subheader("6. Overtime Analysis")

    ot_col = None
    for col in ["overtime_hours", "Overtime_Hours"]:
        if col in staff_df.columns:
            ot_col = col
            break

    if ot_col:
        ot_values = pd.to_numeric(staff_df[ot_col], errors='coerce')
        print(f"\nOvertime column: '{ot_col}'")
        print(f"  Total overtime hours: {ot_values.sum():.0f}")
        print(f"  Daily average:        {ot_values.mean():.1f} hours")
        print(f"  Max single day:       {ot_values.max():.0f} hours")
        print(f"  Days with overtime:   {(ot_values > 0).sum()} ({(ot_values > 0).mean() * 100:.1f}%)")

    # =========================================================================
    # SHORTAGE ANALYSIS
    # =========================================================================
    print_subheader("7. Shortage Analysis")

    shortage_col = None
    for col in ["staff_shortage_flag", "Staff_Shortage_Flag"]:
        if col in staff_df.columns:
            shortage_col = col
            break

    if shortage_col:
        shortage_values = pd.to_numeric(staff_df[shortage_col], errors='coerce').fillna(0)
        shortage_days = int(shortage_values.sum())
        shortage_pct = shortage_values.mean() * 100
        print(f"\nShortage column: '{shortage_col}'")
        print(f"  Total shortage days: {shortage_days}")
        print(f"  Shortage percentage: {shortage_pct:.1f}%")

    # =========================================================================
    # FETCH FINANCIAL DATA (HOURLY RATES)
    # =========================================================================
    print_subheader("8. Financial Data - Hourly Rates")

    try:
        response = (
            client.table("financial_data")
            .select("Doctor_Hourly_Rate, Nurse_Hourly_Rate, Support_Staff_Hourly_Rate, Overtime_Premium_Rate")
            .limit(10)
            .execute()
        )

        if response.data:
            import statistics
            doc_rates = [r.get("Doctor_Hourly_Rate", 0) for r in response.data if r.get("Doctor_Hourly_Rate")]
            nurse_rates = [r.get("Nurse_Hourly_Rate", 0) for r in response.data if r.get("Nurse_Hourly_Rate")]
            support_rates = [r.get("Support_Staff_Hourly_Rate", 0) for r in response.data if r.get("Support_Staff_Hourly_Rate")]

            print(f"\nFinancial data found: {len(response.data)} records sampled")
            if doc_rates:
                print(f"  Doctor Hourly Rate:  ${statistics.mean(doc_rates):.2f} (avg)")
            if nurse_rates:
                print(f"  Nurse Hourly Rate:   ${statistics.mean(nurse_rates):.2f} (avg)")
            if support_rates:
                print(f"  Support Hourly Rate: ${statistics.mean(support_rates):.2f} (avg)")

            print(f"\n  [INFO] These rates should be used in Staff Planner for accurate cost calculations")
        else:
            print("\n[WARNING] No financial data found in Supabase!")
            print("  The app will use default rates: Doctor=$150, Nurse=$50, Support=$25")

    except Exception as e:
        print(f"\n[ERROR] Could not fetch financial data: {e}")

    # =========================================================================
    # FETCH PATIENT DATA FOR COMPARISON
    # =========================================================================
    print_subheader("9. Patient Data Comparison")

    try:
        # Fetch patient arrivals
        patient_data = []
        offset = 0

        while True:
            response = (
                client.table("patient_arrivals")
                .select("datetime, patient_count")
                .order("datetime")
                .range(offset, offset + batch_size - 1)
                .execute()
            )

            if response.data:
                patient_data.extend(response.data)
                if len(response.data) < batch_size:
                    break
                offset += batch_size
            else:
                break

        if patient_data:
            patient_df = pd.DataFrame(patient_data)
            patient_df["datetime"] = pd.to_datetime(patient_df["datetime"])
            patient_df["date"] = patient_df["datetime"].dt.date

            # Daily patient counts
            daily_patients = patient_df.groupby("date")["patient_count"].sum()

            print(f"\nPatient arrivals data:")
            print(f"  Total records:        {len(patient_df)}")
            print(f"  Date range:           {daily_patients.index.min()} to {daily_patients.index.max()}")
            print(f"  Avg patients/day:     {daily_patients.mean():.0f}")
            print(f"  Min patients/day:     {daily_patients.min():.0f}")
            print(f"  Max patients/day:     {daily_patients.max():.0f}")

            # Compare with staff
            if staff_stats:
                avg_doctors = staff_stats.get("doctors", {}).get("mean", 0)
                avg_nurses = staff_stats.get("nurses", {}).get("mean", 0)
                avg_support = staff_stats.get("support", {}).get("mean", 0)
                total_staff = avg_doctors + avg_nurses + avg_support

                if total_staff > 0:
                    patients_per_staff = daily_patients.mean() / total_staff
                    print(f"\n  Historical patients per staff: {patients_per_staff:.1f}")
        else:
            print("\n[WARNING] No patient data found")
            daily_patients = None

    except Exception as e:
        print(f"\n[ERROR] Could not fetch patient data: {e}")
        daily_patients = None

    # =========================================================================
    # COST CALCULATION
    # =========================================================================
    print_subheader("10. Cost Calculation (Using App's Parameters)")

    # Default cost parameters (same as app)
    cost_params = {
        "doctor_hourly": 150,
        "nurse_hourly": 50,
        "support_hourly": 25,
        "shift_hours": 8,
    }

    if staff_stats:
        avg_doctors = staff_stats.get("doctors", {}).get("mean", 0)
        avg_nurses = staff_stats.get("nurses", {}).get("mean", 0)
        avg_support = staff_stats.get("support", {}).get("mean", 0)

        daily_cost = (
            avg_doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
            avg_nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
            avg_support * cost_params["support_hourly"] * cost_params["shift_hours"]
        )
        weekly_cost = daily_cost * 7
        annual_cost = daily_cost * 365

        print(f"\nCost Parameters:")
        print(f"  Doctor hourly rate:  ${cost_params['doctor_hourly']}")
        print(f"  Nurse hourly rate:   ${cost_params['nurse_hourly']}")
        print(f"  Support hourly rate: ${cost_params['support_hourly']}")
        print(f"  Shift hours:         {cost_params['shift_hours']}")

        print(f"\nCalculated Costs (based on historical staff averages):")
        print(f"  Avg staff/day:  {avg_doctors:.0f} doctors + {avg_nurses:.0f} nurses + {avg_support:.0f} support")
        print(f"  Daily cost:     {format_currency(daily_cost)}")
        print(f"  Weekly cost:    {format_currency(weekly_cost)}")
        print(f"  Annual cost:    {format_currency(annual_cost)}")

    # =========================================================================
    # STAFFING RATIO ANALYSIS
    # =========================================================================
    print_subheader("11. Staffing Ratio Analysis (What Optimizer Uses)")

    # App's staffing ratios
    STAFFING_RATIOS = {
        "CARDIAC": {"doctors": 5, "nurses": 3, "support": 8},
        "TRAUMA": {"doctors": 6, "nurses": 2, "support": 5},
        "RESPIRATORY": {"doctors": 8, "nurses": 4, "support": 10},
        "GASTROINTESTINAL": {"doctors": 10, "nurses": 5, "support": 12},
        "INFECTIOUS": {"doctors": 12, "nurses": 6, "support": 15},
        "NEUROLOGICAL": {"doctors": 8, "nurses": 4, "support": 10},
        "OTHER": {"doctors": 15, "nurses": 8, "support": 20},
    }

    distribution = {
        "CARDIAC": 0.15,
        "TRAUMA": 0.20,
        "RESPIRATORY": 0.25,
        "GASTROINTESTINAL": 0.15,
        "INFECTIOUS": 0.15,
        "NEUROLOGICAL": 0.05,
        "OTHER": 0.05,
    }

    print("\nStaffing Ratios (patients per staff member):")
    print(f"{'Category':<20} {'Doctors':<10} {'Nurses':<10} {'Support':<10} {'% of Patients':<15}")
    print("-" * 65)
    for cat, ratios in STAFFING_RATIOS.items():
        pct = distribution.get(cat, 0) * 100
        print(f"{cat:<20} {ratios['doctors']:<10} {ratios['nurses']:<10} {ratios['support']:<10} {pct:.0f}%")

    # Calculate what staff would be needed for average patient load
    if daily_patients is not None:
        avg_patients = daily_patients.mean()

        print(f"\nStaff needed for avg {avg_patients:.0f} patients/day:")

        total_doctors = 0
        total_nurses = 0
        total_support = 0

        for cat, pct in distribution.items():
            cat_patients = avg_patients * pct
            ratios = STAFFING_RATIOS[cat]
            doctors = np.ceil(cat_patients / ratios["doctors"])
            nurses = np.ceil(cat_patients / ratios["nurses"])
            support = np.ceil(cat_patients / ratios["support"])
            total_doctors += doctors
            total_nurses += nurses
            total_support += support

        print(f"  Doctors needed:  {total_doctors:.0f}")
        print(f"  Nurses needed:   {total_nurses:.0f}")
        print(f"  Support needed:  {total_support:.0f}")
        print(f"  Total staff:     {total_doctors + total_nurses + total_support:.0f}")

        # Compare with historical
        if staff_stats:
            hist_doctors = staff_stats.get("doctors", {}).get("mean", 0)
            hist_nurses = staff_stats.get("nurses", {}).get("mean", 0)
            hist_support = staff_stats.get("support", {}).get("mean", 0)

            print(f"\nHistorical vs Needed:")
            print(f"  {'Role':<10} {'Historical':<12} {'Needed':<12} {'Difference':<15}")
            print("-" * 50)
            print(f"  {'Doctors':<10} {hist_doctors:<12.0f} {total_doctors:<12.0f} {total_doctors - hist_doctors:+.0f}")
            print(f"  {'Nurses':<10} {hist_nurses:<12.0f} {total_nurses:<12.0f} {total_nurses - hist_nurses:+.0f}")
            print(f"  {'Support':<10} {hist_support:<12.0f} {total_support:<12.0f} {total_support - hist_support:+.0f}")

            # Calculate cost difference
            needed_daily_cost = (
                total_doctors * cost_params["doctor_hourly"] * cost_params["shift_hours"] +
                total_nurses * cost_params["nurse_hourly"] * cost_params["shift_hours"] +
                total_support * cost_params["support_hourly"] * cost_params["shift_hours"]
            )

            cost_diff = needed_daily_cost - daily_cost

            print(f"\nCost Impact:")
            print(f"  Historical daily cost:  {format_currency(daily_cost)}")
            print(f"  Needed daily cost:      {format_currency(needed_daily_cost)}")
            print(f"  Difference:             {format_currency(cost_diff)} ({'+' if cost_diff > 0 else ''}{cost_diff/daily_cost*100:.1f}%)")

            if cost_diff > 0:
                print(f"\n  [EXPLANATION] Costs are INCREASING because:")
                print(f"    - Historical staff: {hist_doctors + hist_nurses + hist_support:.0f} people/day")
                print(f"    - Staff needed for patient demand: {total_doctors + total_nurses + total_support:.0f} people/day")
                print(f"    - The optimizer recommends MORE staff to handle patient volume")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print_subheader("12. Recommendations")

    issues_found = []

    # Check utilization
    if util_col:
        util_mean = pd.to_numeric(staff_df[util_col], errors='coerce').mean()
        if util_mean <= 1.0:
            issues_found.append({
                "issue": "Utilization stored as decimal instead of percentage",
                "fix": "Multiply staff_utilization_rate by 100 in the database, OR update the app to handle decimal format"
            })
        elif util_mean < 50:
            issues_found.append({
                "issue": f"Very low utilization ({util_mean:.1f}%) suggests data quality issues",
                "fix": "Verify that utilization values are calculated correctly in your source data"
            })

    # Check if optimization increases costs
    if daily_patients is not None and staff_stats:
        if cost_diff > 0:
            issues_found.append({
                "issue": "Optimization increases costs (historical staff < needed staff)",
                "fix": "This may be correct if historical staffing was inadequate. Review if staff data represents actual capacity."
            })

    if issues_found:
        print(f"\nFound {len(issues_found)} potential issue(s):\n")
        for i, item in enumerate(issues_found, 1):
            print(f"  Issue {i}: {item['issue']}")
            print(f"  Fix:     {item['fix']}")
            print()
    else:
        print("\nNo obvious issues found. Data appears consistent.")

    print_header("END OF DIAGNOSTIC REPORT")


if __name__ == "__main__":
    main()
