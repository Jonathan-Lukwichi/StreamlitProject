import pandas as pd
import streamlit as st
from app_core.ui.theme import apply_css
from app_core.state.session import init_state
from app_core.ui.components import header
from app_core.data.ingest import _find_datetime_col, _to_datetime_series, _combine_date_time_columns


def _normalize_to_date(dt_series: pd.Series) -> pd.Series:
    """
    Convert datetime to date-only (midnight) for daily aggregation.
    Assumes input is already timezone-naive from Data Hub cleaning.
    """
    return pd.to_datetime(dt_series).dt.normalize()


def _prep_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare patient data with standardized 'datetime' column.
    Assumes 'datetime' column is already clean (timezone-naive) from Data Hub.
    """
    df = df.copy()

    # Data Hub should provide 'datetime' column - fallback to search if needed
    if "datetime" not in df.columns:
        dt_col = _find_datetime_col(df, want="datetime")
        if dt_col is None:
            raise ValueError("Patient CSV must include a 'datetime' column.")
        df["datetime"] = _to_datetime_series(df[dt_col])

    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in patient datetime column.")

    # Create Date for merging (normalize to midnight for daily aggregation)
    df["Date"] = _normalize_to_date(df["datetime"])

    # Find patient count column
    if "patient_count" not in df.columns:
        for alt in ["Target_1", "patients", "count", "y", "value"]:
            if alt in df.columns:
                df["patient_count"] = pd.to_numeric(df[alt], errors="coerce")
                break

    if "patient_count" not in df.columns:
        raise ValueError("Patient CSV must include 'patient_count' (or a recognizable equivalent).")

    df["patient_count"] = pd.to_numeric(df["patient_count"], errors="coerce")

    # Return only Date and patient_count (drop datetime to avoid duplication)
    return df[["Date", "patient_count"]]

def _prep_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare weather data with standardized 'datetime' column.
    Assumes 'datetime' column is already clean (timezone-naive) from Data Hub.
    """
    df = df.copy()

    # Data Hub should provide 'datetime' column - fallback to search if needed
    if "datetime" not in df.columns:
        dt_col = _find_datetime_col(df, want="datetime")
        if dt_col is None:
            raise ValueError("Weather CSV must include a 'datetime' column.")
        df["datetime"] = _to_datetime_series(df[dt_col])

    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in weather datetime column.")

    # Create Date for merging (normalize to midnight for daily aggregation)
    df["Date"] = _normalize_to_date(df["datetime"])

    # Initialize output with Date only (drop datetime to avoid duplication)
    out = df[["Date"]].copy()

    # Extract weather columns
    weather_cols = ["Average_Temp", "Max_temp", "Average_wind", "Max_wind", "Average_mslp", "Total_precipitation", "Moon_Phase"]
    for col in weather_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    return out

def _prep_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare calendar data with standardized 'datetime' column.
    Assumes 'datetime' column is already clean (timezone-naive) from Data Hub.
    """
    df = df.copy()

    # Data Hub should provide 'datetime' column - fallback to search if needed
    if "datetime" not in df.columns:
        dt_col = _find_datetime_col(df, want="date")
        if dt_col is None:
            raise ValueError("Calendar CSV must include a 'datetime' column.")
        df["datetime"] = _to_datetime_series(df[dt_col])

    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in calendar datetime column.")

    # Create Date for merging (normalize to midnight for daily aggregation)
    out = pd.DataFrame({"Date": _normalize_to_date(df["datetime"])})

    # Extract calendar features
    out["Holiday"] = pd.to_numeric(df["Holiday"], errors="coerce").fillna(0).astype(int) if "Holiday" in df.columns else 0
    out["Holiday_prev"] = pd.to_numeric(df["Holiday_prev"], errors="coerce").fillna(0).astype(float) if "Holiday_prev" in df.columns else 0.0

    if "Day_of_week" in df.columns:
        out["Day_of_week"] = pd.to_numeric(df["Day_of_week"], errors="coerce").astype('Int64')

    return out

def _create_aggregated_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated medical category columns from granular reason columns.

    This function automatically sums related granular conditions into broader categories:
    - Respiratory_Cases = Asthma + Pneumonia + Shortness_of_Breath
    - Cardiac_Cases = Chest_Pain + Arrhythmia + Hypertensive_Emergency
    - Trauma_Cases = Fracture + Laceration + Burn + Fall_Injury
    - Gastrointestinal_Cases = Abdominal_Pain + Vomiting + Diarrhea
    - Infectious_Cases = Flu_Symptoms + Fever + Viral_Infection
    - Other_Cases = Headache + Dizziness + Allergic_Reaction + Mental_Health

    Returns the dataframe with aggregated columns added (keeps all original granular columns).
    """
    df = df.copy()

    # Define category mappings
    category_mappings = {
        "Respiratory_Cases": ["Asthma", "Pneumonia", "Shortness_of_Breath"],
        "Cardiac_Cases": ["Chest_Pain", "Arrhythmia", "Hypertensive_Emergency"],
        "Trauma_Cases": ["Fracture", "Laceration", "Burn", "Fall_Injury"],
        "Gastrointestinal_Cases": ["Abdominal_Pain", "Vomiting", "Diarrhea"],
        "Infectious_Cases": ["Flu_Symptoms", "Fever", "Viral_Infection"],
        "Other_Cases": ["Headache", "Dizziness", "Allergic_Reaction", "Mental_Health"]
    }

    # Create each aggregated category
    for category_name, component_cols in category_mappings.items():
        # Find which component columns exist in the dataframe
        available_cols = [col for col in component_cols if col in df.columns]

        if available_cols:
            # Sum available columns, filling NaN with 0
            df[category_name] = df[available_cols].fillna(0).sum(axis=1)

    return df

def _prep_reason(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare reason for visit data - keeps all granular reason columns as-is.
    Assumes 'datetime' column is already clean (timezone-naive) from Data Hub.

    This function extracts all numeric reason columns (asthma, pneumonia, fracture, etc.)
    and returns them without any aggregation for maximum flexibility in forecasting.
    """
    df = df.copy()

    # Data Hub should provide 'datetime' column - fallback to search if needed
    if "datetime" not in df.columns:
        dt_col = _find_datetime_col(df, want="datetime")
        if dt_col is None:
            raise ValueError("Reason CSV must include a 'datetime' column.")
        df["datetime"] = _to_datetime_series(df[dt_col])

    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in reason datetime column.")

    # Create Date for merging (normalize to midnight for daily aggregation)
    df["Date"] = _normalize_to_date(df["datetime"])

    # Initialize output with Date column only (drop datetime to avoid duplication)
    out = df[["Date"]].copy()

    # Extract ALL numeric columns (keep all granular reason columns as-is)
    # Exclude date/datetime columns, ID columns, and total_arrivals (duplicates Target_1)
    exclude_cols = [
        "Date", "datetime", "date", "timestamp", "ds",
        "id", "ID", "Id",
        "total_arrivals", "Total_Arrivals", "Total_arrivals", "TOTAL_ARRIVALS"
    ]

    for col in df.columns:
        if col not in exclude_cols and col not in out.columns:
            # Try to convert to numeric - if successful, include it
            try:
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                # Only include if at least some values are numeric
                if not numeric_series.isna().all():
                    out[col] = numeric_series
            except Exception:
                # Skip non-numeric columns
                pass

    # Return all columns (Date + all granular reason columns)
    return out

def fuse_data(patient_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame, reason_df: pd.DataFrame = None) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Fuse patient, weather, calendar, and reason datasets into a single training dataset.

    All datasets should have a clean 'datetime' column (timezone-naive) from Data Hub.
    This function creates 'Date' columns for daily aggregation and merging.
    """
    log = []
    required_count = 4 if reason_df is not None else 3

    if patient_df is None or weather_df is None or calendar_df is None:
        log.append(f"❌ Error: All {required_count} datasets (Patient, Weather, Calendar{', Reason' if reason_df is not None else ''}) must be loaded before fusion.")
        return None, log

    try:
        log.append("🚀 Starting data preparation...")
        log.append("ℹ️  Using clean timezone-naive 'datetime' columns from Data Hub")

        # Prepare each dataset (creates 'Date' column for daily aggregation)
        p = _prep_patient(patient_df)
        log.append(f"✅ Patient data prepared (Date created from datetime) - {len(p)} rows, {len(p.columns)} cols")
        log.append(f"   Patient columns: {', '.join(p.columns)}")

        w = _prep_weather(weather_df)
        log.append(f"✅ Weather data prepared (Date created from datetime) - {len(w)} rows, {len(w.columns)} cols")

        c = _prep_calendar(calendar_df)
        log.append(f"✅ Calendar data prepared (Date created from datetime) - {len(c)} rows, {len(c.columns)} cols")

        r = None
        if reason_df is not None and not reason_df.empty:
            log.append(f"ℹ️  Processing Reason for Visit dataset - {len(reason_df)} rows, {len(reason_df.columns)} cols")
            log.append(f"   Reason input columns: {', '.join(reason_df.columns)}")

            r = _prep_reason(reason_df)
            log.append(f"✅ Reason for Visit data prepared (Date created from datetime) - {len(r)} rows, {len(r.columns)} cols")

            # List the reason columns that will be merged (excluding Date)
            reason_cols = [col for col in r.columns if col != "Date"]
            if reason_cols:
                log.append(f"ℹ️  Extracted {len(reason_cols)} granular reason columns (excluded 'total_arrivals' - duplicates Target_1)")
                log.append(f"   Reason columns: {', '.join(reason_cols)}")
            else:
                log.append(f"⚠️  No reason columns found in dataset")
        else:
            log.append("⚠️  Reason for Visit dataset is None or empty - skipping reason data")

        # Merge datasets on 'Date' column (daily aggregation level)
        log.append("🔗 Merging Patient + Weather (inner join on Date)...")
        merged = p.merge(w, on="Date", how="inner", suffixes=("", "_wx"))
        log.append(f"  → Result: {merged.shape[0]} rows, {merged.shape[1]} cols")

        if merged.empty:
            log.append("❌ Error: Merge of Patient and Weather resulted in zero rows. Check date ranges.")
            return None, log

        log.append("🔗 Merging + Calendar (inner join on Date)...")
        merged = merged.merge(c, on="Date", how="inner")
        log.append(f"  → Result: {merged.shape[0]} rows, {merged.shape[1]} cols")

        if merged.empty:
            log.append("❌ Error: Merge with Calendar resulted in zero rows. Check date ranges.")
            return None, log

        if r is not None and not r.empty:
            log.append("🔗 Merging + Reason for Visit (inner join on Date)...")
            log.append(f"   Reason columns to merge: {', '.join([col for col in r.columns if col != 'Date'])}")

            merged = merged.merge(r, on="Date", how="inner")
            log.append(f"  → Final Result: {merged.shape[0]} rows, {merged.shape[1]} cols")
            log.append(f"   Columns after reason merge: {', '.join(merged.columns)}")

            if merged.empty:
                log.append("❌ Error: Merge with Reason data resulted in zero rows. Check date ranges.")
                return None, log
        else:
            log.append("⚠️  Skipping Reason for Visit merge (no reason data available)")

        # Derive Day_of_week if not present
        if "Day_of_week" not in merged.columns:
            merged["Day_of_week"] = merged["Date"].dt.dayofweek + 1
            log.append("ℹ️  Derived 'Day_of_week' from Date")

        # Rename patient_count to Target_1 for consistency
        if "patient_count" in merged.columns:
            merged = merged.rename(columns={"patient_count": "Target_1"})
            log.append("ℹ️  Renamed 'patient_count' to 'Target_1'")
        elif "Target_1" not in merged.columns:
            log.append("❌ Error: Neither 'patient_count' nor 'Target_1' found in merged data")
            return None, log

        # Safety check: Remove Total_Arrivals if it somehow got included (should be excluded in _prep_reason)
        if "Total_Arrivals" in merged.columns or "total_arrivals" in merged.columns:
            # Find the column name (case-insensitive)
            total_arrivals_col = None
            for col in merged.columns:
                if col.lower() == "total_arrivals":
                    total_arrivals_col = col
                    break

            if total_arrivals_col:
                merged = merged.drop(columns=[total_arrivals_col])
                log.append(f"ℹ️  Removed '{total_arrivals_col}' (duplicates Target_1 - should have been excluded earlier)")

        # Column ordering: Priority columns first, then append all remaining columns (including reason columns)
        priority_order = [
            "Date", "Day_of_week", "Holiday", "Moon_Phase", "Average_Temp", "Max_temp",
            "Average_wind", "Max_wind", "Average_mslp", "Total_precipitation",
            "Holiday_prev", "Target_1",
        ]

        # Keep priority columns that exist
        ordered_cols = [col for col in priority_order if col in merged.columns]

        # Append all remaining columns not in priority list (includes all reason columns)
        remaining_cols = [col for col in merged.columns if col not in ordered_cols]
        final_cols = ordered_cols + remaining_cols

        merged = merged[final_cols].sort_values("Date").reset_index(drop=True)
        log.append("✅ Selected, ordered, and sorted final columns")
        log.append(f"ℹ️  Final merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

        # Count how many reason columns are in the final dataset (excluding total_arrivals - it duplicates Target_1)
        reason_cols_in_final = [col for col in remaining_cols if col.lower() in [
            'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
            'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
            'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
            'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
        ]]

        if reason_cols_in_final:
            log.append(f"ℹ️  Included {len(reason_cols_in_final)} reason columns: {', '.join(reason_cols_in_final)}")

        log.append(f"   Final columns: {', '.join(merged.columns)}")
        log.append("🎉 Data fusion complete!")

        return merged, log

    except ValueError as ve:
        log.append(f"❌ Data Preparation Error: {ve}")
        return None, log
    except Exception as e:
        log.append(f"❌ Unexpected Error during fusion: {e}")
        import traceback
        log.append(f"Traceback: {traceback.format_exc()}")
        return None, log

