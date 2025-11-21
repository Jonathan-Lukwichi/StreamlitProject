import pandas as pd
import streamlit as st
from app_core.ui.theme import apply_css
from app_core.state.session import init_state
from app_core.ui.components import header
from app_core.data.ingest import _find_datetime_col, _to_datetime_series, _combine_date_time_columns




# Move these from your current code without changing behavior:
# - _prep_patient
# - _prep_weather
# - _prep_calendar
# - fuse_data

def _prep_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_col = "datetime" if "datetime" in df.columns else _find_datetime_col(df, want="datetime")
    if dt_col is None:
        raise ValueError("Patient CSV must include a date/datetime column (e.g., 'datetime', 'Date', 'timestamp').")
    dt = _to_datetime_series(df[dt_col])
    if dt.isna().all():
        raise ValueError("Could not parse any valid datetimes in patient date/datetime column.")
    df["datetime"] = dt
    df["Date"] = df["datetime"].dt.normalize()
    if "patient_count" not in df.columns:
        for alt in ["Target_1", "patients", "count", "y", "value"]:
            if alt in df.columns:
                df["patient_count"] = pd.to_numeric(df[alt], errors="coerce")
                break
    if "patient_count" not in df.columns:
        raise ValueError("Patient CSV must include 'patient_count' (or a recognizable equivalent).")
    df["patient_count"] = pd.to_numeric(df["patient_count"], errors="coerce")
    return df[["Date", "datetime", "patient_count"]]

def _prep_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_col = "datetime" if "datetime" in df.columns else _find_datetime_col(df, want="datetime")
    if dt_col is None:
        raise ValueError("Weather CSV must include a date/datetime column.")
    df["datetime"] = _to_datetime_series(df[dt_col])
    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in weather date/datetime column.")
    df["Date"] = df["datetime"].dt.normalize()
    out = df[["Date", "datetime"]].copy()
    weather_cols = ["Average_Temp", "Max_temp", "Average_wind", "Max_wind", "Average_mslp", "Total_precipitation", "Moon_Phase"]
    for col in weather_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out

def _prep_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_col = "date" if "date" in df.columns else _find_datetime_col(df, want="date")
    if base_col is None:
        raise ValueError("Calendar CSV must include a date column (e.g., 'date', 'Date').")
    dt = _to_datetime_series(df[base_col])
    if dt.isna().all():
        raise ValueError("Could not parse any valid dates in calendar 'date' column.")
    out = pd.DataFrame({"Date": dt.dt.normalize()})
    out["Holiday"] = pd.to_numeric(df["Holiday"], errors="coerce").fillna(0).astype(int) if "Holiday" in df.columns else 0
    out["Holiday_prev"] = pd.to_numeric(df["Holiday_prev"], errors="coerce").fillna(0).astype(float) if "Holiday_prev" in df.columns else 0.0
    if "Day_of_week" in df.columns:
        out["Day_of_week"] = pd.to_numeric(df["Day_of_week"], errors="coerce").astype('Int64')
    return out

def _prep_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare reason for visit data - keeps ALL granular columns for maximum flexibility.

    This function preserves all numeric reason columns (Asthma, Pneumonia, Fracture,
    Chest_Pain, etc.) instead of aggregating them, allowing for detailed multi-target
    forecasting at the granular level.
    """
    df = df.copy()
    dt_col = "datetime" if "datetime" in df.columns else _find_datetime_col(df, want="datetime")
    if dt_col is None:
        raise ValueError("Reason CSV must include a date/datetime column.")
    df["datetime"] = _to_datetime_series(df[dt_col])
    if df["datetime"].isna().all():
        raise ValueError("Could not parse any valid datetimes in reason date/datetime column.")
    df["Date"] = df["datetime"].dt.normalize()

    # Initialize output with Date column
    out = df[["Date"]].copy()

    # Extract ALL numeric columns (keeps all granular reason columns)
    # Exclude date/datetime columns from numeric extraction
    exclude_cols = ["Date", "datetime", "date", "timestamp", "ds"]
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

    return out

def fuse_data(patient_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame, reason_df: pd.DataFrame = None) -> tuple[pd.DataFrame | None, list[str]]:
    log = []
    required_count = 4 if reason_df is not None else 3
    if patient_df is None or weather_df is None or calendar_df is None:
        log.append(f"❌ Error: All {required_count} datasets (Patient, Weather, Calendar{', Reason' if reason_df is not None else ''}) must be loaded before fusion.")
        return None, log
    try:
        log.append("🚀 Starting data preparation...")
        p = _prep_patient(patient_df); log.append("✅ Patient data prepared.")
        w = _prep_weather(weather_df); log.append("✅ Weather data prepared.")
        c = _prep_calendar(calendar_df); log.append("✅ Calendar data prepared.")
        r = None
        if reason_df is not None:
            r = _prep_reason(reason_df); log.append("✅ Reason for Visit data prepared.")
        log.append("ℹ️ Verified 'Date' columns are aligned (datetime64[ns] @ 00:00).")
        log.append("🔗 Merging Patient + Weather (inner join on Date)...")
        merged = p.merge(w.drop(columns=['datetime'], errors='ignore'), on="Date", how="inner", suffixes=("", "_wx"))
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
        if r is not None:
            log.append("🔗 Merging + Reason for Visit (inner join on Date)...")
            merged = merged.merge(r, on="Date", how="inner")
            log.append(f"  → Final Result: {merged.shape[0]} rows, {merged.shape[1]} cols")
            if merged.empty:
                log.append("❌ Error: Merge with Reason data resulted in zero rows. Check date ranges.")
                return None, log
        if "Day_of_week" not in merged.columns:
            merged["Day_of_week"] = merged["Date"].dt.dayofweek + 1
            log.append("ℹ️ Derived 'Day_of_week' from Date.")
        if "patient_count" in merged.columns:
            merged = merged.rename(columns={"patient_count": "Target_1"})
            log.append("ℹ️ Renamed 'patient_count' to 'Target_1'.")
        elif "Target_1" not in merged.columns:
            log.append("❌ Error: Neither 'patient_count' nor 'Target_1' found in merged data.")
            return None, log

        # Intelligent duplicate detection: Check if Total_Arrivals equals Target_1
        if "Total_Arrivals" in merged.columns and "Target_1" in merged.columns:
            if merged["Total_Arrivals"].equals(merged["Target_1"]):
                merged = merged.drop(columns=["Total_Arrivals"])
                log.append("ℹ️ Detected Total_Arrivals == Target_1, removed duplicate column.")
            else:
                log.append("ℹ️ Total_Arrivals and Target_1 are different, keeping both columns.")

        # Column ordering: Priority columns first, then append all remaining columns
        priority_order = [
            "Date", "Day_of_week", "Holiday", "Moon_Phase", "Average_Temp", "Max_temp",
            "Average_wind", "Max_wind", "Average_mslp", "Total_precipitation",
            "Holiday_prev", "Target_1",
            "Respiratory_Cases", "Cardiac_Cases", "Trauma_Cases", "Gastrointestinal_Cases",
            "Infectious_Cases", "Other_Cases", "Total_Arrivals",
        ]
        # Keep priority columns that exist
        ordered_cols = [col for col in priority_order if col in merged.columns]
        # Append all remaining columns not in priority list
        remaining_cols = [col for col in merged.columns if col not in ordered_cols]
        final_cols = ordered_cols + remaining_cols

        merged = merged[final_cols].sort_values("Date").reset_index(drop=True)
        log.append("✅ Selected, ordered, and sorted final columns.")
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

