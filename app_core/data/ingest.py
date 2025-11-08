import pandas as pd
import streamlit as st
import re
import numpy as np

# Your helper functions below
def _upload_generic(title: str, prefix: str, datetime_target: str = "datetime"):
    st.markdown(f"### {title}")
    # ... rest of your function exactly as before ...


# Move these from your current code without changing behavior:
# - _clean_name
# - _find_datetime_col
# - _combine_date_time_columns
# - _to_datetime_series
# - _upload_generic (if you want to reuse as a function)

def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())

def _find_datetime_col(df: pd.DataFrame, want: str = "datetime") -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_clean_name(c): c for c in df.columns}
    preferred_aliases = {
        "datetime": ["datetime", "dateandtime", "timestamp", "datestamp"],
        "date": ["date", "day", "calendar_date"]
    }
    for alias in preferred_aliases.get(want, []):
        if alias in normalized:
            return normalized[alias]
    common = ["date", "time", "day", "daydate", "recordedat", "createdat",
              "ts", "logtime", "eventtime", "eventdate", "ds"]
    for key in common:
        if key in normalized:
            return normalized[key]
    for k, v in normalized.items():
        if any(token in k for token in ["date", "time", "stamp"]):
            return v
    return None

def _combine_date_time_columns(df: pd.DataFrame) -> pd.Series | None:
    cols = {_clean_name(c): c for c in df.columns}
    date_col = cols.get("date", None)
    time_col = cols.get("time", None)
    if date_col and time_col:
        try:
            dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
            if dt.notna().any():
                return dt
        except Exception:
            pass
    return None

def _to_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def _upload_generic(title: str, key_prefix: str, desired_datetime_name: str):
    st.markdown(f"### {title}")
    up = st.file_uploader(f"Select {title} CSV", type=['csv'], key=f"{key_prefix}_upload")
    if up is None:
        if st.session_state.get(f"{key_prefix}_loaded"):
            st.info(f"Using previously loaded {title}.")
        return
    try:
        df = pd.read_csv(up)
        if df.empty:
            st.error("❌ Uploaded file is empty.")
            st.session_state[f"{key_prefix}_data"] = None
            st.session_state[f"{key_prefix}_loaded"] = False
            return
        df = df.copy(); df.columns = [c.strip() for c in df.columns]
        dt_series = None; found_col_name = None
        combined = _combine_date_time_columns(df)
        if combined is not None and combined.notna().any():
            dt_series = combined; found_col_name = "Date + Time (Combined)"
        else:
            found_col = _find_datetime_col(df, want="datetime" if desired_datetime_name == "datetime" else "date")
            if found_col:
                found_col_name = found_col; dt_series = _to_datetime_series(df[found_col])
        if dt_series is None or dt_series.isna().all():
            numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            parsed_any = None; candidate_col_name = None
            for c in numeric_candidates:
                s = df[c]
                for transform, unit in [
                    (lambda x: pd.to_datetime(x, unit="s", errors="coerce"), "s"),
                    (lambda x: pd.to_datetime(x, unit="ms", errors="coerce"), "ms"),
                    (lambda x: pd.to_datetime("1899-12-30") + pd.to_timedelta(x, unit="D"), "excel")
                ]:
                    plausible = False
                    if unit == "s" and s.between(946684800, 2524608000).any(): plausible = True
                    elif unit == "ms" and s.between(946684800000, 2524608000000).any(): plausible = True
                    elif unit == "excel" and s.between(36526, 54787).any(): plausible = True
                    if plausible:
                        try:
                            cand = transform(s)
                            if cand.notna().any() and cand.min() > pd.Timestamp("1970-01-01") and cand.max() < pd.Timestamp("2070-01-01"):
                                parsed_any = cand; candidate_col_name = c; break
                        except Exception: continue
                if parsed_any is not None: break
            if parsed_any is not None:
                dt_series = parsed_any; found_col_name = f"{candidate_col_name} (parsed as numeric)"
        if dt_series is None or dt_series.isna().all():
            st.error(f"❌ No usable date/datetime column found in {title}.")
            with st.expander("Columns Detected"): st.write(list(df.columns))
            st.session_state[f"{key_prefix}_data"] = None
            st.session_state[f"{key_prefix}_loaded"] = False
            return
        if desired_datetime_name == "datetime":
            df["datetime"] = dt_series
        else:
            df["date"] = dt_series.dt.normalize()
        st.session_state[f"{key_prefix}_data"] = df
        st.session_state[f"{key_prefix}_loaded"] = True
        if key_prefix == "patient": st.session_state["patient_data"] = df
        elif key_prefix == "weather": st.session_state["weather_data"] = df
        elif key_prefix == "calendar": st.session_state["calendar_data"] = df
        st.success(f"✅ {title} loaded ({df.shape[0]} rows, {df.shape[1]} cols). Date detected: '{found_col_name}'.")
        with st.expander("🔍 Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading {title}: {e}")
        st.session_state[f"{key_prefix}_data"] = None
        st.session_state[f"{key_prefix}_loaded"] = False
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
   
