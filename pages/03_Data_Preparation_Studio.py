# =============================================================================
# pages/03_Data_Preparation_Studio.py — Data Preparation Studio (Premium)
# Advanced data fusion and feature engineering with premium design
# =============================================================================
from __future__ import annotations

import re
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, CARD_BG, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        for k, v in {
            "patient_data": None,
            "weather_data": None,
            "calendar_data": None,
            "reason_data": None,
            "merged_data": None,
            "processed_df": None,
            "patient_loaded": False,
            "weather_loaded": False,
            "calendar_loaded": False,
            "_nav_intent": None,
        }.items():
            st.session_state.setdefault(k, v)

try:
    from app_core.ui.components import header
except Exception:
    def header(title: str, subtitle: str = "", icon: str = ""):
        st.markdown(f"### {icon} {title}")
        if subtitle:
            st.caption(subtitle)

# Data core (defensive fallbacks)
try:
    from app_core.data.fusion import fuse_data
except Exception:
    def fuse_data(patient_df, weather_df, calendar_df, reason_df=None):
        # naive outer-merge fallback on best-effort datetime/date keys
        def _key(df):
            for c in ["datetime", "Date", "date", "timestamp", "ds", "time"]:
                if c in df.columns: return c
            return None
        p, w, c = patient_df.copy(), weather_df.copy(), calendar_df.copy()
        r = reason_df.copy() if reason_df is not None else None
        for df in (p, w, c, r):
            if df is not None:
                for k in ["datetime","Date","date","timestamp","ds","time"]:
                    if k in df.columns:
                        df[k] = pd.to_datetime(df[k], errors="coerce")
        pk, wk, ck = _key(p), _key(w), _key(c)
        rk = _key(r) if r is not None else None
        merged = p
        if w is not None and wk: merged = merged.merge(w, left_on=pk, right_on=wk, how="outer")
        if c is not None and ck: merged = merged.merge(c, left_on=pk or wk, right_on=ck, how="outer")
        if r is not None and rk: merged = merged.merge(r, left_on=pk or wk or ck, right_on=rk, how="outer")

        # Intelligent duplicate detection
        if "Total_Arrivals" in merged.columns and "Target_1" in merged.columns:
            if merged["Total_Arrivals"].equals(merged["Target_1"]):
                merged = merged.drop(columns=["Total_Arrivals"])

        return merged, {}

try:
    from app_core.data import process_dataset
except Exception:
    def process_dataset(df: pd.DataFrame, n_lags: int, date_col=None, ed_col=None, strict_drop_na_edges=True):
        # basic placeholder: add ED_1..ED_n and Target_1..Target_n from first numeric col
        # plus clinical categories and multi-targets
        work = df.copy()
        y = None
        for c in work.columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                y = c; break
        if y:
            for i in range(1, n_lags+1):
                work[f"ED_{i}"] = work[y].shift(i)
                work[f"Target_{i}"] = work[y].shift(-i)

        # Create clinical categories from granular reason columns
        clinical_categories = {
            "RESPIRATORY": ["asthma", "pneumonia", "shortness_of_breath"],
            "CARDIAC": ["chest_pain", "arrhythmia", "hypertensive_emergency"],
            "TRAUMA": ["fracture", "laceration", "burn", "fall_injury"],
            "GASTROINTESTINAL": ["abdominal_pain", "vomiting", "diarrhea"],
            "INFECTIOUS": ["flu_symptoms", "fever", "viral_infection"],
            "NEUROLOGICAL": ["headache", "dizziness"],
            "OTHER": ["allergic_reaction", "mental_health"],
        }
        col_map = {c.lower(): c for c in work.columns}
        for category, reasons in clinical_categories.items():
            matching_cols = [col_map[r] for r in reasons if r in col_map]
            if matching_cols:
                work[category] = work[matching_cols].fillna(0).sum(axis=1)

        # Generate multi-targets for clinical categories
        category_cols = list(clinical_categories.keys())
        for col in category_cols:
            if col in work.columns:
                for i in range(1, n_lags+1):
                    work[f"{col}_{i}"] = work[col].shift(-i)

        # Also generate for granular reasons if present
        granular_reasons = [
            "asthma", "pneumonia", "shortness_of_breath", "chest_pain", "arrhythmia",
            "hypertensive_emergency", "fracture", "laceration", "burn", "fall_injury",
            "abdominal_pain", "vomiting", "diarrhea", "flu_symptoms", "fever",
            "viral_infection", "headache", "dizziness", "allergic_reaction", "mental_health"
        ]
        for reason in granular_reasons:
            if reason in col_map:
                col = col_map[reason]
                for i in range(1, n_lags+1):
                    work[f"{col}_{i}"] = work[col].shift(-i)

        if strict_drop_na_edges:
            work = work.dropna(how="any")
        return work, {}

# ---------- PAGE CONFIG --------------------------------------------------------
st.set_page_config(
    page_title="Data Preparation Studio - HealthForecast AI",
    page_icon="🧠",
    layout="wide",
)



# ---------- UTILITIES (time/columns/order) -------------------------------------
def _find_time_like_col(df: pd.DataFrame) -> str | None:
    priority = ["datetime", "timestamp", "date_time", "date", "ds"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_map: return lower_map[p]
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "stamp" in cl: return c
    return None

def _ensure_datetime_and_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = _find_time_like_col(df)
    dt = None
    if time_col is not None:
        try:
            dt = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        except Exception:
            dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt is None or (hasattr(dt, "isna") and dt.isna().all()):
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index.to_series()
        else:
            dt = pd.to_datetime(pd.Series([pd.NaT] * len(df)))
    df["datetime"] = dt
    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["Date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    else:
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
            except Exception:
                df["Date"] = df["Date"].astype(str)
        else:
            df["Date"] = df.index.astype(str)
    return df

def _detect_time_columns(columns: list[str]) -> list[str]:
    preferred = ["datetime", "Date", "timestamp", "date_time", "date", "ds", "time"]
    lower_map = {c.lower(): c for c in columns}
    ordered = []
    for p in preferred:
        if p.lower() in lower_map:
            col = lower_map[p.lower()]
            if col in columns and col not in ordered:
                ordered.append(col)
    for c in columns:
        cl = c.lower()
        if ("date" in cl or "time" in cl or "stamp" in cl) and c not in ordered:
            ordered.append(c)
    return ordered

def _detect_calendar_columns(columns: list[str]) -> list[str]:
    calendar_preferred = [
        "day_of_week","Day_of_week","is_weekend","day_of_month","week_of_year","month","year",
        "Holiday","Holiday_prev","Moon_Phase"
    ]
    lower_map = {c.lower(): c for c in columns}
    found = []
    for cand in calendar_preferred:
        if cand.lower() in lower_map:
            c0 = lower_map[cand.lower()]
            if c0 in columns and c0 not in found:
                found.append(c0)
    return found

def _detect_weather_columns(columns: list[str]) -> list[str]:
    patterns = ["temp","wind","mslp","pressure","precip","rain","snow","humidity","dew","cloud","visibility","uv","moon"]
    out = []
    for c in columns:
        cl = c.lower()
        if any(p in cl for p in patterns): out.append(c)
    # keep Moon_Phase with calendar:
    out = [c for c in out if c != "Moon_Phase"]
    return out

def _detect_ed_columns(columns: list[str]) -> list[str]:
    ed_lags = [c for c in columns if re.fullmatch(r"ED_\d+", c)]
    return sorted(ed_lags, key=lambda x: int(x.split("_")[1]))

def _detect_target_columns(columns: list[str]) -> list[str]:
    """Detect all target columns including multi-target forecasting patterns."""
    # Standard targets (Target_1, Target_2, etc.)
    targets = [c for c in columns if re.fullmatch(r"Target_\d+", c)]

    # Multi-target patterns (Respiratory_Cases_1, Cardiac_Cases_2, etc.)
    reason_patterns = [
        "Respiratory_Cases", "Cardiac_Cases", "Trauma_Cases",
        "Gastrointestinal_Cases", "Infectious_Cases", "Other_Cases", "Total_Arrivals"
    ]
    for pattern in reason_patterns:
        multi_targets = [c for c in columns if re.fullmatch(rf"{pattern}_\d+", c)]
        targets.extend(multi_targets)

    # Sort by extracting the numeric suffix
    def sort_key(col):
        match = re.search(r"_(\d+)$", col)
        return (col.rsplit("_", 1)[0], int(match.group(1)) if match else 0)

    return sorted(targets, key=sort_key)

def order_columns_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    time_cols = _detect_time_columns(cols);   remain1 = [c for c in cols if c not in time_cols]
    cal_cols  = _detect_calendar_columns(remain1);  remain2 = [c for c in remain1 if c not in cal_cols]
    met_cols  = _detect_weather_columns(remain2);   remain3 = [c for c in remain2 if c not in met_cols]
    ed_lags   = _detect_ed_columns(remain3);        remain4 = [c for c in remain3 if c not in ed_lags]
    tgt_cols  = _detect_target_columns(remain4);    remain5 = [c for c in remain4 if c not in tgt_cols]
    new_order = time_cols + cal_cols + met_cols + ed_lags + tgt_cols + remain5
    seen = set()
    new_order = [c for c in new_order if (c not in seen and not seen.add(c))]
    return df[new_order]

def _harmonize_day_of_week_and_ed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "day_of_week" in df.columns and "Day_of_week" in df.columns:
        df["day_of_week"] = df["Day_of_week"]; df = df.drop(columns=["Day_of_week"])
    elif "Day_of_week" in df.columns and "day_of_week" not in df.columns:
        df = df.rename(columns={"Day_of_week": "day_of_week"})
    if "ED" in df.columns and "ED_1" in df.columns:
        df = df.drop(columns=["ED"])
    elif "ED" in df.columns and "ED_1" not in df.columns:
        df = df.rename(columns={"ED": "ED_1"})
    return df

# ---------- CACHED PROCESSORS ---------------------------------------------------
@st.cache_data(show_spinner=False)
def _run_fusion_cached(patient_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame, reason_df: pd.DataFrame = None):
    merged, log = fuse_data(patient_df, weather_df, calendar_df, reason_df)
    return merged, log

@st.cache_data(show_spinner=False)
def _preprocess_after_fusion_cached(merged_df: pd.DataFrame, n_lags: int, strict_drop: bool):
    processed_df, _ = process_dataset(
        merged_df, n_lags=n_lags, date_col=None, ed_col=None, strict_drop_na_edges=strict_drop
    )
    processed_df = _ensure_datetime_and_date(processed_df)
    
    # Drop columns with _x and _y suffixes created by the merge
    cols_to_drop = [c for c in processed_df.columns if c.endswith('_x') or c.endswith('_y')]
    if cols_to_drop:
        processed_df = processed_df.drop(columns=cols_to_drop)
        
    processed_df = _harmonize_day_of_week_and_ed(processed_df)
    processed_df = order_columns_pipeline(processed_df)
    return processed_df

# ---------- ANALYST-STYLE REPORTING --------------------------------------------
def _render_preprocessed_info(proc: pd.DataFrame):
    """Render comprehensive preprocessed dataset information using modern categorization."""
    n_rows, n_cols = proc.shape
    min_dt, max_dt = None, None
    if "datetime" in proc.columns and pd.api.types.is_datetime64_any_dtype(proc["datetime"]):
        min_dt = pd.to_datetime(proc["datetime"]).min()
        max_dt = pd.to_datetime(proc["datetime"]).max()
    elif "Date" in proc.columns:
        try:
            dt_series = pd.to_datetime(proc["Date"], errors="coerce", dayfirst=True)
            min_dt, max_dt = dt_series.min(), dt_series.max()
        except Exception:
            pass

    # Categorize columns using same logic as visual grouping
    def categorize_columns(df):
        """Categorize columns into meaningful groups for ML training."""
        cols = list(df.columns)

        date_cols = []
        calendar_cols = []
        weather_cols = []
        current_reason_cols = []
        past_arrivals_cols = []
        past_reason_cols = []
        future_arrivals_cols = []
        future_reason_cols = []
        other_cols = []

        # Known reason column names (granular medical categories)
        reason_keywords = [
            'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
            'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
            'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
            'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
        ]

        for col in cols:
            col_lower = col.lower()

            # Date columns
            if col_lower in ['date', 'datetime', 'ds', 'timestamp']:
                date_cols.append(col)

            # Calendar features
            elif col_lower in ['day_of_week', 'holiday', 'holiday_prev', 'is_weekend',
                              'day_of_month', 'week_of_year', 'month', 'year']:
                calendar_cols.append(col)

            # Weather features
            elif col_lower in ['average_temp', 'max_temp', 'average_wind', 'max_wind',
                              'average_mslp', 'total_precipitation', 'moon_phase']:
                weather_cols.append(col)

            # Past arrivals features (ED_1, ED_2, etc.)
            elif col_lower.startswith('ed_') and len(col_lower) > 3 and col_lower[3:].isdigit():
                past_arrivals_cols.append(col)

            # Future arrivals features (Target_1, Target_2, etc.)
            elif col_lower.startswith('target_'):
                future_arrivals_cols.append(col)

            # Past reason features (asthma_lag_1, pneumonia_lag_1, etc.)
            elif '_lag_' in col_lower and any(keyword in col_lower for keyword in reason_keywords):
                past_reason_cols.append(col)

            # Future reason features (asthma_1, pneumonia_1, etc.)
            elif '_' in col and col.split('_')[-1].isdigit() and any(keyword in col_lower for keyword in reason_keywords):
                future_reason_cols.append(col)

            # Current reason features (asthma, pneumonia, etc. without suffixes)
            elif any(col_lower == keyword for keyword in reason_keywords):
                current_reason_cols.append(col)

            # Other columns
            else:
                other_cols.append(col)

        return {
            'Date': date_cols,
            'Calendar Features': calendar_cols,
            'Weather Features': weather_cols,
            'Current Reason Columns': current_reason_cols,
            'Past Features (Arrivals)': past_arrivals_cols,
            'Past Features (Reasons)': past_reason_cols,
            'Future Features (Arrivals)': future_arrivals_cols,
            'Future Features (Reasons)': future_reason_cols,
            'Other': other_cols
        }

    categorized = categorize_columns(proc)

    # Calculate summary stats
    total_past_features = len(categorized['Past Features (Arrivals)']) + len(categorized['Past Features (Reasons)'])
    total_future_features = len(categorized['Future Features (Arrivals)']) + len(categorized['Future Features (Reasons)'])

    miss = proc.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(8)

    # Dataset Overview
    st.markdown("**Dataset Overview**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", f"{n_rows:,}")
        st.metric("Total Columns", f"{n_cols}")
    with col2:
        if min_dt is not None and max_dt is not None:
            st.write(f"**Time Coverage:**")
            st.write(f"{min_dt.date()} → {max_dt.date()}")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Column Families with visual grouping
    st.markdown("**Column Families**")

    # Define colors and icons
    category_colors = {
        'Date': '#3B82F6',
        'Calendar Features': '#8B5CF6',
        'Weather Features': '#10B981',
        'Current Reason Columns': '#F59E0B',
        'Past Features (Arrivals)': '#06B6D4',
        'Past Features (Reasons)': '#14B8A6',
        'Future Features (Arrivals)': '#EF4444',
        'Future Features (Reasons)': '#EC4899',
        'Other': '#6B7280'
    }

    category_icons = {
        'Date': '📅',
        'Calendar Features': '🗓️',
        'Weather Features': '🌤️',
        'Current Reason Columns': '🏥',
        'Past Features (Arrivals)': '⏮️',
        'Past Features (Reasons)': '📋',
        'Future Features (Arrivals)': '🎯',
        'Future Features (Reasons)': '🔮',
        'Other': '📦'
    }

    # Display categories with columns
    for category, columns in categorized.items():
        if columns:  # Only show categories that have columns
            color = category_colors.get(category, '#6B7280')
            icon = category_icons.get(category, '•')

            with st.expander(f"{icon} **{category}** ({len(columns)} columns)", expanded=False):
                cols_display = ", ".join([f"`{col}`" for col in columns])
                st.markdown(
                    f"""
                    <div style='padding: 0.75rem; background: rgba(255,255,255,0.02); border-left: 3px solid {color}; border-radius: 4px; margin: 0.5rem 0;'>
                        <div style='color: {color}; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem;'>{category}</div>
                        <div style='color: #CBD5E1; font-size: 0.8125rem; font-family: monospace;'>{cols_display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Date & Calendar", len(categorized['Date']) + len(categorized['Calendar Features']))
    with col2:
        st.metric("🌤️ Weather Features", len(categorized['Weather Features']))
    with col3:
        st.metric("⏮️ Past Features (Lag)", total_past_features)
    with col4:
        st.metric("🎯 Future Targets", total_future_features)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Data Quality
    st.markdown("**Data Quality (quick look)**")
    if len(miss) > 0:
        st.warning(f"Found {len(miss)} columns with missing values:")
        for c in miss.index:
            st.write(f"- `{c}`: {int(miss[c])} missing ({100 * miss[c] / n_rows:.1f}%)")
    else:
        st.success("✅ No major missingness detected (top columns).")

# ---------- UI HELPERS ----------------------------------------------------------
def _dataset_badge(ok: bool, label: str):
    rgb = "16,185,129" if ok else "245,158,11"
    color = SUCCESS_COLOR if ok else WARNING_COLOR
    icon = "✅" if ok else "⏳"
    st.markdown(
        f"""
        <span class='hf-pill' style='background:rgba({rgb},.12);color:{color};border:1px solid rgba({rgb},.25);'>
            {icon} {label}
        </span>
        """,
        unsafe_allow_html=True,
    )

def _detect_key(df: pd.DataFrame) -> str | None:
    return _find_time_like_col(df) if df is not None and not df.empty else None

def _merge_plan_preview(patient_df, weather_df, calendar_df, reason_df=None):
    pk = _detect_key(patient_df)
    wk = _detect_key(weather_df)
    ck = _detect_key(calendar_df)
    rk = _detect_key(reason_df)

    st.markdown("<div class='hf-feature-card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='margin-bottom: 1rem;'>
          <h3 style='font-size: 1.125rem; font-weight: 700; margin-bottom: 0.75rem;'>🔑 Detected Time Keys</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Patient Dataset**")
        st.code(pk or "—", language=None)
    with col2:
        st.markdown(f"**Weather Dataset**")
        st.code(wk or "—", language=None)
    with col3:
        st.markdown(f"**Calendar Dataset**")
        st.code(ck or "—", language=None)
    with col4:
        st.markdown(f"**Reason for Visit**")
        st.code(rk or "—", language=None)

    st.caption("✨ Keys are auto-normalized to `datetime` and `Date` columns during the fusion process")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    try:
        import graphviz
        st.markdown("**Merge Flow Diagram**")
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR", nodesep="0.4", splines="spline")
        dot.node("P", f"Patient\nkey: {pk or '—'}", shape="box", style="filled", fillcolor="#3b82f6", fontcolor="white")
        dot.node("W", f"Weather\nkey: {wk or '—'}", shape="box", style="filled", fillcolor="#22d3ee", fontcolor="white")
        dot.node("C", f"Calendar\nkey: {ck or '—'}", shape="box", style="filled", fillcolor="#22c55e", fontcolor="white")
        dot.node("R", f"Reason\nkey: {rk or '—'}", shape="box", style="filled", fillcolor="#a855f7", fontcolor="white")
        dot.node("M", "Merged\nDataset", shape="ellipse", style="filled", fillcolor="#f59e0b", fontcolor="white")
        dot.edges([("P","M"), ("W","M"), ("C","M"), ("R","M")])
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:
        st.caption("📊 Graphviz not available; skipping merge flow diagram")
    st.markdown("</div>", unsafe_allow_html=True)

def _summarize_df(df: pd.DataFrame, title: str):
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='hf-feature-card' style='padding: 1.5rem;'>
          <h3 style='margin-bottom: 0.75rem; font-size: 1.125rem; font-weight: 700;'>{title}</h3>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("**Data Preview (First 10 rows)**")
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.markdown("**Schema Summary**")
        non_null = df.notnull().sum()
        missing = df.isnull().sum()
        summary = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non_null": non_null,
            "missing": missing
        }).reset_index().rename(columns={"index": "column"})
        st.dataframe(summary, use_container_width=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- PAGE BODY -----------------------------------------------------------
def page_data_preparation_studio():
    # Apply premium theme and sidebar
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    add_logout_button()

    # Apply fluorescent effects + Custom Tab Styling
    st.markdown(f"""
    <style>
    /* Fluorescent Effects for Data Preparation Studio */
    @keyframes float-orb {{
        0%, 100% {{ transform: translate(0, 0) scale(1); opacity: 0.25; }}
        50% {{ transform: translate(30px, -30px) scale(1.05); opacity: 0.35; }}
    }}
    .fluorescent-orb {{
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(70px);
    }}
    .orb-1 {{
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
        top: 15%;
        right: 20%;
        animation: float-orb 25s ease-in-out infinite;
    }}
    .orb-2 {{
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
        bottom: 20%;
        left: 15%;
        animation: float-orb 30s ease-in-out infinite;
        animation-delay: 5s;
    }}
    @keyframes sparkle {{
        0%, 100% {{ opacity: 0; transform: scale(0); }}
        50% {{ opacity: 0.6; transform: scale(1); }}
    }}
    .sparkle {{
        position: fixed;
        width: 3px;
        height: 3px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
        border-radius: 50%;
        pointer-events: none;
        z-index: 2;
        animation: sparkle 3s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }}
    .sparkle-1 {{ top: 25%; left: 35%; animation-delay: 0s; }}
    .sparkle-2 {{ top: 65%; left: 70%; animation-delay: 1s; }}
    .sparkle-3 {{ top: 45%; left: 15%; animation-delay: 2s; }}

    /* ========================================
       CUSTOM TAB STYLING (Modern Design)
       ======================================== */

    /* Tab container */
    .stTabs {{
        background: transparent;
        margin-top: 1rem;
    }}

    /* Tab list (container for all tab buttons) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(11, 17, 32, 0.6), rgba(5, 8, 22, 0.5));
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}

    /* Individual tab buttons */
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background: transparent;
        border-radius: 12px;
        padding: 0 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: {BODY_TEXT};
        border: 1px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* Tab button hover effect */
    .stTabs [data-baseweb="tab"]:hover {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1));
        border-color: rgba(59, 130, 246, 0.3);
        color: {TEXT_COLOR};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }}

    /* Active/selected tab */
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(34, 211, 238, 0.2)) !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        color: {TEXT_COLOR} !important;
        box-shadow:
            0 0 20px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }}

    /* Active tab indicator (underline) - hide it */
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: transparent;
    }}

    /* Tab panels (content area) */
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 1.5rem;
    }}

    @media (max-width: 768px) {{
        .fluorescent-orb {{ width: 200px !important; height: 200px !important; filter: blur(50px); }}
        .sparkle {{ display: none; }}

        /* Make tabs stack vertically on mobile */
        .stTabs [data-baseweb="tab-list"] {{
            flex-direction: column;
        }}

        .stTabs [data-baseweb="tab"] {{
            width: 100%;
        }}
    }}
    </style>
    <div class="fluorescent-orb orb-1"></div>
    <div class="fluorescent-orb orb-2"></div>
    <div class="sparkle sparkle-1"></div>
    <div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div>
    """, unsafe_allow_html=True)

    # Premium Hero Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>🧠</div>
          <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Data Preparation Studio</h1>
          <p class='hf-feature-description' style='font-size: 1rem; max-width: 800px; margin: 0 auto;'>
            Intelligent data fusion and feature engineering pipeline for production-ready forecasting models
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pull uploaded sources
    patient_df  = st.session_state.get("patient_data")
    weather_df  = st.session_state.get("weather_data")
    calendar_df = st.session_state.get("calendar_data")
    reason_df   = st.session_state.get("reason_data")

    patient_ready  = patient_df is not None and not getattr(patient_df, "empty", True)
    weather_ready  = weather_df is not None and not getattr(weather_df, "empty", True)
    calendar_ready = calendar_df is not None and not getattr(calendar_df, "empty", True)
    reason_ready   = reason_df is not None and not getattr(reason_df, "empty", True)
    all_ready = patient_ready and weather_ready and calendar_ready and reason_ready

    # Premium Data Source Status Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_icon = "✅" if patient_ready else "⏳"
        status_text = "Ready" if patient_ready else "Pending"
        rgb = "16,185,129" if patient_ready else "245,158,11"
        color = SUCCESS_COLOR if patient_ready else WARNING_COLOR
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Patient Dataset</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {status_text}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Core patient arrival data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        status_icon = "✅" if weather_ready else "⏳"
        status_text = "Ready" if weather_ready else "Pending"
        rgb = "16,185,129" if weather_ready else "245,158,11"
        color = SUCCESS_COLOR if weather_ready else WARNING_COLOR
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Weather Dataset</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {status_text}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Environmental conditions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        status_icon = "✅" if calendar_ready else "⏳"
        status_text = "Ready" if calendar_ready else "Pending"
        rgb = "16,185,129" if calendar_ready else "245,158,11"
        color = SUCCESS_COLOR if calendar_ready else WARNING_COLOR
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Calendar Dataset</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {status_text}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Temporal features & holidays</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        status_icon = "✅" if reason_ready else "⏳"
        status_text = "Ready" if reason_ready else "Pending"
        rgb = "16,185,129" if reason_ready else "245,158,11"
        color = SUCCESS_COLOR if reason_ready else WARNING_COLOR
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; padding: 1rem;'>
              <div class='hf-feature-description' style='color: {SUBTLE_TEXT}; font-size: 0.8125rem; margin-bottom: 0.5rem;'>Reason for Visit</div>
              <div style='margin: 1rem 0;'>
                <span class='hf-pill' style='background:rgba({rgb},.15);color:{color};border:1px solid rgba({rgb},.3); font-size: 0.875rem; padding: 0.5rem 1rem;'>
                  {status_icon} {status_text}
                </span>
              </div>
              <div class='hf-feature-description' style='font-size: 0.75rem;'>Visit reasons by category</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Tabular Workflow Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Validate Sources",
        "2️⃣ Preview Merge Plan",
        "3️⃣ Run Fusion",
        "4️⃣ Preprocess Features",
        "5️⃣ Preprocessed Info"
    ])

    # TAB 1 — Validate Sources
    with tab1:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 2rem 0; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1)); border: 2px solid rgba(59, 130, 246, 0.3); box-shadow: 0 0 40px rgba(59, 130, 246, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.6));'>✓</div>
              <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);'>Source Validation</h1>
              <p style='font-size: 1.25rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Verify that all required datasets are loaded and ready for processing</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        if not all_ready:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.05)); border-radius: 16px; border: 1px solid rgba(245, 158, 11, 0.2);'>
                  <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>⚠️</div>
                  <div style='font-size: 1.125rem; font-weight: 700; color: {WARNING_COLOR}; margin-bottom: 0.5rem;'>Missing Datasets</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Please upload all four datasets in the Data Hub to proceed with data fusion</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05)); border-radius: 16px; border: 1px solid rgba(34, 197, 94, 0.2);'>
                  <div style='font-size: 3rem; margin-bottom: 1rem;'>✅</div>
                  <div style='font-size: 1.25rem; font-weight: 700; color: {SUCCESS_COLOR}; margin-bottom: 0.5rem;'>All Sources Ready</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>All datasets are loaded. Proceed to Preview Merge Plan</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if patient_ready:  _summarize_df(patient_df,  "Patient — sample & schema")
        if weather_ready:  _summarize_df(weather_df,  "Weather — sample & schema")
        if calendar_ready: _summarize_df(calendar_df, "Calendar — sample & schema")
        if reason_ready:   _summarize_df(reason_df,   "Reason for Visit — sample & schema")

    # TAB 2 — Preview Merge Plan
    with tab2:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 2rem 0; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(59, 130, 246, 0.1)); border: 2px solid rgba(34, 211, 238, 0.3); box-shadow: 0 0 40px rgba(34, 211, 238, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(34, 211, 238, 0.6));'>🔍</div>
              <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(34, 211, 238, 0.3);'>Merge Plan Preview</h1>
              <p style='font-size: 1.25rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Review how datasets will be joined based on detected time keys</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not all_ready:
            st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
            st.info("Upload all sources first to preview the merge plan.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            _merge_plan_preview(patient_df, weather_df, calendar_df, reason_df)
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            with st.expander("📖 What happens during fusion?", expanded=False):
                st.markdown(
                    """
                    <div class='hf-feature-card'>
                      <h4 style='margin-bottom: 1rem;'>Fusion Process Overview</h4>
                      <ul style='line-height: 1.8;'>
                        <li>Normalize time keys to <strong>datetime</strong> and <strong>Date</strong> columns</li>
                        <li>Intelligently join Patient, Weather, Calendar, and Reason for Visit datasets on time keys</li>
                        <li>Preserve all original columns and resolve duplicate column names</li>
                        <li>Validate row counts and check for missing values</li>
                        <li>Ensure temporal alignment across all data sources</li>
                      </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # TAB 3 — Run Fusion
    with tab3:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 2rem 0; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1)); border: 2px solid rgba(34, 197, 94, 0.3); box-shadow: 0 0 40px rgba(34, 197, 94, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(34, 197, 94, 0.6));'>🔄</div>
              <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, {SUCCESS_COLOR}, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(34, 197, 94, 0.3);'>Execute Data Fusion</h1>
              <p style='font-size: 1.25rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Merge all datasets into a unified time-aligned dataframe</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 1.5rem;'>
              <p style='font-size: 1rem; color: {BODY_TEXT};'>
                Ready to merge? This will combine Patient, Weather, Calendar, and Reason for Visit datasets on their time keys.
              </p>
            </div>
            """.format(BODY_TEXT=BODY_TEXT),
            unsafe_allow_html=True,
        )

        keys_ok = all([
            _detect_key(patient_df)  is not None if patient_ready  else False,
            _detect_key(weather_df)  is not None if weather_ready  else False,
            _detect_key(calendar_df) is not None if calendar_ready else False,
            _detect_key(reason_df)   is not None if reason_ready   else False
        ])

        if not all_ready:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05)); border-radius: 16px; border: 1px solid rgba(239, 68, 68, 0.2); margin-bottom: 1.5rem;'>
                  <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>❌</div>
                  <div style='font-size: 1.125rem; font-weight: 700; color: {DANGER_COLOR}; margin-bottom: 0.5rem;'>Missing Sources</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Upload all four datasets before running fusion</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not keys_ok:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.05)); border-radius: 16px; border: 1px solid rgba(245, 158, 11, 0.2); margin-bottom: 1.5rem;'>
                  <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>⚠️</div>
                  <div style='font-size: 1.125rem; font-weight: 700; color: {WARNING_COLOR}; margin-bottom: 0.5rem;'>Time Key Detection Warning</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Could not detect time keys in one or more datasets. Fusion may fail.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Diagnostic: Show Date ranges for debugging
        if all_ready:
            with st.expander("🔍 Date Range Diagnostic", expanded=False):
                st.markdown("**Date ranges in each dataset:**")

                def show_date_info(df, name):
                    time_col = _find_time_like_col(df)
                    if time_col:
                        try:
                            dates = pd.to_datetime(df[time_col], errors='coerce')
                            valid_dates = dates.dropna()
                            if len(valid_dates) > 0:
                                st.write(f"- **{name}**: {valid_dates.min().date()} to {valid_dates.max().date()} ({len(valid_dates):,} valid dates)")
                            else:
                                st.write(f"- **{name}**: ❌ No valid dates found in column '{time_col}'")
                        except Exception as e:
                            st.write(f"- **{name}**: ❌ Error parsing dates from '{time_col}': {e}")
                    else:
                        st.write(f"- **{name}**: ❌ No time column detected")

                show_date_info(patient_df, "Patient Dataset")
                show_date_info(weather_df, "Weather Dataset")
                show_date_info(calendar_df, "Calendar Dataset")
                show_date_info(reason_df, "Reason for Visit Dataset")

        do_merge = st.button("🔄 Merge Datasets Now", type="primary", use_container_width=True, disabled=not all_ready)

        if do_merge:
            prog = st.progress(0, text="Starting fusion…")
            try:
                prog.progress(25, text="Aligning time keys…")
                merged, fusion_log = _run_fusion_cached(patient_df, weather_df, calendar_df, reason_df)
                prog.progress(70, text="Validating fused frame…")
                st.session_state["merged_data"] = merged if merged is not None and not merged.empty else None

                # Display fusion log for diagnostics
                with st.expander("📋 Fusion Process Log", expanded=(st.session_state["merged_data"] is None)):
                    for log_msg in fusion_log:
                        if log_msg.startswith("✅"):
                            st.success(log_msg)
                        elif log_msg.startswith("❌"):
                            st.error(log_msg)
                        elif log_msg.startswith("⚠️"):
                            st.warning(log_msg)
                        elif log_msg.startswith("ℹ️"):
                            st.info(log_msg)
                        else:
                            st.write(log_msg)

                if st.session_state["merged_data"] is not None:
                    prog.progress(100, text="Fusion complete ✓")
                    st.markdown(
                        f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05)); border-radius: 16px; border: 1px solid rgba(34, 197, 94, 0.2); margin: 1.5rem 0;'>
                          <div style='font-size: 3rem; margin-bottom: 1rem;'>✅</div>
                          <div style='font-size: 1.25rem; font-weight: 700; color: {SUCCESS_COLOR}; margin-bottom: 0.5rem;'>Fusion Complete</div>
                          <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Successfully merged {len(st.session_state["merged_data"]):,} rows across all datasets</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Merged Dataset Preview (First 20 rows)**")
                    st.dataframe(st.session_state["merged_data"].head(20), use_container_width=True)
                else:
                    prog.empty()
                    st.error("❌ Data fusion returned an empty result. Check joins/time keys.")
                    st.warning("💡 **Tip**: Check the 'Fusion Process Log' above for detailed error messages, and expand the 'Date Range Diagnostic' to see if datasets have overlapping dates.")
            except Exception as e:
                prog.empty()
                st.error(f"❌ Fusion failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 4 — Preprocess (feature engineering)
    with tab4:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 2rem 0; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1)); border: 2px solid rgba(168, 85, 247, 0.3); box-shadow: 0 0 40px rgba(168, 85, 247, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(168, 85, 247, 0.6));'>🧪</div>
              <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #a855f7, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(168, 85, 247, 0.3);'>Feature Engineering</h1>
              <p style='font-size: 1.25rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Generate lag features, targets, and prepare data for model training</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        md = st.session_state.get("merged_data")
        if md is None or getattr(md, "empty", True):
            st.markdown(
                f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05)); border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.2);'>
                  <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>ℹ️</div>
                  <div style='font-size: 1.125rem; font-weight: 700; color: {PRIMARY_COLOR}; margin-bottom: 0.5rem;'>Fusion Required</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Run data fusion first to enable feature engineering</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='margin-bottom: 1.5rem;'>
                  <h3 style='font-size: 1.125rem; font-weight: 700; margin-bottom: 1rem;'>⚙️ Configuration</h3>
                  <p style='color: {BODY_TEXT}; font-size: 0.9375rem; margin-bottom: 1rem;'>
                    Configure how lag features and future targets are generated from the merged dataset
                  </p>
                </div>
                """.format(BODY_TEXT=BODY_TEXT),
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                n_lags = st.number_input("Lags/Targets (days)", min_value=1, max_value=30, value=7, step=1,
                                        help="Number of lag features (ED_1..ED_n) and targets (Target_1..Target_n) to create")
            with c2:
                strict_drop = st.checkbox(
                    "Drop edge rows (recommended)",
                    value=True,
                    help="Removes NaNs at the edges caused by lag/target shifts."
                )

            run = st.button("🧪 Build Features", type="primary", use_container_width=True)
            if run:
                with st.spinner("Engineering features, harmonising schema, and ordering columns…"):
                    try:
                        processed = _preprocess_after_fusion_cached(md, n_lags=n_lags, strict_drop=strict_drop)
                        st.session_state["processed_df"] = processed
                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05)); border-radius: 16px; border: 1px solid rgba(34, 197, 94, 0.2); margin: 1.5rem 0;'>
                              <div style='font-size: 3rem; margin-bottom: 1rem;'>✅</div>
                              <div style='font-size: 1.25rem; font-weight: 700; color: {SUCCESS_COLOR}; margin-bottom: 0.5rem;'>Feature Engineering Complete</div>
                              <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Generated {len(processed.columns)} features across {len(processed):,} rows</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {e}")

            proc = st.session_state.get("processed_df")
            if proc is not None and not getattr(proc, "empty", True):
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

                # Categorize columns for visual grouping
                def categorize_columns(df):
                    """Categorize columns into meaningful groups for ML training."""
                    cols = list(df.columns)

                    date_cols = []
                    calendar_cols = []
                    weather_cols = []
                    current_reason_cols = []
                    past_arrivals_cols = []
                    past_reason_cols = []
                    future_arrivals_cols = []
                    future_reason_cols = []
                    other_cols = []

                    # Known reason column names (granular medical categories)
                    reason_keywords = [
                        'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
                        'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
                        'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
                        'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
                    ]

                    for col in cols:
                        col_lower = col.lower()

                        # Date columns
                        if col_lower in ['date', 'datetime', 'ds', 'timestamp']:
                            date_cols.append(col)

                        # Calendar features
                        elif col_lower in ['day_of_week', 'holiday', 'holiday_prev', 'is_weekend',
                                          'day_of_month', 'week_of_year', 'month', 'year']:
                            calendar_cols.append(col)

                        # Weather features
                        elif col_lower in ['average_temp', 'max_temp', 'average_wind', 'max_wind',
                                          'average_mslp', 'total_precipitation', 'moon_phase']:
                            weather_cols.append(col)

                        # Past arrivals features (ED_1, ED_2, etc.)
                        elif col_lower.startswith('ed_') and len(col_lower) > 3 and col_lower[3:].isdigit():
                            past_arrivals_cols.append(col)

                        # Future arrivals features (Target_1, Target_2, etc.)
                        elif col_lower.startswith('target_'):
                            future_arrivals_cols.append(col)

                        # Past reason features (asthma_lag_1, pneumonia_lag_1, etc.)
                        elif '_lag_' in col_lower and any(keyword in col_lower for keyword in reason_keywords):
                            past_reason_cols.append(col)

                        # Future reason features (asthma_1, pneumonia_1, etc.)
                        elif '_' in col and col.split('_')[-1].isdigit() and any(keyword in col_lower for keyword in reason_keywords):
                            future_reason_cols.append(col)

                        # Current reason features (asthma, pneumonia, etc. without suffixes)
                        elif any(col_lower == keyword for keyword in reason_keywords):
                            current_reason_cols.append(col)

                        # Other columns
                        else:
                            other_cols.append(col)

                    return {
                        'Date': date_cols,
                        'Calendar Features': calendar_cols,
                        'Weather Features': weather_cols,
                        'Current Reason Columns': current_reason_cols,
                        'Past Features (Arrivals)': past_arrivals_cols,
                        'Past Features (Reasons)': past_reason_cols,
                        'Future Features (Arrivals)': future_arrivals_cols,
                        'Future Features (Reasons)': future_reason_cols,
                        'Other': other_cols
                    }

                # Get categorized columns
                categorized = categorize_columns(proc)

                # Display column grouping visualization
                st.markdown("**📊 Column Structure for Multi-Target ML Training**")
                st.markdown("<div style='margin-bottom: 1rem; color: #94A3B8; font-size: 0.875rem;'>Columns are organized to optimize learning patterns across patient arrivals and visit reasons</div>", unsafe_allow_html=True)

                # Create visual category display
                category_colors = {
                    'Date': '#3B82F6',  # Blue
                    'Calendar Features': '#8B5CF6',  # Purple
                    'Weather Features': '#10B981',  # Green
                    'Current Reason Columns': '#F59E0B',  # Amber
                    'Past Features (Arrivals)': '#06B6D4',  # Cyan
                    'Past Features (Reasons)': '#14B8A6',  # Teal
                    'Future Features (Arrivals)': '#EF4444',  # Red
                    'Future Features (Reasons)': '#EC4899',  # Pink
                    'Other': '#6B7280'  # Gray
                }

                category_icons = {
                    'Date': '📅',
                    'Calendar Features': '🗓️',
                    'Weather Features': '🌤️',
                    'Current Reason Columns': '🏥',
                    'Past Features (Arrivals)': '⏮️',
                    'Past Features (Reasons)': '📋',
                    'Future Features (Arrivals)': '🎯',
                    'Future Features (Reasons)': '🔮',
                    'Other': '📦'
                }

                # Display categories with columns
                for category, columns in categorized.items():
                    if columns:  # Only show categories that have columns
                        color = category_colors.get(category, '#6B7280')
                        icon = category_icons.get(category, '•')

                        with st.expander(f"{icon} **{category}** ({len(columns)} columns)", expanded=False):
                            # Create a nicely formatted display of columns
                            cols_display = ", ".join([f"`{col}`" for col in columns])
                            st.markdown(
                                f"""
                                <div style='padding: 0.75rem; background: rgba(255,255,255,0.02); border-left: 3px solid {color}; border-radius: 4px; margin: 0.5rem 0;'>
                                    <div style='color: {color}; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem;'>{category}</div>
                                    <div style='color: #CBD5E1; font-size: 0.8125rem; font-family: monospace;'>{cols_display}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Summary metrics
                total_past_features = len(categorized['Past Features (Arrivals)']) + len(categorized['Past Features (Reasons)'])
                total_future_features = len(categorized['Future Features (Arrivals)']) + len(categorized['Future Features (Reasons)'])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📅 Date & Calendar", len(categorized['Date']) + len(categorized['Calendar Features']))
                with col2:
                    st.metric("🌤️ Weather Features", len(categorized['Weather Features']))
                with col3:
                    st.metric("⏮️ Past Features (Lag)", total_past_features)
                with col4:
                    st.metric("🎯 Future Targets", total_future_features)

                # Dataset preview
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Dataset Preview (First 30 rows)**")
                st.dataframe(proc.head(30), use_container_width=True)

                # Download button
                csv = proc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=csv,
                    file_name="processed_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 5 — Preprocessed Info
    with tab5:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 2rem 0; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 191, 36, 0.1)); border: 2px solid rgba(245, 158, 11, 0.3); box-shadow: 0 0 40px rgba(245, 158, 11, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(245, 158, 11, 0.6));'>📊</div>
              <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, {WARNING_COLOR}, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(245, 158, 11, 0.3);'>Preprocessed Dataset Summary</h1>
              <p style='font-size: 1.25rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Comprehensive overview of the final feature-engineered dataset</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='hf-feature-card'>", unsafe_allow_html=True)
        proc = st.session_state.get("processed_df")
        if proc is not None and not getattr(proc, "empty", True):
            _render_preprocessed_info(proc)
        else:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 211, 238, 0.05)); border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.2);'>
                  <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>ℹ️</div>
                  <div style='font-size: 1.125rem; font-weight: 700; color: {PRIMARY_COLOR}; margin-bottom: 0.5rem;'>No Preprocessed Data</div>
                  <div style='color: {BODY_TEXT}; font-size: 0.9375rem;'>Run feature engineering first to view the dataset summary</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


init_state()
page_data_preparation_studio()


