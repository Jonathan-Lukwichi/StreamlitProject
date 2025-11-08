# =============================================================================
# pages/03_Data_Preparation_Studio_enhanced.py — Data Preparation Studio (Pro)
# Matches the clean template used across your app:
#  • Hero strip
#  • Compact status badges / section cards
#  • Sidebar stepper preserved
#  • Defensive fallbacks if app_core modules are absent
# =============================================================================
from __future__ import annotations

import re
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

# ---------- THEME / FRAMEWORK FALLBACKS ---------------------------------------
try:
    from app_core.ui.theme import apply_css
    from app_core.ui.theme import (
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
        DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
    )
except Exception:
    PRIMARY_COLOR   = "#2563eb"
    SECONDARY_COLOR = "#8b5cf6"
    SUCCESS_COLOR   = "#10b981"
    WARNING_COLOR   = "#f59e0b"
    DANGER_COLOR    = "#ef4444"
    TEXT_COLOR      = "#0f172a"
    SUBTLE_TEXT     = "#64748b"

    def apply_css():
        # Minimal base so visuals aren’t broken if theme module is missing
        st.markdown(
            """
            <style>
            .muted {color:#6b7280}
            .section-card {border:1px solid #e5e7eb;border-radius:16px;padding:14px;background:#fff;}
            </style>
            """,
            unsafe_allow_html=True,
        )

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        for k, v in {
            "patient_data": None,
            "weather_data": None,
            "calendar_data": None,
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
    def fuse_data(patient_df, weather_df, calendar_df):
        # naive outer-merge fallback on best-effort datetime/date keys
        def _key(df):
            for c in ["datetime", "Date", "date", "timestamp", "ds", "time"]:
                if c in df.columns: return c
            return None
        p, w, c = patient_df.copy(), weather_df.copy(), calendar_df.copy()
        for df in (p, w, c):
            for k in ["datetime","Date","date","timestamp","ds","time"]:
                if k in df.columns:
                    df[k] = pd.to_datetime(df[k], errors="coerce")
        pk, wk, ck = _key(p), _key(w), _key(c)
        merged = p
        if w is not None and wk: merged = merged.merge(w, left_on=pk, right_on=wk, how="outer")
        if c is not None and ck: merged = merged.merge(c, left_on=pk or wk, right_on=ck, how="outer")
        return merged, {}

try:
    from app_core.data import process_dataset
except Exception:
    def process_dataset(df: pd.DataFrame, n_lags: int, date_col=None, ed_col=None, strict_drop_na_edges=True):
        # basic placeholder: add ED_1..ED_n and Target_1..Target_n from first numeric col
        work = df.copy()
        y = None
        for c in work.columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                y = c; break
        if y:
            for i in range(1, n_lags+1):
                work[f"ED_{i}"] = work[y].shift(i)
                work[f"Target_{i}"] = work[y].shift(-i)
        if strict_drop_na_edges:
            work = work.dropna(how="any")
        return work, {}

# ---------- PAGE CONFIG --------------------------------------------------------
st.set_page_config(
    page_title="Data Preparation Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- LOCAL CSS to match template ---------------------------------------
def _prep_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*='css'] {{ font-family: 'Inter', system-ui, sans-serif; }}

        .hero {{
            border-radius: 20px; padding: 28px; color: #0b1220;
            background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(99,102,241,.08), rgba(20,184,166,.12));
            border: 1px solid rgba(37,99,235,.25);
            box-shadow: 0 20px 25px -5px rgba(0,0,0,.08), 0 10px 10px -5px rgba(0,0,0,.04);
            position: relative; overflow: hidden; margin-bottom: 18px;
        }}
        .hero-title {{
            font-size: 1.9rem; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.02em;
            background: linear-gradient(135deg, {TEXT_COLOR}, {PRIMARY_COLOR});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .hero-sub {{ color: {SUBTLE_TEXT}; font-size: 1.0rem; margin: 0; }}

        .section-card {{ border:1px solid #e5e7eb; border-radius:16px; padding:14px; background:#fff; }}
        .section-title {{ font-weight:800; font-size:1.05rem; margin-bottom:8px; color:{TEXT_COLOR}; }}

        .badge {{
            border:1px solid #e9ecef; border-radius:10px; padding:.55rem .8rem;
            margin-bottom:.4rem; display:flex; align-items:center; gap:.5rem;
        }}
        .ok {{ color:{SUCCESS_COLOR}; font-weight:700; }}
        .warn {{ color:{WARNING_COLOR}; font-weight:700; }}
        .fail {{ color:{DANGER_COLOR}; font-weight:700; }}
        </style>
        """,
        unsafe_allow_html=True,
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
    targets = [c for c in columns if re.fullmatch(r"Target_\d+", c)]
    return sorted(targets, key=lambda x: int(x.split("_")[1]))

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
def _run_fusion_cached(patient_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame):
    merged, _ = fuse_data(patient_df, weather_df, calendar_df)
    return merged

@st.cache_data(show_spinner=False)
def _preprocess_after_fusion_cached(merged_df: pd.DataFrame, n_lags: int, strict_drop: bool):
    processed_df, _ = process_dataset(
        merged_df, n_lags=n_lags, date_col=None, ed_col=None, strict_drop_na_edges=strict_drop
    )
    processed_df = _ensure_datetime_and_date(processed_df)
    processed_df = _harmonize_day_of_week_and_ed(processed_df)
    processed_df = order_columns_pipeline(processed_df)
    return processed_df

# ---------- ANALYST-STYLE REPORTING --------------------------------------------
def _render_preprocessed_info(proc: pd.DataFrame):
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

    cols = list(proc.columns)
    time_cols = _detect_time_columns(cols)
    cal_cols  = _detect_calendar_columns([c for c in cols if c not in time_cols])
    met_cols  = _detect_weather_columns([c for c in cols if c not in time_cols + cal_cols])
    ed_lags   = _detect_ed_columns([c for c in cols if c not in time_cols + cal_cols + met_cols])
    tgt_cols  = _detect_target_columns([c for c in cols if c not in time_cols + cal_cols + met_cols + ed_lags])

    miss = proc.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(8)

    st.markdown("**Dataset Overview**")
    st.write(f"- Shape: **{n_rows:,} rows × {n_cols} columns**")
    if min_dt is not None and max_dt is not None:
        st.write(f"- Time coverage: **{min_dt.date()} → {max_dt.date()}**")

    st.markdown("**Column Families**")
    st.write(f"- Time: {', '.join(time_cols) if time_cols else '—'}")
    st.write(f"- Calendar: {', '.join(cal_cols) if cal_cols else '—'}")
    st.write(f"- Weather: {', '.join(met_cols) if met_cols else '—'}")
    st.write(f"- ED features: {', '.join(ed_lags) if ed_lags else '—'}")
    st.write(f"- Targets: {', '.join(tgt_cols) if tgt_cols else '—'}")

    st.markdown("**Data Quality (quick look)**")
    if len(miss) > 0:
        bullets = [f"- {c}: {int(miss[c])} missing" for c in miss.index]
        st.write("\n".join(bullets))
    else:
        st.write("- No major missingness detected (top columns).")

# ---------- UI HELPERS ----------------------------------------------------------
def _dataset_badge(ok: bool, label: str):
    cls = "ok" if ok else "fail"
    icon = "✅" if ok else "❌"
    st.markdown(
        f"<div class='badge'><span>{icon}</span><span class='{cls}'>{label}</span></div>",
        unsafe_allow_html=True,
    )

def _detect_key(df: pd.DataFrame) -> str | None:
    return _find_time_like_col(df) if df is not None and not df.empty else None

def _merge_plan_preview(patient_df, weather_df, calendar_df):
    pk = _detect_key(patient_df)
    wk = _detect_key(weather_df)
    ck = _detect_key(calendar_df)

    st.markdown("<div class='section-title'>Merge Keys Preview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.write(f"- Patient: `{pk or '—'}`")
    st.write(f"- Weather: `{wk or '—'}`")
    st.write(f"- Calendar: `{ck or '—'}`")
    st.caption("Keys are auto-normalized to `datetime` and `Date` during processing.")
    try:
        import graphviz
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR", nodesep="0.4", splines="spline")
        dot.node("P", f"Patient\nkey: {pk or '—'}", shape="box")
        dot.node("W", f"Weather\nkey: {wk or '—'}", shape="box")
        dot.node("C", f"Calendar\nkey: {ck or '—'}", shape="box")
        dot.node("M", "Merged\nData", shape="ellipse")
        dot.edges([("P","M"), ("W","M"), ("C","M")])
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:
        st.caption("Graphviz not available; skipping diagram.")
    st.markdown("</div>", unsafe_allow_html=True)

def _summarize_df(df: pd.DataFrame, title: str):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
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
    # HERO
    st.markdown(
        f"""
        <div class='hero'>
          <div class='hero-title'>🧠 Data Preparation Studio</div>
          <p class='hero-sub'>Upload sources → preview plan → merge → engineer features → validate</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pull uploaded sources
    patient_df  = st.session_state.get("patient_data")
    weather_df  = st.session_state.get("weather_data")
    calendar_df = st.session_state.get("calendar_data")

    patient_ready  = patient_df is not None and not getattr(patient_df, "empty", True)
    weather_ready  = weather_df is not None and not getattr(weather_df, "empty", True)
    calendar_ready = calendar_df is not None and not getattr(calendar_df, "empty", True)
    all_ready = patient_ready and weather_ready and calendar_ready

    # Top status badges
    col1, col2, col3 = st.columns(3)
    with col1: _dataset_badge(patient_ready, "Patient dataset")
    with col2: _dataset_badge(weather_ready, "Weather dataset")
    with col3: _dataset_badge(calendar_ready, "Calendar dataset")

    st.write("")

    # Sidebar workflow (kept)
    st.sidebar.markdown("### Workflow")
    step = st.sidebar.radio(
        label="Steps",
        options=[
            "1) Validate Sources",
            "2) Preview Merge Plan",
            "3) Run Fusion",
            "4) Preprocess Features",
            "5) Preprocessed Info",
        ],
        index=0,
    )

    # STEP 1 — Validate Sources
    if step == "1) Validate Sources":
        st.markdown("<div class='section-title'>Source Validation</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if not all_ready:
            st.warning("Please upload all three datasets in **Data Hub** to proceed.")
        else:
            st.success("All sources are loaded. Move to *Preview Merge Plan*.")
        st.markdown("</div>", unsafe_allow_html=True)

        if patient_ready:  _summarize_df(patient_df,  "Patient — sample & schema")
        if weather_ready:  _summarize_df(weather_df,  "Weather — sample & schema")
        if calendar_ready: _summarize_df(calendar_df, "Calendar — sample & schema")

    # STEP 2 — Preview Merge Plan
    if step == "2) Preview Merge Plan":
        if not all_ready:
            st.info("Upload all sources first.")
        else:
            _merge_plan_preview(patient_df, weather_df, calendar_df)
            with st.expander("What happens during fusion?", expanded=False):
                st.markdown(
                    "- Normalize time keys to **datetime** and **Date**\n"
                    "- Smart-join Patient, Weather, Calendar on time keys\n"
                    "- Preserve original columns; resolve duplicates\n"
                    "- Validate row count & missingness"
                )

    # STEP 3 — Run Fusion
    if step == "3) Run Fusion":
        st.markdown("<div class='section-title'>Execute Fusion</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.write("Ready to merge? This will combine Patient, Weather, and Calendar datasets on their time keys.")

        keys_ok = all([
            _detect_key(patient_df)  is not None if patient_ready  else False,
            _detect_key(weather_df)  is not None if weather_ready  else False,
            _detect_key(calendar_df) is not None if calendar_ready else False
        ])
        if not all_ready:
            st.error("Missing sources. Upload all three datasets.")
        elif not keys_ok:
            st.warning("Could not detect time keys in one or more datasets. Fusion may fail.")

        do_merge = st.button("🔄 Merge datasets now", type="primary", use_container_width=False, disabled=not all_ready)

        if do_merge:
            prog = st.progress(0, text="Starting fusion…")
            try:
                prog.progress(25, text="Aligning time keys…")
                merged = _run_fusion_cached(patient_df, weather_df, calendar_df)
                prog.progress(70, text="Validating fused frame…")
                st.session_state["merged_data"] = merged if merged is not None and not merged.empty else None

                if st.session_state["merged_data"] is not None:
                    prog.progress(100, text="Fusion complete ✓")
                    st.success("✅ Data fusion completed successfully!")
                    st.dataframe(st.session_state["merged_data"].head(20), use_container_width=True)
                else:
                    prog.empty()
                    st.error("❌ Data fusion returned an empty result. Check joins/time keys.")
            except Exception as e:
                prog.empty()
                st.error(f"❌ Fusion failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 4 — Preprocess (feature engineering)
    if step == "4) Preprocess Features":
        st.markdown("<div class='section-title'>Preprocess (Feature Engineering)</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        md = st.session_state.get("merged_data")
        if md is None or getattr(md, "empty", True):
            st.info("Run fusion first to enable preprocessing.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                n_lags = st.number_input("Lags/Targets (days)", min_value=1, max_value=30, value=7, step=1)
            with c2:
                strict_drop = st.checkbox(
                    "Drop edge rows (recommended)",
                    value=True,
                    help="Removes NaNs at the edges caused by lag/target shifts."
                )

            run = st.button("🧪 Build features", type="primary")
            if run:
                with st.spinner("Engineering features, harmonising schema, and ordering columns…"):
                    try:
                        processed = _preprocess_after_fusion_cached(md, n_lags=n_lags, strict_drop=strict_drop)
                        st.session_state["processed_df"] = processed
                        st.success("✅ Preprocessing complete!")
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {e}")

            proc = st.session_state.get("processed_df")
            if proc is not None and not getattr(proc, "empty", True):
                st.markdown("#### Processed Dataset (First 30 rows)")
                st.dataframe(proc.head(30), use_container_width=True)
                csv = proc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download processed CSV",
                    data=csv,
                    file_name="processed_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 5 — Preprocessed Info
    if step == "5) Preprocessed Info":
        st.markdown("<div class='section-title'>Preprocessed Data Info</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        proc = st.session_state.get("processed_df")
        if proc is not None and not getattr(proc, "empty", True):
            _render_preprocessed_info(proc)
        else:
            st.info("Run preprocessing first to view the preprocessed dataset info.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- ENTRYPOINT ----------------------------------------------------------
def main():
    apply_css()
    _prep_css()
    init_state()
    header("Data Preparation Studio", "Upload sources → preview plan → merge → engineer features → validate", icon="🧠")
    page_data_preparation_studio()

if __name__ == "__main__":
    main()
