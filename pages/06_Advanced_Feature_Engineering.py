# =============================================================================
# pages/06_Advanced_Feature_Engineering.py — Advanced Feature Engineering (Flexible, Deduplicated)
# =============================================================================
from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
except Exception:
    OneHotEncoder = None
    MinMaxScaler = None

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
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
        st.session_state.setdefault("feature_engineering", {})

try:
    from app_core.ui.components import header
except Exception:
    def header(title: str, subtitle: str = "", icon: str = ""):
        st.markdown(f"### {icon} {title}")
        if subtitle: st.caption(subtitle)

st.set_page_config(
    page_title="Advanced Feature Engineering - HealthForecast AI",
    page_icon="🧪",
    layout="wide",
)

# === HELPERS ==================================================================
DATE_PREFS = ["datetime", "Date", "date", "timestamp", "ds", "time"]
HOLIDAY_PREFS = ["is_holiday","Is_holiday","Holiday","holiday","IsHoliday"]

def _get_processed() -> Optional[pd.DataFrame]:
    for k in ["processed_df", "data_final", "prepared_data", "preprocessed_data"]:
        df = st.session_state.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None

def _guess_date_col(df: pd.DataFrame) -> str:
    for c in DATE_PREFS:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["date","time","stamp"]):
            return c
    return df.columns[0]

def _guess_targets(df: pd.DataFrame) -> List[str]:
    t = [c for c in df.columns if c.lower().startswith("target_")]
    if t: return sorted(t, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    alt = [c for c in df.columns if c.lower() in ("target","y","label")]
    return alt

def _ensure_dt_sorted(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df

def _holiday_series(df: pd.DataFrame) -> pd.Series:
    for c in HOLIDAY_PREFS:
        if c in df.columns:
            s = df[c]
            if s.dtype == bool: return s.astype(int)
            return s.astype(str).str.lower().isin(["1","true","yes","y"]).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def _calendar_frame(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    cal = pd.DataFrame(index=df.index)
    cal["year"] = s.dt.year
    cal["quarter"] = s.dt.quarter
    cal["month"] = s.dt.month
    cal["week_of_year"] = s.dt.isocalendar().week.astype(int)
    cal["day_of_year"] = s.dt.dayofyear
    cal["day_of_month"] = s.dt.day
    cal["day_of_week"] = s.dt.weekday
    cal["is_weekend"] = cal["day_of_week"].isin([5,6]).astype(int)
    cal["is_holiday"] = _holiday_series(df)
    cal["is_month_end"] = s.dt.is_month_end.astype(int)
    cal["is_quarter_end"] = s.dt.is_quarter_end.astype(int)
    cal["is_year_end"] = s.dt.is_year_end.astype(int)
    return cal

def _weekly_harmonics(day_of_week: pd.Series, orders: List[int] = [1,2]) -> pd.DataFrame:
    out = pd.DataFrame(index=day_of_week.index)
    x = day_of_week.astype(float).to_numpy()
    for k in orders:
        out[f"dow_sin_k{k}"] = np.sin(2*np.pi*k*x/7.0)
        out[f"dow_cos_k{k}"] = np.cos(2*np.pi*k*x/7.0)
    return out

def _cyclical_block(df: pd.DataFrame, date_col: str, cal: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    def _sin_cos(series: pd.Series, period: int, name: str):
        x = series.astype(float).to_numpy()
        out[f"{name}_sin"] = np.sin(2*np.pi*x/period)
        out[f"{name}_cos"] = np.cos(2*np.pi*x/period)
    _sin_cos(cal["month"], 12, "month")
    _sin_cos(cal["day_of_week"], 7, "dow")
    _sin_cos(cal["day_of_month"], 31, "dom")
    _sin_cos(cal["day_of_year"], 365, "doy")
    out = pd.concat([out, _weekly_harmonics(cal["day_of_week"], [1,2])], axis=1)
    for b in ["is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"]:
        out[b] = cal[b]
    out["week_of_year"] = cal["week_of_year"]
    return out

def _make_ohe():
    params = set(inspect.signature(OneHotEncoder).parameters.keys())
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if "sparse" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    return OneHotEncoder(handle_unknown="ignore")

def _safe_concat(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two dataframes safely, dropping duplicate columns."""
    common = [c for c in extra.columns if c in base.columns]
    if common:
        extra = extra.drop(columns=common)
    out = pd.concat([base, extra], axis=1)
    # Final deduplication safeguard
    out = out.loc[:, ~out.columns.duplicated()]
    return out

def _split_indices(df: pd.DataFrame, date_col: str, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    n_tr = int(np.floor(n*ratio))
    return np.arange(n_tr), np.arange(n_tr, n)

# === PIPELINES ================================================================
def _build_variant_A(base, date_col, targets, train_idx, test_idx, apply_scaling=True):
    cal = _calendar_frame(base, date_col)
    cat_df = cal[["month","quarter","day_of_week","year"]]
    ohe = _make_ohe()
    ohe.fit(cat_df.iloc[train_idx])
    names = ohe.get_feature_names_out(["month","quarter","day_of_week","year"])
    arr = ohe.transform(cat_df)
    if hasattr(arr, "toarray"): arr = arr.toarray()
    ohe_df = pd.DataFrame(arr, columns=names, index=base.index)
    keep = ["week_of_year","day_of_year","is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"]
    cal_keep = cal[keep]
    A = _safe_concat(base, cal_keep)
    A = _safe_concat(A, ohe_df)
    if apply_scaling and MinMaxScaler is not None:
        exclude = set([date_col] + targets + [c for c in A.columns if c.startswith("ED_")])
        exclude.update({"is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"})
        num_cols = A.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in num_cols if c not in exclude]
        if scale_cols:
            scaler = MinMaxScaler().fit(A.loc[train_idx, scale_cols])
            A.loc[:, scale_cols] = scaler.transform(A.loc[:, scale_cols])
    A = A.loc[:, ~A.columns.duplicated()]
    return A

def _build_variant_B(base, date_col, targets, train_idx, test_idx, apply_scaling=True):
    cal = _calendar_frame(base, date_col)
    cyc = _cyclical_block(base, date_col, cal)
    B = _safe_concat(base, cyc)
    if apply_scaling and MinMaxScaler is not None:
        exclude = set([date_col] + targets + [c for c in B.columns if c.startswith("ED_")])
        exclude.update({"is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"})
        num_cols = B.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in num_cols if c not in exclude]
        if scale_cols:
            scaler = MinMaxScaler().fit(B.loc[train_idx, scale_cols])
            B.loc[:, scale_cols] = scaler.transform(B.loc[:, scale_cols])
    B = B.loc[:, ~B.columns.duplicated()]
    return B

# === PAGE BODY ================================================================
def page_feature_engineering():

    base = _get_processed()
    if base is None or base.empty:
        st.warning("⚠️ No processed dataset found. Run Data Preparation Studio first.")
        return

    date_col = _guess_date_col(base)
    targets = _guess_targets(base)
    base = _ensure_dt_sorted(base, date_col)

    c1, c2, c3 = st.columns(3)
    with c1:
        skip_engineering = st.checkbox("Skip Feature Engineering", value=False)
    with c2:
        apply_scaling = st.checkbox("Apply Min–Max Scaling", value=True)
    with c3:
        ratio = st.slider("Train ratio", 0.5, 0.95, 0.8, 0.05)

    train_idx, test_idx = _split_indices(base, date_col, ratio)

    run = st.button("🚀 Generate Dataset", type="primary")
    if not run:
        st.dataframe(base.head(10), use_container_width=True)
        return

    with st.spinner("Processing..."):
        if skip_engineering:
            A_full = base.copy()
            B_full = base.copy()
            variant_used = "Original dataset (no feature engineering)"
        else:
            A_full = _build_variant_A(base, date_col, targets, train_idx, test_idx, apply_scaling)
            B_full = _build_variant_B(base, date_col, targets, train_idx, test_idx, apply_scaling)
            variant_used = "Feature engineered (deduplicated)"

    st.session_state["feature_engineering"] = {
        "summary": {"variant_used": variant_used, "apply_scaling": apply_scaling,
                    "skip_engineering": skip_engineering, "train_size": len(train_idx), "test_size": len(test_idx)},
        "A": A_full, "B": B_full, "train_idx": train_idx, "test_idx": test_idx
    }

    st.success(f"✅ Dataset ready: {variant_used}")
    tab1, tab2 = st.tabs(["Variant A (Decomp)", "Variant B (Cyclical)"])
    with tab1:
        st.dataframe(A_full.head(20), use_container_width=True)
        st.download_button("⬇️ Download Variant A", A_full.to_csv(index=False).encode(), "variant_A.csv", "text/csv")
    with tab2:
        st.dataframe(B_full.head(20), use_container_width=True)
        st.download_button("⬇️ Download Variant B", B_full.to_csv(index=False).encode(), "variant_B.csv", "text/csv")

# === ENTRYPOINT ===============================================================
def main():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    init_state()

    # Fluorescent effects
    st.markdown("""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR ADVANCED FEATURE ENGINEERING
       ======================================== */

    @keyframes float-orb {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.25;
        }
        50% {
            transform: translate(30px, -30px) scale(1.05);
            opacity: 0.35;
        }
    }

    .fluorescent-orb {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(70px);
    }

    .orb-1 {
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
        top: 15%;
        right: 20%;
        animation: float-orb 25s ease-in-out infinite;
    }

    .orb-2 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
        bottom: 20%;
        left: 15%;
        animation: float-orb 30s ease-in-out infinite;
        animation-delay: 5s;
    }

    @keyframes sparkle {
        0%, 100% {
            opacity: 0;
            transform: scale(0);
        }
        50% {
            opacity: 0.6;
            transform: scale(1);
        }
    }

    .sparkle {
        position: fixed;
        width: 3px;
        height: 3px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
        border-radius: 50%;
        pointer-events: none;
        z-index: 2;
        animation: sparkle 3s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }

    .sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
    .sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
    .sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

    @media (max-width: 768px) {
        .fluorescent-orb {
            width: 200px !important;
            height: 200px !important;
            filter: blur(50px);
        }
        .sparkle {
            display: none;
        }
    }
    </style>

    <!-- Fluorescent Floating Orbs -->
    <div class="fluorescent-orb orb-1"></div>
    <div class="fluorescent-orb orb-2"></div>

    <!-- Sparkle Particles -->
    <div class="sparkle sparkle-1"></div>
    <div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div>
    """, unsafe_allow_html=True)

    # Premium Hero Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧪</div>
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Advanced Feature Engineering</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Create sophisticated temporal and cyclical features with automated deduplication and scaling for enhanced model performance
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page_feature_engineering()

if __name__ == "__main__":
    main()
