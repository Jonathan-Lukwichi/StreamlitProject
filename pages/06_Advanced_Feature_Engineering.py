# =============================================================================
# pages/06_Advanced_Feature_Engineering.py — Advanced Feature Engineering (Flexible, Deduplicated)
# =============================================================================
from __future__ import annotations
import inspect
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
except Exception:
    OneHotEncoder = None
    MinMaxScaler = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# Import temporal split module
try:
    from app_core.pipelines import (
        compute_temporal_split,
        validate_temporal_split,
        TemporalSplitResult,
    )
    TEMPORAL_SPLIT_AVAILABLE = True
except ImportError:
    TEMPORAL_SPLIT_AVAILABLE = False

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

# Legacy function (kept for fallback if temporal_split not available)
def _split_indices_legacy(df: pd.DataFrame, date_col: str, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy ratio-based split (fallback only)."""
    n = len(df)
    n_tr = int(np.floor(n*ratio))
    return np.arange(n_tr), np.arange(n_tr, n)


def _get_temporal_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[TemporalSplitResult]]:
    """
    Compute temporal split using the academically-rigorous pipeline module.

    Returns:
        Tuple of (train_idx, cal_idx, test_idx, split_result)
        If temporal_split module not available, cal_idx will be empty.
    """
    if TEMPORAL_SPLIT_AVAILABLE:
        split_result = compute_temporal_split(
            df=df,
            date_col=date_col,
            train_ratio=train_ratio,
            cal_ratio=cal_ratio,
            verbose=False
        )
        return (
            split_result.train_idx,
            split_result.cal_idx,
            split_result.test_idx,
            split_result
        )
    else:
        # Fallback to legacy 2-way split
        train_idx, test_idx = _split_indices_legacy(df, date_col, train_ratio + cal_ratio)
        return train_idx, np.array([], dtype=int), test_idx, None


# ============================================================================
# TRANSFORMER PERSISTENCE (Academic Best Practice: Save scalers for inference)
# ============================================================================
TRANSFORMERS_DIR = Path("app_core/pipelines/saved_transformers")

def _ensure_transformers_dir():
    """Create directory for saving transformers if it doesn't exist."""
    if not TRANSFORMERS_DIR.exists():
        TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)

def _save_transformer(transformer, name: str, variant: str = "A") -> Optional[str]:
    """
    Save a fitted transformer (scaler/encoder) for later inference.

    Academic Reference:
        Prevents data leakage by ensuring production uses the SAME transformer
        fitted on training data only.
    """
    if not JOBLIB_AVAILABLE:
        return None

    _ensure_transformers_dir()
    filepath = TRANSFORMERS_DIR / f"{name}_{variant}.joblib"
    try:
        joblib.dump(transformer, filepath)
        return str(filepath)
    except Exception as e:
        st.warning(f"Could not save transformer {name}: {e}")
        return None

def _load_transformer(name: str, variant: str = "A"):
    """Load a previously saved transformer."""
    if not JOBLIB_AVAILABLE:
        return None

    filepath = TRANSFORMERS_DIR / f"{name}_{variant}.joblib"
    if filepath.exists():
        return joblib.load(filepath)
    return None

# === PIPELINES ================================================================
def _build_variant_A(base, date_col, targets, train_idx, cal_idx, test_idx, apply_scaling=True, save_transformers=True):
    """
    Build Variant A: One-Hot Encoded categorical calendar features.

    Academic Notes:
        - OneHotEncoder fitted ONLY on train_idx (prevents data leakage)
        - MinMaxScaler fitted ONLY on train_idx
        - Transformers saved for inference consistency
    """
    cal = _calendar_frame(base, date_col)
    cat_df = cal[["month","quarter","day_of_week","year"]]

    # Fit OHE on training data ONLY
    ohe = _make_ohe()
    ohe.fit(cat_df.iloc[train_idx])

    # Save OHE for later inference
    if save_transformers:
        _save_transformer(ohe, "ohe", variant="A")

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
            # Fit scaler on training data ONLY (Academic Best Practice)
            scaler = MinMaxScaler().fit(A.loc[train_idx, scale_cols])

            # Save scaler for later inference
            if save_transformers:
                _save_transformer(scaler, "scaler", variant="A")
                # Also save the column names for reference
                _save_transformer(scale_cols, "scale_cols", variant="A")

            # Transform ALL data (train, cal, test) using train-fitted scaler
            A.loc[:, scale_cols] = scaler.transform(A.loc[:, scale_cols])

    A = A.loc[:, ~A.columns.duplicated()]
    return A


def _build_variant_B(base, date_col, targets, train_idx, cal_idx, test_idx, apply_scaling=True, save_transformers=True):
    """
    Build Variant B: Cyclical (sin/cos) encoded calendar features.

    Academic Notes:
        - Cyclical encoding doesn't require fitting (mathematical transform)
        - MinMaxScaler fitted ONLY on train_idx
        - Transformer saved for inference consistency
    """
    cal = _calendar_frame(base, date_col)
    cyc = _cyclical_block(base, date_col, cal)
    B = _safe_concat(base, cyc)

    if apply_scaling and MinMaxScaler is not None:
        exclude = set([date_col] + targets + [c for c in B.columns if c.startswith("ED_")])
        exclude.update({"is_weekend","is_holiday","is_month_end","is_quarter_end","is_year_end"})
        num_cols = B.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in num_cols if c not in exclude]
        if scale_cols:
            # Fit scaler on training data ONLY (Academic Best Practice)
            scaler = MinMaxScaler().fit(B.loc[train_idx, scale_cols])

            # Save scaler for later inference
            if save_transformers:
                _save_transformer(scaler, "scaler", variant="B")
                _save_transformer(scale_cols, "scale_cols", variant="B")

            # Transform ALL data using train-fitted scaler
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

    # =========================================================================
    # FEATURE ENGINEERING TABS
    # Academic Reference: Bergmeir & Benítez (2012), Tashman (2000)
    # =========================================================================
    tab_split, tab_options, tab_generate = st.tabs([
        "📊 Data Split",
        "🧪 Options",
        "🚀 Generate"
    ])

    with tab_split:
        st.markdown("#### Configure Temporal Data Split")

        if TEMPORAL_SPLIT_AVAILABLE:
            col_split1, col_split2 = st.columns(2)
            with col_split1:
                train_ratio = st.slider(
                    "Training Set Ratio",
                    min_value=0.50,
                    max_value=0.80,
                    value=0.70,
                    step=0.05,
                    help="Proportion of data for training (recommended: 70%)"
                )
            with col_split2:
                cal_ratio = st.slider(
                    "Calibration Set Ratio",
                    min_value=0.05,
                    max_value=0.25,
                    value=0.15,
                    step=0.05,
                    help="Proportion for calibration (needed for Conformal Prediction)"
                )

            # Compute test ratio
            test_ratio = 1.0 - train_ratio - cal_ratio
            if test_ratio < 0.05:
                st.warning(f"⚠️ Test set too small ({test_ratio:.0%}). Reduce train or calibration ratio.")
                test_ratio = 0.05

            # Show split preview
            min_date = base[date_col].min()
            max_date = base[date_col].max()
            total_days = (max_date - min_date).days

            from datetime import timedelta
            train_end = min_date + timedelta(days=int(total_days * train_ratio))
            cal_end = min_date + timedelta(days=int(total_days * (train_ratio + cal_ratio)))

            col_prev1, col_prev2, col_prev3 = st.columns(3)
            with col_prev1:
                st.metric("Train", f"{train_ratio:.0%}", f"{min_date.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
            with col_prev2:
                st.metric("Calibration", f"{cal_ratio:.0%}", f"{train_end.strftime('%Y-%m-%d')} → {cal_end.strftime('%Y-%m-%d')}")
            with col_prev3:
                st.metric("Test", f"{test_ratio:.0%}", f"{cal_end.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")

            # Compute temporal split
            train_idx, cal_idx, test_idx, split_result = _get_temporal_split(
                base, date_col, train_ratio, cal_ratio
            )

        else:
            st.warning("⚠️ Temporal split module not available. Using legacy ratio-based split.")
            train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.8, 0.05)
            train_idx, test_idx = _split_indices_legacy(base, date_col, train_ratio)
            cal_idx = np.array([], dtype=int)
            split_result = None

    with tab_options:
        st.markdown("#### Feature Engineering Options")

        c1, c2, c3 = st.columns(3)
        with c1:
            skip_engineering = st.checkbox("Skip Feature Engineering", value=False)
        with c2:
            apply_scaling = st.checkbox("Apply Min–Max Scaling", value=True)
        with c3:
            save_transformers = st.checkbox("Save Transformers", value=True,
                                            help="Save scalers/encoders for production inference")

    with tab_generate:
        st.markdown("#### Generate Dataset")

        run = st.button("🚀 Generate Dataset", type="primary")
        if not run:
            st.dataframe(base.head(10), use_container_width=True)
            return

        with st.spinner("Processing features..."):
            if skip_engineering:
                A_full = base.copy()
                B_full = base.copy()
                variant_used = "Original dataset (no feature engineering)"
            else:
                A_full = _build_variant_A(base, date_col, targets, train_idx, cal_idx, test_idx,
                                          apply_scaling, save_transformers)
                B_full = _build_variant_B(base, date_col, targets, train_idx, cal_idx, test_idx,
                                          apply_scaling, save_transformers)
                variant_used = "Feature engineered (deduplicated)"

            # Validate temporal split if available
            if TEMPORAL_SPLIT_AVAILABLE and split_result is not None:
                validation_result = validate_temporal_split(
                    df=base,
                    date_col=date_col,
                    split_result=split_result
                )
                if not validation_result["valid"]:
                    st.error(f"⚠️ Temporal split validation failed: {validation_result['issues']}")
                if validation_result.get("warnings"):
                    for warning in validation_result["warnings"]:
                        st.warning(warning)

        # =========================================================================
        # STORE IN SESSION STATE (with cal_idx for Conformal Prediction)
        # =========================================================================
        st.session_state["feature_engineering"] = {
            "summary": {
                "variant_used": variant_used,
                "apply_scaling": apply_scaling,
                "skip_engineering": skip_engineering,
                "train_size": len(train_idx),
                "cal_size": len(cal_idx),
                "test_size": len(test_idx),
                "train_ratio": train_ratio if TEMPORAL_SPLIT_AVAILABLE else train_ratio,
                "cal_ratio": cal_ratio if TEMPORAL_SPLIT_AVAILABLE else 0.0,
                "transformers_saved": save_transformers and not skip_engineering,
            },
            "A": A_full,
            "B": B_full,
            "train_idx": train_idx,
            "cal_idx": cal_idx,    # NEW: Calibration indices for Conformal Prediction
            "test_idx": test_idx,
            "split_result": split_result,  # Full TemporalSplitResult object
        }

        # Also store split result at top level for easy access by other pages
        if split_result is not None:
            st.session_state["temporal_split"] = split_result

        # =========================================================================
        # DISPLAY RESULTS
        # =========================================================================
        st.success(f"✅ Dataset ready: {variant_used}")

        # Show split summary
        if TEMPORAL_SPLIT_AVAILABLE:
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Train Records", f"{len(train_idx):,}", f"{len(train_idx)/len(base)*100:.1f}%")
            with col_sum2:
                st.metric("Calibration Records", f"{len(cal_idx):,}", f"{len(cal_idx)/len(base)*100:.1f}%")
            with col_sum3:
                st.metric("Test Records", f"{len(test_idx):,}", f"{len(test_idx)/len(base)*100:.1f}%")

            if save_transformers and not skip_engineering:
                st.info(f"💾 Transformers saved to: `{TRANSFORMERS_DIR}/`")

        result_tab1, result_tab2, result_tab3 = st.tabs(["Variant A (Decomp)", "Variant B (Cyclical)", "Split Details"])
        with result_tab1:
            st.dataframe(A_full.head(20), use_container_width=True)
            st.download_button("⬇️ Download Variant A", A_full.to_csv(index=False).encode(), "variant_A.csv", "text/csv")
        with result_tab2:
            st.dataframe(B_full.head(20), use_container_width=True)
            st.download_button("⬇️ Download Variant B", B_full.to_csv(index=False).encode(), "variant_B.csv", "text/csv")
        with result_tab3:
            if split_result is not None:
                st.markdown("#### Temporal Split Summary")
                st.json({
                    "date_column": split_result.date_col,
                    "train_start": str(split_result.train_start),
                    "train_end": str(split_result.train_end),
                    "cal_start": str(split_result.cal_start),
                    "cal_end": str(split_result.cal_end),
                    "test_start": str(split_result.test_start),
                    "test_end": str(split_result.test_end),
                    "train_records": len(split_result.train_idx),
                    "cal_records": len(split_result.cal_idx),
                    "test_records": len(split_result.test_idx),
                })
            else:
                st.info("Temporal split details not available (using legacy split).")

# === ENTRYPOINT ===============================================================
def main():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    init_state()

    # Results Storage Panel (Supabase persistence)
    render_results_storage_panel(
        page_type="feature_engineering",
        page_title="Feature Engineering",
    )

    # Auto-load saved results if available
    auto_load_if_available("feature_engineering")

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
