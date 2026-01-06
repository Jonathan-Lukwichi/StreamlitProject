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
from app_core.ui.page_navigation import render_page_navigation
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# Import temporal split module
try:
    from app_core.pipelines import (
        compute_temporal_split,
        validate_temporal_split,
        TemporalSplitResult,
        # Time series cross-validation
        CVFoldResult,
        CVConfig,
        create_cv_folds,
        visualize_cv_folds,
        evaluate_cv_metrics,
        format_cv_metric,
    )
    TEMPORAL_SPLIT_AVAILABLE = True
    CV_AVAILABLE = True
except ImportError:
    TEMPORAL_SPLIT_AVAILABLE = False
    CV_AVAILABLE = False

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
    page_title="Feature Studio - HealthForecast AI",
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
    tab_split, tab_cv, tab_options, tab_generate = st.tabs([
        "📊 Data Split",
        "🔄 Cross-Validation",
        "🧪 Options",
        "🚀 Generate"
    ])

    with tab_split:
        st.markdown("#### Configure Train/Test Split")

        st.info("""
        **Data Leakage Prevention:** Scalers and encoders are fitted on training data ONLY,
        then applied to test data. This ensures no information from the test set leaks into training.
        """)

        # Simple 80/20 split
        train_ratio = st.slider(
            "Training Set Ratio",
            min_value=0.60,
            max_value=0.90,
            value=0.80,
            step=0.05,
            help="Proportion of data for training (recommended: 80%)"
        )

        test_ratio = 1.0 - train_ratio

        # Show split preview
        min_date = base[date_col].min()
        max_date = base[date_col].max()
        total_days = (max_date - min_date).days

        from datetime import timedelta
        train_end = min_date + timedelta(days=int(total_days * train_ratio))

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.metric("Train", f"{train_ratio:.0%}", f"{min_date.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
        with col_prev2:
            st.metric("Test", f"{test_ratio:.0%}", f"{train_end.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")

        # Compute simple 2-way split (no calibration set)
        train_idx, test_idx = _split_indices_legacy(base, date_col, train_ratio)
        cal_idx = np.array([], dtype=int)  # No calibration set
        split_result = None

    # =========================================================================
    # CROSS-VALIDATION TAB
    # Academic Reference: Bergmeir & Benítez (2012)
    # =========================================================================
    with tab_cv:
        st.markdown("#### Time Series Cross-Validation")

        st.info("""
        **Why Time Series CV?** Standard k-fold CV randomly shuffles data, which causes
        data leakage in time series. Time series CV uses expanding windows where training
        data always precedes validation data, mimicking real-world forecasting.

        *Reference: Bergmeir & Benítez (2012) - "On the use of cross-validation for time series predictor evaluation"*
        """)

        if not CV_AVAILABLE:
            st.warning("⚠️ Cross-validation module not available. Please check installation.")
        else:
            # CV Configuration
            cv_col1, cv_col2 = st.columns(2)

            with cv_col1:
                n_splits = st.slider(
                    "Number of Folds",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Number of train/validation splits. More folds = more robust estimates but slower training."
                )

                gap_days = st.slider(
                    "Gap (days)",
                    min_value=0,
                    max_value=14,
                    value=0,
                    step=1,
                    help="Days between training and validation sets to prevent autocorrelation leakage. "
                         "Recommended: 0-7 days for daily data."
                )

            with cv_col2:
                use_cv = st.checkbox(
                    "Enable Cross-Validation for Model Training",
                    value=True,
                    help="When enabled, models will be trained on all CV folds and metrics will show mean ± std."
                )

                min_train_pct = st.slider(
                    "Minimum Training Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Minimum percentage of data required for training in the first fold."
                )

            # Save CV config to session state
            cv_config = CVConfig(
                n_splits=n_splits,
                gap=gap_days,
                min_train_size=int(len(base) * min_train_pct / 100) if min_train_pct > 0 else None,
                test_size=None
            )
            st.session_state["cv_config"] = cv_config
            st.session_state["cv_enabled"] = use_cv

            # Preview CV folds
            st.markdown("---")
            st.markdown("##### CV Fold Preview")

            try:
                cv_folds = create_cv_folds(
                    df=base,
                    date_col=date_col,
                    n_splits=n_splits,
                    gap=gap_days,
                    min_train_size=cv_config.min_train_size,
                    verbose=False
                )

                # Display fold summary
                fold_data = []
                for fold in cv_folds:
                    fold_data.append({
                        "Fold": fold.fold_idx + 1,
                        "Train Start": fold.train_start.strftime("%Y-%m-%d"),
                        "Train End": fold.train_end.strftime("%Y-%m-%d"),
                        "Train Samples": fold.n_train,
                        "Val Start": fold.val_start.strftime("%Y-%m-%d"),
                        "Val End": fold.val_end.strftime("%Y-%m-%d"),
                        "Val Samples": fold.n_val,
                    })
                fold_df = pd.DataFrame(fold_data)
                st.dataframe(fold_df, use_container_width=True, hide_index=True)

                # Visualization
                fig = visualize_cv_folds(base, date_col, cv_folds, "Time Series Cross-Validation Folds")
                st.plotly_chart(fig, use_container_width=True, key="cv_folds_viz")

                # Summary metrics
                total_train = sum(f.n_train for f in cv_folds)
                total_val = sum(f.n_val for f in cv_folds)
                avg_train = total_train / len(cv_folds) if cv_folds else 0
                avg_val = total_val / len(cv_folds) if cv_folds else 0

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Active Folds", len(cv_folds))
                with metric_cols[1]:
                    st.metric("Avg Train Size", f"{avg_train:,.0f}")
                with metric_cols[2]:
                    st.metric("Avg Val Size", f"{avg_val:,.0f}")
                with metric_cols[3]:
                    st.metric("Total Samples", f"{len(base):,}")

                # Save folds to session state for use in training
                st.session_state["cv_folds"] = cv_folds

                st.success(f"✅ {len(cv_folds)} CV folds configured. Ready for model training with cross-validation.")

            except Exception as e:
                st.error(f"Error creating CV folds: {str(e)}")
                st.session_state["cv_folds"] = None

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
        # STORE IN SESSION STATE (Simple 80/20 split for Train Models page)
        # =========================================================================
        st.session_state["feature_engineering"] = {
            "summary": {
                "variant_used": variant_used,
                "apply_scaling": apply_scaling,
                "skip_engineering": skip_engineering,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "transformers_saved": save_transformers and not skip_engineering,
            },
            "A": A_full,
            "B": B_full,
            "train_idx": train_idx,
            "cal_idx": cal_idx,    # Empty array (no calibration set)
            "test_idx": test_idx,
            "split_result": split_result,
        }

        # =========================================================================
        # DISPLAY RESULTS
        # =========================================================================
        st.success(f"✅ Dataset ready: {variant_used}")

        # Show split summary
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.metric("Train Records", f"{len(train_idx):,}", f"{len(train_idx)/len(base)*100:.1f}%")
        with col_sum2:
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
            st.markdown("#### Train/Test Split Summary")

            # Get date boundaries
            train_dates = base.iloc[train_idx][date_col]
            test_dates = base.iloc[test_idx][date_col]

            st.json({
                "date_column": date_col,
                "train_start": str(train_dates.min()),
                "train_end": str(train_dates.max()),
                "test_start": str(test_dates.min()),
                "test_end": str(test_dates.max()),
                "train_records": len(train_idx),
                "test_records": len(test_idx),
                "train_ratio": f"{train_ratio:.0%}",
                "test_ratio": f"{test_ratio:.0%}",
            })

            st.markdown("---")
            st.markdown("#### Data Leakage Prevention")
            st.success("""
            ✅ **Scalers fitted on TRAIN data only** → Applied to both train and test

            ✅ **Encoders fitted on TRAIN data only** → Applied to both train and test

            ✅ **No future information leaked** → Test data is strictly after train data
            """)

# === ENTRYPOINT ===============================================================
def main():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    init_state()

    # Results Storage Panel (Supabase persistence)
    render_results_storage_panel(
        page_type="feature_engineering",
        page_title="Feature Studio",
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
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Feature Studio</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Create sophisticated temporal and cyclical features with automated deduplication and scaling for enhanced model performance
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page_feature_engineering()

    # =============================================================================
    # PAGE NAVIGATION
    # =============================================================================
    render_page_navigation(5)  # Feature Studio is page index 5

if __name__ == "__main__":
    main()
