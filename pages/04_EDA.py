# ============================================================================
# pages/04_EDA_enhanced.py — Exploratory Data Analysis (Template-Matched)
# Consistent hero + section cards + typography with Dashboard/Data Hub/Studio
# ============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical tools
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# App core
try:
    from app_core.ui.theme import (
        apply_css,
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
        DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, CARD_BG_LIGHT,
    )
except Exception:
    # Safe fallbacks for prototyping
    def apply_css(): pass
    PRIMARY_COLOR   = "#2563eb"
    SECONDARY_COLOR = "#8b5cf6"
    SUCCESS_COLOR   = "#10b981"
    WARNING_COLOR   = "#f59e0b"
    DANGER_COLOR    = "#ef4444"
    TEXT_COLOR      = "#0f172a"
    SUBTLE_TEXT     = "#64748b"
    CARD_BG_LIGHT   = "#ffffff"

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        st.session_state.setdefault("processed_df", None)

# Optional: grid helper
try:
    from app_core.ui.components import add_grid
except Exception:
    def add_grid(fig): return fig

# -------------------------- Local CSS (template) -------------------------------
def _eda_css():
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
        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------- Robust Time Index Helpers ---------------------------
def _coerce_datetime_like(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")

def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index(); out.index.name = "_dt"; return out

    time_candidates = [c for c in ["datetime", "Date", "date"] if c in df.columns]
    dt = None
    for c in time_candidates:
        co = _coerce_datetime_like(df[c])
        if co.notna().sum() > 0: dt = co; break

    if dt is not None:
        mask = dt.notna(); df = df.loc[mask].copy()
        df.index = pd.DatetimeIndex(dt[mask]); df.index.name = "_dt"
        return df.sort_index()

    idx_dt = _coerce_datetime_like(pd.Series(df.index))
    if idx_dt.notna().any():
        valid_idx = idx_dt.notna()
        df = df.loc[valid_idx.values].copy()
        df.index = pd.DatetimeIndex(idx_dt[valid_idx]); df.index.name = "_dt"
        return df.sort_index()

    st.warning("⏱️ Could not infer a date column; synthesizing a daily index.")
    df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    df.index.name = "_dt"
    return df

def _quick_missing_table(df: pd.DataFrame, topn: int = 12) -> pd.DataFrame:
    miss = df.isna().sum()
    if (miss == 0).all():
        return pd.DataFrame({"column": [], "missing": [], "pct": []})
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "column": miss.index,
        "missing": miss.values,
        "pct": (miss.values / len(df) * 100).round(2),
    }).head(topn)

def _month_order():
    return ["January","February","March","April","May","June",
            "July","August","September","October","November","December"]

def _day_map():
    return {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# ------------------------- FFT / Cycle Utilities -------------------------------
def _classify_cycle(period_days: float, *, harmonics: bool=True, tol: float=0.08, label_mode: str="friendly") -> str:
    if not np.isfinite(period_days) or period_days <= 0:
        return "other"
    if abs(period_days - 1.0) <= 0.30: return "daily"
    if 6.0 <= period_days <= 8.0: return "weekly"
    if 13.0 <= period_days <= 16.0: return "bi-weekly"
    if 28.0 <= period_days <= 31.0: return "monthly"
    if 89.0 <= period_days <= 93.0: return "quarterly"
    if 180.0 <= period_days <= 186.0: return "semi-annual"
    if 360.0 <= period_days <= 370.0: return "yearly"

    if harmonics:
        YEAR   = 365.2425
        MONTH  = YEAR/12.0
        QUART  = YEAR/4.0
        WEEK   = 7.0
        def _near_multiple(x, base, max_mult, rel_tol):
            k = int(np.round(x / base))
            if k < 1 or k > max_mult: return None, np.inf
            target = base * k
            err = abs(x - target) / target
            return (k, err) if err <= rel_tol else (None, err)

        checks = [("weekly", WEEK, 60), ("monthly", MONTH, 36), ("quarterly", QUART, 16), ("yearly", YEAR, 12)]
        best = None
        for base_label, base_val, max_k in checks:
            k, err = _near_multiple(period_days, base_val, max_k, tol)
            if k is not None:
                if label_mode == "friendly":
                    if base_label == "weekly" and k == 2: label = "bi-weekly"
                    elif base_label == "yearly" and k == 2: label = "bi-annual"
                    elif base_label == "yearly" and k == 4: label = "quarterly"
                    elif base_label == "monthly" and k == 1: label = "monthly"
                    elif base_label == "yearly" and k == 1: label = "yearly"
                    else: label = f"{base_label}×{k}"
                else:
                    label = f"{base_label}×{k}"
                cand = (label, err)
                if (best is None) or (err < best[1]): best = cand
        if best is not None: return best[0]
    return "other"

def _extract_fft_cycles(y: pd.Series, top_k: int=5, detrend_mean: bool=True, classify_kwargs: dict | None=None) -> pd.DataFrame:
    y = y.asfreq("D").interpolate(method="time", limit_direction="both")
    sig = y.values.astype(float)
    if detrend_mean: sig = sig - np.nanmean(sig)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    valid = freqs > 0
    freqs = freqs[valid]; amps = np.abs(fft_vals[valid])
    if len(freqs) == 0: return pd.DataFrame(columns=["cycle_type","period_days","amplitude"])
    k = max(1, min(top_k, len(freqs)))
    idx = np.argpartition(amps, -k)[-k:]; idx = idx[np.argsort(amps[idx])[::-1]]
    periods = 1.0 / freqs[idx]; amplitudes = amps[idx]
    classify_kwargs = classify_kwargs or {}
    cycle_types = [_classify_cycle(p, **classify_kwargs) for p in periods]
    out = pd.DataFrame({"cycle_type": cycle_types, "period_days": periods, "amplitude": amplitudes})
    return out.sort_values("amplitude", ascending=False).reset_index(drop=True)

def _fft_full_spectrum(y: pd.Series, detrend_mean: bool=True) -> pd.DataFrame:
    y = y.asfreq("D").interpolate(method="time", limit_direction="both")
    sig = y.values.astype(float)
    if detrend_mean: sig = sig - np.nanmean(sig)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    valid = freqs > 0
    freqs = freqs[valid]; amps = np.abs(fft_vals[valid])
    periods = 1.0 / freqs
    spec = pd.DataFrame({"period_days": periods, "amplitude": amps})
    return spec.sort_values("period_days")

# ----------------------------- Render Helpers ----------------------------------
def _render_donut_charts(dff):
    # DOW
    if "day_of_week" in dff.columns:
        agg_dow = dff.groupby("day_of_week")["Target_1"].mean().reset_index().sort_values("day_of_week")
    else:
        agg_dow = pd.DataFrame()

    # Month (derive if needed)
    if "Month_name" in dff.columns:
        agg_month = dff.groupby("Month_name")["Target_1"].mean().reindex(_month_order()).dropna()
        month_labels = agg_month.index.tolist(); month_values = agg_month.values.tolist()
    else:
        idx = dff.index if isinstance(dff.index, pd.DatetimeIndex) else _coerce_datetime_like(dff.index.to_series())
        temp = pd.DataFrame({"_dt": pd.to_datetime(idx, errors="coerce"), "Target_1": dff["Target_1"].values}).dropna(subset=["_dt"])
        if temp.empty:
            month_labels, month_values = [], []
        else:
            temp["Month_name"] = temp["_dt"].dt.month_name()
            agg = temp.groupby("Month_name")["Target_1"].mean().reindex(_month_order()).dropna()
            month_labels, month_values = agg.index.tolist(), agg.values.tolist()

    def _donut(labels, values, title):
        if not labels or not values: return None
        s = pd.Series(values); colors = [PRIMARY_COLOR] * len(values)
        if len(values) > 1:
            try:
                colors[int(s.idxmax())] = DANGER_COLOR
                colors[int(s.idxmin())] = SUCCESS_COLOR
            except Exception: pass
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.6, sort=False,
            marker=dict(colors=colors, line=dict(color=CARD_BG_LIGHT, width=3)),
            textinfo="percent", textposition="outside",
            textfont=dict(size=13, color=TEXT_COLOR),
            hovertemplate="<b>%{label}</b><br>Avg Arrivals: %{value:.1f}<extra></extra>",
        )])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=TEXT_COLOR), x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=350, showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=11)),
            paper_bgcolor=CARD_BG_LIGHT, plot_bgcolor=CARD_BG_LIGHT,
        )
        return fig

    c1, c2 = st.columns(2)
    with c1:
        if not agg_dow.empty:
            labels = [_day_map().get(int(d), str(d)) for d in agg_dow["day_of_week"]]
            fig = _donut(labels, agg_dow["Target_1"].tolist(), "📅 Avg by Day of Week")
            if fig: st.plotly_chart(add_grid(fig), use_container_width=True)
        else:
            st.info("`day_of_week` not found.")
    with c2:
        if month_labels:
            fig = _donut(month_labels, month_values, "📆 Avg by Month")
            if fig: st.plotly_chart(add_grid(fig), use_container_width=True)
        else:
            st.info("Could not derive month names for pie chart.")

def _render_decomposition(df):
    cA, cB = st.columns([3, 1])
    with cB:
        max_period = max(7, len(df) // 4) if len(df) > 0 else 7
        period = st.number_input("Seasonal Period (days)", min_value=2, max_value=int(max_period), value=7, step=1)
        model_type = st.selectbox("Decomposition Model", ["additive", "multiplicative"])
    with cA:
        ts = df["Target_1"].asfreq("D").interpolate(method="time", limit_direction="both")
        needed = 2*int(period) + 1
        if len(ts.dropna()) >= needed:
            try:
                res = seasonal_decompose(ts.dropna(), model=model_type, period=int(period), extrapolate_trend="freq")
                dec_fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                dec_fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed, line=dict(color=PRIMARY_COLOR, width=1.5)), row=1, col=1)
                dec_fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, line=dict(color=DANGER_COLOR, width=2)), row=2, col=1)
                dec_fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, line=dict(color=SUCCESS_COLOR, width=1.5)), row=3, col=1)
                dec_fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode="markers",
                                             marker=dict(color=WARNING_COLOR, size=3, opacity=0.7)), row=4, col=1)
                dec_fig.update_layout(height=600, showlegend=False, margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(add_grid(dec_fig), use_container_width=True)
            except Exception as e:
                st.error(f"Decomposition error: {e}. Check period and data.")
        else:
            st.info(f"Need ≥ {needed} valid points for period {period}; have {len(ts.dropna())}.")

def _render_correlation_analysis(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis.")
        return
    use_all = st.checkbox("Use all numeric columns", value=True)
    if use_all:
        selected = numeric_cols
    else:
        default_sel = ["Target_1"] if "Target_1" in numeric_cols else []
        selected = st.multiselect("Select variables to include:", options=numeric_cols, default=default_sel)
        if len(selected) < 2:
            st.info("Select at least two variables."); return
    try:
        corr_method = st.selectbox("Correlation Method", ["pearson","spearman","kendall"], index=1)
        corr = df[selected].corr(method=corr_method)
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                             title=f"{corr_method.capitalize()} Correlation Matrix (Selected Variables)",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(height=600)
        st.plotly_chart(add_grid(fig_corr), use_container_width=True)

        if "Target_1" in corr.columns:
            target_corr = corr["Target_1"].drop("Target_1", errors="ignore").sort_values(key=np.abs, ascending=False)
            st.markdown("#### 🎯 Correlation with Target_1")
            st.dataframe(pd.DataFrame({"Feature": target_corr.index, "Correlation": target_corr.values.round(3)}),
                         use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not compute correlation: {e}")

def _render_time_series_diagnostics(df):
    ts_interpolated = df["Target_1"].asfreq("D").interpolate(method="time", limit_direction="both")
    y = ts_interpolated.dropna().values
    if len(y) <= 10:
        st.info("Need more than 10 data points for ACF/PACF and ADF tests."); return
    nlags_default = min(40, max(10, len(y)//3))
    nlags = st.slider("Number of Lags (ACF/PACF)", 5, max(60, nlags_default*2), nlags_default)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Autocorrelation (ACF & PACF)")
        try:
            acf_vals, confint_acf = sm_acf(y, nlags=nlags, alpha=0.05, fft=True)
            pacf_vals, confint_pacf = sm_pacf(y, nlags=nlags, method="ywm", alpha=0.05)
            fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF","PACF"])
            fig.add_trace(go.Bar(x=np.arange(len(acf_vals)),  y=acf_vals,  marker_color=PRIMARY_COLOR), row=1, col=1)
            ciu = confint_acf[1:,1]-acf_vals[1:]; cil = confint_acf[1:,0]-acf_vals[1:]
            fig.add_trace(go.Scatter(x=np.arange(1,len(acf_vals)), y=ciu, mode="lines", line=dict(color="rgba(0,0,0,0)")), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(1,len(acf_vals)), y=cil, mode="lines", line=dict(color="rgba(0,0,0,0)"),
                                     fill="tonexty", fillcolor="rgba(102,126,234,0.12)"), row=1, col=1)
            fig.add_trace(go.Bar(x=np.arange(len(pacf_vals)), y=pacf_vals, marker_color=SECONDARY_COLOR), row=1, col=2)
            ciu = confint_pacf[1:,1]-pacf_vals[1:]; cil = confint_pacf[1:,0]-pacf_vals[1:]
            fig.add_trace(go.Scatter(x=np.arange(1,len(pacf_vals)), y=ciu, mode="lines", line=dict(color="rgba(0,0,0,0)")), row=1, col=2)
            fig.add_trace(go.Scatter(x=np.arange(1,len(pacf_vals)), y=cil, mode="lines", line=dict(color="rgba(0,0,0,0)"),
                                     fill="tonexty", fillcolor="rgba(118,75,162,0.12)"), row=1, col=2)
            fig.update_layout(height=420, margin=dict(t=50,b=20,l=20,r=20), showlegend=False)
            st.plotly_chart(add_grid(fig), use_container_width=True)
        except Exception as e:
            st.error(f"Error generating ACF/PACF plots: {e}")

    with c2:
        st.markdown("#### 🔬 ADF Stationarity Test")
        try:
            result = adfuller(y)
            adf_stat, p_value, crit = result[0], result[1], result[4]
            st.metric("ADF Statistic", f"{adf_stat:.4f}")
            st.metric("P-value", f"{p_value:.6f}")
            if p_value < 0.05: st.success("✅ Stationary (p < 0.05)")
            else:              st.warning("🟡 Non-Stationary (p ≥ 0.05)")
            with st.expander("Critical Values"):
                for k, v in crit.items():
                    st.write(f"{k}: {v:.4f} {'(Reject H0)' if adf_stat < v else ''}")
        except Exception as e:
            st.error(f"ADF test failed: {e}")

# ---------------------------------- Page ---------------------------------------
def page_eda():
    # HERO
    st.markdown(
        f"""
        <div class='hero'>
          <div class='hero-title'>📊 Exploratory Data Analysis</div>
          <p class='hero-sub'>Work with the <strong>preprocessed dataset</strong> to understand structure, quality, and time-series behavior.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    processed = st.session_state.get("processed_df")
    if processed is None or (isinstance(processed, pd.DataFrame) and processed.empty):
        st.warning("⚠️ Please run **Data Preparation Studio** first. No preprocessed dataset found.")
        if st.button("Go to Data Preparation Studio"):
            st.session_state["_nav_intent"] = "🧰 Data Preparation Studio"
            st.rerun()
        return

    df = processed.copy()

    if "Target_1" not in df.columns:
        st.error("❌ The preprocessed dataset is missing **'Target_1'**."); return

    # Ensure robust datetime index
    try:
        df = _ensure_time_index(df)
        if not isinstance(df.index, pd.DatetimeIndex):
            idx_dt = _coerce_datetime_like(pd.Series(df.index))
            df.index = pd.DatetimeIndex(idx_dt)
            df = df[df.index.notna()].copy()
            df.index.name = "_dt"
        st.success(f"✅ Time index set successfully. Dataset covers {len(df)} records.")
    except Exception as e:
        st.error(f"❌ Error setting time index: {e}"); return

    df["Target_1"] = pd.to_numeric(df["Target_1"], errors="coerce")
    if df["Target_1"].isna().any():
        st.warning("⚠️ Some 'Target_1' values are non-numeric or missing after preprocessing.")

    # Sidebar Date Filter
    with st.sidebar:
        st.markdown("### 🔧 Analysis Controls")
        if not df.empty:
            date_min, date_max = df.index.min().date(), df.index.max().date()
            date_range = st.date_input("Date Range", value=(date_min, date_max),
                                       min_value=date_min, max_value=date_max)
        else:
            st.warning("No data available for date filtering"); date_range = None

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0]); end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        df = df.loc[(df.index >= start) & (df.index <= end)]

    # Overview metrics (section card)
    st.markdown("<div class='section-title'>📈 Dataset Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{len(df):,}")
    with c2: st.metric("Columns", f"{df.shape[1]}")
    with c3:
        sd = df.index.min(); sd_str = f"{(sd.date() if hasattr(sd,'date') else sd)}" if sd is not pd.NaT else "N/A"
        st.metric("Start Date", sd_str)
    with c4:
        ed = df.index.max(); ed_str = f"{(ed.date() if hasattr(ed,'date') else ed)}" if ed is not pd.NaT else "N/A"
        st.metric("End Date", ed_str)
    st.markdown("</div>", unsafe_allow_html=True)

    # Data Quality (section card)
    st.markdown("<div class='section-title'>🔍 Data Quality</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    miss_tbl = _quick_missing_table(df, topn=12)
    if len(miss_tbl) == 0:
        st.success("🎉 No missing values detected in the dataset")
    else:
        st.warning(f"⚠️ Found {len(miss_tbl)} columns with missing values")
        fig_miss = px.bar(miss_tbl, x='pct', y='column', orientation='h',
                          title='Missing Values by Column',
                          labels={'pct': 'Missing %', 'column': 'Column'},
                          color_discrete_sequence=[WARNING_COLOR])
        st.plotly_chart(add_grid(fig_miss), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tabs
    tab_over, tab_hist, tab_ts, tab_adv = st.tabs(["📋 Overview","📊 Distributions","📈 Time Series","🔬 Advanced Analytics"])

    # OVERVIEW
    with tab_over:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown("#### Dataset Preview")
            all_columns = df.columns.tolist()
            default_cols = [c for c in ['Target_1','day_of_week','Month_name'] if c in all_columns]
            selected_cols = st.multiselect("Select columns to display:", options=all_columns, default=default_cols)
            row_count = st.slider("Number of rows to display:", 5, 50, 10)
            display_df = df[selected_cols] if selected_cols else df
            st.dataframe(display_df.head(row_count), use_container_width=True, height=400)
            st.download_button("📥 Download Full Dataset as CSV", df.to_csv(), "eda_dataset.csv", "text/csv")
        with colB:
            st.markdown("#### 🔑 Key Statistics")
            s = df["Target_1"].dropna() if "Target_1" in df.columns else pd.Series(dtype=float)
            if not s.empty:
                stats = {"Mean": f"{s.mean():.1f}","Median": f"{s.median():.1f}","Std Dev": f"{s.std():.1f}",
                         "Min": f"{s.min():.0f}","Max": f"{s.max():.0f}","Total": f"{s.sum():,.0f}"}
                for k, v in stats.items(): st.metric(k, v)
            else:
                st.warning("No valid Target_1 data")
        with st.expander("📋 View Full Schema Details"):
            st.markdown(f"**Dataset Shape:** {df.shape}")
            type_counts = df.dtypes.value_counts()
            st.markdown("**Column Types:**")
            for dtype, count in type_counts.items(): st.write(f"- {dtype}: {count} columns")
            st.markdown("**Complete Column List:**")
            for col in df.columns: st.write(f"- `{col}` ({df[col].dtype})")
        st.markdown("</div>", unsafe_allow_html=True)

    # DISTRIBUTIONS
    with tab_hist:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Distribution Analysis")

        # Optional day-of-week quick filter
        if "day_of_week" in df.columns:
            st.markdown("#### 🎯 Quick Filters")
            filter_cols = st.columns(8)
            with filter_cols[0]:
                if st.button("All Days", use_container_width=True, key="btn_all_days"):
                    st.session_state["_eda_dow_quick"] = None; st.rerun()
            day_names = _day_map()
            for i, day_num in enumerate(range(1, 7+1), start=1):
                with filter_cols[i]:
                    active = st.session_state.get("_eda_dow_quick") == day_num
                    if st.button(day_names[day_num], use_container_width=True,
                                 key=f"btn_day_{day_num}", type="primary" if active else "secondary"):
                        st.session_state["_eda_dow_quick"] = day_num; st.rerun()
            dff = df[df["day_of_week"] == int(st.session_state.get("_eda_dow_quick"))] \
                  if st.session_state.get("_eda_dow_quick") else df.copy()
        else:
            dff = df.copy()

        if dff.empty:
            st.warning("No data available for distribution analysis")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(dff, x="Target_1", nbins=30,
                                        title="Distribution of Patient Arrivals",
                                        color_discrete_sequence=[PRIMARY_COLOR])
                fig_hist.update_layout(xaxis_title="Patient Arrivals", yaxis_title="Frequency")
                st.plotly_chart(add_grid(fig_hist), use_container_width=True)
            with c2:
                fig_box = px.box(dff, y="Target_1", title="Arrivals Distribution Overview",
                                 color_discrete_sequence=[SECONDARY_COLOR])
                st.plotly_chart(add_grid(fig_box), use_container_width=True)

            st.markdown("#### 📈 Categorical Breakdowns")
            _render_donut_charts(dff)

            st.markdown("#### 🌡️ Weather-Based Arrival Distributions")
            # Temp
            if "Average_Temp" in dff.columns:
                bins = st.slider("Temperature bins", 5, 50, 15, key="temp_bins")
                temp_bins = pd.cut(dff["Average_Temp"].dropna(), bins=bins)
                temp_avg = dff.dropna(subset=["Average_Temp"]).groupby(temp_bins)["Target_1"].mean().reset_index()
                temp_avg["bin_mid"] = temp_avg["Average_Temp"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
                fig_temp = px.bar(temp_avg, x="bin_mid", y="Target_1",
                                  title="Avg Arrivals by Average Temperature",
                                  labels={"bin_mid": "Average Temperature (bin mid)", "Target_1": "Average Arrivals"},
                                  color_discrete_sequence=[PRIMARY_COLOR])
                st.plotly_chart(add_grid(fig_temp), use_container_width=True)
            else:
                st.info("Average_Temp not found.")
            # Wind
            if "Average_wind" in dff.columns:
                bins_w = st.slider("Wind bins", 5, 50, 15, key="wind_bins")
                wind_bins = pd.cut(dff["Average_wind"].dropna(), bins=bins_w)
                wind_avg = dff.dropna(subset=["Average_wind"]).groupby(wind_bins)["Target_1"].mean().reset_index()
                wind_avg["bin_mid"] = wind_avg["Average_wind"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
                fig_wind = px.bar(wind_avg, x="bin_mid", y="Target_1",
                                  title="Avg Arrivals by Average Wind",
                                  labels={"bin_mid": "Average Wind (bin mid)", "Target_1": "Average Arrivals"},
                                  color_discrete_sequence=[SECONDARY_COLOR])
                st.plotly_chart(add_grid(fig_wind), use_container_width=True)
            else:
                st.info("Average_wind not found.")
            # Precip
            if "Total_precipitation" in dff.columns:
                bins_p = st.slider("Precipitation bins", 5, 50, 15, key="precip_bins")
                p_bins = pd.cut(dff["Total_precipitation"].dropna(), bins=bins_p)
                p_avg = dff.dropna(subset=["Total_precipitation"]).groupby(p_bins)["Target_1"].mean().reset_index()
                p_avg["bin_mid"] = p_avg["Total_precipitation"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
                fig_precip = px.bar(p_avg, x="bin_mid", y="Target_1",
                                    title="Avg Arrivals by Total Precipitation",
                                    labels={"bin_mid": "Total Precipitation (bin mid)", "Target_1": "Average Arrivals"},
                                    color_discrete_sequence=[SUCCESS_COLOR])
                st.plotly_chart(add_grid(fig_precip), use_container_width=True)
            else:
                st.info("Total_precipitation not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TIME SERIES
    with tab_ts:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Time Series Analysis")
        if df.empty:
            st.warning("No data available for time series analysis")
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                plot_df = df.copy()
                temp_x = "__dt__"
                while temp_x in plot_df.columns: temp_x = "_" + temp_x
                plot_df[temp_x] = plot_df.index
                fig_ts = px.line(plot_df, x=temp_x, y="Target_1",
                                 title="Daily Patient Arrivals Over Time",
                                 labels={"Target_1": "Arrivals", temp_x: "Date"})
                fig_ts.update_traces(line=dict(color=PRIMARY_COLOR, width=2),
                                     hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Arrivals: %{y}<extra></extra>")
                st.plotly_chart(add_grid(fig_ts), use_container_width=True)
            with c2:
                st.markdown("#### 📊 Performance Metrics")
                s = df["Target_1"].dropna()
                if not s.empty:
                    current_avg = s.mean()
                    prev_avg = s.iloc[:-30].mean() if len(s) > 30 else current_avg
                    delta_pct = ((current_avg - prev_avg) / prev_avg * 100) if prev_avg else 0.0
                    st.metric("Average Daily Arrivals", f"{current_avg:.1f}", f"{delta_pct:+.1f}%")
                    extras = [("Trend Strength","Strong" if s.std() > s.mean()*0.3 else "Moderate"),
                              ("Seasonality","High" if len(s) > 365 else "Developing"),
                              ("Data Quality","Excellent" if s.isna().sum()==0 else "Good")]
                    for k,v in extras: st.metric(k, v)
                else:
                    st.warning("No valid Target_1 data")
            st.markdown("#### 🔬 Time Series Decomposition")
            _render_decomposition(df)
        st.markdown("</div>", unsafe_allow_html=True)

    # ADVANCED
    with tab_adv:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Advanced Analytics")
        st.markdown("#### 📊 Correlation Matrix")
        _render_correlation_analysis(df)

        st.markdown("#### 📉 Time Series Diagnostics")
        _render_time_series_diagnostics(df)

        st.markdown("#### 🧠 Dominant Cycles (FFT)")
        ts = df["Target_1"].asfreq("D").interpolate(method="time", limit_direction="both")
        if ts.dropna().empty or len(ts) < 8:
            st.info("Need at least ~8 daily points to extract dominant cycles.")
        else:
            c1, c2, c3 = st.columns([1,1,1])
            with c1: top_k = st.slider("Top K frequencies", 1, 100, 5)
            with c2: detrend_mean = st.checkbox("Remove mean (detrend before FFT)", value=True)
            with c3:
                tol_pct = st.slider("Harmonic tolerance (±%)", 1, 20, 8)
                label_mode = st.selectbox("Label mode", ["friendly","harmonic"], index=0)
            try:
                spec = _fft_full_spectrum(ts, detrend_mean=detrend_mean)
                spec_view = spec[spec["period_days"] <= 400].copy()
                if not spec_view.empty:
                    fig_spec = px.line(spec_view, x="period_days", y="amplitude",
                                       title="FFT Amplitude Spectrum (Period in Days)",
                                       labels={"period_days":"Period (days)", "amplitude":"Amplitude"})
                    fig_spec.update_traces(line=dict(color=PRIMARY_COLOR, width=2))
                    st.plotly_chart(add_grid(fig_spec), use_container_width=True)
                else:
                    st.info("Spectrum available but outside 0–400 days window.")
            except Exception as e:
                st.error(f"FFT spectrum failed: {e}")

            try:
                cycles_df = _extract_fft_cycles(
                    ts, top_k=top_k, detrend_mean=detrend_mean,
                    classify_kwargs={"harmonics":True, "tol": tol_pct/100.0, "label_mode": label_mode},
                )
                if cycles_df.empty:
                    st.info("No dominant non-zero frequencies found.")
                else:
                    st.markdown("##### Top Cycles")
                    st.dataframe(cycles_df, use_container_width=True, hide_index=True)
                    agg = cycles_df.groupby("cycle_type")["amplitude"].sum().sort_values(ascending=False)
                    dominant_type = agg.index[0] if not agg.empty else "other"
                    st.success(f"**Dominant cycle type detected:** `{dominant_type}`")

                    fig_amp = px.bar(cycles_df, x="period_days", y="amplitude", color="cycle_type",
                                     title="Dominant Cycles — Amplitude by Period (days)",
                                     labels={"period_days":"Period (days)", "amplitude":"Amplitude"})
                    st.plotly_chart(add_grid(fig_amp), use_container_width=True)

                    st.markdown("##### 💡 Feature Engineering Hints (Based on Detected Cycles)")
                    tips = []
                    if "weekly" in agg.index and agg["weekly"] == agg.max():
                        tips.append("- Strong **weekly** seasonality → include `day_of_week`, 7-day Fourier terms (k=1..2), or 7-lag features.")
                    if "monthly" in agg.index and agg["monthly"] == agg.max():
                        tips.append("- Strong **monthly** pattern (~30 days) → month-of-year dummies or 30-day Fourier; consider 28–31 rolling.")
                    if "yearly" in agg.index and agg["yearly"] == agg.max():
                        tips.append("- **Yearly** seasonality → month-of-year indicators, holiday flags, annual Fourier (k=1..3).")
                    if "daily" in agg.index and agg["daily"] == agg.max():
                        tips.append("- **Daily** cycle on daily data → lean on recent lags (1..7) and check for mixed sampling.")
                    if "bi-weekly" in agg.index and agg["bi-weekly"] == agg.max():
                        tips.append("- **Bi-weekly** signal → 14-day Fourier or lag-14 features.")
                    if any(t.startswith("weekly×") for t in agg.index):
                        tips.append("- Higher-order **weekly harmonics** → multiple weekly Fourier terms or seasonal AR components.")
                    if any(t.startswith("monthly×") for t in agg.index):
                        tips.append("- **Monthly harmonics** present → monthly Fourier pairs + month-of-year dummies.")
                    if any(t.startswith("yearly×") for t in agg.index):
                        tips.append("- **Multi-annual** cycles → higher-k yearly Fourier and holiday/season effects.")
                    if not tips:
                        tips.append("- Mixed or unclear cycles → start with weekly dummies, a few Fourier terms (weekly & yearly), and ED_1..ED_7 lags.")
                    st.write("\n".join(tips))
            except Exception as e:
                st.error(f"FFT analysis failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------- Entrypoint -----------------------------------
def main():
    apply_css()
    _eda_css()
    init_state()
    page_eda()

if __name__ == "__main__":
    main()
