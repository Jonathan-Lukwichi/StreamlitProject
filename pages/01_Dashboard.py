# ========================================================================== 
# pages/00_Dashboard_enhanced.py — Healthcare Forecasting Dashboard (Pro Prototype)
# Keeps your original visual language; adds high‑signal sections:
#  • Hero (same style)
#  • KPI row (4 cards)
#  • Data Health (schema, date range, missing)
#  • Recent Activity (sparkline, last ~30 days)
#  • Model Readiness (pipeline gates)
#  • Quick Insights (mean, volatility, weekday/weekend hint)
# Defensive: runs fine with partial/missing data.
# ========================================================================== 
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Sidebar brand
try:
    from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
except ImportError:
    def inject_sidebar_style():
        pass
    def render_sidebar_brand():
        pass

# ---- Theme fallbacks (keeps template look even without app_core) ----
try:
    from app_core.ui.theme import apply_css
    from app_core.ui.theme import (
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
        DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT,
    )
    from app_core.ui.components import header
except Exception:
    PRIMARY_COLOR   = "#2563eb"
    SECONDARY_COLOR = "#8b5cf6"
    SUCCESS_COLOR   = "#10b981"
    WARNING_COLOR   = "#f59e0b"
    DANGER_COLOR    = "#ef4444"
    TEXT_COLOR      = "#0f172a"
    SUBTLE_TEXT     = "#64748b"
    def apply_css():
        st.markdown(
            """
            <style>
            .metric-card {border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:#fff;}
            .muted {color:#6b7280}
            </style>
            """,
            unsafe_allow_html=True,
        )

try:
    from app_core.state.session import init_state
except Exception:
    def init_state():
        for k, v in {
            "patient_loaded": False,
            "weather_loaded": False,
            "calendar_loaded": False,
            "patient_data": None,
            "weather_data": None,
            "calendar_data": None,
            "merged_data": None,
            "_nav_intent": None,
        }.items():
            st.session_state.setdefault(k, v)

# ---- Local CSS (extends your template) ----

def _dashboard_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*='css'] {{ font-family: 'Inter', system-ui, sans-serif; }}
        
        /* 1. Global background */
        .main {{
            background-color: #000000;
        }}
        
        .dashboard-hero {{
            border-radius: 18px; padding: 36px; color: #E6EAF2;
            background: #000000;
            border: 1px solid #1F2937;
            box-shadow: 0 10px 20px -5px rgba(0,0,0,.2);
            position: relative; overflow: hidden; margin-bottom: 22px;
        }}
        .dashboard-hero::before {{
            content: ''; position: absolute; inset: 0;
            background: radial-gradient(600px 300px at 85% 15%, rgba(37,99,235,.1), transparent 60%),
                        radial-gradient(600px 300px at 15% 85%, rgba(16,185,129,.1), transparent 60%);
            pointer-events: none;
        }}
        .dashboard-hero > * {{ position: relative; z-index: 1; }}
        .dashboard-title {{ font-size: 2.2rem; font-weight: 800; margin: 0 0 8px; letter-spacing: -0.02em;
            background: linear-gradient(135deg, #E6EAF2, #93C5FD); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
        .dashboard-subtitle {{ color: #93C5FD; font-size: 1.02rem; margin: 0; }}

        /* 3. Cards/KPI tiles */
        .enhanced-metric-card {{ border:1px solid #1F2937; border-radius:16px; padding:24px; background:#000000;
            box-shadow: 0 8px 16px -4px rgba(0,0,0,.25); height:100%; position:relative; overflow:hidden; }}
        .metric-label {{ color:#93C5FD; font-size:.85rem; font-weight:600; text-transform:uppercase; letter-spacing:.8px; margin-bottom:8px; }}
        .metric-value {{ font-size:2.2rem; font-weight:800; letter-spacing:-.02em; line-height:1; margin-bottom:8px; color: #E6EAF2; }}
        .metric-desc {{ color:#93C5FD; font-size:.82rem; font-weight:500; }}
        
        /* 6. Badges/status pills */
        .status-badge {{ display:inline-flex; align-items:center; gap:8px; padding:8px 14px; border-radius:999px; font-size:.9rem; font-weight:700; background: #000000; border: 1px solid #1F2937; }}

        /* 3 & 4. Section Cards & Tables */
        .section-card {{ border:1px solid #1F2937; border-radius:16px; padding:20px; background:#000000; }}
        .section-title {{ font-weight:800; font-size:1.1rem; margin-bottom:10px; color:#E6EAF2; }}
        
        /* Typography & Table Text */
        p, div, span, td, th {{ color: #E6EAF2; }}
        .stDataFrame, .stTable {{ background-color: #000000; }}
        thead th {{ background-color: #000000; color: #E6EAF2; }}
        tbody td {{ border-color: #1F2937; }}

        .check {{ font-weight:700; color:{SUCCESS_COLOR}; }}
        .warn {{ font-weight:700; color:{WARNING_COLOR}; }}
        .fail {{ font-weight:700; color:{DANGER_COLOR}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- Small helpers ----

def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "date", "Date", "timestamp", "time", "ds"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(s)))


def _pick_count_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["patient_count", "Target_1", "patients", "count", "y", "value"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# ---- Page body ----

def page_dashboard():
    _dashboard_css()

    datasets_loaded = sum([
        bool(st.session_state.get("patient_loaded")),
        bool(st.session_state.get("weather_loaded")),
        bool(st.session_state.get("calendar_loaded")),
    ])
    system_active = bool(st.session_state.get("patient_loaded"))

    # Use professional header component
    header(
        "Healthcare Forecasting Dashboard",
        "Real-time analytics and predictive insights for patient-arrival forecasting",
        icon="📊"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "Active" if system_active else "Offline"
        status_icon = "🟢" if system_active else "🔴"
        rgb = "16,185,129" if system_active else "239,68,68"
        color = SUCCESS_COLOR if system_active else DANGER_COLOR
        st.markdown(
            f"""
            <div class='enhanced-metric-card'>
              <div class='metric-label'>System Status</div>
              <div class='metric-value'>
                <span class='status-badge' style='background:rgba({rgb},.12);color:{color};border:1px solid rgba({rgb},.25);'>
                  {status_icon} {status}
                </span>
              </div>
              <div class='metric-desc'>Data pipeline connectivity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        df_p = st.session_state.get("patient_data")
        total_patients = 0
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            count_col = _pick_count_col(df_p)
            if count_col:
                total_patients = int(pd.to_numeric(df_p[count_col], errors="coerce").fillna(0).sum())
        st.markdown(
            f"""
            <div class='enhanced-metric-card'>
              <div class='metric-label'>Total Patients</div>
              <div class='metric-value' style='color: #E6EAF2;'>
                {total_patients:,}
              </div>
              <div class='metric-desc'>Sum across loaded history</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class='enhanced-metric-card'>
              <div class='metric-label'>Datasets</div>
              <div class='metric-value' style='color: #E6EAF2;'>
                {datasets_loaded}/3
              </div>
              <div class='metric-desc'>Patient • Weather • Calendar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        pipeline_complete = st.session_state.get("merged_data") is not None
        status_icon = "🟢" if pipeline_complete else "🟡"
        rgb = "16,185,129" if pipeline_complete else "245,158,11"
        color = SUCCESS_COLOR if pipeline_complete else WARNING_COLOR
        label = "Complete" if pipeline_complete else "Pending"
        st.markdown(
            f"""
            <div class='enhanced-metric-card'>
              <div class='metric-label'>Pipeline</div>
              <div class='metric-value'>
                <span class='status-badge' style='background:rgba({rgb},.12);color:{color};border:1px solid rgba({rgb},.25);'>
                  {status_icon} {label}
                </span>
              </div>
              <div class='metric-desc'>Data fusion status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("\n")

    # DATA HEALTH & RECENT ACTIVITY
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("<div class='section-title'>Data Health</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            date_col = _pick_date_col(df_p)
            date_range_txt = "N/A"
            if date_col:
                dt = _to_datetime_safe(df_p[date_col])
                if dt.notna().any():
                    date_range_txt = f"{dt.min().date()} → {dt.max().date()}"
            nulls = int(df_p.isna().sum().sum())
            cols = df_p.shape[1]
            rows = len(df_p)
            st.write(
                pd.DataFrame(
                    {"Indicator": ["Rows", "Columns", "Date Range", "Missing Values"],
                     "Value": [f"{rows:,}", f"{cols}", date_range_txt, f"{nulls:,}"]}
                )
            )
        else:
            st.info("Upload patient data to view schema, date range, and nulls.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-title'>Recent Activity</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        try:
            if isinstance(df_p, pd.DataFrame) and not df_p.empty:
                date_col = _pick_date_col(df_p)
                y_col = _pick_count_col(df_p)
                if date_col and y_col:
                    dt = _to_datetime_safe(df_p[date_col])
                    s = pd.to_numeric(df_p[y_col], errors="coerce")
                    recent = pd.DataFrame({"dt": dt, "y": s}).dropna().sort_values("dt")
                    if not recent.empty:
                        cutoff = recent["dt"].max() - pd.Timedelta(days=30)
                        recent_plot_df = recent[recent["dt"] >= cutoff].set_index("dt")[["y"]]
                        
                        # 5. Plotly chart theming
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=recent_plot_df.index, y=recent_plot_df['y'], mode='lines',
                                                 line=dict(color=PRIMARY_COLOR, width=2)))
                        fig.update_layout(
                            paper_bgcolor="rgba(11,15,25,1)",
                            plot_bgcolor="rgba(11,15,25,1)",
                            font_color="#E6EAF2",
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=150
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("No valid time series points detected yet.")
                else:
                    st.caption("Waiting for a valid date/count column to render a sparkline.")
            else:
                st.caption("Data not loaded.")
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            st.caption(f"Unable to draw sparkline: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # MODEL READINESS
    st.markdown("<div class='section-title'>Model Readiness</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    checks = [
        ("Patient data loaded", bool(st.session_state.get("patient_loaded"))),
        ("Weather data loaded", bool(st.session_state.get("weather_loaded"))),
        ("Calendar data loaded", bool(st.session_state.get("calendar_loaded"))),
        ("Data fusion complete", st.session_state.get("merged_data") is not None),
    ]
    for label, ok in checks:
        cls = "check" if ok else ("warn" if label != "Data fusion complete" else "fail")
        icon = "✅" if ok else ("⏳" if label != "Data fusion complete" else "⚠️")
        st.markdown(f"- <span class='{cls}'>{icon}</span> {label}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # QUICK INSIGHTS
    st.markdown("<div class='section-title'>Quick Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    if isinstance(df_p, pd.DataFrame) and not df_p.empty:
        y_col = _pick_count_col(df_p)
        date_col = _pick_date_col(df_p)
        if y_col and date_col:
            dfq = pd.DataFrame({
                "dt": _to_datetime_safe(df_p[date_col]),
                "y": pd.to_numeric(df_p[y_col], errors="coerce")
            }).dropna().sort_values("dt")
            if not dfq.empty:
                mean_v = float(dfq["y"].mean())
                std_v = float(dfq["y"].std(ddof=0))
                ratio = std_v / mean_v if mean_v > 0 else np.nan
                hint = ""
                if len(dfq) >= 14:
                    try:
                        w = dfq.copy()
                        w["dow"] = w["dt"].dt.dayofweek
                        wd = w[w["dow"] < 5]["y"].mean()
                        we = w[w["dow"] >= 5]["y"].mean()
                        if np.isfinite(wd) and np.isfinite(we) and abs(wd - we) / (mean_v + 1e-9) > 0.05:
                            hint = "Possible weekday/weekend seasonality detected."
                    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
                        pass
                st.markdown(
                    f"**Mean arrivals:** {mean_v:,.1f}  |  **Volatility (σ/μ):** {ratio:,.2f}  " + (f"|  **Signal:** {hint}" if hint else "")
                )
            else:
                st.caption("Not enough valid points for insights.")
        else:
            st.caption("Insights will appear once a date and count column are detected.")
    else:
        st.caption("Load data to compute summary insights.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Entrypoint ----

def main():
    """Main function to run the dashboard page."""
    inject_sidebar_style()
    render_sidebar_brand()
    apply_css()
    init_state()
    page_dashboard()

if __name__ == "__main__":
    main()