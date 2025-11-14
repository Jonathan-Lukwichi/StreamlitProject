# ===========================================
# pages/05_Benchmarks.py — Benchmark Models (Enhanced Multi-Horizon)
# ===========================================

from __future__ import annotations

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from app_core.ui.theme import (
    apply_css,
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, CARD_BG, BG_BLACK, BODY_TEXT
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.state.session import init_state

# Plot builders for multi-horizon results (expects matplotlib figs)
from app_core.plots import build_multihorizon_results_dashboard

# === SARIMAX multi-horizon (canonical artifact producer) ===
from app_core.models.sarimax_pipeline import (
    run_sarimax_multihorizon,
    to_multihorizon_artifacts,
)

# === ARIMA helpers (with graceful fallbacks for plotting) ===
from app_core.models.arima_pipeline import run_arima_pipeline
try:
    from app_core.plots import plot_arima_forecast
    _plt_import_ok = True
except ImportError:
    _plt_import_ok = False


# ---------------------------- Small utilities ----------------------------
_DEF_FLOAT_FMT = {
    "MAE": "{:.3f}",
    "RMSE": "{:.3f}",
    "MAPE_%": "{:.2f}",
    "Accuracy_%": "{:.2f}"
}

def _safe_float(x):
    try:
        f = float(x)
        if np.isfinite(f):
            return f
        return np.nan
    except Exception:
        return np.nan

def _mean_from_any(df: pd.DataFrame, candidates: list[str]) -> float:
    """Return mean of first available candidate column (numeric), or NaN."""
    if df is None or df.empty:
        return np.nan
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s[np.isfinite(s)]
            if s.notna().any():
                return float(s.mean())
    return np.nan

def _fmt_num(x, fmt="{:.3f}", na="—"):
    x = _safe_float(x)
    return fmt.format(x) if np.isfinite(x) else na

def _fmt_pct(x, na="—"):
    x = _safe_float(x)
    return f"{x:.2f}%" if np.isfinite(x) else na

def _sanitize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure metrics_df has numeric Horizon and numeric metric columns.
    - Coerce Horizon by extracting digits (handles 'h=1', 'H1', etc.)
    - Drop rows without a numeric Horizon
    - Convert all known metric columns to numeric with coercion and NaN-safe
    """
    if df is None or df.empty:
        return df

    work = df.copy()

    # Fix Horizon
    if "Horizon" in work.columns:
        # extract first integer sequence; e.g. 'h=1' -> 1
        hor = (
            work["Horizon"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
        work["Horizon"] = hor
        work = work.dropna(subset=["Horizon"])
        work["Horizon"] = work["Horizon"].astype(int)

    # Known metric columns to coerce
    cols = [
        "Train_MAE","Train_RMSE","Train_MAPE","Train_Acc",
        "Test_MAE","Test_RMSE","Test_MAPE","Test_Acc",
        "MAE","RMSE","MAPE_%","Accuracy_%","AIC","BIC"
    ]
    for c in cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    return work

def _sanitize_artifacts(F, L, U):
    """Replace infinities with NaN and ensure arrays are float64."""
    def _san(a):
        if a is None:
            return None
        arr = np.array(a, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr
    return _san(F), _san(L), _san(U)

# ---------------------------- Local fallback plotters (ARIMA single) ----------------------------

def _plot_arima_forecast_local(arima_out, title="ARIMA Forecast"):
    y_train = arima_out.get("y_train")
    y_test  = arima_out.get("y_test")
    
    y_pred = arima_out.get("forecasts")
    if y_pred is None:
        y_pred = arima_out.get("y_pred")
        
    ci_low = arima_out.get("forecast_lower")
    if ci_low is None:
        ci_low = arima_out.get("ci_lower")
        
    ci_up = arima_out.get("forecast_upper")
    if ci_up is None:
        ci_up = arima_out.get("ci_upper")

    fig = go.Figure()

    # Add confidence interval first (so it's behind)
    if (ci_low is not None) and (ci_up is not None):
        try:
            if len(ci_low) == len(ci_up):
                fig.add_trace(go.Scatter(
                    x=getattr(ci_up, "index", np.arange(len(ci_up))), y=ci_up,
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=getattr(ci_low, "index", np.arange(len(ci_low))), y=ci_low,
                    mode="lines", fill="tonexty", fillcolor="rgba(102,126,234,0.15)",
                    line=dict(width=0), name="95% CI", showlegend=True,
                    hovertemplate='CI: %{y:.1f}<extra></extra>'
                ))
        except Exception:
            pass

    # Training data
    if y_train is not None and len(y_train) > 0:
        fig.add_trace(go.Scatter(
            x=getattr(y_train, "index", np.arange(len(y_train))), y=y_train,
            name="Training Data",
            line=dict(width=2, color=SUBTLE_TEXT, dash='dot'),
            mode='lines',
            hovertemplate='Train: %{y:.1f}<extra></extra>'
        ))

    # Actual test data
    if y_test is not None and len(y_test) > 0:
        fig.add_trace(go.Scatter(
            x=getattr(y_test, "index", np.arange(len(y_test))), y=y_test,
            name="Actual (Test)",
            line=dict(width=2.5, color=TEXT_COLOR),
            mode='lines+markers',
            marker=dict(size=5),
            hovertemplate='Actual: %{y:.1f}<extra></extra>'
        ))

    # Forecast
    if y_pred is not None and len(y_pred) > 0:
        fig.add_trace(go.Scatter(
            x=getattr(y_pred, "index", np.arange(len(y_pred))), y=y_pred,
            name="Forecast",
            line=dict(width=2.5, color=PRIMARY_COLOR),
            mode='lines+markers',
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='Forecast: %{y:.1f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18, color=TEXT_COLOR)),
        margin=dict(t=60, b=60, l=60, r=40),
        hovermode="x unified",
        xaxis=dict(title="Date", title_font=dict(size=14, color=TEXT_COLOR),
                   showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        yaxis=dict(title="Number of Patient Arrivals", title_font=dict(size=14, color=TEXT_COLOR),
                   showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def _plot_arima_residuals_local(arima_out):
    resid = arima_out.get("residuals")
    fig = go.Figure()

    if resid is None or len(resid) == 0:
        fig.update_layout(title="Residual diagnostics unavailable")
        return fig

    x_idx = getattr(resid, "index", np.arange(len(resid)))

    # Residuals line
    fig.add_trace(go.Scatter(
        x=x_idx, y=np.array(resid),
        mode="lines+markers",
        name="Residuals",
        line=dict(color=SECONDARY_COLOR, width=1.5),
        marker=dict(size=3),
        hovertemplate='Residual: %{y:.2f}<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5, opacity=0.7)

    fig.update_layout(
        title=dict(text="ARIMA Residuals", x=0.5, xanchor='center', font=dict(size=18, color=TEXT_COLOR)),
        hovermode="x unified",
        margin=dict(t=60, b=60, l=60, r=40),
        xaxis=dict(title="Date", title_font=dict(size=14, color=TEXT_COLOR),
                   showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        yaxis=dict(title="Residual Value", title_font=dict(size=14, color=TEXT_COLOR),
                   showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG
    )
    return fig

# ---------------------------- KPI helpers ----------------------------

def _indicator(title: str, value, suffix: str = "", color: str = PRIMARY_COLOR):
    try:
        is_num = isinstance(value, (int, float, np.floating)) and np.isfinite(value)
    except Exception:
        is_num = False

    display_val = float(value) if is_num else 0.0
    number_color = color if is_num else "rgba(0,0,0,0)"

    fig = go.Figure(go.Indicator(
        mode="number",
        value=display_val,
        number={"suffix": suffix, "font": {"size": 32, "color": number_color, "family": "Arial Black"}},
        title={"text": f"<span style='color:{TEXT_COLOR};font-size:14px;font-weight:600'>{title}</span>"},
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=CARD_BG,
        height=150
    )

    if not is_num:
        fig.add_annotation(
            x=0.5, y=0.45,
            text="—",
            showarrow=False,
            font=dict(size=28, color=color, family="Arial")
        )

    return fig

def _render_kpi_row(metrics: dict):
    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")
    mape = metrics.get("MAPE")
    acc = metrics.get("Accuracy") or (100 - mape if isinstance(mape, (int, float)) and np.isfinite(mape) else None)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.plotly_chart(_indicator("MAE", mae, "", PRIMARY_COLOR), use_container_width=True)
    with c2: st.plotly_chart(_indicator("RMSE", rmse, "", SECONDARY_COLOR), use_container_width=True)
    with c3: st.plotly_chart(_indicator("MAPE", mape, "%", WARNING_COLOR), use_container_width=True)
    with c4: st.plotly_chart(_indicator("Accuracy", acc, "%", SUCCESS_COLOR), use_container_width=True)

# ---------------------------- Comparison logging ----------------------------

def _init_comparison_table():
    if "model_comparison" not in st.session_state or not isinstance(st.session_state["model_comparison"], pd.DataFrame):
        st.session_state["model_comparison"] = pd.DataFrame(columns=[
            "Timestamp", "Model", "TrainRatio",
            "MAE", "RMSE", "MAPE_%", "Accuracy_%",
            "Runtime_s", "ModelParams"
        ])

def _append_to_comparison(model_name: str, train_ratio: float, kpis: dict, model_params: str, runtime_s: float | None = None):
    _init_comparison_table()
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "TrainRatio": f"{int(train_ratio*100)}%",
        "MAE": _safe_float(kpis.get("MAE")),
        "RMSE": _safe_float(kpis.get("RMSE")),
        "MAPE_%": _safe_float(kpis.get("MAPE")),
        "Accuracy_%": _safe_float(kpis.get("Accuracy")) if kpis.get("Accuracy") is not None else (
            100.0 - _safe_float(kpis.get("MAPE")) if np.isfinite(_safe_float(kpis.get("MAPE"))) else np.nan
        ),
        "Runtime_s": round(float(runtime_s), 3) if runtime_s is not None else np.nan,
        "ModelParams": model_params
    }
    st.session_state["model_comparison"] = pd.concat(
        [st.session_state["model_comparison"], pd.DataFrame([row])],
        ignore_index=True
    )

# ---------------------------- ARIMA multi-horizon ----------------------------

def run_arima_multihorizon(
    df: pd.DataFrame,
    date_col: str = "Date",
    horizons: int = 7,
    train_ratio: float = 0.8,
    order: tuple[int,int,int] | None = None,
    auto_select: bool = True,
    cv_strategy: str = "expanding",
) -> dict:
    """Train independent univariate ARIMA models for Target_1..Target_H."""
    rows, per_h, ok = [], {}, []

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)

    for h in range(1, int(horizons) + 1):
        tc = f"Target_{h}"
        if tc not in frame.columns:
            continue
        try:
            out = run_arima_pipeline(
                df=frame[[date_col, tc]],
                order=order,
                train_ratio=train_ratio,
                auto_select=auto_select,
                target_col=tc,
            )
        except Exception:
            continue

        y_tr = out.get("y_train"); y_te = out.get("y_test")
        fc   = out.get("forecasts"); lo = out.get("forecast_lower"); hi = out.get("forecast_upper")
        info = out.get("model_info", {}) or {}

        tm = out.get("test_metrics", {}) or {}
        mae  = _safe_float(tm.get("MAE"))
        rmse = _safe_float(tm.get("RMSE"))
        mape = _safe_float(tm.get("MAPE"))
        acc  = _safe_float(tm.get("Accuracy")) if tm.get("Accuracy") is not None else (100.0 - mape if np.isfinite(mape) else np.nan)

        rows.append({
            "Horizon": h,
            "Order": str(info.get("order", order if order else "auto")),
            "Train_MAE": _safe_float(out.get("train_metrics", {}).get("MAE")),
            "Train_RMSE": _safe_float(out.get("train_metrics", {}).get("RMSE")),
            "Train_MAPE": _safe_float(out.get("train_metrics", {}).get("MAPE")),
            "Train_Acc": _safe_float(out.get("train_metrics", {}).get("Accuracy")),
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_MAPE": mape,
            "Test_Acc": acc,
            "Train_N": int(len(y_tr) if y_tr is not None else 0),
            "Test_N": int(len(y_te) if y_te is not None else 0),
        })

        per_h[h] = {
            "y_train": y_tr, "y_test": y_te,
            "forecast": fc, "ci_lo": lo, "ci_hi": hi,
            "order": info.get("order", order if order else "auto"),
            "sorder": None,
            "res": out.get("fit_results") or out.get("model") or None,
        }
        ok.append(h)

    results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)
    return {"results_df": results_df, "per_h": per_h, "successful": ok}

def arima_to_multihorizon_artifacts(arima_out: dict):
    # use SARIMAX artifact shaper; it already matches the dashboard’s expectations
    return to_multihorizon_artifacts(arima_out)

# ---------------------------- Enhanced visualization helpers ----------------------------

def _styled_core_metrics_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Compact train/test metrics table with enhanced styling."""
    df = _sanitize_metrics_df(df)
    if df is None or df.empty:
        return pd.DataFrame({"Notice": ["No metrics available"]}).style

    test_mae  = "Test_MAE"  if "Test_MAE"  in df.columns else ("MAE"        if "MAE"        in df.columns else None)
    test_rmse = "Test_RMSE" if "Test_RMSE" in df.columns else ("RMSE"       if "RMSE"       in df.columns else None)
    test_mape = "Test_MAPE" if "Test_MAPE" in df.columns else ("MAPE_%"     if "MAPE_%"     in df.columns else None)
    test_acc  = "Test_Acc"  if "Test_Acc"  in df.columns else ("Accuracy_%" if "Accuracy_%" in df.columns else None)

    cols = ["Horizon", "Train_MAE", "Train_RMSE", "Train_MAPE", "Train_Acc"]
    if test_mae:  cols.append(test_mae)
    if test_rmse: cols.append(test_rmse)
    if test_mape: cols.append(test_mape)
    if test_acc:  cols.append(test_acc)
    sub = df[[c for c in cols if c in df.columns]].copy()

    ren = {"Train_MAE": "Train MAE", "Train_RMSE": "Train RMSE", "Train_MAPE": "Train MAPE%", "Train_Acc": "Train Acc%"}
    if test_mae:  ren[test_mae]  = "Test MAE"
    if test_rmse: ren[test_rmse] = "Test RMSE"
    if test_mape: ren[test_mape] = "Test MAPE%"
    if test_acc:  ren[test_acc]  = "Test Acc%"
    sub = sub.rename(columns=ren)

    sty = sub.style.format({
        "Train MAE": "{:.3f}", "Train RMSE": "{:.3f}", "Train MAPE%": "{:.2f}", "Train Acc%": "{:.2f}",
        "Test MAE":  "{:.3f}", "Test RMSE":  "{:.3f}", "Test MAPE%":  "{:.2f}", "Test Acc%":  "{:.2f}",
    }, na_rep="—").set_properties(**{
        "text-align": "right",
        "padding": "8px 12px",
        "border": "1px solid #e0e0e0"
    }).set_table_styles([
        {"selector": "th", "props": "text-align:left; background-color:#f8f9fa; font-weight:600; padding:10px;"},
        {"selector": "td", "props": "border: 1px solid #e0e0e0;"},
        {"selector": "", "props": "border-collapse: collapse;"}
    ])
    return sty

def _build_performance_heatmap(df: pd.DataFrame, title: str):
    """Enhanced performance heatmap with proper horizon handling."""
    df_work = _sanitize_metrics_df(df)
    if df_work is None or df_work.empty or "Horizon" not in df_work.columns:
        fig = go.Figure()
        fig.update_layout(title="No performance data available",
                          xaxis_title="Horizon", yaxis_title="Metric")
        return fig

    test_mae_col = "Test_MAE" if "Test_MAE" in df_work.columns else "MAE"
    test_rmse_col = "Test_RMSE" if "Test_RMSE" in df_work.columns else "RMSE"
    test_mape_col = "Test_MAPE" if "Test_MAPE" in df_work.columns else ("MAPE_%" if "MAPE_%" in df_work.columns else "MAPE")
    test_acc_col = "Test_Acc" if "Test_Acc" in df_work.columns else ("Accuracy_%" if "Accuracy_%" in df_work.columns else "Accuracy")

    heatmap_data = {}
    if test_mae_col in df_work.columns:
        heatmap_data["MAE"] = df_work.set_index("Horizon")[test_mae_col]
    if test_rmse_col in df_work.columns:
        heatmap_data["RMSE"] = df_work.set_index("Horizon")[test_rmse_col]
    if test_mape_col in df_work.columns:
        heatmap_data["MAPE%"] = df_work.set_index("Horizon")[test_mape_col]
    if test_acc_col in df_work.columns:
        heatmap_data["Accuracy%"] = df_work.set_index("Horizon")[test_acc_col]

    if not heatmap_data:
        fig = go.Figure()
        fig.update_layout(title="No performance data available",
                          xaxis_title="Horizon", yaxis_title="Metric")
        return fig

    hm = pd.DataFrame(heatmap_data).T

    fig = px.imshow(
        hm,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        title=title,
        labels=dict(x="Horizon (Days)", y="Metric", color="Value"),
        text_auto=".2f"
    )

    fig.update_layout(
        margin=dict(l=60, r=40, t=70, b=60),
        title=dict(x=0.5, xanchor='center', font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(title_font=dict(size=13)),
        yaxis=dict(title_font=dict(size=13)),
        coloraxis_colorbar=dict(title="Value", len=0.7)
    )

    return fig

def _build_error_evolution(per_h: dict, title: str):
    """Enhanced error evolution chart with better styling."""
    df_list = []
    for h, d in per_h.items():
        y_te = d.get("y_test")
        fc   = d.get("forecast")
        if y_te is None or fc is None or len(y_te) == 0:
            continue
        try:
            idx = y_te.index
            ae = (y_te.reindex(idx) - fc.reindex(idx)).abs()
            df_list.append(pd.DataFrame({"Date": idx, "AbsError": ae.values, "Horizon": f"h={h}"}))
        except Exception:
            continue

    if not df_list:
        fig = go.Figure()
        fig.update_layout(title="No error data available for plotting")
        return fig

    err = pd.concat(df_list).dropna()

    fig = px.line(
        err,
        x="Date",
        y="AbsError",
        color="Horizon",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=4)
    )

    fig.update_layout(
        margin=dict(l=60, r=40, t=70, b=60),
        title=dict(x=0.5, xanchor='center', font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(
            title="Date",
            title_font=dict(size=13),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            title="Absolute Error (Patient Arrivals)",
            title_font=dict(size=13),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            title="Forecast Horizon",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )

    return fig

# =====================================================================
# PAGE
# =====================================================================

# Page Configuration
st.set_page_config(
    page_title="Benchmarks - HealthForecast AI",
    page_icon="🤖",
    layout="wide",
)

def page_benchmarks():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()
    init_state()

    # Fluorescent effects + Custom Tab Styling
    st.markdown(f"""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR BENCHMARKS
       ======================================== */

    @keyframes float-orb {{
        0%, 100% {{
            transform: translate(0, 0) scale(1);
            opacity: 0.25;
        }}
        50% {{
            transform: translate(30px, -30px) scale(1.05);
            opacity: 0.35;
        }}
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
        0%, 100% {{
            opacity: 0;
            transform: scale(0);
        }}
        50% {{
            opacity: 0.6;
            transform: scale(1);
        }}
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
        padding-top: 2rem;
    }}

    @media (max-width: 768px) {{
        .fluorescent-orb {{
            width: 200px !important;
            height: 200px !important;
            filter: blur(50px);
        }}
        .sparkle {{
            display: none;
        }}

        /* Make tabs stack vertically on mobile */
        .stTabs [data-baseweb="tab-list"] {{
            flex-direction: column;
        }}

        .stTabs [data-baseweb="tab"] {{
            width: 100%;
        }}
    }}
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
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 1rem; padding: 1.5rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2.5rem;'>🤖</div>
          <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Benchmark Models</h1>
          <p class='hf-feature-description' style='font-size: 1rem; max-width: 700px; margin: 0 auto;'>
            Train and evaluate forecasting models for patient arrivals with advanced statistical and machine learning techniques
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prefer preprocessed data; fallback to merged
    processed = st.session_state.get("processed_df")
    merged    = st.session_state.get("merged_data")

    if processed is not None and not processed.empty:
        data = processed.copy()
        st.caption("Using preprocessed dataset from **Data Preparation Studio**.")
    elif merged is not None and not merged.empty:
        data = merged.copy()
        st.warning("Using **merged_data** (not fully preprocessed). For best results, run Preprocess in Data Studio.")
    else:
        st.error("No dataset found. Go to **Data Hub** → **Data Preparation Studio** to create a preprocessed dataset.")
        return

    if "Date" not in data.columns:
        st.error("Dataset must contain a 'Date' column. Ensure preprocessing created it.")
        return
    if "Target_1" not in data.columns:
        st.error("Dataset must contain 'Target_1'. Build targets in Data Studio (set lags/targets).")
        return

    # ---------- Dataset Summary ----------
    st.markdown("### 📋 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="hf-section-card">
                <div class="hf-section-body">Total Records</div>
                <div class="hf-section-title" style="color:{PRIMARY_COLOR};">{len(data):,}</div>
                <div class="hf-section-body">Time Points</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        date_range = f"{pd.to_datetime(data['Date']).min().date()} to {pd.to_datetime(data['Date']).max().date()}"
        st.markdown(f"""
            <div class="hf-section-card">
                <div class="hf-section-body">Date Range</div>
                <div class="hf-section-title" style="color:{SECONDARY_COLOR};">{date_range}</div>
                <div class="hf-section-body">Coverage Period</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_arrivals = pd.to_numeric(data['Target_1'], errors='coerce').mean()
        st.markdown(f"""
            <div class="hf-section-card">
                <div class="hf-section-body">Avg Arrivals</div>
                <div class="hf-section-title" style="color:{SUCCESS_COLOR};">{avg_arrivals:.1f}</div>
                <div class="hf-section-body">Per Day</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        features_count = len([c for c in data.columns if c not in ['Date'] and not c.startswith('Target_')])
        st.markdown(f"""
            <div class="hf-section-card">
                <div class="hf-section-body">Features</div>
                <div class="hf-section-title" style="color:{WARNING_COLOR};">{max(0, features_count)}</div>
                <div class="hf-section-body">Variables</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    tab_select, tab_config, tab_train, tab_results, tab_compare = st.tabs(
        ["🎯 Model Selection", "⚙️ Configuration", "🚀 Train Model", "📊 Results", "🆚 Model Comparison"]
    )

    # ------------------------------------------------------------
    # 🎯 Model Selection
    # ------------------------------------------------------------
    with tab_select:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(34, 211, 238, 0.1)); border: 2px solid rgba(59, 130, 246, 0.3); box-shadow: 0 0 40px rgba(59, 130, 246, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.6));'>🎯</div>
              <h1 style='font-size: 1.75rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);'>Model Selection</h1>
              <p style='font-size: 1rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Choose a time series forecasting model to predict patient arrivals</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_model1, col_model2 = st.columns(2)

        with col_model1:
            st.markdown(f"""
                <div class="hf-section-card" style="min-height:250px;">
                    <h4 class="hf-section-title" style="color:{PRIMARY_COLOR};">📈 ARIMA</h4>
                    <p class="hf-section-body">
                        <strong>AutoRegressive Integrated Moving Average</strong><br><br>
                        ✓ Classic univariate model<br>
                        ✓ Handles trends/seasonality (via differencing)<br>
                        ✓ Simple and robust<br><br>
                        <em>Use when exogenous vars are not critical</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select ARIMA", use_container_width=True, key="btn_arima"):
                st.session_state["selected_model"] = "ARIMA"
                st.success("✅ ARIMA model selected!")
                st.rerun()

        with col_model2:
            st.markdown(f"""
                <div class="hf-section-card" style="min-height:250px;">
                    <h4 class="hf-section-title" style="color:{SECONDARY_COLOR};">📊 SARIMAX</h4>
                    <p class="hf-section-body">
                        <strong>Seasonal ARIMA with eXogenous variables</strong><br><br>
                        ✓ Explicit seasonality<br>
                        ✓ Add external (weather/calendar) features<br>
                        ✓ Powerful for complex patterns<br>
                        ✓ Simple and robust<br><br>
                        <em>Use when exogenous vars matter</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select SARIMAX", use_container_width=True, key="btn_sarimax"):
                st.session_state["selected_model"] = "SARIMAX"
                st.success("✅ SARIMAX model selected!")
                st.rerun()

        st.markdown("---")
        current_model = st.session_state.get("selected_model")
        if current_model:
            st.info(f"🎯 **Currently Selected Model:** {current_model}")
        else:
            st.warning("⚠️ No model selected yet.")

    # ------------------------------------------------------------
    # ⚙️ Configuration
    # ------------------------------------------------------------
    with tab_config:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(59, 130, 246, 0.1)); border: 2px solid rgba(34, 211, 238, 0.3); box-shadow: 0 0 40px rgba(34, 211, 238, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(34, 211, 238, 0.6));'>⚙️</div>
              <h1 style='font-size: 1.75rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(34, 211, 238, 0.3);'>Model Configuration</h1>
              <p style='font-size: 1rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Fine-tune model parameters and feature selection for optimal performance</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_model = st.session_state.get("selected_model")
        if not current_model:
            st.warning("⚠️ Please select a model first in **Model Selection**.")
            return

        st.info(f"🎯 Configuring: **{current_model}**")

        if current_model == "ARIMA":
            mode = st.radio("Parameter Mode", ["Automatic (recommended)", "Manual (enter p,d,q)"],
                            horizontal=True, key="arima_mode")
            max_h_present = max([int(c.split("_")[1]) for c in data.columns if c.startswith("Target_")] or [1])
            arima_h = st.number_input("Max Forecast Horizon (days)", 1, max(30, max_h_present), min(7, max_h_present), step=1, key="arima_h")
            st.session_state["cv_strategy"] = st.radio("Validation Strategy", ["expanding", "rolling"], index=0, horizontal=True, key="cv_strategy_arima")
            if mode.startswith("Automatic"):
                st.markdown("ARIMA parameters will be found automatically (AIC-driven).")
                st.session_state["arima_order"] = None
            else:
                c1, c2, c3 = st.columns(3)
                with c1: p = st.number_input("p (AR order)", 0, 10, 1, key="arima_p")
                with c2: d = st.number_input("d (Differencing)", 0, 3, 1, key="arima_d")
                with c3: q = st.number_input("q (MA order)", 0, 10, 1, key="arima_q")
                st.session_state["arima_order"] = (p, d, q)
                st.info(f"📊 ARIMA Order set to: ({p}, {d}, {q})")

        elif current_model == "SARIMAX":
            all_candidates = [c for c in data.columns if c not in ["Date"] and not c.startswith("Target_")]
            st.markdown("#### Exogenous Feature Selection")
            select_all = st.checkbox("Select all features", value=True, key="sarimax_select_all")
            default_feats = all_candidates if select_all else st.session_state.get("sarimax_features", [])
            chosen = st.multiselect("Choose exogenous variables", options=all_candidates, default=default_feats)
            st.session_state["sarimax_features"] = chosen

            st.markdown("---")
            st.markdown("#### Parameter Mode")
            mode = st.radio("Parameter Mode", ["Automatic (recommended)", "Manual (enter orders)"],
                            horizontal=True, key="sarimax_mode")

            cA, cB = st.columns(2)
            with cA:
                season_length = st.number_input("Season Length s", 2, 365, 7, step=1,
                                                help="7 for weekly seasonality (daily data)")
            with cB:
                max_h_present = max([int(c.split("_")[1]) for c in data.columns if c.startswith("Target_")] or [1])
                horizons = st.number_input("Max Forecast Horizon (days)", 1, max(30, max_h_present), min(7, max_h_present), step=1)

            st.session_state["cv_strategy"] = st.radio("Validation Strategy", ["expanding", "rolling"], index=0, horizontal=True, key="cv_strategy_sarimax")

            if mode.startswith("Automatic"):
                st.session_state["sarimax_order"] = None
                st.session_state["sarimax_seasonal_order"] = None
                st.info("Order and seasonal_order will be chosen automatically by the pipeline (or defaults).")
            else:
                st.markdown("##### Non-Seasonal Order (p,d,q)")
                c1, c2, c3 = st.columns(3)
                with c1: p = st.number_input("p", 0, 10, 1, key="sx_p")
                with c2: d = st.number_input("d", 0, 2, 1, key="sx_d")
                with c3: q = st.number_input("q", 0, 10, 1, key="sx_q")
                st.markdown("##### Seasonal Order (P,D,Q,s)")
                c4, c5, c6, c7 = st.columns(4)
                with c4: P = st.number_input("P", 0, 5, 1, key="sx_P")
                with c5: D = st.number_input("D", 0, 2, 1, key="sx_D")
                with c6: Q = st.number_input("Q", 0, 5, 1, key="sx_Q")
                with c7: s = st.number_input("s (season length)", 2, 365, int(season_length), key="sx_s")
                st.session_state["sarimax_order"] = (p, d, q)
                st.session_state["sarimax_seasonal_order"] = (P, D, Q, s)
                st.info(f"📊 SARIMAX Orders set to: order=({p},{d},{q})  seasonal=({P},{D},{Q},{s})")

            st.session_state["sarimax_season_length"] = int(season_length)
            st.session_state["sarimax_horizons"] = int(horizons)

    # ------------------------------------------------------------
    # 🚀 Train Model
    # ------------------------------------------------------------
    with tab_train:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1)); border: 2px solid rgba(34, 197, 94, 0.3); box-shadow: 0 0 40px rgba(34, 197, 94, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(34, 197, 94, 0.6));'>🚀</div>
              <h1 style='font-size: 1.75rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, {SUCCESS_COLOR}, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(34, 197, 94, 0.3);'>Train Model</h1>
              <p style='font-size: 1rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Execute model training and generate forecasts for patient arrivals</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_model = st.session_state.get("selected_model")
        if not current_model:
            st.warning("⚠️ Select a model first.")
            return

        train_size = st.slider("Training Data (%)", 50, 95, 80, 5, key="train_ratio_all")
        train_ratio = train_size / 100.0
        st.caption(f"Train/Test split: {train_size}% / {100-train_size}%")

        if current_model == "ARIMA":
            order = st.session_state.get("arima_order")
            use_auto = order is None
            arima_h = int(st.session_state.get("arima_h", 7))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Start Training (ARIMA — h=1)", use_container_width=True, type="primary", key="train_arima_btn"):
                    with st.spinner("Training ARIMA (single horizon h=1)..."):
                        t0 = time.time()
                        try:
                            results = run_arima_pipeline(
                                df=data,
                                order=order,
                                train_ratio=train_ratio,
                                auto_select=use_auto,
                                target_col="Target_1",
                            )
                            runtime_s = time.time() - t0
                            st.session_state["arima_results"] = results
                            st.session_state["model_results"] = results
                            st.success(f"✅ ARIMA (h=1) training completed in {runtime_s:.2f}s!")

                            order_str = str(results.get("model_info", {}).get("order", order if order else "auto"))
                            model_params = f"order={order_str}"

                            test_m = results.get("test_metrics", {}) or {}
                            _append_to_comparison(
                                model_name="ARIMA (h=1)",
                                train_ratio=train_ratio,
                                kpis={
                                    "MAE": test_m.get("MAE"),
                                    "RMSE": test_m.get("RMSE"),
                                    "MAPE": test_m.get("MAPE"),
                                    "Accuracy": test_m.get("Accuracy"),
                                },
                                model_params=model_params,
                                runtime_s=runtime_s
                            )
                        except Exception as e:
                            st.error(f"❌ ARIMA training failed: {e}")
            with c2:
                if st.button("🚀 Train ARIMA (Multi-horizon)", use_container_width=True, type="secondary", key="train_arima_mh_btn"):
                    with st.spinner(f"Training ARIMA multi-horizon (h=1..{arima_h})..."):
                        t0 = time.time()
                        try:
                            arima_mh_out = run_arima_multihorizon(
                                df=data,
                                date_col="Date",
                                horizons=int(arima_h),
                                train_ratio=train_ratio,
                                order=order,
                                auto_select=use_auto,
                                cv_strategy=st.session_state.get("cv_strategy", "expanding"),
                            )
                            runtime_s = time.time() - t0
                            st.session_state["arima_mh_results"] = arima_mh_out
                            st.success(f"✅ ARIMA multi-horizon training completed in {runtime_s:.2f}s!")

                            res_df = arima_mh_out.get("results_df")
                            if res_df is not None and not res_df.empty:
                                res_df = _sanitize_metrics_df(res_df)
                                best_idx = res_df["Test_Acc"].idxmax() if "Test_Acc" in res_df.columns else res_df.index[0]
                                best_row = res_df.loc[best_idx]
                                kpis = {
                                    "MAE": _safe_float(best_row.get("Test_MAE")),
                                    "RMSE": _safe_float(best_row.get("Test_RMSE")),
                                    "MAPE": _safe_float(best_row.get("Test_MAPE")),
                                    "Accuracy": _safe_float(best_row.get("Test_Acc")),
                                }
                                model_params = f"order={order if order else 'auto'}; H={arima_h}"
                                _append_to_comparison(
                                    model_name=f"ARIMA (best h={int(best_row.get('Horizon', 1))})",
                                    train_ratio=train_ratio,
                                    kpis=kpis,
                                    model_params=model_params,
                                    runtime_s=runtime_s
                                )
                        except Exception as e:
                            st.error(f"❌ ARIMA multi-horizon training failed: {e}")

        elif current_model == "SARIMAX":
            exog_vars      = st.session_state.get("sarimax_features", [])
            season_len     = int(st.session_state.get("sarimax_season_length", 7))
            horizons       = int(st.session_state.get("sarimax_horizons", 7))
            order          = st.session_state.get("sarimax_order")
            seasonal_order = st.session_state.get("sarimax_seasonal_order")

            if st.button("🚀 Train SARIMAX (multi-horizon)", use_container_width=True, type="primary", key="train_sarimax_btn"):
                t0 = time.time()
                try:
                    kwargs = dict(
                        df_merged=data,
                        date_col="Date",
                        train_ratio=train_ratio,
                        season_length=season_len,
                        horizons=horizons,
                        use_all_features=(len(exog_vars) == 0),
                        include_dow_ohe=True,
                        selected_features=exog_vars if exog_vars else None,
                    )
                    if order is not None and seasonal_order is not None:
                        kwargs.update(dict(order=order, seasonal_order=seasonal_order))

                    try:
                        sarimax_out = run_sarimax_multihorizon(**kwargs)
                    except TypeError:
                        kwargs.pop("order", None)
                        kwargs.pop("seasonal_order", None)
                        kwargs.pop("selected_features", None)
                        sarimax_out = run_sarimax_multihorizon(**kwargs)

                    runtime_s = time.time() - t0
                    st.session_state["sarimax_results"] = sarimax_out
                    st.success(f"✅ SARIMAX trained in {runtime_s:.2f}s.")

                    res_df = sarimax_out.get("results_df")
                    if res_df is not None and not res_df.empty:
                        res_df = _sanitize_metrics_df(res_df)
                        best_idx = res_df["Test_Acc"].idxmax() if "Test_Acc" in res_df.columns else res_df.index[0]
                        best_row = res_df.loc[best_idx]
                        kpis = {
                            "MAE": _safe_float(best_row.get("Test_MAE") if "Test_MAE" in res_df.columns else best_row.get("MAE")),
                            "RMSE": _safe_float(best_row.get("Test_RMSE") if "Test_RMSE" in res_df.columns else best_row.get("RMSE")),
                            "MAPE": _safe_float(best_row.get("Test_MAPE") if "Test_MAPE" in res_df.columns else best_row.get("MAPE_%")),
                            "Accuracy": _safe_float(best_row.get("Test_Acc") if "Test_Acc" in res_df.columns else best_row.get("Accuracy_%")),
                        }
                        order_str = "auto" if order is None else str(order)
                        seas_str  = "auto" if seasonal_order is None else str(seasonal_order)
                        feats = exog_vars if exog_vars else ["ALL"]
                        model_params = f"order={order_str}; seasonal={seas_str}; s={season_len}; H={horizons}; X={','.join(map(str, feats))}"
                        _append_to_comparison(
                            model_name=f"SARIMAX (best h={int(best_row.get('Horizon', 1))})",
                            train_ratio=train_ratio,
                            kpis=kpis,
                            model_params=model_params,
                            runtime_s=runtime_s
                        )
                except Exception as e:
                    st.error(f"❌ SARIMAX training failed: {e}")

    # ------------------------------------------------------------
    # 📊 Results — Unified Template (SARIMAX-style artifacts for both)
    # ------------------------------------------------------------
    with tab_results:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1)); border: 2px solid rgba(168, 85, 247, 0.3); box-shadow: 0 0 40px rgba(168, 85, 247, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(168, 85, 247, 0.6));'>📊</div>
              <h1 style='font-size: 1.75rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, #a855f7, {SECONDARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(168, 85, 247, 0.3);'>Model Results</h1>
              <p style='font-size: 1rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Comprehensive performance metrics, forecasts, and diagnostic visualizations</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_model = st.session_state.get("selected_model")
        if not current_model:
            st.warning("⚠️ Select and train a model first.")
            return

        # ---------- ARIMA ----------
        if current_model == "ARIMA":
            arima_results = st.session_state.get("arima_results")
            if arima_results is not None:
                st.markdown("#### 🎯 Performance Metrics (h=1)")
                test_m = arima_results.get("test_metrics", {}) or {}
                _render_kpi_row(test_m)

                st.markdown("---")
                st.markdown("#### 📈 Forecast vs Actual (h=1)")
                if _plt_import_ok:
                    fig_forecast = plot_arima_forecast(arima_results, title="ARIMA Forecast: Training and Test Period")
                    try:
                        fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Number of Patient Arrivals")
                    except Exception:
                        pass
                else:
                    fig_forecast = _plot_arima_forecast_local(arima_results, title="ARIMA Forecast: Training and Test Period")
                st.plotly_chart(fig_forecast, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🧪 Comprehensive Multi-Horizon Results — ARIMA")
            arima_mh = st.session_state.get("arima_mh_results")
            if arima_mh is None:
                st.info("Train ARIMA with the **Multi-horizon** button to see full dashboard.")
            else:
                try:
                    mdf, F, L, U, test_eval, train, res, horizons = arima_to_multihorizon_artifacts(arima_mh)

                    # Sanitize metrics & arrays to avoid non-finite → int crashes
                    mdf = _sanitize_metrics_df(mdf)
                    F, L, U = _sanitize_artifacts(F, L, U)

                    kpi_avg = {
                        "MAE": _mean_from_any(mdf, ["Test_MAE", "MAE"]),
                        "RMSE": _mean_from_any(mdf, ["Test_RMSE", "RMSE"]),
                        "MAPE": _mean_from_any(mdf, ["Test_MAPE", "MAPE_%", "MAPE"]),
                        "Accuracy": _mean_from_any(mdf, ["Test_Acc", "Accuracy_%", "Accuracy"]),
                    }
                    st.markdown("#### 🎯 Performance Metrics — Average Across Horizons")
                    _render_kpi_row(kpi_avg)

                    st.markdown("---")
                    st.markdown("#### 📋 Detailed Metrics Table (Train & Test)")
                    st.write(_styled_core_metrics_table(mdf))

                    st.markdown("---")
                    st.markdown("#### 📊 Performance Visualization")
                    colA, colB = st.columns(2)
                    with colA:
                        st.plotly_chart(_build_performance_heatmap(mdf, "Performance Heatmap (Test)"), use_container_width=True)
                    with colB:
                        st.plotly_chart(_build_error_evolution(arima_mh.get("per_h", {}), "Error Evolution by Horizon"), use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### 🎨 Comprehensive Forecast Analysis")
                    art = build_multihorizon_results_dashboard(
                        metrics_df=mdf, F=F, L=L, U=U,
                        test_eval=test_eval, train=train, res=res, horizons=horizons
                    )

                    st.markdown("##### 🧩 Forecast Comparison (All Horizons)")
                    st.pyplot(art["fig_forecast_wall"])

                    st.markdown("##### 🌈 Fan Chart")
                    st.pyplot(art["fig_fanchart"])

                    st.markdown("##### 🎯 Predicted vs Actual")
                    st.pyplot(art["fig_pred_vs_actual"])

                    st.markdown("##### 🔎 Error Distributions")
                    c1, c2 = st.columns(2)
                    with c1: st.pyplot(art["fig_err_box"])
                    with c2: st.pyplot(art["fig_err_kde"])

                    st.markdown("##### ⏱ Temporal Split (Train/Test + h=1)")
                    st.pyplot(art["fig_temporal"])

                except Exception as e:
                    st.error(f"Could not render ARIMA multi-horizon dashboard: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ---------- SARIMAX ----------
        elif current_model == "SARIMAX":
            sar_out = st.session_state.get("sarimax_results")
            if sar_out is None:
                st.info("📊 Train SARIMAX first to see results in this tab.")
                return

            st.markdown("### 🧪 Comprehensive Multi-Horizon Results — SARIMAX")
            try:
                metrics_df, F, L, U, test_eval, train, res, horizons = to_multihorizon_artifacts(sar_out)

                # Sanitize metrics & arrays
                metrics_df = _sanitize_metrics_df(metrics_df)
                F, L, U = _sanitize_artifacts(F, L, U)

                st.markdown("#### 🎯 Performance Metrics — Average Across Horizons")
                kpi_avg = {
                    "MAE": _mean_from_any(metrics_df, ["Test_MAE", "MAE"]),
                    "RMSE": _mean_from_any(metrics_df, ["Test_RMSE", "RMSE"]),
                    "MAPE": _mean_from_any(metrics_df, ["Test_MAPE", "MAPE_%", "MAPE"]),
                    "Accuracy": _mean_from_any(metrics_df, ["Test_Acc", "Accuracy_%", "Accuracy"]),
                }
                _render_kpi_row(kpi_avg)

                st.markdown("---")
                st.markdown("#### 📋 Detailed Metrics Table (Train & Test)")
                st.write(_styled_core_metrics_table(metrics_df))

                st.markdown("---")
                st.markdown("#### 📊 Performance Visualization")
                colA, colB = st.columns(2)
                with colA:
                    st.plotly_chart(_build_performance_heatmap(metrics_df, "Performance Heatmap (Test)"), use_container_width=True)
                with colB:
                    st.plotly_chart(_build_error_evolution(sar_out.get("per_h", {}), "Error Evolution by Horizon"), use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🎨 Comprehensive Forecast Analysis")
                art = build_multihorizon_results_dashboard(
                    metrics_df=metrics_df, F=F, L=L, U=U,
                    test_eval=test_eval, train=train, res=res, horizons=horizons
                )

                st.markdown("##### 🧩 Forecast Comparison (All Horizons)")
                st.pyplot(art["fig_forecast_wall"])

                st.markdown("##### 🌈 Fan Chart")
                st.pyplot(art["fig_fanchart"])

                st.markdown("##### 🎯 Predicted vs Actual")
                st.pyplot(art["fig_pred_vs_actual"])

                st.markdown("##### 🔎 Error Distributions")
                colA, colB = st.columns(2)
                with colA: st.pyplot(art["fig_err_box"])
                with colB: st.pyplot(art["fig_err_kde"])

                st.markdown("##### ⏱ Temporal Split (Train/Test + h=1)")
                st.pyplot(art["fig_temporal"])

                st.markdown("##### 🧪 Residual Diagnostics")
                st.pyplot(art["fig_residuals"])

            except Exception as e:
                st.error(f"Could not render the multi-horizon dashboard: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ------------------------------------------------------------
    # 🆚 Model Comparison
    # ------------------------------------------------------------
    with tab_compare:
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 191, 36, 0.1)); border: 2px solid rgba(245, 158, 11, 0.3); box-shadow: 0 0 40px rgba(245, 158, 11, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3);'>
              <div style='font-size: 2.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(245, 158, 11, 0.6));'>🆚</div>
              <h1 style='font-size: 1.75rem; font-weight: 800; margin-bottom: 0.75rem; background: linear-gradient(135deg, {WARNING_COLOR}, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 0 30px rgba(245, 158, 11, 0.3);'>Model Comparison</h1>
              <p style='font-size: 1rem; color: {BODY_TEXT}; margin: 0; line-height: 1.6;'>Compare performance metrics across all trained models and identify the best performer</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _init_comparison_table()

        col_reset, col_export = st.columns([1, 5])
        with col_reset:
            if st.button("🧹 Clear All", use_container_width=True, help="Reset comparison table"):
                st.session_state["model_comparison"] = st.session_state["model_comparison"].iloc[0:0]
                st.success("✅ Comparison table cleared!")
                st.rerun()

        comp = st.session_state.get("model_comparison")
        if comp is None or comp.empty:
            st.info("📭 No trained models yet. Train ARIMA or SARIMAX models to populate this comparison dashboard.")

            # Show example of what the comparison will look like
            st.markdown("---")
            st.markdown("#### 📖 What you'll see here:")
            st.markdown("""
            - 📊 **Performance Metrics Table** - Detailed comparison of all models
            - 📈 **Multi-Metric Charts** - Visual comparison across different metrics
            - ⚡ **Runtime vs Accuracy** - Trade-off analysis
            - 🏆 **Best Model Identification** - Automatic recommendation
            - 📥 **Export Options** - Download results as CSV
            """)
        else:
            show = comp.copy()
            for c in ["MAE", "RMSE", "MAPE_%", "Accuracy_%", "Runtime_s"]:
                if c in show.columns:
                    show[c] = pd.to_numeric(show[c], errors="coerce")

            # Add ranking column
            if "Accuracy_%" in show.columns and show["Accuracy_%"].notna().any():
                show = show.sort_values("Accuracy_%", ascending=False).reset_index(drop=True)
                show.insert(0, "Rank", range(1, len(show) + 1))

            # Style the dataframe
            styled_df = show.style.format({
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "MAPE_%": "{:.2f}",
                "Accuracy_%": "{:.2f}",
                "Runtime_s": "{:.3f}"
            }, na_rep="—")

            if "Accuracy_%" in show.columns:
                styled_df = styled_df.background_gradient(
                    subset=['Accuracy_%'], cmap='RdYlGn', vmin=70, vmax=100
                )
            if "MAE" in show.columns:
                styled_df = styled_df.background_gradient(
                    subset=['MAE'], cmap='RdYlGn_r'
                )
            if "RMSE" in show.columns:
                styled_df = styled_df.background_gradient(
                    subset=['RMSE'], cmap='RdYlGn_r'
                )

            st.markdown("#### 📋 Performance Comparison Table")
            st.dataframe(styled_df, use_container_width=True, height=min(400, 50 + len(show) * 35))

            # Download buttons
            col_dl1, col_dl2, _ = st.columns([1, 1, 3])
            with col_dl1:
                csv = show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "model_comparison.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_dl2:
                json_export = show.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_export,
                    "model_comparison.json",
                    "application/json",
                    use_container_width=True
                )

            st.markdown("---")

            # Create comparison visualizations
            if len(show) >= 2:
                st.markdown("#### 📊 Visual Performance Analysis")

                charts = _build_model_comparison_charts(show)

                if charts:
                    # Multi-metric comparison
                    if charts["multi_metric"]:
                        st.plotly_chart(charts["multi_metric"], use_container_width=True)

                    # Runtime vs Accuracy + Error metrics side by side
                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        if charts["scatter"]:
                            st.plotly_chart(charts["scatter"], use_container_width=True)
                        else:
                            st.info("Runtime vs Accuracy chart requires both metrics to be available.")

                    with col_chart2:
                        if charts["errors"]:
                            st.plotly_chart(charts["errors"], use_container_width=True)
                        else:
                            st.info("Error metrics chart requires MAE or RMSE data.")

            st.markdown("---")

            # Best model identification
            st.markdown("#### 🏆 Best Model Recommendation")
            try:
                best_model = None
                if "Accuracy_%" in show.columns and show["Accuracy_%"].notna().any():
                    best_model = show.sort_values("Accuracy_%", ascending=False).iloc[0]
                elif "RMSE" in show.columns and show["RMSE"].notna().any():
                    best_model = show.sort_values("RMSE", ascending=True).iloc[0]

                if best_model is not None:
                    col_best1, col_best2 = st.columns([2, 1])

                    with col_best1:
                        accuracy = _safe_float(best_model.get("Accuracy_%"))
                        mae = _safe_float(best_model.get("MAE"))
                        rmse = _safe_float(best_model.get("RMSE"))
                        runtime = _safe_float(best_model.get("Runtime_s"))
                        train_ratio = best_model.get("TrainRatio", "N/A")
                        params = best_model.get("ModelParams", "N/A")

                        acc_txt = _fmt_pct(accuracy)
                        mae_txt = _fmt_num(mae)
                        rmse_txt = _fmt_num(rmse)
                        rt_txt = f"{runtime:.2f}s" if np.isfinite(runtime) else "N/A"

                        fast_flag = " (Fast!)" if np.isfinite(runtime) and runtime < 5 else ""
                        excellent_flag = " (Excellent!)" if np.isfinite(accuracy) and accuracy >= 95 else ""

                        st.success(
                            f"**🥇 Recommended Model: {best_model['Model']}**\n\n"
                            f"**Performance Metrics:**\n"
                            f"- ✅ Accuracy: {acc_txt}{excellent_flag}\n"
                            f"- 📊 MAE: {mae_txt}\n"
                            f"- 📈 RMSE: {rmse_txt}\n"
                            f"- ⚡ Runtime: {rt_txt}{fast_flag}\n\n"
                            f"**Configuration:**\n"
                            f"- Train/Test Split: {train_ratio}\n"
                            f"- Parameters: {params}"
                        )

                    with col_best2:
                        if np.isfinite(_safe_float(best_model.get("Accuracy_%"))):
                            fig_mini = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=_safe_float(best_model.get("Accuracy_%")),
                                title={'text': "Model Accuracy"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': SUCCESS_COLOR},
                                    'steps': [
                                        {'range': [0, 70], 'color': "rgba(255,0,0,0.1)"},
                                        {'range': [70, 85], 'color': "rgba(255,165,0,0.1)"},
                                        {'range': [85, 95], 'color': "rgba(135,206,250,0.1)"},
                                        {'range': [95, 100], 'color': "rgba(144,238,144,0.1)"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 95
                                    }
                                }
                            ))
                            fig_mini.update_layout(
                                height=250,
                                margin=dict(l=20, r=20, t=50, b=20),
                                paper_bgcolor=BG_BLACK,
                                font=dict(color=TEXT_COLOR)
                            )
                            st.plotly_chart(fig_mini, use_container_width=True)

                # Show top 3 models if available
                if len(show) >= 3:
                    st.markdown("---")
                    st.markdown("#### 🥇🥈🥉 Top 3 Models")

                    top3 = show.head(3)
                    col1, col2, col3 = st.columns(3)

                    medal_list = ["🥇", "🥈", "🥉"]
                    color_list = [SUCCESS_COLOR, SECONDARY_COLOR, WARNING_COLOR]

                    for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top3.iterrows())):
                        with col:
                            medal = medal_list[idx]
                            rank_color = color_list[idx]
                            acc_txt = _fmt_pct(row.get("Accuracy_%"))
                            mae_txt = _fmt_num(row.get("MAE"))
                            rmse_txt = _fmt_num(row.get("RMSE"))
                            st.markdown(f"""
                            <div style="border: 2px solid {rank_color}; border-radius: 10px; padding: 15px; background: {BG_BLACK};">
                                <div style="text-align: center; font-size: 2rem;">{medal}</div>
                                <h5 style="text-align: center; color: {TEXT_COLOR}; margin: 10px 0; word-wrap: break-word;">{row['Model']}</h5>
                                <div style="text-align: center; color: {rank_color}; font-size: 1.5rem; font-weight: 600; margin: 10px 0;">
                                    {acc_txt}
                                </div>
                                <div style="font-size: 0.8rem; color: {SUBTLE_TEXT}; text-align: center;">
                                    MAE: {mae_txt} | RMSE: {rmse_txt}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not determine best model: {e}")

            # Model summaries
            if st.checkbox("📖 Show Detailed Model Summaries", value=False):
                st.markdown("---")
                st.markdown("#### 📊 Detailed Model Performance Cards")
                summaries = _create_model_performance_summary(show)
                for summary_html in summaries:
                    st.markdown(summary_html, unsafe_allow_html=True)

def _build_model_comparison_charts(comp_df: pd.DataFrame):
    """Create comprehensive comparison visualizations."""
    if comp_df is None or comp_df.empty or len(comp_df) < 2:
        return {"multi_metric": None, "scatter": None, "errors": None}

    # Prepare data
    df_work = comp_df.copy()
    for c in ["MAE", "RMSE", "MAPE_%", "Accuracy_%", "Runtime_s"]:
        if c in df_work.columns:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

    # 1. Multi-metric comparison (grouped bar)
    fig_metrics = go.Figure()

    metrics_to_plot = []
    if "Accuracy_%" in df_work.columns and df_work["Accuracy_%"].notna().any():
        metrics_to_plot.append(("Accuracy_%", "Accuracy (%)", SUCCESS_COLOR))
    if "MAE" in df_work.columns and df_work["MAE"].notna().any():
        mae_max = df_work["MAE"].max()
        if np.isfinite(mae_max) and mae_max > 0:
            df_work["MAE_norm"] = 100 - (df_work["MAE"] / mae_max * 100)
        else:
            df_work["MAE_norm"] = 100.0
        metrics_to_plot.append(("MAE_norm", "MAE (normalized)", PRIMARY_COLOR))
    if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
        rmse_max = df_work["RMSE"].max()
        if np.isfinite(rmse_max) and rmse_max > 0:
            df_work["RMSE_norm"] = 100 - (df_work["RMSE"] / rmse_max * 100)
        else:
            df_work["RMSE_norm"] = 100.0
        metrics_to_plot.append(("RMSE_norm", "RMSE (normalized)", SECONDARY_COLOR))

    for metric, label, color in metrics_to_plot:
        fig_metrics.add_trace(go.Bar(
            name=label,
            x=df_work["Model"],
            y=df_work[metric],
            marker_color=color,
            text=df_work[metric].round(2),
            textposition='outside',
            texttemplate='%{text}',
            textfont=dict(size=12, color='white')
        ))

    fig_metrics.update_layout(
        title=dict(text="📊 Multi-Metric Model Comparison", x=0.5, xanchor='center'),
        xaxis_title="Model",
        yaxis_title="Score (Higher is Better)",
        barmode='group',
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        margin=dict(t=70, b=80, l=60, r=40),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(tickangle=-45),
        font_color='white'
    )

    # 2. Runtime vs Accuracy scatter
    fig_scatter = None
    if "Runtime_s" in df_work.columns and "Accuracy_%" in df_work.columns:
        if df_work["Runtime_s"].notna().any() and df_work["Accuracy_%"].notna().any():
            fig_scatter = go.Figure()

            fig_scatter.add_trace(go.Scatter(
                x=df_work["Runtime_s"],
                y=df_work["Accuracy_%"],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=df_work["Accuracy_%"],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Accuracy %"),
                        tickfont=dict(color='white')
                    ),
                    line=dict(width=2, color=BG_BLACK)
                ),
                text=df_work["Model"],
                textposition="top center",
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{text}</b><br>Runtime: %{x:.2f}s<br>Accuracy: %{y:.2f}%<extra></extra>'
            ))

            fig_scatter.update_layout(
                title=dict(text="⚡ Runtime vs Accuracy Trade-off", x=0.5, xanchor='center'),
                xaxis=dict(title="Runtime (seconds)", showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                yaxis=dict(title="Accuracy (%)", showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                plot_bgcolor=BG_BLACK,
                paper_bgcolor=BG_BLACK,
                margin=dict(t=70, b=60, l=60, r=60),
                height=400,
                font_color='white'
            )

    # 3. Error metrics comparison (MAE, RMSE side by side)
    fig_errors = None
    if "MAE" in df_work.columns or "RMSE" in df_work.columns:
        fig_errors = go.Figure()

        if "MAE" in df_work.columns and df_work["MAE"].notna().any():
            fig_errors.add_trace(go.Bar(
                name="MAE",
                x=df_work["Model"],
                y=df_work["MAE"],
                marker_color=PRIMARY_COLOR,
                text=df_work["MAE"].round(3),
                textposition='outside',
                yaxis='y',
                offsetgroup=1,
                textfont=dict(size=12, color='white')
            ))

        if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
            fig_errors.add_trace(go.Bar(
                name="RMSE",
                x=df_work["Model"],
                y=df_work["RMSE"],
                marker_color=SECONDARY_COLOR,
                text=df_work["RMSE"].round(3),
                textposition='outside',
                yaxis='y',
                offsetgroup=2,
                textfont=dict(size=12, color='white')
            ))

        fig_errors.update_layout(
            title=dict(text="📉 Error Metrics Comparison (Lower is Better)", x=0.5, xanchor='center'),
            xaxis_title="Model",
            yaxis_title="Error Value",
            barmode='group',
            plot_bgcolor=BG_BLACK,
            paper_bgcolor=BG_BLACK,
            margin=dict(t=70, b=80, l=60, r=40),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(tickangle=-45),
            font_color='white'
        )

    return {
        "multi_metric": fig_metrics,
        "scatter": fig_scatter,
        "errors": fig_errors
    }

def _create_model_performance_summary(comp_df: pd.DataFrame):
    """Create a summary card for each model."""
    if comp_df is None or comp_df.empty:
        return []

    summaries = []
    for _, row in comp_df.iterrows():
        model_name = row.get("Model", "Unknown")
        accuracy = _safe_float(row.get("Accuracy_%"))
        mae = _safe_float(row.get("MAE"))
        rmse = _safe_float(row.get("RMSE"))
        runtime = _safe_float(row.get("Runtime_s"))
        params = row.get("ModelParams", "N/A")
        train_ratio = row.get("TrainRatio", "N/A")

        # Badge
        if np.isfinite(accuracy):
            if accuracy >= 95:
                badge_color = SUCCESS_COLOR
                badge_text = "Excellent"
            elif accuracy >= 90:
                badge_color = PRIMARY_COLOR
                badge_text = "Very Good"
            elif accuracy >= 85:
                badge_color = WARNING_COLOR
                badge_text = "Good"
            else:
                badge_color = DANGER_COLOR
                badge_text = "Needs Improvement"
        else:
            badge_color = SUBTLE_TEXT
            badge_text = "N/A"

        acc_txt = _fmt_pct(accuracy)
        mae_txt = _fmt_num(mae)
        rmse_txt = _fmt_num(rmse)
        rt_txt = f"{runtime:.2f}s" if np.isfinite(runtime) else "N/A"

        summary_html = f"""
        <div class='hf-section-card' style="border: 2px solid {badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 class='hf-section-title' style="color: {TEXT_COLOR}; margin: 0;">{model_name}</h4>
                <span class='hf-pill' style="background: {badge_color}; color: {BG_BLACK};">{badge_text}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 15px;">
                <div class='hf-section-card' style="padding: 10px; border-left: 3px solid {SUCCESS_COLOR};">
                    <div style="color: {SUBTLE_TEXT}; font-size: 0.75rem;">Accuracy</div>
                    <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: 600;">{acc_txt}</div>
                </div>
                <div class='hf-section-card' style="padding: 10px; border-left: 3px solid {PRIMARY_COLOR};">
                    <div style="color: {SUBTLE_TEXT}; font-size: 0.75rem;">MAE</div>
                    <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: 600;">{mae_txt}</div>
                </div>
                <div class='hf-section-card' style="padding: 10px; border-left: 3px solid {SECONDARY_COLOR};">
                    <div style="color: {SUBTLE_TEXT}; font-size: 0.75rem;">RMSE</div>
                    <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: 600;">{rmse_txt}</div>
                </div>
                <div class='hf-section-card' style="padding: 10px; border-left: 3px solid {WARNING_COLOR};">
                    <div style="color: {SUBTLE_TEXT}; font-size: 0.75rem;">Runtime</div>
                    <div style="color: {TEXT_COLOR}; font-size: 1.3rem; font-weight: 600;">{rt_txt}</div>
                </div>
            </div>
            <div class='hf-section-card' style="margin-top: 15px; padding: 10px;">
                <div style="color: {SUBTLE_TEXT}; font-size: 0.75rem; margin-bottom: 5px;">Configuration</div>
                <div style="color: {TEXT_COLOR}; font-size: 0.85rem; line-height: 1.5; overflow-wrap: break-word; word-wrap: break-word; white-space: normal;">
                    <strong>Train/Test:</strong> {train_ratio}<br>
                    <strong>Parameters:</strong> {params}
                </div>
            </div>
        </div>
        """
        summaries.append(summary_html)

    return summaries

page_benchmarks()


