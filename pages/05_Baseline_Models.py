# ===========================================
# pages/05_Baseline_Models.py — Baseline Models (Enhanced Multi-Horizon)
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
from app_core.ui.page_navigation import render_page_navigation

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

from app_core.state.session import init_state

# Plot builders for multi-horizon results (expects matplotlib figs)
from app_core.plots import build_multihorizon_results_dashboard

# === SARIMAX multi-horizon (canonical artifact producer) ===
from app_core.models.sarimax_pipeline import (
    run_sarimax_multihorizon,
    to_multihorizon_artifacts,
    run_sarimax_multi_target_pipeline,
    get_reason_target_columns as get_sarimax_target_columns,
)

# === ARIMA helpers (with graceful fallbacks for plotting) ===
from app_core.models.arima_pipeline import (
    run_arima_pipeline,
    run_arima_multi_target_pipeline,
    get_reason_target_columns as get_arima_target_columns,
)
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

# ---------------------------- Statistical Analysis Helpers ----------------------------

def _calculate_mase(y_true, y_pred, y_train, seasonality=1):
    """
    Mean Absolute Scaled Error - Scale-independent forecast accuracy metric.

    MASE < 1.0: Better than naive seasonal forecast (Good)
    MASE = 1.0: Equal to naive forecast (Baseline)
    MASE > 1.0: Worse than naive forecast (Poor)

    Args:
        y_true: Actual test values
        y_pred: Predicted test values
        y_train: Training data for naive baseline calculation
        seasonality: Seasonal period (1 for non-seasonal, 7 for weekly, etc.)

    Returns:
        float: MASE score
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        y_train = np.asarray(y_train).flatten()

        # Forecast errors
        forecast_errors = np.abs(y_true - y_pred)
        mae_forecast = np.mean(forecast_errors)

        # Naive forecast errors (in-sample)
        if len(y_train) <= seasonality:
            return np.nan

        naive_errors = np.abs(np.diff(y_train, n=seasonality))
        mae_naive = np.mean(naive_errors)

        # Avoid division by zero
        if mae_naive == 0 or not np.isfinite(mae_naive):
            return np.nan

        mase = mae_forecast / mae_naive
        return float(mase) if np.isfinite(mase) else np.nan

    except Exception:
        return np.nan

def _calculate_baseline_metrics(y_train, y_test):
    """
    Calculate performance of naive baseline forecasts for comparison.

    Returns:
        dict: Baseline metrics for Naive, Seasonal Naive, and Moving Average
    """
    try:
        y_train = np.asarray(y_train).flatten()
        y_test = np.asarray(y_test).flatten()

        if len(y_train) < 7 or len(y_test) == 0:
            return {}

        baselines = {}

        # 1. Naive forecast (last value repeated)
        naive_pred = np.full(len(y_test), y_train[-1])
        baselines['Naive'] = {
            'MAE': float(np.mean(np.abs(y_test - naive_pred))),
            'RMSE': float(np.sqrt(np.mean((y_test - naive_pred)**2)))
        }

        # 2. Seasonal Naive (7-day seasonal)
        if len(y_train) >= 7:
            seasonal_naive_pred = np.array([y_train[-(7 - (i % 7))] for i in range(len(y_test))])
            baselines['Seasonal Naive'] = {
                'MAE': float(np.mean(np.abs(y_test - seasonal_naive_pred))),
                'RMSE': float(np.sqrt(np.mean((y_test - seasonal_naive_pred)**2)))
            }

        # 3. Moving Average (7-day)
        if len(y_train) >= 7:
            ma_value = np.mean(y_train[-7:])
            ma_pred = np.full(len(y_test), ma_value)
            baselines['Moving Avg (7d)'] = {
                'MAE': float(np.mean(np.abs(y_test - ma_pred))),
                'RMSE': float(np.sqrt(np.mean((y_test - ma_pred)**2)))
            }

        return baselines

    except Exception:
        return {}

def _build_horizon_performance_chart(metrics_df: pd.DataFrame, title: str = "Forecast Accuracy by Horizon"):
    """
    Create professional horizon-specific performance degradation chart.
    Shows how accuracy changes across forecast horizons.
    """
    df_work = _sanitize_metrics_df(metrics_df)
    if df_work is None or df_work.empty or "Horizon" not in df_work.columns:
        return None

    # Find accuracy column
    acc_col = None
    for col in ["Test_Acc", "Accuracy_%", "Accuracy"]:
        if col in df_work.columns and df_work[col].notna().any():
            acc_col = col
            break

    # Find error columns
    mae_col = "Test_MAE" if "Test_MAE" in df_work.columns else "MAE"
    rmse_col = "Test_RMSE" if "Test_RMSE" in df_work.columns else "RMSE"

    if acc_col is None:
        return None

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Accuracy line (primary y-axis)
    fig.add_trace(go.Scatter(
        x=df_work["Horizon"],
        y=df_work[acc_col],
        mode='lines+markers',
        name='Accuracy',
        line=dict(width=4, color=SUCCESS_COLOR),
        marker=dict(size=12, symbol='circle', line=dict(width=2, color='white')),
        text=[f"{v:.1f}%" for v in df_work[acc_col]],
        textposition='top center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        hovertemplate='<b>Horizon %{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>',
        yaxis='y'
    ))

    # MAE line (secondary y-axis)
    if mae_col in df_work.columns and df_work[mae_col].notna().any():
        fig.add_trace(go.Scatter(
            x=df_work["Horizon"],
            y=df_work[mae_col],
            mode='lines+markers',
            name='MAE',
            line=dict(width=3, color=PRIMARY_COLOR, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Horizon %{x}</b><br>MAE: %{y:.2f}<extra></extra>',
            yaxis='y2',
            visible='legendonly'
        ))

    # Add reference lines
    if acc_col and df_work[acc_col].notna().any():
        h1_acc = df_work[df_work["Horizon"] == 1][acc_col].values
        if len(h1_acc) > 0:
            fig.add_hline(
                y=float(h1_acc[0]),
                line_dash="dash",
                line_color="rgba(144,238,144,0.4)",
                annotation_text=f"h=1 baseline: {h1_acc[0]:.1f}%",
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color=TEXT_COLOR
            )

    # Add performance zones
    fig.add_hrect(y0=95, y1=100, fillcolor="rgba(144,238,144,0.1)",
                  layer="below", line_width=0, annotation_text="Excellent",
                  annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=90, y1=95, fillcolor="rgba(135,206,250,0.1)",
                  layer="below", line_width=0, annotation_text="Good",
                  annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=85, y1=90, fillcolor="rgba(255,255,0,0.05)",
                  layer="below", line_width=0)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR, family='Arial Black')
        ),
        xaxis=dict(
            title="<b>Forecast Horizon (days ahead)</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title="<b>Accuracy (%)</b>",
            title_font=dict(size=14, color=SUCCESS_COLOR),
            side='left',
            range=[max(75, df_work[acc_col].min() - 5) if acc_col else 75, 100],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            color=TEXT_COLOR
        ),
        yaxis2=dict(
            title="<b>MAE</b>",
            title_font=dict(size=14, color=PRIMARY_COLOR),
            overlaying='y',
            side='right',
            showgrid=False,
            color=TEXT_COLOR
        ),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=450,
        margin=dict(t=80, b=70, l=80, r=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11)
        ),
        hovermode='x unified',
        font=dict(color=TEXT_COLOR)
    )

    return fig

def _build_baseline_comparison_chart(model_metrics: dict, baseline_metrics: dict, mase: float):
    """
    Create professional baseline comparison chart showing model vs naive forecasts.
    """
    if not baseline_metrics:
        return None

    # Prepare data
    methods = ['Your Model'] + list(baseline_metrics.keys())
    mae_values = [model_metrics.get('MAE', np.nan)] + [baseline_metrics[k]['MAE'] for k in baseline_metrics.keys()]
    rmse_values = [model_metrics.get('RMSE', np.nan)] + [baseline_metrics[k]['RMSE'] for k in baseline_metrics.keys()]

    # Create grouped bar chart
    fig = go.Figure()

    # MAE bars
    fig.add_trace(go.Bar(
        name='MAE',
        x=methods,
        y=mae_values,
        marker_color=[SUCCESS_COLOR if i == 0 else SECONDARY_COLOR for i in range(len(methods))],
        text=[f"{v:.2f}" for v in mae_values],
        textposition='outside',
        textfont=dict(size=12, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>',
        offsetgroup=0
    ))

    # RMSE bars
    fig.add_trace(go.Bar(
        name='RMSE',
        x=methods,
        y=rmse_values,
        marker_color=[PRIMARY_COLOR if i == 0 else WARNING_COLOR for i in range(len(methods))],
        text=[f"{v:.2f}" for v in rmse_values],
        textposition='outside',
        textfont=dict(size=12, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.3f}<extra></extra>',
        offsetgroup=1
    ))

    # Calculate y-axis range with padding
    all_values = mae_values + rmse_values
    max_val = max([v for v in all_values if np.isfinite(v)])

    fig.update_layout(
        title=dict(
            text="<b>📊 Model vs Baseline Forecasts</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR, family='Arial Black')
        ),
        xaxis=dict(
            title="<b>Forecast Method</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            tickangle=-30,
            showgrid=False,
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title="<b>Error (Lower is Better)</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            range=[0, max_val * 1.3],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color=TEXT_COLOR
        ),
        barmode='group',
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=450,
        margin=dict(t=80, b=100, l=70, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11)
        ),
        font=dict(color=TEXT_COLOR)
    )

    return fig

def _build_pairwise_delta_chart(comp_df: pd.DataFrame):
    """
    Create pairwise difference analysis showing how much better the winner is.
    For 2-model comparison (ARIMA vs SARIMAX).
    """
    if comp_df is None or len(comp_df) != 2:
        return None

    df = comp_df.copy()

    # Ensure numeric columns
    for c in ["MAE", "RMSE", "Accuracy_%", "Runtime_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort by accuracy (best first)
    if "Accuracy_%" in df.columns and df["Accuracy_%"].notna().any():
        df = df.sort_values("Accuracy_%", ascending=False)

    model1, model2 = df.iloc[0], df.iloc[1]

    # Calculate deltas (positive = model1 better)
    deltas = []
    metrics_info = []

    if "Accuracy_%" in df.columns:
        delta = model1["Accuracy_%"] - model2["Accuracy_%"]
        deltas.append(delta)
        metrics_info.append(("Accuracy", delta, "%", True))  # True = higher is better

    if "MAE" in df.columns:
        delta = model2["MAE"] - model1["MAE"]  # Inverted: lower MAE is better
        deltas.append(delta)
        metrics_info.append(("MAE", delta, "", False))  # False = lower is better

    if "RMSE" in df.columns:
        delta = model2["RMSE"] - model1["RMSE"]  # Inverted: lower RMSE is better
        deltas.append(delta)
        metrics_info.append(("RMSE", delta, "", False))

    if "Runtime_s" in df.columns:
        delta = model2["Runtime_s"] - model1["Runtime_s"]  # Inverted: faster is better
        deltas.append(delta)
        metrics_info.append(("Runtime", delta, "s", False))

    if not metrics_info:
        return None

    # Create bar chart
    fig = go.Figure()

    for metric_name, delta_val, suffix, higher_better in metrics_info:
        color = SUCCESS_COLOR if delta_val > 0 else DANGER_COLOR

        fig.add_trace(go.Bar(
            name=metric_name,
            x=[metric_name],
            y=[abs(delta_val)],
            marker_color=color,
            text=[f"{delta_val:+.2f}{suffix}"],
            textposition='outside',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate=f'<b>{metric_name}</b><br>Δ: {delta_val:+.3f}{suffix}<extra></extra>',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>📊 {model1['Model']} vs {model2['Model']}</b><br><sub>Positive = {model1['Model']} Better</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16, color=TEXT_COLOR, family='Arial Black')
        ),
        xaxis=dict(
            title="<b>Metric</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            showgrid=False,
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title="<b>Absolute Difference</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color=TEXT_COLOR
        ),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=400,
        margin=dict(t=100, b=60, l=70, r=40),
        font=dict(color=TEXT_COLOR)
    )

    return fig

def _build_accuracy_gauge_card(model_row: pd.Series, is_winner: bool = False):
    """
    Create a single accuracy gauge card for a model.

    Args:
        model_row: Series containing model metrics (Model, Accuracy_%, MAE, RMSE, Runtime_s)
        is_winner: Whether this model is the best performer

    Returns:
        Plotly figure with gauge chart
    """
    model_name = model_row.get("Model", "Unknown")
    accuracy = _safe_float(model_row.get("Accuracy_%", 0))
    mae = _safe_float(model_row.get("MAE", 0))
    rmse = _safe_float(model_row.get("RMSE", 0))
    runtime = _safe_float(model_row.get("Runtime_s", 0))

    # Determine gauge color based on accuracy
    if accuracy >= 90:
        gauge_color = SUCCESS_COLOR  # Green
    elif accuracy >= 80:
        gauge_color = PRIMARY_COLOR  # Blue
    elif accuracy >= 70:
        gauge_color = WARNING_COLOR  # Orange
    else:
        gauge_color = DANGER_COLOR  # Red

    # Winner gets gold accent
    border_color = "#FFD700" if is_winner else "rgba(255,255,255,0.2)"
    title_prefix = "🏆 " if is_winner else ""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        number={"suffix": "%", "font": {"size": 32, "color": "white"}},
        title={
            "text": f"<b>{title_prefix}{model_name}</b>",
            "font": {"size": 15, "color": "white", "family": "Arial Black"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": TEXT_COLOR,
                "tickfont": {"size": 8, "color": TEXT_COLOR}
            },
            "bar": {"color": gauge_color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.1)",
            "borderwidth": 3,
            "bordercolor": border_color,
            "steps": [
                {"range": [0, 70], "color": "rgba(220, 38, 38, 0.2)"},
                {"range": [70, 80], "color": "rgba(251, 146, 60, 0.2)"},
                {"range": [80, 90], "color": "rgba(59, 130, 246, 0.2)"},
                {"range": [90, 100], "color": "rgba(34, 197, 94, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": accuracy
            }
        }
    ))

    # Add supporting metrics as annotations
    metrics_text = (
        f"<b>MAE:</b> {mae:.2f}  |  "
        f"<b>RMSE:</b> {rmse:.2f}  |  "
        f"<b>Runtime:</b> {runtime:.1f}s"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=10, color="rgba(255,255,255,0.8)"),
        xanchor="center"
    )

    fig.update_layout(
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=400,
        margin=dict(t=60, b=70, l=30, r=30),
        font=dict(color=TEXT_COLOR)
    )

    return fig

def _extract_horizon_comparison_data():
    """
    Extract horizon-specific metrics from both ARIMA and SARIMAX results.
    Returns dict with model names as keys and horizon DataFrames as values.
    """
    horizon_data = {}

    # Extract ARIMA data
    arima_mh = st.session_state.get("arima_mh_results")
    if arima_mh:
        try:
            # ARIMA results are already in the correct dict format with metrics_df key
            if isinstance(arima_mh, dict) and "metrics_df" in arima_mh:
                mdf = arima_mh["metrics_df"]
                mdf = _sanitize_metrics_df(mdf)
                if mdf is not None and not mdf.empty and "Horizon" in mdf.columns:
                    horizon_data["ARIMA"] = mdf
        except Exception as e:
            # Fallback: try results_df key (legacy compatibility)
            if isinstance(arima_mh, dict) and "results_df" in arima_mh:
                try:
                    mdf = arima_mh["results_df"]
                    mdf = _sanitize_metrics_df(mdf)
                    if mdf is not None and not mdf.empty and "Horizon" in mdf.columns:
                        horizon_data["ARIMA"] = mdf
                except Exception:
                    pass

    # Extract SARIMAX data
    sar_out = st.session_state.get("sarimax_results")
    if sar_out:
        try:
            from app_core.models.sarimax_pipeline import to_multihorizon_artifacts
            metrics_df, _, _, _, _, _, _, _ = to_multihorizon_artifacts(sar_out)
            metrics_df = _sanitize_metrics_df(metrics_df)
            if metrics_df is not None and not metrics_df.empty:
                horizon_data["SARIMAX"] = metrics_df
        except Exception:
            # If that fails, try direct access
            if isinstance(sar_out, dict) and "results_df" in sar_out:
                try:
                    metrics_df = sar_out["results_df"]
                    metrics_df = _sanitize_metrics_df(metrics_df)
                    if metrics_df is not None and not metrics_df.empty and "Horizon" in metrics_df.columns:
                        horizon_data["SARIMAX"] = metrics_df
                except Exception:
                    pass

    return horizon_data

def _build_horizon_comparison_line_chart(horizon_data: dict):
    """
    Build horizon-specific comparison line chart for multiple models.
    """
    if not horizon_data or len(horizon_data) < 2:
        return None

    fig = go.Figure()

    colors = [SUCCESS_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, WARNING_COLOR]

    for idx, (model_name, mdf) in enumerate(horizon_data.items()):
        # Find accuracy column
        acc_col = None
        for col in ["Test_Acc", "Accuracy_%", "Accuracy"]:
            if col in mdf.columns and mdf[col].notna().any():
                acc_col = col
                break

        if acc_col and "Horizon" in mdf.columns:
            fig.add_trace(go.Scatter(
                x=mdf["Horizon"],
                y=mdf[acc_col],
                mode='lines+markers',
                name=model_name,
                line=dict(width=4, color=colors[idx % len(colors)]),
                marker=dict(size=10, symbol='circle', line=dict(width=2, color='white')),
                hovertemplate=f'<b>{model_name}</b><br>Horizon: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
            ))

    if not fig.data:
        return None

    # Add performance zones
    fig.add_hrect(y0=95, y1=100, fillcolor="rgba(144,238,144,0.1)",
                  layer="below", line_width=0, annotation_text="Excellent",
                  annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=90, y1=95, fillcolor="rgba(135,206,250,0.1)",
                  layer="below", line_width=0, annotation_text="Good",
                  annotation_position="top left", annotation_font_size=9)

    fig.update_layout(
        title=dict(
            text="<b>📈 Accuracy Across Forecast Horizons</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR, family='Arial Black')
        ),
        xaxis=dict(
            title="<b>Forecast Horizon (days ahead)</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title="<b>Accuracy (%)</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            range=[75, 100],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            color=TEXT_COLOR
        ),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=450,
        margin=dict(t=80, b=70, l=80, r=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=12)
        ),
        hovermode='x unified',
        font=dict(color=TEXT_COLOR)
    )

    return fig

def _build_metrics_over_horizon_chart(horizon_data: dict, selected_metric: str = "Accuracy"):
    """
    Build interactive line chart showing selected metric across horizons for both models.

    Args:
        horizon_data: Dict with model names as keys and metrics DataFrames as values
        selected_metric: One of ["Accuracy", "MAE", "RMSE", "MAPE"]

    Returns:
        Plotly figure or None
    """
    if not horizon_data or len(horizon_data) < 2:
        return None

    # Define metric configurations
    metric_configs = {
        "Accuracy": {
            "column_names": ["Test_Acc", "Accuracy_%", "Accuracy"],
            "title": "Accuracy Across Forecast Horizons",
            "y_title": "Accuracy (%)",
            "is_percentage": True,
            "format": ".2f",
            "suffix": "%",
            "color_primary": SUCCESS_COLOR,
            "color_secondary": PRIMARY_COLOR,
        },
        "MAE": {
            "column_names": ["MAE", "Test_MAE"],
            "title": "MAE (Mean Absolute Error) Across Horizons",
            "y_title": "MAE",
            "is_percentage": False,
            "format": ".2f",
            "suffix": "",
            "color_primary": WARNING_COLOR,
            "color_secondary": DANGER_COLOR,
        },
        "RMSE": {
            "column_names": ["RMSE", "Test_RMSE"],
            "title": "RMSE (Root Mean Squared Error) Across Horizons",
            "y_title": "RMSE",
            "is_percentage": False,
            "format": ".2f",
            "suffix": "",
            "color_primary": SECONDARY_COLOR,
            "color_secondary": "#9333ea",
        },
        "MAPE": {
            "column_names": ["MAPE_%", "MAPE", "Test_MAPE"],
            "title": "MAPE (Mean Absolute Percentage Error) Across Horizons",
            "y_title": "MAPE (%)",
            "is_percentage": True,
            "format": ".2f",
            "suffix": "%",
            "color_primary": "#f59e0b",
            "color_secondary": "#ef4444",
        },
    }

    if selected_metric not in metric_configs:
        return None

    config = metric_configs[selected_metric]

    fig = go.Figure()

    colors = [config["color_primary"], config["color_secondary"]]

    for idx, (model_name, mdf) in enumerate(horizon_data.items()):
        # Find the metric column
        metric_col = None
        for col in config["column_names"]:
            if col in mdf.columns and mdf[col].notna().any():
                metric_col = col
                break

        if metric_col and "Horizon" in mdf.columns:
            # Extract horizons (convert from "h=1" format if needed)
            if mdf["Horizon"].dtype == 'object':
                horizons = mdf["Horizon"].str.replace("h=", "").astype(int)
            else:
                horizons = mdf["Horizon"]

            metric_values = mdf[metric_col]

            fig.add_trace(go.Scatter(
                x=horizons,
                y=metric_values,
                mode='lines+markers',
                name=model_name,
                line=dict(width=4, color=colors[idx % len(colors)]),
                marker=dict(size=10, symbol='circle', line=dict(width=2, color='white')),
                hovertemplate=f'<b>{model_name}</b><br>Horizon: %{{x}}<br>{selected_metric}: %{{y:{config["format"]}}}{config["suffix"]}<extra></extra>'
            ))

    if not fig.data:
        return None

    # Add visual zones for percentage metrics
    if config["is_percentage"] and selected_metric == "Accuracy":
        fig.add_hrect(y0=95, y1=100, fillcolor="rgba(144,238,144,0.1)",
                      layer="below", line_width=0, annotation_text="Excellent",
                      annotation_position="top left", annotation_font_size=9)
        fig.add_hrect(y0=90, y1=95, fillcolor="rgba(135,206,250,0.1)",
                      layer="below", line_width=0, annotation_text="Good",
                      annotation_position="top left", annotation_font_size=9)

    fig.update_layout(
        title=dict(
            text=f"<b>📊 {config['title']}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR, family='Arial Black')
        ),
        xaxis=dict(
            title="<b>Forecast Horizon (days ahead)</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title=f"<b>{config['y_title']}</b>",
            title_font=dict(size=14, color=TEXT_COLOR),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            color=TEXT_COLOR
        ),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        height=450,
        margin=dict(t=80, b=70, l=80, r=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=12)
        ),
        hovermode='x unified',
        font=dict(color=TEXT_COLOR)
    )

    return fig

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
    search_mode: str = "aic_only",  # "aic_only" (fast), "hybrid" (slow but accurate), or "pmdarima_oos"
) -> dict:
    """Train independent univariate ARIMA models for Target_1..Target_H."""
    rows, per_h, ok = [], {}, []

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)

    # Parameter sharing: find optimal parameters once for h=1, reuse for h>1 (automatic mode only)
    shared_order = None

    for h in range(1, int(horizons) + 1):
        tc = f"Target_{h}"
        if tc not in frame.columns:
            continue
        try:
            # For automatic mode: find parameters for h=1, reuse for h>1
            if auto_select and h == 1:
                # Full auto search for first horizon
                out = run_arima_pipeline(
                    df=frame[[date_col, tc]],
                    order=order,
                    train_ratio=train_ratio,
                    auto_select=True,
                    target_col=tc,
                    search_mode=search_mode,
                )
                # Save the optimal parameters found
                shared_order = out.get("model_info", {}).get("order")
            elif auto_select and h > 1 and shared_order is not None:
                # Reuse parameters from h=1 for subsequent horizons
                out = run_arima_pipeline(
                    df=frame[[date_col, tc]],
                    order=shared_order,
                    train_ratio=train_ratio,
                    auto_select=False,  # Skip auto search, use shared_order
                    target_col=tc,
                )
            else:
                # Manual mode: use user-specified order (unchanged)
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
    """
    Convert ARIMA multi-horizon results from per_h format to unified F, L, U matrices.

    ARIMA results structure:
    {
        "results_df": DataFrame with metrics per horizon,
        "per_h": {h: {"y_test": Series, "forecast": Series, "ci_lo": Series, "ci_hi": Series}},
        "successful": [list of successful horizons]
    }

    Returns tuple: (metrics_df, F, L, U, test_eval, train, res, horizons)
    """
    per_h = arima_out.get("per_h", {})
    results_df = arima_out.get("results_df")
    successful = arima_out.get("successful", [])

    if not per_h or not successful:
        # Return empty structure
        import pandas as pd
        import numpy as np
        empty_df = pd.DataFrame()
        empty_array = np.array([])
        return (results_df if results_df is not None else empty_df,
                empty_array, empty_array, empty_array, empty_df,
                None, None, [])

    # Get dimensions
    horizons = sorted(successful)
    H = len(horizons)

    # Get test set length from first horizon
    first_h = horizons[0]
    y_test_first = per_h[first_h].get("y_test")
    T = len(y_test_first) if y_test_first is not None else 0

    if T == 0:
        # No test data
        import pandas as pd
        import numpy as np
        return (results_df, np.array([]), np.array([]), np.array([]),
                pd.DataFrame(), None, None, horizons)

    # Initialize matrices
    import numpy as np
    import pandas as pd
    F = np.full((T, H), np.nan)
    L = np.full((T, H), np.nan)
    U = np.full((T, H), np.nan)
    test_eval = {}

    # Fill matrices from per_h data
    for idx, h in enumerate(horizons):
        h_data = per_h[h]

        # Get forecast and confidence intervals
        fc = h_data.get("forecast")
        lo = h_data.get("ci_lo")
        hi = h_data.get("ci_hi")
        yt = h_data.get("y_test")

        if fc is not None and len(fc) == T:
            F[:, idx] = fc.values if hasattr(fc, 'values') else fc

        if lo is not None and len(lo) == T:
            L[:, idx] = lo.values if hasattr(lo, 'values') else lo

        if hi is not None and len(hi) == T:
            U[:, idx] = hi.values if hasattr(hi, 'values') else hi

        if yt is not None and len(yt) == T:
            test_eval[f"Target_{h}"] = yt.values if hasattr(yt, 'values') else yt

    # Create test_eval DataFrame
    test_eval_df = pd.DataFrame(test_eval)

    # Get train data and res from first horizon
    train = per_h[first_h].get("y_train")
    res = per_h[first_h].get("res")

    # Convert results_df to have proper column names for compatibility
    metrics_df = results_df.copy() if results_df is not None else pd.DataFrame()
    if not metrics_df.empty and "Horizon" in metrics_df.columns:
        # Rename columns to match expected format
        rename_map = {
            "Test_MAE": "MAE",
            "Test_RMSE": "RMSE",
            "Test_MAPE": "MAPE_%",
            "Test_Acc": "Accuracy_%"
        }
        for old_col, new_col in rename_map.items():
            if old_col in metrics_df.columns and new_col not in metrics_df.columns:
                metrics_df[new_col] = metrics_df[old_col]

        # Format Horizon column as "h=1", "h=2", etc.
        if metrics_df["Horizon"].dtype in ['int64', 'int32']:
            metrics_df["Horizon"] = metrics_df["Horizon"].apply(lambda x: f"h={x}")

    return (metrics_df, F, L, U, test_eval_df, train, res, horizons)

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
        xaxis=dict(title_font=dict(size=13), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title_font=dict(size=13), gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
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
        line=dict(width=3),
        marker=dict(size=5)
    )

    fig.update_layout(
        margin=dict(l=60, r=40, t=70, b=60),
        title=dict(x=0.5, xanchor='center', font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(
            title="Date",
            title_font=dict(size=13),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title="Absolute Error (Patient Arrivals)",
            title_font=dict(size=13),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.15)',
            color=TEXT_COLOR
        ),
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        hovermode='x unified',
        legend=dict(
            title="Forecast Horizon",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        )
    )

    return fig

# =====================================================================
# PAGE
# =====================================================================

# Page Configuration
st.set_page_config(
    page_title="Baseline Models - HealthForecast AI",
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
          <h1 class='hf-feature-title' style='font-size: 1.75rem; margin-bottom: 0.5rem;'>Baseline Models</h1>
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

        # ============================================================
        # FORECAST MODE SELECTION (Single vs Multi-Target)
        # ============================================================
        st.markdown("#### Forecasting Mode")
        forecast_mode = st.radio(
            "Select forecasting mode:",
            ["Single-Target (Patient Arrivals Only)", "Multi-Target (Arrivals + Reasons)"],
            horizontal=True,
            key="forecast_mode_radio",
            help="Multi-Target mode forecasts patient arrivals AND each reason for visit separately"
        )

        if "Multi-Target" in forecast_mode:
            st.markdown("---")
            st.markdown("#### Target Variable Selection (Dependent Variables)")
            st.markdown("""
            Select the **future values** you want to predict. These are your dependent variables:
            - `Target_N`: Patient arrivals N days ahead
            - `reason_N`: Medical reason counts N days ahead (e.g., `asthma_1`, `pneumonia_2`)
            """)

            # Granularity and Horizon selection
            col_g1, col_g2, col_g3 = st.columns([1, 1, 1])
            with col_g1:
                forecast_granularity = st.radio(
                    "Forecast granularity:",
                    ["Aggregated Categories (Recommended)", "Granular Reasons"],
                    index=0,
                    key="forecast_granularity",
                    help="Aggregated categories (RESPIRATORY, CARDIAC, etc.) have better accuracy than granular reasons"
                )
            with col_g2:
                selected_horizon = st.selectbox(
                    "Select forecast horizon:",
                    options=["All Horizons (1-7)", 1, 2, 3, 4, 5, 6, 7],
                    index=0,
                    key="multi_target_horizon",
                    help="Choose specific horizon or 'All' for multi-horizon forecasting"
                )

            # Determine horizon for filtering
            horizon_filter = None if selected_horizon == "All Horizons (1-7)" else int(selected_horizon)

            # Aggregated clinical categories
            aggregated_bases = ['respiratory', 'cardiac', 'trauma', 'gastrointestinal', 'infectious', 'neurological', 'other']

            # Detect available target columns (with fallback for backward compatibility)
            try:
                available_targets = get_arima_target_columns(data, horizon=horizon_filter)
            except TypeError:
                # Fallback if old function signature without horizon parameter
                available_targets = get_arima_target_columns(data)

            # Filter based on granularity selection
            if "Aggregated" in forecast_granularity:
                # Only show aggregated categories + patient arrivals
                available_targets = [
                    t for t in available_targets
                    if t.lower().startswith('target_') or
                    any(t.lower().startswith(cat + '_') for cat in aggregated_bases)
                ]
            else:
                # Only show granular reasons + patient arrivals (exclude aggregated)
                available_targets = [
                    t for t in available_targets
                    if t.lower().startswith('target_') or
                    not any(t.lower().startswith(cat + '_') for cat in aggregated_bases)
                ]

            if not available_targets:
                st.warning("⚠️ No future target columns detected. Ensure your dataset has columns like `Target_1`, `RESPIRATORY_1`, etc.")
                available_targets = []
            else:
                # Group targets for display
                patient_targets = [t for t in available_targets if t.lower().startswith('target_')]
                reason_targets = [t for t in available_targets if not t.lower().startswith('target_')]

                with col_g3:
                    if "Aggregated" in forecast_granularity:
                        st.success(f"✅ **{len(patient_targets)}** arrivals + **{len(reason_targets)}** categories")
                    else:
                        st.info(f"📊 **{len(patient_targets)}** arrivals + **{len(reason_targets)}** reasons")

            # Selection controls
            select_all = st.checkbox("Select all targets", value=True, key="select_all_targets_cb")
            default_targets = available_targets if select_all else []

            selected_targets = st.multiselect(
                "Choose targets to forecast (dependent variables):",
                options=available_targets,
                default=default_targets,
                key="selected_targets_ms",
                help="Each target will be trained as a separate univariate model"
            )

            # Store in session state for use in Train tab
            st.session_state["multi_target_columns"] = selected_targets

            if selected_targets:
                st.success(f"✅ {len(selected_targets)} targets selected for forecasting")
                # Show breakdown
                n_patient = len([t for t in selected_targets if t.lower().startswith('target_')])
                n_reason = len(selected_targets) - n_patient
                st.caption(f"Patient Arrivals: {n_patient} | Medical Reasons: {n_reason}")
            else:
                st.warning("⚠️ Please select at least one target column")

        st.markdown("---")

        # ============================================================
        # SEASONAL PROPORTIONS CONFIGURATION
        # ============================================================
        st.markdown("#### 📊 Seasonal Category Distribution")
        st.caption("Distribute total patient forecast into clinical categories using seasonal patterns")

        enable_seasonal = st.checkbox(
            "Enable Seasonal Proportions",
            value=True,
            key="enable_seasonal_proportions",
            help="Apply day-of-week and monthly patterns to distribute total forecasts across clinical categories"
        )

        if enable_seasonal:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                use_dow = st.checkbox(
                    "Day-of-Week Seasonality",
                    value=True,
                    key="seasonal_use_dow",
                    help="Use different proportions for each day of the week"
                )
                use_monthly = st.checkbox(
                    "Monthly Seasonality",
                    value=True,
                    key="seasonal_use_monthly",
                    help="Use different proportions for each month (requires ≥12 months data)"
                )
            with col_s2:
                stl_period = st.selectbox(
                    "STL Period",
                    options=[7, 30, 365],
                    index=0,
                    key="seasonal_stl_period",
                    help="Seasonal period for STL decomposition: 7=weekly, 30=monthly, 365=yearly"
                )
                stl_model = st.selectbox(
                    "STL Model",
                    options=["additive", "multiplicative"],
                    index=0,
                    key="seasonal_stl_model",
                    help="Additive: Y = Trend + Seasonal + Residual; Multiplicative: Y = Trend × Seasonal × Residual"
                )

            # Store configuration in session state
            from app_core.analytics.seasonal_proportions import SeasonalProportionConfig
            st.session_state["seasonal_proportions_config"] = SeasonalProportionConfig(
                use_dow_seasonality=use_dow,
                use_monthly_seasonality=use_monthly,
                stl_period=stl_period,
                stl_model=stl_model
            )

            st.success("✅ Seasonal proportions will be applied to forecasts")
        else:
            st.info("ℹ️ Seasonal proportions disabled. Only total patient forecast will be generated.")

        st.markdown("---")

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

        # Check forecast mode (from Configuration tab radio button)
        forecast_mode = st.session_state.get("forecast_mode_radio", "Single-Target (Patient Arrivals Only)")
        is_multi_target = "Multi-Target" in forecast_mode

        if is_multi_target:
            multi_targets = st.session_state.get("multi_target_columns", [])
            if multi_targets:
                st.info(f"📊 **Multi-Target Mode**: Forecasting {len(multi_targets)} targets")
            else:
                st.warning("⚠️ No targets selected. Go to Configuration tab to select targets.")

        if current_model == "ARIMA":
            order = st.session_state.get("arima_order")
            use_auto = order is None
            arima_h = int(st.session_state.get("arima_h", 7))

            # ============================================================
            # MULTI-TARGET ARIMA TRAINING
            # ============================================================
            if is_multi_target:
                multi_targets = st.session_state.get("multi_target_columns", [])
                if not multi_targets:
                    st.warning("⚠️ Please select target columns in the Configuration tab first.")
                else:
                    if st.button(f"🚀 Train ARIMA Multi-Target ({len(multi_targets)} targets)", use_container_width=True, type="primary", key="train_arima_multi_btn"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(target_name, idx, total):
                            progress = (idx + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Training ARIMA for: {target_name} ({idx + 1}/{total})")

                        t0 = time.time()
                        try:
                            multi_results = run_arima_multi_target_pipeline(
                                df=data.copy(),
                                target_columns=multi_targets,
                                order=order,
                                train_ratio=train_ratio,
                                auto_select=use_auto,
                                search_mode="aic_only",
                                cv_strategy=st.session_state.get("cv_strategy", "expanding"),
                                progress_callback=update_progress,
                            )
                            runtime_s = time.time() - t0
                            st.session_state["arima_multi_target_results"] = multi_results
                            progress_bar.progress(1.0)
                            status_text.empty()

                            # Display summary
                            successful = len(multi_results.get("successful_targets", []))
                            failed = len(multi_results.get("failed_targets", []))
                            st.success(f"✅ ARIMA Multi-Target training completed in {runtime_s:.2f}s! ({successful} successful, {failed} failed)")

                            # Show summary table
                            summary_df = multi_results.get("summary")
                            if summary_df is not None and not summary_df.empty:
                                st.markdown("### Multi-Target Results Summary")
                                st.dataframe(summary_df.style.format({
                                    "MAE": "{:.3f}",
                                    "RMSE": "{:.3f}",
                                    "MAPE_%": "{:.2f}",
                                    "Accuracy_%": "{:.2f}",
                                    "R2": "{:.3f}",
                                    "Direction_Acc_%": "{:.2f}",
                                }, na_rep="—"), use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ ARIMA Multi-Target training failed: {e}")
                            progress_bar.empty()
                            status_text.empty()

            # ============================================================
            # SINGLE-TARGET ARIMA TRAINING (Original)
            # ============================================================
            else:
                if st.button("🚀 Train ARIMA (multi-horizon)", use_container_width=True, type="primary", key="train_arima_btn"):
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

                            # ============================================================
                            # APPLY SEASONAL PROPORTIONS
                            # ============================================================
                            if st.session_state.get("enable_seasonal_proportions", False):
                                try:
                                    from app_core.analytics.seasonal_proportions import (
                                        calculate_seasonal_proportions,
                                        distribute_forecast_to_categories
                                    )

                                    with st.spinner("📊 Calculating seasonal proportions..."):
                                        sp_config = st.session_state.get("seasonal_proportions_config")
                                        sp_result = calculate_seasonal_proportions(data, sp_config, date_col="Date")

                                        # Get forecast dates (last 7 days of best horizon forecast)
                                        best_h = int(best_row.get('Horizon', 7))
                                        if 'forecasts' in arima_mh_out and best_h in arima_mh_out['forecasts']:
                                            forecast_series = arima_mh_out['forecasts'][best_h]
                                            if isinstance(forecast_series, pd.Series):
                                                forecast_dates = forecast_series.index
                                            else:
                                                # Generate date range for forecast
                                                last_date = pd.to_datetime(data['Date'].max())
                                                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=best_h, freq='D')
                                                forecast_series = pd.Series(forecast_series, index=forecast_dates)

                                            # Distribute forecast to categories
                                            category_forecasts = distribute_forecast_to_categories(
                                                forecast_series,
                                                sp_result.dow_proportions,
                                                sp_result.monthly_proportions
                                            )

                                            sp_result.category_forecasts = category_forecasts

                                        # Store results
                                        st.session_state["seasonal_proportions_result"] = sp_result
                                        st.success(f"✅ Seasonal proportions calculated and applied!")

                                except Exception as e:
                                    st.warning(f"⚠️ Seasonal proportions calculation failed: {e}")

                        except Exception as e:
                            st.error(f"❌ ARIMA multi-horizon training failed: {e}")

        elif current_model == "SARIMAX":
            exog_vars      = st.session_state.get("sarimax_features", [])
            season_len     = int(st.session_state.get("sarimax_season_length", 7))
            horizons       = int(st.session_state.get("sarimax_horizons", 7))
            order          = st.session_state.get("sarimax_order")
            seasonal_order = st.session_state.get("sarimax_seasonal_order")

            # ============================================================
            # MULTI-TARGET SARIMAX TRAINING
            # ============================================================
            if is_multi_target:
                multi_targets = st.session_state.get("multi_target_columns", [])
                if not multi_targets:
                    st.warning("⚠️ Please select target columns in the Configuration tab first.")
                else:
                    if st.button(f"🚀 Train SARIMAX Multi-Target ({len(multi_targets)} targets)", use_container_width=True, type="primary", key="train_sarimax_multi_btn"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(target_name, idx, total):
                            progress = (idx + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Training SARIMAX for: {target_name} ({idx + 1}/{total})")

                        t0 = time.time()
                        try:
                            multi_results = run_sarimax_multi_target_pipeline(
                                df=data.copy(),
                                target_columns=multi_targets,
                                date_col="Date",
                                train_ratio=train_ratio,
                                season_length=season_len,
                                use_all_features=(len(exog_vars) == 0),
                                include_dow_ohe=True,
                                order=order,
                                seasonal_order=seasonal_order,
                                selected_features=exog_vars if exog_vars else None,
                                search_mode="aic_only",
                                progress_callback=update_progress,
                            )
                            runtime_s = time.time() - t0
                            st.session_state["sarimax_multi_target_results"] = multi_results
                            progress_bar.progress(1.0)
                            status_text.empty()

                            # Display summary
                            successful = len(multi_results.get("successful_targets", []))
                            failed = len(multi_results.get("failed_targets", []))
                            st.success(f"✅ SARIMAX Multi-Target training completed in {runtime_s:.2f}s! ({successful} successful, {failed} failed)")

                            # Show summary table
                            summary_df = multi_results.get("summary")
                            if summary_df is not None and not summary_df.empty:
                                st.markdown("### Multi-Target Results Summary")
                                st.dataframe(summary_df.style.format({
                                    "MAE": "{:.3f}",
                                    "RMSE": "{:.3f}",
                                    "MAPE_%": "{:.2f}",
                                    "Accuracy_%": "{:.2f}",
                                    "R2": "{:.3f}",
                                    "Direction_Acc_%": "{:.2f}",
                                }, na_rep="—"), use_container_width=True)

                            # ============================================================
                            # APPLY SEASONAL PROPORTIONS
                            # ============================================================
                            if st.session_state.get("enable_seasonal_proportions", False):
                                try:
                                    from app_core.analytics.seasonal_proportions import (
                                        calculate_seasonal_proportions,
                                        distribute_forecast_to_categories
                                    )

                                    with st.spinner("📊 Calculating seasonal proportions..."):
                                        sp_config = st.session_state.get("seasonal_proportions_config")
                                        sp_result = calculate_seasonal_proportions(data, sp_config, date_col="Date")

                                        # Get forecast from first successful target as reference
                                        successful_targets = multi_results.get("successful_targets", [])
                                        if successful_targets and 'target_results' in multi_results:
                                            first_target = successful_targets[0]
                                            target_result = multi_results['target_results'].get(first_target, {})

                                            if 'forecast' in target_result:
                                                forecast_data = target_result['forecast']
                                                if isinstance(forecast_data, pd.Series):
                                                    forecast_series = forecast_data
                                                else:
                                                    # Generate date range
                                                    last_date = pd.to_datetime(data['Date'].max())
                                                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_data), freq='D')
                                                    forecast_series = pd.Series(forecast_data, index=forecast_dates)

                                                # Distribute forecast to categories
                                                category_forecasts = distribute_forecast_to_categories(
                                                    forecast_series,
                                                    sp_result.dow_proportions,
                                                    sp_result.monthly_proportions
                                                )

                                                sp_result.category_forecasts = category_forecasts

                                        # Store results
                                        st.session_state["seasonal_proportions_result"] = sp_result
                                        st.success(f"✅ Seasonal proportions calculated and applied!")

                                except Exception as e:
                                    st.warning(f"⚠️ Seasonal proportions calculation failed: {e}")

                        except Exception as e:
                            st.error(f"❌ SARIMAX Multi-Target training failed: {e}")
                            progress_bar.empty()
                            status_text.empty()

            # ============================================================
            # SINGLE-TARGET SARIMAX TRAINING (Original)
            # ============================================================
            else:
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
                            search_mode="aic_only",  # Fast AIC-only search (4-5x faster than hybrid)
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

                            # ============================================================
                            # APPLY SEASONAL PROPORTIONS
                            # ============================================================
                            if st.session_state.get("enable_seasonal_proportions", False):
                                try:
                                    from app_core.analytics.seasonal_proportions import (
                                        calculate_seasonal_proportions,
                                        distribute_forecast_to_categories
                                    )

                                    with st.spinner("📊 Calculating seasonal proportions..."):
                                        sp_config = st.session_state.get("seasonal_proportions_config")
                                        sp_result = calculate_seasonal_proportions(data, sp_config, date_col="Date")

                                        # Get forecast from results
                                        best_h = int(best_row.get('Horizon', 7))
                                        if 'forecasts' in sarimax_out and best_h in sarimax_out['forecasts']:
                                            forecast_data = sarimax_out['forecasts'][best_h]
                                            if isinstance(forecast_data, pd.Series):
                                                forecast_series = forecast_data
                                            else:
                                                # Generate date range
                                                last_date = pd.to_datetime(data['Date'].max())
                                                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=best_h, freq='D')
                                                forecast_series = pd.Series(forecast_data, index=forecast_dates)

                                            # Distribute forecast to categories
                                            category_forecasts = distribute_forecast_to_categories(
                                                forecast_series,
                                                sp_result.dow_proportions,
                                                sp_result.monthly_proportions
                                            )

                                            sp_result.category_forecasts = category_forecasts

                                        # Store results
                                        st.session_state["seasonal_proportions_result"] = sp_result
                                        st.success(f"✅ Seasonal proportions calculated and applied!")

                                except Exception as e:
                                    st.warning(f"⚠️ Seasonal proportions calculation failed: {e}")

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
            st.markdown("### 🧪 Comprehensive Multi-Horizon Results — ARIMA")
            arima_mh = st.session_state.get("arima_mh_results")
            if arima_mh is None:
                st.info("📊 Train ARIMA first to see results in this tab.")
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

        # ============================================================
        # MULTI-TARGET RESULTS (Reason for Visit Forecasting)
        # ============================================================
        multi_target_results = st.session_state.get("arima_multi_target_results") or st.session_state.get("sarimax_multi_target_results")

        if multi_target_results:
            st.markdown("---")
            st.markdown(
                f"""
                <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 1.5rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(34, 197, 94, 0.1)); border: 2px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);'>
                  <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🏥</div>
                  <h2 style='font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, {SUCCESS_COLOR}, #22c55e); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Multi-Target Results (Reason for Visit)</h2>
                  <p style='font-size: 0.9rem; color: {BODY_TEXT};'>Forecasting performance for patient arrivals AND medical reasons</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Summary statistics
            summary_df = multi_target_results.get("summary")
            successful = multi_target_results.get("successful_targets", [])
            failed = multi_target_results.get("failed_targets", [])

            # KPI row for multi-target
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Targets Trained", len(successful), delta=None)
            with col2:
                st.metric("Failed", len(failed), delta=None, delta_color="inverse" if failed else "off")
            with col3:
                if summary_df is not None and not summary_df.empty and "MAE" in summary_df.columns:
                    avg_mae = summary_df["MAE"].mean()
                    st.metric("Avg MAE", f"{avg_mae:.2f}" if pd.notna(avg_mae) else "—")
                else:
                    st.metric("Avg MAE", "—")
            with col4:
                if summary_df is not None and not summary_df.empty and "RMSE" in summary_df.columns:
                    avg_rmse = summary_df["RMSE"].mean()
                    st.metric("Avg RMSE", f"{avg_rmse:.2f}" if pd.notna(avg_rmse) else "—")
                else:
                    st.metric("Avg RMSE", "—")

            # Summary table
            if summary_df is not None and not summary_df.empty:
                st.markdown("#### 📋 Multi-Target Performance Summary")

                # Separate patient arrivals from medical reasons
                patient_df = summary_df[summary_df["Target"].str.lower().str.startswith("target_")].copy()
                reason_df = summary_df[~summary_df["Target"].str.lower().str.startswith("target_")].copy()

                if not patient_df.empty:
                    st.markdown("##### 👥 Patient Arrivals")
                    st.dataframe(
                        patient_df.style.format({
                            "MAE": "{:.3f}",
                            "RMSE": "{:.3f}",
                            "MAPE_%": "{:.2f}",
                            "Accuracy_%": "{:.2f}",
                            "R2": "{:.3f}",
                        }, na_rep="—").background_gradient(subset=["Accuracy_%"], cmap="Greens"),
                        use_container_width=True,
                        hide_index=True
                    )

                if not reason_df.empty:
                    st.markdown("##### 🩺 Medical Reasons for Visit")
                    st.dataframe(
                        reason_df.style.format({
                            "MAE": "{:.3f}",
                            "RMSE": "{:.3f}",
                            "MAPE_%": "{:.2f}",
                            "Accuracy_%": "{:.2f}",
                            "R2": "{:.3f}",
                        }, na_rep="—").background_gradient(subset=["Accuracy_%"], cmap="Greens"),
                        use_container_width=True,
                        hide_index=True
                    )

                # Quick comparison chart
                if "Accuracy_%" in summary_df.columns:
                    st.markdown("#### 📊 Performance Comparison Chart")
                    chart_df = summary_df.dropna(subset=["Accuracy_%"]).sort_values("Accuracy_%", ascending=True)
                    if not chart_df.empty:
                        fig = px.bar(
                            chart_df,
                            x="Accuracy_%",
                            y="Target",
                            orientation="h",
                            title="Forecast Accuracy by Target",
                            color="Accuracy_%",
                            color_continuous_scale="Greens",
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color=BODY_TEXT,
                            height=max(400, len(chart_df) * 25),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Show failed targets if any
            if failed:
                with st.expander(f"⚠️ Failed Targets ({len(failed)})", expanded=False):
                    for target in failed:
                        result = multi_target_results.get(target, {})
                        error_msg = result.get("message", "Unknown error")
                        st.error(f"**{target}**: {error_msg}")

        # ============================================================
        # SEASONAL PROPORTIONS VISUALIZATIONS
        # ============================================================
        if st.session_state.get("enable_seasonal_proportions", False) and st.session_state.get("seasonal_proportions_result") is not None:
            st.markdown("---")
            st.markdown(
                f"""
                <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 1.5rem; background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(14, 165, 233, 0.1)); border: 2px solid rgba(34, 211, 238, 0.3); box-shadow: 0 0 30px rgba(34, 211, 238, 0.2);'>
                  <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
                  <h2 style='font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Seasonal Category Distribution</h2>
                  <p style='font-size: 0.9rem; color: {BODY_TEXT};'>Day-of-week and monthly patterns for clinical category forecasts</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            sp_result = st.session_state.get("seasonal_proportions_result")

            sp_tab1, sp_tab2, sp_tab3 = st.tabs([
                "📈 STL Decomposition",
                "🔥 Seasonal Heatmap",
                "📊 Category Forecasts"
            ])

            with sp_tab1:
                st.markdown("#### STL Decomposition (Seasonal-Trend-Loess)")
                st.caption("Decompose time series into observed, trend, seasonal, and residual components")

                # Category selector for STL
                available_categories = [cat for cat, res in sp_result.stl_results.items() if res is not None]

                if available_categories:
                    selected_cat = st.selectbox(
                        "Select category to visualize:",
                        options=available_categories,
                        key="stl_category_selector"
                    )

                    from app_core.analytics.seasonal_proportions import create_stl_decomposition_plot
                    fig_stl = create_stl_decomposition_plot(sp_result.stl_results, selected_cat)
                    st.plotly_chart(fig_stl, use_container_width=True)
                else:
                    st.info("ℹ️ STL decomposition not available. Insufficient data or decomposition failed.")

            with sp_tab2:
                st.markdown("#### Seasonal Proportion Heatmaps")
                st.caption("Historical patterns showing how category proportions vary by day and month")

                from app_core.analytics.seasonal_proportions import create_seasonal_heatmap

                col_dow, col_monthly = st.columns(2)

                with col_dow:
                    st.markdown("##### Day-of-Week Patterns")
                    fig_dow = create_seasonal_heatmap(
                        dow_proportions=sp_result.dow_proportions,
                        monthly_proportions=None,
                        title="Day-of-Week Proportions"
                    )
                    st.plotly_chart(fig_dow, use_container_width=True)

                with col_monthly:
                    st.markdown("##### Monthly Patterns")
                    fig_monthly = create_seasonal_heatmap(
                        dow_proportions=None,
                        monthly_proportions=sp_result.monthly_proportions,
                        title="Monthly Proportions"
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)

                # Show proportion tables
                with st.expander("📋 View Proportion Tables", expanded=False):
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("**Day-of-Week Proportions (%)**")
                        dow_pct = (sp_result.dow_proportions * 100).round(2)
                        st.dataframe(dow_pct, use_container_width=True)

                    with col_t2:
                        st.markdown("**Monthly Proportions (%)**")
                        monthly_pct = (sp_result.monthly_proportions * 100).round(2)
                        st.dataframe(monthly_pct, use_container_width=True)

            with sp_tab3:
                st.markdown("#### 7-Day Category Forecast Distribution")
                st.caption("Total patient forecast distributed across clinical categories using seasonal patterns")

                if sp_result.category_forecasts is not None and not sp_result.category_forecasts.empty:
                    from app_core.analytics.seasonal_proportions import create_category_forecast_chart

                    # Create and display chart
                    fig_forecast = create_category_forecast_chart(sp_result.category_forecasts)
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Show forecast table
                    st.markdown("##### Forecast Table")
                    forecast_display = sp_result.category_forecasts.copy()
                    forecast_display = forecast_display.round(2)
                    forecast_display['Total'] = forecast_display.sum(axis=1)

                    st.dataframe(
                        forecast_display.style.background_gradient(cmap="Blues", axis=None),
                        use_container_width=True
                    )

                    # Summary statistics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        total_forecast = forecast_display['Total'].sum()
                        st.metric("Total 7-Day Forecast", f"{int(total_forecast):,}")
                    with col_s2:
                        avg_daily = forecast_display['Total'].mean()
                        st.metric("Avg Daily Forecast", f"{int(avg_daily):,}")
                    with col_s3:
                        dominant_cat = sp_result.category_forecasts.sum().idxmax()
                        dominant_pct = (sp_result.category_forecasts.sum()[dominant_cat] / sp_result.category_forecasts.sum().sum() * 100)
                        st.metric("Dominant Category", f"{dominant_cat} ({dominant_pct:.1f}%)")

                else:
                    st.info("ℹ️ No category forecasts available. Train a model first to generate forecasts.")

        # ============================================================
        # CATEGORY PROPORTIONS (Probability-Based Approach)
        # ============================================================
        st.markdown("---")
        st.markdown(
            f"""
            <div class='hf-feature-card' style='text-align: center; margin: 1.5rem 0; padding: 1.5rem; background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(139, 92, 246, 0.1)); border: 2px solid rgba(167, 139, 250, 0.3); box-shadow: 0 0 30px rgba(167, 139, 250, 0.2);'>
              <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
              <h2 style='font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, #a78bfa, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Category Proportions (Probability Approach)</h2>
              <p style='font-size: 0.9rem; color: {BODY_TEXT};'>Historical distribution used in Dashboard for consistent category forecasts</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Calculate category proportions from processed_df (contains clinical categories)
        merged_df = st.session_state.get("processed_df")
        if merged_df is None or getattr(merged_df, 'empty', True):
            # Fallback to merged_data if processed_df not available
            merged_df = st.session_state.get("merged_data")
        category_config = {
            "RESPIRATORY": {"icon": "🫁", "color": "#3b82f6"},
            "CARDIAC": {"icon": "❤️", "color": "#ef4444"},
            "TRAUMA": {"icon": "🩹", "color": "#f59e0b"},
            "GASTROINTESTINAL": {"icon": "🤢", "color": "#10b981"},
            "INFECTIOUS": {"icon": "🦠", "color": "#8b5cf6"},
            "NEUROLOGICAL": {"icon": "🧠", "color": "#ec4899"},
            "OTHER": {"icon": "📋", "color": "#6b7280"},
        }

        if merged_df is not None and not merged_df.empty:
            # Build a column map for case-insensitive matching
            col_map = {col.upper(): col for col in merged_df.columns}

            # Look for category columns and sum ALL matching columns
            cat_totals = {}
            for cat in category_config.keys():
                cat_upper = cat.upper()

                # Find all columns that match this category
                matching_cols = []
                for col_upper_key, original_col in col_map.items():
                    if col_upper_key == cat_upper or col_upper_key.startswith(cat_upper + "_"):
                        matching_cols.append(original_col)

                # Sum all matching columns
                if matching_cols:
                    total_for_cat = 0
                    for col in matching_cols:
                        col_sum = merged_df[col].sum()
                        if pd.notna(col_sum):
                            total_for_cat += col_sum
                    if total_for_cat > 0:
                        cat_totals[cat] = total_for_cat

            if cat_totals:
                total_sum = sum(cat_totals.values())
                proportions = {cat: (val / total_sum * 100) for cat, val in cat_totals.items()}

                st.markdown("#### 📈 Historical Category Distribution")
                st.info("💡 **How it works**: The Dashboard uses these proportions to distribute the total patient forecast into categories. This ensures category totals always sum to the forecasted total.")

                # Display proportions in columns
                prop_cols = st.columns(len(category_config))
                for idx, (cat, cfg) in enumerate(category_config.items()):
                    with prop_cols[idx]:
                        prop = proportions.get(cat, 0)
                        total = cat_totals.get(cat, 0)
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
                            border-radius: 12px;
                            padding: 1rem 0.5rem;
                            text-align: center;
                            border: 1px solid {cfg['color']}40;
                        '>
                            <div style='font-size: 1.5rem; margin-bottom: 0.25rem;'>{cfg['icon']}</div>
                            <div style='font-size: 0.625rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;'>{cat[:4]}</div>
                            <div style='font-size: 1.5rem; font-weight: 800; color: {cfg['color']}; margin: 0.375rem 0;'>{prop:.1f}%</div>
                            <div style='font-size: 0.5rem; color: #64748b;'>~{int(total):,} total</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Summary
                st.markdown("#### 📋 Proportion Summary Table")
                prop_df = pd.DataFrame([
                    {
                        "Category": cat,
                        "Icon": cfg["icon"],
                        "Historical Total": int(cat_totals.get(cat, 0)),
                        "Proportion (%)": round(proportions.get(cat, 0), 2)
                    }
                    for cat, cfg in category_config.items()
                ])
                prop_df = prop_df.sort_values("Proportion (%)", ascending=False)
                st.dataframe(
                    prop_df.style.background_gradient(subset=["Proportion (%)"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True
                )

                # Example calculation
                st.markdown("#### 🧮 Example Calculation")
                example_total = 80
                st.markdown(f"If Total Forecast = **{example_total} patients**:")
                example_cols = st.columns(len(category_config))
                for idx, (cat, cfg) in enumerate(category_config.items()):
                    with example_cols[idx]:
                        prop = proportions.get(cat, 0)
                        calculated = int(round(example_total * prop / 100))
                        st.metric(f"{cfg['icon']} {cat[:4]}", f"{calculated}")

                sum_example = sum(int(round(example_total * proportions.get(cat, 0) / 100)) for cat in category_config.keys())
                st.success(f"✅ Sum of categories = **{sum_example}** (matches total of {example_total})")
            else:
                st.warning("⚠️ No category columns found in merged data. Enable 'Aggregated Clinical Categories' in Data Preparation Studio.")
        else:
            st.warning("⚠️ Merged data not available. Please run the Data Fusion pipeline first.")

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

            # ========== NEW: Statistical Comparison Section ==========
            if len(show) >= 2:
                st.markdown(
                    """
                    <div style='text-align: center; margin: 1.5rem 0; padding: 1.5rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1)); border: 2px solid rgba(99, 102, 241, 0.3); border-radius: 12px; box-shadow: 0 0 30px rgba(99, 102, 241, 0.2);'>
                      <h3 style='margin: 0; font-size: 1.5rem; color: #818cf8;'>📊 Statistical Comparison</h3>
                      <p style='margin: 0.5rem 0 0 0; font-size: 0.95rem; color: rgba(255,255,255,0.7);'>In-depth analysis of model performance across all dimensions</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 1. Interactive metrics over horizon chart
                horizon_data = _extract_horizon_comparison_data()

                if horizon_data and len(horizon_data) >= 2:
                    st.markdown("##### 📊 Metrics Across Forecast Horizons")

                    # Radio buttons for metric selection
                    selected_metric = st.radio(
                        "Choose metric to visualize across forecast horizons",
                        options=["MAE", "RMSE", "MAPE"],
                        index=0,
                        horizontal=True,
                        key="horizon_metric_selector",
                        label_visibility="collapsed",
                        help="Switch between different metrics to see how they change across horizons"
                    )

                    metrics_horizon_fig = _build_metrics_over_horizon_chart(horizon_data, selected_metric)
                    if metrics_horizon_fig:
                        st.plotly_chart(metrics_horizon_fig, use_container_width=True)
                    else:
                        st.warning(f"📭 Chart could not be generated for {selected_metric}. Check if this metric exists in both models.")

                    st.markdown("---")
                else:
                    if horizon_data:
                        st.info(f"📭 Only found data for {len(horizon_data)} model(s). Need both ARIMA and SARIMAX with multi-horizon results.")
                    else:
                        st.info("📭 No horizon data found. Train both ARIMA and SARIMAX models with multi-horizon forecasting.")

                    st.markdown("---")

                # 2. Interactive error metrics comparison with radio buttons
                st.markdown("##### 📈 Average Performance Comparison")
                view_mode = st.radio(
                    "Choose metrics to compare",
                    options=["Errors Only", "Percentages Only"],
                    index=0,
                    horizontal=True,
                    key="error_metrics_view_mode",
                    label_visibility="collapsed",
                    help="Switch between different metric comparison views"
                )

                # Dynamic interpretation above the chart
                interpretation = _generate_error_metrics_interpretation(show, view_mode)
                if interpretation:
                    st.markdown(interpretation, unsafe_allow_html=True)

                interactive_fig = _build_interactive_error_metrics_chart(show, view_mode)
                if interactive_fig:
                    st.plotly_chart(interactive_fig, use_container_width=True)

                st.markdown("---")

                # 3. Accuracy Gauge Cards (side-by-side)
                st.markdown("##### 🎯 Model Performance Summary")

                # Dynamic interpretation above the gauges
                radar_interpretation = _generate_radar_interpretation(show)
                if radar_interpretation:
                    st.markdown(radar_interpretation, unsafe_allow_html=True)

                # Determine winner and display gauges side-by-side
                if "Accuracy_%" in show.columns and show["Accuracy_%"].notna().any():
                    show_sorted = show.sort_values("Accuracy_%", ascending=False)
                    best_model_name = show_sorted.iloc[0]["Model"]

                    # Create two columns for side-by-side display
                    gauge_col1, gauge_col2 = st.columns(2)

                    # Get all rows as a list
                    rows_list = list(show.iterrows())

                    # Display first model in left column
                    if len(rows_list) >= 1:
                        _, row1 = rows_list[0]
                        is_winner1 = (row1["Model"] == best_model_name)
                        gauge_fig1 = _build_accuracy_gauge_card(row1, is_winner=is_winner1)
                        with gauge_col1:
                            st.plotly_chart(gauge_fig1, use_container_width=True)

                    # Display second model in right column
                    if len(rows_list) >= 2:
                        _, row2 = rows_list[1]
                        is_winner2 = (row2["Model"] == best_model_name)
                        gauge_fig2 = _build_accuracy_gauge_card(row2, is_winner=is_winner2)
                        with gauge_col2:
                            st.plotly_chart(gauge_fig2, use_container_width=True)

                # 4. Pairwise delta analysis (if 2 models)
                if len(show) == 2:
                    delta_fig = _build_pairwise_delta_chart(show)
                    if delta_fig:
                        st.plotly_chart(delta_fig, use_container_width=True)

            st.markdown("---")

            # Best model recommendation (simplified)
            st.markdown("#### 🏆 Final Recommendation")
            try:
                best_model = None
                if "Accuracy_%" in show.columns and show["Accuracy_%"].notna().any():
                    show_sorted = show.sort_values("Accuracy_%", ascending=False)
                    best_model = show_sorted.iloc[0]

                    if len(show) == 2:
                        second_model = show_sorted.iloc[1]

                        # Calculate improvements
                        acc_delta = best_model["Accuracy_%"] - second_model["Accuracy_%"]
                        mae_delta = second_model["MAE"] - best_model["MAE"] if "MAE" in show.columns else 0
                        rmse_delta = second_model["RMSE"] - best_model["RMSE"] if "RMSE" in show.columns else 0
                        runtime_delta = second_model["Runtime_s"] - best_model["Runtime_s"] if "Runtime_s" in show.columns else 0

                        # Build recommendation message
                        st.success(
                            f"**🥇 Recommended Model: {best_model['Model']}**\n\n"
                            f"**Performance Edge:**\n"
                            f"• {acc_delta:+.2f}% higher accuracy ({best_model['Accuracy_%']:.2f}% vs {second_model['Accuracy_%']:.2f}%)\n"
                            f"• {mae_delta:+.2f} lower MAE ({best_model.get('MAE', 0):.2f} vs {second_model.get('MAE', 0):.2f})\n"
                            f"• {rmse_delta:+.2f} lower RMSE ({best_model.get('RMSE', 0):.2f} vs {second_model.get('RMSE', 0):.2f})\n"
                            f"• {runtime_delta:+.1f}s runtime difference ({'faster' if runtime_delta < 0 else 'slower'})\n\n"
                            f"**Decision:** Choose **{best_model['Model']}** for {acc_delta:.1f}% accuracy improvement."
                        )
                    else:
                        # Multiple models - simple recommendation
                        st.success(
                            f"**🥇 Recommended Model: {best_model['Model']}**\n\n"
                            f"• Accuracy: {best_model['Accuracy_%']:.2f}%\n"
                            f"• MAE: {best_model.get('MAE', 0):.2f}\n"
                            f"• RMSE: {best_model.get('RMSE', 0):.2f}\n"
                            f"• Runtime: {best_model.get('Runtime_s', 0):.1f}s"
                        )

                    # Parameters in expander
                    with st.expander("📋 View Model Parameters", expanded=False):
                        st.code(best_model.get("ModelParams", "N/A"), language=None)

            except Exception as e:
                st.error(f"Could not determine best model: {e}")

            # Model summaries
            if st.checkbox("📖 Show Detailed Model Summaries", value=False):
                st.markdown("---")
                st.markdown("#### 📊 Detailed Model Performance Cards")
                summaries = _create_model_performance_summary(show)
                for summary_html in summaries:
                    st.markdown(summary_html, unsafe_allow_html=True)

def _generate_error_metrics_interpretation(comp_df: pd.DataFrame, view_mode: str) -> str:
    """
    Generate dynamic interpretation for error metrics comparison based on actual results.

    Args:
        comp_df: DataFrame with model comparison data
        view_mode: Current view mode selection

    Returns:
        HTML string with interpretation
    """
    if comp_df is None or comp_df.empty or len(comp_df) < 2:
        return ""

    df = comp_df.copy()

    # Get best model by accuracy
    if "Accuracy_%" in df.columns and df["Accuracy_%"].notna().any():
        df_sorted = df.sort_values("Accuracy_%", ascending=False)
        best_model = df_sorted.iloc[0]
        worst_model = df_sorted.iloc[-1]

        best_name = best_model["Model"]
        best_acc = best_model["Accuracy_%"]

        # Calculate performance gaps
        acc_gap = best_acc - worst_model["Accuracy_%"]

        # Build interpretation
        if len(df) == 2:
            # Two-model comparison
            other_model = df_sorted.iloc[1]
            other_name = other_model["Model"]

            # Determine significance
            if acc_gap > 5:
                significance = "significantly outperforms"
                icon = "🏆"
            elif acc_gap > 2:
                significance = "outperforms"
                icon = "✅"
            else:
                significance = "marginally outperforms"
                icon = "📊"

            interpretation = f"{icon} **{best_name}** {significance} **{other_name}** with **{best_acc:.2f}% accuracy** (Δ = {acc_gap:+.2f}%)"

            # Add error comparison if available
            if "MAE" in df.columns and "RMSE" in df.columns:
                mae_diff = best_model["MAE"] - other_model["MAE"]
                rmse_diff = best_model["RMSE"] - other_model["RMSE"]

                if mae_diff < 0 and rmse_diff < 0:
                    interpretation += f", with **{abs(mae_diff):.2f} lower MAE** and **{abs(rmse_diff):.2f} lower RMSE**"
                elif mae_diff < 0:
                    interpretation += f", with **{abs(mae_diff):.2f} lower MAE**"
                elif rmse_diff < 0:
                    interpretation += f", with **{abs(rmse_diff):.2f} lower RMSE**"

            interpretation += "."
        else:
            # Multiple models
            interpretation = f"🏆 **{best_name}** leads with **{best_acc:.2f}% accuracy**, followed by "
            second = df_sorted.iloc[1]
            interpretation += f"**{second['Model']}** ({second['Accuracy_%']:.2f}%)."

        return f"""
        <div style='padding: 0.8rem; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1)); border-left: 4px solid #3b82f6; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.95rem; color: rgba(255,255,255,0.9);'>{interpretation}</p>
        </div>
        """

    return ""


def _generate_radar_interpretation(comp_df: pd.DataFrame) -> str:
    """
    Generate dynamic interpretation for radar chart based on actual results.

    Args:
        comp_df: DataFrame with model comparison data

    Returns:
        HTML string with interpretation
    """
    if comp_df is None or comp_df.empty or len(comp_df) < 2:
        return ""

    df = comp_df.copy()

    # Analyze strengths and weaknesses
    if "Accuracy_%" in df.columns and "Runtime_s" in df.columns:
        df_sorted_acc = df.sort_values("Accuracy_%", ascending=False)
        df_sorted_speed = df.sort_values("Runtime_s", ascending=True)

        most_accurate = df_sorted_acc.iloc[0]
        fastest = df_sorted_speed.iloc[0]

        if most_accurate["Model"] == fastest["Model"]:
            # Same model is best in both
            interpretation = f"💎 **{most_accurate['Model']}** dominates across all dimensions: highest accuracy ({most_accurate['Accuracy_%']:.2f}%) and fastest runtime ({most_accurate['Runtime_s']:.1f}s)."
        else:
            # Trade-off between accuracy and speed
            runtime_diff = fastest["Runtime_s"] - most_accurate["Runtime_s"]
            acc_diff = most_accurate["Accuracy_%"] - fastest["Accuracy_%"]

            interpretation = f"⚖️ Performance trade-off: **{most_accurate['Model']}** offers **{acc_diff:.2f}% better accuracy** but is **{abs(runtime_diff):.1f}s slower** than **{fastest['Model']}**."

        return f"""
        <div style='padding: 0.8rem; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.1)); border-left: 4px solid #8b5cf6; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.95rem; color: rgba(255,255,255,0.9);'>{interpretation}</p>
        </div>
        """

    return ""


def _build_interactive_error_metrics_chart(comp_df: pd.DataFrame, view_mode: str = "All Metrics"):
    """
    Create interactive error metrics chart with dual y-axis support.

    Args:
        comp_df: DataFrame with model comparison data
        view_mode: One of ["All Metrics", "Errors Only", "Percentages Only", "Accuracy vs Errors"]

    Returns:
        Plotly figure with dual y-axis or None
    """
    if comp_df is None or comp_df.empty or len(comp_df) < 2:
        return None

    from plotly.subplots import make_subplots

    # Prepare data
    df_work = comp_df.copy()
    for c in ["MAE", "RMSE", "MAPE_%", "Accuracy_%"]:
        if c in df_work.columns:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

    # Determine which metrics to show based on view mode
    show_percentages = view_mode in ["All Metrics", "Percentages Only", "Accuracy vs Errors"]
    show_errors = view_mode in ["All Metrics", "Errors Only", "Accuracy vs Errors"]

    # Create figure with or without secondary y-axis
    use_dual_axis = view_mode in ["All Metrics", "Accuracy vs Errors"]

    if use_dual_axis:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Count total number of bars to adjust width
    n_bars = 0
    if show_percentages:
        if "MAPE_%" in df_work.columns and df_work["MAPE_%"].notna().any():
            n_bars += 1
        if "Accuracy_%" in df_work.columns and df_work["Accuracy_%"].notna().any():
            n_bars += 1
    if show_errors:
        if "MAE" in df_work.columns and df_work["MAE"].notna().any():
            n_bars += 1
        if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
            n_bars += 1

    # Adjust bar width (smaller for more bars)
    bar_width = 0.15 if n_bars >= 4 else 0.2

    # Add percentage metrics (left y-axis)
    if show_percentages:
        if "MAPE_%" in df_work.columns and df_work["MAPE_%"].notna().any():
            trace = go.Bar(
                name="MAPE %",
                x=df_work["Model"],
                y=df_work["MAPE_%"],
                marker_color='#f59e0b',  # Amber
                text=df_work["MAPE_%"].round(1),
                textposition='outside',
                texttemplate='%{text}%',
                textfont=dict(size=10, color='#f59e0b'),
                hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>',
                width=bar_width,
                offsetgroup=1 if not use_dual_axis else None
            )
            if use_dual_axis:
                fig.add_trace(trace, secondary_y=False)
            else:
                fig.add_trace(trace)

        if "Accuracy_%" in df_work.columns and df_work["Accuracy_%"].notna().any():
            trace = go.Bar(
                name="Accuracy %",
                x=df_work["Model"],
                y=df_work["Accuracy_%"],
                marker_color=SUCCESS_COLOR,
                text=df_work["Accuracy_%"].round(2),
                textposition='outside',
                texttemplate='%{text}%',
                textfont=dict(size=10, color=SUCCESS_COLOR),
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>',
                width=bar_width,
                offsetgroup=2 if not use_dual_axis else None
            )
            if use_dual_axis:
                fig.add_trace(trace, secondary_y=False)
            else:
                fig.add_trace(trace)

    # Add error metrics (right y-axis for dual, same axis for single)
    if show_errors:
        if "MAE" in df_work.columns and df_work["MAE"].notna().any():
            trace = go.Bar(
                name="MAE",
                x=df_work["Model"],
                y=df_work["MAE"],
                marker_color=PRIMARY_COLOR,
                text=df_work["MAE"].round(2),
                textposition='outside',
                texttemplate='%{text}',
                textfont=dict(size=10, color=PRIMARY_COLOR),
                hovertemplate='<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>',
                width=bar_width,
                offsetgroup=3 if not use_dual_axis else None
            )
            if use_dual_axis:
                fig.add_trace(trace, secondary_y=True)
            else:
                fig.add_trace(trace)

        if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
            trace = go.Bar(
                name="RMSE",
                x=df_work["Model"],
                y=df_work["RMSE"],
                marker_color=SECONDARY_COLOR,
                text=df_work["RMSE"].round(2),
                textposition='outside',
                texttemplate='%{text}',
                textfont=dict(size=10, color=SECONDARY_COLOR),
                hovertemplate='<b>%{x}</b><br>RMSE: %{y:.3f}<extra></extra>',
                width=bar_width,
                offsetgroup=4 if not use_dual_axis else None
            )
            if use_dual_axis:
                fig.add_trace(trace, secondary_y=True)
            else:
                fig.add_trace(trace)

    # Calculate y-axis ranges with MORE padding for text labels
    if use_dual_axis:
        # Left y-axis (percentages)
        if show_percentages:
            max_pct = 0
            if "MAPE_%" in df_work.columns and df_work["MAPE_%"].notna().any():
                max_pct = max(max_pct, df_work["MAPE_%"].max())
            if "Accuracy_%" in df_work.columns and df_work["Accuracy_%"].notna().any():
                max_pct = max(max_pct, df_work["Accuracy_%"].max())
            y_pct_range = [0, max_pct * 1.4] if np.isfinite(max_pct) and max_pct > 0 else [0, 130]
        else:
            y_pct_range = None

        # Right y-axis (errors)
        if show_errors:
            max_error = 0
            if "MAE" in df_work.columns and df_work["MAE"].notna().any():
                max_error = max(max_error, df_work["MAE"].max())
            if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
                max_error = max(max_error, df_work["RMSE"].max())
            y_error_range = [0, max_error * 1.4] if np.isfinite(max_error) and max_error > 0 else None
        else:
            y_error_range = None

        # Update layout for dual y-axis
        fig.update_layout(
            title=dict(
                text=f"<b>📊 {view_mode}</b><br><sub>Interactive Model Comparison</sub>",
                x=0.5,
                xanchor='center',
                font=dict(size=14, color=TEXT_COLOR, family='Arial Black')
            ),
            xaxis=dict(
                title="<b>Model</b>",
                title_font=dict(size=12),
                tickangle=0,
                tickfont=dict(size=11),
                showgrid=False,
                color=TEXT_COLOR
            ),
            barmode='group',
            bargap=0.25,  # Gap between bars of different models
            bargroupgap=0.1,  # Gap between metric groups
            plot_bgcolor=BG_BLACK,
            paper_bgcolor=BG_BLACK,
            margin=dict(t=70, b=90, l=80, r=80),
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color=TEXT_COLOR, size=10)
            ),
            font=dict(color=TEXT_COLOR)
        )

        # Left y-axis (percentages)
        fig.update_yaxes(
            title_text="<b>Percentage (%)</b>",
            title_font=dict(size=12, color=TEXT_COLOR),
            tickfont=dict(size=10),
            range=y_pct_range,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            color=TEXT_COLOR,
            secondary_y=False
        )

        # Right y-axis (errors)
        fig.update_yaxes(
            title_text="<b>Error Value</b>",
            title_font=dict(size=12, color=TEXT_COLOR),
            tickfont=dict(size=10),
            range=y_error_range,
            showgrid=False,
            color=TEXT_COLOR,
            secondary_y=True
        )
    else:
        # Single y-axis
        max_val = 0
        if show_percentages:
            if "MAPE_%" in df_work.columns and df_work["MAPE_%"].notna().any():
                max_val = max(max_val, df_work["MAPE_%"].max())
            if "Accuracy_%" in df_work.columns and df_work["Accuracy_%"].notna().any():
                max_val = max(max_val, df_work["Accuracy_%"].max())
            y_title = "<b>Percentage (%)</b>"
        else:  # Errors only
            if "MAE" in df_work.columns and df_work["MAE"].notna().any():
                max_val = max(max_val, df_work["MAE"].max())
            if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
                max_val = max(max_val, df_work["RMSE"].max())
            y_title = "<b>Error Value (Lower is Better)</b>"

        y_range = [0, max_val * 1.4] if np.isfinite(max_val) and max_val > 0 else None

        fig.update_layout(
            title=dict(
                text=f"<b>📊 {view_mode}</b><br><sub>Interactive Model Comparison</sub>",
                x=0.5,
                xanchor='center',
                font=dict(size=14, color=TEXT_COLOR, family='Arial Black')
            ),
            xaxis=dict(
                title="<b>Model</b>",
                title_font=dict(size=12),
                tickangle=0,
                tickfont=dict(size=11),
                showgrid=False,
                color=TEXT_COLOR
            ),
            yaxis=dict(
                title=y_title,
                title_font=dict(size=12),
                tickfont=dict(size=10),
                range=y_range,
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)',
                color=TEXT_COLOR
            ),
            barmode='group',
            bargap=0.25,
            bargroupgap=0.1,
            plot_bgcolor=BG_BLACK,
            paper_bgcolor=BG_BLACK,
            margin=dict(t=70, b=90, l=80, r=50),
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color=TEXT_COLOR, size=10)
            ),
            font=dict(color=TEXT_COLOR)
        )

    return fig

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

    # Calculate y-axis range with padding for text labels
    max_val = max([df_work[m].max() for m, _, _ in metrics_to_plot if m in df_work.columns])
    y_range = [0, max_val * 1.25] if np.isfinite(max_val) else [0, 120]

    fig_metrics.update_layout(
        title=dict(text="📊 Multi-Metric Model Comparison", x=0.5, xanchor='center'),
        xaxis_title="Model",
        yaxis_title="Score (Higher is Better)",
        barmode='group',
        plot_bgcolor=BG_BLACK,
        paper_bgcolor=BG_BLACK,
        margin=dict(t=70, b=80, l=60, r=40),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(tickangle=-45, showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=y_range, showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
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
                xaxis=dict(title="Runtime (seconds)", showgrid=True, gridcolor='rgba(255,255,255,0.15)'),
                yaxis=dict(title="Accuracy (%)", showgrid=True, gridcolor='rgba(255,255,255,0.15)'),
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

        # Calculate y-axis range with padding for text labels
        max_error = 0
        if "MAE" in df_work.columns and df_work["MAE"].notna().any():
            max_error = max(max_error, df_work["MAE"].max())
        if "RMSE" in df_work.columns and df_work["RMSE"].notna().any():
            max_error = max(max_error, df_work["RMSE"].max())
        y_error_range = [0, max_error * 1.25] if np.isfinite(max_error) and max_error > 0 else None

        fig_errors.update_layout(
            title=dict(text="📉 Error Metrics Comparison (Lower is Better)", x=0.5, xanchor='center'),
            xaxis_title="Model",
            yaxis_title="Error Value",
            barmode='group',
            plot_bgcolor=BG_BLACK,
            paper_bgcolor=BG_BLACK,
            margin=dict(t=70, b=80, l=60, r=40),
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(tickangle=-45, showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(range=y_error_range, showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
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

# =============================================================================
# PAGE NAVIGATION
# =============================================================================
render_page_navigation(4)  # Baseline Models is page index 4


