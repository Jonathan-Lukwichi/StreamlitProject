# =============================================================================
# diagnostic_plots.py - Reusable Diagnostic Visualization Components
# =============================================================================
"""
Plotly-based diagnostic visualizations for model evaluation.

This module provides:
1. Residual diagnostic panel (6-plot grid)
2. ACF/PACF correlation plots
3. Actual vs Predicted scatter
4. Q-Q normality plot
5. Skill score gauges
6. Statistical test summary cards
7. Diebold-Mariano heatmap
8. Training loss curves

All plots follow the application's dark theme.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import theme colors (with fallback)
try:
    from app_core.ui.theme import (
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
        WARNING_COLOR, DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT
    )
except ImportError:
    # Fallback colors
    PRIMARY_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    SUCCESS_COLOR = "#22C55E"
    WARNING_COLOR = "#F59E0B"
    DANGER_COLOR = "#EF4444"
    TEXT_COLOR = "#E2E8F0"
    SUBTLE_TEXT = "#94A3B8"

# Diagnostic plot colors
PLOT_COLORS = {
    "scatter": "#3B82F6",
    "line": "#22C55E",
    "histogram": "#8B5CF6",
    "acf": "#F59E0B",
    "pacf": "#EC4899",
    "ci_band": "rgba(59, 130, 246, 0.2)",
    "reference": "#EF4444",
    "grid": "rgba(255, 255, 255, 0.1)"
}


# =============================================================================
# COMMON LAYOUT SETTINGS
# =============================================================================

def get_base_layout() -> dict:
    """Get base layout settings for all plots."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "xaxis": {
            "gridcolor": PLOT_COLORS["grid"],
            "zerolinecolor": PLOT_COLORS["grid"],
            "tickfont": {"color": SUBTLE_TEXT}
        },
        "yaxis": {
            "gridcolor": PLOT_COLORS["grid"],
            "zerolinecolor": PLOT_COLORS["grid"],
            "tickfont": {"color": SUBTLE_TEXT}
        }
    }


# =============================================================================
# ACTUAL VS PREDICTED PLOT
# =============================================================================

def create_actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    show_r2: bool = True
) -> go.Figure:
    """
    Create actual vs predicted scatter plot with perfect prediction line.

    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    show_r2 : bool
        Whether to show R² annotation

    Returns:
    --------
    go.Figure
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Compute R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Create figure
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=y_true,
        mode="markers",
        marker=dict(
            color=PLOT_COLORS["scatter"],
            size=8,
            opacity=0.6,
            line=dict(width=1, color="white")
        ),
        name="Predictions",
        hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>"
    ))

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    padding = (max_val - min_val) * 0.05

    fig.add_trace(go.Scatter(
        x=[min_val - padding, max_val + padding],
        y=[min_val - padding, max_val + padding],
        mode="lines",
        line=dict(color=PLOT_COLORS["reference"], dash="dash", width=2),
        name="Perfect Prediction",
        hoverinfo="skip"
    ))

    # Layout
    layout = get_base_layout()
    layout.update({
        "title": dict(
            text=f"Actual vs Predicted: {model_name}",
            font=dict(size=14, color=TEXT_COLOR)
        ),
        "xaxis_title": "Predicted Values",
        "yaxis_title": "Actual Values",
        "showlegend": False,
        "height": 350,
    })

    # R² annotation
    if show_r2:
        layout["annotations"] = [{
            "x": 0.05,
            "y": 0.95,
            "xref": "paper",
            "yref": "paper",
            "text": f"R² = {r2:.4f}",
            "showarrow": False,
            "font": dict(size=14, color=SUCCESS_COLOR if r2 > 0.7 else WARNING_COLOR),
            "bgcolor": "rgba(0,0,0,0.5)",
            "borderpad": 4
        }]

    fig.update_layout(**layout)
    return fig


# =============================================================================
# RESIDUALS VS PREDICTED PLOT
# =============================================================================

def create_residuals_vs_predicted_plot(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Create residuals vs predicted values plot for heteroscedasticity detection.
    """
    residuals = np.asarray(residuals).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        marker=dict(
            color=PLOT_COLORS["scatter"],
            size=7,
            opacity=0.6
        ),
        name="Residuals",
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
    ))

    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=PLOT_COLORS["reference"],
        line_width=2
    )

    layout = get_base_layout()
    layout.update({
        "title": dict(text="Residuals vs Predicted", font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "Predicted Values",
        "yaxis_title": "Residuals",
        "showlegend": False,
        "height": 350,
    })

    fig.update_layout(**layout)
    return fig


# =============================================================================
# RESIDUAL HISTOGRAM
# =============================================================================

def create_residual_histogram(
    residuals: np.ndarray,
    show_normal_overlay: bool = True
) -> go.Figure:
    """
    Create histogram of residuals with optional normal distribution overlay.
    """
    residuals = np.asarray(residuals).flatten()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker=dict(
            color=PLOT_COLORS["histogram"],
            line=dict(color="white", width=1)
        ),
        opacity=0.7,
        name="Residuals",
        histnorm="probability density"
    ))

    # Normal overlay
    if show_normal_overlay:
        mean = np.mean(residuals)
        std = np.std(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode="lines",
            line=dict(color=PLOT_COLORS["line"], width=2),
            name="Normal Distribution"
        ))

    layout = get_base_layout()
    layout.update({
        "title": dict(text="Residual Distribution", font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "Residual Value",
        "yaxis_title": "Density",
        "showlegend": True,
        "legend": dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0.3)"),
        "height": 350,
    })

    fig.update_layout(**layout)
    return fig


# =============================================================================
# Q-Q PLOT
# =============================================================================

def create_qq_plot(
    residuals: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Create Q-Q plot for normality assessment.
    """
    from scipy import stats

    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    # Compute theoretical and sample quantiles
    sample_quantiles = np.sort(residuals)
    positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = stats.norm.ppf(positions)

    fig = go.Figure()

    # Q-Q points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode="markers",
        marker=dict(
            color=PLOT_COLORS["scatter"],
            size=6,
            opacity=0.7
        ),
        name="Sample Quantiles",
        hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"
    ))

    # Reference line (through Q1 and Q3)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    if theoretical_quantiles[q3_idx] != theoretical_quantiles[q1_idx]:
        slope = (sample_quantiles[q3_idx] - sample_quantiles[q1_idx]) / \
                (theoretical_quantiles[q3_idx] - theoretical_quantiles[q1_idx])
        intercept = sample_quantiles[q1_idx] - slope * theoretical_quantiles[q1_idx]

        x_line = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        y_line = intercept + slope * x_line

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=PLOT_COLORS["reference"], dash="dash", width=2),
            name="Reference Line"
        ))

    layout = get_base_layout()
    layout.update({
        "title": dict(text="Q-Q Plot (Normality Check)", font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "Theoretical Quantiles",
        "yaxis_title": "Sample Quantiles",
        "showlegend": False,
        "height": 350,
    })

    fig.update_layout(**layout)
    return fig


# =============================================================================
# ACF/PACF PLOTS
# =============================================================================

def create_acf_plot(
    acf_values: np.ndarray,
    confidence_bounds: Tuple[float, float],
    title: str = "Autocorrelation Function (ACF)"
) -> go.Figure:
    """
    Create ACF bar plot with confidence bounds.
    """
    acf_values = np.asarray(acf_values).flatten()
    nlags = len(acf_values)
    lags = np.arange(nlags)

    fig = go.Figure()

    # Confidence band
    fig.add_hrect(
        y0=confidence_bounds[0],
        y1=confidence_bounds[1],
        fillcolor=PLOT_COLORS["ci_band"],
        line_width=0,
    )

    # Zero line
    fig.add_hline(y=0, line_color=SUBTLE_TEXT, line_width=1)

    # ACF bars
    colors = [
        DANGER_COLOR if abs(v) > abs(confidence_bounds[1]) and i > 0
        else PLOT_COLORS["acf"]
        for i, v in enumerate(acf_values)
    ]

    fig.add_trace(go.Bar(
        x=lags,
        y=acf_values,
        marker=dict(color=colors, line=dict(width=1, color="white")),
        name="ACF",
        hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>"
    ))

    layout = get_base_layout()
    layout.update({
        "title": dict(text=title, font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "Lag",
        "yaxis_title": "Autocorrelation",
        "showlegend": False,
        "height": 300,
        "yaxis": {"range": [-1, 1], "gridcolor": PLOT_COLORS["grid"]}
    })

    fig.update_layout(**layout)
    return fig


def create_pacf_plot(
    pacf_values: np.ndarray,
    confidence_bounds: Tuple[float, float]
) -> go.Figure:
    """Create PACF bar plot with confidence bounds."""
    return create_acf_plot(
        pacf_values,
        confidence_bounds,
        title="Partial Autocorrelation Function (PACF)"
    )


# =============================================================================
# COMPREHENSIVE DIAGNOSTIC PANEL (6-PLOT GRID)
# =============================================================================

def create_diagnostic_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    horizon: int = 1
) -> go.Figure:
    """
    Create comprehensive 6-plot diagnostic panel.

    Layout:
    [Actual vs Predicted] [Residuals vs Predicted] [Residual Histogram]
    [Q-Q Plot]           [ACF]                     [PACF]
    """
    from app_core.analytics.residual_diagnostics import compute_acf, compute_pacf

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = y_true - y_pred

    # Compute ACF/PACF
    acf_result = compute_acf(residuals, nlags=15)
    try:
        pacf_result = compute_pacf(residuals, nlags=15)
    except Exception:
        pacf_result = None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Actual vs Predicted",
            "Residuals vs Predicted",
            "Residual Distribution",
            "Q-Q Plot (Normality)",
            "ACF",
            "PACF" if pacf_result else "ACF (Copy)"
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # 1. Actual vs Predicted
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    fig.add_trace(go.Scatter(
        x=y_pred, y=y_true, mode="markers",
        marker=dict(color=PLOT_COLORS["scatter"], size=5, opacity=0.6),
        showlegend=False, hovertemplate="Pred: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>"
    ), row=1, col=1)

    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=PLOT_COLORS["reference"], dash="dash"),
        showlegend=False
    ), row=1, col=1)

    # 2. Residuals vs Predicted
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color=PLOT_COLORS["scatter"], size=5, opacity=0.6),
        showlegend=False
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=PLOT_COLORS["reference"], row=1, col=2)

    # 3. Residual Histogram
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=25,
        marker=dict(color=PLOT_COLORS["histogram"], line=dict(color="white", width=0.5)),
        showlegend=False, histnorm="probability density"
    ), row=1, col=3)

    # Normal overlay
    mean, std = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf, mode="lines",
        line=dict(color=PLOT_COLORS["line"], width=2), showlegend=False
    ), row=1, col=3)

    # 4. Q-Q Plot
    from scipy import stats
    n = len(residuals)
    sample_q = np.sort(residuals)
    positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_q = stats.norm.ppf(positions)

    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sample_q, mode="markers",
        marker=dict(color=PLOT_COLORS["scatter"], size=4, opacity=0.7),
        showlegend=False
    ), row=2, col=1)

    # Q-Q reference line
    q1_idx, q3_idx = n // 4, 3 * n // 4
    if theoretical_q[q3_idx] != theoretical_q[q1_idx]:
        slope = (sample_q[q3_idx] - sample_q[q1_idx]) / (theoretical_q[q3_idx] - theoretical_q[q1_idx])
        intercept = sample_q[q1_idx] - slope * theoretical_q[q1_idx]
        x_line = np.array([theoretical_q.min(), theoretical_q.max()])
        fig.add_trace(go.Scatter(
            x=x_line, y=intercept + slope * x_line, mode="lines",
            line=dict(color=PLOT_COLORS["reference"], dash="dash"), showlegend=False
        ), row=2, col=1)

    # 5. ACF
    ci = acf_result.confidence_bounds
    fig.add_hrect(y0=ci[0], y1=ci[1], fillcolor=PLOT_COLORS["ci_band"], line_width=0, row=2, col=2)
    fig.add_hline(y=0, line_color=SUBTLE_TEXT, line_width=1, row=2, col=2)

    acf_colors = [
        DANGER_COLOR if abs(v) > abs(ci[1]) and i > 0 else PLOT_COLORS["acf"]
        for i, v in enumerate(acf_result.acf_values)
    ]
    fig.add_trace(go.Bar(
        x=acf_result.lags, y=acf_result.acf_values,
        marker=dict(color=acf_colors), showlegend=False
    ), row=2, col=2)

    # 6. PACF
    if pacf_result:
        fig.add_hrect(y0=ci[0], y1=ci[1], fillcolor=PLOT_COLORS["ci_band"], line_width=0, row=2, col=3)
        fig.add_hline(y=0, line_color=SUBTLE_TEXT, line_width=1, row=2, col=3)

        pacf_colors = [
            DANGER_COLOR if abs(v) > abs(ci[1]) and i > 0 else PLOT_COLORS["pacf"]
            for i, v in enumerate(pacf_result.pacf_values)
        ]
        fig.add_trace(go.Bar(
            x=pacf_result.lags, y=pacf_result.pacf_values,
            marker=dict(color=pacf_colors), showlegend=False
        ), row=2, col=3)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Residual Diagnostics: {model_name} (Horizon {horizon}) | R² = {r2:.4f}",
            font=dict(size=16, color=TEXT_COLOR)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        height=600,
        showlegend=False,
    )

    # Update axes
    for i in range(1, 7):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        fig.update_xaxes(gridcolor=PLOT_COLORS["grid"], row=row, col=col)
        fig.update_yaxes(gridcolor=PLOT_COLORS["grid"], row=row, col=col)

    return fig


# =============================================================================
# SKILL SCORE GAUGE
# =============================================================================

def create_skill_score_gauge(
    skill_score: float,
    model_name: str = "Model",
    baseline_name: str = "Naive"
) -> go.Figure:
    """
    Create gauge chart for skill score visualization.

    Skill score ranges:
    - > 0.3: Substantial improvement (green)
    - 0.1 to 0.3: Moderate improvement (yellow)
    - 0 to 0.1: Marginal improvement (orange)
    - < 0: Worse than baseline (red)
    """
    # Determine color based on skill score
    if skill_score > 0.3:
        bar_color = SUCCESS_COLOR
        interpretation = "Substantial Improvement"
    elif skill_score > 0.1:
        bar_color = "#F59E0B"
        interpretation = "Moderate Improvement"
    elif skill_score > 0:
        bar_color = "#F97316"
        interpretation = "Marginal Improvement"
    else:
        bar_color = DANGER_COLOR
        interpretation = "Worse than Baseline"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=skill_score * 100,
        number={"suffix": "%", "font": {"size": 36, "color": TEXT_COLOR}},
        title={"text": f"{model_name} vs {baseline_name}<br><span style='font-size:12px;color:{SUBTLE_TEXT}'>{interpretation}</span>"},
        gauge={
            "axis": {"range": [-50, 100], "tickcolor": SUBTLE_TEXT},
            "bar": {"color": bar_color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": SUBTLE_TEXT,
            "steps": [
                {"range": [-50, 0], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [0, 10], "color": "rgba(249, 115, 22, 0.2)"},
                {"range": [10, 30], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [30, 100], "color": "rgba(34, 197, 94, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": skill_score * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR},
        height=250
    )

    return fig


# =============================================================================
# DIEBOLD-MARIANO HEATMAP
# =============================================================================

def create_dm_heatmap(
    p_values: pd.DataFrame,
    significance_markers: pd.DataFrame
) -> go.Figure:
    """
    Create heatmap of Diebold-Mariano test p-values.

    Parameters:
    -----------
    p_values : pd.DataFrame
        Matrix of p-values (model × model)
    significance_markers : pd.DataFrame
        Matrix of significance markers ("***", "**", "*", "ns", "—")
    """
    model_names = list(p_values.index)

    # Create color scale (green = significant, red = not significant)
    # Reverse because low p-value = significant = good
    z_values = p_values.values

    # Create text annotations
    annotations = []
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i == j:
                text = "—"
            else:
                p_val = p_values.loc[model_i, model_j]
                sig = significance_markers.loc[model_i, model_j]
                if pd.isna(p_val):
                    text = "—"
                elif p_val < 0.001:
                    text = f"<0.001{sig}"
                else:
                    text = f"{p_val:.3f}{sig}"

            annotations.append(dict(
                x=j, y=i,
                text=text,
                font=dict(
                    color="white" if (not pd.isna(z_values[i, j]) and z_values[i, j] < 0.05) else SUBTLE_TEXT,
                    size=10
                ),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=model_names,
        y=model_names,
        colorscale=[
            [0, SUCCESS_COLOR],      # p < 0.001
            [0.01, "#22C55E"],       # p < 0.01
            [0.05, "#84CC16"],       # p < 0.05
            [0.1, WARNING_COLOR],    # p < 0.1
            [1.0, DANGER_COLOR]      # p >= 0.1
        ],
        zmin=0,
        zmax=0.2,
        showscale=True,
        colorbar=dict(
            title="p-value",
            tickvals=[0, 0.01, 0.05, 0.1, 0.2],
            ticktext=["0", "0.01", "0.05", "0.1", "≥0.2"]
        ),
        hovertemplate="Row: %{y}<br>Column: %{x}<br>p-value: %{z:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Diebold-Mariano Test Matrix (p-values)",
            font=dict(size=16, color=TEXT_COLOR)
        ),
        annotations=annotations,
        xaxis=dict(title="", tickangle=45),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        height=max(400, len(model_names) * 50 + 100)
    )

    return fig


# =============================================================================
# TRAINING LOSS CURVES
# =============================================================================

def create_training_loss_plot(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    model_name: str = "Model",
    early_stopping_epoch: Optional[int] = None
) -> go.Figure:
    """
    Create training/validation loss curve plot.
    """
    epochs = list(range(1, len(train_loss) + 1))

    fig = go.Figure()

    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode="lines",
        name="Training Loss",
        line=dict(color=PRIMARY_COLOR, width=2)
    ))

    # Validation loss
    if val_loss:
        fig.add_trace(go.Scatter(
            x=epochs[:len(val_loss)],
            y=val_loss,
            mode="lines",
            name="Validation Loss",
            line=dict(color=SECONDARY_COLOR, width=2)
        ))

    # Early stopping marker
    if early_stopping_epoch:
        fig.add_vline(
            x=early_stopping_epoch,
            line_dash="dash",
            line_color=WARNING_COLOR,
            annotation_text=f"Early Stop: {early_stopping_epoch}",
            annotation_position="top"
        )

    # Calculate and annotate gap at end
    if val_loss and len(val_loss) > 0:
        final_train = train_loss[-1]
        final_val = val_loss[-1]
        gap = final_val - final_train

        gap_interpretation = "Good" if gap < 0.2 else "Moderate" if gap < 0.5 else "High"
        gap_color = SUCCESS_COLOR if gap < 0.2 else WARNING_COLOR if gap < 0.5 else DANGER_COLOR

        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=f"Gap: {gap:.3f} ({gap_interpretation})",
            showarrow=False,
            font=dict(size=12, color=gap_color),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4
        )

    layout = get_base_layout()
    layout.update({
        "title": dict(text=f"Training Progress: {model_name}", font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "Epoch",
        "yaxis_title": "Loss",
        "legend": dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0.3)"),
        "height": 350,
    })

    fig.update_layout(**layout)
    return fig


# =============================================================================
# BASELINE COMPARISON CHART
# =============================================================================

def create_baseline_comparison_chart(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    metric: str = "rmse"
) -> go.Figure:
    """
    Create bar chart comparing model against all baselines.

    Parameters:
    -----------
    model_metrics : Dict[str, float]
        Model metrics (e.g., {"rmse": 4.5, "mae": 3.2})
    baseline_metrics : Dict[str, Dict[str, float]]
        Baseline metrics (e.g., {"naive": {"rmse": 7.2}, "seasonal": {"rmse": 5.8}})
    metric : str
        Metric to compare
    """
    # Prepare data
    names = ["Model"] + list(baseline_metrics.keys())
    values = [model_metrics.get(metric, 0)]
    values += [baseline_metrics[b].get(metric, 0) for b in baseline_metrics]

    # Calculate improvement percentages
    baseline_value = values[-1] if len(values) > 1 else values[0]  # Use last baseline (usually naive)
    improvements = [((baseline_value - v) / baseline_value * 100) if baseline_value > 0 else 0 for v in values]

    # Colors (model gets special color)
    colors = [SUCCESS_COLOR] + [SUBTLE_TEXT] * len(baseline_metrics)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{v:.2f}<br><span style='font-size:10px'>({imp:+.1f}%)</span>" for v, imp in zip(values, improvements)],
        textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>"
    ))

    layout = get_base_layout()
    layout.update({
        "title": dict(text=f"{metric.upper()} Comparison: Model vs Baselines", font=dict(size=14, color=TEXT_COLOR)),
        "xaxis_title": "",
        "yaxis_title": metric.upper(),
        "showlegend": False,
        "height": 350,
    })

    fig.update_layout(**layout)
    return fig
