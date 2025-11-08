# app_core/plots/multihorizon_results.py
# Comprehensive Multi-Horizon Forecasting Results Dashboard
# --------------------------------------------------------
# Requirements:
#   metrics_df: DataFrame with columns:
#       ["Horizon","MAE","RMSE","MAPE_%","Accuracy_%","R2","Bias",
#        "CI_Coverage_%","Direction_Accuracy_%","Within_±2_%","Within_±5_%"]
#   F, L, U: numpy arrays (n_dates x n_horizons) forecasts/CI bounds
#   test_eval: DataFrame with actuals per horizon ("Target_1" .. "Target_H")
#   train: DataFrame/Series with the training target (indexed by datetime)
#   res: fitted model results-like object with .resid and .fittedvalues (or close)
#
# All functions return matplotlib.figure.Figure objects (or pandas Styler).
# Figures default to display DPI=100; you can save at 300 DPI when exporting.
# --------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Optional, Tuple
from matplotlib.ticker import MaxNLocator, FuncFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox


# ------------------------------ Styling helpers ------------------------------

def _use_style():
    sns.set_theme(style="darkgrid", rc={
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 14,
        "font.size": 10,
    })


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.2f}%"


def _safe_col(df: pd.DataFrame, col: str):
    return col in df.columns


def _hue_colors(n: int) -> List:
    # pleasant, distinct color cycle for horizons
    base = sns.color_palette("tab10", n)
    if n <= len(base):
        return base[:n]
    extra = sns.color_palette("husl", n)
    return extra


def _add_value_labels(ax, fmt="{:.2f}", fontsize=9, offset=0.01):
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(fmt.format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=fontsize,
                    xytext=(0, ax.get_ylim()[1]*offset), textcoords='offset points')


# ------------------------------ 1) Styled metrics table ------------------------------

def style_metrics_table(metrics_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Format metrics with publication-friendly precision using pandas Styler.
    - MAE, RMSE -> 3 decimals
    - MAPE_%, Accuracy_% -> 2 decimals (with % suffix)
    Others: 2 decimals by default
    """
    df = metrics_df.copy()
    # Ensure canonical column names exist; if not, create blanks
    for c in ["Horizon","MAE","RMSE","MAPE_%","Accuracy_%","R2","Bias","CI_Coverage_%",
              "Direction_Accuracy_%","Within_±2_%","Within_±5_%"]:
        if c not in df.columns:
            df[c] = np.nan

    styler = (df.style
              .format({
                  "Horizon": "{:.0f}",
                  "MAE": "{:.3f}",
                  "RMSE": "{:.3f}",
                  "MAPE_%": "{:.2f}",
                  "Accuracy_%": "{:.2f}",
                  "R2": "{:.3f}",
                  "Bias": "{:.3f}",
                  "CI_Coverage_%": "{:.2f}",
                  "Direction_Accuracy_%": "{:.2f}",
                  "Within_±2_%": "{:.2f}",
                  "Within_±5_%": "{:.2f}",
              }, na_rep="")
              .set_caption("Multi-Horizon Evaluation Metrics")
              .hide(axis="index")
              .set_table_styles([
                  {"selector": "caption",
                   "props": [("font-size", "14px"), ("text-align", "left"),
                             ("font-weight", "600"), ("margin-bottom", "8px")]},
                  {"selector": "th",
                   "props": [("text-align", "center"), ("font-weight", "600")]},
                  {"selector": "td",
                   "props": [("text-align", "center")]}
              ])
    )
    return styler


# ------------------------------ 2) Metrics dashboard (2x2) ------------------------------

def plot_metrics_dashboard(metrics_df: pd.DataFrame,
                          figsize=(16, 10), dpi=100) -> plt.Figure:
    """
    2x2 grid:
      [0,0] Accuracy (%) bar
      [0,1] MAE bar
      [1,0] RMSE bar
      [1,1] MAPE (%) line with markers
    """
    _use_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle("Results Dashboard — Accuracy & Error Metrics", fontsize=14, weight="bold")

    df = metrics_df.copy()
    if "Horizon" not in df.columns:
        df["Horizon"] = np.arange(1, len(df)+1)

    horizons = df["Horizon"].astype(int).tolist()
    x = np.arange(len(horizons))

    # Accuracy
    ax = axes[0, 0]
    if _safe_col(df, "Accuracy_%"):
        bars = ax.bar(x, df["Accuracy_%"], color=sns.color_palette("crest", len(x)))
        ax.set_title("Accuracy (%) by Horizon")
        ax.set_xticks(x, labels=horizons)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Accuracy (%)")
        _add_value_labels(ax, fmt="{:.1f}")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        ax.text(0.5, 0.5, "Accuracy_% not provided", ha="center", va="center")
        ax.set_axis_off()

    # MAE
    ax = axes[0, 1]
    if _safe_col(df, "MAE"):
        bars = ax.bar(x, df["MAE"], color=sns.color_palette("flare", len(x)))
        ax.set_title("MAE by Horizon")
        ax.set_xticks(x, labels=horizons)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("MAE")
        _add_value_labels(ax, fmt="{:.3f}")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        ax.text(0.5, 0.5, "MAE not provided", ha="center", va="center")
        ax.set_axis_off()

    # RMSE
    ax = axes[1, 0]
    if _safe_col(df, "RMSE"):
        bars = ax.bar(x, df["RMSE"], color=sns.color_palette("mako", len(x)))
        ax.set_title("RMSE by Horizon")
        ax.set_xticks(x, labels=horizons)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("RMSE")
        _add_value_labels(ax, fmt="{:.3f}")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        ax.text(0.5, 0.5, "RMSE not provided", ha="center", va="center")
        ax.set_axis_off()

    # MAPE
    ax = axes[1, 1]
    if _safe_col(df, "MAPE_%"):
        ax.plot(horizons, df["MAPE_%"], marker="o", linewidth=2, color=sns.color_palette("tab10", 1)[0])
        for h, v in zip(horizons, df["MAPE_%"]):
            if pd.notna(v):
                ax.text(h, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("MAPE (%) by Horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("MAPE (%)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.text(0.5, 0.5, "MAPE_% not provided", ha="center", va="center")
        ax.set_axis_off()

    return fig


# ------------------------------ 3) Multi-Horizon forecast wall (4x2) ------------------------------

def plot_multihorizon_forecasts(test_eval: pd.DataFrame,
                                F: np.ndarray,
                                L: Optional[np.ndarray],
                                U: Optional[np.ndarray],
                                horizons: Optional[List[int]] = None,
                                date_index: Optional[pd.DatetimeIndex] = None,
                                figsize=(16, 10), dpi=100) -> plt.Figure:
    """
    4x2 grid: 7 horizon subplots + final panel for legend/notes.
    """
    _use_style()

    # Determine horizons count (max 7 panels + 1 legend)
    if horizons is None:
        # attempt to infer from F
        H = F.shape[1]
        horizons = list(range(1, H + 1))
    H = min(7, len(horizons))
    colors = _hue_colors(H)

    # Layout: 4 rows x 2 cols -> 8 axes
    fig, axes = plt.subplots(4, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle("Forecast Comparison by Horizon", fontsize=14, weight="bold")
    axes = axes.ravel()

    if date_index is None:
        date_index = getattr(test_eval, "index", pd.RangeIndex(F.shape[0]))

    # Plot each horizon
    for i in range(H):
        h = horizons[i]
        ax = axes[i]
        actual = test_eval.get(f"Target_{h}")
        if actual is None:
            ax.text(0.5, 0.5, f"Target_{h} not found", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.plot(date_index, actual.values, color="black", linewidth=1.6, label="Actual")
        ax.plot(date_index, F[:, i], color=colors[i], linewidth=1.6, label=f"Forecast h={h}")

        if L is not None and U is not None:
            ax.fill_between(date_index, L[:, i], U[:, i],
                            color=colors[i], alpha=0.15, label="95% CI")

        # Metrics in title if available
        mape = None
        acc = None
        if "MAPE_%" in test_eval.columns:
            # not typical; ignore
            pass
        # try from wide metrics df if user passes separately (not here), so just compute quickly:
        try:
            denom = np.where(actual.values == 0, np.nan, np.abs(actual.values))
            mape = float(np.nanmean(np.abs(F[:, i] - actual.values) / denom) * 100.0)
            acc = 100.0 - mape if np.isfinite(mape) else np.nan
        except Exception:
            pass

        title = f"h={h}"
        if mape is not None and np.isfinite(mape):
            title += f" | MAPE {mape:.2f}%"
        if acc is not None and np.isfinite(acc):
            title += f" | Acc {acc:.1f}%"
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("ED Arrivals")

    # Legend/notes panel
    ax_leg = axes[-1]
    ax_leg.axis("off")
    lines = [plt.Line2D([0], [0], color="black", lw=2)]
    labels = ["Actual"]
    for i in range(H):
        lines.append(plt.Line2D([0],[0], color=colors[i], lw=2))
        labels.append(f"h={horizons[i]}")
    ax_leg.legend(lines, labels, loc="center", ncol=min(4, H+1), frameon=True, title="Legend")
    ax_leg.text(0.5, 0.1,
                "Notes: Each panel shows Actual, Forecast, and 95% CI.\n"
                "Titles include quick MAPE/Accuracy computed per horizon.",
                ha="center", va="center", fontsize=9)

    return fig


# ------------------------------ 4) Fan chart ------------------------------

def plot_fanchart(test_eval: pd.DataFrame,
                  F: np.ndarray,
                  horizons: Optional[List[int]] = None,
                  date_index: Optional[pd.DatetimeIndex] = None,
                  figsize=(16, 10), dpi=100) -> plt.Figure:
    """
    Fan chart: actual Target_1 + each horizon forecast as a colored line.
    """
    _use_style()
    if horizons is None:
        horizons = list(range(1, F.shape[1] + 1))
    H = len(horizons)
    colors = _hue_colors(H)

    if date_index is None:
        date_index = getattr(test_eval, "index", pd.RangeIndex(F.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_title("Fan Chart — Multi-Horizon Forecasts")
    # thick black actual for horizon 1 ground truth
    if f"Target_1" in test_eval.columns:
        ax.plot(date_index, test_eval["Target_1"].values, color="black", linewidth=2.5, label="Actual (Target_1)")

    for i, h in enumerate(horizons):
        ax.plot(date_index, F[:, i], color=colors[i], linewidth=1.6, label=f"h={h}")

    ax.set_xlabel("Date")
    ax.set_ylabel("ED Arrivals")
    ax.legend(ncol=min(5, H+1), frameon=True)
    return fig


# ------------------------------ 5) Predicted vs Actual grid (2x4) ------------------------------

def plot_pred_vs_actual(test_eval: pd.DataFrame,
                        F: np.ndarray,
                        horizons: Optional[List[int]] = None,
                        figsize=(16, 10), dpi=100) -> plt.Figure:
    """
    2x4 grid (7 horizons + final explanation panel).
    """
    _use_style()
    if horizons is None:
        horizons = list(range(1, F.shape[1] + 1))
    H = min(7, len(horizons))
    colors = _hue_colors(H)

    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle("Prediction Accuracy — Predicted vs Actual", fontsize=14, weight="bold")
    axes = axes.ravel()

    for i in range(H):
        h = horizons[i]
        ax = axes[i]
        y = test_eval.get(f"Target_{h}")
        if y is None:
            ax.text(0.5, 0.5, f"Target_{h} not found", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.scatter(y, F[:, i], s=16, alpha=0.35, color=colors[i])
        # perfect line
        lo = np.nanmin([y.min(), F[:, i].min()])
        hi = np.nanmax([y.max(), F[:, i].max()])
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1.2, label="y = x")
        ax.set_title(f"h={h}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    ax_exp = axes[-1]
    ax_exp.axis("off")
    ax_exp.text(0.5, 0.5,
                "Each panel compares Predicted vs Actual for a horizon.\n"
                "Red dashed line marks perfect predictions (y=x).",
                ha="center", va="center", fontsize=10)
    return fig


# ------------------------------ 6) Error distributions ------------------------------

def plot_error_distributions(test_eval: pd.DataFrame,
                             F: np.ndarray,
                             horizons: Optional[List[int]] = None,
                             kde_horizons: Optional[List[int]] = None,
                             figsize=(16, 10), dpi=100) -> Tuple[plt.Figure, plt.Figure]:
    """
    Returns:
      fig_box: boxplot of absolute errors by horizon
      fig_kde: KDE / histogram comparison for selected horizons
    """
    _use_style()
    if horizons is None:
        horizons = list(range(1, F.shape[1] + 1))
    H = len(horizons)
    colors = _hue_colors(H)

    # Build error matrix
    abs_errors = {}
    for i, h in enumerate(horizons):
        y = test_eval.get(f"Target_{h}")
        if y is None:
            continue
        abs_errors[h] = np.abs(F[:, i] - y.values)

    # Boxplot of absolute errors
    fig_box, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_title("Error Distribution — Absolute Errors by Horizon (with means)")
    if len(abs_errors):
        data = [abs_errors[h] for h in horizons if h in abs_errors]
        b = ax.boxplot(data, labels=[f"h={h}" for h in horizons if h in abs_errors],
                       patch_artist=True, showmeans=True, meanline=True)
        for patch, c in zip(b["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.25)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("|Error|")
    else:
        ax.text(0.5, 0.5, "No errors available", ha="center", va="center")
        ax.set_axis_off()

    # KDE / histogram
    if kde_horizons is None:
        kde_horizons = [1, 3, 7]  # sensible defaults; will be skipped if missing
    fig_kde, ax2 = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax2.set_title("Error Distributions — Selected Horizons")
    plotted = False
    for h in kde_horizons:
        if h in abs_errors:
            sns.kdeplot(abs_errors[h], ax=ax2, label=f"h={h}", linewidth=2)
            plotted = True
    if plotted:
        ax2.set_xlabel("|Error|")
        ax2.legend(frameon=True)
    else:
        ax2.text(0.5, 0.5, "Selected horizons not available", ha="center", va="center")
        ax2.set_axis_off()

    return fig_box, fig_kde


# ------------------------------ 7) Temporal split plot ------------------------------

def plot_temporal_split(train: pd.Series | pd.DataFrame,
                        test_eval: pd.DataFrame,
                        F_h1: np.ndarray,
                        figsize=(16, 6), dpi=100) -> plt.Figure:
    """
    Shows Train (actual), Test (actual, thick black), and h=1 forecast aligned to test.
    """
    _use_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_title("Temporal Performance — Train/Test Split with h=1 Forecast")

    # Resolve train series
    if isinstance(train, pd.DataFrame):
        # try a single numeric column
        col = train.select_dtypes(include=[np.number]).columns
        y_train = train[col[0]] if len(col) else train.iloc[:, 0]
    else:
        y_train = train

    ax.plot(y_train.index, y_train.values, color=sns.color_palette("tab10", 1)[0],
            label="Train (Actual)", linewidth=1.3)

    if "Target_1" in test_eval.columns:
        ax.plot(test_eval.index, test_eval["Target_1"].values, color="black",
                linewidth=2.0, label="Test (Actual)")

    if F_h1 is not None:
        ax.plot(test_eval.index, F_h1, color="#00CC96", linestyle="--",
                linewidth=2.0, label="Forecast (h=1)")

    ax.set_xlabel("Date")
    ax.set_ylabel("ED Arrivals")
    ax.legend(frameon=True)
    return fig


# ------------------------------ 8) Residual diagnostics (2×3) ------------------------------

def plot_residual_diagnostics(res,
                              figsize=(16, 10), dpi=100,
                              lags: int = 30) -> plt.Figure:
    """
    2×3 grid:
      Residuals time series, ACF, PACF,
      Q-Q plot, Histogram/KDE, Residuals vs Fitted
    Includes Ljung–Box p-value text.
    """
    _use_style()
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_acf = fig.add_subplot(gs[0, 1])
    ax_pacf = fig.add_subplot(gs[0, 2])
    ax_qq  = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 2])

    resid = np.asarray(getattr(res, "resid", np.array([])))
    fitted = np.asarray(getattr(res, "fittedvalues", np.array([])))

    # Residual time series
    ax_ts.plot(resid, color="#6c757d", linewidth=1.2)
    ax_ts.set_title("Residuals over Time")
    ax_ts.set_xlabel("Index")
    ax_ts.set_ylabel("Residual")

    # ACF / PACF
    try:
        plot_acf(resid, ax=ax_acf, lags=min(lags, len(resid)-2))
        ax_acf.set_title("ACF of Residuals")
    except Exception:
        ax_acf.text(0.5, 0.5, "ACF unavailable", ha="center", va="center")
        ax_acf.set_axis_off()

    try:
        plot_pacf(resid, ax=ax_pacf, lags=min(lags, len(resid)//2 - 1), method="ywm")
        ax_pacf.set_title("PACF of Residuals")
    except Exception:
        ax_pacf.text(0.5, 0.5, "PACF unavailable", ha="center", va="center")
        ax_pacf.set_axis_off()

    # Q-Q
    try:
        qqplot(resid, line="s", ax=ax_qq, markerfacecolor="#4e79a7", markeredgecolor="none", alpha=0.6)
        ax_qq.set_title("Q–Q Plot of Residuals")
    except Exception:
        ax_qq.text(0.5, 0.5, "Q-Q unavailable", ha="center", va="center")
        ax_qq.set_axis_off()

    # Histogram + KDE
    if resid.size:
        sns.histplot(resid, bins=30, kde=True, ax=ax_hist, color="#f28e2b", alpha=0.8)
        ax_hist.set_title("Residuals Histogram / KDE")
        ax_hist.set_xlabel("Residual")
    else:
        ax_hist.text(0.5, 0.5, "No residuals", ha="center", va="center")
        ax_hist.set_axis_off()

    # Residuals vs Fitted
    if resid.size and fitted.size and len(resid) == len(fitted):
        ax_scatter.scatter(fitted, resid, s=16, alpha=0.35, color="#59a14f")
        ax_scatter.axhline(0, color="red", linestyle="--", linewidth=1)
        ax_scatter.set_title("Residuals vs Fitted")
        ax_scatter.set_xlabel("Fitted")
        ax_scatter.set_ylabel("Residual")
    else:
        ax_scatter.text(0.5, 0.5, "No fitted values", ha="center", va="center")
        ax_scatter.set_axis_off()

    # Ljung–Box test annotation
    try:
        lb = acorr_ljungbox(resid, lags=[min(10, len(resid)-2)], return_df=True)
        pval = float(lb["lb_pvalue"].iloc[0])
        msg = f"Ljung–Box p={pval:.3f} (lag={min(10, len(resid)-2)})"
    except Exception:
        msg = "Ljung–Box test unavailable"

    fig.suptitle(f"Residual Diagnostics — {msg}", fontsize=14, weight="bold")
    return fig


# ------------------------------ 9) Orchestrator ------------------------------

def build_multihorizon_results_dashboard(
        metrics_df: pd.DataFrame,
        F: np.ndarray, L: Optional[np.ndarray], U: Optional[np.ndarray],
        test_eval: pd.DataFrame,
        train: pd.Series | pd.DataFrame,
        res,
        horizons: Optional[List[int]] = None
    ) -> Dict[str, object]:
    """
    Convenience wrapper returning all artifacts:
      {
        "metrics_styler": Styler,
        "fig_metrics_dashboard": Figure,
        "fig_forecast_wall": Figure,
        "fig_fanchart": Figure,
        "fig_pred_vs_actual": Figure,
        "fig_err_box": Figure,
        "fig_err_kde": Figure,
        "fig_temporal": Figure,
        "fig_residuals": Figure
      }
    """
    # 1) Styled table
    metrics_styler = style_metrics_table(metrics_df)

    # 2) 2x2 metrics board
    fig_metrics = plot_metrics_dashboard(metrics_df)

    # infer horizons / index
    if horizons is None:
        horizons = list(range(1, F.shape[1] + 1))
    date_index = getattr(test_eval, "index", pd.RangeIndex(F.shape[0]))

    # 3) forecast wall
    fig_wall = plot_multihorizon_forecasts(test_eval, F, L, U, horizons=horizons, date_index=date_index)

    # 4) fan chart
    fig_fan = plot_fanchart(test_eval, F, horizons=horizons, date_index=date_index)

    # 5) pred vs actual
    fig_pva = plot_pred_vs_actual(test_eval, F, horizons=horizons)

    # 6) error distributions
    fig_err_box, fig_err_kde = plot_error_distributions(test_eval, F, horizons=horizons)

    # 7) temporal split (uses h=1 forecasts)
    F_h1 = F[:, 0] if F.ndim == 2 and F.shape[1] >= 1 else None
    fig_temp = plot_temporal_split(train, test_eval, F_h1)

    # 8) residual diagnostics
    fig_resid = plot_residual_diagnostics(res)

    return {
        "metrics_styler": metrics_styler,
        "fig_metrics_dashboard": fig_metrics,
        "fig_forecast_wall": fig_wall,
        "fig_fanchart": fig_fan,
        "fig_pred_vs_actual": fig_pva,
        "fig_err_box": fig_err_box,
        "fig_err_kde": fig_err_kde,
        "fig_temporal": fig_temp,
        "fig_residuals": fig_resid,
    }
