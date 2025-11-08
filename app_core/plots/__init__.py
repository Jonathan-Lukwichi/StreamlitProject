"""
Plotting utilities for multi-horizon forecasting results.

Public API:
- build_multihorizon_results_dashboard
- style_metrics_table
- plot_metrics_dashboard
- plot_multihorizon_forecasts
- plot_fanchart
- plot_pred_vs_actual
- plot_error_distributions
- plot_temporal_split
- plot_residual_diagnostics
"""



from .multihorizon_results import (
    build_multihorizon_results_dashboard,
    style_metrics_table,
    plot_metrics_dashboard,
    plot_multihorizon_forecasts,
    plot_fanchart,
    plot_pred_vs_actual,
    plot_error_distributions,
    plot_temporal_split,
    plot_residual_diagnostics,
)

__all__ = [
    "build_multihorizon_results_dashboard",
    "style_metrics_table",
    "plot_metrics_dashboard",
    "plot_multihorizon_forecasts",
    "plot_fanchart",
    "plot_pred_vs_actual",
    "plot_error_distributions",
    "plot_temporal_split",
    "plot_residual_diagnostics",
]

__version__ = "0.1.0"
