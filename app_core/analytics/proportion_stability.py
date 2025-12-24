# =============================================================================
# proportion_stability.py
# Analyze Clinical Category Proportion Stability Over Time
# Determines if historical proportions are stable enough for forecasting
# =============================================================================

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

# Clinical categories used in the application
CLINICAL_CATEGORIES = [
    "RESPIRATORY", "CARDIAC", "TRAUMA", "GASTROINTESTINAL",
    "INFECTIOUS", "NEUROLOGICAL", "OTHER"
]


def analyze_proportion_stability(
    df: pd.DataFrame,
    date_col: str = "datetime",
    total_col: str = "Total_Arrivals",
    category_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Analyze if clinical category proportions are stable over time.

    Parameters:
    -----------
    df : pd.DataFrame
        The processed dataframe with datetime and category columns
    date_col : str
        Name of the datetime column
    total_col : str
        Name of the total arrivals column
    category_cols : List[str], optional
        List of category column names. If None, auto-detects from CLINICAL_CATEGORIES

    Returns:
    --------
    Dict with analysis results including:
        - monthly_proportions: DataFrame of proportions by month
        - stability_metrics: CV (coefficient of variation) per category
        - recommendation: str with suggested approach
        - is_stable: bool indicating if proportions are stable enough
        - seasonal_patterns: detected seasonality per category
    """
    df = df.copy()

    # Ensure datetime index
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame must have datetime index or '{date_col}' column")

    # Auto-detect category columns if not provided
    if category_cols is None:
        category_cols = [c for c in CLINICAL_CATEGORIES if c in df.columns]

    if not category_cols:
        return {
            "error": "No clinical category columns found in dataframe",
            "available_columns": list(df.columns),
            "expected_columns": CLINICAL_CATEGORIES
        }

    # Calculate or use total
    if total_col in df.columns:
        total = df[total_col]
    else:
        total = df[category_cols].sum(axis=1)

    # Calculate daily proportions
    daily_proportions = pd.DataFrame()
    for cat in category_cols:
        if cat in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                daily_proportions[cat] = df[cat] / total.replace(0, np.nan)

    # Monthly aggregation
    monthly_counts = df[category_cols].resample('M').sum()
    monthly_total = monthly_counts.sum(axis=1)
    monthly_proportions = monthly_counts.div(monthly_total, axis=0)

    # Quarterly aggregation
    quarterly_counts = df[category_cols].resample('Q').sum()
    quarterly_total = quarterly_counts.sum(axis=1)
    quarterly_proportions = quarterly_counts.div(quarterly_total, axis=0)

    # Calculate stability metrics
    stability_metrics = {}
    for cat in category_cols:
        if cat in monthly_proportions.columns:
            props = monthly_proportions[cat].dropna()
            if len(props) > 1:
                mean_prop = props.mean()
                std_prop = props.std()
                cv = (std_prop / mean_prop * 100) if mean_prop > 0 else 0

                # Range (max - min)
                prop_range = props.max() - props.min()

                # Seasonal variation (compare Q1 vs Q3, Q2 vs Q4)
                seasonal_cv = _calculate_seasonal_variation(quarterly_proportions[cat])

                stability_metrics[cat] = {
                    "mean": round(mean_prop * 100, 2),
                    "std": round(std_prop * 100, 2),
                    "cv_percent": round(cv, 2),
                    "min": round(props.min() * 100, 2),
                    "max": round(props.max() * 100, 2),
                    "range": round(prop_range * 100, 2),
                    "seasonal_cv": round(seasonal_cv, 2),
                    "n_months": len(props)
                }

    # Determine overall stability
    cv_values = [m["cv_percent"] for m in stability_metrics.values()]
    avg_cv = np.mean(cv_values) if cv_values else 0
    max_cv = max(cv_values) if cv_values else 0

    # Thresholds for stability
    # CV < 10%: Very stable
    # CV 10-20%: Moderately stable
    # CV > 20%: Unstable (seasonal or trending)

    if max_cv < 10:
        stability_level = "VERY_STABLE"
        is_stable = True
        recommendation = (
            "✅ PROPORTIONS ARE STABLE\n"
            "Historical proportions are very consistent (CV < 10%).\n"
            "The current probability-based approach is EFFICIENT for your data.\n"
            "No changes recommended."
        )
    elif max_cv < 20:
        stability_level = "MODERATELY_STABLE"
        is_stable = True
        recommendation = (
            "⚠️ PROPORTIONS ARE MODERATELY STABLE\n"
            f"Average CV: {avg_cv:.1f}%, Max CV: {max_cv:.1f}%\n"
            "The current approach is acceptable but could be improved.\n"
            "Consider: Dynamic monthly proportions instead of overall average."
        )
    else:
        stability_level = "UNSTABLE"
        is_stable = False
        unstable_cats = [cat for cat, m in stability_metrics.items() if m["cv_percent"] > 20]
        recommendation = (
            "❌ PROPORTIONS ARE UNSTABLE\n"
            f"High variation detected (CV > 20%) in: {', '.join(unstable_cats)}\n"
            "The current probability-based approach may underperform.\n"
            "Recommended alternatives:\n"
            "1. Use SEASONAL proportions (monthly averages)\n"
            "2. Train separate models per category\n"
            "3. Use hierarchical reconciliation"
        )

    # Detect seasonal patterns
    seasonal_patterns = _detect_seasonal_patterns(monthly_proportions, category_cols)

    return {
        "monthly_proportions": monthly_proportions,
        "quarterly_proportions": quarterly_proportions,
        "daily_proportions": daily_proportions,
        "stability_metrics": stability_metrics,
        "summary": {
            "avg_cv": round(avg_cv, 2),
            "max_cv": round(max_cv, 2),
            "stability_level": stability_level,
            "is_stable": is_stable,
            "n_categories": len(category_cols),
            "date_range": f"{df.index.min()} to {df.index.max()}",
            "n_months": len(monthly_proportions)
        },
        "seasonal_patterns": seasonal_patterns,
        "recommendation": recommendation
    }


def _calculate_seasonal_variation(quarterly_series: pd.Series) -> float:
    """Calculate coefficient of variation across quarters."""
    if len(quarterly_series) < 4:
        return 0.0

    # Group by quarter number (1-4)
    quarterly_series = quarterly_series.dropna()
    if len(quarterly_series) == 0:
        return 0.0

    quarter_means = quarterly_series.groupby(quarterly_series.index.quarter).mean()
    if len(quarter_means) < 2:
        return 0.0

    mean_val = quarter_means.mean()
    std_val = quarter_means.std()

    return (std_val / mean_val * 100) if mean_val > 0 else 0.0


def _detect_seasonal_patterns(
    monthly_proportions: pd.DataFrame,
    category_cols: List[str]
) -> Dict[str, Dict]:
    """Detect seasonal patterns in category proportions."""
    patterns = {}

    for cat in category_cols:
        if cat not in monthly_proportions.columns:
            continue

        series = monthly_proportions[cat].dropna()
        if len(series) < 12:
            patterns[cat] = {"has_seasonality": False, "reason": "Insufficient data (<12 months)"}
            continue

        # Group by month
        monthly_avg = series.groupby(series.index.month).mean()

        # Find peak and trough months
        peak_month = monthly_avg.idxmax()
        trough_month = monthly_avg.idxmin()

        # Calculate seasonal amplitude
        amplitude = monthly_avg.max() - monthly_avg.min()
        relative_amplitude = amplitude / monthly_avg.mean() * 100 if monthly_avg.mean() > 0 else 0

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        patterns[cat] = {
            "has_seasonality": relative_amplitude > 15,
            "peak_month": month_names[peak_month - 1],
            "trough_month": month_names[trough_month - 1],
            "amplitude_percent": round(relative_amplitude, 2),
            "monthly_averages": {month_names[m-1]: round(v*100, 2) for m, v in monthly_avg.items()}
        }

    return patterns


def generate_stability_report(analysis_results: Dict) -> str:
    """Generate a text report from analysis results."""
    if "error" in analysis_results:
        return f"Error: {analysis_results['error']}"

    summary = analysis_results["summary"]
    metrics = analysis_results["stability_metrics"]
    patterns = analysis_results["seasonal_patterns"]

    report = []
    report.append("=" * 60)
    report.append("CLINICAL CATEGORY PROPORTION STABILITY REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Date Range: {summary['date_range']}")
    report.append(f"Months Analyzed: {summary['n_months']}")
    report.append(f"Categories: {summary['n_categories']}")
    report.append("")
    report.append("-" * 60)
    report.append("STABILITY METRICS BY CATEGORY")
    report.append("-" * 60)
    report.append(f"{'Category':<20} {'Mean%':>8} {'CV%':>8} {'Range%':>8} {'Status':>12}")
    report.append("-" * 60)

    for cat, m in metrics.items():
        cv = m["cv_percent"]
        status = "✅ Stable" if cv < 10 else ("⚠️ Moderate" if cv < 20 else "❌ Unstable")
        report.append(f"{cat:<20} {m['mean']:>8.1f} {cv:>8.1f} {m['range']:>8.1f} {status:>12}")

    report.append("")
    report.append("-" * 60)
    report.append("SEASONAL PATTERNS")
    report.append("-" * 60)

    for cat, p in patterns.items():
        if p.get("has_seasonality"):
            report.append(f"{cat}: Seasonal (Peak: {p['peak_month']}, Low: {p['trough_month']}, "
                         f"Amplitude: {p['amplitude_percent']:.1f}%)")
        else:
            report.append(f"{cat}: No significant seasonality")

    report.append("")
    report.append("-" * 60)
    report.append("RECOMMENDATION")
    report.append("-" * 60)
    report.append(analysis_results["recommendation"])

    return "\n".join(report)


def create_stability_visualizations(analysis_results: Dict):
    """
    Create Plotly visualizations for proportion stability.
    Returns dict of figures that can be displayed with st.plotly_chart()
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    if "error" in analysis_results:
        return {}

    monthly_props = analysis_results["monthly_proportions"]
    metrics = analysis_results["stability_metrics"]
    patterns = analysis_results["seasonal_patterns"]

    figures = {}

    # 1. Monthly Proportions Over Time (Stacked Area)
    fig_timeline = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, cat in enumerate(monthly_props.columns):
        fig_timeline.add_trace(go.Scatter(
            x=monthly_props.index,
            y=monthly_props[cat] * 100,
            mode='lines',
            name=cat,
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)]
        ))

    fig_timeline.update_layout(
        title="Category Proportions Over Time (Monthly)",
        xaxis_title="Month",
        yaxis_title="Proportion (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    figures["timeline"] = fig_timeline

    # 2. Stability Metrics Bar Chart
    cats = list(metrics.keys())
    cvs = [metrics[c]["cv_percent"] for c in cats]

    colors_bar = ['#22c55e' if cv < 10 else '#f59e0b' if cv < 20 else '#ef4444' for cv in cvs]

    fig_stability = go.Figure(go.Bar(
        x=cats,
        y=cvs,
        marker_color=colors_bar,
        text=[f"{cv:.1f}%" for cv in cvs],
        textposition='outside'
    ))

    fig_stability.add_hline(y=10, line_dash="dash", line_color="green",
                            annotation_text="Stable threshold (10%)")
    fig_stability.add_hline(y=20, line_dash="dash", line_color="red",
                            annotation_text="Unstable threshold (20%)")

    fig_stability.update_layout(
        title="Coefficient of Variation (CV) by Category",
        xaxis_title="Category",
        yaxis_title="CV (%)",
        yaxis=dict(range=[0, max(cvs) * 1.3 if cvs else 30])
    )
    figures["stability"] = fig_stability

    # 3. Seasonal Heatmap
    seasonal_data = []
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for cat in patterns:
        if "monthly_averages" in patterns[cat]:
            for month in month_names:
                if month in patterns[cat]["monthly_averages"]:
                    seasonal_data.append({
                        "Category": cat,
                        "Month": month,
                        "Proportion": patterns[cat]["monthly_averages"][month]
                    })

    if seasonal_data:
        seasonal_df = pd.DataFrame(seasonal_data)
        seasonal_pivot = seasonal_df.pivot(index="Category", columns="Month", values="Proportion")
        seasonal_pivot = seasonal_pivot[month_names]  # Reorder columns

        fig_heatmap = go.Figure(go.Heatmap(
            z=seasonal_pivot.values,
            x=month_names,
            y=seasonal_pivot.index,
            colorscale="RdYlGn",
            text=seasonal_pivot.values.round(1),
            texttemplate="%{text}%",
            textfont={"size": 10}
        ))

        fig_heatmap.update_layout(
            title="Seasonal Pattern Heatmap (Monthly Avg %)",
            xaxis_title="Month",
            yaxis_title="Category"
        )
        figures["heatmap"] = fig_heatmap

    # 4. Box plot of monthly proportions
    box_data = []
    for cat in monthly_props.columns:
        for val in monthly_props[cat].dropna():
            box_data.append({"Category": cat, "Proportion": val * 100})

    if box_data:
        box_df = pd.DataFrame(box_data)
        fig_box = px.box(box_df, x="Category", y="Proportion",
                         title="Distribution of Monthly Proportions",
                         color="Category")
        fig_box.update_layout(showlegend=False)
        figures["distribution"] = fig_box

    return figures
