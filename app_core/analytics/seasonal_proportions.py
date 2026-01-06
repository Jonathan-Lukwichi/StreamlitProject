# =============================================================================
# seasonal_proportions.py
# Seasonal Proportions for Category Distribution
# Combines day-of-week and monthly patterns to distribute total forecasts
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# STL decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Import theme colors
try:
    from app_core.ui.theme import (
        PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR,
        WARNING_COLOR, DANGER_COLOR, BODY_TEXT
    )
except ImportError:
    # Fallback colors if theme is unavailable
    PRIMARY_COLOR = "#3b82f6"
    SECONDARY_COLOR = "#22d3ee"
    SUCCESS_COLOR = "#22c55e"
    WARNING_COLOR = "#facc15"
    DANGER_COLOR = "#f97373"
    BODY_TEXT = "#d1d5db"


# Clinical categories used in the application
CLINICAL_CATEGORIES = [
    "RESPIRATORY", "CARDIAC", "TRAUMA", "GASTROINTESTINAL",
    "INFECTIOUS", "NEUROLOGICAL", "OTHER"
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SeasonalProportionConfig:
    """Configuration for seasonal proportion calculation."""
    use_dow_seasonality: bool = True
    use_monthly_seasonality: bool = True
    normalization_method: str = "multiplicative"
    min_samples_per_period: int = 4
    stl_period: int = 7
    stl_model: str = "additive"


@dataclass
class SeasonalProportionResult:
    """Results from seasonal proportion analysis."""
    dow_proportions: pd.DataFrame           # 7 days x 7 categories
    monthly_proportions: pd.DataFrame       # 12 months x 7 categories
    combined_proportions: Dict[str, np.ndarray]  # Combined proportions per category
    category_forecasts: Optional[pd.DataFrame]   # Forecast breakdown by category
    stl_results: Dict[str, Any]             # STL decomposition results
    config: SeasonalProportionConfig


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

def calculate_dow_proportions(
    df: pd.DataFrame,
    date_col: str = "Date",
    category_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate day-of-week proportions for each category.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime and category columns
    date_col : str
        Name of the date column
    category_cols : List[str], optional
        List of category columns. If None, uses CLINICAL_CATEGORIES

    Returns:
    --------
    pd.DataFrame
        7 rows (Mon-Sun) x N category columns
        Each column sums to 1.0 (normalized proportions)
    """
    df = df.copy()

    # Ensure datetime
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    # Auto-detect categories
    if category_cols is None:
        category_cols = [c for c in CLINICAL_CATEGORIES if c in df.columns]

    if not category_cols:
        raise ValueError("No category columns found in dataframe")

    # Add day of week
    df['dow'] = pd.to_datetime(df[date_col]).dt.dayofweek  # 0=Monday, 6=Sunday

    # Calculate average arrivals per day of week for each category
    dow_counts = df.groupby('dow')[category_cols].sum()

    # Normalize: each category's 7 values sum to 1.0
    dow_proportions = dow_counts.copy()
    for cat in category_cols:
        total = dow_counts[cat].sum()
        if total > 0:
            dow_proportions[cat] = dow_counts[cat] / total
        else:
            # Handle zero case: uniform distribution
            dow_proportions[cat] = 1.0 / 7

    # Rename index for clarity
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_proportions.index = day_names

    return dow_proportions


def calculate_monthly_proportions(
    df: pd.DataFrame,
    date_col: str = "Date",
    category_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate monthly proportions for each category.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime and category columns
    date_col : str
        Name of the date column
    category_cols : List[str], optional
        List of category columns. If None, uses CLINICAL_CATEGORIES

    Returns:
    --------
    pd.DataFrame
        12 rows (Jan-Dec) x N category columns
        Each column sums to 1.0 (normalized proportions)
    """
    df = df.copy()

    # Ensure datetime
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    # Auto-detect categories
    if category_cols is None:
        category_cols = [c for c in CLINICAL_CATEGORIES if c in df.columns]

    if not category_cols:
        raise ValueError("No category columns found in dataframe")

    # Add month
    df['month'] = pd.to_datetime(df[date_col]).dt.month

    # Calculate average arrivals per month for each category
    monthly_counts = df.groupby('month')[category_cols].sum()

    # Normalize: each category's 12 values sum to 1.0
    monthly_proportions = monthly_counts.copy()
    for cat in category_cols:
        total = monthly_counts[cat].sum()
        if total > 0:
            monthly_proportions[cat] = monthly_counts[cat] / total
        else:
            # Handle zero case: uniform distribution
            monthly_proportions[cat] = 1.0 / 12

    # Rename index for clarity
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_proportions.index = month_names

    return monthly_proportions


def combine_proportions_multiplicatively(
    dow_proportions: pd.DataFrame,
    monthly_proportions: pd.DataFrame,
    target_date: pd.Timestamp,
    category_cols: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Combine DOW and monthly proportions multiplicatively for a specific date.

    Algorithm:
    1. Get DOW proportion: dow_prop[cat][day]
    2. Get monthly proportion: monthly_prop[cat][month]
    3. Multiply: raw_prop[cat] = dow_prop * monthly_prop
    4. Normalize across all categories so sum = 1.0

    Parameters:
    -----------
    dow_proportions : pd.DataFrame
        7 rows (days) x N categories
    monthly_proportions : pd.DataFrame
        12 rows (months) x N categories
    target_date : pd.Timestamp
        Date to get proportions for
    category_cols : List[str], optional
        Categories to include. If None, uses all available

    Returns:
    --------
    Dict[str, float]
        Category name -> final normalized proportion (sum=1.0)
    """
    if category_cols is None:
        category_cols = list(dow_proportions.columns)

    # Get day of week and month
    dow = target_date.dayofweek  # 0=Monday
    month_num = target_date.month  # 1=January

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    day_name = day_names[dow]
    month_name = month_names[month_num - 1]

    # Multiply proportions
    raw_proportions = {}
    for cat in category_cols:
        dow_prop = dow_proportions.loc[day_name, cat]
        monthly_prop = monthly_proportions.loc[month_name, cat]
        raw_proportions[cat] = dow_prop * monthly_prop

    # Normalize so sum = 1.0
    total_raw = sum(raw_proportions.values())
    if total_raw > 0:
        normalized = {cat: prop / total_raw for cat, prop in raw_proportions.items()}
    else:
        # Fallback: uniform distribution
        normalized = {cat: 1.0 / len(category_cols) for cat in category_cols}

    return normalized


def distribute_forecast_to_categories(
    total_forecast: pd.Series,
    dow_proportions: pd.DataFrame,
    monthly_proportions: pd.DataFrame,
    category_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Distribute total patient forecast into clinical categories.

    Parameters:
    -----------
    total_forecast : pd.Series
        Series with DatetimeIndex and total patient counts
    dow_proportions : pd.DataFrame
        7 rows x N categories
    monthly_proportions : pd.DataFrame
        12 rows x N categories
    category_cols : List[str], optional
        Categories to include

    Returns:
    --------
    pd.DataFrame
        DatetimeIndex x N category columns with distributed forecasts
    """
    if category_cols is None:
        category_cols = list(dow_proportions.columns)

    # Initialize result dataframe
    result = pd.DataFrame(index=total_forecast.index)

    # For each forecast date, distribute total into categories
    for date in total_forecast.index:
        total_value = total_forecast.loc[date]

        # Get combined proportions for this date
        proportions = combine_proportions_multiplicatively(
            dow_proportions, monthly_proportions, date, category_cols
        )

        # Distribute total forecast
        for cat in category_cols:
            result.loc[date, cat] = total_value * proportions[cat]

    return result


# =============================================================================
# STL DECOMPOSITION
# =============================================================================

def run_stl_decomposition(
    df: pd.DataFrame,
    date_col: str = "Date",
    category_cols: Optional[List[str]] = None,
    period: int = 7,
    model: str = "additive"
) -> Dict[str, Any]:
    """
    Run STL decomposition on category time series.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with datetime and category columns
    date_col : str
        Name of date column
    category_cols : List[str], optional
        Categories to decompose
    period : int
        Seasonal period (7 for weekly, 30 for monthly, 365 for yearly)
    model : str
        'additive' or 'multiplicative'

    Returns:
    --------
    Dict[str, DecomposeResult]
        Category name -> statsmodels DecomposeResult object
    """
    df = df.copy()

    # Ensure datetime index
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)

    # Auto-detect categories
    if category_cols is None:
        category_cols = [c for c in CLINICAL_CATEGORIES if c in df.columns]

    results = {}

    for cat in category_cols:
        if cat not in df.columns:
            continue

        try:
            # Prepare time series
            ts = df[cat].asfreq('D').interpolate(method='time', limit_direction='both')

            # Check minimum data requirement
            needed = 2 * period + 1
            if len(ts.dropna()) >= needed:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    decomposition = seasonal_decompose(
                        ts.dropna(),
                        model=model,
                        period=period,
                        extrapolate_trend='freq'
                    )
                    results[cat] = decomposition
            else:
                results[cat] = None  # Not enough data
        except Exception as e:
            warnings.warn(f"STL decomposition failed for {cat}: {e}")
            results[cat] = None

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def calculate_seasonal_proportions(
    df: pd.DataFrame,
    config: SeasonalProportionConfig,
    date_col: str = "Date",
    category_cols: Optional[List[str]] = None
) -> SeasonalProportionResult:
    """
    Main entry point for calculating seasonal proportions.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime and category columns
    config : SeasonalProportionConfig
        Configuration settings
    date_col : str
        Name of date column
    category_cols : List[str], optional
        Categories to analyze

    Returns:
    --------
    SeasonalProportionResult
        Complete results including proportions, STL decomposition, etc.
    """
    # Auto-detect categories
    if category_cols is None:
        category_cols = [c for c in CLINICAL_CATEGORIES if c in df.columns]

    # Calculate DOW proportions
    dow_props = None
    if config.use_dow_seasonality:
        dow_props = calculate_dow_proportions(df, date_col, category_cols)
    else:
        # Create uniform DOW proportions
        dow_props = pd.DataFrame(
            1.0 / 7,
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=category_cols
        )

    # Calculate monthly proportions
    monthly_props = None
    if config.use_monthly_seasonality:
        # Check if we have at least 12 months of data
        df_copy = df.copy()
        if date_col in df_copy.columns:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            n_months = df_copy[date_col].dt.to_period('M').nunique()
        else:
            n_months = 0

        if n_months >= 12:
            monthly_props = calculate_monthly_proportions(df, date_col, category_cols)
        else:
            warnings.warn(f"Only {n_months} months of data available. Falling back to DOW-only proportions.")
            config.use_monthly_seasonality = False
            monthly_props = pd.DataFrame(
                1.0 / 12,
                index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                columns=category_cols
            )
    else:
        # Create uniform monthly proportions
        monthly_props = pd.DataFrame(
            1.0 / 12,
            index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            columns=category_cols
        )

    # Combine proportions (for a reference date, we'll use the mean)
    combined_props = {}
    for cat in category_cols:
        # Average across all DOW and month combinations
        all_proportions = []
        for dow_idx in range(7):
            for month_idx in range(12):
                dow_val = dow_props.iloc[dow_idx][cat]
                month_val = monthly_props.iloc[month_idx][cat]
                all_proportions.append(dow_val * month_val)
        combined_props[cat] = np.array(all_proportions)

    # Run STL decomposition
    stl_results = {}
    if len(df) >= 2 * config.stl_period + 1:
        stl_results = run_stl_decomposition(
            df, date_col, category_cols,
            period=config.stl_period,
            model=config.stl_model
        )

    return SeasonalProportionResult(
        dow_proportions=dow_props,
        monthly_proportions=monthly_props,
        combined_proportions=combined_props,
        category_forecasts=None,  # Will be populated later when applying to forecasts
        stl_results=stl_results,
        config=config
    )


def apply_seasonal_proportions_to_forecast(
    forecast_results: Dict[str, Any],
    seasonal_props: SeasonalProportionResult,
    forecast_dates: pd.DatetimeIndex
) -> Dict[str, Any]:
    """
    Post-processor: Apply seasonal proportions to total forecast.

    Parameters:
    -----------
    forecast_results : Dict
        Results from ARIMA/SARIMAX pipeline
    seasonal_props : SeasonalProportionResult
        Calculated seasonal proportions
    forecast_dates : pd.DatetimeIndex
        Dates for the forecast period

    Returns:
    --------
    Dict
        Updated forecast_results with category_forecasts added
    """
    # Extract total forecast from results
    # Assuming forecast_results contains a 'forecast' key with total predictions
    if 'forecast' not in forecast_results:
        return forecast_results

    total_forecast = forecast_results['forecast']

    # Convert to Series if needed
    if isinstance(total_forecast, (list, np.ndarray)):
        total_forecast = pd.Series(total_forecast, index=forecast_dates)

    # Distribute forecast to categories
    category_forecasts = distribute_forecast_to_categories(
        total_forecast,
        seasonal_props.dow_proportions,
        seasonal_props.monthly_proportions
    )

    # Update results
    forecast_results['seasonal_proportions'] = seasonal_props
    forecast_results['category_forecasts'] = category_forecasts

    return forecast_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_stl_decomposition_plot(
    stl_results: Dict[str, Any],
    category: str = None
) -> go.Figure:
    """
    Create 4-panel STL decomposition plot.

    Parameters:
    -----------
    stl_results : Dict
        Dictionary of category -> DecomposeResult
    category : str, optional
        Specific category to plot. If None, plots first available

    Returns:
    --------
    go.Figure
        Plotly figure with 4 subplots
    """
    # Select category
    if category is None:
        # Use first available category with results
        category = next((cat for cat, res in stl_results.items() if res is not None), None)

    if category is None or category not in stl_results or stl_results[category] is None:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No STL decomposition available. Insufficient data or decomposition failed.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=BODY_TEXT)
        )
        return fig

    result = stl_results[category]

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )

    # Observed
    fig.add_trace(
        go.Scatter(
            x=result.observed.index,
            y=result.observed.values,
            line=dict(color=PRIMARY_COLOR, width=1.5),
            name="Observed"
        ),
        row=1, col=1
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=result.trend.index,
            y=result.trend.values,
            line=dict(color=DANGER_COLOR, width=2),
            name="Trend"
        ),
        row=2, col=1
    )

    # Seasonal
    fig.add_trace(
        go.Scatter(
            x=result.seasonal.index,
            y=result.seasonal.values,
            line=dict(color=SUCCESS_COLOR, width=1.5),
            name="Seasonal"
        ),
        row=3, col=1
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=result.resid.index,
            y=result.resid.values,
            mode="markers",
            marker=dict(color=WARNING_COLOR, size=3, opacity=0.7),
            name="Residual"
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"STL Decomposition: {category}",
        title_font=dict(size=16, color=BODY_TEXT),
        margin=dict(t=80, b=20, l=60, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    return fig


def create_seasonal_heatmap(
    dow_proportions: Optional[pd.DataFrame] = None,
    monthly_proportions: Optional[pd.DataFrame] = None,
    title: str = "Seasonal Proportions"
) -> go.Figure:
    """
    Create heatmap of seasonal proportions.

    Parameters:
    -----------
    dow_proportions : pd.DataFrame, optional
        7 rows (days) x N categories
    monthly_proportions : pd.DataFrame, optional
        12 rows (months) x N categories
    title : str
        Plot title

    Returns:
    --------
    go.Figure
        Plotly heatmap figure
    """
    # Determine which to plot
    if dow_proportions is not None and monthly_proportions is None:
        data = dow_proportions
        y_label = "Day of Week"
    elif monthly_proportions is not None and dow_proportions is None:
        data = monthly_proportions
        y_label = "Month"
    elif dow_proportions is not None and monthly_proportions is not None:
        # Combine both (DOW on top, monthly on bottom)
        # For simplicity, we'll plot DOW by default
        data = dow_proportions
        y_label = "Day of Week"
    else:
        # Empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No proportion data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Convert to percentage
    data_pct = data * 100

    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=data_pct.values,
        x=data_pct.columns,
        y=data_pct.index,
        colorscale="RdYlGn",
        text=data_pct.values.round(1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Proportion (%)")
    ))

    fig.update_layout(
        title=title,
        title_font=dict(size=14, color=BODY_TEXT),
        xaxis_title="Category",
        yaxis_title=y_label,
        height=400,
        margin=dict(t=60, b=40, l=100, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_category_forecast_chart(
    category_forecasts: pd.DataFrame
) -> go.Figure:
    """
    Create stacked area chart of category forecasts.

    Parameters:
    -----------
    category_forecasts : pd.DataFrame
        DatetimeIndex x N category columns

    Returns:
    --------
    go.Figure
        Plotly stacked area chart
    """
    if category_forecasts is None or category_forecasts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No category forecasts available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create stacked area chart
    fig = px.area(
        category_forecasts,
        x=category_forecasts.index,
        y=category_forecasts.columns,
        title="7-Day Category Forecast Distribution"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Patient Arrivals",
        legend_title="Category",
        height=500,
        margin=dict(t=60, b=40, l=60, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    return fig


# =============================================================================
# EDGE CASE HANDLERS
# =============================================================================

def handle_zero_categories(proportions: pd.DataFrame, min_value: float = 0.001) -> pd.DataFrame:
    """
    Floor zero proportions to minimum value to avoid division by zero.

    Parameters:
    -----------
    proportions : pd.DataFrame
        Proportion dataframe
    min_value : float
        Minimum value to use for zero proportions

    Returns:
    --------
    pd.DataFrame
        Adjusted proportions
    """
    proportions = proportions.copy()
    proportions[proportions == 0] = min_value

    # Re-normalize each column
    for col in proportions.columns:
        total = proportions[col].sum()
        if total > 0:
            proportions[col] = proportions[col] / total

    return proportions


# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def render_seasonal_forecast_breakdown(
    predictions: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    seasonal_proportions: SeasonalProportionResult,
    title: str = "Category Forecast Distribution"
) -> None:
    """
    Render seasonal category forecast breakdown in Streamlit UI.

    Parameters:
    -----------
    predictions : np.ndarray
        Total forecast predictions
    forecast_dates : pd.DatetimeIndex
        Dates for the forecast period
    seasonal_proportions : SeasonalProportionResult
        Calculated seasonal proportions
    title : str
        Title for the section
    """
    import streamlit as st

    if predictions is None or len(predictions) == 0:
        st.info("No predictions available for category breakdown.")
        return

    # Ensure predictions is a Series with proper index
    if isinstance(predictions, (list, np.ndarray)):
        predictions = pd.Series(predictions, index=forecast_dates[:len(predictions)])

    # Distribute forecast to categories
    try:
        category_forecasts = distribute_forecast_to_categories(
            predictions,
            seasonal_proportions.dow_proportions,
            seasonal_proportions.monthly_proportions
        )
    except Exception as e:
        st.warning(f"Could not distribute forecasts to categories: {e}")
        return

    if category_forecasts is None or category_forecasts.empty:
        st.info("No category breakdown available.")
        return

    # Render UI
    st.markdown(f"#### {title}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Forecast", f"{predictions.sum():.0f}")
    with col2:
        st.metric("Forecast Days", len(predictions))
    with col3:
        st.metric("Categories", len(category_forecasts.columns))

    # Create and display chart
    fig = create_category_forecast_chart(category_forecasts)
    st.plotly_chart(fig, use_container_width=True)

    # Category breakdown table
    with st.expander("View Category Breakdown Details", expanded=False):
        summary_df = category_forecasts.sum().reset_index()
        summary_df.columns = ["Category", "Total Forecast"]
        summary_df["Percentage"] = (summary_df["Total Forecast"] / summary_df["Total Forecast"].sum() * 100).round(1)
        summary_df["Percentage"] = summary_df["Percentage"].astype(str) + "%"
        summary_df["Total Forecast"] = summary_df["Total Forecast"].round(0).astype(int)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
