"""
KPI Service - Calculate HealthForecast AI KPIs.

Aggregates data from forecasts, models, staff planner, and historical data
to calculate operational KPIs that are RELEVANT to this project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForecastKPIs:
    """Container for HealthForecast AI KPIs - all project-relevant metrics."""

    # Forecast KPIs (from active forecast)
    today_forecast: float = 0.0          # Today's predicted ED arrivals
    week_total_forecast: float = 0.0     # 7-day total forecast
    peak_day_forecast: float = 0.0       # Highest single day forecast
    peak_day_name: str = "N/A"           # Name of peak day

    # Historical KPIs (from prepared_data)
    historical_avg_ed: float = 0.0       # Average daily ED arrivals
    historical_max_ed: float = 0.0       # Max daily ED arrivals
    historical_min_ed: float = 0.0       # Min daily ED arrivals
    total_records: int = 0               # Total days of data

    # Model Performance KPIs (from model results)
    best_model_name: str = "N/A"
    best_model_mape: float = 0.0
    best_model_rmse: float = 0.0
    models_trained: int = 0

    # Category Distribution (from seasonal proportions or forecast)
    category_distribution: List[Dict] = field(default_factory=list)

    # Staff Planning KPIs (from MILP solver)
    staff_coverage_pct: float = 0.0
    total_staff_needed: int = 0
    overtime_hours: float = 0.0
    daily_staff_cost: float = 0.0

    # Supply Planning KPIs (from inventory optimizer)
    supply_service_level: float = 0.0         # Target service level %
    supply_total_cost: float = 0.0            # Total inventory cost
    supply_weekly_savings: float = 0.0        # Weekly savings from optimization
    supply_items_count: int = 0               # Number of items optimized
    supply_reorder_alerts: int = 0            # Items below reorder point
    supply_item_breakdown: List[Dict] = field(default_factory=list)  # Item-level details

    # Time series data for charts
    forecast_trend: List[Dict] = field(default_factory=list)      # Historical + forecast
    daily_ed_pattern: List[Dict] = field(default_factory=list)    # Day-of-week pattern
    model_comparison: List[Dict] = field(default_factory=list)    # Model accuracy comparison
    staff_schedule: List[Dict] = field(default_factory=list)      # Staff by day

    # Data availability flags
    has_forecast: bool = False
    has_historical: bool = False
    has_models: bool = False
    has_staff_plan: bool = False
    has_supply_plan: bool = False


# =============================================================================
# CLINICAL CATEGORIES
# =============================================================================

CLINICAL_CATEGORIES = [
    "RESPIRATORY", "CARDIAC", "TRAUMA",
    "GASTROINTESTINAL", "INFECTIOUS", "NEUROLOGICAL", "OTHER"
]

CATEGORY_COLORS = {
    "RESPIRATORY": "#3b82f6",      # Blue
    "CARDIAC": "#ef4444",          # Red
    "TRAUMA": "#f97316",           # Orange
    "GASTROINTESTINAL": "#22c55e", # Green
    "INFECTIOUS": "#a855f7",       # Purple
    "NEUROLOGICAL": "#06b6d4",     # Cyan
    "OTHER": "#6b7280"             # Gray
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_forecast_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from active forecast data."""
    forecast_data = st.session_state.get('active_forecast', {})
    forecast_df = forecast_data.get('forecast_df')

    if forecast_df is not None and isinstance(forecast_df, pd.DataFrame) and len(forecast_df) > 0:
        kpis.has_forecast = True

        # Find ED/Total column
        ed_col = None
        for col in ['ED', 'Total', 'Forecast', 'predicted']:
            if col in forecast_df.columns:
                ed_col = col
                break

        if ed_col:
            kpis.today_forecast = round(forecast_df[ed_col].iloc[0], 1)
            kpis.week_total_forecast = round(forecast_df[ed_col].sum(), 0)
            kpis.peak_day_forecast = round(forecast_df[ed_col].max(), 1)

            # Find peak day name
            peak_idx = forecast_df[ed_col].idxmax()
            if 'date' in forecast_df.columns:
                peak_date = pd.to_datetime(forecast_df.loc[peak_idx, 'date'])
                kpis.peak_day_name = peak_date.strftime('%A')
            elif 'ds' in forecast_df.columns:
                peak_date = pd.to_datetime(forecast_df.loc[peak_idx, 'ds'])
                kpis.peak_day_name = peak_date.strftime('%A')
            else:
                kpis.peak_day_name = f"Day {peak_idx + 1}"

        # Build forecast trend for chart
        date_col = 'date' if 'date' in forecast_df.columns else 'ds' if 'ds' in forecast_df.columns else None
        if date_col and ed_col:
            for _, row in forecast_df.iterrows():
                kpis.forecast_trend.append({
                    "date": pd.to_datetime(row[date_col]).strftime('%b %d'),
                    "forecast": round(row[ed_col], 1),
                    "type": "forecast"
                })


def _extract_historical_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from historical prepared data."""
    historical_df = st.session_state.get('prepared_data')

    if historical_df is not None and isinstance(historical_df, pd.DataFrame) and len(historical_df) > 0:
        kpis.has_historical = True
        kpis.total_records = len(historical_df)

        if 'ED' in historical_df.columns:
            kpis.historical_avg_ed = round(historical_df['ED'].mean(), 1)
            kpis.historical_max_ed = round(historical_df['ED'].max(), 0)
            kpis.historical_min_ed = round(historical_df['ED'].min(), 0)

            # Day-of-week pattern
            if 'date' in historical_df.columns:
                df = historical_df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df['dow'] = df['date'].dt.day_name()
                dow_avg = df.groupby('dow')['ED'].mean()

                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for day in day_order:
                    if day in dow_avg.index:
                        kpis.daily_ed_pattern.append({
                            "day": day[:3],
                            "avg_ed": round(dow_avg[day], 1)
                        })

            # Add recent historical data to trend (last 7 days)
            if 'date' in historical_df.columns:
                df = historical_df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                recent = df.tail(7)

                historical_trend = []
                for _, row in recent.iterrows():
                    historical_trend.append({
                        "date": row['date'].strftime('%b %d'),
                        "actual": round(row['ED'], 1),
                        "type": "historical"
                    })
                # Prepend historical to forecast trend
                kpis.forecast_trend = historical_trend + kpis.forecast_trend


def _extract_model_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from trained model results."""
    best_mape = float('inf')
    best_model = None
    models = []

    # Check all model result types
    model_sources = [
        ('arima_results', 'ARIMA'),
        ('sarimax_results', 'SARIMAX'),
        ('ml_results', None),  # ML results have model name inside
        ('arima_mh_results', 'ARIMA'),
        ('sarimax_mh_results', 'SARIMAX'),
        ('ml_mh_results', None),
    ]

    for key, default_name in model_sources:
        results = st.session_state.get(key)
        if results and isinstance(results, dict):
            # Extract metrics
            metrics = results.get('metrics', results.get('test_metrics', {}))
            if isinstance(metrics, dict):
                mape = metrics.get('MAPE', metrics.get('mape'))
                rmse = metrics.get('RMSE', metrics.get('rmse'))
                mae = metrics.get('MAE', metrics.get('mae'))

                model_name = results.get('model_name', results.get('name', default_name))

                if mape is not None and not np.isnan(mape) and not np.isinf(mape):
                    models.append({
                        "name": model_name or key,
                        "mape": round(float(mape), 2),
                        "rmse": round(float(rmse), 2) if rmse else 0,
                        "mae": round(float(mae), 2) if mae else 0
                    })

                    if mape < best_mape:
                        best_mape = mape
                        best_model = {
                            "name": model_name or key,
                            "mape": mape,
                            "rmse": rmse
                        }

    if best_model:
        kpis.has_models = True
        kpis.best_model_name = best_model['name']
        kpis.best_model_mape = round(best_model['mape'], 2)
        kpis.best_model_rmse = round(best_model['rmse'], 2) if best_model['rmse'] else 0
        kpis.models_trained = len(models)
        kpis.model_comparison = sorted(models, key=lambda x: x['mape'])


def _extract_category_distribution(kpis: ForecastKPIs) -> None:
    """Extract category distribution from seasonal proportions or forecast."""
    # Try to get from forecast hub
    forecast_hub = st.session_state.get('forecast_hub_demand')

    if forecast_hub is not None and isinstance(forecast_hub, pd.DataFrame):
        cat_cols = [c for c in forecast_hub.columns if c in CLINICAL_CATEGORIES]
        if cat_cols:
            totals = forecast_hub[cat_cols].sum()
            grand_total = totals.sum()

            for cat in cat_cols:
                pct = (totals[cat] / grand_total * 100) if grand_total > 0 else 0
                kpis.category_distribution.append({
                    "name": cat.title(),
                    "value": round(pct, 1),
                    "count": int(totals[cat]),
                    "color": CATEGORY_COLORS.get(cat, "#6b7280")
                })
            kpis.category_distribution = sorted(
                kpis.category_distribution,
                key=lambda x: x['value'],
                reverse=True
            )
            return

    # Try from seasonal proportions results
    seasonal = st.session_state.get('seasonal_proportions_results', {})
    if seasonal and 'category_totals' in seasonal:
        cat_totals = seasonal['category_totals']
        grand_total = sum(cat_totals.values())

        for cat, count in cat_totals.items():
            pct = (count / grand_total * 100) if grand_total > 0 else 0
            kpis.category_distribution.append({
                "name": cat.title(),
                "value": round(pct, 1),
                "count": int(count),
                "color": CATEGORY_COLORS.get(cat.upper(), "#6b7280")
            })
        kpis.category_distribution = sorted(
            kpis.category_distribution,
            key=lambda x: x['value'],
            reverse=True
        )
        return

    # Default placeholder
    kpis.category_distribution = [
        {"name": "Respiratory", "value": 28, "count": 0, "color": CATEGORY_COLORS["RESPIRATORY"]},
        {"name": "Cardiac", "value": 22, "count": 0, "color": CATEGORY_COLORS["CARDIAC"]},
        {"name": "Trauma", "value": 18, "count": 0, "color": CATEGORY_COLORS["TRAUMA"]},
        {"name": "Gastrointestinal", "value": 14, "count": 0, "color": CATEGORY_COLORS["GASTROINTESTINAL"]},
        {"name": "Infectious", "value": 10, "count": 0, "color": CATEGORY_COLORS["INFECTIOUS"]},
        {"name": "Neurological", "value": 5, "count": 0, "color": CATEGORY_COLORS["NEUROLOGICAL"]},
        {"name": "Other", "value": 3, "count": 0, "color": CATEGORY_COLORS["OTHER"]},
    ]


def _extract_staff_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from staff optimization results."""
    staff_results = st.session_state.get('staff_optimization_results')

    if staff_results and isinstance(staff_results, dict):
        kpis.has_staff_plan = True

        # Extract metrics
        kpis.staff_coverage_pct = round(staff_results.get('coverage_pct',
            staff_results.get('utilization', 0)), 1)
        kpis.total_staff_needed = int(staff_results.get('total_staff',
            staff_results.get('staff_needed', 0)))
        kpis.overtime_hours = round(staff_results.get('total_overtime',
            staff_results.get('overtime_hours', 0)), 1)
        kpis.daily_staff_cost = round(staff_results.get('daily_cost',
            staff_results.get('total_cost', 0) / 7), 2)

        # Build staff schedule
        schedule = staff_results.get('schedule', staff_results.get('daily_schedule', {}))
        if schedule:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for i, day in enumerate(days):
                day_data = schedule.get(f'day_{i}', schedule.get(day, {}))
                if isinstance(day_data, dict):
                    staff_count = sum(day_data.values())
                else:
                    staff_count = day_data if isinstance(day_data, (int, float)) else 0

                kpis.staff_schedule.append({
                    "day": day,
                    "staff": int(staff_count),
                    "demand": int(staff_count * 1.1)  # Approximate demand
                })
    else:
        # Default schedule
        kpis.staff_schedule = [
            {"day": "Mon", "staff": 45, "demand": 48},
            {"day": "Tue", "staff": 50, "demand": 52},
            {"day": "Wed", "staff": 48, "demand": 50},
            {"day": "Thu", "staff": 44, "demand": 46},
            {"day": "Fri", "staff": 51, "demand": 54},
            {"day": "Sat", "staff": 36, "demand": 38},
            {"day": "Sun", "staff": 33, "demand": 35},
        ]


def _extract_supply_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from supply/inventory optimization results."""
    supply_results = st.session_state.get('inv_optimized_results')

    if supply_results and isinstance(supply_results, dict) and supply_results.get('success'):
        kpis.has_supply_plan = True

        # Core metrics
        kpis.supply_service_level = round(supply_results.get('service_level', 95.0), 1)
        kpis.supply_total_cost = round(supply_results.get('total_cost', 0), 2)
        kpis.supply_weekly_savings = round(supply_results.get('weekly_savings', 0), 2)

        # Item breakdown
        item_results = supply_results.get('item_results', {})
        kpis.supply_items_count = len(item_results)

        # Count reorder alerts (items where current stock might be below reorder point)
        reorder_alerts = 0
        for item_id, item_data in item_results.items():
            if item_data.get('reorder_point', 0) > 0:
                # Assume we need to reorder if safety stock is being used
                if item_data.get('safety_stock', 0) > item_data.get('avg_daily_demand', 0) * 2:
                    reorder_alerts += 1

            kpis.supply_item_breakdown.append({
                "name": item_data.get('name', item_id)[:20],
                "demand": round(item_data.get('forecast_demand', 0), 0),
                "order_qty": int(item_data.get('optimal_order', 0)),
                "safety_stock": round(item_data.get('safety_stock', 0), 0),
                "cost": round(item_data.get('purchase_cost', 0) + item_data.get('holding_cost', 0), 0)
            })

        kpis.supply_reorder_alerts = reorder_alerts


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def calculate_forecast_kpis() -> ForecastKPIs:
    """
    Calculate all KPIs from available session state data.

    Returns:
        ForecastKPIs dataclass with project-relevant metrics
    """
    kpis = ForecastKPIs()

    # Extract from each data source
    _extract_forecast_kpis(kpis)
    _extract_historical_kpis(kpis)
    _extract_model_kpis(kpis)
    _extract_category_distribution(kpis)
    _extract_staff_kpis(kpis)

    return kpis


# Legacy compatibility
def calculate_hospital_kpis(*args, **kwargs) -> ForecastKPIs:
    """Legacy function name - redirects to calculate_forecast_kpis."""
    return calculate_forecast_kpis()


# Alias for dataclass
HospitalKPIs = ForecastKPIs
