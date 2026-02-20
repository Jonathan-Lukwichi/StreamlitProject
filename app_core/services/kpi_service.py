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
    today_forecast: float = 0.0          # First day predicted ED arrivals
    week_total_forecast: float = 0.0     # 7-day total forecast
    peak_day_forecast: float = 0.0       # Highest single day forecast
    peak_day_name: str = "N/A"           # Name of peak day (with date)
    forecast_model_name: str = "N/A"     # Model used for forecast (e.g., "ANN", "LSTM")
    forecast_dates: List[str] = field(default_factory=list)  # Actual forecast dates

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
    # Try multiple session state keys (Patient Forecast stores in these)
    forecast_data = st.session_state.get('active_forecast', {})
    if not forecast_data:
        forecast_data = st.session_state.get('forecast_hub_demand', {})

    # Get forecast values - can be list or DataFrame
    forecast_values = forecast_data.get('forecasts', forecast_data.get('forecast', []))
    forecast_dates = forecast_data.get('dates', [])

    # Extract model name (from Patient Forecast page)
    kpis.forecast_model_name = forecast_data.get('model', forecast_data.get('model_name', 'N/A'))

    # Store forecast dates for Dashboard display
    kpis.forecast_dates = list(forecast_dates) if forecast_dates else []

    # Handle list-based format (from 10_Patient_Forecast.py)
    if forecast_values and isinstance(forecast_values, (list, np.ndarray)) and len(forecast_values) > 0:
        kpis.has_forecast = True
        forecast_arr = np.array(forecast_values)

        kpis.today_forecast = int(round(float(forecast_arr[0])))
        kpis.week_total_forecast = int(round(float(forecast_arr.sum())))
        kpis.peak_day_forecast = int(round(float(forecast_arr.max())))

        # Find peak day name - show actual date (e.g., "TUE Feb 08")
        peak_idx = int(np.argmax(forecast_arr))
        if forecast_dates and len(forecast_dates) > peak_idx:
            try:
                # forecast_dates from Patient Forecast are already formatted as "Mon 02/07"
                kpis.peak_day_name = str(forecast_dates[peak_idx])
            except:
                kpis.peak_day_name = f"Day {peak_idx + 1}"
        else:
            kpis.peak_day_name = f"Day {peak_idx + 1}"

        # Build forecast trend for chart
        for i, val in enumerate(forecast_arr):
            date_str = f"Day {i + 1}"
            if forecast_dates and len(forecast_dates) > i:
                try:
                    # Use the date string directly (already formatted)
                    date_str = str(forecast_dates[i])
                except:
                    pass
            kpis.forecast_trend.append({
                "date": date_str,
                "forecast": int(round(float(val))),
                "type": "forecast"
            })
        return

    # Fallback: Handle DataFrame format (legacy)
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
            kpis.today_forecast = int(round(forecast_df[ed_col].iloc[0]))
            kpis.week_total_forecast = int(round(forecast_df[ed_col].sum()))
            kpis.peak_day_forecast = int(round(forecast_df[ed_col].max()))

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
                    "forecast": int(round(row[ed_col])),
                    "type": "forecast"
                })


def _extract_historical_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from historical prepared data."""
    # Try multiple session state sources for historical data
    historical_df = st.session_state.get('prepared_data')

    # Fallback 1: Check feature_engineering dict (from Feature Studio)
    if historical_df is None:
        fe_data = st.session_state.get('feature_engineering', {})
        if isinstance(fe_data, dict):
            historical_df = fe_data.get('df')

    # Fallback 2: Check fused_data (from Data Fusion)
    if historical_df is None:
        historical_df = st.session_state.get('fused_data')

    # Fallback 3: Check uploaded_data (legacy key)
    if historical_df is None:
        historical_df = st.session_state.get('uploaded_data')

    # Fallback 4: Check patient_data (raw upload from 02_Upload_Data.py)
    if historical_df is None:
        historical_df = st.session_state.get('patient_data')

    if historical_df is not None and isinstance(historical_df, pd.DataFrame) and len(historical_df) > 0:
        kpis.has_historical = True
        kpis.total_records = len(historical_df)

        # Find ED column with multiple candidate names (matches data_processing.py)
        ed_col = None
        for col in ['ED', 'ed', 'Ed', 'Arrivals', 'arrivals', 'patients', 'patient_count', 'ED_Count']:
            if col in historical_df.columns:
                ed_col = col
                break

        if ed_col:
            kpis.historical_avg_ed = round(historical_df[ed_col].mean(), 1)
            kpis.historical_max_ed = round(historical_df[ed_col].max(), 0)
            kpis.historical_min_ed = round(historical_df[ed_col].min(), 0)

            # Day-of-week pattern - find date column
            date_col = None
            for dc in ['date', 'Date', 'datetime', 'ds', 'timestamp']:
                if dc in historical_df.columns:
                    date_col = dc
                    break

            if date_col:
                df = historical_df.copy()
                df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
                df['dow'] = df['_date'].dt.day_name()
                dow_avg = df.groupby('dow')[ed_col].mean()

                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for day in day_order:
                    if day in dow_avg.index:
                        kpis.daily_ed_pattern.append({
                            "day": day[:3],
                            "avg_ed": round(dow_avg[day], 1)
                        })

            # Add recent historical data to trend (last 7 days)
            if date_col:
                df = historical_df.copy()
                df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.sort_values('_date')
                recent = df.tail(7)

                historical_trend = []
                for _, row in recent.iterrows():
                    historical_trend.append({
                        "date": row['_date'].strftime('%b %d'),
                        "actual": round(row[ed_col], 1),
                        "type": "historical"
                    })
                # Prepend historical to forecast trend
                kpis.forecast_trend = historical_trend + kpis.forecast_trend


def _extract_model_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from trained model results."""
    best_mape = float('inf')
    best_model = None
    models = []

    # Check all model result types (from all pages)
    model_sources = [
        # Baseline Models (05_Baseline_Models.py)
        ('arima_results', 'ARIMA'),
        ('sarimax_results', 'SARIMAX'),
        ('arima_mh_results', 'ARIMA'),
        # ML Models (08_Train_Models.py - Tab 2)
        ('ml_mh_results_xgboost', 'XGBoost'),
        ('ml_mh_results_lstm', 'LSTM'),
        ('ml_mh_results_ann', 'ANN'),
        ('ml_mh_results_XGBoost', 'XGBoost'),
        ('ml_mh_results_LSTM', 'LSTM'),
        ('ml_mh_results_ANN', 'ANN'),
        ('ml_mh_results', None),
        ('ml_results', None),
        # Hybrid Models (08_Train_Models.py - Tab 4)
        ('lstm_xgb_results', 'LSTM-XGBoost'),
        ('lstm_sarimax_results', 'LSTM-SARIMAX'),
        ('lstm_ann_results', 'LSTM-ANN'),
        # Optimized Models (08_Train_Models.py - Tab 3)
        ('opt_results_xgboost', 'XGBoost (Opt)'),
        ('opt_results_lstm', 'LSTM (Opt)'),
        ('opt_results_ann', 'ANN (Opt)'),
        ('opt_results_XGBoost', 'XGBoost (Opt)'),
        ('opt_results_LSTM', 'LSTM (Opt)'),
        ('opt_results_ANN', 'ANN (Opt)'),
    ]

    for key, default_name in model_sources:
        results = st.session_state.get(key)
        if results and isinstance(results, dict):
            # Try multiple metric extraction patterns
            metrics = _extract_metrics_from_results(results)

            if metrics:
                mape = metrics.get('mape')
                rmse = metrics.get('rmse')
                mae = metrics.get('mae')

                model_name = results.get('model_name', results.get('name', default_name))

                if mape is not None and not np.isnan(mape) and not np.isinf(mape):
                    models.append({
                        "name": model_name or default_name or key,
                        "mape": round(float(mape), 2),
                        "rmse": round(float(rmse), 2) if rmse else 0,
                        "mae": round(float(mae), 2) if mae else 0
                    })

                    if mape < best_mape:
                        best_mape = mape
                        best_model = {
                            "name": model_name or default_name or key,
                            "mape": mape,
                            "rmse": rmse
                        }

    # Remove duplicates (keep model with lowest MAPE if same name)
    seen_models = {}
    for m in models:
        name = m['name']
        if name not in seen_models or m['mape'] < seen_models[name]['mape']:
            seen_models[name] = m
    models = list(seen_models.values())

    if best_model:
        kpis.has_models = True
        kpis.best_model_name = best_model['name']
        kpis.best_model_mape = round(best_model['mape'], 2)
        kpis.best_model_rmse = round(best_model['rmse'], 2) if best_model['rmse'] else 0
        kpis.models_trained = len(models)
        kpis.model_comparison = sorted(models, key=lambda x: x['mape'])


def _extract_metrics_from_results(results: dict) -> Optional[Dict[str, float]]:
    """Extract MAPE, RMSE, MAE from various result formats."""
    if not results:
        return None

    # Pattern 1: Direct metrics dict
    metrics = results.get('metrics', results.get('test_metrics', {}))
    if isinstance(metrics, dict) and metrics:
        mape = metrics.get('MAPE', metrics.get('mape', metrics.get('Test_MAPE')))
        rmse = metrics.get('RMSE', metrics.get('rmse', metrics.get('Test_RMSE')))
        mae = metrics.get('MAE', metrics.get('mae', metrics.get('Test_MAE')))
        if mape is not None:
            return {'mape': mape, 'rmse': rmse, 'mae': mae}

    # Pattern 2: results_df with Test metrics (ML models)
    results_df = results.get('results_df')
    if results_df is not None and isinstance(results_df, pd.DataFrame) and not results_df.empty:
        if 'Test_MAPE' in results_df.columns:
            return {
                'mape': results_df['Test_MAPE'].mean(),
                'rmse': results_df['Test_RMSE'].mean() if 'Test_RMSE' in results_df.columns else None,
                'mae': results_df['Test_MAE'].mean() if 'Test_MAE' in results_df.columns else None
            }
        if 'Test_Acc' in results_df.columns:
            # Convert accuracy to MAPE (MAPE = 100 - Accuracy)
            return {
                'mape': 100 - results_df['Test_Acc'].mean(),
                'rmse': results_df['Test_RMSE'].mean() if 'Test_RMSE' in results_df.columns else None,
                'mae': results_df['Test_MAE'].mean() if 'Test_MAE' in results_df.columns else None
            }

    # Pattern 3: best_metrics (optimized models)
    best_metrics = results.get('best_metrics')
    if isinstance(best_metrics, dict) and best_metrics:
        mape = best_metrics.get('mape', best_metrics.get('MAPE'))
        rmse = best_metrics.get('rmse', best_metrics.get('RMSE'))
        mae = best_metrics.get('mae', best_metrics.get('MAE'))
        if mape is not None:
            return {'mape': mape, 'rmse': rmse, 'mae': mae}

    # Pattern 4: artifacts object (hybrid models)
    artifacts = results.get('artifacts')
    if artifacts and hasattr(artifacts, 'metrics'):
        metrics = artifacts.metrics
        if isinstance(metrics, dict):
            mape = metrics.get('MAPE', metrics.get('mape'))
            rmse = metrics.get('RMSE', metrics.get('rmse'))
            mae = metrics.get('MAE', metrics.get('mae'))
            if mape is not None:
                return {'mape': mape, 'rmse': rmse, 'mae': mae}

    return None


def _extract_category_distribution(kpis: ForecastKPIs) -> None:
    """Extract category distribution from seasonal proportions or forecast."""
    # Try to get from forecast hub (dict with category_forecasts)
    forecast_hub = st.session_state.get('forecast_hub_demand', {})

    # Check for category_forecasts dict (from 10_Patient_Forecast.py)
    if isinstance(forecast_hub, dict):
        category_forecasts = forecast_hub.get('category_forecasts', {})
        if category_forecasts and isinstance(category_forecasts, dict):
            grand_total = 0
            cat_totals = {}

            for cat, values in category_forecasts.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    cat_sum = float(np.sum(values))
                    cat_totals[cat.upper()] = cat_sum
                    grand_total += cat_sum

            if grand_total > 0:
                for cat, total in cat_totals.items():
                    pct = (total / grand_total * 100)
                    kpis.category_distribution.append({
                        "name": cat.title(),
                        "value": round(pct, 1),
                        "count": int(total),
                        "color": CATEGORY_COLORS.get(cat, "#6b7280")
                    })
                kpis.category_distribution = sorted(
                    kpis.category_distribution,
                    key=lambda x: x['value'],
                    reverse=True
                )
                return

    # Fallback: DataFrame format (legacy)
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
    """Extract KPIs from staff optimization results (11_Staff_Planner.py)."""
    # Check multiple possible session state keys
    staff_results = st.session_state.get('optimized_results')  # Main key from Staff Planner
    if not staff_results:
        staff_results = st.session_state.get('staff_optimization_results')  # Alternate key

    # Also check if optimization was done
    optimization_done = st.session_state.get('optimization_done', False)

    if (staff_results and isinstance(staff_results, dict) and staff_results.get('success')) or optimization_done:
        kpis.has_staff_plan = True

        if staff_results:
            # Extract from schedules list (primary data from MILP solver)
            schedules = staff_results.get('schedules', [])
            if schedules and isinstance(schedules, list):
                total_staff = 0
                total_cost = 0

                for sched in schedules:
                    if isinstance(sched, dict):
                        staff_count = sched.get('total_staff', 0) or (
                            sched.get('doctors', 0) +
                            sched.get('nurses', 0) +
                            sched.get('support', 0)
                        )
                        total_staff += staff_count
                        total_cost += sched.get('cost', 0)

                        # Build staff schedule for chart
                        date_str = sched.get('date', '')
                        # Extract day name from date
                        try:
                            if date_str:
                                day_abbr = pd.to_datetime(date_str).strftime('%a')
                            else:
                                day_abbr = f"Day {len(kpis.staff_schedule) + 1}"
                        except:
                            day_abbr = f"Day {len(kpis.staff_schedule) + 1}"

                        kpis.staff_schedule.append({
                            "day": day_abbr,
                            "staff": int(staff_count),
                            "demand": int(sched.get('patients', staff_count * 1.1))
                        })

                kpis.total_staff_needed = int(total_staff)
                kpis.daily_staff_cost = round(total_cost / len(schedules), 2) if schedules else 0

            # Fallback: Calculate from avg values
            if not kpis.total_staff_needed:
                avg_docs = staff_results.get('avg_opt_doctors', 0)
                avg_nurses = staff_results.get('avg_opt_nurses', 0)
                avg_support = staff_results.get('avg_opt_support', 0)
                kpis.total_staff_needed = int((avg_docs + avg_nurses + avg_support) * 7)

            # Coverage estimate (staff vs demand ratio)
            kpis.staff_coverage_pct = round(staff_results.get('coverage_pct',
                staff_results.get('utilization', 95.0)), 1)

            # Weekly savings
            weekly_savings = staff_results.get('weekly_savings', 0)
            if weekly_savings and not kpis.daily_staff_cost:
                # Approximate daily cost from savings
                kpis.daily_staff_cost = round(abs(weekly_savings) / 7, 2)

            kpis.overtime_hours = round(staff_results.get('total_overtime',
                staff_results.get('overtime_hours', 0)), 1)

    # If no schedule built yet, use empty (not placeholder data)
    if not kpis.staff_schedule and kpis.has_staff_plan:
        kpis.staff_schedule = []


def _extract_supply_kpis(kpis: ForecastKPIs) -> None:
    """Extract KPIs from supply/inventory optimization results (12_Supply_Planner.py)."""
    supply_results = st.session_state.get('inv_optimized_results')

    # Also check alternate key and optimization_done flag
    if not supply_results:
        supply_results = st.session_state.get('inventory_optimization_results')
    inv_optimization_done = st.session_state.get('inv_optimization_done', False)

    if (supply_results and isinstance(supply_results, dict) and supply_results.get('success')) or inv_optimization_done:
        kpis.has_supply_plan = True

        if supply_results:
            # Core metrics from Supply Planner
            kpis.supply_service_level = round(supply_results.get('service_level', 95.0), 1)

            # Total cost = sum of ordering, holding, purchase costs
            total_cost = supply_results.get('total_cost', 0)
            if not total_cost:
                total_cost = (
                    supply_results.get('ordering_cost', 0) +
                    supply_results.get('holding_cost', 0) +
                    supply_results.get('purchase_cost', 0)
                )
            kpis.supply_total_cost = round(total_cost, 2)

            # Weekly savings (if available)
            kpis.supply_weekly_savings = round(supply_results.get('weekly_savings', 0), 2)

            # Item breakdown from item_results dict
            item_results = supply_results.get('item_results', {})
            kpis.supply_items_count = len(item_results)

            # Count reorder alerts and build item breakdown
            reorder_alerts = 0
            for item_id, item_data in item_results.items():
                if isinstance(item_data, dict):
                    # Check if item has reorder point set (potential alert)
                    reorder_point = item_data.get('reorder_point', 0)
                    safety_stock = item_data.get('safety_stock', 0)
                    if reorder_point > 0 and safety_stock > 0:
                        reorder_alerts += 1

                    # Calculate item total cost
                    item_cost = (
                        item_data.get('ordering_cost', 0) +
                        item_data.get('holding_cost', 0) +
                        item_data.get('purchase_cost', 0)
                    )

                    kpis.supply_item_breakdown.append({
                        "name": str(item_data.get('name', item_id))[:20],
                        "demand": round(float(item_data.get('forecast_demand', 0)), 0),
                        "order_qty": int(item_data.get('optimal_order', 0)),
                        "safety_stock": round(float(item_data.get('safety_stock', 0)), 0),
                        "cost": round(float(item_cost), 0)
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
    _extract_supply_kpis(kpis)

    return kpis


# Legacy compatibility
def calculate_hospital_kpis(*args, **kwargs) -> ForecastKPIs:
    """Legacy function name - redirects to calculate_forecast_kpis."""
    return calculate_forecast_kpis()


# Alias for dataclass
HospitalKPIs = ForecastKPIs
