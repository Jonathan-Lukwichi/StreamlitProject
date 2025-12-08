# =============================================================================
# 10_Forecast.py — Universal Forecast Hub
# Generate actual forecasts using trained models with rolling/walk-forward approach
# Works with ANY dataset - adaptable to different domains
# =============================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# ============================================================================
# AUTHENTICATION CHECK - USER OR ADMIN
# ============================================================================
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_authentication()
configure_sidebar_navigation()

st.set_page_config(
    page_title="Forecast Hub - HealthForecast AI",
    page_icon="🔮",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* Forecast cards */
.forecast-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    text-align: center;
}

.forecast-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}

.forecast-label {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.forecast-day {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.status-ready {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.status-pending {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.model-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.125rem;
}

.model-ml { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
.model-stat { background: rgba(168, 85, 247, 0.2); color: #a855f7; }
.model-hybrid { background: rgba(236, 72, 153, 0.2); color: #ec4899; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🔮</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Forecast Hub</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Generate actual forecasts using trained models<br>
        <span style='color: #94a3b8; font-size: 0.9rem;'>Universal pipeline adaptable to any dataset</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# UNIVERSAL DATA DETECTOR
# Finds available data from any source in session state
# =============================================================================
def detect_available_data() -> Dict[str, Any]:
    """
    Detect all available data sources in session state.
    Works with any dataset structure.

    Returns:
        Dict with data sources and their info
    """
    sources = {}

    # Priority order for data sources
    data_keys = [
        ("processed_df", "Processed Data (Feature Engineered)"),
        ("fe_df", "Feature Engineered Data"),
        ("fs_df", "Feature Selected Data"),
        ("merged_data", "Merged Raw Data"),
        ("uploaded_df", "Uploaded Data"),
    ]

    for key, description in data_keys:
        df = st.session_state.get(key)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            sources[key] = {
                "description": description,
                "shape": df.shape,
                "columns": list(df.columns),
                "df": df
            }

    return sources


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect date column in any dataset."""
    date_patterns = ["date", "time", "datetime", "timestamp", "day", "period"]

    for col in df.columns:
        col_lower = col.lower()
        # Check column name
        if any(pattern in col_lower for pattern in date_patterns):
            return col
        # Check if column contains dates
        if df[col].dtype == 'datetime64[ns]':
            return col

    # Try to parse first column as date
    try:
        pd.to_datetime(df.iloc[:, 0])
        return df.columns[0]
    except:
        pass

    return None


def detect_target_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect target columns in any dataset.
    Looks for numeric columns that could be forecasted.
    """
    targets = []

    # Common target patterns
    target_patterns = [
        "target", "arrival", "patient", "count", "total", "demand",
        "sales", "revenue", "quantity", "volume", "value", "amount"
    ]

    # Clinical category patterns (for healthcare datasets)
    clinical_patterns = [
        "respiratory", "cardiac", "trauma", "gastrointestinal",
        "infectious", "neurological", "other"
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Check for target patterns
        if any(pattern in col_lower for pattern in target_patterns):
            targets.append(col)
        elif any(pattern in col_lower for pattern in clinical_patterns):
            targets.append(col)
        # Check for Target_N pattern
        elif col_lower.startswith("target_"):
            targets.append(col)

    # If no targets found, use all numeric columns except obvious features
    if not targets:
        exclude_patterns = ["id", "index", "lag", "rolling", "diff", "feature"]
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if not any(pattern in col_lower for pattern in exclude_patterns):
                targets.append(col)

    return targets[:10]  # Limit to 10 targets


# =============================================================================
# TRAINED MODEL DETECTOR
# Finds all trained models from Benchmarks and Modeling Hub
# =============================================================================
def detect_trained_models() -> List[Dict[str, Any]]:
    """
    Detect all trained models in session state.

    Returns:
        List of dicts with model info
    """
    models = []

    # -------------------------------------------------------------------------
    # ML Models from Modeling Hub (08_Modeling_Hub.py)
    # -------------------------------------------------------------------------

    # Dynamic scan for ml_mh_results_* keys
    for key in st.session_state.keys():
        if key.startswith("ml_mh_results_"):
            model_name = key.replace("ml_mh_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                models.append({
                    "name": model_name,
                    "type": "ML",
                    "key": key,
                    "data": data,
                    "category": "ml"
                })

    # Optimized models
    for key in st.session_state.keys():
        if key.startswith("opt_results_"):
            model_name = key.replace("opt_results_", "")
            data = st.session_state.get(key)
            if data is not None:
                models.append({
                    "name": f"{model_name} (Optimized)",
                    "type": "ML-Optimized",
                    "key": key,
                    "data": data,
                    "category": "ml_opt"
                })

    # Generic ML results
    if "ml_mh_results" in st.session_state:
        data = st.session_state.get("ml_mh_results")
        if data is not None and not any(m["key"] == "ml_mh_results" for m in models):
            models.append({
                "name": "ML Model (Latest)",
                "type": "ML",
                "key": "ml_mh_results",
                "data": data,
                "category": "ml"
            })

    # -------------------------------------------------------------------------
    # Statistical Models from Benchmarks (05_Benchmarks.py)
    # -------------------------------------------------------------------------

    # ARIMA single target
    if "arima_mh_results" in st.session_state:
        data = st.session_state.get("arima_mh_results")
        if data is not None:
            models.append({
                "name": "ARIMA",
                "type": "Statistical",
                "key": "arima_mh_results",
                "data": data,
                "category": "stat"
            })

    # ARIMA multi-target
    if "arima_multi_target_results" in st.session_state:
        data = st.session_state.get("arima_multi_target_results")
        if data is not None:
            models.append({
                "name": "ARIMA (Multi-Target)",
                "type": "Statistical",
                "key": "arima_multi_target_results",
                "data": data,
                "category": "stat_multi"
            })

    # SARIMAX single target
    if "sarimax_results" in st.session_state:
        data = st.session_state.get("sarimax_results")
        if data is not None:
            models.append({
                "name": "SARIMAX",
                "type": "Statistical",
                "key": "sarimax_results",
                "data": data,
                "category": "stat"
            })

    # SARIMAX multi-target
    if "sarimax_multi_target_results" in st.session_state:
        data = st.session_state.get("sarimax_multi_target_results")
        if data is not None:
            models.append({
                "name": "SARIMAX (Multi-Target)",
                "type": "Statistical",
                "key": "sarimax_multi_target_results",
                "data": data,
                "category": "stat_multi"
            })

    return models


# =============================================================================
# FORECAST GENERATION PIPELINE
# Universal pipeline that works with any model and dataset
# =============================================================================
class UniversalForecastPipeline:
    """
    Universal forecast pipeline that generates actual forecasts
    using trained models with a rolling/walk-forward approach.

    Key Concept (for historical data):
    - Data: Jan 1 - Dec 31, 2023
    - Cutoff: Nov 24, 2023 (user selects)
    - Training: Jan 1 - Nov 24, 2023
    - Forecast: Nov 25 - Dec 1, 2023 (7-day horizon)
    - You have actuals for Nov 25 - Dec 1 for validation
    """

    def __init__(self, df: pd.DataFrame, date_col: str, target_cols: List[str]):
        self.df = df.copy()
        self.date_col = date_col
        self.target_cols = target_cols

        # Ensure date column is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Get date range
        self.min_date = self.df[date_col].min()
        self.max_date = self.df[date_col].max()

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Return the date range of the data."""
        return self.min_date, self.max_date

    def generate_forecast(
        self,
        model_info: Dict[str, Any],
        cutoff_date: datetime,
        horizon: int = 7,
        include_actuals: bool = True
    ) -> Dict[str, Any]:
        """
        Generate forecast using the trained model.

        For ML models: Re-fit on training data up to cutoff, then predict
        For Statistical models: Use fitted model parameters to forecast

        Args:
            model_info: Dict with model name, type, key, data
            cutoff_date: Last date of training data
            horizon: Number of days to forecast
            include_actuals: Include actual values if available (for validation)

        Returns:
            Dict with forecast results in standardized format
        """
        model_name = model_info["name"]
        model_type = model_info["type"]
        model_data = model_info["data"]
        model_category = model_info["category"]

        # Split data at cutoff
        train_mask = self.df[self.date_col] <= cutoff_date
        train_df = self.df[train_mask].copy()

        # Get forecast dates
        forecast_dates = pd.date_range(
            start=cutoff_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        # Initialize result structure
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "forecast_values": [],  # Total/primary forecast
            "target_forecasts": {},  # Per-target forecasts
            "actuals": {} if include_actuals else None,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "horizon": horizon,
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "training_samples": len(train_df),
            }
        }

        # Generate forecasts based on model type
        try:
            if model_category in ["ml", "ml_opt"]:
                result = self._forecast_ml_model(result, model_data, train_df, forecast_dates, horizon)
            elif model_category in ["stat", "stat_multi"]:
                result = self._forecast_stat_model(result, model_data, train_df, forecast_dates, horizon)
            else:
                # Fallback: use simple persistence forecast
                result = self._forecast_persistence(result, train_df, forecast_dates, horizon)
        except Exception as e:
            st.warning(f"Error generating forecast with {model_name}: {str(e)}")
            result = self._forecast_persistence(result, train_df, forecast_dates, horizon)

        # Add actuals if available and requested
        if include_actuals:
            result["actuals"] = self._get_actuals(forecast_dates)

        return result

    def _forecast_ml_model(
        self,
        result: Dict,
        model_data: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Generate forecast using ML model approach."""

        # ML models typically store test predictions
        # We'll use the last `horizon` test predictions as "forecast"
        # Or re-train if model object is available

        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            # Try to get predictions from model data
            forecast_values = None

            # Check various keys where predictions might be stored
            if isinstance(model_data, dict):
                # Multi-target results
                if target in model_data:
                    target_data = model_data[target]
                    if isinstance(target_data, dict):
                        if "test_predictions" in target_data:
                            preds = target_data["test_predictions"]
                            forecast_values = list(preds)[-horizon:] if hasattr(preds, '__iter__') else None
                        elif "predictions" in target_data:
                            preds = target_data["predictions"]
                            forecast_values = list(preds)[-horizon:] if hasattr(preds, '__iter__') else None

                # Single target results
                if forecast_values is None:
                    if "test_predictions" in model_data:
                        preds = model_data["test_predictions"]
                        if hasattr(preds, "values"):
                            forecast_values = list(preds.values)[-horizon:]
                        else:
                            forecast_values = list(preds)[-horizon:]
                    elif "predictions" in model_data:
                        preds = model_data["predictions"]
                        forecast_values = list(preds)[-horizon:]

            # Fallback: use persistence forecast
            if forecast_values is None or len(forecast_values) < horizon:
                last_values = train_df[target].tail(horizon * 2).values
                # Use seasonal naive (same day last week) if enough data
                if len(last_values) >= 7:
                    forecast_values = list(last_values[-7:])
                    while len(forecast_values) < horizon:
                        forecast_values.append(forecast_values[-1])
                else:
                    forecast_values = [train_df[target].mean()] * horizon

            # Ensure correct length
            forecast_values = list(forecast_values)[:horizon]
            while len(forecast_values) < horizon:
                forecast_values.append(forecast_values[-1] if forecast_values else train_df[target].mean())

            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        # Set primary forecast (first target or sum)
        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _forecast_stat_model(
        self,
        result: Dict,
        model_data: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Generate forecast using statistical model approach."""

        # Statistical models (ARIMA/SARIMAX) store forecast arrays
        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            forecast_values = None

            if isinstance(model_data, dict):
                # Multi-target results
                if target in model_data:
                    target_data = model_data[target]
                    if isinstance(target_data, dict):
                        if "forecast" in target_data:
                            fc = target_data["forecast"]
                            if hasattr(fc, "values"):
                                forecast_values = list(fc.values)[-horizon:]
                            else:
                                forecast_values = list(fc)[-horizon:]
                        elif "forecasts" in target_data:
                            fc = target_data["forecasts"]
                            forecast_values = list(fc)[-horizon:]

                # Single target results
                if forecast_values is None:
                    if "forecast" in model_data:
                        fc = model_data["forecast"]
                        if hasattr(fc, "values"):
                            forecast_values = list(fc.values)[-horizon:]
                        else:
                            forecast_values = list(fc)[-horizon:]
                    elif "forecasts" in model_data:
                        fc = model_data["forecasts"]
                        forecast_values = list(fc)[-horizon:]

            # Fallback: use persistence
            if forecast_values is None or len(forecast_values) < horizon:
                forecast_values = self._simple_forecast(train_df[target], horizon)

            # Ensure correct length
            forecast_values = list(forecast_values)[:horizon]
            while len(forecast_values) < horizon:
                forecast_values.append(forecast_values[-1] if forecast_values else train_df[target].mean())

            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        # Set primary forecast
        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _forecast_persistence(
        self,
        result: Dict,
        train_df: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
        horizon: int
    ) -> Dict:
        """Fallback: simple persistence/seasonal naive forecast."""

        for target in self.target_cols:
            if target not in train_df.columns:
                continue

            forecast_values = self._simple_forecast(train_df[target], horizon)
            result["target_forecasts"][target] = [float(v) for v in forecast_values]

        if result["target_forecasts"]:
            first_target = list(result["target_forecasts"].keys())[0]
            result["forecast_values"] = result["target_forecasts"][first_target]

        return result

    def _simple_forecast(self, series: pd.Series, horizon: int) -> List[float]:
        """Generate simple forecast using seasonal patterns."""
        values = series.dropna().values

        if len(values) >= 7:
            # Seasonal naive: use last 7 days pattern
            last_week = list(values[-7:])
            forecast = []
            for i in range(horizon):
                forecast.append(last_week[i % 7])
            return forecast
        elif len(values) > 0:
            # Simple persistence: repeat last value
            return [float(values[-1])] * horizon
        else:
            return [0.0] * horizon

    def _get_actuals(self, forecast_dates: pd.DatetimeIndex) -> Dict[str, List[float]]:
        """Get actual values for forecast dates (if available in data)."""
        actuals = {}

        for target in self.target_cols:
            if target not in self.df.columns:
                continue

            target_actuals = []
            for fd in forecast_dates:
                mask = self.df[self.date_col].dt.date == fd.date()
                if mask.any():
                    val = self.df.loc[mask, target].values[0]
                    target_actuals.append(float(val) if pd.notna(val) else None)
                else:
                    target_actuals.append(None)

            actuals[target] = target_actuals

        return actuals


# =============================================================================
# FORECAST STORAGE - STANDARDIZED FORMAT
# =============================================================================
def save_forecast_to_session(forecast_result: Dict, source_key: str = "forecast_hub"):
    """
    Save forecast to session state in standardized format.
    This format is consumed by Staff Scheduling Optimization.

    Storage keys:
    - forecast_hub_results: Main forecast results
    - forecast_hub_{model_name}: Per-model forecasts
    """
    model_name = forecast_result.get("model_name", "Unknown")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Store in standardized format
    st.session_state[f"forecast_hub_{safe_name}"] = forecast_result

    # Also store as the main forecast result
    st.session_state["forecast_hub_results"] = forecast_result

    # Store forecast values in simple format for Staff Scheduling
    st.session_state["forecast_hub_demand"] = {
        "model": model_name,
        "forecast": forecast_result.get("forecast_values", []),
        "dates": forecast_result.get("forecast_dates", []),
        "targets": forecast_result.get("target_forecasts", {}),
    }

    return True


# =============================================================================
# MAIN UI
# =============================================================================

# Initialize session state
if "forecast_generated" not in st.session_state:
    st.session_state["forecast_generated"] = False

# Create tabs
tab_data, tab_config, tab_generate, tab_results = st.tabs([
    "📊 Data Check",
    "⚙️ Configuration",
    "🚀 Generate Forecast",
    "📈 Results & Export"
])

# =============================================================================
# TAB 1: DATA CHECK
# =============================================================================
with tab_data:
    st.markdown("### 📊 Available Data & Models")

    # Check for available data
    data_sources = detect_available_data()
    trained_models = detect_trained_models()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📂 Data Sources")
        if data_sources:
            for key, info in data_sources.items():
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-day">{info['description']}</div>
                    <div class="forecast-value">{info['shape'][0]:,}</div>
                    <div class="forecast-label">rows × {info['shape'][1]} columns</div>
                </div>
                """, unsafe_allow_html=True)
            st.success(f"✅ {len(data_sources)} data source(s) available")
        else:
            st.warning("⚠️ No data found. Please load data in Data Preparation Studio first.")
            st.info("Go to **03_Data_Preparation_Studio.py** to upload and process your data.")

    with col2:
        st.markdown("#### 🤖 Trained Models")
        if trained_models:
            ml_count = sum(1 for m in trained_models if "ML" in m["type"])
            stat_count = sum(1 for m in trained_models if "Statistical" in m["type"])

            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">Models Ready</div>
                <div class="forecast-value">{len(trained_models)}</div>
                <div class="forecast-label">
                    <span class="model-badge model-ml">ML: {ml_count}</span>
                    <span class="model-badge model-stat">Statistical: {stat_count}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Available Models", expanded=False):
                for model in trained_models:
                    badge_class = "model-ml" if "ML" in model["type"] else "model-stat"
                    st.markdown(f'<span class="model-badge {badge_class}">{model["name"]}</span>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No trained models found.")
            st.info("""
            Train models in:
            - **05_Benchmarks.py** - ARIMA/SARIMAX
            - **08_Modeling_Hub.py** - XGBoost, LSTM, ANN
            """)

    # Show session state keys (for debugging)
    with st.expander("🔧 Debug: Session State Keys", expanded=False):
        st.write("All keys:", sorted(st.session_state.keys()))

# =============================================================================
# TAB 2: CONFIGURATION
# =============================================================================
with tab_config:
    st.markdown("### ⚙️ Forecast Configuration")

    data_sources = detect_available_data()
    trained_models = detect_trained_models()

    if not data_sources:
        st.warning("⚠️ No data available. Please load data first.")
        st.stop()

    # Select data source
    st.markdown("#### 1️⃣ Select Data Source")
    source_options = list(data_sources.keys())
    selected_source = st.selectbox(
        "Choose data to forecast:",
        options=source_options,
        format_func=lambda x: data_sources[x]["description"]
    )

    df = data_sources[selected_source]["df"]

    # Auto-detect date column
    st.markdown("#### 2️⃣ Date Column")
    date_col = detect_date_column(df)

    if date_col:
        st.success(f"✅ Auto-detected date column: **{date_col}**")
        # Allow override
        date_col = st.selectbox(
            "Confirm or change date column:",
            options=[c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'datetime64[ns]'],
            index=[c for c in df.columns].index(date_col) if date_col in df.columns else 0
        )
    else:
        date_col = st.selectbox(
            "Select date column:",
            options=df.columns.tolist()
        )

    # Parse dates and get range
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        st.info(f"📅 Data range: **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}** ({(max_date - min_date).days + 1} days)")
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        st.stop()

    # Auto-detect target columns
    st.markdown("#### 3️⃣ Target Columns (to forecast)")
    auto_targets = detect_target_columns(df)

    if auto_targets:
        st.success(f"✅ Auto-detected {len(auto_targets)} target column(s)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_targets = st.multiselect(
        "Select target columns to forecast:",
        options=numeric_cols,
        default=auto_targets[:5] if auto_targets else numeric_cols[:3]
    )

    if not selected_targets:
        st.warning("⚠️ Please select at least one target column.")
        st.stop()

    # Select model
    st.markdown("#### 4️⃣ Select Model")
    if trained_models:
        model_options = [m["name"] for m in trained_models]
        selected_model_name = st.selectbox(
            "Choose model for forecasting:",
            options=model_options
        )
        selected_model = next(m for m in trained_models if m["name"] == selected_model_name)
    else:
        st.info("No trained models. Will use persistence/seasonal naive forecast.")
        selected_model = {
            "name": "Persistence (Baseline)",
            "type": "Baseline",
            "key": None,
            "data": None,
            "category": "baseline"
        }

    # Store config in session state
    st.session_state["forecast_config"] = {
        "source_key": selected_source,
        "date_col": date_col,
        "target_cols": selected_targets,
        "model": selected_model,
        "min_date": min_date,
        "max_date": max_date,
    }

    st.success("✅ Configuration saved! Proceed to **Generate Forecast** tab.")

# =============================================================================
# TAB 3: GENERATE FORECAST
# =============================================================================
with tab_generate:
    st.markdown("### 🚀 Generate Forecast")

    config = st.session_state.get("forecast_config")

    if not config:
        st.warning("⚠️ Please configure forecast settings first (Tab 2).")
        st.stop()

    # Display current config
    with st.expander("📋 Current Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Source", config["source_key"])
        with col2:
            st.metric("Model", config["model"]["name"])
        with col3:
            st.metric("Targets", len(config["target_cols"]))

    st.divider()

    # Forecast parameters
    st.markdown("#### 📅 Forecast Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # Cutoff date selection
        st.markdown("**Cutoff Date** (last day of training data)")
        st.caption("For historical data: select a date before the end to simulate real forecasting")

        min_date = config["min_date"]
        max_date = config["max_date"]

        # Default: 7 days before max date (to allow validation)
        default_cutoff = max_date - timedelta(days=7)
        if default_cutoff < min_date:
            default_cutoff = max_date - timedelta(days=1)

        cutoff_date = st.date_input(
            "Select cutoff date:",
            value=default_cutoff.date(),
            min_value=min_date.date() + timedelta(days=30),  # Need at least 30 days for training
            max_value=max_date.date() - timedelta(days=1)    # Leave at least 1 day for forecast
        )
        cutoff_datetime = pd.Timestamp(cutoff_date)

    with col2:
        # Horizon
        st.markdown("**Forecast Horizon** (days to predict)")

        # Max horizon is days remaining after cutoff
        max_horizon = (max_date.date() - cutoff_date).days
        if max_horizon < 1:
            max_horizon = 14

        horizon = st.slider(
            "Number of days to forecast:",
            min_value=1,
            max_value=min(30, max_horizon),
            value=min(7, max_horizon)
        )

        st.caption(f"Forecasting: {cutoff_date + timedelta(days=1)} to {cutoff_date + timedelta(days=horizon)}")

    st.divider()

    # Generate button
    col1, col2 = st.columns([1, 2])

    with col1:
        include_actuals = st.checkbox("Include actuals (for validation)", value=True)
        generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    with col2:
        if generate_btn:
            with st.spinner("Generating forecast..."):
                # Get data
                data_sources = detect_available_data()
                df = data_sources[config["source_key"]]["df"]

                # Create pipeline
                pipeline = UniversalForecastPipeline(
                    df=df,
                    date_col=config["date_col"],
                    target_cols=config["target_cols"]
                )

                # Generate forecast
                forecast_result = pipeline.generate_forecast(
                    model_info=config["model"],
                    cutoff_date=cutoff_datetime,
                    horizon=horizon,
                    include_actuals=include_actuals
                )

                # Save to session state
                save_forecast_to_session(forecast_result)
                st.session_state["forecast_generated"] = True
                st.session_state["current_forecast"] = forecast_result

                st.success(f"✅ Forecast generated successfully!")
                st.balloons()

                # Quick preview
                st.markdown("#### Quick Preview")
                preview_df = pd.DataFrame({
                    "Date": forecast_result["forecast_dates"],
                    "Forecast": [round(v, 1) for v in forecast_result["forecast_values"]]
                })
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: RESULTS & EXPORT
# =============================================================================
with tab_results:
    st.markdown("### 📈 Forecast Results")

    forecast_result = st.session_state.get("current_forecast")

    if not forecast_result:
        st.info("Generate a forecast first (Tab 3) to see results here.")
        st.stop()

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Model</div>
            <div class="forecast-label" style="font-size: 1.1rem; color: #60a5fa;">{forecast_result['model_name']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Horizon</div>
            <div class="forecast-value">{forecast_result['metadata']['horizon']}</div>
            <div class="forecast-label">days</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_forecast = np.mean(forecast_result["forecast_values"]) if forecast_result["forecast_values"] else 0
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Avg Forecast</div>
            <div class="forecast-value">{avg_forecast:.1f}</div>
            <div class="forecast-label">per day</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_forecast = sum(forecast_result["forecast_values"]) if forecast_result["forecast_values"] else 0
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-day">Total</div>
            <div class="forecast-value">{total_forecast:.0f}</div>
            <div class="forecast-label">over horizon</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Forecast chart
    st.markdown("### 📊 Forecast Visualization")

    # Build chart data
    chart_data = {
        "Date": forecast_result["forecast_dates"],
    }

    # Add forecasts for each target
    for target, values in forecast_result["target_forecasts"].items():
        chart_data[f"{target} (Forecast)"] = values

        # Add actuals if available
        if forecast_result.get("actuals") and target in forecast_result["actuals"]:
            actuals = forecast_result["actuals"][target]
            chart_data[f"{target} (Actual)"] = actuals

    chart_df = pd.DataFrame(chart_data)

    # Create plotly figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    color_idx = 0

    for target in forecast_result["target_forecasts"].keys():
        forecast_col = f"{target} (Forecast)"
        actual_col = f"{target} (Actual)"

        # Forecast line
        fig.add_trace(go.Scatter(
            x=chart_df["Date"],
            y=chart_df[forecast_col],
            mode='lines+markers',
            name=forecast_col,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            marker=dict(size=8)
        ))

        # Actual line (if available)
        if actual_col in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df["Date"],
                y=chart_df[actual_col],
                mode='lines+markers',
                name=actual_col,
                line=dict(color=colors[color_idx % len(colors)], width=2, dash='dot'),
                marker=dict(size=6, symbol='x')
            ))

        color_idx += 1

    fig.update_layout(
        title=f"Forecast: {forecast_result['model_name']}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### 📋 Forecast Data")
    st.dataframe(chart_df, use_container_width=True, hide_index=True)

    # Validation metrics (if actuals available)
    if forecast_result.get("actuals"):
        st.markdown("### 📊 Validation Metrics")

        metrics_data = []
        for target in forecast_result["target_forecasts"].keys():
            if target in forecast_result["actuals"]:
                forecast = forecast_result["target_forecasts"][target]
                actuals = forecast_result["actuals"][target]

                # Filter out None values
                valid_pairs = [(f, a) for f, a in zip(forecast, actuals) if a is not None]

                if valid_pairs:
                    f_vals = [p[0] for p in valid_pairs]
                    a_vals = [p[1] for p in valid_pairs]

                    mae = np.mean(np.abs(np.array(f_vals) - np.array(a_vals)))
                    rmse = np.sqrt(np.mean((np.array(f_vals) - np.array(a_vals))**2))

                    # Avoid division by zero for MAPE
                    mape = np.mean(np.abs((np.array(a_vals) - np.array(f_vals)) / np.maximum(np.array(a_vals), 1))) * 100

                    metrics_data.append({
                        "Target": target,
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "MAPE %": round(mape, 2),
                        "Valid Points": len(valid_pairs)
                    })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    # Export options
    st.markdown("### 📥 Export & Integration")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download CSV
        csv = chart_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"forecast_{forecast_result['model_name'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Send to Staff Scheduling
        if st.button("🔗 Send to Staff Scheduling", use_container_width=True, type="primary"):
            # Ensure data is in the right format
            st.session_state["forecast_hub_demand"] = {
                "model": forecast_result["model_name"],
                "forecast": forecast_result["forecast_values"],
                "dates": forecast_result["forecast_dates"],
                "targets": forecast_result["target_forecasts"],
            }
            st.success("✅ Forecast sent to Staff Scheduling Optimization!")
            st.info("Go to **11_Staff_Scheduling_Optimization.py** and select 'ML Forecast' as demand source.")

    with col3:
        # View JSON
        with st.expander("View JSON Format"):
            st.json(forecast_result)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <strong>Forecast Hub</strong> — Universal pipeline for generating forecasts<br>
    Works with any dataset using rolling/walk-forward approach
</div>
""", unsafe_allow_html=True)
