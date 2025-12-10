# =============================================================================
# app_core/state/typed_state.py
# Typed State Management for HealthForecast AI
# Provides type-safe access to session state with validation
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st

from app_core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# STATE DATA CLASSES
# =============================================================================

@dataclass
class DataState:
    """State for raw uploaded data"""
    patient_data: Optional[pd.DataFrame] = None
    weather_data: Optional[pd.DataFrame] = None
    calendar_data: Optional[pd.DataFrame] = None
    reason_data: Optional[pd.DataFrame] = None

    def is_complete(self) -> bool:
        """Check if all required data sources are loaded"""
        return all([
            self.patient_data is not None,
            self.weather_data is not None,
            self.calendar_data is not None,
            self.reason_data is not None,
        ])


@dataclass
class PreparedDataState:
    """State for prepared/processed data"""
    merged_data: Optional[pd.DataFrame] = None
    processed_df: Optional[pd.DataFrame] = None
    date_column: Optional[str] = None
    target_columns: Optional[List[str]] = None
    use_aggregated_categories: bool = True


@dataclass
class FeatureEngineeringState:
    """State for feature engineering results"""
    X_engineered: Optional[pd.DataFrame] = None
    y_engineered: Optional[pd.DataFrame] = None
    feature_names: Optional[List[str]] = None
    train_idx: Optional[np.ndarray] = None
    cal_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    variant: str = "A"  # A = OneHot, B = Cyclical
    fe_config: Optional[Dict[str, Any]] = None


@dataclass
class FeatureSelectionState:
    """State for feature selection results"""
    selected_features: Optional[List[str]] = None
    feature_importances: Optional[Dict[str, float]] = None
    selection_method: Optional[str] = None
    fs_results: Optional[Dict[str, Any]] = None


@dataclass
class ModelingState:
    """State for model training results"""
    ml_results: Optional[Dict[str, Any]] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    hybrid_results: Optional[Dict[str, Any]] = None
    cqr_results: Optional[Dict[str, Any]] = None
    best_model_name: Optional[str] = None
    best_model_metrics: Optional[Dict[str, float]] = None


@dataclass
class ForecastState:
    """State for forecast results"""
    forecasts: Optional[pd.DataFrame] = None
    prediction_intervals: Optional[pd.DataFrame] = None
    forecast_horizons: Optional[List[int]] = None


@dataclass
class PipelineState:
    """
    Master state container for the entire pipeline.
    Aggregates all sub-states into a single typed structure.
    """
    data: DataState = field(default_factory=DataState)
    prepared: PreparedDataState = field(default_factory=PreparedDataState)
    feature_engineering: FeatureEngineeringState = field(default_factory=FeatureEngineeringState)
    feature_selection: FeatureSelectionState = field(default_factory=FeatureSelectionState)
    modeling: ModelingState = field(default_factory=ModelingState)
    forecast: ForecastState = field(default_factory=ForecastState)

    # Metadata
    user_id: Optional[str] = None
    dataset_hash: Optional[str] = None


# =============================================================================
# STATE MANAGER SINGLETON
# =============================================================================

class StateManager:
    """
    Singleton state manager providing typed access to session state.

    Usage:
        from app_core.state.typed_state import get_state_manager

        state = get_state_manager()

        # Set data
        state.set_merged_data(df)

        # Get data with type hints
        df = state.pipeline.prepared.merged_data

        # Check pipeline progress
        if state.pipeline.data.is_complete():
            st.success("All data loaded!")
    """

    _instance: Optional[StateManager] = None
    SESSION_KEY = "_typed_pipeline_state"

    def __new__(cls) -> StateManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize state in session if not present"""
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = PipelineState()
            logger.info("Initialized new PipelineState")

    @property
    def pipeline(self) -> PipelineState:
        """Get the current pipeline state"""
        self._initialize()  # Ensure initialized
        return st.session_state[self.SESSION_KEY]

    # -------------------------------------------------------------------------
    # DATA STATE SETTERS
    # -------------------------------------------------------------------------

    def set_patient_data(self, df: pd.DataFrame) -> None:
        """Set patient data with validation"""
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("patient_data must be a pandas DataFrame")
        self.pipeline.data.patient_data = df
        logger.info(f"Set patient_data: {df.shape if df is not None else None}")

    def set_weather_data(self, df: pd.DataFrame) -> None:
        """Set weather data with validation"""
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("weather_data must be a pandas DataFrame")
        self.pipeline.data.weather_data = df
        logger.info(f"Set weather_data: {df.shape if df is not None else None}")

    def set_calendar_data(self, df: pd.DataFrame) -> None:
        """Set calendar data with validation"""
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("calendar_data must be a pandas DataFrame")
        self.pipeline.data.calendar_data = df
        logger.info(f"Set calendar_data: {df.shape if df is not None else None}")

    def set_reason_data(self, df: pd.DataFrame) -> None:
        """Set reason data with validation"""
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("reason_data must be a pandas DataFrame")
        self.pipeline.data.reason_data = df
        logger.info(f"Set reason_data: {df.shape if df is not None else None}")

    # -------------------------------------------------------------------------
    # PREPARED DATA SETTERS
    # -------------------------------------------------------------------------

    def set_merged_data(self, df: pd.DataFrame) -> None:
        """Set merged/fused data"""
        self.pipeline.prepared.merged_data = df
        logger.info(f"Set merged_data: {df.shape if df is not None else None}")

    def set_processed_data(self, df: pd.DataFrame) -> None:
        """Set processed data with lag features"""
        self.pipeline.prepared.processed_df = df
        logger.info(f"Set processed_df: {df.shape if df is not None else None}")

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING SETTERS
    # -------------------------------------------------------------------------

    def set_temporal_split(
        self,
        train_idx: np.ndarray,
        cal_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> None:
        """Set temporal split indices"""
        self.pipeline.feature_engineering.train_idx = train_idx
        self.pipeline.feature_engineering.cal_idx = cal_idx
        self.pipeline.feature_engineering.test_idx = test_idx
        logger.info(
            f"Set temporal split: train={len(train_idx)}, "
            f"cal={len(cal_idx)}, test={len(test_idx)}"
        )

    def set_engineered_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_names: List[str]
    ) -> None:
        """Set engineered feature matrices"""
        self.pipeline.feature_engineering.X_engineered = X
        self.pipeline.feature_engineering.y_engineered = y
        self.pipeline.feature_engineering.feature_names = feature_names
        logger.info(f"Set engineered features: X={X.shape}, y={y.shape}")

    # -------------------------------------------------------------------------
    # MODELING SETTERS
    # -------------------------------------------------------------------------

    def set_ml_results(self, results: Dict[str, Any]) -> None:
        """Set ML model results"""
        self.pipeline.modeling.ml_results = results
        logger.info(f"Set ml_results: {list(results.keys()) if results else None}")

    def set_cqr_results(self, results: Dict[str, Any]) -> None:
        """Set CQR (Conformal Prediction) results"""
        self.pipeline.modeling.cqr_results = results
        logger.info("Set cqr_results")

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all pipeline state"""
        st.session_state[self.SESSION_KEY] = PipelineState()
        logger.info("Cleared all pipeline state")

    def clear_downstream(self, from_stage: str) -> None:
        """
        Clear state from a given stage onwards.

        Stages: 'data', 'prepared', 'feature_engineering',
                'feature_selection', 'modeling', 'forecast'
        """
        stages = [
            'data', 'prepared', 'feature_engineering',
            'feature_selection', 'modeling', 'forecast'
        ]

        if from_stage not in stages:
            raise ValueError(f"Unknown stage: {from_stage}")

        start_idx = stages.index(from_stage)

        for stage in stages[start_idx:]:
            if stage == 'data':
                self.pipeline.data = DataState()
            elif stage == 'prepared':
                self.pipeline.prepared = PreparedDataState()
            elif stage == 'feature_engineering':
                self.pipeline.feature_engineering = FeatureEngineeringState()
            elif stage == 'feature_selection':
                self.pipeline.feature_selection = FeatureSelectionState()
            elif stage == 'modeling':
                self.pipeline.modeling = ModelingState()
            elif stage == 'forecast':
                self.pipeline.forecast = ForecastState()

        logger.info(f"Cleared state from stage '{from_stage}' onwards")

    def get_pipeline_progress(self) -> Dict[str, bool]:
        """Get completion status of each pipeline stage"""
        return {
            "data_loaded": self.pipeline.data.is_complete(),
            "data_merged": self.pipeline.prepared.merged_data is not None,
            "data_processed": self.pipeline.prepared.processed_df is not None,
            "features_engineered": self.pipeline.feature_engineering.X_engineered is not None,
            "features_selected": self.pipeline.feature_selection.selected_features is not None,
            "model_trained": self.pipeline.modeling.ml_results is not None,
            "forecast_generated": self.pipeline.forecast.forecasts is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export state summary as dictionary (for debugging/logging)"""
        p = self.pipeline
        return {
            "data": {
                "patient": p.data.patient_data.shape if p.data.patient_data is not None else None,
                "weather": p.data.weather_data.shape if p.data.weather_data is not None else None,
                "calendar": p.data.calendar_data.shape if p.data.calendar_data is not None else None,
                "reason": p.data.reason_data.shape if p.data.reason_data is not None else None,
            },
            "prepared": {
                "merged": p.prepared.merged_data.shape if p.prepared.merged_data is not None else None,
                "processed": p.prepared.processed_df.shape if p.prepared.processed_df is not None else None,
            },
            "feature_engineering": {
                "train_size": len(p.feature_engineering.train_idx) if p.feature_engineering.train_idx is not None else None,
                "cal_size": len(p.feature_engineering.cal_idx) if p.feature_engineering.cal_idx is not None else None,
                "test_size": len(p.feature_engineering.test_idx) if p.feature_engineering.test_idx is not None else None,
            },
            "progress": self.get_pipeline_progress(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_state_manager() -> StateManager:
    """Get the singleton StateManager instance"""
    return StateManager()


def get_pipeline_state() -> PipelineState:
    """Get the current PipelineState directly"""
    return get_state_manager().pipeline
