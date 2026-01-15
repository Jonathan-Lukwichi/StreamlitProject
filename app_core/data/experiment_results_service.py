# =============================================================================
# app_core/data/experiment_results_service.py
# Service for managing model experiment results with dataset tracking.
# Stores per-dataset model results for comparison and analysis.
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app_core.data.supabase_client import get_cached_supabase_client
from app_core.data.results_storage_service import NpEncoder, decode_special_types


# =============================================================================
# EXPERIMENT RECORD DATACLASS
# =============================================================================

@dataclass
class ExperimentRecord:
    """
    Single experiment result record.

    Represents a complete model training experiment with:
    - Dataset identification (feature selection method + date + feature count)
    - Model information and hyperparameters
    - Per-horizon performance metrics
    - Full predictions (actual vs predicted values)
    """
    # Dataset Identification
    dataset_id: str                     # Composite: "{method_slug}_{date}_{feature_count}f"
    feature_selection_method: str       # Method name (e.g., "Permutation Importance")
    feature_engineering_variant: str    # "Variant A" or "Variant B"
    feature_count: int                  # Number of features selected
    feature_names: List[str]            # Actual feature names used

    # Model Information
    model_name: str                     # Model identifier (e.g., "XGBoost", "LSTM")
    model_category: str                 # "Statistical", "ML", "Hybrid"
    model_params: Dict[str, Any]        # Hyperparameters used

    # Per-Horizon Metrics
    horizon_metrics: List[Dict]         # [{horizon, mae, rmse, mape, accuracy, r2}, ...]

    # Aggregated Metrics (averages across all horizons)
    avg_mae: float
    avg_rmse: float
    avg_mape: float
    avg_accuracy: float
    avg_r2: Optional[float]
    runtime_seconds: float

    # Predictions (full arrays per horizon)
    predictions: Dict[int, Dict]        # {horizon: {actual: [...], predicted: [...]}}

    # Metadata
    trained_at: datetime = field(default_factory=datetime.now)
    horizons_trained: List[int] = field(default_factory=list)
    train_size: int = 0
    test_size: int = 0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        if isinstance(data.get('trained_at'), datetime):
            data['trained_at'] = data['trained_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        """Create record from dictionary."""
        # Convert ISO string back to datetime
        if isinstance(data.get('trained_at'), str):
            data['trained_at'] = datetime.fromisoformat(data['trained_at'])

        # Handle predictions key conversion (JSON keys are always strings)
        if 'predictions' in data and data['predictions']:
            predictions = {}
            for k, v in data['predictions'].items():
                predictions[int(k) if isinstance(k, str) else k] = v
            data['predictions'] = predictions

        return cls(**data)


# =============================================================================
# MODEL CATEGORY MAPPING
# =============================================================================

MODEL_CATEGORIES = {
    # Statistical Models
    "ARIMA": "Statistical",
    "SARIMAX": "Statistical",

    # Machine Learning Models
    "XGBoost": "ML",
    "LSTM": "ML",
    "ANN": "ML",

    # Hybrid Models
    "LSTM+XGBoost": "Hybrid",
    "LSTM+SARIMAX": "Hybrid",
    "LSTM+ANN": "Hybrid",
}


def get_model_category(model_name: str) -> str:
    """Get category for a model name."""
    return MODEL_CATEGORIES.get(model_name, "Unknown")


# =============================================================================
# EXPERIMENT RESULTS SERVICE
# =============================================================================

class ExperimentResultsService:
    """
    Service for managing model experiment results in Supabase.

    Provides:
    - CRUD operations for experiment records
    - Dataset identification generation
    - Cross-dataset comparison and analysis
    - CSV export functionality

    Usage:
        service = get_experiment_service()

        # Save experiment
        record = ExperimentRecord(...)
        service.save_experiment(record)

        # Compare across datasets
        comparison_df = service.compare_across_datasets("XGBoost")
    """

    TABLE_NAME = "model_experiment_results"

    def __init__(self, client=None):
        """Initialize with Supabase client."""
        self.client = client or get_cached_supabase_client()

    def is_connected(self) -> bool:
        """Check if Supabase client is available."""
        return self.client is not None

    def _get_user_id(self) -> str:
        """
        Get current user ID.
        Uses authenticated username for persistence across sessions.
        """
        username = st.session_state.get("username")
        if username:
            return f"user_{username}"

        # Fallback to session-based ID
        if "user_session_id" not in st.session_state:
            import uuid
            st.session_state["user_session_id"] = str(uuid.uuid4())[:12]
        return st.session_state["user_session_id"]

    # =========================================================================
    # DATASET IDENTIFICATION
    # =========================================================================

    @staticmethod
    def generate_dataset_id(
        method: str,
        date: datetime,
        feature_count: int
    ) -> str:
        """
        Generate unique dataset identifier.

        Format: {method_slug}_{YYYYMMDD_HHMM}_{feature_count}f

        Args:
            method: Feature selection method name
            date: Timestamp of dataset creation
            feature_count: Number of features in dataset

        Returns:
            Unique dataset identifier string
        """
        date_str = date.strftime("%Y%m%d_%H%M")
        method_slug = method.lower().replace(" ", "_").replace("-", "_")[:20]
        return f"{method_slug}_{date_str}_{feature_count}f"

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def save_experiment(self, record: ExperimentRecord) -> Tuple[bool, str]:
        """
        Save or update an experiment result.

        Uses upsert logic: if record with same (user_id, dataset_id, model_name)
        exists, it will be updated.

        Args:
            record: ExperimentRecord to save

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_connected():
            return False, "Supabase not connected"

        user_id = self._get_user_id()

        try:
            # Convert record to dict and serialize
            record_dict = record.to_dict()

            # Prepare database record
            db_record = {
                "user_id": user_id,
                "dataset_id": record.dataset_id,
                "feature_selection_method": record.feature_selection_method,
                "feature_engineering_variant": record.feature_engineering_variant,
                "feature_count": record.feature_count,
                "feature_names": json.loads(json.dumps(record.feature_names, cls=NpEncoder)),
                "model_name": record.model_name,
                "model_category": record.model_category,
                "model_params": json.loads(json.dumps(record.model_params, cls=NpEncoder)),
                "horizon_metrics": json.loads(json.dumps(record.horizon_metrics, cls=NpEncoder)),
                "avg_mae": float(record.avg_mae) if record.avg_mae is not None else None,
                "avg_rmse": float(record.avg_rmse) if record.avg_rmse is not None else None,
                "avg_mape": float(record.avg_mape) if record.avg_mape is not None else None,
                "avg_accuracy": float(record.avg_accuracy) if record.avg_accuracy is not None else None,
                "avg_r2": float(record.avg_r2) if record.avg_r2 is not None else None,
                "runtime_seconds": float(record.runtime_seconds) if record.runtime_seconds is not None else None,
                "predictions": json.loads(json.dumps(record.predictions, cls=NpEncoder)),
                "trained_at": record.trained_at.isoformat() if isinstance(record.trained_at, datetime) else record.trained_at,
                "horizons_trained": record.horizons_trained,
                "train_size": record.train_size,
                "test_size": record.test_size,
                "notes": record.notes,
            }

            # Delete existing record first (upsert workaround)
            self.client.table(self.TABLE_NAME).delete().eq(
                "user_id", user_id
            ).eq(
                "dataset_id", record.dataset_id
            ).eq(
                "model_name", record.model_name
            ).execute()

            # Insert new record
            self.client.table(self.TABLE_NAME).insert(db_record).execute()

            return True, f"Saved {record.model_name} for dataset {record.dataset_id}"

        except Exception as e:
            return False, f"Save failed: {str(e)}"

    def load_experiment(
        self,
        dataset_id: str,
        model_name: str
    ) -> Optional[ExperimentRecord]:
        """
        Load a specific experiment result.

        Args:
            dataset_id: Dataset identifier
            model_name: Model name

        Returns:
            ExperimentRecord or None if not found
        """
        if not self.is_connected():
            return None

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("dataset_id", dataset_id) \
                .eq("model_name", model_name) \
                .limit(1) \
                .execute()

            if not response.data:
                return None

            return self._record_from_db(response.data[0])

        except Exception as e:
            print(f"Load error: {e}")
            return None

    def load_all_for_dataset(self, dataset_id: str) -> List[ExperimentRecord]:
        """
        Load all model results for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of ExperimentRecords
        """
        if not self.is_connected():
            return []

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("dataset_id", dataset_id) \
                .execute()

            return [self._record_from_db(r) for r in response.data or []]

        except Exception as e:
            print(f"Load all for dataset error: {e}")
            return []

    def load_all_for_model(self, model_name: str) -> List[ExperimentRecord]:
        """
        Load all results for a specific model across datasets.

        Args:
            model_name: Model name

        Returns:
            List of ExperimentRecords
        """
        if not self.is_connected():
            return []

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("model_name", model_name) \
                .order("trained_at", desc=True) \
                .execute()

            return [self._record_from_db(r) for r in response.data or []]

        except Exception as e:
            print(f"Load all for model error: {e}")
            return []

    def list_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        List all experiments with optional filters.

        Args:
            filters: Optional dict with filter criteria:
                - model_name: str or list of model names
                - feature_selection_method: str or list
                - dataset_id: str

        Returns:
            DataFrame with experiment summaries
        """
        if not self.is_connected():
            return pd.DataFrame()

        user_id = self._get_user_id()

        try:
            query = self.client.table(self.TABLE_NAME) \
                .select(
                    "dataset_id, model_name, feature_selection_method, "
                    "feature_engineering_variant, feature_count, "
                    "avg_mae, avg_rmse, avg_mape, avg_accuracy, avg_r2, "
                    "runtime_seconds, trained_at, horizons_trained, notes"
                ) \
                .eq("user_id", user_id)

            # Apply filters
            if filters:
                if 'model_name' in filters:
                    if isinstance(filters['model_name'], list):
                        query = query.in_("model_name", filters['model_name'])
                    else:
                        query = query.eq("model_name", filters['model_name'])

                if 'feature_selection_method' in filters:
                    if isinstance(filters['feature_selection_method'], list):
                        query = query.in_("feature_selection_method", filters['feature_selection_method'])
                    else:
                        query = query.eq("feature_selection_method", filters['feature_selection_method'])

                if 'dataset_id' in filters:
                    query = query.eq("dataset_id", filters['dataset_id'])

            response = query.order("trained_at", desc=True).execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)

            # Convert trained_at to datetime
            if 'trained_at' in df.columns:
                df['trained_at'] = pd.to_datetime(df['trained_at'])

            return df

        except Exception as e:
            print(f"List experiments error: {e}")
            return pd.DataFrame()

    def delete_experiment(self, dataset_id: str, model_name: str) -> bool:
        """
        Delete a specific experiment.

        Args:
            dataset_id: Dataset identifier
            model_name: Model name

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("dataset_id", dataset_id) \
                .eq("model_name", model_name) \
                .execute()
            return True
        except Exception:
            return False

    def delete_all_for_dataset(self, dataset_id: str) -> bool:
        """Delete all experiments for a dataset."""
        if not self.is_connected():
            return False

        user_id = self._get_user_id()

        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("dataset_id", dataset_id) \
                .execute()
            return True
        except Exception:
            return False

    # =========================================================================
    # COMPARISON & ANALYSIS
    # =========================================================================

    def compare_across_datasets(
        self,
        model_name: str,
        metric: str = "avg_rmse"
    ) -> pd.DataFrame:
        """
        Compare a model's performance across all datasets.

        Args:
            model_name: Model to compare
            metric: Metric to focus on (for sorting)

        Returns:
            DataFrame with comparison data
        """
        if not self.is_connected():
            return pd.DataFrame()

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select(
                    "dataset_id, feature_selection_method, feature_engineering_variant, "
                    "feature_count, avg_mae, avg_rmse, avg_mape, avg_accuracy, avg_r2, "
                    "runtime_seconds, trained_at"
                ) \
                .eq("user_id", user_id) \
                .eq("model_name", model_name) \
                .order("trained_at", desc=True) \
                .execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df['trained_at'] = pd.to_datetime(df['trained_at'])

            return df

        except Exception as e:
            print(f"Compare across datasets error: {e}")
            return pd.DataFrame()

    def compare_models_for_dataset(self, dataset_id: str) -> pd.DataFrame:
        """
        Compare all models for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataFrame with model comparison
        """
        if not self.is_connected():
            return pd.DataFrame()

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select(
                    "model_name, model_category, avg_mae, avg_rmse, avg_mape, "
                    "avg_accuracy, avg_r2, runtime_seconds, trained_at"
                ) \
                .eq("user_id", user_id) \
                .eq("dataset_id", dataset_id) \
                .order("avg_rmse") \
                .execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df['trained_at'] = pd.to_datetime(df['trained_at'])

            return df

        except Exception as e:
            print(f"Compare models for dataset error: {e}")
            return pd.DataFrame()

    def get_metric_history(
        self,
        model_name: str,
        metric: str = "avg_rmse"
    ) -> pd.DataFrame:
        """
        Get time-series of metric values for a model.

        Args:
            model_name: Model name
            metric: Metric column name

        Returns:
            DataFrame with trained_at and metric value
        """
        if not self.is_connected():
            return pd.DataFrame()

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select(f"trained_at, dataset_id, {metric}") \
                .eq("user_id", user_id) \
                .eq("model_name", model_name) \
                .order("trained_at") \
                .execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df['trained_at'] = pd.to_datetime(df['trained_at'])

            return df

        except Exception as e:
            print(f"Get metric history error: {e}")
            return pd.DataFrame()

    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column (for filter dropdowns)."""
        if not self.is_connected():
            return []

        user_id = self._get_user_id()

        try:
            response = self.client.table(self.TABLE_NAME) \
                .select(column) \
                .eq("user_id", user_id) \
                .execute()

            if not response.data:
                return []

            unique_vals = list(set(r[column] for r in response.data if r.get(column)))
            return sorted(unique_vals)

        except Exception:
            return []

    # =========================================================================
    # EXPORT FUNCTIONALITY
    # =========================================================================

    def export_to_csv(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export filtered results to CSV string.

        Args:
            filters: Optional filter criteria

        Returns:
            CSV string
        """
        df = self.list_experiments(filters)

        if df.empty:
            return ""

        # Format for export
        export_df = df.copy()

        # Convert datetime to string
        if 'trained_at' in export_df.columns:
            export_df['trained_at'] = export_df['trained_at'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convert list columns to string
        if 'horizons_trained' in export_df.columns:
            export_df['horizons_trained'] = export_df['horizons_trained'].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x)
            )

        return export_df.to_csv(index=False)

    def export_to_dataframe(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Export filtered results to DataFrame.

        Args:
            filters: Optional filter criteria

        Returns:
            DataFrame with experiment data
        """
        return self.list_experiments(filters)

    def bulk_import(self, df: pd.DataFrame) -> int:
        """
        Bulk import experiments from DataFrame.

        Args:
            df: DataFrame with experiment data

        Returns:
            Number of records successfully imported
        """
        if not self.is_connected() or df.empty:
            return 0

        success_count = 0
        user_id = self._get_user_id()

        required_cols = ['dataset_id', 'model_name', 'avg_rmse']
        if not all(col in df.columns for col in required_cols):
            return 0

        for _, row in df.iterrows():
            try:
                record = {
                    "user_id": user_id,
                    "dataset_id": str(row.get('dataset_id', '')),
                    "model_name": str(row.get('model_name', '')),
                    "feature_selection_method": str(row.get('feature_selection_method', 'Unknown')),
                    "feature_engineering_variant": str(row.get('feature_engineering_variant', 'Unknown')),
                    "feature_count": int(row.get('feature_count', 0)) if pd.notna(row.get('feature_count')) else 0,
                    "feature_names": [],
                    "model_category": str(row.get('model_category', 'Unknown')),
                    "model_params": {},
                    "horizon_metrics": [],
                    "avg_mae": float(row.get('avg_mae', 0)) if pd.notna(row.get('avg_mae')) else None,
                    "avg_rmse": float(row.get('avg_rmse', 0)) if pd.notna(row.get('avg_rmse')) else None,
                    "avg_mape": float(row.get('avg_mape', 0)) if pd.notna(row.get('avg_mape')) else None,
                    "avg_accuracy": float(row.get('avg_accuracy', 0)) if pd.notna(row.get('avg_accuracy')) else None,
                    "avg_r2": float(row.get('avg_r2', 0)) if pd.notna(row.get('avg_r2')) else None,
                    "runtime_seconds": float(row.get('runtime_seconds', 0)) if pd.notna(row.get('runtime_seconds')) else None,
                    "predictions": {},
                    "horizons_trained": [],
                    "train_size": 0,
                    "test_size": 0,
                    "notes": str(row.get('notes', '')) if pd.notna(row.get('notes')) else None,
                }

                # Delete existing and insert
                self.client.table(self.TABLE_NAME).delete().eq(
                    "user_id", user_id
                ).eq(
                    "dataset_id", record['dataset_id']
                ).eq(
                    "model_name", record['model_name']
                ).execute()

                self.client.table(self.TABLE_NAME).insert(record).execute()
                success_count += 1

            except Exception as e:
                print(f"Import error for row: {e}")
                continue

        return success_count

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _record_from_db(self, data: Dict[str, Any]) -> ExperimentRecord:
        """Convert database record to ExperimentRecord."""
        # Decode special types
        feature_names = decode_special_types(data.get('feature_names', []))
        model_params = decode_special_types(data.get('model_params', {}))
        horizon_metrics = decode_special_types(data.get('horizon_metrics', []))
        predictions = decode_special_types(data.get('predictions', {}))

        # Convert predictions keys to int
        if predictions:
            predictions = {int(k) if isinstance(k, str) else k: v for k, v in predictions.items()}

        # Parse trained_at
        trained_at = data.get('trained_at')
        if isinstance(trained_at, str):
            trained_at = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
        elif trained_at is None:
            trained_at = datetime.now()

        return ExperimentRecord(
            dataset_id=data.get('dataset_id', ''),
            feature_selection_method=data.get('feature_selection_method', 'Unknown'),
            feature_engineering_variant=data.get('feature_engineering_variant', 'Unknown'),
            feature_count=data.get('feature_count', 0),
            feature_names=feature_names if isinstance(feature_names, list) else [],
            model_name=data.get('model_name', ''),
            model_category=data.get('model_category', 'Unknown'),
            model_params=model_params if isinstance(model_params, dict) else {},
            horizon_metrics=horizon_metrics if isinstance(horizon_metrics, list) else [],
            avg_mae=data.get('avg_mae'),
            avg_rmse=data.get('avg_rmse'),
            avg_mape=data.get('avg_mape'),
            avg_accuracy=data.get('avg_accuracy'),
            avg_r2=data.get('avg_r2'),
            runtime_seconds=data.get('runtime_seconds', 0),
            predictions=predictions if isinstance(predictions, dict) else {},
            trained_at=trained_at,
            horizons_trained=data.get('horizons_trained', []),
            train_size=data.get('train_size', 0),
            test_size=data.get('test_size', 0),
            notes=data.get('notes'),
        )


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_experiment_service_instance: Optional[ExperimentResultsService] = None


def get_experiment_service() -> ExperimentResultsService:
    """Get singleton ExperimentResultsService instance."""
    global _experiment_service_instance
    if _experiment_service_instance is None:
        _experiment_service_instance = ExperimentResultsService()
    return _experiment_service_instance


def reset_experiment_service():
    """Reset singleton (useful for testing or reconnection)."""
    global _experiment_service_instance
    _experiment_service_instance = None
