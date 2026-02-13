# =============================================================================
# app_core/services/model_storage_service.py
# Unified Model Storage and Loading Service
# Handles persistence of all model types to pipeline_artifacts/
# =============================================================================

from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd

from app_core.services.base_service import BaseService, ServiceResult


@dataclass
class ModelArtifactMetadata:
    """Metadata for a saved model artifact."""
    model_type: str                              # "ARIMA", "SARIMAX", "LSTM", "XGBoost", etc.
    horizons: List[int] = field(default_factory=list)  # [1, 2, 3, 4, 5, 6, 7]
    saved_at: str = ""                           # ISO timestamp
    config: Dict[str, Any] = field(default_factory=dict)  # Model configuration
    metrics: Dict[str, float] = field(default_factory=dict)  # Performance metrics
    session_state_key: str = ""                  # Key to restore to session state
    file_manifest: Dict[str, str] = field(default_factory=dict)  # {component: filepath}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_type": self.model_type,
            "horizons": self.horizons,
            "saved_at": self.saved_at,
            "config": self.config,
            "metrics": self.metrics,
            "session_state_key": self.session_state_key,
            "file_manifest": self.file_manifest,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelArtifactMetadata:
        """Create from dictionary."""
        return cls(
            model_type=data.get("model_type", "Unknown"),
            horizons=data.get("horizons", []),
            saved_at=data.get("saved_at", ""),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            session_state_key=data.get("session_state_key", ""),
            file_manifest=data.get("file_manifest", {}),
        )


class ModelStorageService(BaseService):
    """
    Unified service for saving and loading model artifacts.

    Supports:
    - ARIMA / SARIMAX (statsmodels)
    - XGBoost (sklearn-compatible)
    - LSTM / ANN (Keras .keras format)
    - Hybrid models (LSTM-XGBoost, LSTM-SARIMAX, LSTM-ANN)
    - Optimized models (from Optuna/Grid search)
    """

    ARTIFACT_BASE = "pipeline_artifacts"
    METADATA_FILE = "results_metadata.json"
    METRICS_FILE = "metrics.csv"
    CONFIG_FILE = "model_config.json"

    # Session state key mapping
    SESSION_KEYS = {
        "ARIMA": "arima_mh_results",
        "SARIMAX": "sarimax_results",
        "XGBOOST": "ml_mh_results_xgboost",
        "LSTM": "ml_mh_results_lstm",
        "ANN": "ml_mh_results_ann",
        "LSTM_XGB": "lstm_xgb_results",
        "LSTM_SARIMAX": "lstm_sarimax_results",
        "LSTM_ANN": "lstm_ann_results",
        "ENSEMBLE": "ensemble_results",
        "STACKING": "stacking_results",
    }

    def __init__(self, artifact_dir: str = None):
        super().__init__()
        self.artifact_dir = artifact_dir or self.ARTIFACT_BASE

    def _get_model_dir(self, model_type: str, subdir: str = None) -> str:
        """Get directory path for a model type."""
        if subdir:
            return os.path.join(self.artifact_dir, subdir, model_type.lower())
        return os.path.join(self.artifact_dir, model_type.lower())

    def _get_session_key(self, model_type: str) -> str:
        """Map model type to session state key."""
        return self.SESSION_KEYS.get(
            model_type.upper().replace("-", "_").replace(" ", "_"),
            f"ml_results_{model_type.lower()}"
        )

    def _ensure_dir(self, path: str) -> None:
        """Ensure directory exists."""
        os.makedirs(path, exist_ok=True)

    def _save_metadata(
        self,
        model_dir: str,
        model_type: str,
        horizons: List[int],
        config: dict,
        metrics: dict,
        file_manifest: dict
    ) -> str:
        """Save comprehensive metadata for model restoration."""
        metadata = ModelArtifactMetadata(
            model_type=model_type,
            horizons=horizons,
            saved_at=datetime.now().isoformat(),
            config=config,
            metrics=metrics,
            session_state_key=self._get_session_key(model_type),
            file_manifest=file_manifest,
        )

        metadata_path = os.path.join(model_dir, self.METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        return metadata_path

    def _save_metrics_csv(self, model_dir: str, results_df: pd.DataFrame) -> Optional[str]:
        """Save metrics DataFrame to CSV."""
        if results_df is None or results_df.empty:
            return None

        metrics_path = os.path.join(model_dir, self.METRICS_FILE)
        results_df.to_csv(metrics_path, index=False)
        return metrics_path

    def _extract_metrics_summary(self, results_df: pd.DataFrame) -> dict:
        """Extract summary metrics from results DataFrame."""
        if results_df is None or results_df.empty:
            return {}

        summary = {}
        # Try different column naming conventions
        for metric, cols in [
            ("MAE", ["Test_MAE", "MAE"]),
            ("RMSE", ["Test_RMSE", "RMSE"]),
            ("MAPE_%", ["Test_MAPE", "MAPE_%", "MAPE"]),
            ("Accuracy_%", ["Test_Acc", "Accuracy_%", "Accuracy"]),
        ]:
            for col in cols:
                if col in results_df.columns:
                    val = results_df[col].mean()
                    if np.isfinite(val):
                        summary[metric] = float(val)
                    break

        return summary

    # =========================================================================
    # SAVE METHODS
    # =========================================================================

    def save_arima_results(self, results: dict) -> ServiceResult:
        """
        Save ARIMA results to disk.

        Handles both structures:
        1. Multi-horizon (F matrix): {F, L, U, test_eval, metrics_df, res, order, ...}
        2. Per-horizon: {per_h: {1: {res: ..., order: ...}, 2: {...}}, results_df, ...}
        """
        import joblib

        model_dir = self._get_model_dir("arima")
        self._ensure_dir(model_dir)
        saved_paths = {}
        horizons_saved = []

        try:
            # Save config
            config = {
                "model_type": "ARIMA",
                "order": results.get("order"),
                "max_horizon": results.get("max_horizon", 7),
            }
            config_path = os.path.join(model_dir, self.CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            saved_paths["config"] = config_path

            # Handle multi-horizon (F matrix) structure
            if "F" in results and results.get("res") is not None:
                res = results["res"]
                model_path = os.path.join(model_dir, "fitted_model.pkl")
                joblib.dump(res, model_path)
                saved_paths["fitted_model"] = model_path
                horizons_saved = list(range(1, results.get("max_horizon", 7) + 1))
                self.logger.info(f"Saved ARIMA fitted model to {model_path}")

            # Handle per_h structure (legacy + new)
            elif "per_h" in results:
                per_h = results["per_h"]
                for h, h_data in per_h.items():
                    # Try multiple keys for fitted model
                    res = h_data.get("res") or h_data.get("fitted_model") or h_data.get("model")
                    if res is not None:
                        model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                        joblib.dump(res, model_path)
                        saved_paths[f"horizon_{h}"] = model_path
                        horizons_saved.append(int(h))

                        # Update config with per-horizon order if available
                        if h_data.get("order") and "order" not in config:
                            config["order"] = h_data.get("order")

            # Save metrics
            results_df = results.get("results_df")
            metrics_path = self._save_metrics_csv(model_dir, results_df)
            if metrics_path:
                saved_paths["metrics"] = metrics_path

            # Save metadata
            metrics_summary = self._extract_metrics_summary(results_df)
            self._save_metadata(
                model_dir, "ARIMA", horizons_saved, config, metrics_summary, saved_paths
            )

            return ServiceResult.ok(saved_paths, metadata={"horizons": horizons_saved})

        except Exception as e:
            self.logger.error(f"Failed to save ARIMA results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "SAVE_FAILED")

    def save_sarimax_results(self, results: dict) -> ServiceResult:
        """Save SARIMAX results to disk."""
        import joblib

        model_dir = self._get_model_dir("sarimax")
        self._ensure_dir(model_dir)
        saved_paths = {}
        horizons_saved = []

        try:
            # Save config
            config = {
                "model_type": "SARIMAX",
                "order": results.get("order"),
                "seasonal_order": results.get("seasonal_order"),
                "max_horizon": results.get("max_horizon", 7),
            }
            config_path = os.path.join(model_dir, self.CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            saved_paths["config"] = config_path

            # Handle multi-horizon (F matrix) structure
            if "F" in results and results.get("res") is not None:
                res = results["res"]
                model_path = os.path.join(model_dir, "fitted_model.pkl")
                joblib.dump(res, model_path)
                saved_paths["fitted_model"] = model_path
                horizons_saved = list(range(1, results.get("max_horizon", 7) + 1))

            # Handle per_h structure
            elif "per_h" in results:
                per_h = results["per_h"]
                for h, h_data in per_h.items():
                    res = h_data.get("res") or h_data.get("fitted_model") or h_data.get("model")
                    if res is not None:
                        model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                        joblib.dump(res, model_path)
                        saved_paths[f"horizon_{h}"] = model_path
                        horizons_saved.append(int(h))

            # Save metrics
            results_df = results.get("results_df")
            metrics_path = self._save_metrics_csv(model_dir, results_df)
            if metrics_path:
                saved_paths["metrics"] = metrics_path

            # Save metadata
            metrics_summary = self._extract_metrics_summary(results_df)
            self._save_metadata(
                model_dir, "SARIMAX", horizons_saved, config, metrics_summary, saved_paths
            )

            return ServiceResult.ok(saved_paths, metadata={"horizons": horizons_saved})

        except Exception as e:
            self.logger.error(f"Failed to save SARIMAX results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "SAVE_FAILED")

    def save_ml_results(self, model_type: str, results: dict) -> ServiceResult:
        """
        Save ML model results (XGBoost, LSTM, ANN) to disk.

        Handles Keras models (.keras) and sklearn-compatible models (.pkl) appropriately.
        """
        import joblib

        model_type_upper = model_type.upper()
        model_dir = self._get_model_dir(model_type)
        self._ensure_dir(model_dir)
        saved_paths = {}
        horizons_saved = []
        is_keras = model_type_upper in ["LSTM", "ANN"]

        try:
            pipelines = results.get("pipelines", {})

            for h, pipeline in pipelines.items():
                if pipeline is None:
                    continue

                h_int = int(h)
                horizons_saved.append(h_int)

                if is_keras:
                    # Keras models - save model and scaler separately
                    model_path = os.path.join(model_dir, f"horizon_{h}.keras")
                    if hasattr(pipeline, 'model') and pipeline.model is not None:
                        pipeline.model.save(model_path)
                        saved_paths[f"horizon_{h}"] = model_path

                        # Save scaler if present
                        scaler = getattr(pipeline, 'scaler_X', None) or getattr(pipeline, 'scaler', None)
                        if scaler is not None:
                            scaler_path = os.path.join(model_dir, f"scaler_{h}.pkl")
                            joblib.dump(scaler, scaler_path)
                            saved_paths[f"scaler_{h}"] = scaler_path
                else:
                    # XGBoost / sklearn - use joblib or pipeline's save method
                    model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                    if hasattr(pipeline, 'save'):
                        pipeline.save(model_path)
                    else:
                        joblib.dump(pipeline, model_path)
                    saved_paths[f"horizon_{h}"] = model_path

            # Save metrics
            results_df = results.get("results_df")
            metrics_path = self._save_metrics_csv(model_dir, results_df)
            if metrics_path:
                saved_paths["metrics"] = metrics_path

            # Save config
            config = {
                "model_type": model_type_upper,
                "horizons": horizons_saved,
                "is_keras": is_keras,
            }
            config_path = os.path.join(model_dir, self.CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            saved_paths["config"] = config_path

            # Save metadata
            metrics_summary = self._extract_metrics_summary(results_df)
            self._save_metadata(
                model_dir, model_type_upper, horizons_saved, config, metrics_summary, saved_paths
            )

            self.logger.info(f"Saved {model_type} model ({len(horizons_saved)} horizons) to {model_dir}")
            return ServiceResult.ok(saved_paths, metadata={"horizons": horizons_saved})

        except Exception as e:
            self.logger.error(f"Failed to save {model_type} results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "SAVE_FAILED")

    def save_hybrid_results(
        self,
        hybrid_type: str,
        artifacts: Any,
        config: dict = None
    ) -> ServiceResult:
        """
        Save hybrid model results (LSTM-XGBoost, LSTM-SARIMAX, LSTM-ANN).

        Hybrid models save files during fit(), this method verifies and adds metadata.
        """
        import joblib

        model_dir = self._get_model_dir(hybrid_type, subdir="hybrids")
        self._ensure_dir(model_dir)
        saved_paths = {}

        try:
            # Verify and record stage1 path
            if hasattr(artifacts, 'stage1_path') and artifacts.stage1_path:
                if os.path.exists(artifacts.stage1_path):
                    saved_paths["stage1_lstm"] = artifacts.stage1_path
                else:
                    self.logger.warning(f"Stage 1 LSTM not found at {artifacts.stage1_path}")

            # Verify and record stage2 path
            if hasattr(artifacts, 'stage2_path') and artifacts.stage2_path:
                if os.path.exists(artifacts.stage2_path):
                    saved_paths["stage2_model"] = artifacts.stage2_path
                else:
                    self.logger.warning(f"Stage 2 model not found at {artifacts.stage2_path}")

            # Verify and record scaler path
            if hasattr(artifacts, 'scaler_path') and artifacts.scaler_path:
                if os.path.exists(artifacts.scaler_path):
                    saved_paths["scaler"] = artifacts.scaler_path

            # Save metrics if available
            metrics = getattr(artifacts, 'metrics', {}) or {}
            if metrics:
                metrics_path = os.path.join(model_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                saved_paths["metrics"] = metrics_path

            # Save config
            config = config or getattr(artifacts, 'config', {}) or {}
            if config:
                config_path = os.path.join(model_dir, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
                saved_paths["config"] = config_path

            # Save metadata
            self._save_metadata(
                model_dir, hybrid_type.upper(), [], config, metrics, saved_paths
            )

            self.logger.info(f"Saved {hybrid_type} hybrid model to {model_dir}")
            return ServiceResult.ok(saved_paths)

        except Exception as e:
            self.logger.error(f"Failed to save {hybrid_type} results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "SAVE_FAILED")

    def save_optimized_results(
        self,
        model_name: str,
        opt_results: dict
    ) -> ServiceResult:
        """
        Save Optuna/Grid search optimized results.

        Properly handles Keras models (LSTM/ANN) vs sklearn models (XGBoost).
        """
        import joblib

        model_dir = self._get_model_dir(model_name.lower().replace(" ", "_"), subdir="optimized")
        self._ensure_dir(model_dir)
        saved_paths = {}
        horizons_saved = []

        model_name_lower = model_name.lower()
        is_keras = "lstm" in model_name_lower or "ann" in model_name_lower

        try:
            # Save best params
            best_params = opt_results.get("best_params", {})
            if best_params:
                params_path = os.path.join(model_dir, "best_params.json")
                with open(params_path, 'w') as f:
                    json.dump(best_params, f, indent=2, default=str)
                saved_paths["best_params"] = params_path

            # Save optimization metrics
            opt_metrics = {
                "best_cv_score": opt_results.get("best_cv_score"),
                "best_rmse": opt_results.get("best_rmse"),
                "best_mae": opt_results.get("best_mae"),
                "search_method": opt_results.get("search_method", "unknown"),
            }
            opt_metrics_path = os.path.join(model_dir, "optimization_metrics.json")
            with open(opt_metrics_path, 'w') as f:
                json.dump(opt_metrics, f, indent=2, default=str)
            saved_paths["optimization_metrics"] = opt_metrics_path

            # Save trained pipelines
            pipelines = opt_results.get("pipelines", {})
            if not pipelines:
                # Try per_h structure
                per_h = opt_results.get("per_h", {})
                for h, h_data in per_h.items():
                    if isinstance(h_data, dict):
                        pipelines[h] = h_data.get("pipeline")

            for h, pipeline in pipelines.items():
                if pipeline is None:
                    continue

                h_int = int(h)
                horizons_saved.append(h_int)

                if is_keras:
                    # Keras - save model directly
                    model_path = os.path.join(model_dir, f"horizon_{h}.keras")
                    if hasattr(pipeline, 'model') and pipeline.model is not None:
                        pipeline.model.save(model_path)
                        saved_paths[f"horizon_{h}"] = model_path

                        # Save scaler
                        scaler = getattr(pipeline, 'scaler_X', None) or getattr(pipeline, 'scaler', None)
                        if scaler is not None:
                            scaler_path = os.path.join(model_dir, f"scaler_{h}.pkl")
                            joblib.dump(scaler, scaler_path)
                            saved_paths[f"scaler_{h}"] = scaler_path
                else:
                    # XGBoost/sklearn - joblib
                    model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                    if hasattr(pipeline, 'save'):
                        pipeline.save(model_path)
                    else:
                        joblib.dump(pipeline, model_path)
                    saved_paths[f"horizon_{h}"] = model_path

            # Save metadata
            config = {"model_name": model_name, "is_keras": is_keras, "best_params": best_params}
            self._save_metadata(
                model_dir, f"{model_name}_optimized", horizons_saved, config, opt_metrics, saved_paths
            )

            self.logger.info(f"Saved optimized {model_name} model to {model_dir}")
            return ServiceResult.ok(saved_paths, metadata={"horizons": horizons_saved})

        except Exception as e:
            self.logger.error(f"Failed to save optimized {model_name}: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "SAVE_FAILED")

    # =========================================================================
    # LOAD METHODS
    # =========================================================================

    def load_arima_results(self) -> ServiceResult:
        """Load ARIMA results from disk."""
        import joblib

        model_dir = self._get_model_dir("arima")

        if not os.path.exists(model_dir):
            return ServiceResult.fail("No ARIMA artifacts found", "NOT_FOUND")

        results = {}

        try:
            # Load metrics
            metrics_path = os.path.join(model_dir, self.METRICS_FILE)
            if os.path.exists(metrics_path):
                results["results_df"] = pd.read_csv(metrics_path)

            # Load config
            config_path = os.path.join(model_dir, self.CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    results["order"] = tuple(config["order"]) if config.get("order") else None
                    results["max_horizon"] = config.get("max_horizon", 7)

            # Load fitted model (single model case)
            model_path = os.path.join(model_dir, "fitted_model.pkl")
            if os.path.exists(model_path):
                results["res"] = joblib.load(model_path)

            # Load per-horizon models
            per_h = {}
            for h in range(1, 8):
                horizon_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                if os.path.exists(horizon_path):
                    per_h[h] = {"res": joblib.load(horizon_path)}
            if per_h:
                results["per_h"] = per_h
                results["successful"] = list(per_h.keys())

            # Load metadata
            metadata_path = os.path.join(model_dir, self.METADATA_FILE)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    results["_metadata"] = json.load(f)

            self.logger.info(f"Loaded ARIMA results from {model_dir}")
            return ServiceResult.ok(results)

        except Exception as e:
            self.logger.error(f"Failed to load ARIMA results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "LOAD_FAILED")

    def load_sarimax_results(self) -> ServiceResult:
        """Load SARIMAX results from disk."""
        import joblib

        model_dir = self._get_model_dir("sarimax")

        if not os.path.exists(model_dir):
            return ServiceResult.fail("No SARIMAX artifacts found", "NOT_FOUND")

        results = {}

        try:
            # Load metrics
            metrics_path = os.path.join(model_dir, self.METRICS_FILE)
            if os.path.exists(metrics_path):
                results["results_df"] = pd.read_csv(metrics_path)

            # Load config
            config_path = os.path.join(model_dir, self.CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    results["order"] = tuple(config["order"]) if config.get("order") else None
                    results["seasonal_order"] = tuple(config["seasonal_order"]) if config.get("seasonal_order") else None

            # Load fitted model
            model_path = os.path.join(model_dir, "fitted_model.pkl")
            if os.path.exists(model_path):
                results["res"] = joblib.load(model_path)

            # Load per-horizon models
            per_h = {}
            for h in range(1, 8):
                horizon_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                if os.path.exists(horizon_path):
                    per_h[h] = {"res": joblib.load(horizon_path)}
            if per_h:
                results["per_h"] = per_h
                results["successful"] = list(per_h.keys())

            self.logger.info(f"Loaded SARIMAX results from {model_dir}")
            return ServiceResult.ok(results)

        except Exception as e:
            self.logger.error(f"Failed to load SARIMAX results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "LOAD_FAILED")

    def load_ml_results(self, model_type: str) -> ServiceResult:
        """Load ML model (XGBoost, LSTM, ANN) results from disk."""
        import joblib

        model_dir = self._get_model_dir(model_type)

        if not os.path.exists(model_dir):
            return ServiceResult.fail(f"No {model_type} artifacts found", "NOT_FOUND")

        model_type_upper = model_type.upper()
        is_keras = model_type_upper in ["LSTM", "ANN"]
        results = {"pipelines": {}, "per_h": {}}

        try:
            # Load per-horizon models
            for h in range(1, 8):
                if is_keras:
                    model_path = os.path.join(model_dir, f"horizon_{h}.keras")
                    scaler_path = os.path.join(model_dir, f"scaler_{h}.pkl")
                else:
                    model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                    scaler_path = None

                if os.path.exists(model_path):
                    try:
                        if is_keras:
                            from tensorflow.keras.models import load_model
                            model = load_model(model_path)
                            scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
                            results["pipelines"][h] = {"model": model, "scaler": scaler}
                        else:
                            results["pipelines"][h] = joblib.load(model_path)
                    except Exception as load_err:
                        self.logger.warning(f"Could not load {model_type} horizon {h}: {load_err}")

            # Load metrics
            metrics_path = os.path.join(model_dir, self.METRICS_FILE)
            if os.path.exists(metrics_path):
                results["results_df"] = pd.read_csv(metrics_path)

            results["successful"] = list(results["pipelines"].keys())
            results["model_type"] = model_type_upper

            self.logger.info(f"Loaded {model_type} results from {model_dir}")
            return ServiceResult.ok(results)

        except Exception as e:
            self.logger.error(f"Failed to load {model_type} results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "LOAD_FAILED")

    def load_hybrid_results(self, hybrid_type: str) -> ServiceResult:
        """Load hybrid model (LSTM-XGBoost, LSTM-SARIMAX, LSTM-ANN) from disk."""
        import joblib

        model_dir = self._get_model_dir(hybrid_type, subdir="hybrids")

        if not os.path.exists(model_dir):
            return ServiceResult.fail(f"No {hybrid_type} artifacts found", "NOT_FOUND")

        results = {}

        try:
            # Load Stage 1 LSTM
            stage1_path = os.path.join(model_dir, "stage1_lstm.keras")
            if os.path.exists(stage1_path):
                from tensorflow.keras.models import load_model
                results["stage1_model"] = load_model(stage1_path)

            # Load Stage 2 model (try multiple formats)
            stage2_patterns = [
                ("stage2_xgb.json", "xgboost_json"),
                ("stage2_xgb.pkl", "pkl"),
                ("stage2_sarimax.pkl", "pkl"),
                ("stage2_ann.keras", "keras"),
            ]
            for pattern, format_type in stage2_patterns:
                stage2_path = os.path.join(model_dir, pattern)
                if os.path.exists(stage2_path):
                    if format_type == "xgboost_json":
                        from xgboost import XGBRegressor
                        model = XGBRegressor()
                        model.load_model(stage2_path)
                        results["stage2_model"] = model
                    elif format_type == "keras":
                        from tensorflow.keras.models import load_model
                        results["stage2_model"] = load_model(stage2_path)
                    else:
                        results["stage2_model"] = joblib.load(stage2_path)
                    break

            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                results["scaler"] = joblib.load(scaler_path)

            # Load config
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    results["config"] = json.load(f)

            # Load metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    results["metrics"] = json.load(f)

            self.logger.info(f"Loaded {hybrid_type} hybrid from {model_dir}")
            return ServiceResult.ok(results)

        except Exception as e:
            self.logger.error(f"Failed to load {hybrid_type} results: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "LOAD_FAILED")

    def load_optimized_results(self, model_name: str) -> ServiceResult:
        """Load optimized model results from disk."""
        import joblib

        model_dir = self._get_model_dir(model_name.lower().replace(" ", "_"), subdir="optimized")

        if not os.path.exists(model_dir):
            return ServiceResult.fail(f"No optimized {model_name} artifacts found", "NOT_FOUND")

        model_name_lower = model_name.lower()
        is_keras = "lstm" in model_name_lower or "ann" in model_name_lower
        results = {"pipelines": {}}

        try:
            # Load best params
            params_path = os.path.join(model_dir, "best_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    results["best_params"] = json.load(f)

            # Load optimization metrics
            opt_metrics_path = os.path.join(model_dir, "optimization_metrics.json")
            if os.path.exists(opt_metrics_path):
                with open(opt_metrics_path, 'r') as f:
                    results["best_metrics"] = json.load(f)

            # Load models
            for h in range(1, 8):
                if is_keras:
                    model_path = os.path.join(model_dir, f"horizon_{h}.keras")
                    scaler_path = os.path.join(model_dir, f"scaler_{h}.pkl")
                else:
                    model_path = os.path.join(model_dir, f"horizon_{h}.pkl")
                    scaler_path = None

                if os.path.exists(model_path):
                    try:
                        if is_keras:
                            from tensorflow.keras.models import load_model
                            model = load_model(model_path)
                            scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
                            results["pipelines"][h] = {"model": model, "scaler": scaler}
                        else:
                            results["pipelines"][h] = joblib.load(model_path)
                    except Exception as load_err:
                        self.logger.warning(f"Could not load optimized {model_name} horizon {h}: {load_err}")

            self.logger.info(f"Loaded optimized {model_name} from {model_dir}")
            return ServiceResult.ok(results)

        except Exception as e:
            self.logger.error(f"Failed to load optimized {model_name}: {e}", exc_info=True)
            return ServiceResult.fail(str(e), "LOAD_FAILED")

    # =========================================================================
    # DISCOVERY METHODS
    # =========================================================================

    def list_available_models(self) -> List[ModelArtifactMetadata]:
        """List all saved model artifacts with their metadata."""
        available = []

        if not os.path.exists(self.artifact_dir):
            return available

        # Check main model directories
        for model_type in ["arima", "sarimax", "xgboost", "lstm", "ann"]:
            model_dir = self._get_model_dir(model_type)
            metadata_path = os.path.join(model_dir, self.METADATA_FILE)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = ModelArtifactMetadata.from_dict(json.load(f))
                        available.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Could not load metadata for {model_type}: {e}")

        # Check hybrids
        hybrids_dir = os.path.join(self.artifact_dir, "hybrids")
        if os.path.exists(hybrids_dir):
            for hybrid_name in os.listdir(hybrids_dir):
                metadata_path = os.path.join(hybrids_dir, hybrid_name, self.METADATA_FILE)
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = ModelArtifactMetadata.from_dict(json.load(f))
                            available.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata for hybrid {hybrid_name}: {e}")

        # Check optimized
        optimized_dir = os.path.join(self.artifact_dir, "optimized")
        if os.path.exists(optimized_dir):
            for opt_name in os.listdir(optimized_dir):
                metadata_path = os.path.join(optimized_dir, opt_name, self.METADATA_FILE)
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = ModelArtifactMetadata.from_dict(json.load(f))
                            available.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata for optimized {opt_name}: {e}")

        return available

    def has_saved_model(self, model_type: str) -> bool:
        """Check if a specific model type has saved artifacts."""
        model_dir = self._get_model_dir(model_type)
        metadata_path = os.path.join(model_dir, self.METADATA_FILE)
        return os.path.exists(metadata_path)

    def get_model_metadata(self, model_type: str) -> Optional[ModelArtifactMetadata]:
        """Get metadata for a specific model type."""
        model_dir = self._get_model_dir(model_type)
        metadata_path = os.path.join(model_dir, self.METADATA_FILE)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, 'r') as f:
                return ModelArtifactMetadata.from_dict(json.load(f))
        except Exception:
            return None

    # =========================================================================
    # SESSION STATE INTEGRATION
    # =========================================================================

    def restore_to_session_state(self, model_type: str) -> ServiceResult:
        """
        Load a model and restore it to Streamlit session state.

        Returns:
            ServiceResult with session_key in metadata if successful
        """
        import streamlit as st

        model_type_upper = model_type.upper().replace("-", "_").replace(" ", "_")

        # Determine which load function to use
        load_map = {
            "ARIMA": self.load_arima_results,
            "SARIMAX": self.load_sarimax_results,
            "XGBOOST": lambda: self.load_ml_results("XGBoost"),
            "LSTM": lambda: self.load_ml_results("LSTM"),
            "ANN": lambda: self.load_ml_results("ANN"),
            "LSTM_XGB": lambda: self.load_hybrid_results("lstm_xgb"),
            "LSTM_SARIMAX": lambda: self.load_hybrid_results("lstm_sarimax"),
            "LSTM_ANN": lambda: self.load_hybrid_results("lstm_ann"),
        }

        load_fn = load_map.get(model_type_upper)
        if not load_fn:
            return ServiceResult.fail(f"Unknown model type: {model_type}", "UNKNOWN_TYPE")

        result = load_fn()
        if not result.success:
            return result

        session_key = self._get_session_key(model_type)
        st.session_state[session_key] = result.data

        self.logger.info(f"Restored {model_type} to session state key: {session_key}")
        return ServiceResult.ok(
            result.data,
            metadata={"session_key": session_key, "loaded_at": datetime.now().isoformat()}
        )

    def restore_all_available(self) -> Dict[str, ServiceResult]:
        """Restore all available saved models to session state."""
        results = {}
        available = self.list_available_models()

        for metadata in available:
            result = self.restore_to_session_state(metadata.model_type)
            results[metadata.model_type] = result

        return results


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_storage_service: Optional[ModelStorageService] = None


def get_storage_service() -> ModelStorageService:
    """Get singleton instance of ModelStorageService."""
    global _storage_service
    if _storage_service is None:
        _storage_service = ModelStorageService()
    return _storage_service
