from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class TrainConfig:
    target_col: str
    feature_cols: List[str]
    split_ratio: float = 0.8
    date_col: Optional[str] = "Date"
    n_splits: int = 3
    random_state: int = 42

@dataclass
class MLSpec:
    name: str
    params: Dict[str, Any]

@dataclass
class MLArtifacts:
    spec: MLSpec
    cfg: TrainConfig
    model: Any               # fitted model (Keras/Pipeline/etc.)
    transformer: Any         # scaler/pipeline
    feature_names: List[str]
    metrics: Dict[str, float]
    y_train_pred: Optional[pd.Series]
    y_test_pred: Optional[pd.Series]
    y_train_actual: Optional[pd.Series]
    y_test_actual: Optional[pd.Series]
    importances: Optional[Dict[str, float]]  # permutation or feature importance

@dataclass
class FitArtifacts:
    model_path: str
    scaler_path: str | None
    config: Dict[str, Any]
    metrics: Dict[str, float]

class BaseTimeSeriesModel:
    name: str
    def fit(self, df, target: str, config: Dict[str,Any]) -> FitArtifacts: ...
    def predict(self, df_future): ...

@dataclass
class HybridArtifacts:
    stage1_path: str
    stage2_path: str
    scaler_path: str | None
    config: Dict[str, Any]
    metrics: Dict[str, float]

class BaseHybridModel:
    name: str
    def fit(self, df, target: str, config: Dict[str,Any]) -> HybridArtifacts: ...
    def predict(self, df_future): ...  # return final forecast and each stage
