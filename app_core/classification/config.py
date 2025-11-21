"""
Configuration file for classification pipeline.

Contains all configurable parameters for data processing, feature engineering,
and model training. Designed to be reusable across different datasets.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ClassificationConfig:
    """Configuration for classification pipeline."""

    # ==================== DATA CONFIGURATION ====================
    # Target column (reason for visit)
    target_col: str = "reason_for_visit"

    # Feature columns (auto-detected if None)
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    text_cols: Optional[List[str]] = None
    datetime_cols: Optional[List[str]] = None

    # Columns to exclude from features
    exclude_cols: List[str] = field(default_factory=lambda: ['patient_id', 'visit_id'])

    # ==================== TRAIN/TEST SPLIT ====================
    test_size: float = 0.2
    val_size: float = 0.1  # From remaining training data
    random_state: int = 42
    stratify: bool = True  # Stratified split to maintain class balance

    # ==================== FEATURE ENGINEERING ====================
    # Numeric features
    numeric_impute_strategy: str = "median"  # "mean", "median", or "constant"
    numeric_scale_method: str = "standard"  # "standard", "minmax", or "robust"

    # Categorical features
    categorical_impute_value: str = "missing"
    categorical_encoding: str = "onehot"  # "onehot" or "ordinal"
    max_categories: int = 50  # Max unique values for categorical encoding

    # Text features
    text_max_features: int = 5000  # Max vocabulary size
    text_max_length: int = 100  # Max sequence length for LSTM
    text_vectorization: str = "tfidf"  # "tfidf", "count", or "embedding"

    # Datetime features
    extract_time_features: bool = True  # Hour, day, month, etc.

    # ==================== XGBOOST CONFIGURATION ====================
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    })

    # ==================== ANN CONFIGURATION ====================
    ann_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_layers': [128, 64, 32],  # Layer sizes
        'activation': 'relu',
        'dropout_rate': 0.3,
        'batch_normalization': True,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
    })

    # ==================== LSTM CONFIGURATION ====================
    lstm_params: Dict[str, Any] = field(default_factory=lambda: {
        'embedding_dim': 128,
        'lstm_units': [64, 32],  # Multiple LSTM layers
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.2,
        'bidirectional': True,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
    })

    # ==================== EVALUATION ====================
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    ])
    average_method: str = "weighted"  # "macro", "weighted", or "micro"

    # ==================== OUTPUT ====================
    save_models: bool = True
    model_dir: str = "pipeline_artifacts/classification"
    plot_style: str = "plotly"  # "plotly" or "matplotlib"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'target_col': self.target_col,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'text_cols': self.text_cols,
            'datetime_cols': self.datetime_cols,
            'exclude_cols': self.exclude_cols,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'xgb_params': self.xgb_params,
            'ann_params': self.ann_params,
            'lstm_params': self.lstm_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClassificationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def validate(self, df_columns: List[str]) -> tuple[bool, str]:
        """
        Validate configuration against dataset columns.

        Returns:
            (is_valid, error_message)
        """
        if self.target_col not in df_columns:
            return False, f"Target column '{self.target_col}' not found in dataset"

        if self.test_size <= 0 or self.test_size >= 1:
            return False, f"test_size must be between 0 and 1, got {self.test_size}"

        if self.val_size < 0 or self.val_size >= 1:
            return False, f"val_size must be between 0 and 1, got {self.val_size}"

        return True, "Configuration valid"
