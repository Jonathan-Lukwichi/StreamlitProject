"""
Classification module for patient arrival reason prediction.

This module provides enterprise-grade classification models including:
- XGBoost Classifier
- Artificial Neural Network (ANN)
- LSTM for text/sequence features

Designed for modular integration with Streamlit apps.
"""

from .config import ClassificationConfig
from .features import FeatureEngineer
from .evaluation import ModelEvaluator

__all__ = [
    'ClassificationConfig',
    'FeatureEngineer',
    'ModelEvaluator',
]
