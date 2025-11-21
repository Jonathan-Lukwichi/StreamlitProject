"""
XGBoost classifier for patient reason classification.

Provides a high-level interface for training and evaluating XGBoost models.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from .config import ClassificationConfig


class XGBoostClassifier:
    """
    XGBoost classifier wrapper with

 standardized interface.
    """

    def __init__(self, config: ClassificationConfig):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_classes = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels (string or numeric)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dict
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.num_classes = len(self.label_encoder.classes_)

        # Get XGBoost parameters
        xgb_params = self.config.xgb_params.copy()

        # Set objective based on number of classes
        if self.num_classes == 2:
            xgb_params['objective'] = 'binary:logistic'
            xgb_params['eval_metric'] = 'logloss'
        else:
            xgb_params['objective'] = 'multi:softmax'
            xgb_params['num_class'] = self.num_classes
            xgb_params['eval_metric'] = 'mlogloss'

        # Create and train model
        self.model = xgb.XGBClassifier(**xgb_params)

        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_encoded)]

        # Train
        self.model.fit(
            X_train,
            y_train_encoded,
            eval_set=eval_set if eval_set else None,
            verbose=False
        )

        # Return training info
        history = {
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist(),
            'n_estimators': self.model.n_estimators,
        }

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained!")

        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances))

    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained!")

        save_dict = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'num_classes': self.num_classes
        }
        joblib.dump(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostClassifier':
        """Load model from disk."""
        save_dict = joblib.load(filepath)

        instance = cls(save_dict['config'])
        instance.model = save_dict['model']
        instance.label_encoder = save_dict['label_encoder']
        instance.num_classes = save_dict['num_classes']

        return instance
