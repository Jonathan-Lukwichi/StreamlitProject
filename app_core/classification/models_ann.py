"""
Artificial Neural Network (ANN) classifier for patient reason classification.

Feed-forward dense neural network for tabular classification.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .config import ClassificationConfig


class ANNClassifier:
    """
    Feed-forward Artificial Neural Network classifier.
    """

    def __init__(self, config: ClassificationConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.history = None

    def build_model(self, input_dim: int, num_classes: int) -> keras.Model:
        """
        Build feed-forward ANN architecture.

        Args:
            input_dim: Number of input features
            num_classes: Number of target classes

        Returns:
            Compiled Keras model
        """
        params = self.config.ann_params

        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for i, units in enumerate(params['hidden_layers']):
            model.add(layers.Dense(units, activation=params['activation'],
                                    name=f'dense_{i+1}'))

            if params['batch_normalization']:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

            model.add(layers.Dropout(params['dropout_rate'], name=f'dropout_{i+1}'))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train ANN classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dict
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.num_classes = len(self.label_encoder.classes_)

        # Build model
        input_dim = X_train.shape[1]
        self.model = self.build_model(input_dim, self.num_classes)

        # Prepare callbacks
        callback_list = []

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.config.ann_params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        callback_list.append(early_stop)

        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=self.config.ann_params['reduce_lr_patience'],
            verbose=0,
            min_lr=1e-7
        )
        callback_list.append(reduce_lr)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            validation_data = (X_val, y_val_encoded)

        # Train model
        history = self.model.fit(
            X_train,
            y_train_encoded,
            validation_data=validation_data,
            epochs=self.config.ann_params['epochs'],
            batch_size=self.config.ann_params['batch_size'],
            callbacks=callback_list,
            verbose=0
        )

        self.history = history.history

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        y_pred_proba = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            y_pred_encoded = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        y_pred_proba = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            # Convert to 2-class probability matrix
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])

        return y_pred_proba

    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained!")

        # Save Keras model
        model_path = filepath.replace('.pkl', '.keras')
        self.model.save(model_path)

        # Save metadata
        meta_dict = {
            'label_encoder': self.label_encoder,
            'config': self.config,
            'num_classes': self.num_classes,
            'history': self.history
        }
        joblib.dump(meta_dict, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ANNClassifier':
        """Load model from disk."""
        meta_dict = joblib.load(filepath)

        instance = cls(meta_dict['config'])

        # Load Keras model
        model_path = filepath.replace('.pkl', '.keras')
        instance.model = keras.models.load_model(model_path)

        instance.label_encoder = meta_dict['label_encoder']
        instance.num_classes = meta_dict['num_classes']
        instance.history = meta_dict.get('history')

        return instance
