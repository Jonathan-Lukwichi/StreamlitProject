"""
LSTM classifier for text-based patient reason classification.

Uses recurrent neural networks for sequence/text features.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .config import ClassificationConfig


class LSTMClassifier:
    """
    LSTM-based text classifier for patient visit reasons.
    """

    def __init__(self, config: ClassificationConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.history = None

    def prepare_text(self, texts: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Tokenize and pad text sequences.

        Args:
            texts: Array of text strings
            fit: Whether to fit tokenizer (True for training data)

        Returns:
            Padded sequences
        """
        if fit:
            self.tokenizer = Tokenizer(
                num_words=self.config.text_max_features,
                oov_token="<OOV>",
                lower=True
            )
            self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.config.text_max_length,
            padding='post',
            truncating='post'
        )

        return padded

    def build_model(self, vocab_size: int, num_classes: int) -> keras.Model:
        """
        Build LSTM architecture for text classification.

        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of target classes

        Returns:
            Compiled Keras model
        """
        params = self.config.lstm_params

        model = models.Sequential()

        # Embedding layer
        model.add(layers.Embedding(
            input_dim=vocab_size,
            output_dim=params['embedding_dim'],
            input_length=self.config.text_max_length,
            name='embedding'
        ))

        # LSTM layers
        for i, units in enumerate(params['lstm_units']):
            return_sequences = (i < len(params['lstm_units']) - 1)  # Not last layer

            if params['bidirectional']:
                model.add(layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=params['dropout_rate'],
                        recurrent_dropout=params['recurrent_dropout']
                    ),
                    name=f'bilstm_{i+1}'
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=params['dropout_rate'],
                    recurrent_dropout=params['recurrent_dropout'],
                    name=f'lstm_{i+1}'
                ))

        # Dense layers before output
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(params['dropout_rate'], name='dropout_1'))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def train(self, X_train_text: np.ndarray, y_train: np.ndarray,
              X_val_text: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train LSTM classifier on text data.

        Args:
            X_train_text: Training text data (raw strings)
            y_train: Training labels
            X_val_text: Validation text data (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dict
        """
        # Prepare text sequences
        X_train_seq = self.prepare_text(X_train_text, fit=True)

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.num_classes = len(self.label_encoder.classes_)

        # Build model
        vocab_size = min(self.config.text_max_features, len(self.tokenizer.word_index)) + 1
        self.model = self.build_model(vocab_size, self.num_classes)

        # Prepare callbacks
        callback_list = []

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val_text is not None else 'loss',
            patience=self.config.lstm_params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        callback_list.append(early_stop)

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val_text is not None else 'loss',
            factor=0.5,
            patience=5,
            verbose=0,
            min_lr=1e-7
        )
        callback_list.append(reduce_lr)

        # Prepare validation data
        validation_data = None
        if X_val_text is not None and y_val is not None:
            X_val_seq = self.prepare_text(X_val_text, fit=False)
            y_val_encoded = self.label_encoder.transform(y_val)
            validation_data = (X_val_seq, y_val_encoded)

        # Train
        history = self.model.fit(
            X_train_seq,
            y_train_encoded,
            validation_data=validation_data,
            epochs=self.config.lstm_params['epochs'],
            batch_size=self.config.lstm_params['batch_size'],
            callbacks=callback_list,
            verbose=0
        )

        self.history = history.history

        return self.history

    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict class labels from text."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        X_seq = self.prepare_text(X_text, fit=False)
        y_pred_proba = self.model.predict(X_seq, verbose=0)

        if self.num_classes == 2:
            y_pred_encoded = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X_text: np.ndarray) -> np.ndarray:
        """Predict class probabilities from text."""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        X_seq = self.prepare_text(X_text, fit=False)
        y_pred_proba = self.model.predict(X_seq, verbose=0)

        if self.num_classes == 2:
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
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'num_classes': self.num_classes,
            'history': self.history
        }
        joblib.dump(meta_dict, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'LSTMClassifier':
        """Load model from disk."""
        meta_dict = joblib.load(filepath)

        instance = cls(meta_dict['config'])

        # Load Keras model
        model_path = filepath.replace('.pkl', '.keras')
        instance.model = keras.models.load_model(model_path)

        instance.tokenizer = meta_dict['tokenizer']
        instance.label_encoder = meta_dict['label_encoder']
        instance.num_classes = meta_dict['num_classes']
        instance.history = meta_dict.get('history')

        return instance
