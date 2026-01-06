# =============================================================================
# lstm_pipeline.py
# LSTM Pipeline for Healthcare Time Series Forecasting
# - Intelligent preprocessing detection (per-feature analysis)
# - Sequence generation for time series
# - Monte Carlo Dropout for uncertainty quantification
# - Healthcare-specific architecture
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMPipeline:
    """
    LSTM Pipeline for multi-horizon time series forecasting with intelligent preprocessing.

    Features:
    - Per-feature preprocessing detection
    - Automatic sequence generation
    - Stacked LSTM architecture
    - Monte Carlo Dropout for confidence intervals
    - Early stopping and learning rate scheduling
    """

    def __init__(self, config):
        """
        Initialize LSTM pipeline with configuration.

        Args:
            config: Dictionary with hyperparameters
                - lookback_window: Past observations to use (default: 14)
                - lstm_hidden_units: LSTM units in first layer (default: 64)
                - lstm_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout rate (default: 0.2)
                - lstm_epochs: Training epochs (default: 100)
                - lstm_lr: Learning rate (default: 0.001)
                - batch_size: Batch size (default: 32)
        """
        self.lookback = config.get("lookback_window", 14)
        self.lstm_units = config.get("lstm_hidden_units", 64)
        self.lstm_layers = config.get("lstm_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.epochs = config.get("lstm_epochs", 100)
        self.learning_rate = config.get("lstm_lr", 0.001)
        self.batch_size = config.get("batch_size", 32)

        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.n_features = None
        self.preprocessing_report = None

    def detect_preprocessing_state(self, X, y=None):
        """
        Check EACH variable individually for preprocessing state.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - optional

        Returns:
            Comprehensive analysis of each feature's preprocessing state
        """
        n_features = X.shape[1]
        feature_analysis = []

        # Check each feature column separately
        for i in range(n_features):
            feature_col = X[:, i]

            col_min, col_max = feature_col.min(), feature_col.max()
            col_mean, col_std = feature_col.mean(), feature_col.std()
            col_range = col_max - col_min

            # Detect scaling type for THIS specific feature
            if -0.05 <= col_min <= 0.05 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[0,1]'
                is_scaled = True
                confidence = 'HIGH'

            elif -1.05 <= col_min <= -0.95 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[-1,1]'
                is_scaled = True
                confidence = 'HIGH'

            elif abs(col_mean) < 0.3 and 0.7 < col_std < 1.3:
                scaling_type = 'StandardScaler'
                is_scaled = True
                confidence = 'MEDIUM'

            elif col_range < 5 and abs(col_mean) < 2:
                scaling_type = 'Custom/Unknown'
                is_scaled = True
                confidence = 'LOW'

            else:
                scaling_type = 'Unscaled'
                is_scaled = False
                confidence = 'HIGH'

            feature_analysis.append({
                'feature_idx': i,
                'is_scaled': is_scaled,
                'scaling_type': scaling_type,
                'confidence': confidence,
                'min': float(col_min),
                'max': float(col_max),
                'mean': float(col_mean),
                'std': float(col_std),
                'range': float(col_range)
            })

        # Check target variable separately
        target_analysis = None
        if y is not None:
            y_min, y_max = y.min(), y.max()
            y_mean, y_std = y.mean(), y.std()
            y_range = y_max - y_min

            if -0.05 <= y_min <= 0.05 and 0.95 <= y_max <= 1.05:
                target_scaling = 'MinMaxScaler[0,1]'
                target_scaled = True
                target_confidence = 'HIGH'
            elif -1.05 <= y_min <= -0.95 and 0.95 <= y_max <= 1.05:
                target_scaling = 'MinMaxScaler[-1,1]'
                target_scaled = True
                target_confidence = 'HIGH'
            elif abs(y_mean) < 0.3 and 0.7 < y_std < 1.3:
                target_scaling = 'StandardScaler'
                target_scaled = True
                target_confidence = 'MEDIUM'
            else:
                target_scaling = 'Unscaled'
                target_scaled = False
                target_confidence = 'HIGH'

            target_analysis = {
                'is_scaled': target_scaled,
                'scaling_type': target_scaling,
                'confidence': target_confidence,
                'min': float(y_min),
                'max': float(y_max),
                'mean': float(y_mean),
                'std': float(y_std),
                'range': float(y_range)
            }

        # Aggregate analysis
        scaled_count = sum(1 for f in feature_analysis if f['is_scaled'])
        high_confidence_count = sum(1 for f in feature_analysis
                                    if f['is_scaled'] and f['confidence'] == 'HIGH')

        # Determine overall state
        if scaled_count == n_features and high_confidence_count >= n_features * 0.8:
            overall_state = 'FULLY_SCALED'
            needs_scaling = False
            recommendation = 'All features appear scaled - SKIP scaling'
        elif scaled_count == 0:
            overall_state = 'UNSCALED'
            needs_scaling = True
            recommendation = 'No features scaled - APPLY scaling'
        elif scaled_count < n_features * 0.5:
            overall_state = 'MOSTLY_UNSCALED'
            needs_scaling = True
            recommendation = 'Majority unscaled - APPLY scaling for consistency'
        else:
            overall_state = 'PARTIALLY_SCALED'
            needs_scaling = True
            recommendation = 'Mixed scaling detected - RE-SCALE all for consistency'

        return {
            'feature_analysis': feature_analysis,
            'target_analysis': target_analysis,
            'scaled_features': scaled_count,
            'total_features': n_features,
            'high_confidence_scaled': high_confidence_count,
            'overall_state': overall_state,
            'needs_scaling': needs_scaling,
            'recommendation': recommendation
        }

    def create_sequences(self, X, y):
        """
        Convert tabular data to sequences for LSTM.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            X_seq: (n_sequences, lookback, n_features)
            y_seq: (n_sequences,)
        """
        X_seq, y_seq = [], []

        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, n_features):
        """
        Build stacked LSTM architecture.

        Args:
            n_features: Number of input features
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM. Install with: pip install tensorflow")

        self.n_features = n_features

        # Build model
        model = keras.Sequential()

        # First LSTM layer (returns sequences for stacking)
        model.add(layers.LSTM(
            self.lstm_units,
            return_sequences=True if self.lstm_layers > 1 else False,
            input_shape=(self.lookback, n_features)
        ))
        model.add(layers.Dropout(self.dropout))

        # Additional LSTM layers
        for i in range(1, self.lstm_layers):
            return_seq = i < self.lstm_layers - 1  # Last layer doesn't return sequences
            units = self.lstm_units // (2 ** i)  # Decrease units in deeper layers
            model.add(layers.LSTM(units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))

        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        self.model = model

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train LSTM with intelligent preprocessing detection.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training target (n_samples,)
            X_val: Validation features
            y_val: Validation target
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM")

        # Step 1: Detect preprocessing state
        detection = self.detect_preprocessing_state(X_train, y_train)
        self.preprocessing_report = detection

        print("=" * 70)
        print("🔍 LSTM PREPROCESSING DETECTION REPORT")
        print("=" * 70)
        print(f"Overall State: {detection['overall_state']}")
        print(f"Recommendation: {detection['recommendation']}")
        print(f"Scaled Features: {detection['scaled_features']}/{detection['total_features']}")
        print("-" * 70)

        # Step 2: Apply scaling if needed
        if detection['needs_scaling']:
            print("⚙️  APPLYING SCALING for consistency...")
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))

            X_train = self.scaler_X.fit_transform(X_train)
            X_val = self.scaler_X.transform(X_val)

            y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

            print("✓ Scaling applied")
        else:
            print("✓ SKIPPING SCALING - data already preprocessed")
            self.scaler_X = None
            self.scaler_y = None

        print("=" * 70)

        # Step 3: Create sequences
        print(f"📊 Creating sequences (lookback={self.lookback})...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        print(f"   Train sequences: {X_train_seq.shape}")
        print(f"   Val sequences: {X_val_seq.shape}")

        # Step 4: Build model
        if self.model is None:
            print(f"🏗️  Building LSTM model...")
            self.build_model(n_features=X_train.shape[1])
            print(f"   Architecture: {self.lstm_layers} layers, {self.lstm_units} units")

        # Step 5: Train with callbacks
        print(f"🚀 Training LSTM ({self.epochs} epochs max)...")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        final_epoch = len(history.history['loss'])
        print(f"✓ Training complete ({final_epoch} epochs)")
        print("=" * 70)

    def predict(self, X, return_ci=False, mc_samples=100):
        """
        Make predictions with optional Monte Carlo Dropout confidence intervals.

        Args:
            X: Features (n_samples, n_features)
            return_ci: Whether to return confidence intervals
            mc_samples: Number of MC samples for uncertainty

        Returns:
            predictions or (predictions, lower_ci, upper_ci)
        """
        # Scale if scaler exists
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)

        # Create sequences
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))

        if not return_ci:
            # Standard prediction
            y_pred_scaled = self.model.predict(X_seq, verbose=0).flatten()

            # Inverse transform
            if self.scaler_y is not None:
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            else:
                y_pred = y_pred_scaled

            # Pad to match original length
            full_pred = np.full(len(X), np.nan)
            full_pred[self.lookback:] = y_pred

            return full_pred
        else:
            # Monte Carlo Dropout for uncertainty
            predictions = []
            for _ in range(mc_samples):
                y_pred = self.model(X_seq, training=True).numpy().flatten()
                predictions.append(y_pred)

            predictions = np.array(predictions)

            # Calculate statistics
            mean_pred_scaled = np.mean(predictions, axis=0)
            lower_ci_scaled = np.percentile(predictions, 5, axis=0)
            upper_ci_scaled = np.percentile(predictions, 95, axis=0)

            # Inverse transform
            if self.scaler_y is not None:
                mean_pred = self.scaler_y.inverse_transform(mean_pred_scaled.reshape(-1, 1)).flatten()
                lower_ci = self.scaler_y.inverse_transform(lower_ci_scaled.reshape(-1, 1)).flatten()
                upper_ci = self.scaler_y.inverse_transform(upper_ci_scaled.reshape(-1, 1)).flatten()
            else:
                mean_pred = mean_pred_scaled
                lower_ci = lower_ci_scaled
                upper_ci = upper_ci_scaled

            # Pad to match original length
            full_mean = np.full(len(X), np.nan)
            full_lower = np.full(len(X), np.nan)
            full_upper = np.full(len(X), np.nan)

            full_mean[self.lookback:] = mean_pred
            full_lower[self.lookback:] = lower_ci
            full_upper[self.lookback:] = upper_ci

            return full_mean, full_lower, full_upper

    def evaluate(self, y_true, y_pred):
        """
        Calculate regression metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics (MAE, RMSE, MAPE, Accuracy)
        """
        # Remove NaN values (from lookback padding)
        mask = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "Accuracy": np.nan
            }

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAPE (avoid division by zero)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

        # Accuracy (100 - MAPE)
        accuracy = max(0, 100 - mape)

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "Accuracy": float(accuracy)
        }


# =============================================================================
# LSTM WITH ATTENTION (Step 4.1)
# Academic Reference: Bahdanau et al. (2015) - Attention Mechanism
#                     Vaswani et al. (2017) - Self-Attention (Transformers)
# =============================================================================
class AttentionLSTM:
    """
    Bidirectional LSTM with Self-Attention for Time Series Forecasting.

    Architecture:
    1. Bidirectional LSTM layers (captures forward and backward dependencies)
    2. Self-Attention layer (learns which time steps are most important)
    3. Dense layers for prediction

    Advantages over standard LSTM:
    - Bidirectional: Uses future context (for training) to understand patterns
    - Attention: Automatically learns which past observations matter most
    - Better long-range dependency modeling

    Academic References:
        - Bahdanau, Cho & Bengio (2015) - "Neural Machine Translation by
          Jointly Learning to Align and Translate" (Attention)
        - Vaswani et al. (2017) - "Attention Is All You Need" (Self-Attention)
        - Lai et al. (2018) - "Modeling Long- and Short-Term Temporal Patterns
          with Deep Neural Networks" (LSTNet)

    Usage:
        model = AttentionLSTM(config)
        model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
    """

    def __init__(self, config):
        """
        Initialize Attention LSTM.

        Args:
            config: Dictionary with hyperparameters
                - lookback_window: Past observations (default: 14)
                - lstm_hidden_units: LSTM units (default: 64)
                - attention_heads: Number of attention heads (default: 4)
                - dropout: Dropout rate (default: 0.2)
                - lstm_epochs: Training epochs (default: 100)
                - lstm_lr: Learning rate (default: 0.001)
                - bidirectional: Use bidirectional LSTM (default: True)
        """
        self.lookback = config.get("lookback_window", 14)
        self.lstm_units = config.get("lstm_hidden_units", 64)
        self.attention_heads = config.get("attention_heads", 4)
        self.dropout = config.get("dropout", 0.2)
        self.epochs = config.get("lstm_epochs", 100)
        self.learning_rate = config.get("lstm_lr", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.bidirectional = config.get("bidirectional", True)

        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.n_features = None
        self.attention_weights = None

    def create_sequences(self, X, y):
        """Create sequences for LSTM (same as LSTMPipeline)."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def _build_attention_layer(self):
        """
        Build self-attention layer using TensorFlow/Keras.

        Returns:
            Attention layer (MultiHeadAttention or custom)
        """
        try:
            from tensorflow.keras import layers
            # Use built-in MultiHeadAttention (TensorFlow >= 2.4)
            return layers.MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.lstm_units // self.attention_heads,
                dropout=self.dropout / 2,
            )
        except AttributeError:
            # Fallback: Simple Attention layer for older TensorFlow
            from tensorflow import keras
            from tensorflow.keras import layers

            class SimpleAttention(layers.Layer):
                """Simple additive attention layer."""
                def __init__(self, units):
                    super().__init__()
                    self.W = layers.Dense(units)
                    self.V = layers.Dense(1)

                def call(self, inputs):
                    # inputs: (batch, seq_len, features)
                    score = self.V(keras.activations.tanh(self.W(inputs)))
                    attention_weights = keras.activations.softmax(score, axis=1)
                    context = inputs * attention_weights
                    return keras.backend.sum(context, axis=1), attention_weights

            return SimpleAttention(self.lstm_units)

    def build_model(self, n_features):
        """
        Build Bidirectional LSTM with Self-Attention.

        Args:
            n_features: Number of input features
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.n_features = n_features

        # Input layer
        inputs = layers.Input(shape=(self.lookback, n_features))

        # Bidirectional LSTM layers
        if self.bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True)
            )(inputs)
            lstm_out_dim = self.lstm_units * 2
        else:
            x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
            lstm_out_dim = self.lstm_units

        x = layers.Dropout(self.dropout)(x)

        # Second LSTM layer
        if self.bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(self.lstm_units // 2, return_sequences=True)
            )(x)
            lstm_out_dim = self.lstm_units
        else:
            x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)
            lstm_out_dim = self.lstm_units // 2

        x = layers.Dropout(self.dropout)(x)

        # Self-Attention Layer
        # MultiHeadAttention: query, key, value all come from LSTM output
        try:
            attention_layer = layers.MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=max(1, lstm_out_dim // self.attention_heads),
                dropout=self.dropout / 2,
            )
            attention_output = attention_layer(x, x, return_attention_scores=False)

            # Residual connection
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)

        except (AttributeError, TypeError):
            # Fallback: Simple attention for older TensorFlow
            # Compute attention scores
            attention_dense = layers.Dense(1, activation='tanh')
            attention_scores = attention_dense(x)
            attention_weights = layers.Softmax(axis=1)(attention_scores)
            x = layers.Multiply()([x, attention_weights])

        # Global pooling to aggregate sequence
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout / 2)(x)
        x = layers.Dense(16, activation='relu')(x)

        # Output
        outputs = layers.Dense(1)(x)

        # Build model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Attention LSTM with preprocessing detection.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training history
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for Attention LSTM")

        # Preprocessing detection (reuse from LSTMPipeline logic)
        print("=" * 70)
        print("🔍 ATTENTION LSTM - Building Model")
        print("=" * 70)
        print(f"   Architecture: Bidirectional={self.bidirectional}")
        print(f"   LSTM Units: {self.lstm_units}")
        print(f"   Attention Heads: {self.attention_heads}")
        print(f"   Lookback: {self.lookback}")
        print("-" * 70)

        # Apply scaling
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        else:
            X_val_scaled, y_val_scaled = None, None

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        print(f"📊 Train sequences: {X_train_seq.shape}")

        if X_val_scaled is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
            print(f"   Val sequences: {X_val_seq.shape}")
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None

        # Build model
        if self.model is None:
            print("🏗️  Building Attention LSTM...")
            self.build_model(n_features=X_train.shape[1])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]

        # Train
        print(f"🚀 Training ({self.epochs} epochs max)...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        final_epoch = len(history.history['loss'])
        print(f"✓ Training complete ({final_epoch} epochs)")
        print("=" * 70)

        return history

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions in original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale input
        if self.scaler_X is not None:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X

        # Create sequences
        X_seq = []
        for i in range(self.lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-self.lookback:i])
        X_seq = np.array(X_seq)

        # Predict
        y_pred_scaled = self.model.predict(X_seq, verbose=0).flatten()

        # Inverse transform
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled

        # Pad to match original length
        full_pred = np.full(len(X), np.nan)
        full_pred[self.lookback:] = y_pred

        return full_pred

    def evaluate(self, y_true, y_pred):
        """Calculate regression metrics."""
        mask = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

        return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"

        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
