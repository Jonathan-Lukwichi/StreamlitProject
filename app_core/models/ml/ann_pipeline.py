# =============================================================================
# app_core/models/ml/ann_pipeline.py
# Simple Baseline ANN Pipeline for Multi-Horizon Time Series Forecasting
# Architecture: Feedforward Neural Network (Dense layers only)
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class ANNPipeline:
    """
    Simple Baseline Artificial Neural Network (ANN) for time series forecasting.

    Architecture: Feedforward network with Dense layers + Dropout
    - No sequence generation (unlike LSTM - direct tabular input)
    - No batch normalization (baseline simplicity)
    - No callbacks or optimization tricks (fixed training)
    - Per-feature preprocessing detection

    Designed for: Fast baseline comparisons with LSTM and XGBoost
    """

    def __init__(self, config: dict):
        """
        Initialize ANN pipeline with configuration.

        Args:
            config: Dictionary with keys:
                - ann_hidden_layers: Number of hidden layers (default: 2)
                - ann_neurons: Neurons per layer (default: 64)
                - ann_activation: Activation function (default: 'relu')
                - dropout: Dropout rate (default: 0.2)
                - ann_epochs: Training epochs (default: 30)
        """
        self.hidden_layers = config.get("ann_hidden_layers", 2)
        self.neurons = config.get("ann_neurons", 64)
        self.activation = config.get("ann_activation", "relu")
        self.dropout = config.get("dropout", 0.2)
        self.epochs = config.get("ann_epochs", 30)

        # Model state
        self.model = None
        self.history = None
        self.preprocessing_report = None

        print("=" * 70)
        print("🤖 ANN PIPELINE INITIALIZED")
        print("=" * 70)
        print(f"Hidden Layers:  {self.hidden_layers}")
        print(f"Neurons/Layer:  {self.neurons}")
        print(f"Activation:     {self.activation}")
        print(f"Dropout:        {self.dropout}")
        print(f"Epochs:         {self.epochs}")
        print(f"Optimizer:      Adam (LR=0.001, default)")
        print(f"Batch Size:     32 (default)")
        print(f"Loss:           MSE")
        print("=" * 70)

    def detect_preprocessing_state(self, X, y=None):
        """
        Intelligent preprocessing detection - analyzes EACH feature individually.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (optional, not used currently)

        Returns:
            dict: Preprocessing analysis with keys:
                - overall_state: 'Scaled' / 'Unscaled' / 'Mixed'
                - per_feature_analysis: List of dicts with feature-level details
                - scaled_count: Number of scaled features
                - total_features: Total number of features
        """
        n_features = X.shape[1]
        feature_analysis = []

        for i in range(n_features):
            col = X[:, i]
            col_min, col_max = col.min(), col.max()
            col_mean, col_std = col.mean(), col.std()

            # Detect scaling type for THIS specific feature
            if -0.05 <= col_min <= 0.05 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[0,1]'
                is_scaled = True
                confidence = 'HIGH'
            elif -1.05 <= col_min <= -0.95 and 0.95 <= col_max <= 1.05:
                scaling_type = 'MinMaxScaler[-1,1]'
                is_scaled = True
                confidence = 'HIGH'
            elif -0.5 <= col_mean <= 0.5 and 0.8 <= col_std <= 1.2:
                scaling_type = 'StandardScaler'
                is_scaled = True
                confidence = 'MEDIUM'
            else:
                scaling_type = 'Unscaled'
                is_scaled = False
                confidence = 'HIGH'

            feature_analysis.append({
                'feature_idx': i,
                'min': float(col_min),
                'max': float(col_max),
                'mean': float(col_mean),
                'std': float(col_std),
                'scaling_type': scaling_type,
                'is_scaled': is_scaled,
                'confidence': confidence
            })

        # Overall state
        scaled_count = sum(1 for f in feature_analysis if f['is_scaled'])
        if scaled_count == n_features:
            overall_state = 'Scaled'
        elif scaled_count == 0:
            overall_state = 'Unscaled'
        else:
            overall_state = 'Mixed'

        return {
            'overall_state': overall_state,
            'per_feature_analysis': feature_analysis,
            'scaled_count': scaled_count,
            'total_features': n_features
        }

    def build_model(self, n_features: int):
        """
        Build simple feedforward neural network.

        Architecture:
            Input(n_features) → Dense(neurons, activation) → Dropout →
            [Dense(neurons, activation) → Dropout] × (hidden_layers - 1) →
            Output(1, linear)

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = Sequential(name="ANN_Baseline")

        # Input + First hidden layer
        model.add(Dense(
            self.neurons,
            activation=self.activation,
            input_dim=n_features,
            name="hidden_1"
        ))
        model.add(Dropout(self.dropout, name="dropout_1"))

        # Additional hidden layers
        for i in range(2, self.hidden_layers + 1):
            model.add(Dense(
                self.neurons,
                activation=self.activation,
                name=f"hidden_{i}"
            ))
            model.add(Dropout(self.dropout, name=f"dropout_{i}"))

        # Output layer (regression)
        model.add(Dense(1, activation='linear', name="output"))

        # Compile with fixed settings (baseline)
        model.compile(
            optimizer='adam',  # Default LR = 0.001
            loss='mse',
            metrics=['mae']
        )

        self.model = model

        print("=" * 70)
        print("🏗️  ANN MODEL ARCHITECTURE")
        print("=" * 70)
        model.summary()
        print("=" * 70)

        return model

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train ANN model with fixed epochs (no callbacks).

        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training targets (n_train,)
            X_val: Validation features (n_val, n_features)
            y_val: Validation targets (n_val,)
        """
        # ===== DATA VALIDATION & CLEANING =====
        print("=" * 70)
        print("🔍 DATA VALIDATION & PREPROCESSING")
        print("=" * 70)

        # Step 1: Ensure numpy arrays (not pandas DataFrames)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        print(f"Input shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Input dtypes: X_train={X_train.dtype}, y_train={y_train.dtype}")

        # Step 2: Force conversion to float (handles object dtype, strings, etc.)
        try:
            # Try to convert to float - this will fail if there are non-numeric values
            X_train = X_train.astype(np.float64)
            y_train = y_train.astype(np.float64)
            X_val = X_val.astype(np.float64)
            y_val = y_val.astype(np.float64)
            print("✅ Successfully converted to numeric arrays")
        except (ValueError, TypeError) as e:
            print(f"❌ ERROR: Cannot convert data to numeric format: {e}")
            raise ValueError(
                "Data contains non-numeric values (strings, objects, etc.). "
                "Please ensure all features are numeric before training. "
                "Check your feature selection - remove any text or categorical columns."
            )

        # Step 3: Check for NaN/inf after conversion
        X_train_nans = np.isnan(X_train).sum()
        X_train_infs = np.isinf(X_train).sum()
        y_train_nans = np.isnan(y_train).sum()
        y_train_infs = np.isinf(y_train).sum()

        print(f"\nX_train - NaNs: {X_train_nans}, Infs: {X_train_infs}")
        print(f"y_train - NaNs: {y_train_nans}, Infs: {y_train_infs}")

        X_val_nans = np.isnan(X_val).sum()
        X_val_infs = np.isinf(X_val).sum()
        y_val_nans = np.isnan(y_val).sum()
        y_val_infs = np.isinf(y_val).sum()

        print(f"X_val   - NaNs: {X_val_nans}, Infs: {X_val_infs}")
        print(f"y_val   - NaNs: {y_val_nans}, Infs: {y_val_infs}")

        # Step 4: Clean NaN/inf values
        has_issues = (X_train_nans + X_train_infs + y_train_nans + y_train_infs +
                     X_val_nans + X_val_infs + y_val_nans + y_val_infs) > 0

        if has_issues:
            print("\n⚠️  WARNING: Found NaN/inf values - replacing with 0...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
            print("✅ Data cleaned (NaN/inf → 0)")
        else:
            print("✅ Data is clean (no NaN/inf values)")

        # Step 5: Convert to float32 for TensorFlow
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        print(f"\nFinal dtypes: X_train={X_train.dtype}, y_train={y_train.dtype}")
        print(f"Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Data range: X_train=[{X_train.min():.3f}, {X_train.max():.3f}], "
              f"y_train=[{y_train.min():.3f}, {y_train.max():.3f}]")
        print("=" * 70)

        # Detect preprocessing state
        detection = self.detect_preprocessing_state(X_train, y_train)
        self.preprocessing_report = detection

        print("=" * 70)
        print("🔍 ANN PREPROCESSING DETECTION REPORT")
        print("=" * 70)
        print(f"Overall State:    {detection['overall_state']}")
        print(f"Scaled Features:  {detection['scaled_count']}/{detection['total_features']}")
        print(f"")
        print("ℹ️  ANN benefits from feature scaling (like LSTM), but can work without it.")
        print("   For optimal performance, consider applying MinMaxScaler or StandardScaler.")
        print("=" * 70)

        # Print per-feature analysis (first 5 features)
        print("\n📊 Per-Feature Analysis (first 5 features):")
        for feat in detection['per_feature_analysis'][:5]:
            status = "✅" if feat['is_scaled'] else "⚠️"
            print(f"  {status} Feature {feat['feature_idx']}: {feat['scaling_type']} "
                  f"(min={feat['min']:.3f}, max={feat['max']:.3f}, "
                  f"confidence={feat['confidence']})")

        if detection['total_features'] > 5:
            print(f"  ... and {detection['total_features'] - 5} more features")
        print("=" * 70)

        # Simple training - no callbacks, fixed epochs
        print(f"\n🚀 Training ANN for {self.epochs} epochs...")
        print("=" * 70)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=32,
            verbose=1  # Show epoch progress
        )

        self.history = history

        # Training summary
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]

        print("=" * 70)
        print("✅ ANN TRAINING COMPLETED")
        print("=" * 70)
        print(f"Final Training Loss (MSE):     {final_train_loss:.6f}")
        print(f"Final Validation Loss (MSE):   {final_val_loss:.6f}")
        print(f"Final Training MAE:            {final_train_mae:.6f}")
        print(f"Final Validation MAE:          {final_val_mae:.6f}")
        print("=" * 70)

    def predict(self, X):
        """
        Make predictions using trained model.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            predictions: 1D array of predictions (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Ensure numpy array and convert to float
        X = np.asarray(X)
        try:
            X = X.astype(np.float64)
        except (ValueError, TypeError):
            raise ValueError("Prediction data contains non-numeric values")

        # Clean NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to float32 for TensorFlow
        X = X.astype(np.float32)

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, y_true, y_pred):
        """
        Calculate evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            dict: Metrics (MAE, RMSE, MAPE, Accuracy)
        """
        # Clean data (handle NaN/inf in predictions)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # RMSE
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = 0.0

        # Accuracy (100 - MAPE)
        accuracy = 100.0 - mape

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Accuracy": accuracy
        }

    def get_training_history(self):
        """
        Get training history for visualization.

        Returns:
            dict: Training history with keys 'loss', 'val_loss', 'mae', 'val_mae'
        """
        if self.history is None:
            return None

        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history['mae'],
            'val_mae': self.history.history['val_mae']
        }
