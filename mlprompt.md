# Actionable Prompts for ML Thesis Implementation
## HealthForecast AI - ED Patient Arrival Forecasting

This document contains refined, implementation-ready prompts for elevating the HealthForecast AI project to Master's Thesis standards. Each prompt includes specific file paths, function names, and code patterns for the 7-day ED patient arrival forecasting system.

**Target Variables:** `Target_1` through `Target_7` (7-day ahead patient arrivals)
**Clinical Categories:** RESPIRATORY, CARDIAC, TRAUMA, GASTROINTESTINAL, INFECTIOUS, NEUROLOGICAL, OTHER

---

## I. Core Methodological Improvements

### **1. Implement Time Series Cross-Validation (CRITICAL PRIORITY)**

**Problem:** The current pipeline uses a single train/test split, which is insufficient for validating time series models academically.

**Implementation Details:**

**File:** `app_core/pipelines/temporal_split.py` (create new or modify existing)

```python
from sklearn.model_selection import TimeSeriesSplit

def create_cv_folds(df, n_splits=5, gap=7):
    """
    Create rolling-origin CV folds with a 7-day gap to prevent leakage.

    Parameters:
    - df: DataFrame with DatetimeIndex
    - n_splits: Number of CV folds (recommend 5 for thesis)
    - gap: Days between train and test to prevent leakage

    Returns: List of (train_idx, test_idx) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return list(tscv.split(df))
```

**Modifications Required:**

1. **`pages/06_Feature_Studio.py`:**
   - Add `st.number_input("Number of CV Folds", min_value=3, max_value=10, value=5)`
   - Store `cv_folds` in `st.session_state["cv_config"]`
   - Display fold date ranges in an expander for transparency

2. **`pages/07_Feature_Selection.py`:**
   - Refactor `compute_feature_importance()` to iterate over all CV folds
   - Calculate importance scores per fold, then compute mean ± std
   - Display: "Feature X: 0.15 ± 0.03 importance (across 5 folds)"

3. **`pages/08_Train_Models.py`:**
   - Modify `run_ml_multihorizon()` to train/evaluate on each fold
   - Final metrics format: `MAE: 12.3 ± 1.8` (mean ± std across folds)
   - Add confidence intervals to forecast visualizations

4. **`pages/09_Model_Results.py`:**
   - Update comparison tables to show mean ± std for all metrics
   - Add "Fold Stability" metric: std/mean (lower = more robust)

---

### **2. Automate and Expand Feature Engineering**

**Problem:** Current features are calendar-based only. Missing critical endogenous (lag) and exogenous (weather) temporal features.

**Implementation Details:**

#### **2A. Automate Fourier Features from FFT Analysis**

**File:** `pages/06_Feature_Studio.py`

```python
def generate_fourier_features(df, detected_periods, n_harmonics=2):
    """
    Generate Fourier terms from FFT-detected cycles.

    Parameters:
    - df: DataFrame with DatetimeIndex
    - detected_periods: List of dominant periods from EDA FFT (e.g., [7, 30, 365])
    - n_harmonics: Number of sine/cosine pairs per period

    Returns: DataFrame with Fourier features
    """
    t = np.arange(len(df))
    features = {}

    for period in detected_periods:
        for k in range(1, n_harmonics + 1):
            features[f'sin_{period}d_h{k}'] = np.sin(2 * np.pi * k * t / period)
            features[f'cos_{period}d_h{k}'] = np.cos(2 * np.pi * k * t / period)

    return pd.DataFrame(features, index=df.index)
```

**UI Addition:**
```python
st.subheader("Fourier Feature Generation")
if "fft_dominant_cycles" in st.session_state:
    detected = st.session_state["fft_dominant_cycles"]  # From 04_Explore_Data.py
    st.info(f"Detected cycles from EDA: {detected}")
    use_fourier = st.checkbox("Generate Fourier features from detected cycles", value=True)
    if use_fourier:
        n_harmonics = st.slider("Harmonics per cycle", 1, 4, 2)
```

#### **2B. Generate Lag and Rolling Window Features**

**File:** `pages/03_Prepare_Data.py` - Add to `process_dataset()` function

```python
def generate_temporal_features(df, target_col='ED_Arrivals',
                                exog_cols=['Average_Temp', 'Total_precipitation'],
                                lag_days=[1, 2, 3, 7, 14],
                                window_sizes=[3, 7, 14]):
    """
    Generate lag and rolling window features for target and exogenous variables.

    Creates:
    - ED_lag_1, ED_lag_2, ..., ED_lag_14 (past arrivals)
    - Temp_lag_1, Temp_lag_7 (lagged weather)
    - ED_roll_7_mean, ED_roll_7_std, ED_roll_7_min, ED_roll_7_max
    """
    result = df.copy()

    # Target lags
    for lag in lag_days:
        result[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Exogenous lags
    for col in exog_cols:
        if col in df.columns:
            for lag in [1, 7]:
                result[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Rolling windows
    for window in window_sizes:
        result[f'{target_col}_roll_{window}_mean'] = df[target_col].rolling(window).mean()
        result[f'{target_col}_roll_{window}_std'] = df[target_col].rolling(window).std()
        result[f'{target_col}_roll_{window}_min'] = df[target_col].rolling(window).min()
        result[f'{target_col}_roll_{window}_max'] = df[target_col].rolling(window).max()

    return result.dropna()
```

**UI Addition:**
```python
with st.expander("Temporal Feature Engineering", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        lag_days = st.multiselect("Lag days", [1,2,3,7,14,21,28], default=[1,2,3,7,14])
    with col2:
        window_sizes = st.multiselect("Rolling windows", [3,7,14,21,30], default=[3,7,14])

    generate_lags = st.checkbox("Generate lag features", value=True)
    generate_rolling = st.checkbox("Generate rolling statistics", value=True)
```

---

### **3. Implement Naive Baseline Models**

**Problem:** Cannot prove ML models add value without benchmarking against simple heuristics.

**File:** `app_core/models/baselines.py` (create new)

```python
import numpy as np
import pandas as pd

class NaiveForecaster:
    """Naive forecast: predict last known value for all horizons."""

    def __init__(self, name="Naive"):
        self.name = name
        self.last_value = None

    def fit(self, y_train):
        self.last_value = y_train.iloc[-1]
        return self

    def predict(self, horizon=7):
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecaster:
    """Seasonal naive: predict value from same day last week."""

    def __init__(self, seasonal_period=7, name="Seasonal Naive"):
        self.name = name
        self.seasonal_period = seasonal_period
        self.history = None

    def fit(self, y_train):
        self.history = y_train.values
        return self

    def predict(self, horizon=7):
        predictions = []
        for h in range(horizon):
            # Value from same day, one period ago
            idx = -(self.seasonal_period - h % self.seasonal_period)
            predictions.append(self.history[idx])
        return np.array(predictions)


def evaluate_baselines(y_train, y_test, horizons=[1,2,3,4,5,6,7]):
    """
    Evaluate naive baselines and return metrics for comparison.

    Returns: Dict with MAE, RMSE, MAPE for each baseline per horizon
    """
    results = {}

    for BaselineClass in [NaiveForecaster, SeasonalNaiveForecaster]:
        model = BaselineClass().fit(y_train)
        preds = model.predict(len(y_test))

        mae = np.abs(y_test.values - preds).mean()
        rmse = np.sqrt(((y_test.values - preds) ** 2).mean())
        mape = (np.abs(y_test.values - preds) / y_test.values).mean() * 100

        results[model.name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    return results
```

**Integration in `pages/05_Baseline_Models.py`:**
```python
# Add new tab: "Naive Baselines"
with tabs[0]:  # or create new tab
    st.subheader("Naive Baseline Benchmarks")
    if st.button("Compute Naive Baselines"):
        from app_core.models.baselines import evaluate_baselines
        baseline_results = evaluate_baselines(train_data, test_data)
        st.session_state["baseline_results"] = baseline_results

        # Display comparison table
        st.dataframe(pd.DataFrame(baseline_results).T)
        st.info("Your ML models must outperform these baselines to demonstrate value.")
```

---

### **4. Automate Stationarity Response**

**Problem:** EDA detects non-stationarity but user must manually act on it.

**File:** `pages/06_Feature_Studio.py`

```python
def apply_differencing_if_needed(df, target_col, adf_pvalue, threshold=0.05):
    """
    Automatically difference target if ADF test indicates non-stationarity.

    Parameters:
    - df: DataFrame
    - target_col: Column to difference
    - adf_pvalue: P-value from ADF test (from 04_Explore_Data.py)
    - threshold: P-value threshold (0.05 = 95% confidence)

    Returns: (transformed_df, is_differenced, original_first_value)
    """
    if adf_pvalue > threshold:
        # Non-stationary: apply differencing
        original_first = df[target_col].iloc[0]
        df[target_col] = df[target_col].diff()
        df = df.dropna()
        return df, True, original_first
    return df, False, None


def inverse_differencing(predictions, is_differenced, original_first, last_actual):
    """
    Inverse transform differenced predictions back to original scale.

    Parameters:
    - predictions: Differenced predictions
    - is_differenced: Boolean flag
    - original_first: First value before differencing
    - last_actual: Last actual value before forecast period

    Returns: Predictions in original scale
    """
    if not is_differenced:
        return predictions

    # Cumulative sum starting from last actual value
    return np.cumsum(np.insert(predictions, 0, last_actual))[1:]
```

**UI Addition:**
```python
# Auto-detect from session state
adf_pvalue = st.session_state.get("adf_test_pvalue", 0.01)
is_stationary = adf_pvalue < 0.05

st.subheader("Stationarity Handling")
if is_stationary:
    st.success(f"Series is stationary (ADF p-value: {adf_pvalue:.4f})")
    apply_diff = st.checkbox("Apply differencing anyway", value=False)
else:
    st.warning(f"Series is NON-stationary (ADF p-value: {adf_pvalue:.4f})")
    apply_diff = st.checkbox("Apply differencing (recommended)", value=True)

st.session_state["apply_differencing"] = apply_diff
```

---

### **5. Implement Recursive Forecasting Strategy**

**Problem:** Only direct strategy implemented. Thesis needs strategy comparison.

**File:** `app_core/models/ml/recursive_strategy.py` (create new)

```python
import numpy as np
import pandas as pd

class RecursiveForecaster:
    """
    Recursive (autoregressive) multi-step forecaster.
    Trains single model for h=1, then iterates for h=2...H.
    """

    def __init__(self, base_model, horizon=7, lag_features=['ED_lag_1', 'ED_lag_2', 'ED_lag_7']):
        self.base_model = base_model
        self.horizon = horizon
        self.lag_features = lag_features
        self.fitted_model = None

    def fit(self, X_train, y_train_h1):
        """Train model to predict only Target_1 (one-step ahead)."""
        self.fitted_model = self.base_model.fit(X_train, y_train_h1)
        return self

    def predict(self, X_last_row, actual_history):
        """
        Generate H-step forecast recursively.

        Parameters:
        - X_last_row: Features for the last known time point
        - actual_history: Recent actual values for updating lag features

        Returns: Array of H predictions
        """
        predictions = []
        current_features = X_last_row.copy()
        history = list(actual_history)

        for h in range(self.horizon):
            # Predict one step
            pred = self.fitted_model.predict(current_features.reshape(1, -1))[0]
            predictions.append(pred)

            # Update history with prediction
            history.append(pred)

            # Update lag features for next iteration
            for lag_col in self.lag_features:
                lag_num = int(lag_col.split('_')[-1])
                if lag_col in current_features.index:
                    current_features[lag_col] = history[-(lag_num)]

        return np.array(predictions)


def compare_strategies(X_train, y_train, X_test, y_test, model_class, model_params):
    """
    Compare Direct vs Recursive strategies.

    Returns: Dict with metrics for each strategy
    """
    results = {}

    # DIRECT STRATEGY (existing approach)
    direct_preds = []
    for h in range(1, 8):
        model = model_class(**model_params)
        model.fit(X_train, y_train[f'Target_{h}'])
        direct_preds.append(model.predict(X_test))
    direct_preds = np.array(direct_preds).T

    # RECURSIVE STRATEGY
    recursive_model = RecursiveForecaster(model_class(**model_params), horizon=7)
    recursive_model.fit(X_train, y_train['Target_1'])
    recursive_preds = recursive_model.predict(X_test.iloc[-1], y_train['Target_1'].values[-14:])

    # Calculate metrics
    for name, preds in [('Direct', direct_preds), ('Recursive', recursive_preds)]:
        mae = np.abs(y_test.values - preds).mean()
        results[name] = {'MAE': mae, 'RMSE': np.sqrt(((y_test.values - preds)**2).mean())}

    return results
```

**UI Addition in `pages/08_Train_Models.py`:**
```python
st.subheader("Forecasting Strategy")
strategy = st.radio(
    "Select multi-horizon strategy:",
    ["Direct (separate model per horizon)", "Recursive (single model, iterative)"],
    help="Direct: Trains 7 models. Recursive: Trains 1 model, uses predictions as inputs."
)
st.session_state["forecast_strategy"] = "direct" if "Direct" in strategy else "recursive"
```

---

## II. Model-Specific Improvements

### **XGBoost Enhancements**

**File:** `app_core/models/ml/xgboost_pipeline.py`

#### **2.1 Trend-Residual Hybrid**
```python
from sklearn.linear_model import LinearRegression

class TrendXGBoostHybrid:
    """Detrend with linear regression, then XGBoost on residuals."""

    def __init__(self, xgb_params):
        self.trend_model = LinearRegression()
        self.xgb_model = xgb.XGBRegressor(**xgb_params)

    def fit(self, X, y):
        # Fit trend on time index
        t = np.arange(len(y)).reshape(-1, 1)
        self.trend_model.fit(t, y)
        trend = self.trend_model.predict(t)

        # Fit XGBoost on residuals
        residuals = y - trend
        self.xgb_model.fit(X, residuals)
        return self

    def predict(self, X, future_t):
        trend_pred = self.trend_model.predict(future_t.reshape(-1, 1))
        residual_pred = self.xgb_model.predict(X)
        return trend_pred + residual_pred
```

#### **2.2 Early Stopping Configuration**
```python
# In training function
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # Set high, early stopping will find optimal
    early_stopping_rounds=50,
    eval_metric='mae',
    **other_params
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# Actual trees used: xgb_model.best_iteration
```

#### **2.3 Benchmark Against LightGBM/CatBoost**
```python
# Add to model selection dropdown
model_options = {
    "XGBoost": xgb.XGBRegressor,
    "LightGBM": lgb.LGBMRegressor,
    "CatBoost": cb.CatBoostRegressor
}
```

---

### **ANN Enhancements**

**File:** `app_core/models/ml/ann_pipeline.py`

#### **2.4 Enhanced Regularization**
```python
from tensorflow.keras.regularizers import l1_l2

def build_regularized_ann(input_dim, layers=[64, 32], l1=0.001, l2=0.001, dropout=0.3):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim,
                    kernel_regularizer=l1_l2(l1=l1, l2=l2),
                    activation='relu'))
    model.add(Dropout(dropout))

    for units in layers[1:]:
        model.add(Dense(units, kernel_regularizer=l1_l2(l1=l1, l2=l2), activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

#### **2.5 Learning Curve Visualization**
```python
def plot_learning_curves(history):
    """Plot training vs validation loss for overfitting diagnosis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
    fig.update_layout(title='Learning Curves', xaxis_title='Epoch', yaxis_title='Loss')
    return fig

# In training: save history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
st.plotly_chart(plot_learning_curves(history))
```

---

### **LSTM Enhancements**

**File:** `app_core/models/ml/lstm_pipeline.py`

#### **2.6 Lookback Window in HPO**
```python
# Add to Optuna search space
def objective(trial):
    lookback = trial.suggest_int('lookback_window', 7, 60, step=7)  # 7, 14, 21, ..., 60
    lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
    # ... rest of hyperparameters
```

#### **2.7 GRU Benchmark Model**
```python
from tensorflow.keras.layers import GRU

def build_gru_model(input_shape, units=64, layers=2):
    model = Sequential()
    for i in range(layers):
        return_seq = (i < layers - 1)
        model.add(GRU(units, return_sequences=return_seq, input_shape=input_shape if i==0 else None))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
```

#### **2.8 LSTM with Attention**
```python
from tensorflow.keras.layers import Attention, Concatenate

def build_lstm_attention(input_shape, lstm_units=64):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

    # Self-attention
    attention = Attention()([lstm_out, lstm_out])

    # Combine
    concat = Concatenate()([lstm_out, attention])
    flat = Flatten()(concat)
    output = Dense(1)(flat)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model
```

---

## III. Hybrid Models & HPO

### **3.1 Stacking Generalization**

**File:** `app_core/models/ml/stacking.py` (create new)

```python
from sklearn.linear_model import Ridge

class ForecastStacker:
    """
    Meta-learner that combines base model predictions.

    Base models: SARIMAX, XGBoost, LSTM
    Meta model: Ridge regression
    """

    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models  # Dict of fitted models
        self.meta_model = meta_model or Ridge(alpha=1.0)

    def fit(self, X_train, y_train, X_val, y_val):
        # Generate base predictions on validation set
        base_preds = []
        for name, model in self.base_models.items():
            preds = model.predict(X_val)
            base_preds.append(preds)

        # Stack predictions as features for meta-model
        meta_features = np.column_stack(base_preds)
        self.meta_model.fit(meta_features, y_val)
        return self

    def predict(self, X_test):
        base_preds = [model.predict(X_test) for model in self.base_models.values()]
        meta_features = np.column_stack(base_preds)
        return self.meta_model.predict(meta_features)
```

---

### **3.2 HPO with Time Series Cross-Validation**

**File:** `pages/08_Train_Models.py` - Hyperparameter Tuning tab

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit

def create_cv_objective(model_class, X, y, n_splits=5):
    """Create Optuna objective with proper time series CV."""

    def objective(trial):
        # Sample hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = model_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = np.abs(y_val - preds).mean()
            scores.append(mae)

        return np.mean(scores)  # Average across all folds

    return objective

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(create_cv_objective(xgb.XGBRegressor, X, y), n_trials=100)
```

---

### **3.3 Optuna Visualization**

```python
import optuna.visualization as vis

def display_optuna_results(study):
    """Display Optuna study visualizations in Streamlit."""

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimization History")
        fig1 = vis.plot_optimization_history(study)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Parameter Importance")
        fig2 = vis.plot_param_importances(study)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Parameter Relationships")
    fig3 = vis.plot_slice(study)
    st.plotly_chart(fig3, use_container_width=True)

    # Best parameters
    st.success(f"Best MAE: {study.best_value:.4f}")
    st.json(study.best_params)
```

---

## Quick Reference: Files to Modify

| Prompt | Primary File(s) | Priority |
|--------|-----------------|----------|
| Time Series CV | `temporal_split.py`, `06/07/08_*.py` | CRITICAL |
| Fourier Features | `06_Feature_Studio.py` | HIGH |
| Lag/Rolling Features | `03_Prepare_Data.py` | HIGH |
| Naive Baselines | `app_core/models/baselines.py` (new) | HIGH |
| Stationarity Auto | `06_Feature_Studio.py` | MEDIUM |
| Recursive Strategy | `app_core/models/ml/recursive_strategy.py` (new) | MEDIUM |
| XGBoost Hybrid | `xgboost_pipeline.py` | MEDIUM |
| LSTM Attention | `lstm_pipeline.py` | LOW |
| Stacking | `app_core/models/ml/stacking.py` (new) | LOW |
| HPO with CV | `08_Train_Models.py` | HIGH |
