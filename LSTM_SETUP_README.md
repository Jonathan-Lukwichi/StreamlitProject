# LSTM Support Setup Complete ✓

## Environment Summary

Your new Python 3.11 virtual environment with full LSTM support has been successfully created!

### Installed Versions
- **Python**: 3.11.1
- **TensorFlow**: 2.15.1 ✓
- **XGBoost**: 3.1.1 ✓
- **Streamlit**: 1.51.0
- **Pandas**: 2.3.3
- **NumPy**: 1.26.4
- **Scikit-learn**: 1.7.2
- Plus all other required packages (statsmodels, pmdarima, plotly, prophet, etc.)

## How to Use

### Option 1: Quick Start (Recommended)
Simply double-click the batch file:
```
run_with_lstm.bat
```

### Option 2: Manual Activation
1. Open Command Prompt in this directory
2. Activate the environment:
   ```cmd
   forecast_env_py311\Scripts\activate
   ```
3. Run Streamlit:
   ```cmd
   streamlit run pages\01_Dashboard.py
   ```

### Option 3: Run from Modeling Hub
1. Navigate to: `pages/08_Modeling_Hub.py`
2. Select **LSTM** as your model type
3. Configure hyperparameters:
   - Lookback Window: 14 days (default)
   - LSTM Hidden Units: 64 (default)
   - Number of Layers: 2 (default)
   - Dropout: 0.2 (default)
   - Epochs: 100 (default)
   - Learning Rate: 0.001
   - Batch Size: 32
4. Click "Train Model"

## Features Available

### XGBoost Pipeline ✓
- Intelligent preprocessing detection
- Per-feature analysis
- Multi-horizon forecasting (Target_1 to Target_7)
- Hyperparameter tuning (Grid, Random, Bayesian search)
- Feature importance extraction

### LSTM Pipeline ✓
- Sequence-based time series architecture
- Per-feature preprocessing detection
- Automatic sequence generation (lookback window)
- Stacked LSTM layers with dropout
- Monte Carlo Dropout for confidence intervals
- Early stopping and learning rate scheduling
- Healthcare-specific optimizations

## Preprocessing Detection

Both pipelines now include intelligent preprocessing detection that:
1. **Checks each variable individually** (not just aggregate)
2. **Detects scaling type**:
   - MinMaxScaler [0,1]
   - MinMaxScaler [-1,1]
   - StandardScaler
   - Custom/Unknown
   - Unscaled
3. **Provides confidence levels**: HIGH, MEDIUM, LOW
4. **Auto-decides** whether to apply scaling based on:
   - XGBoost: Always skips (tree-based model doesn't need scaling)
   - LSTM: Applies only if needed (neural network requires scaling)

### Example Output:
```
🔍 LSTM PREPROCESSING DETECTION REPORT
======================================================================
Overall State: FULLY_SCALED
Recommendation: All features appear scaled - SKIP scaling
Scaled Features: 10/10
----------------------------------------------------------------------
✓ SKIPPING SCALING - data already preprocessed
======================================================================
```

## Troubleshooting

### TensorFlow Warnings
You may see warnings like:
```
WARNING:tensorflow:From ... The name tf.losses.sparse_softmax_cross_entropy is deprecated
```
These are **normal** and don't affect functionality. They're just deprecation notices.

### If LSTM Training Fails
1. Check that you're using the new environment (you should see `(forecast_env_py311)` in your prompt)
2. Verify TensorFlow is working:
   ```cmd
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```
3. Check error messages in console/terminal for detailed diagnostics

### Memory Issues
If you encounter memory errors with LSTM:
- Reduce `batch_size` (try 16 instead of 32)
- Reduce `lookback_window` (try 7 instead of 14)
- Reduce `lstm_hidden_units` (try 32 instead of 64)

## Why Python 3.11?
Your original Python 3.14 is too new for TensorFlow (which currently supports up to Python 3.12). Python 3.11 provides:
- Full TensorFlow 2.15 compatibility
- Better stability for data science packages
- Faster performance for numerical operations

## Next Steps
1. Run `run_with_lstm.bat` to start the application
2. Navigate to the Modeling Hub page
3. Try both XGBoost and LSTM models
4. Compare their performance on your healthcare time series data
5. Use the preprocessing detection reports to understand your data better

## File Locations
- Virtual Environment: `forecast_env_py311/`
- LSTM Pipeline: `app_core/models/ml/lstm_pipeline.py`
- XGBoost Pipeline: `app_core/models/ml/xgboost_pipeline.py`
- Modeling Hub: `pages/08_Modeling_Hub.py`
- Base Pipeline: `app_core/models/ml/base_ml_pipeline.py`

---
**Environment Created**: 2025-11-18
**TensorFlow Version**: 2.15.1 (Python 3.11 compatible)
**XGBoost Version**: 3.1.1
**Status**: ✓ Ready to Use
