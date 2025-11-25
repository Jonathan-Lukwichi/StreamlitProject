# Troubleshooting LSTM Environment

## If start_lstm.bat doesn't work, try these steps:

### Option 1: Use the Simple Batch File
Double-click: **RUN_STREAMLIT_LSTM.bat**

This is a simpler version that just runs Streamlit directly.

### Option 2: Run from Command Prompt
1. Open Command Prompt (cmd.exe)
2. Navigate to your project folder:
   ```cmd
   cd C:\Users\BIBINBUSINESS\OneDrive\Desktop\streamlit_forecast_scaffold
   ```
3. Run this command:
   ```cmd
   forecast_env_py311\Scripts\python.exe -m streamlit run pages\01_Dashboard.py
   ```

### Option 3: Manual Step-by-Step
1. Open Command Prompt
2. Change directory:
   ```cmd
   cd C:\Users\BIBINBUSINESS\OneDrive\Desktop\streamlit_forecast_scaffold
   ```
3. Test the environment:
   ```cmd
   forecast_env_py311\Scripts\python.exe --version
   ```
   Should show: `Python 3.11.1`

4. Test TensorFlow:
   ```cmd
   forecast_env_py311\Scripts\python.exe -c "import tensorflow; print(tensorflow.__version__)"
   ```
   Should show: `2.15.1`

5. Run Streamlit:
   ```cmd
   forecast_env_py311\Scripts\python.exe -m streamlit run pages\01_Dashboard.py
   ```

## Common Issues

### Issue 1: Batch file opens and closes immediately
**Solution**: Run it from Command Prompt so you can see the error message

### Issue 2: "forecast_env_py311 not found"
**Solution**: Make sure you're in the correct directory. The forecast_env_py311 folder should be in the same directory as the batch file.

### Issue 3: "module 'tensorflow' has no attribute '__version__'"
**Solution**: TensorFlow might not be fully installed. Run:
```cmd
forecast_env_py311\Scripts\python.exe -m pip install --upgrade tensorflow
```

### Issue 4: Still shows Python 3.14
**Solution**: You're not using the batch file correctly. Make sure to run one of these:
- RUN_STREAMLIT_LSTM.bat
- start_lstm.bat
- Or use the manual command from Option 2 above

### Issue 5: Streamlit opens but LSTM still fails
**Check**: Look at the terminal/console window. You should see:
```
🔍 LSTM PREPROCESSING DETECTION REPORT
```

If you see "No module named 'tensorflow'" instead, the batch file didn't work and you're still using Python 3.14.

## Verification Commands

Run these to verify your environment is working:

```cmd
REM Check Python version
forecast_env_py311\Scripts\python.exe --version

REM Check TensorFlow
forecast_env_py311\Scripts\python.exe -c "import tensorflow as tf; print('TF:', tf.__version__)"

REM Check XGBoost
forecast_env_py311\Scripts\python.exe -c "import xgboost as xgb; print('XGB:', xgb.__version__)"

REM Check Streamlit
forecast_env_py311\Scripts\python.exe -c "import streamlit as st; print('Streamlit:', st.__version__)"

REM Run test script
forecast_env_py311\Scripts\python.exe test_lstm_import.py
```

## Still Not Working?

If none of the above works, please provide:
1. The exact error message you see
2. The output of: `forecast_env_py311\Scripts\python.exe --version`
3. Whether the forecast_env_py311 folder exists in your project directory

## Quick Test
Before running Streamlit, verify everything works:
```cmd
forecast_env_py311\Scripts\python.exe test_lstm_import.py
```

You should see ">>> ALL TESTS PASSED <<<"
