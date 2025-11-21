@echo off
echo ========================================
echo Starting Streamlit with LSTM Support
echo ========================================
echo.

REM Change to the project directory
cd /d "%~dp0"

echo [1/3] Checking Python 3.11 environment...
if not exist "forecast_env_py311\Scripts\python.exe" (
    echo ERROR: forecast_env_py311 not found!
    echo Please make sure the virtual environment exists.
    pause
    exit /b 1
)

echo [2/3] Verifying TensorFlow installation...
forecast_env_py311\Scripts\python.exe -c "import tensorflow as tf; import xgboost as xgb; print('Python 3.11.1'); print('TensorFlow:', tf.__version__); print('XGBoost:', xgb.__version__)" 2>nul
if errorlevel 1 (
    echo ERROR: TensorFlow not found in environment!
    echo Please run: forecast_env_py311\Scripts\python.exe -m pip install tensorflow xgboost
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Streamlit with Python 3.11...
echo.
echo ========================================
echo Your browser should open automatically.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.

REM Run Streamlit using the virtual environment's Python directly
forecast_env_py311\Scripts\python.exe -m streamlit run app.py

pause
