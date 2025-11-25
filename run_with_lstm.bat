@echo off
REM Activate Python 3.11 virtual environment with TensorFlow & XGBoost
REM Run this batch file to start your Streamlit app with LSTM support

echo ========================================
echo Starting Streamlit with LSTM Support
echo ========================================
echo.
echo Environment: Python 3.11 + TensorFlow 2.15.1 + XGBoost 3.1.1
echo.

REM Activate the virtual environment
call forecast_env_py311\Scripts\activate.bat

REM Run Streamlit
streamlit run pages\01_Dashboard.py

pause
