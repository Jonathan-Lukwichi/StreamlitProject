@echo off
cd /d "%~dp0"
echo Starting Streamlit with Python 3.11 + TensorFlow + XGBoost...
echo.
forecast_env_py311\Scripts\python.exe -m streamlit run app.py
