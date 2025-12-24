# CLAUDE.md - HealthForecast AI Codebase Guide

## Project Overview

**HealthForecast AI** is an enterprise-grade hospital resource planning and forecasting application built on Streamlit. It uses machine learning and operations research to predict Emergency Department (ED) patient arrivals and optimize hospital staffing and inventory management.

**Key Stats:**
- ~54,000 lines of Python code
- 13 Streamlit pages
- 12+ backend packages
- 6+ forecasting models (ARIMA, SARIMAX, XGBoost, LSTM, ANN, Hybrids)
- 2 MILP optimization solvers (Staff + Inventory)

**Workflow:** Upload Data → Prepare → Train Models → Forecast → Optimize → Take Action

---

## Quick Start

```bash
# Recommended: Use the batch script (activates Python 3.11 with TensorFlow)
.\start_lstm.bat

# Manual run
.\forecast_env_py311\Scripts\activate
streamlit run Welcome.py

# Default login: admin / admin123
```

**Requirements:**
- Python 3.11 (TensorFlow compatibility - Python 3.14 will NOT work)
- Virtual environment: `forecast_env_py311`
- Supabase credentials in `.streamlit/secrets.toml`

---

## Project Structure

```
streamlit_forecast_scaffold/
├── Welcome.py                    # Entry point (landing page + auth)
├── pages/                        # 13 Streamlit frontend pages
│   ├── 01_Dashboard.py          # Executive dashboard
│   ├── 02_Upload_Data.py        # Data ingestion
│   ├── 03_Prepare_Data.py       # Data cleaning/fusion
│   ├── 04_Explore_Data.py       # EDA
│   ├── 05_Baseline_Models.py    # ARIMA/SARIMAX training
│   ├── 06_Feature_Studio.py     # Feature engineering
│   ├── 07_Feature_Selection.py  # Automated feature selection
│   ├── 08_Train_Models.py       # ML model training
│   ├── 09_Model_Results.py      # Performance comparison
│   ├── 10_Patient_Forecast.py   # 7-day forecasts
│   ├── 11_Staff_Planner.py      # Staff scheduling optimization
│   ├── 12_Supply_Planner.py     # Inventory optimization
│   └── 13_Action_Center.py      # Recommendations
│
├── app_core/                     # Backend modules
│   ├── auth/                    # Authentication
│   ├── ui/                      # UI components, theme, charts
│   ├── data/                    # Data processing, services
│   ├── models/                  # Forecasting models
│   │   ├── arima_pipeline.py
│   │   ├── sarimax_pipeline.py
│   │   └── ml/                  # ML models (XGBoost, LSTM, ANN, Hybrids)
│   ├── optimization/            # MILP solvers (PuLP)
│   ├── ai/                      # LLM recommendation engine
│   ├── services/                # Business logic layer
│   ├── pipelines/               # Temporal split, conformal prediction
│   ├── state/                   # Session state management
│   ├── offline/                 # Offline/SQLite fallback
│   └── api/                     # External API connectors
│
├── scripts/                      # Database setup and utility scripts
├── tests/                        # Pytest unit and integration tests
├── .streamlit/config.toml        # Streamlit theme config
├── requirements.txt              # Python dependencies
└── forecast_env_py311/           # Python 3.11 virtual environment
```

---

## Key Technologies

| Category | Technologies |
|----------|--------------|
| **Framework** | Streamlit (≥1.28.0), Python 3.11 |
| **Data** | Pandas, NumPy, Supabase (PostgreSQL) |
| **Statistical** | statsmodels (ARIMA/SARIMAX), pmdarima, Prophet |
| **ML/DL** | TensorFlow (LSTM, ANN), XGBoost, scikit-learn, Optuna |
| **Optimization** | PuLP (MILP solver) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **AI/LLM** | Anthropic (Claude), OpenAI (GPT) |
| **Auth** | streamlit-authenticator, bcrypt |

---

## Data Flow

1. **Upload** (`02_Upload_Data.py`): CSV/Excel/Parquet file ingestion
2. **Prepare** (`03_Prepare_Data.py`): Clean, fuse patient + weather + calendar data
3. **Features**: Generate ED lag features (ED_1...ED_7) and targets (Target_1...Target_7)
4. **Train**: Statistical models (ARIMA/SARIMAX) or ML models (XGBoost/LSTM/ANN/Hybrids)
5. **Forecast**: 7-day ahead predictions by clinical category
6. **Optimize**: MILP solver for staff scheduling and inventory planning
7. **Act**: AI-powered recommendations in Action Center

**Clinical Categories:**
- Respiratory, Cardiac, Trauma, Gastrointestinal, Infectious, Neurological, Other

---

## Important Patterns

### Session State
- Use `st.session_state` for all cross-page data persistence
- Typed state containers in `app_core/state/typed_state.py`

### Service Layer
- Business logic in `app_core/services/` (DataService, ModelingService)
- Services wrap data operations and model training

### Model Pipeline Pattern
- Base class: `app_core/models/ml/base_ml_pipeline.py`
- All ML models inherit from base and implement `train()`, `predict()`

### Theme/UI
- Global CSS in `app_core/ui/theme.py`
- Dark theme with glassmorphic cards and neon effects
- All pages call `apply_css()` on load

---

## Common Development Tasks

### Add New Page
1. Create `pages/NN_Page_Name.py` (NN = ordering number)
2. Import auth check and navigation
3. Apply theme with `apply_css()`
4. Streamlit auto-discovers pages

### Add New Model
1. Create `app_core/models/ml/new_model_pipeline.py`
2. Inherit from `BaseMLPipeline`
3. Implement `train()` and `predict()` methods
4. Register in `08_Train_Models.py`

### Add Database Table
1. Create SQL schema in `scripts/create_table.sql`
2. Create setup script in `scripts/setup_table.py`
3. Add service in `app_core/data/table_service.py`

### Run Tests
```bash
pytest tests/
pytest --cov=app_core tests/
```

---

## Configuration Files

### `.streamlit/secrets.toml` (required, gitignored)
```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"

[llm]
provider = "anthropic"
api_key = "your-api-key"
```

### `.streamlit/config.toml` (theme)
```toml
[theme]
base = "dark"
primaryColor = "#667eea"
backgroundColor = "#0a0e27"
```

---

## Important Notes

1. **Python Version**: Must use Python 3.11 for TensorFlow/LSTM support
2. **Authentication**: Current auth is prototype-only (demo credentials). Use Supabase Auth for production
3. **Secrets**: Never commit `.streamlit/secrets.toml`
4. **Model Artifacts**: Saved to `pipeline_artifacts/`
5. **Offline Mode**: Falls back to SQLite if Supabase unavailable

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TensorFlow errors | Ensure Python 3.11 env is activated via `start_lstm.bat` |
| Supabase connection | Check `secrets.toml` configuration |
| Model training fails | Verify sufficient data rows (~50+), no constant columns |
| Import errors | Activate virtual environment first |

---

## Key File References

| Purpose | File |
|---------|------|
| Entry point | `Welcome.py` |
| Data processing | `app_core/data/data_processing.py` |
| ARIMA training | `app_core/models/arima_pipeline.py` |
| ML models | `app_core/models/ml/*.py` |
| MILP optimization | `app_core/optimization/milp_solver.py` |
| Theme/CSS | `app_core/ui/theme.py` |
| Session state | `app_core/state/session.py` |
| Supabase client | `app_core/data/supabase_client.py` |
