# CLAUDE.md - HealthForecast AI Codebase Guide

## Project Overview

**HealthForecast AI** is an enterprise-grade hospital resource planning and forecasting application built on Streamlit. It uses machine learning and operations research to predict Emergency Department (ED) patient arrivals and optimize hospital staffing and inventory management.

**Key Stats:**
- ~59,000 lines of Python code
- 13 Streamlit pages + Welcome entry point
- 18 backend packages in `app_core/`
- 8 forecasting models (ARIMA, SARIMAX, XGBoost, LSTM, ANN, 3 Hybrids)
- 2 MILP optimization solvers (Staff + Inventory)
- Seasonal Proportions for clinical category distribution

**Workflow:** Upload Data → Prepare → Explore → Train Models → Forecast → Optimize → Take Action

---

## Quick Start

```bash
# Recommended: Use the batch script (activates Python 3.11 with TensorFlow)
.\start_lstm.bat

# Manual run
.\forecast_env_py311\Scripts\activate
streamlit run Welcome.py
```

**Requirements:**
- Python 3.11 (TensorFlow compatibility - Python 3.13/3.14 will NOT work)
- Virtual environment: `forecast_env_py311`
- Supabase credentials in `.streamlit/secrets.toml`

---

## Project Structure

```
streamlit_forecast_scaffold/
├── Welcome.py                      # Entry point (764 lines) - landing page + auth
├── pages/                          # 13 Streamlit frontend pages (~28,300 lines)
│   ├── 01_Dashboard.py             # Executive dashboard (2,374 lines)
│   ├── 02_Upload_Data.py           # Data ingestion (785 lines)
│   ├── 03_Prepare_Data.py          # Data cleaning/fusion (1,386 lines)
│   ├── 04_Explore_Data.py          # EDA & visualization (1,193 lines)
│   ├── 05_Baseline_Models.py       # ARIMA/SARIMAX + Seasonal Proportions (3,747 lines)
│   ├── 06_Feature_Studio.py        # Feature engineering (649 lines)
│   ├── 07_Feature_Selection.py     # Automated feature selection (1,104 lines)
│   ├── 08_Train_Models.py          # ML model training - 4 tabs (8,171 lines)
│   ├── 09_Model_Results.py         # Performance comparison (2,164 lines)
│   ├── 10_Patient_Forecast.py      # 7-day forecasts (1,784 lines)
│   ├── 11_Staff_Planner.py         # Staff scheduling optimization (1,620 lines)
│   ├── 12_Supply_Planner.py        # Inventory optimization (1,712 lines)
│   └── 13_Action_Center.py         # AI recommendations (1,634 lines)
│
├── app_core/                        # Backend modules (~29,700 lines)
│   ├── ai/                          # LLM recommendation engine
│   │   └── recommendation_engine.py # Claude/GPT integration (41K)
│   ├── analytics/                   # Analysis modules
│   │   ├── eda_pipeline.py          # Exploratory data analysis
│   │   └── seasonal_proportions.py  # DOW × Monthly category distribution (25K)
│   ├── api/                         # External API connectors
│   │   ├── base_connector.py        # Abstract base class
│   │   ├── patient_connector.py     # Patient data API
│   │   ├── weather_connector.py     # Weather data API
│   │   ├── calendar_connector.py    # Calendar/holiday API
│   │   ├── reason_connector.py      # Clinical reason API
│   │   └── config_manager.py        # API configuration
│   ├── auth/                        # Authentication
│   │   ├── authentication.py        # Login/session management
│   │   └── navigation.py            # Page routing
│   ├── cache/                       # Caching layer
│   │   └── cache_manager.py         # In-memory + file caching
│   ├── classification/              # Classification models (legacy)
│   │   ├── config.py, features.py
│   │   ├── models_ann.py, models_lstm.py, models_xgb.py
│   │   └── evaluation.py
│   ├── data/                        # Data layer
│   │   ├── data_processing.py       # Core data transformations
│   │   ├── fusion.py                # Multi-source data fusion
│   │   ├── ingest.py                # File ingestion (CSV/Excel/Parquet)
│   │   ├── supabase_client.py       # Database client
│   │   ├── financial_service.py     # Cost calculations
│   │   ├── hospital_service.py      # Hospital data ops
│   │   ├── inventory_service.py     # Inventory data ops
│   │   ├── staff_scheduling_service.py # Staff data ops
│   │   └── results_storage_service.py  # Model results persistence
│   ├── errors/                      # Error handling
│   │   ├── exceptions.py            # Custom exceptions
│   │   └── handlers.py              # Error handlers
│   ├── logging/                     # Logging configuration
│   │   └── config.py                # Structured logging setup
│   ├── models/                      # Forecasting models
│   │   ├── arima_pipeline.py        # ARIMA implementation (50K)
│   │   ├── sarimax_pipeline.py      # SARIMAX implementation (47K)
│   │   ├── model_selection.py       # Auto model selection
│   │   └── ml/                      # Machine Learning models
│   │       ├── base_ml_pipeline.py  # Abstract ML base class
│   │       ├── xgboost_pipeline.py  # XGBoost regression
│   │       ├── lstm_pipeline.py     # LSTM neural network
│   │       ├── ann_pipeline.py      # Dense neural network
│   │       ├── lstm_xgb.py          # Hybrid: LSTM → XGBoost residuals
│   │       ├── lstm_sarimax.py      # Hybrid: LSTM → SARIMAX residuals
│   │       ├── lstm_ann.py          # Hybrid: LSTM → ANN residuals
│   │       ├── hybrid_diagnostics.py # Hybrid model analysis (46K)
│   │       ├── model_registry.py    # Model registration
│   │       └── utils.py, sequence_builder.py
│   ├── offline/                     # Offline mode support
│   │   ├── local_database.py        # SQLite fallback
│   │   ├── connection_manager.py    # Network detection
│   │   ├── cache_manager.py         # Offline caching
│   │   ├── sync_engine.py           # Data synchronization
│   │   └── unified_data_service.py  # Unified online/offline access
│   ├── optimization/                # Operations Research
│   │   ├── milp_solver.py           # Staff scheduling MILP (PuLP)
│   │   ├── inventory_optimizer.py   # Inventory MILP (28K)
│   │   ├── cost_parameters.py       # Cost configuration
│   │   └── solution_analyzer.py     # Solution analysis
│   ├── pipelines/                   # ML pipelines
│   │   ├── temporal_split.py        # Time-series train/test split
│   │   └── conformal_prediction.py  # Prediction intervals (28K)
│   ├── plots/                       # Visualization
│   │   └── multihorizon_results.py  # Multi-horizon charts
│   ├── services/                    # Business logic layer
│   │   ├── base_service.py          # Service base class
│   │   ├── data_service.py          # Data operations service
│   │   └── modeling_service.py      # Model training service
│   ├── state/                       # Session state management
│   │   ├── session.py               # Session utilities
│   │   └── typed_state.py           # Typed state containers (13K)
│   └── ui/                          # UI components
│       ├── theme.py                 # Global CSS/styling (17K)
│       ├── sidebar_brand.py         # Sidebar branding (15K)
│       ├── page_navigation.py       # Page nav component
│       ├── ml_components.py         # ML-specific UI widgets
│       ├── insight_components.py    # Dashboard insights (16K)
│       ├── effects.py               # Visual effects (11K)
│       ├── api_fetch_ui.py          # API data fetch UI
│       ├── results_storage_ui.py    # Results display UI
│       └── components.py, charts.py, model_config_panel.py
│
├── scripts/                         # Utility scripts
├── tests/                           # Pytest test suite
├── data/                            # Sample data files
├── pipeline_artifacts/              # Saved model artifacts
├── assets/                          # Static assets (images, etc.)
├── .streamlit/                      # Streamlit configuration
│   ├── config.toml                  # Theme settings
│   └── secrets.toml                 # Credentials (gitignored)
├── requirements.txt                 # Python dependencies
└── forecast_env_py311/              # Python 3.11 virtual environment
```

---

## Key Technologies

| Category | Technologies |
|----------|--------------|
| **Framework** | Streamlit (≥1.28.0), Python 3.11 |
| **Data** | Pandas, NumPy, Supabase (PostgreSQL) |
| **Statistical** | statsmodels (ARIMA/SARIMAX), pmdarima |
| **ML/DL** | TensorFlow/Keras (LSTM, ANN), XGBoost, scikit-learn, Optuna |
| **Optimization** | PuLP (MILP solver) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **AI/LLM** | Anthropic (Claude), OpenAI (GPT) |
| **Auth** | streamlit-authenticator, bcrypt |
| **Caching** | Streamlit cache, custom file cache |

---

## Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Upload    │───▶│   Prepare   │───▶│   Explore   │───▶│   Feature   │
│    Data     │    │    Data     │    │    Data     │    │   Studio    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Action    │◀───│   Staff/    │◀───│  Patient    │◀───│   Train     │
│   Center    │    │  Supply     │    │  Forecast   │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Upload** (`02_Upload_Data.py`): CSV/Excel/Parquet file ingestion
2. **Prepare** (`03_Prepare_Data.py`): Clean, fuse patient + weather + calendar data
3. **Explore** (`04_Explore_Data.py`): EDA, visualizations, data quality checks
4. **Features** (`06_Feature_Studio.py`): Generate ED lag features (ED_1...ED_7) and targets (Target_1...Target_7)
5. **Selection** (`07_Feature_Selection.py`): Automated feature importance analysis
6. **Train** (`05_Baseline_Models.py` + `08_Train_Models.py`): Statistical and ML models
7. **Forecast** (`10_Patient_Forecast.py`): 7-day ahead predictions by clinical category
8. **Optimize** (`11_Staff_Planner.py` + `12_Supply_Planner.py`): MILP optimization
9. **Act** (`13_Action_Center.py`): AI-powered recommendations

**Clinical Categories:**
- RESPIRATORY, CARDIAC, TRAUMA, GASTROINTESTINAL, INFECTIOUS, NEUROLOGICAL, OTHER

---

## Page Details

### 08_Train_Models.py (4 Tabs)
1. **Benchmarks** - Quick model comparison metrics
2. **Machine Learning** - XGBoost, LSTM, ANN training with seasonal proportions
3. **Hyperparameter Tuning** - Optuna-based optimization
4. **Hybrid Models** - LSTM+XGBoost, LSTM+SARIMAX, LSTM+ANN hybrids

### 05_Baseline_Models.py (Tabs)
- **Configure** - ARIMA/SARIMAX parameter setup
- **Train** - Model training with seasonal proportions toggle
- **Results** - Performance metrics and visualizations

---

## Important Patterns

### Session State
- Use `st.session_state` for all cross-page data persistence
- Typed state containers in `app_core/state/typed_state.py`
- Key states: `prepared_data`, `feature_engineering`, `arima_results`, `sarimax_results`, `ml_results`

### Service Layer
- Business logic in `app_core/services/` (DataService, ModelingService)
- Services wrap data operations and model training
- Base service pattern in `base_service.py`

### Model Pipeline Pattern
- Base class: `app_core/models/ml/base_ml_pipeline.py`
- All ML models inherit from base and implement `train()`, `predict()`
- Hybrid models combine stage-1 (LSTM) + stage-2 (residual correction)

### Seasonal Proportions
- Location: `app_core/analytics/seasonal_proportions.py`
- Calculates DOW × Monthly multiplicative factors for category distribution
- Config: `SeasonalProportionConfig` dataclass
- Applied in Baseline Models and ML training pages

### Theme/UI
- Global CSS in `app_core/ui/theme.py`
- Dark theme with glassmorphic cards and neon effects
- All pages call `apply_css()` on load
- Color constants: `PRIMARY_COLOR`, `SECONDARY_COLOR`, `SUCCESS_COLOR`, etc.

---

## Common Development Tasks

### Add New Page
1. Create `pages/NN_Page_Name.py` (NN = ordering number)
2. Import auth check: `from app_core.auth import require_authentication`
3. Import navigation: `from app_core.ui.page_navigation import render_page_navigation`
4. Apply theme: `from app_core.ui.theme import apply_css; apply_css()`
5. Streamlit auto-discovers pages alphabetically

### Add New Model
1. Create `app_core/models/ml/new_model_pipeline.py`
2. Inherit from `BaseMLPipeline`
3. Implement `train()`, `predict()`, `evaluate()` methods
4. Register in model selection UI in `08_Train_Models.py`

### Add Database Table
1. Create SQL schema in `scripts/`
2. Add service methods in `app_core/data/`
3. Update `supabase_client.py` if needed

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
2. **Secrets**: Never commit `.streamlit/secrets.toml`
3. **Model Artifacts**: Saved to `pipeline_artifacts/`
4. **Offline Mode**: Falls back to SQLite via `app_core/offline/` if Supabase unavailable
5. **Seasonal Proportions**: Enable in training pages to distribute forecasts by clinical category

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TensorFlow errors | Ensure Python 3.11 env is activated via `start_lstm.bat` |
| Supabase connection | Check `secrets.toml` configuration |
| Model training fails | Verify sufficient data rows (~50+), no constant columns |
| Import errors | Activate virtual environment first |
| Session state errors | Don't assign directly to widget-bound keys |
| Seasonal proportions missing | Ensure training data has date column and category data |

---

## Key File References

| Purpose | File |
|---------|------|
| Entry point | `Welcome.py` |
| Data processing | `app_core/data/data_processing.py` |
| Data fusion | `app_core/data/fusion.py` |
| ARIMA training | `app_core/models/arima_pipeline.py` |
| SARIMAX training | `app_core/models/sarimax_pipeline.py` |
| ML base class | `app_core/models/ml/base_ml_pipeline.py` |
| XGBoost model | `app_core/models/ml/xgboost_pipeline.py` |
| LSTM model | `app_core/models/ml/lstm_pipeline.py` |
| Hybrid models | `app_core/models/ml/lstm_xgb.py`, `lstm_sarimax.py`, `lstm_ann.py` |
| Staff optimization | `app_core/optimization/milp_solver.py` |
| Inventory optimization | `app_core/optimization/inventory_optimizer.py` |
| Seasonal proportions | `app_core/analytics/seasonal_proportions.py` |
| Theme/CSS | `app_core/ui/theme.py` |
| Session state | `app_core/state/typed_state.py` |
| Supabase client | `app_core/data/supabase_client.py` |
| AI recommendations | `app_core/ai/recommendation_engine.py` |
| Offline support | `app_core/offline/unified_data_service.py` |
