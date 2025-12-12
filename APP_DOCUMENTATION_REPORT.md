# HealthForecast AI - Comprehensive Application Report

## Executive Summary

**HealthForecast AI** is an enterprise-grade hospital resource planning application built on Streamlit. It leverages machine learning and operations research to predict Emergency Department (ED) patient arrivals by clinical category and optimize staffing and inventory accordingly.

| Metric | Value |
|--------|-------|
| **Total Python Files** | ~95 application files |
| **Frontend Pages** | 13 Streamlit pages |
| **Backend Modules** | 12 core packages |
| **ML Models** | 6 forecasting models |
| **Optimization Solvers** | 2 MILP solvers |

---

## Table of Contents

1. [Application Architecture](#1-application-architecture)
2. [Backend Components](#2-backend-components)
3. [Frontend Pages](#3-frontend-pages)
4. [Data Flow & Integration](#4-data-flow--integration)
5. [Forecasting Models](#5-forecasting-models)
6. [Optimization Modules](#6-optimization-modules)
7. [Technical Stack](#7-technical-stack)
8. [Expert Recommendations](#8-expert-recommendations)

---

## 1. Application Architecture

### 1.1 High-Level Structure

```
streamlit_forecast_scaffold/
├── pages/                      # 13 Streamlit frontend pages
├── app_core/                   # Core backend modules
│   ├── api/                    # External API connectors
│   ├── analytics/              # EDA pipelines
│   ├── auth/                   # Authentication & navigation
│   ├── cache/                  # Caching system
│   ├── classification/         # Classification models
│   ├── data/                   # Data processing & services
│   ├── errors/                 # Exception handling
│   ├── logging/                # Logging configuration
│   ├── models/                 # Forecasting models
│   │   └── ml/                 # ML pipelines (XGBoost, LSTM, ANN)
│   ├── offline/                # Offline capabilities
│   ├── optimization/           # MILP solvers (staffing, inventory)
│   ├── pipelines/              # ML pipelines (conformal prediction)
│   ├── plots/                  # Visualization utilities
│   ├── services/               # Business logic services
│   ├── state/                  # Session state management
│   └── ui/                     # UI components & theming
└── forecast_env_py311/         # Python virtual environment
```

### 1.2 Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **MVC-like Separation** | Backend (app_core) vs Frontend (pages) |
| **Service Layer** | `app_core/services/` for business logic |
| **Pipeline Pattern** | ML pipelines with standardized interfaces |
| **State Management** | Centralized Streamlit session state |
| **Dependency Injection** | Configuration managers for API connectors |

---

## 2. Backend Components

### 2.1 Data Processing (`app_core/data/`)

#### 2.1.1 Data Processing Pipeline (`data_processing.py`)
- **Purpose**: Clean, transform, and engineer features from raw hospital data
- **Key Functions**:
  - `process_dataset()`: Main pipeline entry point
  - `_create_clinical_categories()`: Aggregates granular medical reasons into 6 clinical categories
  - `_generate_lags_and_targets()`: Creates ED lag features (ED_1..ED_7) and targets (Target_1..Target_7)
  - `_generate_multi_targets()`: Multi-target forecasting for clinical categories

#### Clinical Categories Mapping:
```python
CLINICAL_CATEGORIES = {
    "RESPIRATORY": ["asthma", "pneumonia", "shortness_of_breath"],
    "CARDIAC": ["chest_pain", "arrhythmia", "hypertensive_emergency"],
    "TRAUMA": ["fracture", "laceration", "burn", "fall_injury"],
    "GASTROINTESTINAL": ["abdominal_pain", "vomiting", "diarrhea"],
    "INFECTIOUS": ["flu_symptoms", "fever", "viral_infection"],
    "NEUROLOGICAL": ["headache", "dizziness"],
    "OTHER": ["allergic_reaction", "mental_health"],
}
```

#### 2.1.2 Data Fusion (`fusion.py`)
- Merges patient, weather, and calendar datasets
- Handles timezone normalization
- Joins on datetime index

#### 2.1.3 Data Services
- `staff_scheduling_service.py`: Staff optimization data service
- `inventory_service.py`: Inventory management data service
- `financial_service.py`: Financial calculations and ROI
- `supabase_client.py`: Cloud database connectivity
- `results_storage_service.py`: Persisting model results

### 2.2 Forecasting Models (`app_core/models/`)

#### 2.2.1 Statistical Models

**ARIMA Pipeline (`arima_pipeline.py`)**
- Single-horizon and multi-horizon ARIMA
- Automatic order selection via:
  - AIC-based grid search
  - Hybrid AIC + CV-RMSE optimization
  - pmdarima auto_arima integration
- Stationarity tests (ADF, KPSS)
- Residual diagnostics (Ljung-Box)
- Multi-target support for clinical categories

**SARIMAX Pipeline (`sarimax_pipeline.py`)**
- Seasonal ARIMA with exogenous variables
- Calendar features (day-of-week, weekend, yearly sin/cos)
- Weather feature integration
- Parameter sharing across horizons for efficiency
- Search modes: `aic_only`, `hybrid` (AIC + CV-RMSE)

#### 2.2.2 Machine Learning Models (`app_core/models/ml/`)

**XGBoost Pipeline (`xgboost_pipeline.py`)**
```python
class XGBoostPipeline(BaseMLPipeline):
    - Intelligent preprocessing detection
    - Grid search, random search, Bayesian optimization
    - Feature importance extraction
    - Tree-based (no scaling required)
```

**LSTM Pipeline (`lstm_pipeline.py`)**
- Deep learning for sequence modeling
- Configurable architecture (layers, units, dropout)
- Sequence builder for time series windowing
- Early stopping and learning rate scheduling

**ANN Pipeline (`ann_pipeline.py`)**
- Feedforward neural network
- Flexible hidden layer configuration
- Batch normalization and dropout

**Hybrid Models**:
- `lstm_sarimax.py`: LSTM + SARIMAX ensemble
- `lstm_xgb.py`: LSTM + XGBoost ensemble
- `lstm_ann.py`: LSTM + ANN ensemble

#### 2.2.3 Model Registry (`model_registry.py`)
- Centralized model registration
- Factory pattern for model instantiation
- Configuration management

### 2.3 Uncertainty Quantification (`app_core/pipelines/`)

**Conformal Prediction (`conformal_prediction.py`)**
- Conformalized Quantile Regression (CQR)
- Guaranteed coverage prediction intervals
- RPIW (Relative Prediction Interval Width) calculation
- Model ranking based on interval quality

```python
# Key formulas:
# Conformity Score: E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
# Coverage Guarantee: P(Y ∈ [lower, upper]) ≥ 1 - α
```

### 2.4 Optimization (`app_core/optimization/`)

#### 2.4.1 Staff Scheduling MILP (`milp_solver.py`)

**Mathematical Formulation:**
```
Minimize Z = Σ_t Σ_k (c_k · H · x[k,t])           # Regular labor
           + Σ_t Σ_k (c_k^o · o[k,t])             # Overtime
           + Σ_t Σ_c (w_c · c^u · u[c,t])         # Understaffing
           + Σ_t (c^v · v[t])                      # Overstaffing

Subject to:
- Demand satisfaction per clinical category
- Staff availability (min/max constraints)
- Overtime limits
- Nurse-to-doctor ratio (skill mix)
- Non-negativity and integrality
```

**Decision Variables:**
- `x[k,t]`: Number of staff type k on day t (Integer)
- `o[k,t]`: Overtime hours (Continuous)
- `u[c,t]`: Understaffing slack by category (Continuous)
- `v[t]`: Overstaffing slack (Continuous)

#### 2.4.2 Inventory Optimization (`inventory_optimizer.py`)

**Economic Order Quantity (EOQ):**
```
Q* = sqrt(2 * K * D / h)
Where:
  K = ordering cost per order
  D = annual demand
  h = holding cost per unit per year
```

**MILP Multi-Item Formulation:**
```
Minimize Z = Σ_t Σ_i [K_i · y_{i,t} + c_i · Q_{i,t} + h_i · I_{i,t} + p_i · B_{i,t}]

Subject to:
- Inventory balance: I_{i,t} = I_{i,t-1} + Q_{i,t-L} - d_{i,t} + B_{i,t} - B_{i,t-1}
- Storage capacity: Σ_i v_i · I_{i,t} <= W
- Order-quantity link: Q_{i,t} <= M · y_{i,t}
- Budget constraints
```

**Healthcare Item Catalog:**
- Examination Gloves
- PPE Kits
- Emergency Medications
- Syringes
- Bandages
- IV Fluids
- Surgical Masks

### 2.5 API Connectors (`app_core/api/`)

| Connector | Purpose | Data Source |
|-----------|---------|-------------|
| `patient_connector.py` | Patient arrival data | Supabase/CSV |
| `weather_connector.py` | Weather data | Supabase/CSV |
| `calendar_connector.py` | Calendar/holiday data | Supabase/CSV |
| `reason_connector.py` | Visit reason data | Supabase/CSV |
| `base_connector.py` | Abstract base class | - |
| `config_manager.py` | API configuration | Environment |

### 2.6 State Management (`app_core/state/`)

**Session Defaults:**
```python
SESSION_DEFAULTS = {
    "patient_data": None,
    "weather_data": None,
    "calendar_data": None,
    "merged_data": None,
    "processed_df": None,
    "patient_loaded": False,
    "model_results": None,
    "arima_results": None,
    "arima_params": {"p": 1, "d": 1, "q": 1},
}
```

### 2.7 Authentication (`app_core/auth/`)

- Role-based access control (Admin/User)
- Page-level authentication decorators
- `require_authentication()`: User or Admin access
- `require_admin_access()`: Admin-only pages

---

## 3. Frontend Pages

### 3.1 Page Overview

| # | Page | Purpose | Access |
|---|------|---------|--------|
| 01 | Dashboard | Executive overview, KPIs, trends | User/Admin |
| 02 | Data Hub | Upload/fetch patient, weather, calendar data | Admin |
| 03 | Data Preparation Studio | Data cleaning, feature engineering | Admin |
| 04 | EDA | Exploratory data analysis | User/Admin |
| 05 | Benchmarks | Baseline model comparison | Admin |
| 06 | Advanced Feature Engineering | Feature creation tools | Admin |
| 07 | Automated Feature Selection | Feature importance analysis | Admin |
| 08 | Modeling Hub | Train ARIMA, SARIMAX, XGBoost, LSTM, ANN | Admin |
| 09 | Results | Model comparison, metrics dashboard | User/Admin |
| 10 | Forecast | Multi-horizon predictions visualization | User/Admin |
| 11 | Staff Scheduling | MILP optimization for staffing | Admin |
| 12 | Inventory Management | EOQ/MILP inventory optimization | Admin |
| 13 | Hospital Command Center | Executive dashboard (plain language) | User/Admin |

### 3.2 Page Details

#### Page 08: Modeling Hub
**The core ML training interface with:**
- Model selection (ARIMA, SARIMAX, XGBoost, LSTM, ANN)
- Single-horizon and multi-horizon training
- Hyperparameter configuration panels
- Hyperparameter tuning (Grid, Random, Bayesian search)
- Train/validation/test split controls
- Progress tracking and logging

#### Page 09: Results & Model Comparison
- Comprehensive metrics table (MAE, RMSE, MAPE, R², RPIW)
- Model comparison charts
- Uncertainty analysis (conformal prediction intervals)
- Per-horizon breakdown
- Best model recommendation

#### Page 11: Staff Scheduling Optimization
- 7-day planning horizon
- Clinical category demand breakdown
- Cost parameters configuration
- MILP solver with CBC backend
- Schedule visualization (Gantt-style)
- Coverage analysis by category

#### Page 12: Inventory Management
- Patient forecast integration
- EOQ calculations per item
- MILP multi-item optimization
- Reorder alerts with urgency levels
- Storage capacity constraints
- Budget optimization

#### Page 13: Hospital Command Center
- **Redesigned for non-technical users**
- Plain language descriptions (no ML jargon)
- 7-day patient forecast visualization
- Action items with priority (Urgent/Important/Normal)
- Financial impact and ROI calculations
- Staffing and inventory status summaries
- Downloadable weekly planning report

---

## 4. Data Flow & Integration

### 4.1 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Supabase API  ─┐                                                       │
│  CSV Upload    ─┼─→ Data Hub (02) → Session State                       │
│  Manual Entry  ─┘                                                       │
└───────────────────────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Data Preparation Studio (03)                                           │
│  ├─ Merge patient + weather + calendar                                  │
│  ├─ Clinical category aggregation                                       │
│  ├─ Lag features (ED_1..ED_7)                                          │
│  ├─ Target features (Target_1..Target_7)                               │
│  └─ Calendar features (day_of_week, is_weekend, etc.)                  │
└───────────────────────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODELING                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Modeling Hub (08)                                                      │
│  ├─ Statistical: ARIMA, SARIMAX                                        │
│  ├─ Machine Learning: XGBoost, LSTM, ANN                               │
│  ├─ Hybrid: LSTM+SARIMAX, LSTM+XGBoost                                 │
│  └─ Hyperparameter tuning                                              │
└───────────────────────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESULTS & FORECASTING                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Results (09) ─→ Forecast (10)                                          │
│  ├─ Model comparison metrics                                           │
│  ├─ RPIW-based ranking                                                 │
│  ├─ Conformal prediction intervals                                     │
│  └─ Multi-horizon forecasts (h=1..7 days)                              │
└───────────────────────────────────────────┬─────────────────────────────┘
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     │                      │                      │
                     ▼                      ▼                      ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌─────────────────┐
│   STAFF SCHEDULING (11)   │ │ INVENTORY MANAGEMENT (12) │ │ COMMAND CENTER  │
├───────────────────────────┤ ├───────────────────────────┤ │      (13)       │
│ • 7-day staffing plan     │ │ • EOQ calculations        │ ├─────────────────┤
│ • MILP optimization       │ │ • Reorder alerts          │ │ • Plain language│
│ • Category coverage       │ │ • Safety stock levels     │ │ • Action items  │
│ • Cost minimization       │ │ • Budget optimization     │ │ • Financial ROI │
└───────────────────────────┘ └───────────────────────────┘ └─────────────────┘
```

### 4.2 Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `patient_data` | DataFrame | Raw patient arrival data |
| `weather_data` | DataFrame | Weather observations |
| `calendar_data` | DataFrame | Holiday/calendar data |
| `merged_data` | DataFrame | Combined dataset |
| `processed_df` | DataFrame | Feature-engineered data |
| `model_results` | Dict | ML model results |
| `arima_results` | Dict | ARIMA pipeline output |
| `sarimax_results` | Dict | SARIMAX pipeline output |
| `xgboost_results` | Dict | XGBoost pipeline output |
| `lstm_results` | Dict | LSTM pipeline output |
| `staff_optimization` | Dict | Staffing MILP results |
| `inventory_optimization` | Dict | Inventory MILP results |

---

## 5. Forecasting Models

### 5.1 Model Comparison Matrix

| Model | Type | Strengths | Weaknesses | Best For |
|-------|------|-----------|------------|----------|
| **ARIMA** | Statistical | Interpretable, fast, handles trends | No exogenous vars | Baseline, univariate |
| **SARIMAX** | Statistical | Seasonality, exogenous vars | Complex tuning | Weekly patterns, weather |
| **XGBoost** | ML (Tree) | Non-linear, feature importance | Black box | Complex relationships |
| **LSTM** | Deep Learning | Sequential patterns, long memory | Data hungry | Temporal dependencies |
| **ANN** | Deep Learning | Flexible, universal approximator | Overfitting | Non-linear mapping |
| **Hybrid** | Ensemble | Best of both worlds | Complexity | Production systems |

### 5.2 Multi-Horizon Architecture

```
Input: Historical data (t-7 to t)
       ↓
┌──────────────────────────────────┐
│    Feature Engineering           │
│  • ED_1..ED_7 (lag features)    │
│  • Calendar (DOW, weekend)       │
│  • Weather (temp, precip)        │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│         Model Training           │
│  One model per horizon OR        │
│  Direct multi-output             │
└────────────────┬─────────────────┘
                 │
                 ▼
Output: Target_1..Target_7 (h=1..7 day forecasts)
```

### 5.3 Metrics Framework

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Average absolute error |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Root mean square error |
| **MAPE** | mean(\|y - ŷ\|/y) × 100 | Percentage error |
| **Accuracy** | 100 - MAPE | Forecast accuracy % |
| **R²** | 1 - SS_res/SS_tot | Explained variance |
| **RPIW** | mean(width)/range(y) | Interval tightness |
| **CI Coverage** | % within bounds | Interval reliability |
| **Direction Accuracy** | % correct trend | Trend prediction |

---

## 6. Optimization Modules

### 6.1 Staff Scheduling

**Objective**: Minimize total staffing cost while meeting patient demand by clinical category.

**Staff Types:**
- Doctors (highest cost)
- Nurses (medium cost)
- Support Staff (lowest cost)

**Category Priorities:**
- CARDIAC: 1.5x (highest priority)
- TRAUMA: 1.3x
- RESPIRATORY: 1.2x
- Others: 1.0x

**Cost Components:**
1. Regular labor: hourly_rate × shift_length × staff_count
2. Overtime: overtime_rate × overtime_hours
3. Understaffing penalty: penalty × unmet_demand × category_weight
4. Overstaffing penalty: penalty × excess_capacity

### 6.2 Inventory Management

**RPIW Decision Tree:**
```
RPIW < 15%  → Deterministic EOQ/MILP (low uncertainty)
15% ≤ RPIW ≤ 40% → Robust (s,S) Policy (medium uncertainty)
RPIW > 40% → Stochastic (s,S) Policy (high uncertainty)
```

**Healthcare Item Parameters:**
- Unit cost (USD)
- Holding cost rate (annual %)
- Ordering cost (per order)
- Stockout penalty (per unit) - HIGH for healthcare
- Lead time (days)
- Usage rate (per patient)
- Shelf life (days)
- Criticality level (LOW/MEDIUM/HIGH/CRITICAL)

---

## 7. Technical Stack

### 7.1 Core Technologies

| Layer | Technology | Version |
|-------|------------|---------|
| **Frontend** | Streamlit | Latest |
| **Backend** | Python | 3.11 |
| **Database** | Supabase (PostgreSQL) | Cloud |
| **Caching** | Disk-based (Pickle) | Custom |

### 7.2 Data Science Libraries

| Category | Libraries |
|----------|-----------|
| **Data Processing** | pandas, numpy |
| **Statistical Models** | statsmodels, pmdarima |
| **Machine Learning** | scikit-learn, xgboost |
| **Deep Learning** | TensorFlow/Keras |
| **Optimization** | PuLP (CBC solver), scipy |
| **Visualization** | Plotly, Altair |

### 7.3 External Services

- **Supabase**: Cloud database for patient/weather/calendar data
- **Authentication**: Streamlit-Authenticator (bcrypt)

---

## 8. Expert Recommendations

### 8.1 Data Science Perspective

#### Strengths:
1. **Multi-model architecture** - Comprehensive model zoo covering statistical and ML approaches
2. **Conformal prediction** - Rigorous uncertainty quantification with coverage guarantees
3. **Multi-target forecasting** - Clinical category disaggregation improves planning granularity
4. **Hybrid optimization** - AIC + CV-RMSE balances complexity and predictive accuracy

#### Recommendations:

**1. Model Ensemble Implementation**
```python
# Weighted ensemble based on RPIW ranking
final_forecast = w1*arima + w2*sarimax + w3*xgboost + w4*lstm
# where weights inversely proportional to RPIW
```

**2. Feature Store Integration**
- Implement a feature store for consistent feature engineering across models
- Enable feature versioning for experiment tracking

**3. Online Learning Capability**
- Add incremental learning for XGBoost when new data arrives
- Implement concept drift detection using statistical tests

**4. Probabilistic Forecasting**
- Add quantile regression forests for native uncertainty
- Implement NGBoost for probabilistic gradient boosting

### 8.2 Industrial Engineering Perspective

#### Strengths:
1. **MILP formulations** - Mathematically rigorous optimization models
2. **Multi-objective consideration** - Balances cost, coverage, and quality
3. **Clinical category integration** - Operational relevance for resource planning

#### Recommendations:

**1. Stochastic Programming Extension**
```
min E[cost] + ρ × CVaR[understaffing]
```
- Add two-stage stochastic programming for demand uncertainty
- Implement chance constraints for service level guarantees

**2. Rolling Horizon Planning**
- Implement receding horizon control for dynamic replanning
- Add scenario-based robust optimization

**3. Multi-Shift Scheduling**
- Extend model to handle day/evening/night shifts
- Add nurse fatigue constraints (max consecutive shifts)

**4. Operating Room Integration**
- Link ED forecasts to OR scheduling
- Add patient flow simulation (queuing models)

### 8.3 Forecasting Expert Perspective

#### Strengths:
1. **Multi-horizon direct strategy** - Avoids error accumulation from recursive forecasting
2. **Exogenous variables** - Weather and calendar effects captured
3. **Automatic order selection** - pmdarima + hybrid CV reduces manual tuning

#### Recommendations:

**1. Hierarchical Forecasting**
- Implement hierarchical reconciliation (top-down/bottom-up/optimal)
- Forecast at hospital → department → clinical category levels

**2. External Data Enrichment**
- Integrate Google Trends for flu/respiratory surge detection
- Add local event calendars (concerts, sports) for trauma prediction

**3. Temporal Fusion Transformers**
- Implement TFT for interpretable deep learning forecasts
- Attention mechanism for identifying key drivers

**4. Forecast Combination**
- Implement BMA (Bayesian Model Averaging)
- Add forecast encompassing tests

### 8.4 Hospital Management Perspective (20+ Years Experience)

#### Strengths:
1. **Clinical category focus** - Aligns with departmental resource planning
2. **7-day horizon** - Matches nursing schedule cycles
3. **Plain language dashboard** - Accessible to non-technical managers
4. **Financial ROI tracking** - Demonstrates value to administration

#### Recommendations:

**1. Acuity Level Integration**
- Add ESI (Emergency Severity Index) levels 1-5
- Different staffing ratios by acuity
- Length of stay predictions by acuity

**2. Admission Forecasting**
- Extend from ED arrivals to admission predictions
- Link to bed management systems
- Predict ICU/ward transfers

**3. Real-Time Dashboard**
- Current census vs. forecast comparison
- Variance alerts when actuals deviate significantly
- Mobile-friendly interface for on-call managers

**4. Scenario Planning Tools**
- "What-if" analysis for surge events (pandemic, MCI)
- Cost impact of different staffing scenarios
- Sensitivity analysis on key parameters

**5. Performance Benchmarking**
- Compare to peer hospitals
- Track forecast accuracy trends over time
- Set and monitor KPIs for planning effectiveness

**6. Physician Buy-In Features**
- Add predicted case mix by specialty
- Resident scheduling integration
- On-call coverage optimization

### 8.5 Technical Debt & Improvements

| Area | Current State | Recommended Improvement |
|------|--------------|------------------------|
| **Testing** | No test files found | Add pytest suite, 80%+ coverage |
| **CI/CD** | Manual deployment | GitHub Actions pipeline |
| **Logging** | Basic logging | Structured logging (JSON) |
| **Monitoring** | None | Add Prometheus/Grafana |
| **Documentation** | Code comments | Sphinx API docs |
| **Error Handling** | Basic exceptions | Comprehensive error hierarchy |
| **Configuration** | Mixed approaches | Centralized config (Pydantic) |
| **Type Hints** | Partial | Full typing with mypy |

### 8.6 Roadmap Priorities

**Phase 1 (Immediate):**
- [ ] Add automated testing framework
- [ ] Implement model versioning
- [ ] Add data validation layer

**Phase 2 (Short-term):**
- [ ] Ensemble model implementation
- [ ] Real-time dashboard updates
- [ ] Mobile-responsive design

**Phase 3 (Medium-term):**
- [ ] Hierarchical forecasting
- [ ] Two-stage stochastic optimization
- [ ] Admission prediction module

**Phase 4 (Long-term):**
- [ ] Temporal Fusion Transformer integration
- [ ] Hospital-wide resource optimization
- [ ] Multi-facility planning

---

## Appendix A: File Index

### Backend Files (95 total)

**Core Modules:**
- `app_core/__init__.py`
- `app_core/constants.py`
- `app_core/routing.py`

**Data Processing (12 files):**
- `app_core/data/data_processing.py`
- `app_core/data/fusion.py`
- `app_core/data/ingest.py`
- `app_core/data/io.py`
- `app_core/data/utils.py`
- `app_core/data/supabase_client.py`
- `app_core/data/staff_scheduling_service.py`
- `app_core/data/inventory_service.py`
- `app_core/data/financial_service.py`
- `app_core/data/results_storage_service.py`

**Models (20 files):**
- `app_core/models/arima_pipeline.py`
- `app_core/models/sarimax_pipeline.py`
- `app_core/models/model_selection.py`
- `app_core/models/ml/base.py`
- `app_core/models/ml/base_ml_pipeline.py`
- `app_core/models/ml/xgboost_pipeline.py`
- `app_core/models/ml/lstm_pipeline.py`
- `app_core/models/ml/ann_pipeline.py`
- `app_core/models/ml/xgb_model.py`
- `app_core/models/ml/lstm_model.py`
- `app_core/models/ml/ann_model.py`
- `app_core/models/ml/sequence_builder.py`
- `app_core/models/ml/model_registry.py`
- `app_core/models/ml/hybrid_utils.py`
- `app_core/models/ml/lstm_sarimax.py`
- `app_core/models/ml/lstm_xgb.py`
- `app_core/models/ml/lstm_ann.py`
- `app_core/models/ml/utils.py`

**Optimization (4 files):**
- `app_core/optimization/milp_solver.py`
- `app_core/optimization/inventory_optimizer.py`
- `app_core/optimization/cost_parameters.py`
- `app_core/optimization/solution_analyzer.py`

**Pipelines (2 files):**
- `app_core/pipelines/conformal_prediction.py`
- `app_core/pipelines/temporal_split.py`

### Frontend Pages (13 files)

1. `pages/01_Dashboard.py`
2. `pages/02_Data_Hub.py`
3. `pages/03_Data_Preparation_Studio.py`
4. `pages/04_EDA.py`
5. `pages/05_Benchmarks.py`
6. `pages/06_Advanced_Feature_Engineering.py`
7. `pages/07_Automated_Feature_Selection.py`
8. `pages/08_Modeling_Hub.py`
9. `pages/09_Results.py`
10. `pages/10_Forecast.py`
11. `pages/11_Staff_Scheduling_Optimization.py`
12. `pages/12_Inventory_Management_Optimization.py`
13. `pages/13_Decision_Command_Center.py`

---

## Appendix B: Key Formulas

### Forecasting Metrics

**Mean Absolute Error (MAE):**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE):**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Mean Absolute Percentage Error (MAPE):**
$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Relative Prediction Interval Width (RPIW):**
$$RPIW = \frac{\bar{w}}{y_{max} - y_{min}}$$

### Optimization

**Economic Order Quantity (EOQ):**
$$Q^* = \sqrt{\frac{2KD}{h}}$$

**Reorder Point with Safety Stock:**
$$s = \bar{d} \cdot L + z_{CSL} \cdot \sigma_d \cdot \sqrt{L}$$

---

*Report generated: December 2024*
*Version: 1.0*
*Author: Claude AI Analysis*
