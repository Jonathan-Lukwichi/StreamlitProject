# Project Improvement Suggestions as Direct Prompts

This document outlines key improvements for the HealthForecast AI application, framed as direct instructions. Suggestions are **prioritized for prototype validation phase** before production deployment.

---

## System Architecture Overview

![ML Pipeline Architecture](assets/ml_pipeline_architecture.png)

### Architecture Description

The diagram above illustrates the core machine learning pipeline architecture for HealthForecast AI:

**Human Oversight Layer**
- Human design choices, engineering, and oversight govern the entire system
- Ensures responsible AI deployment with human-in-the-loop decision making

**Training Pipeline**
- **Training Data**: Historical patient arrival data, weather, calendar features
- **Machine Learning**: Model training processes (ARIMA, SARIMAX, XGBoost, LSTM, ANN, Hybrids)
- **Model**: Trained artifacts stored for inference

**Continuous Learning Loop**
- Models are continuously refined as new data becomes available
- Feedback from production outcomes improves future predictions

**Inference Pipeline**
- **Inputs**: Production data (real-time patient arrivals) and contextual information
- **Processing**: Model inference engine applies trained models to inputs
- **Outputs**:
  - **Predictions**: 7-day patient arrival forecasts by clinical category
  - **Actions**: Automated staffing schedule adjustments
  - **Recommendations**: AI-generated operational suggestions
  - **Decisions**: Inventory reorder triggers and resource allocation

---

## Priority Legend

| Priority | Meaning | Timeline |
|----------|---------|----------|
| **P0** | Critical - Prototype broken without this | Immediate |
| **P1** | High - Required for prototype validation | This sprint |
| **P2** | Medium - Important for demo readiness | Next sprint |
| **P3** | Low - Nice to have for prototype | Future |
| **P4** | Deferred - Production phase only | Post-validation |

---

## P0: Critical Fixes (Prototype Broken)

### P0.1 Fix Category Forecast Data Flow
**Prompt:** "Fix the broken data flow between forecast and optimization pages. In `pages/10_Patient_Forecast.py`, modify the `save_forecast_to_session()` function to persist `category_forecasts_by_horizon` to `st.session_state['category_forecasts']`. Then update `pages/11_Staff_Planner.py` to read category distributions from `st.session_state['category_forecasts']` instead of using the hardcoded distribution dictionary at lines 584-592. Similarly update `pages/12_Supply_Planner.py` to consume the dynamic category forecasts. Ensure the Seasonal Proportions calculations flow through the entire pipeline from training to optimization."

### P0.2 Implement Backtest vs. Live Forecast Modes
**Prompt:** "Refactor the 'Patient Forecast' page (`pages/10_Patient_Forecast.py`) to have two distinct modes: 'Backtest Evaluation' and 'Generate Live Forecast'. Add a mode selector at the top of the page. The 'Backtest' mode will show model performance on historical test data (current functionality) with walk-forward validation metrics. The 'Generate Live Forecast' mode will add a new button that, when clicked, re-trains the selected model on 100% of the available data and generates a true forecast for the next 7 days with no actuals comparison. Display a clear warning that Live mode forecasts cannot be evaluated until actuals arrive."

### P0.3 Implement Walk-Forward Validation
**Prompt:** "Create a robust backtesting framework in `app_core/pipelines/walk_forward_validation.py`. Implement walk-forward (rolling window) cross-validation that respects time-series ordering. The function should: (1) Take a dataset and model pipeline as input, (2) Train on expanding or sliding windows, (3) Forecast the next 7 days, (4) Compare against actuals, (5) Return per-fold metrics (MAE, RMSE, MAPE) and aggregated statistics. Integrate this into the Model Results page (`pages/09_Model_Results.py`) with a 'Run Backtest' button that displays fold-by-fold performance and overall validation metrics."

---

## P1: High Priority (Prototype Validation)

### P1.1 Implement Automated Testing
**Prompt:** "Create a comprehensive test suite for the `app_core` modules using `pytest`. Write unit tests for all public functions in the service, data processing, and model pipeline files. Focus first on: (1) `app_core/data/data_processing.py` - test data transformations, (2) `app_core/analytics/seasonal_proportions.py` - test DOW/monthly calculations, (3) `app_core/models/arima_pipeline.py` - test model training and prediction. Ensure the final test coverage for the `app_core` directory exceeds 80% and that the tests are added to the `tests/unit` and `tests/integration` directories. Use fixtures for sample data and mock external dependencies."

### P1.2 Set Up CI/CD Pipeline
**Prompt:** "Set up a GitHub Actions workflow for this project. The workflow should be triggered on every push to the `main` branch and on all pull requests. It must: (1) Set up Python 3.11 environment, (2) Install dependencies from `requirements.txt`, (3) Run linting with `flake8`, (4) Run the entire `pytest` test suite with coverage reporting, (5) Fail the build if coverage drops below 70% or any tests fail. Create the workflow file at `.github/workflows/ci.yml`. Add a badge to README.md showing build status."

### P1.3 Refactor Train Models Page (Code Quality)
**Prompt:** "Refactor the monolithic `pages/08_Train_Models.py` (8,171 lines) into smaller, maintainable modules. Create a new directory `pages/train_models/` with: (1) `__init__.py` - exports main page function, (2) `benchmarks.py` - Benchmarks tab logic (~1,500 lines), (3) `ml_training.py` - Machine Learning tab logic (~2,500 lines), (4) `hyperparameter_tuning.py` - Hyperparameter Tuning tab logic (~2,000 lines), (5) `hybrid_models.py` - Hybrid Models tab logic (~2,000 lines), (6) `shared_utils.py` - Common functions used across tabs. Update `pages/08_Train_Models.py` to import from these modules and orchestrate the tabs. Ensure all session state keys and widget keys remain consistent."

### P1.4 Implement Hierarchical Forecasting with Reconciliation
**Prompt:** "Implement hierarchical forecasting for patient arrivals in `app_core/pipelines/hierarchical_forecast.py`. The pipeline should: (1) Generate a forecast for total patient volume using the selected model, (2) Generate separate forecasts for each clinical category (RESPIRATORY, CARDIAC, TRAUMA, GASTROINTESTINAL, INFECTIOUS, NEUROLOGICAL, OTHER), (3) Apply a reconciliation step using the 'top-down proportions' method to ensure category forecasts sum to total, (4) Optionally support 'middle-out' and 'bottom-up' reconciliation methods. Add a 'Hierarchical Forecasting' toggle to the Patient Forecast page that enables this pipeline. Display both reconciled and unreconciled forecasts for comparison."

---

## P2: Medium Priority (Demo Readiness)

### P2.1 Implement SHAP Explainability for Models
**Prompt:** "Integrate SHAP (SHapley Additive exPlanations) to provide explainability for ML models. In `app_core/models/ml/`, add a new file `explainability.py` with functions to: (1) Create SHAP explainers for XGBoost and tree-based models, (2) Generate feature importance plots, (3) Create force plots for individual predictions, (4) Create summary plots showing global feature impact. In `pages/09_Model_Results.py`, add a new 'Model Explanation' tab that displays: (1) Global feature importance bar chart, (2) SHAP summary beeswarm plot, (3) Interactive force plot for user-selected forecast dates. For hybrid models, explain the stage-2 XGBoost residual correction."

### P2.2 Add Patient Acuity (ESI) Integration
**Prompt:** "Extend the data model to include patient acuity scores (ESI levels 1-5). In `app_core/data/data_processing.py`, add functions to: (1) Parse ESI levels from uploaded data, (2) Create acuity distribution features per clinical category, (3) Generate acuity-weighted patient counts (ESI 1-2 = high acuity, ESI 3 = medium, ESI 4-5 = low). In `pages/10_Patient_Forecast.py`, add an 'Acuity Breakdown' section that shows forecasted patients by ESI level within each category. Update `pages/11_Staff_Planner.py` to use acuity-weighted staffing ratios (e.g., 1 nurse per 2 high-acuity patients vs 1 per 4 low-acuity)."

### P2.3 Enhanced Seasonality Features
**Prompt:** "Enhance the seasonal proportions module (`app_core/analytics/seasonal_proportions.py`) to include additional healthcare-specific seasonal patterns. Add support for: (1) Flu season indicator (October-March elevated respiratory), (2) Holiday effects (Thanksgiving, Christmas, New Year trauma spikes), (3) Pay period effects (configurable bi-weekly or monthly patterns), (4) Weather event indicators (heat wave, cold snap from weather data). Create a `SeasonalityConfig` class that allows enabling/disabling each seasonal component. Update the training pages to expose these configuration options."

### P2.4 Visualize Forecast Uncertainty
**Prompt:** "Improve the communication of forecast uncertainty across all visualization pages. In `pages/10_Patient_Forecast.py` and `pages/01_Dashboard.py`, modify the 7-day forecast charts to: (1) Display error bars representing the 95% confidence interval on each bar/point, (2) Show a shaded uncertainty band on line charts, (3) Add a tooltip showing 'Expected: X, Range: Y-Z (95% CI)'. For models that don't natively produce intervals (XGBoost, ANN), use the conformal prediction module (`app_core/pipelines/conformal_prediction.py`) to generate prediction intervals from historical residuals."

### P2.5 Forecast vs. Actuals Performance Tracking
**Prompt:** "Add a forecast performance review feature to track prediction accuracy over time. Create a new file `app_core/analytics/forecast_tracker.py` with functions to: (1) Store historical forecasts with timestamps in session state or database, (2) Compare forecasts against actuals when they become available, (3) Calculate rolling MAPE, MAE, and bias metrics over the past 7, 14, and 30 days. In `pages/01_Dashboard.py`, add a 'Review Past Performance' tab that displays: (1) Overlay chart of forecasted vs actual patient volumes, (2) KPI cards showing recent forecast accuracy metrics, (3) Alert if accuracy degradation exceeds threshold (e.g., MAPE > 20%)."

---

## P3: Lower Priority (Nice to Have)

### P3.1 Introduce Background Workers for Long-Running Tasks
**Prompt:** "Refactor the application to offload long-running tasks to a background worker. Integrate Redis and RQ (Redis Queue). Create `app_core/workers/` directory with: (1) `queue_manager.py` - RQ queue setup and job submission, (2) `tasks.py` - task definitions for model training and optimization. Modify `pages/08_Train_Models.py` and optimization pages (11 & 12) so that when a user starts a model training or optimization job, it is enqueued as a background job. Update the UI to show job status (queued, running, finished, failed) with a progress indicator and display results upon completion without blocking the main application."

### P3.2 Add Structured Logging and Monitoring
**Prompt:** "Integrate structured logging into the application using Python's standard `logging` library configured to output JSON format. Create `app_core/logging/structured_logger.py` with a configured logger that outputs: (1) Timestamp, (2) Log level, (3) Module name, (4) Event type, (5) Contextual data as JSON. Add log statements for key events: data ingestion start/complete, model training start/complete with duration and metrics, prediction requests with input summary, optimization solver runs with solution status. Optionally expose key metrics via a `/metrics` endpoint using `prometheus-client` library for future monitoring integration."

### P3.3 Build Real-Time Dashboard Updates
**Prompt:** "Upgrade the Dashboard page (`pages/01_Dashboard.py`) to support automatic refresh. Add a toggle 'Enable Auto-Refresh' that, when enabled, uses `st.experimental_rerun()` with a 15-minute interval via `time.sleep()` in a separate thread or Streamlit's native rerun mechanism. Implement a visual component that: (1) Fetches latest actual patient arrival data from Supabase, (2) Compares current-hour actuals against the forecast, (3) Displays a prominent alert banner if actuals deviate from forecast by more than 20%, (4) Shows a 'Last Updated' timestamp."

### P3.4 Create Scenario Planning Tool
**Prompt:** "Add a new 'Scenario Planner' page at `pages/14_Scenario_Planner.py`. This page should allow users to create 'what-if' scenarios by: (1) Selecting a baseline forecast from session state, (2) Applying multipliers to specific categories (e.g., '+50% respiratory for 7 days'), (3) Specifying date ranges for the scenario, (4) Running the staffing and inventory optimizers on the modified forecast. Display a comparison report showing: baseline vs scenario patient counts, required staff changes, inventory impact, and estimated cost difference. Allow saving scenarios with names for future reference."

### P3.5 Implement Model Ensemble Pipeline
**Prompt:** "Create a new 'Ensemble' model pipeline in `app_core/models/ensemble_pipeline.py`. This pipeline should: (1) Accept a list of trained model results (e.g., LSTM, XGBoost, SARIMAX), (2) Combine their forecasts using configurable weighting methods: equal weights, inverse-RMSE weights, or user-specified weights, (3) Optionally use a meta-learner (Ridge regression) trained on OOF predictions. Add an 'Ensemble' option to the model selection dropdown in `pages/10_Patient_Forecast.py`. Display the ensemble composition and individual model contributions in the results."

---

## P4: Deferred (Production Phase)

### P4.1 Develop Feature Store
**Prompt:** "Design and implement a feature store for consistent feature generation and versioning. Create `app_core/feature_store/` module with: (1) `feature_registry.py` - catalog of all feature definitions, (2) `feature_generator.py` - functions to compute each feature type (lags, rolling stats, calendar, weather), (3) `feature_storage.py` - persistence layer using Supabase or local Parquet files, (4) `feature_versioning.py` - track feature set versions with metadata. Refactor `app_core/data/data_processing.py` to use the feature store. Ensure all models consume features from the store for consistency."

### P4.2 Implement Stacking Ensemble Framework
**Prompt:** "Generalize the hybrid modeling framework to a full stacking ensemble. Create `app_core/models/ml/stacking_pipeline.py` that: (1) Trains multiple base models (LSTM, SARIMAX, XGBoost, ANN) using K-fold cross-validation, (2) Generates out-of-fold (OOF) predictions from each base model, (3) Uses OOF predictions as features for a meta-model (default: Ridge regression), (4) Supports custom meta-models via configuration. Add a 'Stacking Ensemble' tab to `pages/08_Train_Models.py` with checkboxes to select which base models to include in the stack."

### P4.3 Add HIPAA Compliance Features
**Prompt:** "Implement healthcare regulatory compliance features for production deployment. Create `app_core/compliance/` module with: (1) `audit_logger.py` - log all data access events with user ID, timestamp, data accessed, and action type to a secure audit table, (2) `data_anonymizer.py` - functions to de-identify patient data for display, (3) `access_control.py` - role-based permissions (admin, analyst, viewer) with page-level restrictions. Update all pages to check permissions before displaying sensitive data. Add an 'Audit Log' page accessible only to admins showing recent data access events."

### P4.4 Implement Admission Forecasting
**Prompt:** "Create an 'Admission Forecast' page at `pages/15_Admission_Forecast.py` that predicts hospital admissions from ED arrivals. The page should: (1) Load ED arrival forecasts by category and acuity from session state, (2) Apply configurable admission rates per category/acuity combination (e.g., ESI-1 Cardiac = 85% admission rate), (3) Generate 7-day admission forecasts by unit (ICU, Med-Surg, Observation), (4) Display bed demand projections with confidence intervals, (5) Alert if projected admissions exceed configured bed capacity."

### P4.5 Enable Dashboard Cross-Filtering
**Prompt:** "Enhance dashboard interactivity by enabling cross-filtering between charts. In `pages/01_Dashboard.py`, modify the Plotly charts to use `plotly.callbacks` or Streamlit's click event handling so that: (1) Clicking a day in the '7-Day Forecast' chart filters the 'Staffing Overview' and 'Inventory Status' tabs to show data for only that day, (2) Clicking a category in the category breakdown filters all charts to that category, (3) Add a 'Clear Filters' button to reset the view. Use `st.session_state` to track active filters across chart interactions."

---

## UI/UX Enhancements (Any Phase)

### UX.1 Clarify Forecast Page Call-to-Action
**Prompt:** "Improve UI clarity on the 'Patient Forecast' page (`pages/10_Patient_Forecast.py`). In the 'Forecast View' tab, rename the 'Generate Forecast' button to 'Activate Forecast' to accurately reflect that it selects and pushes the chosen model's results to downstream pages (Staff Planner, Supply Planner, Dashboard). Add a tooltip explaining: 'Activating a forecast makes it the active forecast used by optimization and planning pages.'"

### UX.2 Co-locate Related Controls
**Prompt:** "Improve the user experience on the 'Patient Forecast' page by co-locating related UI elements. Move the 'Activate Forecast' button so it is positioned directly next to the 'Select Model' dropdown menu, creating a clear 'Select → Activate' workflow. Group all model selection controls (model dropdown, horizon selector, confidence level) in a single card or container with a header 'Forecast Configuration'."

### UX.3 Elevate Category Breakdown Visibility
**Prompt:** "Increase the visibility of clinical category breakdown on the 'Patient Forecast' page. Remove the `st.expander` that currently hides the category distribution. Instead, display this breakdown prominently as a primary visual element: (1) Add a stacked bar chart showing daily forecasts segmented by category, (2) Add a donut chart showing the 7-day total distribution across categories, (3) Position these charts directly below the main forecast chart, not hidden in an expander."

### UX.4 Unify Dashboard Pages
**Prompt:** "Consolidate redundant dashboard functionality. Review `pages/01_Dashboard.py` and `pages/13_Action_Center.py` for overlapping features. Merge the 'WORKFLOW PROGRESS' indicator and AI recommendations from Action Center into the Executive Dashboard. Consider whether Action Center should remain as a separate page or become a tab within the Dashboard. If merging, ensure all unique functionality is preserved and the Dashboard doesn't become cluttered - use tabs or progressive disclosure."

### UX.5 Add Forecast Composition Chart to Dashboard
**Prompt:** "Increase the operational value of the Dashboard forecast display. In `pages/01_Dashboard.py`, below the main '7-Day Patient Forecast' bar chart, add a new stacked bar chart. Each bar represents a day, segmented by color to show the forecasted patient count for each clinical category (RESPIRATORY=blue, CARDIAC=red, TRAUMA=orange, GASTROINTESTINAL=green, INFECTIOUS=purple, NEUROLOGICAL=cyan, OTHER=gray). Add a legend and ensure colors are consistent across all pages."

---

## Code Quality & Architecture (Any Phase)

### CQ.1 Reduce Information Overload on Dashboard
**Prompt:** "Reduce information overload on the Executive Dashboard (`pages/01_Dashboard.py`). Modify the 'Executive Summary' tab to: (1) Initially display only the 4 most critical KPIs (Total Patients Next 7 Days, Predicted Monthly Cost, Staff Utilization %, Urgent Actions Count), (2) Add a 'View Detailed Report' expander that reveals secondary KPIs and detailed charts when clicked, (3) Use progressive disclosure pattern - summary first, details on demand. Ensure the page loads quickly by deferring heavy chart rendering until the detail section is expanded."

### CQ.2 Standardize Error Handling
**Prompt:** "Implement consistent error handling across all pages. Create an error handling decorator in `app_core/errors/handlers.py` called `@handle_page_errors` that: (1) Catches all exceptions, (2) Logs the error with full traceback using structured logging, (3) Displays a user-friendly error message via `st.error()`, (4) Optionally shows technical details in an expander for debugging. Apply this decorator to all main page functions. Create specific exception classes in `app_core/errors/exceptions.py` for: DataValidationError, ModelTrainingError, OptimizationError, DatabaseConnectionError."

### CQ.3 Add Data Validation Layer
**Prompt:** "Implement a data validation layer to catch issues early. Create `app_core/data/validators.py` with Pydantic models or custom validators for: (1) Uploaded data schema (required columns, data types), (2) Prepared data quality (no nulls in key columns, date continuity), (3) Feature engineering outputs (expected feature names, value ranges), (4) Model inputs (sufficient rows, no constant columns). Add validation calls at each pipeline stage with clear error messages. Display validation warnings in the UI before allowing users to proceed to the next step."

### CQ.4 Implement Configuration Management
**Prompt:** "Centralize application configuration for easier management. Create `app_core/config/` module with: (1) `settings.py` - Pydantic Settings class loading from environment variables and `.streamlit/secrets.toml`, (2) `model_defaults.py` - default hyperparameters for each model type, (3) `ui_constants.py` - colors, chart configurations, text labels. Replace all hardcoded values throughout the codebase with references to these config modules. Add a 'System Settings' page (admin only) to view current configuration."

---

## Summary: Recommended Execution Order

**Sprint 1 (Immediate):**
1. P0.1 - Fix Category Forecast Data Flow
2. P0.2 - Implement Backtest vs. Live Modes
3. P0.3 - Implement Walk-Forward Validation

**Sprint 2 (This Week):**
4. P1.1 - Implement Automated Testing
5. P1.2 - Set Up CI/CD Pipeline
6. P1.3 - Refactor Train Models Page

**Sprint 3 (Next Week):**
7. P1.4 - Hierarchical Forecasting
8. P2.1 - SHAP Explainability
9. P2.4 - Visualize Forecast Uncertainty

**Sprint 4 (Demo Prep):**
10. P2.2 - Patient Acuity Integration
11. P2.5 - Forecast vs. Actuals Tracking
12. UX.3 - Elevate Category Breakdown

**Post-Validation (Production Phase):**
- All P3 and P4 items
- P4.3 (HIPAA) is mandatory before any real patient data
