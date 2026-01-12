# Comprehensive Recommendations for HealthForecast AI

This document consolidates all expert recommendations for improving the UI/UX, design, and data science methodologies of the HealthForecast AI application. Each recommendation is formulated as an actionable prompt.

---

## 1. High-Level UI/UX and Workflow Recommendations

### 1.1. Consolidate Disjointed User Workflows

**Observation:** The current application forces users to navigate between multiple pages to complete a single logical task (e.g., Data Prep requires visiting pages 02, 03, and 06). This is confusing and error-prone.

**Recommendation:** Merge related pages into unified "Studios" or "Hubs" with guided, multi-step interfaces.

*   **Prompt 1: Create a Unified "Data Studio".**
    > Consolidate the functionality of `pages/02_Upload_Data.py`, `pages/03_Prepare_Data.py`, and `pages/06_Feature_Studio.py` into a single page. Use a visual stepper UI (`[Step 1: Upload] -> [Step 2: Fuse & Prepare] -> [Step 3: Engineer Features]`) to guide the user through the data preparation pipeline seamlessly.

*   **Prompt 2: Create a Unified "Modeling Studio".**
    > Consolidate `pages/05_Baseline_Models.py`, `pages/07_Feature_Selection.py`, and `pages/08_Train_Models.py` into a single page. Use a stepper (`[Step 1: Feature Selection] -> [Step 2: Train Model]`) to create a guided workflow. The "Train Model" step should then have internal tabs or a selectbox for choosing between `Statistical`, `Machine Learning`, and `Hybrid` model families.

### 1.2. Enhance Dashboard Usability and Onboarding

**Observation:** The main dashboard is powerful but can be overwhelming and feels broken to a new user if prerequisite steps have not been completed.

**Recommendation:** Improve the onboarding experience and make the dashboard more interactive and user-centric.

*   **Prompt 3: Implement a "Getting Started" State.**
    > On the main dashboard (`pages/01_Dashboard.py`), if `st.session_state` indicates that no data has been processed, display a "Welcome" guide instead of empty charts. This guide should include a prominent button that uses `st.switch_page` to navigate the user to the first step of the data workflow.

*   **Prompt 4: Enable Drill-Down Functionality.**
    > Make the main dashboard's KPI cards interactive. Clicking on a card (e.g., "Staff Utilization") should navigate the user to the corresponding detailed page (e.g., `Staff_Planner.py`) for deeper analysis.

*   **Prompt 5: Introduce User-Centric Views.**
    > Add a global `st.toggle` in the sidebar for "Analyst Mode". When disabled (default), the app shows a simplified "Manager View" with only high-level KPIs and summaries. When enabled, it reveals all advanced charts and technical details (like ACF/PACF plots, model parameter tables, etc.).

### 1.3. Refine General Look and Feel for Professional Context

**Observation:** While visually striking, some design choices may be distracting in a professional healthcare setting.

**Recommendation:** Offer more control to the user and improve accessibility.

*   **Prompt 6: Add a "Quiet Mode" for Animations.**
    > In a global "Settings" area or the sidebar, add a `st.toggle` to disable the background fluorescent and sparkle CSS animations. This provides a more focused experience for professional use.

*   **Prompt 7: Add Contextual Help.**
    > For complex charts and concepts (e.g., ACF/PACF plots, FFT analysis, HPO results), include a small `st.expander` titled "What does this mean?" that provides a brief, plain-language explanation of the topic.

---

## 2. Data Science & Modeling Best Practices

### 2.1. Implement a Robust Validation Strategy (Most Critical)

*   **Prompt 8: Integrate Time Series Cross-Validation.**
    > Refactor the data splitting logic across `06_Feature_Studio.py`, `07_Feature_Selection.py`, and `08_Train_Models.py`. Replace the single train/test split with a rolling-origin validation scheme using `sklearn.model_selection.TimeSeriesSplit`. A model's final reported metrics must be the average and standard deviation of its performance across all CV folds.

### 2.2. Enhance Feature Engineering and Automation

*   **Prompt 9: Generate Dynamic Lag and Window Features.**
    > Enhance the `process_dataset` function in `03_Prepare_Data.py`. Add UI controls to allow users to specify windows (e.g., 3, 7, 14 days) for generating lag features and rolling window statistics (mean, std, min, max) for both the target variable and key exogenous variables.

*   **Prompt 10: Automate Fourier Feature Creation.**
    > In `06_Feature_Studio.py`, add a feature generation option that programmatically takes the dominant cycle periods detected by the FFT analysis in the EDA page and creates corresponding Fourier terms (sine/cosine pairs) as new features.

*   **Prompt 11: Automate Response to Stationarity.**
    > In `06_Feature_Studio.py`, add a checkbox: **"Apply Differencing if Non-Stationary"**. This option should be automatically recommended if the ADF test on the EDA page had a p-value > 0.05. This step should handle the transformation and inverse-transformation of the target variable.

### 2.3. Improve Modeling and Evaluation Rigor

*   **Prompt 12: Implement Naive Baselines.**
    > In the modeling workflow, add two simple baseline models for comparison: a "Naive Forecast" (last value) and a "Seasonal Naive Forecast" (value from last season). All complex models must prove they outperform these heuristics.

*   **Prompt 13: Compare Multi-Horizon Strategies.**
    > In `08_Train_Models.py`, implement an alternative "Recursive" forecasting strategy (where the model's own output is used as input for the next step) to compare against the current "Direct" strategy. Analyze the trade-offs in your thesis.

*   **Prompt 14: Implement Stacking Generalization.**
    > As an advanced hybrid model, implement a Stacking ensemble. Train diverse base models (e.g., SARIMAX, XGBoost, LSTM) and then train a final "meta-model" (e.g., Linear Regression) that uses their predictions as input features.

### 2.4. Refine Model-Specific and HPO Best Practices

*   **Prompt 15: Enhance XGBoost Training.**
    > Modify the XGBoost pipeline to use its built-in `early_stopping_rounds` feature during training to find the optimal number of trees automatically and prevent overfitting.

*   **Prompt 16: Enhance LSTM Training.**
    > For the LSTM model, add the `lookback_window` as a tunable parameter in the HPO process. Also, for a more comprehensive thesis, benchmark the LSTM against a simpler GRU model.

*   **Prompt 17: Use Cross-Validation within HPO.**
    > Refactor the Optuna `objective` function. It must perform Time Series Cross-Validation for each hyperparameter trial and return the *average* validation score across all folds. This is critical for finding robust hyperparameters.

*   **Prompt 18: Visualize HPO Results.**
    > After an Optuna study completes, use its built-in `optuna.visualization` library to generate and display `plot_optimization_history` and `plot_param_importances` within the Streamlit UI.
