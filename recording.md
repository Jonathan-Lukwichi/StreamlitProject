# HealthForecast AI - Video Recording Script

---

# SHORT VERSION (3-5 Minutes)

## Intro (15 seconds)

> "Welcome to HealthForecast AI - a hospital resource planning tool that predicts Emergency Department patient arrivals and optimizes staffing and inventory using machine learning."

---

## Login & Dashboard (30 seconds)

> "After logging in, we land on the Executive Dashboard showing key metrics - today's predicted patients, staffing levels, and inventory status. Everything at a glance for quick decisions."

---

## Data Workflow (45 seconds)

> "The workflow starts with **Upload Data** - we support CSV, Excel, and Parquet files.
>
> In **Prepare Data**, we clean the dataset and fuse it with weather and calendar information.
>
> **Explore Data** shows us patterns - weekly cycles, seasonal trends, and correlations that drive patient arrivals."

---

## Feature Engineering (30 seconds)

> "**Feature Studio** creates lag features - yesterday's patients, last week's average - and target variables for multi-day forecasting.
>
> **Feature Selection** identifies which features matter most, reducing noise and improving accuracy."

---

## Model Training (45 seconds)

> "Now the core of our system. **Baseline Models** trains ARIMA and SARIMAX - classical statistical methods.
>
> **Train Models** offers machine learning: XGBoost, LSTM neural networks, and hybrid models that combine multiple approaches.
>
> **Model Results** compares all models side-by-side - MAE, RMSE, accuracy - so we pick the best performer."

---

## Forecasting & Optimization (45 seconds)

> "**Patient Forecast** generates 7-day predictions broken down by clinical category - respiratory, cardiac, trauma, and more.
>
> **Staff Planner** uses these forecasts to optimize shift schedules - minimizing costs while ensuring adequate coverage.
>
> **Supply Planner** optimizes inventory - right supplies, right time, no waste."

---

## Action Center (30 seconds)

> "Finally, the **Action Center** uses AI to analyze everything and provide plain-language recommendations - what to do this week, risks to watch, and cost-saving opportunities."

---

## Closing (15 seconds)

> "HealthForecast AI: Upload, Prepare, Train, Forecast, Optimize, Act. Data-driven hospital planning made simple. Thank you."

---

## SHORT VERSION TIMING

| Section | Duration |
|---------|----------|
| Intro | 15 sec |
| Login & Dashboard | 30 sec |
| Data Workflow | 45 sec |
| Feature Engineering | 30 sec |
| Model Training | 45 sec |
| Forecasting & Optimization | 45 sec |
| Action Center | 30 sec |
| Closing | 15 sec |
| **TOTAL** | **~4 minutes** |

### Tips for Short Version:
- Move quickly between pages
- Don't wait for loading screens - cut in editing
- Show, don't explain every detail
- Use smooth transitions

---
---

# FULL VERSION (15-20 Minutes)

---

## Introduction (Before Opening the App)

> "Hello and welcome to HealthForecast AI - an enterprise-grade hospital resource planning and forecasting application. This tool helps hospitals predict Emergency Department patient arrivals and optimize staffing and inventory management using machine learning and operations research. Let me walk you through each feature of the application."

---

## Page 1: Welcome Page (Welcome.py)

> "This is the Welcome page - the entry point of our application. Here you can see our branding and a brief overview of what HealthForecast AI offers.
>
> To access the application, users need to authenticate. I'll log in with my credentials.
>
> *[Enter credentials and click Login]*
>
> Once authenticated, we gain access to all the features through the sidebar navigation. You'll notice 13 different pages, each serving a specific purpose in our hospital forecasting workflow."

---

## Page 2: Dashboard (01_Dashboard.py)

> "This is the Executive Dashboard - the command center for hospital administrators.
>
> At the top, you can see key performance indicators showing today's predicted patient arrivals, current staffing levels, and inventory status.
>
> The dashboard provides a high-level overview of:
> - **Patient volume trends** over time
> - **Model performance metrics** from our forecasting models
> - **Resource utilization** across departments
> - **Quick insights** that highlight important patterns or anomalies
>
> This page is designed for quick decision-making - administrators can see everything at a glance without diving into technical details."

---

## Page 3: Upload Data (02_Upload_Data.py)

> "Now let's start the data workflow. The Upload Data page is where we bring in our hospital data.
>
> The application supports multiple file formats:
> - **CSV files** - comma-separated values
> - **Excel files** - .xlsx and .xls formats
> - **Parquet files** - for large datasets
>
> *[Drag and drop or browse for a file]*
>
> Once uploaded, the system automatically detects column types and shows a preview of the data. You can see the number of rows, columns, and basic statistics about your dataset.
>
> The key data we need includes:
> - **Date column** - for time series analysis
> - **Patient counts** - our target variable (ED arrivals)
> - **Optional**: Weather data, calendar events, and other external factors"

---

## Page 4: Prepare Data (03_Prepare_Data.py)

> "The Data Preparation Studio is where we clean and transform our raw data into analysis-ready format.
>
> Here we can:
> - **Handle missing values** - using interpolation or forward/backward fill
> - **Remove duplicates** - ensuring data integrity
> - **Convert data types** - especially date parsing
> - **Fuse multiple data sources** - combining patient data with weather and calendar information
>
> *[Show the data fusion options if available]*
>
> The fusion capability is powerful - we can merge patient arrival data with external factors like:
> - Weather conditions (temperature, humidity, precipitation)
> - Calendar events (holidays, weekends, special events)
>
> After preparation, the cleaned dataset is stored in session state and ready for exploration and modeling."

---

## Page 5: Explore Data (04_Explore_Data.py)

> "Before building models, we need to understand our data. The Explore Data page provides comprehensive Exploratory Data Analysis.
>
> Here you'll find:
>
> **1. Time Series Visualization**
> - Patient arrival trends over time
> - Seasonal patterns (weekly, monthly, yearly cycles)
>
> **2. Statistical Summary**
> - Mean, median, standard deviation
> - Distribution of patient volumes
>
> **3. Correlation Analysis**
> - How different variables relate to patient arrivals
> - Which features might be good predictors
>
> **4. Pattern Detection**
> - Day-of-week effects (weekends vs weekdays)
> - Monthly variations
> - Holiday impacts
>
> *[Show some visualizations]*
>
> These insights help us understand what drives patient arrivals and guide our feature engineering and model selection."

---

## Page 6: Baseline Models (05_Baseline_Models.py)

> "Now we enter the modeling phase. The Baseline Models page implements classical statistical forecasting methods.
>
> We have two baseline models:
>
> **1. ARIMA (Auto-Regressive Integrated Moving Average)**
> - A pure time series model
> - Captures trends and autocorrelation in the data
> - Works with just the historical patient counts
>
> **2. SARIMAX (Seasonal ARIMA with eXogenous variables)**
> - Adds seasonal components for weekly patterns
> - Includes calendar features like day-of-week and weekend indicators
> - Better for data with strong seasonal patterns
>
> *[Show the configuration options]*
>
> For each model, you can choose:
> - **Automatic mode** - the system finds optimal parameters using AIC criterion
> - **Manual mode** - specify your own parameters if you have domain expertise
>
> The models support **multi-horizon forecasting** - predicting 1 day ahead, 2 days ahead, up to 7 days ahead simultaneously.
>
> *[Click Train button and show results]*
>
> After training, we see performance metrics like MAE, RMSE, and forecast accuracy for each horizon."

---

## Page 7: Feature Studio (06_Feature_Studio.py)

> "The Feature Studio is where we engineer features for our machine learning models.
>
> Feature engineering is crucial for ML performance. Here we create:
>
> **1. Lag Features (ED_1, ED_2, ... ED_7)**
> - Yesterday's patient count (ED_1)
> - Two days ago (ED_2), and so on
> - These capture the autoregressive nature of the time series
>
> **2. Target Variables (Target_1, Target_2, ... Target_7)**
> - Tomorrow's patient count (Target_1)
> - Day after tomorrow (Target_2), and so on
> - These are what our models will predict
>
> **3. Rolling Statistics**
> - Moving averages (7-day, 14-day, 30-day)
> - Rolling standard deviation for volatility
>
> **4. Calendar Features**
> - Day of week, month, quarter
> - Holiday indicators
> - Weekend flags
>
> *[Show the feature generation process]*
>
> Once features are generated, they're added to our dataset and ready for model training."

---

## Page 8: Feature Selection (07_Feature_Selection.py)

> "Not all features are equally important. The Feature Selection page helps us identify which features contribute most to prediction accuracy.
>
> We use automated techniques including:
>
> **1. Correlation Analysis**
> - Features highly correlated with our target
> - Removing redundant features that correlate with each other
>
> **2. Feature Importance Ranking**
> - Using tree-based methods to rank features
> - Identifying the top predictors
>
> **3. Statistical Tests**
> - Significance testing for each feature
> - Removing noise that doesn't help prediction
>
> *[Show the feature importance chart]*
>
> This process helps us:
> - Reduce model complexity
> - Improve training speed
> - Avoid overfitting
> - Focus on what really matters for prediction"

---

## Page 9: Train Models (08_Train_Models.py)

> "This is the main Machine Learning training page with four tabs for different model types.
>
> **Tab 1: Benchmarks**
> Quick comparison of all available models on your data.
>
> **Tab 2: Machine Learning Models**
> Here we train our core ML models:
>
> - **XGBoost** - Gradient boosting for tabular data
>   - Fast and accurate
>   - Handles missing values automatically
>   - Great for structured hospital data
>
> - **LSTM (Long Short-Term Memory)** - Deep learning for sequences
>   - Captures long-term dependencies
>   - Ideal for time series patterns
>   - Requires more data but very powerful
>
> - **ANN (Artificial Neural Network)** - Dense neural network
>   - Flexible architecture
>   - Can model complex non-linear relationships
>
> *[Show model configuration options]*
>
> Each model can be configured with:
> - Training/test split ratio
> - Hyperparameters (learning rate, layers, etc.)
> - Forecast horizons
>
> **Tab 3: Hyperparameter Tuning**
> Uses Optuna for automated hyperparameter optimization - finding the best model settings automatically.
>
> **Tab 4: Hybrid Models**
> Our most advanced models combine multiple approaches:
>
> - **LSTM + XGBoost** - LSTM captures patterns, XGBoost corrects residuals
> - **LSTM + SARIMAX** - Deep learning meets statistical modeling
> - **LSTM + ANN** - Two neural networks working together
>
> *[Train a model and show progress]*
>
> Training includes progress bars and real-time metric updates."

---

## Page 10: Model Results (09_Model_Results.py)

> "After training multiple models, we need to compare them. The Model Results page provides comprehensive performance analysis.
>
> **Performance Metrics Table**
> Side-by-side comparison of all trained models:
> - MAE (Mean Absolute Error) - average prediction error
> - RMSE (Root Mean Square Error) - penalizes large errors
> - MAPE (Mean Absolute Percentage Error) - relative accuracy
> - Forecast Accuracy - percentage of correct predictions
>
> *[Show the comparison table]*
>
> **Visualization**
> - Actual vs Predicted plots for each model
> - Residual analysis to check for patterns
> - Confidence intervals showing prediction uncertainty
>
> **Model Selection**
> Based on these metrics, we can select the best performing model for deployment. The system highlights the winner for each metric.
>
> This page helps answer the crucial question: Which model should we trust for our hospital forecasts?"

---

## Page 11: Patient Forecast (10_Patient_Forecast.py)

> "Now we use our trained models for actual forecasting. The Patient Forecast page generates 7-day ahead predictions.
>
> *[Show the forecast interface]*
>
> Here you can:
>
> **1. Select Your Model**
> Choose which trained model to use for forecasting - ARIMA, SARIMAX, XGBoost, LSTM, or any hybrid model.
>
> **2. View Forecasts by Clinical Category**
> Patient arrivals are broken down by:
> - Respiratory cases
> - Cardiac cases
> - Trauma
> - Gastrointestinal
> - Infectious diseases
> - Neurological
> - Other
>
> This uses our **Seasonal Proportions** algorithm that distributes total forecasts based on historical patterns for each day-of-week and month.
>
> **3. Confidence Intervals**
> Each forecast includes uncertainty bounds - showing the range where actual values are likely to fall.
>
> *[Show the 7-day forecast chart]*
>
> The forecast table shows predicted patient counts for each of the next 7 days, helping hospitals prepare resources in advance."

---

## Page 12: Staff Planner (11_Staff_Planner.py)

> "With patient forecasts in hand, we can optimize staffing. The Staff Planner uses Mixed Integer Linear Programming to create optimal schedules.
>
> **The Optimization Problem:**
> Given predicted patient volumes, how do we:
> - Minimize staffing costs
> - Ensure adequate coverage for all shifts
> - Respect labor laws and constraints
> - Balance full-time and part-time staff
>
> *[Show the configuration options]*
>
> **Input Parameters:**
> - Forecast patient volumes (from our models)
> - Staff-to-patient ratios by category
> - Shift definitions (morning, afternoon, night)
> - Cost parameters (hourly rates, overtime)
> - Constraints (max hours, minimum rest)
>
> *[Run the optimizer]*
>
> **Output:**
> - Optimal number of staff per shift
> - Total staffing cost
> - Coverage analysis
> - Recommendations for hiring or reducing hours
>
> The MILP solver finds the mathematically optimal solution - balancing cost efficiency with patient care quality."

---

## Page 13: Supply Planner (12_Supply_Planner.py)

> "Similar to staffing, we optimize inventory management. The Supply Planner ensures hospitals have the right supplies without overstocking.
>
> **The Challenge:**
> - Too little inventory = stockouts and patient care issues
> - Too much inventory = wasted money and storage space
>
> *[Show the inventory interface]*
>
> **What We Optimize:**
> - Medical supplies tied to patient categories
> - Reorder points and quantities
> - Safety stock levels
> - Storage constraints
>
> **Input Parameters:**
> - Current inventory levels
> - Predicted patient volumes by category
> - Supply usage rates per patient
> - Lead times for orders
> - Storage capacity
> - Cost parameters
>
> *[Run the optimizer]*
>
> **Output:**
> - Optimal reorder schedule
> - Recommended safety stock by item
> - Total inventory cost projection
> - Risk analysis for stockouts
>
> This ensures hospitals are prepared for predicted demand while minimizing waste and costs."

---

## Page 14: Action Center (13_Action_Center.py)

> "Finally, the Action Center brings everything together with AI-powered recommendations.
>
> This page uses Large Language Models (Claude or GPT) to analyze all our data and provide actionable insights.
>
> *[Show the Action Center interface]*
>
> **What the AI Analyzes:**
> - Forecast trends and anomalies
> - Staffing optimization results
> - Inventory status
> - Historical patterns
>
> **Recommendations Include:**
> - Specific actions to take this week
> - Risk alerts for potential issues
> - Cost-saving opportunities
> - Quality improvement suggestions
>
> *[Show some AI recommendations]*
>
> The AI explains its reasoning in plain language, making complex analytics accessible to all hospital staff - from administrators to department heads.
>
> This transforms data into decisions - the ultimate goal of HealthForecast AI."

---

## Conclusion (After Showing All Pages)

> "And that concludes our tour of HealthForecast AI.
>
> To summarize the workflow:
> 1. **Upload** your hospital data
> 2. **Prepare** and clean it for analysis
> 3. **Explore** patterns and insights
> 4. **Engineer** predictive features
> 5. **Train** multiple forecasting models
> 6. **Compare** and select the best model
> 7. **Forecast** patient arrivals 7 days ahead
> 8. **Optimize** staffing and inventory
> 9. **Act** on AI-powered recommendations
>
> This end-to-end solution helps hospitals:
> - Reduce costs through optimization
> - Improve patient care with better preparation
> - Make data-driven decisions with confidence
>
> Thank you for watching. If you have questions, please reach out to our team."

---

## Recording Tips

1. **Pace**: Speak slowly and clearly. Pause between sections.
2. **Cursor**: Move your mouse slowly so viewers can follow.
3. **Zoom**: Use browser zoom (Ctrl +) to make text readable.
4. **Data**: Use realistic sample data for demonstrations.
5. **Errors**: If something fails, explain it's a demo environment.
6. **Duration**: Aim for 15-20 minutes total.

## Suggested Recording Order

1. Introduction (30 seconds)
2. Welcome + Login (1 minute)
3. Dashboard (2 minutes)
4. Upload Data (2 minutes)
5. Prepare Data (2 minutes)
6. Explore Data (2 minutes)
7. Baseline Models (3 minutes)
8. Feature Studio (2 minutes)
9. Feature Selection (1 minute)
10. Train Models (3 minutes)
11. Model Results (2 minutes)
12. Patient Forecast (2 minutes)
13. Staff Planner (2 minutes)
14. Supply Planner (2 minutes)
15. Action Center (2 minutes)
16. Conclusion (1 minute)

**Total: ~27 minutes** (can trim to 15-20 by being concise)
