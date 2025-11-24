# 🚀 API Integration Guide
## Multi-Source Data Ingestion System

This guide explains how to use the API-based data ingestion system to fetch all 4 datasets from external APIs instead of manual CSV uploads.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Supported APIs](#supported-apis)
4. [Configuration](#configuration)
5. [Integration into Data Hub](#integration)
6. [Usage Examples](#usage-examples)
7. [Importance & Benefits](#importance)

---

## 📖 Overview

The API Integration System allows you to fetch all 4 datasets from various external APIs:

- **Patient Data**: Hospital/EHR systems, Supabase, or Mock
- **Weather Data**: OpenWeather, Visual Crossing, WeatherAPI.com, or Mock
- **Calendar Data**: AbstractAPI, Calendarific, or Mock
- **Reason for Visit**: Hospital Clinical APIs, FHIR, Supabase, or Mock

### Key Features:
✅ **Multi-Provider Support**: Choose from multiple API providers for each dataset
✅ **Standardized Output**: All APIs return data in your app's expected schema
✅ **Bulk Fetch**: Fetch all 4 datasets with one button click
✅ **Mock Mode**: Test without real API keys using realistic synthetic data
✅ **Error Handling**: Robust error handling with clear user feedback

---

## 🏗️ Architecture

```
User selects API provider + date range
           ↓
APIConfigManager creates connector
           ↓
Connector fetches data from external API
           ↓
Data validated and mapped to standard schema
           ↓
Stored in session_state (same as CSV uploads)
           ↓
Proceeds to fusion → feature engineering → modeling
```

### File Structure:

```
app_core/api/
├── base_connector.py         # Abstract base class for all connectors
├── patient_connector.py       # Patient data connectors (Hospital, Supabase, Mock)
├── weather_connector.py       # Weather connectors (OpenWeather, Visual Crossing, etc.)
├── calendar_connector.py      # Calendar/holiday connectors
├── reason_connector.py        # Reason for visit connectors (Clinical, FHIR, Mock)
├── config_manager.py          # Central API configuration manager
└── __init__.py                # Module exports

app_core/ui/
└── api_fetch_ui.py            # UI components for API fetch

pages/
└── 02_Data_Hub.py             # Data Hub page (add API fetch option here)
```

---

## 🔌 Supported APIs

### 1. **Patient Data APIs**

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `mock` | Synthetic patient data | None (free) |
| `hospital_api` | Generic hospital REST API | Base URL + API key |
| `supabase` | Supabase database | Supabase URL + API key |

**Output Schema:**
- `datetime`: Timestamp
- `Target_1`: Patient count

---

### 2. **Weather Data APIs**

| Provider | Description | Requirements | Free Tier |
|----------|-------------|--------------|-----------|
| `mock` | Synthetic weather | None | ✅ |
| `openweather` | OpenWeatherMap | API key + lat/lon | ❌ (paid for historical) |
| `visualcrossing` | Visual Crossing | API key + location | ✅ (1000 requests/day) |
| `weatherapi` | WeatherAPI.com | API key + location | ✅ (7 days history) |

**Output Schema:**
- `datetime`, `Date`
- `Average_Temp`, `Max_temp`, `Min_temp`
- `Total_precipitation`
- `Average_wind`, `Max_wind`
- `Average_mslp` (pressure)

---

### 3. **Calendar Data APIs**

| Provider | Description | Requirements | Free Tier |
|----------|-------------|--------------|-----------|
| `mock` | Synthetic holidays | None | ✅ |
| `abstractapi` | AbstractAPI Holidays | API key + country code | ✅ (1000 requests/month) |
| `calendarific` | Calendarific | API key + country code | ✅ (1000 requests/month) |

**Output Schema:**
- `Date`
- `Day_of_week` (0=Monday, 6=Sunday)
- `Holiday` (0 or 1)
- `Holiday_prev` (was yesterday a holiday?)
- `Moon_Phase` (0-7)

---

### 4. **Reason for Visit APIs**

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `mock` | Synthetic medical data | None (free) |
| `hospital_api` | Generic hospital clinical API | Base URL + API key |
| `supabase` | Supabase clinical table | Supabase URL + API key |
| `fhir` | FHIR-compliant EHR (Epic, Cerner) | FHIR endpoint + OAuth token |

**Output Schema:**
- `Date`
- Granular columns: `Asthma`, `Pneumonia`, `Fracture`, `Chest_Pain`, etc.
- **Automatically aggregated** into 6 categories by your existing fusion pipeline

---

## ⚙️ Configuration

### Option 1: Using Streamlit Secrets (Recommended)

Create `.streamlit/secrets.toml`:

```toml
[api.patient]
provider = "mock"  # or "hospital_api", "supabase"
# base_url = "https://hospital.example.com/api"
# api_key = "your_api_key_here"

[api.weather]
provider = "visualcrossing"
base_url = "https://weather.visualcrossing.com"
api_key = "YOUR_VISUAL_CROSSING_KEY"
location = "Madrid,Spain"

[api.calendar]
provider = "calendarific"
base_url = "https://calendarific.com/api"
api_key = "YOUR_CALENDARIFIC_KEY"
country = "ES"

[api.reason]
provider = "mock"
# base_url = "https://hospital.example.com/api"
# api_key = "your_api_key_here"
```

### Option 2: Runtime Configuration (UI-Based)

Users can select providers and enter parameters directly in the Data Hub UI.

---

## 🔗 Integration into Data Hub

### Step 1: Import API Fetch UI

Add to the top of `pages/02_Data_Hub.py`:

```python
from app_core.ui.api_fetch_ui import render_api_fetch_option, render_bulk_api_fetch
```

### Step 2: Add API Fetch to Each Card

Replace the simple `_upload_generic()` call with a tabbed interface:

**Example for Patient Data Card:**

```python
with col1:  # Patient Data Card
    st.markdown("""
    <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
        <div style='text-align: center; margin-bottom: 1rem;'>
            <div class='hf-feature-icon'>📊</div>
            <h2 class='hf-feature-title'>Patient Data</h2>
        </div>
    """, unsafe_allow_html=True)

    # TABBED INTERFACE: CSV Upload OR API Fetch
    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 Fetch from API"])

    with tab1:
        _upload_generic("Upload CSV", "patient", "datetime")

    with tab2:
        render_api_fetch_option(
            dataset_type="patient",
            key_prefix="patient"
        )

    # Display metrics (same as before)
    df_patient = st.session_state.get("patient_data")
    if isinstance(df_patient, pd.DataFrame) and not df_patient.empty:
        st.metric("Rows", f"{df_patient.shape[0]:,}")
        # ... rest of metrics

    st.markdown("</div>", unsafe_allow_html=True)
```

**Repeat for Weather, Calendar, and Reason cards.**

### Step 3: Add Bulk Fetch Section (Optional)

After the 4-column grid, add:

```python
# After the individual cards...
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

with st.expander("⚡ Bulk API Fetch - Fetch All 4 Datasets at Once", expanded=False):
    render_bulk_api_fetch()
```

---

## 💡 Usage Examples

### Example 1: Fetch Patient Data from Mock API

```python
from app_core.api import APIConfigManager
from datetime import datetime, timedelta

# Initialize manager
config_manager = APIConfigManager()

# Get patient connector (mock provider)
patient_connector = config_manager.get_patient_connector("mock")

# Fetch data for last year
df = patient_connector.fetch_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

print(df.head())
# Output:
#    datetime  Target_1
# 0  2023-01-01      58
# 1  2023-01-02      46
# ...
```

### Example 2: Fetch Weather from Visual Crossing

```python
# Get weather connector
weather_connector = config_manager.get_weather_connector(
    provider="visualcrossing",
    location="Madrid,Spain"
)

# Fetch historical weather
df = weather_connector.fetch_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

print(df.columns)
# Output: ['datetime', 'Date', 'Average_Temp', 'Max_temp', 'Min_temp',
#          'Total_precipitation', 'Average_wind', 'Max_wind', 'Average_mslp']
```

### Example 3: Bulk Fetch All 4 Datasets

```python
# Fetch all datasets with mock providers
config_manager = APIConfigManager()

patient_df = config_manager.get_patient_connector("mock").fetch_data(start_date, end_date)
weather_df = config_manager.get_weather_connector("mock").fetch_data(start_date, end_date)
calendar_df = config_manager.get_calendar_connector("mock").fetch_data(start_date, end_date)
reason_df = config_manager.get_reason_connector("mock").fetch_data(start_date, end_date)

print(f"Fetched {len(patient_df)} patient records")
print(f"Fetched {len(weather_df)} weather records")
print(f"Fetched {len(calendar_df)} calendar records")
print(f"Fetched {len(reason_df)} reason records")
```

### Example 4: Test API Connections

```python
config_manager = APIConfigManager()
results = config_manager.test_all_connections()

for dataset_type, result in results.items():
    print(f"{dataset_type}: {result['status']} - {result['message']}")

# Output:
# patient: success - Successfully connected to patient_mock
# weather: success - Successfully connected to weather_mock
# calendar: success - Successfully connected to calendar_mock
# reason: success - Successfully connected to reason_mock
```

---

## 🎯 Importance & Benefits

### Why This Matters for Your Project:

#### 1. **Real-World Hospital Scenario** 🏥
   - Hospitals **never give direct access** to entire datasets
   - Data is accessed via **secure APIs** with authentication
   - Your app now mimics **real production environments**

#### 2. **Automated Data Updates** 🔄
   - **No manual CSV uploads** for daily/weekly updates
   - Set up **scheduled API fetches** (cron jobs, Airflow)
   - Always have **latest data** without human intervention

#### 3. **Multi-Source Flexibility** 🌐
   - Patient data from **hospital EHR (Epic/Cerner via FHIR)**
   - Weather from **Visual Crossing or OpenWeather**
   - Calendar/holidays from **public APIs**
   - Reason data from **clinical systems or Supabase**

#### 4. **Scalability** 📈
   - Supports **multiple hospitals/facilities**
   - Add new API providers **without changing core code**
   - Easy to extend for new data sources

#### 5. **Data Lineage & Audit Trail** 📊
   - Track **which API** provided which data
   - Know **when data was fetched**
   - Log **API parameters** for reproducibility

#### 6. **Reduced Manual Errors** ✅
   - No more "wrong file uploaded"
   - No more "date mismatch" issues
   - **Automated schema validation**

#### 7. **Development & Testing** 🧪
   - **Mock providers** for local development
   - Test entire pipeline **without real APIs**
   - Generate **realistic synthetic data** on demand

#### 8. **Cost Optimization** 💰
   - Many APIs have **free tiers** (Visual Crossing: 1000/day)
   - **Cache API responses** to reduce calls
   - Only fetch **new/updated data**

---

## 🔒 Security Best Practices

1. **Never commit API keys** to Git
   - Use `.streamlit/secrets.toml` (gitignored)
   - Or environment variables

2. **Use Row-Level Security** (for Supabase)
   - Restrict API access by user/hospital
   - Implement proper authentication

3. **Rate Limiting**
   - Respect API rate limits
   - Implement exponential backoff for retries

4. **Data Encryption**
   - Use HTTPS for all API calls
   - Encrypt sensitive data in transit

---

## 🚀 Next Steps

1. **Test with Mock Providers**
   - Run the app with all APIs in "mock" mode
   - Verify data flows through fusion → feature engineering → modeling

2. **Get Free API Keys**
   - Visual Crossing: https://www.visualcrossing.com/sign-up
   - Calendarific: https://calendarific.com/signup

3. **Connect Real APIs**
   - Update secrets.toml with real API keys
   - Test individual connectors
   - Verify data quality

4. **Integrate into Data Hub**
   - Add tabs to each upload card
   - Test bulk fetch
   - Add API connection status indicators

5. **Monitor & Optimize**
   - Log API call latency
   - Cache frequently accessed data
   - Implement retry logic for failures

---

## 📞 API Provider Resources

### Weather APIs:
- **Visual Crossing**: https://www.visualcrossing.com/weather-api
- **OpenWeatherMap**: https://openweathermap.org/api
- **WeatherAPI.com**: https://www.weatherapi.com/

### Calendar APIs:
- **AbstractAPI**: https://www.abstractapi.com/holidays-api
- **Calendarific**: https://calendarific.com/

### Healthcare APIs:
- **FHIR Specification**: https://www.hl7.org/fhir/
- **Epic FHIR**: https://fhir.epic.com/
- **Cerner FHIR**: https://fhir.cerner.com/

---

## 🎉 Conclusion

The API Integration System transforms your app from a **manual CSV upload tool** into a **production-ready healthcare data platform** that:

✅ Mimics real-world hospital data access patterns
✅ Automates data ingestion from multiple sources
✅ Provides flexibility to choose best-of-breed APIs
✅ Reduces manual errors and operational overhead
✅ Enables continuous model retraining with fresh data

**This is a critical feature for any real-world deployment!** 🚀
