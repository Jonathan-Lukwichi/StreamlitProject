# 🏥 PRODUCTION SETUP: Supabase Hospital Database

## Complete Guide to Deploy Your App with Real-World Hospital API Access

---

## 🎯 **WHAT YOU'RE BUILDING**

A **production-ready healthcare forecasting platform** that mimics real-world hospital scenarios:

```
Hospital Database (Supabase)
    ↓
  READ-ONLY API
    ↓
Your Streamlit App
    ↓
Data Fetch → Fusion → Feature Engineering → Model Training → Predictions
```

**Key Characteristics:**
- ✅ App CANNOT upload/modify hospital data
- ✅ App can ONLY read via API
- ✅ Realistic hospital security restrictions
- ✅ Production-ready architecture

---

## 📋 **SETUP STEPS (Follow in Order)**

### **PHASE 1: SUPABASE SETUP** (Already Done ✅)

You've already completed:
1. ✅ Created Supabase project
2. ✅ Created 4 tables (patient_arrivals, weather_data, calendar_data, clinical_visits)
3. ✅ Enabled RLS with read policies

---

### **PHASE 2: UPLOAD DATA TO SUPABASE** (Do This Next 🔄)

#### **Step 1: Install Required Packages**

```bash
pip install supabase pandas python-dotenv toml
```

#### **Step 2: Configure Secrets**

Create `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://YOUR_PROJECT_ID.supabase.co"
key = "YOUR_ANON_KEY"

[api.patient]
provider = "supabase"
base_url = "https://YOUR_PROJECT_ID.supabase.co/rest/v1"
api_key = "YOUR_ANON_KEY"

[api.weather]
provider = "supabase"
base_url = "https://YOUR_PROJECT_ID.supabase.co/rest/v1"
api_key = "YOUR_ANON_KEY"

[api.calendar]
provider = "supabase"
base_url = "https://YOUR_PROJECT_ID.supabase.co/rest/v1"
api_key = "YOUR_ANON_KEY"

[api.reason]
provider = "supabase"
base_url = "https://YOUR_PROJECT_ID.supabase.co/rest/v1"
api_key = "YOUR_ANON_KEY"
```

**How to get YOUR_PROJECT_ID and YOUR_ANON_KEY:**
1. Go to Supabase dashboard
2. Click "Settings" → "API"
3. Copy "Project URL" (contains your project ID)
4. Copy "anon public" key

#### **Step 3: Upload Your CSV Files**

Run the one-time setup script:

```bash
python setup_hospital_database.py --all
```

This will:
- Read your 4 CSV files from the default directory
- Map columns to Supabase schema
- Upload to Supabase tables
- Show progress and summary

**Output you should see:**

```
📊 Uploading Patient Data...
   Loaded 1,844 rows
   Uploading 1,844 records...
   Progress: 1,844/1,844 (100%)
   ✅ Uploaded 1,844 records to patient_arrivals

🌤️  Uploading Weather Data...
   ...

✅ HOSPITAL DATABASE SETUP COMPLETE!
📊 Patient Arrivals:     1,844 records
🌤️  Weather Data:         1,844 records
📅 Calendar Data:        1,844 records
🏥 Clinical Visits:      1,844 records

📈 TOTAL:                7,376 records
```

---

### **PHASE 3: LOCK DOWN SUPABASE (READ-ONLY)** 🔒

#### **Step 4: Make Database Read-Only**

In Supabase SQL Editor, run:

```sql
-- File: supabase_readonly_security.sql
-- Copy and paste this entire file into SQL Editor and click "Run"
```

This removes all INSERT/UPDATE/DELETE permissions, leaving only SELECT (read) access.

**Why?**
- Real hospitals don't let external apps modify their database
- Your app should only READ data via API
- This mimics production security

---

### **PHASE 4: TEST API CONNECTIONS** ✅

#### **Step 5: Verify Everything Works**

```bash
python test_supabase_api.py
```

**Expected output:**

```
🧪 SUPABASE API CONNECTION TEST

Testing PATIENT API (supabase)
✅ Connector created: SupabasePatientConnector
📅 Fetching data from 2024-01-08 to 2024-01-15...
✅ SUCCESS! Fetched 7 rows
📊 Columns: ['datetime', 'Target_1']

Testing WEATHER API (supabase)
✅ SUCCESS! Fetched 7 rows

Testing CALENDAR API (supabase)
✅ SUCCESS! Fetched 7 rows

Testing REASON API (supabase)
✅ SUCCESS! Fetched 7 rows

🎉 ALL TESTS PASSED!
✅ Your Streamlit app can now fetch data from Supabase
🚀 Ready for production use!
```

---

### **PHASE 5: USE IN STREAMLIT APP** 🎨

#### **Step 6: Fetch Data in Data Hub**

Open your Streamlit app and go to Data Hub:

1. For each dataset (Patient, Weather, Calendar, Reason):
   - Click **"🔌 Fetch from API"** tab
   - Provider: **supabase** (should be auto-selected)
   - Select date range
   - Click **"Fetch Data"**

2. Or use **Bulk API Fetch**:
   - Scroll to bottom of Data Hub
   - Select date range
   - Click **"🚀 Fetch All Datasets"**
   - All 4 datasets load at once

3. Verify data loaded:
   - Each card shows row count
   - Preview shows actual data
   - Date range displayed

4. Proceed to **Data Preparation Studio** → Fusion works normally!

---

## 🔄 **TESTING DIFFERENT DATASETS**

### **Scenario: You Want to Test a New Hospital's Data**

#### **Method 1: Clear and Re-upload**

```bash
# Clear all tables
python setup_hospital_database.py --clear

# Upload new dataset
python setup_hospital_database.py \
  --patient "path/to/new_patient.csv" \
  --weather "path/to/new_weather.csv" \
  --calendar "path/to/new_calendar.csv" \
  --reason "path/to/new_reason.csv"
```

**Time:** 30 seconds

#### **Method 2: Create Separate Supabase Projects**

- Hospital A: Project "healthforecast-hospitalA"
- Hospital B: Project "healthforecast-hospitalB"
- Switch by changing `.streamlit/secrets.toml`

---

## 🎓 **HOW IT MIMICS REAL HOSPITALS**

### **Real-World Hospital IT Architecture**

```
┌──────────────────────────────────────────┐
│  HOSPITAL'S INTERNAL SYSTEMS              │
│  - Epic EHR                               │
│  - Cerner EMR                             │
│  - Weather API Subscriptions              │
│  - Calendar Services                      │
│                                           │
│  [Data stored in hospital's database]    │
└──────────────────────────────────────────┘
              ↓
    ┌─────────────────┐
    │  API GATEWAY    │
    │  (READ-ONLY)    │
    └─────────────────┘
              ↓
┌──────────────────────────────────────────┐
│  ANALYTICS TEAM (YOUR APP)                │
│  - Requests data via API                  │
│  - Specifies date ranges                  │
│  - Processes data for forecasting         │
│  - ❌ CANNOT modify hospital's data       │
└──────────────────────────────────────────┘
```

### **Your Supabase Setup**

```
┌──────────────────────────────────────────┐
│  SUPABASE DATABASE (Simulates Hospital)   │
│  - patient_arrivals                       │
│  - weather_data                           │
│  - calendar_data                          │
│  - clinical_visits                        │
│                                           │
│  [You uploaded data ONCE as "IT team"]   │
└──────────────────────────────────────────┘
              ↓
    ┌─────────────────┐
    │  POSTGREST API  │
    │  (READ-ONLY)    │
    └─────────────────┘
              ↓
┌──────────────────────────────────────────┐
│  YOUR STREAMLIT APP                       │
│  - Fetches data via API                   │
│  - Date range filters                     │
│  - Processes for ML models                │
│  - ❌ CANNOT upload/modify data           │
└──────────────────────────────────────────┘
```

**IDENTICAL ARCHITECTURE!** ✅

---

## 🚨 **TROUBLESHOOTING**

### **Problem 1: "Connection failed" when fetching**

**Solution:**
```bash
# 1. Check secrets.toml exists
ls .streamlit/secrets.toml

# 2. Verify Supabase credentials
python test_supabase_api.py

# 3. Check Supabase project is running (dashboard)
```

### **Problem 2: "No data found for date range"**

**Solution:**
```sql
-- In Supabase SQL Editor, check if data exists:
SELECT COUNT(*), MIN(date), MAX(date) FROM patient_arrivals;
SELECT COUNT(*), MIN(date), MAX(date) FROM weather_data;
SELECT COUNT(*), MIN(date), MAX(date) FROM calendar_data;
SELECT COUNT(*), MIN(date), MAX(date) FROM clinical_visits;
```

If counts are 0, re-run:
```bash
python setup_hospital_database.py --all
```

### **Problem 3: "Permission denied" when fetching**

**Solution:**
```sql
-- Check RLS policies exist:
SELECT * FROM pg_policies
WHERE tablename IN ('patient_arrivals', 'weather_data', 'calendar_data', 'clinical_visits');
```

Should show policies named "Allow public read access".

If missing, re-run:
```sql
-- In Supabase SQL Editor:
-- Paste contents of supabase_readonly_security.sql
```

### **Problem 4: Upload script fails**

**Solution:**
```bash
# Check file paths in setup_hospital_database.py line 441-446
# Update to your actual CSV paths:

default_paths = {
    "patient": Path(r"YOUR_ACTUAL_PATH/Patient_Data.csv"),
    "weather": Path(r"YOUR_ACTUAL_PATH/Weather_Data.csv"),
    # ...
}
```

---

## 📊 **WHAT'S IN YOUR DATABASE**

After successful upload, you should have:

| Table | Records | Date Range | Key Columns |
|-------|---------|------------|-------------|
| **patient_arrivals** | 1,844 | 2018-2023 | date, datetime, patient_count |
| **weather_data** | 1,844 | 2018-2023 | date, average_temp, max_temp, precipitation, wind, pressure |
| **calendar_data** | 1,844 | 2018-2023 | date, day_of_week, holiday, holiday_prev |
| **clinical_visits** | 1,844 | 2018-2023 | date, asthma, pneumonia, fracture, chest_pain... (20+ conditions) |

**Total:** ~7,376 records representing 5+ years of hospital data

---

## 🎯 **SUCCESS CRITERIA**

You're ready for production when:

✅ **Setup Script Runs Successfully**
```bash
python setup_hospital_database.py --all
# Shows "✅ HOSPITAL DATABASE SETUP COMPLETE!"
```

✅ **Test Script Passes**
```bash
python test_supabase_api.py
# Shows "🎉 ALL TESTS PASSED!"
```

✅ **App Fetches Data**
- Data Hub loads data from Supabase
- No errors in console
- Preview shows actual data

✅ **Fusion Works**
- All 4 datasets merge successfully
- Aggregated categories created
- Feature engineering completes

✅ **Models Train**
- LSTM, XGBoost, etc. train on Supabase data
- Predictions generate correctly

---

## 🚀 **NEXT STEPS**

### **After Basic Setup Works:**

1. **Remove CSV Upload Option** (optional)
   - Make app API-only
   - Pure production mode
   - See `DATA_HUB_API_ONLY.md` for instructions

2. **Add Data Quality Checks**
   - Validate date ranges
   - Check for missing values
   - Alert on anomalies

3. **Implement Caching**
   - Cache frequently requested date ranges
   - Reduce API calls
   - Faster app performance

4. **Multi-Hospital Support**
   - Add facility_id to API calls
   - Filter by hospital
   - Scale to multiple sites

5. **Automated Daily Updates**
   - Cron job to fetch yesterday's data
   - Append to Supabase
   - Trigger model retraining

---

## 🎉 **CONGRATULATIONS!**

You now have a **production-grade healthcare forecasting platform** with:

✅ **Cloud Database** (Supabase)
✅ **API-Based Data Access** (REST API)
✅ **Read-Only Security** (Hospital-grade)
✅ **Real-World Architecture** (Mimics actual hospitals)

**This is how professional healthcare analytics platforms work!** 🏥

---

## 📞 **SUPPORT**

### **Files Reference:**
- `setup_hospital_database.py` - One-time data upload
- `supabase_readonly_security.sql` - Security policies
- `test_supabase_api.py` - Connection testing
- `.streamlit/secrets.toml.example` - Configuration template
- `API_INTEGRATION_GUIDE.md` - Full API documentation

### **Quick Commands:**
```bash
# Upload data
python setup_hospital_database.py --all

# Test connections
python test_supabase_api.py

# Clear and re-upload
python setup_hospital_database.py --clear --all
```

**You're ready to build production-ready healthcare AI!** 🚀
