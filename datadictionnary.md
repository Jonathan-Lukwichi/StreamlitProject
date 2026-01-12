# Data Dictionary - HealthForecast AI

This document describes all datasets required for the hospital forecasting and optimization system. Use this as a reference when collecting data from the hospital.

---

## 1. Patient Arrivals Data

This is the core dataset for forecasting. It contains daily counts of patients arriving at the Emergency Department.

**Table Name:** patient_arrivals

**Required Fields:**
- datetime: The date of patient arrivals in YYYY-MM-DD format
- hospital_name: The name of the hospital for multi-hospital support
- ED: Total number of Emergency Department arrivals for that day (integer count)

**Purpose:** This dataset is used to train forecasting models that predict future patient volumes. The ED column is the primary target variable for all forecasting operations.

---

## 2. Clinical Visits / Reason for Visit Data

This dataset contains granular breakdown of patient visits by medical condition. It allows the system to forecast not just total arrivals, but arrivals by clinical category.

**Table Name:** clinical_visits

**Required Fields:**
- datetime: The date of visits in YYYY-MM-DD format
- hospital_name: The name of the hospital

**Granular Medical Condition Fields (all integers representing daily counts):**
- Asthma: Number of asthma-related visits
- Pneumonia: Number of pneumonia cases
- Shortness_of_Breath: Patients with breathing difficulties
- Chest_Pain: Patients presenting with chest pain
- Arrhythmia: Heart rhythm disorder cases
- Hypertensive_Emergency: High blood pressure emergencies
- Fracture: Bone fracture cases
- Laceration: Cuts and wounds requiring treatment
- Burn: Burn injury cases
- Fall_Injury: Injuries resulting from falls
- Abdominal_Pain: Stomach and abdominal pain cases
- Vomiting: Patients with vomiting symptoms
- Diarrhea: Gastrointestinal distress cases
- Flu_Symptoms: Influenza-like illness presentations
- Fever: Patients with elevated temperature
- Viral_Infection: Confirmed or suspected viral infections
- Headache: Patients presenting with headaches
- Dizziness: Vertigo and balance issues
- Allergic_Reaction: Allergic response cases
- Mental_Health: Psychiatric and mental health presentations
- Total_Arrivals: Sum of all conditions (optional, calculated automatically)

**Purpose:** The system automatically aggregates these into 7 clinical categories for forecasting: RESPIRATORY (Asthma + Pneumonia + Shortness_of_Breath), CARDIAC (Chest_Pain + Arrhythmia + Hypertensive_Emergency), TRAUMA (Fracture + Laceration + Burn + Fall_Injury), GASTROINTESTINAL (Abdominal_Pain + Vomiting + Diarrhea), INFECTIOUS (Flu_Symptoms + Fever + Viral_Infection), NEUROLOGICAL (Headache + Dizziness), and OTHER (Allergic_Reaction + Mental_Health).

---

## 3. Calendar Data

This dataset contains calendar-related information that affects patient arrival patterns, such as holidays and special events.

**Table Name:** calendar_data

**Required Fields:**
- date: The calendar date in YYYY-MM-DD format
- hospital_name: The name of the hospital
- holiday: Binary flag (1 if public holiday, 0 if not)

**Optional Fields:**
- holiday_prev: Binary flag indicating the day before a holiday
- moon_phase: Moon phase value from 0 to 1 (some studies suggest correlation with ED visits)

**Purpose:** Holidays significantly impact patient volumes. Public holidays typically see different arrival patterns, and the day before/after holidays can also show variations. This data helps improve forecast accuracy.

---

## 4. Staff Scheduling Data

This dataset contains daily staffing information including the number of staff on duty and utilization metrics.

**Table Name:** staff_scheduling

**Required Fields:**
- date: The scheduling date in YYYY-MM-DD format
- doctors_on_duty: Number of doctors working that day (integer)
- nurses_on_duty: Number of nurses working that day (integer)
- support_staff_on_duty: Number of support staff (orderlies, technicians, etc.) working that day (integer)

**Optional Fields:**
- overtime_hours: Total overtime hours worked across all staff (decimal)
- average_shift_length_hours: Average length of shifts in hours (decimal, typically 8 or 12)
- staff_shortage_flag: Binary flag (1 if staffing was below required levels, 0 if adequate)
- staff_utilization_rate: Percentage of staff capacity utilized (0-100)

**Purpose:** This data is used by the staff optimization module to analyze current staffing patterns, identify inefficiencies, and generate optimized schedules that minimize costs while meeting patient demand.

---

## 5. Inventory Management Data

This dataset tracks daily inventory consumption and stock levels for critical medical supplies.

**Table Name:** inventory_management

**Required Fields:**
- Date: The inventory date in YYYY-MM-DD format

**Consumption Fields (daily usage counts):**
- Inventory_Used_Gloves: Number of glove pairs consumed
- Inventory_Used_PPE_Sets: Number of PPE sets consumed (gowns, masks, face shields)
- Inventory_Used_Medications: Number of medication units dispensed

**Stock Level Fields (current inventory counts):**
- Inventory_Level_Gloves: Current gloves in stock
- Inventory_Level_PPE: Current PPE sets in stock
- Inventory_Level_Medications: Current medication units in stock

**Optional Fields:**
- Restock_Event: Binary flag (1 if restocking occurred that day, 0 if not)
- Stockout_Risk_Score: Calculated risk of running out of supplies (0 to 1 scale)

**Purpose:** This data feeds the inventory optimization module which calculates optimal reorder points, safety stock levels, and generates procurement recommendations to prevent stockouts while minimizing holding costs.

---

## 6. Financial Data

This comprehensive dataset contains cost and revenue information needed for optimization calculations and financial impact analysis.

**Table Name:** financial_data

**Required Fields:**
- Date: The financial date in YYYY-MM-DD format

**Labor Cost Fields:**
- Doctor_Hourly_Rate: Hourly wage for doctors in local currency
- Nurse_Hourly_Rate: Hourly wage for nurses in local currency
- Support_Staff_Hourly_Rate: Hourly wage for support staff in local currency
- Overtime_Premium_Rate: Overtime multiplier (typically 1.5 for time-and-a-half)
- Doctor_Daily_Cost: Total doctor labor cost for the day
- Nurse_Daily_Cost: Total nurse labor cost for the day
- Support_Staff_Daily_Cost: Total support staff labor cost for the day
- Overtime_Cost: Total overtime expenses for the day
- Agency_Staff_Cost: Cost of temporary/agency staff if used
- Total_Labor_Cost: Sum of all labor costs

**Inventory Cost Fields:**
- Cost_Per_Glove_Pair: Unit cost for gloves
- Cost_Per_PPE_Set: Unit cost for PPE sets
- Cost_Per_Medication_Unit: Average unit cost for medications
- Gloves_Cost: Total daily cost for gloves
- PPE_Cost: Total daily cost for PPE
- Medication_Cost: Total daily cost for medications
- Inventory_Holding_Cost: Daily cost of storing inventory
- Restock_Order_Cost: Cost of placing orders
- Total_Inventory_Cost: Sum of all inventory costs

**Revenue Fields:**
- Revenue_Per_Patient_Avg: Average revenue generated per patient visit
- Total_Daily_Revenue: Total revenue from patient services
- Government_Subsidy: Any government funding received
- Insurance_Reimbursement: Payments from insurance providers
- Total_Revenue: Sum of all revenue sources

**Operating Cost Fields:**
- Utility_Cost: Electricity, water, gas costs
- Equipment_Maintenance_Cost: Medical equipment upkeep costs
- Administrative_Cost: Office and administrative expenses
- Facility_Cost: Building maintenance and rent
- Total_Operating_Cost: Sum of all operating costs

**Profitability Fields:**
- Daily_Profit: Revenue minus all costs
- Profit_Margin_Percent: Profit as percentage of revenue
- Cost_Per_Patient: Average cost to serve one patient
- Revenue_Per_Patient: Average revenue from one patient

**Optimization Penalty Fields:**
- Overstaffing_Cost: Penalty cost when staff exceeds demand
- Understaffing_Penalty: Penalty cost when staff is insufficient
- Stockout_Penalty: Cost incurred during supply stockouts
- Overstock_Penalty: Cost of excess inventory
- Total_Optimization_Cost: Sum of all penalty costs

**Purpose:** Financial data enables the system to calculate cost savings from optimization, perform what-if analysis, and provide ROI projections for recommended changes.

---

## Data Collection Priority

When collecting data from the hospital, prioritize in this order:

**Critical (Must Have):**
1. Patient Arrivals - datetime and ED columns are essential for any forecasting
2. Clinical Visits - At least 5-10 medical condition columns for category forecasting

**High Priority:**
3. Calendar Data - date and holiday columns significantly improve forecast accuracy

**Medium Priority:**
4. Staff Scheduling - Required for staff optimization features
5. Inventory Management - Required for supply optimization features

**Lower Priority:**
6. Financial Data - Enhances optimization with cost-based recommendations

---

## Data Format Notes

- All dates should be in YYYY-MM-DD format (e.g., 2024-01-15)
- Integer fields should contain whole numbers only
- Decimal fields can contain values like 8.5 or 85.75
- Binary flags should be 0 or 1
- Percentages should be 0-100 (not 0-1)
- Currency values should be in consistent units (all USD, all EUR, etc.)
- Missing values are acceptable but will be imputed - collect as much as possible
- Minimum recommended data: 1 year of historical data (365+ rows)
- Optimal data: 2-3 years of historical data for seasonal pattern detection
