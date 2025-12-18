-- =============================================================================
-- SQL Script: Add hospital_name column to all data tables
-- Run this in Supabase SQL Editor
-- =============================================================================
-- NOTE: Table names based on your actual Supabase schema:
--   - patient_arrivals (patient data)
--   - weather_data (weather data)
--   - calendar_data (calendar/holiday data)
--   - clinical_visits (reason for visit data)
-- =============================================================================

-- Step 1: Add hospital_name column to patient_arrivals table
ALTER TABLE patient_arrivals
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

-- Update existing rows to have the hospital name
UPDATE patient_arrivals
SET hospital_name = 'Pamplona Spain Hospital'
WHERE hospital_name IS NULL;

-- Step 2: Add hospital_name column to weather_data table
ALTER TABLE weather_data
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

UPDATE weather_data
SET hospital_name = 'Pamplona Spain Hospital'
WHERE hospital_name IS NULL;

-- Step 3: Add hospital_name column to calendar_data table
ALTER TABLE calendar_data
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

UPDATE calendar_data
SET hospital_name = 'Pamplona Spain Hospital'
WHERE hospital_name IS NULL;

-- Step 4: Add hospital_name column to clinical_visits table
ALTER TABLE clinical_visits
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

UPDATE clinical_visits
SET hospital_name = 'Pamplona Spain Hospital'
WHERE hospital_name IS NULL;

-- Step 5: Create an index for faster hospital filtering
CREATE INDEX IF NOT EXISTS idx_patient_hospital ON patient_arrivals(hospital_name);
CREATE INDEX IF NOT EXISTS idx_weather_hospital ON weather_data(hospital_name);
CREATE INDEX IF NOT EXISTS idx_calendar_hospital ON calendar_data(hospital_name);
CREATE INDEX IF NOT EXISTS idx_clinical_hospital ON clinical_visits(hospital_name);

-- Step 6: Create a view to get list of all hospitals (for dropdown)
CREATE OR REPLACE VIEW available_hospitals AS
SELECT DISTINCT hospital_name
FROM (
    SELECT DISTINCT hospital_name FROM patient_arrivals
    UNION
    SELECT DISTINCT hospital_name FROM weather_data
    UNION
    SELECT DISTINCT hospital_name FROM calendar_data
    UNION
    SELECT DISTINCT hospital_name FROM clinical_visits
) AS all_hospitals
WHERE hospital_name IS NOT NULL
ORDER BY hospital_name;

-- Verify the changes
SELECT 'patient_arrivals' as table_name, COUNT(*) as rows, COUNT(DISTINCT hospital_name) as hospitals FROM patient_arrivals
UNION ALL
SELECT 'weather_data', COUNT(*), COUNT(DISTINCT hospital_name) FROM weather_data
UNION ALL
SELECT 'calendar_data', COUNT(*), COUNT(DISTINCT hospital_name) FROM calendar_data
UNION ALL
SELECT 'clinical_visits', COUNT(*), COUNT(DISTINCT hospital_name) FROM clinical_visits;
