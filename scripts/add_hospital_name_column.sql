-- =============================================================================
-- SQL Script: Add hospital_name column to all data tables
-- Run this in Supabase SQL Editor
-- =============================================================================

-- Step 1: Add hospital_name column to patient_data table
ALTER TABLE patient_data
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

-- Update existing rows to have the hospital name
UPDATE patient_data
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

-- Step 4: Add hospital_name column to reason_data table
ALTER TABLE reason_data
ADD COLUMN IF NOT EXISTS hospital_name TEXT DEFAULT 'Pamplona Spain Hospital';

UPDATE reason_data
SET hospital_name = 'Pamplona Spain Hospital'
WHERE hospital_name IS NULL;

-- Step 5: Create an index for faster hospital filtering
CREATE INDEX IF NOT EXISTS idx_patient_hospital ON patient_data(hospital_name);
CREATE INDEX IF NOT EXISTS idx_weather_hospital ON weather_data(hospital_name);
CREATE INDEX IF NOT EXISTS idx_calendar_hospital ON calendar_data(hospital_name);
CREATE INDEX IF NOT EXISTS idx_reason_hospital ON reason_data(hospital_name);

-- Step 6: Create a view to get list of all hospitals (for dropdown)
CREATE OR REPLACE VIEW available_hospitals AS
SELECT DISTINCT hospital_name
FROM (
    SELECT DISTINCT hospital_name FROM patient_data
    UNION
    SELECT DISTINCT hospital_name FROM weather_data
    UNION
    SELECT DISTINCT hospital_name FROM calendar_data
    UNION
    SELECT DISTINCT hospital_name FROM reason_data
) AS all_hospitals
WHERE hospital_name IS NOT NULL
ORDER BY hospital_name;

-- Verify the changes
SELECT 'patient_data' as table_name, COUNT(*) as rows, COUNT(DISTINCT hospital_name) as hospitals FROM patient_data
UNION ALL
SELECT 'weather_data', COUNT(*), COUNT(DISTINCT hospital_name) FROM weather_data
UNION ALL
SELECT 'calendar_data', COUNT(*), COUNT(DISTINCT hospital_name) FROM calendar_data
UNION ALL
SELECT 'reason_data', COUNT(*), COUNT(DISTINCT hospital_name) FROM reason_data;
