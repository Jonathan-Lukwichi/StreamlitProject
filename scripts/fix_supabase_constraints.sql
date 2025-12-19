-- =============================================================================
-- Fix Supabase Constraints for Multi-Hospital Support
-- =============================================================================
-- Run these commands in your Supabase SQL Editor to enable multiple hospitals
-- to have data for the same dates.

-- 1. Drop existing unique constraints (date only)
ALTER TABLE patient_arrivals DROP CONSTRAINT IF EXISTS unique_patient_date;
ALTER TABLE weather_data DROP CONSTRAINT IF EXISTS unique_weather_date;
ALTER TABLE calendar_data DROP CONSTRAINT IF EXISTS unique_calendar_date;
ALTER TABLE clinical_visits DROP CONSTRAINT IF EXISTS unique_clinical_date;

-- 2. Add new unique constraints on (date, hospital_name)
ALTER TABLE patient_arrivals ADD CONSTRAINT unique_patient_hospital_date UNIQUE (date, hospital_name);
ALTER TABLE weather_data ADD CONSTRAINT unique_weather_hospital_date UNIQUE (date, hospital_name);
ALTER TABLE calendar_data ADD CONSTRAINT unique_calendar_hospital_date UNIQUE (date, hospital_name);
ALTER TABLE clinical_visits ADD CONSTRAINT unique_clinical_hospital_date UNIQUE (date, hospital_name);

-- 3. Fix weather_data column precision for average_mslp (atmospheric pressure)
-- The Madrid data has values like 101703.64 (in Pascals, not hPa)
ALTER TABLE weather_data ALTER COLUMN average_mslp TYPE NUMERIC(10, 2);

-- 4. Create or update the available_hospitals view
CREATE OR REPLACE VIEW available_hospitals AS
SELECT DISTINCT hospital_name
FROM patient_arrivals
WHERE hospital_name IS NOT NULL
ORDER BY hospital_name;

-- Verify the changes
SELECT
    tc.constraint_name,
    tc.table_name,
    kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
WHERE tc.constraint_type = 'UNIQUE'
    AND tc.table_name IN ('patient_arrivals', 'weather_data', 'calendar_data', 'clinical_visits');
