-- ============================================================================
-- SUPABASE READ-ONLY SECURITY CONFIGURATION
-- ============================================================================
-- This SQL makes your Supabase database READ-ONLY for the Streamlit app
-- (Simulates hospital database with restricted access)
--
-- Run this in Supabase SQL Editor AFTER uploading your CSV data
-- ============================================================================

-- ============================================================================
-- STEP 1: Remove ALL write permissions
-- ============================================================================

-- Drop any existing insert/update/delete policies
DROP POLICY IF EXISTS "Allow public insert" ON patient_arrivals;
DROP POLICY IF EXISTS "Allow public insert" ON weather_data;
DROP POLICY IF EXISTS "Allow public insert" ON calendar_data;
DROP POLICY IF EXISTS "Allow public insert" ON clinical_visits;

DROP POLICY IF EXISTS "Allow authenticated insert" ON patient_arrivals;
DROP POLICY IF EXISTS "Allow authenticated insert" ON weather_data;
DROP POLICY IF EXISTS "Allow authenticated insert" ON calendar_data;
DROP POLICY IF EXISTS "Allow authenticated insert" ON clinical_visits;

DROP POLICY IF EXISTS "Allow public update" ON patient_arrivals;
DROP POLICY IF EXISTS "Allow public update" ON weather_data;
DROP POLICY IF EXISTS "Allow public update" ON calendar_data;
DROP POLICY IF EXISTS "Allow public update" ON clinical_visits;

DROP POLICY IF EXISTS "Allow public delete" ON patient_arrivals;
DROP POLICY IF EXISTS "Allow public delete" ON weather_data;
DROP POLICY IF EXISTS "Allow public delete" ON calendar_data;
DROP POLICY IF EXISTS "Allow public delete" ON clinical_visits;

-- ============================================================================
-- STEP 2: Ensure RLS is ENABLED (security layer active)
-- ============================================================================

ALTER TABLE patient_arrivals ENABLE ROW LEVEL SECURITY;
ALTER TABLE weather_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE calendar_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinical_visits ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- STEP 3: Create READ-ONLY policies (SELECT only)
-- ============================================================================

-- Patient Arrivals: READ-ONLY
CREATE POLICY "Allow public read access"
ON patient_arrivals
FOR SELECT
USING (true);

-- Weather Data: READ-ONLY
CREATE POLICY "Allow public read access"
ON weather_data
FOR SELECT
USING (true);

-- Calendar Data: READ-ONLY
CREATE POLICY "Allow public read access"
ON calendar_data
FOR SELECT
USING (true);

-- Clinical Visits: READ-ONLY
CREATE POLICY "Allow public read access"
ON clinical_visits
FOR SELECT
USING (true);

-- ============================================================================
-- VERIFICATION: Check that policies are correct
-- ============================================================================

-- This query shows all policies (should only see SELECT policies)
SELECT
    schemaname,
    tablename,
    policyname,
    cmd as command,
    qual as using_expression
FROM pg_policies
WHERE tablename IN ('patient_arrivals', 'weather_data', 'calendar_data', 'clinical_visits')
ORDER BY tablename, policyname;

-- ============================================================================
-- SUCCESS!
-- ============================================================================
-- Your database is now READ-ONLY for API access
--
-- ✅ Streamlit app CAN:
--    - SELECT (read) data via API
--    - Filter by date ranges
--    - Request specific columns
--
-- ❌ Streamlit app CANNOT:
--    - INSERT new records
--    - UPDATE existing records
--    - DELETE records
--
-- This mimics real hospital database access restrictions!
-- ============================================================================
