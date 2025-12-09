
-- ============================================================================
-- INVENTORY MANAGEMENT TABLE
-- For HealthForecast AI - Inventory Optimization Module
-- ============================================================================

-- Drop existing table if needed (uncomment if you want to recreate)
-- DROP TABLE IF EXISTS inventory_management;

-- Create the inventory_management table
CREATE TABLE IF NOT EXISTS inventory_management (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    "Date" DATE NOT NULL,
    "Inventory_Used_Gloves" INTEGER DEFAULT 0,
    "Inventory_Used_PPE_Sets" INTEGER DEFAULT 0,
    "Inventory_Used_Medications" INTEGER DEFAULT 0,
    "Inventory_Level_Gloves" INTEGER DEFAULT 0,
    "Inventory_Level_PPE" INTEGER DEFAULT 0,
    "Inventory_Level_Medications" INTEGER DEFAULT 0,
    "Restock_Event" INTEGER DEFAULT 0,
    "Stockout_Risk_Score" REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on Date for faster queries
CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_management("Date");

-- Enable Row Level Security (RLS)
ALTER TABLE inventory_management ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust as needed for your security requirements)
CREATE POLICY "Allow all operations on inventory_management"
ON inventory_management
FOR ALL
USING (true)
WITH CHECK (true);

-- Grant permissions
GRANT ALL ON inventory_management TO authenticated;
GRANT ALL ON inventory_management TO anon;

-- Verify table was created
SELECT 'Table inventory_management created successfully!' AS status;
