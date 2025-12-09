-- ============================================================================
-- FINANCIAL DATA TABLE
-- For HealthForecast AI - Hospital Financial Data Module
-- ============================================================================

-- Drop existing table if needed (uncomment if you want to recreate)
-- DROP TABLE IF EXISTS financial_data;

-- Create the financial_data table
CREATE TABLE IF NOT EXISTS financial_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    "Date" DATE NOT NULL,

    -- Labor Cost Parameters
    "Doctor_Hourly_Rate" REAL DEFAULT 0,
    "Nurse_Hourly_Rate" REAL DEFAULT 0,
    "Support_Staff_Hourly_Rate" REAL DEFAULT 0,
    "Overtime_Premium_Rate" REAL DEFAULT 0,
    "Doctor_Daily_Cost" REAL DEFAULT 0,
    "Nurse_Daily_Cost" REAL DEFAULT 0,
    "Support_Staff_Daily_Cost" REAL DEFAULT 0,
    "Overtime_Cost" REAL DEFAULT 0,
    "Agency_Staff_Cost" REAL DEFAULT 0,
    "Total_Labor_Cost" REAL DEFAULT 0,

    -- Inventory Cost Parameters
    "Cost_Per_Glove_Pair" REAL DEFAULT 0,
    "Cost_Per_PPE_Set" REAL DEFAULT 0,
    "Cost_Per_Medication_Unit" REAL DEFAULT 0,
    "Gloves_Cost" REAL DEFAULT 0,
    "PPE_Cost" REAL DEFAULT 0,
    "Medication_Cost" REAL DEFAULT 0,
    "Total_Inventory_Usage_Cost" REAL DEFAULT 0,
    "Inventory_Holding_Cost" REAL DEFAULT 0,
    "Restock_Order_Cost" REAL DEFAULT 0,
    "Emergency_Procurement_Cost" REAL DEFAULT 0,
    "Inventory_Wastage_Cost" REAL DEFAULT 0,
    "Total_Inventory_Cost" REAL DEFAULT 0,

    -- Revenue
    "Revenue_Per_Patient_Avg" REAL DEFAULT 0,
    "Total_Daily_Revenue" REAL DEFAULT 0,
    "Government_Subsidy" REAL DEFAULT 0,
    "Insurance_Reimbursement" REAL DEFAULT 0,
    "Total_Revenue" REAL DEFAULT 0,

    -- Operating Costs
    "Utility_Cost" REAL DEFAULT 0,
    "Equipment_Maintenance_Cost" REAL DEFAULT 0,
    "Administrative_Cost" REAL DEFAULT 0,
    "Facility_Cost" REAL DEFAULT 0,
    "Total_Operating_Cost" REAL DEFAULT 0,

    -- Profitability Metrics
    "Daily_Profit" REAL DEFAULT 0,
    "Profit_Margin_Percent" REAL DEFAULT 0,
    "Cost_Per_Patient" REAL DEFAULT 0,
    "Revenue_Per_Patient" REAL DEFAULT 0,

    -- Cost Ratios
    "Labor_Cost_Ratio" REAL DEFAULT 0,
    "Inventory_Cost_Ratio" REAL DEFAULT 0,
    "Budget_Variance" REAL DEFAULT 0,
    "Budget_Variance_Percent" REAL DEFAULT 0,
    "Labor_Cost_Per_Patient" REAL DEFAULT 0,
    "Inventory_Cost_Per_Patient" REAL DEFAULT 0,

    -- Optimization Penalties
    "Overstaffing_Cost" REAL DEFAULT 0,
    "Understaffing_Penalty" REAL DEFAULT 0,
    "Stockout_Penalty" REAL DEFAULT 0,
    "Overstock_Penalty" REAL DEFAULT 0,
    "Total_Optimization_Cost" REAL DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on Date for faster queries
CREATE INDEX IF NOT EXISTS idx_financial_date ON financial_data("Date");

-- Enable Row Level Security (RLS)
ALTER TABLE financial_data ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust as needed for your security requirements)
CREATE POLICY "Allow all operations on financial_data"
ON financial_data
FOR ALL
USING (true)
WITH CHECK (true);

-- Grant permissions
GRANT ALL ON financial_data TO authenticated;
GRANT ALL ON financial_data TO anon;

-- Verify table was created
SELECT 'Table financial_data created successfully!' AS status;
