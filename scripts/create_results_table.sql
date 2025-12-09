-- =============================================================================
-- SQL Schema for pipeline_results table
-- Stores page results for session persistence in HealthForecast AI
-- Run this in Supabase SQL Editor
-- =============================================================================

-- Create the pipeline_results table
CREATE TABLE IF NOT EXISTS pipeline_results (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    page_type VARCHAR(50) NOT NULL,
    dataset_hash VARCHAR(50) NOT NULL,
    results_json TEXT NOT NULL,
    keys_saved TEXT[], -- Array of saved key names
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Create unique constraint for user + page + dataset
    CONSTRAINT unique_user_page_dataset UNIQUE (user_id, page_type, dataset_hash)
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_results_user_id ON pipeline_results(user_id);
CREATE INDEX IF NOT EXISTS idx_results_page_type ON pipeline_results(page_type);
CREATE INDEX IF NOT EXISTS idx_results_dataset_hash ON pipeline_results(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_results_saved_at ON pipeline_results(saved_at);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE pipeline_results ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust as needed for your security requirements)
CREATE POLICY "Allow all operations" ON pipeline_results
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant access to authenticated and anon users
GRANT ALL ON pipeline_results TO authenticated;
GRANT ALL ON pipeline_results TO anon;
GRANT USAGE, SELECT ON SEQUENCE pipeline_results_id_seq TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE pipeline_results_id_seq TO anon;

-- Add comment
COMMENT ON TABLE pipeline_results IS 'Stores ML pipeline page results for session persistence';
