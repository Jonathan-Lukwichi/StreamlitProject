-- =============================================================================
-- Model Experiment Results Table
-- Tracks model performance across different datasets/feature configurations
-- =============================================================================
-- Run this script in Supabase SQL Editor to create the table
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_experiment_results (
    id BIGSERIAL PRIMARY KEY,

    -- User & Session
    user_id VARCHAR(100) NOT NULL,

    -- Dataset Identification (Method + Date + Feature Count)
    dataset_id VARCHAR(255) NOT NULL,           -- Unique composite identifier
    feature_selection_method VARCHAR(100),       -- e.g., "Permutation Importance", "Lasso", etc.
    feature_engineering_variant VARCHAR(50),     -- "Variant A" or "Variant B"
    feature_count INTEGER,                       -- Number of features selected
    feature_names JSONB,                         -- Array of feature names used

    -- Model Information
    model_name VARCHAR(100) NOT NULL,            -- e.g., "XGBoost", "LSTM", "ARIMA"
    model_category VARCHAR(50),                  -- "Statistical", "ML", "Hybrid"
    model_params JSONB,                          -- Hyperparameters used

    -- Per-Horizon Metrics (stored as JSONB array)
    horizon_metrics JSONB NOT NULL,              -- [{horizon: 1, mae: x, rmse: y, ...}, ...]

    -- Aggregated Metrics (averages across all horizons)
    avg_mae FLOAT,
    avg_rmse FLOAT,
    avg_mape FLOAT,
    avg_accuracy FLOAT,
    avg_r2 FLOAT,
    runtime_seconds FLOAT,

    -- Predictions (full arrays)
    predictions JSONB,                           -- {horizon_1: {actual: [...], predicted: [...]}, ...}

    -- Metadata
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    horizons_trained INTEGER[],                  -- Array of horizon numbers [1,2,3,4,5,6,7]
    train_size INTEGER,
    test_size INTEGER,
    notes TEXT,

    -- Constraints
    CONSTRAINT unique_user_dataset_model UNIQUE (user_id, dataset_id, model_name)
);

-- =============================================================================
-- Indexes for efficient querying
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_experiment_user ON model_experiment_results(user_id);
CREATE INDEX IF NOT EXISTS idx_experiment_dataset ON model_experiment_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_experiment_model ON model_experiment_results(model_name);
CREATE INDEX IF NOT EXISTS idx_experiment_method ON model_experiment_results(feature_selection_method);
CREATE INDEX IF NOT EXISTS idx_experiment_trained ON model_experiment_results(trained_at);

-- =============================================================================
-- Enable Row Level Security (RLS) for multi-tenant access
-- =============================================================================

ALTER TABLE model_experiment_results ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own experiment results
CREATE POLICY "Users can view own experiments" ON model_experiment_results
    FOR SELECT USING (true);

CREATE POLICY "Users can insert own experiments" ON model_experiment_results
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can update own experiments" ON model_experiment_results
    FOR UPDATE USING (true);

CREATE POLICY "Users can delete own experiments" ON model_experiment_results
    FOR DELETE USING (true);

-- =============================================================================
-- Comments for documentation
-- =============================================================================

COMMENT ON TABLE model_experiment_results IS 'Stores model training experiment results with dataset tracking';
COMMENT ON COLUMN model_experiment_results.dataset_id IS 'Composite identifier: {method_slug}_{date}_{feature_count}f';
COMMENT ON COLUMN model_experiment_results.horizon_metrics IS 'Array of per-horizon metrics: [{horizon, mae, rmse, mape, accuracy, r2}, ...]';
COMMENT ON COLUMN model_experiment_results.predictions IS 'Full predictions per horizon: {horizon_N: {actual: [...], predicted: [...]}, ...}';
