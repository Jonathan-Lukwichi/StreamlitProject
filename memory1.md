# HealthForecast AI - Complete Supabase Persistence Implementation Guide

## Overview

This document provides a comprehensive prompt for Claude Code to implement persistent storage for the HealthForecast AI Streamlit application using Supabase. This will save ALL pipeline stages (data upload, preparation, feature engineering, model training, forecasting) so users can resume from any point without re-running long computations.

---

## Problem Statement

Currently, HealthForecast AI loses all intermediate results when the app restarts because everything lives in `st.session_state`. The full pipeline takes 10-30+ minutes:

- Data Upload & Preparation: ~2-5 min
- Feature Engineering: ~2-3 min  
- Feature Selection: ~3-5 min
- Baseline Models (ARIMA/SARIMAX): ~5-10 min
- ML Models (XGBoost, LSTM, ANN): ~10-20 min
- Hybrid Models: ~5-10 min

**Goal:** Save everything to Supabase so users can skip completed stages.

---

## Complete Prompt for Claude Code

Copy everything below this line into Claude Code:

---

```
## Task: Implement Full Pipeline Persistence with Supabase for HealthForecast AI

### Context

HealthForecast AI is a Streamlit hospital forecasting app with the following pipeline:

Upload Data → Prepare Data → Explore Data → Feature Studio → Feature Selection → Baseline Models → Train ML Models → Forecast → Optimize → Action Center

Currently ALL results are lost on app restart. We need to persist every stage to Supabase.

**Project Location:** c:\Users\BIBINBUSINESS\OneDrive\Desktop\streamlit_forecast_scaffold\

**Existing Supabase Client:** `app_core/data/supabase_client.py` has `get_supabase_client()` function

---

### STEP 1: Create Database Table in Supabase

Use the connected Supabase MCP to execute this SQL:

```sql
-- =====================================================
-- APP_RESULTS TABLE FOR PIPELINE PERSISTENCE
-- =====================================================

-- Drop existing table if needed (uncomment if you want fresh start)
-- DROP TABLE IF EXISTS app_results;

-- Create the main storage table
CREATE TABLE IF NOT EXISTS app_results (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    page_key TEXT NOT NULL,
    result_key TEXT NOT NULL,
    result_data JSONB NOT NULL,
    metadata JSONB,
    data_size_bytes BIGINT
);

-- Unique constraint for upsert functionality
CREATE UNIQUE INDEX IF NOT EXISTS idx_app_results_unique 
ON app_results (user_id, page_key, result_key);

-- Index for faster page queries
CREATE INDEX IF NOT EXISTS idx_app_results_page 
ON app_results (user_id, page_key);

-- Index for listing all user results
CREATE INDEX IF NOT EXISTS idx_app_results_user 
ON app_results (user_id, updated_at DESC);

-- Enable Row Level Security
ALTER TABLE app_results ENABLE ROW LEVEL SECURITY;

-- Drop existing policy if any
DROP POLICY IF EXISTS "Users can manage their own results" ON app_results;

-- Create policy: users can only access their own data
CREATE POLICY "Users can manage their own results"
ON app_results FOR ALL
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Auto-update updated_at timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Trigger for auto-updating updated_at
DROP TRIGGER IF EXISTS update_app_results_updated_at ON app_results;
CREATE TRIGGER update_app_results_updated_at
    BEFORE UPDATE ON app_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verify table creation
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'app_results' ORDER BY ordinal_position;
```

**Verify:** Confirm the table was created with all columns.

---

### STEP 2: Create the Results Storage Service

Create file: `app_core/data/results_storage_service.py`

```python
"""
Persistent storage service for HealthForecast AI pipeline results.
Handles serialization/deserialization of complex Python objects (DataFrames, NumPy arrays, etc.)
and saves them to Supabase for cross-session persistence.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app_core.data.supabase_client import get_supabase_client


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy, Pandas, and other non-standard Python types.
    Encodes objects with type markers for proper reconstruction on load.
    """
    
    def default(self, obj: Any) -> Any:
        # === NumPy Types ===
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape)
            }
        
        # === Pandas Types ===
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "dataframe",
                "data": obj.to_json(orient='split', date_format='iso'),
                "shape": list(obj.shape),
                "columns": list(obj.columns),
                "dtypes": {col: str(dtype) for col, dtype in obj.dtypes.items()}
            }
        if isinstance(obj, pd.Series):
            return {
                "__type__": "series",
                "data": obj.to_json(orient='split', date_format='iso'),
                "name": obj.name,
                "dtype": str(obj.dtype)
            }
        if isinstance(obj, pd.Timestamp):
            return {"__type__": "pd_timestamp", "data": obj.isoformat()}
        if isinstance(obj, pd.Timedelta):
            return {"__type__": "pd_timedelta", "data": str(obj)}
        if isinstance(obj, pd.Interval):
            return {"__type__": "pd_interval", "data": str(obj)}
        if isinstance(obj, (pd.Index, pd.RangeIndex)):
            return {"__type__": "pd_index", "data": obj.tolist()}
        
        # === Handle NaN/None ===
        if pd.isna(obj):
            return None
        
        # === Datetime Types ===
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}
        
        # === Sets and Tuples ===
        if isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": list(obj)}
        
        # === Bytes ===
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "data": obj.decode('utf-8', errors='replace')}
        
        # === Default ===
        try:
            return super().default(obj)
        except TypeError:
            # Last resort: convert to string
            return {"__type__": "str_fallback", "data": str(obj)}


def decode_special_types(obj: Any) -> Any:
    """
    Recursively decode special types from JSON back to Python objects.
    Handles all types encoded by NpEncoder.
    """
    if isinstance(obj, dict):
        type_marker = obj.get("__type__")
        
        if type_marker == "dataframe":
            try:
                df = pd.read_json(obj["data"], orient='split')
                return df
            except Exception:
                return obj
        
        elif type_marker == "series":
            try:
                series = pd.read_json(obj["data"], orient='split', typ='series')
                series.name = obj.get("name")
                return series
            except Exception:
                return obj
        
        elif type_marker == "ndarray":
            try:
                arr = np.array(obj["data"])
                dtype = obj.get("dtype")
                if dtype:
                    arr = arr.astype(dtype)
                return arr
            except Exception:
                return obj
        
        elif type_marker == "pd_timestamp":
            try:
                return pd.Timestamp(obj["data"])
            except Exception:
                return obj
        
        elif type_marker == "pd_timedelta":
            try:
                return pd.Timedelta(obj["data"])
            except Exception:
                return obj
        
        elif type_marker == "datetime":
            try:
                return datetime.fromisoformat(obj["data"])
            except Exception:
                return obj
        
        elif type_marker == "set":
            return set(obj["data"])
        
        elif type_marker == "tuple":
            return tuple(obj["data"])
        
        elif type_marker == "pd_index":
            return pd.Index(obj["data"])
        
        elif type_marker == "bytes":
            return obj["data"].encode('utf-8')
        
        elif type_marker == "str_fallback":
            return obj["data"]
        
        else:
            # Regular dict - recurse into values
            return {k: decode_special_types(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [decode_special_types(item) for item in obj]
    
    return obj


class ResultsStorageService:
    """
    Service for persisting HealthForecast AI pipeline results to Supabase.
    
    Supports:
    - Saving/loading any Python object (DataFrames, dicts, lists, numpy arrays)
    - Upsert functionality (update existing or insert new)
    - Listing saved results with metadata
    - Deleting results by page or specific key
    - User isolation via Row Level Security
    """
    
    TABLE_NAME = "app_results"
    
    def __init__(self, client=None):
        """Initialize with Supabase client."""
        self.client = client or get_supabase_client()
    
    def _get_user_id(self) -> Optional[str]:
        """Get current authenticated user ID."""
        try:
            user = self.client.auth.get_user()
            if user and hasattr(user, 'user') and user.user:
                return user.user.id
        except Exception:
            pass
        return None
    
    def save_results(
        self,
        page_key: str,
        result_key: str,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Save results to Supabase using upsert (update or insert).
        
        Args:
            page_key: Page identifier (e.g., 'Upload Data', 'Train Models')
            result_key: Result identifier (e.g., 'prepared_data', 'ml_results')
            data: Any Python object to save
            metadata: Optional additional metadata
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        user_id = self._get_user_id()
        if not user_id:
            return False, "User not authenticated"
        
        try:
            # Serialize data to JSON
            json_string = json.dumps(data, cls=NpEncoder)
            data_size = len(json_string.encode('utf-8'))
            
            # Build comprehensive metadata
            full_metadata = {
                "saved_at": datetime.now().isoformat(),
                "data_type": type(data).__name__,
                "size_bytes": data_size,
            }
            
            # Add DataFrame-specific metadata
            if isinstance(data, pd.DataFrame):
                full_metadata["shape"] = list(data.shape)
                full_metadata["columns"] = list(data.columns)[:20]  # First 20 columns
                full_metadata["memory_mb"] = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Add dict/list length
            if isinstance(data, (dict, list)):
                full_metadata["length"] = len(data)
            
            # Merge with user-provided metadata
            if metadata:
                full_metadata.update(metadata)
            
            # Build record
            record = {
                "user_id": user_id,
                "page_key": page_key,
                "result_key": result_key,
                "result_data": json_string,
                "metadata": full_metadata,
                "data_size_bytes": data_size
            }
            
            # Upsert to database
            self.client.table(self.TABLE_NAME).upsert(
                record,
                on_conflict="user_id,page_key,result_key"
            ).execute()
            
            return True, f"Saved {result_key} ({data_size / 1024:.1f} KB)"
        
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def load_results(self, page_key: str, result_key: str) -> Optional[Any]:
        """
        Load results from Supabase.
        
        Args:
            page_key: Page identifier
            result_key: Result identifier
        
        Returns:
            Deserialized Python object or None if not found
        """
        user_id = self._get_user_id()
        if not user_id:
            return None
        
        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("result_data") \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .eq("result_key", result_key) \
                .limit(1) \
                .execute()
            
            if not response.data:
                return None
            
            # Parse JSON and decode special types
            raw_data = json.loads(response.data[0]['result_data'])
            return decode_special_types(raw_data)
        
        except Exception as e:
            print(f"Load error for {page_key}/{result_key}: {e}")
            return None
    
    def load_all_for_page(self, page_key: str) -> Dict[str, Any]:
        """
        Load all saved results for a specific page.
        
        Returns:
            Dict mapping result_key to deserialized data
        """
        user_id = self._get_user_id()
        if not user_id:
            return {}
        
        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("result_key, result_data") \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .execute()
            
            results = {}
            for record in response.data or []:
                try:
                    raw_data = json.loads(record['result_data'])
                    results[record['result_key']] = decode_special_types(raw_data)
                except Exception:
                    continue
            
            return results
        
        except Exception as e:
            print(f"Load all error for {page_key}: {e}")
            return {}
    
    def list_saved_results(self, page_key: Optional[str] = None) -> List[Dict]:
        """
        List all saved results with metadata.
        
        Args:
            page_key: Optional filter by page
        
        Returns:
            List of result metadata dicts
        """
        user_id = self._get_user_id()
        if not user_id:
            return []
        
        try:
            query = self.client.table(self.TABLE_NAME) \
                .select("page_key, result_key, metadata, created_at, updated_at, data_size_bytes") \
                .eq("user_id", user_id)
            
            if page_key:
                query = query.eq("page_key", page_key)
            
            response = query.order("updated_at", desc=True).execute()
            return response.data or []
        
        except Exception:
            return []
    
    def delete_results(self, page_key: str, result_key: str) -> bool:
        """Delete a specific result."""
        user_id = self._get_user_id()
        if not user_id:
            return False
        
        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .eq("result_key", result_key) \
                .execute()
            return True
        except Exception:
            return False
    
    def delete_all_for_page(self, page_key: str) -> bool:
        """Delete all results for a page."""
        user_id = self._get_user_id()
        if not user_id:
            return False
        
        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .eq("page_key", page_key) \
                .execute()
            return True
        except Exception:
            return False
    
    def delete_all_user_results(self) -> bool:
        """Delete ALL results for current user (use with caution)."""
        user_id = self._get_user_id()
        if not user_id:
            return False
        
        try:
            self.client.table(self.TABLE_NAME) \
                .delete() \
                .eq("user_id", user_id) \
                .execute()
            return True
        except Exception:
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for current user."""
        user_id = self._get_user_id()
        if not user_id:
            return {}
        
        try:
            response = self.client.table(self.TABLE_NAME) \
                .select("page_key, data_size_bytes") \
                .eq("user_id", user_id) \
                .execute()
            
            total_bytes = sum(r.get('data_size_bytes', 0) or 0 for r in response.data or [])
            by_page = {}
            for r in response.data or []:
                page = r['page_key']
                by_page[page] = by_page.get(page, 0) + (r.get('data_size_bytes', 0) or 0)
            
            return {
                "total_bytes": total_bytes,
                "total_mb": total_bytes / 1024 / 1024,
                "total_items": len(response.data or []),
                "by_page": by_page
            }
        except Exception:
            return {}


# === Singleton Pattern ===
_storage_service_instance: Optional[ResultsStorageService] = None


def get_storage_service() -> ResultsStorageService:
    """Get singleton ResultsStorageService instance."""
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = ResultsStorageService()
    return _storage_service_instance


def reset_storage_service():
    """Reset singleton (useful for testing or reconnection)."""
    global _storage_service_instance
    _storage_service_instance = None
```

---

### STEP 3: Create Storage UI Components

Create or update file: `app_core/ui/results_storage_ui.py`

```python
"""
UI components for saving/loading HealthForecast AI pipeline results.
Provides buttons, auto-load functionality, and pipeline status dashboard.
"""

import streamlit as st
from datetime import datetime
from typing import List, Optional, Dict, Any

from app_core.data.results_storage_service import get_storage_service


# =====================================================
# PAGE STORAGE CONFIGURATION
# Define which session_state keys to persist for each page
# =====================================================

PAGE_STORAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Upload Data": {
        "keys": ["uploaded_data", "raw_dataframe", "upload_metadata", "file_info"],
        "description": "Raw uploaded data files",
        "icon": "📤"
    },
    "Prepare Data": {
        "keys": ["prepared_data", "fused_data", "cleaning_report", "preparation_config"],
        "description": "Cleaned and fused datasets",
        "icon": "🔧"
    },
    "Explore Data": {
        "keys": ["eda_results", "data_quality_report", "visualization_config", "statistical_summary"],
        "description": "EDA analysis results",
        "icon": "🔍"
    },
    "Baseline Models": {
        "keys": [
            "arima_results", "sarimax_results", "seasonal_proportion_results",
            "baseline_config", "baseline_metrics", "baseline_forecasts"
        ],
        "description": "ARIMA/SARIMAX model results and seasonal proportions",
        "icon": "📊"
    },
    "Feature Studio": {
        "keys": [
            "feature_engineering", "engineered_features", "lag_features",
            "target_features", "feature_config", "ed_features"
        ],
        "description": "Engineered features (ED_1...ED_7, Target_1...Target_7)",
        "icon": "🛠️"
    },
    "Feature Selection": {
        "keys": [
            "selected_features", "feature_importance", "selection_report",
            "feature_rankings", "selection_config"
        ],
        "description": "Selected features and importance analysis",
        "icon": "🎯"
    },
    "Train Models": {
        "keys": [
            "ml_mh_results", "xgboost_results", "lstm_results", "ann_results",
            "hybrid_results", "tuning_results", "model_comparison",
            "training_history", "best_model_config"
        ],
        "description": "ML model training results (XGBoost, LSTM, ANN, Hybrids)",
        "icon": "🤖"
    },
    "Model Results": {
        "keys": ["comparison_results", "metrics_summary", "model_rankings"],
        "description": "Model comparison and performance metrics",
        "icon": "📈"
    },
    "Patient Forecast": {
        "keys": [
            "forecast_results", "category_forecasts", "prediction_intervals",
            "forecast_config", "forecast_summary"
        ],
        "description": "7-day patient forecasts by category",
        "icon": "🏥"
    },
    "Staff Planner": {
        "keys": [
            "staff_optimization_results", "schedule_output", "staff_config",
            "shift_assignments", "staff_costs"
        ],
        "description": "Staff scheduling optimization results",
        "icon": "👥"
    },
    "Supply Planner": {
        "keys": [
            "inventory_optimization_results", "reorder_schedule",
            "inventory_config", "supply_costs"
        ],
        "description": "Inventory optimization results",
        "icon": "📦"
    },
    "Action Center": {
        "keys": ["recommendations", "action_items", "ai_insights"],
        "description": "AI-powered recommendations",
        "icon": "🎬"
    }
}


def render_results_storage_panel(
    page_key: str,
    custom_keys: Optional[List[str]] = None,
    show_in_sidebar: bool = True
):
    """
    Render save/load/delete buttons for pipeline results.
    
    Args:
        page_key: Must match a key in PAGE_STORAGE_CONFIG
        custom_keys: Override default keys for this page
        show_in_sidebar: If True, render in sidebar; else in main area
    """
    config = PAGE_STORAGE_CONFIG.get(page_key, {})
    keys_to_save = custom_keys or config.get("keys", [])
    description = config.get("description", "Pipeline results")
    icon = config.get("icon", "💾")
    
    container = st.sidebar if show_in_sidebar else st
    
    with container.expander(f"{icon} Cloud Storage", expanded=False):
        st.caption(description)
        
        col1, col2 = st.columns(2)
        
        # === SAVE BUTTON ===
        with col1:
            if st.button("⬆️ Save", key=f"save_btn_{page_key}", use_container_width=True):
                _execute_save(page_key, keys_to_save)
        
        # === LOAD BUTTON ===
        with col2:
            if st.button("⬇️ Load", key=f"load_btn_{page_key}", use_container_width=True):
                _execute_load(page_key, keys_to_save)
        
        # === DELETE BUTTON ===
        if st.button("🗑️ Clear Saved", key=f"delete_btn_{page_key}", use_container_width=True):
            _execute_delete(page_key)
        
        # === SHOW STATUS ===
        _show_storage_status(page_key)


def _execute_save(page_key: str, keys: List[str]):
    """Execute save operation for specified keys."""
    try:
        storage = get_storage_service()
        saved_count = 0
        saved_items = []
        
        with st.spinner("💾 Saving to cloud..."):
            for key in keys:
                if key in st.session_state and st.session_state[key] is not None:
                    success, msg = storage.save_results(
                        page_key=page_key,
                        result_key=key,
                        data=st.session_state[key],
                        metadata={"source_session_key": key}
                    )
                    if success:
                        saved_count += 1
                        saved_items.append(key)
        
        if saved_count > 0:
            st.success(f"✅ Saved {saved_count} item(s): {', '.join(saved_items)}")
            st.balloons()
        else:
            st.warning("⚠️ No data found to save. Run the pipeline first.")
    
    except Exception as e:
        st.error(f"❌ Save failed: {str(e)}")


def _execute_load(page_key: str, keys: List[str]):
    """Execute load operation for specified keys."""
    try:
        storage = get_storage_service()
        loaded_count = 0
        loaded_items = []
        
        with st.spinner("☁️ Loading from cloud..."):
            for key in keys:
                data = storage.load_results(page_key, key)
                if data is not None:
                    st.session_state[key] = data
                    loaded_count += 1
                    loaded_items.append(key)
        
        if loaded_count > 0:
            st.success(f"✅ Loaded {loaded_count} item(s): {', '.join(loaded_items)}")
            st.rerun()
        else:
            st.info("ℹ️ No saved data found for this page.")
    
    except Exception as e:
        st.error(f"❌ Load failed: {str(e)}")


def _execute_delete(page_key: str):
    """Execute delete operation for a page."""
    try:
        storage = get_storage_service()
        storage.delete_all_for_page(page_key)
        st.success("✅ Cleared saved data for this page")
    except Exception as e:
        st.error(f"❌ Delete failed: {str(e)}")


def _show_storage_status(page_key: str):
    """Display what's currently saved for this page."""
    try:
        storage = get_storage_service()
        saved = storage.list_saved_results(page_key)
        
        if saved:
            st.caption(f"📦 Saved items ({len(saved)}):")
            for item in saved:
                result_key = item.get('result_key', 'unknown')
                size_bytes = item.get('data_size_bytes', 0) or 0
                size_kb = size_bytes / 1024
                updated = item.get('updated_at', '')[:10]  # Date only
                st.caption(f"  • {result_key} ({size_kb:.1f} KB) - {updated}")
        else:
            st.caption("📭 No saved data")
    except Exception:
        st.caption("⚠️ Could not check storage status")


def auto_load_if_available(page_key: str, force_reload: bool = False):
    """
    Automatically load saved results when a page loads.
    
    Call this at the TOP of each page, right after authentication check.
    Only loads data if session_state is empty (won't overwrite existing data).
    
    Args:
        page_key: Must match a key in PAGE_STORAGE_CONFIG
        force_reload: If True, reload even if data exists in session
    """
    # Flag to ensure we only run once per session per page
    flag_key = f"__autoloaded_{page_key}"
    
    if not force_reload and st.session_state.get(flag_key, False):
        return
    
    config = PAGE_STORAGE_CONFIG.get(page_key, {})
    keys = config.get("keys", [])
    
    if not keys:
        st.session_state[flag_key] = True
        return
    
    try:
        storage = get_storage_service()
        loaded_count = 0
        loaded_items = []
        
        for key in keys:
            # Skip if data already exists in session (unless force_reload)
            if not force_reload and key in st.session_state and st.session_state[key] is not None:
                continue
            
            data = storage.load_results(page_key, key)
            if data is not None:
                st.session_state[key] = data
                loaded_count += 1
                loaded_items.append(key)
        
        if loaded_count > 0:
            st.toast(f"☁️ Auto-loaded {loaded_count} saved result(s)", icon="✅")
    
    except Exception as e:
        # Fail silently - don't disrupt user experience
        print(f"Auto-load failed for {page_key}: {e}")
    
    finally:
        st.session_state[flag_key] = True


def render_pipeline_status_dashboard():
    """
    Render a visual dashboard showing pipeline progress and saved stages.
    Useful for Welcome page or Dashboard.
    """
    st.subheader("📊 Pipeline Progress & Cloud Storage")
    
    try:
        storage = get_storage_service()
        all_saved = storage.list_saved_results()
        stats = storage.get_storage_stats()
        
        # Group saved items by page
        by_page = {}
        for item in all_saved:
            page = item['page_key']
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(item)
        
        # Pipeline stages in order
        pipeline_order = [
            "Upload Data", "Prepare Data", "Explore Data",
            "Baseline Models", "Feature Studio", "Feature Selection",
            "Train Models", "Patient Forecast", "Staff Planner", "Supply Planner"
        ]
        
        # Display pipeline progress
        st.markdown("**Pipeline Stages:**")
        
        cols = st.columns(5)
        for i, page in enumerate(pipeline_order):
            config = PAGE_STORAGE_CONFIG.get(page, {})
            icon = config.get("icon", "📄")
            
            with cols[i % 5]:
                if page in by_page:
                    count = len(by_page[page])
                    total_kb = sum((item.get('data_size_bytes', 0) or 0) for item in by_page[page]) / 1024
                    st.success(f"{icon} **{page}**\n\n✅ {count} items ({total_kb:.0f} KB)")
                else:
                    st.warning(f"{icon} **{page}**\n\n⬜ Not saved")
        
        # Storage summary
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Items", stats.get('total_items', 0))
        with col2:
            st.metric("Total Storage", f"{stats.get('total_mb', 0):.2f} MB")
        with col3:
            st.metric("Saved Stages", f"{len(by_page)}/{len(pipeline_order)}")
        
        # Clear all button
        st.markdown("---")
        if st.button("🗑️ Clear ALL Saved Data", type="secondary"):
            if st.session_state.get("confirm_clear_all", False):
                storage.delete_all_user_results()
                st.success("✅ All saved data cleared")
                st.session_state["confirm_clear_all"] = False
                st.rerun()
            else:
                st.session_state["confirm_clear_all"] = True
                st.warning("⚠️ Click again to confirm deletion of ALL saved data")
    
    except Exception as e:
        st.error(f"Could not load pipeline status: {e}")


def get_page_storage_keys(page_key: str) -> List[str]:
    """Get the list of session_state keys for a page."""
    config = PAGE_STORAGE_CONFIG.get(page_key, {})
    return config.get("keys", [])
```

---

### STEP 4: Integrate into Each Pipeline Page

For EACH page listed below, add the following code:

**At the TOP of the file (after existing imports):**
```python
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available
```

**Right after authentication check (usually after `require_authentication()`):**
```python
auto_load_if_available("Page Name")  # Use exact name from config
```

**In the sidebar (usually after other sidebar content):**
```python
render_results_storage_panel("Page Name")
```

**Files to update:**

| File | Page Key to Use |
|------|-----------------|
| `pages/02_Upload_Data.py` | `"Upload Data"` |
| `pages/03_Prepare_Data.py` | `"Prepare Data"` |
| `pages/04_Explore_Data.py` | `"Explore Data"` |
| `pages/05_Baseline_Models.py` | `"Baseline Models"` |
| `pages/06_Feature_Studio.py` | `"Feature Studio"` |
| `pages/07_Feature_Selection.py` | `"Feature Selection"` |
| `pages/08_Train_Models.py` | `"Train Models"` |
| `pages/09_Model_Results.py` | `"Model Results"` |
| `pages/10_Patient_Forecast.py` | `"Patient Forecast"` |
| `pages/11_Staff_Planner.py` | `"Staff Planner"` |
| `pages/12_Supply_Planner.py` | `"Supply Planner"` |
| `pages/13_Action_Center.py` | `"Action Center"` |

---

### STEP 5: Add Pipeline Dashboard to Welcome Page

In `Welcome.py`, after successful login, add:

```python
from app_core.ui.results_storage_ui import render_pipeline_status_dashboard

# After authentication successful:
if st.session_state.get("authenticated"):
    render_pipeline_status_dashboard()
```

---

### STEP 6: Test the Implementation

Run these tests in order:

1. **Database Test:**
   - Verify `app_results` table exists in Supabase
   - Check columns: id, user_id, page_key, result_key, result_data, metadata, data_size_bytes

2. **Save Test:**
   - Go to `02_Upload_Data.py`
   - Upload a CSV file
   - Click "⬆️ Save" in the Cloud Storage panel
   - Check Supabase table for the saved record

3. **Load Test:**
   - Refresh the page completely (Ctrl+F5)
   - Verify data auto-loads (toast notification)
   - OR click "⬇️ Load" manually

4. **Full Pipeline Test:**
   - Run through: Upload → Prepare → Feature Studio → Baseline Models → Train Models
   - Save at each step
   - Close browser completely
   - Reopen → Login → Verify all stages auto-load
   - Go directly to Forecast page (should have all required data)

5. **Isolation Test:**
   - Login as different user
   - Verify they don't see first user's data

---

### Expected Outcomes

After full implementation:

✅ Every pipeline stage can be saved to Supabase with one click
✅ Results auto-load when returning to any page
✅ Visual pipeline dashboard shows what's saved
✅ No more waiting 10-30 minutes to re-run completed stages
✅ Results persist across browser sessions and devices
✅ Each user's data is completely isolated (Row Level Security)
✅ Support for complex data types (DataFrames, NumPy arrays, nested dicts)
✅ Storage statistics and management tools
```

---

## Summary of Files to Create/Modify

| Action | File Path |
|--------|-----------|
| CREATE | `app_core/data/results_storage_service.py` |
| CREATE/UPDATE | `app_core/ui/results_storage_ui.py` |
| UPDATE | `pages/02_Upload_Data.py` |
| UPDATE | `pages/03_Prepare_Data.py` |
| UPDATE | `pages/04_Explore_Data.py` |
| UPDATE | `pages/05_Baseline_Models.py` |
| UPDATE | `pages/06_Feature_Studio.py` |
| UPDATE | `pages/07_Feature_Selection.py` |
| UPDATE | `pages/08_Train_Models.py` |
| UPDATE | `pages/09_Model_Results.py` |
| UPDATE | `pages/10_Patient_Forecast.py` |
| UPDATE | `pages/11_Staff_Planner.py` |
| UPDATE | `pages/12_Supply_Planner.py` |
| UPDATE | `pages/13_Action_Center.py` |
| UPDATE | `Welcome.py` |

---

## Session State Keys Reference

These are the key `st.session_state` variables that will be persisted:

### Data Pipeline
- `uploaded_data` - Raw uploaded file data
- `prepared_data` - Cleaned/processed DataFrame
- `fused_data` - Multi-source fused data

### Feature Engineering
- `feature_engineering` - Feature engineering results
- `engineered_features` - Generated features DataFrame
- `selected_features` - Final selected feature list
- `feature_importance` - Feature importance scores

### Models
- `arima_results` - ARIMA model results
- `sarimax_results` - SARIMAX model results
- `seasonal_proportion_results` - Seasonal proportions
- `ml_mh_results` - Multi-horizon ML results
- `xgboost_results` - XGBoost training results
- `lstm_results` - LSTM training results
- `ann_results` - ANN training results
- `hybrid_results` - Hybrid model results

### Forecasts & Optimization
- `forecast_results` - 7-day forecast results
- `category_forecasts` - By-category forecasts
- `staff_optimization_results` - Staff scheduling output
- `inventory_optimization_results` - Inventory optimization output
