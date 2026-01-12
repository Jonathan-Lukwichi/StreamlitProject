# Prompts for Implementing Persistent Results Storage with Supabase

This document provides a series of actionable prompts to implement a persistent storage layer for your Streamlit application using Supabase. This will allow you to save and load the results of long-running computations, preventing the need to re-run them every time the app starts.

---

### **Prompt 1: Set Up the Database Table in Supabase**

**Instruction:**
Go to your Supabase project dashboard, navigate to the **SQL Editor**, and execute the following SQL script. This script will create a new table named `app_results` designed to store your application's computational results. It also enables Row Level Security and creates policies to ensure that each user can only access their own data.

```sql
-- Create the table to store serialized results
CREATE TABLE app_results (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE, -- Optional, but good practice
  page_key TEXT NOT NULL,          -- A key for the page, e.g., 'Train Models'
  result_key TEXT NOT NULL,       -- A key for the specific result, e.g., 'ml_mh_results'
  result_data JSONB NOT NULL,     -- The main results data, stored as a JSONB object
  metadata JSONB                  -- Optional metadata, e.g., model accuracy, runtime
);

-- Create a unique index to prevent duplicate entries for the same user, page, and result.
-- This is crucial for the upsert functionality.
CREATE UNIQUE INDEX ON app_results (user_id, page_key, result_key);

-- Enable Row Level Security on the new table
ALTER TABLE app_results ENABLE ROW LEVEL SECURITY;

-- Create a policy that allows users to insert, select, update, and delete their own results.
-- The `auth.uid()` function gets the ID of the currently authenticated user.
CREATE POLICY "Users can manage their own results"
ON app_results FOR ALL
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

```

---

### **Prompt 2: Create the Python Service for Storage Logic**

**Instruction:**
Create a new Python file at `app_core/data/results_storage_service.py`. This module will contain the `ResultsStorageService` class, which handles the logic for saving and loading data. This includes a custom JSON encoder to correctly handle data types from libraries like NumPy and Pandas, which are not natively JSON serializable.

```python
# In new file: app_core/data/results_storage_service.py

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from supabase import Client

# Make sure you have a centralized way to get your Supabase client
# For example, from a file like this:
from app_core.data.supabase_client import get_supabase_client

class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy and other non-standard types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (pd.Timedelta, pd.Interval)):
            return str(obj)
        return super(NpEncoder, self).default(obj)

class ResultsStorageService:
    """
    A service class to manage saving and loading application results
    to a persistent Supabase table.
    """
    def __init__(self, client: Client):
        self.client = client
        self.table_name = "app_results"

    def save_results(self, page_key: str, result_key: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        Serializes Python objects (including DataFrames) and saves them to Supabase.
        Uses 'upsert' to either create a new record or update an existing one.
        """
        user = self.client.auth.get_user()
        if not user or not hasattr(user, 'user'):
            raise Exception("User not authenticated. Cannot save results.")
        
        # Serialize the data payload to a JSON string using our custom encoder
        json_string = json.dumps(data, cls=NpEncoder)

        record = {
            "user_id": user.user.id,
            "page_key": page_key,
            "result_key": result_key,
            "result_data": json_string,
            "metadata": metadata
        }
        
        # Use upsert: if a record with the same user_id, page_key, and result_key exists,
        # it will be updated; otherwise, a new one will be inserted.
        self.client.table(self.table_name).upsert(
            record, 
            on_conflict="user_id,page_key,result_key"
        ).execute()

    def load_results(self, page_key: str, result_key: str) -> Optional[Any]:
        """
        Loads and deserializes results from the Supabase table for the current user.
        """
        user = self.client.auth.get_user()
        if not user or not hasattr(user, 'user'):
            return None

        response = self.client.table(self.table_name) \
            .select("result_data") \
            .eq("user_id", user.user.id) \
            .eq("page_key", page_key) \
            .eq("result_key", result_key) \
            .limit(1) \
            .execute()

        if not response.data:
            return None

        # The result_data is stored as a JSON string, so we must parse it.
        # It's good practice to wrap this in a try-except block.
        try:
            return json.loads(response.data[0]['result_data'])
        except json.JSONDecodeError:
            # Handle cases where data might be corrupted or not valid JSON
            return None

def get_storage_service() -> ResultsStorageService:
    """Singleton factory for the ResultsStorageService."""
    return ResultsStorageService(get_supabase_client())

```

---

### **Prompt 3: Integrate the "Save" Functionality into the UI**

**Instruction:**
Modify the `render_results_storage_panel` function located in `app_core/ui/results_storage_ui.py`. Update the logic inside the "Save Results to Cloud" button's `if` block. This new logic will instantiate the `ResultsStorageService`, loop through the specified `custom_keys`, and save the corresponding data from `st.session_state` to Supabase.

```python
# In file: app_core/ui/results_storage_ui.py

# Add this import at the top of the file
from app_core.data.results_storage_service import get_storage_service

# ... (keep existing imports and code) ...

def render_results_storage_panel(page_type: str, page_title: str, custom_keys: list):
    # ... (keep existing expander UI code) ...

    if st.button("💾 Save Results to Cloud", key=f"save_results_{page_type}"):
        # --- START: New Implementation ---
        try:
            with st.spinner("Saving results to Supabase..."):
                storage_service = get_storage_service()
                
                saved_count = 0
                for key in custom_keys:
                    if key in st.session_state and st.session_state[key] is not None:
                        # Extract some metadata for easier querying later, if desired
                        metadata = {
                            "saved_at": datetime.now().isoformat(),
                            "data_type": str(type(st.session_state[key]))
                        }
                        
                        storage_service.save_results(
                            page_key=page_title, 
                            result_key=key, 
                            data=st.session_state[key],
                            metadata=metadata
                        )
                        saved_count += 1
                
            st.success(f"✅ Successfully saved {saved_count} result object(s) to the cloud!")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Failed to save results: {e}")
        # --- END: New Implementation ---

# ... (rest of the file) ...
```

---

### **Prompt 4: Integrate the "Load" Functionality into the UI**

**Instruction:**
Modify the `auto_load_if_available` function in `app_core/ui/results_storage_ui.py`. This function is designed to run when a page loads. Implement the logic to instantiate the `ResultsStorageService` and query Supabase for any previously saved results corresponding to the current page. If data is found, it should be loaded directly into `st.session_state`.

```python
# In file: app_core/ui/results_storage_ui.py

# Add this import at the top of the file
from app_core.data.results_storage_service import get_storage_service

# ... (keep existing imports and code) ...

def auto_load_if_available(page_key: str):
    """
    On page load, checks if there are saved results in Supabase for this
    page and automatically loads them into the session state.
    """
    # Use a flag to ensure this runs only once per page load
    session_key = f"__{page_key}_loaded_from_db"
    if st.session_state.get(session_key, False):
        return

    try:
        # Initialize the service
        storage_service = get_storage_service()
        
        # Check for saved data for this specific page_key
        response = storage_service.client.table("app_results") \
            .select("result_key, result_data") \
            .eq("user_id", storage_service.client.auth.get_user().user.id) \
            .eq("page_key", page_key) \
            .execute()

        if response.data:
            with st.spinner(f"Loading {len(response.data)} saved result(s) from the cloud..."):
                loaded_count = 0
                for record in response.data:
                    result_key = record.get("result_key")
                    result_data_str = record.get("result_data")
                    
                    if result_key and result_data_str:
                        # Deserialize the JSON string
                        data = json.loads(result_data_str)
                        
                        # If it was a DataFrame, reconstruct it
                        if isinstance(data, dict) and data.get("__type__") == "dataframe":
                            st.session_state[result_key] = pd.read_json(data['data'], orient='split')
                        else:
                            st.session_state[result_key] = data
                        
                        loaded_count += 1
                
                if loaded_count > 0:
                    st.toast(f"✅ Loaded {loaded_count} result(s) from previous session.", icon="☁️")

    except Exception as e:
        # Fail silently on auto-load to not disrupt user experience
        print(f"Auto-load failed for page '{page_key}': {e}")
    finally:
        # Set the flag to prevent re-running
        st.session_state[session_key] = True

# ... (rest of the file) ...