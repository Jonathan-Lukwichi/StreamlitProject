from __future__ import annotations
import pandas as pd

def find_time_like_col(df: pd.DataFrame) -> str | None:
    """
    Finds the best candidate for a time/date column in a DataFrame.
    Prioritizes common names, then searches for substrings in column names.
    """
    if df is None or df.empty:
        return None
        
    priority = ["datetime", "timestamp", "date_time", "date", "ds"]
    lower_map = {c.lower(): c for c in df.columns}
    
    for p in priority:
        if p in lower_map:
            return lower_map[p]
            
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "stamp" in cl:
            return c
            
    return None
