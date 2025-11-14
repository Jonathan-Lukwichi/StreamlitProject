# =============================================================================
# app_core/data/io.py — Simple file persistence helpers for Data Hub
# =============================================================================
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import pandas as pd


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def ensure_data_dir():
    """Ensure the /data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(df: pd.DataFrame, dataset_name: str, extension: str = "csv"):
    """
    Save a dataset to the /data folder.

    Args:
        df: DataFrame to save
        dataset_name: Name of the dataset (e.g., "patient", "weather", "calendar", "reason_of_visit")
        extension: File extension ("csv" or "xlsx")
    """
    ensure_data_dir()
    filepath = DATA_DIR / f"{dataset_name}.{extension}"

    if extension.lower() == "csv":
        df.to_csv(filepath, index=False)
    elif extension.lower() in ["xlsx", "xls"]:
        df.to_excel(filepath, index=False)
    else:
        # Default to CSV
        df.to_csv(filepath, index=False)


def load_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load a dataset from the /data folder.
    Tries CSV first, then XLSX.

    Args:
        dataset_name: Name of the dataset (e.g., "patient", "weather", "calendar", "reason_of_visit")

    Returns:
        DataFrame if file exists, None otherwise
    """
    ensure_data_dir()

    # Try CSV first
    csv_path = DATA_DIR / f"{dataset_name}.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    # Try XLSX
    xlsx_path = DATA_DIR / f"{dataset_name}.xlsx"
    if xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path)
        except Exception:
            pass

    return None


def delete_dataset(dataset_name: str):
    """
    Delete a dataset from the /data folder.

    Args:
        dataset_name: Name of the dataset (e.g., "patient", "weather", "calendar", "reason_of_visit")
    """
    ensure_data_dir()

    for ext in ["csv", "xlsx"]:
        filepath = DATA_DIR / f"{dataset_name}.{ext}"
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception:
                pass
