# =============================================================================
# app_core/services/__init__.py
# Service Layer for HealthForecast AI
# Separates business logic from UI presentation
# =============================================================================
"""
Service Layer for HealthForecast AI

This module provides a clean service layer that separates business logic
from UI components. Services can be gradually adopted in existing pages.

Usage Example:
-------------
    from app_core.services import DataService, ModelingService
    from app_core.services.modeling_service import ModelConfig

    # Data validation and processing
    data_service = DataService()
    validation = data_service.validate_data(df, required_columns=['datetime', 'value'])
    if validation.success:
        print("Data is valid!")

    # Model training
    modeling_service = ModelingService()
    config = ModelConfig(
        model_type='xgboost',
        target_column='Target_1',
        hyperparameters={'n_estimators': 100, 'max_depth': 6}
    )
    result = modeling_service.train_model(X_train, y_train, X_test, y_test, config)
    if result.success:
        print(f"RMSE: {result.data.metrics['rmse']}")

Offline-First Data Access:
-------------------------
    from app_core.services import get_unified_data_service

    # Get the unified service (handles online/offline automatically)
    service = get_unified_data_service()

    # Fetch data (auto-selects source based on connectivity)
    df = service.fetch_patient_data()

    # Save data (queues for sync if offline)
    service.save_patient_data(df)

    # Check status
    print(f"Online: {service.is_online}")

Migration Guide:
---------------
1. Import services at the top of your page
2. Create service instances (once per page)
3. Replace inline validation with service calls
4. Replace model training with service calls
5. Use UnifiedDataService for all data fetch/save operations
"""

from .base_service import BaseService, ServiceResult
from .data_service import DataService
from .modeling_service import ModelingService, ModelConfig, ModelResult

# Import unified data service for offline-first architecture
from app_core.offline import (
    UnifiedDataService,
    get_data_service as get_unified_data_service,
)

__all__ = [
    # Base classes
    "BaseService",
    "ServiceResult",
    # Data operations
    "DataService",
    # Modeling operations
    "ModelingService",
    "ModelConfig",
    "ModelResult",
    # Unified Data Service (offline-first)
    "UnifiedDataService",
    "get_unified_data_service",
    # Convenience functions
    "quick_validate",
    "quick_train_xgboost",
    "fetch_patient_data",
    "fetch_inventory_data",
    "fetch_financial_data",
    "is_online",
    "get_connection_status",
]


# =============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK ADOPTION
# =============================================================================

def quick_validate(df, required_columns=None):
    """
    Quick data validation - convenience wrapper.

    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names

    Returns:
        Tuple of (is_valid: bool, error_message: str or None)

    Example:
        is_valid, error = quick_validate(df, ['datetime', 'value'])
        if not is_valid:
            st.error(error)
    """
    service = DataService()
    result = service.validate_data(df, required_columns=required_columns)
    return result.success, result.error


def quick_train_xgboost(X_train, y_train, X_test, y_test, **hyperparameters):
    """
    Quick XGBoost training - convenience wrapper.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        **hyperparameters: XGBoost hyperparameters

    Returns:
        Tuple of (success: bool, result: ModelResult or error_message)

    Example:
        success, result = quick_train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100)
        if success:
            print(f"RMSE: {result.metrics['rmse']}")
    """
    service = ModelingService()
    config = ModelConfig(
        model_type='xgboost',
        target_column='target',
        hyperparameters=hyperparameters or {'n_estimators': 100, 'max_depth': 6}
    )
    result = service.train_model(X_train, y_train, X_test, y_test, config)
    if result.success:
        return True, result.data
    return False, result.error


# =============================================================================
# OFFLINE-FIRST DATA ACCESS CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_patient_data(start_date=None, end_date=None):
    """
    Fetch patient arrivals data (offline-first).

    Automatically uses local database when offline, syncs when online.

    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with patient arrivals data

    Example:
        df = fetch_patient_data()
        df_2024 = fetch_patient_data(start_date='2024-01-01', end_date='2024-12-31')
    """
    return get_unified_data_service().fetch_patient_data(start_date, end_date)


def fetch_inventory_data(category=None):
    """
    Fetch inventory data (offline-first).

    Args:
        category: Optional category filter

    Returns:
        DataFrame with inventory data
    """
    return get_unified_data_service().fetch_inventory_data(category)


def fetch_financial_data(department=None):
    """
    Fetch financial data (offline-first).

    Args:
        department: Optional department filter

    Returns:
        DataFrame with financial data
    """
    return get_unified_data_service().fetch_financial_data(department)


def is_online():
    """
    Check if the app is online.

    Returns:
        True if connected to cloud services
    """
    return get_unified_data_service().is_online


def get_connection_status():
    """
    Get detailed connection status.

    Returns:
        Dict with connection details
    """
    return get_unified_data_service().get_status()
