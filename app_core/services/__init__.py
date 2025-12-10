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

Migration Guide:
---------------
1. Import services at the top of your page
2. Create service instances (once per page)
3. Replace inline validation with service calls
4. Replace model training with service calls
"""

from .base_service import BaseService, ServiceResult
from .data_service import DataService
from .modeling_service import ModelingService, ModelConfig, ModelResult

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
