# =============================================================================
# app_core/models/ml/model_registry.py
# Central Registry for All ML Model Configurations
# Single source of truth for model parameters, UI settings, and pipelines
# =============================================================================

MODEL_REGISTRY = {
    "LSTM": {
        "display_name": "Long Short-Term Memory (LSTM)",
        "icon": "🧠",
        "description": "Recurrent neural network for sequential time series data",

        # Automatic mode bounds (for hyperparameter search)
        "auto_params": {
            "max_layers": {"min": 1, "max": 6, "default": 3, "step": 1},
            "max_units": {"min": 8, "max": 1024, "default": 256, "step": 8},
            "max_epochs": {"min": 5, "max": 500, "default": 50, "step": 1},
            "lr_min": {"min": 1e-6, "max": 1e-1, "default": 1e-5, "format": "%.6f"},
            "lr_max": {"min": 1e-6, "max": 1e-1, "default": 1e-2, "format": "%.6f"},
        },

        # Manual mode defaults (for direct configuration)
        "manual_params": {
            "layers": {"min": 1, "max": 5, "default": 1, "step": 1, "label": "LSTM layers"},
            "hidden_units": {"min": 8, "max": 512, "default": 64, "step": 8, "label": "Hidden units"},
            "epochs": {"min": 1, "max": 500, "default": 20, "step": 1, "label": "Epochs"},
            "lr": {"min": 1e-6, "max": 1e-1, "default": 1e-3, "format": "%.6f", "label": "Learning rate"},
        },

        # Search methods available for this model
        "search_methods": ["Grid (bounded)", "Random", "Bayesian (planned)"],
        "default_search": "Grid (bounded)",

        # Pipeline class reference (will be implemented later)
        "pipeline_class": "LSTMPipeline",
    },

    "ANN": {
        "display_name": "Artificial Neural Network (ANN)",
        "icon": "🤖",
        "description": "Feedforward neural network for regression forecasting",

        # Automatic mode bounds
        "auto_params": {
            "max_layers": {"min": 1, "max": 8, "default": 4, "step": 1},
            "max_neurons": {"min": 8, "max": 1024, "default": 256, "step": 8},
            "max_epochs": {"min": 5, "max": 500, "default": 60, "step": 1},
        },

        # Manual mode defaults
        "manual_params": {
            "hidden_layers": {"min": 1, "max": 8, "default": 2, "step": 1, "label": "Hidden layers"},
            "neurons": {"min": 4, "max": 1024, "default": 64, "step": 4, "label": "Neurons per layer"},
            "epochs": {"min": 1, "max": 500, "default": 30, "step": 1, "label": "Epochs"},
            "activation": {
                "type": "select",
                "options": ["relu", "tanh", "sigmoid"],
                "default": "relu",
                "label": "Activation"
            },
        },

        # Search methods
        "search_methods": ["Grid (bounded)", "Random", "Bayesian (planned)"],
        "default_search": "Grid (bounded)",

        "pipeline_class": "ANNPipeline",
    },

    "XGBoost": {
        "display_name": "Extreme Gradient Boosting (XGBoost)",
        "icon": "⚡",
        "description": "Gradient boosting for structured/tabular data forecasting",

        # Automatic mode bounds
        "auto_params": {
            "n_estimators_min": {"min": 50, "max": 5000, "default": 100, "step": 50},
            "n_estimators_max": {"min": 50, "max": 5000, "default": 1000, "step": 50},
            "depth_min": {"min": 1, "max": 30, "default": 3, "step": 1},
            "depth_max": {"min": 1, "max": 30, "default": 12, "step": 1},
            "eta_min": {"min": 0.001, "max": 1.0, "default": 0.01, "step": 0.001},
            "eta_max": {"min": 0.001, "max": 1.0, "default": 0.3, "step": 0.001},
        },

        # Manual mode defaults
        "manual_params": {
            "n_estimators": {"min": 50, "max": 5000, "default": 300, "step": 50, "label": "n_estimators"},
            "max_depth": {"min": 1, "max": 30, "default": 6, "step": 1, "label": "max_depth"},
            "min_child_weight": {"min": 1, "max": 50, "default": 1, "step": 1, "label": "min_child_weight"},
            "eta": {"min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001, "label": "Learning rate (eta)"},
        },

        # Search methods
        "search_methods": ["Grid (bounded)", "Random", "Bayesian (planned)"],
        "default_search": "Grid (bounded)",

        "pipeline_class": "XGBoostPipeline",
    },
}


def get_model_config(model_key: str) -> dict:
    """
    Get configuration for a specific model.

    Args:
        model_key: Model identifier (e.g., 'LSTM', 'ANN', 'XGBoost')

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model_key not found in registry
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_key}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]


def get_all_model_names() -> list:
    """Get list of all registered model names."""
    return list(MODEL_REGISTRY.keys())


def get_all_display_names() -> list:
    """Get list of all model display names."""
    return [MODEL_REGISTRY[m]["display_name"] for m in MODEL_REGISTRY.keys()]


def map_display_to_key(display_name: str) -> str:
    """
    Map display name back to model key.

    Args:
        display_name: Human-readable model name

    Returns:
        Model key

    Raises:
        ValueError: If display name not found
    """
    for key, config in MODEL_REGISTRY.items():
        if config["display_name"] == display_name:
            return key
    raise ValueError(f"Display name '{display_name}' not found in registry")
