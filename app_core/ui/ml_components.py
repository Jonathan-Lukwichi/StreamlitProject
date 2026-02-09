# =============================================================================
# app_core/ui/ml_components.py
# Reusable UI Components for ML Model Configuration
# IMPORTANT: Preserves existing design (fluorescent effects, colors, styling)
# =============================================================================

from __future__ import annotations
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from app_core.models.ml.model_registry import (
    MODEL_REGISTRY,
    get_all_model_names,
    get_all_display_names,
    map_display_to_key,
    get_model_config
)


class MLUIComponents:
    """
    Reusable UI components for ML model configuration.
    All components preserve existing design and styling from Modeling Hub.
    """

    @staticmethod
    def render_model_selector(
        current_model: Optional[str] = None,
        training_status: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Generic model selection dropdown with training status indicators.

        Args:
            current_model: Currently selected model key (e.g., "XGBoost")
            training_status: Dict mapping model keys (lowercase) to trained status
                            e.g., {"xgboost": True, "lstm": False, "ann": True}

        Returns:
            Selected model key
        """
        model_names = get_all_model_names()  # ["LSTM", "ANN", "XGBoost"]
        display_names = get_all_display_names()

        # Build display options with training status badges
        display_options = []
        for i, model in enumerate(model_names):
            model_key = model.lower()
            is_trained = training_status.get(model_key, False) if training_status else False
            badge = " ✅" if is_trained else ""
            display_options.append(f"{display_names[i]}{badge}")

        # Find current index
        if current_model and current_model in model_names:
            default_idx = model_names.index(current_model)
        else:
            # Default to XGBoost (index 2)
            default_idx = model_names.index("XGBoost") if "XGBoost" in model_names else 0

        # Render dropdown with status indicators
        selected_display = st.selectbox(
            "Choose ML Model",
            options=display_options,
            index=default_idx,
            help="✅ = Model trained. Select a model to train or view results."
        )

        # Map back to model key (strip badge from display name)
        selected_idx = display_options.index(selected_display)
        return model_names[selected_idx]

    @staticmethod
    def render_parameter_mode(model_key: str, config: Dict) -> str:
        """
        Generic parameter mode selector (preserves existing design).

        Args:
            model_key: Model identifier (e.g., "LSTM", "ANN", "XGBoost")
            config: Configuration dictionary

        Returns:
            Selected mode ("Automatic" or "Manual")
        """
        # Determine current mode based on model-specific config
        param_mode_key = f"ml_{model_key.lower()}_param_mode"
        current_mode = config.get(param_mode_key, "Automatic")

        # Render radio (preserves existing options and layout)
        mode = st.radio(
            "How do you want to set parameters?",
            options=["Automatic (optimize)", "Manual"],
            horizontal=True,
            index=0 if current_mode == "Automatic" else 1,
            key=f"ml_mode_radio_{model_key}"
        )

        # Return simplified mode name
        return "Automatic" if "Automatic" in mode else "Manual"

    @staticmethod
    def render_search_method(model_key: str, config: Dict) -> str:
        """
        Generic search method selector (preserves existing design).

        Args:
            model_key: Model identifier
            config: Configuration dictionary

        Returns:
            Selected search method
        """
        model_config = get_model_config(model_key)
        search_methods = model_config["search_methods"]
        default_search = model_config["default_search"]

        # Get current selection from config
        search_key = f"{model_key.lower()}_search_method"
        current = config.get(search_key, default_search)

        # Find index
        try:
            default_idx = search_methods.index(current)
        except ValueError:
            default_idx = 0

        # Render selectbox (preserves existing label)
        return st.selectbox(
            "Search method",
            search_methods,
            index=default_idx,
            key=f"{model_key.lower()}_search_method_select"
        )

    @staticmethod
    def render_train_test_split(model_key: str, config: Dict) -> float:
        """
        Generic train/test split selector (preserves existing design).

        Args:
            model_key: Model identifier
            config: Configuration dictionary

        Returns:
            Split ratio (0.0-1.0)
        """
        current_ratio = config.get("split_ratio", 0.80)

        # Render slider (preserves existing label and range)
        ratio = st.slider(
            "Train/Validation split ratio",
            0.50, 0.95,
            float(current_ratio),
            0.01,
            key=f"{model_key.lower()}_split_ratio"
        )

        return float(ratio)

    @staticmethod
    def render_automatic_params(model_key: str, config: Dict) -> Dict[str, Any]:
        """
        Dynamically render automatic mode parameters (preserves existing design).

        Args:
            model_key: Model identifier
            config: Configuration dictionary

        Returns:
            Dictionary of parameter values
        """
        model_config = get_model_config(model_key)
        auto_params = model_config["auto_params"]
        param_values = {}

        # Caption (preserves existing design)
        st.caption("No training yet — placeholders for search settings.")

        # Render parameters dynamically based on model registry
        if model_key == "LSTM":
            # LSTM layout: 4 columns (preserves existing design)
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                spec = auto_params["max_layers"]
                param_values["lstm_max_layers"] = int(st.number_input(
                    "max layers", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c2:
                spec = auto_params["max_units"]
                param_values["lstm_max_units"] = int(st.number_input(
                    "max units", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c3:
                spec = auto_params["max_epochs"]
                param_values["lstm_max_epochs"] = int(st.number_input(
                    "max epochs", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c4:
                lr_min_spec = auto_params["lr_min"]
                lr_min = st.number_input(
                    "min LR", lr_min_spec["min"], lr_min_spec["max"],
                    lr_min_spec["default"], format=lr_min_spec["format"]
                )

                lr_max_spec = auto_params["lr_max"]
                lr_max = st.number_input(
                    "max LR", lr_max_spec["min"], lr_max_spec["max"],
                    lr_max_spec["default"], format=lr_max_spec["format"]
                )

                param_values["lstm_lr_min"] = float(lr_min)
                param_values["lstm_lr_max"] = float(lr_max)

        elif model_key == "ANN":
            # ANN layout: 3 columns (preserves existing design)
            c1, c2, c3 = st.columns(3)

            with c1:
                spec = auto_params["max_layers"]
                param_values["ann_max_layers"] = int(st.number_input(
                    "max layers", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c2:
                spec = auto_params["max_neurons"]
                param_values["ann_max_neurons"] = int(st.number_input(
                    "max neurons", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c3:
                spec = auto_params["max_epochs"]
                param_values["ann_max_epochs"] = int(st.number_input(
                    "max epochs", spec["min"], spec["max"], spec["default"], spec["step"]
                ))

        elif model_key == "XGBoost":
            # XGBoost layout: 3 columns (preserves existing design)
            c1, c2, c3 = st.columns(3)

            with c1:
                spec_min = auto_params["n_estimators_min"]
                param_values["xgb_n_estimators_min"] = int(st.number_input(
                    "min n_estimators", spec_min["min"], spec_min["max"],
                    spec_min["default"], spec_min["step"]
                ))

                spec_max = auto_params["n_estimators_max"]
                param_values["xgb_n_estimators_max"] = int(st.number_input(
                    "max n_estimators", spec_max["min"], spec_max["max"],
                    spec_max["default"], spec_max["step"]
                ))

            with c2:
                spec_min = auto_params["depth_min"]
                param_values["xgb_depth_min"] = int(st.number_input(
                    "min max_depth", spec_min["min"], spec_min["max"], spec_min["default"]
                ))

                spec_max = auto_params["depth_max"]
                param_values["xgb_depth_max"] = int(st.number_input(
                    "max max_depth", spec_max["min"], spec_max["max"], spec_max["default"]
                ))

            with c3:
                spec_min = auto_params["eta_min"]
                param_values["xgb_eta_min"] = float(st.number_input(
                    "min eta", spec_min["min"], spec_min["max"], spec_min["default"]
                ))

                spec_max = auto_params["eta_max"]
                param_values["xgb_eta_max"] = float(st.number_input(
                    "max eta", spec_max["min"], spec_max["max"], spec_max["default"]
                ))

        return param_values

    @staticmethod
    def render_manual_params(model_key: str, config: Dict) -> Dict[str, Any]:
        """
        Dynamically render manual mode parameters (preserves existing design).

        Args:
            model_key: Model identifier
            config: Configuration dictionary

        Returns:
            Dictionary of parameter values
        """
        model_config = get_model_config(model_key)
        manual_params = model_config["manual_params"]
        param_values = {}

        # Caption (preserves existing design)
        st.caption("Manual hyperparameters (placeholders).")

        if model_key == "LSTM":
            # LSTM layout: 4 columns (lookback, layers, hidden units, epochs) + learning rate below
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                spec = manual_params["lookback"]
                param_values["lookback_window"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c2:
                spec = manual_params["layers"]
                param_values["lstm_layers"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c3:
                spec = manual_params["hidden_units"]
                param_values["lstm_hidden_units"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c4:
                spec = manual_params["epochs"]
                param_values["lstm_epochs"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            # Learning rate below (full width)
            spec = manual_params["lr"]
            param_values["lstm_lr"] = float(st.number_input(
                spec["label"], spec["min"], spec["max"], spec["default"], format=spec["format"]
            ))

        elif model_key == "ANN":
            # ANN layout: 3 columns + activation (preserves existing design)
            c1, c2, c3 = st.columns(3)

            with c1:
                spec = manual_params["hidden_layers"]
                param_values["ann_hidden_layers"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c2:
                spec = manual_params["neurons"]
                param_values["ann_neurons"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c3:
                spec = manual_params["epochs"]
                param_values["ann_epochs"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            # Activation below
            spec = manual_params["activation"]
            param_values["ann_activation"] = st.selectbox(
                spec["label"], spec["options"],
                index=spec["options"].index(spec["default"])
            )

        elif model_key == "XGBoost":
            # XGBoost layout: 3 columns + eta (preserves existing design)
            c1, c2, c3 = st.columns(3)

            with c1:
                spec = manual_params["n_estimators"]
                param_values["xgb_n_estimators"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c2:
                spec = manual_params["max_depth"]
                param_values["xgb_max_depth"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            with c3:
                spec = manual_params["min_child_weight"]
                param_values["xgb_min_child_weight"] = int(st.number_input(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                ))

            # Learning rate (eta) below
            spec = manual_params["eta"]
            param_values["xgb_eta"] = float(st.number_input(
                spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
            ))

        return param_values

    @staticmethod
    def render_train_test_buttons(model_key: str) -> Tuple[bool, bool]:
        """
        Render train and test buttons (preserves existing design).

        Args:
            model_key: Model identifier

        Returns:
            Tuple of (train_clicked, test_clicked)
        """
        c1, c2 = st.columns(2)

        with c1:
            train_clicked = st.button(
                f"🚀 Train {model_key} (placeholder)",
                key="ml_train_button",
                use_container_width=True
            )

        with c2:
            test_clicked = st.button(
                f"🧪 Test {model_key} (placeholder)",
                key="ml_test_button",
                use_container_width=True
            )

        return train_clicked, test_clicked
