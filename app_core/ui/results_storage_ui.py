# =============================================================================
# app_core/ui/results_storage_ui.py
# UI components for saving/loading HealthForecast AI pipeline results.
# Provides buttons, auto-load functionality, and pipeline status dashboard.
# Based on memory1.md implementation guide.
# =============================================================================

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
        "keys": ["prepared_data", "fused_data", "cleaning_report", "preparation_config",
                 "merged_data", "processed_df", "data_summary", "target_columns", "date_column"],
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
            "arima_results", "sarimax_results", "arima_mh_results", "sarimax_mh_results",
            "arima_multi_target_results", "sarimax_multi_target_results",
            "seasonal_proportions_result", "seasonal_proportions_config",
            "model_comparison", "selected_model", "arima_order", "sarimax_order",
            "sarimax_seasonal_order", "sarimax_features", "sarimax_horizons"
        ],
        "description": "ARIMA/SARIMAX model results and seasonal proportions",
        "icon": "📊"
    },
    "Feature Studio": {
        "keys": [
            "feature_engineering", "cv_config", "cv_enabled", "cv_folds",
            "differencing_config", "fourier_config"
        ],
        "description": "Engineered features (ED_1...ED_7, Target_1...Target_7)",
        "icon": "🛠️"
    },
    "Feature Selection": {
        "keys": ["feature_selection"],
        "description": "Selected features and importance analysis",
        "icon": "🎯"
    },
    "Train Models": {
        "keys": [
            "ml_mh_results", "modeling_config", "seasonal_proportions_config",
            "ml_all_models_trained", "opt_all_models_results", "opt_last_model",
            "opt_last_method", "opt_last_mode", "hybrid_pipeline_results",
            "ensemble_results", "decomposition_results", "stacking_results",
            "lstm_sarimax_results", "lstm_xgb_results", "lstm_ann_results"
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

        if not storage.is_connected():
            st.error("❌ Supabase not connected")
            return

        saved_count = 0
        saved_items = []
        failed_items = []
        skipped_items = []

        with st.spinner("💾 Saving to cloud..."):
            for key in keys:
                if key in st.session_state and st.session_state[key] is not None:
                    try:
                        success, msg = storage.save_results(
                            page_key=page_key,
                            result_key=key,
                            data=st.session_state[key],
                            metadata={"source_session_key": key}
                        )
                        if success:
                            saved_count += 1
                            saved_items.append(key)
                        else:
                            failed_items.append(f"{key}: {msg}")
                    except Exception as save_err:
                        failed_items.append(f"{key}: {str(save_err)[:50]}")
                else:
                    skipped_items.append(key)

        # Show results
        if saved_count > 0:
            st.success(f"✅ Saved {saved_count} item(s): {', '.join(saved_items)}")
            st.balloons()

        if failed_items:
            st.error(f"❌ Failed to save {len(failed_items)} item(s):")
            for item in failed_items[:3]:  # Show first 3 errors
                st.caption(f"  • {item}")
            if len(failed_items) > 3:
                st.caption(f"  ... and {len(failed_items) - 3} more")

        if saved_count == 0 and not failed_items:
            available_keys = [k for k in keys if k in st.session_state and st.session_state[k] is not None]
            if available_keys:
                st.warning(f"⚠️ Found data but failed to save. Keys with data: {', '.join(available_keys)}")
            else:
                st.warning("⚠️ No data found to save. Run the pipeline first.")

    except Exception as e:
        st.error(f"❌ Save failed: {str(e)}")


def _execute_load(page_key: str, keys: List[str]):
    """Execute load operation for specified keys."""
    try:
        storage = get_storage_service()

        if not storage.is_connected():
            st.error("❌ Supabase not connected")
            return

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

        if not storage.is_connected():
            st.error("❌ Supabase not connected")
            return

        storage.delete_all_for_page(page_key)
        st.success("✅ Cleared saved data for this page")
    except Exception as e:
        st.error(f"❌ Delete failed: {str(e)}")


def _show_storage_status(page_key: str):
    """Display what's currently saved for this page."""
    try:
        storage = get_storage_service()

        if not storage.is_connected():
            st.caption("⚠️ Supabase not connected")
            return

        saved = storage.list_saved_results(page_key)

        if saved:
            st.caption(f"📦 Saved items ({len(saved)}):")
            for item in saved:
                result_key = item.get('result_key', 'unknown')
                size_bytes = item.get('data_size_bytes', 0) or 0
                size_kb = size_bytes / 1024
                updated = item.get('updated_at', '')[:10] if item.get('updated_at') else ''
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

        if not storage.is_connected():
            st.session_state[flag_key] = True
            return

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

        if not storage.is_connected():
            st.warning("⚠️ Supabase not connected. Cloud storage unavailable.")
            return

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
