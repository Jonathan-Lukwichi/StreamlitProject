# =============================================================================
# app_core/ui/results_storage_ui.py
# Reusable UI Component for Results Storage
# Provides load/save/clear functionality for each page
# =============================================================================

import streamlit as st
from typing import Optional, List, Callable
from datetime import datetime

from app_core.data.results_storage_service import (
    get_results_storage,
    compute_current_dataset_hash,
)


def render_results_storage_panel(
    page_type: str,
    page_title: str,
    custom_keys: Optional[List[str]] = None,
    on_load_callback: Optional[Callable] = None,
    show_in_expander: bool = True,
) -> None:
    """
    Render the results storage panel for a page.

    This component provides:
    - Save current results to Supabase
    - Load previous results from Supabase
    - Clear saved results

    Args:
        page_type: One of 'data_preparation', 'feature_engineering',
                   'feature_selection', 'modeling_hub', 'benchmarks'
        page_title: Display name for the page (e.g., "Data Preparation")
        custom_keys: Additional session state keys to save/load
        on_load_callback: Optional callback to run after loading results
        show_in_expander: Whether to wrap in an expander (default: True)
    """
    storage = get_results_storage()

    if not storage.is_connected():
        st.info("💾 Connect to Supabase to enable results persistence.")
        return

    dataset_hash = compute_current_dataset_hash()

    def _render_panel():
        # Check for saved results
        saved_info = storage.get_saved_info(page_type, dataset_hash)
        has_saved = saved_info is not None

        # Status indicator
        if has_saved:
            saved_at = saved_info.get("saved_at", "")
            try:
                # Parse and format the datetime
                dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_time = saved_at[:16] if saved_at else "Unknown"

            keys_count = len(saved_info.get("keys_saved", []))
            st.success(f"✅ Saved results found ({keys_count} items, last saved: {formatted_time})")
        else:
            st.info("📭 No saved results for this dataset")

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "💾 Save Results",
                key=f"save_{page_type}",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("Saving..."):
                    success = storage.save_page_results(
                        page_type=page_type,
                        dataset_hash=dataset_hash,
                        custom_keys=custom_keys,
                    )
                    if success:
                        st.success("✅ Results saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save results")

        with col2:
            if st.button(
                "📥 Load Results",
                key=f"load_{page_type}",
                use_container_width=True,
                disabled=not has_saved,
            ):
                with st.spinner("Loading..."):
                    loaded = storage.load_page_results(
                        page_type=page_type,
                        dataset_hash=dataset_hash,
                    )
                    if loaded:
                        count = storage.apply_loaded_results(loaded)
                        st.success(f"✅ Loaded {count} items from saved results!")
                        if on_load_callback:
                            on_load_callback()
                        st.rerun()
                    else:
                        st.warning("No results to load")

        with col3:
            if st.button(
                "🗑️ Clear Saved",
                key=f"clear_{page_type}",
                use_container_width=True,
                disabled=not has_saved,
            ):
                success = storage.clear_page_results(
                    page_type=page_type,
                    dataset_hash=dataset_hash,
                )
                if success:
                    st.success("✅ Saved results cleared!")
                    st.rerun()

        # Show saved keys in a small detail section
        if has_saved and saved_info:
            keys_saved = saved_info.get("keys_saved", [])
            if keys_saved:
                with st.expander("📋 Saved items", expanded=False):
                    st.write(", ".join(keys_saved))

    # Render in expander or directly
    if show_in_expander:
        with st.expander("💾 Results Storage (Supabase)", expanded=False):
            _render_panel()
    else:
        _render_panel()


def render_global_storage_status() -> None:
    """
    Render a global status view of all saved results.
    Useful for the sidebar or a settings page.
    """
    storage = get_results_storage()

    if not storage.is_connected():
        st.warning("Supabase not connected")
        return

    dataset_hash = compute_current_dataset_hash()
    all_saved = storage.get_all_saved_pages(dataset_hash)

    if not all_saved:
        st.info("No saved results for current dataset")
        return

    st.write(f"**Saved Results** (Dataset: `{dataset_hash[:8]}...`)")

    for item in all_saved:
        page = item.get("page_type", "unknown")
        saved_at = item.get("saved_at", "")
        keys = item.get("keys_saved", [])

        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%m/%d %H:%M")
        except:
            formatted_time = "?"

        st.caption(f"• {page}: {len(keys)} items ({formatted_time})")

    # Clear all button
    if st.button("🗑️ Clear All Saved Results", key="clear_all_global"):
        if storage.clear_all_results(dataset_hash):
            st.success("All results cleared!")
            st.rerun()


def render_new_dataset_warning() -> bool:
    """
    Render a warning when a new dataset is detected with existing saved results.

    Returns:
        True if user wants to clear old results, False otherwise
    """
    storage = get_results_storage()

    if not storage.is_connected():
        return False

    current_hash = compute_current_dataset_hash()
    all_saved = storage.get_all_saved_pages()

    # Check if there are results for different datasets
    other_datasets = [
        item for item in all_saved
        if item.get("dataset_hash") != current_hash
    ]

    if other_datasets:
        unique_hashes = set(item.get("dataset_hash") for item in other_datasets)
        st.warning(
            f"⚠️ Found saved results from {len(unique_hashes)} different dataset(s). "
            "These won't be loaded for your current dataset."
        )

        if st.button("🗑️ Clear Old Dataset Results"):
            for old_hash in unique_hashes:
                storage.clear_all_results(old_hash)
            st.success("Old results cleared!")
            st.rerun()
            return True

    return False


def auto_load_if_available(
    page_type: str,
    force_key: str = "auto_loaded",
) -> bool:
    """
    Automatically load saved results if available and not already loaded.

    Args:
        page_type: The page type to check
        force_key: Session state key to track if auto-load was done

    Returns:
        True if results were loaded, False otherwise
    """
    # Check if already auto-loaded this session
    load_key = f"{force_key}_{page_type}"
    if st.session_state.get(load_key):
        return False

    storage = get_results_storage()
    if not storage.is_connected():
        return False

    dataset_hash = compute_current_dataset_hash()

    if storage.has_saved_results(page_type, dataset_hash):
        loaded = storage.load_page_results(page_type, dataset_hash)
        if loaded:
            storage.apply_loaded_results(loaded)
            st.session_state[load_key] = True
            return True

    return False
