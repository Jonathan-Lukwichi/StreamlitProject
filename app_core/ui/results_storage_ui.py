# =============================================================================
# app_core/ui/results_storage_ui.py
# Pipeline Progress Indicator for HealthForecast AI.
# Shows workflow completion status (simplified - no cloud storage).
# =============================================================================

import streamlit as st
from typing import List, Optional, Dict, Any


# =====================================================
# PAGE CONFIGURATION (icons only, for progress display)
# =====================================================

PAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Upload Data": {"icon": "📤", "session_key": "uploaded_data"},
    "Prepare Data": {"icon": "🔧", "session_key": "merged_data"},
    "Explore Data": {"icon": "🔍", "session_key": "eda_results"},
    "Baseline Models": {"icon": "📊", "session_key": "arima_results"},
    "Feature Studio": {"icon": "🛠️", "session_key": "feature_engineering"},
    "Feature Selection": {"icon": "🎯", "session_key": "feature_selection"},
    "Train Models": {"icon": "🤖", "session_key": "ml_mh_results"},
    "Model Results": {"icon": "📈", "session_key": "comparison_results"},
    "Patient Forecast": {"icon": "🏥", "session_key": "forecast_results"},
    "Staff Planner": {"icon": "👥", "session_key": "staff_optimization_results"},
    "Supply Planner": {"icon": "📦", "session_key": "inventory_optimization_results"},
}


def render_pipeline_status_dashboard():
    """
    Render a visual dashboard showing pipeline workflow progress.
    Shows which stages have been completed in the current session.
    """
    st.subheader("📊 Pipeline Progress")

    # Pipeline stages in order
    pipeline_order = [
        "Upload Data", "Prepare Data", "Explore Data",
        "Baseline Models", "Feature Studio", "Feature Selection",
        "Train Models", "Patient Forecast", "Staff Planner", "Supply Planner"
    ]

    # Check completion status from session state
    completed_count = 0

    # Display pipeline progress in two rows
    st.markdown("**Workflow Stages:**")

    cols = st.columns(5)
    for i, page in enumerate(pipeline_order):
        config = PAGE_CONFIG.get(page, {})
        icon = config.get("icon", "📄")
        session_key = config.get("session_key", "")

        # Check if this stage has data in session state
        is_complete = (
            session_key in st.session_state and
            st.session_state[session_key] is not None
        )

        if is_complete:
            completed_count += 1

        with cols[i % 5]:
            if is_complete:
                st.success(f"{icon} **{page}**\n\n✅ Complete")
            else:
                st.info(f"{icon} **{page}**\n\n⬜ Pending")

    # Progress summary
    st.markdown("---")
    progress_pct = (completed_count / len(pipeline_order)) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Completed Stages", f"{completed_count}/{len(pipeline_order)}")
    with col2:
        st.metric("Progress", f"{progress_pct:.0f}%")

    # Progress bar
    st.progress(completed_count / len(pipeline_order))


# =====================================================
# DEPRECATED FUNCTIONS (kept for backwards compatibility)
# These are no-ops to avoid breaking existing page imports
# =====================================================

def render_results_storage_panel(
    page_key: str = "",
    custom_keys: Optional[List[str]] = None,
    show_in_sidebar: bool = True
):
    """
    DEPRECATED: Cloud storage has been removed.
    This function is a no-op for backwards compatibility.
    """
    pass  # No-op


def auto_load_if_available(page_key: str = "", force_reload: bool = False):
    """
    DEPRECATED: Cloud storage has been removed.
    This function is a no-op for backwards compatibility.
    """
    pass  # No-op


def get_page_storage_keys(page_key: str) -> List[str]:
    """
    DEPRECATED: Returns empty list for backwards compatibility.
    """
    return []
