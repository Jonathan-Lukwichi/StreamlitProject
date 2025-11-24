"""
DATA HUB INTEGRATION EXAMPLE
Shows how to add API fetch option to the Data Hub page

This is a code snippet showing the modifications needed to pages/02_Data_Hub.py
"""

# ============================================================================
# ADD THIS IMPORT AT THE TOP OF pages/02_Data_Hub.py
# ============================================================================
from app_core.ui.api_fetch_ui import render_api_fetch_option, render_bulk_api_fetch

# ============================================================================
# REPLACE PATIENT DATA CARD (Column 1) WITH THIS:
# ============================================================================
with col1:
    st.markdown("""
    <div class='hf-feature-card' style='height: 100%; padding: 1.25rem;'>
        <div style='text-align: center; margin-bottom: 1rem;'>
            <div class='hf-feature-icon' style='margin: 0 auto 0.75rem auto; font-size: 2rem;'>📊</div>
            <h2 class='hf-feature-title' style='margin: 0; font-size: 1.25rem;'>Patient Data</h2>
            <p class='hf-feature-description' style='margin: 0.5rem 0 0 0; font-size: 0.8125rem;'>Historical arrival records</p>
        </div>
    """, unsafe_allow_html=True)

    # === NEW: TABBED INTERFACE ===
    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 API Fetch"])

    with tab1:
        _upload_generic("Upload CSV", "patient", "datetime")

    with tab2:
        render_api_fetch_option(dataset_type="patient", key_prefix="patient")
    # === END NEW ===

    df_patient = st.session_state.get("patient_data")

    if isinstance(df_patient, pd.DataFrame) and not df_patient.empty:
        st.markdown("<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06);'></div>", unsafe_allow_html=True)
        st.metric("Rows", f"{df_patient.shape[0]:,}")
        st.metric("Columns", df_patient.shape[1])

        dcol = _date_col(df_patient)
        if dcol:
            dt = pd.to_datetime(df_patient[dcol], errors="coerce")
            if dt.notna().any():
                st.caption(f"📅 {dt.min().date()} → {dt.max().date()}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# REPEAT SAME PATTERN FOR WEATHER, CALENDAR, AND REASON CARDS
# ============================================================================

# Weather Card (Column 2):
with col2:
    st.markdown("""[card header HTML]""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 API Fetch"])

    with tab1:
        _upload_generic("Upload CSV", "weather", "datetime")

    with tab2:
        render_api_fetch_option(dataset_type="weather", key_prefix="weather")

    # [rest of metrics code...]

# Calendar Card (Column 3):
with col3:
    st.markdown("""[card header HTML]""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 API Fetch"])

    with tab1:
        _upload_generic("Upload CSV", "calendar", "date")

    with tab2:
        render_api_fetch_option(dataset_type="calendar", key_prefix="calendar")

    # [rest of metrics code...]

# Reason Card (Column 4):
with col4:
    st.markdown("""[card header HTML]""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔌 API Fetch"])

    with tab1:
        _upload_generic("Upload CSV", "reason", "datetime")

    with tab2:
        render_api_fetch_option(dataset_type="reason", key_prefix="reason")

    # [rest of metrics code...]


# ============================================================================
# ADD BULK FETCH SECTION AFTER THE 4-COLUMN GRID
# ============================================================================
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

st.markdown("""
<div class='hf-feature-card'>
    <div style='text-align: center; margin-bottom: 2rem;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>⚡</div>
        <h2 style='font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--primary-color);'>Bulk API Fetch</h2>
        <p style='color: var(--subtle-text); font-size: 1rem;'>Fetch all 4 datasets from APIs in one operation</p>
    </div>
""", unsafe_allow_html=True)

render_bulk_api_fetch()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# THAT'S IT! The rest of the page remains unchanged.
# ============================================================================
