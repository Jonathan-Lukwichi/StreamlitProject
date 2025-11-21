# Streamlit Forecast App — Modular Scaffold

This scaffold preserves your current design and behavior. It only organizes code by responsibility.

## How to migrate safely (no design changes)
1. Keep your working repo untouched. Open this scaffold side-by-side.
2. Copy functions verbatim from your current files into these modules:
   - CSS/colors -> `app_core/ui/theme.py` (`apply_css`, palette)
   - `header`, `add_grid` -> `app_core/ui/components.py`
   - Date parsing & upload helpers -> `app_core/data/ingest.py`
   - `_prep_*` + `fuse_data` -> `app_core/data/fusion.py`
   - `eda_pipeline.py` -> `app_core/analytics/eda_pipeline.py`
   - `arima_pipeline.py` -> `app_core/models/arima_pipeline.py`
   - `sarimax_pipeline.py` -> `app_core/models/sarimax_pipeline.py`
3. Cut the big page bodies from your monolith and paste into:
   - `pages/01_Dashboard.py` -> Dashboard body
   - `pages/02_Data_Hub.py` -> Data Hub body
   - `pages/03_Data_Fusion.py` -> Fusion body
   - `pages/04_EDA.py` -> EDA body
   - `pages/05_Benchmarks.py` -> Benchmarks body
4. Run: `streamlit run Welcome.py`
5. Move incrementally: one page at a time, test, then continue.

## Notes
- You are not required to use `.streamlit/config.toml` — your CSS already styles the UI.
- If you prefer a single-page app with a radio sidebar, use `app_core/routing.py`.
- Keep function names identical to avoid breakage.
