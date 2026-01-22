# SARIMAX Configuration Optimization in `pages/05_Baseline_Models.py`

## Problem Description

Users have reported that running the SARIMAX model with "Manual" configuration in `pages/05_Baseline_Models.py` takes more than 5 minutes, which is unexpectedly long given the UI's indication of "instant training" for this mode.

## Root Cause Analysis

The bottleneck lies within the `page_benchmarks` function in `pages/05_Baseline_Models.py`, specifically in how the `pipeline_mode` is determined for the `run_sarimax_multihorizon` function when the SARIMAX model is being trained.

The relevant section is found under the `tab_train` when `current_model == "SARIMAX"`:

```python
# Retrieve configuration from session state
# ... (other parameter retrievals) ...
search_mode = st.session_state.get("sarimax_search_mode", "fast")

# For manual mode, skip parameter search by passing "aic_only" with fixed orders
pipeline_mode = "aic_only" if search_mode == "manual" else search_mode # <--- This is the problematic line

# ... (rest of the training logic) ...

sarimax_out = run_sarimax_multihorizon(**kwargs)
```

**Explanation:**

1.  When a user selects **"Manual - Specify all parameters"** in the configuration tab, `st.session_state["sarimax_search_mode"]` correctly gets set to `"manual"`.
2.  However, the problematic line `pipeline_mode = "aic_only" if search_mode == "manual" else search_mode` then *overrides* this setting. If `search_mode` is `"manual"`, it explicitly changes `pipeline_mode` to `"aic_only"`.
3.  As a result, the `run_sarimax_multihorizon` function receives `"aic_only"` as its `search_mode` argument, even though the user intended to provide manual parameters.
4.  The `aic_only` mode typically triggers an AIC-based stepwise parameter search (as indicated by the "Balanced (~2min)" option), which is computationally intensive and time-consuming. This completely bypasses the user's manual parameter input and leads to the observed long training times.

The comment `# For manual mode, skip parameter search by passing "aic_only" with fixed orders` is misleading. Passing `"aic_only"` to a function designed for automatic parameter search will indeed *perform* a search, not skip it. The `run_sarimax_multihorizon` function (presumably in `app_core/models/sarimax_pipeline.py`) likely interprets `aic_only` as a command to find parameters via AIC, regardless of whether `order` and `seasonal_order` are provided.

## Proposed Solution

To resolve this, the `pipeline_mode` should directly reflect the user's `search_mode` choice, especially for the "manual" option, allowing the `run_sarimax_multihorizon` function to use the provided `order` and `seasonal_order` without initiating an unnecessary parameter search.

**Proposed Change:**

Modify the following line in `pages/05_Baseline_Models.py`:

```python
pipeline_mode = "aic_only" if search_mode == "manual" else search_mode
```

to:

```python
pipeline_mode = search_mode
```

**Impact of the Change:**

*   When "Manual" mode is selected, `pipeline_mode` will correctly be set to `"manual"`.
*   Assuming `run_sarimax_multihorizon` in `app_core/models/sarimax_pipeline.py` has logic to handle a `"manual"` `search_mode` (or implicitly uses the `order` and `seasonal_order` arguments when `search_mode` is not an auto-search variant), this change will ensure that the manual parameters are used directly.
*   This will drastically reduce the training time for the manual configuration to nearly instant, aligning with user expectations and the UI's promise.
*   The efficiency of the model will not be negatively impacted, as it will simply use the parameters the user explicitly provided.

This change directly addresses the performance bottleneck without sacrificing the model's ability to achieve optimal results when configured manually.