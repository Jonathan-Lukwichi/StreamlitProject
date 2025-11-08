# =============================================================================
# 08_Modeling_Hub.py
# Unified Modeling Hub (foundation only)
# - Benchmarks (ARIMA / SARIMAX): Model selection dropdown + Auto/Manual + Train/Test
# - ML (LSTM / ANN / XGBoost): Model selection dropdown ("Choose a ml model") with XGBoost default + Auto/Manual + Train/Test
# - Hybrids (LSTM-XGBoost / LSTM-ANN / LSTM-SARIMAX): Model selection dropdown + Auto/Manual + Train/Test
# Robust defaults (no KeyErrors even with old sessions)
# =============================================================================
from __future__ import annotations
import streamlit as st

# Optional theme hook (safe if missing)
try:
    from app_core.ui.theme import apply_css
    apply_css()
except Exception:
    pass

st.set_page_config(page_title="Modeling Hub", layout="wide")

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧠 Modeling Hub")
subpage = st.sidebar.radio(
    "Open a model page",
    options=[
        "📏 Benchmarks",
        "🧮 Machine Learning",
        "🧬 Hybrid Models",
    ],
    index=0,
)

# -----------------------------------------------------------------------------
# Seed defaults robustly (works even if an old session dict exists)
# -----------------------------------------------------------------------------
DEFAULTS = {
    # Global-ish
    "target_col": None,
    "feature_cols": [],
    "split_ratio": 0.8,
    "cv_folds": 3,

    # ---------- Benchmarks (ARIMA / SARIMAX) ----------
    "benchmark_model": "ARIMA",              # default
    "benchmark_param_mode": "Automatic",     # 'Automatic' or 'Manual'
    "optimization_method": "Auto (AIC-based)",
    "seasonal_enabled": False,
    "exogenous_cols": [],
    "seasonal_period": 7,
    # Auto bounds
    "max_p": 5, "max_d": 2, "max_q": 5,
    "max_P": 2, "max_D": 1, "max_Q": 2,
    # Manual params
    "arima_order": (1, 0, 0),
    "sarimax_order": (1, 0, 0),
    "sarimax_seasonal_order": (0, 0, 0, 0),

    # ---------- ML (LSTM / ANN / XGBoost) ----------
    "ml_choice": "XGBoost",                  # << default as requested
    # LSTM
    "ml_lstm_param_mode": "Automatic",
    "lstm_layers": 1, "lstm_hidden_units": 64, "lstm_epochs": 20, "lstm_lr": 1e-3,
    "lstm_search_method": "Grid (bounded)",
    "lstm_max_layers": 3, "lstm_max_units": 256, "lstm_max_epochs": 50,
    "lstm_lr_min": 1e-5, "lstm_lr_max": 1e-2,
    # ANN
    "ml_ann_param_mode": "Automatic",
    "ann_hidden_layers": 2, "ann_neurons": 64, "ann_epochs": 30, "ann_activation": "relu",
    "ann_search_method": "Grid (bounded)",
    "ann_max_layers": 4, "ann_max_neurons": 256, "ann_max_epochs": 60,
    # XGB
    "ml_xgb_param_mode": "Automatic",
    "xgb_n_estimators": 300, "xgb_max_depth": 6, "xgb_min_child_weight": 1, "xgb_eta": 0.1,
    "xgb_search_method": "Grid (bounded)",
    "xgb_n_estimators_min": 100, "xgb_n_estimators_max": 1000,
    "xgb_depth_min": 3, "xgb_depth_max": 12,
    "xgb_eta_min": 0.01, "xgb_eta_max": 0.3,

    # ---------- Hybrids (LSTM-XGB / LSTM-ANN / LSTM-SARIMAX) ----------
    "hybrid_choice": "LSTM-XGBoost",         # sensible default
    # LSTM-XGB
    "hyb_lstm_xgb_param_mode": "Automatic",
    "hyb_lstm_units": 64, "hyb_xgb_estimators": 300, "hyb_xgb_depth": 6, "hyb_xgb_eta": 0.1,
    "hyb_lstm_search_method": "Grid (bounded)",
    "hyb_lstm_max_units": 256,
    "hyb_xgb_estimators_min": 100, "hyb_xgb_estimators_max": 1000,
    "hyb_xgb_depth_min": 3, "hyb_xgb_depth_max": 12,
    "hyb_xgb_eta_min": 0.01, "hyb_xgb_eta_max": 0.3,
    # LSTM-ANN
    "hyb_lstm_ann_param_mode": "Automatic",
    "hyb2_lstm_units": 64, "hyb2_ann_neurons": 64, "hyb2_ann_layers": 2, "hyb2_ann_activation": "relu",
    "hyb2_lstm_search_method": "Grid (bounded)",
    "hyb2_lstm_max_units": 256, "hyb2_ann_max_layers": 4, "hyb2_ann_max_neurons": 256,
    # LSTM-SARIMAX
    "hyb_lstm_sarimax_param_mode": "Automatic",
    "hyb3_lstm_units": 64, "hyb3_sarimax_order": (1, 0, 0),
    "hyb3_sarimax_seasonal_order": (0, 0, 0, 0), "hyb3_seasonal_period": 7,
    "hyb3_exogenous_cols": [],
    "hyb3_lstm_search_method": "Grid (bounded)",
    "hyb3_lstm_max_units": 256,
    "hyb3_max_p": 5, "hyb3_max_d": 2, "hyb3_max_q": 5,
    "hyb3_max_P": 2, "hyb3_max_D": 1, "hyb3_max_Q": 2,
}

cfg = st.session_state.setdefault("modeling_config", {})
for k, v in DEFAULTS.items():
    cfg.setdefault(k, v)

with st.sidebar.expander("⚙️ Config"):
    if st.button("Reset modeling config"):
        st.session_state["modeling_config"] = DEFAULTS.copy()
        st.rerun()

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def data_source_placeholder():
    with st.expander("📂 Data Source (placeholder)", expanded=False):
        st.info(
            "In the complete version, you'll select your working DataFrame here "
            "(e.g., from st.session_state['fs_df'] / 'fe_df'), or upload a CSV."
        )
        st.write("Detected session keys:")
        st.code([k for k in st.session_state.keys()])

def target_feature_placeholders():
    st.subheader("🎯 Target & Features (placeholders)")
    st.selectbox("Target column (placeholder)", ["(choose later)"], key="tf_target")
    st.multiselect("Feature columns (placeholder)", [], key="tf_feats")

def debug_panel():
    st.divider()
    with st.expander("🔎 Debug — current modeling config", expanded=False):
        st.json(st.session_state["modeling_config"])

# -----------------------------------------------------------------------------
# BENCHMARKS (ARIMA / SARIMAX)
# -----------------------------------------------------------------------------
def page_benchmarks():
    st.title("📏 Benchmarks")

    # Model selection header + dropdown
    st.subheader("Model selection")
    cfg["benchmark_model"] = st.selectbox(
        "Choose a benchmark model",
        options=["ARIMA", "SARIMAX"],
        index=["ARIMA","SARIMAX"].index(cfg.get("benchmark_model","ARIMA")),
        help="Choose ARIMA or SARIMAX as your baseline model."
    )

    data_source_placeholder()

    # Parameter mode
    st.subheader("Parameter mode")
    mode = st.radio(
        "How do you want to set parameters?",
        options=["Automatic (optimize)", "Manual"],
        index=0 if cfg.get("benchmark_param_mode","Automatic") == "Automatic" else 1,
        horizontal=True,
        key="bench_param_mode_radio"
    )
    cfg["benchmark_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"

    # --- Automatic ---
    if cfg["benchmark_param_mode"] == "Automatic":
        st.caption("No training yet — placeholders for the optimizer’s settings.")
        col1, col2 = st.columns(2)
        with col1:
            cfg["optimization_method"] = st.selectbox(
                "Optimization method",
                ["Auto (AIC-based)", "Grid search (bounded)", "Optuna / Bayesian (planned)"],
                index=["Auto (AIC-based)", "Grid search (bounded)", "Optuna / Bayesian (planned)"].index(
                    cfg.get("optimization_method","Auto (AIC-based)")
                )
            )
        with col2:
            cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

        st.markdown("**Bounds for search (placeholders)**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: cfg["max_p"] = int(st.number_input("max p", 0, 10, int(cfg.get("max_p",5))))
        with c2: cfg["max_d"] = int(st.number_input("max d", 0, 3, int(cfg.get("max_d",2))))
        with c3: cfg["max_q"] = int(st.number_input("max q", 0, 10, int(cfg.get("max_q",5))))
        with c4: cfg["seasonal_enabled"] = bool(st.checkbox("Include seasonal terms", value=bool(cfg.get("seasonal_enabled",False))))

        if cfg["seasonal_enabled"]:
            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["max_P"] = int(st.number_input("max P", 0, 5, int(cfg.get("max_P",2))))
            with c2: cfg["max_D"] = int(st.number_input("max D", 0, 2, int(cfg.get("max_D",1))))
            with c3: cfg["max_Q"] = int(st.number_input("max Q", 0, 5, int(cfg.get("max_Q",2))))
            with c4: cfg["seasonal_period"] = int(st.number_input("Seasonal period (s)", 1, 365, int(cfg.get("seasonal_period",7))))

        if cfg["benchmark_model"] == "SARIMAX":
            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="sarimax_exog_placeholder")
            cfg["exogenous_cols"] = st.session_state.get("sarimax_exog_placeholder", [])

    # --- Manual ---
    else:
        st.caption("Enter parameters manually (placeholders).")
        if cfg["benchmark_model"] == "ARIMA":
            c1, c2, c3 = st.columns(3)
            with c1: p = st.number_input("p", 0, 10, int(cfg.get("arima_order",(1,0,0))[0]))
            with c2: d = st.number_input("d", 0, 3,  int(cfg.get("arima_order",(1,0,0))[1]))
            with c3: q = st.number_input("q", 0, 10, int(cfg.get("arima_order",(1,0,0))[2]))
            cfg["arima_order"] = (int(p), int(d), int(q))

            seasonal = st.checkbox("Add seasonal terms", value=bool(cfg.get("seasonal_enabled", False)))
            cfg["seasonal_enabled"] = seasonal
            if seasonal:
                s1, s2, s3, s4 = st.columns(4)
                with s1: P = st.number_input("P", 0, 5, 0)
                with s2: D = st.number_input("D", 0, 2, 0)
                with s3: Q = st.number_input("Q", 0, 5, 0)
                with s4:
                    s = st.number_input("Seasonal period (s)", 1, 365, int(cfg.get("seasonal_period", 7)))
                    cfg["seasonal_period"] = int(s)
                cfg["sarimax_seasonal_order"] = (int(P), int(D), int(Q), int(cfg["seasonal_period"]))

        else:  # SARIMAX
            r1, r2, r3 = st.columns(3)
            with r1: p = st.number_input("p", 0, 10, int(cfg.get("sarimax_order",(1,0,0))[0]))
            with r2: d = st.number_input("d", 0, 3,  int(cfg.get("sarimax_order",(1,0,0))[1]))
            with r3: q = st.number_input("q", 0, 10, int(cfg.get("sarimax_order",(1,0,0))[2]))
            cfg["sarimax_order"] = (int(p), int(d), int(q))

            s1, s2, s3, s4 = st.columns(4)
            so = cfg.get("sarimax_seasonal_order",(0,0,0,0))
            with s1: P = st.number_input("P", 0, 5, int(so[0]))
            with s2: D = st.number_input("D", 0, 2, int(so[1]))
            with s3: Q = st.number_input("Q", 0, 5, int(so[2]))
            with s4: s = st.number_input("s (period)", 1, 365, int(cfg.get("seasonal_period",7)))
            cfg["sarimax_seasonal_order"] = (int(P), int(D), int(Q), int(s))
            cfg["seasonal_period"] = int(s)

            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="sarimax_exog_manual")
            cfg["exogenous_cols"] = st.session_state.get("sarimax_exog_manual", [])

    st.divider()
    st.subheader("Run (placeholders)")
    c1, c2 = st.columns(2)
    with c1: st.button(f"🚀 Train {cfg['benchmark_model']}")
    with c2: st.button(f"🧪 Test {cfg['benchmark_model']}")
    debug_panel()

# -----------------------------------------------------------------------------
# MACHINE LEARNING (LSTM / ANN / XGBoost)
# -----------------------------------------------------------------------------
def page_ml():
    st.title("🧮 Machine Learning")

    # Model selection header + dropdown (XGBoost pre-selected)
    st.subheader("Model selection")
    ml_options = ["Long Short-Term Memory (LSTM)", "Artificial Neural Network (ANN)", "Extreme Gradient Boosting (XGBoost)"]
    default_idx = 2  # XGBoost default
    current = cfg.get("ml_choice", "XGBoost")
    # map current string to display name for selectbox
    current_display = (
        "Extreme Gradient Boosting (XGBoost)" if current == "XGBoost" else
        "Long Short-Term Memory (LSTM)" if current == "LSTM" else
        "Artificial Neural Network (ANN)"
    )
    choice_display = st.selectbox(
        "Choose a ml model",
        options=ml_options,
        index=ml_options.index(current_display) if current_display in ml_options else default_idx,
        help="Pick your ML model. XGBoost is pre-selected."
    )
    # Normalize choice to short token
    cfg["ml_choice"] = "XGBoost" if "XGBoost" in choice_display else ("LSTM" if "LSTM" in choice_display else "ANN")

    data_source_placeholder()

    # Parameter mode per selected ML model
    st.subheader("Parameter mode")
    mode = st.radio(
        "How do you want to set parameters?",
        options=["Automatic (optimize)", "Manual"], horizontal=True,
        index=0 if (
            (cfg["ml_choice"] == "LSTM" and cfg.get("ml_lstm_param_mode","Automatic") == "Automatic") or
            (cfg["ml_choice"] == "ANN"  and cfg.get("ml_ann_param_mode","Automatic")  == "Automatic") or
            (cfg["ml_choice"] == "XGBoost" and cfg.get("ml_xgb_param_mode","Automatic") == "Automatic")
        ) else 1,
        key="ml_mode_radio"
    )

    # ---------------- LSTM ----------------
    if cfg["ml_choice"] == "LSTM":
        cfg["ml_lstm_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        if cfg["ml_lstm_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["lstm_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("lstm_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["lstm_max_layers"] = int(st.number_input("max layers", 1, 6,  int(cfg.get("lstm_max_layers",3))))
            with c2: cfg["lstm_max_units"]  = int(st.number_input("max units", 8, 1024, int(cfg.get("lstm_max_units",256)), step=8))
            with c3: cfg["lstm_max_epochs"] = int(st.number_input("max epochs", 5, 500, int(cfg.get("lstm_max_epochs",50))))
            with c4:
                lr_min = st.number_input("min LR", 1e-6, 1e-1, float(cfg.get("lstm_lr_min",1e-5)), format="%.6f")
                lr_max = st.number_input("max LR", 1e-6, 1e-1, float(cfg.get("lstm_lr_max",1e-2)), format="%.6f")
                cfg["lstm_lr_min"], cfg["lstm_lr_max"] = float(lr_min), float(lr_max)
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["lstm_layers"] = int(st.number_input("LSTM layers", 1, 5, int(cfg.get("lstm_layers",1))))
            with c2: cfg["lstm_hidden_units"] = int(st.number_input("Hidden units", 8, 512, int(cfg.get("lstm_hidden_units",64)), step=8))
            with c3: cfg["lstm_epochs"] = int(st.number_input("Epochs", 1, 500, int(cfg.get("lstm_epochs",20))))
            cfg["lstm_lr"] = float(st.number_input("Learning rate", 1e-6, 1e-1, float(cfg.get("lstm_lr",1e-3)), format="%.6f"))

    # ---------------- ANN ----------------
    elif cfg["ml_choice"] == "ANN":
        cfg["ml_ann_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        if cfg["ml_ann_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["ann_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("ann_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["ann_max_layers"]  = int(st.number_input("max layers", 1, 8,  int(cfg.get("ann_max_layers",4))))
            with c2: cfg["ann_max_neurons"] = int(st.number_input("max neurons", 8, 1024, int(cfg.get("ann_max_neurons",256)), step=8))
            with c3: cfg["ann_max_epochs"]  = int(st.number_input("max epochs", 5, 500, int(cfg.get("ann_max_epochs",60))))
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["ann_hidden_layers"] = int(st.number_input("Hidden layers", 1, 8, int(cfg.get("ann_hidden_layers",2))))
            with c2: cfg["ann_neurons"] = int(st.number_input("Neurons per layer", 4, 1024, int(cfg.get("ann_neurons",64)), step=4))
            with c3: cfg["ann_epochs"] = int(st.number_input("Epochs", 1, 500, int(cfg.get("ann_epochs",30))))
            cfg["ann_activation"] = st.selectbox("Activation", ["relu", "tanh", "sigmoid"],
                index=["relu","tanh","sigmoid"].index(cfg.get("ann_activation","relu")))

    # ---------------- XGBoost ----------------
    else:  # XGBoost
        cfg["ml_xgb_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        if cfg["ml_xgb_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["xgb_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("xgb_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3 = st.columns(3)
            with c1:
                cfg["xgb_n_estimators_min"] = int(st.number_input("min n_estimators", 50, 5000, int(cfg.get("xgb_n_estimators_min",100)), step=50))
                cfg["xgb_n_estimators_max"] = int(st.number_input("max n_estimators", 50, 5000, int(cfg.get("xgb_n_estimators_max",1000)), step=50))
            with c2:
                cfg["xgb_depth_min"] = int(st.number_input("min max_depth", 1, 30, int(cfg.get("xgb_depth_min",3))))
                cfg["xgb_depth_max"] = int(st.number_input("max max_depth", 1, 30, int(cfg.get("xgb_depth_max",12))))
            with c3:
                cfg["xgb_eta_min"] = float(st.number_input("min eta", 0.001, 1.0, float(cfg.get("xgb_eta_min",0.01))))
                cfg["xgb_eta_max"] = float(st.number_input("max eta", 0.001, 1.0, float(cfg.get("xgb_eta_max",0.3))))
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["xgb_n_estimators"] = int(st.number_input("n_estimators", 50, 5000, int(cfg.get("xgb_n_estimators",300)), step=50))
            with c2: cfg["xgb_max_depth"] = int(st.number_input("max_depth", 1, 30, int(cfg.get("xgb_max_depth",6))))
            with c3: cfg["xgb_min_child_weight"] = int(st.number_input("min_child_weight", 1, 50, int(cfg.get("xgb_min_child_weight",1))))
            cfg["xgb_eta"] = float(st.number_input("Learning rate (eta)", 0.001, 1.0, float(cfg.get("xgb_eta",0.1))))

    target_feature_placeholders()
    c1, c2 = st.columns(2)
    with c1: st.button(f"🚀 Train {cfg['ml_choice']} (placeholder)")
    with c2: st.button(f"🧪 Test {cfg['ml_choice']} (placeholder)")
    debug_panel()

# -----------------------------------------------------------------------------
# HYBRID MODELS (LSTM-XGB / LSTM-ANN / LSTM-SARIMAX)
# -----------------------------------------------------------------------------
def page_hybrid():
    st.title("🧬 Hybrid Models")

    # Model selection header + dropdown
    st.subheader("Model selection")
    hyb_options = ["LSTM-XGBoost", "LSTM-ANN", "LSTM-SARIMAX"]
    cfg["hybrid_choice"] = st.selectbox(
        "Choose a hybrid model",
        options=hyb_options,
        index=hyb_options.index(cfg.get("hybrid_choice","LSTM-XGBoost")),
        help="Pick a hybrid architecture to configure."
    )

    data_source_placeholder()

    # Parameter mode
    st.subheader("Parameter mode")
    mode = st.radio(
        "How do you want to set parameters?",
        options=["Automatic (optimize)", "Manual"], horizontal=True,
        index=0 if (
            (cfg["hybrid_choice"] == "LSTM-XGBoost"  and cfg.get("hyb_lstm_xgb_param_mode","Automatic")     == "Automatic") or
            (cfg["hybrid_choice"] == "LSTM-ANN"      and cfg.get("hyb_lstm_ann_param_mode","Automatic")     == "Automatic") or
            (cfg["hybrid_choice"] == "LSTM-SARIMAX"  and cfg.get("hyb_lstm_sarimax_param_mode","Automatic") == "Automatic")
        ) else 1,
        key="hyb_mode_radio"
    )

    # ---------------- LSTM-XGBoost ----------------
    if cfg["hybrid_choice"] == "LSTM-XGBoost":
        cfg["hyb_lstm_xgb_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• Phase 1: LSTM → predictions/residuals")
            st.write("• Phase 2: XGBoost on residuals or stacked features")

        if cfg["hyb_lstm_xgb_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb_lstm_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb_lstm_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb_lstm_max_units",256)), step=8))
            with c2:
                cfg["hyb_xgb_estimators_min"] = int(st.number_input("XGB min n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators_min",100)), step=50))
                cfg["hyb_xgb_estimators_max"] = int(st.number_input("XGB max n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators_max",1000)), step=50))
            with c3:
                cfg["hyb_xgb_depth_min"] = int(st.number_input("XGB min max_depth", 1, 30, int(cfg.get("hyb_xgb_depth_min",3))))
                cfg["hyb_xgb_depth_max"] = int(st.number_input("XGB max max_depth", 1, 30, int(cfg.get("hyb_xgb_depth_max",12))))
            st.slider("XGB eta range (placeholder)",
                      float(cfg.get("hyb_xgb_eta_min",0.01)),
                      float(cfg.get("hyb_xgb_eta_max",0.3)),
                      float(cfg.get("hyb_xgb_eta",0.1)),
                      key="hyb_xgb_eta_range")
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb_lstm_units",64)), step=8))
            with c2: cfg["hyb_xgb_estimators"] = int(st.number_input("XGB n_estimators", 50, 5000, int(cfg.get("hyb_xgb_estimators",300)), step=50))
            with c3: cfg["hyb_xgb_depth"] = int(st.number_input("XGB max_depth", 1, 30, int(cfg.get("hyb_xgb_depth",6))))
            cfg["hyb_xgb_eta"] = float(st.number_input("XGB learning rate (eta)", 0.001, 1.0, float(cfg.get("hyb_xgb_eta",0.1))))

    # ---------------- LSTM-ANN ----------------
    elif cfg["hybrid_choice"] == "LSTM-ANN":
        cfg["hyb_lstm_ann_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• LSTM sequence embedding → ANN dense head")

        if cfg["hyb_lstm_ann_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb2_lstm_search_method"] = st.selectbox(
                    "Search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb2_lstm_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2 = st.columns(2)
            with c1: cfg["hyb2_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb2_lstm_max_units",256)), step=8))
            with c2:
                cfg["hyb2_ann_max_layers"]  = int(st.number_input("ANN max layers", 1, 8, int(cfg.get("hyb2_ann_max_layers",4))))
                cfg["hyb2_ann_max_neurons"] = int(st.number_input("ANN max neurons", 8, 1024, int(cfg.get("hyb2_ann_max_neurons",256)), step=8))
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2, c3 = st.columns(3)
            with c1: cfg["hyb2_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb2_lstm_units",64)), step=8))
            with c2: cfg["hyb2_ann_layers"] = int(st.number_input("ANN hidden layers", 1, 8, int(cfg.get("hyb2_ann_layers",2))))
            with c3: cfg["hyb2_ann_neurons"] = int(st.number_input("ANN neurons per layer", 4, 1024, int(cfg.get("hyb2_ann_neurons",64)), step=4))
            cfg["hyb2_ann_activation"] = st.selectbox(
                "ANN activation", ["relu", "tanh", "sigmoid"],
                index=["relu","tanh","sigmoid"].index(cfg.get("hyb2_ann_activation","relu"))
            )

    # ---------------- LSTM-SARIMAX ----------------
    else:
        cfg["hyb_lstm_sarimax_param_mode"] = "Automatic" if "Automatic" in mode else "Manual"
        with st.expander("🔗 Strategy (placeholder)", expanded=True):
            st.write("• SARIMAX: seasonality & exogenous; LSTM: nonlinear patterns")
            st.write("• Combine via residuals or stacked features")

        if cfg["hyb_lstm_sarimax_param_mode"] == "Automatic":
            st.caption("No training yet — placeholders for search settings.")
            c1, c2 = st.columns(2)
            with c1:
                cfg["hyb3_lstm_search_method"] = st.selectbox(
                    "LSTM search method", ["Grid (bounded)", "Random", "Bayesian (planned)"],
                    index=["Grid (bounded)", "Random", "Bayesian (planned)"].index(cfg.get("hyb3_lstm_search_method","Grid (bounded)"))
                )
            with c2:
                cfg["split_ratio"] = st.slider("Train/Validation split ratio", 0.50, 0.95, float(cfg.get("split_ratio",0.80)), 0.01)

            st.markdown("**Bounds (placeholders)**")
            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["hyb3_lstm_max_units"] = int(st.number_input("LSTM max units", 8, 1024, int(cfg.get("hyb3_lstm_max_units",256)), step=8))
            with c2: cfg["hyb3_max_p"] = int(st.number_input("SARIMAX max p", 0, 10, int(cfg.get("hyb3_max_p",5))))
            with c3: cfg["hyb3_max_d"] = int(st.number_input("SARIMAX max d", 0, 3, int(cfg.get("hyb3_max_d",2))))
            with c4: cfg["hyb3_max_q"] = int(st.number_input("SARIMAX max q", 0, 10, int(cfg.get("hyb3_max_q",5))))

            c1, c2, c3, c4 = st.columns(4)
            with c1: cfg["hyb3_max_P"] = int(st.number_input("SARIMAX max P", 0, 5, int(cfg.get("hyb3_max_P",2))))
            with c2: cfg["hyb3_max_D"] = int(st.number_input("SARIMAX max D", 0, 2, int(cfg.get("hyb3_max_D",1))))
            with c3: cfg["hyb3_max_Q"] = int(st.number_input("SARIMAX max Q", 0, 5, int(cfg.get("hyb3_max_Q",2))))
            with c4: cfg["hyb3_seasonal_period"] = int(st.number_input("Seasonal period (s)", 1, 365, int(cfg.get("hyb3_seasonal_period",7))))

            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="hyb3_exog_auto")
            cfg["hyb3_exogenous_cols"] = st.session_state.get("hyb3_exog_auto", [])
        else:
            st.caption("Manual hyperparameters (placeholders).")
            c1, c2 = st.columns(2)
            with c1: cfg["hyb3_lstm_units"] = int(st.number_input("LSTM hidden units", 8, 1024, int(cfg.get("hyb3_lstm_units",64)), step=8))
            with c2:
                st.markdown("**SARIMAX orders**")
                r1, r2, r3 = st.columns(3)
                with r1:
                    p = st.number_input("p", 0, 10, int(cfg.get("hyb3_sarimax_order",(1,0,0))[0]))
                with r2:
                    d = st.number_input("d", 0, 3, int(cfg.get("hyb3_sarimax_order",(1,0,0))[1]))
                with r3:
                    q = st.number_input("q", 0, 10, int(cfg.get("hyb3_sarimax_order",(1,0,0))[2]))
                cfg["hyb3_sarimax_order"] = (int(p), int(d), int(q))

            s1, s2, s3, s4 = st.columns(4)
            so = cfg.get("hyb3_sarimax_seasonal_order", (0,0,0,0))
            with s1: P = st.number_input("P", 0, 5, int(so[0]))
            with s2: D = st.number_input("D", 0, 2, int(so[1]))
            with s3: Q = st.number_input("Q", 0, 5, int(so[2]))
            with s4:
                s = st.number_input("s (period)", 1, 365, int(cfg.get("hyb3_seasonal_period",7)))
            cfg["hyb3_sarimax_seasonal_order"] = (int(P), int(D), int(Q), int(s))
            cfg["hyb3_seasonal_period"] = int(s)

            st.markdown("**Exogenous (X) features (placeholder)**")
            st.multiselect("Select exogenous columns", options=[], key="hyb3_exog_manual")
            cfg["hyb3_exogenous_cols"] = st.session_state.get("hyb3_exog_manual", [])

    target_feature_placeholders()
    c1, c2 = st.columns(2)
    with c1: st.button(f"🚀 Train {cfg['hybrid_choice']} (placeholder)")
    with c2: st.button(f"🧪 Test {cfg['hybrid_choice']} (placeholder)")
    debug_panel()

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if subpage == "📏 Benchmarks":
    page_benchmarks()
elif subpage == "🧮 Machine Learning":
    page_ml()
elif subpage == "🧬 Hybrid Models":
    page_hybrid()
