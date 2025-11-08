# =============================================================================
# pages/07_Automated_Feature_Selection.py — Feature Selection Studio (Part A)
# =============================================================================
from __future__ import annotations
import time, json, warnings
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
warnings.filterwarnings("ignore")

# Plot helpers (used in Part B)
import plotly.express as px
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

# ----------------------- Version-safe helpers ---------------------------------
def _rmse(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error
        return root_mean_squared_error(y_true, y_pred)
    except ImportError:
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _safe_serialize(obj):
    import numpy as _np
    # Use string name checks to avoid generic type issues
    obj_type = type(obj).__name__
    if obj_type == 'DataFrame':
        return {"__type__":"DataFrame","data": obj.to_dict(orient="records")}
    if obj_type == 'Series':
        return {"__type__":"Series","data": obj.to_dict()}
    # Check for numpy integer types
    if hasattr(obj, 'dtype') and _np.issubdtype(type(obj), _np.integer):
        return int(obj)
    # Check for numpy float types
    if hasattr(obj, 'dtype') and _np.issubdtype(type(obj), _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (set, tuple)): return list(obj)
    if isinstance(obj, dict): return {k:_safe_serialize(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_safe_serialize(v) for v in obj]
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

# --------------------------- Metrics block ------------------------------------
def _metrics_block(y_tr, yhat_tr, y_te, yhat_te) -> Dict[str, float]:
    return {
        "rmse_train": _rmse(y_tr, yhat_tr),
        "mae_train":  mean_absolute_error(y_tr, yhat_tr),
        "mape_train": _mape(y_tr, yhat_tr),
        "rmse_test":  _rmse(y_te, yhat_te),
        "mae_test":   mean_absolute_error(y_te, yhat_te),
        "mape_test":  _mape(y_te, yhat_te),
    }

# --------------------------- Pipeline class -----------------------------------
class FeatureSelectionPipeline:
    """
    All methods return:
      - selected features
      - per-feature importances (when applicable)
      - uniform metrics dict: RMSE/MAE/MAPE (train/test)
      - elapsed time
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[int, Dict[str, Any]] = {}
        self.selected_features: Dict[int, List[str]] = {}

    # ---- data prep ----
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series):
        test_ratio = 1 - self.config.get("train_test_split", 0.8)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=self.config.get("random_state", 42)
        )
        scaler = StandardScaler()
        num_cols = X.select_dtypes(include=[np.number]).columns
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])
        return X_train, X_test, y_train, y_test

    # ---- Option 0: Baseline (no selection) ----
    def _baseline(self, X_train, X_test, y_train, y_test):
        start = time.time()
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        yhat_tr = model.predict(X_train)
        yhat_te = model.predict(X_test)
        metrics = _metrics_block(y_train, yhat_tr, y_test, yhat_te)
        elapsed = time.time() - start
        return list(X_train.columns), None, metrics, elapsed

    # ---- Option 1: Permutation Importance (true permutation on X_test) ----
    def _permutation_importance(self, X_train, X_test, y_train, y_test):
        """
        1) Train model on full training set
        2) Compute baseline metrics on X_test
        3) For each feature:
           - shuffle that single column on X_test
           - predict with the same trained model
           - importance = delta in metric (here we store RMSE/MAE/MAPE deltas)
        4) Select top features by RMSE delta (or threshold)
        """
        start = time.time()
        base_model = RandomForestRegressor(
            n_estimators=self.config.get("rf_n_estimators", 200),
            random_state=self.config.get("random_state", 42)
        )
        base_model.fit(X_train, y_train)
        base_pred = base_model.predict(X_test)
        base_rmse = _rmse(y_test, base_pred)
        base_mae  = mean_absolute_error(y_test, base_pred)
        base_mape = _mape(y_test, base_pred)

        deltas = []
        for col in X_test.columns:
            Xp = X_test.copy()
            Xp[col] = np.random.RandomState(42).permutation(Xp[col].values)
            yhat = base_model.predict(Xp)
            deltas.append({
                "feature": col,
                "delta_rmse": _rmse(y_test, yhat) - base_rmse,
                "delta_mae":  mean_absolute_error(y_test, yhat) - base_mae,
                "delta_mape": _mape(y_test, yhat) - base_mape
            })
        imp = pd.DataFrame(deltas).sort_values("delta_rmse", ascending=False)

        # selection rule
        thr = self.config.get("rf_importance_threshold", None)  # on delta_rmse
        top_n = self.config.get("rf_top_n_features", None)
        if top_n:
            selected = imp.head(int(top_n))["feature"].tolist()
        elif thr is not None:
            selected = imp[imp["delta_rmse"] > float(thr)]["feature"].tolist()
            if not selected:
                selected = imp.head(max(1, int(0.8 * len(imp))))["feature"].tolist()
        else:
            selected = imp.head(max(1, int(0.8 * len(imp))))["feature"].tolist()

        # report baseline metrics too
        metrics = {
            "rmse_train": np.nan, "mae_train": np.nan, "mape_train": np.nan,
            "rmse_test":  base_rmse, "mae_test": base_mae, "mape_test": base_mape
        }
        elapsed = time.time() - start
        return selected, imp, metrics, elapsed

    # ---- Option 2: Lasso (pure linear regression with L1) ----
    def _lasso_joint(self, X_train, X_test, y_train, y_test):
        """
        Pick alpha by jointly minimizing RMSE+MAE+MAPE (TEST) over candidates from LassoCV.
        Keep ALL non-zero coefficients (positive and negative).
        """
        start = time.time()
        lcv = LassoCV(cv=self.config.get("lasso_cv_folds", 5), random_state=42, n_alphas=100)
        lcv.fit(X_train, y_train)
        alphas = np.unique(np.clip(lcv.alphas_, 1e-6, None))

        grid = []
        for a in alphas:
            m = Lasso(alpha=float(a), random_state=42, max_iter=5000)
            m.fit(X_train, y_train)
            yhat_tr = m.predict(X_train)
            yhat_te = m.predict(X_test)
            rmse_t = _rmse(y_test, yhat_te)
            mae_t  = mean_absolute_error(y_test, yhat_te)
            mape_t = _mape(y_test, yhat_te)
            grid.append({"alpha": float(a), "rmse": rmse_t, "mae": mae_t, "mape": mape_t,
                         "yhat_tr": yhat_tr, "yhat_te": yhat_te, "model": m})
        grid = pd.DataFrame(grid)

        # normalize by median to balance scales
        for k in ["rmse","mae","mape"]:
            med = float(np.median(grid[k].to_numpy())) if grid[k].notna().any() else 1.0
            if med <= 0: med = 1.0
            grid[f"{k}_n"] = grid[k] / med
        grid["score"] = grid["rmse_n"] + grid["mae_n"] + grid["mape_n"]
        best = grid.loc[grid["score"].idxmin()]
        model = best["model"]
        yhat_tr = best["yhat_tr"]; yhat_te = best["yhat_te"]
        metrics = _metrics_block(y_train, yhat_tr, y_test, yhat_te)

        coefs = pd.Series(model.coef_, index=X_train.columns)
        selected = list(coefs[coefs != 0].index)
        elapsed = time.time() - start
        # importance proxy = |coef|
        imp = coefs.abs().rename("importance").reset_index().rename(columns={"index":"feature"})
        imp = imp.sort_values("importance", ascending=False)
        return selected, imp, metrics, elapsed

    # ---- Option 3: Gradient Boosting (with GridSearchCV) ----
    def _gb_gridsearch(self, X_train, X_test, y_train, y_test):
        start = time.time()
        grid = self.config.get("gb_param_grid", {
            "learning_rate":[0.01,0.1,0.2],
            "max_depth":[3,5,7],
            "n_estimators":[50,100,200]
        })
        gbr = GradientBoostingRegressor(random_state=42)
        gscv = GridSearchCV(gbr, grid, cv=self.config.get("cv_folds",5), scoring="neg_root_mean_squared_error")
        gscv.fit(X_train, y_train)
        best = gscv.best_estimator_
        yhat_tr = best.predict(X_train)
        yhat_te = best.predict(X_test)
        metrics = _metrics_block(y_train, yhat_tr, y_test, yhat_te)
        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": getattr(best, "feature_importances_", np.zeros(X_train.shape[1]))
        }).sort_values("importance", ascending=False)
        thr = self.config.get("gb_importance_threshold", 0.0)
        selected = imp[imp["importance"] > thr]["feature"].tolist()
        if not selected:
            selected = imp.head(max(1, int(0.8*len(imp))))["feature"].tolist()
        elapsed = time.time() - start
        return selected, imp, metrics, elapsed

    # ---- Public API ----
    def fit(self, X: pd.DataFrame, y: pd.Series, option: str | int = "all"):
        X_train, X_test, y_train, y_test = self._prepare_data(X, y)
        opts = [0,1,2,3] if option == "all" else [option]

        for opt in opts:
            if opt == 0:
                feats, imp, metrics, t = self._baseline(X_train,X_test,y_train,y_test)
                self.results[0] = {"features":feats,"importances":imp,"metrics":metrics,"time":t,"name":"No Selection"}
                self.selected_features[0] = feats
            elif opt == 1:
                feats, imp, metrics, t = self._permutation_importance(X_train,X_test,y_train,y_test)
                self.results[1] = {"features":feats,"importances":imp,"metrics":metrics,"time":t,"name":"Permutation Importance"}
                self.selected_features[1] = feats
            elif opt == 2:
                feats, imp, metrics, t = self._lasso_joint(X_train,X_test,y_train,y_test)
                self.results[2] = {"features":feats,"importances":imp,"metrics":metrics,"time":t,"name":"Lasso (Linear L1)"}
                self.selected_features[2] = feats
            elif opt == 3:
                feats, imp, metrics, t = self._gb_gridsearch(X_train,X_test,y_train,y_test)
                self.results[3] = {"features":feats,"importances":imp,"metrics":metrics,"time":t,"name":"Gradient Boosting"}
                self.selected_features[3] = feats
        return self

    def summary_table(self) -> pd.DataFrame:
        rows=[]
        for k,v in self.results.items():
            name = v.get("name", {0:"No Selection",1:"Permutation Importance",2:"Lasso (Linear L1)",3:"Gradient Boosting"}.get(k,str(k)))
            feats=len(v.get("features",[])); t=v.get("time",0.0); m=v.get("metrics",{})
            rows.append({
                "Method":name, "Features":feats,
                "RMSE Train": m.get("rmse_train", np.nan), "RMSE Test": m.get("rmse_test", np.nan),
                "MAE Train":  m.get("mae_train",  np.nan), "MAE Test":  m.get("mae_test",  np.nan),
                "MAPE Train": m.get("mape_train", np.nan), "MAPE Test": m.get("mape_test", np.nan),
                "Time (s)": round(float(t),2)
            })
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype.kind in "fc": df[c] = df[c].astype(float).round(4)
        return df


# =============================================================================
# pages/07_Automated_Feature_Selection.py — Feature Selection Studio (Part B)
# =============================================================================

# ---------- Visual helpers ----------
def _chip_list(items: List[str], color: str):
    if not items: return "<span style='color:#64748b'>—</span>"
    return " ".join([f"<span style='color:{color}; padding:2px 6px; border:1px solid {color}; border-radius:8px; margin:2px; display:inline-block'>{x}</span>" for x in items])

def _bar_importances(imp_df: Optional[pd.DataFrame], selected: List[str], label_col="importance", title=""):
    if imp_df is None or (isinstance(imp_df, pd.DataFrame) and imp_df.empty):
        st.info("No importances to display.")
        return
    df = imp_df.copy()
    if "feature" not in df.columns:
        df.rename(columns={df.columns[0]: "feature"}, inplace=True)
    if label_col not in df.columns:
        # try fallbacks (delta_rmse or importance)
        if "delta_rmse" in df.columns: label_col = "delta_rmse"
        elif "importance" in df.columns: label_col = "importance"
        else:
            for c in df.columns:
                if c != "feature": label_col = c; break
    df = df.sort_values(label_col, ascending=False)
    top = df.head(20).copy()
    top["selected"] = top["feature"].isin(selected)
    top["status"] = np.where(top["selected"], "Selected (blue)", "Rejected (red)")
    fig = px.bar(
        top, x=label_col, y="feature", color="status",
        color_discrete_map={"Selected (blue)":"#1d4ed8","Rejected (red)":"#dc2626"},
        orientation="h", title=title, height=460
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Legend: **blue** = selected | **red** = rejected.")

def _corr_heatmap(df: pd.DataFrame, title: str):
    if df is None or df.empty or df.shape[1] < 2:
        st.info("Not enough columns for a correlation heatmap.")
        return
    try:
        if sns is None:
            fig, ax = plt.subplots(figsize=(7, 5))
            c = df.corr(numeric_only=True)
            im = ax.imshow(c, cmap="viridis")
            ax.set_xticks(range(len(c.columns))); ax.set_xticklabels(c.columns, rotation=90)
            ax.set_yticks(range(len(c.index)));   ax.set_yticklabels(c.index)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.heatmap(df.corr(numeric_only=True), cmap="viridis", annot=False, ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Heatmap unavailable: {e}")

def _download(label: str, data: bytes, name: str, mime: str="text/csv"):
    st.download_button(label, data=data, file_name=name, mime=mime, use_container_width=False)

# ---------- Session helpers ----------
def _variants_from_session():
    fe = st.session_state.get("feature_engineering", {})
    variants = {}
    if isinstance(fe, dict):
        if "A" in fe and isinstance(fe["A"], pd.DataFrame):
            variants["Variant A (Decomposition)"] = fe["A"]
        if "B" in fe and isinstance(fe["B"], pd.DataFrame):
            variants["Variant B (Cyclical)"] = fe["B"]
    proc = st.session_state.get("processed_df")
    if isinstance(proc, pd.DataFrame) and not proc.empty:
        variants["Original (Processed)"] = proc
    return variants, fe.get("train_idx"), fe.get("test_idx")

def _pick_target(df: pd.DataFrame) -> List[str]:
    if "Target_1" in df.columns: return ["Target_1"]
    t = [c for c in df.columns if c.lower().startswith("target_")]
    return sorted(t, key=lambda x: (x != "Target_1", x)) or [df.columns[-1]]

# ---------- Page ----------
def page_feature_selection():
    # Compact hero (no redundancy)
    st.markdown(
        """
        <div style="
          border-radius: 18px; padding: 18px;
          background: linear-gradient(135deg, rgba(37,99,235,.10), rgba(16,185,129,.10));
          border: 1px solid rgba(37,99,235,.18); margin-bottom: 14px;">
          <div style="font-weight:800; font-size:1.25rem;">🧩 Feature Selection Studio</div>
          <div class="muted">Run a selector and receive a <b>ready-to-train</b> dataset (selected features + Target_1..Target_7).</div>
        </div>
        """, unsafe_allow_html=True
    )

    variants, train_idx, test_idx = _variants_from_session()
    if not variants:
        st.warning("No engineered datasets found. Please run **Advanced Feature Engineering** first.")
        return

    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1:
        variant_name = st.selectbox("Dataset variant", list(variants.keys()), index=0)
    dfv = variants[variant_name].copy()
    with c2:
        target_col = st.selectbox("Target (uses only Target_1 for selection)", _pick_target(dfv), index=0)
    with c3:
        run_which = st.selectbox("Method(s)", ["All (0–3)", "0 Baseline", "1 Permutation", "2 Lasso (L1)", "3 Gradient Boosting"], index=0)

    # Config (compact)
    with st.expander("Configuration", expanded=False):
        left, right = st.columns(2)
        with left:
            tr_ratio = st.slider("Train ratio (ignored if FE indices provided)", 0.5, 0.95, 0.8, 0.05)
            cv_folds = st.number_input("CV folds", 3, 10, value=5, step=1)
            rf_n = st.number_input("RF n_estimators (baseline model for permutation)", 50, 1000, value=200, step=50)
            rf_thr = st.text_input("Permutation threshold on ΔRMSE (blank = top 80%)", "")
        with right:
            gb_lr = [0.01, 0.1, 0.2]
            gb_depth = [3, 5, 7]
            gb_n = [50, 100, 200]
            st.caption(f"GB grid: learning_rate {gb_lr}, max_depth {gb_depth}, n_estimators {gb_n}")

    # Build config dict
    try:
        rf_thr_val = None if rf_thr.strip()=="" else float(rf_thr)
    except Exception:
        rf_thr_val = None

    config = {
        "train_test_split": tr_ratio,
        "random_state": 42,
        "cv_folds": int(cv_folds),
        "rf_n_estimators": int(rf_n),
        "rf_importance_threshold": rf_thr_val,   # ΔRMSE threshold
        "rf_top_n_features": None,
        "lasso_cv_folds": int(cv_folds),
        "gb_param_grid": {
            "learning_rate": gb_lr,
            "max_depth": gb_depth,
            "n_estimators": gb_n,
        },
        "gb_importance_threshold": 0.0,  # keep all >0; fallback to top 80%
    }

    # Strict: never use Target_2..7 in selection; keep them for outputs
    forbidden = [f"Target_{i}" for i in range(2,8)]
    date_like = [c for c in dfv.columns if any(k in c.lower() for k in ["date","time","stamp","datetime","ds"])]

    X = dfv.drop(columns=[target_col] + date_like + forbidden, errors="ignore")
    y = dfv[target_col]
    targets_1_7 = [f"Target_{i}" for i in range(1,8) if f"Target_{i}" in dfv.columns]

    pipe = FeatureSelectionPipeline(config=config)

    # Use FE-provided indices if available
    if isinstance(train_idx, np.ndarray) and isinstance(test_idx, np.ndarray):
        def _prep_with_idx(X_in, y_in):
            from sklearn.preprocessing import StandardScaler
            X_tr, X_te = X_in.iloc[train_idx].copy(), X_in.iloc[test_idx].copy()
            y_tr, y_te = y_in.iloc[train_idx].copy(), y_in.iloc[test_idx].copy()
            num_cols = X_in.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
            X_te[num_cols] = scaler.transform(X_te[num_cols])
            return X_tr, X_te, y_tr, y_te
        pipe._prepare_data = _prep_with_idx  # type: ignore

    opt_map = {
        "All (0–3)": "all",
        "0 Baseline": 0,
        "1 Permutation": 1,
        "2 Lasso (L1)": 2,
        "3 Gradient Boosting": 3
    }
    opt_choice = opt_map[run_which]

    # Preview
    with st.expander("Preview variant (first 12 rows)", expanded=False):
        st.dataframe(dfv.head(12), use_container_width=True)
        st.caption(f"Variant: {variant_name} | Columns: {len(dfv.columns)}")

    run = st.button("🚀 Run Selection", type="primary")
    if not run:
        st.caption(f"Training target: **{target_col}**. Excluded from selection: {', '.join(forbidden)}")
        return

    with st.spinner("Selecting features…"):
        pipe.fit(X, y, option=opt_choice)

    # Build ready-to-train datasets: selected + Target_1..7 (from variant)
    filtered_datasets: Dict[int, pd.DataFrame] = {}
    for opt, feats in pipe.selected_features.items():
        feats = list(dict.fromkeys(feats))
        keep_cols = [c for c in feats + targets_1_7 if c in dfv.columns]
        filtered = dfv[keep_cols].copy() if keep_cols else dfv.copy()
        filtered_datasets[opt] = filtered

    summary = pipe.summary_table()
    st.session_state["feature_selection"] = {
        "results": pipe.results,
        "selected_features": pipe.selected_features,
        "summary": summary,
        "filtered_datasets": filtered_datasets,
        "variant": variant_name,
        "target": target_col
    }

    st.success("Done. Explore outputs below.")

    # Tabs per option
    labels = []
    if 0 in pipe.results: labels.append("0 — Baseline")
    if 1 in pipe.results: labels.append("1 — Permutation")
    if 2 in pipe.results: labels.append("2 — Lasso (L1)")
    if 3 in pipe.results: labels.append("3 — Gradient Boosting")
    tabs = st.tabs(labels) if labels else []

    def _lists(opt_idx: int):
        feats_all = list(X.columns)
        feats_sel = pipe.selected_features.get(opt_idx, [])
        feats_rej = [c for c in feats_all if c not in feats_sel]
        st.markdown(_chip_list(feats_sel, "#1d4ed8"), unsafe_allow_html=True)
        st.markdown(_chip_list(feats_rej, "#dc2626"), unsafe_allow_html=True)

    # 0 — Baseline
    if 0 in pipe.results:
        with tabs[labels.index("0 — Baseline")]:
            res = pipe.results[0]
            st.caption("No feature selection (reference).")
            _lists(0)
            st.json(_safe_serialize(res.get("metrics", {})), expanded=False)
            df_out = filtered_datasets.get(0)
            if df_out is not None:
                st.dataframe(df_out.head(20), use_container_width=True)
                _download("⬇️ Dataset (csv)", df_out.to_csv(index=False).encode("utf-8"), "baseline_filtered.csv")

    # 1 — Permutation
    if 1 in pipe.results:
        with tabs[labels.index("1 — Permutation")]:
            res = pipe.results[1]
            feats = res.get("features", [])
            imp = res.get("importances")
            st.caption("Importance = Δ metric after shuffling a single feature on X_test (higher = more important).")
            _lists(1)
            # show ΔRMSE as primary bar
            _bar_importances(imp.rename(columns={"delta_rmse":"importance"}) if isinstance(imp, pd.DataFrame) else imp,
                             feats, label_col="importance", title="Permutation Importance (ΔRMSE)")
            # optional correlation heatmap
            if feats:
                cols = [c for c in (feats + [target_col]) if c in dfv.columns]
                _corr_heatmap(dfv[cols], "Correlation — Selected + Target_1")
            df_out = filtered_datasets.get(1)
            if df_out is not None:
                st.dataframe(df_out.head(20), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    _download("⬇️ Selected (txt)", "\n".join(feats).encode("utf-8"), "permutation_selected.txt", "text/plain")
                with c2:
                    if isinstance(imp, pd.DataFrame):
                        _download("⬇️ Importances (csv)", imp.to_csv(index=False).encode("utf-8"), "permutation_importances.csv")
                with c3:
                    _download("⬇️ Dataset (csv)", df_out.to_csv(index=False).encode("utf-8"), "permutation_filtered.csv")
            st.json(_safe_serialize(res.get("metrics", {})), expanded=False)

    # 2 — Lasso (L1)
    if 2 in pipe.results:
        with tabs[labels.index("2 — Lasso (L1)")]:
            res = pipe.results[2]
            feats = res.get("features", [])
            imp = res.get("importances")
            st.caption("Linear regression with L1 penalty; α chosen by joint minimization of RMSE+MAE+MAPE.")
            _lists(2)
            _bar_importances(imp, feats, label_col="importance", title="Lasso |coef|")
            df_out = filtered_datasets.get(2)
            if df_out is not None:
                st.dataframe(df_out.head(20), use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    _download("⬇️ Selected (txt)", "\n".join(feats).encode("utf-8"), "lasso_selected.txt", "text/plain")
                with c2:
                    _download("⬇️ Dataset (csv)", df_out.to_csv(index=False).encode("utf-8"), "lasso_filtered.csv")
            st.json(_safe_serialize(res.get("metrics", {})), expanded=False)

    # 3 — Gradient Boosting
    if 3 in pipe.results:
        with tabs[labels.index("3 — Gradient Boosting")]:
            res = pipe.results[3]
            feats = res.get("features", [])
            imp = res.get("importances")
            st.caption("Gradient Boosting with GridSearchCV; importance from best estimator.")
            _lists(3)
            _bar_importances(imp, feats, label_col="importance", title="GB Feature Importance")
            if feats:
                cols = [c for c in (feats + [target_col]) if c in dfv.columns]
                _corr_heatmap(dfv[cols], "Correlation — Selected + Target_1")
            df_out = filtered_datasets.get(3)
            if df_out is not None:
                st.dataframe(df_out.head(20), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    _download("⬇️ Selected (txt)", "\n".join(feats).encode("utf-8"), "gb_selected.txt", "text/plain")
                with c2:
                    if isinstance(imp, pd.DataFrame):
                        _download("⬇️ Importances (csv)", imp.to_csv(index=False).encode("utf-8"), "gb_importances.csv")
                with c3:
                    _download("⬇️ Dataset (csv)", df_out.to_csv(index=False).encode("utf-8"), "gb_filtered.csv")
            st.json(_safe_serialize(res.get("metrics", {})), expanded=False)

    st.markdown("---")
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True, height=260)

    c1, c2 = st.columns(2)
    with c1:
        _download("⬇️ Summary (csv)", summary.to_csv(index=False).encode("utf-8"), "feature_selection_summary.csv")
    with c2:
        _download("⬇️ Report (json)",
                  json.dumps(_safe_serialize({
                      "results": st.session_state["feature_selection"]["results"],
                      "selected_features": st.session_state["feature_selection"]["selected_features"],
                      "summary": st.session_state["feature_selection"]["summary"].to_dict(orient="records"),
                      "variant_used": st.session_state["feature_selection"]["variant"],
                      "target": st.session_state["feature_selection"]["target"]
                  }), indent=2).encode("utf-8"),
                  "feature_selection_report.json", "application/json")

# ---------- Entrypoint ----------
def main():
    try:
        from app_core.ui.theme import apply_css
        apply_css()

    except Exception:
        pass
    page_feature_selection()

if __name__ == "__main__":
    main()
