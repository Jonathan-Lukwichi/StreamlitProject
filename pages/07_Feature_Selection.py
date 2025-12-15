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

from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand
from app_core.ui.results_storage_ui import render_results_storage_panel, auto_load_if_available

# ============================================================================
# AUTHENTICATION CHECK - ADMIN ONLY
# ============================================================================
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button
require_admin_access()
configure_sidebar_navigation()

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

    Multi-target support: when y_multi is provided (dict of target -> Series),
    importance scores are aggregated (averaged) across all targets for more
    robust feature selection.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[int, Dict[str, Any]] = {}
        self.selected_features: Dict[int, List[str]] = {}
        self.y_multi: Optional[Dict[str, pd.Series]] = None  # Multi-target data

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

    def _prepare_data_multi(self, X: pd.DataFrame, y_multi: Dict[str, pd.Series]):
        """Prepare data for all targets. Returns dict of prepared data per target."""
        prepared = {}
        test_ratio = 1 - self.config.get("train_test_split", 0.8)
        random_state = self.config.get("random_state", 42)

        # Get consistent train/test indices using first target
        first_target = list(y_multi.keys())[0]
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )

        # Scale X once
        X_train_base = X.iloc[train_idx].copy()
        X_test_base = X.iloc[test_idx].copy()
        scaler = StandardScaler()
        num_cols = X.select_dtypes(include=[np.number]).columns
        X_train_base[num_cols] = scaler.fit_transform(X_train_base[num_cols])
        X_test_base[num_cols] = scaler.transform(X_test_base[num_cols])

        for target_name, y in y_multi.items():
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            prepared[target_name] = (X_train_base.copy(), X_test_base.copy(), y_train, y_test)

        return prepared, X_train_base, X_test_base

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across all targets."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        aggregated = {}
        for key in keys:
            values = [m.get(key, np.nan) for m in metrics_list]
            aggregated[key] = float(np.nanmean(values))
        return aggregated

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
            n_estimators=self.config.get("rf_n_estimators", 100),  # Reduced from 200 for speed
            random_state=self.config.get("random_state", 42),
            n_jobs=-1  # Parallel processing
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
        lcv = LassoCV(cv=self.config.get("lasso_cv_folds", 5), random_state=42, n_alphas=30, n_jobs=-1)  # Reduced alphas + parallel
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
        # Reduced grid for faster execution (9 combinations instead of 27)
        grid = self.config.get("gb_param_grid", {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [50, 100]
        })
        gbr = GradientBoostingRegressor(random_state=42)
        gscv = GridSearchCV(gbr, grid, cv=self.config.get("cv_folds", 3), scoring="neg_root_mean_squared_error", n_jobs=-1)  # Reduced CV + parallel
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

    # ---- Multi-target aggregation methods ----
    def _permutation_importance_multi(self, X: pd.DataFrame, y_multi: Dict[str, pd.Series]):
        """
        Run permutation importance across ALL targets and aggregate importance scores.
        Returns averaged importance (delta_rmse) across all targets.
        """
        start = time.time()
        prepared, X_train, X_test = self._prepare_data_multi(X, y_multi)

        all_deltas = []  # List of DataFrames, one per target
        all_metrics = []

        for target_name, (X_tr, X_te, y_tr, y_te) in prepared.items():
            base_model = RandomForestRegressor(
                n_estimators=self.config.get("rf_n_estimators", 100),  # Reduced for speed
                random_state=self.config.get("random_state", 42),
                n_jobs=-1  # Parallel processing
            )
            base_model.fit(X_tr, y_tr)
            base_pred = base_model.predict(X_te)
            base_rmse = _rmse(y_te, base_pred)
            base_mae = mean_absolute_error(y_te, base_pred)
            base_mape = _mape(y_te, base_pred)

            deltas = []
            for col in X_te.columns:
                Xp = X_te.copy()
                Xp[col] = np.random.RandomState(42).permutation(Xp[col].values)
                yhat = base_model.predict(Xp)
                deltas.append({
                    "feature": col,
                    "delta_rmse": _rmse(y_te, yhat) - base_rmse,
                    "delta_mae": mean_absolute_error(y_te, yhat) - base_mae,
                    "delta_mape": _mape(y_te, yhat) - base_mape
                })

            imp_df = pd.DataFrame(deltas)
            all_deltas.append(imp_df.set_index("feature"))
            all_metrics.append({
                "rmse_train": np.nan, "mae_train": np.nan, "mape_train": np.nan,
                "rmse_test": base_rmse, "mae_test": base_mae, "mape_test": base_mape
            })

        # Aggregate importances across all targets (average delta values)
        combined = pd.concat(all_deltas, axis=1)
        agg_imp = combined.groupby(level=0, axis=1).mean().reset_index()
        agg_imp.columns = ["feature", "delta_rmse", "delta_mae", "delta_mape"]
        agg_imp = agg_imp.sort_values("delta_rmse", ascending=False)

        # Selection rule
        thr = self.config.get("rf_importance_threshold", None)
        top_n = self.config.get("rf_top_n_features", None)
        if top_n:
            selected = agg_imp.head(int(top_n))["feature"].tolist()
        elif thr is not None:
            selected = agg_imp[agg_imp["delta_rmse"] > float(thr)]["feature"].tolist()
            if not selected:
                selected = agg_imp.head(max(1, int(0.8 * len(agg_imp))))["feature"].tolist()
        else:
            selected = agg_imp.head(max(1, int(0.8 * len(agg_imp))))["feature"].tolist()

        # Aggregate metrics across targets
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        elapsed = time.time() - start
        return selected, agg_imp, aggregated_metrics, elapsed

    def _lasso_joint_multi(self, X: pd.DataFrame, y_multi: Dict[str, pd.Series]):
        """
        Run Lasso across ALL targets and aggregate coefficients.
        Returns averaged |coefficient| across all targets.
        """
        start = time.time()
        prepared, X_train, X_test = self._prepare_data_multi(X, y_multi)

        all_coefs = []  # List of coefficient Series, one per target
        all_metrics = []

        for target_name, (X_tr, X_te, y_tr, y_te) in prepared.items():
            lcv = LassoCV(cv=self.config.get("lasso_cv_folds", 5), random_state=42, n_alphas=30, n_jobs=-1)  # Reduced + parallel
            lcv.fit(X_tr, y_tr)
            alphas = np.unique(np.clip(lcv.alphas_, 1e-6, None))

            grid = []
            for a in alphas:
                m = Lasso(alpha=float(a), random_state=42, max_iter=5000)
                m.fit(X_tr, y_tr)
                yhat_tr = m.predict(X_tr)
                yhat_te = m.predict(X_te)
                rmse_t = _rmse(y_te, yhat_te)
                mae_t = mean_absolute_error(y_te, yhat_te)
                mape_t = _mape(y_te, yhat_te)
                grid.append({"alpha": float(a), "rmse": rmse_t, "mae": mae_t, "mape": mape_t,
                             "yhat_tr": yhat_tr, "yhat_te": yhat_te, "model": m})
            grid = pd.DataFrame(grid)

            # Normalize and find best alpha
            for k in ["rmse", "mae", "mape"]:
                med = float(np.median(grid[k].to_numpy())) if grid[k].notna().any() else 1.0
                if med <= 0: med = 1.0
                grid[f"{k}_n"] = grid[k] / med
            grid["score"] = grid["rmse_n"] + grid["mae_n"] + grid["mape_n"]
            best = grid.loc[grid["score"].idxmin()]
            model = best["model"]
            yhat_tr = best["yhat_tr"]; yhat_te = best["yhat_te"]

            coefs = pd.Series(np.abs(model.coef_), index=X_tr.columns)
            all_coefs.append(coefs)
            all_metrics.append(_metrics_block(y_tr, yhat_tr, y_te, yhat_te))

        # Aggregate coefficients across targets (average)
        coef_df = pd.concat(all_coefs, axis=1)
        avg_coefs = coef_df.mean(axis=1)
        selected = list(avg_coefs[avg_coefs != 0].index)

        # Build importance DataFrame
        imp = avg_coefs.rename("importance").reset_index().rename(columns={"index": "feature"})
        imp = imp.sort_values("importance", ascending=False)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        elapsed = time.time() - start
        return selected, imp, aggregated_metrics, elapsed

    def _gb_gridsearch_multi(self, X: pd.DataFrame, y_multi: Dict[str, pd.Series]):
        """
        Run Gradient Boosting across ALL targets and aggregate feature importances.
        Returns averaged importance across all targets.
        """
        start = time.time()
        prepared, X_train, X_test = self._prepare_data_multi(X, y_multi)

        all_importances = []  # List of importance arrays, one per target
        all_metrics = []

        # Reduced grid for faster execution (8 combinations instead of 27)
        grid_params = self.config.get("gb_param_grid", {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [50, 100]
        })

        for target_name, (X_tr, X_te, y_tr, y_te) in prepared.items():
            gbr = GradientBoostingRegressor(random_state=42)
            gscv = GridSearchCV(gbr, grid_params, cv=self.config.get("cv_folds", 3),
                                scoring="neg_root_mean_squared_error", n_jobs=-1)  # Reduced CV + parallel
            gscv.fit(X_tr, y_tr)
            best = gscv.best_estimator_
            yhat_tr = best.predict(X_tr)
            yhat_te = best.predict(X_te)

            importances = getattr(best, "feature_importances_", np.zeros(X_tr.shape[1]))
            all_importances.append(pd.Series(importances, index=X_tr.columns))
            all_metrics.append(_metrics_block(y_tr, yhat_tr, y_te, yhat_te))

        # Aggregate importances across targets (average)
        imp_df = pd.concat(all_importances, axis=1)
        avg_imp = imp_df.mean(axis=1)

        imp = pd.DataFrame({
            "feature": avg_imp.index,
            "importance": avg_imp.values
        }).sort_values("importance", ascending=False)

        # Selection
        thr = self.config.get("gb_importance_threshold", 0.0)
        selected = imp[imp["importance"] > thr]["feature"].tolist()
        if not selected:
            selected = imp.head(max(1, int(0.8 * len(imp))))["feature"].tolist()

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        elapsed = time.time() - start
        return selected, imp, aggregated_metrics, elapsed

    def _baseline_multi(self, X: pd.DataFrame, y_multi: Dict[str, pd.Series]):
        """Baseline (no selection) across multiple targets."""
        start = time.time()
        prepared, X_train, X_test = self._prepare_data_multi(X, y_multi)

        all_metrics = []
        for target_name, (X_tr, X_te, y_tr, y_te) in prepared.items():
            model = DummyRegressor(strategy="mean")
            model.fit(X_tr, y_tr)
            yhat_tr = model.predict(X_tr)
            yhat_te = model.predict(X_te)
            all_metrics.append(_metrics_block(y_tr, yhat_tr, y_te, yhat_te))

        aggregated_metrics = self._aggregate_metrics(all_metrics)
        elapsed = time.time() - start
        return list(X_train.columns), None, aggregated_metrics, elapsed

    # ---- Public API ----
    def fit(self, X: pd.DataFrame, y: pd.Series, option: str | int = "all", y_multi: Optional[Dict[str, pd.Series]] = None):
        """
        Fit feature selection methods.

        Args:
            X: Feature DataFrame
            y: Primary target Series (used for single-target fallback)
            option: Which methods to run ("all", 0, 1, 2, or 3)
            y_multi: Optional dict of target_name -> Series for multi-target aggregation.
                     If provided with >1 targets, uses multi-target methods that aggregate
                     importance scores across all targets.
        """
        opts = [0, 1, 2, 3] if option == "all" else [option]

        # Determine if we should use multi-target methods
        use_multi = y_multi is not None and len(y_multi) > 1
        self.y_multi = y_multi if use_multi else None

        if use_multi:
            # Multi-target mode: aggregate across all targets
            n_targets = len(y_multi)
            for opt in opts:
                if opt == 0:
                    feats, imp, metrics, t = self._baseline_multi(X, y_multi)
                    self.results[0] = {
                        "features": feats, "importances": imp, "metrics": metrics,
                        "time": t, "name": f"No Selection (avg {n_targets} targets)"
                    }
                    self.selected_features[0] = feats
                elif opt == 1:
                    feats, imp, metrics, t = self._permutation_importance_multi(X, y_multi)
                    self.results[1] = {
                        "features": feats, "importances": imp, "metrics": metrics,
                        "time": t, "name": f"Permutation Importance (avg {n_targets} targets)"
                    }
                    self.selected_features[1] = feats
                elif opt == 2:
                    feats, imp, metrics, t = self._lasso_joint_multi(X, y_multi)
                    self.results[2] = {
                        "features": feats, "importances": imp, "metrics": metrics,
                        "time": t, "name": f"Lasso (avg {n_targets} targets)"
                    }
                    self.selected_features[2] = feats
                elif opt == 3:
                    feats, imp, metrics, t = self._gb_gridsearch_multi(X, y_multi)
                    self.results[3] = {
                        "features": feats, "importances": imp, "metrics": metrics,
                        "time": t, "name": f"Gradient Boosting (avg {n_targets} targets)"
                    }
                    self.selected_features[3] = feats
        else:
            # Single-target mode: use original methods
            X_train, X_test, y_train, y_test = self._prepare_data(X, y)
            for opt in opts:
                if opt == 0:
                    feats, imp, metrics, t = self._baseline(X_train, X_test, y_train, y_test)
                    self.results[0] = {"features": feats, "importances": imp, "metrics": metrics, "time": t, "name": "No Selection"}
                    self.selected_features[0] = feats
                elif opt == 1:
                    feats, imp, metrics, t = self._permutation_importance(X_train, X_test, y_train, y_test)
                    self.results[1] = {"features": feats, "importances": imp, "metrics": metrics, "time": t, "name": "Permutation Importance"}
                    self.selected_features[1] = feats
                elif opt == 2:
                    feats, imp, metrics, t = self._lasso_joint(X_train, X_test, y_train, y_test)
                    self.results[2] = {"features": feats, "importances": imp, "metrics": metrics, "time": t, "name": "Lasso (Linear L1)"}
                    self.selected_features[2] = feats
                elif opt == 3:
                    feats, imp, metrics, t = self._gb_gridsearch(X_train, X_test, y_train, y_test)
                    self.results[3] = {"features": feats, "importances": imp, "metrics": metrics, "time": t, "name": "Gradient Boosting"}
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

# Page Configuration
st.set_page_config(
    page_title="Feature Selection - HealthForecast AI",
    page_icon="🧩",
    layout="wide",
)

# ---------- Visual helpers ----------
def _chip_list(items: List[str], color: str):
    if not items: return f"<span style='color:{SUBTLE_TEXT}'>—</span>"
    return " ".join([f"<span class='hf-pill' style='background:rgba(34,211,238,0.12); color:{PRIMARY_COLOR}; border:1px solid rgba(34,211,238,0.25);'>{x}</span>" for x in items])

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
    """
    Get dataset variants and split indices from Feature Engineering.

    Returns:
        tuple: (variants, train_idx, cal_idx, test_idx)
            - variants: Dict of dataset names to DataFrames
            - train_idx: Training set indices (from temporal split)
            - cal_idx: Calibration set indices (for CQR in Modeling Hub)
            - test_idx: Test set indices (from temporal split)
    """
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
    return variants, fe.get("train_idx"), fe.get("cal_idx"), fe.get("test_idx")

def _pick_target(df: pd.DataFrame) -> List[str]:
    """Return all available Target columns (Target_1 through Target_7)."""
    targets = [f"Target_{i}" for i in range(1, 8) if f"Target_{i}" in df.columns]
    if not targets:
        # Fallback: look for any target-like columns
        targets = [c for c in df.columns if c.lower().startswith("target_")]
        targets = sorted(targets, key=lambda x: (x != "Target_1", x))
    return targets if targets else [df.columns[-1]]

# ---------- Page ----------
def page_feature_selection():
    # Premium Hero Header
    st.markdown(
        f"""
        <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
          <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🧩</div>
          <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Feature Selection</h1>
          <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
            Intelligently select the most impactful features using advanced selection algorithms and machine learning techniques
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    variants, train_idx, cal_idx, test_idx = _variants_from_session()
    if not variants:
        st.warning("No engineered datasets found. Please run **Advanced Feature Engineering** first.")
        return

    c1, c2, c3 = st.columns([1.25, 1.5, 1])
    with c1:
        variant_name = st.selectbox("Dataset variant", list(variants.keys()), index=0)
    dfv = variants[variant_name].copy()
    available_targets = _pick_target(dfv)
    with c2:
        # Multi-select for targets with "Select All" option
        select_all = st.checkbox("Select All Targets", value=False, key="select_all_targets")
        if select_all:
            selected_targets = available_targets
        else:
            selected_targets = st.multiselect(
                "Target(s) for feature selection",
                options=available_targets,
                default=[available_targets[0]] if available_targets else [],
                help="Select one or more targets. Features will be selected based on importance across all selected targets."
            )
        if not selected_targets:
            selected_targets = [available_targets[0]] if available_targets else []
            st.warning("At least one target required. Using Target_1.")
    with c3:
        run_which = st.selectbox("Method(s)", ["All (0–3)", "0 Baseline", "1 Permutation", "2 Lasso (L1)", "3 Gradient Boosting"], index=0)

    # Config (compact) - Optimized defaults for faster execution
    with st.expander("Configuration", expanded=False):
        left, right = st.columns(2)
        with left:
            tr_ratio = st.slider("Train ratio (ignored if FE indices provided)", 0.5, 0.95, 0.8, 0.05)
            cv_folds = st.number_input("CV folds", 3, 10, value=3, step=1)  # Reduced from 5 for speed
            rf_n = st.number_input("RF n_estimators (baseline model for permutation)", 50, 500, value=100, step=50)  # Reduced from 200
            rf_thr = st.text_input("Permutation threshold on ΔRMSE (blank = top 80%)", "")
        with right:
            gb_lr = [0.05, 0.1]  # Reduced grid
            gb_depth = [3, 5]
            gb_n = [50, 100]
            st.caption(f"GB grid: learning_rate {gb_lr}, max_depth {gb_depth}, n_estimators {gb_n}")
            st.caption("⚡ Optimized for speed with parallel processing")

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

    # Exclude non-selected targets from features (they are outputs, not inputs)
    all_targets = [f"Target_{i}" for i in range(1, 8) if f"Target_{i}" in dfv.columns]
    forbidden = [t for t in all_targets if t not in selected_targets]
    date_like = [c for c in dfv.columns if any(k in c.lower() for k in ["date","time","stamp","datetime","ds"])]

    # For multi-target: use first selected target as primary for single-target methods
    # For averaging across targets, we'll aggregate importance scores
    primary_target = selected_targets[0]
    X = dfv.drop(columns=selected_targets + date_like + forbidden, errors="ignore")
    y = dfv[primary_target]  # Primary target for single-target methods

    # Store all target data for multi-target aggregation
    y_multi = {t: dfv[t] for t in selected_targets if t in dfv.columns}
    targets_1_7 = all_targets  # Keep all targets in output datasets

    pipe = FeatureSelectionPipeline(config=config)

    # Use FE-provided indices if available
    fe_indices_available = isinstance(train_idx, np.ndarray) and isinstance(test_idx, np.ndarray)
    if fe_indices_available:
        # Override _prepare_data for single-target mode
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

        # Override _prepare_data_multi for multi-target mode
        def _prep_multi_with_idx(X_in, y_multi_in):
            from sklearn.preprocessing import StandardScaler
            prepared = {}
            X_tr = X_in.iloc[train_idx].copy()
            X_te = X_in.iloc[test_idx].copy()
            scaler = StandardScaler()
            num_cols = X_in.select_dtypes(include=[np.number]).columns
            X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
            X_te[num_cols] = scaler.transform(X_te[num_cols])
            for target_name, y_series in y_multi_in.items():
                y_tr = y_series.iloc[train_idx]
                y_te = y_series.iloc[test_idx]
                prepared[target_name] = (X_tr.copy(), X_te.copy(), y_tr, y_te)
            return prepared, X_tr, X_te
        pipe._prepare_data_multi = _prep_multi_with_idx  # type: ignore

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
        targets_str = ", ".join(selected_targets)
        excluded_str = ", ".join(forbidden) if forbidden else "None"
        st.caption(f"**Selected targets ({len(selected_targets)}):** {targets_str}")
        st.caption(f"**Excluded from features:** {excluded_str}")
        return

    # Multi-target message
    if len(selected_targets) > 1:
        st.info(f"🎯 **Multi-target mode**: Aggregating importance scores across {len(selected_targets)} targets for robust feature selection.")

    with st.spinner("Selecting features…"):
        pipe.fit(X, y, option=opt_choice, y_multi=y_multi)

    # Build ready-to-train datasets: selected + Target_1..7 + datetime columns (from variant)
    filtered_datasets: Dict[int, pd.DataFrame] = {}
    for opt, feats in pipe.selected_features.items():
        feats = list(dict.fromkeys(feats))
        # Include datetime columns in the output datasets (for plotting/tracking)
        keep_cols = [c for c in date_like + feats + targets_1_7 if c in dfv.columns]
        filtered = dfv[keep_cols].copy() if keep_cols else dfv.copy()
        filtered_datasets[opt] = filtered

    summary = pipe.summary_table()
    st.session_state["feature_selection"] = {
        "results": pipe.results,
        "selected_features": pipe.selected_features,
        "summary": summary,
        "filtered_datasets": filtered_datasets,
        "variant": variant_name,
        "targets": selected_targets,  # Now stores list of targets
        "primary_target": primary_target,
        # Pass through split indices from Feature Engineering for CQR
        "train_idx": train_idx,
        "cal_idx": cal_idx,  # Calibration set for CQR (Conformal Prediction)
        "test_idx": test_idx,
    }

    st.success(f"✅ Done! Feature selection completed for **{len(selected_targets)} target(s)**: {', '.join(selected_targets)}")

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
                cols = [c for c in (feats + [primary_target]) if c in dfv.columns]
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
                cols = [c for c in (feats + [primary_target]) if c in dfv.columns]
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
                      "targets": st.session_state["feature_selection"]["targets"],
                      "primary_target": st.session_state["feature_selection"]["primary_target"]
                  }), indent=2).encode("utf-8"),
                  "feature_selection_report.json", "application/json")

# ---------- Entrypoint ----------
def main():
    apply_css()
    inject_sidebar_style()
    render_sidebar_brand()

    # Results Storage Panel (Supabase persistence)
    render_results_storage_panel(
        page_type="feature_selection",
        page_title="Feature Selection",
    )

    # Auto-load saved results if available
    auto_load_if_available("feature_selection")

    # Fluorescent effects
    st.markdown("""
    <style>
    /* ========================================
       FLUORESCENT EFFECTS FOR AUTOMATED FEATURE SELECTION
       ======================================== */

    @keyframes float-orb {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.25;
        }
        50% {
            transform: translate(30px, -30px) scale(1.05);
            opacity: 0.35;
        }
    }

    .fluorescent-orb {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(70px);
    }

    .orb-1 {
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
        top: 15%;
        right: 20%;
        animation: float-orb 25s ease-in-out infinite;
    }

    .orb-2 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
        bottom: 20%;
        left: 15%;
        animation: float-orb 30s ease-in-out infinite;
        animation-delay: 5s;
    }

    @keyframes sparkle {
        0%, 100% {
            opacity: 0;
            transform: scale(0);
        }
        50% {
            opacity: 0.6;
            transform: scale(1);
        }
    }

    .sparkle {
        position: fixed;
        width: 3px;
        height: 3px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
        border-radius: 50%;
        pointer-events: none;
        z-index: 2;
        animation: sparkle 3s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }

    .sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
    .sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
    .sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

    @media (max-width: 768px) {
        .fluorescent-orb {
            width: 200px !important;
            height: 200px !important;
            filter: blur(50px);
        }
        .sparkle {
            display: none;
        }
    }
    </style>

    <!-- Fluorescent Floating Orbs -->
    <div class="fluorescent-orb orb-1"></div>
    <div class="fluorescent-orb orb-2"></div>

    <!-- Sparkle Particles -->
    <div class="sparkle sparkle-1"></div>
    <div class="sparkle sparkle-2"></div>
    <div class="sparkle sparkle-3"></div>
    """, unsafe_allow_html=True)

    page_feature_selection()

if __name__ == "__main__":
    main()
