# app_core/models/model_selection.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pmdarima as pm
except Exception:
    pm = None


# ==============================
# Data structures
# ==============================
@dataclass
class CandidateSpec:
    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    seasonal_m: Optional[int] = None
    info: Optional[Dict[str, Any]] = None  # e.g., aic, bic sampled from pmdarima search

@dataclass
class CandidateEval:
    spec: CandidateSpec
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    complexity: int
    runtime_s: float
    score: float

@dataclass
class SelectionResult:
    best_per_h: Dict[int, CandidateSpec]         # horizon -> best spec
    best_single: Optional[CandidateSpec]         # single spec across horizons (if requested)
    ranked_table: pd.DataFrame                   # overall ranking (averaged across horizons)
    per_h_tables: Dict[int, pd.DataFrame]        # ranking per horizon
    cv_diagnostics: Dict[str, Any]               # fold-by-fold metrics (optional)
    policy_snapshot: Dict[str, Any]              # weights/penalties/CV setup/epsilon
    runtime_s: float


# ==============================
# Helpers
# ==============================
def _ts_cv_splits(
    n: int,
    train_ratio: float,
    n_folds: int,
    strategy: str,
    min_train: int = 24,
    fold_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build time-ordered expanding or rolling splits.
    Each split returns (train_idx, test_idx).
    """
    assert strategy in {"expanding", "rolling"}
    n_train0 = max(int(n * train_ratio), min_train)
    if n_train0 >= n - 1:
        n_train0 = max(n - 2, min_train)
    if n_train0 < min_train:
        n_train0 = min_train

    if fold_size is None:
        fold_size = max(1, (n - n_train0) // max(1, n_folds))

    splits = []
    if strategy == "expanding":
        for k in range(n_folds):
            tr_end = n_train0 + k * fold_size
            te_end = min(tr_end + fold_size, n)
            if te_end <= tr_end or tr_end <= 0:
                break
            splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    else:  # rolling
        for k in range(n_folds):
            te_end = n_train0 + (k + 1) * fold_size
            te_end = min(te_end, n)
            te_start = te_end - fold_size
            tr_end = te_start
            tr_start = max(0, tr_end - n_train0)
            if tr_end - tr_start <= 1:
                break
            splits.append((np.arange(tr_start, tr_end), np.arange(te_start, te_end)))
    return splits


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _safe_mape(y_true, y_pred, eps)
    acc = 100.0 - mape if np.isfinite(mape) else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Accuracy": acc}


def _complexity(order: Tuple[int, int, int], seasonal: Optional[Tuple[int, int, int, int]]) -> int:
    p, d, q = order
    c = p + d + q
    if seasonal:
        P, D, Q, _s = seasonal
        c += P + D + Q
    return int(c)


def _composite_score(
    metrics_mean: Dict[str, float],
    metrics_std: Dict[str, float],
    weights: Dict[str, float],
    lam_complexity: float,
    mu_stability: float,
    complexity: int,
) -> float:
    """
    Lower is better. Minimize weighted (MAPE, RMSE, MAE, -Accuracy) with penalties.
    """
    mae = metrics_mean.get("MAE", np.nan)
    rmse = metrics_mean.get("RMSE", np.nan)
    mape = metrics_mean.get("MAPE", np.nan)
    acc = metrics_mean.get("Accuracy", np.nan)

    parts = []
    if np.isfinite(mape):
        parts.append(weights.get("MAPE", 0.0) * (mape / 100.0))
    if np.isfinite(rmse):
        parts.append(weights.get("RMSE", 0.0) * rmse)
    if np.isfinite(mae):
        parts.append(weights.get("MAE", 0.0) * mae)
    if np.isfinite(acc):
        parts.append(weights.get("-Accuracy", 0.0) * ((100.0 - acc) / 100.0))

    # Stability penalty: sum of stds (scaled for Accuracy)
    stab = 0.0
    for k in ("MAPE", "RMSE", "MAE", "Accuracy"):
        v = metrics_std.get(k, np.nan)
        if np.isfinite(v):
            s = v if k != "Accuracy" else v / 100.0
            stab += s

    penal = lam_complexity * complexity + mu_stability * stab
    return float(np.nansum(parts) + penal)


# ==============================
# Candidate generation (pmdarima)
# ==============================
def _harvest_candidates_auto_arima(
    y: np.ndarray,
    X: Optional[np.ndarray],
    seasonal: bool,
    m: int,
    max_candidates: int,
) -> List[CandidateSpec]:
    """
    Use pmdarima.auto_arima to explore and collect up to top-k specs by AIC.
    """
    if pm is None:
        raise RuntimeError("pmdarima is not installed. Run: pip install pmdarima")

    model = pm.auto_arima(
        y,
        X=X,
        seasonal=seasonal,
        m=m if seasonal else 1,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        max_order=12,
        max_p=5, max_q=5, max_d=2,
        max_P=3, max_Q=3, max_D=2,
        trace=False,
        information_criterion="aic",
        with_intercept=True,
    )

    tried: List[Tuple[Tuple[int,int,int], Optional[Tuple[int,int,int,int]], float, float]] = []
    try:
        if hasattr(model, "arima_res_") and hasattr(model.arima_res_, "results_dict"):
            df = pd.DataFrame(model.arima_res_.results_dict)
            df = df.sort_values("aic").reset_index(drop=True)
            for _, r in df.iterrows():
                order = tuple(int(x) for x in r["order"])
                seas = tuple(int(x) for x in r["seasonal_order"]) if seasonal else None
                tried.append((order, seas, float(r.get("aic", np.nan)), float(r.get("bic", np.nan))))
        else:
            tried.append((model.order_, model.seasonal_order_ if seasonal else None,
                          float(getattr(model, "aic", np.nan)), float(getattr(model, "bic", np.nan))))
    except Exception:
        tried.append((model.order_, model.seasonal_order_ if seasonal else None,
                      float(getattr(model, "aic", np.nan)), float(getattr(model, "bic", np.nan))))

    # Deduplicate and keep top-k
    seen = set()
    cands: List[CandidateSpec] = []
    for (ord_, seas, aic, bic) in tried:
        key = (ord_, seas)
        if key in seen:
            continue
        seen.add(key)
        cands.append(CandidateSpec(
            order=ord_,
            seasonal_order=seas,
            seasonal_m=(m if seasonal else None),
            info={"aic": aic, "bic": bic}
        ))
        if len(cands) >= max_candidates:
            break

    if not cands:
        cands.append(CandidateSpec(
            order=model.order_,
            seasonal_order=(model.seasonal_order_ if seasonal else None),
            seasonal_m=(m if seasonal else None),
            info={"aic": float(getattr(model, "aic", np.nan))}
        ))
    return cands


# ==============================
# Evaluation & selection
# ==============================
def _evaluate_spec_for_horizon(
    y: np.ndarray,
    X: Optional[np.ndarray],
    horizon: int,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    spec: CandidateSpec,
    eps: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit SARIMAX on each train split, forecast into test, and evaluate the 'horizon'-ahead alignment.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fold_metrics: Dict[str, List[float]] = {"MAE": [], "RMSE": [], "MAPE": [], "Accuracy": []}

    for train_idx, test_idx in splits:
        y_tr = y[train_idx]
        y_te = y[test_idx]
        X_tr = X[train_idx] if X is not None else None
        X_te = X[test_idx] if X is not None else None

        # Fit
        try:
            mod = SARIMAX(
                y_tr,
                exog=X_tr,
                order=spec.order,
                seasonal_order=(spec.seasonal_order or (0, 0, 0, 0)),
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            res = mod.fit(disp=False)
        except Exception:
            continue

        # Forecast the length of the test slice
        try:
            fc = res.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
            # horizon alignment (1-based horizon)
            if horizon > 1:
                y_true_h = y_te[horizon - 1:]
                y_pred_h = fc[:-horizon + 1]
            else:
                y_true_h = y_te
                y_pred_h = fc

            n = min(len(y_true_h), len(y_pred_h))
            if n <= 0:
                continue

            mb = _metric_bundle(y_true_h[:n], y_pred_h[:n], eps)
            for k in fold_metrics:
                fold_metrics[k].append(mb[k])
        except Exception:
            continue

    mean = {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in fold_metrics.items()}
    std = {k: float(np.nanstd(v)) if len(v) else np.nan for k, v in fold_metrics.items()}
    return mean, std


def select_orders_via_pmdarima(
    series: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    seasonal: bool = False,
    m: int = 1,
    horizons: Optional[List[int]] = None,
    cv_strategy: str = "expanding",
    n_folds: int = 3,
    train_ratio: float = 0.8,
    max_candidates: int = 20,
    weights: Optional[Dict[str, float]] = None,   # e.g., {"MAPE":0.5,"RMSE":0.25,"MAE":0.15,"-Accuracy":0.10}
    lam_complexity: float = 0.0,
    mu_stability: float = 0.0,
    eps: float = 1e-3,
    horizon_policy: str = "per_h",                # "per_h" or "single_spec"
) -> SelectionResult:
    """
    Master: harvest candidates with pmdarima, evaluate OOS via time-aware CV, rank by composite,
    return best per horizon and a single best (if requested).
    """
    t0 = time.time()
    if horizons is None:
        horizons = [1]

    y = np.asarray(series.values if isinstance(series, pd.Series) else series, dtype="float64")
    X = np.asarray(exog.values) if (exog is not None) else None

    splits = _ts_cv_splits(
        n=len(y),
        train_ratio=train_ratio,
        n_folds=n_folds,
        strategy=cv_strategy,
    )

    # Candidate list
    cands = _harvest_candidates_auto_arima(
        y=y,
        X=X,
        seasonal=seasonal,
        m=m,
        max_candidates=max_candidates,
    )

    # Defaults for weights
    weights = weights or {"MAPE": 0.5, "RMSE": 0.25, "MAE": 0.15, "-Accuracy": 0.10}

    # Evaluate each candidate for each horizon
    per_h_tables: Dict[int, pd.DataFrame] = {}
    per_h_best: Dict[int, CandidateSpec] = {}

    all_rows_for_global: List[Dict[str, Any]] = []

    for h in horizons:
        rows: List[Dict[str, Any]] = []
        for spec in cands:
            t1 = time.time()
            mean, std = _evaluate_spec_for_horizon(y, X, h, splits, spec, eps)
            rt = time.time() - t1
            comp = _complexity(spec.order, spec.seasonal_order)
            score = _composite_score(mean, std, weights, lam_complexity, mu_stability, comp)

            row = {
                "Horizon": h,
                "order": spec.order,
                "seasonal_order": spec.seasonal_order,
                "m": spec.seasonal_m,
                "AIC_seed": spec.info.get("aic") if spec.info else np.nan,
                "BIC_seed": spec.info.get("bic") if spec.info else np.nan,
                "MAE": mean["MAE"], "RMSE": mean["RMSE"], "MAPE": mean["MAPE"], "Accuracy": mean["Accuracy"],
                "MAE_std": std["MAE"], "RMSE_std": std["RMSE"], "MAPE_std": std["MAPE"], "Accuracy_std": std["Accuracy"],
                "Complexity": comp,
                "Runtime_s": rt,
                "Score": score,
            }
            rows.append(row)
            all_rows_for_global.append(row)

        tbl = pd.DataFrame(rows).sort_values(["Score", "MAPE", "RMSE"], ascending=[True, True, True]).reset_index(drop=True)
        per_h_tables[h] = tbl
        if not tbl.empty:
            top = tbl.iloc[0]
            per_h_best[h] = CandidateSpec(
                order=tuple(map(int, top["order"])),
                seasonal_order=tuple(map(int, top["seasonal_order"])) if pd.notna(top["seasonal_order"]).all() if isinstance(top["seasonal_order"], (list, tuple, np.ndarray)) else (top["seasonal_order"] if top["seasonal_order"] else None) else None,
                seasonal_m=(int(top["m"]) if pd.notna(top["m"]) else None),
                info={"score": float(top["Score"])}
            )

    # Global ranking (averaged across horizons for each unique spec)
    global_df = pd.DataFrame(all_rows_for_global)
    if not global_df.empty:
        key_cols = ["order", "seasonal_order", "m", "Complexity"]
        metric_cols = ["MAE", "RMSE", "MAPE", "Accuracy", "Runtime_s", "Score"]
        agg = {c: "mean" for c in metric_cols}
        ranked = (global_df.groupby(key_cols, dropna=False)[metric_cols]
                  .agg(agg).reset_index()
                  .sort_values(["Score", "MAPE", "RMSE"], ascending=[True, True, True])
                  .reset_index(drop=True))
    else:
        ranked = pd.DataFrame(columns=["order", "seasonal_order", "m", "MAE", "RMSE", "MAPE", "Accuracy", "Runtime_s", "Score", "Complexity"])

    best_single: Optional[CandidateSpec] = None
    if not ranked.empty and horizon_policy == "single_spec":
        r0 = ranked.iloc[0]
        best_single = CandidateSpec(
            order=tuple(map(int, r0["order"])),
            seasonal_order=tuple(map(int, r0["seasonal_order"])) if isinstance(r0["seasonal_order"], (list, tuple, np.ndarray)) else (r0["seasonal_order"] if r0["seasonal_order"] else None),
            seasonal_m=(int(r0["m"]) if pd.notna(r0["m"]) else None),
            info={"score": float(r0["Score"])}
        )

    sel = SelectionResult(
        best_per_h=per_h_best,
        best_single=best_single,
        ranked_table=ranked,
        per_h_tables=per_h_tables,
        cv_diagnostics={},  # keep simple; can add fold traces later
        policy_snapshot={
            "weights": weights,
            "lam_complexity": lam_complexity,
            "mu_stability": mu_stability,
            "eps": eps,
            "cv_strategy": cv_strategy,
            "n_folds": n_folds,
            "train_ratio": train_ratio,
            "horizon_policy": horizon_policy,
            "seasonal": seasonal,
            "m": m,
            "max_candidates": max_candidates,
        },
        runtime_s=time.time() - t0,
    )
    return sel
