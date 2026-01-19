# app_core/models/sarimax_pipeline.py
# Thesis-grade SARIMAX with exogenous features, multi-horizon outputs, and ARIMA-compatible artifacts

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional auto-order (pmdarima). If absent, we fall back to safe defaults.
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

# ---------- styling tokens (optional; keep consistent with ARIMA plots) ----------
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
WARNING_COLOR = "#f59e0b"
SUCCESS_COLOR = "#10b981"
DANGER_COLOR = "#ef4444"
CARD_BG_LIGHT = "#ffffff"
TEXT_COLOR = "#2c3e50"
GRID_COLOR = "#e5e7eb"

# ---------- utils ----------
SEED = 42
np.random.seed(SEED)

def _mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error in %, robust to zeros (skips near-zero denom)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _dir_acc(y_true, y_pred) -> float:
    """Directional accuracy (% of times the sign of change matches)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    up_t = (np.diff(y_true) > 0)
    up_p = (np.diff(y_pred) > 0)
    return float(np.mean(up_t == up_p) * 100.0)

def _clean_daily(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Daily regularization + numeric coercion + interpolate/ffill/bfill on all numeric columns."""
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not in dataframe.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid dates after coercion.")
    full_idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    out = df.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()
    for c in out.columns:
        if c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].interpolate("linear").ffill().bfill()
    return out

def _exog_calendar(index, include_dow_ohe: bool = True) -> pd.DataFrame:
    """Calendar features (DOW one-hot, weekend flag, yearly sin/cos).

    Args:
        index: DatetimeIndex or Series of datetime values
        include_dow_ohe: Whether to include day-of-week one-hot encoding
    """
    # Handle both DatetimeIndex and Series inputs
    if isinstance(index, pd.Series):
        dt_index = pd.DatetimeIndex(index)
    else:
        dt_index = index

    X = pd.DataFrame(index=dt_index)
    dow = dt_index.dayofweek
    if include_dow_ohe:
        # Mon..Sun one-hot
        for i, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
            X[f"is_{name}"] = (dow == i).astype(int)
    X["is_weekend"] = (dow >= 5).astype(int)
    t = dt_index.dayofyear.astype(float)
    X["sin_y"] = np.sin(2*np.pi*t/365.25)
    X["cos_y"] = np.cos(2*np.pi*t/365.25)
    return X

def _exog_all(
    df_full: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    include_dow_ohe: bool,
    selected_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build exogenous feature matrix from ALL non-target, non-date columns (+ optional DOW OHE).
    If selected_features is provided, we subset to those (when present).
    """
    drop_cols = [date_col] + [c for c in target_cols if c in df_full.columns]
    X = df_full.drop(columns=drop_cols, errors="ignore").copy()

    # Clean numerics
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Add calendar OHE on DOW if requested
    cal = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)
    cal = cal.reset_index(drop=True)  # Align integer index with X

    # Combine
    X = pd.concat([X, cal], axis=1)

    # If user specified a subset of features, keep intersection only
    if selected_features:
        keep = [c for c in selected_features if c in X.columns]
        if include_dow_ohe:
            keep += [c for c in cal.columns if c not in keep]
        keep = list(dict.fromkeys(keep))  # de-dup
        if keep:
            X = X[keep]

    X = X.interpolate("linear").ffill().bfill()

    # Ensure all columns are numeric float64 (SARIMAX requires numeric data)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)

    # Drop columns that are entirely NaN (can't be used as features)
    X = X.dropna(axis=1, how="all")

    # Fill any remaining NaN with 0 to avoid dtype issues
    X = X.fillna(0)

    return X

def _auto_order_rmse_only(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
    n_folds: int = 3,
    cv_strategy: str = "expanding",
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """
    Find optimal SARIMAX orders by minimizing PURE CV-RMSE (no AIC penalty).

    Uses time series cross-validation for robust RMSE estimation.
    This method prioritizes out-of-sample prediction accuracy over model complexity.

    Args:
        y_tr: Training target series
        X_tr: Training exogenous features
        m: Seasonal period
        n_folds: Number of CV folds (default 3)
        cv_strategy: Cross-validation strategy - "expanding" (default) or "rolling"
                     - "expanding": Training window grows over time (recommended for time series)
                     - "rolling": Fixed-size training window slides forward
        max_p, max_q, max_d: Non-seasonal parameter bounds
        max_P, max_Q, max_D: Seasonal parameter bounds

    Returns:
        Tuple (order, seasonal_order, aic, bic) with lowest CV-RMSE
    """
    results = []

    # Generate candidate orders - comprehensive search
    candidates = []

    # Common SARIMAX orders
    for p in range(0, min(max_p + 1, 4)):
        for d in range(0, min(max_d + 1, 3)):
            for q in range(0, min(max_q + 1, 4)):
                for P in range(0, min(max_P + 1, 3)):
                    for D in range(0, min(max_D + 1, 2)):
                        for Q in range(0, min(max_Q + 1, 3)):
                            candidates.append(((p, d, q), (P, D, Q, m)))

    # Limit candidates for efficiency
    if len(candidates) > 50:
        # Prioritize common orders
        priority_orders = [
            ((1, 1, 1), (1, 1, 0, m)),
            ((1, 1, 1), (1, 1, 1, m)),
            ((0, 1, 1), (0, 1, 1, m)),
            ((1, 1, 0), (1, 1, 0, m)),
            ((2, 1, 2), (1, 1, 1, m)),
            ((1, 0, 1), (1, 0, 1, m)),
            ((0, 0, 0), (0, 0, 0, m)),
            ((0, 1, 1), (1, 1, 0, m)),
            ((1, 1, 1), (0, 1, 1, m)),
            ((2, 1, 1), (1, 1, 0, m)),
        ]
        # Add variations
        import random
        random.seed(42)
        other_candidates = [c for c in candidates if c not in priority_orders]
        random.shuffle(other_candidates)
        candidates = priority_orders + other_candidates[:40]

    # Remove duplicates
    candidates = list(set(candidates))

    # Evaluate each candidate with CV-RMSE
    for order, seasonal_order in candidates:
        try:
            # Time Series CV for RMSE
            cv_rmses = []
            fold_size = max(20, len(y_tr) // (n_folds + 1))

            for i in range(n_folds):
                # Calculate train window based on cv_strategy
                if cv_strategy == "expanding":
                    # Expanding window: training grows over time
                    train_start = 0
                    train_end = fold_size * (i + 2)
                else:
                    # Rolling window: fixed-size training window
                    train_start = fold_size * i
                    train_end = fold_size * (i + 2)

                if train_end > len(y_tr):
                    break

                test_start = train_end
                test_end = min(train_end + fold_size, len(y_tr))

                if test_end <= test_start:
                    continue

                y_cv_train = y_tr.iloc[train_start:train_end]
                y_cv_test = y_tr.iloc[test_start:test_end]

                if X_tr is not None and X_tr.shape[1] > 0:
                    X_cv_train = X_tr.iloc[train_start:train_end]
                    X_cv_test = X_tr.iloc[test_start:test_end]
                else:
                    X_cv_train = None
                    X_cv_test = None

                if len(y_cv_train) < 20 or len(y_cv_test) == 0:
                    continue

                cv_model = SARIMAX(
                    endog=y_cv_train,
                    exog=X_cv_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                cv_forecast = cv_model.get_forecast(
                    steps=len(y_cv_test),
                    exog=X_cv_test
                ).predicted_mean

                rmse_fold = float(np.sqrt(mean_squared_error(y_cv_test, cv_forecast)))
                cv_rmses.append(rmse_fold)

            # Average CV-RMSE across folds
            avg_cv_rmse = float(np.mean(cv_rmses)) if cv_rmses else float('inf')

            if np.isfinite(avg_cv_rmse):
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'cv_rmse': avg_cv_rmse,
                })

        except Exception:
            continue

    if not results:
        # Fallback to default
        return (1, 1, 1), (1, 1, 0, m), float("nan"), float("nan")

    # Select order with lowest CV-RMSE (pure RMSE, no AIC)
    df = pd.DataFrame(results)
    best_idx = df['cv_rmse'].idxmin()
    best_row = df.loc[best_idx]

    # Fit full model to get AIC/BIC for reporting
    try:
        full_model = SARIMAX(
            endog=y_tr,
            exog=X_tr if X_tr is not None and X_tr.shape[1] > 0 else None,
            order=tuple(best_row['order']),
            seasonal_order=tuple(best_row['seasonal_order']),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        aic = float(full_model.aic)
        bic = float(full_model.bic)
    except Exception:
        aic = float("nan")
        bic = float("nan")

    return (
        tuple(best_row['order']),
        tuple(best_row['seasonal_order']),
        aic,
        bic
    )


def _auto_order(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """Auto-select orders via pmdarima if available; else a safe default."""
    if PMDARIMA_AVAILABLE:
        a = pm.auto_arima(
            y_tr.values.astype(float),
            X=None if X_tr is None or X_tr.shape[1] == 0 else X_tr.values,
            start_p=0, max_p=max_p, start_q=0, max_q=max_q,
            d=None, max_d=max_d, seasonal=True, m=m,
            start_P=0, max_P=max_P, start_Q=0, max_Q=max_Q,
            D=None, max_D=max_D, stepwise=True, suppress_warnings=True,
            error_action="ignore", information_criterion="aic",
            random_state=SEED, maxiter=200,
        )
        return a.order, a.seasonal_order, float(a.aic()), float(a.bic())
    # Fallback: modest seasonal default (weekly seasonality)
    return (1, 1, 1), (1, 1, 0, m), float("nan"), float("nan")


def _auto_order_hybrid(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
    n_folds: int = 3,
    cv_strategy: str = "expanding",
    alpha: float = 0.3,  # AIC weight
    beta: float = 0.7,   # CV-RMSE weight
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """
    Find optimal SARIMAX orders by minimizing weighted composite score:
    Score = alpha * Normalized_AIC + beta * Normalized_CV_RMSE

    Uses time series cross-validation for robust RMSE estimation.
    Leverages pmdarima for efficient candidate generation.

    Args:
        y_tr: Training target series
        X_tr: Training exogenous features
        m: Seasonal period
        n_folds: Number of CV folds (default 3)
        cv_strategy: Cross-validation strategy - "expanding" (default) or "rolling"
                     - "expanding": Training window grows over time (recommended for time series)
                     - "rolling": Fixed-size training window slides forward
        alpha: Weight for AIC (complexity penalty), default 0.3
        beta: Weight for CV-RMSE (prediction accuracy), default 0.7
        max_p, max_q, max_d: Non-seasonal parameter bounds
        max_P, max_Q, max_D: Seasonal parameter bounds

    Returns:
        Tuple (order, seasonal_order, aic, bic)
    """
    results = []

    # Generate candidate orders using pmdarima stepwise (if available)
    if PMDARIMA_AVAILABLE:
        try:
            # Get best order from pmdarima as starting point
            auto_model = pm.auto_arima(
                y_tr.values.astype(float),
                X=None if X_tr is None or X_tr.shape[1] == 0 else X_tr.values,
                start_p=0, max_p=max_p, start_q=0, max_q=max_q,
                d=None, max_d=max_d, seasonal=True, m=m,
                start_P=0, max_P=max_P, start_Q=0, max_Q=max_Q,
                D=None, max_D=max_D, stepwise=True, suppress_warnings=True,
                error_action="ignore", information_criterion="aic",
                random_state=SEED, maxiter=100,
            )

            order_pm = auto_model.order
            seasonal_order_pm = auto_model.seasonal_order

            # Generate candidate variations around pmdarima suggestion
            candidates = [
                (order_pm, seasonal_order_pm),
            ]

            # Add variations for robustness
            p, d, q = order_pm
            P, D, Q, s = seasonal_order_pm

            # Variations in non-seasonal parameters
            if p > 0:
                candidates.append(((p-1, d, q), seasonal_order_pm))
            if p < max_p:
                candidates.append(((p+1, d, q), seasonal_order_pm))
            if q > 0:
                candidates.append(((p, d, q-1), seasonal_order_pm))
            if q < max_q:
                candidates.append(((p, d, q+1), seasonal_order_pm))

            # Variations in seasonal parameters
            if P > 0:
                candidates.append((order_pm, (P-1, D, Q, s)))
            if P < max_P:
                candidates.append((order_pm, (P+1, D, Q, s)))

        except Exception:
            # Fallback to simple candidates
            candidates = [
                ((1, 1, 1), (1, 1, 0, m)),
                ((1, 1, 0), (1, 1, 0, m)),
                ((0, 1, 1), (1, 1, 0, m)),
            ]
    else:
        # No pmdarima available, use simple defaults
        candidates = [
            ((1, 1, 1), (1, 1, 0, m)),
            ((1, 1, 0), (1, 1, 0, m)),
        ]

    # Remove duplicates
    candidates = list(set(candidates))

    # Evaluate each candidate with CV-RMSE
    for order, seasonal_order in candidates:
        try:
            # 1. Full fit for AIC
            model_full = SARIMAX(
                endog=y_tr,
                exog=X_tr if X_tr is not None and X_tr.shape[1] > 0 else None,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            aic = float(model_full.aic)
            bic = float(model_full.bic)

            # 2. Time Series CV for RMSE
            cv_rmses = []
            fold_size = max(20, len(y_tr) // (n_folds + 1))

            for i in range(n_folds):
                # Calculate train window based on cv_strategy
                if cv_strategy == "expanding":
                    # Expanding window: training grows over time
                    train_start = 0
                    train_end = fold_size * (i + 2)
                else:
                    # Rolling window: fixed-size training window
                    train_start = fold_size * i
                    train_end = fold_size * (i + 2)

                if train_end > len(y_tr):
                    break

                test_start = train_end
                test_end = min(train_end + fold_size, len(y_tr))

                if test_end <= test_start:
                    continue

                y_cv_train = y_tr.iloc[train_start:train_end]
                y_cv_test = y_tr.iloc[test_start:test_end]

                if X_tr is not None and X_tr.shape[1] > 0:
                    X_cv_train = X_tr.iloc[train_start:train_end]
                    X_cv_test = X_tr.iloc[test_start:test_end]
                else:
                    X_cv_train = None
                    X_cv_test = None

                if len(y_cv_train) < 20 or len(y_cv_test) == 0:
                    continue

                cv_model = SARIMAX(
                    endog=y_cv_train,
                    exog=X_cv_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                cv_forecast = cv_model.get_forecast(
                    steps=len(y_cv_test),
                    exog=X_cv_test
                ).predicted_mean

                rmse_fold = float(np.sqrt(mean_squared_error(y_cv_test, cv_forecast)))
                cv_rmses.append(rmse_fold)

            # Average CV-RMSE across folds
            avg_cv_rmse = float(np.mean(cv_rmses)) if cv_rmses else float('inf')

            if np.isfinite(aic) and np.isfinite(avg_cv_rmse):
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': aic,
                    'bic': bic,
                    'cv_rmse': avg_cv_rmse,
                })

        except Exception:
            continue

    if not results:
        # Fallback to default
        return (1, 1, 1), (1, 1, 0, m), float("nan"), float("nan")

    # Normalize and score
    df = pd.DataFrame(results)

    aic_min, aic_max = df['aic'].min(), df['aic'].max()
    rmse_min, rmse_max = df['cv_rmse'].min(), df['cv_rmse'].max()

    # Avoid division by zero
    aic_range = aic_max - aic_min if aic_max > aic_min else 1.0
    rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1.0

    df['aic_norm'] = (df['aic'] - aic_min) / aic_range
    df['rmse_norm'] = (df['cv_rmse'] - rmse_min) / rmse_range

    # Weighted composite score (lower is better)
    df['composite_score'] = alpha * df['aic_norm'] + beta * df['rmse_norm']

    # Select best
    best_idx = df['composite_score'].idxmin()
    best_row = df.loc[best_idx]

    return (
        tuple(best_row['order']),
        tuple(best_row['seasonal_order']),
        float(best_row['aic']),
        float(best_row['bic'])
    )


# ---------- single-horizon SARIMAX (h=1) ----------
def run_sarimax_single(
    df_merged: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "Target_1",
    train_ratio: float = 0.80,
    season_length: int = 7,
    use_all_features: bool = True,
    include_dow_ohe: bool = True,
    # manual order overrides
    order: Optional[Tuple[int,int,int]] = None,
    seasonal_order: Optional[Tuple[int,int,int,int]] = None,
    # bounds for auto-order (when order/seasonal_order are None)
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Dict[str, Any]:
    """
    Train + evaluate a single-horizon SARIMAX (like ARIMA single). Returns a payload
    compatible with your ARIMA single-horizon dashboard (y_train, y_test, forecasts,
    forecast_lower/upper, train/test metrics, residuals, etc.).
    """
    df_full = _clean_daily(df_merged, date_col)
    if target_col not in df_full.columns:
        raise ValueError(f"'{target_col}' not found in dataframe.")

    y = df_full.set_index(date_col)[target_col].astype(float)
    y = y.interpolate("linear").ffill().bfill()

    if len(y) < 24:
        raise ValueError("Not enough data for SARIMAX single-horizon (need >= 24 points).")

    # Exogenous
    if use_all_features:
        X_all = _exog_all(
            df_full=df_full, date_col=date_col,
            target_cols=[target_col], include_dow_ohe=include_dow_ohe
        )
    else:
        X_all = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)
        X_all = X_all.reset_index(drop=True)  # Align integer index
        # Ensure numeric types for SARIMAX
        for c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce").astype(float)
        X_all = X_all.fillna(0)
    X_all.index = df_full[date_col]

    # Split
    n = len(y)
    split = int(n * float(train_ratio))
    split = max(12, min(n - 1, split))
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    X_tr, X_te = X_all.loc[y_tr.index], X_all.loc[y_te.index]

    # Orders
    if (order is None) or (seasonal_order is None):
        ord_auto, sord_auto, aic_auto, bic_auto = _auto_order(
            y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
            max_p=max_p, max_q=max_q, max_d=max_d,
            max_P=max_P, max_Q=max_Q, max_D=max_D
        )
        use_order = order if order is not None else ord_auto
        use_sorder = seasonal_order if seasonal_order is not None else sord_auto
    else:
        use_order, use_sorder = order, seasonal_order

    # Fit
    fit = SARIMAX(
        endog=y_tr,
        exog=X_tr if X_tr.shape[1] > 0 else None,
        order=use_order,
        seasonal_order=use_sorder,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # In-sample/fitted
    fitted = fit.fittedvalues.reindex(y_tr.index)

    # Forecast on test
    fc = fit.get_forecast(steps=len(y_te), exog=(X_te if X_te.shape[1] > 0 else None))
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    mean.index = y_te.index
    ci.index = y_te.index

    # Metrics
    tr_mae = float(mean_absolute_error(y_tr, fitted))
    tr_rmse = _rmse(y_tr, fitted)
    tr_mape = float(_mape(y_tr, fitted))
    try:
        tr_r2 = float(r2_score(y_tr, fitted))
    except Exception:
        tr_r2 = float("nan")
    tr_acc = 100.0 - tr_mape if np.isfinite(tr_mape) else 0.0

    te_mae = float(mean_absolute_error(y_te, mean))
    te_rmse = _rmse(y_te, mean)
    te_mape = float(_mape(y_te, mean))
    try:
        te_r2 = float(r2_score(y_te, mean))
    except Exception:
        te_r2 = float("nan")
    te_acc = 100.0 - te_mape if np.isfinite(te_mape) else 0.0

    bias = float((mean - y_te).mean())
    dacc = _dir_acc(y_te.values, mean.values)
    abs_err = np.abs(mean - y_te)
    within2 = float((abs_err <= 2).mean() * 100.0)
    within5 = float((abs_err <= 5).mean() * 100.0)
    cicov = float(((y_te >= ci.iloc[:, 0]) & (y_te <= ci.iloc[:, 1])).mean() * 100.0)

    return {
        "y_train": y_tr,
        "y_test": y_te,
        "forecasts": mean,
        "forecast_lower": ci.iloc[:, 0],
        "forecast_upper": ci.iloc[:, 1],
        "train_metrics": {
            "AIC": float(fit.aic),
            "BIC": float(fit.bic),
            "Log-Likelihood": float(fit.llf),
            "HQIC": float(getattr(fit, "hqic", np.nan)),
        },
        "test_metrics": {
            "MAE": te_mae, "RMSE": te_rmse, "MAPE": te_mape, "Accuracy": te_acc,
            "R2": te_r2, "Bias": bias, "CI_Coverage_%": cicov,
            "Direction_Accuracy_%": dacc, "Within_±2_%": within2, "Within_±5_%": within5,
        },
        "model_info": {
            "order": tuple(use_order),
            "seasonal_order": tuple(use_sorder),
            "aic": float(fit.aic), "bic": float(fit.bic),
            "llf": float(fit.llf), "hqic": float(getattr(fit, "hqic", np.nan)),
            "params": fit.params.to_dict() if hasattr(fit.params, "to_dict") else {},
        },
        "residuals": pd.Series(fit.resid, index=y_tr.index).dropna(),
        "fitted_values": fitted,
        "seasonal_proportions": None,
        "category_forecasts": None,
    }

# ---------- multi-horizon SARIMAX (h = 1..H) ----------
def run_sarimax_multihorizon(
    df_merged: pd.DataFrame,
    date_col: str = "Date",
    target_cols: Optional[List[str]] = None,       # e.g. ["Target_1",...,"Target_7"]
    train_ratio: float = 0.80,
    season_length: int = 7,
    horizons: int = 7,
    use_all_features: bool = True,
    include_dow_ohe: bool = True,
    # optional manual orders
    order: Optional[Tuple[int,int,int]] = None,
    seasonal_order: Optional[Tuple[int,int,int,int]] = None,
    # optional explicit feature subset
    selected_features: Optional[List[str]] = None,
    # Bounds for auto-order search (only used when order/seasonal_order are None)
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
    # Search mode: "rmse_only" (default - seeks lowest RMSE), "aic_only", "hybrid"
    search_mode: str = "rmse_only",
    n_folds: int = 3,
    cv_strategy: str = "expanding",
) -> Dict[str, Any]:
    """
    Train SARIMAX per horizon 1..H (expects columns Target_1..Target_H).
    Returns: {'results_df', 'per_h', 'successful'}
    """
    if target_cols is None:
        target_cols = [f"Target_{i}" for i in range(1, horizons + 1)]

    # Regularize to daily, coerce numeric, interpolate
    df_full = _clean_daily(df_merged, date_col)

    # Pre-build exogenous features once (time-indexed) if needed
    X_all_global = None
    if use_all_features:
        X_all_global = _exog_all(
            df_full=df_full,
            date_col=date_col,
            target_cols=target_cols,
            include_dow_ohe=include_dow_ohe,
            selected_features=selected_features,
        )
        X_all_global.index = df_full[date_col]

    rows: List[Dict[str, Any]] = []
    per_h: Dict[int, Dict[str, Any]] = {}
    ok: List[int] = []

    # Parameter sharing: find optimal parameters once for h=1, reuse for h>1 (automatic mode only)
    shared_order = None
    shared_seasonal_order = None

    for h in range(1, horizons + 1):
        tc = f"Target_{h}"
        if tc not in df_full.columns:
            continue

        # y series
        y = df_full.set_index(date_col)[tc].astype(float)
        y = y.interpolate("linear").ffill().bfill()

        # require sensible history
        if len(y) < 100:
            continue

        # Build exogenous matrix for this horizon
        if use_all_features:
            X = X_all_global.reindex(y.index).copy()
        else:
            X = _exog_calendar(y.index, include_dow_ohe=include_dow_ohe)
            if selected_features:
                keep = [c for c in selected_features if c in X.columns]
                if include_dow_ohe:
                    keep = list(dict.fromkeys(keep + [c for c in X.columns if c not in keep]))
                X = X[keep] if keep else X

        X = X.interpolate("linear").ffill().bfill()

        # Train/test split
        n = len(y)
        split = int(n * float(train_ratio))
        split = max(10, min(n - 1, split))  # guards
        y_tr, y_te = y.iloc[:split], y.iloc[split:]
        X_tr, X_te = X.iloc[:split, :], X.iloc[split:, :]

        # Determine orders (manual overrides > auto)
        # For automatic mode: find parameters for h=1, reuse for h>1 (7x speedup)
        if (order is None or seasonal_order is None) and h == 1:
            # Full auto search for first horizon only
            if search_mode == "rmse_only":
                # RMSE-only mode: optimize pure CV-RMSE (DEFAULT - seeks lowest RMSE)
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_rmse_only(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, cv_strategy=cv_strategy,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            elif search_mode == "hybrid":
                # Hybrid mode: optimize AIC + CV-RMSE (70% RMSE weight)
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_hybrid(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, cv_strategy=cv_strategy, alpha=0.3, beta=0.7,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            else:
                # AIC-only mode: standard pmdarima stepwise
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            # Save the optimal parameters found for reuse
            shared_order = ord_auto if order is None else order
            shared_seasonal_order = sord_auto if seasonal_order is None else seasonal_order
            use_order = shared_order
            use_sorder = shared_seasonal_order
            aic_ref, bic_ref = aic_auto, bic_auto
        elif (order is None or seasonal_order is None) and h > 1 and shared_order is not None and shared_seasonal_order is not None:
            # Reuse parameters from h=1 for subsequent horizons (skip expensive auto search)
            use_order = shared_order
            use_sorder = shared_seasonal_order
            aic_ref, bic_ref = float("nan"), float("nan")
        elif order is not None and seasonal_order is not None:
            # Manual mode: use user-specified orders (unchanged)
            use_order, use_sorder = order, seasonal_order
            aic_ref, bic_ref = float("nan"), float("nan")
        else:
            # Fallback for edge cases
            if search_mode == "rmse_only":
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_rmse_only(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, cv_strategy=cv_strategy,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            elif search_mode == "hybrid":
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_hybrid(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, cv_strategy=cv_strategy, alpha=0.3, beta=0.7,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            else:
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            use_order = ord_auto if order is None else order
            use_sorder = sord_auto if seasonal_order is None else seasonal_order
            aic_ref, bic_ref = aic_auto, bic_auto

        # Fit model
        fit = SARIMAX(
            endog=y_tr,
            exog=X_tr if X_tr.shape[1] > 0 else None,
            order=use_order,
            seasonal_order=use_sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # In-sample fit
        fitted = fit.fittedvalues.reindex(y_tr.index)

        # Out-of-sample forecast
        fc = fit.get_forecast(steps=len(y_te), exog=(X_te if X_te.shape[1] > 0 else None))
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        mean.index = y_te.index
        ci.index = y_te.index

        # ------------ metrics ------------
        tr_mae = float(mean_absolute_error(y_tr, fitted))
        tr_rmse = _rmse(y_tr, fitted)
        tr_mape = float(_mape(y_tr, fitted))
        try:
            tr_r2 = float(r2_score(y_tr, fitted))
        except Exception:
            tr_r2 = float("nan")
        tr_acc = 100.0 - tr_mape if np.isfinite(tr_mape) else 0.0

        te_mae = float(mean_absolute_error(y_te, mean))
        te_rmse = _rmse(y_te, mean)
        te_mape = float(_mape(y_te, mean))
        try:
            te_r2 = float(r2_score(y_te, mean))
        except Exception:
            te_r2 = float("nan")
        te_acc = 100.0 - te_mape if np.isfinite(te_mape) else 0.0

        bias = float((mean - y_te).mean())
        dacc = _dir_acc(y_te.values, mean.values)
        abs_err = np.abs(mean - y_te)
        within2 = float((abs_err <= 2).mean() * 100.0)
        within5 = float((abs_err <= 5).mean() * 100.0)
        cicov = float(((y_te >= ci.iloc[:, 0]) & (y_te <= ci.iloc[:, 1])).mean() * 100.0)

        rows.append({
            "Horizon": h,
            "Order": f"{use_order}×{use_sorder}",
            "AIC": aic_ref if not (isinstance(aic_ref, float) and np.isnan(aic_ref)) else float(fit.aic),
            "BIC": bic_ref if not (isinstance(bic_ref, float) and np.isnan(bic_ref)) else float(fit.bic),
            "Train_MAE": tr_mae, "Train_RMSE": tr_rmse, "Train_MAPE": tr_mape, "Train_Acc": tr_acc, "Train_R2": tr_r2,
            "Test_MAE": te_mae, "Test_RMSE": te_rmse, "Test_MAPE": te_mape, "Test_Acc": te_acc, "Test_R2": te_r2,
            "Bias": bias, "DirAcc": dacc, "Within±2": within2, "Within±5": within5, "CIcov%": cicov,
            "Train_N": int(len(y_tr)), "Test_N": int(len(y_te)),
        })

        per_h[h] = {
            "fit": fit,
            "res": fit,  # expose results for residual diagnostics (has resid, fittedvalues)
            "y_train": y_tr, "y_test": y_te,
            "fitted": fitted, "forecast": mean,
            "ci_lo": ci.iloc[:, 0], "ci_hi": ci.iloc[:, 1],
            "order": use_order, "sorder": use_sorder,
        }
        ok.append(h)

    if not ok:
        raise RuntimeError("SARIMAX: not enough data to train any horizon.")

    results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)
    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": ok,
        "seasonal_proportions": None,
        "category_forecasts": None,
    }

# ---------- helpers to adapt outputs for dashboards ----------
def calculate_multi_horizon_metrics_from_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert results_df (per-h row metrics) to the canonical metrics_df used by your plots.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=[
            "Horizon","MAE","RMSE","MAPE_%","Accuracy_%","R2","CI_Coverage_%","Direction_Accuracy_%"
        ])

    m = results_df.copy()
    out = pd.DataFrame({
        "Horizon": [f"h={h}" for h in m["Horizon"].tolist()],
        "MAE": m["Test_MAE"].astype(float),
        "RMSE": m["Test_RMSE"].astype(float),
        "MAPE_%": m["Test_MAPE"].astype(float),
        "Accuracy_%": m["Test_Acc"].astype(float),
        "R2": m.get("Test_R2", pd.Series([np.nan]*len(m))).astype(float),
        "CI_Coverage_%": m.get("CIcov%", pd.Series([np.nan]*len(m))).astype(float),
        "Direction_Accuracy_%": m.get("DirAcc", pd.Series([np.nan]*len(m))).astype(float),
    })

    # Average row
    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        avg_row = {k: out[k].mean() for k in numeric_cols}
        avg_row["Horizon"] = "Average"
        out = pd.concat([out, pd.Series(avg_row).to_frame().T], ignore_index=True)

    return out.reset_index(drop=True)

def to_multihorizon_artifacts(sarimax_out: Dict[str, Any]):
    """
    Convert pipeline outputs to plotting inputs:
      - metrics_df: columns ["Horizon","MAE","RMSE","MAPE_%","Accuracy_%", ...]
      - F, L, U: arrays (n_dates x n_horizons)
      - test_eval: DataFrame with columns "Target_1"... "Target_H" and Date index for test period
      - train: pd.Series (training target, horizon-agnostic)
      - res: fitted model results-like (must have .resid, .fittedvalues)
      - horizons: sorted list of integer horizons available
    """
    res_df = sarimax_out["results_df"]           # per-horizon metrics table
    per_h = sarimax_out["per_h"]                 # dict: h -> dict(...)

    # --- metrics_df in canonical shape ---
    metrics_df = calculate_multi_horizon_metrics_from_rows(res_df)

    # --- build F, L, U, test_eval aligned on the test index used for horizon 1 ---
    horizons = sorted(per_h.keys())
    idx = per_h[horizons[0]]["y_test"].index                # base test index
    n, H = len(idx), len(horizons)
    F = np.full((n, H), np.nan)
    L = np.full((n, H), np.nan)
    U = np.full((n, H), np.nan)

    test_eval = pd.DataFrame(index=idx)
    for j, h in enumerate(horizons):
        d = per_h[h]
        # Ensure alignment on index
        f = d["forecast"].reindex(idx).values
        lo = d.get("ci_lo").reindex(idx).values if d.get("ci_lo") is not None else np.full(n, np.nan)
        hi = d.get("ci_hi").reindex(idx).values if d.get("ci_hi") is not None else np.full(n, np.nan)
        y = d["y_test"].reindex(idx).values

        F[:, j] = f
        L[:, j] = lo
        U[:, j] = hi
        test_eval[f"Target_{h}"] = y

    # pick a train series (any horizon has same train target)
    train = per_h[horizons[0]]["y_train"]
    # pick a representative results object (for residual diagnostics)
    res = per_h[horizons[0]].get("res", None)

    return metrics_df, F, L, U, test_eval, train, res, horizons
