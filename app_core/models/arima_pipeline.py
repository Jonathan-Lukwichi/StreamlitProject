import warnings
warnings.filterwarnings("ignore")

# ===== Standard =====
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# ===== Statsmodels / Metrics =====
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== Plotly =====
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== Optional selector hook (from your app) =====
try:
    # expected to return an object with .best_single.order OR .best_per_h[1].order
    from app_core.models.model_selection import select_orders_via_pmdarima
except Exception:
    select_orders_via_pmdarima = None


# =========================
# Styling helpers (unchanged)
# =========================
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
WARNING_COLOR = "#f59e0b"
SUCCESS_COLOR = "#10b981"
DANGER_COLOR = "#ef4444"
CARD_BG_LIGHT = "#ffffff"
TEXT_COLOR = "#2c3e50"
GRID_COLOR = "#e5e7eb"

def add_grid(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR))
    fig.update_layout(plot_bgcolor=CARD_BG_LIGHT, paper_bgcolor=CARD_BG_LIGHT,
                      font=dict(family="Segoe UI, sans-serif", size=12, color=TEXT_COLOR))
    return fig


# ===========================================
# pmdarima-based order selection (optional)
# ===========================================
def optimize_arima_order_with_pmdarima(
    series: pd.Series,
    cv_strategy: str = "expanding",
    train_ratio: float = 0.8,
    n_folds: int = 3,
    max_candidates: int = 20,
    weights: Optional[Dict[str, float]] = None,  # e.g. {"MAPE":0.5,"RMSE":0.25,"MAE":0.15,"-Accuracy":0.10}
    lam_complexity: float = 0.0,
    mu_stability: float = 0.0,
    eps: float = 1e-3,
    horizon_policy: str = "single_spec",
) -> Tuple[int, int, int]:
    """
    Uses your app's pmdarima-backed selector if available.
    Falls back to (1,1,1) if not.
    """
    if select_orders_via_pmdarima is None:
        return (1, 1, 1)

    sel = select_orders_via_pmdarima(
        series=series,
        exog=None,
        seasonal=False,
        m=1,
        horizons=[1],
        cv_strategy=cv_strategy,
        n_folds=n_folds,
        train_ratio=train_ratio,
        max_candidates=max_candidates,
        weights=weights,
        lam_complexity=lam_complexity,
        mu_stability=mu_stability,
        eps=eps,
        horizon_policy=horizon_policy,
    )
    if getattr(sel, "best_single", None):
        return sel.best_single.order
    if getattr(sel, "best_per_h", None) and 1 in sel.best_per_h:
        return sel.best_per_h[1].order
    return (1, 1, 1)


# ===========================================
# Hybrid AIC + CV-RMSE optimization
# ===========================================
def optimize_arima_order_hybrid(
    y_train: np.ndarray,
    n_folds: int = 3,
    alpha: float = 0.3,  # AIC weight
    beta: float = 0.7,   # CV-RMSE weight
) -> Tuple[int, int, int]:
    """
    Find optimal ARIMA order by minimizing weighted composite score:
    Score = alpha * Normalized_AIC + beta * Normalized_CV_RMSE

    Uses expanding window cross-validation for robust RMSE estimation.

    Args:
        y_train: Training data array
        n_folds: Number of CV folds (default 3)
        alpha: Weight for AIC (complexity penalty), default 0.3
        beta: Weight for CV-RMSE (prediction accuracy), default 0.7

    Returns:
        Tuple (p, d, q) with optimal ARIMA order
    """
    # Define search grid
    common = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (1, 0, 1)]
    grid = [(p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3)]
    candidates = common + grid

    results = []

    for order in candidates:
        try:
            # 1. Fit full training data to get AIC
            model_full = ARIMA(y_train, order=order).fit()
            aic = float(model_full.aic)

            # 2. Time Series Cross-Validation for RMSE
            cv_rmses = []
            fold_size = max(12, len(y_train) // (n_folds + 1))

            for i in range(n_folds):
                # Expanding window
                train_end = fold_size * (i + 2)
                if train_end > len(y_train):
                    break

                test_start = train_end
                test_end = min(train_end + fold_size, len(y_train))

                if test_end <= test_start or test_start >= len(y_train):
                    continue

                y_cv_train = y_train[:train_end]
                y_cv_test = y_train[test_start:test_end]

                if len(y_cv_train) < 10 or len(y_cv_test) == 0:
                    continue

                # Fit and forecast
                cv_model = ARIMA(y_cv_train, order=order).fit()
                cv_forecast = cv_model.forecast(steps=len(y_cv_test))

                # Calculate RMSE for this fold
                rmse_fold = float(np.sqrt(mean_squared_error(y_cv_test, cv_forecast)))
                cv_rmses.append(rmse_fold)

            # Average CV-RMSE across folds
            avg_cv_rmse = float(np.mean(cv_rmses)) if cv_rmses else float('inf')

            if np.isfinite(aic) and np.isfinite(avg_cv_rmse):
                results.append({
                    'order': order,
                    'aic': aic,
                    'cv_rmse': avg_cv_rmse,
                })

        except Exception:
            continue

    if not results:
        return (1, 1, 1)

    # Normalize scores to [0, 1] range
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

    # Select best order
    best_idx = df['composite_score'].idxmin()
    best_order = df.loc[best_idx, 'order']

    return tuple(best_order)


# ===========================================
# Multi-horizon metrics
# ===========================================
def calculate_multi_horizon_metrics(F: np.ndarray, test_eval: pd.DataFrame, L: np.ndarray, U: np.ndarray) -> pd.DataFrame:
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    H = F.shape[1]
    results = []

    for h in range(H):
        h_idx = h + 1
        actual = test_eval[f"Target_{h_idx}"].values
        pred = F[:, h]
        l95 = L[:, h]
        u95 = U[:, h]

        valid_idx = np.isfinite(actual) & np.isfinite(pred)
        actual_v = actual[valid_idx]
        pred_v = pred[valid_idx]
        l95_v = l95[valid_idx]
        u95_v = u95[valid_idx]

        if actual_v.size == 0:
            continue

        mae_h = mean_absolute_error(actual_v, pred_v)
        rmse_h = np.sqrt(mean_squared_error(actual_v, pred_v))

        with np.errstate(divide="ignore", invalid="ignore"):
            mape_h = np.nanmean(np.abs((pred_v - actual_v) / np.where(actual_v != 0, actual_v, np.nan))) * 100
        acc_h = 100.0 * (1 - mape_h / 100.0) if np.isfinite(mape_h) else np.nan

        try:
            r2_h = r2_score(actual_v, pred_v)
        except ValueError:
            r2_h = np.nan

        coverage = np.mean((actual_v >= l95_v) & (actual_v <= u95_v)) * 100

        direction_acc = np.nan
        if actual_v.size > 1:
            da = np.diff(actual_v)
            dp = np.diff(pred_v)
            mask = da != 0
            if np.any(mask):
                direction_acc = np.mean(np.sign(da[mask]) == np.sign(dp[mask])) * 100

        results.append({
            "Horizon": f"h={h_idx}",
            "MAE": mae_h,
            "RMSE": rmse_h,
            "MAPE_%": mape_h,
            "Accuracy_%": acc_h,
            "R2": r2_h,
            "CI_Coverage_%": coverage,
            "Direction_Accuracy_%": direction_acc,
        })

    metrics_df = pd.DataFrame(results)
    if not metrics_df.empty:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        avg_row = {k: metrics_df[k].mean() for k in numeric_cols}
        avg_row["Horizon"] = "Average"
        metrics_df = pd.concat([metrics_df, pd.Series(avg_row).to_frame().T], ignore_index=True)

    return metrics_df.reset_index(drop=True)


# ===========================================
# Single-horizon ARIMA pipeline (h=1)
# ===========================================
def run_arima_pipeline(
    df: pd.DataFrame,
    order: Optional[Tuple[int, int, int]] = None,
    train_ratio: float = 0.8,
    auto_select: bool = True,
    target_col: str = "Target_1",
    # NEW selector knobs:
    search_mode: str = "aic_only",          # "aic_only" (default) or "pmdarima_oos"
    cv_strategy: str = "expanding",
    n_folds: int = 3,
    max_candidates: int = 20,
    weights: Optional[Dict[str, float]] = None,   # {"MAPE":0.5,"RMSE":0.25,"MAE":0.15,"-Accuracy":0.10}
    lam_complexity: float = 0.0,
    mu_stability: float = 0.0,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    """
    Single-horizon ARIMA pipeline with optional pmdarima-based order selection,
    stationarity tests, diagnostics and rich metrics.
    """

    # ---------------- Prepare index/target ----------------
    if "Date" in df.columns:
        s = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index(s).sort_index()
        df = df.drop(columns=["Date"])

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in dataframe columns: {list(df.columns)}")

    y_series = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    y = y_series.values
    dates = y_series.index

    if len(y) < 12:
        raise ValueError("Not enough data points to fit ARIMA (need >= 12).")

    train_size = int(len(y) * train_ratio)
    if train_size <= 1 or train_size >= len(y):
        raise ValueError("Invalid train_ratio produced an empty train/test split.")

    y_train, y_test = y[:train_size], y[train_size:]
    train_dates = dates[:train_size]
    test_dates  = dates[train_size:]

    # ---------------- Stationarity tests ----------------
    stationarity: Dict[str, Any] = {"adf": {}, "kpss": {}}
    try:
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(y_train, autolag="AIC", regression="c")
        stationarity["adf"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "critical_values": adf_crit,
            "interpretation": "Stationary" if adf_p < 0.05 else "Non-stationary",
        }
    except Exception as e:
        stationarity["adf"] = {"error": str(e)}

    try:
        kpss_stat, kpss_p, _, kpss_crit = kpss(y_train, regression="c", nlags="auto")
        stationarity["kpss"] = {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "critical_values": kpss_crit,
            "interpretation": "Stationary" if kpss_p > 0.05 else "Non-stationary",
        }
    except Exception as e:
        stationarity["kpss"] = {"error": str(e)}

    # ---------------- Order selection ----------------
    results = None

    # Path A-Hybrid: AIC + CV-RMSE optimization
    if search_mode == "hybrid" and order is None:
        try:
            order = optimize_arima_order_hybrid(
                y_train=y_train,
                n_folds=n_folds,
                alpha=0.3,  # AIC weight
                beta=0.7,   # CV-RMSE weight
            )
            auto_select = False
        except Exception:
            # will fall back to AIC mini-grid
            pass

    # Path A: pmdarima-backed selector
    if search_mode == "pmdarima_oos" and order is None:
        try:
            order = optimize_arima_order_with_pmdarima(
                series=pd.Series(y, index=dates),
                cv_strategy=cv_strategy,
                train_ratio=train_ratio,
                n_folds=n_folds,
                max_candidates=max_candidates,
                weights=weights,
                lam_complexity=lam_complexity,
                mu_stability=mu_stability,
                eps=eps,
                horizon_policy="single_spec",
            )
            auto_select = False
        except Exception:
            # will fall back to AIC mini-grid
            pass

    # Path B: your AIC mini-grid (if still needed)
    if auto_select and order is None:
        best_aic = np.inf
        best_order = None
        common = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (1, 0, 1)]
        grid = [(p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3)]
        for cand in (common + grid):
            try:
                r = ARIMA(y_train, order=cand).fit()
                if r.aic < best_aic:
                    best_aic = r.aic
                    best_order = cand
                    results = r
            except Exception:
                continue
        order = best_order if best_order is not None else (1, 1, 1)

    # If we still don't have results, fit with chosen/provided order
    if results is None:
        if order is None:
            order = (1, 1, 1)
        results = ARIMA(y_train, order=order).fit()

    # ---------------- Forecast & CI ----------------
    # forecast over the entire test portion
    fc_obj = results.get_forecast(steps=len(y_test))
    forecast = fc_obj.predicted_mean
    ci = fc_obj.conf_int(alpha=0.05)

    # conf_int can be DataFrame or ndarray; standardize
    if isinstance(ci, pd.DataFrame):
        ci_lower = ci.iloc[:, 0].values
        ci_upper = ci.iloc[:, 1].values
    else:
        ci = np.asarray(ci)
        ci_lower = ci[:, 0]
        ci_upper = ci[:, 1]

    # ---------------- Metrics ----------------
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_test - forecast) / np.where(y_test != 0, y_test, np.nan))) * 100
        if not np.isfinite(mape):
            mape = float("inf")
    accuracy = 100.0 - mape if np.isfinite(mape) else 0.0

    try:
        r2 = r2_score(y_test, forecast)
    except Exception:
        r2 = np.nan

    bias = float(np.mean(forecast - y_test))

    # CI coverage (align lengths)
    ytest_v = np.asarray(y_test)
    coverage = np.mean((ytest_v >= ci_lower) & (ytest_v <= ci_upper)) * 100 if len(ytest_v) == len(ci_lower) else np.nan

    if len(y_test) > 1:
        direction_accuracy = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(forecast))) * 100
    else:
        direction_accuracy = np.nan

    abs_err = np.abs(forecast - y_test)
    within_2 = np.mean(abs_err <= 2) * 100
    within_5 = np.mean(abs_err <= 5) * 100

    # ---------------- Residual diagnostics ----------------
    resid_series = pd.Series(results.resid).dropna()
    try:
        ljung_box = acorr_ljungbox(resid_series, lags=[10, 20], return_df=True)
    except Exception:
        ljung_box = None

    # ---------------- Return payload ----------------
    return {
        "y_train": pd.Series(y_train, index=train_dates),
        "y_test": pd.Series(y_test, index=test_dates),
        "forecasts": pd.Series(forecast, index=test_dates),
        "forecast_lower": pd.Series(ci_lower, index=test_dates),
        "forecast_upper": pd.Series(ci_upper, index=test_dates),
        "train_metrics": {
            "AIC": float(results.aic),
            "BIC": float(results.bic),
            "Log-Likelihood": float(results.llf),
            "HQIC": float(results.hqic),
        },
        "test_metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "Accuracy": float(accuracy),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Bias": float(bias),
            "CI_Coverage_%": float(coverage) if np.isfinite(coverage) else np.nan,
            "Direction_Accuracy_%": float(direction_accuracy) if np.isfinite(direction_accuracy) else np.nan,
            "Within_±2_%": float(within_2),
            "Within_±5_%": float(within_5),
        },
        "model_info": {
            "order": tuple(order),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "llf": float(results.llf),
            "hqic": float(results.hqic),
            "params": results.params.to_dict() if hasattr(results.params, "to_dict") else {},
        },
        "stationarity": stationarity,
        "residuals": resid_series,
        "ljung_box": ljung_box,
        "fitted_values": results.fittedvalues if hasattr(results, "fittedvalues") else None,
    }


# ===========================================
# Expanding multi-horizon (H-step) pipeline
# ===========================================
def run_arima_multi_horizon_pipeline(
    df: pd.DataFrame,
    order: Optional[Tuple[int, int, int]] = None,
    train_ratio: float = 0.8,
    max_horizon: int = 7,
    auto_select: bool = False,
    target_col: str = "Target_1",
    # allow same selector in MH too (optional)
    search_mode: str = "aic_only",
    cv_strategy: str = "expanding",
    n_folds: int = 3,
    max_candidates: int = 20,
    weights: Optional[Dict[str, float]] = None,
    lam_complexity: float = 0.0,
    mu_stability: float = 0.0,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    """
    Expanding-window multi-horizon forecaster that builds an F matrix (Time x H).
    """

    # Index setup
    if "Date" in df.columns:
        s = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index(s).sort_index()
        df = df.drop(columns=["Date"])

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in dataframe columns: {list(df.columns)}")

    y_series = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    y = y_series.values
    dates = y_series.index

    train_size = int(len(y) * train_ratio)
    if len(y) < max(train_size + max_horizon, 12):
        # Try to salvage H
        H_eff = max(1, len(y) - train_size)
        if H_eff < 1:
            return {"status": "error", "message": "Insufficient data for multi-horizon evaluation."}
        max_horizon = H_eff

    # Optional global order selection (once)
    if search_mode == "pmdarima_oos" and order is None:
        try:
            order = optimize_arima_order_with_pmdarima(
                series=pd.Series(y, index=dates),
                cv_strategy=cv_strategy,
                train_ratio=train_ratio,
                n_folds=n_folds,
                max_candidates=max_candidates,
                weights=weights,
                lam_complexity=lam_complexity,
                mu_stability=mu_stability,
                eps=eps,
                horizon_policy="single_spec",
            )
            auto_select = False
        except Exception:
            pass

    if auto_select and order is None:
        # small AIC mini-grid on initial train
        best_aic = np.inf
        best_order = None
        y0 = y[:train_size]
        common = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (1, 0, 1)]
        grid = [(p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3)]
        for cand in (common + grid):
            try:
                r = ARIMA(y0, order=cand).fit()
                if r.aic < best_aic:
                    best_aic = r.aic
                    best_order = cand
            except Exception:
                continue
        order = best_order if best_order is not None else (1, 1, 1)

    if order is None:
        order = (1, 1, 1)

    # Build F, L, U
    H = int(max_horizon)
    forecast_start_index = train_size
    forecast_end_index = len(y) - H + 1
    n_test_points = max(0, forecast_end_index - forecast_start_index)

    F = np.full((n_test_points, H), np.nan)
    L = np.full((n_test_points, H), np.nan)
    U = np.full((n_test_points, H), np.nan)

    eval_dates = dates[forecast_start_index: forecast_end_index]

    for t in range(n_test_points):
        current_train = y[:train_size + t]
        try:
            r = ARIMA(current_train, order=order).fit()
        except Exception as e:
            warnings.warn(f"ARIMA refit failed at t={t}: {e}")
            continue

        fc_obj = r.get_forecast(steps=H)
        F[t, :] = fc_obj.predicted_mean

        ci = fc_obj.conf_int(alpha=0.05)
        if isinstance(ci, pd.DataFrame):
            L[t, :] = ci.iloc[:, 0].values
            U[t, :] = ci.iloc[:, 1].values
        else:
            ci = np.asarray(ci)
            L[t, :] = ci[:, 0]
            U[t, :] = ci[:, 1]

    # Actuals matrix-like (Target_h columns)
    actuals = {}
    for h in range(1, H + 1):
        actuals[f"Target_{h}"] = y[forecast_start_index + h - 1: forecast_end_index + h - 1]
    test_eval = pd.DataFrame(actuals, index=eval_dates)

    metrics_df = calculate_multi_horizon_metrics(F, test_eval, L, U)

    # A final fit on the original training split for diagnostics panel
    final_res = ARIMA(y[:train_size], order=order).fit()

    return {
        "F": F,
        "L": L,
        "U": U,
        "test_eval": test_eval,
        "metrics_df": metrics_df,
        "train": df.iloc[:train_size].copy(),
        "res": final_res,
        "eval_dates": eval_dates,
        "order": tuple(order),
        "max_horizon": H,
    }


# ===========================================
# Plotting (unchanged from your style)
# ===========================================
def plot_multi_horizon_metrics_dashboard(metrics_df: pd.DataFrame) -> go.Figure:
    m = metrics_df[metrics_df["Horizon"] != "Average"].copy()
    if m.empty:
        return go.Figure().update_layout(title_text="No multi-horizon metrics available")
    horizons = m["Horizon"].str.replace("h=", "").astype(int)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy (%) by Horizon", "MAE by Horizon", "RMSE by Horizon", "MAPE (%) by Horizon")
    )

    fig.add_trace(go.Bar(x=horizons, y=m["Accuracy_%"], name="Accuracy", marker_color=SUCCESS_COLOR), row=1, col=1)
    fig.add_trace(go.Bar(x=horizons, y=m["MAE"], name="MAE", marker_color=PRIMARY_COLOR), row=1, col=2)
    fig.add_trace(go.Bar(x=horizons, y=m["RMSE"], name="RMSE", marker_color=SECONDARY_COLOR), row=2, col=1)
    fig.add_trace(go.Scatter(x=horizons, y=m["MAPE_%"], mode="lines+markers", name="MAPE",
                             line=dict(color=DANGER_COLOR)), row=2, col=2)

    fig.update_layout(title_text="Multi-Horizon Results — Accuracy & Error Metrics", height=700, showlegend=False)
    fig.update_yaxes(range=[0, 100], row=1, col=1, title_text="Accuracy (%)")
    fig.update_yaxes(title_text="Error Value", row=1, col=2)
    fig.update_yaxes(title_text="Error Value", row=2, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=2)
    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Horizon (h)", row=r, col=c)

    return add_grid(fig)


def plot_all_horizons_overview(F: np.ndarray, L: np.ndarray, U: np.ndarray,
                               test_eval: pd.DataFrame, metrics_df: pd.DataFrame,
                               eval_dates: pd.DatetimeIndex) -> go.Figure:
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    H = F.shape[1]
    rows = int(np.ceil(H / 2))

    fig = make_subplots(rows=rows, cols=2, horizontal_spacing=0.05, vertical_spacing=0.1)

    for h in range(H):
        h_idx = h + 1
        r = (h // 2) + 1
        c = (h % 2) + 1

        actual = test_eval[f"Target_{h_idx}"]
        pred = F[:, h]
        l95 = L[:, h]
        u95 = U[:, h]

        # label
        try:
            mrow = metrics_df[metrics_df["Horizon"] == f"h={h_idx}"].iloc[0]
            mae_h = mrow["MAE"]
            acc_h = mrow["Accuracy_%"]
            title_txt = f"h={h_idx} | MAE={mae_h:.2f} | Acc={acc_h:.1f}%"
        except Exception:
            title_txt = f"h={h_idx}"

        fig.add_trace(go.Scatter(x=eval_dates, y=actual, mode="lines",
                                 name="Actual", line=dict(color="black", width=1.2),
                                 showlegend=(h == 0)), row=r, col=c)

        fig.add_trace(go.Scatter(x=eval_dates, y=pred, mode="lines",
                                 name=f"Forecast h={h_idx}",
                                 line=dict(color=PRIMARY_COLOR, width=1.2, dash="dot"),
                                 showlegend=(h == 0)), row=r, col=c)

        fig.add_trace(go.Scatter(x=eval_dates, y=u95, mode="lines", line=dict(width=0),
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=eval_dates, y=l95, mode="lines", fill="tonexty",
                                 fillcolor="rgba(102, 126, 234, 0.15)", line=dict(width=0),
                                 name="95% CI", showlegend=(h == 0)), row=r, col=c)

        fig.update_annotations([dict(xref="x domain", yref="y domain", x=0.5, y=1.08,
                                     text=title_txt, showarrow=False, row=r, col=c,
                                     font=dict(size=11))])

        fig.update_xaxes(title_text="Date", row=r, col=c)
        fig.update_yaxes(title_text="Value", row=r, col=c)

    fig.update_layout(title_text=f"All Horizons ({H} Steps) — Actual vs Forecast",
                      height=rows * 400, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return add_grid(fig)


# ===========================================
# Convenience tables & diagnostics plots
# ===========================================
def create_arima_metrics_table(arima_results: dict) -> pd.DataFrame:
    if not arima_results:
        return pd.DataFrame()

    # If multi-horizon payload
    if "metrics_df" in arima_results:
        metrics_df = arima_results["metrics_df"]
        if metrics_df.empty:
            return pd.DataFrame()
        avg_row = metrics_df[metrics_df["Horizon"] == "Average"]
        rows = []
        rows.append({"Metric": "Model Order (Initial Train)", "Value": str(arima_results.get("order", ""))})
        rows.append({"Metric": "Max Horizon (H)", "Value": str(arima_results.get("max_horizon", ""))})
        if not avg_row.empty:
            avg_row = avg_row.iloc[0]
            metric_formats = {
                "MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE_%": "{:.2f}%",
                "Accuracy_%": "{:.1f}%", "R2": "{:.3f}", "CI_Coverage_%": "{:.1f}%",
                "Direction_Accuracy_%": "{:.1f}%"
            }
            for k, fmt in metric_formats.items():
                if k in avg_row:
                    rows.append({"Metric": f"Avg. {k}", "Value": fmt.format(avg_row[k])})
        return pd.DataFrame(rows)

    # Single-horizon payload
    train_m = arima_results.get("train_metrics", {})
    test_m = arima_results.get("test_metrics", {})
    model_info = arima_results.get("model_info", {})

    rows = []
    if model_info:
        rows += [
            {"Metric": "Model Order", "Value": str(model_info.get("order", ""))},
            {"Metric": "AIC", "Value": float(model_info.get("aic", float("nan")))},
            {"Metric": "BIC", "Value": float(model_info.get("bic", float("nan")))},
            {"Metric": "Log-Likelihood", "Value": float(model_info.get("llf", float("nan")))},
            {"Metric": "HQIC", "Value": float(model_info.get("hqic", float("nan")))},
        ]

    rows.append({"Metric": "Train AIC", "Value": train_m.get("AIC")})
    rows.append({"Metric": "Train BIC", "Value": train_m.get("BIC")})

    metric_formats = {
        "MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%",
        "Accuracy": "{:.1f}%", "R2": "{:.3f}", "Bias": "{:.3f}",
        "CI_Coverage_%": "{:.1f}%", "Direction_Accuracy_%": "{:.1f}%",
        "Within_±2_%": "{:.1f}%", "Within_±5_%": "{:.1f}%"
    }
    for k, v in (test_m or {}).items():
        fmt = metric_formats.get(k, "{}")
        try:
            rows.append({"Metric": f"Test {k}", "Value": fmt.format(v)})
        except Exception:
            rows.append({"Metric": f"Test {k}", "Value": str(v)})

    return pd.DataFrame(rows)


def plot_arima_forecast(arima_results: dict, title: str = "ARIMA Forecast (Single-Step)") -> go.Figure:
    y_train = arima_results.get("y_train")
    y_test = arima_results.get("y_test")
    preds = arima_results.get("forecasts")
    lo = arima_results.get("forecast_lower")
    hi = arima_results.get("forecast_upper")

    fig = go.Figure()

    if y_train is not None and len(y_train) > 0:
        fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode="lines",
                                 name="Training Data", line=dict(color=PRIMARY_COLOR, width=2)))
    if y_test is not None and len(y_test) > 0:
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines+markers",
                                 name="Actual Test", line=dict(color=SECONDARY_COLOR, width=3),
                                 marker=dict(size=4)))
    if preds is not None and len(preds) > 0:
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, mode="lines",
                                 name="Forecast", line=dict(color=DANGER_COLOR, width=2.5, dash="dash")))
        if lo is not None and hi is not None:
            fig.add_trace(go.Scatter(x=hi.index, y=hi.values, mode="lines", line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=lo.index, y=lo.values, mode="lines", fill="tonexty",
                                     fillcolor="rgba(239, 68, 68, 0.15)", line=dict(width=0),
                                     name="95% CI"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=TEXT_COLOR)),
        xaxis_title="Date", yaxis_title="Value", hovermode="x unified",
        height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return add_grid(fig)


def plot_arima_residuals(arima_results: dict, title: str = "Residual Diagnostics") -> go.Figure:
    residuals = arima_results.get("residuals")
    fitted_values = arima_results.get("fitted_values")

    if residuals is None or len(residuals) == 0:
        return go.Figure().update_layout(title="No residuals available")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=["Residuals over Time", "Residuals Histogram",
                        "Autocorrelation (ACF)", "Partial Autocorrelation (PACF)",
                        "Q-Q Plot", "Residuals vs Fitted"],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.08, horizontal_spacing=0.08
    )

    # 1) Residuals over time
    x_ts = residuals.index if hasattr(residuals, "index") else list(range(len(residuals)))
    fig.add_trace(go.Scatter(x=x_ts, y=residuals, mode="lines+markers", name="Residuals",
                             marker=dict(size=3, color=WARNING_COLOR), line=dict(width=1)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=TEXT_COLOR, width=1, dash="dash"), row=1, col=1)

    # 2) Histogram
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color=SECONDARY_COLOR, name="Distribution", showlegend=False), row=1, col=2)

    # 3) ACF
    try:
        nlags = min(40, len(residuals) // 2)
        acf_vals, confint = acf(residuals, nlags=nlags, alpha=0.05, fft=True)
        lags = list(range(len(acf_vals)))
        fig.add_trace(go.Bar(x=lags, y=acf_vals, marker_color=PRIMARY_COLOR, name="ACF", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 1], mode="lines", line=dict(width=1, dash="dash", color='rgba(102, 126, 234, 0.5)'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 0], mode="lines", line=dict(width=1, dash="dash", color='rgba(102, 126, 234, 0.5)'), showlegend=False), row=2, col=1)
    except Exception:
        pass

    # 4) PACF
    try:
        pacf_vals, confint_pacf = pacf(residuals, nlags=nlags, alpha=0.05, method="ywm")
        lags = list(range(len(pacf_vals)))
        fig.add_trace(go.Bar(x=lags, y=pacf_vals, marker_color=SUCCESS_COLOR, name="PACF", showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=lags, y=confint_pacf[:, 1], mode="lines", line=dict(width=1, dash="dash", color='rgba(16, 185, 129, 0.5)'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=lags, y=confint_pacf[:, 0], mode="lines", line=dict(width=1, dash="dash", color='rgba(16, 185, 129, 0.5)'), showlegend=False), row=2, col=2)
    except Exception:
        pass

    # 5) Q-Q Plot
    try:
        from scipy import stats
        qq = stats.probplot(residuals.values, dist="norm")
        x = qq[0][0]; y = qq[0][1]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=WARNING_COLOR, size=4), showlegend=False), row=3, col=1)
        slope, intercept, *_ = stats.linregress(x, y)
        line_x = np.array([np.min(x), np.max(x)])
        line_y = intercept + slope * line_x
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color=DANGER_COLOR, width=2), showlegend=False), row=3, col=1)
    except Exception:
        pass

    # 6) Residuals vs Fitted
    if fitted_values is not None and len(fitted_values) == len(residuals):
        fig.add_trace(go.Scatter(x=fitted_values, y=residuals, mode="markers",
                                 marker=dict(color=PRIMARY_COLOR, size=4, opacity=0.6), showlegend=False), row=3, col=2)
        fig.add_hline(y=0, line=dict(color=TEXT_COLOR, width=1, dash="dash"), row=3, col=2)

    fig.update_layout(
        height=900, title=dict(text=title, font=dict(size=20, color=TEXT_COLOR)),
        showlegend=False, margin=dict(t=80, b=40, l=40, r=40)
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="PACF", row=2, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=3, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=3, col=1)
    fig.update_xaxes(title_text="Fitted Values", row=3, col=2)
    fig.update_yaxes(title_text="Residuals", row=3, col=2)
    return add_grid(fig)


def plot_stationarity_tests(arima_results: dict) -> go.Figure:
    stationarity = arima_results.get("stationarity", {})
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ADF Test Results", "KPSS Test Results"],
                        specs=[[{"type": "indicator"}, {"type": "indicator"}]])

    adf_data = stationarity.get("adf", {})
    if "statistic" in adf_data:
        fig.add_trace(go.Indicator(
            mode="number",
            value=float(adf_data["statistic"]),
            number={"suffix": ""},
            title={"text": f"ADF<br><span style='font-size:0.8em'>{adf_data.get('interpretation','')}</span>"},
        ), row=1, col=1)

    kpss_data = stationarity.get("kpss", {})
    if "statistic" in kpss_data:
        fig.add_trace(go.Indicator(
            mode="number",
            value=float(kpss_data["statistic"]),
            number={"suffix": ""},
            title={"text": f"KPSS<br><span style='font-size:0.8em'>{kpss_data.get('interpretation','')}</span>"},
        ), row=1, col=2)

    fig.update_layout(height=300, margin=dict(t=80, b=40, l=40, r=40))
    return add_grid(fig)


# ===========================================
# Multi-Target ARIMA Pipeline
# ===========================================
def run_arima_multi_target_pipeline(
    df: pd.DataFrame,
    target_columns: List[str],
    order: Optional[Tuple[int, int, int]] = None,
    train_ratio: float = 0.8,
    auto_select: bool = True,
    search_mode: str = "aic_only",
    cv_strategy: str = "expanding",
    n_folds: int = 3,
    max_candidates: int = 20,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run ARIMA pipeline for multiple target columns (patient arrivals + reasons).

    Args:
        df: DataFrame with date and target columns
        target_columns: List of target column names to forecast
            e.g., ["patient_count", "asthma", "pneumonia", "chest_pain", ...]
        order: ARIMA order (p,d,q) - if None, auto-select for each target
        train_ratio: Train/test split ratio
        auto_select: Whether to auto-select order for each target
        search_mode: Order search mode ("aic_only", "pmdarima_oos", "hybrid")
        cv_strategy: Cross-validation strategy
        n_folds: Number of CV folds
        max_candidates: Max candidates for order search
        progress_callback: Optional callback function(target_name, idx, total)

    Returns:
        Dictionary with results for each target:
        {
            "target_name": {
                "results": arima_results_dict,
                "order": (p, d, q),
                "metrics": {...}
            },
            ...
            "summary": pd.DataFrame with all targets' metrics
        }
    """
    all_results = {}
    metrics_summary = []

    total_targets = len(target_columns)

    for idx, target_col in enumerate(target_columns):
        if progress_callback:
            progress_callback(target_col, idx, total_targets)

        # Check if target column exists in dataframe
        if target_col not in df.columns:
            all_results[target_col] = {
                "status": "error",
                "message": f"Column '{target_col}' not found in dataframe"
            }
            continue

        # Check if column has enough non-null values
        valid_count = df[target_col].dropna().shape[0]
        if valid_count < 12:
            all_results[target_col] = {
                "status": "error",
                "message": f"Insufficient data for '{target_col}' (need >= 12 points, got {valid_count})"
            }
            continue

        try:
            # Run single-target ARIMA pipeline
            result = run_arima_pipeline(
                df=df.copy(),
                order=order,
                train_ratio=train_ratio,
                auto_select=auto_select,
                target_col=target_col,
                search_mode=search_mode,
                cv_strategy=cv_strategy,
                n_folds=n_folds,
                max_candidates=max_candidates,
            )

            # Store results
            all_results[target_col] = {
                "status": "success",
                "results": result,
                "order": result.get("model_info", {}).get("order", (1, 1, 1)),
            }

            # Extract metrics for summary
            test_metrics = result.get("test_metrics", {})
            metrics_summary.append({
                "Target": target_col,
                "Order": str(result.get("model_info", {}).get("order", "N/A")),
                "MAE": test_metrics.get("MAE", np.nan),
                "RMSE": test_metrics.get("RMSE", np.nan),
                "MAPE_%": test_metrics.get("MAPE", np.nan),
                "Accuracy_%": test_metrics.get("Accuracy", np.nan),
                "R2": test_metrics.get("R2", np.nan),
                "Direction_Acc_%": test_metrics.get("Direction_Accuracy_%", np.nan),
            })

        except Exception as e:
            all_results[target_col] = {
                "status": "error",
                "message": str(e)
            }
            metrics_summary.append({
                "Target": target_col,
                "Order": "Error",
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "Accuracy_%": np.nan,
                "R2": np.nan,
                "Direction_Acc_%": np.nan,
            })

    # Create summary dataframe
    summary_df = pd.DataFrame(metrics_summary)

    # Add average row (excluding errors)
    if not summary_df.empty:
        numeric_cols = ["MAE", "RMSE", "MAPE_%", "Accuracy_%", "R2", "Direction_Acc_%"]
        avg_row = {"Target": "AVERAGE", "Order": "-"}
        for col in numeric_cols:
            avg_row[col] = summary_df[col].mean()
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    all_results["summary"] = summary_df
    all_results["successful_targets"] = [t for t, r in all_results.items()
                                          if isinstance(r, dict) and r.get("status") == "success"]
    all_results["failed_targets"] = [t for t, r in all_results.items()
                                      if isinstance(r, dict) and r.get("status") == "error"]

    return all_results


def get_reason_target_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect all reason/target columns in the dataframe for multi-target forecasting.

    Returns list of column names that can be used as forecast targets:
    - patient_count or similar arrival columns
    - Medical reason columns (asthma, pneumonia, chest_pain, etc.)
    """
    # Known reason column names
    reason_keywords = [
        'patient_count', 'ed_visits', 'arrivals', 'visits',
        'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
        'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
        'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
        'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
    ]

    # Find matching columns (case-insensitive)
    target_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column matches any reason keyword
        if any(keyword in col_lower for keyword in reason_keywords):
            # Exclude lag columns and future target columns
            if '_lag_' not in col_lower and not col_lower.startswith('target_') and not col_lower.startswith('ed_'):
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    target_cols.append(col)

    return target_cols
