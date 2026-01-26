# app_core/models/sarimax_pipeline.py
# Thesis-grade SARIMAX with exogenous features, multi-horizon outputs, and ARIMA-compatible artifacts

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# ===========================================
# Scenario-1 Dataclasses (Thesis Methodology)
# ===========================================

@dataclass
class Scenario1DifferencingResult:
    """
    Step 4.3: Differencing test results from statistical tests.

    - ADF (Augmented Dickey-Fuller) test determines non-seasonal differencing order (d)
    - Canova-Hansen test determines seasonal differencing order (D)
    """
    adf_statistic: float
    adf_pvalue: float
    adf_d: int  # Recommended d value from ADF test
    ch_statistic: float
    ch_pvalue: float
    ch_D: int  # Recommended D value from Canova-Hansen test


@dataclass
class Scenario1DiagnosticResult:
    """
    Step 4.4: Residual diagnostic results for white noise and normality checks.

    - Ljung-Box test checks for autocorrelation in residuals (white noise)
    - Jarque-Bera test checks for normality of residuals
    """
    ljung_box_statistic: float
    ljung_box_pvalue: float
    ljung_box_interpretation: str  # "White noise" or "Autocorrelation present"
    normality_statistic: float
    normality_pvalue: float
    normality_interpretation: str  # "Normal" or "Non-normal"
    residual_mean: float
    residual_std: float
    residual_skewness: float
    residual_kurtosis: float


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

    Returns DataFrame with RangeIndex and all float64 columns.

    Args:
        index: DatetimeIndex or Series of datetime values
        include_dow_ohe: Whether to include day-of-week one-hot encoding
    """
    # Handle both DatetimeIndex and Series inputs
    if isinstance(index, pd.Series):
        dt_index = pd.DatetimeIndex(index)
    else:
        dt_index = index

    dow = dt_index.dayofweek.values

    # Build features as numpy arrays first (guaranteed numeric)
    features = {}

    if include_dow_ohe:
        for i, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
            features[f"is_{name}"] = (dow == i).astype(np.float64)

    features["is_weekend"] = (dow >= 5).astype(np.float64)

    t = dt_index.dayofyear.values.astype(np.float64)
    features["sin_y"] = np.sin(2*np.pi*t/365.25)
    features["cos_y"] = np.cos(2*np.pi*t/365.25)

    # Create DataFrame with RangeIndex (NOT DatetimeIndex)
    X = pd.DataFrame(features)

    return X

def _exog_all(
    df_full: pd.DataFrame,
    date_col: str,
    target_cols: List[str],
    include_dow_ohe: bool,
    selected_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build exogenous feature matrix from ALL non-target, non-date columns.
    Returns DataFrame with RangeIndex and all float64 columns.

    IMPORTANT: ED lag columns (ED, ED_1, ED_2, etc.) are EXCLUDED because they
    are redundant with SARIMAX's AR component. Including them would cause:
    1. Multicollinearity (AR terms duplicate ED lags)
    2. Artificially inflated accuracy (same info provided twice)

    Only truly EXTERNAL features should be used as exogenous variables:
    - Calendar features (day-of-week, weekend, yearly sin/cos)
    - Weather features (temperature, humidity, etc.)
    - Holiday indicators
    """
    # Columns to exclude from exogenous features
    drop_cols = [date_col] + [c for c in target_cols if c in df_full.columns]

    # CRITICAL: Also drop ED lag columns - they duplicate SARIMAX AR component
    # This prevents artificially inflated accuracy from redundant features
    ed_lag_patterns = ['ED', 'ED_1', 'ED_2', 'ED_3', 'ED_4', 'ED_5', 'ED_6', 'ED_7',
                       'ed', 'ed_1', 'ed_2', 'ed_3', 'ed_4', 'ed_5', 'ed_6', 'ed_7']
    drop_cols.extend([c for c in df_full.columns if c in ed_lag_patterns])

    # CRITICAL DATA LEAKAGE FIX: Drop ALL future target columns
    # These are columns ending with _1, _2, ..., _7 that are NOT lag features
    # Examples: RESPIRATORY_1, CARDIAC_1, asthma_1, pneumonia_1 (all contain FUTURE values)
    # Lag features have "_lag_" in them (e.g., asthma_lag_1) and should be kept
    future_target_pattern = re.compile(r'^(.+)_([1-7])$')
    for col in df_full.columns:
        if col in drop_cols:
            continue
        match = future_target_pattern.match(col)
        if match:
            base_name = match.group(1)
            # Keep lag features (they have _lag in the name)
            if '_lag' in base_name.lower():
                continue
            # Drop all other _N columns - they are future targets
            drop_cols.append(col)

    X = df_full.drop(columns=drop_cols, errors="ignore").copy()

    # Convert ALL columns to float64, coercing errors to NaN
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)

    # Reset index to RangeIndex
    X = X.reset_index(drop=True)

    # Add calendar features (already has RangeIndex and float64)
    cal = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)

    # Combine - both have RangeIndex so no alignment issues
    X = pd.concat([X, cal], axis=1)

    # If user specified features, keep only those
    if selected_features:
        keep = [c for c in selected_features if c in X.columns]
        if include_dow_ohe:
            keep += [c for c in cal.columns if c not in keep]
        keep = list(dict.fromkeys(keep))
        if keep:
            X = X[keep]

    # Fill NaN and ensure float64
    X = X.fillna(0).astype(np.float64)

    return X


def _prepare_sarimax_input(y: pd.Series, X: Optional[pd.DataFrame]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert pandas Series/DataFrame to numpy arrays safe for SARIMAX.

    SARIMAX in statsmodels is sensitive to dtypes and requires clean float64 arrays.
    This function handles all edge cases including NaN values, mixed dtypes, and
    object columns that cause "Pandas data cast to numpy dtype of object" errors.

    Args:
        y: Target series (endog)
        X: Exogenous features DataFrame (can be None)

    Returns:
        Tuple of (endog_array, exog_array or None)
    """
    # Convert target to 1-D float64 array
    y_arr = np.asarray(y.values, dtype=np.float64).ravel()

    # Handle NaN in target - replace with mean
    if np.any(np.isnan(y_arr)):
        mean_val = np.nanmean(y_arr)
        if np.isnan(mean_val):
            mean_val = 0.0
        y_arr = np.nan_to_num(y_arr, nan=mean_val)

    # Convert exog to 2-D float64 array
    if X is not None and len(X.columns) > 0:
        X_arr = np.asarray(X.values, dtype=np.float64)

        # Handle NaN in exog - replace with column means
        if np.any(np.isnan(X_arr)):
            col_means = np.nanmean(X_arr, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            for j in range(X_arr.shape[1]):
                mask = np.isnan(X_arr[:, j])
                X_arr[mask, j] = col_means[j]

        return y_arr, X_arr

    return y_arr, None


# ===========================================
# Scenario-1 Helper Functions (Thesis Methodology)
# ===========================================

def _compute_differencing_orders(
    y_train: pd.Series,
    m: int = 7,
    adf_significance: float = 0.05,
) -> Scenario1DifferencingResult:
    """
    Step 4.3: Determine d and D using statistical tests.

    Uses:
    - ADF (Augmented Dickey-Fuller) test for non-seasonal differencing order (d)
    - Canova-Hansen test for seasonal differencing order (D)

    Args:
        y_train: Training target series
        m: Seasonal period (7 for weekly)
        adf_significance: Significance level for ADF test

    Returns:
        Scenario1DifferencingResult with recommended d and D values
    """
    y_arr = y_train.values.astype(float)

    # === ADF Test for d ===
    # H0: Series has a unit root (non-stationary)
    # If p < alpha, reject H0 -> series is stationary (d=0)
    # If p >= alpha, fail to reject H0 -> differencing needed
    try:
        adf_result = adfuller(y_arr, autolag="AIC", regression="c")
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
    except Exception:
        adf_stat = np.nan
        adf_p = np.nan

    # Use pmdarima's ndiffs for robust d estimation
    if PMDARIMA_AVAILABLE:
        try:
            from pmdarima.arima import ndiffs, nsdiffs
            d = ndiffs(y_arr, test='adf', max_d=2)
        except Exception:
            # Fallback: use p-value decision
            d = 0 if (np.isfinite(adf_p) and adf_p < adf_significance) else 1
    else:
        # Fallback without pmdarima
        d = 0 if (np.isfinite(adf_p) and adf_p < adf_significance) else 1

    # === Canova-Hansen Test for D ===
    # H0: Seasonal pattern is stable (D=0)
    # Uses pmdarima's nsdiffs with CH test
    ch_stat = np.nan
    ch_p = np.nan

    if PMDARIMA_AVAILABLE:
        try:
            from pmdarima.arima import nsdiffs
            D = nsdiffs(y_arr, m=m, test='ch', max_D=1)
        except Exception:
            D = 1  # Default to seasonal differencing for weekly data
    else:
        D = 1  # Default to seasonal differencing

    return Scenario1DifferencingResult(
        adf_statistic=adf_stat,
        adf_pvalue=adf_p,
        adf_d=int(d),
        ch_statistic=ch_stat,
        ch_pvalue=ch_p,
        ch_D=int(D),
    )


def _run_residual_diagnostics(
    residuals: np.ndarray,
    lags: int = 10,
) -> Scenario1DiagnosticResult:
    """
    Step 4.4: Residual diagnostics for white noise and normality.

    Tests:
    1. Ljung-Box test for autocorrelation (white noise check)
       - H0: Residuals are independently distributed (white noise)
       - If p > 0.05: residuals are white noise (GOOD)
       - If p < 0.05: autocorrelation present (model may be underfitting)

    2. Jarque-Bera normality test
       - H0: Residuals are normally distributed
       - If p > 0.05: residuals are normal (GOOD for CI validity)

    Args:
        residuals: Model residuals array
        lags: Number of lags for Ljung-Box test

    Returns:
        Scenario1DiagnosticResult with test results and interpretations
    """
    resid = np.asarray(residuals).flatten()
    resid = resid[~np.isnan(resid)]

    if len(resid) == 0:
        return Scenario1DiagnosticResult(
            ljung_box_statistic=np.nan,
            ljung_box_pvalue=np.nan,
            ljung_box_interpretation="Insufficient data",
            normality_statistic=np.nan,
            normality_pvalue=np.nan,
            normality_interpretation="Insufficient data",
            residual_mean=np.nan,
            residual_std=np.nan,
            residual_skewness=np.nan,
            residual_kurtosis=np.nan,
        )

    # === Ljung-Box Test ===
    try:
        lb_result = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_stat = float(lb_result['lb_stat'].values[0])
        lb_pval = float(lb_result['lb_pvalue'].values[0])
        lb_interp = "White noise (residuals uncorrelated)" if lb_pval > 0.05 else "Autocorrelation present (model may underfit)"
    except Exception:
        lb_stat, lb_pval = np.nan, np.nan
        lb_interp = "Test unavailable"

    # === Jarque-Bera Normality Test ===
    try:
        norm_stat, norm_pval = scipy_stats.jarque_bera(resid)
        norm_stat = float(norm_stat)
        norm_pval = float(norm_pval)
        norm_interp = "Normal distribution (CIs valid)" if norm_pval > 0.05 else "Non-normal distribution (CIs approximate)"
    except Exception:
        norm_stat, norm_pval = np.nan, np.nan
        norm_interp = "Test unavailable"

    # === Residual Statistics ===
    resid_mean = float(np.mean(resid))
    resid_std = float(np.std(resid))
    try:
        resid_skew = float(scipy_stats.skew(resid))
        resid_kurt = float(scipy_stats.kurtosis(resid))
    except Exception:
        resid_skew = np.nan
        resid_kurt = np.nan

    return Scenario1DiagnosticResult(
        ljung_box_statistic=lb_stat,
        ljung_box_pvalue=lb_pval,
        ljung_box_interpretation=lb_interp,
        normality_statistic=norm_stat,
        normality_pvalue=norm_pval,
        normality_interpretation=norm_interp,
        residual_mean=resid_mean,
        residual_std=resid_std,
        residual_skewness=resid_skew,
        residual_kurtosis=resid_kurt,
    )


def _scenario1_order_search(
    y_train: pd.Series,
    X_train: Optional[pd.DataFrame],
    d: int,
    D: int,
    m: int,
    max_p: int = 3,
    max_q: int = 3,
    max_P: int = 2,
    max_Q: int = 2,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float, float]:
    """
    Step 4.2: AIC-based order search with FIXED d and D from statistical tests.

    Key difference from standard auto_arima:
    - d and D are PRE-COMPUTED from statistical tests (Step 4.3)
    - auto_arima only searches p, q, P, Q

    Args:
        y_train: Training target series
        X_train: Training exogenous features
        d: Non-seasonal differencing order (from ADF test)
        D: Seasonal differencing order (from Canova-Hansen test)
        m: Seasonal period
        max_p, max_q: Non-seasonal AR/MA bounds
        max_P, max_Q: Seasonal AR/MA bounds

    Returns:
        Tuple (order, seasonal_order, aic, bic)
    """
    if not PMDARIMA_AVAILABLE:
        # Fallback if pmdarima not available
        return (1, d, 1), (1, D, 0, m), float("nan"), float("nan")

    X_arr = X_train.values.astype(np.float64) if X_train is not None and len(X_train.columns) > 0 else None

    try:
        model = pm.auto_arima(
            y_train.values.astype(float),
            X=X_arr,
            # Fix differencing orders from Step 4.3
            d=d,
            D=D,
            # Search p, q, P, Q
            start_p=0, max_p=max_p,
            start_q=0, max_q=max_q,
            start_P=0, max_P=max_P,
            start_Q=0, max_Q=max_Q,
            # Seasonal settings
            seasonal=True,
            m=m,
            # AIC-based selection (thesis requirement)
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            random_state=SEED,
            maxiter=100,
        )
        return model.order, model.seasonal_order, float(model.aic()), float(model.bic())
    except Exception:
        # Fallback to safe default with fixed d, D
        return (1, d, 1), (1, D, 0, m), float("nan"), float("nan")


def _prepare_scenario1_data(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    train_ratio: float,
    use_all_features: bool,
    include_dow_ohe: bool,
    selected_features: Optional[List[str]],
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Step 4.1: Prepare train/test split with exogenous features.

    Returns:
        y_train, y_test: Target series for train/test
        X_train, X_test: Exogenous features for train/test
        train_dates, test_dates: DatetimeIndex for each split
    """
    # Clean and regularize data
    df_full = _clean_daily(df, date_col)

    # Extract target series
    y_series = df_full.set_index(date_col)[target_col].astype(float)
    y_series = y_series.interpolate("linear").ffill().bfill()

    # Build exogenous features
    if use_all_features:
        X_all = _exog_all(
            df_full=df_full,
            date_col=date_col,
            target_cols=[target_col],
            include_dow_ohe=include_dow_ohe,
            selected_features=selected_features,
        )
    else:
        X_all = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)

    # 80/20 split (thesis requirement)
    n = len(y_series)
    split_idx = int(n * train_ratio)

    y_train = y_series.iloc[:split_idx]
    y_test = y_series.iloc[split_idx:]
    X_train = X_all.iloc[:split_idx].copy()
    X_test = X_all.iloc[split_idx:].copy()

    return y_train, y_test, X_train, X_test, y_train.index, y_test.index


def _scenario1_forecast_and_evaluate(
    fitted_model,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_horizon: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Step 4.5 & 4.6: Rolling window forecast with .predict(n_periods=H).

    Unlike expanding window (which refits), this uses a SINGLE fitted model
    and rolls the forecast origin forward through the test set.

    Args:
        fitted_model: Fitted SARIMAX model
        y_train: Training target series
        y_test: Test target series
        X_train: Training exogenous features
        X_test: Test exogenous features
        max_horizon: Maximum forecast horizon (default 7)

    Returns:
        F: Forecast matrix [n_origins x H]
        L: Lower CI matrix [n_origins x H]
        U: Upper CI matrix [n_origins x H]
        test_eval: DataFrame with actual values for each horizon
        metrics_df: Per-horizon metrics DataFrame
    """
    n_test = len(y_test)
    H = max_horizon

    # Number of forecast origins (need H points after each origin)
    n_origins = n_test - H + 1
    if n_origins <= 0:
        raise ValueError(f"Not enough test data ({n_test}) for {H}-step horizon forecast")

    # Initialize matrices
    F = np.full((n_origins, H), np.nan)  # Forecasts
    L = np.full((n_origins, H), np.nan)  # Lower CI
    U = np.full((n_origins, H), np.nan)  # Upper CI

    # Training mean/std for fallback
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())

    # Rolling window forecast
    for t in range(n_origins):
        # Get exogenous features for forecast period
        future_start = t
        future_end = t + H

        if future_end > len(X_test):
            continue

        X_future = X_test.iloc[future_start:future_end].values.astype(np.float64)

        try:
            # Use .get_forecast() for H steps
            fc = fitted_model.get_forecast(steps=H, exog=X_future)
            F[t, :] = fc.predicted_mean

            # Get confidence intervals
            ci = fc.conf_int(alpha=0.05)
            if isinstance(ci, pd.DataFrame):
                L[t, :] = ci.iloc[:, 0].values
                U[t, :] = ci.iloc[:, 1].values
            else:
                L[t, :] = ci[:, 0]
                U[t, :] = ci[:, 1]

        except Exception:
            # Fallback to training mean
            F[t, :] = y_mean
            L[t, :] = y_mean - 2 * y_std
            U[t, :] = y_mean + 2 * y_std

    # Build test_eval DataFrame with actual values for each horizon
    eval_dates = y_test.index[:n_origins]
    actuals = {}
    for h in range(1, H + 1):
        # Target_h = actual value h days ahead from each origin
        actuals[f"Target_{h}"] = y_test.values[h - 1:n_origins + h - 1]
    test_eval = pd.DataFrame(actuals, index=eval_dates)

    # Calculate per-horizon metrics
    metrics_df = _calculate_scenario1_metrics(F, test_eval, L, U, H)

    return F, L, U, test_eval, metrics_df


def _calculate_scenario1_metrics(
    F: np.ndarray,
    test_eval: pd.DataFrame,
    L: np.ndarray,
    U: np.ndarray,
    H: int,
) -> pd.DataFrame:
    """
    Step 4.5: Calculate comprehensive metrics for Scenario-1.

    Metrics per horizon (thesis requirements):
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Accuracy (100 - MAPE)
    - Residual mean and std
    - CI Coverage

    Args:
        F: Forecast matrix [n_origins x H]
        test_eval: DataFrame with Target_1...Target_H columns
        L, U: Lower/Upper CI matrices
        H: Maximum horizon

    Returns:
        DataFrame with per-horizon metrics + average row
    """
    results = []

    for h in range(1, H + 1):
        col_name = f"Target_{h}"
        if col_name not in test_eval.columns:
            continue

        actual = test_eval[col_name].values
        pred = F[:, h - 1]
        lower = L[:, h - 1]
        upper = U[:, h - 1]

        # Filter valid data
        valid = np.isfinite(actual) & np.isfinite(pred)
        if not np.any(valid):
            continue

        actual_v = actual[valid]
        pred_v = pred[valid]

        # Core metrics
        mae = float(mean_absolute_error(actual_v, pred_v))
        rmse = float(np.sqrt(mean_squared_error(actual_v, pred_v)))

        # MAPE with zero handling
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(np.abs(actual_v) < 1e-8, np.nan, np.abs(actual_v))
            ape = np.abs((actual_v - pred_v) / denom)
            mape = float(np.nanmean(ape) * 100)
            mape = min(100.0, mape) if np.isfinite(mape) else 100.0

        accuracy = max(0.0, 100.0 - mape)

        # Residuals for this horizon
        residuals = actual_v - pred_v
        resid_mean = float(np.mean(residuals))
        resid_std = float(np.std(residuals))

        # CI Coverage
        lower_v = lower[valid]
        upper_v = upper[valid]
        if np.any(np.isfinite(lower_v)) and np.any(np.isfinite(upper_v)):
            coverage = float(np.mean((actual_v >= lower_v) & (actual_v <= upper_v)) * 100)
        else:
            coverage = np.nan

        results.append({
            "Horizon": h,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "Accuracy_%": accuracy,
            "Residual_Mean": resid_mean,
            "Residual_Std": resid_std,
            "CI_Coverage_%": coverage,
        })

    metrics_df = pd.DataFrame(results)

    # Add average row
    if not metrics_df.empty:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        avg = {col: metrics_df[col].mean() for col in numeric_cols}
        avg["Horizon"] = "Average"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg])], ignore_index=True)

    return metrics_df


def _auto_order_rmse_only(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
    n_folds: int = 3,
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """
    Find optimal SARIMAX orders by minimizing PURE CV-RMSE (no AIC penalty).

    Uses expanding window cross-validation for robust RMSE estimation.
    This method prioritizes out-of-sample prediction accuracy over model complexity.

    Args:
        y_tr: Training target series
        X_tr: Training exogenous features
        m: Seasonal period
        n_folds: Number of CV folds (default 3)
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
                train_end = fold_size * (i + 2)
                if train_end > len(y_tr):
                    break

                test_start = train_end
                test_end = min(train_end + fold_size, len(y_tr))

                if test_end <= test_start:
                    continue

                y_cv_train = y_tr.iloc[:train_end]
                y_cv_test = y_tr.iloc[test_start:test_end]

                if X_tr is not None and X_tr.shape[1] > 0:
                    X_cv_train = X_tr.iloc[:train_end].copy()
                    X_cv_test = X_tr.iloc[test_start:test_end].copy()
                    # Ensure float64
                    X_cv_train = X_cv_train.astype(np.float64)
                    X_cv_test = X_cv_test.astype(np.float64)
                else:
                    X_cv_train = None
                    X_cv_test = None

                if len(y_cv_train) < 20 or len(y_cv_test) == 0:
                    continue

                # Convert to numpy arrays for SARIMAX
                y_cv_arr, X_cv_arr = _prepare_sarimax_input(y_cv_train, X_cv_train)
                _, X_cv_test_arr = _prepare_sarimax_input(y_cv_test, X_cv_test)

                cv_model = SARIMAX(
                    endog=y_cv_arr,
                    exog=X_cv_arr,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                cv_forecast = cv_model.get_forecast(
                    steps=len(y_cv_test),
                    exog=X_cv_test_arr
                ).predicted_mean

                rmse_fold = float(np.sqrt(mean_squared_error(y_cv_test.values, cv_forecast)))
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
        # Convert to numpy arrays for SARIMAX
        y_full_arr, X_full_arr = _prepare_sarimax_input(y_tr, X_tr)

        full_model = SARIMAX(
            endog=y_full_arr,
            exog=X_full_arr,
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
            random_state=SEED, maxiter=100,
        )
        return a.order, a.seasonal_order, float(a.aic()), float(a.bic())
    # Fallback: modest seasonal default (weekly seasonality)
    return (1, 1, 1), (1, 1, 0, m), float("nan"), float("nan")


def _auto_order_fast(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """
    Ultra-fast order selection using only common seasonal patterns.
    Evaluates just 3 candidates instead of 30-80 from pmdarima stepwise.

    Ideal for: Quick prototyping, large datasets, interactive use.
    Trade-off: May not find globally optimal parameters.

    Args:
        y_tr: Training target series
        X_tr: Training exogenous features (optional)
        m: Seasonal period (default 7 for weekly)

    Returns:
        Tuple (order, seasonal_order, aic, bic)
    """
    # Pre-defined common weekly seasonal patterns (from ED forecasting literature)
    candidates = [
        ((1, 1, 1), (1, 1, 1, m)),  # Full seasonal - best for strong weekly patterns
        ((1, 1, 1), (1, 1, 0, m)),  # Seasonal AR only
        ((1, 0, 1), (1, 0, 1, m)),  # No differencing
        ((0, 1, 1), (0, 1, 1, m)),  # Simple exponential smoothing style
    ]

    best_aic = float('inf')
    best_bic = float('nan')
    best_order = candidates[0][0]
    best_sorder = candidates[0][1]

    for order, sorder in candidates:
        try:
            y_arr, X_arr = _prepare_sarimax_input(y_tr, X_tr)
            model = SARIMAX(
                endog=y_arr,
                exog=X_arr,
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=50)

            if model.aic < best_aic:
                best_aic = model.aic
                best_bic = model.bic
                best_order = order
                best_sorder = sorder
        except Exception:
            continue

    return best_order, best_sorder, best_aic, best_bic


def _auto_order_hybrid(
    y_tr: pd.Series,
    X_tr: Optional[pd.DataFrame],
    m: int = 7,
    n_folds: int = 3,
    alpha: float = 0.3,  # AIC weight
    beta: float = 0.7,   # CV-RMSE weight
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], float, float]:
    """
    Find optimal SARIMAX orders by minimizing weighted composite score:
    Score = alpha * Normalized_AIC + beta * Normalized_CV_RMSE

    Uses expanding window cross-validation for robust RMSE estimation.
    Leverages pmdarima for efficient candidate generation.

    Args:
        y_tr: Training target series
        X_tr: Training exogenous features
        m: Seasonal period
        n_folds: Number of CV folds (default 3)
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
                random_state=SEED, maxiter=50,
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
            # Convert to numpy arrays for full model
            y_full_arr, X_full_arr = _prepare_sarimax_input(y_tr, X_tr)

            # 1. Full fit for AIC
            model_full = SARIMAX(
                endog=y_full_arr,
                exog=X_full_arr,
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
                train_end = fold_size * (i + 2)
                if train_end > len(y_tr):
                    break

                test_start = train_end
                test_end = min(train_end + fold_size, len(y_tr))

                if test_end <= test_start:
                    continue

                y_cv_train = y_tr.iloc[:train_end]
                y_cv_test = y_tr.iloc[test_start:test_end]

                if X_tr is not None and X_tr.shape[1] > 0:
                    X_cv_train = X_tr.iloc[:train_end].copy()
                    X_cv_test = X_tr.iloc[test_start:test_end].copy()
                    # Ensure float64
                    X_cv_train = X_cv_train.astype(np.float64)
                    X_cv_test = X_cv_test.astype(np.float64)
                else:
                    X_cv_train = None
                    X_cv_test = None

                if len(y_cv_train) < 20 or len(y_cv_test) == 0:
                    continue

                # Convert to numpy arrays for SARIMAX
                y_cv_arr, X_cv_arr = _prepare_sarimax_input(y_cv_train, X_cv_train)
                _, X_cv_test_arr = _prepare_sarimax_input(y_cv_test, X_cv_test)

                cv_model = SARIMAX(
                    endog=y_cv_arr,
                    exog=X_cv_arr,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                cv_forecast = cv_model.get_forecast(
                    steps=len(y_cv_test),
                    exog=X_cv_test_arr
                ).predicted_mean

                rmse_fold = float(np.sqrt(mean_squared_error(y_cv_test.values, cv_forecast)))
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
    X_all.index = df_full[date_col]

    # Split
    n = len(y)
    split = int(n * float(train_ratio))
    split = max(12, min(n - 1, split))
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    X_tr, X_te = X_all.loc[y_tr.index].copy(), X_all.loc[y_te.index].copy()

    # Ensure float64
    X_tr = X_tr.astype(np.float64)
    X_te = X_te.astype(np.float64)

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

    # Convert to numpy arrays for SARIMAX
    y_tr_arr, X_tr_arr = _prepare_sarimax_input(y_tr, X_tr)
    _, X_te_arr = _prepare_sarimax_input(y_te, X_te)

    # Fit
    fit = SARIMAX(
        endog=y_tr_arr,
        exog=X_tr_arr,
        order=use_order,
        seasonal_order=use_sorder,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # In-sample/fitted
    fitted = fit.fittedvalues
    fitted = pd.Series(fitted, index=y_tr.index)

    # Forecast on test
    fc = fit.get_forecast(steps=len(y_te), exog=X_te_arr)
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
    # Cap train MAPE at 100%
    tr_mape = min(100.0, tr_mape) if np.isfinite(tr_mape) else 100.0
    tr_acc = max(0.0, min(100.0, 100.0 - tr_mape)) if np.isfinite(tr_mape) else 0.0

    te_mae = float(mean_absolute_error(y_te, mean))
    te_rmse = _rmse(y_te, mean)
    te_mape = float(_mape(y_te, mean))
    # Cap MAPE at 100% for display purposes (model is useless beyond this)
    te_mape = min(100.0, te_mape) if np.isfinite(te_mape) else 100.0
    try:
        te_r2 = float(r2_score(y_te, mean))
    except Exception:
        te_r2 = float("nan")
    # Clamp accuracy to [0, 100] range
    te_acc = max(0.0, min(100.0, 100.0 - te_mape)) if np.isfinite(te_mape) else 0.0

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
    # Search mode: "fast" (3 candidates ~30s), "aic_only" (~2min), "rmse_only" (~5min), "hybrid"
    search_mode: str = "rmse_only",
    n_folds: int = 3,
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
        if use_all_features and X_all_global is not None:
            # Use positional indexing to avoid index alignment issues
            X = X_all_global.iloc[:len(y)].copy()
            X.index = y.index
        else:
            X = _exog_calendar(y.index, include_dow_ohe=include_dow_ohe)
            X.index = y.index
            if selected_features:
                keep = [c for c in selected_features if c in X.columns]
                if include_dow_ohe:
                    keep = list(dict.fromkeys(keep + [c for c in X.columns if c not in keep]))
                X = X[keep] if keep else X

        # Fill NaN and ensure all float64
        X = X.fillna(0).astype(np.float64)

        # Train/test split
        n = len(y)
        split = int(n * float(train_ratio))
        split = max(10, min(n - 1, split))  # guards
        y_tr, y_te = y.iloc[:split], y.iloc[split:]
        X_tr, X_te = X.iloc[:split, :].copy(), X.iloc[split:, :].copy()

        # Ensure float64 after slicing
        X_tr = X_tr.astype(np.float64)
        X_te = X_te.astype(np.float64)

        # Scale features to prevent numerical instability
        # SARIMAX can explode with unscaled exogenous variables of different magnitudes
        if X_tr.shape[1] > 0:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_tr),
                index=X_tr.index,
                columns=X_tr.columns
            )
            X_te = pd.DataFrame(
                scaler.transform(X_te),
                index=X_te.index,
                columns=X_te.columns
            )

        # ================================================================
        # DETERMINE ORDERS: Manual mode takes priority over auto-search
        # ================================================================

        # CASE 1: Manual mode - user provided both order and seasonal_order
        if order is not None and seasonal_order is not None:
            # Manual mode: use user-specified orders directly (no search needed)
            use_order, use_sorder = order, seasonal_order
            aic_ref, bic_ref = float("nan"), float("nan")
            if h == 1:
                print(f"[SARIMAX] Manual mode: using order={order}, seasonal_order={seasonal_order}")

        # CASE 2: search_mode="manual" but parameters missing - raise error
        elif search_mode == "manual":
            raise ValueError(
                f"Manual mode selected (search_mode='manual') but order and/or seasonal_order not provided. "
                f"Got order={order}, seasonal_order={seasonal_order}. "
                f"Please provide both parameters for manual mode."
            )

        # CASE 3: Auto mode, first horizon - run parameter search
        elif (order is None or seasonal_order is None) and h == 1:
            # Full auto search for first horizon only
            if search_mode == "fast":
                # Ultra-fast mode: only evaluate 3 pre-selected candidates (~10x faster)
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_fast(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length
                )
            elif search_mode == "rmse_only":
                # RMSE-only mode: optimize pure CV-RMSE (DEFAULT - seeks lowest RMSE)
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_rmse_only(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            elif search_mode == "hybrid":
                # Hybrid mode: optimize AIC + CV-RMSE (70% RMSE weight)
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_hybrid(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, alpha=0.3, beta=0.7,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            else:
                # AIC-only mode (aic_only): standard pmdarima stepwise
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )

            # Save the optimal parameters found for reuse across horizons
            shared_order = ord_auto if order is None else order
            shared_seasonal_order = sord_auto if seasonal_order is None else seasonal_order
            use_order = shared_order
            use_sorder = shared_seasonal_order
            aic_ref, bic_ref = aic_auto, bic_auto
            print(f"[SARIMAX] Auto search ({search_mode}): selected order={use_order}, seasonal_order={use_sorder}")

        # CASE 4: Auto mode, subsequent horizons - reuse h=1 parameters
        elif (order is None or seasonal_order is None) and h > 1 and shared_order is not None and shared_seasonal_order is not None:
            # Reuse parameters from h=1 for subsequent horizons (skip expensive auto search)
            use_order = shared_order
            use_sorder = shared_seasonal_order
            aic_ref, bic_ref = float("nan"), float("nan")

        # CASE 5: Fallback for edge cases (shouldn't normally reach here)
        else:
            # Fallback: run auto search with current search_mode
            if search_mode == "fast":
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_fast(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length
                )
            elif search_mode == "rmse_only":
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_rmse_only(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    max_P=max_P, max_Q=max_Q, max_D=max_D
                )
            elif search_mode == "hybrid":
                ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_hybrid(
                    y_tr, X_tr if X_tr.shape[1] > 0 else None, m=season_length,
                    n_folds=n_folds, alpha=0.3, beta=0.7,
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

        # Convert to numpy arrays for SARIMAX
        y_tr_arr, X_tr_arr = _prepare_sarimax_input(y_tr, X_tr)
        _, X_te_arr = _prepare_sarimax_input(y_te, X_te)

        # Fit model with optimized solver settings for speed
        # - method='powell': Often faster than default 'lbfgs' for SARIMAX
        # - maxiter=50: Sufficient for convergence, prevents over-iteration
        # - low_memory=True: Reduces memory footprint for large datasets
        model = SARIMAX(
            endog=y_tr_arr,
            exog=X_tr_arr,
            order=use_order,
            seasonal_order=use_sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, method='powell', maxiter=50, low_memory=True)

        # In-sample fit
        fitted = pd.Series(fit.fittedvalues, index=y_tr.index)

        # Out-of-sample forecast
        fc = fit.get_forecast(steps=len(y_te), exog=X_te_arr)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        mean = pd.Series(mean, index=y_te.index)
        ci = pd.DataFrame(ci, index=y_te.index)

        # ------------ metrics ------------
        tr_mae = float(mean_absolute_error(y_tr, fitted))
        tr_rmse = _rmse(y_tr, fitted)
        tr_mape = float(_mape(y_tr, fitted))
        try:
            tr_r2 = float(r2_score(y_tr, fitted))
        except Exception:
            tr_r2 = float("nan")
        # Cap train MAPE at 100%
        tr_mape = min(100.0, tr_mape) if np.isfinite(tr_mape) else 100.0
        tr_acc = max(0.0, min(100.0, 100.0 - tr_mape)) if np.isfinite(tr_mape) else 0.0

        te_mae = float(mean_absolute_error(y_te, mean))
        te_rmse = _rmse(y_te, mean)
        te_mape = float(_mape(y_te, mean))
        # Cap MAPE at 100% for display purposes
        te_mape = min(100.0, te_mape) if np.isfinite(te_mape) else 100.0
        try:
            te_r2 = float(r2_score(y_te, mean))
        except Exception:
            te_r2 = float("nan")
        # Clamp accuracy to [0, 100] range
        te_acc = max(0.0, min(100.0, 100.0 - te_mape)) if np.isfinite(te_mape) else 0.0

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
    return {"results_df": results_df, "per_h": per_h, "successful": ok}


# ===========================================
# SARIMAX Fast Manual Mode (Fit Once, Forecast All)
# ===========================================
def run_sarimax_multihorizon_fast(
    df_merged: pd.DataFrame,
    date_col: str = "Date",
    train_ratio: float = 0.80,
    season_length: int = 7,
    horizons: int = 7,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
    use_all_features: bool = False,
    include_dow_ohe: bool = True,
    selected_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    FAST Manual SARIMAX: Fits ONE model, forecasts H steps ahead.

    Instead of fitting 7 separate models (one per horizon), this fits a single
    model on Target_1 and uses multi-step forecasting to predict all horizons.

    ~7-10x faster than run_sarimax_multihorizon() for manual mode.

    Args:
        df_merged: DataFrame with Date and Target_1 columns
        date_col: Name of date column
        train_ratio: Fraction of data for training
        season_length: Seasonal period (default 7 for weekly)
        horizons: Number of forecast horizons (default 7)
        order: ARIMA order (p,d,q) - REQUIRED for fast mode
        seasonal_order: Seasonal order (P,D,Q,m) - REQUIRED for fast mode
        use_all_features: Use all available features as exogenous (default False for speed)
        include_dow_ohe: Include day-of-week one-hot encoding
        selected_features: Specific features to use

    Returns:
        Dict with same structure as run_sarimax_multihorizon():
        - results_df: DataFrame with per-horizon metrics
        - per_h: Dict mapping h -> {y_train, y_test, forecast, ci_lo, ci_hi, ...}
        - successful: List of successful horizons
        - order: Tuple of ARIMA order used
        - seasonal_order: Tuple of seasonal order used
    """
    print(f"[SARIMAX Fast] Manual mode: Fit once, forecast all horizons")
    print(f"[SARIMAX Fast] Parameters: order={order}, seasonal_order={seasonal_order}")

    # ========== 1. Data Preparation ==========
    df_full = _clean_daily(df_merged, date_col)

    # Use Target_1 as base target (today's arrivals / 1-day ahead)
    target_col = "Target_1"
    if target_col not in df_full.columns:
        raise ValueError(f"Target column '{target_col}' not found. Fast mode requires Target_1.")

    y = df_full.set_index(date_col)[target_col].astype(float)
    y = y.interpolate("linear").ffill().bfill()

    if len(y) < 50:
        raise ValueError(f"Insufficient data for SARIMAX fast mode (need >= 50, got {len(y)})")

    # ========== 2. Build Exogenous Features ==========
    if use_all_features:
        X_all = _exog_all(
            df_full=df_full,
            date_col=date_col,
            target_cols=[target_col],
            include_dow_ohe=include_dow_ohe,
            selected_features=selected_features,
        )
    else:
        X_all = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)
    X_all.index = y.index
    X_all = X_all.fillna(0).astype(np.float64)

    # ========== 3. Train/Test Split ==========
    n = len(y)
    split = int(n * train_ratio)
    split = max(24, min(n - horizons, split))

    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    X_tr, X_te = X_all.iloc[:split].copy(), X_all.iloc[split:].copy()

    # Scale features for numerical stability
    if X_tr.shape[1] > 0:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(
            scaler.fit_transform(X_tr),
            index=X_tr.index,
            columns=X_tr.columns
        )
        X_te = pd.DataFrame(
            scaler.transform(X_te),
            index=X_te.index,
            columns=X_te.columns
        )

    # Convert to numpy arrays
    y_tr_arr, X_tr_arr = _prepare_sarimax_input(y_tr, X_tr)
    _, X_te_arr = _prepare_sarimax_input(y_te, X_te)

    # ========== 4. FIT ONCE ==========
    model = SARIMAX(
        endog=y_tr_arr,
        exog=X_tr_arr,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # Fast fit settings for manual mode
    fit = model.fit(
        disp=False,
        method='lbfgs',
        maxiter=50,
        low_memory=True
    )

    # In-sample fitted values for training metrics
    fitted_train = pd.Series(fit.fittedvalues, index=y_tr.index)

    # ========== 5. FORECAST ALL HORIZONS AT ONCE ==========
    fc = fit.get_forecast(steps=len(y_te), exog=X_te_arr)
    forecast_mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    # Convert to pandas for easier handling
    forecast_mean = pd.Series(forecast_mean, index=y_te.index)
    ci = pd.DataFrame(ci, index=y_te.index)

    # ========== 6. Build Results for Each Horizon ==========
    rows = []
    per_h = {}
    ok = []

    # Training metrics (same for all horizons since single model)
    tr_mae = float(mean_absolute_error(y_tr, fitted_train))
    tr_rmse = _rmse(y_tr, fitted_train)
    tr_mape = float(_mape(y_tr, fitted_train))
    tr_mape = min(100.0, tr_mape) if np.isfinite(tr_mape) else 100.0
    tr_acc = max(0.0, min(100.0, 100.0 - tr_mape))
    try:
        tr_r2 = float(r2_score(y_tr, fitted_train))
    except Exception:
        tr_r2 = float("nan")

    for h in range(1, min(horizons + 1, len(y_te) + 1)):
        # For horizon h, compare forecast starting at position (h-1) with actuals
        # This simulates "h days ahead" prediction
        h_offset = h - 1

        if h_offset >= len(forecast_mean):
            continue

        # Get forecast and actuals for this horizon
        # Shift both to align: forecast[h-1:] vs actual[h-1:]
        fc_h = forecast_mean.iloc[h_offset:].values
        actual_h = y_te.iloc[h_offset:].values
        ci_lo_h = ci.iloc[h_offset:, 0].values
        ci_hi_h = ci.iloc[h_offset:, 1].values
        dates_h = y_te.index[h_offset:]

        # Ensure same length
        min_len = min(len(fc_h), len(actual_h))
        if min_len == 0:
            continue

        fc_h = fc_h[:min_len]
        actual_h = actual_h[:min_len]
        ci_lo_h = ci_lo_h[:min_len]
        ci_hi_h = ci_hi_h[:min_len]
        dates_h = dates_h[:min_len]

        # Calculate test metrics for this horizon
        te_mae = float(np.mean(np.abs(actual_h - fc_h)))
        te_rmse = float(np.sqrt(np.mean((actual_h - fc_h) ** 2)))
        te_mape = float(_mape(actual_h, fc_h))
        te_mape = min(100.0, te_mape) if np.isfinite(te_mape) else 100.0
        te_acc = max(0.0, min(100.0, 100.0 - te_mape))

        try:
            te_r2 = float(r2_score(actual_h, fc_h))
        except Exception:
            te_r2 = float("nan")

        # Additional metrics
        bias = float(np.mean(fc_h - actual_h))
        dacc = _dir_acc(actual_h, fc_h)
        abs_err = np.abs(fc_h - actual_h)
        within2 = float(np.mean(abs_err <= 2) * 100.0)
        within5 = float(np.mean(abs_err <= 5) * 100.0)
        cicov = float(np.mean((actual_h >= ci_lo_h) & (actual_h <= ci_hi_h)) * 100.0)

        rows.append({
            "Horizon": h,
            "Order": f"{order}x{seasonal_order}",
            "AIC": float(fit.aic),
            "BIC": float(fit.bic),
            "Train_MAE": tr_mae, "Train_RMSE": tr_rmse, "Train_MAPE": tr_mape, "Train_Acc": tr_acc, "Train_R2": tr_r2,
            "Test_MAE": te_mae, "Test_RMSE": te_rmse, "Test_MAPE": te_mape, "Test_Acc": te_acc, "Test_R2": te_r2,
            "Bias": bias, "DirAcc": dacc, "Within+/-2": within2, "Within+/-5": within5, "CIcov%": cicov,
            "Train_N": int(len(y_tr)), "Test_N": int(min_len),
        })

        per_h[h] = {
            "fit": fit,
            "res": fit,
            "y_train": y_tr,
            "y_test": pd.Series(actual_h, index=dates_h),
            "fitted": fitted_train,
            "forecast": pd.Series(fc_h, index=dates_h),
            "ci_lo": pd.Series(ci_lo_h, index=dates_h),
            "ci_hi": pd.Series(ci_hi_h, index=dates_h),
            "order": order,
            "sorder": seasonal_order,
        }
        ok.append(h)

    if not ok:
        raise RuntimeError("SARIMAX fast: could not compute any horizon results")

    results_df = pd.DataFrame(rows).sort_values("Horizon").reset_index(drop=True)

    print(f"[SARIMAX Fast] Completed: {len(ok)} horizons, Train={len(y_tr)}, Test={len(y_te)}")

    return {
        "results_df": results_df,
        "per_h": per_h,
        "successful": ok,
        "order": order,
        "seasonal_order": seasonal_order,
    }


# ===========================================
# SARIMAX Baseline Pipeline (Expanding Window)
# ===========================================
def run_sarimax_baseline_pipeline(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "Target_1",
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    train_ratio: float = 0.8,
    max_horizon: int = 7,
    season_length: int = 7,
    # Feature options
    use_all_features: bool = True,
    include_dow_ohe: bool = True,
    selected_features: Optional[List[str]] = None,
    # Search mode
    search_mode: str = "rmse_only",
    n_folds: int = 3,
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
) -> Dict[str, Any]:
    """
    SARIMAX Baseline: Single model with expanding-window multi-horizon forecaster.

    This mirrors ARIMA's `run_arima_multi_horizon_pipeline()` approach:
    - Single model with one set of (p,d,q)(P,D,Q,m) parameters
    - Expanding window: training data grows at each forecast origin
    - Forecasts H steps ahead from each origin
    - Uses exogenous features (calendar, weather) - key difference from ARIMA

    Unlike `run_sarimax_multihorizon()` which trains 7 separate models (one per horizon),
    this baseline uses ONE model that refits at each step and forecasts all horizons.

    Args:
        df: DataFrame with date column and target column
        date_col: Name of date column
        target_col: Name of target column (e.g., "Target_1" or "ED")
        order: ARIMA order (p,d,q) - if None, auto-selected
        seasonal_order: Seasonal order (P,D,Q,m) - if None, auto-selected
        train_ratio: Fraction of data for initial training
        max_horizon: Number of steps to forecast (default 7)
        season_length: Seasonal period (default 7 for weekly)
        use_all_features: Use all non-target columns as exogenous features
        include_dow_ohe: Include day-of-week one-hot encoding
        selected_features: Specific features to use (optional)
        search_mode: Parameter selection mode ("fast", "aic_only", "rmse_only", "hybrid")
        n_folds: Number of CV folds for parameter selection
        max_p, max_q, max_d: Non-seasonal parameter bounds
        max_P, max_Q, max_D: Seasonal parameter bounds

    Returns:
        Dict with:
            'F': np.ndarray - Forecast matrix [n_test_points × H]
            'L': np.ndarray - Lower 95% CI [n_test_points × H]
            'U': np.ndarray - Upper 95% CI [n_test_points × H]
            'test_eval': pd.DataFrame - Actuals for each horizon (Target_1...Target_H)
            'metrics_df': pd.DataFrame - Per-horizon metrics + average row
            'order': tuple - Selected ARIMA order
            'seasonal_order': tuple - Selected seasonal order
            'eval_dates': DatetimeIndex - Dates for test period
            'max_horizon': int - Number of horizons
            'train': pd.DataFrame - Training data
            'res': SARIMAX results object - For residual diagnostics
            'exog_columns': List[str] - Names of exogenous features used
    """
    # ========== 1. Prepare Data ==========
    df_full = _clean_daily(df, date_col)

    if target_col not in df_full.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Extract target series
    y_series = df_full.set_index(date_col)[target_col].astype(float)
    y_series = y_series.interpolate("linear").ffill().bfill()
    y = y_series.values
    dates = y_series.index

    if len(y) < max(24, max_horizon + 10):
        raise ValueError(f"Not enough data for SARIMAX baseline (need >= {max(24, max_horizon + 10)}, got {len(y)}).")

    # ========== 2. Build Exogenous Features ==========
    if use_all_features:
        X_all = _exog_all(
            df_full=df_full,
            date_col=date_col,
            target_cols=[target_col],  # Exclude target from features
            include_dow_ohe=include_dow_ohe,
            selected_features=selected_features,
        )
    else:
        X_all = _exog_calendar(df_full[date_col], include_dow_ohe=include_dow_ohe)

    # Ensure proper alignment
    X_all = X_all.reset_index(drop=True)
    exog_columns = list(X_all.columns)

    # Validate alignment between features and target
    if len(X_all) != len(y):
        raise ValueError(f"Feature/target length mismatch: X_all={len(X_all)}, y={len(y)}")

    # ========== 3. Train/Test Split Setup ==========
    train_size = int(len(y) * train_ratio)
    train_size = max(24, min(len(y) - max_horizon, train_size))

    # Verify we have enough test data
    if len(y) < train_size + max_horizon:
        H_eff = max(1, len(y) - train_size)
        if H_eff < 1:
            return {"status": "error", "message": "Insufficient data for multi-horizon evaluation."}
        max_horizon = H_eff

    H = int(max_horizon)

    # ========== 4. Parameter Selection (Once) ==========
    y_tr_init = pd.Series(y[:train_size], index=dates[:train_size])
    X_tr_init = X_all.iloc[:train_size].copy().astype(np.float64)

    # CASE 1: Manual mode - user provided both order and seasonal_order
    if order is not None and seasonal_order is not None:
        use_order, use_sorder = order, seasonal_order
        print(f"[SARIMAX Baseline] Manual mode: using order={order}, seasonal_order={seasonal_order}")

    # CASE 2: search_mode="manual" but parameters missing - raise error
    elif search_mode == "manual":
        raise ValueError(
            f"Manual mode selected but order and/or seasonal_order not provided. "
            f"Got order={order}, seasonal_order={seasonal_order}."
        )

    # CASE 3: Auto search
    else:
        if search_mode == "fast":
            ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_fast(
                y_tr_init, X_tr_init if X_tr_init.shape[1] > 0 else None, m=season_length
            )
        elif search_mode == "rmse_only":
            ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_rmse_only(
                y_tr_init, X_tr_init if X_tr_init.shape[1] > 0 else None, m=season_length,
                n_folds=n_folds,
                max_p=max_p, max_q=max_q, max_d=max_d,
                max_P=max_P, max_Q=max_Q, max_D=max_D
            )
        elif search_mode == "hybrid":
            ord_auto, sord_auto, aic_auto, bic_auto = _auto_order_hybrid(
                y_tr_init, X_tr_init if X_tr_init.shape[1] > 0 else None, m=season_length,
                n_folds=n_folds, alpha=0.3, beta=0.7,
                max_p=max_p, max_q=max_q, max_d=max_d,
                max_P=max_P, max_Q=max_Q, max_D=max_D
            )
        else:  # aic_only
            ord_auto, sord_auto, aic_auto, bic_auto = _auto_order(
                y_tr_init, X_tr_init if X_tr_init.shape[1] > 0 else None, m=season_length,
                max_p=max_p, max_q=max_q, max_d=max_d,
                max_P=max_P, max_Q=max_Q, max_D=max_D
            )
        use_order = order if order is not None else ord_auto
        use_sorder = seasonal_order if seasonal_order is not None else sord_auto
        print(f"[SARIMAX Baseline] Auto search ({search_mode}): selected order={use_order}, seasonal_order={use_sorder}")

    # ========== 5. Build F, L, U Matrices (Expanding Window) ==========
    forecast_start_index = train_size
    forecast_end_index = len(y) - H + 1
    n_test_points = max(0, forecast_end_index - forecast_start_index)

    if n_test_points <= 0:
        return {"status": "error", "message": "Not enough test data points for evaluation."}

    F = np.full((n_test_points, H), np.nan)
    L = np.full((n_test_points, H), np.nan)
    U = np.full((n_test_points, H), np.nan)

    eval_dates = dates[forecast_start_index:forecast_end_index]
    final_res = None  # Store last fit for diagnostics

    # Compute reasonable bounds for forecasts based on training data
    y_mean = np.nanmean(y[:train_size])
    y_std = np.nanstd(y[:train_size])
    forecast_min = max(0, y_mean - 10 * y_std)  # Non-negative lower bound
    forecast_max = y_mean + 10 * y_std  # Upper bound at 10 std devs

    for t in range(n_test_points):
        # Expanding training window
        current_train_end = train_size + t
        y_train = y[:current_train_end]
        X_train = X_all.iloc[:current_train_end].copy().astype(np.float64)

        # Future exog for forecasting H steps
        future_start = current_train_end
        future_end = current_train_end + H
        X_future = X_all.iloc[future_start:future_end].copy().astype(np.float64)

        if len(X_future) < H:
            # Pad with last row if needed (edge case)
            while len(X_future) < H:
                X_future = pd.concat([X_future, X_future.iloc[[-1]]], ignore_index=True)

        try:
            # Prepare inputs
            y_arr, X_arr = _prepare_sarimax_input(
                pd.Series(y_train), X_train
            )
            _, X_future_arr = _prepare_sarimax_input(
                pd.Series([0]), X_future  # Dummy y, we just need X
            )

            # Fit SARIMAX
            model = SARIMAX(
                endog=y_arr,
                exog=X_arr,
                order=use_order,
                seasonal_order=use_sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)

            # Forecast H steps
            fc = model.get_forecast(steps=H, exog=X_future_arr)
            preds = fc.predicted_mean

            # CRITICAL: Clip forecasts to reasonable bounds to prevent numerical explosion
            preds = np.clip(preds, forecast_min, forecast_max)
            # Replace any NaN/Inf with the training mean
            preds = np.where(np.isfinite(preds), preds, y_mean)

            F[t, :] = preds

            ci = fc.conf_int(alpha=0.05)
            if isinstance(ci, pd.DataFrame):
                L[t, :] = np.clip(ci.iloc[:, 0].values, forecast_min, forecast_max)
                U[t, :] = np.clip(ci.iloc[:, 1].values, forecast_min, forecast_max)
            else:
                ci = np.asarray(ci)
                L[t, :] = np.clip(ci[:, 0], forecast_min, forecast_max)
                U[t, :] = np.clip(ci[:, 1], forecast_min, forecast_max)

            # Keep last model for diagnostics
            if t == n_test_points - 1:
                final_res = model

        except Exception as e:
            # On failure, use naive forecast (training mean)
            F[t, :] = y_mean
            L[t, :] = y_mean - 2 * y_std
            U[t, :] = y_mean + 2 * y_std
            warnings.warn(f"SARIMAX baseline refit failed at t={t}: {e}")

    # ========== 6. Build Actuals Matrix (test_eval) ==========
    actuals = {}
    for h in range(1, H + 1):
        start_idx = forecast_start_index + h - 1
        end_idx = forecast_end_index + h - 1
        actuals[f"Target_{h}"] = y[start_idx:end_idx]

    test_eval = pd.DataFrame(actuals, index=eval_dates)

    # ========== 7. Calculate Metrics ==========
    metrics_df = _calculate_baseline_metrics(F, test_eval, L, U, H)

    # ========== 8. Final fit on original training split for diagnostics ==========
    if final_res is None:
        try:
            y_arr, X_arr = _prepare_sarimax_input(
                pd.Series(y[:train_size]), X_all.iloc[:train_size].astype(np.float64)
            )
            final_res = SARIMAX(
                endog=y_arr,
                exog=X_arr,
                order=use_order,
                seasonal_order=use_sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            final_res = None

    return {
        "F": F,
        "L": L,
        "U": U,
        "test_eval": test_eval,
        "metrics_df": metrics_df,
        "train": df_full.iloc[:train_size].copy(),
        "res": final_res,
        "eval_dates": eval_dates,
        "order": tuple(use_order),
        "seasonal_order": tuple(use_sorder),
        "max_horizon": H,
        "exog_columns": exog_columns,
        "status": "success",
    }


def _calculate_baseline_metrics(
    F: np.ndarray,
    test_eval: pd.DataFrame,
    L: np.ndarray,
    U: np.ndarray,
    H: int,
) -> pd.DataFrame:
    """
    Calculate per-horizon metrics for baseline pipeline.
    Matches the format expected by plotting functions.

    Includes robust handling of NaN/Inf values to prevent numerical explosions.
    """
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    results = []

    for h in range(H):
        h_idx = h + 1
        col_name = f"Target_{h_idx}"

        if col_name not in test_eval.columns:
            continue

        actual = test_eval[col_name].values.astype(np.float64)
        pred = F[:, h].astype(np.float64)
        l95 = L[:, h].astype(np.float64)
        u95 = U[:, h].astype(np.float64)

        # Filter valid indices - must be finite for both actual and predicted
        valid_idx = np.isfinite(actual) & np.isfinite(pred)
        if not np.any(valid_idx):
            # No valid data points - skip this horizon
            results.append({
                "Horizon": f"h={h_idx}",
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "Accuracy_%": np.nan,
                "R2": np.nan,
                "CI_Coverage_%": np.nan,
                "Direction_Accuracy_%": np.nan,
            })
            continue

        actual_v = actual[valid_idx]
        pred_v = pred[valid_idx]
        l95_v = l95[valid_idx]
        u95_v = u95[valid_idx]

        # Replace any remaining non-finite values in CI bounds
        l95_v = np.where(np.isfinite(l95_v), l95_v, pred_v - np.std(actual_v) * 2)
        u95_v = np.where(np.isfinite(u95_v), u95_v, pred_v + np.std(actual_v) * 2)

        # MAE
        try:
            mae_h = float(mean_absolute_error(actual_v, pred_v))
            # Sanity check: MAE should be reasonable
            if not np.isfinite(mae_h) or mae_h > 1e10:
                mae_h = np.nan
        except Exception:
            mae_h = np.nan

        # RMSE
        try:
            rmse_h = float(np.sqrt(mean_squared_error(actual_v, pred_v)))
            if not np.isfinite(rmse_h) or rmse_h > 1e10:
                rmse_h = np.nan
        except Exception:
            rmse_h = np.nan

        # MAPE - with robust handling of zeros and extreme values
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                # Use absolute values to avoid issues
                abs_actual = np.abs(actual_v)
                # Only compute MAPE where actual is reasonably large
                mask = abs_actual > 0.01  # Avoid division by near-zero
                if np.any(mask):
                    ape = np.abs((pred_v[mask] - actual_v[mask]) / actual_v[mask])
                    # Clip extreme APE values before averaging (cap at 100% per sample)
                    ape = np.clip(ape, 0, 1.0)  # Cap individual errors at 100%
                    mape_h = float(np.nanmean(ape) * 100)
                else:
                    mape_h = np.nan

                # Cap final MAPE at 100% (corresponding to 0% accuracy)
                if np.isfinite(mape_h):
                    mape_h = min(100.0, mape_h)
                else:
                    mape_h = 100.0
        except Exception:
            mape_h = np.nan

        # Accuracy (100 - MAPE, clamped to [0, 100])
        if np.isfinite(mape_h):
            acc_h = max(0.0, min(100.0, 100.0 - mape_h))
        else:
            acc_h = np.nan

        # R²
        try:
            if len(actual_v) > 1 and np.std(actual_v) > 0:
                r2_h = float(r2_score(actual_v, pred_v))
                # R² should be between -inf and 1, but we cap negative values
                r2_h = max(-1.0, min(1.0, r2_h))
            else:
                r2_h = np.nan
        except Exception:
            r2_h = np.nan

        # CI Coverage
        try:
            coverage = float(np.mean((actual_v >= l95_v) & (actual_v <= u95_v)) * 100)
            coverage = max(0.0, min(100.0, coverage))
        except Exception:
            coverage = np.nan

        # Direction Accuracy
        direction_acc = np.nan
        try:
            if len(actual_v) > 1:
                da = np.diff(actual_v)
                dp = np.diff(pred_v)
                mask = da != 0
                if np.any(mask):
                    direction_acc = float(np.mean(np.sign(da[mask]) == np.sign(dp[mask])) * 100)
                    direction_acc = max(0.0, min(100.0, direction_acc))
        except Exception:
            direction_acc = np.nan

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

    # Add average row
    if not metrics_df.empty:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        avg_row = {}
        for k in numeric_cols:
            vals = metrics_df[k].dropna()
            avg_row[k] = float(vals.mean()) if len(vals) > 0 else np.nan
        avg_row["Horizon"] = "Average"
        metrics_df = pd.concat([metrics_df, pd.Series(avg_row).to_frame().T], ignore_index=True)

    return metrics_df.reset_index(drop=True)


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


# ===========================================
# Scenario-1 SARIMAX Pipeline (Thesis Methodology)
# ===========================================

def run_sarimax_scenario1(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "ED",
    train_ratio: float = 0.80,
    max_horizon: int = 7,
    season_length: int = 7,
    use_all_features: bool = True,
    include_dow_ohe: bool = True,
    selected_features: Optional[List[str]] = None,
    mode: str = "scenario1",  # "scenario1" or "manual"
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    max_p: int = 3,
    max_q: int = 3,
    max_P: int = 2,
    max_Q: int = 2,
    adf_significance: float = 0.05,
    ljung_box_lags: int = 10,
) -> Dict[str, Any]:
    """
    Scenario-1 SARIMAX Pipeline following thesis methodology.

    This implements the academic Scenario-1 approach:
    - Step 4.1: Train/Test split (80/20) with exogenous features
    - Step 4.2: auto_arima() with AIC criterion (searches p, q, P, Q only)
    - Step 4.3: Statistical tests for d (ADF) and D (Canova-Hansen) BEFORE order search
    - Step 4.4: Residual diagnostics (Ljung-Box white noise, Jarque-Bera normality)
    - Step 4.5: Test set evaluation with metrics (MAE, RMSE, MAPE, residual stats)
    - Step 4.6: Rolling window forecast using .predict(n_periods=H)

    Key differences from standard approaches:
    - Single SARIMAX model (not per-horizon)
    - d and D determined by statistical tests FIRST
    - Comprehensive residual diagnostics
    - Rolling window (not expanding) forecast

    Args:
        df: Input DataFrame with date and target columns
        date_col: Name of date column
        target_col: Name of target column (e.g., "ED" for total arrivals)
        train_ratio: Train/test split ratio (default 0.80)
        max_horizon: Maximum forecast horizon in days (default 7)
        season_length: Seasonal period (default 7 for weekly)
        use_all_features: Use all available features as exogenous
        include_dow_ohe: Include day-of-week one-hot encoding
        selected_features: Specific features to use (if not use_all_features)
        mode: "scenario1" (full thesis methodology) or "manual" (user-specified orders)
        order: Manual (p, d, q) order (required if mode="manual")
        seasonal_order: Manual (P, D, Q, m) order (required if mode="manual")
        max_p, max_q, max_P, max_Q: Search bounds for auto_arima
        adf_significance: Significance level for ADF test
        ljung_box_lags: Number of lags for Ljung-Box test

    Returns:
        Dict with keys:
        - F, L, U: Forecast matrices [n_origins x H]
        - test_eval: DataFrame with actual values
        - metrics_df: Per-horizon metrics
        - order, seasonal_order: Selected model orders
        - differencing: Scenario1DifferencingResult (Step 4.3)
        - diagnostics: Scenario1DiagnosticResult (Step 4.4)
        - train_metrics, test_metrics: Summary metrics
        - res: Fitted model object
        - status: "success" or "error"
    """
    try:
        # ==================== STEP 4.1: Data Preparation ====================
        y_train, y_test, X_train, X_test, train_dates, test_dates = _prepare_scenario1_data(
            df, date_col, target_col, train_ratio,
            use_all_features, include_dow_ohe, selected_features
        )

        if mode == "manual":
            # Manual mode: use provided orders, skip statistical tests
            if order is None or seasonal_order is None:
                raise ValueError("Manual mode requires both 'order' and 'seasonal_order' parameters")
            use_order = order
            use_sorder = seasonal_order
            differencing = None
            aic_search = float("nan")
            bic_search = float("nan")

        else:
            # ==================== STEP 4.3: Differencing Tests FIRST ====================
            differencing = _compute_differencing_orders(
                y_train, m=season_length,
                adf_significance=adf_significance
            )

            # ==================== STEP 4.2: AIC Order Search with Fixed d/D ====================
            use_order, use_sorder, aic_search, bic_search = _scenario1_order_search(
                y_train, X_train,
                d=differencing.adf_d,
                D=differencing.ch_D,
                m=season_length,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q,
            )

        # ==================== FIT SINGLE SARIMAX MODEL ====================
        y_arr, X_arr = _prepare_sarimax_input(y_train, X_train)

        model = SARIMAX(
            endog=y_arr,
            exog=X_arr,
            order=use_order,
            seasonal_order=use_sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)

        # ==================== STEP 4.4: Residual Diagnostics ====================
        diagnostics = _run_residual_diagnostics(
            fitted.resid,
            lags=ljung_box_lags,
        )

        # ==================== STEP 4.5 & 4.6: Forecast & Evaluate ====================
        F, L, U, test_eval, metrics_df = _scenario1_forecast_and_evaluate(
            fitted, y_train, y_test, X_train, X_test, max_horizon
        )

        # Compute aggregate metrics for reporting
        train_metrics = {
            "AIC": float(fitted.aic),
            "BIC": float(fitted.bic),
            "Log-Likelihood": float(fitted.llf),
        }

        # Filter out "Average" row for test metrics calculation
        metrics_numeric = metrics_df[metrics_df["Horizon"] != "Average"]
        test_metrics = {
            "MAE": float(metrics_numeric["MAE"].mean()) if "MAE" in metrics_numeric else np.nan,
            "RMSE": float(metrics_numeric["RMSE"].mean()) if "RMSE" in metrics_numeric else np.nan,
            "MAPE_%": float(metrics_numeric["MAPE_%"].mean()) if "MAPE_%" in metrics_numeric else np.nan,
            "Accuracy_%": float(metrics_numeric["Accuracy_%"].mean()) if "Accuracy_%" in metrics_numeric else np.nan,
            "Residual_Mean": diagnostics.residual_mean,
            "Residual_Std": diagnostics.residual_std,
        }

        return {
            # Forecast outputs (compatible with existing plotting)
            "F": F,
            "L": L,
            "U": U,
            "test_eval": test_eval,
            "metrics_df": metrics_df,
            "eval_dates": test_dates[:len(F)],
            # Model configuration
            "order": use_order,
            "seasonal_order": use_sorder,
            "max_horizon": max_horizon,
            # Scenario-1 specific outputs
            "differencing": differencing,
            "diagnostics": diagnostics,
            # Metrics
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            # Model artifacts
            "res": fitted,
            "train": y_train,
            "exog_columns": list(X_train.columns) if X_train is not None else [],
            # Status
            "status": "success",
            "message": None,
        }

    except Exception as e:
        import traceback
        return {
            "F": np.array([]),
            "L": np.array([]),
            "U": np.array([]),
            "test_eval": pd.DataFrame(),
            "metrics_df": pd.DataFrame(),
            "order": order,
            "seasonal_order": seasonal_order,
            "differencing": None,
            "diagnostics": None,
            "train_metrics": {},
            "test_metrics": {},
            "res": None,
            "train": None,
            "status": "error",
            "message": f"{str(e)}\n{traceback.format_exc()}",
        }


# ===========================================
# Multi-Target SARIMAX Pipeline (Matches ARIMA pattern)
# ===========================================
def run_sarimax_multi_target_pipeline(
    df: pd.DataFrame,
    target_columns: List[str],
    date_col: str = "Date",
    train_ratio: float = 0.80,
    season_length: int = 7,
    use_all_features: bool = False,
    include_dow_ohe: bool = True,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    selected_features: Optional[List[str]] = None,
    max_p: int = 3, max_q: int = 3, max_d: int = 2,
    max_P: int = 2, max_Q: int = 2, max_D: int = 1,
    search_mode: str = "rmse_only",  # "rmse_only" (default - seeks lowest RMSE), "aic_only", "hybrid"
    n_folds: int = 3,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run SARIMAX pipeline for multiple target columns (patient arrivals + reasons).
    Follows the same pattern as run_arima_multi_target_pipeline for consistency.
    """
    all_results = {}
    metrics_summary = []
    total_targets = len(target_columns)

    # ========== DIAGNOSTIC OUTPUT ==========
    print("\n" + "="*60)
    print("SARIMAX MULTI-TARGET PIPELINE STARTING")
    print("="*60)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns ({len(df.columns)}): {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")
    print(f"Target columns requested ({len(target_columns)}): {target_columns[:10]}{'...' if len(target_columns) > 10 else ''}")
    print(f"Date column: '{date_col}' - exists: {date_col in df.columns}")
    if date_col in df.columns:
        print(f"Date column dtype: {df[date_col].dtype}")
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print("="*60 + "\n")
    # ========================================

    for idx, target_col in enumerate(target_columns):
        if progress_callback:
            progress_callback(target_col, idx, total_targets)

        # Check if target column exists
        if target_col not in df.columns:
            all_results[target_col] = {
                "status": "error",
                "message": f"Column '{target_col}' not found in dataframe"
            }
            continue

        # Check for sufficient data
        valid_count = df[target_col].dropna().shape[0]
        print(f"[SARIMAX CHECK] {target_col}: valid_count={valid_count}")
        if valid_count < 24:
            print(f"[SARIMAX SKIP] {target_col}: Insufficient data ({valid_count} < 24)")
            all_results[target_col] = {
                "status": "error",
                "message": f"Insufficient data for '{target_col}' (need >= 24 points, got {valid_count})"
            }
            continue

        try:
            # Create a simple dataframe with just date and target (like ARIMA does)
            df_simple = df[[date_col, target_col]].copy()

            # Debug: Check data before running
            print(f"[SARIMAX DEBUG] {target_col}: {len(df_simple)} rows, {df_simple[target_col].notna().sum()} non-null")

            # Run single-target SARIMAX
            result = run_sarimax_single(
                df_merged=df_simple,
                date_col=date_col,
                target_col=target_col,
                train_ratio=train_ratio,
                season_length=season_length,
                use_all_features=False,  # Only use calendar features
                include_dow_ohe=include_dow_ohe,
                order=order,
                seasonal_order=seasonal_order,
                max_p=max_p, max_q=max_q, max_d=max_d,
                max_P=max_P, max_Q=max_Q, max_D=max_D,
            )

            # Store results (match ARIMA structure exactly)
            all_results[target_col] = {
                "status": "success",
                "results": result,
                "order": result.get("model_info", {}).get("order", (1, 1, 1)),
                "seasonal_order": result.get("model_info", {}).get("seasonal_order", (1, 1, 0, 7)),
            }

            # Extract metrics for summary
            test_metrics = result.get("test_metrics", {})
            model_info = result.get("model_info", {})
            metrics_summary.append({
                "Target": target_col,
                "Order": str(model_info.get("order", "N/A")),
                "Seasonal": str(model_info.get("seasonal_order", "N/A")),
                "MAE": test_metrics.get("MAE", np.nan),
                "RMSE": test_metrics.get("RMSE", np.nan),
                "MAPE_%": test_metrics.get("MAPE", np.nan),
                "Accuracy_%": test_metrics.get("Accuracy", np.nan),
                "R2": test_metrics.get("R2", np.nan),
                "Direction_Acc_%": test_metrics.get("Direction_Accuracy_%", np.nan),
            })

        except Exception as e:
            import traceback
            print(f"[SARIMAX ERROR] {target_col}: {str(e)}")
            traceback.print_exc()
            all_results[target_col] = {
                "status": "error",
                "message": str(e)
            }
            metrics_summary.append({
                "Target": target_col,
                "Order": "Error",
                "Seasonal": "Error",
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "Accuracy_%": np.nan,
                "R2": np.nan,
                "Direction_Acc_%": np.nan,
            })

    # Create summary dataframe
    summary_df = pd.DataFrame(metrics_summary)

    # Add average row
    if not summary_df.empty:
        numeric_cols = ["MAE", "RMSE", "MAPE_%", "Accuracy_%", "R2", "Direction_Acc_%"]
        avg_row = {"Target": "AVERAGE", "Order": "-", "Seasonal": "-"}
        for col in numeric_cols:
            avg_row[col] = summary_df[col].mean()
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    all_results["summary"] = summary_df
    all_results["successful_targets"] = [t for t, r in all_results.items()
                                          if isinstance(r, dict) and r.get("status") == "success"]
    all_results["failed_targets"] = [t for t, r in all_results.items()
                                      if isinstance(r, dict) and r.get("status") == "error"]

    return all_results


def get_reason_target_columns(df: pd.DataFrame, horizon: Optional[int] = None) -> List[str]:
    """
    Detect all FUTURE target columns in the dataframe for multi-target forecasting.

    These are the dependent variables (what we want to predict):
    - Target_1, Target_2, ..., Target_7 (patient arrivals for horizons 1-7)
    - asthma_1, asthma_2, ..., asthma_7 (asthma cases for horizons 1-7)
    - pneumonia_1, pneumonia_2, etc. (and all other medical reasons)

    Args:
        df: DataFrame with preprocessed columns
        horizon: If specified, only return targets for this specific horizon (1-7)
                 If None, return all available target columns

    Returns:
        List of future target column names (dependent variables)
    """
    # Medical reason base names (without horizon suffix)
    # Includes both granular reasons AND aggregated clinical categories
    reason_bases = [
        # Aggregated clinical categories (recommended for forecasting)
        'respiratory', 'cardiac', 'trauma', 'gastrointestinal', 'infectious', 'neurological', 'other',
        # Granular medical reasons
        'asthma', 'pneumonia', 'shortness_of_breath', 'chest_pain', 'arrhythmia',
        'hypertensive_emergency', 'fracture', 'laceration', 'burn', 'fall_injury',
        'abdominal_pain', 'vomiting', 'diarrhea', 'flu_symptoms', 'fever',
        'viral_infection', 'headache', 'dizziness', 'allergic_reaction', 'mental_health'
    ]

    target_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Skip lag columns (features, not targets)
        if '_lag_' in col_lower or col_lower.startswith('ed_'):
            continue

        # Check for Target_N pattern (patient arrivals)
        if col_lower.startswith('target_'):
            try:
                h = int(col_lower.split('_')[1])
                if horizon is None or h == horizon:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        target_cols.append(col)
            except (IndexError, ValueError):
                pass
            continue

        # Check for reason_N pattern (e.g., asthma_1, pneumonia_2)
        for base in reason_bases:
            if col_lower.startswith(base + '_'):
                try:
                    suffix = col_lower[len(base) + 1:]
                    h = int(suffix)
                    if 1 <= h <= 7:  # Valid horizon range
                        if horizon is None or h == horizon:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                target_cols.append(col)
                except ValueError:
                    pass
                break

    # Sort by horizon for consistent ordering
    def sort_key(col):
        col_lower = col.lower()
        if col_lower.startswith('target_'):
            return (0, int(col_lower.split('_')[1]))
        for i, base in enumerate(reason_bases):
            if col_lower.startswith(base + '_'):
                try:
                    h = int(col_lower[len(base) + 1:])
                    return (i + 1, h)
                except ValueError:
                    return (100, 0)
        return (100, 0)

    return sorted(target_cols, key=sort_key)
