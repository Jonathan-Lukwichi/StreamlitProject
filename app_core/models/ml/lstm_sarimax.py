from __future__ import annotations
import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseHybridModel, HybridArtifacts
from .utils import (
    set_seed,
    train_test_split_time,
    compute_metrics,
    save_artifacts,
    load_artifacts,
)
from .hybrid_utils import (
    build_calendar_features,
    make_lags,
    expanding_oof_forecast,
)
from .lstm_model import build_lstm_model
from .sequence_builder import build_supervised_sequences

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _sm_available = True
except ImportError:
    _sm_available = False


class LSTMSARIMAXHybrid(BaseHybridModel):
    """
    A hybrid model combining an LSTM (Stage 1) and a SARIMAX model (Stage 2).
    - The LSTM learns the primary time-series signal.
    - The SARIMAX model learns to predict the residuals (errors) of the LSTM.
    - The final forecast is the sum of the LSTM forecast and the SARIMAX residual forecast.
    """

    name = "LSTM-SARIMAX"

    def __init__(self, artifact_dir: str = "pipeline_artifacts/hybrids/lstm_sarimax"):
        if not _sm_available:
            raise ImportError("statsmodels is required for the LSTMSARIMAXHybrid model.")
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def _build_features(self, df: pd.DataFrame, target: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepares the full feature set for both model stages."""
        # TODO: Make date_col configurable or auto-detected
        date_col = "Date"

        # Keep Date column for time-based splitting
        date_series = df[date_col]
        cal_feats = build_calendar_features(df, date_col)
        lag_feats = make_lags(df, target, config["lags"], config["rolls"])

        X_tab = pd.concat([date_series, cal_feats, lag_feats], axis=1)
        y = df[target]

        full_df = pd.concat([X_tab, y], axis=1).dropna()

        return full_df.drop(columns=[target]), full_df[target]

    def fit(self, df: pd.DataFrame, target: str, config: Dict[str, Any]) -> HybridArtifacts:
        """Fits the two-stage hybrid model."""
        set_seed(config.get("random_state", 42))

        # 1. Prepare features and split data
        X_tab, y = self._build_features(df, target, config["features"])
        # Use Date column for time-based splitting
        df_train, _, df_eval = train_test_split_time(pd.concat([X_tab, y], axis=1), date_col="Date", test_size=config["eval"]["tail_size"], val_size=0)

        # Drop Date column after splitting (not needed for model training)
        X_train_tab = df_train.drop(columns=[y.name, "Date"])
        y_train = df_train[y.name]
        X_eval_tab = df_eval.drop(columns=[y.name, "Date"])
        y_eval = df_eval[y.name]

        scaler = StandardScaler().fit(X_train_tab)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_tab), index=X_train_tab.index, columns=X_train_tab.columns)

        # 2. Stage 1: Get LSTM Out-of-Fold (OOF) predictions
        def _lstm_fitter(X_fold, y_fold, cfg):
            # cfg already contains the stage1 config (unpacked in oof_config below)
            # Convert to DataFrame for build_supervised_sequences
            if not isinstance(X_fold, pd.DataFrame):
                X_fold = pd.DataFrame(X_fold)
            if not isinstance(y_fold, pd.Series):
                y_fold = pd.Series(y_fold, name="target")

            X_seq, y_seq, _ = build_supervised_sequences(
                pd.concat([X_fold, y_fold], axis=1),
                list(X_fold.columns),
                y_fold.name,
                cfg["lookback"]
            )
            model_cfg = {**cfg, "n_features": X_fold.shape[1]}
            model = build_lstm_model(model_cfg)
            model.fit(X_seq, y_seq, epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0)
            return model

        oof_config = {**config["stage1"], "use_sequences": True}
        oof_preds = expanding_oof_forecast(
            model_fn=_lstm_fitter,
            series_df=X_train_scaled,
            y=y_train,
            splits=config["oof_splits"],
            config=oof_config,
        )

        # 3. Stage 2: Train SARIMAX on residuals
        residuals = y_train.loc[oof_preds.index] - oof_preds
        X_resid_train = X_train_scaled.loc[residuals.index] if config["stage2"]["use_exog"] else None

        sarimax_cfg = config["stage2"]
        stage2_model = SARIMAX(
            endog=residuals,
            exog=X_resid_train,
            order=tuple(sarimax_cfg["order"]),
            seasonal_order=tuple(sarimax_cfg["seasonal_order"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # 4. Refit Stage 1 LSTM on all training data
        df_train_combined = pd.concat([X_train_scaled, y_train.rename("target")], axis=1)
        X_train_seq, y_train_seq, _ = build_supervised_sequences(
            df_train_combined,
            list(X_train_scaled.columns),
            "target",
            config["stage1"]["lookback"]
        )
        stage1_model_final = build_lstm_model({**config["stage1"], "n_features": X_train_scaled.shape[1]})
        stage1_model_final.fit(X_train_seq, y_train_seq, epochs=config["stage1"]["epochs"], batch_size=config["stage1"]["batch_size"], verbose=0)

        # 5. Evaluate final hybrid model on the hold-out evaluation set
        y_pred_final, _, _ = self.predict(X_eval_tab, stage1_model_final, stage2_model, scaler, config)
        metrics = compute_metrics(y_eval, y_pred_final)

        # 6. Save artifacts
        s1_path = os.path.join(self.artifact_dir, "stage1_lstm.keras")
        s2_path = os.path.join(self.artifact_dir, "stage2_sarimax.pkl")
        scaler_path = os.path.join(self.artifact_dir, "scaler.joblib")
        config_path = os.path.join(self.artifact_dir, "config.json")

        stage1_model_final.save(s1_path)
        save_artifacts(stage2_model, s2_path) # Use joblib/pickle for statsmodels results
        save_artifacts(scaler, scaler_path)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return HybridArtifacts(
            stage1_path=s1_path,
            stage2_path=s2_path,
            scaler_path=scaler_path,
            config=config,
            metrics=metrics,
        )

    def predict(self, df_future: pd.DataFrame, stage1_model, stage2_model, scaler, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a forecast for a future dataframe."""
        X_future_scaled = pd.DataFrame(scaler.transform(df_future), index=df_future.index, columns=df_future.columns)

        # Stage 1: LSTM forecast
        # Add dummy target for build_supervised_sequences
        df_with_dummy = X_future_scaled.copy()
        df_with_dummy["dummy_target"] = 0
        X_future_seq, _, _ = build_supervised_sequences(
            df_with_dummy,
            list(X_future_scaled.columns),
            "dummy_target",
            config["stage1"]["lookback"]
        )
        y_hat_stage1 = stage1_model.predict(X_future_seq).flatten()

        # Stage 2: SARIMAX forecast on residuals
        X_future_for_resid = X_future_scaled.iloc[-len(y_hat_stage1):] if config["stage2"]["use_exog"] else None
        y_hat_stage2 = stage2_model.forecast(steps=len(y_hat_stage1), exog=X_future_for_resid).values

        # Final forecast
        y_hat_final = y_hat_stage1 + y_hat_stage2

        return y_hat_final, y_hat_stage1, y_hat_stage2


# A tiny smoke test to ensure the class is runnable
if __name__ == "__main__":
    print("Running smoke test for LSTMSARIMAXHybrid...")

    # 1. Create dummy data
    set_seed(42)
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=200, freq="D"))
    data = np.linspace(0, 100, 200) + np.sin(np.arange(200) / 7) * 15 + np.random.randn(200) * 5
    df = pd.DataFrame({"Date": dates, "value": data})

    # 2. Define a configuration
    test_config = {
        "random_state": 42,
        "features": {
            "lags": [7, 14],
            "rolls": [7, 14],
        },
        "eval": {"tail_size": 0.15},
        "oof_splits": 4,
        "stage1": { # LSTM config
            "lookback": 14,
            "units": 50,
            "dropout": 0.1,
            "epochs": 5,
            "batch_size": 16,
            "lr": 0.001,
        },
        "stage2": { # SARIMAX config
            "order": [1, 1, 1],
            "seasonal_order": [1, 0, 1, 7], # Weekly seasonality
            "use_exog": True,
        },
    }

    # 3. Instantiate and run fit
    hybrid_model = LSTMSARIMAXHybrid()
    try:
        artifacts = hybrid_model.fit(df, target="value", config=test_config)
        print("\n[OK] Fit completed successfully!")
        print(f"  - LSTM model saved to: {artifacts.stage1_path}")
        print(f"  - SARIMAX model saved to: {artifacts.stage2_path}")
        print(f"  - Final evaluation metrics: {artifacts.metrics}")
        print("\nSmoke test passed. The model can be instantiated and fit.")

    except Exception as e:
        print(f"\n[ERROR] Smoke test failed: {e}")
        import traceback
        traceback.print_exc()