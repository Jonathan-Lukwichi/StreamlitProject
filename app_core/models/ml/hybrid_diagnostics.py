# =============================================================================
# app_core/models/ml/hybrid_diagnostics.py
# Diagnostic Visualizations for Hybrid Model Analysis
# =============================================================================
"""
This module provides diagnostic tools to analyze whether:
1. Residuals have learnable structure (worth training Stage 2)
2. Data is sufficient for two-stage modeling
3. Why a hybrid model might underperform a single model

Usage:
    from app_core.models.ml.hybrid_diagnostics import HybridModelDiagnostics

    diagnostics = HybridModelDiagnostics()
    fig = diagnostics.create_full_diagnostic_dashboard(
        y_true=actual_values,
        y_pred_stage1=lstm_predictions,
        dates=date_index
    )
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.signal import periodogram


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results."""
    test_name: str
    statistic: float
    p_value: Optional[float]
    conclusion: str
    recommendation: str
    is_favorable: bool  # True if result suggests hybrid will help


class HybridModelDiagnostics:
    """
    Diagnostic tools for analyzing hybrid model suitability.

    Helps answer:
    - Do residuals have learnable patterns?
    - Is there enough data for two-stage modeling?
    - Why might hybrid underperform single models?
    """

    def __init__(self):
        self.colors = {
            "primary": "#6366F1",
            "secondary": "#8B5CF6",
            "success": "#22C55E",
            "warning": "#EAB308",
            "danger": "#EF4444",
            "info": "#3B82F6",
            "residual": "#F97316",
            "signal": "#06B6D4",
        }

    def compute_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute residuals (errors) from Stage 1 predictions."""
        return np.array(y_true) - np.array(y_pred)

    def compute_autocorrelation(
        self,
        residuals: np.ndarray,
        max_lag: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute autocorrelation function (ACF) of residuals.

        If residuals have NO learnable structure, ACF should be near zero
        for all lags (white noise).
        """
        n = len(residuals)
        mean = np.mean(residuals)
        var = np.var(residuals)

        acf = []
        for lag in range(max_lag + 1):
            if lag == 0:
                acf.append(1.0)
            else:
                cov = np.mean((residuals[:-lag] - mean) * (residuals[lag:] - mean))
                acf.append(cov / var if var > 0 else 0)

        # Confidence interval (95%) for white noise
        conf_interval = 1.96 / np.sqrt(n)

        return np.array(acf), conf_interval

    def compute_partial_autocorrelation(
        self,
        residuals: np.ndarray,
        max_lag: int = 20
    ) -> np.ndarray:
        """
        Compute partial autocorrelation function (PACF).

        PACF helps identify the order of AR processes in residuals.
        Significant PACF at lag k suggests residuals have AR(k) structure.
        """
        n = len(residuals)
        pacf = [1.0]  # Lag 0 is always 1

        for k in range(1, min(max_lag + 1, n // 4)):
            # Build design matrix for AR(k) regression
            X = np.column_stack([residuals[k-i:-i] if i > 0 else residuals[k:]
                                 for i in range(k)])
            y = residuals[k:]

            if len(y) > k:
                try:
                    # Solve least squares
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    pacf.append(coeffs[-1])
                except:
                    pacf.append(0)
            else:
                pacf.append(0)

        return np.array(pacf)

    def ljung_box_test(
        self,
        residuals: np.ndarray,
        lags: int = 10
    ) -> DiagnosticResult:
        """
        Ljung-Box test for autocorrelation in residuals.

        H0: Residuals are independently distributed (no autocorrelation)
        H1: Residuals exhibit serial correlation

        If p-value < 0.05: Residuals have learnable structure (GOOD for hybrid)
        If p-value > 0.05: Residuals are random noise (BAD for hybrid)
        """
        n = len(residuals)
        acf, _ = self.compute_autocorrelation(residuals, max_lag=lags)

        # Ljung-Box Q statistic
        q_stat = n * (n + 2) * np.sum([acf[k]**2 / (n - k) for k in range(1, lags + 1)])

        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(q_stat, df=lags)

        if p_value < 0.05:
            conclusion = "Residuals have significant autocorrelation (learnable structure)"
            recommendation = "Stage 2 model CAN learn patterns from residuals"
            is_favorable = True
        else:
            conclusion = "Residuals appear to be white noise (no learnable structure)"
            recommendation = "Stage 2 model will likely NOT improve predictions"
            is_favorable = False

        return DiagnosticResult(
            test_name="Ljung-Box Test",
            statistic=q_stat,
            p_value=p_value,
            conclusion=conclusion,
            recommendation=recommendation,
            is_favorable=is_favorable
        )

    def runs_test(self, residuals: np.ndarray) -> DiagnosticResult:
        """
        Runs test for randomness.

        Counts the number of "runs" (consecutive positive or negative values).
        Too few runs = trending pattern (learnable)
        Too many runs = alternating pattern (learnable)
        Expected runs = random noise
        """
        signs = np.sign(residuals)
        signs[signs == 0] = 1  # Treat zeros as positive

        # Count runs
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1

        # Expected runs under randomness
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n = len(signs)

        expected_runs = (2 * n_pos * n_neg) / n + 1
        std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1)))

        if std_runs > 0:
            z_stat = (runs - expected_runs) / std_runs
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0

        if p_value < 0.05:
            conclusion = "Residuals show non-random patterns"
            recommendation = "Patterns exist that Stage 2 might capture"
            is_favorable = True
        else:
            conclusion = "Residuals appear random"
            recommendation = "No systematic patterns for Stage 2 to learn"
            is_favorable = False

        return DiagnosticResult(
            test_name="Runs Test (Randomness)",
            statistic=z_stat,
            p_value=p_value,
            conclusion=conclusion,
            recommendation=recommendation,
            is_favorable=is_favorable
        )

    def spectral_entropy(self, residuals: np.ndarray) -> DiagnosticResult:
        """
        Compute spectral entropy of residuals.

        High entropy (close to 1) = white noise (bad for hybrid)
        Low entropy (close to 0) = concentrated frequencies (good for hybrid)
        """
        # Compute power spectral density
        freqs, psd = periodogram(residuals)

        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log

        # Compute entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1

        if normalized_entropy < 0.7:
            conclusion = f"Low spectral entropy ({normalized_entropy:.2f}) - concentrated frequencies"
            recommendation = "Residuals have periodic structure Stage 2 can learn"
            is_favorable = True
        elif normalized_entropy < 0.85:
            conclusion = f"Moderate spectral entropy ({normalized_entropy:.2f})"
            recommendation = "Some structure exists, hybrid might help marginally"
            is_favorable = True
        else:
            conclusion = f"High spectral entropy ({normalized_entropy:.2f}) - flat spectrum"
            recommendation = "Residuals resemble white noise, Stage 2 won't help much"
            is_favorable = False

        return DiagnosticResult(
            test_name="Spectral Entropy",
            statistic=normalized_entropy,
            p_value=None,
            conclusion=conclusion,
            recommendation=recommendation,
            is_favorable=is_favorable
        )

    def data_sufficiency_check(
        self,
        n_samples: int,
        stage1_params: int = 1000,  # Typical LSTM params
        stage2_params: int = 10     # Typical SARIMAX params
    ) -> DiagnosticResult:
        """
        Check if data is sufficient for two-stage modeling.

        Rule of thumb: Need at least 10-50 samples per parameter.
        """
        total_params = stage1_params + stage2_params
        samples_per_param = n_samples / total_params

        # Also check absolute sample size
        min_samples_stage1 = 200  # LSTM needs significant data
        min_samples_stage2 = 50   # SARIMAX needs less

        issues = []
        if n_samples < min_samples_stage1:
            issues.append(f"Only {n_samples} samples (LSTM needs 200+)")
        if samples_per_param < 5:
            issues.append(f"Only {samples_per_param:.1f} samples per parameter (need 10+)")

        if not issues:
            conclusion = f"Sufficient data: {n_samples} samples, {samples_per_param:.1f} samples/param"
            recommendation = "Data quantity is adequate for two-stage modeling"
            is_favorable = True
        else:
            conclusion = "Insufficient data: " + "; ".join(issues)
            recommendation = "Consider using simpler models or collecting more data"
            is_favorable = False

        return DiagnosticResult(
            test_name="Data Sufficiency",
            statistic=samples_per_param,
            p_value=None,
            conclusion=conclusion,
            recommendation=recommendation,
            is_favorable=is_favorable
        )

    def create_residual_analysis_plot(
        self,
        residuals: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> go.Figure:
        """Create a 2x2 subplot for residual analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Residuals Over Time",
                "Residual Distribution",
                "Autocorrelation (ACF)",
                "Partial Autocorrelation (PACF)"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. Residuals over time
        x_axis = dates if dates is not None else np.arange(len(residuals))
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=residuals,
                mode="lines",
                name="Residuals",
                line=dict(color=self.colors["residual"], width=1),
            ),
            row=1, col=1
        )
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)

        # Add rolling mean to show trends
        window = min(30, len(residuals) // 10)
        if window > 1:
            rolling_mean = pd.Series(residuals).rolling(window=window, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rolling_mean,
                    mode="lines",
                    name=f"Rolling Mean ({window})",
                    line=dict(color=self.colors["warning"], width=2),
                ),
                row=1, col=1
            )

        # 2. Distribution (histogram + normal fit)
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name="Distribution",
                marker_color=self.colors["primary"],
                opacity=0.7,
            ),
            row=1, col=2
        )

        # Add normal distribution overlay
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
        y_norm = y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                name="Normal Fit",
                line=dict(color=self.colors["danger"], width=2),
            ),
            row=1, col=2
        )

        # 3. ACF
        max_lag = min(30, len(residuals) // 4)
        acf, conf = self.compute_autocorrelation(residuals, max_lag=max_lag)
        lags = np.arange(len(acf))

        fig.add_trace(
            go.Bar(
                x=lags,
                y=acf,
                name="ACF",
                marker_color=[self.colors["success"] if abs(a) > conf else self.colors["info"]
                             for a in acf],
            ),
            row=2, col=1
        )
        # Confidence bands
        fig.add_hline(y=conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=1)
        fig.add_hline(y=-conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=1)

        # 4. PACF
        pacf = self.compute_partial_autocorrelation(residuals, max_lag=max_lag)
        fig.add_trace(
            go.Bar(
                x=np.arange(len(pacf)),
                y=pacf,
                name="PACF",
                marker_color=[self.colors["success"] if abs(p) > conf else self.colors["secondary"]
                             for p in pacf],
            ),
            row=2, col=2
        )
        fig.add_hline(y=conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=2)
        fig.add_hline(y=-conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=2)

        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=False,
            title=dict(
                text="📊 Residual Structure Analysis",
                font=dict(size=18)
            ),
        )

        return fig

    def create_spectral_analysis_plot(self, residuals: np.ndarray) -> go.Figure:
        """Create spectral analysis plot to detect hidden periodicities."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Power Spectral Density",
                "Cumulative Spectral Power"
            ),
            horizontal_spacing=0.1
        )

        # Compute periodogram
        freqs, psd = periodogram(residuals)

        # Convert frequency to period (more interpretable)
        periods = 1 / freqs[1:]  # Skip zero frequency
        psd_no_dc = psd[1:]

        # 1. Power Spectral Density
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=psd_no_dc,
                mode="lines",
                name="PSD",
                line=dict(color=self.colors["signal"], width=1),
                fill="tozeroy",
                fillcolor=f"rgba(6, 182, 212, 0.3)",
            ),
            row=1, col=1
        )

        # Highlight significant peaks
        threshold = np.percentile(psd_no_dc, 95)
        significant_peaks = psd_no_dc > threshold
        if any(significant_peaks):
            fig.add_trace(
                go.Scatter(
                    x=periods[significant_peaks],
                    y=psd_no_dc[significant_peaks],
                    mode="markers",
                    name="Significant Peaks",
                    marker=dict(color=self.colors["warning"], size=10, symbol="star"),
                ),
                row=1, col=1
            )

        # 2. Cumulative spectral power
        cumulative = np.cumsum(psd_no_dc) / np.sum(psd_no_dc) if np.sum(psd_no_dc) > 0 else np.zeros_like(psd_no_dc)
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=cumulative,
                mode="lines",
                name="Cumulative Power",
                line=dict(color=self.colors["primary"], width=2),
            ),
            row=1, col=2
        )

        # Add reference line for white noise (linear cumulative)
        fig.add_trace(
            go.Scatter(
                x=[periods.min(), periods.max()],
                y=[0, 1],
                mode="lines",
                name="White Noise Reference",
                line=dict(color=self.colors["danger"], width=2, dash="dash"),
            ),
            row=1, col=2
        )

        fig.update_xaxes(type="log", title_text="Period (samples)", row=1, col=1)
        fig.update_xaxes(type="log", title_text="Period (samples)", row=1, col=2)
        fig.update_yaxes(title_text="Power", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Proportion", row=1, col=2)

        fig.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title=dict(
                text="🔬 Spectral Analysis (Hidden Periodicities)",
                font=dict(size=18)
            ),
        )

        return fig

    def create_diagnostic_summary_card(
        self,
        results: List[DiagnosticResult]
    ) -> Dict[str, Any]:
        """Create summary of all diagnostic tests."""
        favorable_count = sum(1 for r in results if r.is_favorable)
        total_count = len(results)

        # Overall recommendation
        if favorable_count >= total_count * 0.7:
            overall = "FAVORABLE"
            color = "success"
            recommendation = "Hybrid model is likely to improve over single model"
        elif favorable_count >= total_count * 0.4:
            overall = "UNCERTAIN"
            color = "warning"
            recommendation = "Hybrid might help marginally; try both approaches"
        else:
            overall = "UNFAVORABLE"
            color = "danger"
            recommendation = "Single model likely better; residuals appear random"

        return {
            "overall": overall,
            "color": color,
            "recommendation": recommendation,
            "favorable_count": favorable_count,
            "total_count": total_count,
            "score": favorable_count / total_count if total_count > 0 else 0,
            "details": results
        }

    def create_full_diagnostic_dashboard(
        self,
        y_true: np.ndarray,
        y_pred_stage1: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        stage1_params: int = 1000,
        stage2_params: int = 10
    ) -> Tuple[go.Figure, go.Figure, Dict[str, Any]]:
        """
        Create complete diagnostic dashboard.

        Returns:
            - Residual analysis figure
            - Spectral analysis figure
            - Summary dictionary with test results
        """
        # Compute residuals
        residuals = self.compute_residuals(y_true, y_pred_stage1)

        # Run all diagnostic tests
        test_results = [
            self.ljung_box_test(residuals),
            self.runs_test(residuals),
            self.spectral_entropy(residuals),
            self.data_sufficiency_check(len(residuals), stage1_params, stage2_params)
        ]

        # Create visualizations
        residual_fig = self.create_residual_analysis_plot(residuals, dates)
        spectral_fig = self.create_spectral_analysis_plot(residuals)

        # Create summary
        summary = self.create_diagnostic_summary_card(test_results)

        return residual_fig, spectral_fig, summary


def render_diagnostic_results_streamlit(summary: Dict[str, Any]):
    """
    Render diagnostic results in Streamlit.

    Usage:
        import streamlit as st
        from app_core.models.ml.hybrid_diagnostics import (
            HybridModelDiagnostics,
            render_diagnostic_results_streamlit
        )

        diagnostics = HybridModelDiagnostics()
        residual_fig, spectral_fig, summary = diagnostics.create_full_diagnostic_dashboard(
            y_true, y_pred_stage1
        )

        st.plotly_chart(residual_fig)
        st.plotly_chart(spectral_fig)
        render_diagnostic_results_streamlit(summary)
    """
    import streamlit as st

    # Overall verdict
    color_map = {
        "success": "#22C55E",
        "warning": "#EAB308",
        "danger": "#EF4444"
    }

    verdict_color = color_map.get(summary["color"], "#6366F1")

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {verdict_color}22, {verdict_color}11);
                border-left: 4px solid {verdict_color};
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;'>
        <h3 style='color: {verdict_color}; margin: 0 0 0.5rem 0;'>
            🎯 Hybrid Model Suitability: {summary["overall"]}
        </h3>
        <p style='margin: 0; font-size: 1.1rem;'>
            {summary["recommendation"]}
        </p>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>
            Score: {summary["favorable_count"]}/{summary["total_count"]} tests favorable
            ({summary["score"]*100:.0f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Individual test results
    st.markdown("### 📋 Diagnostic Test Results")

    for result in summary["details"]:
        icon = "✅" if result.is_favorable else "❌"

        with st.expander(f"{icon} {result.test_name}", expanded=False):
            st.markdown(f"**Statistic:** {result.statistic:.4f}")
            if result.p_value is not None:
                st.markdown(f"**P-value:** {result.p_value:.4f}")
            st.markdown(f"**Conclusion:** {result.conclusion}")
            st.info(f"💡 {result.recommendation}")


# =============================================================================
# MULTI-MODEL DIAGNOSTIC PIPELINE WITH RAG
# =============================================================================

@dataclass
class ModelDiagnosticResult:
    """Container for individual model diagnostic analysis."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    residual_mean: float
    residual_std: float
    ljung_box_pvalue: float
    runs_test_pvalue: float
    spectral_entropy: float
    has_learnable_structure: bool
    autocorr_significant_lags: int
    recommendation: str


@dataclass
class HybridRecommendation:
    """Container for hybrid model recommendations."""
    strategy: str
    stage1_model: str
    stage2_model: str
    confidence: float  # 0-1
    rationale: str
    expected_improvement: str
    risk_level: str  # "low", "medium", "high"


class HybridDiagnosticPipeline:
    """
    Multi-model diagnostic pipeline with RAG-powered recommendations.

    This pipeline:
    1. Analyzes ALL trained models individually (SARIMAX, LSTM, ANN, XGBoost)
    2. Compares residual patterns across models
    3. Uses RAG approach to build context and provide intelligent recommendations
    4. Suggests best hybrid strategies and model combinations

    Usage:
        pipeline = HybridDiagnosticPipeline()
        results = pipeline.run_full_pipeline(
            y_true=actual_values,
            model_predictions={
                "SARIMAX": sarimax_preds,
                "LSTM": lstm_preds,
                "XGBoost": xgb_preds,
                "ANN": ann_preds
            },
            dates=date_index
        )
    """

    def __init__(self, api_key: Optional[str] = None, api_provider: str = "anthropic"):
        self.diagnostics = HybridModelDiagnostics()
        self.api_key = api_key
        self.api_provider = api_provider
        self.colors = {
            "sarimax": "#22C55E",   # Green
            "lstm": "#3B82F6",      # Blue
            "xgboost": "#F97316",   # Orange
            "ann": "#A855F7",       # Purple
            "primary": "#6366F1",
            "success": "#22C55E",
            "warning": "#EAB308",
            "danger": "#EF4444",
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        errors = y_true - y_pred
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAPE (handle zero values)
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs(errors[mask] / y_true[mask])) * 100
        else:
            mape = np.nan

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def _count_significant_acf_lags(
        self,
        residuals: np.ndarray,
        max_lag: int = 20
    ) -> int:
        """Count number of significant autocorrelation lags."""
        acf, conf = self.diagnostics.compute_autocorrelation(residuals, max_lag=max_lag)
        # Skip lag 0 (always 1)
        return sum(1 for a in acf[1:] if abs(a) > conf)

    def analyze_single_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ModelDiagnosticResult:
        """
        Run full diagnostic analysis on a single model.

        Returns a comprehensive diagnostic result including:
        - Performance metrics (MAE, RMSE, MAPE)
        - Residual statistics
        - Statistical tests for learnable structure
        - Recommendations
        """
        residuals = self.diagnostics.compute_residuals(y_true, y_pred)
        metrics = self._compute_metrics(y_true, y_pred)

        # Run diagnostic tests
        ljung_box = self.diagnostics.ljung_box_test(residuals)
        runs = self.diagnostics.runs_test(residuals)
        spectral = self.diagnostics.spectral_entropy(residuals)

        # Count significant ACF lags
        significant_lags = self._count_significant_acf_lags(residuals)

        # Determine if residuals have learnable structure
        has_structure = (
            ljung_box.is_favorable or
            runs.is_favorable or
            (spectral.is_favorable and spectral.statistic < 0.8)
        )

        # Generate recommendation
        if has_structure:
            if significant_lags >= 5:
                recommendation = f"Strong learnable structure detected ({significant_lags} significant ACF lags). Good Stage 1 candidate."
            else:
                recommendation = f"Moderate learnable structure. Hybrid may improve by {5 + significant_lags}%."
        else:
            recommendation = "Residuals appear random. Using as Stage 1 likely won't benefit from Stage 2."

        return ModelDiagnosticResult(
            model_name=model_name,
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
            residual_mean=np.mean(residuals),
            residual_std=np.std(residuals),
            ljung_box_pvalue=ljung_box.p_value,
            runs_test_pvalue=runs.p_value,
            spectral_entropy=spectral.statistic,
            has_learnable_structure=has_structure,
            autocorr_significant_lags=significant_lags,
            recommendation=recommendation
        )

    def analyze_all_models(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, ModelDiagnosticResult]:
        """
        Analyze all trained models and return diagnostic results.

        Args:
            y_true: Actual target values
            model_predictions: Dict mapping model name to predictions

        Returns:
            Dict mapping model name to ModelDiagnosticResult
        """
        results = {}
        for model_name, y_pred in model_predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                results[model_name] = self.analyze_single_model(model_name, y_true, y_pred)
        return results

    def _build_rag_context(
        self,
        y_true: np.ndarray,
        model_results: Dict[str, ModelDiagnosticResult],
        data_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build RAG context from all diagnostic results.

        This creates a comprehensive context string that can be used for
        intelligent recommendations (either rule-based or LLM-powered).
        """
        n_samples = len(y_true)

        context = f"""
=== HYBRID MODEL DIAGNOSTIC ANALYSIS ===

DATA CHARACTERISTICS:
- Sample size: {n_samples}
- Target mean: {np.mean(y_true):.2f}
- Target std: {np.std(y_true):.2f}
- Target range: [{np.min(y_true):.2f}, {np.max(y_true):.2f}]

MODEL PERFORMANCE COMPARISON:
"""
        # Sort models by RMSE
        sorted_models = sorted(model_results.items(), key=lambda x: x[1].rmse)

        for i, (name, result) in enumerate(sorted_models, 1):
            context += f"""
{i}. {name}:
   - MAE: {result.mae:.4f} | RMSE: {result.rmse:.4f} | MAPE: {result.mape:.2f}%
   - Residual Mean: {result.residual_mean:.4f} (bias indicator)
   - Residual Std: {result.residual_std:.4f}
   - Ljung-Box p-value: {result.ljung_box_pvalue:.4f} (< 0.05 = learnable structure)
   - Runs Test p-value: {result.runs_test_pvalue:.4f} (< 0.05 = non-random)
   - Spectral Entropy: {result.spectral_entropy:.3f} (< 0.7 = concentrated frequencies)
   - Significant ACF Lags: {result.autocorr_significant_lags}
   - Has Learnable Structure: {'YES' if result.has_learnable_structure else 'NO'}
"""

        # Add hybrid strategy context
        context += """
HYBRID STRATEGY OPTIONS:
1. LSTM-SARIMAX: LSTM captures non-linear patterns, SARIMAX models residual autocorrelation
2. LSTM-XGBoost: LSTM for sequences, XGBoost for complex residual relationships
3. LSTM-ANN: Deep learning + feedforward for residual patterns
4. Weighted Ensemble: Combine multiple model predictions with optimized weights
5. Stacking: Meta-learner trained on base model predictions
6. Decomposition: Separate trend/seasonality/residual components

DECISION CRITERIA:
- Choose Stage 1 model with LOWEST RMSE but residuals with learnable structure
- If residuals are random (high entropy, no ACF significance), hybrid won't help
- XGBoost Stage 2 works best with many features and complex residuals
- SARIMAX Stage 2 works best with clear autocorrelation patterns
- ANN Stage 2 works best with non-linear residual relationships
"""

        return context

    def generate_recommendations(
        self,
        model_results: Dict[str, ModelDiagnosticResult],
        y_true: np.ndarray,
        use_llm: bool = False
    ) -> List[HybridRecommendation]:
        """
        Generate hybrid model recommendations based on diagnostic results.

        Uses either rule-based logic or LLM (if API key provided and use_llm=True).
        """
        recommendations = []

        # Build RAG context
        context = self._build_rag_context(y_true, model_results)

        if use_llm and self.api_key:
            # Use LLM for intelligent recommendations
            recommendations = self._generate_llm_recommendations(context, model_results)
        else:
            # Use rule-based recommendations
            recommendations = self._generate_rule_based_recommendations(model_results)

        return recommendations

    def _generate_rule_based_recommendations(
        self,
        model_results: Dict[str, ModelDiagnosticResult]
    ) -> List[HybridRecommendation]:
        """Generate recommendations using rule-based logic."""
        recommendations = []

        if not model_results:
            return recommendations

        # Sort by RMSE (best first)
        sorted_models = sorted(model_results.items(), key=lambda x: x[1].rmse)

        # Find models with learnable residual structure
        models_with_structure = [
            (name, result) for name, result in sorted_models
            if result.has_learnable_structure
        ]

        # Best overall model
        best_model_name, best_result = sorted_models[0]

        # Rule 1: If best model has learnable structure, recommend hybrid
        if best_result.has_learnable_structure:
            # Determine best Stage 2 based on residual characteristics
            if best_result.autocorr_significant_lags >= 3:
                # Strong autocorrelation → SARIMAX
                stage2 = "SARIMAX"
                strategy = f"LSTM-{stage2}" if "LSTM" in best_model_name else f"{best_model_name}-{stage2}"
                rationale = f"Residuals show {best_result.autocorr_significant_lags} significant ACF lags, indicating strong temporal patterns that SARIMAX can capture."
            elif best_result.spectral_entropy < 0.6:
                # Concentrated frequencies → Could benefit from XGBoost
                stage2 = "XGBoost"
                strategy = f"LSTM-{stage2}" if "LSTM" in best_model_name else f"{best_model_name}-{stage2}"
                rationale = f"Low spectral entropy ({best_result.spectral_entropy:.2f}) suggests hidden periodicities. XGBoost can learn complex residual patterns."
            else:
                # General non-linear patterns → ANN
                stage2 = "ANN"
                strategy = f"LSTM-{stage2}" if "LSTM" in best_model_name else f"{best_model_name}-{stage2}"
                rationale = "Residuals show non-random patterns but unclear structure. ANN can learn flexible residual relationships."

            confidence = min(0.9, 0.5 + (best_result.autocorr_significant_lags * 0.05))

            recommendations.append(HybridRecommendation(
                strategy=strategy,
                stage1_model=best_model_name,
                stage2_model=stage2,
                confidence=confidence,
                rationale=rationale,
                expected_improvement=f"{5 + best_result.autocorr_significant_lags * 2}% RMSE reduction",
                risk_level="low" if confidence > 0.7 else "medium"
            ))

        # Rule 2: If multiple models exist, consider weighted ensemble
        if len(model_results) >= 2:
            model_names = [name for name, _ in sorted_models[:3]]
            avg_rmse = np.mean([r.rmse for _, r in sorted_models[:3]])
            best_rmse = sorted_models[0][1].rmse

            # Ensemble helps when models have similar performance
            rmse_spread = (sorted_models[-1][1].rmse - best_rmse) / best_rmse if best_rmse > 0 else 0

            if rmse_spread < 0.3:  # Models are within 30% of each other
                recommendations.append(HybridRecommendation(
                    strategy="Weighted Ensemble",
                    stage1_model=", ".join(model_names),
                    stage2_model="N/A (combined)",
                    confidence=0.75,
                    rationale=f"Models have similar performance (spread: {rmse_spread*100:.1f}%). Ensemble can reduce variance by combining diverse predictions.",
                    expected_improvement="3-8% RMSE reduction through diversification",
                    risk_level="low"
                ))

        # Rule 3: If NO models have learnable structure, recommend against hybrid
        if not models_with_structure:
            recommendations.append(HybridRecommendation(
                strategy="Single Model (No Hybrid)",
                stage1_model=best_model_name,
                stage2_model="None",
                confidence=0.85,
                rationale="None of the model residuals show learnable structure. Hybrid approaches would likely overfit.",
                expected_improvement="N/A - Stay with single model",
                risk_level="low"
            ))

        # Rule 4: Check for stacking potential
        if len(model_results) >= 3:
            # Calculate correlation between model predictions (via residuals)
            residual_vars = [r.residual_std for _, r in sorted_models]
            var_spread = np.std(residual_vars) / np.mean(residual_vars) if np.mean(residual_vars) > 0 else 0

            if var_spread > 0.2:  # Models capture different aspects
                recommendations.append(HybridRecommendation(
                    strategy="Stacking Ensemble",
                    stage1_model="All models as base learners",
                    stage2_model="Meta-learner (Ridge/XGBoost)",
                    confidence=0.65,
                    rationale=f"Models have diverse error patterns (variance spread: {var_spread*100:.1f}%). Stacking can learn optimal combination.",
                    expected_improvement="5-12% RMSE reduction",
                    risk_level="medium"
                ))

        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        return recommendations

    def _generate_llm_recommendations(
        self,
        context: str,
        model_results: Dict[str, ModelDiagnosticResult]
    ) -> List[HybridRecommendation]:
        """Generate recommendations using LLM (RAG approach)."""
        # First get rule-based as fallback
        rule_based = self._generate_rule_based_recommendations(model_results)

        prompt = f"""
{context}

Based on this diagnostic analysis, recommend the TOP 3 hybrid strategies.

For each recommendation, provide:
1. Strategy name
2. Stage 1 model
3. Stage 2 model (if applicable)
4. Confidence score (0-1)
5. Rationale (2-3 sentences)
6. Expected improvement
7. Risk level (low/medium/high)

Focus on PRACTICAL recommendations that will actually improve forecasting.
If hybrid won't help, say so clearly.

Format each recommendation clearly.
"""

        system_prompt = """You are an expert in time series forecasting and ensemble methods.
Analyze diagnostic results and provide practical hybrid model recommendations.
Be specific about WHY each strategy would work based on the residual analysis.
If the data suggests hybrid won't help, be honest about it."""

        try:
            if self.api_provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                llm_text = response.content[0].text

                # Parse LLM response and enhance rule-based recommendations
                # For now, add LLM insight to the top recommendation
                if rule_based:
                    rule_based[0].rationale = f"[AI Enhanced] {llm_text[:500]}..."

            else:  # OpenAI
                import openai
                client = openai.OpenAI(api_key=self.api_key)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                llm_text = response.choices[0].message.content

                if rule_based:
                    rule_based[0].rationale = f"[AI Enhanced] {llm_text[:500]}..."

        except Exception as e:
            # Fallback to rule-based if LLM fails
            if rule_based:
                rule_based[0].rationale += f" (Note: LLM analysis unavailable: {str(e)[:50]})"

        return rule_based

    def create_comparison_plot(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None
    ) -> go.Figure:
        """Create a comparison plot of all model residuals."""
        n_models = len(model_predictions)
        if n_models == 0:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Model RMSE Comparison",
                "Residual Distributions",
                "ACF Comparison (Lag 1-10)",
                "Spectral Entropy"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        model_colors = {
            "SARIMAX": self.colors["sarimax"],
            "LSTM": self.colors["lstm"],
            "XGBoost": self.colors["xgboost"],
            "ANN": self.colors["ann"],
        }

        rmse_values = []
        model_names = []
        entropy_values = []

        for model_name, y_pred in model_predictions.items():
            if y_pred is None or len(y_pred) != len(y_true):
                continue

            residuals = y_true - y_pred
            metrics = self._compute_metrics(y_true, y_pred)

            color = model_colors.get(model_name, self.colors["primary"])

            # Store for bar chart
            model_names.append(model_name)
            rmse_values.append(metrics["rmse"])

            # Spectral entropy
            spectral = self.diagnostics.spectral_entropy(residuals)
            entropy_values.append(spectral.statistic)

            # 2. Residual distribution (violin plot style via histogram)
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name=model_name,
                    opacity=0.6,
                    marker_color=color,
                    nbinsx=20,
                ),
                row=1, col=2
            )

            # 3. ACF bars
            acf, _ = self.diagnostics.compute_autocorrelation(residuals, max_lag=10)
            fig.add_trace(
                go.Bar(
                    x=[f"Lag {i}" for i in range(1, 11)],
                    y=acf[1:11],
                    name=f"{model_name} ACF",
                    marker_color=color,
                    opacity=0.7,
                ),
                row=2, col=1
            )

        # 1. RMSE bar chart
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=rmse_values,
                marker_color=[model_colors.get(n, self.colors["primary"]) for n in model_names],
                text=[f"{v:.2f}" for v in rmse_values],
                textposition="auto",
                name="RMSE",
            ),
            row=1, col=1
        )

        # 4. Spectral entropy bar chart
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=entropy_values,
                marker_color=[model_colors.get(n, self.colors["primary"]) for n in model_names],
                text=[f"{v:.2f}" for v in entropy_values],
                textposition="auto",
                name="Entropy",
            ),
            row=2, col=2
        )

        # Add entropy threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color=self.colors["warning"],
                      annotation_text="Learnable threshold", row=2, col=2)

        # Add ACF confidence band
        n = len(y_true)
        conf = 1.96 / np.sqrt(n)
        fig.add_hline(y=conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=1)
        fig.add_hline(y=-conf, line_dash="dash", line_color=self.colors["danger"], row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=650,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title=dict(
                text="📊 Multi-Model Diagnostic Comparison",
                font=dict(size=20)
            ),
            barmode="group"
        )

        return fig

    def run_full_pipeline(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete diagnostic pipeline.

        Args:
            y_true: Actual target values
            model_predictions: Dict mapping model name to predictions
            dates: Optional date index for time plots
            use_llm: Whether to use LLM for enhanced recommendations

        Returns:
            Dict containing:
            - model_results: Individual model diagnostics
            - recommendations: List of HybridRecommendation
            - comparison_plot: Plotly figure comparing models
            - best_single_model: Name of best performing model
            - context: RAG context string
        """
        # 1. Analyze all models
        model_results = self.analyze_all_models(y_true, model_predictions)

        if not model_results:
            return {
                "model_results": {},
                "recommendations": [],
                "comparison_plot": go.Figure(),
                "best_single_model": None,
                "context": "No models to analyze."
            }

        # 2. Build RAG context
        context = self._build_rag_context(y_true, model_results)

        # 3. Generate recommendations
        recommendations = self.generate_recommendations(model_results, y_true, use_llm)

        # 4. Create comparison plot
        comparison_plot = self.create_comparison_plot(y_true, model_predictions, dates)

        # 5. Find best single model
        best_model = min(model_results.items(), key=lambda x: x[1].rmse)

        return {
            "model_results": model_results,
            "recommendations": recommendations,
            "comparison_plot": comparison_plot,
            "best_single_model": best_model[0],
            "context": context
        }
