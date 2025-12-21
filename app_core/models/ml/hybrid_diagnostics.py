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
