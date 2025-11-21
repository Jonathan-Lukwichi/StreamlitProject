"""
Evaluation metrics and visualization for classification models.

Provides comprehensive evaluation including accuracy, precision, recall, F1,
confusion matrices, and comparison plots.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """

    def __init__(self, average_method: str = "weighted"):
        self.average_method = average_method
        self.results = {}

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_pred_proba: Optional[np.ndarray] = None,
                       model_name: str = "Model") -> Dict[str, Any]:
        """
        Compute all evaluation metrics for a single model.

        Returns:
            Dict with accuracy, precision, recall, F1, confusion matrix, etc.
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=self.average_method, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=self.average_method, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=self.average_method, zero_division=0)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # ROC-AUC (if probabilities provided and multi-class)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class OvR
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=self.average_method)
            except Exception as e:
                metrics['roc_auc'] = None

        # Store results
        self.results[model_name] = metrics

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = "Confusion Matrix") -> go.Figure:
        """Create interactive confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            labels = [str(x) for x in np.unique(y_true)]

        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            title=title,
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500
        )

        return fig

    def plot_classification_report(self, model_name: str) -> go.Figure:
        """Visualize per-class precision, recall, F1."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not evaluated yet!")

        report = self.results[model_name]['classification_report']

        # Extract per-class metrics
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Precision',
            x=classes,
            y=precision,
            marker_color='#3B82F6'
        ))

        fig.add_trace(go.Bar(
            name='Recall',
            x=classes,
            y=recall,
            marker_color='#10B981'
        ))

        fig.add_trace(go.Bar(
            name='F1-Score',
            x=classes,
            y=f1,
            marker_color='#F59E0B'
        ))

        fig.update_layout(
            title=f"Per-Class Metrics - {model_name}",
            xaxis_title="Class",
            yaxis_title="Score",
            barmode='group',
            height=500
        )

        return fig

    def plot_model_comparison(self, metric: str = 'accuracy') -> go.Figure:
        """
        Compare multiple models on a single metric.

        Args:
            metric: 'accuracy', 'precision', 'recall', 'f1', or 'roc_auc'
        """
        if not self.results:
            raise ValueError("No models evaluated yet!")

        model_names = list(self.results.keys())
        scores = [self.results[m][metric] for m in model_names if metric in self.results[m]]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=model_names,
            y=scores,
            text=[f"{s:.3f}" for s in scores],
            textposition='auto',
            marker_color=['#3B82F6', '#10B981', '#F59E0B'][:len(model_names)]
        ))

        fig.update_layout(
            title=f"Model Comparison - {metric.upper()}",
            xaxis_title="Model",
            yaxis_title=metric.capitalize(),
            yaxis_range=[0, 1.1],
            height=500
        )

        return fig

    def get_summary_table(self) -> pd.DataFrame:
        """
        Create summary table comparing all models.

        Returns:
            DataFrame with model names and all metrics
        """
        if not self.results:
            return pd.DataFrame()

        summary = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
            }
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                row['ROC-AUC'] = metrics['roc_auc']

            summary.append(row)

        df = pd.DataFrame(summary)
        # Round to 4 decimal places
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)

        return df

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model") -> go.Figure:
        """Plot ROC curve for binary classification."""
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve only for binary classification!")

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color='#3B82F6', width=2)
        ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )

        return fig

    def plot_feature_importance(self, importance_dict: Dict[str, float],
                                 top_n: int = 20,
                                 title: str = "Feature Importance") -> go.Figure:
        """
        Plot feature importance from XGBoost or similar.

        Args:
            importance_dict: Dict mapping feature names to importance scores
            top_n: Number of top features to show
        """
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=list(features),
            x=list(importances),
            orientation='h',
            marker_color='#3B82F6'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, 20 * top_n),
            yaxis=dict(autorange="reversed")
        )

        return fig
