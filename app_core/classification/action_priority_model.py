# =============================================================================
# app_core/classification/action_priority_model.py
# ML-Based Action Priority Classification (Placeholder)
# Classifies optimization recommendations into priority levels
# Training deferred until historical outcome data is available
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from enum import Enum


# =============================================================================
# PRIORITY LEVELS
# =============================================================================

class ActionPriority(Enum):
    """Priority levels for optimization actions."""
    CRITICAL = 4  # Immediate action required
    HIGH = 3      # Action within 24 hours
    MEDIUM = 2    # Action within 1 week
    LOW = 1       # Informational, no immediate action

    @classmethod
    def from_string(cls, s: str) -> "ActionPriority":
        """Convert string to ActionPriority."""
        mapping = {
            "CRITICAL": cls.CRITICAL,
            "HIGH": cls.HIGH,
            "MEDIUM": cls.MEDIUM,
            "LOW": cls.LOW,
        }
        return mapping.get(s.upper(), cls.MEDIUM)

    def to_color(self) -> str:
        """Get display color for priority."""
        colors = {
            self.CRITICAL: "red",
            self.HIGH: "orange",
            self.MEDIUM: "yellow",
            self.LOW: "green",
        }
        return colors.get(self, "gray")


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

@dataclass
class ActionPriorityFeatures:
    """
    Features used for action priority classification.

    Attributes:
        cost_impact_pct: Percentage cost change (negative = savings)
        urgency_score: Time-sensitivity score (0-1)
        resource_utilization: Current resource utilization (0-1)
        demand_volatility: Forecast uncertainty (RPIW)
        shortage_days: Number of days with shortages
        service_level: Current service level (0-1)
        budget_headroom: Budget remaining vs limit (0-1)
        category_priority: Clinical category weight
    """
    cost_impact_pct: float = 0.0
    urgency_score: float = 0.0
    resource_utilization: float = 0.0
    demand_volatility: float = 0.0
    shortage_days: int = 0
    service_level: float = 1.0
    budget_headroom: float = 1.0
    category_priority: float = 1.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.cost_impact_pct,
            self.urgency_score,
            self.resource_utilization,
            self.demand_volatility,
            self.shortage_days,
            self.service_level,
            self.budget_headroom,
            self.category_priority,
        ])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "cost_impact_pct": self.cost_impact_pct,
            "urgency_score": self.urgency_score,
            "resource_utilization": self.resource_utilization,
            "demand_volatility": self.demand_volatility,
            "shortage_days": float(self.shortage_days),
            "service_level": self.service_level,
            "budget_headroom": self.budget_headroom,
            "category_priority": self.category_priority,
        }


# =============================================================================
# ACTION PRIORITY CLASSIFIER
# =============================================================================

class ActionPriorityClassifier:
    """
    XGBoost-based classifier for action priority prediction.

    This classifier predicts the priority level (CRITICAL, HIGH, MEDIUM, LOW)
    for optimization recommendations based on various input features.

    IMPORTANT: Training is deferred until historical outcome data is available.
    Until then, the classifier uses a rule-based fallback that mimics the
    expected model behavior.

    Usage:
        classifier = ActionPriorityClassifier()

        # Get priority for an action (uses rule-based fallback)
        features = ActionPriorityFeatures(
            cost_impact_pct=-15.0,  # 15% savings
            shortage_days=3,
            service_level=0.92
        )
        priority = classifier.predict_single(features)

        # Get priority with confidence
        priority, confidence = classifier.predict_with_confidence(features)

        # Future: Train when data is available
        # classifier.train(X_train, y_train)
    """

    # Feature names for model training
    FEATURE_NAMES = [
        "cost_impact_pct",
        "urgency_score",
        "resource_utilization",
        "demand_volatility",
        "shortage_days",
        "service_level",
        "budget_headroom",
        "category_priority",
    ]

    # Priority labels
    PRIORITY_LABELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self._training_history: List[Dict[str, Any]] = []

    @property
    def model_status(self) -> str:
        """Get model status string."""
        if self.is_trained:
            return "Trained"
        return "Using rule-based fallback (training data not available)"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the classifier on historical data.

        DEFERRED: This method is a placeholder for future training when
        historical outcome data becomes available.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Priority labels (n_samples,)
            validation_split: Fraction of data for validation

        Returns:
            Training history with metrics

        Raises:
            NotImplementedError: Until training data is available
        """
        # Check if we have enough data
        if len(X) < 50:
            raise ValueError(
                f"Insufficient training data. Need at least 50 samples, got {len(X)}. "
                "Training is deferred until historical outcome data is available."
            )

        # Import XGBoost (lazy import to avoid dependency issues)
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
        except ImportError:
            raise ImportError(
                "XGBoost is required for training. Install with: pip install xgboost"
            )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )

        # Initialize and train model
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        self.is_trained = True

        history = {
            "accuracy": accuracy,
            "classification_report": report,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
        }
        self._training_history.append(history)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict priority levels for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of priority labels
        """
        if not self.is_trained:
            return self._rule_based_batch(X)
        return self.model.predict(X)

    def predict_single(self, features: ActionPriorityFeatures) -> ActionPriority:
        """
        Predict priority for a single action.

        Args:
            features: Action features

        Returns:
            ActionPriority enum value
        """
        X = features.to_array().reshape(1, -1)

        if not self.is_trained:
            label = self._rule_based_single(features)
        else:
            label = self.model.predict(X)[0]

        return ActionPriority.from_string(label)

    def predict_with_confidence(
        self,
        features: ActionPriorityFeatures,
    ) -> Tuple[ActionPriority, float]:
        """
        Predict priority with confidence score.

        Args:
            features: Action features

        Returns:
            Tuple of (ActionPriority, confidence_score)
        """
        X = features.to_array().reshape(1, -1)

        if not self.is_trained:
            label = self._rule_based_single(features)
            # Rule-based confidence based on how clearly rules apply
            confidence = self._calculate_rule_confidence(features)
        else:
            label = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))

        return ActionPriority.from_string(label), confidence

    def _rule_based_single(self, features: ActionPriorityFeatures) -> str:
        """
        Apply rule-based classification for a single sample.

        This replicates the logic from BaseRecommendationEngine._get_priority_from_impact()
        but with more sophisticated multi-factor rules.
        """
        # CRITICAL: High impact or immediate risk
        if (
            abs(features.cost_impact_pct) > 25 or
            features.shortage_days >= 5 or
            features.service_level < 0.85 or
            features.resource_utilization > 0.95
        ):
            return "CRITICAL"

        # HIGH: Significant impact or elevated risk
        if (
            abs(features.cost_impact_pct) > 15 or
            features.shortage_days >= 3 or
            features.service_level < 0.90 or
            features.resource_utilization > 0.90 or
            features.urgency_score > 0.7 or
            features.category_priority >= 2.5
        ):
            return "HIGH"

        # MEDIUM: Moderate impact
        if (
            abs(features.cost_impact_pct) > 5 or
            features.shortage_days >= 1 or
            features.service_level < 0.95 or
            features.resource_utilization > 0.80 or
            features.budget_headroom < 0.2
        ):
            return "MEDIUM"

        # LOW: Minor or informational
        return "LOW"

    def _rule_based_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply rule-based classification for batch of samples."""
        predictions = []
        for row in X:
            features = ActionPriorityFeatures(
                cost_impact_pct=row[0],
                urgency_score=row[1],
                resource_utilization=row[2],
                demand_volatility=row[3],
                shortage_days=int(row[4]),
                service_level=row[5],
                budget_headroom=row[6],
                category_priority=row[7],
            )
            predictions.append(self._rule_based_single(features))
        return np.array(predictions)

    def _calculate_rule_confidence(self, features: ActionPriorityFeatures) -> float:
        """
        Calculate confidence for rule-based prediction.

        Higher confidence when features clearly indicate a priority level.
        """
        # Calculate how strongly each factor indicates the assigned priority
        factors = []

        # Cost impact factor
        impact = abs(features.cost_impact_pct)
        if impact > 25:
            factors.append(1.0)  # Very clear
        elif impact > 15:
            factors.append(0.85)
        elif impact > 5:
            factors.append(0.70)
        else:
            factors.append(0.55)

        # Service level factor
        sl = features.service_level
        if sl < 0.85:
            factors.append(1.0)
        elif sl < 0.90:
            factors.append(0.85)
        elif sl < 0.95:
            factors.append(0.70)
        else:
            factors.append(0.55)

        # Shortage factor
        if features.shortage_days >= 5:
            factors.append(1.0)
        elif features.shortage_days >= 3:
            factors.append(0.85)
        elif features.shortage_days >= 1:
            factors.append(0.70)
        else:
            factors.append(0.60)

        # Average confidence
        return sum(factors) / len(factors)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from trained model.

        Returns:
            Dict mapping feature names to importance scores, or None if not trained
        """
        if not self.is_trained:
            # Return rule-based importance (approximation)
            return {
                "cost_impact_pct": 0.25,
                "urgency_score": 0.10,
                "resource_utilization": 0.15,
                "demand_volatility": 0.05,
                "shortage_days": 0.20,
                "service_level": 0.15,
                "budget_headroom": 0.05,
                "category_priority": 0.05,
            }

        importances = self.model.feature_importances_
        return dict(zip(self.FEATURE_NAMES, importances))

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model state."""
        return {
            "is_trained": self.is_trained,
            "status": self.model_status,
            "feature_names": self.FEATURE_NAMES,
            "priority_labels": self.PRIORITY_LABELS,
            "feature_importance": self.get_feature_importance(),
            "training_history": self._training_history,
        }


# =============================================================================
# FEATURE EXTRACTION UTILITIES
# =============================================================================

def extract_staff_action_features(
    optimization_results: Dict[str, Any],
    forecast_data: Dict[str, Any],
    cost_params: Dict[str, Any],
) -> ActionPriorityFeatures:
    """
    Extract features from staff optimization results.

    Args:
        optimization_results: Results from MILP solver
        forecast_data: Forecast data with uncertainty
        cost_params: Cost parameters

    Returns:
        ActionPriorityFeatures for classification
    """
    # Cost impact
    total_cost = optimization_results.get("total_cost", 0)
    baseline_cost = optimization_results.get("baseline_cost", total_cost)
    if baseline_cost > 0:
        cost_impact_pct = ((total_cost - baseline_cost) / baseline_cost) * 100
    else:
        cost_impact_pct = 0

    # Understaffing as shortage indicator
    understaffing = optimization_results.get("understaffing_penalty", 0)
    shortage_days = int(understaffing / 200)  # Rough estimate

    # Service level from coverage
    coverage_df = optimization_results.get("category_coverage")
    if coverage_df is not None and len(coverage_df) > 0:
        avg_coverage = coverage_df["Coverage_%"].mean() / 100
    else:
        avg_coverage = 1.0

    # Demand volatility from forecast
    rpiw = forecast_data.get("rpiw", 0)
    demand_volatility = min(rpiw / 100, 1.0)  # Normalize to 0-1

    # Budget headroom
    daily_budget = cost_params.get("daily_budget_cap", 50000)
    daily_cost = optimization_results.get("regular_labor_cost", 0) / 7
    budget_headroom = max(0, (daily_budget - daily_cost) / daily_budget)

    return ActionPriorityFeatures(
        cost_impact_pct=cost_impact_pct,
        urgency_score=1 - avg_coverage,  # Higher urgency if lower coverage
        resource_utilization=min(daily_cost / daily_budget, 1.0),
        demand_volatility=demand_volatility,
        shortage_days=shortage_days,
        service_level=avg_coverage,
        budget_headroom=budget_headroom,
        category_priority=1.5,  # Default medium priority
    )


def extract_inventory_action_features(
    optimization_results: Dict[str, Any],
    forecast_data: Dict[str, Any],
    inv_params: Dict[str, Any],
) -> ActionPriorityFeatures:
    """
    Extract features from inventory optimization results.

    Args:
        optimization_results: Results from inventory optimizer
        forecast_data: Forecast data with uncertainty
        inv_params: Inventory parameters

    Returns:
        ActionPriorityFeatures for classification
    """
    # Cost impact
    total_cost = optimization_results.get("total_cost", 0)
    baseline_cost = optimization_results.get("baseline_cost", total_cost)
    if baseline_cost > 0:
        cost_impact_pct = ((total_cost - baseline_cost) / baseline_cost) * 100
    else:
        cost_impact_pct = 0

    # Stockout as shortage indicator
    stockout_cost = optimization_results.get("stockout_cost", 0)
    shortage_days = int(stockout_cost / 100)  # Rough estimate

    # Service level
    service_level = optimization_results.get("service_level", 0.95)

    # Demand volatility from forecast
    rpiw = forecast_data.get("rpiw", 0)
    demand_volatility = min(rpiw / 100, 1.0)

    # Budget headroom
    daily_budget = inv_params.get("daily_budget", 10000) or 10000
    purchase_cost = optimization_results.get("purchase_cost", 0)
    avg_daily_cost = purchase_cost / inv_params.get("planning_horizon", 30)
    budget_headroom = max(0, (daily_budget - avg_daily_cost) / daily_budget)

    return ActionPriorityFeatures(
        cost_impact_pct=cost_impact_pct,
        urgency_score=1 - service_level,
        resource_utilization=1 - budget_headroom,
        demand_volatility=demand_volatility,
        shortage_days=shortage_days,
        service_level=service_level,
        budget_headroom=budget_headroom,
        category_priority=2.0,  # Inventory typically high priority
    )


# =============================================================================
# GLOBAL CLASSIFIER INSTANCE
# =============================================================================

# Singleton instance for reuse across pages
_classifier_instance: Optional[ActionPriorityClassifier] = None


def get_classifier() -> ActionPriorityClassifier:
    """
    Get or create the global classifier instance.

    Returns:
        ActionPriorityClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ActionPriorityClassifier()
    return _classifier_instance
