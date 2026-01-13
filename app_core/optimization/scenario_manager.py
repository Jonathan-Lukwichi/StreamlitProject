# =============================================================================
# app_core/optimization/scenario_manager.py
# Scenario Management for What-If Analysis
# Allows comparing multiple optimization scenarios with different constraints
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

from .constraint_config import StaffConstraintConfig, InventoryConstraintConfig


# =============================================================================
# OPTIMIZATION SCENARIO
# =============================================================================

@dataclass
class OptimizationScenario:
    """
    Container for a single optimization scenario.

    Attributes:
        name: Unique scenario name
        description: Human-readable description
        constraints: Constraint configuration (staff or inventory)
        results: Optimization results (populated after solving)
        created_at: Timestamp when scenario was created
        solved_at: Timestamp when scenario was solved (None if not solved)
        is_baseline: Whether this is the baseline scenario for comparison
    """
    name: str
    description: str
    constraints: Union[StaffConstraintConfig, InventoryConstraintConfig]
    results: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    solved_at: Optional[datetime] = None
    is_baseline: bool = False

    @property
    def is_solved(self) -> bool:
        """Check if scenario has been solved."""
        return self.results is not None and self.solved_at is not None

    @property
    def scenario_type(self) -> str:
        """Get scenario type based on constraint type."""
        if isinstance(self.constraints, StaffConstraintConfig):
            return "staff"
        elif isinstance(self.constraints, InventoryConstraintConfig):
            return "inventory"
        return "unknown"

    def set_results(self, results: Dict[str, Any]) -> None:
        """Set optimization results and mark as solved."""
        self.results = results
        self.solved_at = datetime.now()

    def get_key_metrics(self) -> Dict[str, Any]:
        """Extract key metrics from results for comparison."""
        if not self.is_solved:
            return {}

        metrics = {
            "scenario_name": self.name,
            "scenario_type": self.scenario_type,
        }

        if self.scenario_type == "staff":
            metrics.update({
                "total_cost": self.results.get("total_cost", 0),
                "regular_labor_cost": self.results.get("regular_labor_cost", 0),
                "overtime_cost": self.results.get("overtime_cost", 0),
                "understaffing_penalty": self.results.get("understaffing_penalty", 0),
                "overstaffing_penalty": self.results.get("overstaffing_penalty", 0),
                "avg_coverage": self.results.get("avg_coverage", 0),
                "status": self.results.get("status", "Unknown"),
            })
        elif self.scenario_type == "inventory":
            metrics.update({
                "total_cost": self.results.get("total_cost", 0),
                "ordering_cost": self.results.get("ordering_cost", 0),
                "holding_cost": self.results.get("holding_cost", 0),
                "stockout_cost": self.results.get("stockout_cost", 0),
                "purchase_cost": self.results.get("purchase_cost", 0),
                "service_level": self.results.get("service_level", 0),
                "status": self.results.get("status", "Unknown"),
            })

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "constraints": self.constraints.to_dict(),
            "results": self.results,
            "created_at": self.created_at.isoformat(),
            "solved_at": self.solved_at.isoformat() if self.solved_at else None,
            "is_baseline": self.is_baseline,
            "scenario_type": self.scenario_type,
        }


# =============================================================================
# SCENARIO MANAGER
# =============================================================================

class ScenarioManager:
    """
    Manager for creating, storing, and comparing optimization scenarios.

    Usage:
        manager = ScenarioManager()

        # Create scenarios
        baseline = manager.create_scenario(
            "Baseline",
            "Current constraints",
            StaffConstraintConfig(),
            is_baseline=True
        )

        # Add more scenarios
        aggressive = manager.create_scenario(
            "Aggressive",
            "Reduced staffing levels",
            StaffConstraintConfig(min_doctors=3, min_nurses=10)
        )

        # After running optimization...
        manager.set_scenario_results("Baseline", results_baseline)
        manager.set_scenario_results("Aggressive", results_aggressive)

        # Compare scenarios
        comparison_df = manager.compare_scenarios(["Baseline", "Aggressive"])
    """

    def __init__(self, scenario_type: str = "staff"):
        """
        Initialize scenario manager.

        Args:
            scenario_type: Type of scenarios ("staff" or "inventory")
        """
        self.scenario_type = scenario_type
        self.scenarios: Dict[str, OptimizationScenario] = {}
        self._baseline_name: Optional[str] = None

    def create_scenario(
        self,
        name: str,
        description: str,
        constraints: Union[StaffConstraintConfig, InventoryConstraintConfig],
        is_baseline: bool = False,
    ) -> OptimizationScenario:
        """
        Create and register a new scenario.

        Args:
            name: Unique scenario name
            description: Human-readable description
            constraints: Constraint configuration
            is_baseline: Whether this is the baseline scenario

        Returns:
            Created OptimizationScenario
        """
        if name in self.scenarios:
            raise ValueError(f"Scenario '{name}' already exists")

        # If this is baseline, unset previous baseline
        if is_baseline:
            if self._baseline_name and self._baseline_name in self.scenarios:
                self.scenarios[self._baseline_name].is_baseline = False
            self._baseline_name = name

        scenario = OptimizationScenario(
            name=name,
            description=description,
            constraints=constraints,
            is_baseline=is_baseline,
        )

        self.scenarios[name] = scenario
        return scenario

    def get_scenario(self, name: str) -> Optional[OptimizationScenario]:
        """Get scenario by name."""
        return self.scenarios.get(name)

    def delete_scenario(self, name: str) -> bool:
        """
        Delete a scenario.

        Args:
            name: Scenario name

        Returns:
            True if deleted, False if not found
        """
        if name in self.scenarios:
            if self.scenarios[name].is_baseline:
                self._baseline_name = None
            del self.scenarios[name]
            return True
        return False

    def set_scenario_results(self, name: str, results: Dict[str, Any]) -> bool:
        """
        Set optimization results for a scenario.

        Args:
            name: Scenario name
            results: Optimization results

        Returns:
            True if successful, False if scenario not found
        """
        if name in self.scenarios:
            self.scenarios[name].set_results(results)
            return True
        return False

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """
        List all scenarios with summary info.

        Returns:
            List of scenario summaries
        """
        summaries = []
        for name, scenario in self.scenarios.items():
            summaries.append({
                "name": name,
                "description": scenario.description,
                "is_baseline": scenario.is_baseline,
                "is_solved": scenario.is_solved,
                "created_at": scenario.created_at,
                "scenario_type": scenario.scenario_type,
            })
        return summaries

    def get_baseline(self) -> Optional[OptimizationScenario]:
        """Get the baseline scenario."""
        if self._baseline_name:
            return self.scenarios.get(self._baseline_name)
        return None

    def compare_scenarios(
        self,
        scenario_names: Optional[List[str]] = None,
        include_baseline: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios.

        Args:
            scenario_names: List of scenario names to compare (None = all)
            include_baseline: Whether to include baseline if not in list

        Returns:
            DataFrame with comparison metrics
        """
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())

        # Add baseline if requested and not already included
        if include_baseline and self._baseline_name:
            if self._baseline_name not in scenario_names:
                scenario_names = [self._baseline_name] + scenario_names

        # Collect metrics from each scenario
        metrics_list = []
        for name in scenario_names:
            if name in self.scenarios and self.scenarios[name].is_solved:
                metrics = self.scenarios[name].get_key_metrics()
                metrics_list.append(metrics)

        if not metrics_list:
            return pd.DataFrame()

        df = pd.DataFrame(metrics_list)

        # Calculate delta from baseline if available
        if self._baseline_name and self._baseline_name in df["scenario_name"].values:
            baseline_idx = df[df["scenario_name"] == self._baseline_name].index[0]

            for col in df.select_dtypes(include=[np.number]).columns:
                baseline_val = df.loc[baseline_idx, col]
                if baseline_val != 0:
                    df[f"{col}_vs_baseline_%"] = (
                        (df[col] - baseline_val) / baseline_val * 100
                    ).round(2)

        return df

    def get_best_scenario(self, metric: str = "total_cost") -> Optional[str]:
        """
        Get the best scenario based on a metric.

        Args:
            metric: Metric to minimize (default: total_cost)

        Returns:
            Name of best scenario, or None if no solved scenarios
        """
        best_name = None
        best_value = float("inf")

        for name, scenario in self.scenarios.items():
            if scenario.is_solved:
                value = scenario.results.get(metric, float("inf"))
                if value < best_value:
                    best_value = value
                    best_name = name

        return best_name

    def generate_sensitivity_analysis(
        self,
        base_constraints: Union[StaffConstraintConfig, InventoryConstraintConfig],
        parameter: str,
        values: List[float],
        optimization_func: callable,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on a single parameter.

        Args:
            base_constraints: Base constraint configuration
            parameter: Parameter name to vary
            values: List of values to test
            optimization_func: Function to run optimization (constraints -> results)

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        for value in values:
            # Clone constraints and modify parameter
            constraints_dict = base_constraints.to_dict()
            constraints_dict[parameter] = value

            if isinstance(base_constraints, StaffConstraintConfig):
                constraints = StaffConstraintConfig.from_dict(constraints_dict)
            else:
                constraints = InventoryConstraintConfig.from_dict(constraints_dict)

            # Run optimization
            try:
                opt_results = optimization_func(constraints)
                results.append({
                    parameter: value,
                    "total_cost": opt_results.get("total_cost", 0),
                    "status": opt_results.get("status", "Unknown"),
                })
            except Exception as e:
                results.append({
                    parameter: value,
                    "total_cost": None,
                    "status": f"Error: {str(e)}",
                })

        return pd.DataFrame(results)

    def clear_all(self) -> None:
        """Clear all scenarios."""
        self.scenarios.clear()
        self._baseline_name = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert manager state to dictionary for serialization."""
        return {
            "scenario_type": self.scenario_type,
            "baseline_name": self._baseline_name,
            "scenarios": {
                name: scenario.to_dict()
                for name, scenario in self.scenarios.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioManager":
        """Create manager from dictionary."""
        manager = cls(scenario_type=data.get("scenario_type", "staff"))
        manager._baseline_name = data.get("baseline_name")

        for name, scenario_data in data.get("scenarios", {}).items():
            # Reconstruct constraints based on type
            constraints_data = scenario_data.get("constraints", {})
            scenario_type = scenario_data.get("scenario_type", "staff")

            if scenario_type == "staff":
                constraints = StaffConstraintConfig.from_dict(constraints_data)
            else:
                constraints = InventoryConstraintConfig.from_dict(constraints_data)

            scenario = OptimizationScenario(
                name=name,
                description=scenario_data.get("description", ""),
                constraints=constraints,
                results=scenario_data.get("results"),
                is_baseline=scenario_data.get("is_baseline", False),
            )

            # Parse timestamps
            created_at = scenario_data.get("created_at")
            if created_at:
                scenario.created_at = datetime.fromisoformat(created_at)

            solved_at = scenario_data.get("solved_at")
            if solved_at:
                scenario.solved_at = datetime.fromisoformat(solved_at)

            manager.scenarios[name] = scenario

        return manager


# =============================================================================
# SCENARIO COMPARISON UTILITIES
# =============================================================================

def format_comparison_table(
    comparison_df: pd.DataFrame,
    highlight_best: bool = True,
) -> pd.DataFrame:
    """
    Format comparison DataFrame for display.

    Args:
        comparison_df: Raw comparison DataFrame
        highlight_best: Whether to highlight best values

    Returns:
        Formatted DataFrame
    """
    if comparison_df.empty:
        return comparison_df

    # Create formatted copy
    formatted = comparison_df.copy()

    # Format currency columns
    currency_cols = [
        "total_cost", "regular_labor_cost", "overtime_cost",
        "understaffing_penalty", "overstaffing_penalty",
        "ordering_cost", "holding_cost", "stockout_cost", "purchase_cost"
    ]

    for col in currency_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )

    # Format percentage columns
    pct_cols = ["avg_coverage", "service_level"]
    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )

    # Format vs_baseline columns
    for col in formatted.columns:
        if col.endswith("_vs_baseline_%"):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
            )

    return formatted


def calculate_scenario_rankings(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Calculate overall scenario rankings based on multiple metrics.

    Args:
        comparison_df: Comparison DataFrame
        metrics: List of metrics to consider (default: total_cost only)
        weights: Metric weights (higher = more important)

    Returns:
        DataFrame with rankings
    """
    if comparison_df.empty:
        return comparison_df

    if metrics is None:
        metrics = ["total_cost"]

    if weights is None:
        weights = {m: 1.0 for m in metrics}

    # Calculate normalized scores for each metric (lower is better)
    scores = pd.DataFrame()
    scores["scenario_name"] = comparison_df["scenario_name"]

    for metric in metrics:
        if metric in comparison_df.columns:
            values = comparison_df[metric]
            min_val = values.min()
            max_val = values.max()

            if max_val > min_val:
                # Normalize 0-1 (lower is better)
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = pd.Series([0.5] * len(values))

            weight = weights.get(metric, 1.0)
            scores[f"{metric}_score"] = normalized * weight

    # Calculate total weighted score
    score_cols = [c for c in scores.columns if c.endswith("_score")]
    scores["total_score"] = scores[score_cols].sum(axis=1)

    # Rank (1 = best)
    scores["rank"] = scores["total_score"].rank(method="min")

    return scores.sort_values("rank")
