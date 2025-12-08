# =============================================================================
# app_core/optimization/__init__.py
# Staff Scheduling Optimization Module for HealthForecast AI
# Implements Deterministic MILP with Clinical Category Integration
# =============================================================================

from .cost_parameters import (
    CostParameters,
    StaffingRatios,
    DEFAULT_COST_PARAMS,
    DEFAULT_STAFFING_RATIOS,
    CLINICAL_CATEGORIES,
    CATEGORY_PRIORITIES,
)

from .milp_solver import (
    StaffSchedulingMILP,
    OptimizationResult,
    solve_staff_scheduling,
)

from .solution_analyzer import (
    SolutionAnalyzer,
    analyze_historical_costs,
    compare_before_after,
)

__all__ = [
    # Cost Parameters
    "CostParameters",
    "StaffingRatios",
    "DEFAULT_COST_PARAMS",
    "DEFAULT_STAFFING_RATIOS",
    "CLINICAL_CATEGORIES",
    "CATEGORY_PRIORITIES",
    # MILP Solver
    "StaffSchedulingMILP",
    "OptimizationResult",
    "solve_staff_scheduling",
    # Solution Analysis
    "SolutionAnalyzer",
    "analyze_historical_costs",
    "compare_before_after",
]
