# =============================================================================
# app_core/optimization/milp_solver.py
# Deterministic MILP Solver for Staff Scheduling Optimization
# Implements the mathematical formulation with clinical category integration
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from .cost_parameters import (
    CostParameters,
    StaffingRatios,
    DEFAULT_COST_PARAMS,
    DEFAULT_STAFFING_RATIOS,
    CLINICAL_CATEGORIES,
    CATEGORY_PRIORITIES,
    STAFF_TYPES,
)


# =============================================================================
# OPTIMIZATION RESULT DATA CLASS
# =============================================================================

@dataclass
class OptimizationResult:
    """Container for optimization results."""

    # Status
    status: str = "Not Solved"
    solve_time: float = 0.0
    is_optimal: bool = False

    # Objective value
    total_cost: float = 0.0

    # Cost breakdown
    regular_labor_cost: float = 0.0
    overtime_cost: float = 0.0
    understaffing_penalty: float = 0.0
    overstaffing_penalty: float = 0.0

    # Decision variables (DataFrames)
    staff_schedule: Optional[pd.DataFrame] = None      # x[k,t] - staff assignments
    overtime_schedule: Optional[pd.DataFrame] = None   # o[k,t] - overtime hours
    understaffing: Optional[pd.DataFrame] = None       # u[c,t] - unmet demand by category
    overstaffing: Optional[pd.DataFrame] = None        # v[t] - excess capacity

    # Coverage analysis
    category_coverage: Optional[pd.DataFrame] = None   # Coverage % per category
    daily_summary: Optional[pd.DataFrame] = None       # Daily totals

    # Model info
    num_variables: int = 0
    num_constraints: int = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "Status": self.status,
            "Is Optimal": self.is_optimal,
            "Total Cost": f"${self.total_cost:,.2f}",
            "Regular Labor": f"${self.regular_labor_cost:,.2f}",
            "Overtime Cost": f"${self.overtime_cost:,.2f}",
            "Understaffing Penalty": f"${self.understaffing_penalty:,.2f}",
            "Overstaffing Penalty": f"${self.overstaffing_penalty:,.2f}",
            "Solve Time": f"{self.solve_time:.3f}s",
            "Variables": self.num_variables,
            "Constraints": self.num_constraints,
        }


# =============================================================================
# MILP SOLVER CLASS
# =============================================================================

class StaffSchedulingMILP:
    """
    Deterministic MILP Solver for Hospital Staff Scheduling.

    Objective:
        min Z = Σ_t Σ_k (c_k · H · x[k,t])           # Regular labor cost
              + Σ_t Σ_k (c_k^o · o[k,t])             # Overtime cost
              + Σ_t Σ_c (w_c · c^u · u[c,t])         # Category-weighted understaffing
              + Σ_t (c^v · v[t])                      # Overstaffing penalty

    Subject to:
        C1: Demand Satisfaction (per category)
        C2: Staff Availability (min/max)
        C3: Overtime Limits
        C4: Skill Mix Ratio (nurse-to-doctor)
        C5: Non-negativity and Integrality
    """

    def __init__(
        self,
        cost_params: Optional[CostParameters] = None,
        staffing_ratios: Optional[StaffingRatios] = None,
    ):
        """
        Initialize the MILP solver.

        Args:
            cost_params: Cost parameters (USD)
            staffing_ratios: Category-specific staffing ratios
        """
        if not PULP_AVAILABLE:
            raise ImportError(
                "PuLP is not installed. Run: pip install pulp"
            )

        self.cost_params = cost_params or DEFAULT_COST_PARAMS
        self.staffing_ratios = staffing_ratios or DEFAULT_STAFFING_RATIOS

        # Model components (set during build)
        self.model: Optional[pulp.LpProblem] = None
        self.x: Dict = {}  # Staff assignments x[k,t]
        self.o: Dict = {}  # Overtime o[k,t]
        self.u: Dict = {}  # Understaffing u[c,t]
        self.v: Dict = {}  # Overstaffing v[t]

        # Problem data
        self.planning_horizon: int = 7
        self.demand_forecast: Optional[pd.DataFrame] = None
        self.category_demand: Optional[Dict[str, List[float]]] = None

    def set_demand_forecast(
        self,
        total_demand: List[float],
        category_proportions: Optional[Dict[str, float]] = None,
        category_demand: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Set demand forecast for optimization.

        Args:
            total_demand: Total patient arrivals per day [D_1, D_2, ..., D_T]
            category_proportions: Optional dict of category -> proportion (sum to 1)
            category_demand: Optional dict of category -> [demand per day]
                            If provided, overrides proportions
        """
        self.planning_horizon = len(total_demand)

        if category_demand is not None:
            # Use provided category-level demand
            self.category_demand = category_demand
        elif category_proportions is not None:
            # Distribute total demand by proportions
            self.category_demand = {}
            for category in CLINICAL_CATEGORIES:
                prop = category_proportions.get(category, 0.0)
                self.category_demand[category] = [d * prop for d in total_demand]
        else:
            # Default: equal distribution across categories
            n_categories = len(CLINICAL_CATEGORIES)
            self.category_demand = {}
            for category in CLINICAL_CATEGORIES:
                self.category_demand[category] = [d / n_categories for d in total_demand]

        # Store total demand
        self.demand_forecast = pd.DataFrame({
            "Day": list(range(1, self.planning_horizon + 1)),
            "Total_Demand": total_demand,
        })

        # Add category columns
        for category in CLINICAL_CATEGORIES:
            self.demand_forecast[category] = self.category_demand[category]

    def build_model(self) -> pulp.LpProblem:
        """Build the MILP model."""

        if self.category_demand is None:
            raise ValueError("Demand forecast not set. Call set_demand_forecast() first.")

        T = self.planning_horizon
        K = STAFF_TYPES
        C = CLINICAL_CATEGORIES
        H = self.cost_params.shift_length

        # Create the model
        self.model = pulp.LpProblem("Staff_Scheduling_Optimization", pulp.LpMinimize)

        # =================================================================
        # DECISION VARIABLES
        # =================================================================

        # x[k,t] - Number of regular staff of type k on day t (Integer)
        self.x = {
            (k, t): pulp.LpVariable(
                f"x_{k}_{t}",
                lowBound=self.staffing_ratios.get_min_staff(k),
                upBound=self.staffing_ratios.get_max_staff(k),
                cat=pulp.LpInteger
            )
            for k in K for t in range(T)
        }

        # o[k,t] - Overtime hours for staff type k on day t (Continuous)
        self.o = {
            (k, t): pulp.LpVariable(
                f"o_{k}_{t}",
                lowBound=0,
                cat=pulp.LpContinuous
            )
            for k in K for t in range(T)
        }

        # u[c,t] - Unmet demand for category c on day t (Continuous, slack)
        self.u = {
            (c, t): pulp.LpVariable(
                f"u_{c}_{t}",
                lowBound=0,
                cat=pulp.LpContinuous
            )
            for c in C for t in range(T)
        }

        # v[t] - Overstaffing slack on day t (Continuous)
        self.v = {
            t: pulp.LpVariable(
                f"v_{t}",
                lowBound=0,
                cat=pulp.LpContinuous
            )
            for t in range(T)
        }

        # =================================================================
        # OBJECTIVE FUNCTION
        # =================================================================

        # Regular labor cost: Σ_t Σ_k (c_k · H · x[k,t])
        regular_cost = pulp.lpSum(
            self.cost_params.get_hourly_rate(k) * H * self.x[k, t]
            for k in K for t in range(T)
        )

        # Overtime cost: Σ_t Σ_k (c_k^o · o[k,t])
        overtime_cost = pulp.lpSum(
            self.cost_params.get_overtime_rate(k) * self.o[k, t]
            for k in K for t in range(T)
        )

        # Understaffing penalty: Σ_t Σ_c (w_c · c^u · u[c,t])
        understaffing_penalty = pulp.lpSum(
            self.cost_params.get_understaffing_penalty(c) * self.u[c, t]
            for c in C for t in range(T)
        )

        # Overstaffing penalty: Σ_t (c^v · v[t])
        overstaffing_penalty = pulp.lpSum(
            self.cost_params.overstaffing_penalty * self.v[t]
            for t in range(T)
        )

        # Total objective
        self.model += (
            regular_cost + overtime_cost + understaffing_penalty + overstaffing_penalty,
            "Total_Cost"
        )

        # =================================================================
        # CONSTRAINTS
        # =================================================================

        # C1: Demand Satisfaction (per category, per day)
        # Service capacity >= Demand - Understaffing slack
        for c in C:
            for t in range(T):
                demand_ct = self.category_demand[c][t]

                # Calculate service capacity for this category
                # Capacity = Σ_k (service_rate[k,c] · H · x[k,t] + service_rate[k,c] · o[k,t])
                capacity = pulp.lpSum(
                    self.staffing_ratios.get_service_rate(c, k, H) * H * self.x[k, t]
                    + self.staffing_ratios.get_service_rate(c, k, H) * self.o[k, t]
                    for k in K
                )

                self.model += (
                    capacity + self.u[c, t] >= demand_ct,
                    f"Demand_{c}_{t}"
                )

        # C2: Overstaffing calculation (total capacity vs total demand)
        for t in range(T):
            total_demand_t = sum(self.category_demand[c][t] for c in C)

            # Total capacity across all staff
            total_capacity = pulp.lpSum(
                # Average service rate across categories
                (sum(self.staffing_ratios.get_service_rate(c, k, H) for c in C) / len(C))
                * H * self.x[k, t]
                for k in K
            )

            self.model += (
                self.v[t] >= total_capacity - total_demand_t,
                f"Overstaffing_{t}"
            )

        # C3: Overtime Limits
        # o[k,t] <= max_overtime * x[k,t]
        for k in K:
            for t in range(T):
                self.model += (
                    self.o[k, t] <= self.cost_params.max_overtime_hours * self.x[k, t],
                    f"Overtime_Limit_{k}_{t}"
                )

        # C4: Skill Mix Ratio (Nurses >= ratio * Doctors)
        for t in range(T):
            self.model += (
                self.x["Nurses", t] >= self.staffing_ratios.nurse_to_doctor_ratio * self.x["Doctors", t],
                f"Skill_Mix_{t}"
            )

        return self.model

    def solve(self, time_limit: int = 60, gap: float = 0.01) -> OptimizationResult:
        """
        Solve the MILP model.

        Args:
            time_limit: Maximum solve time in seconds
            gap: Optimality gap tolerance (default 1%)

        Returns:
            OptimizationResult with solution details
        """
        import time

        if self.model is None:
            self.build_model()

        result = OptimizationResult()
        result.num_variables = len(self.model.variables())
        result.num_constraints = len(self.model.constraints)

        # Solve
        start_time = time.time()

        # Use CBC solver (built into PuLP)
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=gap,
            msg=False  # Suppress solver output
        )

        self.model.solve(solver)

        result.solve_time = time.time() - start_time
        result.status = pulp.LpStatus[self.model.status]
        result.is_optimal = self.model.status == pulp.LpStatusOptimal

        if result.is_optimal or self.model.status == pulp.LpStatusNotSolved:
            # Extract solution
            result = self._extract_solution(result)

        return result

    def _extract_solution(self, result: OptimizationResult) -> OptimizationResult:
        """Extract solution values from solved model."""

        T = self.planning_horizon
        K = STAFF_TYPES
        C = CLINICAL_CATEGORIES
        H = self.cost_params.shift_length

        # =================================================================
        # EXTRACT STAFF SCHEDULE
        # =================================================================
        staff_data = []
        for t in range(T):
            row = {"Day": t + 1}
            for k in K:
                row[k] = int(pulp.value(self.x[k, t]) or 0)
            staff_data.append(row)

        result.staff_schedule = pd.DataFrame(staff_data)

        # =================================================================
        # EXTRACT OVERTIME SCHEDULE
        # =================================================================
        overtime_data = []
        for t in range(T):
            row = {"Day": t + 1}
            for k in K:
                row[f"{k}_OT"] = round(pulp.value(self.o[k, t]) or 0, 2)
            overtime_data.append(row)

        result.overtime_schedule = pd.DataFrame(overtime_data)

        # =================================================================
        # EXTRACT UNDERSTAFFING (by category)
        # =================================================================
        understaffing_data = []
        for t in range(T):
            row = {"Day": t + 1}
            for c in C:
                row[c] = round(pulp.value(self.u[c, t]) or 0, 2)
            understaffing_data.append(row)

        result.understaffing = pd.DataFrame(understaffing_data)

        # =================================================================
        # EXTRACT OVERSTAFFING
        # =================================================================
        overstaffing_data = []
        for t in range(T):
            overstaffing_data.append({
                "Day": t + 1,
                "Overstaffing": round(pulp.value(self.v[t]) or 0, 2)
            })

        result.overstaffing = pd.DataFrame(overstaffing_data)

        # =================================================================
        # CALCULATE COST BREAKDOWN
        # =================================================================

        # Regular labor cost
        result.regular_labor_cost = sum(
            self.cost_params.get_hourly_rate(k) * H * (pulp.value(self.x[k, t]) or 0)
            for k in K for t in range(T)
        )

        # Overtime cost
        result.overtime_cost = sum(
            self.cost_params.get_overtime_rate(k) * (pulp.value(self.o[k, t]) or 0)
            for k in K for t in range(T)
        )

        # Understaffing penalty
        result.understaffing_penalty = sum(
            self.cost_params.get_understaffing_penalty(c) * (pulp.value(self.u[c, t]) or 0)
            for c in C for t in range(T)
        )

        # Overstaffing penalty
        result.overstaffing_penalty = sum(
            self.cost_params.overstaffing_penalty * (pulp.value(self.v[t]) or 0)
            for t in range(T)
        )

        # Total cost
        result.total_cost = (
            result.regular_labor_cost +
            result.overtime_cost +
            result.understaffing_penalty +
            result.overstaffing_penalty
        )

        # =================================================================
        # CATEGORY COVERAGE ANALYSIS
        # =================================================================
        coverage_data = []
        for c in C:
            total_demand = sum(self.category_demand[c])
            total_unmet = sum(pulp.value(self.u[c, t]) or 0 for t in range(T))
            coverage_pct = ((total_demand - total_unmet) / total_demand * 100) if total_demand > 0 else 100

            coverage_data.append({
                "Category": c,
                "Total_Demand": round(total_demand, 1),
                "Unmet_Demand": round(total_unmet, 1),
                "Coverage_%": round(coverage_pct, 1),
                "Priority": CATEGORY_PRIORITIES.get(c, 1.0),
            })

        result.category_coverage = pd.DataFrame(coverage_data)

        # =================================================================
        # DAILY SUMMARY
        # =================================================================
        daily_data = []
        for t in range(T):
            total_staff = sum(pulp.value(self.x[k, t]) or 0 for k in K)
            total_overtime = sum(pulp.value(self.o[k, t]) or 0 for k in K)
            total_demand = sum(self.category_demand[c][t] for c in C)
            total_unmet = sum(pulp.value(self.u[c, t]) or 0 for c in C)

            daily_data.append({
                "Day": t + 1,
                "Total_Staff": int(total_staff),
                "Total_OT_Hours": round(total_overtime, 1),
                "Patient_Demand": round(total_demand, 1),
                "Unmet_Demand": round(total_unmet, 1),
                "Coverage_%": round((total_demand - total_unmet) / total_demand * 100, 1) if total_demand > 0 else 100,
            })

        result.daily_summary = pd.DataFrame(daily_data)

        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def solve_staff_scheduling(
    total_demand: List[float],
    category_proportions: Optional[Dict[str, float]] = None,
    category_demand: Optional[Dict[str, List[float]]] = None,
    cost_params: Optional[CostParameters] = None,
    staffing_ratios: Optional[StaffingRatios] = None,
    time_limit: int = 60,
) -> OptimizationResult:
    """
    Convenience function to solve staff scheduling problem.

    Args:
        total_demand: Total patient arrivals per day
        category_proportions: Optional category proportions
        category_demand: Optional category-level demand (overrides proportions)
        cost_params: Cost parameters
        staffing_ratios: Staffing ratios
        time_limit: Maximum solve time in seconds

    Returns:
        OptimizationResult with solution
    """
    solver = StaffSchedulingMILP(cost_params, staffing_ratios)
    solver.set_demand_forecast(total_demand, category_proportions, category_demand)
    solver.build_model()
    return solver.solve(time_limit=time_limit)
