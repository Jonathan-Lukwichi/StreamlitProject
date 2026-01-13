# =============================================================================
# app_core/optimization/constraint_config.py
# Constraint Configuration for Staff Scheduling and Inventory Optimization
# Provides UI-friendly configuration that converts to solver parameters
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .cost_parameters import (
    CostParameters,
    StaffingRatios,
    CLINICAL_CATEGORIES,
    CATEGORY_PRIORITIES,
    STAFF_TYPES,
)


# =============================================================================
# Z-SCORES FOR SERVICE LEVELS
# =============================================================================

SERVICE_LEVEL_Z_SCORES: Dict[float, float] = {
    0.90: 1.282,
    0.91: 1.341,
    0.92: 1.405,
    0.93: 1.476,
    0.94: 1.555,
    0.95: 1.645,
    0.96: 1.751,
    0.97: 1.881,
    0.98: 2.054,
    0.99: 2.326,
    0.995: 2.576,
    0.999: 3.090,
}


def get_z_score(service_level: float) -> float:
    """
    Get Z-score for a given service level.

    Args:
        service_level: Target service level (0.0 to 1.0)

    Returns:
        Z-score for the service level
    """
    # Find closest match
    if service_level in SERVICE_LEVEL_Z_SCORES:
        return SERVICE_LEVEL_Z_SCORES[service_level]

    # Interpolate if not exact match
    from scipy import stats
    return stats.norm.ppf(service_level)


# =============================================================================
# STAFF CONSTRAINT CONFIGURATION
# =============================================================================

@dataclass
class StaffConstraintConfig:
    """
    Complete constraint configuration for staff scheduling optimization.

    This class provides a user-friendly interface for configuring all
    optimization constraints and converts to the solver's CostParameters
    and StaffingRatios classes.

    Attributes:
        # Budget Constraints
        daily_budget_cap: Maximum daily labor cost ($)
        weekly_budget_cap: Maximum weekly labor cost ($)

        # Headcount Limits
        min_doctors, max_doctors: Doctor count bounds
        min_nurses, max_nurses: Nurse count bounds
        min_support, max_support: Support staff count bounds

        # Overtime Rules
        max_overtime_per_staff: Maximum overtime hours per staff per day
        overtime_multiplier: Overtime pay multiplier (e.g., 1.5x)

        # Skill Mix
        nurse_doctor_ratio: Minimum nurses per doctor

        # Hourly Rates
        doctor_hourly_rate: Doctor hourly rate ($)
        nurse_hourly_rate: Nurse hourly rate ($)
        support_hourly_rate: Support staff hourly rate ($)

        # Shift Parameters
        shift_length: Standard shift length (hours)

        # Penalty Costs
        understaffing_penalty_base: Base penalty for understaffing ($/patient-hr)
        overstaffing_penalty: Penalty for overstaffing ($/staff-hr)

        # Solver Settings
        solve_time_limit: Maximum solve time (seconds)
        optimality_gap: Acceptable optimality gap (0.01 = 1%)
    """

    # Budget Constraints (NEW - not in base classes)
    daily_budget_cap: float = 50000.0
    weekly_budget_cap: float = 350000.0
    enforce_budget: bool = True

    # Headcount Limits
    min_doctors: int = 5
    max_doctors: int = 20
    min_nurses: int = 15
    max_nurses: int = 50
    min_support: int = 8
    max_support: int = 30

    # Overtime Rules
    max_overtime_per_staff: float = 4.0
    overtime_multiplier: float = 1.5
    allow_overtime: bool = True

    # Skill Mix
    nurse_doctor_ratio: float = 3.0
    enforce_skill_mix: bool = True

    # Hourly Rates ($)
    doctor_hourly_rate: float = 150.0
    nurse_hourly_rate: float = 45.0
    support_hourly_rate: float = 25.0

    # Shift Parameters
    shift_length: float = 8.0

    # Penalty Costs
    understaffing_penalty_base: float = 200.0
    overstaffing_penalty: float = 15.0

    # Category-Specific Staffing Ratios (patients per staff)
    staffing_ratios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "TRAUMA": {"Doctors": 3, "Nurses": 2, "Support": 4},
        "CARDIAC": {"Doctors": 4, "Nurses": 2, "Support": 5},
        "NEUROLOGICAL": {"Doctors": 5, "Nurses": 2, "Support": 6},
        "RESPIRATORY": {"Doctors": 6, "Nurses": 3, "Support": 8},
        "INFECTIOUS": {"Doctors": 8, "Nurses": 4, "Support": 10},
        "GASTROINTESTINAL": {"Doctors": 10, "Nurses": 5, "Support": 12},
        "OTHER": {"Doctors": 12, "Nurses": 6, "Support": 15},
    })

    # Solver Settings
    solve_time_limit: int = 60
    optimality_gap: float = 0.01  # 1%

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate constraint consistency.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Budget validation
        if self.daily_budget_cap <= 0:
            errors.append("Daily budget cap must be positive")
        if self.weekly_budget_cap < self.daily_budget_cap * 7:
            errors.append("Weekly budget cap should be >= 7x daily cap")

        # Headcount validation
        if self.min_doctors > self.max_doctors:
            errors.append("Min doctors cannot exceed max doctors")
        if self.min_nurses > self.max_nurses:
            errors.append("Min nurses cannot exceed max nurses")
        if self.min_support > self.max_support:
            errors.append("Min support cannot exceed max support")

        # Overtime validation
        if self.max_overtime_per_staff < 0:
            errors.append("Max overtime must be non-negative")
        if self.max_overtime_per_staff > 8:
            errors.append("Max overtime per staff should not exceed 8 hours")
        if self.overtime_multiplier < 1.0:
            errors.append("Overtime multiplier must be >= 1.0")

        # Skill mix validation
        if self.nurse_doctor_ratio < 1.0:
            errors.append("Nurse-to-doctor ratio should be >= 1.0")

        # Hourly rates validation
        if self.doctor_hourly_rate <= 0:
            errors.append("Doctor hourly rate must be positive")
        if self.nurse_hourly_rate <= 0:
            errors.append("Nurse hourly rate must be positive")
        if self.support_hourly_rate <= 0:
            errors.append("Support hourly rate must be positive")

        # Solver settings validation
        if self.solve_time_limit < 1:
            errors.append("Solve time limit must be at least 1 second")
        if self.optimality_gap < 0 or self.optimality_gap > 0.5:
            errors.append("Optimality gap must be between 0 and 0.5 (50%)")

        return len(errors) == 0, errors

    def to_cost_params(self) -> CostParameters:
        """Convert to CostParameters for MILP solver."""
        return CostParameters(
            doctor_hourly_rate=self.doctor_hourly_rate,
            nurse_hourly_rate=self.nurse_hourly_rate,
            support_hourly_rate=self.support_hourly_rate,
            overtime_multiplier=self.overtime_multiplier,
            shift_length=self.shift_length,
            understaffing_penalty_base=self.understaffing_penalty_base,
            overstaffing_penalty=self.overstaffing_penalty,
            max_overtime_hours=self.max_overtime_per_staff if self.allow_overtime else 0.0,
        )

    def to_staffing_ratios(self) -> StaffingRatios:
        """Convert to StaffingRatios for MILP solver."""
        ratios = StaffingRatios(
            ratios=self.staffing_ratios,
            min_doctors=self.min_doctors,
            max_doctors=self.max_doctors,
            min_nurses=self.min_nurses,
            max_nurses=self.max_nurses,
            min_support=self.min_support,
            max_support=self.max_support,
            nurse_to_doctor_ratio=self.nurse_doctor_ratio if self.enforce_skill_mix else 0.0,
        )
        return ratios

    def estimate_min_daily_cost(self) -> float:
        """Estimate minimum daily cost based on min staff."""
        min_cost = (
            self.min_doctors * self.doctor_hourly_rate * self.shift_length +
            self.min_nurses * self.nurse_hourly_rate * self.shift_length +
            self.min_support * self.support_hourly_rate * self.shift_length
        )
        return min_cost

    def estimate_max_daily_cost(self) -> float:
        """Estimate maximum daily cost based on max staff + overtime."""
        max_regular = (
            self.max_doctors * self.doctor_hourly_rate * self.shift_length +
            self.max_nurses * self.nurse_hourly_rate * self.shift_length +
            self.max_support * self.support_hourly_rate * self.shift_length
        )
        max_overtime = (
            (self.max_doctors + self.max_nurses + self.max_support) *
            self.max_overtime_per_staff *
            self.doctor_hourly_rate * self.overtime_multiplier  # Approx
        )
        return max_regular + max_overtime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            # Budget
            "daily_budget_cap": self.daily_budget_cap,
            "weekly_budget_cap": self.weekly_budget_cap,
            "enforce_budget": self.enforce_budget,
            # Headcount
            "min_doctors": self.min_doctors,
            "max_doctors": self.max_doctors,
            "min_nurses": self.min_nurses,
            "max_nurses": self.max_nurses,
            "min_support": self.min_support,
            "max_support": self.max_support,
            # Overtime
            "max_overtime_per_staff": self.max_overtime_per_staff,
            "overtime_multiplier": self.overtime_multiplier,
            "allow_overtime": self.allow_overtime,
            # Skill mix
            "nurse_doctor_ratio": self.nurse_doctor_ratio,
            "enforce_skill_mix": self.enforce_skill_mix,
            # Rates
            "doctor_hourly_rate": self.doctor_hourly_rate,
            "nurse_hourly_rate": self.nurse_hourly_rate,
            "support_hourly_rate": self.support_hourly_rate,
            "shift_length": self.shift_length,
            # Penalties
            "understaffing_penalty_base": self.understaffing_penalty_base,
            "overstaffing_penalty": self.overstaffing_penalty,
            # Staffing ratios
            "staffing_ratios": self.staffing_ratios,
            # Solver
            "solve_time_limit": self.solve_time_limit,
            "optimality_gap": self.optimality_gap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaffConstraintConfig":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# INVENTORY CONSTRAINT CONFIGURATION
# =============================================================================

@dataclass
class InventoryConstraintConfig:
    """
    Complete constraint configuration for inventory optimization.

    Attributes:
        # Storage Constraints
        storage_capacity: Total storage capacity (cubic meters)

        # Budget Constraints
        daily_budget: Daily procurement budget ($), None = unlimited

        # Service Level Targets
        target_service_level: Target service level (0.90 to 0.999)
        safety_factor: Z-score for safety stock calculation

        # Lead Times (days) - per item overrides
        lead_times: Dict mapping item_id to lead time

        # Order Quantities
        min_order_quantities: Dict mapping item_id to min order qty
        max_order_quantities: Dict mapping item_id to max order qty

        # Cost Parameters
        holding_cost_rate: Annual holding cost as fraction of unit cost

        # Solver Settings
        optimization_method: "eoq_only", "milp_full", or "hybrid"
        planning_horizon: Number of days to plan
        solve_time_limit: Maximum solve time (seconds)
    """

    # Storage Constraints
    storage_capacity: float = 1000.0  # Cubic meters

    # Budget Constraints
    daily_budget: Optional[float] = None  # None = unlimited

    # Service Level Targets
    target_service_level: float = 0.95
    safety_factor: Optional[float] = None  # Auto-calculated if None

    # Lead Times (per item, days)
    lead_times: Dict[str, int] = field(default_factory=dict)

    # Order Quantities
    min_order_quantities: Dict[str, int] = field(default_factory=dict)
    max_order_quantities: Dict[str, int] = field(default_factory=dict)

    # Cost Parameters
    holding_cost_rate: float = 0.20  # 20% annual

    # Solver Settings
    optimization_method: str = "hybrid"  # "eoq_only", "milp_full", "hybrid"
    planning_horizon: int = 30  # Days
    solve_time_limit: int = 120  # Seconds

    def __post_init__(self):
        """Calculate safety factor if not provided."""
        if self.safety_factor is None:
            self.safety_factor = get_z_score(self.target_service_level)

    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time: int,
    ) -> float:
        """
        Calculate safety stock using statistical formula.

        SS = Z × σ × √L

        Where:
            Z = safety factor (z-score for service level)
            σ = standard deviation of demand
            L = lead time in days

        Args:
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days

        Returns:
            Safety stock quantity
        """
        return self.safety_factor * demand_std * np.sqrt(lead_time)

    def calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: float,
        unit_cost: float,
    ) -> float:
        """
        Calculate Economic Order Quantity.

        EOQ = √((2 × D × S) / H)

        Where:
            D = annual demand
            S = ordering cost per order
            H = annual holding cost per unit = unit_cost × holding_cost_rate

        Args:
            annual_demand: Annual demand in units
            ordering_cost: Fixed cost per order ($)
            unit_cost: Cost per unit ($)

        Returns:
            Economic order quantity
        """
        h = unit_cost * self.holding_cost_rate
        if h <= 0 or annual_demand <= 0:
            return 0.0
        return np.sqrt((2 * ordering_cost * annual_demand) / h)

    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        demand_std: float,
        lead_time: int,
    ) -> float:
        """
        Calculate reorder point.

        ROP = (avg_daily_demand × lead_time) + safety_stock

        Args:
            avg_daily_demand: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days

        Returns:
            Reorder point quantity
        """
        lead_time_demand = avg_daily_demand * lead_time
        safety_stock = self.calculate_safety_stock(demand_std, lead_time)
        return lead_time_demand + safety_stock

    def get_lead_time(self, item_id: str, default: int = 3) -> int:
        """Get lead time for an item, or default."""
        return self.lead_times.get(item_id, default)

    def get_min_order_qty(self, item_id: str, default: int = 1) -> int:
        """Get minimum order quantity for an item, or default."""
        return self.min_order_quantities.get(item_id, default)

    def get_max_order_qty(self, item_id: str, default: int = 10000) -> int:
        """Get maximum order quantity for an item, or default."""
        return self.max_order_quantities.get(item_id, default)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate constraint consistency.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Storage validation
        if self.storage_capacity <= 0:
            errors.append("Storage capacity must be positive")

        # Budget validation
        if self.daily_budget is not None and self.daily_budget <= 0:
            errors.append("Daily budget must be positive or None (unlimited)")

        # Service level validation
        if self.target_service_level < 0.8 or self.target_service_level > 0.999:
            errors.append("Target service level must be between 0.80 and 0.999")

        # Safety factor validation
        if self.safety_factor is not None and self.safety_factor < 0:
            errors.append("Safety factor must be non-negative")

        # Holding cost rate validation
        if self.holding_cost_rate < 0 or self.holding_cost_rate > 1:
            errors.append("Holding cost rate must be between 0 and 1")

        # Planning horizon validation
        if self.planning_horizon < 1 or self.planning_horizon > 365:
            errors.append("Planning horizon must be between 1 and 365 days")

        # Optimization method validation
        valid_methods = ["eoq_only", "milp_full", "hybrid"]
        if self.optimization_method not in valid_methods:
            errors.append(f"Optimization method must be one of: {valid_methods}")

        # Solver time validation
        if self.solve_time_limit < 1:
            errors.append("Solve time limit must be at least 1 second")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "storage_capacity": self.storage_capacity,
            "daily_budget": self.daily_budget,
            "target_service_level": self.target_service_level,
            "safety_factor": self.safety_factor,
            "lead_times": self.lead_times,
            "min_order_quantities": self.min_order_quantities,
            "max_order_quantities": self.max_order_quantities,
            "holding_cost_rate": self.holding_cost_rate,
            "optimization_method": self.optimization_method,
            "planning_horizon": self.planning_horizon,
            "solve_time_limit": self.solve_time_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InventoryConstraintConfig":
        """Create from dictionary."""
        return cls(**data)

    def get_service_level_options(self) -> List[Dict[str, Any]]:
        """Get common service level options for UI dropdown."""
        return [
            {"level": 0.90, "z_score": 1.282, "label": "90% (Standard)"},
            {"level": 0.95, "z_score": 1.645, "label": "95% (Recommended)"},
            {"level": 0.97, "z_score": 1.881, "label": "97% (High)"},
            {"level": 0.99, "z_score": 2.326, "label": "99% (Critical Items)"},
            {"level": 0.995, "z_score": 2.576, "label": "99.5% (Life-Critical)"},
        ]


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_STAFF_CONSTRAINTS = StaffConstraintConfig()
DEFAULT_INVENTORY_CONSTRAINTS = InventoryConstraintConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_all_constraints(
    staff_config: Optional[StaffConstraintConfig] = None,
    inventory_config: Optional[InventoryConstraintConfig] = None,
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate all constraint configurations.

    Args:
        staff_config: Staff constraint configuration
        inventory_config: Inventory constraint configuration

    Returns:
        Tuple of (all_valid, dict_of_errors_by_type)
    """
    errors = {}
    all_valid = True

    if staff_config is not None:
        valid, staff_errors = staff_config.validate()
        if not valid:
            all_valid = False
            errors["staff"] = staff_errors

    if inventory_config is not None:
        valid, inv_errors = inventory_config.validate()
        if not valid:
            all_valid = False
            errors["inventory"] = inv_errors

    return all_valid, errors


def create_constraints_summary(
    staff_config: Optional[StaffConstraintConfig] = None,
    inventory_config: Optional[InventoryConstraintConfig] = None,
) -> Dict[str, Any]:
    """
    Create a summary of constraint configurations for display.

    Args:
        staff_config: Staff constraint configuration
        inventory_config: Inventory constraint configuration

    Returns:
        Summary dictionary
    """
    summary = {}

    if staff_config is not None:
        summary["staff"] = {
            "Budget": {
                "Daily Cap": f"${staff_config.daily_budget_cap:,.0f}",
                "Weekly Cap": f"${staff_config.weekly_budget_cap:,.0f}",
                "Enforced": staff_config.enforce_budget,
            },
            "Headcount Limits": {
                "Doctors": f"{staff_config.min_doctors} - {staff_config.max_doctors}",
                "Nurses": f"{staff_config.min_nurses} - {staff_config.max_nurses}",
                "Support": f"{staff_config.min_support} - {staff_config.max_support}",
            },
            "Overtime": {
                "Max Hours/Staff": staff_config.max_overtime_per_staff,
                "Multiplier": f"{staff_config.overtime_multiplier}x",
                "Allowed": staff_config.allow_overtime,
            },
            "Skill Mix": {
                "Nurse:Doctor Ratio": f"{staff_config.nurse_doctor_ratio}:1",
                "Enforced": staff_config.enforce_skill_mix,
            },
            "Solver": {
                "Time Limit": f"{staff_config.solve_time_limit}s",
                "Gap": f"{staff_config.optimality_gap * 100:.1f}%",
            },
        }

    if inventory_config is not None:
        summary["inventory"] = {
            "Storage": {
                "Capacity": f"{inventory_config.storage_capacity:,.0f} m³",
            },
            "Budget": {
                "Daily Limit": f"${inventory_config.daily_budget:,.0f}" if inventory_config.daily_budget else "Unlimited",
            },
            "Service Level": {
                "Target": f"{inventory_config.target_service_level * 100:.1f}%",
                "Safety Factor (Z)": f"{inventory_config.safety_factor:.3f}",
            },
            "Cost Parameters": {
                "Holding Cost Rate": f"{inventory_config.holding_cost_rate * 100:.0f}%",
            },
            "Solver": {
                "Method": inventory_config.optimization_method,
                "Horizon": f"{inventory_config.planning_horizon} days",
                "Time Limit": f"{inventory_config.solve_time_limit}s",
            },
        }

    return summary
