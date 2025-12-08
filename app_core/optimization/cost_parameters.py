# =============================================================================
# app_core/optimization/cost_parameters.py
# Cost Parameters and Staffing Ratios for MILP Optimization
# All costs in USD (American Dollars)
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# =============================================================================
# CLINICAL CATEGORIES (from ML Forecasting)
# =============================================================================

CLINICAL_CATEGORIES: List[str] = [
    "RESPIRATORY",
    "CARDIAC",
    "TRAUMA",
    "GASTROINTESTINAL",
    "INFECTIOUS",
    "NEUROLOGICAL",
    "OTHER",
]

# Priority weights for understaffing penalties (higher = more critical)
CATEGORY_PRIORITIES: Dict[str, float] = {
    "TRAUMA": 3.0,           # Critical - emergency cases
    "CARDIAC": 2.5,          # High priority - life-threatening
    "NEUROLOGICAL": 2.0,     # High complexity
    "RESPIRATORY": 1.5,      # Medium-high priority
    "INFECTIOUS": 1.2,       # Medium priority
    "GASTROINTESTINAL": 1.0, # Standard priority
    "OTHER": 0.8,            # Lower priority
}

# Staff types
STAFF_TYPES: List[str] = ["Doctors", "Nurses", "Support"]


# =============================================================================
# COST PARAMETERS (USD)
# =============================================================================

@dataclass
class CostParameters:
    """
    Cost parameters for staff scheduling optimization.
    All values in USD (American Dollars).
    """

    # Regular hourly rates ($/hour)
    doctor_hourly_rate: float = 150.0
    nurse_hourly_rate: float = 45.0
    support_hourly_rate: float = 25.0

    # Overtime multiplier (typically 1.5x)
    overtime_multiplier: float = 1.5

    # Agency staff multiplier (typically 2x regular rate)
    agency_multiplier: float = 2.0

    # Standard shift length (hours)
    shift_length: float = 8.0

    # Penalty costs ($/patient-hour for understaffing)
    understaffing_penalty_base: float = 200.0  # Base penalty
    overstaffing_penalty: float = 15.0         # $/staff-hour excess

    # Maximum overtime hours per staff per day
    max_overtime_hours: float = 4.0

    def get_hourly_rate(self, staff_type: str) -> float:
        """Get hourly rate for staff type."""
        rates = {
            "Doctors": self.doctor_hourly_rate,
            "Nurses": self.nurse_hourly_rate,
            "Support": self.support_hourly_rate,
        }
        return rates.get(staff_type, 0.0)

    def get_overtime_rate(self, staff_type: str) -> float:
        """Get overtime rate for staff type."""
        return self.get_hourly_rate(staff_type) * self.overtime_multiplier

    def get_agency_rate(self, staff_type: str) -> float:
        """Get agency staff rate for staff type."""
        return self.get_hourly_rate(staff_type) * self.agency_multiplier

    def get_understaffing_penalty(self, category: str) -> float:
        """Get category-weighted understaffing penalty."""
        priority = CATEGORY_PRIORITIES.get(category, 1.0)
        return self.understaffing_penalty_base * priority

    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            "Doctor Hourly Rate": f"${self.doctor_hourly_rate:.2f}",
            "Nurse Hourly Rate": f"${self.nurse_hourly_rate:.2f}",
            "Support Hourly Rate": f"${self.support_hourly_rate:.2f}",
            "Overtime Multiplier": f"{self.overtime_multiplier}x",
            "Agency Multiplier": f"{self.agency_multiplier}x",
            "Shift Length": f"{self.shift_length} hours",
            "Max Overtime/Day": f"{self.max_overtime_hours} hours",
            "Understaffing Penalty (Base)": f"${self.understaffing_penalty_base}/patient-hr",
            "Overstaffing Penalty": f"${self.overstaffing_penalty}/staff-hr",
        }


# =============================================================================
# STAFFING RATIOS (Patients per Staff)
# =============================================================================

@dataclass
class StaffingRatios:
    """
    Category-specific staffing ratios.
    Values represent: patients per staff member per shift.
    Lower ratio = more staff needed (higher acuity).
    """

    # Default ratios per category (patients per staff)
    # Format: {category: {staff_type: patients_per_staff}}
    ratios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "TRAUMA": {
            "Doctors": 3,      # 1 doctor per 3 trauma patients
            "Nurses": 2,       # 1 nurse per 2 trauma patients
            "Support": 4,      # 1 support per 4 trauma patients
        },
        "CARDIAC": {
            "Doctors": 4,
            "Nurses": 2,
            "Support": 5,
        },
        "NEUROLOGICAL": {
            "Doctors": 5,
            "Nurses": 2,
            "Support": 6,
        },
        "RESPIRATORY": {
            "Doctors": 6,
            "Nurses": 3,
            "Support": 8,
        },
        "INFECTIOUS": {
            "Doctors": 8,
            "Nurses": 4,
            "Support": 10,
        },
        "GASTROINTESTINAL": {
            "Doctors": 10,
            "Nurses": 5,
            "Support": 12,
        },
        "OTHER": {
            "Doctors": 12,
            "Nurses": 6,
            "Support": 15,
        },
    })

    # Staff availability constraints (min/max per day)
    min_doctors: int = 5
    max_doctors: int = 20
    min_nurses: int = 15
    max_nurses: int = 50
    min_support: int = 8
    max_support: int = 30

    # Skill mix ratio (minimum nurses per doctor)
    nurse_to_doctor_ratio: float = 3.0

    def get_ratio(self, category: str, staff_type: str) -> float:
        """Get patients per staff ratio for category and staff type."""
        if category in self.ratios and staff_type in self.ratios[category]:
            return self.ratios[category][staff_type]
        # Default fallback
        defaults = {"Doctors": 10, "Nurses": 5, "Support": 12}
        return defaults.get(staff_type, 10)

    def get_service_rate(self, category: str, staff_type: str, shift_hours: float = 8.0) -> float:
        """
        Get service rate: patients served per staff per hour.
        service_rate = patients_per_shift / shift_hours
        """
        patients_per_shift = self.get_ratio(category, staff_type)
        return patients_per_shift / shift_hours

    def get_min_staff(self, staff_type: str) -> int:
        """Get minimum staff count for type."""
        mins = {
            "Doctors": self.min_doctors,
            "Nurses": self.min_nurses,
            "Support": self.min_support,
        }
        return mins.get(staff_type, 0)

    def get_max_staff(self, staff_type: str) -> int:
        """Get maximum staff count for type."""
        maxs = {
            "Doctors": self.max_doctors,
            "Nurses": self.max_nurses,
            "Support": self.max_support,
        }
        return maxs.get(staff_type, 100)

    def set_ratio(self, category: str, staff_type: str, value: float):
        """Update a specific ratio."""
        if category not in self.ratios:
            self.ratios[category] = {}
        self.ratios[category][staff_type] = value

    def to_dataframe(self):
        """Convert ratios to DataFrame for display."""
        import pandas as pd

        data = []
        for category in CLINICAL_CATEGORIES:
            row = {"Category": category}
            for staff_type in STAFF_TYPES:
                row[f"{staff_type} (1:{self.get_ratio(category, staff_type):.0f})"] = \
                    self.get_ratio(category, staff_type)
            row["Priority"] = CATEGORY_PRIORITIES.get(category, 1.0)
            data.append(row)

        return pd.DataFrame(data)


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_COST_PARAMS = CostParameters()
DEFAULT_STAFFING_RATIOS = StaffingRatios()
