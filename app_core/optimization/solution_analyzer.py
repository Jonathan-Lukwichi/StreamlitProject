# =============================================================================
# app_core/optimization/solution_analyzer.py
# Solution Analysis and Before/After Comparison
# Analyzes historical staffing patterns and compares with optimized solutions
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .cost_parameters import (
    CostParameters,
    StaffingRatios,
    DEFAULT_COST_PARAMS,
    DEFAULT_STAFFING_RATIOS,
    CLINICAL_CATEGORIES,
    CATEGORY_PRIORITIES,
    STAFF_TYPES,
)
from .milp_solver import OptimizationResult


# =============================================================================
# HISTORICAL ANALYSIS
# =============================================================================

@dataclass
class HistoricalAnalysis:
    """Container for historical staffing analysis (Before Optimization)."""

    # Summary statistics
    total_cost: float = 0.0
    avg_daily_cost: float = 0.0
    total_overtime_hours: float = 0.0
    avg_overtime_per_day: float = 0.0
    shortage_days: int = 0
    shortage_rate: float = 0.0

    # Staff statistics
    avg_doctors: float = 0.0
    avg_nurses: float = 0.0
    avg_support: float = 0.0
    total_staff_days: int = 0

    # Utilization
    avg_utilization: float = 0.0

    # Cost breakdown
    regular_labor_cost: float = 0.0
    overtime_cost: float = 0.0
    estimated_shortage_cost: float = 0.0

    # DataFrames
    daily_breakdown: Optional[pd.DataFrame] = None
    weekly_summary: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "Total Cost (Est.)": f"${self.total_cost:,.2f}",
            "Avg Daily Cost": f"${self.avg_daily_cost:,.2f}",
            "Regular Labor": f"${self.regular_labor_cost:,.2f}",
            "Overtime Cost": f"${self.overtime_cost:,.2f}",
            "Shortage Penalty": f"${self.estimated_shortage_cost:,.2f}",
            "Total Overtime Hours": f"{self.total_overtime_hours:,.1f}h",
            "Avg Overtime/Day": f"{self.avg_overtime_per_day:.1f}h",
            "Shortage Days": f"{self.shortage_days}",
            "Shortage Rate": f"{self.shortage_rate:.1f}%",
            "Avg Doctors/Day": f"{self.avg_doctors:.1f}",
            "Avg Nurses/Day": f"{self.avg_nurses:.1f}",
            "Avg Support/Day": f"{self.avg_support:.1f}",
            "Avg Utilization": f"{self.avg_utilization:.1f}%",
        }


class SolutionAnalyzer:
    """
    Analyzes historical staffing data and compares with optimized solutions.
    """

    def __init__(
        self,
        cost_params: Optional[CostParameters] = None,
        staffing_ratios: Optional[StaffingRatios] = None,
    ):
        """Initialize the analyzer."""
        self.cost_params = cost_params or DEFAULT_COST_PARAMS
        self.staffing_ratios = staffing_ratios or DEFAULT_STAFFING_RATIOS

    def analyze_historical(
        self,
        staff_df: pd.DataFrame,
        n_days: Optional[int] = None,
    ) -> HistoricalAnalysis:
        """
        Analyze historical staffing data from Supabase.

        Args:
            staff_df: DataFrame with columns:
                - Date
                - Doctors_on_Duty
                - Nurses_on_Duty
                - Support_Staff_on_Duty
                - Overtime_Hours
                - Staff_Shortage_Flag
                - Staff_Utilization_Rate
            n_days: Number of recent days to analyze (None = all)

        Returns:
            HistoricalAnalysis with detailed breakdown
        """
        result = HistoricalAnalysis()

        if staff_df.empty:
            return result

        # Use last n_days if specified
        df = staff_df.copy()
        if n_days is not None and len(df) > n_days:
            df = df.tail(n_days).reset_index(drop=True)

        n_records = len(df)
        H = self.cost_params.shift_length

        # =================================================================
        # BASIC STATISTICS
        # =================================================================

        # Staff averages
        result.avg_doctors = df["Doctors_on_Duty"].mean() if "Doctors_on_Duty" in df.columns else 0
        result.avg_nurses = df["Nurses_on_Duty"].mean() if "Nurses_on_Duty" in df.columns else 0
        result.avg_support = df["Support_Staff_on_Duty"].mean() if "Support_Staff_on_Duty" in df.columns else 0

        # Overtime
        if "Overtime_Hours" in df.columns:
            result.total_overtime_hours = df["Overtime_Hours"].sum()
            result.avg_overtime_per_day = df["Overtime_Hours"].mean()
        else:
            result.total_overtime_hours = 0
            result.avg_overtime_per_day = 0

        # Shortage days
        if "Staff_Shortage_Flag" in df.columns:
            result.shortage_days = int(df["Staff_Shortage_Flag"].sum())
            result.shortage_rate = (result.shortage_days / n_records) * 100
        else:
            result.shortage_days = 0
            result.shortage_rate = 0

        # Utilization
        if "Staff_Utilization_Rate" in df.columns:
            result.avg_utilization = df["Staff_Utilization_Rate"].mean() * 100
        else:
            result.avg_utilization = 0

        # Total staff-days
        result.total_staff_days = int(
            df.get("Doctors_on_Duty", pd.Series([0])).sum() +
            df.get("Nurses_on_Duty", pd.Series([0])).sum() +
            df.get("Support_Staff_on_Duty", pd.Series([0])).sum()
        )

        # =================================================================
        # COST CALCULATION
        # =================================================================

        # Regular labor cost (per day)
        daily_labor = []
        for _, row in df.iterrows():
            doctors = row.get("Doctors_on_Duty", 0)
            nurses = row.get("Nurses_on_Duty", 0)
            support = row.get("Support_Staff_on_Duty", 0)

            daily_cost = (
                doctors * self.cost_params.doctor_hourly_rate * H +
                nurses * self.cost_params.nurse_hourly_rate * H +
                support * self.cost_params.support_hourly_rate * H
            )
            daily_labor.append(daily_cost)

        result.regular_labor_cost = sum(daily_labor)

        # Overtime cost
        if "Overtime_Hours" in df.columns:
            # Estimate overtime distribution by staff type (proportional to count)
            overtime_cost = 0
            for _, row in df.iterrows():
                ot_hours = row.get("Overtime_Hours", 0)
                if ot_hours > 0:
                    doctors = row.get("Doctors_on_Duty", 0)
                    nurses = row.get("Nurses_on_Duty", 0)
                    support = row.get("Support_Staff_on_Duty", 0)
                    total = doctors + nurses + support

                    if total > 0:
                        # Distribute overtime proportionally
                        doc_ot = ot_hours * (doctors / total)
                        nurse_ot = ot_hours * (nurses / total)
                        support_ot = ot_hours * (support / total)

                        overtime_cost += (
                            doc_ot * self.cost_params.get_overtime_rate("Doctors") +
                            nurse_ot * self.cost_params.get_overtime_rate("Nurses") +
                            support_ot * self.cost_params.get_overtime_rate("Support")
                        )

            result.overtime_cost = overtime_cost
        else:
            result.overtime_cost = 0

        # Estimated shortage cost (based on shortage days)
        # Assume average 20 patients unmet per shortage day
        avg_unmet_per_shortage = 20
        result.estimated_shortage_cost = (
            result.shortage_days *
            avg_unmet_per_shortage *
            self.cost_params.understaffing_penalty_base
        )

        # Total cost
        result.total_cost = (
            result.regular_labor_cost +
            result.overtime_cost +
            result.estimated_shortage_cost
        )
        result.avg_daily_cost = result.total_cost / n_records if n_records > 0 else 0

        # =================================================================
        # DAILY BREAKDOWN
        # =================================================================
        breakdown_data = []
        for idx, row in df.iterrows():
            doctors = row.get("Doctors_on_Duty", 0)
            nurses = row.get("Nurses_on_Duty", 0)
            support = row.get("Support_Staff_on_Duty", 0)
            ot = row.get("Overtime_Hours", 0)
            shortage = row.get("Staff_Shortage_Flag", 0)

            daily_regular = (
                doctors * self.cost_params.doctor_hourly_rate * H +
                nurses * self.cost_params.nurse_hourly_rate * H +
                support * self.cost_params.support_hourly_rate * H
            )

            breakdown_data.append({
                "Day": idx + 1,
                "Doctors": int(doctors),
                "Nurses": int(nurses),
                "Support": int(support),
                "Total_Staff": int(doctors + nurses + support),
                "OT_Hours": round(ot, 1),
                "Shortage": int(shortage),
                "Daily_Cost": round(daily_regular, 2),
            })

        result.daily_breakdown = pd.DataFrame(breakdown_data)

        return result

    def compare_before_after(
        self,
        historical: HistoricalAnalysis,
        optimized: OptimizationResult,
    ) -> pd.DataFrame:
        """
        Compare historical (before) vs optimized (after) results.

        Args:
            historical: Historical analysis results
            optimized: Optimization results

        Returns:
            DataFrame with comparison metrics
        """
        comparisons = []

        # Total Cost
        cost_savings = historical.total_cost - optimized.total_cost
        cost_savings_pct = (cost_savings / historical.total_cost * 100) if historical.total_cost > 0 else 0
        comparisons.append({
            "Metric": "Total Cost",
            "Before": f"${historical.total_cost:,.2f}",
            "After": f"${optimized.total_cost:,.2f}",
            "Change": f"${cost_savings:,.2f}",
            "Change_%": f"{cost_savings_pct:+.1f}%",
            "Status": "Improved" if cost_savings > 0 else "Worse",
        })

        # Regular Labor Cost
        labor_change = historical.regular_labor_cost - optimized.regular_labor_cost
        comparisons.append({
            "Metric": "Regular Labor",
            "Before": f"${historical.regular_labor_cost:,.2f}",
            "After": f"${optimized.regular_labor_cost:,.2f}",
            "Change": f"${labor_change:,.2f}",
            "Change_%": f"{(labor_change / historical.regular_labor_cost * 100) if historical.regular_labor_cost > 0 else 0:+.1f}%",
            "Status": "Improved" if labor_change > 0 else "Adjusted",
        })

        # Overtime Cost
        ot_change = historical.overtime_cost - optimized.overtime_cost
        comparisons.append({
            "Metric": "Overtime Cost",
            "Before": f"${historical.overtime_cost:,.2f}",
            "After": f"${optimized.overtime_cost:,.2f}",
            "Change": f"${ot_change:,.2f}",
            "Change_%": f"{(ot_change / historical.overtime_cost * 100) if historical.overtime_cost > 0 else 0:+.1f}%",
            "Status": "Improved" if ot_change > 0 else "Adjusted",
        })

        # Understaffing Penalty
        understaffing_change = historical.estimated_shortage_cost - optimized.understaffing_penalty
        comparisons.append({
            "Metric": "Understaffing Penalty",
            "Before": f"${historical.estimated_shortage_cost:,.2f}",
            "After": f"${optimized.understaffing_penalty:,.2f}",
            "Change": f"${understaffing_change:,.2f}",
            "Change_%": f"{(understaffing_change / historical.estimated_shortage_cost * 100) if historical.estimated_shortage_cost > 0 else 0:+.1f}%",
            "Status": "Improved" if understaffing_change > 0 else "Adjusted",
        })

        # Overtime Hours
        if optimized.overtime_schedule is not None:
            optimized_ot = optimized.overtime_schedule[[c for c in optimized.overtime_schedule.columns if "_OT" in c]].sum().sum()
        else:
            optimized_ot = 0

        # Scale historical overtime to match planning horizon
        horizon = len(optimized.daily_summary) if optimized.daily_summary is not None else 7
        historical_ot_scaled = historical.avg_overtime_per_day * horizon

        ot_hours_change = historical_ot_scaled - optimized_ot
        comparisons.append({
            "Metric": "Total Overtime Hours",
            "Before": f"{historical_ot_scaled:.1f}h",
            "After": f"{optimized_ot:.1f}h",
            "Change": f"{ot_hours_change:+.1f}h",
            "Change_%": f"{(ot_hours_change / historical_ot_scaled * 100) if historical_ot_scaled > 0 else 0:+.1f}%",
            "Status": "Improved" if ot_hours_change > 0 else "Adjusted",
        })

        # Average Coverage
        if optimized.daily_summary is not None:
            avg_coverage = optimized.daily_summary["Coverage_%"].mean()
        else:
            avg_coverage = 100

        historical_coverage = 100 - (historical.shortage_rate if historical.shortage_rate else 0)
        coverage_change = avg_coverage - historical_coverage
        comparisons.append({
            "Metric": "Demand Coverage",
            "Before": f"{historical_coverage:.1f}%",
            "After": f"{avg_coverage:.1f}%",
            "Change": f"{coverage_change:+.1f}%",
            "Change_%": "-",
            "Status": "Improved" if coverage_change > 0 else ("Same" if coverage_change == 0 else "Adjusted"),
        })

        return pd.DataFrame(comparisons)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_historical_costs(
    staff_df: pd.DataFrame,
    cost_params: Optional[CostParameters] = None,
    n_days: Optional[int] = None,
) -> HistoricalAnalysis:
    """
    Analyze historical staffing costs.

    Args:
        staff_df: Staff scheduling DataFrame from Supabase
        cost_params: Cost parameters (uses defaults if None)
        n_days: Number of days to analyze

    Returns:
        HistoricalAnalysis results
    """
    analyzer = SolutionAnalyzer(cost_params)
    return analyzer.analyze_historical(staff_df, n_days)


def compare_before_after(
    staff_df: pd.DataFrame,
    optimization_result: OptimizationResult,
    cost_params: Optional[CostParameters] = None,
) -> Tuple[HistoricalAnalysis, pd.DataFrame]:
    """
    Compare historical vs optimized staffing.

    Args:
        staff_df: Historical staff data
        optimization_result: MILP optimization result
        cost_params: Cost parameters

    Returns:
        Tuple of (HistoricalAnalysis, comparison DataFrame)
    """
    analyzer = SolutionAnalyzer(cost_params)

    # Analyze historical (use same horizon as optimization)
    horizon = len(optimization_result.daily_summary) if optimization_result.daily_summary is not None else 7
    historical = analyzer.analyze_historical(staff_df, n_days=horizon)

    # Compare
    comparison = analyzer.compare_before_after(historical, optimization_result)

    return historical, comparison
