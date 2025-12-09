# =============================================================================
# app_core/data/financial_service.py
# Supabase Service for Hospital Financial Data
# =============================================================================
"""
Financial Data Service for Supabase Integration

Handles CRUD operations for financial data stored in Supabase.
Table: financial_data

Schema includes:
    - Date, Labor costs, Inventory costs, Revenue, Operating costs
    - Optimization metrics (overstaffing, understaffing penalties)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import streamlit as st

from .supabase_client import get_cached_supabase_client


# Table name in Supabase
FINANCIAL_TABLE = "financial_data"


class FinancialService:
    """
    Service class for financial data operations.
    """

    def __init__(self):
        """Initialize with Supabase client."""
        self.supabase = get_cached_supabase_client()
        self.table = FINANCIAL_TABLE

    def is_connected(self) -> bool:
        """Check if Supabase connection is active."""
        return self.supabase is not None

    def fetch_all(self, limit: int = None) -> pd.DataFrame:
        """
        Fetch all financial records with pagination to bypass 1000 row limit.

        Args:
            limit: Optional limit on records (None = all)

        Returns:
            DataFrame with all financial data
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_records = []
            batch_size = 1000
            offset = 0

            while True:
                # Fetch batch
                response = (
                    self.supabase
                    .table(self.table)
                    .select("*")
                    .order("Date")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                all_records.extend(response.data)

                # Check if we got a full batch (more data might exist)
                if len(response.data) < batch_size:
                    break

                offset += batch_size

                # Apply limit if specified
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records)
            return self._clean_dataframe(df)

        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
            return pd.DataFrame()

    def fetch_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch financial records within a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with filtered records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            all_records = []
            batch_size = 1000
            offset = 0

            start_str = start_date.isoformat()
            end_str = end_date.isoformat()

            while True:
                response = (
                    self.supabase
                    .table(self.table)
                    .select("*")
                    .gte("Date", start_str)
                    .lte("Date", end_str)
                    .order("Date")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                all_records.extend(response.data)

                if len(response.data) < batch_size:
                    break

                offset += batch_size

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records)
            return self._clean_dataframe(df)

        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
            return pd.DataFrame()

    def fetch_latest(self, n_days: int = 30) -> pd.DataFrame:
        """
        Fetch the most recent n days of financial data.

        Args:
            n_days: Number of recent days to fetch

        Returns:
            DataFrame with recent records
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            response = (
                self.supabase
                .table(self.table)
                .select("*")
                .order("Date", desc=True)
                .limit(n_days)
                .execute()
            )

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df = self._clean_dataframe(df)

            # Sort by date ascending
            if "Date" in df.columns:
                df = df.sort_values("Date").reset_index(drop=True)

            return df

        except Exception as e:
            st.error(f"Error fetching latest financial data: {str(e)}")
            return pd.DataFrame()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from financial data.

        Returns:
            Dict with summary statistics
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        stats = {
            "total_records": len(df),
            "date_range": {
                "start": df["Date"].min() if "Date" in df.columns else None,
                "end": df["Date"].max() if "Date" in df.columns else None,
            },
            "avg_labor_cost": {
                "doctor_rate": df["Doctor_Hourly_Rate"].mean() if "Doctor_Hourly_Rate" in df.columns else 0,
                "nurse_rate": df["Nurse_Hourly_Rate"].mean() if "Nurse_Hourly_Rate" in df.columns else 0,
                "support_rate": df["Support_Staff_Hourly_Rate"].mean() if "Support_Staff_Hourly_Rate" in df.columns else 0,
                "total_labor": df["Total_Labor_Cost"].mean() if "Total_Labor_Cost" in df.columns else 0,
            },
            "avg_inventory_cost": {
                "gloves": df["Gloves_Cost"].mean() if "Gloves_Cost" in df.columns else 0,
                "ppe": df["PPE_Cost"].mean() if "PPE_Cost" in df.columns else 0,
                "medication": df["Medication_Cost"].mean() if "Medication_Cost" in df.columns else 0,
                "total": df["Total_Inventory_Cost"].mean() if "Total_Inventory_Cost" in df.columns else 0,
            },
            "avg_revenue": df["Total_Revenue"].mean() if "Total_Revenue" in df.columns else 0,
            "avg_profit": df["Daily_Profit"].mean() if "Daily_Profit" in df.columns else 0,
            "avg_profit_margin": df["Profit_Margin_Percent"].mean() if "Profit_Margin_Percent" in df.columns else 0,
        }

        return stats

    def get_labor_cost_params(self) -> Dict[str, float]:
        """
        Get average labor cost parameters for optimization.

        Returns:
            Dict with hourly rates and multipliers
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        params = {}

        if "Doctor_Hourly_Rate" in df.columns:
            params["doctor_hourly_rate"] = float(df["Doctor_Hourly_Rate"].mean())

        if "Nurse_Hourly_Rate" in df.columns:
            params["nurse_hourly_rate"] = float(df["Nurse_Hourly_Rate"].mean())

        if "Support_Staff_Hourly_Rate" in df.columns:
            params["support_hourly_rate"] = float(df["Support_Staff_Hourly_Rate"].mean())

        if "Overtime_Premium_Rate" in df.columns:
            params["overtime_multiplier"] = float(df["Overtime_Premium_Rate"].mean())

        if "Understaffing_Penalty" in df.columns:
            params["understaffing_penalty"] = float(df["Understaffing_Penalty"].mean())

        if "Overstaffing_Cost" in df.columns:
            params["overstaffing_penalty"] = float(df["Overstaffing_Cost"].mean())

        return params

    def get_inventory_cost_params(self) -> Dict[str, float]:
        """
        Get average inventory cost parameters for optimization.

        Returns:
            Dict with unit costs and penalties
        """
        df = self.fetch_all()

        if df.empty:
            return {}

        params = {}

        if "Cost_Per_Glove_Pair" in df.columns:
            params["gloves_unit_cost"] = float(df["Cost_Per_Glove_Pair"].mean())

        if "Cost_Per_PPE_Set" in df.columns:
            params["ppe_unit_cost"] = float(df["Cost_Per_PPE_Set"].mean())

        if "Cost_Per_Medication_Unit" in df.columns:
            params["medication_unit_cost"] = float(df["Cost_Per_Medication_Unit"].mean())

        if "Inventory_Holding_Cost" in df.columns:
            params["holding_cost"] = float(df["Inventory_Holding_Cost"].mean())

        if "Restock_Order_Cost" in df.columns:
            params["ordering_cost"] = float(df["Restock_Order_Cost"].mean())

        if "Stockout_Penalty" in df.columns:
            params["stockout_penalty"] = float(df["Stockout_Penalty"].mean())

        if "Overstock_Penalty" in df.columns:
            params["overstock_penalty"] = float(df["Overstock_Penalty"].mean())

        return params

    def get_comprehensive_financial_kpis(self, n_days: int = None) -> Dict[str, Any]:
        """
        Get comprehensive financial KPIs for optimization analysis.

        Args:
            n_days: Optional limit to recent N days (None = all data)

        Returns:
            Dict with comprehensive financial metrics
        """
        if n_days:
            df = self.fetch_latest(n_days)
        else:
            df = self.fetch_all()

        if df.empty:
            return {}

        kpis = {
            "data_period": {
                "start_date": df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else None,
                "end_date": df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else None,
                "total_days": len(df),
            },
            # Labor Cost Metrics
            "labor": {
                "avg_doctor_hourly_rate": float(df["Doctor_Hourly_Rate"].mean()) if "Doctor_Hourly_Rate" in df.columns else 0,
                "avg_nurse_hourly_rate": float(df["Nurse_Hourly_Rate"].mean()) if "Nurse_Hourly_Rate" in df.columns else 0,
                "avg_support_hourly_rate": float(df["Support_Staff_Hourly_Rate"].mean()) if "Support_Staff_Hourly_Rate" in df.columns else 0,
                "avg_overtime_rate": float(df["Overtime_Premium_Rate"].mean()) if "Overtime_Premium_Rate" in df.columns else 1.5,
                "avg_daily_doctor_cost": float(df["Doctor_Daily_Cost"].mean()) if "Doctor_Daily_Cost" in df.columns else 0,
                "avg_daily_nurse_cost": float(df["Nurse_Daily_Cost"].mean()) if "Nurse_Daily_Cost" in df.columns else 0,
                "avg_daily_support_cost": float(df["Support_Staff_Daily_Cost"].mean()) if "Support_Staff_Daily_Cost" in df.columns else 0,
                "avg_overtime_cost": float(df["Overtime_Cost"].mean()) if "Overtime_Cost" in df.columns else 0,
                "avg_agency_cost": float(df["Agency_Staff_Cost"].mean()) if "Agency_Staff_Cost" in df.columns else 0,
                "avg_total_labor_cost": float(df["Total_Labor_Cost"].mean()) if "Total_Labor_Cost" in df.columns else 0,
                "total_labor_cost": float(df["Total_Labor_Cost"].sum()) if "Total_Labor_Cost" in df.columns else 0,
                "total_overtime_cost": float(df["Overtime_Cost"].sum()) if "Overtime_Cost" in df.columns else 0,
                "labor_cost_per_patient": float(df["Labor_Cost_Per_Patient"].mean()) if "Labor_Cost_Per_Patient" in df.columns else 0,
            },
            # Revenue Metrics
            "revenue": {
                "avg_revenue_per_patient": float(df["Revenue_Per_Patient_Avg"].mean()) if "Revenue_Per_Patient_Avg" in df.columns else 0,
                "avg_daily_revenue": float(df["Total_Daily_Revenue"].mean()) if "Total_Daily_Revenue" in df.columns else 0,
                "avg_total_revenue": float(df["Total_Revenue"].mean()) if "Total_Revenue" in df.columns else 0,
                "total_revenue": float(df["Total_Revenue"].sum()) if "Total_Revenue" in df.columns else 0,
                "avg_government_subsidy": float(df["Government_Subsidy"].mean()) if "Government_Subsidy" in df.columns else 0,
                "avg_insurance_reimbursement": float(df["Insurance_Reimbursement"].mean()) if "Insurance_Reimbursement" in df.columns else 0,
            },
            # Profitability Metrics
            "profitability": {
                "avg_daily_profit": float(df["Daily_Profit"].mean()) if "Daily_Profit" in df.columns else 0,
                "total_profit": float(df["Daily_Profit"].sum()) if "Daily_Profit" in df.columns else 0,
                "avg_profit_margin": float(df["Profit_Margin_Percent"].mean()) if "Profit_Margin_Percent" in df.columns else 0,
                "min_profit_margin": float(df["Profit_Margin_Percent"].min()) if "Profit_Margin_Percent" in df.columns else 0,
                "max_profit_margin": float(df["Profit_Margin_Percent"].max()) if "Profit_Margin_Percent" in df.columns else 0,
                "avg_cost_per_patient": float(df["Cost_Per_Patient"].mean()) if "Cost_Per_Patient" in df.columns else 0,
                "avg_revenue_per_patient": float(df["Revenue_Per_Patient"].mean()) if "Revenue_Per_Patient" in df.columns else 0,
            },
            # Operating Cost Metrics
            "operating": {
                "avg_utility_cost": float(df["Utility_Cost"].mean()) if "Utility_Cost" in df.columns else 0,
                "avg_equipment_maintenance": float(df["Equipment_Maintenance_Cost"].mean()) if "Equipment_Maintenance_Cost" in df.columns else 0,
                "avg_administrative_cost": float(df["Administrative_Cost"].mean()) if "Administrative_Cost" in df.columns else 0,
                "avg_facility_cost": float(df["Facility_Cost"].mean()) if "Facility_Cost" in df.columns else 0,
                "avg_total_operating_cost": float(df["Total_Operating_Cost"].mean()) if "Total_Operating_Cost" in df.columns else 0,
            },
            # Cost Ratios
            "ratios": {
                "avg_labor_cost_ratio": float(df["Labor_Cost_Ratio"].mean()) if "Labor_Cost_Ratio" in df.columns else 0,
                "avg_inventory_cost_ratio": float(df["Inventory_Cost_Ratio"].mean()) if "Inventory_Cost_Ratio" in df.columns else 0,
                "avg_budget_variance": float(df["Budget_Variance"].mean()) if "Budget_Variance" in df.columns else 0,
                "avg_budget_variance_pct": float(df["Budget_Variance_Percent"].mean()) if "Budget_Variance_Percent" in df.columns else 0,
            },
            # Optimization Penalty Metrics (Historical Baseline)
            "penalties": {
                "avg_overstaffing_cost": float(df["Overstaffing_Cost"].mean()) if "Overstaffing_Cost" in df.columns else 0,
                "avg_understaffing_penalty": float(df["Understaffing_Penalty"].mean()) if "Understaffing_Penalty" in df.columns else 0,
                "total_overstaffing_cost": float(df["Overstaffing_Cost"].sum()) if "Overstaffing_Cost" in df.columns else 0,
                "total_understaffing_penalty": float(df["Understaffing_Penalty"].sum()) if "Understaffing_Penalty" in df.columns else 0,
                "avg_total_optimization_cost": float(df["Total_Optimization_Cost"].mean()) if "Total_Optimization_Cost" in df.columns else 0,
                "total_optimization_cost": float(df["Total_Optimization_Cost"].sum()) if "Total_Optimization_Cost" in df.columns else 0,
            },
        }

        return kpis

    def calculate_optimization_savings(
        self,
        optimized_labor_cost: float,
        optimized_overtime_cost: float,
        optimized_penalties: float,
        planning_horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate savings from optimization compared to historical data.

        Args:
            optimized_labor_cost: Total labor cost from optimized schedule
            optimized_overtime_cost: Total overtime cost from optimized schedule
            optimized_penalties: Total penalty costs from optimized schedule
            planning_horizon: Number of days in planning horizon

        Returns:
            Dict with savings calculations
        """
        kpis = self.get_comprehensive_financial_kpis(n_days=planning_horizon * 4)  # Get ~1 month historical

        if not kpis:
            return {}

        # Historical averages scaled to planning horizon
        hist_labor = kpis["labor"]["avg_total_labor_cost"] * planning_horizon
        hist_overtime = kpis["labor"]["avg_overtime_cost"] * planning_horizon
        hist_penalties = kpis["penalties"]["avg_total_optimization_cost"] * planning_horizon

        # Calculate savings
        labor_savings = hist_labor - optimized_labor_cost
        overtime_savings = hist_overtime - optimized_overtime_cost
        penalty_savings = hist_penalties - optimized_penalties
        total_savings = labor_savings + overtime_savings + penalty_savings

        # Calculate percentages
        labor_savings_pct = (labor_savings / hist_labor * 100) if hist_labor > 0 else 0
        overtime_savings_pct = (overtime_savings / hist_overtime * 100) if hist_overtime > 0 else 0
        penalty_savings_pct = (penalty_savings / hist_penalties * 100) if hist_penalties > 0 else 0
        total_savings_pct = (total_savings / (hist_labor + hist_overtime + hist_penalties) * 100) if (hist_labor + hist_overtime + hist_penalties) > 0 else 0

        # Estimate hours saved (based on overtime cost reduction)
        avg_hourly_rate = (
            kpis["labor"]["avg_doctor_hourly_rate"] +
            kpis["labor"]["avg_nurse_hourly_rate"] +
            kpis["labor"]["avg_support_hourly_rate"]
        ) / 3

        overtime_multiplier = kpis["labor"]["avg_overtime_rate"] if kpis["labor"]["avg_overtime_rate"] > 0 else 1.5
        overtime_hourly_rate = avg_hourly_rate * overtime_multiplier
        overtime_hours_saved = overtime_savings / overtime_hourly_rate if overtime_hourly_rate > 0 else 0

        # Revenue impact (improved patient care with proper staffing)
        revenue_per_patient = kpis["revenue"]["avg_revenue_per_patient"]
        profit_margin = kpis["profitability"]["avg_profit_margin"]

        return {
            "historical": {
                "labor_cost": hist_labor,
                "overtime_cost": hist_overtime,
                "penalty_cost": hist_penalties,
                "total_cost": hist_labor + hist_overtime + hist_penalties,
            },
            "optimized": {
                "labor_cost": optimized_labor_cost,
                "overtime_cost": optimized_overtime_cost,
                "penalty_cost": optimized_penalties,
                "total_cost": optimized_labor_cost + optimized_overtime_cost + optimized_penalties,
            },
            "savings": {
                "labor_savings": labor_savings,
                "labor_savings_pct": labor_savings_pct,
                "overtime_savings": overtime_savings,
                "overtime_savings_pct": overtime_savings_pct,
                "penalty_savings": penalty_savings,
                "penalty_savings_pct": penalty_savings_pct,
                "total_savings": total_savings,
                "total_savings_pct": total_savings_pct,
                "overtime_hours_saved": overtime_hours_saved,
            },
            "impact": {
                "avg_hourly_rate": avg_hourly_rate,
                "overtime_multiplier": overtime_multiplier,
                "revenue_per_patient": revenue_per_patient,
                "avg_profit_margin": profit_margin,
            },
            "planning_horizon": planning_horizon,
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format the dataframe.

        Args:
            df: Raw dataframe from Supabase

        Returns:
            Cleaned dataframe
        """
        if df.empty:
            return df

        # Drop Supabase internal columns
        drop_cols = ["id", "created_at", "id_numeric"]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Convert Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # Ensure numeric columns are correct type
        numeric_cols = [
            "Doctor_Hourly_Rate", "Nurse_Hourly_Rate", "Support_Staff_Hourly_Rate",
            "Overtime_Premium_Rate", "Doctor_Daily_Cost", "Nurse_Daily_Cost",
            "Support_Staff_Daily_Cost", "Overtime_Cost", "Agency_Staff_Cost",
            "Total_Labor_Cost", "Cost_Per_Glove_Pair", "Cost_Per_PPE_Set",
            "Cost_Per_Medication_Unit", "Gloves_Cost", "PPE_Cost", "Medication_Cost",
            "Total_Inventory_Usage_Cost", "Inventory_Holding_Cost", "Restock_Order_Cost",
            "Emergency_Procurement_Cost", "Inventory_Wastage_Cost", "Total_Inventory_Cost",
            "Revenue_Per_Patient_Avg", "Total_Daily_Revenue", "Government_Subsidy",
            "Insurance_Reimbursement", "Total_Revenue", "Utility_Cost",
            "Equipment_Maintenance_Cost", "Administrative_Cost", "Facility_Cost",
            "Total_Operating_Cost", "Daily_Profit", "Profit_Margin_Percent",
            "Cost_Per_Patient", "Revenue_Per_Patient", "Labor_Cost_Ratio",
            "Inventory_Cost_Ratio", "Budget_Variance", "Budget_Variance_Percent",
            "Labor_Cost_Per_Patient", "Inventory_Cost_Per_Patient", "Overstaffing_Cost",
            "Understaffing_Penalty", "Stockout_Penalty", "Overstock_Penalty",
            "Total_Optimization_Cost"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def insert_records(self, df: pd.DataFrame) -> bool:
        """
        Insert financial records into Supabase.

        Args:
            df: DataFrame with financial data

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        try:
            # Prepare records
            records = []
            for _, row in df.iterrows():
                record = {
                    "Date": row["Date"].isoformat() if hasattr(row["Date"], "isoformat") else str(row["Date"]),
                }

                # Add all numeric columns
                numeric_cols = [col for col in df.columns if col != "Date"]
                for col in numeric_cols:
                    if col in row:
                        val = row[col]
                        if pd.notna(val):
                            record[col] = float(val)
                        else:
                            record[col] = 0.0

                records.append(record)

            # Insert in batches
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.supabase.table(self.table).insert(batch).execute()

            return True

        except Exception as e:
            st.error(f"Error inserting financial data: {str(e)}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_financial_service() -> FinancialService:
    """Get or create FinancialService instance."""
    # Check if cached service has the required methods (handles class updates)
    if "financial_service" in st.session_state:
        service = st.session_state["financial_service"]
        if not hasattr(service, "get_comprehensive_financial_kpis"):
            # Class was updated, create new instance
            st.session_state["financial_service"] = FinancialService()
    else:
        st.session_state["financial_service"] = FinancialService()
    return st.session_state["financial_service"]


def fetch_financial_data(limit: int = None) -> pd.DataFrame:
    """
    Convenience function to fetch financial data.

    Args:
        limit: Optional limit on records

    Returns:
        DataFrame with financial data
    """
    service = get_financial_service()
    return service.fetch_all(limit)


def check_financial_connection() -> bool:
    """Check if Supabase connection is available."""
    service = get_financial_service()
    return service.is_connected()
