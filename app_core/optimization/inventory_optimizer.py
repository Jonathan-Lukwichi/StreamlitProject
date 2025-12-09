# =============================================================================
# app_core/optimization/inventory_optimizer.py
# Deterministic Inventory Optimization (EOQ + MILP)
# For Healthcare Context - When RPIW < 15%
# =============================================================================
"""
Inventory Management Optimization Module

Mathematical Foundation:
-----------------------
1. EOQ (Economic Order Quantity):
   Q* = sqrt(2 * K * D / h)

2. MILP Multi-Item Formulation:
   min Z = sum_t sum_i [K_i * y_{i,t} + c_i * Q_{i,t} + h_i * I_{i,t} + p_i * B_{i,t}]

   Subject to:
   - Inventory Balance: I_{i,t} = I_{i,t-1} + Q_{i,t-L} - d_{i,t} + B_{i,t} - B_{i,t-1}
   - Storage Capacity: sum_i v_i * I_{i,t} <= W
   - Order-Quantity Link: Q_{i,t} <= M * y_{i,t}
   - Minimum Order: Q_{i,t} >= Q_min * y_{i,t}
   - Budget: sum_i c_i * Q_{i,t} <= Budget_t
   - Non-negativity: Q, I, B >= 0, y in {0,1}

3. Reorder Point with Safety Stock:
   s_i = d_i * L_i + SS_i
   SS_i = z_CSL * sigma_i * sqrt(L_i)

Healthcare Context:
------------------
- Critical items (medications, PPE) have high stockout penalties
- Shelf life constraints for medications
- Demand driven by patient arrivals from ML forecasts
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# =============================================================================
# HEALTHCARE INVENTORY ITEM DEFINITIONS
# =============================================================================

class ItemCriticality(Enum):
    """Criticality levels for healthcare items."""
    LOW = 1       # General supplies
    MEDIUM = 2    # Important but not life-critical
    HIGH = 3      # Life-critical items
    CRITICAL = 4  # Emergency/life-saving items


@dataclass
class HealthcareInventoryItem:
    """
    Healthcare inventory item with all parameters.

    Attributes:
        item_id: Unique identifier
        name: Item name
        category: Category (Gloves, PPE, Medications, etc.)
        unit_cost: Cost per unit (USD)
        holding_cost_rate: Annual holding cost as fraction of unit cost (typically 0.15-0.25)
        ordering_cost: Fixed cost per order (USD)
        stockout_penalty: Penalty per unit stockout (USD) - HIGH for healthcare
        lead_time: Days from order to delivery
        usage_rate: Units consumed per patient
        min_order_qty: Minimum order quantity (MOQ)
        max_order_qty: Maximum order quantity
        shelf_life: Days until expiration (None if non-perishable)
        criticality: Item criticality level
        volume_per_unit: Storage volume per unit (cubic meters)
    """
    item_id: str
    name: str
    category: str
    unit_cost: float
    holding_cost_rate: float = 0.20  # 20% annual holding cost
    ordering_cost: float = 50.0      # Fixed ordering cost
    stockout_penalty: float = 100.0  # High for healthcare
    lead_time: int = 3               # Days
    usage_rate: float = 1.0          # Units per patient
    min_order_qty: int = 1
    max_order_qty: int = 10000
    shelf_life: Optional[int] = None
    criticality: ItemCriticality = ItemCriticality.MEDIUM
    volume_per_unit: float = 0.001   # Cubic meters

    @property
    def daily_holding_cost(self) -> float:
        """Daily holding cost per unit."""
        return (self.unit_cost * self.holding_cost_rate) / 365

    @property
    def stockout_multiplier(self) -> float:
        """Multiplier based on criticality."""
        return {
            ItemCriticality.LOW: 1.0,
            ItemCriticality.MEDIUM: 2.0,
            ItemCriticality.HIGH: 5.0,
            ItemCriticality.CRITICAL: 10.0,
        }[self.criticality]

    @property
    def effective_stockout_penalty(self) -> float:
        """Stockout penalty adjusted for criticality."""
        return self.stockout_penalty * self.stockout_multiplier


# =============================================================================
# DEFAULT HEALTHCARE INVENTORY CATALOG
# =============================================================================

DEFAULT_HEALTHCARE_ITEMS = [
    HealthcareInventoryItem(
        item_id="GLV001",
        name="Examination Gloves (Box of 100)",
        category="Gloves",
        unit_cost=8.50,
        holding_cost_rate=0.15,
        ordering_cost=25.0,
        stockout_penalty=50.0,
        lead_time=2,
        usage_rate=5.0,  # 5 pairs per patient (averaged)
        min_order_qty=10,
        max_order_qty=1000,
        shelf_life=730,  # 2 years
        criticality=ItemCriticality.HIGH,
        volume_per_unit=0.005,
    ),
    HealthcareInventoryItem(
        item_id="PPE001",
        name="PPE Kit (Gown, Mask, Face Shield)",
        category="PPE",
        unit_cost=45.00,
        holding_cost_rate=0.20,
        ordering_cost=75.0,
        stockout_penalty=200.0,
        lead_time=5,
        usage_rate=0.3,  # 30% of patients need full PPE
        min_order_qty=50,
        max_order_qty=2000,
        shelf_life=1095,  # 3 years
        criticality=ItemCriticality.CRITICAL,
        volume_per_unit=0.015,
    ),
    HealthcareInventoryItem(
        item_id="MED001",
        name="Emergency Medication Kit",
        category="Medications",
        unit_cost=125.00,
        holding_cost_rate=0.25,
        ordering_cost=100.0,
        stockout_penalty=500.0,
        lead_time=3,
        usage_rate=0.15,  # 15% of patients
        min_order_qty=20,
        max_order_qty=500,
        shelf_life=365,  # 1 year
        criticality=ItemCriticality.CRITICAL,
        volume_per_unit=0.002,
    ),
    HealthcareInventoryItem(
        item_id="SYR001",
        name="Syringes (Box of 100)",
        category="Medical Supplies",
        unit_cost=12.00,
        holding_cost_rate=0.15,
        ordering_cost=30.0,
        stockout_penalty=75.0,
        lead_time=2,
        usage_rate=2.0,  # 2 per patient
        min_order_qty=20,
        max_order_qty=1000,
        shelf_life=1825,  # 5 years
        criticality=ItemCriticality.HIGH,
        volume_per_unit=0.003,
    ),
    HealthcareInventoryItem(
        item_id="BND001",
        name="Bandages & Dressings Kit",
        category="Medical Supplies",
        unit_cost=18.00,
        holding_cost_rate=0.15,
        ordering_cost=25.0,
        stockout_penalty=40.0,
        lead_time=2,
        usage_rate=1.5,  # 1.5 per patient
        min_order_qty=25,
        max_order_qty=800,
        shelf_life=1095,  # 3 years
        criticality=ItemCriticality.MEDIUM,
        volume_per_unit=0.004,
    ),
    HealthcareInventoryItem(
        item_id="IV001",
        name="IV Fluids (Saline 1L)",
        category="IV Supplies",
        unit_cost=3.50,
        holding_cost_rate=0.20,
        ordering_cost=40.0,
        stockout_penalty=150.0,
        lead_time=3,
        usage_rate=0.8,  # 80% of patients
        min_order_qty=100,
        max_order_qty=2000,
        shelf_life=730,  # 2 years
        criticality=ItemCriticality.CRITICAL,
        volume_per_unit=0.001,
    ),
    HealthcareInventoryItem(
        item_id="MSK001",
        name="Surgical Masks (Box of 50)",
        category="PPE",
        unit_cost=15.00,
        holding_cost_rate=0.15,
        ordering_cost=20.0,
        stockout_penalty=60.0,
        lead_time=2,
        usage_rate=3.0,  # 3 per patient
        min_order_qty=20,
        max_order_qty=1000,
        shelf_life=1095,  # 3 years
        criticality=ItemCriticality.HIGH,
        volume_per_unit=0.008,
    ),
]


# =============================================================================
# EOQ MODEL
# =============================================================================

@dataclass
class EOQResult:
    """Results from EOQ calculation."""
    item_id: str
    item_name: str
    eoq: float                    # Economic Order Quantity
    reorder_point: float          # When to reorder
    safety_stock: float           # Safety stock level
    annual_demand: float          # Annual demand
    annual_ordering_cost: float   # Total ordering cost
    annual_holding_cost: float    # Total holding cost
    total_annual_cost: float      # Total inventory cost
    orders_per_year: float        # Number of orders
    cycle_time_days: float        # Days between orders
    average_inventory: float      # Average inventory level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Item ID": self.item_id,
            "Item Name": self.item_name,
            "EOQ": round(self.eoq),
            "Reorder Point": round(self.reorder_point),
            "Safety Stock": round(self.safety_stock),
            "Annual Demand": round(self.annual_demand),
            "Ordering Cost/Year": f"${self.annual_ordering_cost:,.2f}",
            "Holding Cost/Year": f"${self.annual_holding_cost:,.2f}",
            "Total Cost/Year": f"${self.total_annual_cost:,.2f}",
            "Orders/Year": round(self.orders_per_year, 1),
            "Cycle Time (days)": round(self.cycle_time_days, 1),
            "Avg Inventory": round(self.average_inventory),
        }


def calculate_eoq(
    item: HealthcareInventoryItem,
    daily_demand: float,
    demand_std: float = None,
    service_level: float = 0.95,
) -> EOQResult:
    """
    Calculate Economic Order Quantity for a single item.

    EOQ Formula: Q* = sqrt(2 * K * D / h)

    Args:
        item: Healthcare inventory item
        daily_demand: Average daily demand (units)
        demand_std: Standard deviation of daily demand (for safety stock)
        service_level: Target service level (e.g., 0.95 for 95%)

    Returns:
        EOQResult with all calculated values
    """
    from scipy import stats

    # Annual demand
    D = daily_demand * 365

    # Ordering cost per order
    K = item.ordering_cost

    # Annual holding cost per unit
    h = item.unit_cost * item.holding_cost_rate

    # EOQ calculation
    if D > 0 and h > 0:
        Q_star = np.sqrt(2 * K * D / h)
    else:
        Q_star = item.min_order_qty

    # Enforce min/max constraints
    Q_star = max(item.min_order_qty, min(item.max_order_qty, Q_star))

    # Safety stock calculation
    if demand_std is not None and demand_std > 0:
        # z-score for service level
        z = stats.norm.ppf(service_level)
        # Safety stock: z * sigma * sqrt(L)
        SS = z * demand_std * np.sqrt(item.lead_time)
    else:
        # If no std provided, use 10% of lead time demand as safety stock
        SS = 0.1 * daily_demand * item.lead_time

    # Reorder point
    ROP = daily_demand * item.lead_time + SS

    # Cost calculations
    orders_per_year = D / Q_star if Q_star > 0 else 0
    annual_ordering_cost = K * orders_per_year
    average_inventory = Q_star / 2 + SS
    annual_holding_cost = h * average_inventory
    total_cost = annual_ordering_cost + annual_holding_cost

    # Cycle time
    cycle_time = 365 / orders_per_year if orders_per_year > 0 else 365

    return EOQResult(
        item_id=item.item_id,
        item_name=item.name,
        eoq=Q_star,
        reorder_point=ROP,
        safety_stock=SS,
        annual_demand=D,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_cost=total_cost,
        orders_per_year=orders_per_year,
        cycle_time_days=cycle_time,
        average_inventory=average_inventory,
    )


# =============================================================================
# MILP MULTI-ITEM OPTIMIZATION
# =============================================================================

@dataclass
class MILPResult:
    """Results from MILP optimization."""
    status: str                           # Optimal, Infeasible, etc.
    total_cost: float                     # Total objective value
    ordering_cost: float                  # Total ordering cost
    holding_cost: float                   # Total holding cost
    stockout_cost: float                  # Total stockout/backorder cost
    purchase_cost: float                  # Total purchase cost
    order_schedule: pd.DataFrame          # When and how much to order
    inventory_levels: pd.DataFrame        # Inventory levels over time
    reorder_points: Dict[str, float]      # Reorder point per item
    safety_stocks: Dict[str, float]       # Safety stock per item
    solver_time: float                    # Solve time in seconds

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Status": self.status,
            "Total Cost": f"${self.total_cost:,.2f}",
            "Ordering Cost": f"${self.ordering_cost:,.2f}",
            "Holding Cost": f"${self.holding_cost:,.2f}",
            "Stockout Cost": f"${self.stockout_cost:,.2f}",
            "Purchase Cost": f"${self.purchase_cost:,.2f}",
            "Solver Time": f"{self.solver_time:.2f}s",
        }


def solve_milp_inventory(
    items: List[HealthcareInventoryItem],
    demand_forecast: Dict[str, List[float]],  # item_id -> daily demands
    planning_horizon: int = 30,
    initial_inventory: Dict[str, float] = None,
    storage_capacity: float = 1000.0,  # Cubic meters
    daily_budget: float = None,
    service_level: float = 0.95,
) -> MILPResult:
    """
    Solve multi-item inventory optimization using MILP.

    Objective:
        min Z = sum_t sum_i [K_i * y_{i,t} + c_i * Q_{i,t} + h_i * I_{i,t} + p_i * B_{i,t}]

    Args:
        items: List of inventory items
        demand_forecast: Dict mapping item_id to list of daily demands
        planning_horizon: Number of days to plan
        initial_inventory: Starting inventory levels
        storage_capacity: Total storage capacity (cubic meters)
        daily_budget: Optional daily budget constraint
        service_level: Target service level for safety stock

    Returns:
        MILPResult with optimization results
    """
    import time
    try:
        from pulp import (
            LpProblem, LpMinimize, LpVariable, LpBinary,
            LpContinuous, lpSum, LpStatus, value
        )
    except ImportError:
        raise ImportError("PuLP is required. Install with: pip install pulp")

    start_time = time.time()

    # Initialize
    T = planning_horizon
    I = {item.item_id: item for item in items}

    # Default initial inventory
    if initial_inventory is None:
        initial_inventory = {item.item_id: 0 for item in items}

    # Ensure demand forecast covers all items
    for item in items:
        if item.item_id not in demand_forecast:
            demand_forecast[item.item_id] = [0.0] * T
        # Pad or trim to planning horizon
        d = demand_forecast[item.item_id]
        if len(d) < T:
            d = d + [d[-1] if d else 0] * (T - len(d))
        demand_forecast[item.item_id] = d[:T]

    # Create problem
    prob = LpProblem("Healthcare_Inventory_Optimization", LpMinimize)

    # Big M for order-quantity link
    M = 100000

    # ==========================================================================
    # DECISION VARIABLES
    # ==========================================================================

    # Q[i,t] = order quantity for item i at time t
    Q = {
        (i, t): LpVariable(f"Q_{i}_{t}", lowBound=0, cat=LpContinuous)
        for i in I.keys() for t in range(T)
    }

    # y[i,t] = 1 if order placed for item i at time t
    y = {
        (i, t): LpVariable(f"y_{i}_{t}", cat=LpBinary)
        for i in I.keys() for t in range(T)
    }

    # Inv[i,t] = inventory level for item i at end of period t
    Inv = {
        (i, t): LpVariable(f"Inv_{i}_{t}", lowBound=0, cat=LpContinuous)
        for i in I.keys() for t in range(T)
    }

    # B[i,t] = backorder/stockout quantity for item i at time t
    B = {
        (i, t): LpVariable(f"B_{i}_{t}", lowBound=0, cat=LpContinuous)
        for i in I.keys() for t in range(T)
    }

    # ==========================================================================
    # OBJECTIVE FUNCTION
    # ==========================================================================

    # Total cost = ordering + purchase + holding + stockout
    ordering_cost_expr = lpSum([
        I[i].ordering_cost * y[i, t]
        for i in I.keys() for t in range(T)
    ])

    purchase_cost_expr = lpSum([
        I[i].unit_cost * Q[i, t]
        for i in I.keys() for t in range(T)
    ])

    holding_cost_expr = lpSum([
        I[i].daily_holding_cost * Inv[i, t]
        for i in I.keys() for t in range(T)
    ])

    stockout_cost_expr = lpSum([
        I[i].effective_stockout_penalty * B[i, t]
        for i in I.keys() for t in range(T)
    ])

    prob += ordering_cost_expr + purchase_cost_expr + holding_cost_expr + stockout_cost_expr

    # ==========================================================================
    # CONSTRAINTS
    # ==========================================================================

    for i in I.keys():
        item = I[i]
        L = item.lead_time

        for t in range(T):
            # Demand at time t
            d_t = demand_forecast[i][t]

            # -----------------------------------------------------------------
            # C1: Inventory Balance
            # I[t] = I[t-1] + Q[t-L] - d[t] + B[t] - B[t-1]
            # -----------------------------------------------------------------
            if t == 0:
                I_prev = initial_inventory.get(i, 0)
                B_prev = 0
            else:
                I_prev = Inv[i, t-1]
                B_prev = B[i, t-1]

            # Orders placed at t-L arrive at t
            if t >= L:
                Q_arriving = Q[i, t - L]
            else:
                Q_arriving = 0  # No orders arriving yet

            prob += (
                Inv[i, t] == I_prev + Q_arriving - d_t + B[i, t] - B_prev,
                f"InvBalance_{i}_{t}"
            )

            # -----------------------------------------------------------------
            # C2: Order-Quantity Link (if order placed, Q <= M * y)
            # -----------------------------------------------------------------
            prob += Q[i, t] <= M * y[i, t], f"OrderLink_{i}_{t}"

            # -----------------------------------------------------------------
            # C3: Minimum Order Quantity
            # -----------------------------------------------------------------
            prob += Q[i, t] >= item.min_order_qty * y[i, t], f"MinOrder_{i}_{t}"

            # -----------------------------------------------------------------
            # C4: Maximum Order Quantity
            # -----------------------------------------------------------------
            prob += Q[i, t] <= item.max_order_qty, f"MaxOrder_{i}_{t}"

    # -------------------------------------------------------------------------
    # C5: Storage Capacity (per period)
    # -------------------------------------------------------------------------
    for t in range(T):
        prob += (
            lpSum([I[i].volume_per_unit * Inv[i, t] for i in I.keys()]) <= storage_capacity,
            f"StorageCap_{t}"
        )

    # -------------------------------------------------------------------------
    # C6: Budget Constraint (optional)
    # -------------------------------------------------------------------------
    if daily_budget is not None:
        for t in range(T):
            prob += (
                lpSum([I[i].unit_cost * Q[i, t] for i in I.keys()]) <= daily_budget,
                f"Budget_{t}"
            )

    # ==========================================================================
    # SOLVE
    # ==========================================================================

    prob.solve()
    solve_time = time.time() - start_time
    status = LpStatus[prob.status]

    # ==========================================================================
    # EXTRACT RESULTS
    # ==========================================================================

    if status == "Optimal":
        # Order schedule
        order_data = []
        for t in range(T):
            row = {"Day": t + 1}
            for i in I.keys():
                q_val = value(Q[i, t]) or 0
                y_val = value(y[i, t]) or 0
                if q_val > 0.5:
                    row[f"{I[i].name[:20]}"] = int(round(q_val))
                else:
                    row[f"{I[i].name[:20]}"] = 0
            order_data.append(row)
        order_schedule = pd.DataFrame(order_data)

        # Inventory levels
        inv_data = []
        for t in range(T):
            row = {"Day": t + 1}
            for i in I.keys():
                inv_val = value(Inv[i, t]) or 0
                row[f"{I[i].name[:20]}"] = int(round(inv_val))
            inv_data.append(row)
        inventory_levels = pd.DataFrame(inv_data)

        # Calculate cost components
        total_ordering = sum(
            I[i].ordering_cost * (value(y[i, t]) or 0)
            for i in I.keys() for t in range(T)
        )
        total_purchase = sum(
            I[i].unit_cost * (value(Q[i, t]) or 0)
            for i in I.keys() for t in range(T)
        )
        total_holding = sum(
            I[i].daily_holding_cost * (value(Inv[i, t]) or 0)
            for i in I.keys() for t in range(T)
        )
        total_stockout = sum(
            I[i].effective_stockout_penalty * (value(B[i, t]) or 0)
            for i in I.keys() for t in range(T)
        )

        # Calculate reorder points and safety stocks
        from scipy import stats
        z = stats.norm.ppf(service_level)

        reorder_points = {}
        safety_stocks = {}
        for i in I.keys():
            item = I[i]
            demands = demand_forecast[i]
            avg_demand = np.mean(demands) if demands else 0
            std_demand = np.std(demands) if len(demands) > 1 else avg_demand * 0.1

            SS = z * std_demand * np.sqrt(item.lead_time)
            ROP = avg_demand * item.lead_time + SS

            safety_stocks[i] = SS
            reorder_points[i] = ROP

        return MILPResult(
            status=status,
            total_cost=value(prob.objective) or 0,
            ordering_cost=total_ordering,
            holding_cost=total_holding,
            stockout_cost=total_stockout,
            purchase_cost=total_purchase,
            order_schedule=order_schedule,
            inventory_levels=inventory_levels,
            reorder_points=reorder_points,
            safety_stocks=safety_stocks,
            solver_time=solve_time,
        )
    else:
        # Return empty result if not optimal
        return MILPResult(
            status=status,
            total_cost=0,
            ordering_cost=0,
            holding_cost=0,
            stockout_cost=0,
            purchase_cost=0,
            order_schedule=pd.DataFrame(),
            inventory_levels=pd.DataFrame(),
            reorder_points={},
            safety_stocks={},
            solver_time=solve_time,
        )


# =============================================================================
# DEMAND CONVERSION: Patient Forecasts -> Inventory Demand
# =============================================================================

def convert_patient_forecast_to_inventory_demand(
    patient_forecast: List[float],
    items: List[HealthcareInventoryItem],
) -> Dict[str, List[float]]:
    """
    Convert patient arrival forecasts to inventory item demands.

    Formula: d_{i,t} = alpha_i * D_t
    Where alpha_i is the usage rate per patient for item i.

    Args:
        patient_forecast: List of daily patient arrival forecasts
        items: List of inventory items with usage rates

    Returns:
        Dict mapping item_id to list of daily demands
    """
    demand_forecast = {}

    for item in items:
        # d_{i,t} = usage_rate * patient_count
        item_demands = [
            item.usage_rate * patients
            for patients in patient_forecast
        ]
        demand_forecast[item.item_id] = item_demands

    return demand_forecast


def calculate_rpiw(
    forecast_values: List[float],
    lower_bounds: List[float] = None,
    upper_bounds: List[float] = None,
) -> float:
    """
    Calculate Relative Prediction Interval Width (RPIW).

    RPIW = (U - L) / D_hat * 100%

    Args:
        forecast_values: Point forecasts
        lower_bounds: Lower prediction interval bounds
        upper_bounds: Upper prediction interval bounds

    Returns:
        RPIW as percentage
    """
    if lower_bounds is None or upper_bounds is None:
        # Estimate bounds from forecast variance
        mean_forecast = np.mean(forecast_values)
        std_forecast = np.std(forecast_values)

        # 90% prediction interval
        lower_bounds = [max(0, f - 1.645 * std_forecast) for f in forecast_values]
        upper_bounds = [f + 1.645 * std_forecast for f in forecast_values]

    # Calculate RPIW
    rpiw_values = []
    for d_hat, l, u in zip(forecast_values, lower_bounds, upper_bounds):
        if d_hat > 0:
            rpiw = ((u - l) / d_hat) * 100
            rpiw_values.append(rpiw)

    return np.mean(rpiw_values) if rpiw_values else 0.0


def get_optimization_recommendation(rpiw: float) -> Dict[str, Any]:
    """
    Get optimization method recommendation based on RPIW.

    RPIW < 15%: Deterministic EOQ/MILP
    15% <= RPIW <= 40%: Robust (s,S) Policy
    RPIW > 40%: Stochastic (s,S) Policy

    Args:
        rpiw: Relative Prediction Interval Width (%)

    Returns:
        Dict with recommendation details
    """
    if rpiw < 15:
        return {
            "method": "Deterministic EOQ/MILP",
            "rpiw": rpiw,
            "uncertainty_level": "LOW",
            "color": "green",
            "safety_stock": "Minimal (~1 day)",
            "computation": "Fast",
            "description": "Demand forecasts are highly accurate. Minimal safety stock needed.",
        }
    elif rpiw <= 40:
        return {
            "method": "Robust (s,S) Policy",
            "rpiw": rpiw,
            "uncertainty_level": "MEDIUM",
            "color": "yellow",
            "safety_stock": "From interval width",
            "computation": "Fast",
            "description": "Moderate uncertainty. Use prediction intervals directly for robust protection.",
        }
    else:
        return {
            "method": "Stochastic (s,S) Policy",
            "rpiw": rpiw,
            "uncertainty_level": "HIGH",
            "color": "red",
            "safety_stock": "From distribution",
            "computation": "Slow",
            "description": "High uncertainty. Explicit scenario modeling required.",
        }


# =============================================================================
# REORDER ALERTS
# =============================================================================

def generate_reorder_alerts(
    current_inventory: Dict[str, float],
    reorder_points: Dict[str, float],
    items: List[HealthcareInventoryItem],
) -> List[Dict[str, Any]]:
    """
    Generate reorder alerts based on current inventory vs reorder points.

    Args:
        current_inventory: Current inventory levels by item_id
        reorder_points: Reorder points by item_id
        items: List of inventory items

    Returns:
        List of alert dictionaries
    """
    alerts = []

    for item in items:
        current = current_inventory.get(item.item_id, 0)
        rop = reorder_points.get(item.item_id, 0)

        if current <= rop:
            # Calculate urgency
            if current <= rop * 0.5:
                urgency = "CRITICAL"
                color = "red"
            elif current <= rop * 0.75:
                urgency = "HIGH"
                color = "orange"
            else:
                urgency = "MEDIUM"
                color = "yellow"

            alerts.append({
                "item_id": item.item_id,
                "item_name": item.name,
                "category": item.category,
                "current_inventory": current,
                "reorder_point": rop,
                "urgency": urgency,
                "color": color,
                "recommended_order": max(item.min_order_qty, rop - current),
                "lead_time": item.lead_time,
            })

    # Sort by urgency
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    alerts.sort(key=lambda x: urgency_order.get(x["urgency"], 3))

    return alerts
