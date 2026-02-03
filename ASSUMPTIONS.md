# Modeling Assumptions and Scope Definition

## Healthcare Resource Optimization System

**Document Version:** 1.0
**Last Updated:** February 2026
**Scope:** Staff Scheduling Optimization (MILP) | Inventory Optimization (EOQ/Safety Stock)

---

## Abstract

This document formally defines the assumptions underlying the mathematical models implemented in the Staff Planner and Supply Planner modules. These assumptions establish the boundary conditions for model validity and define the operational context within which optimization results should be interpreted. The assumptions are categorized by model component and are consistent with standard practices in operations research literature.

---

## 1. Introduction

### 1.1 Purpose

Mathematical optimization models are abstractions of real-world systems. The validity and applicability of any optimization model depends critically on the assumptions made during its formulation. This document serves to:

1. **Establish model boundaries** — Define what phenomena the models capture and what they deliberately exclude
2. **Ensure reproducibility** — Enable other researchers to understand and replicate the methodology
3. **Guide interpretation** — Inform practitioners about the conditions under which model recommendations are valid
4. **Identify limitations** — Transparently acknowledge simplifications made for analytical tractability

### 1.2 Notation Convention

Throughout this document, assumptions are coded as follows:
- **S-XX**: Staff Planner assumptions
- **I-XX**: Inventory/Supply Planner assumptions
- **G-XX**: General assumptions applicable to both models

---

## 2. Staff Planning Model Assumptions

The staff planning module employs Mixed Integer Linear Programming (MILP) to determine optimal daily staffing levels across multiple personnel categories (physicians, nurses, support staff) and clinical departments.

### 2.1 Demand Model Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **S-D1** | **Deterministic Demand.** Patient arrival forecasts are treated as point estimates without associated uncertainty distributions. | The optimization problem is formulated as a deterministic MILP rather than a stochastic program. Let d_t denote demand on day t; we assume d_t = d̂_t (forecast) with probability 1. |
| **S-D2** | **Daily Aggregation.** Patient demand is aggregated at the daily level; intra-day arrival patterns are not modeled. | The decision epoch is one day. Let T = {1, 2, ..., H} denote the planning horizon in days. |
| **S-D3** | **Categorical Decomposition.** Aggregate daily demand is distributed across clinical categories using learned seasonal proportions. | Let d_t = Σ(c∈C) d_ct where d_ct = π_c(dow_t, month_t) · d_t and π_c(·) represents the seasonal proportion function for category c. |
| **S-D4** | **Exogenous Demand.** Patient arrivals are independent of staffing decisions; demand is not affected by service capacity. | Demand d_t is a parameter, not a decision variable. No demand-capacity feedback loops are modeled. |

### 2.2 Workforce Model Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **S-W1** | **Homogeneous Workforce.** Within each staff category s ∈ S = {Doctors, Nurses, Support}, all personnel are identical in skill, productivity, and cost. | Staff are counted as integers x_st ∈ Z⁺. No individual-level modeling. |
| **S-W2** | **Single Shift Structure.** One 8-hour shift per day is assumed; multi-shift scheduling is outside scope. | Shift index is suppressed. The model solves for daily headcount, not shift assignment. |
| **S-W3** | **Perfect Flexibility.** Staff can be assigned to any clinical category within their professional scope. | No assignment constraints x_sct by category c; only aggregate x_st is decided. |
| **S-W4** | **Unlimited Labor Pool.** Sufficient staff are available in the external labor market at the specified wage rates. | No upper bound on hiring; constraint is budget, not availability. |
| **S-W5** | **No Carryover Effects.** Daily staffing decisions are independent; fatigue accumulation, consecutive working day limits, and rest requirements are not modeled. | Temporal coupling constraints are absent: x_st does not depend on x_s,t-1. |

### 2.3 Cost Model Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **S-C1** | **Linear Cost Structure.** Labor costs are linear in the number of staff-hours. | Cost function: C_s(x) = w_s · h · x where w_s is hourly wage and h is shift length. |
| **S-C2** | **Constant Wage Rates.** Hourly rates do not vary by day of week, seniority, or market conditions. | w_s is a scalar parameter, not a function of t or individual attributes. |
| **S-C3** | **Overtime Availability.** Overtime is available at a fixed premium (default: 1.5×). | C_OT = μ · w_s · h_OT where μ = 1.5 is the overtime multiplier. |
| **S-C4** | **Linear Penalty Functions.** Understaffing and overstaffing incur linear penalties. | Penalty: P = α · U + β · O where U, O are understaffing/overstaffing quantities and α, β are penalty coefficients. |

### 2.4 Constraint Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **S-X1** | **Fixed Staffing Ratios.** Patient-to-staff ratios by category are known constants derived from clinical guidelines. | Constraint: x_st ≥ Σ_c d_ct / r_sc where r_sc is the ratio for staff type s in category c. |
| **S-X2** | **Box Constraints on Headcount.** Minimum and maximum staffing levels are exogenously specified. | x̲_s ≤ x_st ≤ x̄_s  ∀t, s |
| **S-X3** | **Budget Constraints.** Daily and/or weekly budget caps may be imposed. | Σ_s C_s(x_st) ≤ B_t^day and Σ_t Σ_s C_s(x_st) ≤ B^week |
| **S-X4** | **Skill Mix Ratios.** Minimum nurse-to-doctor ratios may be enforced. | x_Nurse,t ≥ ρ · x_Doctor,t where ρ is the required ratio. |

---

## 3. Supply Planning Model Assumptions

The supply planning module employs Economic Order Quantity (EOQ) theory with statistical safety stock to determine optimal order quantities and reorder points for medical supplies.

### 3.1 Demand Model Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-D1** | **Derived Demand.** Supply consumption is a deterministic function of patient volume. | D_i = Σ_t d_t · u_i where u_i is usage rate of item i per patient. |
| **I-D2** | **Stationary Demand Distribution.** Demand variability is characterized by a constant standard deviation estimated from historical data. | σ_i = std(D_i,historical) or estimated as σ_i = 0.2 · D̄_i when data is insufficient. |
| **I-D3** | **Independent Demand.** Demand for different inventory items is statistically independent. | Cov(D_i, D_j) = 0  ∀i ≠ j |
| **I-D4** | **Normally Distributed Demand.** Demand during lead time follows a normal distribution. | D_L ~ N(μ_L, σ_L²) where μ_L = d̄ · L and σ_L = σ_d · √L |

### 3.2 Lead Time Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-L1** | **Deterministic Lead Time.** Supplier lead times are constant and known with certainty. | L_i = L̄_i (no variability). This simplifies the safety stock formula. |
| **I-L2** | **Lead Time Independence.** Lead time does not depend on order quantity or system state. | No economies or diseconomies of scale in delivery time. |

### 3.3 Cost Model Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-C1** | **Constant Unit Costs.** Purchase prices do not vary with order quantity (no quantity discounts). | C_purchase = c_i · Q_i (linear in Q). |
| **I-C2** | **Fixed Ordering Cost.** Each order incurs a fixed administrative/setup cost independent of quantity. | C_order = K per order, where K is constant. |
| **I-C3** | **Linear Holding Cost.** Inventory carrying cost is proportional to average inventory level and unit cost. | C_hold = h · c_i · Ī where h is the holding cost rate (fraction per year) and Ī is average inventory. |
| **I-C4** | **Implicit Stockout Cost.** Stockout costs are not explicitly modeled; instead, a service level constraint ensures low stockout probability. | Service level α implies P(stockout) ≤ 1 - α. |

### 3.4 Inventory Policy Assumptions

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-P1** | **Continuous Review Policy.** Inventory position is monitored continuously; orders are placed when inventory reaches the reorder point. | (Q, R) policy where Q = order quantity, R = reorder point. |
| **I-P2** | **Single Echelon.** Only one inventory location is modeled (central stockroom). | No warehouse-to-department transfers or multi-location pooling. |
| **I-P3** | **No Perishability.** Items do not expire or deteriorate within the planning horizon. | Shelf life >> planning horizon. No FEFO constraints. |
| **I-P4** | **No Backorders.** Unsatisfied demand is lost (or handled outside the model). | Stockouts incur implicit cost through service level constraint, not backorder tracking. |
| **I-P5** | **Instantaneous Replenishment.** Orders arrive as a single batch (no partial shipments). | Inventory jumps by Q units upon delivery. |

### 3.5 Safety Stock Formulation

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-S1** | **Standard Safety Stock Formula.** Safety stock is calculated using the standard normal approximation. | SS = z_α · σ_d · √L where z_α = Φ⁻¹(α) is the standard normal quantile for service level α. |
| **I-S2** | **Cycle Service Level.** The service level refers to the probability of no stockout during a replenishment cycle. | P(D_L ≤ R) = α where R = d̄ · L + SS |

### 3.6 EOQ Formulation

| Code | Assumption | Mathematical Implication |
|------|------------|-------------------------|
| **I-E1** | **Classic EOQ Conditions.** Demand rate is constant, lead time is constant, costs are as specified above. | Q* = √(2KD / hc) minimizes total cost TC = (D/Q)K + (Q/2)hc + Dc |
| **I-E2** | **Infinite Planning Horizon.** EOQ is derived assuming ongoing operations without a terminal date. | The formula assumes steady-state operations. |

---

## 4. General Assumptions

These assumptions apply to both optimization models.

| Code | Assumption | Justification |
|------|------------|---------------|
| **G-1** | **Synthetic Data Validity.** Synthetic test data is structurally representative of realistic hospital operations. | Data generation follows patterns documented in healthcare operations literature. Model validation focuses on methodological correctness rather than empirical parameter accuracy. |
| **G-2** | **Forecast Accuracy.** The upstream forecasting module provides sufficiently accurate predictions for planning purposes. | Integration assumes forecast errors are within acceptable bounds. Forecast error propagation is not explicitly modeled. |
| **G-3** | **Model Independence.** Staff and inventory planning are solved sequentially, not jointly. | No feedback between staffing levels and supply consumption rates. Each optimization is self-contained. |
| **G-4** | **Solver Optimality.** The MILP solver (PuLP/CBC) finds optimal or near-optimal solutions within specified time limits and gap tolerances. | Commercial-grade solver performance is assumed adequate for problem sizes encountered. |
| **G-5** | **Parameter Stationarity.** Model parameters (costs, ratios, lead times) are constant over the planning horizon. | No inflation, contract renegotiation, or supplier changes during the planning period. |

---

## 5. Scope Exclusions

The following phenomena are explicitly **outside the scope** of the current implementation:

### 5.1 Staff Planning Exclusions

| Exclusion | Rationale |
|-----------|-----------|
| Multi-shift scheduling (day/night/swing) | Shift scheduling is a separate combinatorial optimization problem with different decision variables and constraints. |
| Individual staff assignment and rostering | Requires personnel-level data and introduces significant computational complexity. |
| Fatigue management and consecutive day limits | Requires temporal coupling constraints that complicate the model structure. |
| Staff preferences and fairness constraints | Outside scope; focus is on cost-optimal aggregate staffing. |
| Stochastic demand / robust optimization | Deferred to future work; current focus is deterministic methodology. |
| Real-time re-optimization | Model provides medium-term plans, not real-time adjustments. |

### 5.2 Supply Planning Exclusions

| Exclusion | Rationale |
|-----------|-----------|
| Quantity discounts | Would require modified EOQ formulation (incremental or all-units discount models). |
| Stochastic lead times | Adds complexity to safety stock calculation: SS = z_α √(L·σ_d² + d̄²·σ_L²) |
| Multi-echelon inventory | Requires network optimization; current scope is single-location. |
| Perishable inventory / expiration | Requires FEFO policies and wastage tracking. |
| Correlated item demand | Joint replenishment and demand correlation would require different modeling approach. |
| ABC/XYZ classification | All items treated uniformly; differentiated policies not implemented. |

---

## 6. Assumption Validation

### 6.1 Internal Consistency

The assumptions have been reviewed for mutual consistency:

- **S-D1** (deterministic demand) is consistent with **I-D1** (derived demand) as both treat forecasts as point estimates
- **S-C1** (linear costs) enables MILP formulation compatible with **G-4** (solver optimality)
- **I-S1** (normal distribution) is consistent with **I-D4** and enables closed-form safety stock calculation

### 6.2 Literature Support

Key assumptions are supported by standard operations research references:

| Assumption Category | Supporting Literature |
|--------------------|----------------------|
| EOQ formulation | Harris (1913), Wilson (1934) |
| Safety stock formula | Silver, Pyke & Thomas (2016), Chopra & Meindl (2018) |
| MILP workforce planning | Ernst et al. (2004), Van den Bergh et al. (2013) |
| Service level approach | Axsäter (2015), Nahmias & Olsen (2015) |

---

## 7. Future Work

The following extensions would relax current assumptions and enhance model realism:

1. **Stochastic Programming** — Replace deterministic demand with scenario-based or distributionally robust formulations
2. **Multi-Shift Scheduling** — Extend staff model to include shift assignment variables
3. **Variable Lead Times** — Incorporate lead time variability into safety stock calculations
4. **Multi-Echelon Inventory** — Model central warehouse and department-level stock interactions
5. **Joint Staff-Inventory Optimization** — Develop integrated model capturing interactions between capacity and consumption
6. **Empirical Validation** — Validate model parameters and performance against real hospital operational data

---

## 8. Conclusion

This document establishes the formal assumptions underlying the staff and supply planning optimization models. These assumptions represent deliberate simplifications made to ensure analytical tractability while maintaining alignment with established operations research methodology. Users of these models should interpret optimization results within the context of these stated assumptions and recognize that relaxing specific assumptions (e.g., introducing demand uncertainty) would require corresponding modifications to the mathematical formulations.

The assumptions defined herein provide a solid foundation for demonstrating the integration of demand forecasting with operational optimization in healthcare settings. Extensions addressing the identified scope exclusions represent natural directions for future research.

---

## References

- Axsäter, S. (2015). *Inventory Control* (3rd ed.). Springer.
- Chopra, S., & Meindl, P. (2018). *Supply Chain Management: Strategy, Planning, and Operation* (7th ed.). Pearson.
- Ernst, A. T., Jiang, H., Krishnamoorthy, M., & Sier, D. (2004). Staff scheduling and rostering: A review of applications, methods and models. *European Journal of Operational Research*, 153(1), 3-27.
- Harris, F. W. (1913). How many parts to make at once. *Factory, The Magazine of Management*, 10(2), 135-136.
- Nahmias, S., & Olsen, T. L. (2015). *Production and Operations Analysis* (7th ed.). Waveland Press.
- Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
- Van den Bergh, J., Beliën, J., De Bruecker, P., Demeulemeester, E., & De Boeck, L. (2013). Personnel scheduling: A literature review. *European Journal of Operational Research*, 226(3), 367-385.
- Wilson, R. H. (1934). A scientific routine for stock control. *Harvard Business Review*, 13(1), 116-128.

---

*Document prepared in accordance with academic standards for operations research model documentation.*
