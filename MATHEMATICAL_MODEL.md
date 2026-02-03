# Mathematical Model Formulation

## Integrated Demand Forecasting and Resource Optimization for Hospital Emergency Departments

**Document Version:** 1.0
**Domain:** Operations Research, Machine Learning, Healthcare Analytics
**Application:** Hospital Resource Planning System

---

## Abstract

This document presents the complete mathematical formulation of an integrated hospital resource planning system. The system comprises two primary stages: (1) **Demand Forecasting** using statistical and machine learning models to predict Emergency Department (ED) patient arrivals, and (2) **Resource Optimization** using Mixed Integer Linear Programming (MILP) for staff scheduling and Economic Order Quantity (EOQ) theory with statistical safety stock for inventory management. The objective is to minimize operational costs while ensuring adequate service levels under the assumptions defined in the companion ASSUMPTIONS.md document.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Notation and Index Sets](#2-notation-and-index-sets)
3. [Stage 1: Demand Forecasting Models](#3-stage-1-demand-forecasting-models)
4. [Stage 2: Seasonal Category Decomposition](#4-stage-2-seasonal-category-decomposition)
5. [Stage 3: Staff Scheduling Optimization (MILP)](#5-stage-3-staff-scheduling-optimization-milp)
6. [Stage 4: Inventory Optimization (EOQ + Safety Stock)](#6-stage-4-inventory-optimization-eoq--safety-stock)
7. [Integrated Solution Procedure](#7-integrated-solution-procedure)
8. [Performance Metrics](#8-performance-metrics)
9. [References](#9-references)

---

## 1. Problem Statement

### 1.1 Context

A hospital Emergency Department (ED) must plan resources (staff and supplies) to serve stochastic patient demand. The planning problem involves:

1. **Forecasting** daily patient arrivals over a planning horizon H (typically 7 days)
2. **Disaggregating** total demand into clinical categories (RESPIRATORY, CARDIAC, TRAUMA, etc.)
3. **Optimizing** staff schedules to minimize labor costs while meeting service requirements
4. **Optimizing** inventory orders to minimize total inventory costs while maintaining service levels

### 1.2 Objective

**Minimize** total operational cost comprising:
- Labor costs (regular + overtime)
- Understaffing penalties (patient care quality)
- Overstaffing costs (idle resources)
- Inventory ordering costs
- Inventory holding costs
- Stockout penalties

**Subject to:**
- Service level requirements
- Budget constraints
- Capacity constraints
- Staffing ratio requirements

### 1.3 Two-Stage Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: FORECASTING                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ ARIMA   │  │ SARIMAX │  │ XGBoost │  │  LSTM   │  │ Hybrid  │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │
│       └───────────┬┴───────────┴┬───────────┴────────────┘        │
│                   ▼             ▼                                   │
│           Model Selection (Best Accuracy)                          │
│                         │                                           │
│                         ▼                                           │
│              D̂ = [D̂₁, D̂₂, ..., D̂ₕ]  (Total Daily Forecasts)       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SEASONAL DECOMPOSITION                                 │
│                                                                     │
│     d̂[c,t] = π[c,t] × D̂[t]  (Category-level demand)               │
│                                                                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  STAGE 2A: STAFF    │       │  STAGE 2B: INVENTORY │
│  OPTIMIZATION       │       │  OPTIMIZATION        │
│  (MILP)             │       │  (EOQ + MILP)        │
└─────────────────────┘       └─────────────────────┘
```

---

## 2. Notation and Index Sets

### 2.1 Index Sets

| Symbol | Definition | Elements |
|--------|------------|----------|
| T | Planning horizon (days) | t ∈ {1, 2, ..., H}, typically H = 7 |
| C | Clinical categories | c ∈ {RESPIRATORY, CARDIAC, TRAUMA, GASTROINTESTINAL, INFECTIOUS, NEUROLOGICAL, OTHER} |
| K | Staff types | k ∈ {Doctors, Nurses, Support} |
| I | Inventory items | i ∈ {Gloves, PPE, Medications, Syringes, Bandages, IV_Fluids, ...} |
| N | Historical observations | n ∈ {1, 2, ..., N} |

### 2.2 Parameters - Forecasting

| Symbol | Definition | Units |
|--------|------------|-------|
| yₙ | Historical ED arrivals at time n | patients/day |
| Xₙ | Feature vector at time n | varies |
| p, d, q | ARIMA orders (AR, differencing, MA) | integers |
| L | LSTM lookback window | days (default: 14) |
| H | Forecast horizon | days (default: 7) |

### 2.3 Parameters - Staff Optimization

| Symbol | Definition | Units/Value |
|--------|------------|-------------|
| wₖ | Hourly wage for staff type k | $/hour |
| μ | Overtime multiplier | 1.5 (default) |
| τ | Shift length | 8 hours |
| r[c,k] | Patients per staff (ratio) for category c, staff k | patients/staff |
| x̲ₖ, x̄ₖ | Min/max staff bounds for type k | staff count |
| ρ | Nurse-to-doctor ratio requirement | 3.0 (default) |
| πᵘ | Understaffing penalty base | $/patient |
| πᵒ | Overstaffing penalty | $/staff-hour |
| ωc | Category priority weight | multiplier |

### 2.4 Parameters - Inventory Optimization

| Symbol | Definition | Units |
|--------|------------|-------|
| cᵢ | Unit cost of item i | $/unit |
| Kᵢ | Ordering cost for item i | $/order |
| hᵢ | Holding cost rate for item i | fraction/year |
| Lᵢ | Lead time for item i | days |
| αᵢ | Usage rate per patient for item i | units/patient |
| σᵢ | Standard deviation of daily demand for item i | units/day |
| zα | Z-score for service level α | e.g., 1.645 for 95% |

### 2.5 Decision Variables

**Staff Optimization:**

| Variable | Domain | Definition |
|----------|--------|------------|
| x[k,t] | ℤ⁺ | Number of regular staff type k on day t |
| o[k,t] | ℝ⁺ | Overtime hours for staff type k on day t |
| u[c,t] | ℝ⁺ | Unmet demand (understaffing) for category c on day t |
| v[t] | ℝ⁺ | Overstaffing slack on day t |

**Inventory Optimization:**

| Variable | Domain | Definition |
|----------|--------|------------|
| Q[i,t] | ℝ⁺ | Order quantity for item i at period t |
| y[i,t] | {0,1} | Binary: 1 if order placed for item i at period t |
| Inv[i,t] | ℝ⁺ | Inventory level for item i at end of period t |
| B[i,t] | ℝ⁺ | Backorder quantity for item i at period t |

---

## 3. Stage 1: Demand Forecasting Models

### 3.1 Feature Engineering

**3.1.1 Lag Features (Autoregressive Inputs)**

For the ED arrival time series {yₜ}, create lagged features:

```
ED_j(t) = y_{t-j}    for j ∈ {1, 2, ..., 7}
```

**3.1.2 Target Variables (Multi-Horizon)**

```
Target_h(t) = y_{t+h}    for h ∈ {1, 2, ..., H}
```

Each horizon h is treated as a separate prediction task.

**3.1.3 Calendar Features**

```
X^cal_t = [dow_t, month_t, is_weekend_t, is_holiday_t,
           sin(2π · doy_t / 365), cos(2π · doy_t / 365)]ᵀ
```

**3.1.4 Fourier Features (Seasonality)**

For period P and harmonic order k:

```
fourier^sin_{P,k}(t) = sin(2πk · t / P)
fourier^cos_{P,k}(t) = cos(2πk · t / P)
```

Default periods: P ∈ {7, 14, 30.44, 91.31, 365.25}

---

### 3.2 ARIMA Model

**Model Specification: ARIMA(p, d, q)**

The general ARIMA model for differenced series:

```
Φ(B)(1-B)^d y_t = c + Θ(B)ε_t
```

Where:
- B = backshift operator: Bᵏyₜ = y_{t-k}
- Φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ (AR polynomial)
- Θ(B) = 1 + θ₁B + θ₂B² + ... + θ_qBᵍ (MA polynomial)
- d = differencing order
- εₜ ~ N(0, σ²) (white noise)

**Expanded Form:**

```
y_t = c + Σ(i=1 to p) φᵢy_{t-i} + Σ(j=1 to q) θⱼε_{t-j} + εₜ
```

**Order Selection (Grid Search with CV):**

```
(p̂, d̂, q̂) = argmin_{p,d,q} CV-RMSE(p, d, q)
```

Where CV-RMSE is computed using expanding window cross-validation.

**Stationarity Testing:**
- ADF Test: H₀: unit root exists (non-stationary)
- KPSS Test: H₀: series is stationary
- Critical value: α = 0.05

---

### 3.3 SARIMAX Model

**Model Specification: SARIMAX(p, d, q)(P, D, Q)ₛ**

Extends ARIMA with seasonal components and exogenous variables:

```
Φ(B)Φₛ(Bˢ)(1-B)^d(1-Bˢ)^D y_t = c + Θ(B)Θₛ(Bˢ)ε_t + β'X_t
```

Where:
- s = seasonal period (s = 7 for weekly)
- Φₛ(Bˢ) = 1 - Φ₁Bˢ - ... - Φ_P B^{Ps} (seasonal AR)
- Θₛ(Bˢ) = 1 + Θ₁Bˢ + ... + Θ_Q B^{Qs} (seasonal MA)
- Xₜ = exogenous features (weather, holidays)
- β = regression coefficients

**Default Configuration:**
- Non-seasonal: (1, 1, 1)
- Seasonal: (1, 0, 1)₇

---

### 3.4 XGBoost Model

**Gradient Boosted Trees for Regression**

```
ŷ_t = Σ(m=1 to M) f_m(X_t),    f_m ∈ ℱ
```

Where ℱ is the space of regression trees.

**Objective Function:**

```
ℒ = Σ(i=1 to n) l(yᵢ, ŷᵢ) + Σ(m=1 to M) Ω(f_m)
```

Where:
- l(yᵢ, ŷᵢ) = (yᵢ - ŷᵢ)² (MSE loss)
- Ω(f) = γT + (1/2)λ‖w‖² (regularization)
- T = number of leaves
- w = leaf weights

**Additive Training (Boosting):**

At iteration m, add tree f_m that minimizes:

```
ℒ^(m) = Σ(i=1 to n) l(yᵢ, ŷᵢ^{(m-1)} + f_m(Xᵢ)) + Ω(f_m)
```

**Key Hyperparameters:**
- n_estimators (M): 300
- max_depth: 6
- learning_rate (η): 0.1
- subsample: 0.8
- colsample_bytree: 0.8

---

### 3.5 LSTM Neural Network

**Long Short-Term Memory Architecture**

For sequence input X = [x_{t-L+1}, ..., x_{t-1}, x_t] with lookback L:

**Cell State Update:**

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     (forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     (input gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  (candidate)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t        (cell state)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     (output gate)
h_t = o_t ⊙ tanh(C_t)                   (hidden state)
```

Where:
- σ(·) = sigmoid activation
- ⊙ = element-wise multiplication

**Network Architecture:**

```
Input(L, n_features)
  → LSTM(64 units, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32 units)
  → Dropout(0.2)
  → Dense(16, ReLU)
  → Dense(1, Linear)
```

**Loss Function:**

```
ℒ = (1/n) Σ(i=1 to n) (yᵢ - ŷᵢ)²    (MSE)
```

**Training:**
- Optimizer: Adam (lr = 0.001)
- Epochs: 100 (with early stopping, patience = 3)
- Batch size: 32

**Uncertainty Quantification (Monte Carlo Dropout):**

```
ŷ_MC = (1/K) Σ(k=1 to K) f(X; dropout_k)
```

Where K inference passes are made with dropout enabled.

---

### 3.6 Hybrid Models (Two-Stage)

**Architecture: Stage 1 (Pattern) → Stage 2 (Residual Correction)**

```
ŷ^(1)_t = f_{LSTM}(X_t)           (Stage 1: LSTM)
ε_t = y_t - ŷ^(1)_t               (Residuals)
ε̂_t = g(X_t)                      (Stage 2: XGBoost, SARIMAX, or ANN)
ŷ_t = ŷ^(1)_t + ε̂_t              (Final prediction)
```

**Hybrid Variants:**
1. **LSTM-XGBoost:** g(·) = XGBoost on residuals
2. **LSTM-SARIMAX:** g(·) = SARIMAX(1,1,1)×(1,0,1)₇ on residuals
3. **LSTM-ANN:** g(·) = Dense neural network on residuals

**Training Protocol:**
- Stage 1: Fit LSTM on expanding window folds
- Compute out-of-fold residuals
- Stage 2: Fit corrector model on residuals with same features

---

### 3.7 Model Selection

**Best Model Selection Criterion:**

```
m̂ = argmin_{m ∈ ℳ} MAPE_m
```

Where ℳ = {ARIMA, SARIMAX, XGBoost, LSTM, ANN, Hybrids}

**Alternative criteria:** RMSE, MAE, or weighted combination

---

## 4. Stage 2: Seasonal Category Decomposition

### 4.1 Seasonal Proportion Calculation

**Day-of-Week Proportions:**

For category c and day-of-week d ∈ {0, 1, ..., 6}:

```
π^DOW_{c,d} = Σ_{t: dow(t)=d} y_{c,t} / Σ_t y_{c,t}
```

**Monthly Proportions:**

For category c and month m ∈ {1, 2, ..., 12}:

```
π^MONTH_{c,m} = Σ_{t: month(t)=m} y_{c,t} / Σ_t y_{c,t}
```

### 4.2 Multiplicative Combination

For a target date with day-of-week d and month m:

```
π^raw_c = π^DOW_{c,d} × π^MONTH_{c,m}
```

**Normalization:**

```
π_c = π^raw_c / Σ_{c' ∈ C} π^raw_{c'}
```

**Constraint:** Σ_{c ∈ C} πc = 1

### 4.3 Category Demand Distribution

Given total forecast D̂ₜ for day t:

```
d̂_{c,t} = π_{c,t} × D̂_t    ∀c ∈ C
```

**Result:** Category-level demand matrix [d̂_{c,t}]_{|C| × H}

---

## 5. Stage 3: Staff Scheduling Optimization (MILP)

### 5.1 Problem Formulation

**Objective Function:**

```
min Z_staff = Σ_t Σ_k (w_k · τ · x_{k,t})           [Regular Labor]
            + Σ_t Σ_k (μ · w_k · o_{k,t})          [Overtime]
            + Σ_t Σ_c (ω_c · π^u · u_{c,t})        [Understaffing Penalty]
            + Σ_t (π^o · v_t)                       [Overstaffing Penalty]
```

### 5.2 Constraints

**C1: Demand Satisfaction (per category, per day)**

```
Σ_k [(τ / r_{c,k}) · x_{k,t} + (1 / r_{c,k}) · o_{k,t}] + u_{c,t} ≥ d̂_{c,t}
                                                    ∀c ∈ C, t ∈ T
```

Where r_{c,k} = patients per staff of type k in category c.

**C2: Overstaffing Calculation**

```
v_t ≥ Σ_k [(τ / r̄_k) · x_{k,t}] - Σ_c d̂_{c,t}    ∀t ∈ T
```

Where r̄_k = (1/|C|) Σ_c r_{c,k} (average ratio)

**C3: Overtime Limits**

```
o_{k,t} ≤ o^max · x_{k,t}    ∀k ∈ K, t ∈ T
```

Default: o^max = 4 hours/day

**C4: Staff Bounds**

```
x̲_k ≤ x_{k,t} ≤ x̄_k    ∀k ∈ K, t ∈ T
```

**C5: Skill Mix Ratio**

```
x_{Nurses,t} ≥ ρ · x_{Doctors,t}    ∀t ∈ T
```

Default: ρ = 3.0 (3 nurses per doctor)

**C6: Budget Constraint (Optional)**

```
Σ_k (w_k · τ · x_{k,t}) ≤ B^daily    ∀t ∈ T
```

### 5.3 Staffing Ratios by Category

| Category c | r[c, Doctors] | r[c, Nurses] | r[c, Support] | ω_c |
|------------|---------------|--------------|---------------|-----|
| TRAUMA | 3 | 2 | 4 | 3.0 |
| CARDIAC | 4 | 2 | 5 | 2.5 |
| NEUROLOGICAL | 5 | 2 | 6 | 2.0 |
| RESPIRATORY | 6 | 3 | 8 | 1.5 |
| INFECTIOUS | 8 | 4 | 10 | 1.2 |
| GASTROINTESTINAL | 10 | 5 | 12 | 1.0 |
| OTHER | 12 | 6 | 15 | 0.8 |

### 5.4 Cost Parameters

| Parameter | Symbol | Default Value |
|-----------|--------|---------------|
| Doctor hourly rate | w_D | $150/hr |
| Nurse hourly rate | w_N | $45/hr |
| Support hourly rate | w_S | $25/hr |
| Overtime multiplier | μ | 1.5 |
| Shift length | τ | 8 hours |
| Understaffing penalty | π^u | $200/patient |
| Overstaffing penalty | π^o | $15/staff-hour |

---

## 6. Stage 4: Inventory Optimization (EOQ + Safety Stock)

### 6.1 Demand Conversion

Convert patient forecast to item demand:

```
d̂_{i,t} = α_i · D̂_t    ∀i ∈ I, t ∈ T
```

Where αᵢ = usage rate per patient for item i.

### 6.2 Economic Order Quantity (EOQ)

**Classic EOQ Formula:**

```
Q*_i = √(2 K_i D_i / (h_i · c_i))
```

Where:
- Dᵢ = annual demand for item i (units/year)
- Kᵢ = ordering cost per order ($)
- hᵢ = holding cost rate (fraction/year)
- cᵢ = unit cost ($)

**Annual Demand Estimation:**

```
D_i = 365 · d̄_i = 365 · (1/H) Σ_{t=1}^H d̂_{i,t}
```

### 6.3 Statistical Safety Stock

**Safety Stock Formula (Assumption I-S1):**

```
SS_i = z_α · σ_i · √L_i
```

Where:
- zα = standard normal quantile for service level α
- σᵢ = standard deviation of daily demand
- Lᵢ = lead time in days

**Service Level Z-Scores:**

| Service Level α | Z-Score z_α |
|-----------------|-------------|
| 90% | 1.282 |
| 95% | 1.645 |
| 99% | 2.326 |
| 99.5% | 2.576 |

### 6.4 Reorder Point

```
ROP_i = d̄_i · L_i + SS_i
```

**Interpretation:** Reorder when inventory falls to ROP_i.

### 6.5 MILP Multi-Item Inventory Optimization

**Objective Function:**

```
min Z_inv = Σ_t Σ_i [K_i · y_{i,t} + c_i · Q_{i,t} + h^daily_i · Inv_{i,t} + p_i · B_{i,t}]
```

Where h^daily_i = (h_i · c_i) / 365

### 6.6 Inventory Constraints

**C1: Inventory Balance**

```
Inv_{i,t} = Inv_{i,t-1} + Q_{i,t-L_i} - d̂_{i,t} + B_{i,t} - B_{i,t-1}    ∀i, t
```

**C2: Order-Quantity Linking**

```
Q_{i,t} ≤ M · y_{i,t}    ∀i, t
```

Where M is a large constant (big-M).

**C3: Minimum Order Quantity**

```
Q_{i,t} ≥ Q^min_i · y_{i,t}    ∀i, t
```

**C4: Storage Capacity**

```
Σ_i (v_i · Inv_{i,t}) ≤ W    ∀t
```

Where vᵢ = volume per unit, W = total warehouse capacity.

### 6.7 Inventory Item Parameters

| Item i | αᵢ (units/patient) | cᵢ ($) | Kᵢ ($) | hᵢ | Lᵢ (days) |
|--------|---------------------|---------|---------|-----|------------|
| Gloves | 5.0 | 8.50 | 25 | 0.15 | 2 |
| PPE | 0.3 | 45.00 | 75 | 0.20 | 5 |
| Medications | 0.15 | 125.00 | 100 | 0.25 | 3 |
| Syringes | 2.0 | 12.00 | 30 | 0.15 | 2 |
| Bandages | 1.5 | 18.00 | 25 | 0.15 | 2 |
| IV Fluids | 0.8 | 3.50 | 40 | 0.20 | 3 |

---

## 7. Integrated Solution Procedure

### 7.1 Algorithm

```
ALGORITHM: Integrated Hospital Resource Planning

INPUT:
  - Historical data: {y_n, X_n} for n = 1, ..., N
  - Planning horizon: H
  - Cost parameters: {w_k, c_i, K_i, h_i, ...}
  - Constraints: {x̲_k, x̄_k, ρ, W, ...}

OUTPUT:
  - Demand forecasts: D̂ = [D̂_1, ..., D̂_H]
  - Staff schedule: x* = [x*_{k,t}]
  - Inventory orders: Q* = [Q*_{i,t}]
  - Total cost: Z*

PROCEDURE:

1. FEATURE ENGINEERING
   For each observation n:
     Create ED_j(n) = y_{n-j} for j = 1,...,7
     Create calendar features X^cal_n
     Create Fourier features X^fourier_n

2. TEMPORAL SPLIT
   Split data: Train (70%) | Calibration (15%) | Test (15%)
   Ensure: max(train_dates) < min(test_dates)

3. MODEL TRAINING (for each model m ∈ M)
   For each horizon h = 1, ..., H:
     Fit model m on Train data with Target_h
     Evaluate on Test data
     Store metrics: MAE_m,h, RMSE_m,h, MAPE_m,h

4. MODEL SELECTION
   m* = argmin_m (average MAPE across horizons)
   D̂ = forecasts from model m*

5. SEASONAL DECOMPOSITION
   Calculate π^DOW_{c,d} and π^MONTH_{c,m} from historical data
   For each forecast day t:
     Compute π_{c,t} (normalized multiplicative proportion)
     d̂_{c,t} = π_{c,t} × D̂_t

6. STAFF OPTIMIZATION (MILP)
   Solve:
     min Z_staff subject to C1-C6
   Extract: x*_{k,t}, o*_{k,t}, u*_{c,t}, v*_t

7. INVENTORY OPTIMIZATION
   For each item i:
     Compute d̂_{i,t} = α_i × D̂_t
     Compute Q*_i (EOQ)
     Compute SS_i (Safety Stock)
     Compute ROP_i (Reorder Point)

   If MILP enabled:
     Solve multi-item optimization
     Extract: Q*_{i,t}, Inv*_{i,t}

8. COST COMPUTATION
   Z*_staff = regular labor + overtime + penalties
   Z*_inv = ordering + holding + purchase + stockout
   Z* = Z*_staff + Z*_inv

RETURN D̂, x*, Q*, Z*
```

### 7.2 Data Flow Diagram

```
Historical Data
      │
      ▼
┌─────────────────┐
│ Feature         │
│ Engineering     │──── ED_1...ED_7, Calendar, Fourier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal Split  │──── Train/Cal/Test (70/15/15)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │──── ARIMA, XGBoost, LSTM, Hybrids
│ & Selection     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Best Forecast   │──── D̂ = [D̂_1, ..., D̂_H]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Seasonal        │──── d̂[c,t] = π[c,t] × D̂[t]
│ Decomposition   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│ MILP  │  │ EOQ + │
│ Staff │  │ Inv.  │
└───┬───┘  └───┬───┘
    │         │
    ▼         ▼
  x*_{k,t}   Q*_i, SS_i, ROP_i
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Total Cost Z*   │
│ Recommendations │
└─────────────────┘
```

---

## 8. Performance Metrics

### 8.1 Forecasting Metrics

**Mean Absolute Error (MAE):**

```
MAE = (1/n) Σ_{i=1}^n |y_i - ŷ_i|
```

**Root Mean Squared Error (RMSE):**

```
RMSE = √[(1/n) Σ_{i=1}^n (y_i - ŷ_i)²]
```

**Mean Absolute Percentage Error (MAPE):**

```
MAPE = (100%/n) Σ_{i=1}^n |y_i - ŷ_i| / |y_i|
```

**Forecast Accuracy:**

```
Accuracy = 100% - MAPE
```

**Coefficient of Determination (R²):**

```
R² = 1 - [Σ_i(y_i - ŷ_i)² / Σ_i(y_i - ȳ)²]
```

### 8.2 Prediction Interval Metrics

**Empirical Coverage:**

```
Coverage = (1/n) Σ_{i=1}^n 𝟙[y_i ∈ [L_i, U_i]]
```

**Relative Prediction Interval Width (RPIW):**

```
RPIW = [(1/n) Σ_i (U_i - L_i)] / [max(y) - min(y)] × 100%
```

### 8.3 Optimization Metrics

**Service Level (Staff):**

```
SL_staff = [Σ_{c,t}(d̂_{c,t} - u_{c,t})] / [Σ_{c,t} d̂_{c,t}] × 100%
```

**Service Level (Inventory):**

```
SL_inv = P(D_L ≤ Inv) ≈ Φ[(Inv - μ_L) / σ_L]
```

**Cost Efficiency:**

```
η = (Z^current - Z^optimized) / Z^current × 100%
```

---

## 9. References

### Statistical Forecasting
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

### Machine Learning
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

### Operations Research
- Silver, E.A., Pyke, D.F., & Thomas, D.J. (2016). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
- Ernst, A.T., et al. (2004). Staff scheduling and rostering: A review. *EJOR*, 153(1), 3-27.
- Axsäter, S. (2015). *Inventory Control* (3rd ed.). Springer.

### Conformal Prediction
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.

---

## Appendix A: Assumption Cross-References

This mathematical model operates under the assumptions defined in `ASSUMPTIONS.md`:

| Model Component | Key Assumptions |
|-----------------|-----------------|
| Demand Forecasting | S-D1 (deterministic), S-D2 (daily aggregation) |
| Seasonal Decomposition | S-D3 (categorical decomposition) |
| Staff Optimization | S-W1 to S-W5, S-C1 to S-C4, S-X1 to S-X4 |
| Inventory Optimization | I-D1 to I-D4, I-L1 to I-L2, I-C1 to I-C4, I-P1 to I-P5, I-S1 to I-S2, I-E1 to I-E2 |
| General | G-1 to G-5 |

---

*Document prepared in accordance with academic standards for operations research model documentation.*
