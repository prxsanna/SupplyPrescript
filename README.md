# Supply Prescript — Closed-Loop Prescriptive Analytics

**Author & Module Lead:** Prasanna (*Machine Learning + Prescriptive Optimization + Demo UI*)  
**Team Allocation:**
- **Prasanna:** Machine Learning (XGBoost), Prescriptive Optimization (PuLP / SciPy), Closed-Loop Evaluation Logic, Demo Frontend.
- **Trisha:** Backend (FastAPI), Database (PostgreSQL / SQL), Write-Back Architecture.
- **Nanditha:** Operational Dashboard (React / Retool), Integration & Visualization.

---

## 1. Project Overview & Workflow

Supply Prescript implements a **Closed-Loop Prescriptive Analytics** architecture for supply chain disruption management:

```
                  ┌───────────────────────────────┐
                  │ 1. Shipment Information Input │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 2. ML Delay Prediction        │
                  │    (XGBoost Classifier & Reg) │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 3. Prescriptive Optimization  │
                  │    (PuLP MILP Solver)         │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 4. 3 Alternative Actions      │
                  │    (Air / Supplier / Buffer)  │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 5. Decision Execution         │
                  │    (Write-Back to DB)         │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 6. Closed-Loop Evaluation     │
                  │    (Variance & Mitigation ROI)│
                  └───────────────────────────────┘
```

---

## 2. Directory Structure & Files Created

```
supplyprescript/
├── data/
│   ├── supply_chain_data.csv          # Realistic supply chain benchmark dataset (3,500 records)
│   └── closed_loop_history.json       # Persistent closed-loop decision & evaluation logs
├── ml/
│   ├── __init__.py
│   ├── data_preparation.py            # Data generator, domain feature engineer & preprocessor
│   ├── train_model.py                 # Dual XGBoost model training (Classifier + Regressor)
│   ├── predictor.py                   # High-level real-time inference & explainability engine
│   └── models/                        # Serialized artifacts (.joblib, metrics.json)
├── optimization/
│   ├── __init__.py
│   ├── constraints.py                 # Optimization constraints & business rules dataclass
│   ├── validator.py                   # Hard constraint audit & feasibility validator
│   └── optimizer.py                   # PuLP Mixed-Integer Linear Program (MILP) solver
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py                   # Closed-loop variance and mitigation ROI calculator
│   └── feedback.py                    # Decision history tracking & model feedback loop
├── api/
│   ├── __init__.py
│   ├── interfaces.py                  # Standard Python functions contract for Trisha & Nanditha
│   └── main.py                        # FastAPI endpoints for write-back and API serving
├── frontend/
│   ├── __init__.py
│   └── app.py                         # Streamlit interactive demonstration dashboard
├── tests/
│   ├── __init__.py
│   ├── test_ml.py                     # ML pipeline and inference tests
│   ├── test_optimization.py          # Prescriptive optimizer and constraint tests
│   ├── test_closed_loop.py            # Closed-loop evaluation tests
│   └── test_api_contract.py           # API integration and contract tests
├── run_demo.py                        # Single-command demo launcher
└── README.md                          # Full system documentation
```

---

## 3. How the ML Model Works

The predictive layer uses a **Dual-Head XGBoost Architecture**:

1. **XGBoost Classifier (`XGBClassifier`)**:
   - Target: `is_delayed` (0 or 1)
   - Predicts: Disruption Probability $P(\text{Delayed}) \in [0, 1]$ and maps to categorical Risk Levels:
     - `LOW`: $<25\%$
     - `MEDIUM`: $25\%\text{--}50\%$
     - `HIGH`: $50\%\text{--}75\%$
     - `CRITICAL`: $\ge 75\%$
   - Provides confidence metric and explainable risk drivers.

2. **XGBoost Regressor (`XGBRegressor`)**:
   - Target: `delay_days` (Continuous positive duration in days)
   - Predicts: Estimated extra days of delay $\hat{D}$.

### Features Used by the Model:
- **Numerical Features:**
  - `lead_time_days`: Baseline promised supplier lead time
  - `shipping_cost`: Base shipping freight cost ($)
  - `inventory_level`: Current warehouse safety stock on hand (units)
  - `order_quantity`: Shipment batch size (units)
  - `supplier_delay_rate`: Historical supplier late delivery frequency (0.0 to 1.0)
  - `route_distance_km`: Origin-to-destination transit distance (km)
  - `weather_risk_index`: Environmental weather severity score (0.0 to 1.0)
  - `congestion_index`: Port/corridor congestion score (0.0 to 1.0)
- **Engineered Domain Features:**
  - `lead_time_to_distance_ratio`: Transit speed efficiency index
  - `inventory_coverage_ratio`: Safety stock cushion vs order volume
  - `risk_multiplier`: Compound weather-congestion-supplier disruption interaction
- **Categorical Features (One-Hot Encoded):**
  - `shipping_mode` (Standard Ground, Air Freight, Ocean Intermodal, Express Carrier)
  - `product_category` (Semiconductors, Automotive, Industrial, Pharmaceuticals, Consumer)
  - `supplier_tier` (Tier-1, Tier-2, Tier-3)

### Model Evaluation Results:
- **Classification Accuracy:** **96.14%**
- **Classification Precision:** **97.72%**
- **Classification Recall:** **90.68%**
- **Classification F1 Score:** **94.07%**
- **Classification ROC-AUC:** **0.9960**
- **Regression MAE:** **0.67 Days**
- **Regression RMSE:** **1.06 Days**
- **Regression $R^2$ Score:** **0.8169**

---

## 4. Prescriptive Optimization Formulation

When the ML model predicts a delay, the **Prescriptive Optimization Solver** (`PuLP` MILP) computes the optimal action among 3 alternatives:

### Three Strategic Alternatives:
1. **Option A: Expedited Air Freight / Express Carrier**
   - Transmit Speed: $1\text{--}4$ days (Fastest delay recovery)
   - Direct Cost: Base cost $\times 2.15 + \text{handling}$
   - Residual Delay: $0$ days (Risk: LOW)
2. **Option B: Secondary Supplier Reallocation / Regional Split**
   - Transmit Speed: $4\text{--}7$ days (Moderate recovery)
   - Direct Cost: Base cost $\times 1.32 + \text{handling}$
   - Residual Delay: Reduced by $\approx 75\%$ (Risk: MEDIUM)
3. **Option C: Inventory Buffer Reallocation & Ground Expedite**
   - Transmit Speed: $7\text{--}12$ days (Cushioned delay)
   - Direct Cost: Base cost $\times 1.08 + \text{handling}$
   - Residual Delay: Reduced by $\approx 40\%$ (Risk: MODERATE)

### Mathematical MILP Formulation:
$$\min \sum_{i \in \{A, B, C\}} y_i \cdot \Big[ C_i + \lambda_{\text{delay}} \cdot (D_i + \max(0, T_i - T_{\text{deadline}})) + \lambda_{\text{risk}} \cdot R_i \Big]$$

**Subject to Hard Business Constraints:**
1. Selection Constraint: $\sum_{i} y_i = 1, \quad y_i \in \{0, 1\}$
2. Hard Budget Constraint: $y_i \cdot C_i \le B_{\max}$
3. Hard Delivery SLA Constraint: $y_i \cdot T_i \le T_{\text{deadline}}$
4. Hard Capacity Constraint: $y_i \cdot Q_{\text{order}} \le \text{Cap}_i$

If an option violates a hard constraint, the audit validator flags it as **INFEASIBLE** and explains the exact violation.

---

## 5. Integration Guides for Team Members

### Integration for Trisha (Backend / Database Lead)
Trisha can directly import the core functions from `api.interfaces` or run the FastAPI app in `api/main.py`:

```python
from api.interfaces import (
    predict_delay,
    optimize_actions,
    validate_recommendation,
    evaluate_decision,
    log_and_evaluate_cycle,
)

# 1. Delay Prediction Endpoint
prediction = predict_delay(shipment_dict)

# 2. Optimization Endpoint
optimization_result = optimize_actions(
    shipment_data=shipment_dict,
    prediction=prediction,
    constraints={"max_budget": 20000.0, "max_delivery_days": 10.0}
)

# 3. Decision Evaluation & Write-back
cycle_result = log_and_evaluate_cycle(
    shipment_data=shipment_dict,
    prediction=prediction,
    decision=chosen_option_dict,
    actual_outcome={"actual_cost": 16000.0, "actual_delivery_days": 3.5, "notes": "Completed"}
)
```

### Integration for Nanditha (Operational UI Lead)
Nanditha can connect her React or Retool dashboard to the FastAPI endpoints:
- `POST /predict`: Sends shipment inputs, returns probability gauge data and predicted delay days.
- `POST /optimize`: Sends prediction + constraints, returns the 3 recommendation cards with `is_recommended`, `is_feasible`, `cost`, `delivery_days`, and `tradeoff_comparison` chart data.
- `POST /execute-decision`: Sends chosen option for operational database write-back.
- `POST /evaluate-decision`: Sends actual outcome values, returns cost/delivery variance and mitigation ROI %.

---

## 6. How to Run the Project

### Running the Interactive Demo:
```powershell
python run_demo.py
```
*Or directly via Streamlit:*
```powershell
streamlit run frontend/app.py
```

### Running the FastAPI Backend:
```powershell
python -m uvicorn api.main:app --reload --port 8000
```
*Interactive Swagger API documentation available at `http://127.0.0.1:8000/docs`.*

### Running the Test Suite:
```powershell
pytest tests/ -v
```

---

## 7. Exact 11-Step Demonstration Flow

1. **Open Application**: Launch `python run_demo.py` and navigate to `http://localhost:8501`.
2. **Select Scenario**: Choose **"🚨 Scenario 1: Critical Semiconductor Delay"** from the sidebar dropdown.
3. **Inspect Inputs**: Review the shipment attributes (Supplier, Ocean Mode, Lead Time: 24d, Weather Risk: 0.75, Congestion: 0.70, Max Budget: \$24,000, SLA: 10d).
4. **Click "PREDICT DELAY"**: The live XGBoost model computes inference.
5. **View Prediction**: Delay Probability = **100.0%** (`CRITICAL` Risk), Predicted Delay = **10.0 Days**, Expected Total = **34.0 Days**, Model Confidence = **99.9%**, with key risk drivers highlighted.
6. **Automatic Optimization**: The PuLP MILP optimizer solves for 3 alternative options.
7. **Inspect 3 Recommendation Cards**:
   - **Option A (Air Freight)**: Cost \$21,675 | Delivery 3.6d | **FEASIBLE & RECOMMENDED ✓** (Satisfies SLA $\le 10$d & Budget $\le \$24,000$).
   - **Option B (Secondary Supplier)**: Cost \$12,500 | Delivery 10.9d | **INFEASIBLE ✗** (Violates SLA $\le 10$d).
   - **Option C (Inventory Buffer)**: Cost \$9,620 | Delivery 21.0d | **INFEASIBLE ✗** (Violates SLA $\le 10$d).
8. **View Audit & Trade-off Chart**: Read the *"Why These Recommendations?"* explanation and inspect the Cost vs. Speed scatter plot.
9. **Click "EXECUTE OPTION A"**: Triggers decision execution and simulated operational write-back.
10. **Enter Actual Outcome**: Input actual recorded cost (e.g. \$22,500) and delivery (e.g. 4.0 days).
11. **Click "EVALUATE DECISION"**: View Cost Variance (+\$825), Delivery Variance (+0.4d), SLA Penalties Prevented (\$8,075), and Mitigation ROI (+35.8%), completing the closed-loop demonstration.
