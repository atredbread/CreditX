# Credit Risk Feature Engineering Documentation

## Input Data Files

### 1. credit_Agents.xlsx
Contains agent information including region and contact details.

**Sample Data:**
```
Bzid    | AgentName  | Region      | Phone       | Email
--------|------------|-------------|-------------|------------------
AG001   | John Doe   | North       | 1234567890  | john@example.com
AG002   | Jane Smith | South       | 2345678901  | jane@example.com
```

### 2. DPD.xlsx
Days Past Due data for each agent.

**Sample Data:**
```
Bzid  | Dpd  | Amount  | DueDate    | Status
------|------|---------|------------|--------
AG001 | 15   | 5000    | 2025-05-01 | Open
AG002 | 95   | 15000   | 2025-02-01 | Overdue
```

### 3. repayment report.csv
Payment history and status for each agent.

**Sample Data:**
```
agent_id | payment_date | amount | status     | payment_method
---------|--------------|--------|------------|--------------
AG001   | 2025-06-01   | 2500   | On Time    | Bank Transfer
AG002   | 2025-06-15   | 5000   | Late       | Credit Card
```

### 4. Credit_sales_data.xlsx
Sales data including both cash and credit sales.

**Sample Data:**
```
Bzid  | SaleDate    | TotalAmount | CreditAmount | PaymentStatus
------|-------------|-------------|--------------|-------------
AG001 | 2025-06-01  | 10000       | 7000         | Paid
AG002 | 2025-06-10  | 20000       | 15000        | Pending
```

## Calculated Metrics

### 1. Credit Utilization Ratio
**Formula:** `CreditAmount / TotalCreditLimit`  
**Range:** 0-1 (Lower is better)

### 2. Repayment Score
**Scoring Logic:**
- Base Score: 100 points
- Penalties:
  - 30+ DPD: -5 points
  - 60+ DPD: -15 points
  - 90+ DPD: -30 points
  - 120+ DPD: -50 points
- Bonus: +10 points for recent on-time payments

**Range:** 0-100 (Higher is better)

### 3. Delinquency Flags
- `delinquent_30p`: True if DPD ≥ 30
- `delinquent_60p`: True if DPD ≥ 60
- `delinquent_90p`: True if DPD ≥ 90

### 4. Recovery Index
**Formula:** `Days between default and first payment`  
**Range:** Number of days (Lower is better)

### 5. Region Risk
**Factors Considered:**
1. Default rate (50% weight)
2. Average DPD in region (30% weight)
3. Recovery metrics (20% weight)

**Range:** 0-1 (Lower is better)

## Sample Output

### Engineered Features
```
Bzid  | credit_util | repay_score | delinquent_30p | recovery_days | region_risk
------|-------------|-------------|----------------|---------------|------------
AG001 | 0.35        | 85          | False          | 45            | 0.25
AG002 | 0.75        | 45          | True           | 120           | 0.78
```

### Risk Classification
```
Bzid  | Risk_Level  | Recommended_Action
------|-------------|-------------------
AG001 | Low Risk    | Increase credit limit
AG002 | High Risk   | Review account
```

## Notes
- All monetary values are in local currency (INR)
- Dates are in YYYY-MM-DD format
- Risk thresholds:
  - Low: 0-0.3
  - Medium: 0.31-0.7
  - High: 0.71-1.0
