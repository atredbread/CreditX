# Credit Risk Feature Engineering Documentation

## Unified Standards Compliance

This document follows the unified standards for the Credit Health Intelligence Engine. All features and processing steps adhere to the thresholds and conventions defined in the latest documentation.

## Data Processing Standards

### 1. Data Sources and Schema

#### 1.1 Agent Details.xlsx (Primary Source)
Comprehensive agent information including contact details, organizational hierarchy, and regional assignments.

**Required Columns:**
```
account          # Primary key, agent identifier (maps to Bzid in other datasets)
agentname        # Full name of agent (string, encrypted)
organizationname # Legal business name (string)
RoName           # Region Officer name (string, encrypted)
RmName           # Relationship Manager name (string, encrypted)
email            # Contact email (string, encrypted)
mobile           # Contact number (string, encrypted)
region           # Region code (string)
AgentRegion      # Full region name (string)
agenttype        # Type of agent (e.g., BP_AGENT, OTHERS_NBP)
status           # Account status (ACTIVE/INACTIVE/SUSPENDED)
```

**Data Quality Rules:**
- `account` must be unique and non-null (matches `Bzid` in other datasets)
- All encrypted fields must be properly formatted before encryption
- Region codes must match standardized values from `Region_contact.xlsx`
- Status must be one of: ACTIVE, INACTIVE, SUSPENDED
- Email must be in valid format
- Phone numbers must follow E.164 format

#### 1.2 credit_Agents.xlsx (Legacy)
**Status**: Deprecated - Use `Agent Details.xlsx` as the primary source

**Description**: Legacy list of onboarded agents. This file is being phased out in favor of `Agent Details.xlsx`.

#### 1.2 DPD.xlsx (Days Past Due)
Tracks payment delays and defaults.

**Required Columns:**
```
Bzid       # Agent identifier (string, matches credit_Agents)
Date       # Transaction date (YYYY-MM-DD)
DPD        # Days past due (integer, 0 for on-time)
Amount     # Amount in default (float)
Status     # Payment status (string: 'current', '30_dpd', '60_dpd', '90+_dpd')
```

## Feature Engineering

### 2. Core Features

#### 2.1 GMV Trend Analysis
- **Calculation**: 6-month rolling slope of GMV
- **Requirements**:
  - Minimum 4 months of non-zero data required
  - Negative slope indicates decreasing trend
- **Usage**: Triggers P1 classification if negative

#### 2.2 Credit Utilization
- **Formula**: (Total Credit Used / Total Credit Limit) * 100
- **Classification**:
  - 20-50%: Optimal (P0)
  - 50-75%: Elevated risk
  - >75%: High risk (triggers P2 if combined with DPD)

#### 2.3 Payment Behavior
- **Features**:
  - 30/60/90+ DPD counts
  - Rolling 6-month late payment frequency
  - Payment consistency score

## Output Standards

### 3. Directory Structure

```
/output/
  ├── processed/          # Cleaned and processed data files
  │   └── {date}_processed.csv
  ├── logs/               # Processing logs
  │   ├── {date}_processing.log
  │   └── error_logs/
  ├── region_reports/     # Regional analysis
  │   └── {region}_{date}.xlsx
  └── email_summaries/    # Automated reports
      └── {date}_summary.md
```

### 4. File Naming Conventions
- **Processed Data**: `{source}_{date}_processed.csv`
- **Reports**: `{region}_{report_type}_{date}.xlsx`
- **Logs**: `{process_name}_{timestamp}.log`

## Data Processing Pipeline

1. **Data Validation**
   - Check for required columns
   - Validate data types and formats
   - Identify and handle missing values

2. **Feature Calculation**
   - Compute GMV trends
   - Calculate credit utilization
   - Derive payment behavior metrics

3. **Output Generation**
   - Save processed data to appropriate directories
   - Generate logs for audit trail
   - Create standardized reports

## Version Control
- All feature engineering changes must be versioned
- Document any changes to feature calculations
- Maintain backward compatibility where possible
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

**Calculation Method:**
The repayment score is calculated using a weighted combination of multiple repayment metrics, as defined in the [Rule Book](../rule%20book.md#repayment-score-calculation-standards).

#### Metrics and Weights:
| Metric | Weight | Description |
|--------|--------|-------------|
| Total Repayment Amount | 25% | Sum of all repayments made by the agent |
| Total Principal Repaid | 20% | Sum of principal amounts repaid |
| Number of Repayment Transactions | 15% | Count of repayment transactions |
| Average Repayment Per Transaction | 15% | Mean repayment amount per transaction |
| Average Principal Per Transaction | 10% | Mean principal amount per transaction |
| Principal-to-Total Repayment Ratio | 15% | Ratio of principal to total repayment amount |

#### Calculation Process:
1. **Normalization**: Each metric is normalized to a 0-1 scale using min-max scaling
2. **Weighted Sum**: Normalized metrics are combined using their respective weights
3. **Scaling**: Result is scaled to a 0-100 range
4. **Rounding**: Final score is rounded to 2 decimal places

**Implementation Notes:**
- Recalculated daily for all active agents
- Historical scores are preserved for trend analysis
- All calculations are logged for audit purposes

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
