# Credit Risk Feature Engineering Documentation

## 1. Data Files Overview

### 1.1 credit_Agents.xlsx
- **Purpose**: Contains agent credit information and setup details
- **Key Columns**:
  - `Bzid` (int64): Unique agent identifier (primary key)
  - `Phone` (int64): Contact number
  - `Credit Line Setup Co` (datetime): Timestamp of credit line setup
  - `Approval Amount` (float64): Original approved credit amount
  - `Credit Limit` (float64): Maximum credit allowed
  - `Credit Line Balance` (float64): Current credit utilization
  - `Status` (str): Account status (e.g., 'Active', 'Suspended')
  - `Region` (str): Business region code
  - `Created At` (datetime): Account creation timestamp
  - `Updated At` (datetime): Last update timestamp

### 1.2 DPD.xlsx
- **Purpose**: Tracks Days Past Due and Point of Sale information
- **Key Columns**:
  - `Anchor` (str): Business anchor (e.g., REDBUS)
  - `Phone` (int64): Contact number
  - `Bzid` (int64): Agent identifier
  - `Username` (str): Agent's name
  - `Business Name` (str): Name of the business
  - `Dpd` (int64): Days past due
  - `Pos` (float64): Point of Sale amount

### 1.3 repayment_report.csv
- **Purpose**: Records customer repayment transactions
- **Key Columns**:
  - `All Anchors Onboarding Info - Anchor → Bzid` (int64): Agent identifier
  - `Customer Repayment Date` (str): When payment was made (e.g., 'April 27, 2023, 2:03 PM')
  - `Customer Repayment Amount` (str): Total payment amount (formatted with commas)
  - `Customer Principle Repaid` (str): Principal amount repaid (formatted with commas)

### 1.4 Credit_sales_data.xlsx
- **Purpose**: Tracks credit sales transactions
- **Key Columns**:
  - `DATE` (datetime): Transaction date
  - `Bzid` (int64): Agent identifier (matches credit_Agents.Bzid)
  - `TransactionDate` (datetime): Date and time of transaction
  - `Amount` (float64): Transaction amount (credit sales only)
  - `TransactionId` (str): Unique transaction identifier
  - `Status` (str): Transaction status (e.g., 'Completed', 'Refunded')
  - `Region` (str): Region code

### 1.5 sales_data.xlsx
- **Purpose**: Tracks all sales transactions (both cash and credit)
- **Granularity**: Individual transactions by agent and date
- **Key Columns**:
  - `Bzid` (int64): Agent identifier (matches credit_Agents.Bzid)
  - `SaleDate` (datetime): Transaction date and time
  - `TotalSales` (float64): Total sales amount (GMV)
  - `TotalSeats` (int64): Number of seats booked
  - `Commission` (float64): Agent commission amount
  - `PaymentType` (str): 'Credit' or 'Cash'
  - `Status` (str): Transaction status (e.g., 'Completed', 'Cancelled')
  - `City` (str): City of operation
  - `State` (str): State of operation
  - `Region` (str): Business region code
  - `RO` (str): Regional officer name
  - `RM` (str): Relationship manager name
  - `CreatedAt` (datetime): Record creation timestamp

### 1.6 Credit_history_sales_vs_credit_sales.xlsx
- **Purpose**: Historical monthly sales and credit sales data
- **Granularity**: Monthly aggregation by agent
- **Key Columns**:
  - `Bzid` (int64): Agent identifier (matches credit_Agents.Bzid)
  - `MonthYear` (datetime): Reference month (e.g., '2025-01-01' for Jan 2025)
  - `TotalGMV` (float64): Monthly total sales amount
  - `CreditGMV` (float64): Monthly credit sales amount
  - `CreditRatio` (float64): Credit to total GMV ratio (0-1)
  - `TransactionCount` (int64): Number of transactions
  - `AvgTicketSize` (float64): Average transaction value

### 1.7 Region_contact.xlsx
- **Purpose**: Maps regions to their respective managers
- **Key Columns**:
  - `Region` (str): Region code (e.g., 'MP')
  - `RegionName` (str): Full region name
  - `ManagerEmail` (str): Manager's email
  - `ManagerName` (str): Manager's full name
  - `ManagerPhone` (str): Manager's contact number
  - `RO` (str): Regional officer name
  - `ROEmail` (str): Regional officer email
  - `RM` (str): Relationship manager name
  - `RMEmail` (str): Relationship manager email

## 2. Calculated Features

### 2.1 Credit Utilization Ratio
**Formula**: 
```
Credit Utilization = (Credit Line Balance / Credit Limit) * 100
```
**Range**: 0-100% (Higher is better)
**Risk Levels**:
- <30%: high risk
- 30-70%: medium risk
- >70%: low risk

### 2.2 Delinquency Status
Based on DPD (Days Past Due):
- `Current`: DPD = 0
- `1-29 days`: Early Delinquency
- `30-59 days`: Mild Delinquency
- `60-89 days`: Severe Delinquency
- `90+ days`: Default

### 2.3 Payment History (from repayment report)
**Sample Data**:
```
Bzid     | Repayment Date       | Amount   | Principal
-----------------------------------------------
42405740 | 2023-04-27 14:03:00 | 1,260.76 | 1,260.00
25848971 | 2023-04-28 12:39:00 | 12,607.56| 12,600.00
```

## 3. Risk Assessment

### 3.1 Risk Score Components
1. **Credit Utilization (30% weight)**
   - <30%: Low risk (1)
   - 30-70%: Medium risk (2)
   - >70%: High risk (3)

---

### 3.2 Repayment-Based Agent Scoring System

#### **How the Score is Calculated**

1. **Normalization (Min-Max Scaling):**
   - Each metric is normalized to a 0–1 scale using the formula:
     
     `X_norm = (X - X_min) / (X_max - X_min)`
   
   - This ensures all metrics are comparable regardless of their original scale.

2. **Metrics and Weights:**
   - **Total repayment amount:** 25%
   - **Total principal repaid:** 20%
   - **Transaction count:** 15%
   - **Average repayment per transaction:** 15%
   - **Average principal per transaction:** 10%
   - **Principal-to-total repayment ratio:** 15%

3. **Weighted Score Calculation:**
   - The final score for each agent is computed as:
     
     `Score = 0.25 * Total Repayment_norm + 0.20 * Total Principal_norm + 0.15 * Txn Count_norm + 0.15 * Avg Repay_norm + 0.10 * Avg Principal_norm + 0.15 * Principal Ratio_norm`
   
   - Higher scores indicate better repayment behavior and credit health.

**Implementation Note:**
- This scoring system is implemented in the repayment analysis script. Each agent receives a `score` column, and agents are ranked by this score for downstream risk classification or reporting.

---

2. **Payment History (40% weight)**
   - No late payments: Low risk (1)
   - 1-2 late payments: Medium risk (2)
   - 3+ late payments: High risk (3)

3. **DPD Status (30% weight)**
   - Current: Low risk (1)
   - 1-29 days: Medium risk (2)
   - 30+ days: High risk (3)

### 3.2 Example Risk Calculation
For Bzid 25305319:
- Credit Utilization: 33.37% (Medium)
- Payment History: No late payments (Low)
- DPD Status: Current (Low)

**Weighted Score**:
(0.3 * 2) + (0.4 * 1) + (0.3 * 1) = 1.3 (Low Risk)

## 4. Region-wise Analysis

### 4.1 Region Contact Information
```
Region              | Manager                      | Contact
--------------------------------------------------
MP                  | Aswinsatheesh.work@gamil.com | Aswin
Tamil Nadu & Kerala | Aswinsatheesh.work@gamil.com | Aswin
Gujarat             | Aswinsatheesh.work@gamil.com | Aswin
```

### 4.2 Regional Risk Metrics
Calculated based on:
- Average credit utilization
- Default rates
- Recovery rates
- DPD distribution

## 5. Data Validation Rules

1. **Credit Limit Validation**:
   ```
   Credit Line Balance ≤ Credit Limit ≤ Approval Amount
   ```

2. **DPD Validation**:
   - Must be non-negative integer
   - Should be updated daily

3. **Payment Validation**:
   - Repayment amount ≥ Principal amount
   - No future-dated payments allowed

## 6. Data Quality Checks

1. **Missing Values**:
   - Check for null/empty values in critical fields
   - Verify required fields are populated

2. **Data Consistency**:
   - Validate Bzid exists in credit_Agents
   - Ensure dates are in correct format

3. **Outlier Detection**:
   - Identify unusually high/low transaction amounts
   - Flag abnormal DPD values

## 7. GMV Trend Analysis

### 7.1 Data Sources and Granularity

#### Sales Data (`sales_data.xlsx`)
- **Content**: Day-level transaction data for all sales (both cash and credit)
- **Granularity**: Individual transactions aggregated by agent and day
- **Key Columns**:
  - `account`: Agent identifier
  - `DATE(a.creationtime)`: Transaction date
  - `GMV`: Total transaction value (cash + credit)
  - `region`: Geographical region

#### Credit Sales Data (`Credit_sales_data.xlsx`)
- **Content**: Day-level transaction data for credit sales only
- **Granularity**: Individual credit transactions by agent and day
- **Key Columns**:
  - `Bzid`: Agent identifier
  - `SaleDate`: Transaction date
  - `CreditAmount`: Credit portion of the transaction
  - `TotalAmount`: Total transaction amount (should match sales_data for credit sales)

### 7.2 Key Metrics

- **Total GMV**: 
  - Sum of all sales (cash + credit) from `sales_data.xlsx`
  - Represents the complete sales volume
  - Formula: `SUM(GMV) GROUP BY account, date`

- **Credit GMV**: 
  - Sum of credit sales from `Credit_sales_data.xlsx`
  - Represents the credit exposure
  - Formula: `SUM(CreditAmount) GROUP BY Bzid, SaleDate`

- **Credit to Total Ratio**: 
  - Measures credit exposure as a percentage of total sales
  - Formula: `(Credit GMV / Total GMV) * 100`
  - Risk Indicators:
    - <30%: Low risk
    - 30-50%: Monitor
    - >50%: High risk

### 7.3 Analysis Capabilities

#### 7.3.1 Day-level Analysis
```python
# Load and prepare data
sales = pd.read_excel('source_data/sales_data.xlsx')
credit_sales = pd.read_excel('source_data/Credit_sales_data.xlsx')

# Convert dates
sales['date'] = pd.to_datetime(sales['DATE(a.creationtime)']).dt.date
credit_sales['date'] = pd.to_datetime(credit_sales['SaleDate']).dt.date

# Daily aggregation
daily_sales = sales.groupby(['account', 'date'])['GMV'].sum().reset_index()
daily_credit = credit_sales.groupby(['Bzid', 'date'])['CreditAmount'].sum().reset_index()

# Merge and calculate credit ratio
daily_combined = pd.merge(
    daily_sales, 
    daily_credit, 
    left_on=['account', 'date'], 
    right_on=['Bzid', 'date'],
    how='left'
)
daily_combined['credit_ratio'] = (daily_combined['CreditAmount'] / daily_combined['GMV']) * 100
```

#### 7.3.2 Week-level Analysis
```python
# Convert to weekly
weekly_sales = sales.groupby([
    'account',
    pd.Grouper(key='DATE(a.creationtime)', freq='W-MON')
])['GMV'].sum().reset_index()

weekly_credit = credit_sales.groupby([
    'Bzid',
    pd.Grouper(key='SaleDate', freq='W-MON')
])['CreditAmount'].sum().reset_index()

# Merge and calculate weekly metrics
weekly_combined = pd.merge(
    weekly_sales,
    weekly_credit,
    left_on=['account', 'DATE(a.creationtime)'],
    right_on=['Bzid', 'SaleDate'],
    how='left'
)
weekly_combined['credit_ratio'] = (weekly_combined['CreditAmount'] / weekly_combined['GMV']) * 100
```

#### 7.3.3 Regional Analysis
```python
# Regional credit consumption
if 'region' in sales.columns and 'region' in credit_sales.columns:
    regional_daily = sales.groupby(['region', 'date'])['GMV'].sum().reset_index()
    regional_credit = credit_sales.groupby(['region', 'date'])['CreditAmount'].sum().reset_index()
    
    regional_combined = pd.merge(
        regional_daily,
        regional_credit,
        on=['region', 'date'],
        how='left'
    )
    regional_combined['credit_ratio'] = (regional_combined['CreditAmount'] / regional_combined['GMV']) * 100
    
    # Calculate regional benchmarks
    regional_benchmarks = regional_combined.groupby('region').agg({
        'credit_ratio': ['mean', 'median', 'std']
    }).round(2)
```

#### 7.3.4 Trend Analysis
```python
def calculate_trend(series):
    """Calculate normalized trend slope"""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    y = np.array(series, dtype=float)
    mask = ~np.isnan(y)
    if sum(mask) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    slope = np.polyfit(x, y, 1)[0]
    return slope / np.mean(y) if np.mean(y) != 0 else 0.0

# Apply trend analysis
window = 4  # weeks
trend_metrics = weekly_combined.groupby('account')['credit_ratio']\
    .rolling(window=window, min_periods=2)\n    .apply(calculate_trend, raw=True)\n    .reset_index()
```

### 7.2 Trend Analysis Implementation

#### Data Sources
1. **Total GMV**: Aggregated from `sales_data.xlsx`:
   ```python
   # Example calculation
   total_gmv = sales_data.groupby(['account', 'DATE(a.creationtime)'])['GMV'].sum()
   ```

2. **Credit GMV**: From `Credit_history_sales_vs_credit_sales.xlsx`:
   - Monthly columns: Jan Total GMV, Jan Credit Gmv, Feb Total GMV, etc.
   - Handles naming inconsistencies (e.g., 'JuneCredit Gmv' vs 'Jun Credit Gmv')

#### Trend Calculation
- **Time Windows**:
  - Total GMV: 6-month trend
  - Credit GMV: 5-month trend (aligned with payment cycles)

- **Normalized Trend**:
  ```python
  def calculate_trend(values):
      if len(values) < 2 or all(v == 0 for v in values):
          return 0.0
      x = np.arange(len(values))
      y = np.array(values, dtype=float)
      # Remove any NaN values
      mask = ~np.isnan(y)
      if sum(mask) < 2:  # Need at least 2 valid points
          return 0.0
      x = x[mask]
      y = y[mask]
      slope = np.polyfit(x, y, 1)[0]
      return slope / np.mean(y) if np.mean(y) != 0 else 0.0
  ```

#### Risk Indicators
- **Credit Exposure**:
  - <30%: Low risk
  - 30-50%: Monitor
  - >50%: High risk

- **Trend Analysis**:
  - Increasing credit ratio: Potential risk
  - Decreasing total GMV with stable credit: Warning sign
  - Seasonal patterns: Identify normal vs. abnormal fluctuations
