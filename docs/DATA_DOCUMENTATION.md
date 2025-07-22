# Credit Health Intelligence Engine
## Data Documentation

This document provides detailed information about the data sources, schemas, and processing standards for the Credit Health Intelligence Engine.

## Repayment Score Calculation

The repayment score is a critical metric used throughout the system to assess agent creditworthiness. The calculation follows the standards defined in the [Rule Book](../rule%20book.md#repayment-score-calculation-standards).

### Data Sources for Repayment Score

1. **repayment_report.csv**
   - Primary source for repayment transactions
   - Used to calculate metrics like total repayments, principal amounts, and transaction counts

2. **DPD.xlsx**
   - Provides Days Past Due information
   - Used to identify and penalize late payments

3. **credit_Agents.xlsx**
   - Contains agent credit limits and status
   - Used for normalization and validation

### Key Fields Used

| Field | Source Table | Description |
|-------|-------------|-------------|
| repayment_amount | repayment_report | Amount repaid in each transaction |
| principal_amount | repayment_report | Principal portion of repayment |
| transaction_date | repayment_report | Date of repayment transaction |
| due_date | repayment_report | Original due date for payment |
| agent_id | All tables | Links to Bzid in credit_Agents |

### Calculation Process

1. **Data Extraction**:
   - Extract all repayment transactions for the agent
   - Filter for the relevant time period (typically last 6-12 months)

2. **Metric Calculation**:
   - Total Repayment Amount: Sum of all repayment_amount
   - Total Principal Repaid: Sum of principal_amount
   - Transaction Count: Count of unique transactions
   - Average Repayment: Total Repayment / Transaction Count
   - Average Principal: Total Principal / Transaction Count
   - Principal Ratio: Total Principal / Total Repayment

3. **Normalization**:
   - Each metric is normalized to a 0-1 scale using min-max scaling
   - Historical data is used to determine min/max bounds when available

4. **Weighted Aggregation**:
   - Apply weights to each normalized metric
   - Sum weighted scores
   - Scale to 0-100 range

5. **Final Score**:
   - Rounded to 2 decimal places
   - Capped at 100
   - Stored with agent's credit profile

### Data Quality Considerations

- Missing transactions are treated as 0 for that period
- Negative values are capped at 0
- Inconsistent data is flagged for review
- All calculations are logged for audit purposes

## Data Integration and Key Relationships

The `Bzid` field serves as the **primary business identifier** that connects all data points across the system. This key-based integration enables comprehensive analysis by linking agent information with their transaction history, credit performance, and regional data.

### How Data is Connected Using Bzid:

1. **Primary Reference**: `credit_Agents.xlsx` contains the master list of all agents, with `Bzid` as the unique identifier.

2. **Data Linking**: Other datasets connect to this primary reference through:
   - Direct `Bzid` match (e.g., in DPD.xlsx, repayment_report.csv)
   - `account` field that maps to `Bzid` (e.g., in sales_data.xlsx, Credit_sales_data.xlsx)

3. **Automatic Data Enrichment**: When processing any agent's data, the system automatically retrieves and combines related information from all datasets using these key relationships.

4. **Example Integration**: For agent `Bzid: 323238238`, the system can:
   - Retrieve basic agent information from `credit_Agents.xlsx`
   - Pull DPD history from `DPD.xlsx`
   - Match sales data from `sales_data.xlsx` using the `account` field
   - Link to regional manager details from `Region_contact.xlsx`
   - Analyze credit history from `Credit_history_sales_vs_credit_sales.xlsx`
   - Track repayments from `repayment_report.csv`

## Data Sources

*All canonical data source schemas, columns, validation rules, and sample data are now maintained in [DATA_DICTIONARY.md](./DATA_DICTIONARY.md). Please refer to that file for the latest and most complete specifications.*

This document focuses on:
- Repayment score and feature calculation logic
- Data processing and normalization flows
- Integration and enrichment examples
- Data quality and audit considerations

---

(Existing calculation logic, data flows, and integration sections retained below)
45966733,9820762252,"Shabbir M Kothari","Kothari Sales Corporation",2,5778.38
```

#### Sample Data
```csv
Region,Contact_Person,Email,Phone,Last_Updated
MP,"Aswin Satheesh","aswinsatheesh.work@gmail.com",1234567890,2023-12-01T00:00:00Z
"Tamil Nadu & Kerala","Aswin Satheesh","aswinsatheesh.work@gmail.com",1234567890,2023-12-01T00:00:00Z
Gujarat,"Aswin Satheesh","aswinsatheesh.work@gmail.com",1234567890,2023-12-01T00:00:00Z
```

### 7. repayment_report.csv

**Record Count:** 349,598 (as of last update)

#### Schema
| Column Name | Data Type | Description | Validation Rules |
|-------------|-----------|-------------|-------------------|
| Bzid | String (Foreign Key) | Business identifier | Required, matches credit_Agents.Bzid |
| Repayment_Date | DateTime | When payment was made | Required, ISO 8601 format |
| Repayment_Amount | Decimal | Total amount repaid | Required, > 0 |
| Principal_Amount | Decimal | Portion applied to principal | Required, <= Repayment_Amount |
| Interest_Amount | Decimal | Portion applied to interest | Required, = (Repayment_Amount - Principal_Amount) |

#### Data Quality Metrics
- **Completeness:** 99.9% for required fields
- **Accuracy:** 99.7% (validated against bank records)
- **Freshness:** Updated daily at 04:00 UTC

#### Sample Data
```csv
Bzid,Repayment_Date,Repayment_Amount,Principal_Amount,Interest_Amount
42405740,2023-04-27T14:03:00Z,1260.76,1260.00,0.76
25848971,2023-04-28T12:39:00Z,12607.56,12600.00,7.56
25848971,2023-04-28T12:39:00Z,1154.65,1153.95,0.70
```

### 8. sales_data.xlsx

**Record Count:** 13,913 (as of last update)

#### Schema
| Column Name | Data Type | Description | Validation Rules |
|-------------|-----------|-------------|-------------------|
| Transaction_Date | DateTime | When booking was made | Required, ISO 8601 format |
| Bzid | String (Foreign Key) | Business identifier | Required, matches credit_Agents.Bzid |
| Organization_Name | String | Business name | Required, title case |
| Status | String | Booking status | One of: BOOKED, CANCELLED, COMPLETED |
| Total_Seats | Integer | Number of seats booked | Required, >= 1 |
| GMV | Decimal | Gross Merchandise Value | Required, >= 0 |
| Agent_Commission | Decimal | Commission earned | Required, >= 0 |
| City | String | Business city | Required, title case |
| State | String | Business state | Required, title case |
| Region | String | Business region | Required |
| Is_Verified | Boolean | If booking was verified | Required |
| RO_Name | String | Relationship Officer name | Optional |
| RM_Name | String | Relationship Manager name | Optional |

#### Data Quality Metrics
- **Completeness:** 98.8% for required fields
- **Accuracy:** 99.3% (validated against booking system)
- **Freshness:** Updated hourly

#### Sample Data
```csv
Transaction_Date,Bzid,Organization_Name,Status,Total_Seats,GMV,Agent_Commission,City,State,Region,Is_Verified,RO_Name,RM_Name
2025-06-01T00:00:00Z,28115985,National Travels Belgaum,BOOKED,4,5800.00,580.00,Belagavi,Karnataka,Karnataka,false,Sohail Fazal,Shivaraj
2025-06-01T00:00:00Z,20692257,Shree Tours & Travels Karad,BOOKED,1,2280.00,0.00,Karad,Maharashtra,MH+Goa,false,Bajirao Yewale,
2025-06-01T00:00:00Z,28372265,Prisha Tours and Travels,BOOKED,1,1699.95,80.95,Mumbai,Maharashtra,MH+Goa,false,Rajkumarreddy Ranjolkar,
```
