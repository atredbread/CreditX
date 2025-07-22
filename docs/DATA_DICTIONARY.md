# Data Dictionary

This document defines the unified data schema for the Credit Health Intelligence Engine. All data processing must adhere to these standards.

## Data Integration Overview

The `Bzid` field serves as the primary business identifier that connects all data points across the Credit Health Intelligence Engine. This key-based integration enables comprehensive analysis by linking agent information with their transaction history, credit performance, and regional data.

### Key Integration Points:

1. **Primary Data Source**: `credit_Agents.xlsx` serves as the master reference containing the complete list of onboarded agents. The `Bzid` in this file is the source of truth for all agent-related data.

2. **Data Linking**: All other datasets reference this primary key through either:
   - Direct `Bzid` match (e.g., in DPD.xlsx, repayment_report.csv)
   - `account` field that maps to `Bzid` (e.g., in sales_data.xlsx, Credit_sales_data.xlsx)

3. **Automatic Data Enrichment**: When processing any agent's data, the system automatically retrieves and combines related information from all datasets using these key relationships.

4. **Example Integration Flow**:
   - Start with `Bzid` from credit_Agents
   - Retrieve DPD history using the same `Bzid`
   - Match sales data via the `account` = `Bzid` relationship
   - Enrich with regional information through region-based joins

## 1. Data Sources

### 1.1 Agent Details.xlsx

**Description**: Comprehensive agent information including contact details, organizational hierarchy, and regional assignments. This is the primary source of truth for agent master data.

**Location**: `/source_data/Agent Details.xlsx`

**Key Columns**:
- `account` (Primary Key, int64): Unique agent identifier that maps to `Bzid` in other datasets
- `agentname` (string, encrypted): Agent's full name
- `organizationname` (string): Legal business name
- `RoName` (string, encrypted): Region Officer name
- `RmName` (string, encrypted): Relationship Manager name
- `email` (string, encrypted): Contact email
- `mobile` (string, encrypted): Contact number
- `city` (string): City code
- `cityname` (string): Full city name
- `state` (string): State code
- `StateName` (string): Full state name
- `region` (string): Region code
- `AgentRegion` (string): Full region name
- `agenttype` (string): Type of agent (e.g., BP_AGENT, OTHERS_NBP)
- `status` (string): Account status (e.g., ACTIVE)
- `SO?TSE` (string): Sales Officer/Territory Sales Executive

**Relationships**:
- **Primary Relationship**: `account` in this file maps to `Bzid` in all other datasets
- **Related to**: All transaction and credit datasets
  - **Join Key**: `account` = `Bzid`
  - **Type**: one-to-many (one agent can have multiple transactions/records)

**Data Quality Rules**:
- `account` must be unique and non-null (matches `Bzid` in other datasets)
- All encrypted fields must be properly formatted before encryption
- Region codes must match standardized values from `Region_contact.xlsx`
- Status must be one of: ACTIVE, INACTIVE, SUSPENDED
- Email must be in valid format
- Phone numbers must follow E.164 format

**Usage Notes**:
- This file serves as the master reference for all agent information
- All new agents must be added here before they can be processed by the system
- Changes to agent details should be made in this file and will be reflected across all reports

### 1.2 credit_Agents.xlsx (Legacy)

**Status**: Deprecated - Use `Agent Details.xlsx` as the primary source

**Description**: Legacy list of onboarded agents. This file is being phased out in favor of `Agent Details.xlsx` which contains more comprehensive and up-to-date information.

**Location**: `/data/raw/credit_Agents.xlsx`

**Key Columns**: 
- `Bzid` (Primary Key, String): The canonical identifier for all agent-related data across the platform. This is the single source of truth for agent identification.

**Core Role in Data Integration**:
- Serves as the **master reference** for all agent-related data
- Defines the complete set of valid agents in the system
- All other datasets must align with and connect to this master list
- Any agent not present in this file is considered inactive or non-existent in the system

**Relationships**:
- **Related to**: All other datasets in the system
  - **Join Key**: `Bzid` (or `account` in some datasets)
  - **Type**: one-to-many (one agent can have multiple records in other datasets)
  - **Example**: A single agent (Bzid: 323238238) can have:
    - Multiple DPD records
    - Multiple sales transactions
    - Multiple repayment entries
    - Multiple credit history entries

**Data Quality Rules**:
- `Bzid` must be unique and non-null across the entire platform
- `Bzid` format must be strictly enforced (8-10 alphanumeric characters)
- No duplicate `Bzid` values allowed
- `Region` must match standardized region codes
- Email must be in valid format
- Phone numbers must be in E.164 format
- All required fields must be populated for active agents

### 1.2 repayment_report.csv

**Description**: Records customer repayment transactions, including repayment date, total repaid amount, and principal repaid for each agent (Bzid). Used for repayment score calculations, cash flow analysis, and risk profiling.

**Location**: `/source_data/repayment report.csv`

**Key Columns**:
- `Bzid` (Primary Key, int64): Unique agent identifier, maps to `account` in Agent Details.xlsx
- `Customer Repayment Date` (datetime/string): Timestamp of repayment transaction (format: 'Month DD, YYYY, HH:MM AM/PM')
- `Customer Repayment Amount` (float): Total amount repaid by the customer in the transaction
- `Customer Principle Repaid` (float): Portion of the repayment applied to principal outstanding

**Relationships**:
- **Primary Relationship**: `Bzid` links to agent master data and all credit/transaction datasets
- **Related to**: Agent Details.xlsx, DPD.xlsx, sales_data.xlsx, etc.
  - **Join Key**: `Bzid`
  - **Type**: one-to-many (one agent can have multiple repayment transactions)

**Data Quality Rules**:
- `Bzid` must be valid and present in Agent Details.xlsx
- Dates must be valid and parsable to ISO 8601 (recommended to convert for processing)
- Amount fields must be numeric and non-negative
- No NULLs in required fields

**Usage Notes**:
- Used for daily repayment score calculations
- Repayment events are aggregated by agent and time period for risk analytics
- Data is retained for 7 years per governance policy

### 1.3 DPD.xlsx (Days Past Due)

**Description**: Tracks payment delays and defaults for each agent.

**Location**: `/data/raw/DPD.xlsx`

**Key Columns**: `Bzid`, `Date`, `DPD`

**Relationships**:
- **Related to**: credit_Agents.xlsx  
  **Join Key**: `Bzid`  
  **Type**: many-to-one (many DPD records can belong to one agent)

## 2. Core Data Structures

### 2.1 Agent Information

| Column | Type | Description | Validation Rules |
|--------|------|-------------|------------------|
| Bzid | string | Unique agent identifier | Required, non-null, unique |
| AgentName | string | Legal name of agent | Required, non-null |
| Phone | string | Contact number | E.164 format |
| Email | string | Contact email | Valid email format |
| Region | string | Geographic region | Must match Region_contact.xlsx |
| City | string | City name | Title case |
| State | string | State/Province | Title case |
| JoinDate | date | Date agent joined | YYYY-MM-DD |
| CreditLimit | float | Total credit limit | Non-negative |
| CurrentBalance | float | Current outstanding balance | Non-negative |
| ActiveStatus | boolean | Whether agent is active | True/False |

### 2.2 DPD (Days Past Due) Data

| Column | Type | Description | Validation Rules |
|--------|------|-------------|------------------|
| Bzid | string | Agent identifier | Must exist in credit_Agents |
| Date | date | Transaction date | YYYY-MM-DD |
| DPD | integer | Days past due | 0 for on-time |
| Amount | float | Amount in default | Non-negative |
| Status | string | Payment status | ['current', '30_dpd', '60_dpd', '90+_dpd'] |
| DueDate | date | Original due date | YYYY-MM-DD |
| PaymentDate | date | Actual payment date | YYYY-MM-DD, nullable |

## 3. Data Processing Standards

### 3.1 Field Validation
- All dates must be in ISO 8601 format (YYYY-MM-DD)
- Numeric fields must not contain non-numeric characters
- String fields must be properly encoded (UTF-8)
- Boolean fields must be stored as True/False (not 1/0 or 'Yes'/'No')

### 3.2 Data Types
- **String**: Text data, including codes and identifiers
- **Integer**: Whole numbers (e.g., DPD days, counts)
- **Float**: Decimal numbers (e.g., amounts, percentages)
- **Date**: Calendar dates (YYYY-MM-DD)
- **Boolean**: True/False values

## 4. Data Quality Rules

### 4.1 Completeness
- No NULL values in required fields
- All foreign keys must reference existing records

### 4.2 Consistency
- Region codes must be consistent across all datasets
- Date ranges must be valid (e.g., PaymentDate ≥ DueDate for late payments)

### 4.3 Accuracy
- DPD values must be non-negative integers
- Credit utilization must be between 0-100%
- No future dates allowed in historical data

## 5. Data Retention

| Data Type | Retention Period | Archive Location |
|-----------|------------------|------------------|
| Raw Data | 3 years | /data/archive/raw/ |
| Processed Data | 2 years | /data/archive/processed/ |
| Reports | 1 year | /data/archive/reports/ |

## 6. Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-07-20 | 2.0 | Updated to unified standards | System |
| 2025-01-15 | 1.0 | Initial version | Team |

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `Bzid` | int64 | 23058821, 21425838, 26840517 |  |
| `Phone` | int64 | 9966119584, 8722348686, 7411103360 |  |
| `Credit Line Setup Co` | object | 19-5-2023, 9:47 AM, 18-5-2023, 5:29 PM, 19-5-2023, 1:54 PM |  |
| `Approval Amount` | int64 | 10000, 37000, 166000 |  |
| `Credit Limit` | int64 | 10000, 37000, 166000 |  |
| `Credit Line Balance` | float64 | 10000.0, 37000.0, 140576.14 |  |
| `Unnamed: 6` | object | D, D, D |  |

---

## Credit_history_sales_vs_credit_sales.xlsx

**Description**: Historical sales and credit sales data

**Number of Rows**: 1,416

**Key Columns**: `Account`

### Columns

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `Account` | int64 | 25125208, 28499610, 13810786 |  |
| `Jan Total GMV - Year25` | int64 | 458674, 226523, 319249 |  |
| `Jan Credit Gmv` | int64 | 322444, 0, 0 |  |
| `% Jan Total GMV - Year25 consumption` | float64 | 0.703, 0.0, 0.0 |  |
| `Feb Total GMV - Year25` | int64 | 388310, 228508, 236825 |  |
| `Feb Credit Gmv` | int64 | 147399, 0, 0 |  |
| `% Feb Total GMV - Year25 consumption` | float64 | 0.3796, 0.0, 0.0 |  |
| `Mar Total GMV - Year25` | int64 | 223093, 89740, 108210 |  |
| `Mar Credit Gmv` | int64 | 64794, 0, 0 |  |
| `% Mar Total GMV - Year25 consumption` | float64 | 0.2904, 0.0, 0.0 |  |
| `Apr Total GMV - Year25` | int64 | 507822, 520917, 351687 |  |
| `Apr Credit Gmv` | int64 | 222337, 0, 0 |  |
| `% Apr Total GMV - Year25 consumption` | float64 | 0.4378, 0.0, 0.0 |  |
| `May Total GMV - Year25` | int64 | 341265, 249222, 239615 |  |
| `May Credit Gmv` | int64 | 121208, 0, 0 |  |
| `% May Total GMV - Year25 consumption` | float64 | 0.3552, 0.0, 0.0 |  |
| `June Total GMV - Year25` | float64 | 324456.6, 63996.15, 98008.61 |  |
| `JuneCredit Gmv` | float64 | 118592.2, 0.0, 0.0 |  |
| `% May Total GMV - Year25 consumption.1` | float64 | 0.3655, 0.0, 0.0 |  |

---

## Credit_sales_data.xlsx

**Description**: Monthly credit sales transactions

**Number of Rows**: 6,226

**Key Columns**: `account`, `DATE`

### Relationships

- **Related to**: credit_Agents.xlsx  
  **On**: ('account', 'Bzid')  
  **Type**: many-to-one

### Columns

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `DATE` | datetime64[ns] | 2025-06-01 00:00:00, 2025-06-01 00:00:00, 2025-06-01 00:00:00 |  |
| `account` | int64 | 28115985, 28372265, 27602131 |  |
| `GMV` | float64 | 5800.0, 1699.95, 1020.0 |  |
| `tin` | object | 7AB67HK6, 7A46MMDV, 7AHH4PNE |  |

---

## Region_contact.xlsx

**Description**: Mapping of regions to contact persons

**Number of Rows**: 9

**Key Columns**: `Region`

### Columns

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `Region` | object | MP, Tamil Nadu & Kerala, Gujarat |  |
| `Manager` | object | Aswinsatheesh.work@gamil.com, Aswinsatheesh.work@gamil.com, Aswinsatheesh.work@gamil.com |  |
| `Name` | object | Aswin, Aswin, Aswin |  |

---

## DPD.xlsx

**Description**: Days Past Due information for credit accounts

**Number of Rows**: 91

**Key Columns**: `Bzid`, `Phone`

### Columns

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `Anchor` | object | REDBUS, REDBUS, REDBUS |  |
| `Phone` | int64 | 9885777379, 9944190111, 9820762252 |  |
| `Bzid` | int64 | 24939241, 13910540, 45966733 |  |
| `Username` | object | RAHAMTHULLA SHAIK, MANICKAM RAJESH, SHABBIR M KOTHARI |  |
| `Business Name` | object | SRT travels - Rayachoti, TRAVEL ZONE TOURS AND TRAVELS, KOTHARI SALES CORPORATION |  |
| `Dpd` | int64 | 3, 3, 2 |  |
| `Pos` | float64 | 5379.3, 26004.09, 5778.38 |  |

---

## sales_data.xlsx

**Description**: Complete sales transaction data

**Number of Rows**: 13,913

**Key Columns**: `account`, `DATE(a.creationtime)`

### Columns

| Column | Data Type | Sample Values | Description |
|--------|-----------|----------------|-------------|
| `DATE(a.creationtime)` | datetime64[ns] | 2025-06-01 00:00:00, 2025-06-01 00:00:00, 2025-06-01 00:00:00 |  |
| `account` | int64 | 28115985, 20692257, 28372265 |  |
| `organizationname` | object | National Travels Belgaum, Shree Tours & Travels Karad 9021250999, Prisha Tours and Travels |  |
| `status` | object | BOOKED, BOOKED, BOOKED |  |
| `TotalSeats` | int64 | 4, 1, 1 |  |
| `GMV` | float64 | 5800.0, 2280.0, 1699.95 |  |
| `AgentCommission(Exe GDS)` | float64 | 580.0, 0.0, 80.95 |  |
| `city` | object | Belagavi, Karad, Mumbai |  |
| `State` | object | Karnataka, Pune, Mumbai |  |
| `Region` | object | Karnataka, MH+Goa, MH+Goa |  |
| `Check` | bool | False, False, False |  |
| `Ro Name` | object | Sohail fazal, Bajirao  yewale, Rajkumarreddy Ranjolkar |  |
| `RM Name` | object | Shivaraj, Shivaraj, Shriram S |  |

---

