# Data Dictionary

This document describes the structure and content of all input datasets.

## credit_Agents.xlsx

**Description**: Master list of onboarded agents with credit information

**Number of Rows**: 1,915

**Key Columns**: `Bzid`, `Phone`

### Relationships

- **Related to**: DPD.xlsx  
  **On**: Bzid, Phone  
  **Type**: one-to-one

### Columns

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

