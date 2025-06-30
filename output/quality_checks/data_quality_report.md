# Data Quality Report

This report provides an overview of data quality across all input datasets.

## Table of Contents
- [credit_Agents.xlsx](#credit_agentsxlsx)
- [Credit_history_sales_vs_credit_sales.xlsx](#credit_history_sales_vs_credit_salesxlsx)
- [Credit_sales_data.xlsx](#credit_sales_dataxlsx)
- [DPD.xlsx](#dpdxlsx)
- [Region_contact.xlsx](#region_contactxlsx)
- [sales_data.xlsx](#sales_dataxlsx)

---

## credit_Agents.xlsx <a id='credit_agentsxlsx'></a>

**Shape:** (1915, 7) (rows × columns)

✅ No missing values found.

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `Bzid` | 24915690.96 | 10010576.00 | 20689935.50 | 24238807.00 | 27493304.00 | 49912503.00 |
| `Phone` | 9201911831.75 | 6005188916.00 | 8889679749.50 | 9445336786.00 | 9844655954.00 | 9998877014.00 |
| `Approval Amount` | 24773.89 | 10000.00 | 10000.00 | 10000.00 | 27000.00 | 234000.00 |
| `Credit Limit` | 23422.77 | 9500.00 | 10000.00 | 10000.00 | 26000.00 | 192000.00 |
| `Credit Line Balance` | 20247.25 | 13.67 | 10000.00 | 10000.00 | 22406.12 | 154344.85 |

### Missing Data Visualization

![Missing Values in credit_Agents.xlsx](output\quality_checks\missing_values_credit_Agents.png)

---

## Credit_history_sales_vs_credit_sales.xlsx <a id='credit_history_sales_vs_credit_salesxlsx'></a>

**Shape:** (1416, 19) (rows × columns)

### Missing Values

| Column | Missing Count | Missing % |
|--------|--------------:|----------:|
| `Account` | 3 | 0.21% |
| `June Total GMV - Year25` | 3 | 0.21% |
| `% May Total GMV - Year25 consumption` | 3 | 0.21% |
| `May Total GMV - Year25` | 3 | 0.21% |
| `May Credit Gmv` | 3 | 0.21% |
| `% May Total GMV - Year25 consumption.1` | 3 | 0.21% |
| `JuneCredit Gmv` | 3 | 0.21% |
| `Jan Credit Gmv` | 2 | 0.14% |
| `Jan Total GMV - Year25` | 2 | 0.14% |
| `% Jan Total GMV - Year25 consumption` | 2 | 0.14% |
| `Feb Total GMV - Year25` | 2 | 0.14% |
| `% Feb Total GMV - Year25 consumption` | 2 | 0.14% |
| `% Mar Total GMV - Year25 consumption` | 2 | 0.14% |
| `Mar Credit Gmv` | 2 | 0.14% |
| `Mar Total GMV - Year25` | 2 | 0.14% |
| `Feb Credit Gmv` | 2 | 0.14% |
| `Apr Total GMV - Year25` | 1 | 0.07% |
| `Apr Credit Gmv` | 1 | 0.07% |
| `% Apr Total GMV - Year25 consumption` | 1 | 0.07% |

### Duplicate Rows

🔍 Found **3** duplicate rows based on key columns.

**Example duplicate keys:**

1. Account: `nan`
2. Account: `nan`
3. Account: `nan`

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `Account` | 25031445.40 | 10010576.00 | 20578156.00 | 24331556.00 | 27584438.00 | 49912503.00 |
| `Jan Total GMV - Year25` | 60854.48 | 0.00 | 1978.50 | 11267.00 | 34307.00 | 42794778.00 |
| `Jan Credit Gmv` | 33482.89 | 0.00 | 0.00 | 2509.00 | 15626.50 | 23511180.00 |
| `% Jan Total GMV - Year25 consumption` | 0.46 | 0.00 | 0.00 | 0.47 | 0.95 | 1.00 |
| `Feb Total GMV - Year25` | 64812.91 | 0.00 | 1745.50 | 10420.50 | 36560.50 | 45628569.00 |
| `Feb Credit Gmv` | 35263.71 | 0.00 | 0.00 | 2199.00 | 15145.50 | 24857746.00 |
| `% Feb Total GMV - Year25 consumption` | 0.44 | 0.00 | 0.00 | 0.37 | 0.92 | 1.00 |
| `Mar Total GMV - Year25` | 24070.60 | 0.00 | 0.00 | 3629.00 | 14750.25 | 16906365.00 |
| `Mar Credit Gmv` | 13120.52 | 0.00 | 0.00 | 0.00 | 6770.00 | 9243809.00 |
| `% Mar Total GMV - Year25 consumption` | 0.40 | 0.00 | 0.00 | 0.00 | 0.93 | 1.00 |
| `Apr Total GMV - Year25` | 93724.98 | 0.00 | 1890.00 | 10810.00 | 33417.00 | 44373648.00 |
| `Apr Credit Gmv` | 46004.28 | 0.00 | 0.00 | 1340.00 | 15237.50 | 21772797.00 |
| `% Apr Total GMV - Year25 consumption` | 0.41 | 0.00 | 0.00 | 0.19 | 0.91 | 1.00 |
| `May Total GMV - Year25` | 18108.32 | 0.00 | 0.00 | 5707.00 | 19147.00 | 341265.00 |
| `May Credit Gmv` | 8715.07 | 0.00 | 0.00 | 0.00 | 7729.00 | 213413.00 |
| `% May Total GMV - Year25 consumption` | 0.37 | 0.00 | 0.00 | 0.00 | 0.92 | 1.00 |
| `June Total GMV - Year25` | 8922.99 | 0.00 | 0.00 | 2500.00 | 9909.00 | 324456.60 |
| `JuneCredit Gmv` | 4326.15 | 0.00 | 0.00 | 0.00 | 3648.00 | 118592.20 |
| `% May Total GMV - Year25 consumption.1` | 0.31 | 0.00 | 0.00 | 0.00 | 0.86 | 1.00 |

### Missing Data Visualization

![Missing Values in Credit_history_sales_vs_credit_sales.xlsx](output\quality_checks\missing_values_Credit_history_sales_vs_credit_sales.png)

---

## Credit_sales_data.xlsx <a id='credit_sales_dataxlsx'></a>

**Shape:** (6226, 4) (rows × columns)

✅ No missing values found.

### Duplicate Rows

🔍 Found **4070** duplicate rows based on key columns.

**Example duplicate keys:**

1. account: `28372265`, DATE: `2025-06-01 00:00:00`
2. account: `27602131`, DATE: `2025-06-01 00:00:00`
3. account: `25125208`, DATE: `2025-06-01 00:00:00`
4. account: `11769330`, DATE: `2025-06-01 00:00:00`
5. account: `11769330`, DATE: `2025-06-01 00:00:00`

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `account` | 24517515.50 | 10091139.00 | 20273220.00 | 23944592.00 | 27388173.00 | 49912503.00 |
| `GMV` | 1860.48 | 145.00 | 924.00 | 1400.00 | 2220.00 | 21595.00 |

### Missing Data Visualization

![Missing Values in Credit_sales_data.xlsx](output\quality_checks\missing_values_Credit_sales_data.png)

---

## DPD.xlsx <a id='dpdxlsx'></a>

**Shape:** (91, 7) (rows × columns)

✅ No missing values found.

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `Phone` | 9154259215.47 | 6380663812.00 | 8696044260.00 | 9397176752.00 | 9866298247.50 | 9994947237.00 |
| `Bzid` | 24609479.09 | 10644128.00 | 21013970.00 | 23988142.00 | 27184508.50 | 49530786.00 |
| `Dpd` | 284.76 | 1.00 | 5.00 | 305.00 | 547.50 | 737.00 |
| `Pos` | 13993.37 | 814.29 | 4471.28 | 8586.05 | 14215.47 | 92006.24 |

### Missing Data Visualization

![Missing Values in DPD.xlsx](output\quality_checks\missing_values_DPD.png)

---

## Region_contact.xlsx <a id='region_contactxlsx'></a>

**Shape:** (9, 3) (rows × columns)

✅ No missing values found.

### Missing Data Visualization

![Missing Values in Region_contact.xlsx](output\quality_checks\missing_values_Region_contact.png)

---

## sales_data.xlsx <a id='sales_dataxlsx'></a>

**Shape:** (13913, 13) (rows × columns)

### Missing Values

| Column | Missing Count | Missing % |
|--------|--------------:|----------:|
| `RM Name` | 3,789 | 27.23% |
| `Ro Name` | 1,121 | 8.06% |
| `Region` | 5 | 0.04% |
| `State` | 1 | 0.01% |
| `organizationname` | 1 | 0.01% |
| `city` | 1 | 0.01% |

### Duplicate Rows

🔍 Found **10061** duplicate rows based on key columns.

**Example duplicate keys:**

1. account: `28372265`, DATE(a.creationtime): `2025-06-01 00:00:00`
2. account: `41784449`, DATE(a.creationtime): `2025-06-01 00:00:00`
3. account: `27602131`, DATE(a.creationtime): `2025-06-01 00:00:00`
4. account: `25125208`, DATE(a.creationtime): `2025-06-01 00:00:00`
5. account: `11769330`, DATE(a.creationtime): `2025-06-01 00:00:00`

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `account` | 25749034.58 | 10010576.00 | 20362456.00 | 24862812.00 | 28327391.00 | 49912503.00 |
| `TotalSeats` | 1.66 | 1.00 | 1.00 | 1.00 | 2.00 | 6.00 |
| `GMV` | 1701.93 | 60.00 | 849.45 | 1293.00 | 2016.00 | 21595.00 |
| `AgentCommission(Exe GDS)` | 118.60 | 0.00 | 50.00 | 90.00 | 152.00 | 1727.62 |

### Missing Data Visualization

![Missing Values in sales_data.xlsx](output\quality_checks\missing_values_sales_data.png)

---

