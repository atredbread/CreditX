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

**Shape:** (1599, 7) (rows × columns)

✅ No missing values found.

### Numeric Column Statistics

| Column | Mean | Min | 25% | 50% | 75% | Max |
|--------|-----:|----:|---:|---:|---:|----:|
| `Bzid` | 24938342.28 | 10010576.00 | 20485062.00 | 24301776.00 | 27642163.00 | 49912503.00 |
| `Phone` | 9226814710.67 | 6005188916.00 | 8984493197.50 | 9448394267.00 | 9845321933.00 | 9998877014.00 |
| `Approval Amount` | 24673.55 | 10000.00 | 10000.00 | 10000.00 | 27000.00 | 234000.00 |
| `Credit Limit` | 23441.40 | 9500.00 | 10000.00 | 10000.00 | 26000.00 | 192000.00 |
| `Credit Line Balance` | 20222.06 | 23.32 | 10000.00 | 10000.00 | 23000.00 | 155768.20 |

### Missing Data Visualization

![Missing Values in credit_Agents.xlsx](output\quality_checks\missing_values_credit_Agents.png)

---

## Credit_history_sales_vs_credit_sales.xlsx <a id='credit_history_sales_vs_credit_salesxlsx'></a>

**Shape:** (1416, 55) (rows × columns)

### Missing Values

| Column | Missing Count | Missing % |
|--------|--------------:|----------:|
| `%Mar Consumtion` | 195 | 13.77% |
| `%Feb consumtion` | 184 | 12.99% |
| `%May Total GMV - Year24 Consumption` | 174 | 12.29% |
| `%Apr Consumtion` | 170 | 12.01% |
| `%Jan Consumption` | 169 | 11.94% |
| `Jan Credit Gmv` | 3 | 0.21% |
| `Account` | 3 | 0.21% |
| `Feb Credit Gmv` | 3 | 0.21% |
| `Feb Total GMV - Year24` | 3 | 0.21% |
| `Jan Total GMV - Year24 Sales GMV` | 3 | 0.21% |
| `Mar Credit Gmv` | 3 | 0.21% |
| `Apr Total GMV - Year24` | 3 | 0.21% |
| `Apr Credit Gmv` | 3 | 0.21% |
| `May Total GMV - Year24` | 3 | 0.21% |
| `Mar Total GMV - Year24` | 3 | 0.21% |
| `May Credit Gmv` | 3 | 0.21% |
| `June Total GMV - Year24` | 3 | 0.21% |
| `June Credit Gmv` | 3 | 0.21% |
| `%Jun Total GMV - Year24 consumtion` | 3 | 0.21% |
| `May Total GMV - Year25` | 3 | 0.21% |
| `May Credit Gmv.1` | 3 | 0.21% |
| `% May Total GMV - Year25 consumption.1` | 3 | 0.21% |
| `JuneCredit Gmv` | 3 | 0.21% |
| `June Total GMV - Year25` | 3 | 0.21% |
| `% May Total GMV - Year25 consumption` | 3 | 0.21% |
| `July Credit Gmv` | 2 | 0.14% |
| `July Total GMV - Year24` | 2 | 0.14% |
| `Aug Total GMV - Year24` | 2 | 0.14% |
| `% Jul Total GMV - Year24 consumption` | 2 | 0.14% |
| `Aug Credit Gmv` | 2 | 0.14% |
| `% Aug Total GMV - Year24 consumption` | 2 | 0.14% |
| `Nov Total GMV - Year24` | 2 | 0.14% |
| `Nov Credit Gmv` | 2 | 0.14% |
| `Sep Total GMV - Year24` | 2 | 0.14% |
| `Sep Credit Gmv` | 2 | 0.14% |
| `% Sep Total GMV - Year24 consumption` | 2 | 0.14% |
| `Oct Total GMV - Year24` | 2 | 0.14% |
| `Oct Credit Gmv` | 2 | 0.14% |
| `% oct Total GMV - Year24 consumption` | 2 | 0.14% |
| `Jan Credit Gmv.1` | 2 | 0.14% |
| `Jan Total GMV - Year25` | 2 | 0.14% |
| `% Dec Total GMV - Year24 consumption` | 2 | 0.14% |
| `Dec Credit Gmv` | 2 | 0.14% |
| `Dec Total GMV - Year24` | 2 | 0.14% |
| `% Nov Total GMV - Year24 consumption` | 2 | 0.14% |
| `% Jan Total GMV - Year25 consumption` | 2 | 0.14% |
| `Feb Total GMV - Year25` | 2 | 0.14% |
| `% Feb Total GMV - Year25 consumption` | 2 | 0.14% |
| `% Mar Total GMV - Year25 consumption` | 2 | 0.14% |
| `Mar Credit Gmv.1` | 2 | 0.14% |
| `Mar Total GMV - Year25` | 2 | 0.14% |
| `Feb Credit Gmv.1` | 2 | 0.14% |
| `Apr Total GMV - Year25` | 1 | 0.07% |
| `Apr Credit Gmv.1` | 1 | 0.07% |
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
| `Jan Total GMV - Year24 Sales GMV` | 30879.10 | 0.00 | 3570.00 | 12552.00 | 33978.00 | 1240596.00 |
| `Jan Credit Gmv` | 19547.41 | 0.00 | 0.00 | 6369.00 | 19775.00 | 1145430.00 |
| `%Jan Consumption` | 0.65 | 0.00 | 0.28 | 0.84 | 1.00 | 1.00 |
| `Feb Total GMV - Year24` | 26091.32 | 0.00 | 2625.00 | 10316.00 | 27471.00 | 1003574.00 |
| `Feb Credit Gmv` | 16191.97 | 0.00 | 0.00 | 4410.00 | 16148.00 | 766874.00 |
| `%Feb consumtion` | 0.65 | 0.00 | 0.25 | 0.84 | 1.00 | 1.00 |
| `Mar Total GMV - Year24` | 25788.10 | 0.00 | 2583.00 | 9530.00 | 27289.00 | 885870.00 |
| `Mar Credit Gmv` | 16340.88 | 0.00 | 0.00 | 4616.00 | 16606.00 | 698236.00 |
| `%Mar Consumtion` | 0.65 | 0.00 | 0.29 | 0.85 | 1.00 | 1.00 |
| `Apr Total GMV - Year24` | 31284.12 | 0.00 | 3255.00 | 11806.00 | 34061.00 | 1016384.00 |
| `Apr Credit Gmv` | 18744.02 | 0.00 | 0.00 | 5723.00 | 19760.00 | 665711.00 |
| `%Apr Consumtion` | 0.65 | 0.00 | 0.27 | 0.85 | 1.00 | 1.00 |
| `May Total GMV - Year24` | 38758.89 | 0.00 | 4095.00 | 14860.00 | 42246.00 | 1570356.00 |
| `May Credit Gmv` | 22733.56 | 0.00 | 0.00 | 6610.00 | 24774.00 | 622632.00 |
| `%May Total GMV - Year24 Consumption` | 0.62 | 0.00 | 0.22 | 0.80 | 0.99 | 1.00 |
| `June Total GMV - Year24` | 28607.56 | 0.00 | 2520.00 | 10213.00 | 30596.00 | 1438355.00 |
| `June Credit Gmv` | 16209.35 | 0.00 | 0.00 | 4376.00 | 15841.00 | 788325.00 |
| `%Jun Total GMV - Year24 consumtion` | 0.53 | 0.00 | 0.00 | 0.66 | 0.97 | 1.00 |
| `July Total GMV - Year24` | 46034.86 | 0.00 | 1892.50 | 8150.00 | 23041.50 | 32546646.00 |
| `July Credit Gmv` | 26432.88 | 0.00 | 0.00 | 2977.50 | 11996.50 | 18688047.00 |
| `% Jul Total GMV - Year24 consumption` | 0.49 | 0.00 | 0.00 | 0.60 | 0.92 | 1.00 |
| `Aug Total GMV - Year24` | 50190.28 | 0.00 | 1787.50 | 8981.50 | 26467.25 | 35484526.00 |
| `Aug Credit Gmv` | 27708.17 | 0.00 | 0.00 | 2891.00 | 13169.50 | 19589673.00 |
| `% Aug Total GMV - Year24 consumption` | 0.48 | 0.00 | 0.00 | 0.50 | 0.94 | 1.00 |
| `Sep Total GMV - Year24` | 45555.83 | 0.00 | 1482.88 | 7689.98 | 23704.47 | 32207969.16 |
| `Sep Credit Gmv` | 23930.31 | 0.00 | 0.00 | 1566.00 | 11483.59 | 16918726.88 |
| `% Sep Total GMV - Year24 consumption` | 0.43 | 0.00 | 0.00 | 0.30 | 0.92 | 1.00 |
| `Oct Total GMV - Year24` | 53915.15 | 0.00 | 1952.93 | 9343.23 | 28835.25 | 38118008.28 |
| `Oct Credit Gmv` | 31052.98 | 0.00 | 0.00 | 2605.24 | 15951.67 | 21954459.73 |
| `% oct Total GMV - Year24 consumption` | 0.47 | 0.00 | 0.00 | 0.53 | 0.94 | 1.00 |
| `Nov Total GMV - Year24` | 59360.75 | 0.00 | 2100.00 | 11273.50 | 31859.50 | 41968050.00 |
| `Nov Credit Gmv` | 35782.42 | 0.00 | 0.00 | 3000.00 | 16514.30 | 25298169.84 |
| `% Nov Total GMV - Year24 consumption` | 0.47 | 0.00 | 0.00 | 0.52 | 0.94 | 1.00 |
| `Dec Total GMV - Year24` | 60947.60 | 0.00 | 2390.25 | 10365.50 | 34692.00 | 42784371.00 |
| `Dec Credit Gmv` | 34840.10 | 0.00 | 0.00 | 3214.00 | 16655.00 | 24412436.00 |
| `% Dec Total GMV - Year24 consumption` | 0.47 | 0.00 | 0.00 | 0.52 | 0.95 | 1.00 |
| `Jan Total GMV - Year25` | 60854.48 | 0.00 | 1978.50 | 11267.00 | 34307.00 | 42794778.00 |
| `Jan Credit Gmv.1` | 33482.89 | 0.00 | 0.00 | 2509.00 | 15626.50 | 23511180.00 |
| `% Jan Total GMV - Year25 consumption` | 0.46 | 0.00 | 0.00 | 0.47 | 0.95 | 1.00 |
| `Feb Total GMV - Year25` | 64812.91 | 0.00 | 1745.50 | 10420.50 | 36560.50 | 45628569.00 |
| `Feb Credit Gmv.1` | 35263.71 | 0.00 | 0.00 | 2199.00 | 15145.50 | 24857746.00 |
| `% Feb Total GMV - Year25 consumption` | 0.44 | 0.00 | 0.00 | 0.37 | 0.92 | 1.00 |
| `Mar Total GMV - Year25` | 24070.60 | 0.00 | 0.00 | 3629.00 | 14750.25 | 16906365.00 |
| `Mar Credit Gmv.1` | 13120.52 | 0.00 | 0.00 | 0.00 | 6770.00 | 9243809.00 |
| `% Mar Total GMV - Year25 consumption` | 0.40 | 0.00 | 0.00 | 0.00 | 0.93 | 1.00 |
| `Apr Total GMV - Year25` | 93724.98 | 0.00 | 1890.00 | 10810.00 | 33417.00 | 44373648.00 |
| `Apr Credit Gmv.1` | 46004.28 | 0.00 | 0.00 | 1340.00 | 15237.50 | 21772797.00 |
| `% Apr Total GMV - Year25 consumption` | 0.41 | 0.00 | 0.00 | 0.19 | 0.91 | 1.00 |
| `May Total GMV - Year25` | 18108.32 | 0.00 | 0.00 | 5707.00 | 19147.00 | 341265.00 |
| `May Credit Gmv.1` | 8715.07 | 0.00 | 0.00 | 0.00 | 7729.00 | 213413.00 |
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

