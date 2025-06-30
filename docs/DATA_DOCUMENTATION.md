# Credit Risk Data Documentation

## credit_Agents.xlsx

- **Rows:** 1,915

### Columns:
- Bzid
- Phone
- Credit Line Setup Co
- Approval Amount
- Credit Limit
- Credit Line Balance
- Unnamed: 6

### Sample Data:
```
Bzid     | Phone      | Credit Line Setup Co | Approval Amount | Credit Limit | Credit Line Balance | Unnamed: 6
----------------------------------------------------------------------------------------------------------------
23058821 | 9966119584 | 19-5-2023, 9:47 AM   | 10000           | 10000        | 10000.0             | D         
21425838 | 8722348686 | 18-5-2023, 5:29 PM   | 37000           | 37000        | 37000.0             | D         
26840517 | 7411103360 | 19-5-2023, 1:54 PM   | 166000          | 166000       | 140576.14           | D         
```

## Credit_history_sales_vs_credit_sales.xlsx

- **Rows:** 1,416

### Columns:
- Account
- Jan Total GMV - Year25
- Jan Credit Gmv
- % Jan Total GMV - Year25 consumption
- Feb Total GMV - Year25
- Feb Credit Gmv
- % Feb Total GMV - Year25 consumption
- Mar Total GMV - Year25
- Mar Credit Gmv
- % Mar Total GMV - Year25 consumption
- Apr Total GMV - Year25
- Apr Credit Gmv
- % Apr Total GMV - Year25 consumption
- May Total GMV - Year25
- May Credit Gmv
- % May Total GMV - Year25 consumption
- June Total GMV - Year25
- JuneCredit Gmv
- % May Total GMV - Year25 consumption.1

### Sample Data:
```
Account  | Jan Total GMV - Year25 | Jan Credit Gmv | % Jan Total GMV - Year25 consumption | Feb Total GMV - Year25 | Feb Credit Gmv | % Feb Total GMV - Year25 consumption | Mar Total GMV - Year25 | Mar Credit Gmv | % Mar Total GMV - Year25 consumption | Apr Total GMV - Year25 | Apr Credit Gmv | % Apr Total GMV - Year25 consumption | May Total GMV - Year25 | May Credit Gmv | % May Total GMV - Year25 consumption | June Total GMV - Year25 | JuneCredit Gmv | % May Total GMV - Year25 consumption.1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
25125208 | 458674                 | 322444         | 0.703                                | 388310                 | 147399         | 0.3796                               | 223093                 | 64794          | 0.2904                               | 507822                 | 222337         | 0.4378                               | 341265                 | 121208         | 0.3552                               | 324456.6                | 118592.2       | 0.3655                                
28499610 | 226523                 | 0              | 0.0                                  | 228508                 | 0              | 0.0                                  | 89740                  | 0              | 0.0                                  | 520917                 | 0              | 0.0                                  | 249222                 | 0              | 0.0                                  | 63996.15                | 0.0            | 0.0                                   
13810786 | 319249                 | 0              | 0.0                                  | 236825                 | 0              | 0.0                                  | 108210                 | 0              | 0.0                                  | 351687                 | 0              | 0.0                                  | 239615                 | 0              | 0.0                                  | 98008.61                | 0.0            | 0.0                                   
```

## Credit_sales_data.xlsx

- **Rows:** 6,226

### Columns:
- DATE
- account
- GMV
- tin

### Sample Data:
```
DATE                | account  | GMV     | tin     
---------------------------------------------------
2025-06-01 00:00:00 | 28115985 | 5800.0  | 7AB67HK6
2025-06-01 00:00:00 | 28372265 | 1699.95 | 7A46MMDV
2025-06-01 00:00:00 | 27602131 | 1020.0  | 7AHH4PNE
```

## DPD.xlsx

- **Rows:** 91

### Columns:
- Anchor
- Phone
- Bzid
- Username
- Business Name
- Dpd
- Pos

### Sample Data:
```
Anchor | Phone      | Bzid     | Username          | Business Name                 | Dpd | Pos     
---------------------------------------------------------------------------------------------------
REDBUS | 9885777379 | 24939241 | RAHAMTHULLA SHAIK | SRT travels - Rayachoti       | 3   | 5379.3  
REDBUS | 9944190111 | 13910540 | MANICKAM RAJESH   | TRAVEL ZONE TOURS AND TRAVELS | 3   | 26004.09
REDBUS | 9820762252 | 45966733 | SHABBIR M KOTHARI | KOTHARI SALES CORPORATION     | 2   | 5778.38 
```

## Region_contact.xlsx

- **Rows:** 9

### Columns:
- Region
- Manager
- Name

### Sample Data:
```
Region              | Manager                      | Name 
----------------------------------------------------------
MP                  | Aswinsatheesh.work@gamil.com | Aswin
Tamil Nadu & Kerala | Aswinsatheesh.work@gamil.com | Aswin
Gujarat             | Aswinsatheesh.work@gamil.com | Aswin
```

## repayment report.csv

- **Rows:** 349,598

### Columns:
- All Anchors Onboarding Info - Anchor → Bzid
- Customer Repayment Date
- Customer Repayment Amount
- Customer Principle Repaid

### Sample Data:
```
All Anchors Onboarding Info - Anchor → Bzid | Customer Repayment Date  | Customer Repayment Amount | Customer Principle Repaid
------------------------------------------------------------------------------------------------------------------------------
42405740                                    | April 27, 2023, 2:03 PM  | 1,260.76                  | 1,260                    
25848971                                    | April 28, 2023, 12:39 PM | 12,607.56                 | 12,600                   
25848971                                    | April 28, 2023, 12:39 PM | 1,154.65                  | 1,153.95                 
```

## sales_data.xlsx

- **Rows:** 13,913

### Columns:
- DATE(a.creationtime)
- account
- organizationname
- status
- TotalSeats
- GMV
- AgentCommission(Exe GDS)
- city
- State
- Region
- Check
- Ro Name
- RM Name

### Sample Data:
```
DATE(a.creationtime) | account  | organizationname                       | status | TotalSeats | GMV     | AgentCommission(Exe GDS) | city     | State     | Region    | Check | Ro Name                 | RM Name 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2025-06-01 00:00:00  | 28115985 | National Travels Belgaum               | BOOKED | 4          | 5800.0  | 580.0                    | Belagavi | Karnataka | Karnataka | False | Sohail fazal            | Shivaraj
2025-06-01 00:00:00  | 20692257 | Shree Tours & Travels Karad 9021250999 | BOOKED | 1          | 2280.0  | 0.0                      | Karad    | Pune      | MH+Goa    | False | Bajirao  yewale         | nan     
2025-06-01 00:00:00  | 28372265 | Prisha Tours and Travels               | BOOKED | 1          | 1699.95 | 80.95                    | Mumbai   | Mumbai    | MH+Goa    | False | Rajkumarreddy Ranjolkar | nan     
```
