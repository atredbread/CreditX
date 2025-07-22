# Repayment Metrics Glossary

This document provides detailed explanations of each column in the repayment analysis report.

## Core Metrics

### agent_id
- **Description**: Unique identifier for each agent (business/merchant)
- **Type**: String
- **Example**: "A1001", "BZ20045"

### total_repayment_amount
- **Description**: Sum of all repayment amounts made by the agent
- **Type**: Currency (e.g., INR)
- **Calculation**: `SUM(repayment_amount)`
- **Importance**: Indicates total repayment capacity

### avg_repayment_amount
- **Description**: Average amount per repayment transaction
- **Type**: Currency (e.g., INR)
- **Calculation**: `total_repayment_amount / transaction_count`
- **Importance**: Shows typical transaction size

### std_repayment_amount
- **Description**: Standard deviation of repayment amounts
- **Type**: Currency (e.g., INR)
- **Importance**: Measures consistency in repayment amounts

### total_principal_repaid
- **Description**: Total principal amount repaid
- **Type**: Currency (e.g., INR)
- **Importance**: Shows actual debt reduction

### avg_principal_repaid
- **Description**: Average principal amount per transaction
- **Type**: Currency (e.g., INR)
- **Calculation**: `total_principal_repaid / transaction_count`

### transaction_count
- **Description**: Total number of repayment transactions
- **Type**: Integer
- **Importance**: Indicates transaction frequency

### first_transaction
- **Description**: Date of first repayment transaction
- **Type**: Date
- **Format**: YYYY-MM-DD

### last_transaction
- **Description**: Date of most recent repayment transaction
- **Type**: Date
- **Format**: YYYY-MM-DD

## Derived Metrics

### data_quality
- **Description**: Score (0-1) indicating data quality for this agent
- **Range**: 0 (poor) to 1 (excellent)
- **Factors**: Completeness, validity, consistency

### principal_ratio
- **Description**: Ratio of principal repaid to total repayment
- **Calculation**: `total_principal_repaid / total_repayment_amount`
- **Range**: 0 to 1
- **Interpretation**: Higher values indicate more repayment is going towards principal

### interest_paid
- **Description**: Total interest paid
- **Calculation**: `total_repayment_amount - total_principal_repaid`
- **Type**: Currency (e.g., INR)

### repayment_frequency
- **Description**: Average days between transactions
- **Calculation**: `(last_transaction - first_transaction) / transaction_count`
- **Unit**: Days

## Normalized Metrics (0-1 scale)

### *_norm columns
- **Description**: Normalized versions of metrics for scoring
- **Range**: 0 to 1 (min-max scaled)
- **Purpose**: Enables comparison across different scales

## Credit Health Metrics

### credit_health_score
- **Description**: Overall credit health score
- **Range**: 0-100
- **Components**:
  - 25%: Total repayment amount
  - 20%: Total principal repaid
  - 15%: Transaction count
  - 15%: Average repayment amount
  - 10%: Average principal repaid
  - 10%: Principal ratio
  - 5%: Data quality

### confidence
- **Description**: Confidence in the calculated score (0-1)
- **Range**: 0 (low) to 1 (high)
- **Factors**: Data quality, transaction history length

### risk_category
- **Description**: Risk classification based on score percentiles
- **Values**: 
  - Low Risk: Top 20% of scores
  - Medium Risk: Middle 60% of scores
  - High Risk: Bottom 20% of scores
- **Purpose**: Quick risk assessment
