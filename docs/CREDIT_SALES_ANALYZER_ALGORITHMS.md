   # Credit Sales Analyzer - Algorithms and Calculations

   ## Table of Contents
   1. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   2. [Monthly Metrics Calculation](#monthly-metrics-calculation)
   3. [Aggregate Metrics Calculation](#aggregate-metrics-calculation)
   4. [Credit Health Score](#credit-health-score)
   5. [Risk Indicators](#risk-indicators)

## Data Loading and Preprocessing

### File Structure
- **Input File**: `Credit_history_sales_vs_credit_sales.xlsx`
- **Key Columns**:
  - `Account`: Unique agent identifier
  - `[Month] Total GMV - Year25`: Total Gross Merchandise Value for the month
  - `[Month] Credit Gmv`: Credit-based GMV for the month
  - `% [Month] Total GMV - Year25 consumption`: Pre-calculated consumption percentage

### Preprocessing Steps
1. **Data Loading**:
   - Load Excel file using pandas
   - Basic validation for missing or empty data

2. **Column Standardization**:
   - Handle inconsistent column naming (e.g., 'JuneCredit Gmv' vs 'Jan Credit Gmv')
   - Convert all column names to consistent format

3. **Data Cleaning**:
   - Replace zeros with NaN for GMV calculations to avoid division by zero
   - Handle missing values appropriately

## Monthly Metrics Calculation

### GMV and Credit GMV
For each month (Jan-Jun):
```python
month_gmv = df['[Month] Total GMV - Year25']
month_credit = df['[Month] Credit Gmv']
```

### Credit Ratio
```python
month_ratio = month_credit / month_gmv  # For month_gmv > 0
```

## Aggregate Metrics Calculation

### Total GMV and Credit GMV
```python
total_gmv = sum(month_gmv for all months)
total_credit = sum(month_credit for all months)
overall_ratio = total_credit / total_gmv if total_gmv > 0 else 0
```

### Monthly Statistics
```python
# For all months with valid data (month_gmv > 0)
avg_credit_ratio = mean(monthly_ratios)
max_credit_ratio = max(monthly_ratios)
min_credit_ratio = min(monthly_ratios)
credit_ratio_std = std(monthly_ratios, ddof=1)  # Sample standard deviation
```

## Credit Health Score

### Score Components
1. **Credit Ratio Score** (40% weight):
   ```
   score_credit_ratio = 1 - min(1, avg_monthly_credit_ratio)
   ```
   - Lower credit utilization is better
   - Capped at 1 (100% utilization)

2. **Volatility Score** (30% weight):
   ```
   score_volatility = 1 - min(1, credit_ratio_std)
   ```
   - Lower volatility is better
   - Capped at 1 (100% standard deviation)

3. **Business Volume Score** (30% weight):
   ```
   score_gmv = (total_gmv - min_gmv) / (max_gmv - min_gmv + ε)
   ```
   - Normalized between 0 and 1
   - ε (epsilon) is a small constant to prevent division by zero

### Final Score
```python
credit_health_score = (
   score_credit_ratio * 0.4 +
   score_volatility * 0.3 +
   score_gmv * 0.3
) * 100  # Scale to 0-100
```

## Risk Indicators

### High Credit Dependence
```python
high_credit_dependence = avg_monthly_credit_ratio > CREDIT_THRESHOLD_HIGH  # 0.5 (50%)
```

### Low Credit Utilization
```python
low_credit_utilization = avg_monthly_credit_ratio < CREDIT_THRESHOLD_LOW  # 0.3 (30%)
```

### Dormant Agent
```python
dormant_agent = zero_credit_months >= DORMANT_MONTHS  # 3 months
```

## Agent Clustering

### Features Used
1. Average monthly credit ratio
2. Credit ratio standard deviation
3. Total GMV
4. Total Credit GMV
5. Number of high credit months

### Clustering Algorithm
1. **Feature Scaling**:
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **K-means Clustering**:
   ```python
   kmeans = KMeans(n_clusters=4, random_state=42)
   cluster_labels = kmeans.fit_predict(X_scaled)
   ```

## Trend Analysis

### Credit Trend
1. Calculate monthly credit ratios
2. Perform linear regression on the ratios
3. Classify trend based on slope:
   - Increasing: slope > 0.01
   - Decreasing: slope < -0.01
   - Stable: otherwise

### Peer Comparison
```python
peer_avg = metrics_df.agg({
   'avg_monthly_credit_ratio': 'mean',
   'total_gmv': 'mean',
   'total_credit_gmv': 'mean',
   'credit_health_score': 'mean'
})
```

## Visualization

### Credit Ratio Distribution
- Histogram of average monthly credit ratios
- Markers for high (50%) and low (30%) thresholds

### Trend Analysis
- Bar chart showing distribution of trends (Increasing/Decreasing/Stable)
- Time series plot for individual agent credit ratios

## Error Handling
1. **Missing Data**: Skip months with missing or invalid data
2. **Division by Zero**: Handle cases where GMV is zero
3. **Input Validation**: Validate agent IDs and menu choices
4. **File Operations**: Handle missing or corrupted input files
