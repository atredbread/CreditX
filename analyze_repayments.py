"""
Analyze repayment report data to assess agent credit health.

# 📊 Agent Credit Health Scoring System

## How the Score is Calculated

1. **Normalization (Min-Max Scaling):**
   - Each metric is normalized to a 0–1 scale using the formula:
     X_norm = (X - X_min) / (X_max - X_min)
   - This ensures all metrics are comparable regardless of their original scale.

2. **Metrics and Weights:**
   - Total repayment amount: 25%
   - Total principal repaid: 20%
   - Transaction count: 15%
   - Average repayment per transaction: 15%
   - Average principal per transaction: 10%
   - Principal-to-total repayment ratio: 15%

3. **Weighted Score Calculation:**
   - The final score for each agent is computed as:
     Score = 0.25 * Total Repayment_norm + 0.20 * Total Principal_norm + 0.15 * Txn Count_norm + 0.15 * Avg Repay_norm + 0.10 * Avg Principal_norm + 0.15 * Principal Ratio_norm
   - Higher scores indicate better repayment behavior and credit health.

---

This script analyzes repayment data, calculates all metrics above, and assigns a score to each agent based on the weighted system above.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_currency(value) -> float:
    """Convert currency strings to float, handling commas and empty values."""
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(',', '').strip())

def load_repayment_data(filepath: str) -> pd.DataFrame:
    """Load and clean the repayment report data."""
    logger.info(f"Loading repayment data from {filepath}")
    
    # Read the CSV file
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Clean and convert numeric columns
    df['Customer Repayment Amount'] = df['Customer Repayment Amount'].apply(clean_currency)
    df['Customer Principle Repaid'] = df['Customer Principle Repaid'].apply(clean_currency)
    
    # Convert date column to datetime
    df['Repayment Date'] = pd.to_datetime(
        df['Customer Repayment Date'], 
        errors='coerce',
        format='mixed'
    )
    
    # Calculate interest paid
    df['Interest Paid'] = df['Customer Repayment Amount'] - df['Customer Principle Repaid']
    
    return df

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate credit health metrics for each agent."""
    logger.info("Calculating repayment metrics...")
    
    # Group by agent and calculate metrics
    metrics = df.groupby('Bzid').agg({
        'Customer Repayment Amount': ['sum', 'count', 'mean'],
        'Customer Principle Repaid': 'sum',
        'Interest Paid': 'sum'
    })
    
    # Flatten multi-index columns
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    
    # Rename columns for clarity
    metrics = metrics.rename(columns={
        'Customer Repayment Amount_sum': 'Total Repaid',
        'Customer Repayment Amount_count': 'Transaction Count',
        'Customer Repayment Amount_mean': 'Avg Repayment',
        'Customer Principle Repaid_sum': 'Total Principal',
        'Interest Paid_sum': 'Total Interest'
    })
    
    # Calculate additional metrics
    metrics['Principal Ratio'] = (metrics['Total Principal'] / metrics['Total Repaid'] * 100).round(2)
    metrics['Avg Principal'] = (metrics['Total Principal'] / metrics['Transaction Count']).round(2)
    
    # Format currency columns
    for col in ['Total Repaid', 'Total Principal', 'Total Interest', 'Avg Repayment', 'Avg Principal']:
        metrics[col] = metrics[col].round(2)
    
    # Sort by total repaid (descending)
    metrics = metrics.sort_values('Total Repaid', ascending=False)
    
    return metrics

def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Format the metrics for better readability."""
    # Create a copy to avoid SettingWithCopyWarning
    formatted = metrics.copy()
    
    # Format large numbers with commas
    for col in ['Total Repaid', 'Total Principal', 'Total Interest']:
        formatted[col] = formatted[col].apply(lambda x: f'₹{x:,.2f}')
    
    # Format average values
    for col in ['Avg Repayment', 'Avg Principal']:
        formatted[col] = formatted[col].apply(lambda x: f'₹{x:,.2f}')
    
    # Format ratio as percentage
    formatted['Principal Ratio'] = formatted['Principal Ratio'].apply(lambda x: f'{x:.2f}%')
    
    return formatted

def analyze_repayments(input_file: str = 'source_data/repayment report.csv') -> pd.DataFrame:
    """
    Analyze repayment data and calculate credit health metrics.
    
    Args:
        input_file: Path to the repayment report CSV file
        
    Returns:
        DataFrame containing repayment metrics and scores for each agent
    """
    try:
        # Load data
        df = pd.read_csv(input_file, encoding='utf-8')
        df.columns = [c.strip() for c in df.columns]

        # Rename columns for clarity
        df.columns = ['bzid', 'repayment_date', 'repayment_amount', 'principal_repaid']

        # Remove commas and convert monetary columns to numeric
        df['repayment_amount'] = df['repayment_amount'].replace(r'[\$,]', '', regex=True).astype(float)
        df['principal_repaid'] = df['principal_repaid'].replace(r'[\$,]', '', regex=True).astype(float)

        # Convert repayment_date to datetime using explicit format
        df['repayment_date'] = pd.to_datetime(df['repayment_date'], format="%B %d, %Y, %I:%M %p")

        # Group by agent (bzid) to compute credit health metrics
        grouped = df.groupby('bzid').agg(
            total_repayment_amount=('repayment_amount', 'sum'),
            total_principal_repaid=('principal_repaid', 'sum'),
            transaction_count=('repayment_amount', 'count'),
            avg_repayment_per_txn=('repayment_amount', 'mean'),
            avg_principal_per_txn=('principal_repaid', 'mean'),
        )

        # Add principal-to-repayment ratio
        grouped['principal_ratio'] = grouped['total_principal_repaid'] / grouped['total_repayment_amount']

        # Min-Max normalization for each metric
        def minmax(col):
            return (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else 1.0
        norm = grouped.apply(minmax)

        # Weighted score calculation
        grouped['score'] = (
            0.25 * norm['total_repayment_amount'] +
            0.20 * norm['total_principal_repaid'] +
            0.15 * norm['transaction_count'] +
            0.15 * norm['avg_repayment_per_txn'] +
            0.10 * norm['avg_principal_per_txn'] +
            0.15 * norm['principal_ratio']
        ).round(4)

        # Sort by score for best agents
        return grouped.sort_values(by='score', ascending=False)

    except Exception as e:
        logger.error(f"Error analyzing repayments: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point for command-line usage."""
    try:
        # File path
        input_file = Path('source_data/repayment report.csv')
        
        # Analyze repayments
        result = analyze_repayments(input_file)
        
        # Display the results
        pd.set_option('display.max_rows', 30)
        pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
        print("\nAgent Repayment Metrics with Score (sorted by score):")
        print(result)
        
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    main()
