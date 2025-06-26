"""
Script to validate and display the last 6 months of GMV and credit GMV data for the first 5 agents.
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for module imports
sys.path.append(str(Path(__file__).parent))

def load_gmv_data():
    """Load and preprocess the GMV data with correct credit GMV mapping."""
    try:
        file_path = 'source_data/Credit_history_sales_vs_credit_sales.xlsx'
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Ensure Account is string and clean it
        df['Account'] = df['Account'].astype(str).str.strip()
        
        return df
    except Exception as e:
        print(f"Error loading GMV data: {e}")
        return None

def get_month_columns(columns):
    """
    Identify month columns and their corresponding credit GMV columns.
    Returns a list of dicts with month info and column mappings.
    """
    month_data = []
    
    # First, identify all potential month columns with their types
    month_cols = []
    for i, col in enumerate(columns):
        if not isinstance(col, str):
            continue
            
        col_lower = col.lower()
        
        # Skip percentage columns and other non-relevant columns
        if any(x in col_lower for x in ['%', 'consumpt', 'ratio', 'share']):
            continue
            
        # Check for month patterns
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            # Handle both Year24 and Year25 formats
            for year_suffix in ['Year24', 'Year25']:
                year = 2000 + int(year_suffix[-2:])  # Convert YY to YYYY
                
                # Look for total GMV columns
                if (month.lower() in col_lower and 
                    'total gmv' in col_lower and 
                    year_suffix.lower() in col_lower):
                    
                    month_cols.append({
                        'index': i,
                        'col_name': col,
                        'month': month,
                        'year': year,
                        'month_num': datetime.strptime(month, '%b').month,
                        'type': 'total_gmv'
                    })
                
                # Look for credit GMV columns (they might not have the year in the name)
                elif (month.lower() in col_lower and 
                      'credit' in col_lower and 
                      'gmv' in col_lower and
                      not any(x in col_lower for x in ['%', 'consumpt', 'ratio', 'share'])):
                    
                    # Try to find the most recent matching credit column
                    month_cols.append({
                        'index': i,
                        'col_name': col,
                        'month': month,
                        'year': year,  # This will be updated when we pair with total columns
                        'month_num': datetime.strptime(month, '%b').month,
                        'type': 'credit_gmv'
                    })
    
    # Sort all columns by year, month, and index
    month_cols.sort(key=lambda x: (x['year'], x['month_num'], x['index']))
    
    # Pair total_gmv with the next credit_gmv column
    i = 0
    while i < len(month_cols):
        if month_cols[i]['type'] == 'total_gmv' and i+1 < len(month_cols) and month_cols[i+1]['type'] == 'credit_gmv':
            # Found a pair
            total_col = month_cols[i]
            credit_col = month_cols[i+1]
            
            # Only add if they're for the same month and year
            if (total_col['month'] == credit_col['month'] and 
                total_col['year'] == credit_col['year']):
                
                month_data.append({
                    'month_name': f"{total_col['month']} {total_col['year']}",
                    'month_num': total_col['month_num'],
                    'year': total_col['year'],
                    'total_col': total_col['col_name'],
                    'credit_col': credit_col['col_name']
                })
                i += 2  # Skip the credit column in next iteration
                continue
        i += 1
    
    # Sort by year and month (descending, newest first)
    month_data.sort(key=lambda x: (x['year'], x['month_num']), reverse=True)
    
    # Get the most recent 6 months
    return month_data[:6] if len(month_data) >= 6 else month_data

def format_currency(amount):
    """Format number as currency with 2 decimal places and commas."""
    if pd.isna(amount):
        return "N/A"
    try:
        return f"{float(amount):,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def display_agent_gmv_data(agent_id, agent_data, month_columns):
    """Display GMV data for a single agent with correct credit GMV mapping."""
    print(f"\nAgent ID: {agent_id}")
    print("-" * 60)
    print(f"{'Month/Year':<15} | {'Total GMV':>15} | {'Credit GMV':>15} | {'Credit %':>10}")
    print("-" * 60)
    
    for month_info in month_columns:
        month_name = month_info['month_name']
        total_col = month_info['total_col']
        credit_col = month_info['credit_col']
        
        total_gmv = agent_data.get(total_col, pd.NA)
        credit_gmv = agent_data.get(credit_col, pd.NA)
        
        # Clean and convert values
        try:
            total_gmv = float(total_gmv) if pd.notna(total_gmv) else 0
            credit_gmv = float(credit_gmv) if pd.notna(credit_gmv) else 0
            
            # Calculate credit percentage
            credit_pct = (credit_gmv / total_gmv * 100) if total_gmv != 0 else 0
            credit_pct_str = f"{credit_pct:.1f}%"
            
            # Format values for display
            total_display = f"{total_gmv:,.2f}" if total_gmv > 0 else "0.00"
            credit_display = f"{credit_gmv:,.2f}" if credit_gmv > 0 else "0.00"
            
        except (ValueError, TypeError) as e:
            total_display = "N/A"
            credit_display = "N/A"
            credit_pct_str = "N/A"
        
        print(f"{month_name:<15} | {total_display:>15} | {credit_display:>15} | {credit_pct_str:>10}")

def main():
    """Main function to display GMV data for the first 5 agents."""
    print("Loading GMV data...")
    gmv_data = load_gmv_data()
    
    if gmv_data is None or gmv_data.empty:
        print("No GMV data found or error loading data.")
        return
    
    # Get month columns with their corresponding credit columns
    month_columns = get_month_columns(gmv_data.columns)
    
    if not month_columns:
        print("Error: Could not identify month columns in the data.")
        print("Available columns:", list(gmv_data.columns))
        return
    
    # Get first 5 unique agent IDs
    agent_ids = gmv_data['Account'].drop_duplicates().head(5).tolist()
    
    print("\n" + "="*80)
    print(f"GMV VALIDATION REPORT - Last 6 Months (as of {datetime.now().strftime('%Y-%m-%d')})")
    print("="*80)
    print("Note: Credit GMV is taken from the column immediately following each Total GMV column")
    
    # Print column mapping for verification
    print("\nColumn Mapping (Most Recent First):")
    for i, month in enumerate(month_columns, 1):
        print(f"{i}. {month['month_name']}: {month['total_col']} -> {month['credit_col']}")
    
    # Show date range
    if month_columns:
        first_month = month_columns[-1]['month_name']
        last_month = month_columns[0]['month_name']
        print(f"\nDate Range: {first_month} to {last_month}")
    
    for agent_id in agent_ids:
        agent_data = gmv_data[gmv_data['Account'] == agent_id].iloc[0]
        display_agent_gmv_data(agent_id, agent_data, month_columns)
    
    print("\n" + "="*80)
    print("End of report")

if __name__ == "__main__":
    main()
