import pandas as pd
import os
import argparse
from typing import Optional, Dict, List, Any
from pathlib import Path

def get_monthly_columns(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Extract and organize monthly credit data columns."""
    month_data = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    for month in months:
        # Handle June's different column naming
        month_col = 'June' if month == 'Jun' else month
        
        # Find matching columns for this month
        gmv_col = next((c for c in df.columns if f"{month_col} Total GMV" in c), None)
        credit_col = next((c for c in df.columns if f"{month_col} Credit" in c), None)
        ratio_col = next((c for c in df.columns if f"{month} " in c and "%" in c and "consumption" in c), None)
        
        if gmv_col and credit_col and ratio_col:
            month_data.append({
                'month': month,
                'month_display': month_col,
                'gmv_col': gmv_col,
                'credit_col': credit_col,
                'ratio_col': ratio_col
            })
    
    return month_data

def display_agent_data(df: pd.DataFrame, agent_id: str, month_data: List[Dict[str, str]]):
    """Display credit data for a specific agent."""
    # Convert agent_id to string and strip any whitespace
    agent_id = str(agent_id).strip()
    
    # Find the agent's data
    agent_data = df[df['Account'].astype(str).str.strip() == agent_id]
    
    if agent_data.empty:
        print(f"\n❌ No data found for Agent ID: {agent_id}")
        return False
    
    # Get the first matching row (should be only one per agent)
    row = agent_data.iloc[0]
    
    print("\n" + "="*80)
    print(f"CREDIT DATA FOR AGENT: {agent_id}")
    print("="*80)
    
    # Display agent information
    print(f"\n{'Metric':<20} | {'Value':>15}")
    print("-" * 40)
    
    # Display monthly data
    total_gmv = 0
    total_credit = 0
    
    for month_info in month_data:
        gmv = float(row.get(month_info['gmv_col'], 0))
        credit = float(row.get(month_info['credit_col'], 0))
        ratio = (credit / gmv * 100) if gmv != 0 else 0
        
        total_gmv += gmv
        total_credit += credit
        
        print(f"\n{month_info['month_display']}:")
        print("-" * 40)
        print(f"  {'Total GMV:':<16} {gmv:15,.2f}")
        print(f"  {'Credit GMV:':<16} {credit:15,.2f}")
        print(f"  {'Credit Ratio:':<16} {ratio:14.2f}%")
    
    # Calculate and display totals
    avg_ratio = (total_credit / total_gmv * 100) if total_gmv != 0 else 0
    
    print("\n" + "="*40)
    print("TOTALS:")
    print("-" * 40)
    print(f"  {'Total GMV:':<16} {total_gmv:15,.2f}")
    print(f"  {'Total Credit GMV:':<16} {total_credit:15,.2f}")
    print(f"  {'Avg Credit Ratio:':<16} {avg_ratio:14.2f}%")
    
    return True

def display_sample_data(df: pd.DataFrame, month_data: List[Dict[str, str]], sample_size: int = 3):
    """Display sample data for the first few agents."""
    print("\n" + "="*80)
    print(f"SAMPLE DATA (First {sample_size} Agents):")
    print("="*80)
    
    for idx, row in df.head(sample_size).iterrows():
        agent_id = row['Account']
        print(f"\nAGENT ID: {agent_id}")
        print("-"*60)
        
        # Display monthly data
        total_gmv = 0
        total_credit = 0
        
        for month_info in month_data:
            gmv = float(row.get(month_info['gmv_col'], 0))
            credit = float(row.get(month_info['credit_col'], 0))
            ratio = (credit / gmv * 100) if gmv != 0 else 0
            
            total_gmv += gmv
            total_credit += credit
            
            print(f"{month_info['month_display']}:")
            print(f"  {'Total GMV:':<16} {gmv:15,.2f}")
            print(f"  {'Credit GMV:':<16} {credit:15,.2f}")
            print(f"  {'Credit Ratio:':<16} {ratio:14.2f}%")
            print("-" * 60)
        
        # Calculate and print totals
        avg_ratio = (total_credit / total_gmv * 100) if total_gmv != 0 else 0
        
        print("=" * 40)
        print("TOTALS:")
        print("-" * 40)
        print(f"  {'Total GMV:':<16} {total_gmv:15,.2f}")
        print(f"  {'Total Credit GMV:':<16} {total_credit:15,.2f}")
        print(f"  {'Avg Credit Ratio:':<16} {avg_ratio:14.2f}%")

def display_credit_data(file_path: str, agent_id: Optional[str] = None):
    """Display credit history data in a clean, readable format."""
    try:
        # Read the Excel file
        print(f"\n📂 Loading data from: {file_path}")
        
        # Read the Excel file without any type conversion for Account column
        df = pd.read_excel(file_path, dtype={'Account': str})
        
        # Clean and standardize the Account column
        if 'Account' in df.columns:
            # First convert to string, then clean (handles both int and float)
            df['Account'] = df['Account'].apply(clean_agent_id)
            # Remove any empty strings that might have resulted from cleaning
            df = df[df['Account'] != '']
        else:
            print("\n❌ Error: 'Account' column not found in the data")
            print("\nAvailable columns:", df.columns.tolist())
            return
        
        # Display basic info
        print("="*80)
        print(f"CREDIT HISTORY DATA: {os.path.basename(file_path)}")
        print("="*80)
        print(f"Total Agents: {len(df)}")
        
        # Get monthly column mappings
        month_data = get_monthly_columns(df)
        
        if not month_data:
            print("\n❌ Could not identify monthly data columns.")
            return
        
        # If agent_id is provided, show only that agent's data
        if agent_id:
            display_agent_data(df, agent_id, month_data)
        else:
            # Otherwise show sample data
            display_sample_data(df, month_data)
    
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {file_path}")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        if 'df' in locals():
            print("\nAvailable columns in the file:")
            for col in df.columns:
                print(f"- {col}")

def clean_agent_id(agent_id) -> str:
    """Convert agent ID to clean string representation without decimal .0"""
    if pd.isna(agent_id):
        return ""
    # Convert to string, remove .0 if it's a whole number float
    s = str(agent_id).strip()
    if s.endswith('.0'):
        return s[:-2]
    return s

def get_sample_agent_ids(df: pd.DataFrame, count: int = 5) -> List[str]:
    """Get sample agent IDs from the data."""
    try:
        return [clean_agent_id(agent_id) for agent_id in df['Account'].drop_duplicates().head(count).tolist()]
    except Exception:
        return []

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Display credit data for agents')
    parser.add_argument('--file', type=str, 
                       default="source_data/Credit_history_sales_vs_credit_sales.xlsx",
                       help='Path to the credit data Excel file')
    parser.add_argument('--agent', type=str, 
                       help='Agent ID to display data for (can be with or without .0)')
    parser.add_argument('--list-agents', action='store_true',
                       help='List all available agent IDs')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File not found: {args.file}")
        print("Please provide a valid file path using --file")
        sys.exit(1)
    
    try:
        if args.list_agents:
            # Load just the Account column to list all agent IDs
            df = pd.read_excel(args.file, usecols=['Account'], dtype={'Account': str})
            df['Account'] = df['Account'].apply(clean_agent_id)
            agents = df['Account'].drop_duplicates().sort_values()
            print("\n".join(agents.tolist()))
        else:
            display_credit_data(args.file, args.agent)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
