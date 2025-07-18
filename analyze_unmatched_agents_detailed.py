"""
Detailed analysis of unmatched agents to understand data patterns.
"""
import pandas as pd
import os
from datetime import datetime

def load_agent_data():
    """Load agent data with proper column handling."""
    print("Loading agent data...")
    agents = pd.read_excel('source_data/credit_Agents.xlsx')
    agents['BZID'] = agents['Bzid'].astype(str).str.strip()
    
    # Clean and parse date column if it exists
    if 'Credit Line Setup Co' in agents.columns:
        # First, try to parse with dayfirst=True for DD/MM/YYYY format
        agents['Setup_Date'] = pd.to_datetime(
            agents['Credit Line Setup Co'], 
            dayfirst=True,  # Try DD/MM/YYYY format first
            errors='coerce'
        )
        
        # If parsing failed, try other common formats
        if agents['Setup_Date'].isna().any():
            # Try MM/DD/YYYY format
            mask = agents['Setup_Date'].isna()
            agents.loc[mask, 'Setup_Date'] = pd.to_datetime(
                agents.loc[mask, 'Credit Line Setup Co'],
                dayfirst=False,  # Try MM/DD/YYYY format
                errors='coerce'
            )
        
        # Log parsing results
        parsed_count = agents['Setup_Date'].notna().sum()
        print(f"Successfully parsed {parsed_count} of {len(agents)} dates ({parsed_count/len(agents):.1%})")
    
    return agents

def analyze_credit_limits(agents):
    """Analyze credit limit distribution."""
    print("\n=== Credit Limit Analysis ===")
    
    if 'Credit Limit' not in agents.columns:
        print("Credit Limit column not found in agent data")
        return
    
    # Define credit limit ranges
    bins = [0, 10000, 50000, 100000, float('inf')]
    labels = ['<10K', '10K-50K', '50K-100K', '>100K']
    
    agents['Credit_Limit_Range'] = pd.cut(
        agents['Credit Limit'],
        bins=bins,
        labels=labels,
        right=False
    )
    
    # Calculate distribution
    dist = agents['Credit_Limit_Range'].value_counts().sort_index()
    print("\nCredit Limit Distribution:")
    print(dist)
    
    return agents

def analyze_setup_dates(agents, sales_data=None):
    """Analyze agent setup dates with improved date handling and sales correlation."""
    print("\n=== Setup Date Analysis ===")
    
    if 'Setup_Date' not in agents.columns:
        print("Setup date information not available")
        return agents
    
    # Ensure we have valid datetime objects
    agents = agents.dropna(subset=['Setup_Date']).copy()
    
    if agents.empty:
        print("No valid setup dates found")
        return agents
    
    # Get the most recent date to calculate relative ages
    latest_date = agents['Setup_Date'].max()
    print(f"\nDate Range:")
    print(f"- Most recent setup date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"- Earliest setup date: {agents['Setup_Date'].min().strftime('%Y-%m-%d')}")
    print(f"- Total time span: {(latest_date - agents['Setup_Date'].min()).days} days")
    
    # Calculate days since setup
    agents['Days_Since_Setup'] = (latest_date - agents['Setup_Date']).dt.days
    
    # Define time periods for analysis
    time_periods = [
        (0, 7, 'Last 7 days'),
        (8, 30, '8-30 days'),
        (31, 90, '1-3 months'),
        (91, 180, '3-6 months'),
        (181, 365, '6-12 months'),
        (366, float('inf'), 'Over 1 year')
    ]
    
    # Add a column for age group
    agents['Age_Group'] = 'Other'
    for min_days, max_days, label in time_periods:
        if max_days == float('inf'):
            mask = agents['Days_Since_Setup'] >= min_days
        else:
            mask = (agents['Days_Since_Setup'] >= min_days) & (agents['Days_Since_Setup'] <= max_days)
        agents.loc[mask, 'Age_Group'] = label
    
    # Print distribution by time periods
    print("\nAgents by Time Since Setup:")
    print("=" * 40)
    
    total_agents = len(agents)
    for _, _, label in time_periods:
        count = (agents['Age_Group'] == label).sum()
        if total_agents > 0:
            percentage = (count / total_agents) * 100
            print(f"{label}: {count:>4} agents ({percentage:5.1f}%)")
    
    # Additional analysis: Setup date by year/month
    print("\nAgents by Setup Year/Month (Top 15):")
    print("=" * 40)
    agents['YearMonth'] = agents['Setup_Date'].dt.to_period('M')
    monthly_counts = agents['YearMonth'].value_counts().sort_index()
    
    # Convert Period to string for better display
    monthly_counts.index = monthly_counts.index.astype(str)
    
    # Print in a formatted table
    print("Date      | Count  | % of Total")
    print("-" * 30)
    for date_str, count in monthly_counts.head(15).items():
        percentage = (count / total_agents) * 100
        print(f"{date_str:<9} | {count:6} | {percentage:5.1f}%")
    
    if len(monthly_counts) > 15:
        print(f"... and {len(monthly_counts) - 15} more months")
    
    # Correlation with sales data if available
    if sales_data is not None and 'BZID' in agents.columns and 'BZID' in sales_data.columns:
        print("\nSales Data Correlation:")
        print("=" * 40)
        
        # Mark agents with/without sales data
        agents['Has_Sales_Data'] = agents['BZID'].isin(sales_data['BZID'])
        
        # Calculate match rate by age group
        print("\nSales Data Match Rate by Agent Age:")
        print("-" * 40)
        for _, _, label in time_periods:
            group = agents[agents['Age_Group'] == label]
            if len(group) > 0:
                match_rate = group['Has_Sales_Data'].mean() * 100
                print(f"{label}: {match_rate:.1f}% match rate ({len(group)} agents)")
    
    # Ensure we return the agents with the new columns
    return agents

def generate_detailed_report(agents, output_dir='reports'):
    """Generate a detailed analysis report."""
    print("\n=== Generating Detailed Report ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'detailed_agent_analysis_{timestamp}.xlsx')
    
    # Create Excel writer
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        # Agent summary
        summary_data = {
            'Total Agents': [len(agents)],
            'With Credit Limit': [agents['Credit Limit'].notna().sum()],
            'With Setup Date': [agents['Setup_Date'].notna().sum() if 'Setup_Date' in agents.columns else 0]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Credit limit analysis
        if 'Credit_Limit_Range' in agents.columns:
            credit_analysis = agents['Credit_Limit_Range'].value_counts().reset_index()
            credit_analysis.columns = ['Credit Limit Range', 'Count']
            credit_analysis['Percentage'] = (credit_analysis['Count'] / len(agents)) * 100
            credit_analysis.to_excel(writer, sheet_name='Credit Limits', index=False)
        
        # Setup date analysis
        if 'Setup_Date' in agents.columns:
            agents[['BZID', 'Setup_Date', 'Days_Since_Setup']]\
                .sort_values('Days_Since_Setup')\
                .to_excel(writer, sheet_name='Setup Dates', index=False)
    
    print(f"Detailed report generated: {report_file}")

def load_sales_data():
    """Load sales data for correlation analysis."""
    try:
        print("\nLoading sales data...")
        sales = pd.read_excel('source_data/sales_data.xlsx')
        
        # Standardize BZID column name if needed
        if 'account' in sales.columns:
            sales.rename(columns={'account': 'BZID'}, inplace=True)
        
        if 'BZID' in sales.columns:
            sales['BZID'] = sales['BZID'].astype(str).str.strip()
            print(f"Loaded {len(sales)} sales records")
            return sales
        else:
            print("Warning: Could not find BZID/account in sales data")
            return None
            
    except Exception as e:
        print(f"Warning: Could not load sales data - {str(e)}")
        return None

def main():
    """Main function to run the detailed analysis."""
    try:
        print("=" * 60)
        print("AGENT DATA ANALYSIS TOOL")
        print("=" * 60)
        
        # Load data
        agents = load_agent_data()
        sales = load_sales_data()
        
        # Analyze data
        agents = analyze_credit_limits(agents)
        agents = analyze_setup_dates(agents, sales_data=sales)
        
        # Generate detailed report
        generate_detailed_report(agents)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "!" * 60)
        print(f"ERROR: {str(e)}")
        print("!" * 60)
        raise

if __name__ == "__main__":
    main()
