"""
Script to analyze and report on agents without matching sales data.
"""
import pandas as pd
import os
from datetime import datetime

def load_data():
    """Load agent and sales data."""
    print("Loading data...")
    
    # Load agent data
    agents = pd.read_excel('source_data/credit_Agents.xlsx')
    agents['BZID'] = agents['Bzid'].astype(str).str.strip()
    
    # Load sales data
    sales = pd.read_excel('source_data/sales_data.xlsx')
    sales['BZID'] = sales['account'].astype(str).str.strip()
    
    return agents, sales

def analyze_unmatched_agents(agents, sales):
    """Analyze agents without matching sales data."""
    print("\n=== Analyzing Unmatched Agents ===")
    
    # Get BZID sets
    agent_bzids = set(agents['BZID'])
    sales_bzids = set(sales['BZID'])
    
    # Find unmatched agents
    unmatched_bzids = agent_bzids - sales_bzids
    matched_bzids = agent_bzids.intersection(sales_bzids)
    
    print(f"Total agents: {len(agent_bzids):,}")
    print(f"Agents with sales data: {len(matched_bzids):,} ({(len(matched_bzids)/len(agent_bzids))*100:.1f}%)")
    print(f"Agents without sales data: {len(unmatched_bzids):,} ({(len(unmatched_bzids)/len(agent_bzids))*100:.1f}%)")
    
    # Create a DataFrame for unmatched agents
    unmatched_agents = agents[agents['BZID'].isin(unmatched_bzids)].copy()
    
    # Display first 10 unmatched agents
    if not unmatched_agents.empty:
        print("\nSample of 10 agents missing from sales data:")
        sample_agents = unmatched_agents.head(10)[['BZID', 'Name', 'Credit Limit']] if 'Name' in unmatched_agents.columns else unmatched_agents.head(10)[['BZID', 'Credit Limit']]
        print(sample_agents.to_string(index=False))
    
    # Add analysis columns
    unmatched_agents['Missing_Sales_Data'] = True
    
    # Get credit limit distribution
    credit_limit_bins = [0, 10000, 50000, 100000, float('inf')]
    credit_limit_labels = ['<10K', '10K-50K', '50K-100K', '>100K']
    
    if 'Credit Limit' in unmatched_agents.columns:
        unmatched_agents['Credit_Limit_Range'] = pd.cut(
            unmatched_agents['Credit Limit'],
            bins=credit_limit_bins,
            labels=credit_limit_labels,
            right=False
        )
    
    return unmatched_agents

def generate_report(unmatched_agents, output_dir='reports'):
    """Generate a detailed report of unmatched agents."""
    print("\n=== Generating Report ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'unmatched_agents_report_{timestamp}.xlsx')
    
    # Create a Pandas Excel writer
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        # Overall summary
        summary_data = {
            'Metric': [
                'Total Agents',
                'Agents with Sales Data',
                'Agents without Sales Data',
                'Percentage without Sales Data'
            ],
            'Value': [
                len(unmatched_agents) + len(set(unmatched_agents['BZID'])),
                len(set(unmatched_agents['BZID'])) - len(unmatched_agents),
                len(unmatched_agents),
                f"{(len(unmatched_agents) / (len(unmatched_agents) + len(set(unmatched_agents['BZID'])) - len(unmatched_agents))) * 100:.1f}%"
            ]
        }
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed unmatched agents
        unmatched_agents.to_excel(writer, sheet_name='Unmatched Agents', index=False)
        
        # Credit limit analysis
        if 'Credit_Limit_Range' in unmatched_agents.columns:
            credit_limit_summary = unmatched_agents['Credit_Limit_Range'].value_counts().reset_index()
            credit_limit_summary.columns = ['Credit Limit Range', 'Count']
            credit_limit_summary['Percentage'] = (credit_limit_summary['Count'] / len(unmatched_agents)) * 100
            credit_limit_summary.to_excel(writer, sheet_name='Credit Limit Analysis', index=False)
    
    print(f"Report generated: {report_file}")

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        agents, sales = load_data()
        
        # Analyze unmatched agents
        unmatched_agents = analyze_unmatched_agents(agents, sales)
        
        # Generate report
        if not unmatched_agents.empty:
            generate_report(unmatched_agents)
        else:
            print("No unmatched agents found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
