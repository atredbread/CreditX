"""
Script to visualize GMV and credit GMV trends for agents.
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path
import sys

# Add project root to path for module imports
sys.path.append(str(Path(__file__).parent))

def load_gmv_data():
    """Load and preprocess the GMV data."""
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
    """Identify month columns and their corresponding credit GMV columns."""
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
                    
                    month_cols.append({
                        'index': i,
                        'col_name': col,
                        'month': month,
                        'year': year,
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
    month_data.sort(key=lambda x: (x['year'], x['month_num']))
    
    return month_data

def create_visualizations(gmv_data, month_columns, output_dir='output/visualizations'):
    """Create visualizations for GMV and credit GMV trends."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first 5 unique agent IDs
    agent_ids = gmv_data['Account'].drop_duplicates().head(5).tolist()
    
    # Prepare data for plotting
    plot_data = []
    for agent_id in agent_ids:
        agent_data = gmv_data[gmv_data['Account'] == agent_id].iloc[0]
        agent_plots = {'agent_id': agent_id, 'months': [], 'total_gmv': [], 'credit_gmv': []}
        
        for month in month_columns:
            total_gmv = agent_data.get(month['total_col'], 0)
            credit_gmv = agent_data.get(month['credit_col'], 0)
            
            # Convert to float and handle any non-numeric values
            try:
                total_gmv = float(total_gmv) if pd.notna(total_gmv) else 0
                credit_gmv = float(credit_gmv) if pd.notna(credit_gmv) else 0
            except (ValueError, TypeError):
                total_gmv = 0
                credit_gmv = 0
            
            agent_plots['months'].append(month['month_name'])
            agent_plots['total_gmv'].append(total_gmv)
            agent_plots['credit_gmv'].append(credit_gmv)
        
        plot_data.append(agent_plots)
    
    # Create visualizations for each agent
    for agent_plot in plot_data:
        plt.figure(figsize=(12, 6))
        
        # Plot total GMV
        plt.plot(agent_plot['months'], agent_plot['total_gmv'], 
                marker='o', label='Total GMV', color='#1f77b4')
        
        # Plot credit GMV
        plt.plot(agent_plot['months'], agent_plot['credit_gmv'], 
                marker='s', label='Credit GMV', color='#ff7f0e')
        
        # Add data labels
        for i, (total, credit) in enumerate(zip(agent_plot['total_gmv'], agent_plot['credit_gmv'])):
            if total > 0:
                plt.annotate(f"{total/1000:.1f}K", 
                            (i, total), 
                            textcoords="offset points", xytext=(0,10), 
                            ha='center', fontsize=8, color='#1f77b4')
            if credit > 0:
                plt.annotate(f"{credit/1000:.1f}K", 
                            (i, credit), 
                            textcoords="offset points", xytext=(0,-15), 
                            ha='center', fontsize=8, color='#ff7f0e')
        
        # Customize the plot
        plt.title(f"Agent {agent_plot['agent_id']} - GMV Trends")
        plt.xlabel('Month')
        plt.ylabel('Amount (in thousands)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"agent_{agent_plot['agent_id']}_gmv_trends.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for agent {agent_plot['agent_id']} to {plot_path}")
    
    # Create a combined plot for all agents
    plt.figure(figsize=(14, 8))
    
    # Define a color palette
    colors = plt.cm.tab10.colors
    
    # Plot each agent's credit GMV as a percentage of total GMV
    for idx, agent_plot in enumerate(plot_data):
        percentages = []
        for total, credit in zip(agent_plot['total_gmv'], agent_plot['credit_gmv']):
            if total > 0:
                percentages.append((credit / total) * 100)
            else:
                percentages.append(0)
        
        plt.plot(agent_plot['months'], percentages, 
                marker='o', 
                label=f"Agent {agent_plot['agent_id']}",
                color=colors[idx % len(colors)])
    
    # Customize the combined plot
    plt.title('Credit GMV as % of Total GMV by Agent')
    plt.xlabel('Month')
    plt.ylabel('Credit GMV (% of Total GMV)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the combined plot
    combined_path = os.path.join(output_dir, 'all_agents_credit_percentage.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined visualization to {combined_path}")

def main():
    """Main function to generate visualizations."""
    print("Loading GMV data...")
    gmv_data = load_gmv_data()
    
    if gmv_data is None or gmv_data.empty:
        print("No GMV data found or error loading data.")
        return
    
    # Get month columns with their corresponding credit columns
    month_columns = get_month_columns(gmv_data.columns)
    
    if not month_columns:
        print("Error: Could not identify month columns in the data.")
        return
    
    print(f"\nFound {len(month_columns)} months of data:")
    for month in month_columns:
        print(f"- {month['month_name']}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(gmv_data, month_columns)
    
    print("\nVisualization complete! Check the 'output/visualizations' directory for the generated plots.")

if __name__ == "__main__":
    main()
