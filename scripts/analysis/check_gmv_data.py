import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gmv_data(file_path: str) -> pd.DataFrame:
    """Load and clean GMV data from Excel file."""
    try:
        logger.info(f"Loading GMV data from: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names (strip whitespace and standardize)
        df.columns = df.columns.str.strip()
        
        # Find the ID column (could be 'Account' or 'Bzid')
        id_col = next((col for col in ['Account', 'Bzid', 'Agent ID', 'ID'] 
                      if col in df.columns), None)
        
        if id_col is None:
            raise ValueError("No valid ID column found in the GMV data")
        
        # Convert ID column to integer, handling any decimal points or strings
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').fillna(0).astype('int64')
        df = df[df[id_col] > 0]  # Remove any invalid/zero IDs
        
        logger.info(f"Converted {id_col} to integer type")
        logger.debug(f"Sample IDs: {df[id_col].head().tolist()}")
        
        logger.info(f"Successfully loaded {len(df)} rows with columns: {', '.join(df.columns)}")
        return df, id_col
        
    except Exception as e:
        logger.error(f"Error loading GMV data: {str(e)}")
        raise

def get_agent_gmv_data(agent_id: str, gmv_data: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    """Extract and process GMV data for a specific agent."""
    try:
        # Convert agent_id to integer
        try:
            agent_id = int(float(str(agent_id).strip()))
        except (ValueError, TypeError):
            logger.error(f"Invalid agent ID format: {agent_id}")
            return {}
            
        logger.info(f"Looking up agent ID (as integer): {agent_id}")
        
        # Find the agent's record using integer comparison
        agent_data = gmv_data[gmv_data[id_col] == agent_id]
        
        if agent_data.empty:
            logger.warning(f"No data found for agent ID: {agent_id}")
            logger.info(f"Sample available IDs: {gmv_data[id_col].head().tolist()}")
            return {}
            
        # Get the first matching record
        row = agent_data.iloc[0]
        
        # Define month patterns to look for in columns
        months = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        
        result = {
            'agent_id': agent_id,
            'months': [],
            'total_gmv': [],
            'credit_gmv': [],
            'credit_ratio': []
        }
        
        # Process each month's data
        for month in months:
            # Look for matching columns
            gmv_cols = [col for col in gmv_data.columns 
                       if month.lower() in col.lower() 
                       and 'total' in col.lower() 
                       and 'gmv' in col.lower()]
                       
            credit_cols = [col for col in gmv_data.columns 
                         if month.lower() in col.lower() 
                         and 'credit' in col.lower() 
                         and 'gmv' in col.lower()]
                         
            ratio_cols = [col for col in gmv_data.columns 
                        if month.lower() in col.lower() 
                        and ('%' in col or 'ratio' in col.lower() 
                        or 'consumption' in col.lower())]
            
            if not all([gmv_cols, credit_cols, ratio_cols]):
                logger.debug(f"Missing columns for {month}")
                continue
                
            try:
                # Get values
                total = float(row[gmv_cols[0]]) if pd.notna(row[gmv_cols[0]]) else 0
                credit = float(row[credit_cols[0]]) if pd.notna(row[credit_cols[0]]) else 0
                
                # Parse ratio (handle percentage or decimal)
                ratio_val = 0
                if ratio_cols and pd.notna(row[ratio_cols[0]]):
                    ratio_str = str(row[ratio_cols[0]]).replace('%', '').strip()
                    try:
                        ratio_val = float(ratio_str)
                        if '%' not in str(row[ratio_cols[0]]):  # If not already a percentage
                            ratio_val *= 100  # Convert to percentage
                    except (ValueError, TypeError):
                        pass
                
                # Add to results
                result['months'].append(month)
                result['total_gmv'].append(total)
                result['credit_gmv'].append(credit)
                result['credit_ratio'].append(ratio_val)
                
                logger.debug(f"Processed {month}: Total={total:,.2f}, Credit={credit:,.2f}, Ratio={ratio_val:.1f}%")
                
            except Exception as e:
                logger.error(f"Error processing {month}: {str(e)}")
                continue
                
        return result
        
    except Exception as e:
        logger.error(f"Error getting agent GMV data: {str(e)}")
        return {}

def display_agent_gmv(data: Dict[str, Any]) -> None:
    """Display GMV data for an agent in a formatted table."""
    if not data or not data['months']:
        print("\n❌ No GMV data available for this agent")
        return
        
    print("\n📊 GMV TREND ANALYSIS")
    print("=" * 50)
    print(f"Agent ID: {data['agent_id']}")
    print("-" * 50)
    print(f"{'Month':<8} | {'Total GMV':>15} | {'Credit GMV':>15} | {'Credit %':>10}")
    print("-" * 50)
    
    for month, total, credit, ratio in zip(data['months'], 
                                          data['total_gmv'], 
                                          data['credit_gmv'], 
                                          data['credit_ratio']):
        print(f"{month:<8} | {total:15,.2f} | {credit:15,.2f} | {ratio:9.1f}%")
    
    # Calculate and display summary
    avg_credit_ratio = sum(data['credit_ratio']) / len(data['credit_ratio']) if data['credit_ratio'] else 0
    total_gmv = sum(data['total_gmv'])
    total_credit = sum(data['credit_gmv'])
    
    print("-" * 50)
    print(f"{'TOTAL':<8} | {total_gmv:15,.2f} | {total_credit:15,.2f} | {avg_credit_ratio:9.1f}%")
    print("=" * 50)

def main():
    """Main function to demonstrate GMV data loading and display."""
    try:
        # Path to the GMV data file
        file_path = os.path.join('source_data', 'Credit_history_sales_vs_credit_sales.xlsx')
        
        # Load the data
        gmv_data, id_col = load_gmv_data(file_path)
        
        # Interactive agent lookup
        while True:
            print("\n" + "="*50)
            print("AGENT GMV DATA LOOKUP")
            print("="*50)
            print("Enter agent ID (as integer) to view GMV data (or 'q' to quit)")
            print(f"Sample agent IDs: {gmv_data[id_col].head().tolist()}")
            agent_input = input("\nAgent ID: ").strip()
            
            if agent_input.lower() == 'q':
                break
                
            if not agent_input or not agent_input.isdigit():
                print("❌ Please enter a valid numeric agent ID")
                continue
                
            agent_id = int(agent_input)
                
            # Get and display agent data
            agent_data = get_agent_gmv_data(agent_id, gmv_data, id_col)
            display_agent_gmv(agent_data)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
