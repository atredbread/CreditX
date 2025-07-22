"""
Data Processor for Credit Risk Analysis

This is a fixed version that properly includes agent details in the output.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.analysis.analyze_repayments import load_repayment_data, calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processor.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path('source_data')
DEFAULT_OUTPUT_DIR = Path('output')

class DataProcessor:
    """Core data processing class for agent analysis."""
    
    def __init__(self, data_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """Initialize the data processor."""
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.agents_df = None
        self.agent_details_df = None
        self.sales_df = None
        self.dpd_df = None
        self.region_df = None
        self.credit_history_df = None
        
        # Store validation results
        self.validation_errors = {}
        self.data_loaded = False
    
    def load_data(self) -> None:
        """Load and prepare the required data."""
        logger.info("Loading agent and sales data...")
        
        try:
            # Load agent details
            agent_details_file = self.data_dir / 'Agent Details.xlsx'
            if agent_details_file.exists():
                logger.info(f"Loading agent details from: {agent_details_file}")
                self.agent_details_df = pd.read_excel(
                    agent_details_file,
                    dtype={
                        'account': 'int64',
                        'organizationname': 'str',
                        'city': 'str',
                        'cityname': 'str',
                        'state': 'str',
                        'StateName': 'str',
                        'region': 'str',
                        'AgentRegion': 'str',
                        'agenttype': 'str',
                        'status': 'str',
                        'SO?TSE': 'str'
                    }
                )
                logger.info(f"Loaded {len(self.agent_details_df)} agent details records")
            else:
                logger.warning(f"Agent details file not found: {agent_details_file}")
                self.agent_details_df = pd.DataFrame()
            
            # Load other data files...
            
            self.data_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def _get_agent_basic_info(self, agent_id: str) -> Dict[str, Any]:
        """Get basic agent information including details from Agent Details."""
        try:
            # Initialize with default values
            result = {
                'organization_name': '',
                'region': '',
                'status': '',
                'credit_limit': 0.0,
                'current_balance': 0.0,
                'agent_details': {}
            }
            
            # Get agent details if available
            if hasattr(self, 'agent_details_df') and not self.agent_details_df.empty:
                try:
                    agent_details = self.agent_details_df[
                        self.agent_details_df['account'] == int(agent_id)
                    ]
                    
                    if not agent_details.empty:
                        details = agent_details.iloc[0].to_dict()
                        # Include non-encrypted fields
                        result['agent_details'] = {
                            'organization_name': str(details.get('organizationname', '')),
                            'city': str(details.get('city', '')),
                            'city_name': str(details.get('cityname', '')),
                            'state_code': str(details.get('state', '')),
                            'state_name': str(details.get('StateName', '')),
                            'region_code': str(details.get('region', '')),
                            'region_name': str(details.get('AgentRegion', '')),
                            'agent_type': str(details.get('agenttype', '')),
                            'account_status': str(details.get('status', '')),
                            'sales_officer': str(details.get('SO?TSE', ''))
                        }
                        
                        # Set top-level fields for easy access
                        result['organization_name'] = result['agent_details']['organization_name']
                        result['region'] = result['agent_details']['region_name'] or result['agent_details']['region_code']
                        result['status'] = result['agent_details']['account_status']
                        
                except Exception as e:
                    logger.warning(f"Error getting agent details for {agent_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _get_agent_basic_info for agent {agent_id}: {e}")
            return {
                'organization_name': '',
                'region': '',
                'status': '',
                'credit_limit': 0.0,
                'current_balance': 0.0,
                'agent_details': {}
            }
    
    def _load_repayment_data(self):
        """Load and prepare the full repayment DataFrame and metrics DataFrame for all agents."""
        try:
            if not hasattr(self, 'repayment_df') or self.repayment_df is None:
                repayment_files = list(self.data_dir.glob('*repayment*.csv')) + list(self.data_dir.glob('*Repayment*.xlsx'))
                if not repayment_files:
                    logger.warning("No repayment report found in data directory")
                    self.repayment_df = None
                    self.repayment_metrics_df = None
                    return
                repayment_file = repayment_files[0]
                logger.info(f"Loading repayment data from: {repayment_file}")
                if str(repayment_file).lower().endswith('.csv'):
                    try:
                        self.repayment_df = pd.read_csv(repayment_file, parse_dates=['repayment_date'], dayfirst=False, dtype={'Bzid': str})
                    except Exception:
                        self.repayment_df = pd.read_csv(repayment_file, dtype={'Bzid': str})
                else:  # Excel
                    self.repayment_df = pd.read_excel(repayment_file, parse_dates=['repayment_date'])
                # Standardize column names
                column_mapping = {
                    'Bzid': 'agent_id',
                    'Customer Repayment Date': 'repayment_date',
                    'Customer Repayment Amount': 'repayment_amount',
                    'Customer Principle Repaid': 'principal_repaid',
                }
                self.repayment_df = self.repayment_df.rename(columns={k: v for k, v in column_mapping.items() if k in self.repayment_df.columns})
                print('[FIXED DIAG] repayment_df.columns after mapping:', list(self.repayment_df.columns))
                if 'agent_id' in self.repayment_df.columns:
                    self.repayment_df['agent_id'] = self.repayment_df['agent_id'].astype(str).str.strip()
                print("[DEEP DIAG] repayment_df.columns:", list(self.repayment_df.columns))
                print("[DEEP DIAG] repayment_df row count:", len(self.repayment_df))
                unique_ids = self.repayment_df['agent_id'].unique()
                print("[DEEP DIAG] Unique agent_ids (first 20):", unique_ids[:20])
                agent_id_check = '10010576'
                agent_ids_with_ws = [x for x in unique_ids if x.strip() != x]
                print("[DEEP DIAG] agent_ids with whitespace:", agent_ids_with_ws)
                filtered_df = self.repayment_df[self.repayment_df['agent_id'].str.strip() == agent_id_check]
                print(f"[DEEP DIAG] Rows for agent_id.strip() == '{agent_id_check}':", filtered_df.head())
                print(f"[DEEP DIAG] Count for agent_id.strip() == '{agent_id_check}':", len(filtered_df))
                print("[DEEP DIAG] repayment_df.head:")
                print(self.repayment_df.head(10))
                agent_id_check = '10644128'
                unique_ids = self.repayment_df['agent_id'].unique()
                print(f"[DIAG] Is '{agent_id_check}' in repayment_df agent_id?", agent_id_check in unique_ids)
                print("[DIAG] All agent_ids containing '10644128':", [x for x in unique_ids if agent_id_check in str(x)])
                print("[DIAG] First 5 rows for agent_id == '10644128':")
                print(self.repayment_df[self.repayment_df['agent_id'] == agent_id_check].head())
                if 'repayment_date' in self.repayment_df.columns:
                    try:
                        self.repayment_df['repayment_date'] = pd.to_datetime(self.repayment_df['repayment_date'], errors='coerce')
                    except Exception:
                        self.repayment_df['repayment_date'] = pd.to_datetime(self.repayment_df['repayment_date'], errors='coerce')
                for col in ['repayment_amount', 'principal_repaid']:
                    if col in self.repayment_df.columns:
                        self.repayment_df[col] = pd.to_numeric(self.repayment_df[col], errors='coerce')
                # Drop rows with NA in any required column (matches analyze_repayments.py)
                self.repayment_df = self.repayment_df.dropna(subset=['repayment_amount', 'principal_repaid', 'repayment_date', 'agent_id'])
                # Calculate metrics for all agents at once and index by agent_id
                from scripts.analysis.analyze_repayments import calculate_metrics
                self.repayment_metrics_df = calculate_metrics(self.repayment_df)
                if not self.repayment_metrics_df.empty:
                    self.repayment_metrics_df.set_index('agent_id', inplace=True)
                    # Diagnostic: print full metrics row for agent 10091139
                    if '10091139' in self.repayment_metrics_df.index:
                        print('\n[DIAG] Full metrics row for agent 10091139:')
                        print(self.repayment_metrics_df.loc['10091139'])
                    else:
                        print('[DIAG] Agent 10091139 not found in repayment_metrics_df index!')
                    # Diagnostic: print min/max normalization values if available
                    if hasattr(self.repayment_metrics_df, 'min') and hasattr(self.repayment_metrics_df, 'max'):
                        print('\n[DIAG] Metrics min values:')
                        print(self.repayment_metrics_df.min())
                        print('[DIAG] Metrics max values:')
                        print(self.repayment_metrics_df.max())
            return
        except Exception as e:
            logger.error(f"Error loading repayment data: {e}", exc_info=True)
            self.repayment_df = None
            self.repayment_metrics_df = None
            return

    def get_agent_repayment_metrics(self, agent_id: str):
        """Return repayment metrics row for agent_id, or None if not available."""
        if hasattr(self, 'repayment_metrics_df') and self.repayment_metrics_df is not None:
            try:
                return self.repayment_metrics_df.loc[str(agent_id)]
            except Exception:
                return None
        return None

    
    def process_agent(self, agent_id: str) -> Dict[str, Any]:
        """Process data for a single agent."""
        try:
            # Get basic agent info including details
            basic_info = self._get_agent_basic_info(agent_id)

            # Initialize result with default values
            result = {
                'bzid': agent_id,
                'basic_info': basic_info,
                'sources': {
                    'repayments': {
                        'credit_health_score': 0.0,
                        'risk_category': 'UNKNOWN',
                        'confidence': 0.0,
                        'total_repayment': 0.0,
                        'principal_repaid': 0.0,
                        'total_repayment_count': 0,
                        'avg_repayment_amount': 0.0,
                        'last_repayment_date': '',
                        'first_repayment_date': '',
                        'repayment_consistency': 0.0,
                        'days_since_last_repayment': 0,
                        'days_since_first_repayment': 0,
                        'repayment_frequency': 0.0
                    }
                },
                'metadata': {
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'agent_details_included': bool(basic_info.get('agent_details')),
                    'repayment_data_available': False
                }
            }
            
            # Load repayment data if available
            self._load_repayment_data()

            if hasattr(self, 'repayment_metrics_df') and self.repayment_metrics_df is not None:
                try:
                    # Look up agent_id in self.repayment_metrics_df for repayment metrics mapping
                    metrics_row = self.get_agent_repayment_metrics(agent_id)
                    if metrics_row is not None:
                        repayments = result['sources']['repayments']
                        # Map all repayment metrics
                        repayments['credit_health_score'] = float(metrics_row.get('credit_health_score', 0.0)) if 'credit_health_score' in metrics_row else 0.0
                        repayments['risk_category'] = str(metrics_row.get('risk_category', 'UNKNOWN')) if 'risk_category' in metrics_row else 'UNKNOWN'
                        repayments['confidence'] = float(metrics_row.get('confidence', 0.0)) if 'confidence' in metrics_row else 0.0
                        repayments['total_repayment'] = float(metrics_row.get('total_repayment_amount', 0.0)) if 'total_repayment_amount' in metrics_row else 0.0
                        repayments['principal_repaid'] = float(metrics_row.get('total_principal_repaid', 0.0)) if 'total_principal_repaid' in metrics_row else 0.0
                        repayments['total_repayment_count'] = int(metrics_row.get('transaction_count', 0)) if 'transaction_count' in metrics_row else 0
                        repayments['avg_repayment_amount'] = float(metrics_row.get('avg_repayment_amount', 0.0)) if 'avg_repayment_amount' in metrics_row else 0.0
                        # Handle date fields robustly
                        last_tx = metrics_row.get('last_transaction', None)
                        first_tx = metrics_row.get('first_transaction', None)
                        repayments['last_repayment_date'] = str(last_tx) if pd.notnull(last_tx) else ''
                        repayments['first_repayment_date'] = str(first_tx) if pd.notnull(first_tx) else ''
                        repayments['repayment_consistency'] = float(metrics_row.get('data_quality', 0.0)) if 'data_quality' in metrics_row else 0.0
                        # Compute days since last/first repayment only if date is valid
                        if pd.notnull(last_tx):
                            repayments['days_since_last_repayment'] = (datetime.now().date() - last_tx.date()).days
                        else:
                            repayments['days_since_last_repayment'] = 0
                        if pd.notnull(first_tx):
                            repayments['days_since_first_repayment'] = (datetime.now().date() - first_tx.date()).days
                        else:
                            repayments['days_since_first_repayment'] = 0
                        repayments['repayment_frequency'] = float(metrics_row.get('repayment_frequency', 0.0)) if 'repayment_frequency' in metrics_row else 0.0
                        result['metadata']['repayment_data_available'] = True
                except Exception as e:
                    logger.error(f"Error calculating repayment metrics for agent {agent_id}: {e}", exc_info=True)
            else:
                logger.warning(f"No repayment data found for agent {agent_id}")
                result['metadata']['repayment_data_available'] = False

            return result
        except Exception as e:
            logger.error(f"Error processing agent {agent_id}: {e}", exc_info=True)
            raise

    def process_all_agents(self) -> Dict[str, Any]:
        """Process all agents and return results."""
        if not self.data_loaded:
            self.load_data()
        # Get list of agent IDs to process
        agent_ids = [str(agent_id) for agent_id in self.agent_details_df['account'].unique()]
        results = {}
        processed = 0
        errors = 0
        logger.info(f"Starting to process {len(agent_ids)} agents...")
        for agent_id in agent_ids:
            try:
                result = self.process_agent(agent_id)
                results[agent_id] = result
                processed += 1
            except Exception as e:
                logger.error(f"Error processing agent {agent_id}: {e}", exc_info=True)
                errors += 1
                logger.error(f"Error processing agent {agent_id}: {e}")
        
        logger.info(f"Processing complete. Success: {processed}, Errors: {errors}, Total: {len(agent_ids)}")
        
        return {
            'metadata': {
                'version': '1.0.0',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'agent_count': len(agent_ids),
                'success_count': processed,
                'error_count': errors,
                'agent_details_included': True,
                'data_sources': [
                    'Agent Details.xlsx',
                    'credit_Agents.xlsx',
                    'sales_data.xlsx',
                    'DPD.xlsx',
                    'Region_contact.xlsx'
                ]
            },
            'data': results
        }
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save results to JSON file."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f'agent_analysis_{timestamp}.json'
            
            # Save to temporary file first
            temp_file = output_file.with_suffix('.tmp')
            
            # Write JSON with indentation for readability
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Rename temp file to final filename
            temp_file.replace(output_file)
            
            logger.info(f"Results saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)
            raise

def main():
    """Main function to run the data processor."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, help='Process only this agent ID')
    parser.add_argument('--test-agent', action='store_true', help='Process a random agent for testing')
    args = parser.parse_args()

    try:
        processor = DataProcessor()
        processor.load_data()
        # --- DIAGNOSTIC: Print agent_id samples from both dataframes ---
        if hasattr(processor, 'repayment_df') and processor.repayment_df is not None:
            print("[DIAG] Repayment agent_id sample:", list(processor.repayment_df['agent_id'].unique()[:10]))
            print("[DIAG] Repayment agent_id types:", processor.repayment_df['agent_id'].map(type).unique())
            print("[DIAG] All repayment agent_ids:", list(processor.repayment_df['agent_id'].unique()))
        if hasattr(processor, 'agent_details_df') and processor.agent_details_df is not None:
            print("[DIAG] Agent details account sample:", list(processor.agent_details_df['account'].unique()[:10]))
            print("[DIAG] Agent details account types:", processor.agent_details_df['account'].map(type).unique())
        # -------------------------------------------------------------
        if args.agent:
            agent_id = args.agent
        elif args.test_agent:
            # Pick a random agent from agent_details_df
            import random
            agent_ids = [str(a) for a in processor.agent_details_df['account'].unique()]
            agent_id = random.choice(agent_ids)
            print(f"[TEST MODE] Processing random agent: {agent_id}")
        else:
            agent_id = None

        if agent_id:
            # Process only the specified/random agent
            result = processor.process_agent(agent_id)
            import pprint
            pprint.pprint(result)
            return 0
        else:
            # Full batch mode
            results = processor.process_all_agents()
            output_file = processor.save_results(results)
            logger.info(f"Processing complete. Output saved to: {output_file}")
            return 0

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
