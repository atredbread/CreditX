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
            
            # Load DPD data (modular, standards-compliant)
            dpd_file = self.data_dir / 'DPD.xlsx'
            if dpd_file.exists():
                self.dpd_df = pd.read_excel(dpd_file)
                if 'Bzid' in self.dpd_df.columns:
                    self.dpd_df['Bzid'] = self.dpd_df['Bzid'].astype(str).str.strip()
                if 'Dpd' in self.dpd_df.columns:
                    self.dpd_df = self.dpd_df.sort_values('Dpd', ascending=False).drop_duplicates('Bzid', keep='first')[['Bzid', 'Dpd']]
                    self.dpd_df = self.dpd_df.rename(columns={'Bzid': 'agent_id', 'Dpd': 'current_dpd'})
            else:
                self.dpd_df = pd.DataFrame(columns=['agent_id', 'current_dpd'])
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
                # Calculate metrics for all agents at once
                from scripts.analysis.analyze_repayments import calculate_metrics
                self.repayment_metrics_df = calculate_metrics(self.repayment_df)

                # === DPD Integration: Modular, Standards-Compliant ===
                # Merge DPD data by agent_id
                if hasattr(self, 'dpd_df') and not self.dpd_df.empty:
                    self.repayment_metrics_df = self.repayment_metrics_df.merge(
                        self.dpd_df, on='agent_id', how='left')
                else:
                    self.repayment_metrics_df['current_dpd'] = None

                # Calculate DPD_subscore as per reference logic
                def dpd_subscore(dpd):
                    if pd.isna(dpd):
                        return 1.0
                    try:
                        d = float(dpd)
                    except Exception:
                        return 1.0
                    if d == 0:
                        return 1.0
                    elif d < 30:
                        return 0.7
                    elif d < 60:
                        return 0.4
                    else:
                        return 0.1
                self.repayment_metrics_df['DPD_subscore'] = self.repayment_metrics_df['current_dpd'].apply(dpd_subscore)
                # Normalize repayment score
                self.repayment_metrics_df['repayment_score_norm'] = self.repayment_metrics_df['credit_health_score'] / 100.0
                # Final combined score
                self.repayment_metrics_df['final_score'] = 0.7 * self.repayment_metrics_df['repayment_score_norm'] + 0.3 * self.repayment_metrics_df['DPD_subscore']
                # Assign risk categories by percentile of final_score
                if not self.repayment_metrics_df.empty:
                    scores = self.repayment_metrics_df['final_score']
                    low_cutoff = np.percentile(scores, 80)
                    high_cutoff = np.percentile(scores, 20)
                    def assign_risk(score):
                        if score >= low_cutoff:
                            return 'Low'
                        elif score < high_cutoff:
                            return 'High'
                        else:
                            return 'Medium'
                    self.repayment_metrics_df['risk_category'] = scores.apply(assign_risk)
                # Set index for fast lookup
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

            # --- DPD integration ---
            current_dpd = None
            dpd_subscore = None
            if hasattr(self, 'dpd_df') and self.dpd_df is not None and not self.dpd_df.empty:
                dpd_row = self.dpd_df[self.dpd_df['agent_id'] == str(agent_id)]
                if not dpd_row.empty:
                    current_dpd = float(dpd_row.iloc[0]['current_dpd']) if pd.notnull(dpd_row.iloc[0]['current_dpd']) else None
                    # Example subscore: 1 - min(current_dpd/30, 1) (score 1 if DPD=0, 0 if DPD>=30)
                    if current_dpd is not None:
                        dpd_subscore = max(0.0, 1.0 - min(current_dpd / 30.0, 1.0))

            if hasattr(self, 'repayment_metrics_df') and self.repayment_metrics_df is not None:
                metrics_row = self.get_agent_repayment_metrics(agent_id)
                if metrics_row is not None:
                    repayments = result['sources']['repayments']
                    repayments['credit_health_score'] = float(metrics_row.get('credit_health_score', 0.0)) if 'credit_health_score' in metrics_row else 0.0
                    repayments['risk_category'] = str(metrics_row.get('risk_category', 'UNKNOWN')) if 'risk_category' in metrics_row else 'UNKNOWN'
                    repayments['confidence'] = float(metrics_row.get('confidence', 0.0)) if 'confidence' in metrics_row else 0.0
                    repayments['total_repayment'] = float(metrics_row.get('total_repayment_amount', 0.0)) if 'total_repayment_amount' in metrics_row else 0.0
                    repayments['principal_repaid'] = float(metrics_row.get('total_principal_repaid', 0.0)) if 'total_principal_repaid' in metrics_row else 0.0
                    repayments['total_repayment_count'] = int(metrics_row.get('transaction_count', 0)) if 'transaction_count' in metrics_row else 0
                    repayments['avg_repayment_amount'] = float(metrics_row.get('avg_repayment_amount', 0.0)) if 'avg_repayment_amount' in metrics_row else 0.0
                    last_tx = metrics_row.get('last_transaction', None)
                    first_tx = metrics_row.get('first_transaction', None)
                    repayments['last_repayment_date'] = str(last_tx) if pd.notnull(last_tx) else ''
                    repayments['first_repayment_date'] = str(first_tx) if pd.notnull(first_tx) else ''
                    repayments['repayment_consistency'] = float(metrics_row.get('data_quality', 0.0)) if 'data_quality' in metrics_row else 0.0
                    # Compute days since last/first repayment only if date is valid
                    if pd.notnull(last_tx):
                        days_since_last = (datetime.now(timezone.utc) - pd.to_datetime(last_tx, utc=True)).days
                        repayments['days_since_last_repayment'] = days_since_last
                    if pd.notnull(first_tx):
                        days_since_first = (datetime.now(timezone.utc) - pd.to_datetime(first_tx, utc=True)).days
                        repayments['days_since_first_repayment'] = days_since_first
                    # Repayment frequency (simple heuristic)
                    if repayments['total_repayment_count'] > 0 and pd.notnull(first_tx) and pd.notnull(last_tx):
                        days_span = max((pd.to_datetime(last_tx, utc=True) - pd.to_datetime(first_tx, utc=True)).days, 1)
                        repayments['repayment_frequency'] = repayments['total_repayment_count'] / days_span
                    result['metadata']['repayment_data_available'] = True

            # Add DPD info to output
            result['sources']['repayments']['current_dpd'] = current_dpd
            result['sources']['repayments']['DPD_subscore'] = dpd_subscore

            # Optionally, update final score to include DPD subscore (example: weighted average)
            # This should follow your documented risk scoring logic; adjust weights as needed
            if dpd_subscore is not None and 'credit_health_score' in result['sources']['repayments']:
                # Example: 80% repayment score, 20% DPD subscore
                chs = result['sources']['repayments']['credit_health_score']
                final_score = 0.8 * chs + 0.2 * (dpd_subscore * 100)
                result['sources']['repayments']['final_score'] = final_score
            else:
                result['sources']['repayments']['final_score'] = result['sources']['repayments'].get('credit_health_score', 0.0)

            return result
        except Exception as e:
            logger.error(f"Error in process_agent for agent {agent_id}: {e}", exc_info=True)
            return None

    def process_all_agents(self) -> dict:
        """Process all agents and aggregate results."""
        if not self.data_loaded:
            self.load_data()
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
                    # Add other sources as needed
                ]
            },
            'results': results
        }

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
                        # (existing logic for days_since_last/first_repayment)
                        pass
            return result
        except Exception as e:
            logger.error(f"Error in process_agent for agent {agent_id}: {e}", exc_info=True)
            return None

                # (rest of your function logic)
            return result
        except Exception as e:
            logger.error(f"Error in process_agent for agent {agent_id}: {e}", exc_info=True)
            return None
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
