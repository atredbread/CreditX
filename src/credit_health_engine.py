"""
Credit Health Intelligence Engine
----------------------------------
Automated system for credit agent classification and monitoring.
Processes transaction data to generate agent tiers and regional insights.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_health_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_DIR = Path('source_data')
OUTPUT_DIR = Path('output')
REPORT_DIR = OUTPUT_DIR / 'region_reports'
EMAIL_DIR = OUTPUT_DIR / 'email_summaries'

# Required input files with expected columns mapping based on data_structure.md
REQUIRED_FILES = {
    'credit_Agents.xlsx': {
        'required_columns': ['Bzid', 'Phone', 'Credit Line Setup Co', 'Approval Amount', 
                          'Credit Limit', 'Credit Line Balance'],
        'dtype': {
            'Bzid': 'int64', 
            'Phone': 'str', 
            'Approval Amount': 'float64', 
            'Credit Limit': 'float64', 
            'Credit Line Balance': 'float64'
        }
    },
    'Credit_history_sales_vs_credit_sales.xlsx': {
        'required_columns': ['Account'],
        'dtype': {
            'Account': 'int64',
            'Jan Total GMV - Year25': 'int64',
            'Jan Credit Gmv': 'int64',
            'Feb Total GMV - Year25': 'int64',
            'Feb Credit Gmv': 'int64',
            'Mar Total GMV - Year25': 'int64',
            'Mar Credit Gmv': 'int64',
            'Apr Total GMV - Year25': 'int64',
            'Apr Credit Gmv': 'int64',
            'May Total GMV - Year25': 'int64',
            'May Credit Gmv': 'int64',
            'June Total GMV - Year25': 'int64',
            'JuneCredit Gmv': 'int64'
        },
        'month_columns': {
            'total_gmv': [
                'Jan Total GMV - Year25', 
                'Feb Total GMV - Year25', 
                'Mar Total GMV - Year25',
                'Apr Total GMV - Year25', 
                'May Total GMV - Year25', 
                'June Total GMV - Year25'
            ],
            'credit_gmv': [
                'Jan Credit Gmv', 
                'Feb Credit Gmv', 
                'Mar Credit Gmv',
                'Apr Credit Gmv', 
                'May Credit Gmv', 
                'JuneCredit Gmv'  # Note: No space between 'June' and 'Credit'
            ]
        }
    },
    'Credit_sales_data.xlsx': {
        'required_columns': ['account', 'GMV'],
        'dtype': {
            'account': 'int64', 
            'GMV': 'float64',
            'DATE': 'datetime64[ns]'
        }
    },
    'DPD.xlsx': {
        'required_columns': ['Bzid', 'Dpd', 'Pos'],
        'dtype': {
            'Bzid': 'int64', 
            'Dpd': 'int64', 
            'Pos': 'float64',
            'Phone': 'str',
            'Business Name': 'str',
            'Username': 'str',
            'Anchor': 'str'
        }
    },
    'Region_contact.xlsx': {
        'required_columns': ['Region', 'Manager', 'Name'],
        'dtype': {
            'Region': 'str',
            'Manager': 'str',
            'Name': 'str'
        }
    },
    'sales_data.xlsx': {
        'required_columns': ['account', 'GMV', 'DATE(a.creationtime)'],
        'dtype': {
            'account': 'int64', 
            'GMV': 'float64',
            'DATE(a.creationtime)': 'datetime64[ns]',
            'organizationname': 'str',
            'status': 'str',
            'TotalSeats': 'int64',
            'AgentCommission(Exe GDS)': 'float64',
            'city': 'str',
            'State': 'str',
            'Region': 'str',
            'Check': 'bool',
            'Ro Name': 'str',
            'RM Name': 'str'
        }
    }
}

class CreditHealthEngine:
    """Main class for the Credit Health Intelligence Engine."""
    
    def __init__(self):
        """
        Initialize the Credit Health Engine.
        
        Attributes:
            data (dict): Dictionary to store raw input data
            features (pd.DataFrame): Engineered features for all agents
            classified_agents (pd.DataFrame): Agent classifications with tiers and recommendations
        """
        self.data = {}
        self.features = pd.DataFrame()
        self.classified_agents = pd.DataFrame()
        self._setup_directories()
    
    def _setup_directories(self):
        """Create required directories if they don't exist."""
        for directory in [INPUT_DIR, REPORT_DIR, EMAIL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_dataframe(self, df: pd.DataFrame, file_name: str) -> bool:
        """
        Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            file_name: Name of the file being validated
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if file_name not in REQUIRED_FILES:
            logger.warning(f"No validation rules found for {file_name}")
            return True
            
        file_spec = REQUIRED_FILES[file_name]
        
        # Check for required columns
        missing_columns = [col for col in file_spec['required_columns'] 
                         if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in {file_name}: {', '.join(missing_columns)}")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            return False
            
        # Validate data types
        if 'dtype' in file_spec:
            for col, expected_type in file_spec['dtype'].items():
                if col in df.columns:
                    try:
                        if expected_type == 'int64':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        elif expected_type == 'float64':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif expected_type == 'str':
                            df[col] = df[col].astype(str)
                        elif expected_type == 'datetime64[ns]':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error converting column {col} to {expected_type}: {e}")
                        return False
                        
        return True
        
    def generate_reports(self, output_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Generate comprehensive reports for all agents and overall summary.
        
        Args:
            output_dir: Directory to save reports (defaults to self.reports_dir)
            
        Returns:
            Tuple[bool, str]: (success, message) indicating success/failure and status message
        """
        try:
            logger.info("Starting report generation...")
            
            # Use provided output directory or default
            if output_dir is None:
                output_dir = REPORT_DIR
            
            # Ensure output directory exists
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Using output directory: {output_dir}")
            except Exception as e:
                error_msg = f"Failed to create output directory {output_dir}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg
            
            # Check if we have the necessary data
            if not hasattr(self, 'data') or not self.data:
                error_msg = "No data available to generate reports"
                logger.error(error_msg)
                return False, error_msg
                
            # Generate timestamp for report naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Generate agent reports
            agent_reports_dir = output_dir / 'agent_reports'
            try:
                agent_reports_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created agent reports directory: {agent_reports_dir}")
            except Exception as e:
                error_msg = f"Failed to create agent reports directory: {str(e)}"
                logger.error(error_msg)
                return False, error_msg
            
            # Get agent data
            agents = self.data.get('credit_Agents.xlsx')
            if agents is None or agents.empty:
                error_msg = "No agent data available to generate reports"
                logger.error(error_msg)
                return False, error_msg
                
            total_agents = len(agents)
            agent_summaries = []
            
            logger.info(f"Generating reports for {total_agents} agents...")
            
            # Ensure required columns exist
            required_columns = ['Bzid', 'Name', 'Region']
            missing_columns = [col for col in required_columns if col not in agents.columns]
            if missing_columns:
                logger.warning(f"Missing columns in agent data: {', '.join(missing_columns)}"
                             f" - Using default values where needed")
            
            for idx, (_, agent_row) in enumerate(agents.iterrows(), 1):
                try:
                    agent_id = agent_row.get('Bzid')
                    if pd.isna(agent_id):
                        logger.warning(f"Skipping row {idx} - missing agent ID")
                        continue
                        
                    logger.debug(f"Processing agent {idx}/{total_agents}: {agent_id}")
                    
                    # Prepare agent data for JSON serialization
                    agent_data = agent_row.to_dict()
                    
                    # Convert non-serializable types
                    for k, v in agent_data.items():
                        if isinstance(v, (pd.Timestamp, np.datetime64)):
                            agent_data[k] = str(v)
                        elif isinstance(v, (np.integer, np.floating)):
                            agent_data[k] = int(v) if v.is_integer() else float(v)
                    
                    # Save detailed agent report
                    report_file = agent_reports_dir / f"agent_{agent_id}_report.json"
                    try:
                        with open(report_file, 'w', encoding='utf-8') as f:
                            json.dump(agent_data, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"Failed to write report for agent {agent_id}: {str(e)}")
                        continue
                    
                    # Add to summary
                    agent_summaries.append({
                        'agent_id': agent_id,
                        'name': agent_row.get('Name', 'N/A'),
                        'region': agent_row.get('Region', 'N/A'),
                        'report_path': str(report_file.relative_to(output_dir))
                    })
                    
                    # Log progress
                    if idx % 100 == 0 or idx == total_agents:
                        logger.info(f"Processed {idx}/{total_agents} agents")
                    
                except Exception as e:
                    logger.error(f"Error processing agent {idx} (ID: {agent_id}): {str(e)}", 
                               exc_info=logger.isEnabledFor(logging.DEBUG))
            
            if not agent_summaries:
                error_msg = "No agent reports were generated successfully"
                logger.error(error_msg)
                return False, error_msg
            
            # 2. Generate summary report
            logger.info("Generating summary report...")
            summary_report = {
                'timestamp': timestamp,
                'total_agents': total_agents,
                'reported_agents': len(agent_summaries),
                'success_rate': f"{(len(agent_summaries)/total_agents*100):.1f}%" if total_agents > 0 else "0%",
                'agent_reports': agent_summaries,
                'report_generated': datetime.now().isoformat()
            }
            
            summary_file = output_dir / 'summary_report.json'
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False)
                logger.info(f"Summary report saved to {summary_file}")
            except Exception as e:
                error_msg = f"Failed to write summary report: {str(e)}"
                logger.error(error_msg)
                return False, error_msg
                
            success_msg = (f"Successfully generated reports for {len(agent_summaries)}/{total_agents} agents "
                         f"({(len(agent_summaries)/total_agents*100):.1f}%)" if total_agents > 0 else "0 agents")
            logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error in report generation: {str(e)}"
            logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
            return False, error_msg
            return False
        
        # Special validation for credit history file
        if file_name == 'Credit_history_sales_vs_credit_sales.xlsx':
            if 'month_columns' in file_spec:
                # Log all available columns for debugging
                logger.debug(f"Available columns in {file_name}: {', '.join(df.columns)}")
                
                # Check for each month column group
                found_groups = {}
                for group_name, expected_cols in file_spec['month_columns'].items():
                    found_cols = [col for col in expected_cols if col in df.columns]
                    found_groups[group_name] = found_cols
                    
                    if found_cols:
                        logger.debug(f"Found {len(found_cols)}/{len(expected_cols)} columns for {group_name}: {', '.join(found_cols)}")
                
                # Check if we have at least one complete column group
                valid_groups = [name for name, cols in found_groups.items() 
                              if len(cols) == len(file_spec['month_columns'][name])]
                
                if not valid_groups:
                    logger.error(f"No complete set of month columns found in {file_name}")
                    logger.error(f"Expected column groups: {file_spec['month_columns']}")
                    logger.error(f"Found columns: {list(df.columns)}")
                    return False
                
                # Log which column groups are complete
                logger.info(f"Found complete data for: {', '.join(valid_groups)}")
                
                # Store the found columns for later use
                df.attrs['found_month_columns'] = found_groups
        
        # Check for empty data
        if df.empty:
            logger.warning(f"{file_name} is empty")
            return False
            
        # Log basic info about the loaded data
        logger.info(f"Loaded {len(df)} rows from {file_name}")
        logger.debug(f"Sample data from {file_name}:\n{df.head(2).to_string()}")
                    
        return True
    
    def _load_excel_file(self, file_path: Path, file_spec: dict) -> Optional[pd.DataFrame]:
        """
        Load an Excel file with appropriate settings based on file specification.
        
        Args:
            file_path: Path to the Excel file
            file_spec: File specification from REQUIRED_FILES
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        try:
            # First load without type conversion to inspect the data
            df = pd.read_excel(file_path)
            
            # Clean column names (remove extra spaces, special chars, etc.)
            df.columns = df.columns.str.strip()
            
            # Special handling for credit history file
            if file_path.name == 'Credit_history_sales_vs_credit_sales.xlsx' and 'Account' in df.columns:
                # Log sample values for debugging
                sample_values = df['Account'].dropna().head().tolist()
                logger.debug(f"Sample 'Account' values: {sample_values}")
                
                # Try to clean the Account column
                try:
                    # First try direct conversion
                    df['Account'] = pd.to_numeric(df['Account'], errors='raise').astype('int64')
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert 'Account' to int64 directly: {e}")
                    try:
                        # Try cleaning any non-numeric characters
                        df['Account'] = df['Account'].astype(str).str.replace(r'\D', '', regex=True)
                        df['Account'] = pd.to_numeric(df['Account'], errors='coerce')
                        df = df.dropna(subset=['Account'])
                        df['Account'] = df['Account'].astype('int64')
                        logger.info("Successfully cleaned 'Account' column")
                    except Exception as e2:
                        logger.error(f"Failed to clean 'Account' column: {e2}")
                        return None
            
            # Apply other type conversions if specified
            dtype = file_spec.get('dtype', {})
            for col, col_type in dtype.items():
                if col in df.columns and col != 'Account':  # Already handled Account
                    try:
                        df[col] = df[col].astype(col_type)
                    except Exception as e:
                        logger.warning(f"Could not convert column '{col}' to {col_type}: {e}")
            
            # Handle month columns for credit history file
            if 'month_columns' in file_spec and file_path.name == 'Credit_history_sales_vs_credit_sales.xlsx':
                all_month_cols = []
                for col_group in file_spec['month_columns'].values():
                    all_month_cols.extend(col_group)
                
                # Check if any month columns are missing
                missing_cols = [col for col in all_month_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing month columns in {file_path.name}: {', '.join(missing_cols)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}", exc_info=True)
            return None
    
    def _normalize_region_name(self, region_name: Optional[str]) -> str:
        """Normalize region names to handle variations and missing values."""
        if pd.isna(region_name) or not str(region_name).strip():
            return 'Unassigned'
        return str(region_name).strip().title()

    def _map_data_to_agents(self) -> bool:
        """
        Map all data to credit agents (like VLOOKUP) to ensure complete coverage.
        
        Returns:
            bool: True if mapping was successful, False otherwise
        """
        try:
            if 'credit_Agents.xlsx' not in self.data:
                logger.error("Cannot map data: credit_Agents.xlsx not loaded")
                return False
                
            agents = self.data['credit_Agents.xlsx']
            
            # Ensure Bzid is string type for consistent comparison
            agents['Bzid'] = agents['Bzid'].astype(str)
            agent_ids = set(agents['Bzid'].dropna().unique())
            logger.info(f"Found {len(agent_ids)} unique agent IDs in credit_Agents.xlsx")
            
            # Normalize region data if present
            if 'Region' in agents.columns:
                agents['Region'] = agents['Region'].apply(self._normalize_region_name)
                logger.info(f"Found {len(agents['Region'].unique())} unique regions in agent data")
            
            # Track which agents are missing from each dataset
            missing_agents = {}
            
            # Map DPD data to agents
            if 'DPD.xlsx' in self.data:
                dpd_data = self.data['DPD.xlsx']
                dpd_agents = set(dpd_data['Bzid'].dropna().unique())
                missing_dpd = agent_ids - dpd_agents
                if missing_dpd:
                    missing_agents['DPD.xlsx'] = missing_dpd
                    logger.warning(f"{len(missing_dpd)} agents missing DPD data")
                    
                    # Add empty DPD records for missing agents
                    empty_dpd = pd.DataFrame({
                        'Bzid': list(missing_dpd),
                        'Dpd': 0,
                        'Amount': 0,
                        'DueDate': pd.NaT
                    })
                    self.data['DPD.xlsx'] = pd.concat([dpd_data, empty_dpd], ignore_index=True)
            
            # Map sales data to agents
            if 'Credit_history_sales_vs_credit_sales.xlsx' in self.data:
                sales_data = self.data['Credit_history_sales_vs_credit_sales.xlsx']
                sales_agents = set(sales_data['Account'].dropna().astype(str).unique())
                missing_sales = agent_ids - {str(agent_id) for agent_id in sales_agents}
                if missing_sales:
                    missing_agents['Credit_history_sales_vs_credit_sales.xlsx'] = missing_sales
                    logger.warning(f"{len(missing_sales)} agents missing sales data")
                    
                    # Add empty sales records for missing agents
                    empty_sales = pd.DataFrame({
                        'Account': list(missing_sales),
                        'TotalSales': 0,
                        'CreditSales': 0,
                        'Date': pd.to_datetime('today').normalize()
                    })
                    self.data['Credit_history_sales_vs_credit_sales.xlsx'] = pd.concat(
                        [sales_data, empty_sales], 
                        ignore_index=True
                    )
            
            # Log comprehensive report
            if missing_agents:
                logger.warning("\n[WARNING] Agents missing from datasets:")
                for dataset, missing in missing_agents.items():
                    logger.warning(f"  - {dataset}: {len(missing)} agents missing")
                    if len(missing) <= 5:  # Show up to 5 missing agent IDs for debugging
                        logger.warning(f"     Missing agents: {', '.join(map(str, sorted(missing)[:5]))}")
                    
                    # Save detailed report of missing agents
                    output_dir = Path('output/data_quality')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    missing_file = output_dir / f"missing_agents_{Path(dataset).stem}.csv"
                    pd.Series(list(missing), name='Bzid').to_csv(missing_file, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error mapping data to agents: {str(e)}", exc_info=True)
            return False
    
    def load_data(self) -> bool:
        """
        Load and validate all required input files, then map all data to credit agents.
        
        Returns:
            bool: True if all files were loaded, validated, and mapped successfully, False otherwise
        """
        logger.info("Loading and validating input data...")
        missing_files = []
        validation_errors = []
        
        # First, load all required files
        for file_name, file_spec in REQUIRED_FILES.items():
            file_path = INPUT_DIR / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                missing_files.append(file_name)
                continue
                
            try:
                # Load the file
                df = self._load_excel_file(file_path, file_spec)
                if df is None or df.empty:
                    logger.error(f"Failed to load or empty DataFrame: {file_name}")
                    validation_errors.append(file_name)
                    continue
                
                # Apply type conversions if specified
                if 'dtype' in file_spec:
                    for col, col_type in file_spec['dtype'].items():
                        if col in df.columns and col != 'Account':  # Skip Account as it's handled specially
                            try:
                                df[col] = df[col].astype(col_type)
                            except Exception as e:
                                logger.warning(f"Could not convert column '{col}' to {col_type}: {e}")
                
                # Handle month columns for credit history file
                if file_name == 'Credit_history_sales_vs_credit_sales.xlsx' and 'month_columns' in file_spec:
                    all_month_cols = []
                    for col_group in file_spec['month_columns'].values():
                        all_month_cols.extend(col_group)
                    
                    # Log month columns info for debugging
                    logger.debug(f"Processing month columns in {file_name}: {all_month_cols}")
                
                # Validate the DataFrame
                if not self._validate_dataframe(df, file_name):
                    validation_errors.append(file_name)
                    continue
                
                # Store the loaded data
                self.data[file_name] = df
                logger.info(f"[OK] Successfully loaded {file_name} with shape {df.shape}")
                
                # Log column information for debugging
                logger.debug(f"Columns in {file_name}: {', '.join(df.columns)}")
                
            except Exception as e:
                logger.error(f"Error loading {file_name}: {str(e)}", exc_info=True)
                validation_errors.append(file_name)
        
        # Report loaded files
        loaded_files = set(self.data.keys())
        expected_files = set(REQUIRED_FILES.keys())
        
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING SUMMARY")
        logger.info("="*50)
        
        if loaded_files:
            logger.info("\n[SUCCESS] Successfully loaded files:")
            for f in sorted(loaded_files):
                logger.info(f"  - {f} ({len(self.data[f])} rows)")
        
        # Report missing files
        if missing_files:
            logger.warning("\n[WARNING] The following required files were not found:")
            for f in sorted(missing_files):
                logger.warning(f"  - {f}")
        
        # Report validation errors
        if validation_errors:
            logger.error("\n[ERROR] Files with validation errors:")
            for f in sorted(validation_errors):
                logger.error(f"  - {f}")
        
        # If we have the required data, map everything to agents
        if 'credit_Agents.xlsx' in self.data:
            logger.info("\nMapping all data to credit agents...")
            if not self._map_data_to_agents():
                logger.error("Failed to map all data to credit agents")
                validation_errors.append("data_mapping")
        
        # Check if all required files were loaded successfully
        all_loaded = len(missing_files) == 0 and len(validation_errors) == 0
        
        if all_loaded:
            logger.info("\n✅ All files loaded, validated, and mapped successfully!")
        else:
            logger.error("\n[ERROR] Some files failed to load, validate, or map. Check logs for details.")
        
        logger.info("="*50 + "\n")
        
        return all_loaded
    def engineer_features(self) -> bool:
        """
        Engineer features from the loaded data.
        
        Returns:
            bool: True if features were successfully engineered, False otherwise
        """
        logger.info("Engineering features...")
        
        try:
            # Import here to avoid circular imports
            from src.feature_engineering import FeatureEngineer
            
            # Create feature engineer instance with current data
            feature_engineer = FeatureEngineer(self.data)
            
            # Generate all features
            self.features = feature_engineer.engineer_all_features()
            
            if self.features.empty:
                logger.error("No features were generated")
                return False
                
            logger.info(f"Successfully engineered {len(self.features.columns)} features for {len(self.features)} agents")
            
            # Save features for debugging
            output_dir = Path('output/features')
            output_dir.mkdir(parents=True, exist_ok=True)
            self.features.to_csv(output_dir / 'engineered_features.csv')
            logger.info(f"Features saved to {output_dir}/engineered_features.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}", exc_info=True)
            return False
    
    def classify_agents(self) -> bool:
        """
        Classify agents into tiers (P0-P5) based on their features.
        
        Returns:
            bool: True if classification was successful, False otherwise
        """
        logger.info("Classifying agents into tiers...")
        
        if not hasattr(self, 'features') or self.features.empty:
            logger.error("No features available for classification. Run engineer_features() first.")
            return False
            
        try:
            from src.agent_classifier import AgentClassifier
            
            # Initialize the classifier
            classifier = AgentClassifier()
            
            # Classify all agents
            self.classified_agents = classifier.classify_all_agents(self.features)
            
            if self.classified_agents.empty:
                logger.error("No agents were classified")
                return False
                
            logger.info(f"Successfully classified {len(self.classified_agents)} agents")
            
            # Save classification results
            output_dir = Path('output/classifications')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'agent_classifications.csv'
            self.classified_agents.to_csv(output_file)
            logger.info(f"Classification results saved to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during agent classification: {str(e)}", exc_info=True)
            return False
    
    def _prepare_region_report(self, region_data: pd.DataFrame, region_name: str) -> pd.ExcelWriter:
        """
        Prepare a region report with multiple sheets.
        
        Args:
            region_data: DataFrame containing agent data for the region
            region_name: Name of the region
            
        Returns:
            ExcelWriter: Configured Excel writer object
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path('output/reports')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Excel writer
            output_file = output_dir / f"{region_name}_report.xlsx"
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            
            # 1. Summary Sheet
            summary_metrics = [
                ('Total Agents', lambda: len(region_data)),
                ('P0 Agents', lambda: len(region_data[region_data['tier'] == 'P0'])),
                ('P1 Agents', lambda: len(region_data[region_data['tier'] == 'P1'])),
                ('P4 Agents', lambda: len(region_data[region_data['tier'] == 'P4'])),
                ('Avg Credit Utilization', lambda: region_data['credit_utilization'].mean() if 'credit_utilization' in region_data else 'N/A'),
                ('Avg Repayment Score', lambda: region_data['repayment_score'].mean() if 'repayment_score' in region_data else 'N/A'),
                ('Total GMV (Last 6M)', lambda: region_data['gmv_6m'].sum() if 'gmv_6m' in region_data else 'N/A'),
                ('Avg DPD', lambda: region_data['avg_dpd'].mean() if 'avg_dpd' in region_data else 'N/A')
            ]
            
            # Calculate metrics with error handling
            metrics = []
            values = []
            for metric_name, metric_func in summary_metrics:
                try:
                    value = metric_func()
                    metrics.append(metric_name)
                    values.append(value)
                except Exception as e:
                    logger.warning(f"Could not calculate metric '{metric_name}': {str(e)}")
            
            summary_data = {
                'Metric': metrics,
                'Value': values
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. Agent Details Sheet
            agent_cols = ['Bzid', 'Name', 'tier', 'tier_description', 'recommended_action']
            # Add optional columns if they exist
            optional_cols = ['credit_utilization', 'repayment_score', 'gmv_6m', 'avg_dpd']
            for col in optional_cols:
                if col in region_data.columns:
                    agent_cols.append(col)
            
            # Only include columns that exist in the dataframe
            existing_cols = [col for col in agent_cols if col in region_data.columns]
            agent_details = region_data[existing_cols].copy()
            agent_details.to_excel(writer, sheet_name='Agent Details', index=False)
            
            # 3. Risk Analysis Sheet
            risk_cols = ['Bzid', 'Name', 'tier']
            optional_risk_cols = ['delinquent_30p', 'delinquent_60p', 'delinquent_90p', 
                               'dpd_trend_3m', 'credit_utilization']
            for col in optional_risk_cols:
                if col in region_data.columns:
                    risk_cols.append(col)
            
            # Only include columns that exist in the dataframe
            existing_risk_cols = [col for col in risk_cols if col in region_data.columns]
            risk_analysis = region_data[existing_risk_cols].copy()
            risk_analysis.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # 4. Performance Metrics Sheet
            perf_cols = ['Bzid', 'Name']
            optional_perf_cols = ['gmv_6m', 'gmv_trend_6m', 'credit_gmv_share',
                               'repayment_score', 'on_time_payment_rate']
            for col in optional_perf_cols:
                if col in region_data.columns:
                    perf_cols.append(col)
            
            # Only include columns that exist in the dataframe
            existing_perf_cols = [col for col in perf_cols if col in region_data.columns]
            perf_metrics = region_data[existing_perf_cols].copy()
            perf_metrics.to_excel(writer, sheet_name='Performance', index=False)
            
            # Format the Excel file
            workbook = writer.book
            
            # Format for headers
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Format for numbers
            num_format = workbook.add_format({'num_format': '#,##0.00'})
            pct_format = workbook.add_format({'num_format': '0.00%'})
            
            # Apply formats to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # Get the appropriate dataframe for this sheet
                if sheet_name == 'Agent Details':
                    df = region_data
                elif sheet_name == 'Performance':
                    df = perf_metrics
                else:
                    df = region_data  # Fallback to region_data for any other sheets
                               
                # Set column widths and formats
                for col_num, column in enumerate(df.columns):
                    try:
                        # Find the maximum length of the column content
                        if not df[column].empty:
                            max_content_length = df[column].astype(str).apply(len).max()
                            column_length = max(max_content_length, len(str(column)))
                        else:
                            column_length = len(str(column))
                        
                        # Set column width with bounds
                        worksheet.set_column(col_num, col_num, min(column_length + 2, 30))
                        
                        # Apply number formats based on column name patterns
                        col_name = str(column).lower()
                        if any(x in col_name for x in ['utilization', 'trend', 'rate', 'ratio']):
                            worksheet.set_column(col_num, col_num, None, pct_format)
                        elif any(x in col_name for x in ['gmv', 'score', 'dpd', 'amount', 'value']):
                            worksheet.set_column(col_num, col_num, None, num_format)
                            
                        # Format header
                        worksheet.write(0, col_num, column, header_format)
                        
                    except Exception as e:
                        logger.warning(f"Could not format column '{column}' in sheet '{sheet_name}': {str(e)}")
                        # Still try to set a basic header even if formatting fails
                        worksheet.write(0, col_num, column, header_format)
            
            return writer
            
        except Exception as e:
            logger.error(f"Error preparing {region_name} report: {str(e)}", exc_info=True)
            raise
    
    def generate_reports(self):
        """
        Generate region-wise Excel reports with agent classifications and metrics.
        
        Returns:
            bool: True if reports were generated successfully, False otherwise
        """
        try:
            if self.classified_agents.empty:
                logger.error("No classified agent data available. Run classify_agents() first.")
                return False
                
            logger.info("Generating region-wise reports...")
            
            # Ensure we have region data
            if 'Region' not in self.classified_agents.columns:
                logger.warning("No 'Region' column found in classified data. Using 'All Regions'.")
                self.classified_agents['Region'] = 'All Regions'
            
            # Group by region and generate reports
            success = True
            for region_name, region_data in self.classified_agents.groupby('Region'):
                try:
                    logger.info(f"Generating report for {region_name}...")
                    writer = self._prepare_region_report(region_data, str(region_name).replace('/', '_'))
                    writer.close()
                    logger.info(f"Successfully generated report for {region_name}")
                except Exception as e:
                    logger.error(f"Failed to generate report for {region_name}: {str(e)}", exc_info=True)
                    success = False
            
            # Generate summary report across all regions
            try:
                logger.info("Generating summary report...")
                writer = self._prepare_region_report(self.classified_agents, "All_Regions_Summary")
                writer.close()
                logger.info("Successfully generated summary report")
            except Exception as e:
                logger.error(f"Failed to generate summary report: {str(e)}", exc_info=True)
                success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error in generate_reports: {str(e)}", exc_info=True)
            return False
    
    def create_email_summaries(self):
        """Create email-ready summaries for each region."""
        logger.info("Creating email summaries...")
        # Email summary generation logic will be implemented here
        pass
    
    def run(self):
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if the pipeline completed successfully, False otherwise
        """
        logger.info("Starting Credit Health Intelligence Engine")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load required input files. Please check the logs.")
            return False
        
        # Engineer features
        if not self.engineer_features():
            logger.error("Feature engineering failed. Check logs for details.")
            return False
            
        try:
            # Classify agents
            if not self.classify_agents():
                logger.error("Agent classification failed. Check logs for details.")
                return False
                
            # Generate reports
            if not self.generate_reports():
                logger.error("Report generation failed. Check logs for details.")
                return False
                
            # Create email summaries
            if not self.create_email_summaries():
                logger.error("Email summary creation failed. Check logs for details.")
                return False
            
            logger.info("Processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point for the script."""
    engine = CreditHealthEngine()
    success = engine.run()
    
    if not success:
        logger.error("Processing failed. Check the logs for details.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
