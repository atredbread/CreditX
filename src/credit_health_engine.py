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
    
    def load_data(self) -> bool:
        """
        Load and validate all required input files.
        
        Returns:
            bool: True if all files were loaded and validated successfully, False otherwise
        """
        logger.info("Loading and validating input data...")
        missing_files = []
        validation_errors = []
        
        for file_name, file_spec in REQUIRED_FILES.items():
            file_path = INPUT_DIR / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                missing_files.append(file_name)
                continue
                
            try:
                logger.info(f"Loading {file_name}...")
                
                # Load the file
                df = self._load_excel_file(file_path, file_spec)
                if df is None or df.empty:
                    logger.error(f"Failed to load or empty DataFrame: {file_name}")
                    validation_errors.append(file_name)
                    continue
                
                # Validate the loaded data
                if not self._validate_dataframe(df, file_name):
                    validation_errors.append(file_name)
                    continue
                
                # Store the loaded data
                self.data[file_name] = df
                
                # Log successful load with shape and sample data (using ASCII symbols for Windows compatibility)
                logger.info(f"[OK] Successfully loaded {file_name} with shape {df.shape}")
                
                # Log column information for debugging
                logger.debug(f"Columns in {file_name}: {', '.join(df.columns)}")
                
                # For credit history, log month columns info
                if file_name == 'Credit_history_sales_vs_credit_sales.xlsx':
                    month_cols = file_spec.get('month_columns', {})
                    logger.debug(f"Month columns found: {month_cols}")
                
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                validation_errors.append(file_name)
        
        # Generate summary report
        loaded_files = set(self.data.keys())
        expected_files = set(REQUIRED_FILES.keys())
        
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING SUMMARY")
        logger.info("="*50)
        
        # Report loaded files
        if loaded_files:
            logger.info("\n[OK] Successfully loaded files:")
            for f in sorted(loaded_files):
                logger.info(f"  - {f}: {len(self.data[f])} rows")
        
        # Report missing files
        if missing_files:
            logger.error("\n[ERROR] Missing files:")
            for f in sorted(missing_files):
                logger.error(f"  - {f}")
        
        # Report validation errors
        if validation_errors:
            logger.error("\n[ERROR] Files with validation errors:")
            for f in sorted(validation_errors):
                logger.error(f"  - {f}")
        
        # Check if all required files were loaded successfully
        all_loaded = len(missing_files) == 0 and len(validation_errors) == 0
        
        if all_loaded:
            logger.info("\n✅ All files loaded and validated successfully!")
        else:
            logger.error("\n[ERROR] Some files failed to load or validate. Check logs for details.")
        
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
    
    def generate_reports(self):
        """Generate region-wise Excel reports."""
        logger.info("Generating reports...")
        # Report generation logic will be implemented here
        pass
    
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
