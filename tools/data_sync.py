"""
Data Synchronization Utility for Credit Risk Analysis

This script provides functionality to synchronize and validate data from multiple sources
used in the credit risk analysis pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.credit_health_engine import CreditHealthEngine, REQUIRED_FILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SOURCE_DIR = Path('source_data')
BACKUP_DIR = Path('data_backups')

class DataSynchronizer:
    """Handles data synchronization and validation for credit risk analysis."""
    
    def __init__(self, source_dir: Path = SOURCE_DIR, backup_dir: Path = BACKUP_DIR):
        """
        Initialize the DataSynchronizer.
        
        Args:
            source_dir: Directory containing source data files
            backup_dir: Directory for storing backups
        """
        self.source_dir = source_dir
        self.backup_dir = backup_dir
        self.engine = CreditHealthEngine()
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Ensure required directories exist."""
        for directory in [self.source_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self) -> str:
        """
        Create a timestamped backup of all source data files.
        
        Returns:
            str: Path to the backup directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'backup_{timestamp}'
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating backup in {backup_path}")
        
        for file in self.source_dir.glob('*'):
            if file.is_file():
                backup_file = backup_path / file.name
                with open(file, 'rb') as src, open(backup_file, 'wb') as dst:
                    dst.write(src.read())
        
        return str(backup_path)
    
    def validate_data(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all source data files.
        
        Returns:
            tuple: (success, issues) where success is a boolean indicating if all files are valid,
                  and issues is a dictionary mapping filenames to lists of issues
        """
        logger.info("Validating data files...")
        issues = {}
        all_valid = True
        
        # Check file existence
        for file in REQUIRED_FILES:
            file_path = self.source_dir / file
            if not file_path.exists():
                issues.setdefault(str(file), []).append("File not found")
                all_valid = False
        
        # If files are missing, don't try to validate further
        if not all_valid:
            return False, issues
        
        # Load and validate each file
        for file in REQUIRED_FILES:
            file_path = self.source_dir / file
            try:
                # Basic validation - just check if file can be loaded
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Check for empty data
                if df.empty:
                    issues.setdefault(str(file), []).append("File is empty")
                    all_valid = False
                
                # TODO: Add more specific validations based on file type
                
            except Exception as e:
                issues.setdefault(str(file), []).append(f"Error loading file: {str(e)}")
                all_valid = False
        
        return all_valid, issues
    
    def sync_data(self, create_backup: bool = True) -> bool:
        """
        Synchronize all data sources.
        
        Args:
            create_backup: Whether to create a backup before syncing
            
        Returns:
            bool: True if synchronization was successful, False otherwise
        """
        logger.info("Starting data synchronization...")
        
        # Create backup if requested
        if create_backup:
            backup_path = self.create_backup()
            logger.info(f"Created backup at: {backup_path}")
        
        # Validate data files
        is_valid, issues = self.validate_data()
        
        if not is_valid:
            logger.error("Data validation failed. Please fix the following issues:")
            for file, file_issues in issues.items():
                for issue in file_issues:
                    logger.error(f"  {file}: {issue}")
            return False
        
        # Load data using the engine
        logger.info("Loading data using CreditHealthEngine...")
        if not self.engine.load_data():
            logger.error("Failed to load data in CreditHealthEngine")
            return False
        
        logger.info("Data synchronization completed successfully")
        return True
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the loaded data.
        
        Returns:
            dict: A dictionary containing summary information
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        for file, df in self.engine.data.items():
            summary['sources'][file] = {
                'rows': len(df),
                'columns': list(df.columns),
                'last_modified': datetime.fromtimestamp(
                    (self.source_dir / file).stat().st_mtime
                ).isoformat()
            }
        
        return summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Synchronize and validate credit risk data')
    parser.add_argument('--no-backup', action='store_true', 
                       help='Skip creating a backup before syncing')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate data without syncing')
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    try:
        synchronizer = DataSynchronizer()
        
        if args.validate_only:
            is_valid, issues = synchronizer.validate_data()
            if is_valid:
                logger.info("✅ All data files are valid")
                return 0
            else:
                logger.error("❌ Data validation failed with the following issues:")
                for file, file_issues in issues.items():
                    for issue in file_issues:
                        logger.error(f"  {file}: {issue}")
                return 1
        else:
            if synchronizer.sync_data(create_backup=not args.no_backup):
                # Print summary
                summary = synchronizer.get_data_summary()
                logger.info("\nData Summary:")
                logger.info("=" * 50)
                for file, info in summary['sources'].items():
                    logger.info(f"{file}:")
                    logger.info(f"  Rows: {info['rows']:,}")
                    logger.info(f"  Columns: {', '.join(info['columns'])}")
                    logger.info(f"  Last Modified: {info['last_modified']}")
                    logger.info("-" * 50)
                return 0
            return 1
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
