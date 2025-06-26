"""
Test script to verify GMV trend calculation and agent metadata lookup.
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.feature_engineering import FeatureEngineer

# Check if data_loader exists, otherwise use direct loading
try:
    from src.data_loader import DataLoader
    HAS_DATA_LOADER = True
except ImportError:
    HAS_DATA_LOADER = False
    logger = logging.getLogger(__name__)
    logger.warning("data_loader module not found. Using direct data loading.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load test data from source files."""
    data_dir = Path("source_data")
    
    if HAS_DATA_LOADER:
        logger.info("Loading data using DataLoader...")
        data_loader = DataLoader(data_dir)
        return data_loader.load_all_data()
    else:
        logger.info("Loading data directly from Excel files...")
        data = {}
        try:
            # List of expected data files
            data_files = [
                'credit_Agents.xlsx',
                'credit_history_sales_vs_credit_sales.xlsx',
                'credit_sales_data.xlsx',
                'DPD.xlsx',
                'Region_contact.xlsx',
                'sales_data.xlsx'
            ]
            
            # Load each file that exists
            for file_name in data_files:
                file_path = data_dir / file_name
                if file_path.exists():
                    logger.info(f"Loading {file_name}...")
                    data[file_name] = pd.read_excel(file_path)
                else:
                    logger.warning(f"File not found: {file_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {}

def test_gmv_trend_calculation():
    """Test the GMV trend calculation."""
    logger.info("Starting GMV trend calculation test...")
    
    # Load test data
    data = load_test_data()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(data)
    
    # Calculate GMV trends
    gmv_trends = feature_engineer.calculate_gmv_trend()
    
    # Print results
    if not gmv_trends.empty:
        logger.info(f"Calculated GMV trends for {len(gmv_trends)} agents")
        
        # Show basic statistics
        logger.info("\nGMV Trend Statistics:")
        logger.info(f"Mean: {gmv_trends.mean():.4f}")
        logger.info(f"Median: {gmv_trends.median():.4f}")
        logger.info(f"Min: {gmv_trends.min():.4f}")
        logger.info(f"Max: {gmv_trends.max():.4f}")
        
        # Show top and bottom agents by trend
        logger.info("\nTop 5 agents by positive GMV trend:")
        logger.info(gmv_trends.nlargest(5).to_string())
        
        logger.info("\nTop 5 agents by negative GMV trend:")
        logger.info(gmv_trends.nsmallest(5).to_string())
        
        # Check for zero trends
        zero_trends = (gmv_trends == 0).sum()
        logger.info(f"\nNumber of agents with zero trend: {zero_trends} ({(zero_trends/len(gmv_trends)*100):.1f}%)")
    else:
        logger.warning("No GMV trends were calculated. Check input data and logs.")

if __name__ == "__main__":
    test_gmv_trend_calculation()
