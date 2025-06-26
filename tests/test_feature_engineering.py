"""
Test script for feature engineering in Credit Health Engine.
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.credit_health_engine import CreditHealthEngine
from src.feature_engineering import FeatureEngineer

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler - always create new log file with timestamp
log_file = logs_dir / f'feature_engineering_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler with proper encoding
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# Set higher level for console to avoid debug messages
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """Test the feature engineering pipeline."""
    logger.info("Starting feature engineering test...")
    
    # Initialize the engine
    engine = CreditHealthEngine()
    
    # Load the data
    logger.info("Loading data...")
    if not engine.load_data():
        logger.error("Failed to load data")
        return False
    
    # Initialize the feature engineer
    logger.info("Initializing feature engineer...")
    feature_engineer = FeatureEngineer(engine.data)
    
    # Generate features
    logger.info("Engineering features...")
    features = feature_engineer.engineer_all_features()
    
    # Check if features were generated
    if features.empty:
        logger.error("No features were generated")
        return False
    
    # Print feature summary
    logger.info("\n=== Feature Engineering Test Results ===")
    logger.info(f"Generated {len(features.columns)} features for {len(features)} agents")
    logger.info("\nFeature summary:")
    logger.info(features.describe().to_string())
    
    # Check for missing values
    missing = features.isnull().sum()
    if missing.sum() > 0:
        logger.warning("\nFeatures with missing values:")
        logger.warning(missing[missing > 0].to_string())
    else:
        logger.info("\nNo missing values in features")
    
    # Save features to CSV for inspection
    output_dir = Path('output/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / 'engineered_features.csv'
    features.to_csv(features_path)
    logger.info(f"\nFeatures saved to: {features_path}")
    
    return True

def main():
    """Run the feature engineering test."""
    try:
        success = test_feature_engineering()
        if success:
            logger.info("Feature engineering test completed successfully!")
        else:
            logger.error("\n❌ Feature engineering test failed")
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error during feature engineering test: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
