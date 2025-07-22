"""
Unified Analysis Runner

This script provides a consistent entry point for all analysis tasks,
ensuring proper data processing and JSON output generation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local modules
from scripts.data_processor import DataProcessor
from scripts.utils.json_cache import clear_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analysis_runner.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run credit risk analysis')
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the JSON cache before running analysis'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if cache is valid'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def setup_environment():
    """Set up the required environment."""
    # Create necessary directories
    for directory in ['data/raw', 'data/processed', 'output', 'logs']:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point for the analysis runner."""
    args = parse_arguments()
    setup_environment()
    
    try:
        # Set log level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Clear cache if requested
        if args.clear_cache:
            logger.info("Clearing JSON cache...")
            clear_cache()
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = DataProcessor()
        
        # Load and process data
        logger.info("Loading data...")
        processor.load_data()
        
        # Process all agents
        logger.info("Processing agent data...")
        results = processor.process_all_agents()
        
        # Save results
        output_file = processor.save_results(results)
        logger.info(f"Analysis complete. Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
