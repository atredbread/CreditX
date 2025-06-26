"""
Test script for agent classification in Credit Health Engine.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.credit_health_engine import CreditHealthEngine
from src.agent_classifier import AgentClassifier, TierThresholds

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
log_file = logs_dir / f'agent_classification_test_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler with proper encoding
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def test_agent_classification():
    """Test the agent classification pipeline."""
    logger.info("Starting agent classification test...")
    
    # Initialize the engine
    engine = CreditHealthEngine()
    
    # Load the data
    logger.info("Loading data...")
    if not engine.load_data():
        logger.error("Failed to load data")
        return False
    
    # Engineer features
    logger.info("Engineering features...")
    if not engine.engineer_features():
        logger.error("Feature engineering failed")
        return False
    
    # Classify agents
    logger.info("Classifying agents...")
    if not engine.classify_agents():
        logger.error("Agent classification failed")
        return False
    
    # Print classification summary
    if hasattr(engine, 'classified_agents') and not engine.classified_agents.empty:
        logger.info("\n=== Agent Classification Results ===")
        
        # Tier distribution
        tier_counts = engine.classified_agents['tier'].value_counts().sort_index()
        logger.info("\nTier Distribution:")
        for tier, count in tier_counts.items():
            logger.info(f"{tier}: {count} agents ({count/len(engine.classified_agents):.1%})")
        
        # Sample of classified agents
        logger.info("\nSample Classifications:")
        sample = engine.classified_agents.sample(min(5, len(engine.classified_agents)))
        for idx, row in sample.iterrows():
            logger.info(f"Agent {idx}: Tier {row['tier']} - {row['tier_description']}")
            logger.info(f"   Action: {row['recommended_action']}")
            logger.info(f"   Features: util={row.get('credit_utilization', 'N/A'):.2f}, "
                      f"repay={row.get('repayment_score', 'N/A')}, "
                      f"gmv_share={row.get('credit_gmv_share', 'N/A'):.2f}")
        
        return True
    else:
        logger.error("No agents were classified")
        return False

def test_classifier_direct():
    """Test the AgentClassifier directly with sample data."""
    logger.info("\nTesting AgentClassifier with sample data...")
    
    # Create sample data
    sample_data = [
        # P0: High usage, good repayment
        {'credit_utilization': 0.9, 'repayment_score': 95, 'gmv_trend_6m': 0.2, 'credit_gmv_share': 0.8},
        # P1: Slight delay
        {'credit_utilization': 0.6, 'repayment_score': 75, 'gmv_trend_6m': -0.1, 'credit_gmv_share': 0.5, 'delinquent_30p': True},
        # P2: Maxed out with late payments
        {'credit_utilization': 0.95, 'repayment_score': 60, 'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.9, 'delinquent_30p': True},
        # P3: Dropped off
        {'credit_utilization': 0.3, 'repayment_score': 85, 'gmv_trend_6m': -0.3, 'credit_gmv_share': 0.4},
        # P4: Not using credit
        {'credit_utilization': 0.05, 'repayment_score': 100, 'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.05},
        # P5: Churned
        {'credit_utilization': 0.0, 'repayment_score': 0, 'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.0}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize classifier
    classifier = AgentClassifier()
    
    # Classify each sample
    for idx, row in df.iterrows():
        result = classifier.classify_agent(row)
        logger.info(f"\nSample {idx + 1}:")
        logger.info(f"Features: {row.to_dict()}")
        logger.info(f"Classification: Tier {result['tier']} - {result['description']}")
        logger.info(f"Recommended Action: {result['action']}")
    
    return True

def main():
    """Run the agent classification tests."""
    try:
        # Test with real data
        logger.info("=== Testing with Real Data ===")
        success_real = test_agent_classification()
        
        # Test with sample data
        logger.info("\n=== Testing with Sample Data ===")
        success_sample = test_classifier_direct()
        
        if success_real and success_sample:
            logger.info("\n✅ All tests completed successfully!")
        else:
            logger.error("\n❌ Some tests failed. Check the logs for details.")
        
        return 0 if (success_real and success_sample) else 1
        
    except Exception as e:
        logger.error(f"Error during agent classification test: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
