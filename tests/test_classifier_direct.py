"""
Direct test of the AgentClassifier with sample data.
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

# Configure logging with debug level
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set the agent_classifier logger to DEBUG level
logging.getLogger('src.agent_classifier').setLevel(logging.DEBUG)

# Import from src package
from src.agent_classifier import AgentClassifier, TierThresholds

def test_classifier():
    """Test the AgentClassifier with sample data."""
    from src.agent_classifier import AgentClassifier, TierThresholds
    
    # Create a sample dataset
    sample_data = [
        # P0: High usage, good repayment
        {'agent_id': 'A1', 'credit_utilization': 0.85, 'repayment_score': 90, 
         'gmv_trend_6m': 0.15, 'credit_gmv_share': 0.8, 
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P1: Slight delay
        {'agent_id': 'A2', 'credit_utilization': 0.6, 'repayment_score': 70, 
         'gmv_trend_6m': -0.05, 'credit_gmv_share': 0.5, 
         'delinquent_30p': True, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P2: Maxed out with late payments
        {'agent_id': 'A3', 'credit_utilization': 0.95, 'repayment_score': 60, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.9, 
         'delinquent_30p': True, 'delinquent_60p': True, 'delinquent_90p': False},
        
        # P3: Used & repaid but dropped off (had significant activity before but declining)
        {'agent_id': 'A4', 'credit_utilization': 0.35, 'repayment_score': 88, 
         'gmv_trend_6m': -0.25, 'credit_gmv_share': 0.6,  # High past credit activity, now declining
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},  # No current delinquency
        
        # P4: Not using credit
        {'agent_id': 'A5', 'credit_utilization': 0.05, 'repayment_score': 100, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.05, 
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P5: Churned
        {'agent_id': 'A6', 'credit_utilization': 0.0, 'repayment_score': 0, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.0, 
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': True}
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_data).set_index('agent_id')
    
    # Initialize classifier
    classifier = AgentClassifier()
    
    # Classify agents with detailed logging
    results = []
    for agent_id, row in df.iterrows():
        print(f"\n=== Classifying Agent {agent_id} ===")
        print(f"Features: {row.to_dict()}")
        
        # Log thresholds for reference
        print("\nCurrent Thresholds:")
        print(f"- MEDIUM_UTILIZATION: {classifier.thresholds.MEDIUM_UTILIZATION}")
        print(f"- HIGH_UTILIZATION: {classifier.thresholds.HIGH_UTILIZATION}")
        print(f"- GOOD_REPAYMENT: {classifier.thresholds.GOOD_REPAYMENT}")
        print(f"- NEGATIVE_TREND: {classifier.thresholds.NEGATIVE_TREND}")
        
        # Classify the agent
        print("\nClassification Steps:")
        classification = classifier.classify_agent(row)
        
        # Add to results
        results.append({
            'agent_id': agent_id,
            'tier': classification['tier'],
            'description': classification['description'],
            'action': classification['action']
        })
        
        print(f"\nFinal Classification: {classification['tier']} - {classification['description']}")
        print(f"Recommended action: {classification['action']}")
    
    # Display results
    results_df = pd.DataFrame(results).set_index('agent_id')
    print("\n=== Classification Results ===")
    print(results_df[['tier', 'description', 'action']])
    
    # Show detailed info for agent A4
    if 'A4' in results_df.index:
        agent4 = results_df.loc['A4']
        agent4_features = df.loc['A4']
        
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION ANALYSIS FOR AGENT A4")
        print("="*50)
        
        print("\nCLASSIFICATION RESULT:")
        print(f"- Tier: {agent4['tier']}")
        print(f"- Description: {agent4['description']}")
        print(f"- Recommended Action: {agent4['action']}")
        
        print("\nFEATURES USED FOR CLASSIFICATION:")
        for feature, value in agent4_features.items():
            print(f"- {feature}: {value}")
            
        print("\nTHRESHOLDS:")
        print(f"- MEDIUM_UTILIZATION: {classifier.thresholds.MEDIUM_UTILIZATION}")
        print(f"- HIGH_UTILIZATION: {classifier.thresholds.HIGH_UTILIZATION}")
        print(f"- GOOD_REPAYMENT: {classifier.thresholds.GOOD_REPAYMENT}")
        print(f"- NEGATIVE_TREND: {classifier.thresholds.NEGATIVE_TREND}")
        
        print("\n" + "="*50 + "\n")
    
    # Validate results
    expected_tiers = {
        'A1': 'P0', 'A2': 'P1', 'A3': 'P2',
        'A4': 'P3', 'A5': 'P4', 'A6': 'P5'
    }
    
    print("\n=== Validation ===")
    all_correct = True
    for agent_id, expected_tier in expected_tiers.items():
        actual_tier = results_df.loc[agent_id, 'tier']
        if actual_tier != expected_tier:
            print(f"❌ {agent_id}: Expected {expected_tier}, got {actual_tier}")
            all_correct = False
        else:
            print(f"✅ {agent_id}: Correctly classified as {actual_tier}")
    
    if all_correct:
        print("\n🎉 All agents classified correctly!")
    else:
        print("\n❌ Some classifications are incorrect.")
    
    return all_correct

if __name__ == "__main__":
    test_classifier()
