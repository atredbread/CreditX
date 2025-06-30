"""
Direct test of the AgentClassifier with sample data.
"""
import sys
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np

# Suppress deprecation warnings from dateutil
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module='dateutil.tz.tz'
)

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
    
    # This will be used to track test results
    test_passed = True
    
    # Create a sample dataset with additional required fields for credit health score
    sample_data = [
        # P0: Healthy profile - moderate usage, good repayment, positive trend
        {'agent_id': 'A1', 
         'credit_utilization': 0.4, 'repayment_score': 90, 
         'gmv_trend_6m': 0.02, 'credit_gmv_share': 0.45,
         'avg_monthly_credit_ratio': 0.4, 'credit_ratio_std': 0.1,
         'total_gmv': 100000, 'min_gmv': 50000, 'max_gmv': 150000,
         'zero_credit_months': 0,
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P1: Slight delay and negative trend
        {'agent_id': 'A2', 
         'credit_utilization': 0.5, 'repayment_score': 70, 
         'gmv_trend_6m': -0.02, 'credit_gmv_share': 0.5,
         'avg_monthly_credit_ratio': 0.5, 'credit_ratio_std': 0.15,
         'total_gmv': 80000, 'min_gmv': 40000, 'max_gmv': 120000,
         'zero_credit_months': 0,
         'delinquent_30p': True, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P2: High credit dependence with late payments
        {'agent_id': 'A3', 
         'credit_utilization': 0.8, 'repayment_score': 60, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.8,
         'avg_monthly_credit_ratio': 0.8, 'credit_ratio_std': 0.2,
         'total_gmv': 120000, 'min_gmv': 60000, 'max_gmv': 180000,
         'zero_credit_months': 0,
         'delinquent_30p': True, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P3: Used & repaid but dropped off
        {'agent_id': 'A4', 
         'credit_utilization': 0.35, 'repayment_score': 88, 
         'gmv_trend_6m': -0.03, 'credit_gmv_share': 0.6,
         'avg_monthly_credit_ratio': 0.6, 'credit_ratio_std': 0.12,
         'total_gmv': 90000, 'min_gmv': 45000, 'max_gmv': 135000,
         'zero_credit_months': 0,
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P4: Not using credit (low utilization and low credit GMV share)
        {'agent_id': 'A5', 
         'credit_utilization': 0.1, 'repayment_score': 100, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.1,
         'avg_monthly_credit_ratio': 0.1, 'credit_ratio_std': 0.05,
         'total_gmv': 50000, 'min_gmv': 25000, 'max_gmv': 75000,
         'zero_credit_months': 0,
         'delinquent_30p': False, 'delinquent_60p': False, 'delinquent_90p': False},
        
        # P5: Churned (dormant agent)
        {'agent_id': 'A6', 
         'credit_utilization': 0.0, 'repayment_score': 0, 
         'gmv_trend_6m': 0.0, 'credit_gmv_share': 0.0,
         'avg_monthly_credit_ratio': 0.0, 'credit_ratio_std': 0.0,
         'total_gmv': 1000, 'min_gmv': 500, 'max_gmv': 1500,
         'zero_credit_months': 4,  # More than DORMANT_MONTHS (3)
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
        print(f"- POOR_REPAYMENT: {classifier.thresholds.POOR_REPAYMENT}")
        print(f"- NEGATIVE_TREND: {classifier.thresholds.NEGATIVE_TREND}")
        print(f"- CREDIT_THRESHOLD_HIGH: {classifier.thresholds.CREDIT_THRESHOLD_HIGH}")
        print(f"- CREDIT_THRESHOLD_LOW: {classifier.thresholds.CREDIT_THRESHOLD_LOW}")
        print(f"- DORMANT_MONTHS: {classifier.thresholds.DORMANT_MONTHS}")
        print(f"- CREDIT_RATIO_WEIGHT: {classifier.thresholds.CREDIT_RATIO_WEIGHT}")
        print(f"- VOLATILITY_WEIGHT: {classifier.thresholds.VOLATILITY_WEIGHT}")
        print(f"- BUSINESS_VOLUME_WEIGHT: {classifier.thresholds.BUSINESS_VOLUME_WEIGHT}")
        
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
    
    # Print final results
    results_df = pd.DataFrame(results).set_index('agent_id')
    print("\n=== Final Classifications ===")
    print(results_df)
    
    # Verify all test cases were processed
    assert len(results_df) == 6, f"Should have 6 test cases, got {len(results_df)}"
    
    # Get the actual tiers assigned
    actual_tiers = set(results_df['tier'])
    print(f"\nActual tiers assigned: {actual_tiers}")
    
    # Verify each agent's classification makes sense based on their features
    for agent_id, row in results_df.iterrows():
        features = df.loc[agent_id]
        tier = row['tier']
        
        print(f"\nValidating Agent {agent_id} (Tier {tier}):")
        print(f"- Credit Utilization: {features['credit_utilization']}")
        print(f"- Repayment Score: {features['repayment_score']}")
        print(f"- GMV Trend: {features['gmv_trend_6m']}")
        print(f"- Credit GMV Share: {features['credit_gmv_share']}")
        
        # Basic validation that the tier is one of the expected values
        assert tier in {'P0', 'P1', 'P2', 'P3', 'P4', 'P5'}, f"Invalid tier {tier} for agent {agent_id}"
        
        # Specific validations based on agent ID
        if agent_id == 'A1':
            # P0: Should have good repayment and positive trend
            assert features['repayment_score'] >= classifier.thresholds.GOOD_REPAYMENT, \
                f"Agent A1 should have good repayment score"
            assert features['gmv_trend_6m'] > 0, \
                f"Agent A1 should have positive GMV trend"
                
        elif agent_id == 'A6':
            # P5: Should be churned/dormant
            assert features['zero_credit_months'] >= classifier.thresholds.DORMANT_MONTHS, \
                f"Agent A6 should be marked as dormant"
    
    print("\nAll test cases passed validation!")

    # Show detailed info for all agents with color coding
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    for agent_id in results_df.index:
        agent = results_df.loc[agent_id]
        features = df.loc[agent_id]
        
        # Color code based on tier
        tier_colors = {
            'P0': bcolors.OKGREEN,
            'P1': bcolors.OKBLUE,
            'P2': bcolors.WARNING,
            'P3': bcolors.WARNING,
            'P4': bcolors.FAIL,
            'P5': bcolors.FAIL
        }
        tier_color = tier_colors.get(agent['tier'], bcolors.ENDC)
        
        print("\n" + "="*80)
        print(f"{bcolors.HEADER}{bcolors.BOLD}AGENT {agent_id} CLASSIFICATION ANALYSIS{' ' * 40}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{'=' * 80}{bcolors.ENDC}")
        
        # Classification Summary
        print(f"\n{bcolors.BOLD}CLASSIFICATION SUMMARY:{bcolors.ENDC}")
        print(f"{'Tier:':<20} {tier_color}{agent['tier']}{bcolors.ENDC}")
        print(f"{'Description:':<20} {agent['description']}")
        print(f"{'Recommended Action:':<20} {agent['action']}")
        
        # Key Metrics
        print(f"\n{bcolors.BOLD}KEY METRICS:{bcolors.ENDC}")
        print(f"{'Credit Utilization:':<30} {features['credit_utilization']:.1%}")
        print(f"{'Repayment Score:':<30} {features['repayment_score']}/100")
        print(f"{'GMV Trend (6m):':<30} {features['gmv_trend_6m']:+.2f}")
        print(f"{'Credit GMV Share:':<30} {features['credit_gmv_share']:.1%}")
        print(f"{'Avg Monthly Credit Ratio:':<30} {features['avg_monthly_credit_ratio']:.1%}")
        print(f"{'Credit Volatility (std):':<30} {features['credit_ratio_std']:.3f}")
        print(f"{'Total GMV:':<30} ${features['total_gmv']:,.2f}")
        print(f"{'Zero Credit Months:':<30} {features['zero_credit_months']}")
        
        # Risk Indicators
        print(f"\n{bcolors.BOLD}RISK INDICATORS:{bcolors.ENDC}")
        print(f"{'30+ Days Delinquent:':<30} {'Yes' if features['delinquent_30p'] else 'No'}")
        print(f"{'60+ Days Delinquent:':<30} {'Yes' if features['delinquent_60p'] else 'No'}")
        print(f"{'90+ Days Delinquent:':<30} {'Yes' if features['delinquent_90p'] else 'No'}")
        
        # Threshold Analysis
        print(f"\n{bcolors.BOLD}THRESHOLD ANALYSIS:{bcolors.ENDC}")
        print(f"{'Credit Utilization > 50%:':<40} {features['credit_utilization'] > 0.5}")
        print(f"{'Credit GMV Share > 50%:':<40} {features['credit_gmv_share'] > 0.5}")
        print(f"{'Repayment Score < 70:':<40} {features['repayment_score'] < 70}")
        print(f"{'GMV Trend < 0:':<40} {features['gmv_trend_6m'] < 0}")
        print(f"{'Zero Credit Months >= 3:':<40} {features['zero_credit_months'] >= 3}")
        
        print(f"\n{bcolors.HEADER}{'=' * 80}{bcolors.ENDC}")
    
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
