"""
Agent Classification Module for Credit Health Intelligence Engine

This module handles the classification of agents into different tiers (P0-P5)
based on their credit behavior and transaction patterns.
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TierThresholds:
    """Threshold values for classifying agents into different tiers."""
    # Credit utilization thresholds (0-1)
    HIGH_UTILIZATION: float = 0.8
    MEDIUM_UTILIZATION: float = 0.4
    
    # Repayment score thresholds (0-100)
    GOOD_REPAYMENT: float = 80.0
    POOR_REPAYMENT: float = 50.0
    
    # GMV trend thresholds (slope of the trend line)
    POSITIVE_TREND: float = 0.1
    NEGATIVE_TREND: float = -0.1
    
    # Credit GMV share thresholds (0-1)
    HIGH_GMV_SHARE: float = 0.7
    LOW_GMV_SHARE: float = 0.3
    
    # Days since last activity (for churn detection)
    CHURN_THRESHOLD_DAYS: int = 90  # 3 months

class AgentClassifier:
    """Classifies agents into different tiers based on their credit behavior."""
    
    def __init__(self, thresholds: Optional[TierThresholds] = None):
        """
        Initialize the agent classifier with optional custom thresholds.
        
        Args:
            thresholds: Custom threshold values for classification.
                       If None, default thresholds will be used.
        """
        self.thresholds = thresholds if thresholds else TierThresholds()
        self.tier_descriptions = {
            'P0': 'High usage + timely repayments',
            'P1': 'Usage decline or slight delays',
            'P2': 'Maxed limit + late payments',
            'P3': 'Used & repaid but dropped off',
            'P4': 'Credit available but not used',
            'P5': 'No use or repayment for a long time'
        }
        self.tier_actions = {
            'P0': 'Nurture',
            'P1': 'Follow-up',
            'P2': 'Monitor / intervene',
            'P3': 'Re-engage',
            'P4': 'Educate / activate',
            'P5': 'Churned — deprioritize'
        }
    
    def classify_agent(self, agent_features: pd.Series) -> Dict[str, str]:
        """
        Classify a single agent based on their features.
        
        Args:
            agent_features: A pandas Series containing the agent's features
                          (credit_utilization, repayment_score, etc.)
                          
        Returns:
            Dictionary containing tier, description, and recommended action
        """
        agent_id = agent_features.name if hasattr(agent_features, 'name') else 'unknown'
        logger.debug(f"\n=== Classifying agent {agent_id} ===")
        logger.debug(f"All features: {agent_features.to_dict()}")
        logger.debug(f"Thresholds: MEDIUM_UTILIZATION={self.thresholds.MEDIUM_UTILIZATION}, "
                   f"HIGH_UTILIZATION={self.thresholds.HIGH_UTILIZATION}, "
                   f"GOOD_REPAYMENT={self.thresholds.GOOD_REPAYMENT}, "
                   f"NEGATIVE_TREND={self.thresholds.NEGATIVE_TREND}")
        
        try:
            # Extract features with default values if missing
            utilization = agent_features.get('credit_utilization', 0)
            repayment_score = agent_features.get('repayment_score', 100)  # Assume good if missing
            gmv_trend = agent_features.get('gmv_trend_6m', 0)
            credit_gmv_share = agent_features.get('credit_gmv_share', 0)
            delinquent_30p = agent_features.get('delinquent_30p', False)
            delinquent_60p = agent_features.get('delinquent_60p', False)
            delinquent_90p = agent_features.get('delinquent_90p', False)
            
            # Log feature values
            logger.debug(f"Utilization: {utilization:.2f}, Repayment: {repayment_score:.1f}, "
                       f"GMV Trend: {gmv_trend:.2f}, Credit GMV Share: {credit_gmv_share:.2f}")
            logger.debug(f"Delinquent (30/60/90+): {delinquent_30p}/{delinquent_60p}/{delinquent_90p}")
            
            # P5: Churned - No recent activity or extreme delinquency
            if delinquent_90p or utilization == 0 or pd.isna(utilization):
                logger.debug("Classified as P5: Churned or no activity")
                return self._get_tier_info('P5')
                
            # P4: Credit available but not used (very low utilization)
            if utilization < 0.1 and credit_gmv_share < 0.1:
                logger.debug("Classified as P4: Credit available but not used")
                return self._get_tier_info('P4')
                
            # P2: Maxed out with late payments
            if (utilization >= self.thresholds.HIGH_UTILIZATION and 
                (delinquent_30p or delinquent_60p)):
                logger.debug("Classified as P2: Maxed out with late payments")
                return self._get_tier_info('P2')
                
            # P1: Usage decline or slight delays
            if (gmv_trend < self.thresholds.NEGATIVE_TREND or 
                (delinquent_30p and not delinquent_60p)):
                logger.debug("Classified as P1: Usage decline or slight delays")
                return self._get_tier_info('P1')
                
            # P3: Used & repaid but dropped off (had significant activity before but declining)
            # Check for agents who had meaningful credit activity but show declining trends
            logger.debug("\nChecking P3 classification...")
            
            # First P3 condition: Moderate utilization with negative trend
            p3_cond1 = (
                f"Utilization > 0.2: {utilization > 0.2}, "
                f"Util < MEDIUM_UTIL*1.2: {utilization < self.thresholds.MEDIUM_UTILIZATION * 1.2}, "
                f"Credit GMV Share > 0.25: {credit_gmv_share > 0.25}, "
                f"GMV Trend < -0.15: {gmv_trend < -0.15}, "
                f"No Delinquency: {not any([delinquent_30p, delinquent_60p, delinquent_90p])}, "
                f"Repayment >= 85: {repayment_score >= 85}"
            )
            logger.debug(f"P3 Condition 1: {p3_cond1}")
            
            if (utilization > 0.2 and  # Some meaningful utilization
                utilization < self.thresholds.MEDIUM_UTILIZATION * 1.2 and  # Below medium utilization (with some buffer)
                credit_gmv_share > 0.25 and  # Had meaningful credit activity (25%+ of GMV was credit)
                gmv_trend < -0.15 and  # Significant negative GMV trend
                not any([delinquent_30p, delinquent_60p, delinquent_90p]) and  # No delinquency
                repayment_score >= 85):  # Good repayment history
                
                logger.debug(f"Classified as P3 (Condition 1) - Util: {utilization}, GMV Trend: {gmv_trend}, "
                            f"Credit GMV Share: {credit_gmv_share}, Repayment: {repayment_score}")
                return self._get_tier_info('P3')
            
            # Second P3 condition: Had high credit activity but now low utilization
            p3_cond2 = (
                f"Credit GMV Share > 0.4: {credit_gmv_share > 0.4}, "
                f"Utilization < 0.3: {utilization < 0.3}, "
                f"GMV Trend < 0: {gmv_trend < 0}, "
                f"No Delinquency: {not any([delinquent_30p, delinquent_60p, delinquent_90p])}, "
                f"Repayment >= 80: {repayment_score >= 80}"
            )
            logger.debug(f"P3 Condition 2: {p3_cond2}")
                
            if (credit_gmv_share > 0.4 and  # Had very high credit activity
                utilization < 0.3 and  # Now low utilization
                gmv_trend < 0 and  # Some decline
                not any([delinquent_30p, delinquent_60p, delinquent_90p]) and
                repayment_score >= 80):
                
                logger.debug(f"Classified as P3 (Condition 2) - Util: {utilization}, "
                            f"Credit GMV Share: {credit_gmv_share}, GMV Trend: {gmv_trend}")
                return self._get_tier_info('P3')
                
            logger.debug("Agent does not meet P3 criteria")
                
            # P0: High usage with good repayment
            if (utilization >= self.thresholds.MEDIUM_UTILIZATION and 
                repayment_score >= self.thresholds.GOOD_REPAYMENT and 
                not delinquent_30p):
                logger.debug("Classified as P0: High usage with good repayment")
                return self._get_tier_info('P0')
                
            # Default to P1 if no other conditions met
            logger.debug("No specific tier matched, defaulting to P1")
            return self._get_tier_info('P1')
            
        except Exception as e:
            logger.error(f"Error classifying agent: {str(e)}")
            # Return a default tier in case of errors
            return self._get_tier_info('P1')
    
    def classify_all_agents(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all agents in the features DataFrame.
        
        Args:
            features_df: DataFrame containing features for all agents
                        (rows = agents, columns = features)
                        
        Returns:
            DataFrame with additional columns for tier, description, and action
        """
        if features_df.empty:
            logger.warning("Empty features DataFrame provided for classification")
            return pd.DataFrame()
            
        # Make a copy to avoid modifying the input
        results = features_df.copy()
        
        # Initialize new columns
        results['tier'] = 'P1'  # Default tier
        results['tier_description'] = self.tier_descriptions['P1']
        results['recommended_action'] = self.tier_actions['P1']
        
        # Classify each agent
        for idx, agent_id in enumerate(results.index):
            try:
                classification = self.classify_agent(results.loc[agent_id])
                results.at[agent_id, 'tier'] = classification['tier']
                results.at[agent_id, 'tier_description'] = classification['description']
                results.at[agent_id, 'recommended_action'] = classification['action']
                
                # Log progress
                if (idx + 1) % 100 == 0:
                    logger.info(f"Classified {idx + 1}/{len(results)} agents")
                    
            except Exception as e:
                logger.error(f"Error classifying agent {agent_id}: {str(e)}")
        
        logger.info(f"Classification complete. Tier distribution:\n{results['tier'].value_counts().sort_index()}")
        return results
    
    def _get_tier_info(self, tier: str) -> Dict[str, str]:
        """
        Get tier information including description and recommended action.
        
        Args:
            tier: Tier identifier (P0-P5)
            
        Returns:
            Dictionary with tier, description, and action
        """
        return {
            'tier': tier,
            'description': self.tier_descriptions.get(tier, 'Unknown'),
            'action': self.tier_actions.get(tier, 'Investigate')
        }
