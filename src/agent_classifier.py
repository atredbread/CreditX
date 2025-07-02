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
    HIGH_UTILIZATION: float = 0.5  # 50% threshold for high credit dependence
    MEDIUM_UTILIZATION: float = 0.3  # 30% threshold for low credit utilization
    
    # Repayment score thresholds (0-100)
    GOOD_REPAYMENT: float = 80.0
    POOR_REPAYMENT: float = 50.0
    
    # GMV trend thresholds (slope of the trend line)
    POSITIVE_TREND: float = 0.01  # 1% increase
    NEGATIVE_TREND: float = -0.01  # 1% decrease
    
    # Credit GMV share thresholds (0-1)
    HIGH_GMV_SHARE: float = 0.5  # 50% threshold for high credit GMV share
    LOW_GMV_SHARE: float = 0.25  # 25% threshold for low credit GMV share
    
    # Credit Health Score weights
    CREDIT_RATIO_WEIGHT: float = 0.4
    VOLATILITY_WEIGHT: float = 0.3
    BUSINESS_VOLUME_WEIGHT: float = 0.3
    
    # Risk indicators
    DORMANT_MONTHS: int = 3  # Number of months with no credit activity
    CREDIT_THRESHOLD_HIGH: float = 0.5  # 50% threshold for high credit dependence
    CREDIT_THRESHOLD_LOW: float = 0.3  # 30% threshold for low credit utilization

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
    
    def _calculate_credit_health_score(self, agent_features: pd.Series) -> float:
        """
        Calculate the credit health score based on documentation.
        
        Args:
            agent_features: Agent features including metrics
            
        Returns:
            Credit health score (0-100)
        """
        # Extract metrics with defaults
        avg_credit_ratio = agent_features.get('avg_monthly_credit_ratio', 0)
        credit_ratio_std = agent_features.get('credit_ratio_std', 0)
        total_gmv = agent_features.get('total_gmv', 0)
        repayment_score = agent_features.get('repayment_score', 0)
        credit_utilization = agent_features.get('credit_utilization', 0)
        
        # Log input metrics
        logger.debug(f"Calculating credit health score for agent {agent_features.get('agent_id', 'unknown')}:")
        logger.debug(f"  - Avg Credit Ratio: {avg_credit_ratio:.2f}")
        logger.debug(f"  - Credit Ratio Std: {credit_ratio_std:.2f}")
        logger.debug(f"  - Total GMV: {total_gmv:,.0f}")
        logger.debug(f"  - Repayment Score: {repayment_score:.0f}")
        logger.debug(f"  - Credit Utilization: {credit_utilization:.1%}")
        
        # 1. Credit Ratio Score (40% weight)
        # Moderate utilization is best (around 30-50%)
        # Score peaks at 1 for 40% utilization, decreases as it moves away
        ideal_utilization = 0.4
        utilization_diff = abs(avg_credit_ratio - ideal_utilization)
        score_credit_ratio = max(0, 1 - (utilization_diff / ideal_utilization))
        
        # 2. Volatility Score (30% weight)
        # Lower volatility is better
        score_volatility = 1 - min(1, credit_ratio_std * 2)  # Scale down the impact of volatility
        
        # 3. Repayment Score (30% weight)
        # Higher repayment score is better
        score_repayment = repayment_score / 100.0
        
        # Calculate final weighted score (0-100)
        credit_health_score = (
            score_credit_ratio * 0.4 +  # 40% weight for credit ratio
            score_volatility * 0.3 +    # 30% weight for volatility
            score_repayment * 0.3       # 30% weight for repayment
        ) * 100
        
        # Log the score components and final score
        logger.debug("Score Components:")
        logger.debug(f"  - Credit Ratio: {score_credit_ratio:.2f} (40%)")
        logger.debug(f"  - Volatility: {score_volatility:.2f} (30%)")
        logger.debug(f"  - Repayment: {score_repayment:.2f} (30%)")
        logger.debug(f"  - Final Score: {credit_health_score:.1f}/100")
        
        return min(max(credit_health_score, 0), 100)  # Clamp between 0-100

    def _get_risk_indicators(self, agent_features: pd.Series) -> Dict[str, bool]:
        """
        Calculate risk indicators for an agent.
        
        Args:
            agent_features: Agent features including metrics
            
        Returns:
            Dictionary of risk indicators
        """
        credit_utilization = agent_features.get('credit_utilization', 0)
        zero_credit_months = agent_features.get('zero_credit_months', 0)
        
        high_credit_dependence = credit_utilization > self.thresholds.CREDIT_THRESHOLD_HIGH
        low_credit_utilization = credit_utilization < self.thresholds.CREDIT_THRESHOLD_LOW
        dormant_agent = zero_credit_months >= self.thresholds.DORMANT_MONTHS
        
        logger.debug(f"Risk Indicators for {agent_features.name if hasattr(agent_features, 'name') else 'unknown'}:")
        logger.debug(f"  - Credit Utilization: {credit_utilization:.2f} (Threshold: {self.thresholds.CREDIT_THRESHOLD_HIGH:.2f})")
        logger.debug(f"  - High Credit Dependence: {high_credit_dependence} (Utilization > {self.thresholds.CREDIT_THRESHOLD_HIGH})")
        logger.debug(f"  - Low Credit Utilization: {low_credit_utilization} (Utilization < {self.thresholds.CREDIT_THRESHOLD_LOW})")
        logger.debug(f"  - Dormant Agent: {dormant_agent} (Zero Credit Months: {zero_credit_months} >= {self.thresholds.DORMANT_MONTHS})")
        
        return {
            'high_credit_dependence': high_credit_dependence,
            'low_credit_utilization': low_credit_utilization,
            'dormant_agent': dormant_agent
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
        
        try:
            # Create a dictionary to store working data
            working_data = agent_features.to_dict()
            
            # Calculate credit health score if not provided
            if 'credit_health_score' not in working_data:
                credit_health_score = self._calculate_credit_health_score(pd.Series(working_data))
                working_data['credit_health_score'] = credit_health_score
            else:
                credit_health_score = working_data['credit_health_score']
            
            # Get risk indicators using a Series created from working_data
            risk_indicators = self._get_risk_indicators(pd.Series(working_data))
            
            # Extract features with default values if missing
            utilization = working_data.get('credit_utilization', 0)
            repayment_score = working_data.get('repayment_score', 100)  # Assume good if missing
            gmv_trend = working_data.get('gmv_trend_6m', 0)
            credit_gmv_share = working_data.get('credit_gmv_share', 0)
            delinquent_30p = working_data.get('delinquent_30p', False)
            delinquent_60p = working_data.get('delinquent_60p', False)
            delinquent_90p = working_data.get('delinquent_90p', False)
            
            # Log feature values with more detail
            logger.debug("\n=== AGENT FEATURES ===")
            logger.debug(f"Agent ID: {agent_id}")
            logger.debug(f"Credit Utilization: {utilization:.2f} (Thresholds: <{self.thresholds.CREDIT_THRESHOLD_LOW:.2f}=low, >{self.thresholds.CREDIT_THRESHOLD_HIGH:.2f}=high)")
            logger.debug(f"Repayment Score: {repayment_score:.1f} (Good: >={self.thresholds.GOOD_REPAYMENT})")
            logger.debug(f"GMV Trend (6m): {gmv_trend:.4f} (Negative: <{self.thresholds.NEGATIVE_TREND:.4f})")
            logger.debug(f"Credit GMV Share: {credit_gmv_share:.2f} (Low: <{self.thresholds.LOW_GMV_SHARE:.2f}, High: >{self.thresholds.HIGH_GMV_SHARE:.2f})")
            logger.debug(f"Credit Health Score: {credit_health_score:.1f}/100 (Good: >=70)")
            logger.debug(f"Risk Indicators: {risk_indicators}")
            logger.debug(f"Delinquent (30/60/90+): {delinquent_30p}/{delinquent_60p}/{delinquent_90p}")
            logger.debug("====================\n")
            
            # P5: Churned - No recent activity or extreme delinquency
            if (delinquent_90p or 
                risk_indicators['dormant_agent'] or 
                utilization == 0 or 
                pd.isna(utilization)):
                logger.debug("Classified as P5: Churned or no activity")
                return self._get_tier_info('P5')
                
            # P4: Credit available but not used (very low utilization)
            if risk_indicators['low_credit_utilization'] and credit_gmv_share < self.thresholds.LOW_GMV_SHARE:
                logger.debug("Classified as P4: Credit available but not used")
                return self._get_tier_info('P4')
                
            # P2: High credit dependence with late payments
            if (risk_indicators['high_credit_dependence'] and 
                (delinquent_30p or delinquent_60p)):
                logger.debug("Classified as P2: High credit dependence with late payments")
                return self._get_tier_info('P2')
                
            # P0: High usage with good repayment and healthy metrics
            # Should be checked before P1 and P3 to prevent early classification
            p0_conditions = {
                'utilization > 0': utilization > 0,
                'not high_credit_dependence': not risk_indicators['high_credit_dependence'],
                'repayment_score >= GOOD_REPAYMENT': repayment_score >= self.thresholds.GOOD_REPAYMENT,
                'credit_health_score >= 70': credit_health_score >= 70,
                'no delinquency': not any([delinquent_30p, delinquent_60p, delinquent_90p])
            }
            
            logger.debug(f"P0 Classification Check for {agent_id}:")
            for condition, result in p0_conditions.items():
                logger.debug(f"  - {condition}: {result}")
            
            if all(p0_conditions.values()):
                logger.debug("Classified as P0: Healthy credit profile")
                return self._get_tier_info('P0')
            else:
                logger.debug("Not classified as P0 due to failed conditions")
                for condition, result in p0_conditions.items():
                    if not result:
                        logger.debug(f"  - Failed condition: {condition}")
                        break
            
            # P3: Used & repaid but dropped off
            # Check for agents who had meaningful credit activity but show declining trends
            # Should be checked before P1 to prevent early classification
            if (credit_gmv_share > self.thresholds.LOW_GMV_SHARE and  # Had meaningful credit activity
                gmv_trend < self.thresholds.NEGATIVE_TREND and  # Negative trend
                not any([delinquent_30p, delinquent_60p, delinquent_90p]) and  # No delinquency
                repayment_score >= self.thresholds.GOOD_REPAYMENT):  # Good repayment history
                
                logger.debug(f"Classified as P3: Used & repaid but dropped off - "
                           f"Credit GMV Share: {credit_gmv_share:.2f}, "
                           f"GMV Trend: {gmv_trend:.2f}")
                return self._get_tier_info('P3')
            
            # P1: Usage decline or slight delays
            if (gmv_trend < self.thresholds.NEGATIVE_TREND or 
                (delinquent_30p and not delinquent_60p) or
                credit_health_score < self.thresholds.GOOD_REPAYMENT):  # Changed from POOR_REPAYMENT to GOOD_REPAYMENT
                logger.debug("Classified as P1: Usage decline or slight delays")
                return self._get_tier_info('P1')
                
            # Default to P1 if no other conditions met
            logger.debug("No specific tier matched, defaulting to P1")
            return self._get_tier_info('P1')
            
        except Exception as e:
            logger.error(f"Error classifying agent {agent_id}: {str(e)}")
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
