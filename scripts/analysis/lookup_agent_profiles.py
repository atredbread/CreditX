"""
Agent Profile Lookup Tool

This script provides an interactive interface to look up agent profiles
and display their credit risk analysis using the unified agent data store.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import json
import re
import math
from pathlib import Path
from tabulate import tabulate
from colorama import init, Fore, Style

# Add project root to Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent classifier and data store
try:
    from src.agent_classifier import AgentClassifier
    from src.agent_data_store import AgentDataStore
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running the script from the project root directory.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    sys.exit(1)

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_lookup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()

class AgentReporter:
    """Generates agent-specific credit risk analysis reports."""
    
    def __init__(self, agent_data: Dict):
        """Initialize with agent data."""
        self.agent_data = agent_data
        self.metrics = {}
        self.assessments = []
        self.findings = []
        self.recommendations = []
    
    def calculate_credit_usage_trend(self) -> Dict[str, Any]:
        """Calculate credit usage trend over time.
        
        Returns:
            Dict containing trend analysis with:
            - trend: 'increasing', 'decreasing', or 'stable'
            - rate: Monthly change rate
            - months: Number of months of data
            - confidence: Confidence level of the trend (0-1)
        """
        # Default values if no trend data is available
        default_trend = {
            'trend': 'stable',
            'rate': 0.0,
            'months': 0,
            'confidence': 0.0
        }
        
        try:
            # Check if we have historical credit usage data
            if 'credit_history' not in self.agent_data or not self.agent_data['credit_history']:
                return default_trend
                
            # Get historical credit usage (assuming list of {'date': 'YYYY-MM-DD', 'balance': float})
            history = sorted(
                self.agent_data['credit_history'],
                key=lambda x: x.get('date', '')
            )
            
            if len(history) < 2:
                return default_trend
                
            # Calculate monthly changes
            changes = []
            for i in range(1, len(history)):
                prev_balance = float(history[i-1].get('balance', 0))
                curr_balance = float(history[i].get('balance', 0))
                if prev_balance > 0:
                    change = (curr_balance - prev_balance) / prev_balance
                    changes.append(change)
            
            if not changes:
                return default_trend
                
            # Calculate average monthly change
            avg_change = sum(changes) / len(changes)
            
            # Determine trend
            if abs(avg_change) < 0.01:  # Less than 1% change is considered stable
                trend = 'stable'
            elif avg_change > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
                
            # Calculate confidence (based on consistency of direction)
            consistent_changes = sum(1 for c in changes if (c > 0) == (avg_change > 0))
            confidence = consistent_changes / len(changes)
            
            return {
                'trend': trend,
                'rate': avg_change,
                'months': len(history),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Error calculating credit usage trend: {e}")
            return default_trend
    
    def calculate_sales_trend(self) -> Dict[str, Any]:
        """Calculate sales trend over time.
        
        Returns:
            Dict containing sales trend analysis with:
            - trend: 'growing', 'declining', or 'stable'
            - rate: Monthly growth rate
            - months: Number of months of data
            - confidence: Confidence level of the trend (0-1)
        """
        # Default values if no trend data is available
        default_trend = {
            'trend': 'stable',
            'rate': 0.0,
            'months': 0,
            'confidence': 0.0
        }
        
        try:
            # Check if we have historical sales data
            if 'sales_history' not in self.agent_data or not self.agent_data['sales_history']:
                return default_trend
                
            # Get historical sales (assuming list of {'date': 'YYYY-MM-DD', 'amount': float})
            history = sorted(
                self.agent_data['sales_history'],
                key=lambda x: x.get('date', '')
            )
            
            if len(history) < 2:
                return default_trend
                
            # Calculate monthly changes
            changes = []
            for i in range(1, len(history)):
                prev_sales = float(history[i-1].get('amount', 0))
                curr_sales = float(history[i].get('amount', 0))
                if prev_sales > 0:
                    change = (curr_sales - prev_sales) / prev_sales
                    changes.append(change)
            
            if not changes:
                return default_trend
                
            # Calculate average monthly change
            avg_change = sum(changes) / len(changes)
            
            # Determine trend
            if abs(avg_change) < 0.01:  # Less than 1% change is considered stable
                trend = 'stable'
            elif avg_change > 0:
                trend = 'growing'
            else:
                trend = 'declining'
                
            # Calculate confidence (based on consistency of direction)
            consistent_changes = sum(1 for c in changes if (c > 0) == (avg_change > 0))
            confidence = consistent_changes / len(changes)
            
            return {
                'trend': trend,
                'rate': avg_change,
                'months': len(history),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sales trend: {e}")
            return default_trend
    
    def analyze_credit_utilization(self) -> None:
        """Analyze credit utilization metrics including GMV and credit dependence.
        
        Calculates:
        - Credit Utilization Ratio (Balance/Limit)
        - GMV (Credit Limit - Credit Line Balance)
        - Credit Dependence (Credit GMV / Total GMV if available)
        - Credit Usage Trend (over time)
        
        Risk levels based on utilization:
        - <30%: High risk (underutilization)
        - 30-70%: Medium risk (optimal utilization)
        - >70%: Low risk (good utilization)
        """
        # Initialize metrics
        self.metrics['credit_utilization_ratio'] = None
        self.metrics['gmv'] = None
        self.metrics['credit_dependence'] = None
        
        if 'Credit Limit' in self.agent_data and 'Credit Line Balance' in self.agent_data:
            try:
                limit = float(self.agent_data['Credit Limit'])
                balance = float(self.agent_data['Credit Line Balance'])
                
                # Calculate GMV (Credit Used)
                gmv = max(0, limit - balance)  # Ensure non-negative
                self.metrics['gmv'] = gmv
                
                # Add GMV to assessments
                self.assessments.append({
                    'metric': 'GMV (Credit Used)',
                    'value': f"{gmv:,.2f}",
                    'risk': RiskLevel.LOW  # Just informational, no risk
                })
                
                # Calculate Credit Utilization Ratio if limit > 0
                if limit > 0:
                    utilization = (balance / limit) * 100
                    self.metrics['credit_utilization_ratio'] = utilization
                    
                    # Updated risk levels based on new criteria
                    if utilization < 30:
                        risk = RiskLevel.HIGH
                        risk_label = "High (Underutilization)"
                        self.findings.append("High Risk: Credit utilization below 30% indicates underutilization")
                        self.recommendations.append("Consider increasing credit line usage or reviewing credit needs")
                    elif utilization <= 70:
                        risk = RiskLevel.MODERATE
                        risk_label = "Medium (Optimal)"
                        self.findings.append("Optimal credit utilization range (30-70%)")
                    else:
                        risk = RiskLevel.LOW
                        risk_label = "Low (Good Utilization)"
                        self.findings.append("Good credit utilization (above 70%)")
                        self.recommendations.append("Monitor for any sudden changes in utilization patterns")
                    
                    self.assessments.append({
                        'metric': 'Credit Utilization',
                        'value': f"{utilization:.1f}% - {risk_label}",
                        'risk': risk
                    })
                
                # Calculate Credit Dependence if Total GMV is available
                if 'Total GMV' in self.agent_data and float(self.agent_data.get('Total GMV', 0)) > 0:
                    total_gmv = float(self.agent_data['Total GMV'])
                    credit_dependence = (gmv / total_gmv) * 100 if total_gmv > 0 else 0
                    self.metrics['credit_dependence'] = credit_dependence
                    
                    # Add credit dependence to assessments
                    if credit_dependence > 70:
                        dep_risk = RiskLevel.HIGH
                        dep_label = "High Dependence"
                        self.findings.append("High credit dependence - business heavily relies on credit")
                    elif credit_dependence > 40:
                        dep_risk = RiskLevel.MODERATE
                        dep_label = "Moderate Dependence"
                    else:
                        dep_risk = RiskLevel.LOW
                        dep_label = "Low Dependence"
                    
                    self.assessments.append({
                        'metric': 'Credit Dependence',
                        'value': f"{credit_dependence:.1f}% - {dep_label}",
                        'risk': dep_risk
                    })
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing credit utilization data: {e}")
    
    def calculate_repayment_score(self, dpd: int, recent_payments: int = 0) -> int:
        """Calculate repayment score based on DPD and recent payment history.
        
        Scoring:
        - Base: 100 points
        - Penalties:
          - 30+ DPD: -5 points
          - 60+ DPD: -15 points
          - 90+ DPD: -30 points
          - 120+ DPD: -50 points
        - Bonus: +10 points for recent on-time payments
        
        Returns:
            int: Repayment score (0-100)
        """
        score = 100  # Start with perfect score
        
        # Apply penalties based on DPD
        if dpd >= 120:
            score -= 50
        elif dpd >= 90:
            score -= 30
        elif dpd >= 60:
            score -= 15
        elif dpd >= 30:
            score -= 5
            
        # Apply bonus for recent on-time payments
        if recent_payments > 0:
            bonus = min(10, recent_payments * 2)  # Max 10 points, 2 points per recent payment
            score = min(100, score + bonus)  # Cap at 100
            
        return max(0, score)  # Ensure non-negative
    
    def analyze_repayment(self) -> None:
        """Analyze repayment metrics including DPD, delinquency flags, and repayment score."""
        # Initialize metrics
        self.metrics['repayment_score'] = None
        self.metrics['delinquent_30p'] = False
        self.metrics['delinquent_60p'] = False
        self.metrics['delinquent_90p'] = False
        
        # Get DPD (Days Past Due) if available
        dpd = int(self.agent_data.get('Dpd', 0)) if pd.notnull(self.agent_data.get('Dpd')) else 0
        self.metrics['dpd'] = dpd
        
        # Set delinquency flags
        self.metrics['delinquent_30p'] = dpd >= 30
        self.metrics['delinquent_60p'] = dpd >= 60
        self.metrics['delinquent_90p'] = dpd >= 90
        
        # Get recent on-time payments (default to 0 if not available)
        recent_payments = int(self.agent_data.get('RecentPayments', 0)) if pd.notnull(self.agent_data.get('RecentPayments')) else 0
        
        # Calculate repayment score
        repayment_score = self.calculate_repayment_score(dpd, recent_payments)
        self.metrics['repayment_score'] = repayment_score
        
        # Determine risk level based on repayment score
        if repayment_score >= 90:
            risk = RiskLevel.LOW
            risk_label = "Excellent"
        elif repayment_score >= 70:
            risk = RiskLevel.LOW
            risk_label = "Good"
        elif repayment_score >= 50:
            risk = RiskLevel.MODERATE
            risk_label = "Moderate"
        elif repayment_score >= 30:
            risk = RiskLevel.HIGH
            risk_label = "High Risk"
        else:
            risk = RiskLevel.CRITICAL
            risk_label = "Critical Risk"
        
        # Add repayment score to assessments
        self.assessments.append({
            'metric': 'Repayment Score',
            'value': f"{repayment_score}/100 - {risk_label}",
            'risk': risk
        })
        
        # Add DPD assessment
        if dpd > 90:
            dpd_risk = RiskLevel.CRITICAL
            dpd_label = f"Severe Delinquency ({dpd} days)"
            self.findings.append(f"Critical: {dpd} days past due - account is severely delinquent")
            self.recommendations.append("Immediate collection action required")
        elif dpd > 60:
            dpd_risk = RiskLevel.HIGH
            dpd_label = f"High Risk Delinquency ({dpd} days)"
            self.findings.append(f"High Risk: {dpd} days past due - account is delinquent")
            self.recommendations.append("Initiate collection process and review credit terms")
        elif dpd > 30:
            dpd_risk = RiskLevel.MODERATE
            dpd_label = f"Moderate Risk ({dpd} days)"
            self.findings.append(f"Moderate Risk: {dpd} days past due - monitor closely")
            self.recommendations.append("Send payment reminder and follow up")
        elif dpd > 7:
            dpd_risk = RiskLevel.LOW
            dpd_label = f"Slight Delay ({dpd} days)"
            self.findings.append(f"Slight delay in payment ({dpd} days)")
        else:
            dpd_risk = RiskLevel.LOW
            dpd_label = "Current (0-7 days)"
        
        self.assessments.append({
            'metric': 'Days Past Due (DPD)',
            'value': dpd_label,
            'risk': dpd_risk
        })
        
        # Add credit score if available
        if 'credit_score' in self.agent_data and pd.notnull(self.agent_data['credit_score']):
            score = int(self.agent_data['credit_score'])
            self.metrics['credit_score'] = score
            
            if score < 30000:
                score_risk = RiskLevel.HIGH
                score_label = f"{score:,} - High Risk"
                self.findings.append("Low credit score indicates higher risk of default")
                self.recommendations.append("Consider additional credit checks or collateral requirements")
            elif score < 70000:
                score_risk = RiskLevel.MODERATE
                score_label = f"{score:,} - Medium Risk"
            else:
                score_risk = RiskLevel.LOW
                score_label = f"{score:,} - Low Risk"
                
            self.assessments.append({
                'metric': 'Credit Score',
                'value': score_label,
                'risk': score_risk
            })
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on analysis."""
        if not self.recommendations:
            self.recommendations.append("No specific recommendations at this time.")
    
    def analyze_region_risk(self) -> None:
        """Analyze regional risk factors and their impact on credit health.
        
        Region risk is calculated based on:
        - Default rate in the region (50% weight)
        - Average DPD in the region (30% weight)
        - Recovery metrics (20% weight)
        """
        # Initialize default values
        self.metrics['region_risk_score'] = 0.0
        self.metrics['region_default_rate'] = None
        self.metrics['region_avg_dpd'] = None
        self.metrics['region_recovery_rate'] = None
        
        try:
            # Get region data if available
            region = self.agent_data.get('Region')
            if not region or pd.isna(region):
                logger.warning("No region data available for risk analysis")
                return
                
            # Get region metrics (these would typically come from aggregated data)
            # For now, we'll use placeholders or values from the agent data if available
            region_default_rate = float(self.agent_data.get('RegionDefaultRate', 0.1))  # Default 10%
            region_avg_dpd = float(self.agent_data.get('RegionAvgDPD', 15))  # Default 15 days
            region_recovery_rate = float(self.agent_data.get('RegionRecoveryRate', 0.7))  # Default 70%
            
            # Store the metrics
            self.metrics.update({
                'region_default_rate': region_default_rate,
                'region_avg_dpd': region_avg_dpd,
                'region_recovery_rate': region_recovery_rate
            })
            
            # Calculate region risk score (0-1 scale, higher is riskier)
            # Normalize DPD to 0-1 range (assuming max 90 days for normalization)
            norm_dpd = min(region_avg_dpd / 90, 1.0)
            
            # Calculate risk components with weights
            default_risk = region_default_rate * 0.5  # 50% weight
            dpd_risk = norm_dpd * 0.3                # 30% weight
            recovery_risk = (1 - region_recovery_rate) * 0.2  # 20% weight
            
            # Calculate overall region risk (0-1 scale)
            region_risk_score = default_risk + dpd_risk + recovery_risk
            self.metrics['region_risk_score'] = min(max(region_risk_score, 0), 1)  # Clamp to 0-1
            
            # Add region risk assessment
            if region_risk_score > 0.7:
                risk = RiskLevel.CRITICAL
                risk_label = "Very High Risk"
            elif region_risk_score > 0.5:
                risk = RiskLevel.HIGH
                risk_label = "High Risk"
            elif region_risk_score > 0.3:
                risk = RiskLevel.MODERATE
                risk_label = "Moderate Risk"
            else:
                risk = RiskLevel.LOW
                risk_label = "Low Risk"
            
            self.assessments.append({
                'metric': 'Region Risk',
                'value': f"{risk_label} ({region_risk_score:.1%})",
                'risk': risk,
                'details': {
                    'Default Rate': f"{region_default_rate:.1%}",
                    'Avg DPD': f"{region_avg_dpd:.1f} days",
                    'Recovery Rate': f"{region_recovery_rate:.1%}"
                }
            })
            
        except Exception as e:
            logger.error(f"Error in region risk analysis: {e}")
    
    def calculate_credit_health_score(self) -> float:
        """Calculate overall credit health score (0-100).
        
        Score components and weights:
        - Repayment Score: 40%
        - Credit Utilization: 30%
        - Region Risk: 20%
        - Credit Dependence: 10%
        """
        try:
            # Get component scores with defaults
            repayment_score = self.metrics.get('repayment_score', 0)
            
            # Credit utilization score (0-100, higher is better)
            util_ratio = self.metrics.get('credit_utilization_ratio', 0)
            if util_ratio is None:
                util_score = 50  # Neutral if not available
            elif util_ratio < 0.3:  # Underutilization
                util_score = 30 + (util_ratio / 0.3) * 20  # 30-50
            elif util_ratio <= 0.7:  # Optimal
                util_score = 50 + ((util_ratio - 0.3) / 0.4) * 40  # 50-90
            else:  # High utilization
                util_score = 90 - ((util_ratio - 0.7) / 0.3) * 40  # 90-50
            
            # Region risk score (convert from 0-1 scale to 0-100, inverted)
            region_risk = self.metrics.get('region_risk_score', 0.5)
            region_score = (1 - region_risk) * 100
            
            # Credit dependence score (lower is better)
            credit_dep = min(self.metrics.get('credit_dependence', 0) / 100, 1.0)  # Clamp to 0-1
            dep_score = (1 - credit_dep) * 100
            
            # Calculate weighted score
            weights = {
                'repayment': 0.4,
                'utilization': 0.3,
                'region': 0.2,
                'dependence': 0.1
            }
            
            total_score = (
                (repayment_score * weights['repayment']) +
                (util_score * weights['utilization']) +
                (region_score * weights['region']) +
                (dep_score * weights['dependence'])
            )
            
            # Store the score and component metrics
            self.metrics['credit_health_score'] = total_score
            self.metrics['credit_health_components'] = {
                'repayment': repayment_score,
                'utilization': util_score,
                'region': region_score,
                'dependence': dep_score
            }
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating credit health score: {e}")
            return 50.0  # Return neutral score on error
    
    def analyze(self) -> Dict[str, Any]:
        """Run all analyses and return results."""
        # Run individual analyses
        self.analyze_credit_utilization()
        self.analyze_repayment()
        self.analyze_region_risk()
        
        # Calculate overall credit health score
        health_score = self.calculate_credit_health_score()
        
        # Add credit health assessment
        if health_score >= 80:
            health_risk = RiskLevel.LOW
            health_label = "Excellent"
        elif health_score >= 60:
            health_risk = RiskLevel.LOW
            health_label = "Good"
        elif health_score >= 40:
            health_risk = RiskLevel.MODERATE
            health_label = "Fair"
        elif health_score >= 20:
            health_risk = RiskLevel.HIGH
            health_label = "Poor"
        else:
            health_risk = RiskLevel.CRITICAL
            health_label = "Critical"
            
        self.assessments.append({
            'metric': 'Credit Health Score',
            'value': f"{health_score:.1f}/100 - {health_label}",
            'risk': health_risk
        })
        
        # Add trend analyses
        credit_trend = self.calculate_credit_usage_trend()
        sales_trend = self.calculate_sales_trend()
        
        # Store trend metrics
        self.metrics['credit_usage_trend'] = credit_trend
        self.metrics['sales_trend'] = sales_trend
        
        # Add trend assessments
        if credit_trend['months'] > 0:
            trend_emoji = '📈' if credit_trend['trend'] == 'increasing' else '📉' if credit_trend['trend'] == 'decreasing' else '➡️'
            self.assessments.append({
                'metric': 'Credit Usage Trend',
                'value': f"{trend_emoji} {credit_trend['trend'].title()} ({credit_trend['rate']:+.1%} monthly)",
                'risk': RiskLevel.HIGH if credit_trend['trend'] == 'increasing' else RiskLevel.LOW
            })
        
        if sales_trend['months'] > 0:
            trend_emoji = '📈' if sales_trend['trend'] == 'growing' else '📉' if sales_trend['trend'] == 'declining' else '➡️'
            self.assessments.append({
                'metric': 'Sales Trend',
                'value': f"{trend_emoji} {sales_trend['trend'].title()} ({sales_trend['rate']:+.1%} monthly)",
                'risk': RiskLevel.HIGH if sales_trend['trend'] == 'declining' else RiskLevel.LOW
            })
        
        # Generate final recommendations
        self.generate_recommendations()
        
        return {
            'metrics': self.metrics,
            'assessments': self.assessments,
            'findings': self.findings,
            'recommendations': self.recommendations,
            'credit_health_score': health_score
        }

def load_agents() -> pd.DataFrame:
    """Load and standardize agent data with enhanced error handling and data quality checks.
    
    This function performs the following operations:
    1. Loads agent data from the source Excel file
    2. Enriches data with information from sales data when available
    3. Handles missing values with appropriate defaults
    4. Validates data quality and provides metrics
    5. Returns a standardized DataFrame with all required columns
    
    Returns:
        DataFrame with standardized column names matching the documented schema.
        Required columns: BZID, organizationname, status, Region, city, State,
        Credit Limit, Credit Line Balance, TotalSeats, GMV
    
    Raises:
        FileNotFoundError: If the source data file is not found
        ValueError: If required columns are missing from the source data
    """
    # Define required and optional columns
    required_columns = [
        'BZID', 'organizationname', 'status', 'Region', 'city', 'State',
        'Credit Limit', 'Credit Line Balance', 'TotalSeats', 'GMV'
    ]
    
    # Define data quality thresholds (can be adjusted as needed)
    QUALITY_THRESHOLDS = {
        'missing_org_name': 0.1,  # Max 10% missing organization names
        'missing_region': 0.1,    # Max 10% missing regions
        'zero_credit_limit': 0.05, # Max 5% zero/negative credit limits
    }
    
    try:
        # 1. Load the raw agent data with validation
        agent_file = 'source_data/credit_Agents.xlsx'
        if not os.path.exists(agent_file):
            raise FileNotFoundError(f"Agent data file not found at {agent_file}")
            
        print("Loading agent data from source file...")
        try:
            agents = pd.read_excel(agent_file)
            if agents.empty:
                raise ValueError("Agent data file is empty")
                
            # Log basic info about the loaded data
            print(f"Found {len(agents):,} agents with columns: {', '.join(agents.columns.tolist())}")
            
            # Check for required columns in source data
            required_source_cols = ['Bzid', 'Credit Limit', 'Credit Line Balance']
            missing_source_cols = [col for col in required_source_cols 
                                 if col not in agents.columns]
            
            if missing_source_cols:
                raise ValueError(f"Missing required columns in source data: {', '.join(missing_source_cols)}")
                
        except Exception as e:
            raise ValueError(f"Error reading agent data: {str(e)}")
        
        # 2. Load sales data for additional information
        print("\nLoading sales data...")
        sales_data = load_sales()  # This now includes its own data quality reporting
        
        # 3. Prepare agent data with proper data types and validation
        result = pd.DataFrame()
        
        # 3.1 Handle BZID - ensure it's a string and clean it
        result['BZID'] = agents['Bzid'].astype(str).str.strip()
        
        # Remove any rows with empty BZID
        initial_count = len(result)
        result = result[result['BZID'].str.len() > 0].copy()
        if len(result) < initial_count:
            print(f"Warning: Removed {initial_count - len(result)} rows with empty BZID")
        
        # 3.2 Process credit-related fields with validation
        credit_fields = {
            'Credit Limit': ('Credit Limit', 0),
            'Credit Line Balance': ('Credit Line Balance', 0)
        }
        
        for target_col, (src_col, default) in credit_fields.items():
            if src_col in agents.columns:
                # Convert to numeric, coercing errors to NaN
                result[target_col] = pd.to_numeric(agents[src_col], errors='coerce')
                
                # Count and report null values
                null_count = result[target_col].isna().sum()
                if null_count > 0:
                    pct_null = (null_count / len(result)) * 100
                    print(f"Warning: Found {null_count:,} null values ({pct_null:.1f}%) in {target_col}, "
                          f"filling with {default}")
                    result[target_col] = result[target_col].fillna(default)
            else:
                print(f"Warning: Column '{src_col}' not found, using default value {default}")
                result[target_col] = default
        
        # 3.3 Calculate derived fields
        result['GMV'] = result['Credit Limit'] - result['Credit Line Balance'].clip(lower=0)
        
        # 4. Merge with sales data if available
        if not sales_data.empty and 'BZID' in sales_data.columns:
            print("\n=== Starting Data Merge ===")
            print(f"Agent data shape before merge: {result.shape}")
            print(f"Sales data shape: {sales_data.shape}")
            
            # Ensure BZID is string type in both dataframes
            result['BZID'] = result['BZID'].astype(str).str.strip()
            sales_data['BZID'] = sales_data['BZID'].astype(str).str.strip()
            
            # Log sample BZIDs from both datasets
            print("\nSample agent BZIDs (first 5):", result['BZID'].head(5).tolist())
            print("Sample sales BZIDs (first 5):", sales_data['BZID'].head(5).tolist())
            
            # Check for common BZIDs between datasets
            agent_bzids = set(result['BZID'])
            sales_bzids = set(sales_data['BZID'])
            common_bzids = agent_bzids.intersection(sales_bzids)
            
            print(f"\nAgent BZIDs: {len(agent_bzids):,}")
            print(f"Sales BZIDs: {len(sales_bzids):,}")
            print(f"Common BZIDs: {len(common_bzids):,} ({(len(common_bzids)/len(agent_bzids))*100:.1f}% of agents)")
            
            if common_bzids:
                print("Sample common BZIDs:", list(common_bzids)[:5])
            else:
                print("WARNING: No common BZIDs found between agent and sales data")
                # Try to diagnose the issue with sample data
                print("\nSample agent BZIDs (full sample):", result['BZID'].head(10).tolist())
                print("Sample sales BZIDs (full sample):", sales_data['BZID'].head(10).tolist())
                
                # Check for potential type mismatches
                print("\nChecking for type mismatches...")
                agent_types = result['BZID'].apply(type).value_counts()
                sales_types = sales_data['BZID'].apply(type).value_counts()
                print(f"Agent BZID types:\n{agent_types}")
                print(f"Sales BZID types:\n{sales_types}")
                
                # Check for leading/trailing spaces
                print("\nChecking for whitespace issues...")
                agent_has_whitespace = result['BZID'].str.contains(r'^\s|\s$').any()
                sales_has_whitespace = sales_data['BZID'].str.contains(r'^\s|\s$').any()
                print(f"Agent BZIDs have whitespace: {agent_has_whitespace}")
                print(f"Sales BZIDs have whitespace: {sales_has_whitespace}")
            
            # Only merge columns that exist in sales_data and are needed
            sales_cols = ['organizationname', 'Region', 'city', 'State', 'TotalSeats', 'status']
            merge_cols = ['BZID'] + [col for col in sales_cols if col in sales_data.columns]
            
            print(f"\nMerging with columns: {merge_cols}")
            
            # Perform the merge with indicator to track matches
            initial_count = len(result)
            merged = result.merge(
                sales_data[merge_cols].drop_duplicates('BZID'),  # Ensure one record per BZID
                on='BZID', 
                how='left',
                indicator='_merge',
                suffixes=('', '_sales')
            )
            
            # Log merge results
            merge_counts = merged['_merge'].value_counts()
            print("\n=== Merge Results ===")
            for merge_type, count in merge_counts.items():
                print(f"- {merge_type}: {count:,} rows ({(count/len(merged))*100:.1f}%)")
            
            # Analyze merge quality
            matched_count = (merged['_merge'] == 'both').sum()
            match_rate = (matched_count / len(merged)) * 100
            print(f"\nMatch rate: {match_rate:.1f}% ({matched_count:,} of {len(merged):,} agents matched)")
            
            # Handle merged columns
            for col in sales_cols:
                if f"{col}_sales" in merged.columns:
                    # Only fill NA values in the original column with sales data
                    merged[col] = merged[col].fillna(merged[f"{col}_sales"])
                    merged = merged.drop(columns=[f"{col}_sales"])
            
            # Remove the merge indicator column
            if '_merge' in merged.columns:
                merged = merged.drop(columns=['_merge'])
            
            result = merged
            
            # Final check for data quality
            print("\n=== Data Quality After Merge ===")
            quality_metrics = []
            for col in ['organizationname', 'Region', 'city', 'State']:
                if col in result.columns:
                    missing = result[col].isna() | (result[col] == 'Unknown')
                    pct_missing = (missing.sum() / len(result)) * 100
                    quality_metrics.append((col, missing.sum(), pct_missing))
            
            # Print quality metrics in a table
            print("\n{:<20} {:<10} {:<10}".format("Column", "Missing", "% Missing"))
            print("-" * 45)
            for col, count, pct in quality_metrics:
                print("{:<20} {:<10,} {:<10.1f}%".format(col, count, pct))
            
            # Log sample of merged data
            print("\nSample of merged data (first row):")
            sample = result.iloc[0].to_dict()
            for k, v in sample.items():
                if pd.isna(v):
                    v = "[MISSING]"
                elif isinstance(v, float):
                    v = f"{v:,.2f}"
                print(f"  {k}: {v}")
        
        # 5. Set default values for missing columns
        defaults = {
            'organizationname': 'Unknown Organization',
            'status': 'Unknown',
            'Region': 'Unknown',
            'city': 'Unknown',
            'State': 'Unknown',
            'TotalSeats': 1
        }
        
        for col, default in defaults.items():
            if col not in result.columns:
                result[col] = default
            else:
                result[col] = result[col].fillna(default)
        
        # 6. Ensure all required columns exist and are in the correct order
        missing_cols = [col for col in required_columns if col not in result.columns]
        if missing_cols:
            print(f"Warning: Missing required columns after processing: {', '.join(missing_cols)}")
            for col in missing_cols:
                result[col] = None  # Add missing columns with None values
        
        result = result[required_columns].copy()
        
        # 7. Generate comprehensive data quality report
        print("\n=== Agent Data Quality Report ===")
        print(f"Total agents processed: {len(result):,}")
        
        # Define quality checks
        quality_checks = {
            'Missing BZID': result['BZID'].isna().sum(),
            'Missing organization name': (result['organizationname'] == 'Unknown Organization').sum(),
            'Missing region': (result['Region'] == 'Unknown').sum(),
            'Zero/negative credit limit': (result['Credit Limit'] <= 0).sum(),
            'Negative GMV': (result['GMV'] < 0).sum(),
            'Missing city': (result['city'] == 'Unknown').sum(),
            'Missing state': (result['State'] == 'Unknown').sum()
        }
        
        # Calculate percentages and check against thresholds
        quality_issues = []
        for check, count in quality_checks.items():
            pct = (count / len(result)) * 100
            status = "OK"
            
            # Check against quality thresholds
            if 'organization name' in check and pct > (QUALITY_THRESHOLDS['missing_org_name'] * 100):
                status = f"WARNING: Exceeds {QUALITY_THRESHOLDS['missing_org_name']*100:.0f}% threshold"
                quality_issues.append(f"High percentage of {check.lower()} ({pct:.1f}%)")
            elif 'region' in check and pct > (QUALITY_THRESHOLDS['missing_region'] * 100):
                status = f"WARNING: Exceeds {QUALITY_THRESHOLDS['missing_region']*100:.0f}% threshold"
                quality_issues.append(f"High percentage of {check.lower()} ({pct:.1f}%)")
            elif 'credit limit' in check and pct > (QUALITY_THRESHOLDS['zero_credit_limit'] * 100):
                status = f"WARNING: Exceeds {QUALITY_THRESHOLDS['zero_credit_limit']*100:.0f}% threshold"
                quality_issues.append(f"High percentage of {check.lower()} ({pct:.1f}%)")
            
            print(f"- {check}: {count:,} agents ({pct:.1f}%) {status}")
        
        # 8. Log sample data for verification
        print("\nSample agent data (first row):")
        sample = result.iloc[0].to_dict()
        for k, v in sample.items():
            if isinstance(v, float):
                print(f"  {k}: {v:,.2f}")
            else:
                print(f"  {k}: {v}")
        
        # 9. Raise warning if critical data quality issues found
        if quality_issues:
            print("\nWARNING: Data quality issues detected:")
            for issue in quality_issues:
                print(f"- {issue}")
            print("\nRecommendation: Review the source data and data loading process "
                  "to address these issues.")
        
        return result
        
    except Exception as e:
        error_msg = f"Error loading agent data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=required_columns)

def load_sales() -> pd.DataFrame:
    """Load and process sales data with enhanced error handling and data quality checks.
    
    This function performs the following operations:
    1. Loads sales data from the source Excel file with validation
    2. Standardizes column names and data types
    3. Cleans and validates the data
    4. Provides comprehensive data quality metrics
    5. Handles errors gracefully with detailed logging
    
    Returns:
        DataFrame with standardized column names including:
        - Bzid: Agent ID (required)
        - organizationname: Organization name
        - city: City
        - State: State
        - Region: Sales region
        - Ro_Name: Regional Officer name
        - RM_Name: Relationship Manager name
    """
    SALES_FILE = 'source_data/sales_data.xlsx'
    
    # Define expected columns and their data types
    COLUMN_MAPPING = {
        'account': 'Bzid',
        'organizationname': 'organizationname',
        'city': 'city',
        'State': 'State',
        'Region': 'Region',
        'Ro Name': 'Ro_Name',
        'RM Name': 'RM_Name',
        'TotalSeats': 'TotalSeats',
        'status': 'status'
    }
    
    # Required columns (must be present in the source data)
    REQUIRED_COLS = ['account']
    
    # Optional columns (will use defaults if missing)
    OPTIONAL_COLS = [col for col in COLUMN_MAPPING.keys() if col not in REQUIRED_COLS]
    
    try:
        # 1. Check if file exists and is accessible
        if not os.path.exists(SALES_FILE):
            print(f"Error: Sales data file not found at {SALES_FILE}")
            return pd.DataFrame()
            
        print(f"\nLoading sales data from {SALES_FILE}...")
        
        # 2. First, check available columns
        try:
            # Read just the header to get column names
            all_columns = pd.read_excel(SALES_FILE, nrows=0).columns.tolist()
            print(f"Found {len(all_columns)} columns in sales data")
            
            # Log available columns for debugging
            print("\nAvailable columns in sales data:")
            for i, col in enumerate(all_columns, 1):
                print(f"  {i}. {col}")
            
            # Check for required columns
            missing_required = [col for col in REQUIRED_COLS if col not in all_columns]
            if missing_required:
                print(f"\nError: Missing required columns: {', '.join(missing_required)}")
                return pd.DataFrame()
                
            # Select only columns that exist in the file
            selected_cols = [col for col in (REQUIRED_COLS + OPTIONAL_COLS) 
                           if col in all_columns]
            
            if not selected_cols:
                print("Error: No valid columns found in sales data")
                return pd.DataFrame()
                
            # 3. Load the data with selected columns
            print(f"\nLoading data with columns: {', '.join(selected_cols)}")
            sales = pd.read_excel(
                SALES_FILE, 
                usecols=selected_cols,
                dtype=str,  # Read all as string first, convert later
                na_values=['', ' ', 'NA', 'N/A', 'NULL', 'None', 'nan', 'NaN'],
                keep_default_na=False
            )
            
            if sales.empty:
                print("Warning: Sales data file is empty")
                return pd.DataFrame()
                
            print(f"Loaded {len(sales):,} rows from sales data")
            
        except Exception as e:
            print(f"Error reading sales data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
        # 4. Standardize column names
        print("\nStandardizing column names...")
        rename_map = {k: v for k, v in COLUMN_MAPPING.items() 
                     if k in sales.columns and v != k}
        
        if rename_map:
            print("Renaming columns:")
            for old, new in rename_map.items():
                print(f"  {old} -> {new}")
            sales = sales.rename(columns=rename_map)
        
        # 5. Clean and validate data
        print("\nCleaning and validating data...")
        
        # 5.1 Clean Bzid (required field)
        if 'Bzid' in sales.columns:
            # Convert to string and clean
            sales['Bzid'] = sales['Bzid'].astype(str).str.strip()
            
            # Remove rows with empty/missing Bzid
            initial_count = len(sales)
            sales = sales[~sales['Bzid'].isna() & (sales['Bzid'] != '')].copy()
            
            if len(sales) < initial_count:
                print(f"  Removed {initial_count - len(sales):,} rows with missing/invalid Bzid")
            
            # Check for duplicates
            dup_count = sales.duplicated('Bzid').sum()
            if dup_count > 0:
                print(f"  Found {dup_count:,} duplicate Bzid values, keeping first occurrence")
                sales = sales.drop_duplicates('Bzid', keep='first')
        else:
            print("Error: Required column 'Bzid' not found after renaming")
            return pd.DataFrame()
        
        # 5.2 Clean text fields
        text_columns = ['organizationname', 'city', 'State', 'Region', 'Ro_Name', 'RM_Name', 'status']
        for col in text_columns:
            if col in sales.columns:
                # Convert to string, strip whitespace, and clean common placeholders
                sales[col] = (
                    sales[col]
                    .astype(str)
                    .str.strip()
                    .replace({
                        'nan': '', 'None': '', 'NULL': '', 
                        'N/A': '', 'n/a': '', '': None
                    })
                )
                
                # Count missing values
                missing = sales[col].isna().sum()
                if missing > 0:
                    pct_missing = (missing / len(sales)) * 100
                    print(f"  Column '{col}': {missing:,} missing values ({pct_missing:.1f}%)")
        
        # 5.3 Clean numeric fields
        if 'TotalSeats' in sales.columns:
            # Convert to numeric, coerce errors to NaN
            sales['TotalSeats'] = pd.to_numeric(sales['TotalSeats'], errors='coerce')
            
            # Replace NaN with default value (1)
            sales['TotalSeats'] = sales['TotalSeats'].fillna(1).astype(int)
            
            # Validate values
            invalid_seats = (sales['TotalSeats'] <= 0).sum()
            if invalid_seats > 0:
                print(f"  Found {invalid_seats:,} rows with invalid TotalSeats (<=0), setting to 1")
                sales.loc[sales['TotalSeats'] <= 0, 'TotalSeats'] = 1
        
        # 6. Data quality report
        print("\n=== Sales Data Quality Report ===")
        print(f"Total agents in sales data: {len(sales):,}")
        
        if len(sales) > 0:
            # Check for missing values in key columns
            key_columns = [
                'organizationname', 'city', 'State', 'Region', 
                'Ro_Name', 'RM_Name', 'status', 'TotalSeats'
            ]
            
            print("\nMissing values in key columns:")
            for col in key_columns:
                if col in sales.columns:
                    missing = sales[col].isna().sum()
                    pct_missing = (missing / len(sales)) * 100
                    print(f"  {col}: {missing:,} missing ({pct_missing:.1f}%)")
            
            # Check for data distribution in key columns
            if 'Region' in sales.columns:
                print("\nRegion distribution (top 10):")
                region_counts = sales['Region'].value_counts(dropna=False).head(10)
                for region, count in region_counts.items():
                    pct = (count / len(sales)) * 100
                    print(f"  {region or 'MISSING'}: {count:,} agents ({pct:.1f}%)")
            
            # Show sample data
            print("\nSample sales data (first row):")
            sample = sales.iloc[0].to_dict()
            for k, v in sample.items():
                if pd.isna(v):
                    v = "[MISSING]"
                elif isinstance(v, float):
                    v = f"{v:,.2f}"
                print(f"  {k}: {v}")
        
        return sales
        
    except Exception as e:
        print(f"\nUnexpected error loading sales data: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_dpd() -> pd.DataFrame:
    """Load DPD data from Excel file."""
    try:
        dpd = pd.read_excel('source_data/DPD.xlsx')
        dpd['Bzid'] = dpd['Bzid'].astype(str)
        return dpd
    except Exception as e:
        print(f"Error loading DPD data: {e}")
        return pd.DataFrame()

def load_repayment_metrics():
    """
    Load the most recent repayment metrics from CSV file in the reports directory.
    If no file exists, runs analyze_repayments.py to generate it.
    """
    try:
        # Find the most recent repayment metrics file
        reports_dir = Path('reports')
        if not reports_dir.exists():
            reports_dir.mkdir(parents=True, exist_ok=True)
            
        # Look for files matching the pattern
        metric_files = list(reports_dir.glob('repayment_analysis_*.csv'))
        
        if not metric_files:
            print("No repayment metrics files found. Running analysis...")
            try:
                from scripts.analysis import analyze_repayments
                analyze_repayments.main()
                # Try to find the newly generated file
                metric_files = list(reports_dir.glob('repayment_analysis_*.csv'))
                if not metric_files:
                    print("Error: Failed to generate repayment metrics file")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error running analyze_repayments.py: {e}")
                return pd.DataFrame()
            
        # Get the most recent file
        latest_file = max(metric_files, key=lambda x: x.stat().st_mtime)
        print(f"Found repayment metrics file: {latest_file}")
        
        # Load the CSV file with potential encoding issues
        try:
            df = pd.read_csv(latest_file, encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            try:
                df = pd.read_csv(latest_file, encoding='latin1')
            except Exception as e:
                print(f"Error reading CSV file with any encoding: {e}")
                return pd.DataFrame()
        
        if df.empty:
            print("Warning: Repayment metrics file is empty")
            return df
            
        # Standardize column names (case-insensitive and handle special characters)
        df.columns = df.columns.str.strip()
        
        # Try to find the BZID column with flexible matching
        bzid_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'bzid' in col_lower or 'anchor' in col_lower or 'onboard' in col_lower:
                bzid_col = col
                print(f"Found BZID column: {col}")
                break
        
        if bzid_col is None:
            print("Error: Could not find BZID column in repayment metrics")
            print(f"Available columns: {', '.join(df.columns)}")
            return pd.DataFrame()
            
        # Rename to standard BZID
        if bzid_col != 'BZID':
            df = df.rename(columns={bzid_col: 'BZID'})
            
        # Ensure BZID is string type and clean up any whitespace
        df['BZID'] = df['BZID'].astype(str).str.strip()
        
        # Debug: Print sample BZIDs to verify
        print(f"Sample BZIDs from repayment data: {df['BZID'].head().tolist()}")
        
        # Check for required columns with flexible naming
        required_mappings = {
            'total_repayment_amount': ['total_repayment_amount', 'total_repayment', 'repayment_amount', 'amount'],
            'transaction_count': ['transaction_count', 'txn_count', 'num_transactions', 'count', 'transactions'],
            'credit_health_score': ['credit_health_score', 'health_score', 'score', 'health']
        }
        
        # Map actual columns to standard names
        column_mapping = {'BZID': 'BZID'}
        missing_columns = []
        
        for standard_name, possible_names in required_mappings.items():
            found = False
            for name in possible_names:
                # Check for exact match first
                if name in df.columns:
                    if standard_name != name:  # Only rename if different
                        df = df.rename(columns={name: standard_name})
                    column_mapping[name] = standard_name
                    found = True
                    break
                
                # Try case-insensitive match
                for col in df.columns:
                    if name.lower() in col.lower():
                        df = df.rename(columns={col: standard_name})
                        column_mapping[col] = standard_name
                        found = True
                        print(f"Mapped column '{col}' to '{standard_name}'")
                        break
                if found:
                    break
                    
            if not found:
                missing_columns.append(standard_name)
        
        if missing_columns:
            print(f"Warning: Missing required columns in repayment metrics: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(df.columns)}")
            # Don't return empty DF, just log the warning and continue with available columns
        
        # Select available columns (don't fail if some are missing)
        available_columns = ['BZID'] + [col for col in required_mappings.keys() if col in df.columns]
        df = df[available_columns].copy()
        
        # Ensure numeric columns are properly typed
        for col in ['total_repayment_amount', 'credit_health_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'transaction_count' in df.columns:
            df['transaction_count'] = pd.to_numeric(df['transaction_count'], errors='coerce').fillna(0).astype(int)
        
        print(f"Successfully loaded repayment data with {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading repayment metrics: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def format_score_breakdown(score_components: Dict[str, float], max_width: int = 40) -> str:
    """Format the score breakdown with visual bars and percentages.
    
    Args:
        score_components: Dictionary of component names and their scores
        max_width: Maximum width of the bar chart
        
    Returns:
        Formatted string with score breakdown
    """
    # Filter out zero scores for cleaner output
    non_zero_components = {k: v for k, v in score_components.items() if v > 0}
    if not non_zero_components:
        return "\n\033[1mScore Breakdown:\033[0m\n  No score data available\n"
        
    max_score = max(non_zero_components.values()) if non_zero_components else 1
    total_score = sum(non_zero_components.values())
    
    # Sort components by score in descending order
    sorted_components = sorted(non_zero_components.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    result.append("\n\033[1mSCORE BREAKDOWN\033[0m")
    result.append("-" * 80)
    result.append(f"{'COMPONENT':<35} {'SCORE':<7} {'VISUALIZATION':<25} {'% OF TOTAL'}")
    result.append("-" * 80)
    
    for component, score in sorted_components:
        # Calculate bar length (minimum 1 if score > 0)
        bar_length = max(1, int((score / max_score) * max_width)) if max_score > 0 else 1
        bar = "=" * bar_length  # Using '=' for the bar instead of block characters
        
        # Calculate percentage of total score
        pct_of_total = (score / total_score * 100) if total_score > 0 else 0
        
        # Color code based on percentage of total
        if pct_of_total > 20:
            color = "\033[92m"  # Green for high contributors
            pct_color = "\033[92m"
        elif pct_of_total > 5:
            color = "\033[93m"  # Yellow for medium contributors
            pct_color = "\033[93m"
        else:
            color = "\033[91m"  # Red for low contributors
            pct_color = "\033[91m"
        
        # Format the score with appropriate color
        score_str = f"{score:.2f}"
        
        # Format the line with aligned columns using ASCII characters
        line = (
            f"  \033[1m{component:<33}\033[0m "  # Component name in bold
            f"{color}{score_str:<7}\033[0m "      # Score with color
            f"{color}[{bar:<{max_width}}]\033[0m "  # Bar with ASCII borders
            f"{pct_color}{pct_of_total:>5.1f}%\033[0m"  # Percentage
        )
        result.append(line)
    
    # Add a summary line
    result.append("-" * 80)
    result.append(
        f"  \033[1m{'TOTAL':<33} {total_score:.2f}"
        f"{' ' * (max_width + 6)} {100.0:>5.1f}%\033[0m"
    )
    
    return "\n".join(result)

def print_agent_profile(bzid: str, agent_data: Dict, analysis: Dict) -> None:
    """Print formatted agent profile in a detailed, structured format with visual enhancements."""
    
    # Helper function to format currency values with ₹ symbol
    def fmt_curr(val):
        if pd.isna(val):
            return "N/A"
        return f"₹{float(val):,.2f}"
    
    # Helper function to format percentages with color coding
    def fmt_pct(val, good_threshold=70, warning_threshold=30):
        if pd.isna(val):
            return "N/A"
        val_float = float(val)
        if val_float >= good_threshold:
            return f"\033[92m{val_float:.2f}%\033[0m"  # Green for good
        elif val_float <= warning_threshold:
            return f"\033[91m{val_float:.2f}%\033[0m"  # Red for warning
        else:
            return f"\033[93m{val_float:.2f}%\033[0m"  # Yellow for caution
    
    # Helper function to print section headers
    def print_section(title, emoji):
        print(f"\n\033[1;36m{emoji} {title.upper()}\033[0m")
        print("-" * (len(title) + 2))
    # Helper function to format currency values
    def fmt_curr(val):
        if pd.isna(val):
            return "N/A"
        return f"{float(val):,.2f}"
    
    # Helper function to format percentages
    def fmt_pct(val):
        if pd.isna(val):
            return "N/A"
        return f"{float(val):.2f}%"
    
    # Clear screen and print header
    print("\033[H\033[J")  # Clear screen
    print("\033[1;35m" + "=" * 80)
    print(f"[AGENT] CREDIT RISK PROFILE: {bzid}".center(80))
    print("=" * 80 + "\033[0m")
    
    # Print last updated timestamp
    from datetime import datetime
    print(f"\033[90mAgent ID: {bzid} • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print("\033[90m" + "-" * 80 + "\033[0m")
    
    # --- Basic Information ---
    print_section("Basic Information", "[PERSON]")
    
    # Helper function to print key-value pairs with descriptions
    def print_info(label, value, description=""):
        print(f"\033[1m{label:<20}\033[0m {value}" + (f"  \033[90m# {description}\033[0m" if description else ""))
    
    # Print organization and region information if available
    if 'organizationname' in agent_data and pd.notna(agent_data.get('organizationname')):
        print_info("Organization", agent_data['organizationname'], "Agent's organization")
    
    if 'Region' in agent_data and pd.notna(agent_data.get('Region')):
        print_info("Region", agent_data['Region'], "Agent's primary region")
    
    if 'city' in agent_data and pd.notna(agent_data.get('city')):
        location_parts = []
        if pd.notna(agent_data.get('city')):
            location_parts.append(str(agent_data['city']))
        if 'State' in agent_data and pd.notna(agent_data.get('State')):
            location_parts.append(str(agent_data['State']))
        if location_parts:
            print_info("Location", ", ".join(location_parts), "Agent's location")
    
    if 'Ro Name' in agent_data and pd.notna(agent_data.get('Ro Name')):
        print_info("Regional Officer", agent_data['Ro Name'], "Assigned regional officer")
    
    if 'RM Name' in agent_data and pd.notna(agent_data.get('RM Name')):
        print_info("Regional Manager", agent_data['RM Name'], "Assigned regional manager")
    
    print()  # Add spacing after basic info section
    
    print_info("Agent ID:", bzid, "Unique identifier for the agent")
    print_info("Agent Name:", agent_data.get('Username', 'N/A'), "Full name of the agent")
    print_info("Business:", agent_data.get('Business Name', 'N/A'), "Registered business name")
    print_info("Region:", agent_data.get('Region', 'N/A'), "Geographic region of operation")
    print_info("Phone:", agent_data.get('Phone', 'N/A'), "Primary contact number")
    
    # --- Credit Health ---
    print_section("Credit Health", "💳")
    
    # Try to get credit data from unified JSON first, then fall back to direct fields
    credit_data = agent_data.get('sources', {}).get('repayments', {})
    
    # Get credit limit with fallback
    credit_limit = credit_data.get('credit_limit')
    if credit_limit is None:
        credit_limit = agent_data.get('Credit Limit')
    
    # Get credit line balance with fallback
    credit_line_balance = credit_data.get('credit_line_balance')
    if credit_line_balance is None:
        credit_line_balance = agent_data.get('Credit Line Balance')
    
    # Calculate available credit and credit used
    available_credit = None
    credit_used = None
    
    if credit_limit is not None and credit_line_balance is not None:
        try:
            credit_limit_float = float(credit_limit)
            credit_line_balance_float = float(credit_line_balance)
            available_credit = max(0, credit_limit_float - credit_line_balance_float)
            credit_used = credit_line_balance_float  # Credit used is the same as credit line balance
        except (ValueError, TypeError):
            logger.warning("Invalid credit limit or balance values")
    
    # Print credit limit information with status indicator
    if credit_limit is not None and float(credit_limit or 0) > 0:
        limit_status = "\033[92m✓ Available"
        print_info("Credit Limit:", f"₹{float(credit_limit):,.2f}  {limit_status}\033[0m", 
                  "Total credit available to the agent")
        
        # Print available credit
        if available_credit is not None:
            print_info("Available Credit:", f"₹{available_credit:,.2f}", 
                      "Remaining credit available for use")
            
            # Print credit used (which is the credit line balance)
            if credit_used is not None:
                used_pct = (credit_used / float(credit_limit)) * 100 if float(credit_limit) > 0 else 0
                used_status = "\033[93mHigh Usage" if used_pct > 70 else "\033[92mGood"
                print_info("Credit Used:", 
                         f"₹{credit_used:,.2f}  {used_status}\033[0m", 
                         f"Amount of credit currently in use")
                
                # Display utilization with color coding and risk assessment
                if credit_limit_float > 0:
                    utilization = (credit_used / credit_limit_float) * 100
                    
                    # Determine utilization risk level with emojis and colors
                    if utilization < 30:
                        util_risk = "\033[93m⚠️ Underutilized"
                        risk_emoji = "🟡"
                        risk_desc = "Low credit utilization may indicate business issues"
                    elif utilization <= 70:
                        util_risk = "\033[92m✓ Optimal"
                        risk_emoji = "🟢"
                        risk_desc = "Healthy credit utilization"
                    else:
                        util_risk = "\033[91m⚠️ High Utilization"
                        risk_emoji = "🔴"
                        risk_desc = "High utilization may impact credit score"
                    
                    # Print utilization metrics
                    print_info("Utilization:", 
                             f"{utilization:.2f}% {risk_emoji} {util_risk}\033[0m", 
                             risk_desc)
                    
                    # Add utilization gauge visualization
                    gauge_width = 40
                    used_width = min(gauge_width, max(1, int(utilization / 100 * gauge_width)))
                    gauge = "[" + "=" * used_width + " " * (gauge_width - used_width) + "]"
                    print(f"\n  {gauge} {utilization:.1f}% utilized")
                    print("  " + "^" * (used_width + 1) + f" {utilization:.1f}%")
                    print("  " + "|" * (gauge_width + 2))
                    print("  0%" + " " * (gauge_width - 5) + "100%")
    else:
        print_info("Credit Limit:", "\033[93mNot Available\033[0m", 
                  "Credit limit not set or zero")
    
    # --- Agent Classification ---
    print_section("Agent Classification", "📊")
    
    # Check if we have classification data from AgentClassifier
    if 'classification' in analysis and analysis['classification']:
        classification = analysis['classification']
        tier = classification.get('tier', 'N/A')
        description = classification.get('description', 'No description available')
        action = classification.get('action', 'No action recommended')
        
        # Define tier colors and emojis
        tier_colors = {
            'P0': ('\033[92m', '✅'),  # Green
            'P1': ('\033[93m', 'ℹ️'),  # Yellow
            'P2': ('\033[91m', '⚠️'),  # Red
            'P3': ('\033[93m', '🔍'),  # Yellow
            'P4': ('\033[91m', '📚'),  # Red
            'P5': ('\033[90m', '🚫'),  # Gray
        }
        
        # Get color and emoji for the current tier
        color, emoji = tier_colors.get(tier, ('\033[0m', '❓'))
        
        # Print tier with color and emoji
        print(f"\n  {emoji} {color}Tier: {tier}\033[0m")
        print(f"  {'─' * 75}")
        print(f"  \033[1mDescription:\033[0m {description}")
        print(f"  \033[1mRecommended Action:\033[0m {action}")
        
        # Add a visual indicator of the tier
        if tier in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
            tier_num = int(tier[1])
            scale = "[P0│P1│P2│P3│P4│P5]"
            marker = " " * (tier_num * 4 + 1) + "^"
            print(f"\n  Tier Scale: {scale}")
            print(f"  {' ' * 11}{marker} (Current: {tier})")
    else:
        print("\n  \033[93m⚠️ No classification data available\033[0m")
        print("  Run the classification analysis to see agent tier and recommendations.")
    
    # --- Repayment Performance ---
    print_section("Repayment Performance", "💰")
    
    # Try to get repayment data from unified JSON first
    repayment_data = agent_data.get('sources', {}).get('repayments', {})
    
    if repayment_data:
        # Get repayment metrics with fallbacks
        total_repaid = repayment_data.get('total_repayment', 0)
        principal_repaid = repayment_data.get('principal_repaid', 0)
        txn_count = repayment_data.get('total_repayment_count', 0)
        last_payment = repayment_data.get('last_repayment_date', 'N/A')
        first_payment = repayment_data.get('first_repayment_date', 'N/A')
        consistency = repayment_data.get('repayment_consistency', 0)
        
        # Print summary metrics
        print_info("Total Amount Repaid:", 
                  fmt_curr(total_repaid), 
                  "Cumulative amount repaid (principal + interest)")
        
        print_info("Principal Repaid:", 
                  fmt_curr(principal_repaid),
                  "Portion of repayment that went towards principal")
        
        # Calculate and display interest paid
        if pd.notnull(total_repaid) and pd.notnull(principal_repaid):
            interest_paid = total_repaid - principal_repaid
            interest_pct = (interest_paid / total_repaid * 100) if total_repaid > 0 else 0
            print_info("Interest Paid:", 
                      f"{fmt_curr(interest_paid)} ({interest_pct:.1f}% of total)",
                      "Total interest paid on credit")
        
        # Transaction metrics
        print_info("Total Transactions:", 
                  f"{int(txn_count):,}",
                  "Number of repayment transactions")
        
        # Calculate and display averages if transaction count is available
        if txn_count > 0:
            avg_repayment = agent_data.get('avg_repayment_per_txn', 0)
            avg_principal = agent_data.get('avg_principal_per_txn', 0)
            
            print_info("Avg. Repayment:", 
                      fmt_curr(avg_repayment),
                      "Average amount paid per transaction")
            
            print_info("Avg. Principal:",
                     fmt_curr(avg_principal),
                     "Average principal amount per transaction")
            
            # Calculate payment frequency (if we have date information)
            if 'first_repayment_date' in agent_data and 'last_repayment_date' in agent_data:
                try:
                    from datetime import datetime
                    first_date = pd.to_datetime(agent_data['first_repayment_date'])
                    last_date = pd.to_datetime(agent_data['last_repayment_date'])
                    days_between = (last_date - first_date).days + 1
                    if days_between > 0:
                        freq = txn_count / (days_between / 30)  # Transactions per month
                        print_info("Payment Frequency:",
                                 f"{freq:.1f} txns/month",
                                 "Average transactions per month")
                except:
                    pass
            
            # Calculate principal as % of repayment
            if total_repaid > 0:
                pct_principal = (principal_repaid / total_repaid) * 100
                efficiency = ""
                if pct_principal > 80:
                    efficiency = "\033[92m✓ Excellent"
                elif pct_principal > 60:
                    efficiency = "\033[93m✓ Good"
                else:
                    efficiency = "\033[91m⚠️ Low"
                
                print_info("Principal Efficiency:",
                         f"{pct_principal:.1f}%  {efficiency}\033[0m",
                         "Higher % means more of payments go to principal")
    else:
        print_info("No repayment data available", "", "Repayment history not found for this agent")
    
    # --- Behavior Score ---
    print_section("Behavior & Payment History", "📊")
    
    # Credit Score
    if 'credit_score' in agent_data and pd.notnull(agent_data['credit_score']):
        score = float(agent_data['credit_score'])
        normalized_score = min(1.0, max(0.0, score / 100000.0))  # Assuming max score is 100,000
        
        # Determine score category
        if normalized_score >= 0.8:
            score_category = "\033[92mExcellent"
            score_emoji = "⭐"
        elif normalized_score >= 0.6:
            score_category = "\033[94mGood"
            score_emoji = "✓"
        elif normalized_score >= 0.4:
            score_category = "\033[93mFair"
            score_emoji = "ℹ️"
        else:
            score_category = "\033[91mNeeds Improvement"
            score_emoji = "⚠️"
            
        # Create score bar
        score_width = 40
        filled_width = int(normalized_score * score_width)
        score_bar = "[" + "█" * filled_width + " " * (score_width - filled_width) + "]"
        
        print_info("Credit Score:", 
                  f"{normalized_score:.2f} {score_emoji} {score_category}\033[0m",
                  "Score range: 0.0 (Low) to 1.0 (High)")
        print(f"  {score_bar}")
    
    # DPD and Payment History
    dpd = agent_data.get('Dpd', 0)
    if pd.isna(dpd):
        dpd = 0
    
    # Determine DPD status with emojis and colors
    if dpd == 0:
        dpd_status = "\033[92m✅ Current"
        payment_status = "No Late Payments"
        status_desc = "Payments are up to date"
    elif dpd <= 7:
        dpd_status = "\033[93m⚠️ Watch"
        payment_status = "Minor Delays"
        status_desc = "Slight delays in payments"
    elif dpd <= 30:
        dpd_status = "\033[91m❌ Delinquent"
        payment_status = "Late Payments"
        status_desc = "Concerns with payment timeliness"
    else:
        dpd_status = "\033[91m🚨 High Risk"
        payment_status = "Severely Delinquent"
        status_desc = "Significant payment issues"
    
    print_info("Days Past Due (DPD):", 
              f"{int(dpd)} days  {dpd_status}\033[0m",
              "Number of days since last payment was due")
    
    print_info("Payment Status:", 
              f"{payment_status}",
              status_desc)
    
    # --- Final Credit Risk ---
    print_section("Credit Risk Assessment", "📈")
    
    # Initialize score components with default values
    score_components = {}
    
    # Set default values for all components
    score_components["Payment History (15%)"] = 0.0
    score_components["Credit Utilization (20%)"] = 0.0
    score_components["Total Repaid (20%)"] = 0.0
    score_components["Principal Repaid (15%)"] = 0.0
    score_components["Transaction Volume (15%)"] = 0.0
    score_components["Average Repayment (10%)"] = 0.0
    score_components["Payment Consistency (5%)"] = 0.0
    
    # Calculate risk score (1-5 scale, lower is better)
    risk_score = 0.0
    risk_factors = []
    
    # Utilization component (20% of score)
    if 'utilization' in locals():
        if utilization < 30:
            risk_score += 2.0  # Higher risk for underutilization
            score_components["Credit Utilization (20%)"] = 0.05  # Low score for underutilization
            risk_factors.append(f"High credit underutilization ({utilization:.1f}% < 30%)")
        elif utilization <= 70:
            risk_score += 1.0  # Medium risk for optimal utilization
            score_components["Credit Utilization (20%)"] = 0.20  # Full score for optimal utilization
            risk_factors.append(f"Optimal credit utilization ({utilization:.1f}%)")
        else:
            risk_score += 1.5  # Slightly higher risk for high utilization
            score_components["Credit Utilization (20%)"] = 0.10  # Partial score for high utilization
            risk_factors.append(f"High credit utilization ({utilization:.1f}% > 70%)")
    else:
        score_components["Credit Utilization (20%)"] = 0.0
    
    # DPD component (Days Past Due)
    if dpd == 0:
        risk_score += 1.0
        score_components["Payment History (15%)"] = 0.15  # Full points for no delays
        risk_factors.append("No late payments")
    elif dpd <= 7:
        risk_score += 2.0
        score_components["Payment History (15%)"] = 0.10  # Partial points for minor delays
        risk_factors.append(f"Minor payment delays ({dpd} days)")
    elif dpd <= 30:
        risk_score += 3.0
        score_components["Payment History (15%)"] = 0.05  # Few points for late payments
        risk_factors.append(f"Late payments ({dpd} days)")
    else:
        risk_score += 5.0
        score_components["Payment History (15%)"] = 0.0  # No points for severe delinquency
        risk_factors.append(f"Severe delinquency ({dpd} days)")
    
    # Transaction history component (if available)
    if 'transaction_count' in agent_data and agent_data['transaction_count'] > 0:
        txn_count = agent_data['transaction_count']
        
        # Transaction Volume (15% of score)
        if txn_count >= 12:  # At least 1 year of monthly payments
            risk_score -= 0.5
            score_components["Transaction Volume (15%)"] = 0.15  # Full points for established history
            risk_factors.append(f"Established payment history ({txn_count} transactions)")
        elif txn_count >= 6:  # At least 6 months of history
            risk_score += 0.5
            score_components["Transaction Volume (15%)"] = 0.10  # Partial points for limited history
            risk_factors.append(f"Limited payment history ({txn_count} transactions)")
        else:
            risk_score += 1.0
            score_components["Transaction Volume (15%)"] = 0.05  # Few points for minimal history
            risk_factors.append(f"Minimal payment history ({txn_count} transactions)")
        
        # Payment Consistency (5% of score) - based on transaction frequency
        if txn_count >= 24:  # At least 2 years of bi-monthly payments
            score_components["Payment Consistency (5%)"] = 0.05
        elif txn_count >= 12:  # At least 1 year of monthly payments
            score_components["Payment Consistency (5%)"] = 0.04
        elif txn_count >= 6:  # At least 6 months of history
            score_components["Payment Consistency (5%)"] = 0.03
        else:
            score_components["Payment Consistency (5%)"] = 0.01
            
        # Add repayment metrics to score components if available
        if 'total_repayment_amount' in agent_data and agent_data['total_repayment_amount'] > 0:
            repaid = agent_data['total_repayment_amount']
            score_components["Total Repaid (20%)"] = min(0.20, (repaid / 100000) * 0.20)  # Scale to 20% of score
            
        if 'total_principal_repaid' in agent_data and agent_data['total_principal_repaid'] > 0:
            principal = agent_data['total_principal_repaid']
            score_components["Principal Repaid (15%)"] = min(0.15, (principal / 100000) * 0.15)  # Scale to 15% of score
            
        if 'avg_repayment_per_txn' in agent_data and agent_data['avg_repayment_per_txn'] > 0:
            avg_repay = agent_data['avg_repayment_per_txn']
            score_components["Average Repayment (10%)"] = min(0.10, (avg_repay / 1000) * 0.10)  # Scale to 10% of score
            
        # Calculate principal ratio if possible
        if ('total_principal_repaid' in agent_data and 'total_repayment_amount' in agent_data and 
            agent_data['total_repayment_amount'] > 0):
            principal_ratio = agent_data['total_principal_repaid'] / agent_data['total_repayment_amount']
            score_components["Principal Ratio (15%)"] = principal_ratio * 0.15  # Scale to 15% of score
    
    # Normalize to 1-5 scale (lower is better)
    overall_risk_score = min(5.0, max(1.0, risk_score))
    
    # Determine risk category with color and emoji
    if overall_risk_score <= 1.5:
        risk_category = "\033[92m🌟 Very Low Risk"
        risk_emoji = "✅"
        risk_desc = "Excellent credit profile"
    elif overall_risk_score <= 2.5:
        risk_category = "\033[92m✓ Low Risk"
        risk_emoji = "👍"
        risk_desc = "Good credit profile"
    elif overall_risk_score <= 3.5:
        risk_category = "\033[93m⚠️ Medium Risk"
        risk_emoji = "ℹ️"
        risk_desc = "Moderate credit risk"
    elif overall_risk_score <= 4.5:
        risk_category = "\033[91m⚠️ High Risk"
        risk_emoji = "⚠️"
        risk_desc = "High credit risk"
    else:
        risk_category = "\033[91m🚨 Very High Risk"
        risk_emoji = "❌"
        risk_desc = "Very high credit risk"
    
    # Print risk assessment
    print_info("Risk Score:", 
              f"{overall_risk_score:.1f}/5.0 {risk_emoji} {risk_category}\033[0m",
              "1 = Lowest Risk, 5 = Highest Risk")
    
    # Print score breakdown
    print(format_score_breakdown(score_components))
    
    # Print risk factors
    print("\n\033[1mKey Risk Factors:\033[0m")
    for factor in risk_factors:
        print(f"  • {factor}")
    
    # Print recommendation based on risk
    print(f"\n\033[1mRecommendation: {risk_desc}\033[0m")
    
    # Print footer
    print("\n" + "=" * 80)
    print("\033[90mReport generated on:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "\033[0m")

def main():
    """Main function to run the agent lookup tool."""
    from src.agent_data_store import AgentDataStore
    
    # Initialize the unified agent data store
    print("Loading unified agent data store...")
    data_store = AgentDataStore()
    
    # Load all agent data from the unified JSON
    print("Loading agent data from unified store...")
    agent_data_list = data_store.get_agents()
    
    if not agent_data_list:
        print("No agent data found in unified store. Loading from individual sources...")
        agents = load_agents()
        
        if agents.empty:
            print("Failed to load agent data. Exiting.")
            return
            
        # Load sales data for additional information
        print("Loading sales data...")
        sales_data = load_sales()
        
        # Load repayment metrics
        print("Loading repayment metrics...")
        repay_metrics = load_repayment_metrics()
        
        # If no repayment metrics found, try to generate them
        if repay_metrics.empty:
            print("Repayment metrics not found. Running analysis...")
            try:
                from scripts.analysis import analyze_repayments
                analyze_repayments.main()
                repay_metrics = load_repayment_metrics()
            except Exception as e:
                print(f"Failed to generate repayment metrics: {e}")
        
        # Standardize column names to use uppercase BZID
        agents.rename(columns={'Bzid': 'BZID'}, inplace=True)
        
        # Ensure BZID column has consistent format
        agents['BZID'] = agents['BZID'].astype(str).str.strip()
        
        # Merge agent data with sales data first
        if not sales_data.empty and 'Bzid' in sales_data.columns:
            sales_data.rename(columns={'Bzid': 'BZID'}, inplace=True)
            agents = pd.merge(agents, sales_data, on='BZID', how='left')
        else:
            print("Warning: Could not merge with sales data - region and organization info will be missing")
        
        # Convert agent data to list of dicts for the data store
        agent_data_list = agents.to_dict('records')
        
        # Add repayment metrics to each agent's data
        if not repay_metrics.empty:
            # Standardize column names to uppercase for consistent merging
            repay_metrics = repay_metrics.rename(columns={col: col.upper() for col in repay_metrics.columns})
            
            # Ensure BZID is in the repay_metrics columns (case insensitive)
            bzid_col = next((col for col in repay_metrics.columns if col.upper() == 'BZID'), 'BZID')
            
            # Convert BZID to string and strip whitespace
            repay_metrics[bzid_col] = repay_metrics[bzid_col].astype(str).str.strip()
            
            # Create a mapping of BZID to repayment metrics
            repay_dict = repay_metrics.set_index(bzid_col).to_dict('index')
            
            # Add repayment metrics to each agent's data
            for agent in agent_data_list:
                bzid = str(agent.get('BZID', '')).strip()
                if bzid in repay_dict:
                    if 'sources' not in agent:
                        agent['sources'] = {}
                    agent['sources']['repayments'] = repay_dict[bzid]
                    
                    # Also add repayment metrics to the top level for backward compatibility
                    for key, value in repay_dict[bzid].items():
                        if key not in agent:  # Don't overwrite existing fields
                            agent[key.lower()] = value
            
            # Save agent data to the unified store using the correct method
            for agent_dict in agent_data_list:
                bzid = str(agent_dict.get('BZID') or agent_dict.get('Bzid') or '')
                if bzid:
                    # Add agent data to the store
                    data_store.add_agent_data(
                        bzid=bzid,
                        source='profile',
                        data=agent_dict
                    )
                    print(f"Saved agent {bzid} to unified data store")
    
    # Convert agent data list to a DataFrame for easier handling
    agent_data = pd.DataFrame(agent_data_list)
    
    # Debug: Print sample of loaded data
    print("\nSample of agent data (first 5 agents):")
    if not agent_data.empty:
        # Get available columns for display
        available_columns = []
        for col in ['BZID', 'Bzid', 'organizationname', 'organization_name', 'Organization', 'name']:
            if col in agent_data.columns:
                available_columns.append(col)
                if len(available_columns) >= 2:  # Show at most 2 columns
                    break
        
        if available_columns:
            print(agent_data[available_columns].head())
        else:
            print("No standard identifier columns found in agent data")
            print("Available columns:", agent_data.columns.tolist())
    else:
        print("No agent data available")
    
    while True:
        print("\nEnter BZID to look up (or 'q' to quit): ")
        bzid = input().strip()
        
        if bzid.lower() == 'q':
            break
            
        if bzid == '':
            continue
        
        # Look up agent in the unified data store
        agent_dict = data_store.get_agent(bzid)
        
        if not agent_dict:
            # Fallback to DataFrame lookup if not found in unified store
            agent = agent_data[agent_data['BZID'].astype(str).str.strip() == bzid]
            
            if agent.empty:
                print(f"Agent with BZID {bzid} not found.")
                continue
                
            # Convert agent data to dict for analysis
            agent_dict = agent.iloc[0].to_dict()
            
            # Save to unified store for future lookups
            data_store.add_agent_data(
                bzid=bzid,
                source='profile',
                data=agent_dict
            )
            print(f"Saved agent {bzid} to unified data store")
        
        # Debug: Print available keys in agent_dict
        print(f"\n[DEBUG] Available keys in agent data: {list(agent_dict.keys())}")
        if 'sources' in agent_dict and agent_dict['sources']:
            print(f"[DEBUG] Available sources: {list(agent_dict['sources'].keys())}")
        
        # Analyze agent data
        reporter = AgentReporter(agent_dict)
        analysis = reporter.analyze()
        
        # Classify the agent
        classifier = AgentClassifier()
        classification = classifier.classify_agent(agent_dict)
        
        # Add classification to analysis
        analysis['classification'] = classification
        
        # Print the profile
        print_agent_profile(bzid, agent_dict, analysis)

if __name__ == "__main__":
    main()