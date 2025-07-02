"""
Agent Profile Lookup Tool

This script provides an interactive interface to look up agent profiles
and display their credit risk analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum, auto
import os

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
    
    def analyze_credit_utilization(self) -> None:
        """Analyze credit utilization metrics with updated risk levels.
        
        Risk levels based on utilization:
        - <30%: High risk (underutilization)
        - 30-70%: Medium risk (optimal utilization)
        - >70%: Low risk (good utilization)
        """
        if 'Credit Limit' in self.agent_data and 'Credit Line Balance' in self.agent_data:
            limit = self.agent_data['Credit Limit']
            balance = self.agent_data['Credit Line Balance']
            
            if pd.notnull(limit) and limit > 0:
                utilization = (balance / limit) * 100
                self.metrics['credit_utilization'] = utilization
                
                # Updated risk levels based on new criteria
                if utilization < 30:
                    risk = RiskLevel.HIGH
                    risk_label = "High (Underutilization)"
                elif utilization <= 70:
                    risk = RiskLevel.MODERATE
                    risk_label = "Medium (Optimal)"
                else:
                    risk = RiskLevel.LOW
                    risk_label = "Low (Good Utilization)"
                
                self.assessments.append({
                    'metric': 'Credit Utilization',
                    'value': f"{utilization:.1f}% - {risk_label}",
                    'risk': risk
                })
                
                # Add findings and recommendations based on risk level
                if risk == RiskLevel.HIGH:
                    self.findings.append("High Risk: Credit utilization below 30% indicates underutilization")
                    self.recommendations.append("Consider increasing credit line usage or reviewing credit needs")
                elif risk == RiskLevel.MODERATE:
                    self.findings.append("Optimal credit utilization range (30-70%)")
                    self.recommendations.append("Continue current credit usage patterns")
                else:
                    self.findings.append("Good credit utilization (above 70%)")
                    self.recommendations.append("Monitor for any sudden changes in utilization patterns")
    
    def analyze_repayment(self) -> None:
        """Analyze repayment metrics and DPD (Days Past Due) status."""
        # Analyze credit score if available
        if 'credit_score' in self.agent_data and pd.notnull(self.agent_data['credit_score']):
            score = self.agent_data['credit_score']
            self.metrics['credit_score'] = score
            
            if score < 30000:
                risk = RiskLevel.HIGH
                score_label = "High Risk"
            elif score < 70000:
                risk = RiskLevel.MODERATE
                score_label = "Medium Risk"
            else:
                risk = RiskLevel.LOW
                score_label = "Low Risk"
                
            self.assessments.append({
                'metric': 'Credit Score',
                'value': f"{score:,} - {score_label}",
                'risk': risk
            })
        
        # Analyze DPD (Days Past Due) if available
        if 'Dpd' in self.agent_data and pd.notnull(self.agent_data['Dpd']):
            dpd = self.agent_data['Dpd']
            self.metrics['dpd'] = dpd
            
            if dpd > 30:
                risk = RiskLevel.CRITICAL
                dpd_label = "Severe Delinquency"
                self.findings.append(f"Critical: {dpd} days past due - account is severely delinquent")
                self.recommendations.append("Immediate collection action required")
            elif dpd > 15:
                risk = RiskLevel.HIGH
                dpd_label = "High Risk Delinquency"
                self.findings.append(f"High Risk: {dpd} days past due - account is delinquent")
                self.recommendations.append("Initiate collection process and review credit terms")
            elif dpd > 7:
                risk = RiskLevel.MODERATE
                dpd_label = "Moderate Risk"
                self.findings.append(f"Moderate Risk: {dpd} days past due - monitor closely")
                self.recommendations.append("Send payment reminder and follow up")
            else:
                risk = RiskLevel.LOW
                dpd_label = "Current"
                
            self.assessments.append({
                'metric': 'Days Past Due (DPD)',
                'value': f"{dpd} days - {dpd_label}",
                'risk': risk
            })
            
            self.assessments.append({
                'metric': 'Credit Score',
                'value': f"{score:,.0f}",
                'risk': risk
            })
            
            if score < 30000:
                self.findings.append("Low credit score indicates higher risk of default")
                self.recommendations.append("Consider additional credit checks or collateral requirements")
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on analysis."""
        if not self.recommendations:
            self.recommendations.append("No specific recommendations at this time.")
    
    def analyze(self) -> Dict[str, Any]:
        """Run all analyses and return results."""
        self.analyze_credit_utilization()
        self.analyze_repayment()
        self.generate_recommendations()
        
        return {
            'metrics': self.metrics,
            'assessments': self.assessments,
            'findings': self.findings,
            'recommendations': self.recommendations
        }

def load_agents() -> pd.DataFrame:
    """Load agent data from Excel file."""
    try:
        agents = pd.read_excel('source_data/credit_Agents.xlsx')
        agents['Bzid'] = agents['Bzid'].astype(str)
        return agents
    except Exception as e:
        print(f"Error loading agent data: {e}")
        return pd.DataFrame()

def load_sales() -> pd.DataFrame:
    """Load sales data from Excel file with region and organization info."""
    try:
        # Load sales data with specific columns we need
        sales = pd.read_excel('source_data/sales_data.xlsx', 
                            usecols=['account', 'organizationname', 'city', 'State', 'Region', 'Ro Name', 'RM Name'])
        
        # Standardize column names and clean data
        sales = (sales
                .rename(columns={'account': 'Bzid', 'Region': 'Region'})
                .drop_duplicates('Bzid')  # Keep only one record per agent
                .assign(Bzid=lambda x: x['Bzid'].astype(str).str.strip())
                .dropna(subset=['Bzid']))  # Remove rows with missing Bzid
                
        return sales
    except Exception as e:
        print(f"Error loading sales data: {e}")
        print("Available columns in sales data:")
        try:
            print(pd.read_excel('source_data/sales_data.xlsx', nrows=0).columns.tolist())
        except Exception as e2:
            print(f"Could not read column names: {e2}")
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

def load_repayment_metrics() -> pd.DataFrame:
    """Load repayment metrics from CSV file."""
    try:
        # First check if the file exists
        if not os.path.exists('output/repayment_metrics_with_scores.csv'):
            # If not, run analyze_repayments.py to generate it
            try:
                import analyze_repayments
                analyze_repayments.main()
            except Exception as e:
                print(f"Error running analyze_repayments.py: {e}")
                return pd.DataFrame()
        
        # Now try to load the file
        repay = pd.read_csv('output/repayment_metrics_with_scores.csv')
        
        # Handle case sensitivity in column names
        repay.columns = [col.strip() for col in repay.columns]
        
        # Check if 'Bzid' column exists (case insensitive)
        bzid_col = next((col for col in repay.columns if col.lower() == 'bzid'), None)
        if bzid_col is None:
            # If no Bzid column, check if index is named 'bzid' (case insensitive)
            if repay.index.name and repay.index.name.lower() == 'bzid':
                repay = repay.reset_index()
                bzid_col = 'bzid'
            else:
                print("Error: Could not find 'Bzid' column in repayment metrics")
                return pd.DataFrame()
        
        # Standardize the column name to 'Bzid'
        if bzid_col != 'Bzid':
            repay = repay.rename(columns={bzid_col: 'Bzid'})
        
        repay['Bzid'] = repay['Bzid'].astype(str)
        return repay
    except Exception as e:
        print(f"Error loading repayment metrics: {e}")
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
        bar = "█" * bar_length
        
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
        
        # Format the line with aligned columns
        line = (
            f"  \033[1m{component:<33}\033[0m "  # Component name in bold
            f"{color}{score_str:<7}\033[0m "      # Score with color
            f"{color}▕{bar:<{max_width}}▏\033[0m "  # Bar with borders
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
    print(f"🤖 AGENT CREDIT RISK PROFILE: {bzid}".center(80))
    print("=" * 80 + "\033[0m")
    
    # Print last updated timestamp
    from datetime import datetime
    print(f"\033[90mAgent ID: {bzid} • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print("\033[90m" + "-" * 80 + "\033[0m")
    
    # --- Basic Information ---
    print_section("Basic Information", "👤")
    
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
    
    credit_limit = agent_data.get('Credit Limit')
    credit_used = agent_data.get('Credit Line Balance')
    
    # Print credit limit information
    limit_status = "\033[92m✓ Available" if pd.notnull(credit_limit) and credit_limit > 0 else "\033[93mNot Available"
    print_info("Credit Limit:", f"{fmt_curr(credit_limit)}  {limit_status}\033[0m", 
              "Total credit available to the agent")
    
    # Print credit used information
    used_status = "\033[93mNo Usage" if pd.notnull(credit_used) and credit_used == 0 else ""
    print_info("Credit Used:", f"{fmt_curr(credit_used)}  {used_status}\033[0m", 
              "Current credit utilization")
    
    # Calculate and display utilization with color coding and risk assessment
    if pd.notnull(credit_limit) and credit_limit > 0 and pd.notnull(credit_used):
        utilization = (credit_used / credit_limit) * 100
        
        # Determine utilization risk level with emojis and colors
        if utilization < 30:
            util_risk = "\033[91m⚠️ High Risk (Underutilized)"
            risk_emoji = "🔴"
            risk_desc = "Credit underutilization may indicate business issues"
        elif utilization <= 70:
            util_risk = "\033[92m✓ Optimal Utilization"
            risk_emoji = "🟢"
            risk_desc = "Healthy credit usage pattern"
        else:
            util_risk = "\033[93m⚠️ Approaching Limit"
            risk_emoji = "🟡"
            risk_desc = "High utilization may impact credit score"
        
        # Print utilization metrics
        print_info("Utilization:", 
                  f"{fmt_pct(utilization)} {risk_emoji}", 
                  "Percentage of available credit being used")
        
        print_info("Risk Level:", 
                  f"{util_risk}\033[0m", 
                  risk_desc)
        
        # Add utilization gauge visualization
        gauge_width = 40
        used_width = int((utilization / 100) * gauge_width)
        gauge = "[" + "█" * used_width + " " * (gauge_width - used_width) + "]"
        print(f"\n  {gauge} {utilization:.1f}%")
        print("  " + " " * (gauge_width//2 - 3) + "↑ Ideal Range")
        print("  " + " " * 5 + "0%" + " " * (gauge_width-15) + "30%" + " " * 10 + "70%" + " " * 10 + "100%")
    else:
        print_info("Utilization:", "N/A", "Insufficient data to calculate credit utilization")
    
    # --- Repayment Performance ---
    print_section("Repayment Performance", "💰")
    
    if 'total_repayment_amount' in agent_data:
        total_repaid = agent_data.get('total_repayment_amount', 0)
        principal_repaid = agent_data.get('total_principal_repaid', 0)
        txn_count = agent_data.get('transaction_count', 0)
        
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
    print("Loading agent data...")
    agents = load_agents()
    
    if agents.empty:
        print("Failed to load agent data. Exiting.")
        return
    
    print("Loading sales data...")
    sales_data = load_sales()
    
    print("Loading repayment metrics...")
    repay_metrics = load_repayment_metrics()
    
    if repay_metrics.empty:
        print("Repayment metrics not found. Running analysis...")
        try:
            import analyze_repayments
            analyze_repayments.main()
            repay_metrics = load_repayment_metrics()
        except Exception as e:
            print(f"Failed to generate repayment metrics: {e}")
    
    # Ensure Bzid column exists in all DataFrames and has consistent case
    agents['Bzid'] = agents['Bzid'].astype(str).str.strip()
    
    # Merge agent data with sales data first
    if not sales_data.empty and 'Bzid' in sales_data.columns:
        agents = pd.merge(agents, sales_data, on='Bzid', how='left')
    else:
        print("Warning: Could not merge with sales data - region and organization info will be missing")
    
    # Then merge with repayment metrics
    if not repay_metrics.empty and 'Bzid' in repay_metrics.columns:
        agent_data = pd.merge(agents, repay_metrics, on='Bzid', how='left')
    else:
        print("Warning: Could not merge with repayment metrics - using agent data only")
        agent_data = agents
    
    while True:
        print("\nEnter Bzid to look up (or 'q' to quit): ")
        bzid = input().strip()
        
        if bzid.lower() == 'q':
            break
            
        if bzid == '':
            continue
            
        agent = agent_data[agent_data['Bzid'] == bzid]
        
        if agent.empty:
            print(f"Agent with Bzid {bzid} not found.")
            continue
            
        # Convert agent data to dict for analysis
        agent_dict = agent.iloc[0].to_dict()
        
        # Analyze agent data
        reporter = AgentReporter(agent_dict)
        analysis = reporter.analyze()
        
        # Print the profile
        print_agent_profile(bzid, agent_dict, analysis)

if __name__ == "__main__":
    main()