"""
Credit Sales Analyzer

This script analyzes credit sales data to derive key metrics related to credit behavior,
sales performance, and financial risk. It processes monthly sales data to provide
insights into credit utilization patterns and risk indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Constants
CREDIT_THRESHOLD_HIGH = 0.5  # 50% credit usage is considered high
CREDIT_THRESHOLD_LOW = 0.3   # 30% credit usage is considered low
DORMANT_MONTHS = 3           # Number of consecutive months to consider an agent dormant

class CreditSalesAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with the path to the credit sales data."""
        self.file_path = file_path
        self.df = None
        self.metrics = None
        
    def load_data(self) -> None:
        """Load and preprocess the credit sales data."""
        try:
            # Read the Excel file
            self.df = pd.read_excel(self.file_path)
            
            # Basic validation
            if self.df.empty:
                raise ValueError("The input file is empty")
                
            # Rename columns to standard format
            self.df.columns = [col.strip() for col in self.df.columns]
            
            # Ensure required columns exist
            required_columns = ['Account']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
            print(f"Successfully loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self) -> None:
        """Preprocess the data for analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert date columns if they exist
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'month' in col.lower()]
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Sort by Account and date if available
        sort_columns = ['Account']
        if date_columns:
            sort_columns.extend(date_columns)
        self.df = self.df.sort_values(by=sort_columns)
        
        # Reset index after sorting
        self.df = self.df.reset_index(drop=True)
    
    def calculate_monthly_metrics(self) -> pd.DataFrame:
        """Calculate monthly metrics for each agent."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying the original dataframe
        metrics_df = self.df.copy()
        
        # Get all GMV and Credit GMV columns
        gmv_columns = [col for col in metrics_df.columns if 'GMV' in col and 'Credit' not in col]
        credit_columns = [col for col in metrics_df.columns if 'Credit' in col and 'GMV' in col]
        
        # Calculate credit ratio for each month
        for gmv_col, credit_col in zip(gmv_columns, credit_columns):
            # Get month name from column (e.g., 'Jan Total GMV - Year25' -> 'Jan')
            month = gmv_col.split()[0]
            ratio_col = f"{month}_Credit_Ratio"
            
            # Calculate credit ratio (Credit GMV / Total GMV)
            metrics_df[ratio_col] = metrics_df[credit_col] / metrics_df[gmv_col].replace(0, np.nan)
        
        return metrics_df
    
    def calculate_aggregate_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate metrics across all months."""
        # Define month names
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June']
        
        # Calculate metrics for each agent
        results = []
        
        for account, group in metrics_df.groupby('Account'):
            # Initialize variables for this agent
            monthly_ratios = []
            monthly_gmv = []
            monthly_credit = []
            total_gmv = 0
            total_credit = 0
            
            # Process each month's data
            for month in months:
                # Handle June's inconsistent column name
                month_prefix = f"{month} " if month != 'June' else month
                
                # Define column names for this month
                gmv_col = f"{month_prefix}Total GMV - Year25"
                credit_col = f"{month_prefix}Credit Gmv"
                
                # Skip if columns don't exist
                if gmv_col not in group.columns or credit_col not in group.columns:
                    continue
                    
                # Get monthly values
                month_gmv = group[gmv_col].sum()
                month_credit = group[credit_col].sum()
                
                # Update totals
                total_gmv += month_gmv
                total_credit += month_credit
                
                # Calculate monthly ratio if GMV > 0
                if month_gmv > 0:
                    month_ratio = month_credit / month_gmv
                    monthly_ratios.append(month_ratio)
                    monthly_gmv.append(month_gmv)
                    monthly_credit.append(month_credit)
            
            # Calculate overall ratio
            overall_ratio = total_credit / total_gmv if total_gmv > 0 else 0
            
            # Calculate statistics from monthly data
            if monthly_ratios:
                avg_credit_ratio = np.mean(monthly_ratios)
                max_credit_ratio = np.max(monthly_ratios)
                min_credit_ratio = np.min(monthly_ratios)
                credit_ratio_std = np.std(monthly_ratios, ddof=1) if len(monthly_ratios) > 1 else 0
                high_credit_months = np.sum(np.array(monthly_ratios) > CREDIT_THRESHOLD_HIGH)
                zero_credit_months = np.sum(np.array(monthly_ratios) == 0)
            else:
                avg_credit_ratio = max_credit_ratio = min_credit_ratio = credit_ratio_std = 0
                high_credit_months = zero_credit_months = 0
            
            # Trend analysis (simple linear regression)
            if len(monthly_ratios) > 1:
                x = np.arange(len(monthly_ratios))
                slope = np.polyfit(x, monthly_ratios, 1)[0]
                trend = "Increasing" if slope > 0.01 else ("Decreasing" if slope < -0.01 else "Stable")
            else:
                trend = "Insufficient Data"
            
            # Risk indicators
            risk_indicators = {
                'high_credit_dependence': avg_credit_ratio > CREDIT_THRESHOLD_HIGH,
                'low_credit_utilization': avg_credit_ratio < CREDIT_THRESHOLD_LOW,
                'dormant_agent': zero_credit_months >= DORMANT_MONTHS,
                'credit_trend': trend
            }
            
            # Add to results
            result = {
                'account': account,
                'total_gmv': total_gmv,
                'total_credit_gmv': total_credit,
                'overall_credit_ratio': overall_ratio,
                'avg_monthly_credit_ratio': avg_credit_ratio,
                'max_monthly_credit_ratio': max_credit_ratio,
                'min_monthly_credit_ratio': min_credit_ratio,
                'credit_ratio_std': credit_ratio_std,
                'high_credit_months': high_credit_months,
                'zero_credit_months': zero_credit_months,
                'credit_trend': trend,
                **risk_indicators
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_regional_metrics(self, metrics_df: pd.DataFrame, region_column: Optional[str] = None) -> pd.DataFrame:
        """Calculate regional metrics if region data is available."""
        if region_column and region_column in metrics_df.columns:
            regional_metrics = metrics_df.groupby(region_column).agg({
                'total_gmv': 'sum',
                'total_credit_gmv': 'sum',
                'avg_monthly_credit_ratio': 'mean',
                'high_credit_months': 'sum',
                'zero_credit_months': 'sum'
            }).reset_index()
            
            regional_metrics['region_credit_ratio'] = \
                regional_metrics['total_credit_gmv'] / regional_metrics['total_gmv'].replace(0, np.nan)
                
            return regional_metrics
        return pd.DataFrame()
    
    def cluster_agents(self, metrics_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """Cluster agents based on their credit behavior."""
        # Select features for clustering
        features = [
            'avg_monthly_credit_ratio',
            'credit_ratio_std',
            'total_gmv',
            'total_credit_gmv',
            'high_credit_months'
        ]
        
        # Prepare data for clustering
        X = metrics_df[features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        metrics_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        return metrics_df
    
    def generate_credit_health_score(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a credit health score for each agent."""
        # Normalize metrics (0-1 scale)
        metrics_df['score_credit_ratio'] = 1 - np.minimum(1, metrics_df['avg_monthly_credit_ratio'])
        metrics_df['score_volatility'] = 1 - np.minimum(1, metrics_df['credit_ratio_std'])
        metrics_df['score_gmv'] = (metrics_df['total_gmv'] - metrics_df['total_gmv'].min()) / \
                                 (metrics_df['total_gmv'].max() - metrics_df['total_gmv'].min() + 1e-6)
        
        # Calculate composite score (weighted average)
        weights = {
            'score_credit_ratio': 0.4,  # Credit utilization (lower is better)
            'score_volatility': 0.3,    # Stability (lower volatility is better)
            'score_gmv': 0.3            # Business volume (higher is better)
        }
        
        metrics_df['credit_health_score'] = (
            metrics_df['score_credit_ratio'] * weights['score_credit_ratio'] +
            metrics_df['score_volatility'] * weights['score_volatility'] +
            metrics_df['score_gmv'] * weights['score_gmv']
        ) * 100  # Scale to 0-100
        
        return metrics_df
    
    def generate_report(self, output_dir: str = 'output') -> None:
        """Generate analysis reports and visualizations."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save metrics to CSV
        self.metrics.to_csv(f"{output_dir}/credit_sales_metrics.csv", index=False)
        
        # Generate visualizations
        self._plot_credit_distribution(self.metrics, output_dir)
        self._plot_trend_analysis(self.metrics, output_dir)
        
        print(f"Analysis complete. Reports saved to {output_dir}/")
    
    def _plot_credit_distribution(self, metrics_df: pd.DataFrame, output_dir: str) -> None:
        """Plot distribution of credit ratios."""
        plt.figure(figsize=(10, 6))
        sns.histplot(metrics_df['avg_monthly_credit_ratio'].dropna(), bins=20, kde=True)
        plt.axvline(CREDIT_THRESHOLD_HIGH, color='r', linestyle='--', label=f'High Credit Threshold ({CREDIT_THRESHOLD_HIGH*100}%)')
        plt.axvline(CREDIT_THRESHOLD_LOW, color='g', linestyle='--', label=f'Low Credit Threshold ({CREDIT_THRESHOLD_LOW*100}%)')
        plt.title('Distribution of Average Monthly Credit Ratios')
        plt.xlabel('Credit Ratio (Credit GMV / Total GMV)')
        plt.ylabel('Number of Agents')
        plt.legend()
        plt.savefig(f"{output_dir}/credit_ratio_distribution.png")
        plt.close()
    
    def _plot_trend_analysis(self, metrics_df: pd.DataFrame, output_dir: str) -> None:
        """Plot trend analysis."""
        trend_counts = metrics_df['credit_trend'].value_counts()
        
        plt.figure(figsize=(8, 6))
        trend_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Agent Credit Trend Distribution')
        plt.xlabel('Trend')
        plt.ylabel('Number of Agents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/credit_trend_analysis.png")
        plt.close()
    
    def lookup_agent(self, agent_id: str) -> Optional[Dict]:
        """
        Look up detailed information for a specific agent.
        
        Args:
            agent_id: The agent ID to look up
            
        Returns:
            Dictionary containing agent details or None if not found
        """
        if self.metrics is None:
            print("No metrics available. Please run the analysis first.")
            return None
            
        agent_data = self.metrics[self.metrics['account'] == int(agent_id)]
        
        if agent_data.empty:
            print(f"No data found for agent ID: {agent_id}")
            return None
            
        # Convert to dict for easier access
        agent_dict = agent_data.iloc[0].to_dict()
        
        # Calculate peer averages for comparison
        peer_avg = self.metrics.agg({
            'avg_monthly_credit_ratio': 'mean',
            'total_gmv': 'mean',
            'total_credit_gmv': 'mean',
            'credit_health_score': 'mean'
        }).to_dict()
        
        return {
            'agent_id': agent_id,
            'details': agent_dict,
            'peer_averages': peer_avg
        }
    
    def display_agent_profile(self, agent_data: Dict) -> None:
        """Display a formatted agent profile."""
        if not agent_data:
            return
            
        details = agent_data['details']
        peer_avg = agent_data['peer_averages']
        
        print("\n" + "="*80)
        print(f"{'AGENT CREDIT PROFILE':^80}")
        print("="*80)
        
        # Basic Info
        print("\n📋 BASIC INFORMATION")
        print("-"*40)
        print(f"{'Agent ID:':<25} {details['account']}")
        print(f"{'Credit Health Score:':<25} {details['credit_health_score']:.1f}/100")
        
        # Credit Metrics
        print("\n💳 CREDIT METRICS")
        print("-"*40)
        print(f"{'Avg. Monthly Credit Ratio:':<30} {details['avg_monthly_credit_ratio']*100:.1f}%")
        print(f"  {'(Peer Average):':<28} {peer_avg['avg_monthly_credit_ratio']*100:.1f}%")
        print(f"{'Total GMV:':<30} ₹{details['total_gmv']:,.2f}")
        print(f"  {'(Peer Average):':<28} ₹{peer_avg['total_gmv']:,.2f}")
        print(f"{'Total Credit GMV:':<30} ₹{details['total_credit_gmv']:,.2f}")
        print(f"  {'(Peer Average):':<28} ₹{peer_avg['total_credit_gmv']:,.2f}")
        
        # Risk Indicators
        print("\n⚠️  RISK INDICATORS")
        print("-"*40)
        risk_factors = []
        if details['high_credit_dependence']:
            risk_factors.append("High credit dependence")
        if details['low_credit_utilization']:
            risk_factors.append("Low credit utilization")
        if details['dormant_agent']:
            risk_factors.append("Dormant account")
            
        if risk_factors:
            for factor in risk_factors:
                print(f"• {factor}")
        else:
            print("No significant risk factors detected.")
            
        # Credit Trend
        trend_emoji = "📈" if details['credit_trend'] == "Increasing" else "📉" if details['credit_trend'] == "Decreasing" else "➡️"
        print(f"\n📊 CREDIT TREND: {trend_emoji} {details['credit_trend']}")
        
        print("\n" + "="*80)

def interactive_menu(analyzer: CreditSalesAnalyzer) -> None:
    """Run an interactive menu for the credit sales analyzer."""
    while True:
        print("\n" + "="*50)
        print(f"{'CREDIT SALES ANALYZER':^50}")
        print("="*50)
        print("1. Look up agent profile")
        print("2. Run full analysis")
        print("3. View report summary")
        print("4. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            agent_id = input("\nEnter agent ID to look up: ").strip()
            if agent_id.isdigit():
                agent_data = analyzer.lookup_agent(agent_id)
                if agent_data:
                    analyzer.display_agent_profile(agent_data)
            else:
                print("Invalid agent ID. Please enter a numeric ID.")
                
        elif choice == '2':
            print("\nRunning full analysis...")
            try:
                analyzer.generate_report('credit_analysis_reports')
                print("Analysis complete! Reports saved to 'credit_analysis_reports/'")
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                
        elif choice == '3':
            if analyzer.metrics is not None:
                print("\n📊 REPORT SUMMARY")
                print("-"*40)
                print(f"Total Agents Analyzed: {len(analyzer.metrics)}")
                print(f"Average Credit Health Score: {analyzer.metrics['credit_health_score'].mean():.1f}/100")
                print(f"Average Monthly Credit Ratio: {analyzer.metrics['avg_monthly_credit_ratio'].mean()*100:.1f}%")
                print("\nRisk Distribution:")
                print(analyzer.metrics['credit_trend'].value_counts())
            else:
                print("No analysis data available. Please run the full analysis first.")
                
        elif choice == '4':
            print("\nExiting Credit Sales Analyzer. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")

def main():
    """Main function to run the credit sales analysis."""
    print("\n" + "="*60)
    print(f"{'CREDIT SALES ANALYZER':^60}")
    print("="*60)
    print("Loading data and initializing analyzer...")
    
    try:
        # Initialize analyzer with the correct file path
        analyzer = CreditSalesAnalyzer('source_data/Credit_history_sales_vs_credit_sales.xlsx')
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        analyzer.load_data()
        analyzer.preprocess_data()
        
        # Calculate metrics
        print("Calculating monthly metrics...")
        monthly_metrics = analyzer.calculate_monthly_metrics()
        
        print("Calculating aggregate metrics...")
        analyzer.metrics = analyzer.calculate_aggregate_metrics(monthly_metrics)
        
        # Add advanced analytics
        print("Generating credit health scores...")
        analyzer.metrics = analyzer.generate_credit_health_score(analyzer.metrics)
        
        print("Performing agent clustering...")
        analyzer.metrics = analyzer.cluster_agents(analyzer.metrics)
        
        # Start interactive menu
        print("\nInitialization complete!")
        interactive_menu(analyzer)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please make sure the data file exists at: source_data/Credit_history_sales_vs_credit_sales.xlsx")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nThank you for using the Credit Sales Analyzer. Goodbye!")

if __name__ == "__main__":
    main()
