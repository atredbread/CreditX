"""
GMV (Gross Merchandise Value) Analyzer for Credit Health Intelligence Engine

This module handles the analysis of GMV trends and patterns for agents.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pandas as pd
import numpy as np
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Make matplotlib optional
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization features will be disabled.")

class GMVAnalyzer:
    """Analyzes GMV trends and patterns for agents."""
    
    def __init__(self, sales_data: pd.DataFrame, date_column: str = 'date', 
                 gmv_column: str = 'GMV', agent_column: str = 'account'):
        """
        Initialize the GMV analyzer with sales data.
        
        Args:
            sales_data: DataFrame containing sales data
            date_column: Name of the date column
            gmv_column: Name of the GMV column
            agent_column: Name of the agent identifier column
        """
        self.sales_data = sales_data.copy()
        self.date_column = date_column
        self.gmv_column = gmv_column
        self.agent_column = agent_column
        
        # Convert date column to datetime if it's not already
        if self.date_column in self.sales_data.columns:
            self.sales_data[self.date_column] = pd.to_datetime(self.sales_data[self.date_column])
    
    def calculate_gmv_trends(self, window: int = 3) -> pd.DataFrame:
        """
        Calculate GMV trends for each agent over time.
        
        Args:
            window: Number of months for trend calculation
            
        Returns:
            DataFrame with GMV trends for each agent
        """
        if self.sales_data.empty:
            logger.warning("No sales data available for GMV trend analysis")
            return pd.DataFrame()
        
        try:
            # Group by agent and month
            monthly_gmv = self.sales_data.groupby([
                self.agent_column,
                pd.Grouper(key=self.date_column, freq='M')
            ])[self.gmv_column].sum().reset_index()
            
            # Calculate month-over-month changes
            monthly_gmv['prev_month_gmv'] = monthly_gmv.groupby(
                self.agent_column
            )[self.gmv_column].shift(1)
            
            monthly_gmv['mom_growth'] = (
                (monthly_gmv[self.gmv_column] - monthly_gmv['prev_month_gmv']) / 
                monthly_gmv['prev_month_gmv'].replace(0, np.nan)
            ) * 100
            
            # Calculate rolling statistics
            monthly_gmv['rolling_avg'] = monthly_gmv.groupby(
                self.agent_column
            )[self.gmv_column].transform(lambda x: x.rolling(window=window).mean())
            
            monthly_gmv['rolling_std'] = monthly_gmv.groupby(
                self.agent_column
            )[self.gmv_column].transform(lambda x: x.rolling(window=window).std())
            
            # Calculate trend line using linear regression
            def calculate_trend(group):
                if len(group) < 2:
                    return np.nan
                x = np.arange(len(group))
                slope, _, _, _, _ = stats.linregress(x, group[self.gmv_column].values)
                return slope
            
            trends = monthly_gmv.groupby(self.agent_column).apply(calculate_trend)
            monthly_gmv['trend_slope'] = monthly_gmv[self.agent_column].map(trends)
            
            return monthly_gmv
            
        except Exception as e:
            logger.error(f"Error calculating GMV trends: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def identify_anomalies(self, z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Identify anomalies in GMV data using Z-scores.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame containing identified anomalies
        """
        if self.sales_data.empty:
            return pd.DataFrame()
        
        try:
            # Calculate Z-scores for GMV values
            self.sales_data['gmv_zscore'] = self.sales_data.groupby(
                self.agent_column
            )[self.gmv_column].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            
            # Identify anomalies
            anomalies = self.sales_data[
                abs(self.sales_data['gmv_zscore']) > z_threshold
            ].copy()
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error identifying GMV anomalies: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def generate_gmv_report(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Generate a GMV analysis report with visualizations.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Dictionary with paths to generated report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_files = {}
        
        try:
            # Calculate trends
            gmv_trends = self.calculate_gmv_trends()
            
            if not gmv_trends.empty:
                # Save trends to CSV
                trends_file = output_dir / 'gmv_trends.csv'
                gmv_trends.to_csv(trends_file, index=False)
                report_files['trends_csv'] = str(trends_file)
                
                if MATPLOTLIB_AVAILABLE:
                    # Generate trend visualization
                    plt.figure(figsize=(12, 6))
                    for agent_id, group in gmv_trends.groupby(self.agent_column):
                        plt.plot(
                            group[self.date_column], 
                            group['rolling_avg'], 
                            label=f'Agent {agent_id}'
                        )
                    
                    plt.title('GMV Trends by Agent')
                    plt.xlabel('Date')
                    plt.ylabel('Rolling Average GMV')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot
                    plot_file = output_dir / 'gmv_trends.png'
                    plt.savefig(plot_file)
                    plt.close()
                    report_files['trends_plot'] = str(plot_file)
                plt.tight_layout()
                
                # Save plot
                plot_file = output_dir / 'gmv_trends.png'
                plt.savefig(plot_file)
                plt.close()
                report_files['trends_plot'] = str(plot_file)
            
            # Identify anomalies
            anomalies = self.identify_anomalies()
            if not anomalies.empty:
                # Save anomalies to CSV
                anomalies_file = output_dir / 'gmv_anomalies.csv'
                anomalies.to_csv(anomalies_file, index=False)
                report_files['anomalies_csv'] = str(anomalies_file)
            
            return report_files
            
        except Exception as e:
            logger.error(f"Error generating GMV report: {str(e)}", exc_info=True)
            return {}

    def get_agent_gmv_summary(self, agent_id: str) -> Dict[str, float]:
        """
        Get GMV summary for a specific agent.
        
        Args:
            agent_id: Agent ID to get summary for
            
        Returns:
            Dictionary with GMV metrics for the agent
        """
        if self.sales_data.empty:
            return {}
        
        try:
            agent_data = self.sales_data[
                self.sales_data[self.agent_column] == agent_id
            ]
            
            if agent_data.empty:
                return {}
            
            # Calculate basic GMV metrics
            total_gmv = agent_data[self.gmv_column].sum()
            avg_gmv = agent_data[self.gmv_column].mean()
            max_gmv = agent_data[self.gmv_column].max()
            min_gmv = agent_data[self.gmv_column].min()
            
            # Calculate month-over-month growth
            monthly_gmv = agent_data.groupby(pd.Grouper(
                key=self.date_column, freq='M'
            ))[self.gmv_column].sum()
            
            mom_growth = monthly_gmv.pct_change().mean() * 100 if len(monthly_gmv) > 1 else 0
            
            return {
                'total_gmv': total_gmv,
                'avg_gmv': avg_gmv,
                'max_gmv': max_gmv,
                'min_gmv': min_gmv,
                'mom_growth_pct': mom_growth,
                'transaction_count': len(agent_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting GMV summary for agent {agent_id}: {str(e)}")
            return {}
