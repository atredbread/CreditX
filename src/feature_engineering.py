"""
Feature Engineering for Credit Health Intelligence Engine

This module handles the creation of features for agent classification
and risk assessment based on transaction and credit history data.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List, Union

import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for credit health analysis."""
    
    def __init__(self, data: Dict[str, pd.DataFrame], current_date: Optional[str] = None):
        """
        Initialize the feature engineer with raw data.
        
        Args:
            data: Dictionary of DataFrames containing the input data
            current_date: Reference date for calculations (YYYY-MM-DD format). 
                        If None, uses current date.
        """
        self.data = data
        self.current_date = pd.to_datetime(current_date) if current_date else datetime.now()
        self.features = pd.DataFrame()
    
    def calculate_credit_utilization(self) -> pd.Series:
        """
        Calculate credit utilization ratio for each agent.
        
        Returns:
            Series with credit utilization ratio (0-1)
        """
        logger.info("Calculating credit utilization...")
        agents = self.data.get('credit_Agents.xlsx')
        
        if agents is None:
            logger.error("credit_Agents.xlsx not found in input data")
            return pd.Series()
        
        # Calculate utilization as (Credit Line Balance / Credit Limit) * 100
        # Based on verified column names in credit_Agents.xlsx
        required_columns = ['Credit Limit', 'Credit Line Balance']
        if not all(col in agents.columns for col in required_columns):
            logger.error(f"Required columns not found in credit agents data. Available columns: {agents.columns.tolist()}")
            return pd.Series()
        
        # Handle division by zero and missing values
        utilization = (agents['Credit Line Balance'] / agents['Credit Limit'].replace(0, 1)) * 100
        utilization = utilization.clip(0, 100)  # Ensure between 0 and 100
        utilization.index = agents['Bzid']
        
        return utilization
    
    def calculate_repayment_score(self) -> pd.Series:
        """
        Calculate comprehensive repayment score incorporating DPD history, repayment patterns,
        and risk indicators.
        
        Returns:
            Series with repayment scores (300-900), where higher is better
            
        Scoring Components:
        - Base Score: 500 points
        - DPD History: -100 to +100 points based on DPD patterns
        - Payment Behavior: -50 to +50 points based on payment consistency
        - Risk Indicators: -100 to +100 points based on risk factors
        - Trend Analysis: -50 to +50 points based on improvement/decline
        
        Final score is scaled to 300-900 range (industry standard for credit scoring)
        """
        logger.info("Calculating comprehensive repayment scores...")
        
        # Get required data using verified file names
        dpd_data = self.data.get('DPD.xlsx')
        repayment_report = self.data.get('repayment report.csv')
        agents_data = self.data.get('credit_Agents.xlsx')
        
        if dpd_data is None:
            logger.error("DPD.xlsx not found in input data")
            return pd.Series()
            
        # Initialize scores with base value (500)
        agent_ids = set(dpd_data['Bzid'].unique())
        if agents_data is not None and not agents_data.empty:
            agent_ids.update(agents_data['Bzid'].unique())
        if repayment_report is not None and not repayment_report.empty:
            # Using correct column name from repayment report
            agent_col = 'All Anchors Onboarding Info - Anchor → Bzid'
            if agent_col in repayment_report.columns:
                agent_ids.update(repayment_report[agent_col].dropna().unique())
            
        if not agent_ids:
            logger.error("No agent IDs found in input data")
            return pd.Series()
            
        scores = pd.Series(500, index=list(agent_ids), name='repayment_score')
        
        try:
            # 1. Calculate DPD-based score component
            dpd_scores = self._calculate_dpd_score_component(dpd_data)
            if not dpd_scores.empty:
                scores = scores.add(dpd_scores, fill_value=0)
            
            # 2. Calculate payment behavior component
            payment_scores = self._calculate_payment_behavior(repayment_report)
            if not payment_scores.empty:
                scores = scores.add(payment_scores, fill_value=0)
            
            # 3. Apply risk adjustments
            risk_adjustments = self._calculate_risk_adjustments(dpd_data, repayment_report)
            if not risk_adjustments.empty:
                scores = scores.add(risk_adjustments, fill_value=0)
            
            # 4. Apply trend adjustments
            trend_adjustments = self._calculate_trend_adjustments(dpd_data, repayment_report)
            if not trend_adjustments.empty:
                scores = scores.add(trend_adjustments, fill_value=0)
            
            # Ensure scores are within bounds and handle missing values
            scores = scores.clip(300, 900).round().astype(int)
            
            logger.info(f"Calculated repayment scores for {len(scores)} agents")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating repayment scores: {str(e)}", exc_info=True)
            return pd.Series(index=agent_ids, name='repayment_score')
    
    def _calculate_dpd_score_component(self, dpd_data: pd.DataFrame) -> pd.Series:
        """Calculate DPD-based score component (-100 to +100).
        
        Uses verified column names from DPD.xlsx:
        - 'Dpd': Days Past Due
        - 'Bzid': Agent identifier
        - 'Pos': Point of Sale amount (used for weighting)
        """
        if dpd_data is None or dpd_data.empty or 'Dpd' not in dpd_data.columns:
            return pd.Series()
            
        try:
            # Ensure we have the required columns
            required_cols = ['Bzid', 'Dpd', 'Pos']
            if not all(col in dpd_data.columns for col in required_cols):
                logger.warning(f"Missing required columns in DPD data. Available: {dpd_data.columns.tolist()}")
                return pd.Series()
                
            # Calculate DPD metrics with Pos-weighted average
            dpd_metrics = dpd_data.groupby('Bzid').apply(
                lambda x: pd.Series({
                    'max_dpd': x['Dpd'].max(),
                    'mean_dpd': (x['Dpd'] * x['Pos']).sum() / x['Pos'].sum() if x['Pos'].sum() > 0 else x['Dpd'].mean(),
                    'count': len(x),
                    'recent_max_dpd': x.nlargest(3, 'Dpd')['Dpd'].mean() if len(x) >= 3 else x['Dpd'].max()
                })
            ).fillna(0)
            
            # Calculate base DPD score (0-100)
            dpd_metrics['dpd_score'] = 100 - np.minimum(100, 
                dpd_metrics['max_dpd'] * 0.5 + 
                dpd_metrics['mean_dpd'] * 0.3 + 
                dpd_metrics['recent_max_dpd'] * 0.2
            )
            
            # Convert to -100 to +100 range
            dpd_scores = (dpd_metrics['dpd_score'] - 50) * 2
            return dpd_scores
            
        except Exception as e:
            logger.error(f"Error calculating DPD score component: {str(e)}", exc_info=True)
            return pd.Series()
    
    def _calculate_payment_behavior(self, repayment_data: pd.DataFrame) -> pd.Series:
        """Calculate payment behavior component (-50 to +50).
        
        Uses verified column names from repayment report:
        - 'All Anchors Onboarding Info - Anchor → Bzid': Agent identifier
        - 'Customer Repayment Date': Date of repayment
        - 'Customer Repayment Amount': Total amount repaid
        - 'Customer Principle Repaid': Principal portion of repayment
        """
        if repayment_data is None or repayment_data.empty:
            return pd.Series()
            
        try:
            # Map verified column names
            col_mapping = {
                'agent_id': 'All Anchors Onboarding Info - Anchor → Bzid',
                'date_col': 'Customer Repayment Date',
                'amount_col': 'Customer Repayment Amount',
                'principal_col': 'Customer Principle Repaid'
            }
            
            # Find available columns
            available_cols = repayment_data.columns.tolist()
            col_mapping = {k: v for k, v in col_mapping.items() 
                         if v in available_cols}
            
            if not col_mapping.get('agent_id') or not col_mapping.get('date_col'):
                logger.warning("Missing required columns in repayment data")
                return pd.Series()
                
            # Process date column
            repayment_data[col_mapping['date_col']] = pd.to_datetime(
                repayment_data[col_mapping['date_col']], 
                errors='coerce'
            )
            repayment_data = repayment_data.dropna(subset=[col_mapping['date_col']])
            
            # Group by agent and calculate payment metrics
            agg_dict = {
                col_mapping['date_col']: ['min', 'max', 'count']
            }
            
            # Add amount columns if available
            if 'amount_col' in col_mapping:
                agg_dict[col_mapping['amount_col']] = ['sum', 'mean']
            if 'principal_col' in col_mapping:
                agg_dict[col_mapping['principal_col']] = ['sum']
            
            payment_metrics = repayment_data.groupby(
                col_mapping['agent_id']
            ).agg(agg_dict).fillna(0)
            
            # Flatten column multi-index
            payment_metrics.columns = ['_'.join(col).strip('_') for col in payment_metrics.columns]
            
            # Calculate payment consistency score (0-50)
            date_count_col = f"{col_mapping['date_col']}_count"
            if date_count_col in payment_metrics.columns:
                payment_metrics['consistency_score'] = np.minimum(50, 
                    payment_metrics[date_count_col] * 0.5  # More payments = better
                )
                
                # Convert to -50 to +50 range
                payment_scores = payment_metrics['consistency_score'] - 25
                return payment_scores
                
            return pd.Series()
            
        except Exception as e:
            logger.error(f"Error calculating payment behavior: {str(e)}", exc_info=True)
            return pd.Series()
            
        except Exception as e:
            logger.error(f"Error calculating payment behavior: {str(e)}")
            return pd.Series()
    
    def _calculate_risk_adjustments(self, dpd_data: pd.DataFrame, repayment_data: pd.DataFrame) -> pd.Series:
        """Calculate risk-based adjustments (-100 to +100)."""
        adjustments = pd.Series(0, index=dpd_data['Bzid'].unique())
        
        # Apply penalties for high DPD concentrations
        dpd_distribution = dpd_data.groupby('Bzid')['Dpd'].value_counts(normalize=True).unstack(fill_value=0)
        for dpd_level in [90, 60, 30]:
            if dpd_level in dpd_distribution.columns:
                adjustments -= dpd_distribution[dpd_level] * 50  # Up to -50 points for high DPD concentration
        
        # Apply bonuses for consistent on-time payments
        if repayment_data is not None and not repayment_data.empty:
            on_time_payers = repayment_data[
                repayment_data['status'].str.lower().str.contains('on.?time|paid', na=False, regex=True)
            ]['agent_id'].value_counts()
            adjustments = adjustments.add(on_time_payers * 0.1, fill_value=0)  # +10 points per on-time payment
        
        return adjustments.clip(-100, 100)
    
    def _calculate_trend_adjustments(self, dpd_data: pd.DataFrame, repayment_data: pd.DataFrame) -> pd.Series:
        """Calculate trend-based adjustments (-50 to +50)."""
        if dpd_data.empty or 'DueDate' not in dpd_data.columns:
            return pd.Series()
            
        try:
            # Calculate monthly DPD trends
            dpd_data['month'] = pd.to_datetime(dpd_data['DueDate']).dt.to_period('M')
            monthly_dpd = dpd_data.groupby(['Bzid', 'month'])['Dpd'].mean().unstack()
            
            # Calculate 3-month moving average slope
            def calculate_slope(series):
                if len(series.dropna()) < 2:
                    return 0
                x = np.arange(len(series))
                return np.polyfit(x, series, 1)[0]  # Returns slope
                
            trends = monthly_dpd.rolling(window=3, axis=1).apply(calculate_slope, raw=True)
            
            # Convert slope to adjustment (-50 to +50)
            trend_adjustments = trends.iloc[:, -1].clip(-5, 5) * 10  # Scale slope to meaningful range
            return trend_adjustments.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating trend adjustments: {str(e)}")
            return pd.Series()
    
    def calculate_credit_gmv_share(self) -> pd.Series:
        """
        Calculate the percentage of GMV (Gross Merchandise Value) that is on credit.
        
        Uses verified column names:
        - credit_sales_data.xlsx: 'Bzid' (agent ID), 'GMV' (credit sales amount)
        - sales_data.xlsx: 'Bzid' (agent ID), 'TotalSales' (total sales amount)
        
        Returns:
            Series with credit GMV share (0-1)
        """
        logger.info("Calculating credit GMV share...")
        credit_sales = self.data.get('Credit_sales_data.xlsx')
        sales_data = self.data.get('sales_data.xlsx')
        
        if credit_sales is None or sales_data is None:
            logger.error("Required sales data not found")
            return pd.Series()
        
        try:
            # Verify required columns exist
            if 'Bzid' not in credit_sales.columns or 'GMV' not in credit_sales.columns:
                logger.error("Missing required columns in credit sales data")
                return pd.Series()
                
            if 'Bzid' not in sales_data.columns or 'TotalSales' not in sales_data.columns:
                logger.error("Missing required columns in sales data")
                return pd.Series()
            
            # Calculate total credit GMV by agent
            credit_gmv = credit_sales.groupby('Bzid')['GMV'].sum()
            
            # Calculate total GMV by agent
            total_gmv = sales_data.groupby('Bzid')['TotalSales'].sum()
            
            # Calculate credit GMV share
            gmv_share = credit_gmv / total_gmv
            gmv_share = gmv_share.fillna(0).clip(0, 1)  # Handle division by zero and cap at 1
            
            return gmv_share
            
        except Exception as e:
            logger.error(f"Error calculating credit GMV share: {str(e)}", exc_info=True)
            return pd.Series()
        
        return gmv_share
    
    def calculate_gmv_trend(self, total_window: int = 6, credit_window: int = 5):
        """
        Calculate GMV growth trends for both total sales and credit sales.
        
        Args:
            total_window: Number of months to consider for total sales trend
            credit_window: Number of months to consider for credit sales trend
            
        Returns:
            DataFrame with columns:
            - total_gmv_trend: Normalized slope of total GMV trend
            - credit_gmv_trend: Normalized slope of credit GMV trend
            - credit_to_total_ratio: Ratio of credit GMV to total GMV
        """
        logger.info(f"Calculating GMV trends (total: {total_window}m, credit: {credit_window}m)...")
        
        # Get required data
        sales_data = self.data.get('sales_data.xlsx')
        credit_sales = self.data.get('credit_sales_data.xlsx')
        credit_history = self.data.get('credit_history_sales_vs_credit_sale.csv')
        
        if sales_data is None or sales_data.empty:
            logger.error("sales_data.xlsx not found or empty in input data")
            return pd.DataFrame()
            
        try:
            # 1. Process total sales trend
            total_trends = self._calculate_sales_trend(sales_data, total_window, 'total')
            
            # 2. Process credit sales trend (use credit_sales if available, otherwise use credit_history)
            if credit_sales is not None and not credit_sales.empty:
                credit_trends = self._calculate_sales_trend(credit_sales, credit_window, 'credit')
            elif credit_history is not None and not credit_history.empty:
                credit_trends = self._calculate_credit_history_trend(credit_history, credit_window)
            else:
                logger.warning("No credit sales data available")
                credit_trends = pd.Series(dtype=float, name='credit_gmv_trend')
            
            # 3. Combine results
            trends = pd.DataFrame(index=total_trends.index)
            trends['total_gmv_trend'] = total_trends
            
            if not credit_trends.empty:
                trends = trends.join(credit_trends, how='left')
                
                # Calculate credit to total GMV ratio if both available
                if 'credit_gmv_trend' in trends.columns:
                    # Get latest month's GMV for ratio calculation
                    total_gmv = self._get_latest_monthly_gmv(sales_data, 'total')
                    credit_gmv = self._get_latest_monthly_gmv(credit_sales or credit_history, 'credit')
                    
                    if total_gmv is not None and credit_gmv is not None:
                        # Align indices and calculate ratio
                        ratio = credit_gmv.div(total_gmv, fill_value=0)
                        trends['credit_to_total_ratio'] = ratio
            
            logger.info(f"Calculated GMV trends for {len(trends)} agents")
            return trends.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating GMV trends: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_sales_trend(self, sales_data: pd.DataFrame, window: int, sales_type: str) -> pd.Series:
        """Calculate trend for a given sales dataset.
        
        Uses verified column names:
        - For total sales: 'Bzid' (agent ID), 'SaleDate', 'TotalSales'
        - For credit sales: 'Bzid' (agent ID), 'TransactionDate', 'Amount'
        """
        try:
            # Make a copy to avoid modifying the original data
            sales_data = sales_data.copy()
            
            # Map column names based on sales type
            col_mapping = {
                'total': {
                    'id_col': 'Bzid',
                    'date_col': 'SaleDate',
                    'amount_col': 'TotalSales'
                },
                'credit': {
                    'id_col': 'Bzid',
                    'date_col': 'TransactionDate',
                    'amount_col': 'Amount'
                }
            }
            
            # Get column mapping for current sales type
            cols = col_mapping.get(sales_type, {})
            
            # Verify required columns exist
            missing_cols = [col for col in cols.values() if col not in sales_data.columns]
            if missing_cols:
                logger.error(f"Missing columns in {sales_type} data: {', '.join(missing_cols)}")
                return pd.Series()
            
            # Convert date column
            sales_data[cols['date_col']] = pd.to_datetime(sales_data[cols['date_col']], errors='coerce')
            sales_data = sales_data.dropna(subset=[cols['date_col']])
            
            # Convert amount to numeric
            sales_data['gmv_numeric'] = pd.to_numeric(sales_data[cols['amount_col']], errors='coerce')
            sales_data = sales_data.dropna(subset=['gmv_numeric'])
            
            if sales_data.empty:
                logger.warning(f"No valid {sales_type} GMV data found")
                return pd.Series()
            
            # Get the most recent date and filter window
            max_date = sales_data[cols['date_col']].max()
            min_date = max_date - pd.DateOffset(months=window)
            recent_sales = sales_data[sales_data[cols['date_col']] >= min_date].copy()
            
            if recent_sales.empty:
                logger.warning(f"No {sales_type} data found in the last {window} months")
                return pd.Series()
            
            # Group by month and calculate trends
            recent_sales['month'] = recent_sales[cols['date_col']].dt.to_period('M')
            monthly_gmv = recent_sales.groupby([cols['id_col'], 'month'])['gmv_numeric'].sum().unstack()
            
            if monthly_gmv.empty:
                return pd.Series()
                
            # Calculate trend for each agent
            trends = monthly_gmv.apply(self._calculate_trend_slope, axis=1)
            return trends.rename(f"{sales_type}_gmv_trend")
            
        except Exception as e:
            logger.error(f"Error calculating {sales_type} sales trend: {str(e)}", exc_info=True)
            return pd.Series()
    
    def _calculate_credit_history_trend(self, credit_history: pd.DataFrame, window: int) -> pd.Series:
        """Calculate trend from credit history data."""
        try:
            # Convert date column
            date_col = next((col for col in credit_history.columns if 'date' in col.lower()), None)
            if date_col is None:
                logger.error("Could not find date column in credit history data")
                return pd.Series()
                
            credit_history[date_col] = pd.to_datetime(credit_history[date_col], errors='coerce')
            credit_history = credit_history.dropna(subset=[date_col])
            
            # Find credit GMV column
            credit_gmv_col = next((col for col in ['credit_gmv', 'credit_amount', 'amount'] 
                                 if col in credit_history.columns), None)
            
            if credit_gmv_col is None:
                logger.error("Could not find credit GMV column in credit history data")
                return pd.Series()
                
            # Convert GMV to numeric
            credit_history['gmv_numeric'] = pd.to_numeric(credit_history[credit_gmv_col], errors='coerce')
            credit_history = credit_history.dropna(subset=['gmv_numeric'])
            
            if credit_history.empty:
                logger.warning("No valid credit GMV data found in credit history")
                return pd.Series()
            
            # Get the most recent date and filter window
            max_date = credit_history[date_col].max()
            min_date = max_date - pd.DateOffset(months=window)
            recent_credits = credit_history[credit_history[date_col] >= min_date].copy()
            
            if recent_credits.empty:
                logger.warning(f"No credit data found in the last {window} months")
                return pd.Series()
            
            # Group by month and calculate trends
            recent_credits['month'] = recent_credits[date_col].dt.to_period('M')
            monthly_credit = recent_credits.groupby(['account', 'month'])['gmv_numeric'].sum().unstack()
            
            if monthly_credit.empty:
                return pd.Series()
                
            # Calculate trend for each agent
            trends = monthly_credit.apply(self._calculate_trend_slope, axis=1)
            return trends.rename('credit_gmv_trend')
            
        except Exception as e:
            logger.error(f"Error calculating credit history trend: {str(e)}", exc_info=True)
            return pd.Series()
    
    @staticmethod
    def _calculate_trend_slope(series: pd.Series) -> float:
        """Calculate normalized slope of a time series."""
        series = series.dropna()
        if len(series) < 2:  # Need at least 2 points for a trend
            return 0.0
            
        x = np.arange(len(series))
        y = series.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]  # [slope, intercept]
            
            # Normalize by average value to get percentage change
            avg = np.mean(y)
            if avg != 0:
                return slope / avg
            return 0.0
        except Exception as e:
            logger.debug(f"Error calculating trend slope: {str(e)}")
            return 0.0
    
    def _get_latest_monthly_gmv(self, data: pd.DataFrame, data_type: str) -> Optional[pd.Series]:
        """Get the latest month's GMV for each account."""
        try:
            if data is None or data.empty:
                return None
                
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Find date and amount columns
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            amount_cols = ['gmv', 'GMV', 'amount', 'total_amount', 'sale_amount']
            amount_col = next((col for col in amount_cols if col in df.columns), None)
            
            if date_col is None or amount_col is None:
                return None
                
            # Convert date and amount
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            df = df.dropna(subset=[date_col, amount_col])
            
            if df.empty:
                return None
                
            # Get the most recent month
            df['month'] = df[date_col].dt.to_period('M')
            latest_month = df['month'].max()
            
            # Filter to latest month and sum by account
            latest_gmv = df[df['month'] == latest_month]\
                .groupby('account')[amount_col].sum()
                
            return latest_gmv.rename(f"{data_type}_gmv")
            
        except Exception as e:
            logger.error(f"Error getting latest {data_type} GMV: {str(e)}")
            return None
    
    def calculate_delinquency_flags(self) -> pd.DataFrame:
        """
        Calculate delinquency flags with granular DPD buckets and trend analysis.
        
        Returns:
            DataFrame with delinquency flags, DPD trends, and risk indicators
            
        Features:
        - Granular DPD buckets (15, 30, 45, 60, 90+ days)
        - DPD trend (3-month moving average)
        - Weighted average DPD
        - Risk indicators based on DPD patterns
        """
        logger.info("Calculating delinquency flags and DPD trends...")
        dpd_data = self.data.get('DPD.xlsx')
        
        if dpd_data is None:
            logger.error("DPD.xlsx not found in input data")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_columns = ['Dpd', 'Bzid', 'DueDate']
        missing_cols = [col for col in required_columns if col not in dpd_data.columns]
        if missing_cols:
            logger.error(f"Required columns not found in DPD data: {missing_cols}")
            return pd.DataFrame()
        
        try:
            # Convert dates if needed
            dpd_data['DueDate'] = pd.to_datetime(dpd_data['DueDate'], errors='coerce')
            dpd_data = dpd_data.dropna(subset=['DueDate', 'Dpd'])
            
            # Get unique agents
            agent_ids = dpd_data['Bzid'].unique()
            flags = pd.DataFrame(index=agent_ids)
            
            # 1. Granular DPD Buckets
            dpd_buckets = [
                (15, 'dpd_15p'),
                (30, 'dpd_30p'),
                (45, 'dpd_45p'),
                (60, 'dpd_60p'),
                (90, 'dpd_90p')
            ]
            
            for days, col_name in dpd_buckets:
                flags[col_name] = dpd_data[dpd_data['Dpd'] >= days].groupby('Bzid').size() > 0
            
            # 2. Weighted Average DPD (recent DPDs weighted more heavily)
            def calculate_weighted_avg_dpd(group):
                if len(group) == 0:
                    return 0
                weights = np.linspace(1, 0.1, len(group))  # Linear decay
                return np.average(group['Dpd'], weights=weights)
                
            weighted_avg_dpd = dpd_data.groupby('Bzid').apply(calculate_weighted_avg_dpd)
            flags['weighted_avg_dpd'] = weighted_avg_dpd
            
            # 3. DPD Trend (3-month moving average)
            dpd_data['month'] = dpd_data['DueDate'].dt.to_period('M')
            monthly_dpd = dpd_data.groupby(['Bzid', 'month'])['Dpd'].mean().unstack()
            
            # Calculate 3-month moving average
            dpd_trend = monthly_dpd.rolling(window=3, axis=1).mean()
            
            # Get trend direction (slope of last 3 months)
            def get_trend(series):
                if len(series.dropna()) < 2:
                    return 0
                x = np.arange(len(series))
                return np.polyfit(x, series, 1)[0]
                
            flags['dpd_trend'] = dpd_trend.apply(get_trend, axis=1)
            
            # 4. Risk Indicators
            flags['high_risk'] = (
                flags['dpd_90p'] |  # 90+ DPD
                ((flags['dpd_60p']) & (flags['dpd_trend'] > 0)) |  # 60+ DPD with increasing trend
                (flags['weighted_avg_dpd'] > 45)  # High weighted average DPD
            )
            
            flags['watchlist'] = (
                (flags['dpd_30p'] | flags['dpd_45p']) &  # 30-59 DPD
                (flags['dpd_trend'] > 0)  # With increasing trend
            )
            
            return flags.fillna(False)
            
        except Exception as e:
            logger.error(f"Error calculating delinquency flags: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def calculate_recovery_index(self) -> pd.Series:
        """
        Calculate comprehensive recovery metrics for agents with default history.
        
        Returns:
            Series with recovery index (0-100), where higher is better.
            The index combines multiple recovery metrics into a single score.
            
        Recovery Metrics:
        - Days to first payment after default
        - Recovery rate (amount recovered / default amount)
        - Recovery speed (time to 50% and 90% recovery)
        - Default recurrence rate
        """
        logger.info("Calculating comprehensive recovery metrics...")
        
        # Get required data
        dpd_data = self.data.get('DPD.xlsx')
        repayment_report = self.data.get('repayment report.csv')
        credit_sales = self.data.get('credit_sales_data.xlsx')
        
        if dpd_data is None:
            logger.error("DPD.xlsx not found in input data")
            return pd.Series()
            
        # Initialize with default values (worst case)
        agent_ids = dpd_data['Bzid'].unique()
        recovery_metrics = pd.DataFrame(index=agent_ids)
        recovery_metrics['recovery_index'] = 0  # 0-100 scale
        recovery_metrics['days_to_first_payment'] = np.nan
        recovery_metrics['recovery_rate'] = 0
        recovery_metrics['days_to_90p_recovery'] = np.nan
        recovery_metrics['default_recurrence'] = 0
        
        try:
            # 1. Process DPD data to identify default events (90+ DPD)
            if not dpd_data.empty and 'Dpd' in dpd_data.columns:
                # Convert date columns if they exist
                date_cols = [col for col in dpd_data.columns if 'date' in col.lower()]
                for col in date_cols:
                    dpd_data[col] = pd.to_datetime(dpd_data[col], errors='coerce')
                
                # Find default events (DPD >= 90 days)
                defaults = dpd_data[dpd_data['Dpd'] >= 90].copy()
                
                if not defaults.empty and date_cols:
                    default_date_col = date_cols[0]
                    defaults = defaults.sort_values(by=default_date_col)
                    
                    # Process each agent's default history
                    for agent_id, agent_defaults in defaults.groupby('Bzid'):
                        # Get default dates and amounts
                        default_history = agent_defaults[[default_date_col, 'amount']].dropna()
                        if default_history.empty:
                            continue
                            
                        first_default_date = default_history[default_date_col].min()
                        default_amount = default_history['amount'].sum()
                        
                        # 2. Process repayment report for this agent
                        if repayment_report is not None and not repayment_report.empty:
                            # Find payment date and amount columns
                            pay_date_col = next((col for col in repayment_report.columns 
                                              if 'date' in col.lower()), None)
                            amount_col = next((col for col in repayment_report.columns 
                                            if 'amount' in col.lower()), None)
                            
                            if pay_date_col and amount_col:
                                # Process payments after first default
                                repayments = repayment_report[
                                    (repayment_report['agent_id'] == agent_id) &
                                    (repayment_report[pay_date_col] > first_default_date)
                                ].copy()
                                
                                if not repayments.empty:
                                    repayments[pay_date_col] = pd.to_datetime(
                                        repayments[pay_date_col], errors='coerce'
                                    )
                                    repayments[amount_col] = pd.to_numeric(
                                        repayments[amount_col], errors='coerce'
                                    )
                                    repayments = repayments.dropna(subset=[pay_date_col, amount_col])
                                    
                                    if not repayments.empty:
                                        # Sort by date and calculate cumulative payments
                                        repayments = repayments.sort_values(pay_date_col)
                                        repayments['cumulative_paid'] = repayments[amount_col].cumsum()
                                        
                                        # Calculate recovery metrics
                                        first_payment_row = repayments.iloc[0]
                                        recovery_metrics.at[agent_id, 'days_to_first_payment'] = (
                                            first_payment_row[pay_date_col] - first_default_date
                                        ).days
                                        
                                        total_paid = repayments[amount_col].sum()
                                        recovery_metrics.at[agent_id, 'recovery_rate'] = min(
                                            1.0, total_paid / default_amount
                                        ) if default_amount > 0 else 0
                                        
                                        # Calculate days to 90% recovery
                                        recovery_90p = default_amount * 0.9
                                        recovery_90p_days = None
                                        for _, row in repayments.iterrows():
                                            if row['cumulative_paid'] >= recovery_90p:
                                                recovery_90p_days = (row[pay_date_col] - first_default_date).days
                                                break
                                        recovery_metrics.at[agent_id, 'days_to_90p_recovery'] = recovery_90p_days
                                        
                                        # Check for recurring defaults
                                        if len(agent_defaults) > 1:
                                            recovery_metrics.at[agent_id, 'default_recurrence'] = len(agent_defaults) - 1
            
            # 3. Calculate composite recovery index (0-100)
            def calculate_composite_score(row):
                if pd.isna(row['days_to_first_payment']):
                    return 0  # No recovery data
                    
                score = 0
                
                # Days to first payment (max 30 days, lower is better)
                days_score = max(0, 1 - (row['days_to_first_payment'] / 30)) * 30
                
                # Recovery rate (0-100% of default amount)
                recovery_score = row['recovery_rate'] * 40
                
                # Days to 90% recovery (max 90 days, lower is better)
                days_90p = row['days_to_90p_recovery']
                if pd.notna(days_90p):
                    recovery_speed_score = max(0, 1 - (days_90p / 90)) * 20
                else:
                    recovery_speed_score = 0
                
                # Default recurrence penalty
                recurrence_penalty = min(10, row['default_recurrence'] * 2)
                
                score = days_score + recovery_score + recovery_speed_score - recurrence_penalty
                return max(0, min(100, score))
            
            recovery_metrics['recovery_index'] = recovery_metrics.apply(calculate_composite_score, axis=1)
            
            logger.info(f"Calculated recovery metrics for {len(recovery_metrics)} agents")
            return recovery_metrics['recovery_index']
            
        except Exception as e:
            logger.error(f"Error calculating recovery metrics: {str(e)}", exc_info=True)
            return pd.Series(index=agent_ids, name='recovery_index')
    
    def calculate_region_risk(self) -> pd.Series:
        """
        Calculate region risk index based on multiple factors including default density,
        average DPD, and recovery rates.
        
        Returns:
            Series with region risk scores (0-1), where:
            - 0: Lowest risk
            - 1: Highest risk
            - NaN: Insufficient data
            
        Risk Factors Considered:
        1. Default rate (primary weight)
        2. Average DPD in the region
        3. Recovery rate (how quickly defaults are resolved)
        4. Concentration of high-risk agents
        """
        logger.info("Calculating region risk...")
        agents = self.data.get('credit_Agents.xlsx')
        dpd_data = self.data.get('DPD.xlsx')
        repayment_report = self.data.get('repayment report.csv')
        
        if agents is None or dpd_data is None or 'Region' not in agents.columns:
            logger.error("Required data or columns not found for region risk calculation")
            return pd.Series()
            
        try:
            # 1. Prepare base data
            agent_region = agents[['Bzid', 'Region']].drop_duplicates()
            
            # 2. Calculate default rate by region
            region_defaults = dpd_data[dpd_data['Dpd'] >= 90].merge(
                agent_region, on='Bzid', how='left'
            )
            
            # 3. Calculate region statistics
            region_stats = agent_region.groupby('Region').size().reset_index(name='total_agents')
            
            # Default rate
            default_counts = region_defaults.groupby('Region').size().reset_index(name='defaults')
            region_stats = region_stats.merge(default_counts, on='Region', how='left')
            region_stats['default_rate'] = region_stats['defaults'] / region_stats['total_agents']
            
            # Average DPD by region
            region_dpd = dpd_data.merge(agent_region, on='Bzid').groupby('Region')['Dpd'].mean().reset_index()
            region_stats = region_stats.merge(region_dpd, on='Region', how='left')
            
            # 4. Calculate recovery metrics if repayment data is available
            if repayment_report is not None and not repayment_report.empty:
                # Find payment date and amount columns
                date_col = next((col for col in repayment_report.columns 
                              if 'date' in col.lower()), None)
                amount_col = next((col for col in repayment_report.columns 
                                if 'amount' in col.lower()), None)
                
                if date_col and amount_col:
                    # Process repayment data
                    repayments = repayment_report.merge(agent_region, left_on='agent_id', right_on='Bzid')
                    repayments[date_col] = pd.to_datetime(repayments[date_col], errors='coerce')
                    
                    # Calculate recovery rate (amount recovered / amount due)
                    recovery_stats = repayments.groupby('Region')[amount_col].agg(
                        total_recovered='sum',
                        avg_recovery='mean',
                        recovery_count='count'
                    ).reset_index()
                    
                    region_stats = region_stats.merge(recovery_stats, on='Region', how='left')
            
            # 5. Normalize metrics (0-1 scale)
            metrics = ['default_rate', 'Dpd']
            if 'total_recovered' in region_stats.columns:
                metrics.extend(['avg_recovery', 'recovery_count'])
            
            for metric in metrics:
                if metric in region_stats.columns and region_stats[metric].nunique() > 1:
                    region_stats[f'{metric}_norm'] = (
                        (region_stats[metric] - region_stats[metric].min()) / 
                        (region_stats[metric].max() - region_stats[metric].min())
                    )
            
            # 6. Calculate composite risk score (weighted average of normalized metrics)
            weights = {
                'default_rate_norm': 0.5,
                'Dpd_norm': 0.3,
                'avg_recovery_norm': -0.2,  # Negative weight - higher recovery lowers risk
                'recovery_count_norm': -0.1  # Negative weight - more recoveries lower risk
            }
            
            region_stats['region_risk'] = 0
            for metric, weight in weights.items():
                if metric in region_stats.columns:
                    region_stats['region_risk'] += region_stats[metric] * weight
            
            # Scale to 0-1 range
            region_stats['region_risk'] = (
                (region_stats['region_risk'] - region_stats['region_risk'].min()) /
                (region_stats['region_risk'].max() - region_stats['region_risk'].min() + 1e-9)  # Avoid division by zero
            ).clip(0, 1)
            
            # 7. Map risk scores back to agents
            risk_map = dict(zip(region_stats['Region'], region_stats['region_risk']))
            agent_risk = agent_region['Region'].map(risk_map)
            agent_risk.index = agent_region['Bzid']
            
            logger.info(f"Calculated region risk for {len(region_stats)} regions")
            return agent_risk
            
        except Exception as e:
            logger.error(f"Error calculating region risk: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.Series()
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Calculate all features and combine them into a single DataFrame.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Engineering all features...")
        
        # Initialize features DataFrame with agent IDs
        agents = self.data.get('credit_Agents.xlsx')
        if agents is None:
            logger.error("credit_Agents.xlsx not found in input data")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=agents['Bzid'].unique())
        
        # Calculate and add each feature
        feature_methods = [
            ('credit_utilization', self.calculate_credit_utilization),
            ('repayment_score', self.calculate_repayment_score),
            ('credit_gmv_share', self.calculate_credit_gmv_share),
            ('gmv_trend_6m', lambda: self.calculate_gmv_trend(6)),
            ('recovery_index', self.calculate_recovery_index)
        ]
        
        for name, method in feature_methods:
            try:
                feature = method()
                if not feature.empty:
                    features[name] = feature
                    logger.info(f"Added feature: {name}")
                else:
                    logger.warning(f"Could not calculate feature: {name}")
            except Exception as e:
                logger.error(f"Error calculating {name}: {str(e)}")
        
        # Add delinquency flags
        try:
            delinquency_flags = self.calculate_delinquency_flags()
            if not delinquency_flags.empty:
                features = features.join(delinquency_flags, how='left')
                logger.info("Added delinquency flags")
            else:
                # Add empty columns if no flags were generated
                for col in ['delinquent_30p', 'delinquent_60p', 'delinquent_90p']:
                    features[col] = False
                logger.warning("Using default values for delinquency flags")
        except Exception as e:
            logger.error(f"Error adding delinquency flags: {str(e)}")
            # Add default values on error
            for col in ['delinquent_30p', 'delinquent_60p', 'delinquent_90p']:
                features[col] = False
        
        # Add region risk
        try:
            region_risk = self.calculate_region_risk()
            if not region_risk.empty:
                # Map region risk to each agent
                agent_regions = agents.set_index('Bzid')['Region']
                features['region_risk'] = agent_regions.map(region_risk).fillna(0)
                logger.info("Added region risk")
        except Exception as e:
            logger.error(f"Error adding region risk: {str(e)}")
        
        # Fill any remaining missing values
        features = features.fillna({
            'credit_utilization': 0,
            'repayment_score': 100,  # Assume good score if no data
            'credit_gmv_share': 0,
            'gmv_trend_6m': 0,
            'recovery_index': 0,
            'delinquent_30p': False,
            'delinquent_60p': False,
            'delinquent_90p': False,
            'region_risk': 0
        })
        
        # Ensure boolean columns are properly typed
        for col in ['delinquent_30p', 'delinquent_60p', 'delinquent_90p']:
            if col in features.columns:
                features[col] = features[col].astype(bool)
        
        logger.info(f"Feature engineering complete. Created {len(features.columns)} features.")
        return features
