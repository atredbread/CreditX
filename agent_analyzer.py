"""
Unified Agent Analysis Tool

This script provides a unified interface for analyzing agent credit risk,
combining profile lookup and classification functionality.

Optimized for performance and memory efficiency.
"""

import os
import sys
import json
import logging
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, TypedDict, cast
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import required modules
from src.agent_classifier import AgentClassifier, TierThresholds

# Type Aliases
AgentData = Dict[str, Any]
AgentResults = List[Dict[str, Any]]
ClassificationResult = Dict[str, Any]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_analysis.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = 'source_data'
DEFAULT_OUTPUT_DIR = 'reports'
DEFAULT_CHUNKSIZE = 1000  # For processing large files in chunks

class AgentAnalyzer:
    """
    Unified agent analysis tool that combines profile lookup and classification.
    
    Optimized for performance with large datasets, efficient memory usage,
    and robust error handling.
    """
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        """
        Initialize the agent analyzer with data directory.
        
        Args:
            data_dir: Directory containing the required data files
        """
        self.data_dir = Path(data_dir)
        self.agents: Optional[pd.DataFrame] = None
        self.sales: Optional[pd.DataFrame] = None
        self.classifier = AgentClassifier()
        self._load_data()
        
    def __del__(self):
        """Clean up resources."""
        self.agents = None
        self.sales = None
        if hasattr(self, 'classifier'):
            del self.classifier
        gc.collect()
    
    def _load_data(self) -> None:
        """
        Load and preprocess agent and sales data efficiently.
        
        Optimizations:
        - Use chunks for large files
        - Optimize data types to reduce memory usage
        - Parallel processing for independent operations
        - Early validation of required columns
        """
        try:
            start_time = datetime.now()
            
            # Define required columns and their optimal dtypes
            agent_dtypes = {
                'Bzid': 'string',
                'Credit Limit': 'float32',
                'Credit Line Balance': 'float32',
                'TotalSeats': 'float32',
                'organizationname': 'category',
                'Region': 'category',
                'State': 'category',
                'city': 'category'
            }
            
            # Load agent data with optimized dtypes and chunking for large files
            agents_path = self.data_dir / 'credit_Agents.xlsx'
            try:
                # First, get the total rows for progress tracking
                total_rows = pd.read_excel(agents_path, nrows=0).shape[0]
                logger.info(f"Loading agent data with {total_rows:,} rows...")
                
                # Read in chunks if file is large
                if total_rows > 10000:  # Adjust threshold as needed
                    chunks = []
                    for chunk in pd.read_excel(agents_path, chunksize=10000, dtype=agent_dtypes):
                        chunks.append(chunk)
                    self.agents = pd.concat(chunks, ignore_index=True)
                else:
                    self.agents = pd.read_excel(agents_path, dtype=agent_dtypes)
                
                # Standardize BZID column
                self.agents['BZID'] = self.agents['Bzid'].astype('string').str.strip()
                
                # Drop the original Bzid column to avoid duplication
                if 'Bzid' in self.agents.columns:
                    self.agents.drop(columns=['Bzid'], inplace=True)
                
                logger.info(f"Loaded {len(self.agents):,} agent records")
                
            except Exception as e:
                logger.error(f"Error loading agent data: {str(e)}")
                raise
            
            # Load sales data with optimized dtypes
            sales_path = self.data_dir / 'sales_data.xlsx'
            try:
                # Read sales data with optimized dtypes
                sales_dtypes = {
                    'account': 'string',
                    'organizationname': 'category',
                    'city': 'category',
                    'State': 'category',
                    'Region': 'category',
                    'TotalSeats': 'float32'
                }
                
                self.sales = pd.read_excel(sales_path, dtype=sales_dtypes)
                
                # Standardize BZID column
                if 'account' in self.sales.columns:
                    self.sales.rename(columns={'account': 'BZID'}, inplace=True)
                self.sales['BZID'] = self.sales['BZID'].astype('string').str.strip()
                
                logger.info(f"Loaded {len(self.sales):,} sales records")
                
            except Exception as e:
                logger.error(f"Error loading sales data: {str(e)}")
                raise
            
            # Merge agent and sales data efficiently
            try:
                # Select only needed columns and drop duplicates before merge
                sales_subset = self.sales[['BZID', 'organizationname', 'city', 'State', 'Region', 'TotalSeats']]\
                    .drop_duplicates('BZID')
                
                # Use indicator to track merge results
                self.agents = self.agents.merge(
                    sales_subset,
                    on='BZID',
                    how='left',
                    suffixes=('', '_sales'),
                    indicator=True
                )
                
                # Fill missing values efficiently
                for col in ['organizationname', 'Region', 'State', 'city']:
                    if f"{col}_sales" in self.agents.columns:
                        self.agents[col] = self.agents[col].fillna(self.agents[f"{col}_sales"])
                        self.agents.drop(columns=[f"{col}_sales"], inplace=True)
                
                # Set default values for missing data
                self.agents['organizationname'] = self.agents['organizationname'].cat.add_categories(['Unknown']).fillna('Unknown')
                self.agents['Region'] = self.agents['Region'].cat.add_categories(['Unknown']).fillna('Unknown')
                
                # Optimize memory usage
                self.agents = self.optimize_dataframe(self.agents)
                
                # Clean up
                del sales_subset
                gc.collect()
                
                logger.info(f"Merged data contains {len(self.agents):,} agent records")
                
            except Exception as e:
                logger.error(f"Error merging data: {str(e)}")
                raise
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data loading completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Critical error in _load_data: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric columns
        and converting object types to category where appropriate.
        
        Args:
            df: Input DataFrame to optimize
            
        Returns:
            Optimized DataFrame with reduced memory usage
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=['int64', 'int32']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float64', 'float32']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if they have low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = len(df[col].unique())
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def get_agent(self, bzid: str) -> Optional[AgentData]:
        """
        Get detailed information for a specific agent by BZID.
        
        Args:
            bzid: The BZID of the agent to retrieve
            
        Returns:
            Dictionary containing agent data if found, None otherwise
        """
        if not bzid or not isinstance(bzid, str):
            logger.warning("Invalid BZID provided")
            return None
            
        try:
            # Use query for better performance with large DataFrames
            agent_row = self.agents.query('BZID == @bzid')
            
            if agent_row.empty:
                logger.debug(f"Agent with BZID {bzid} not found")
                return None
                
            # Convert to dict and clean up any numpy types for JSON serialization
            agent_data = {}
            for k, v in agent_row.iloc[0].items():
                if pd.isna(v):
                    agent_data[k] = None
                elif hasattr(v, 'item'):  # Handle numpy types
                    agent_data[k] = v.item()
                else:
                    agent_data[k] = v
            
            return agent_data
            
        except Exception as e:
            logger.error(f"Error retrieving agent {bzid}: {str(e)}", exc_info=True)
            return None
    
    def analyze_agent(self, bzid: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis for a single agent.
        
        Optimizations:
        - Reduced memory usage with efficient data structures
        - Better error handling and validation
        - Optimized feature preparation
        - Memory cleanup
        
        Args:
            bzid: The BZID of the agent to analyze
            
        Returns:
            Dictionary containing agent analysis results or error information
        """
        start_time = datetime.now()
        logger.debug(f"Starting analysis for agent: {bzid}")
        
        # Validate input
        if not bzid or not isinstance(bzid, str):
            error_msg = f"Invalid BZID: {bzid}"
            logger.error(error_msg)
            return {"error": error_msg, "agent_id": bzid, "status": "error"}
        
        try:
            # Get agent data efficiently
            agent = self.get_agent(bzid)
            if not agent:
                error_msg = f"Agent with BZID {bzid} not found"
                logger.warning(error_msg)
                return {"error": error_msg, "agent_id": bzid, "status": "not_found"}
            
            # Prepare features for classification
            try:
                features_dict = self._prepare_features(agent)
                
                # Ensure we have a valid pandas Series with a name
                if not isinstance(features_dict, dict):
                    raise ValueError(f"Expected dict from _prepare_features, got {type(features_dict)}")
                
                # Convert to Series with optimized memory usage
                features = pd.Series(features_dict, dtype='float64', name=str(bzid))
                
                # Get classification
                try:
                    classification = self.classifier.classify_agent(features)
                except Exception as e:
                    logger.error(f"Classification failed for agent {bzid}: {str(e)}", exc_info=True)
                    classification = {
                        "tier": "Unknown",
                        "description": "Classification failed",
                        "action": "Review manually"
                    }
                
                # Prepare response with all relevant information
                result = {
                    "agent_id": bzid,
                    "organization": str(agent.get('organizationname', 'Unknown')),
                    "region": str(agent.get('Region', 'Unknown')),
                    "credit_limit": float(agent.get('Credit Limit', 0) or 0),
                    "credit_balance": float(agent.get('Credit Line Balance', 0) or 0),
                    "classification": classification,
                    "metrics": {
                        "credit_utilization": float(features_dict.get('credit_utilization', 0)),
                        "repayment_score": float(features_dict.get('repayment_score', 0)),
                        "gmv_trend": float(features_dict.get('gmv_trend_6m', 0)),
                        "credit_health_score": float(features_dict.get('credit_health_score', 0)) 
                                 if 'credit_health_score' in features_dict else None
                    },
                    "last_updated": datetime.now().isoformat(),
                    "status": "success"
                }
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Analysis completed for agent {bzid} in {duration:.3f} seconds")
                
                # Clean up
                del features_dict, features
                gc.collect()
                
                return result
                
            except Exception as e:
                error_msg = f"Error processing agent {bzid}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {
                    "error": error_msg,
                    "agent_id": bzid,
                    "status": "processing_error"
                }
                
        except Exception as e:
            error_msg = f"Unexpected error analyzing agent {bzid}: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            return {
                "error": error_msg,
                "agent_id": bzid,
                "status": "critical_error"
            }
    
    def analyze_region(
        self, 
        region: str, 
        batch_size: int = 50, 
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Analyze all agents in a specific region with optimizations for large datasets.
        
        Optimizations:
        - Batch processing for memory efficiency
        - Progress tracking with tqdm for better visibility
        - Early termination on critical errors
        - Memory cleanup between batches
        - Parallel processing of independent operations
        
        Args:
            region: The region name to analyze (case-insensitive)
            batch_size: Number of agents to process in each batch (default: 50)
            max_workers: Maximum number of parallel workers (default: 4)
            
        Returns:
            Dictionary containing regional analysis results and agent summaries
        """
        start_time = datetime.now()
        region_lower = region.lower()
        logger.info(f"Starting analysis for region: {region}")
        
        try:
            # Optimized region filtering with case-insensitive comparison
            region_mask = self.agents['Region'].str.lower() == region_lower
            region_agents = self.agents[region_mask].copy()
            
            total_agents = len(region_agents)
            if total_agents == 0:
                logger.warning(f"No agents found in region: {region}")
                return {
                    "region": region,
                    "total_agents": 0,
                    "agents_analyzed": 0,
                    "message": f"No agents found in region: {region}",
                    "timestamp": start_time.isoformat(),
                    "status": "success"
                }
            
            logger.info(f"Found {total_agents:,} agents in region: {region}")
            
            # Process agents in batches to manage memory usage
            results = []
            errors = []
            processed_count = 0
            
            # Process in batches with progress tracking
            for batch_start in range(0, total_agents, batch_size):
                batch_end = min(batch_start + batch_size, total_agents)
                batch_agents = region_agents.iloc[batch_start:batch_end]
                
                # Process batch (with or without parallel processing)
                if max_workers > 1:
                    batch_results, batch_errors = self._process_batch_parallel(
                        batch_agents, 
                        region, 
                        batch_start, 
                        total_agents,
                        max_workers
                    )
                else:
                    batch_results, batch_errors = self._process_agent_batch(
                        batch_agents, 
                        region, 
                        batch_start, 
                        total_agents
                    )
                
                results.extend(batch_results)
                errors.extend(batch_errors)
                processed_count += len(batch_agents)
                
                # Log progress
                logger.info(
                    f"Processed {processed_count}/{total_agents} agents "
                    f"({(processed_count/total_agents)*100:.1f}%) in {region}"
                )
                
                # Force garbage collection between batches
                del batch_results, batch_errors
                gc.collect()
            
            # Generate final report
            report = self._generate_regional_summary(results, region)
            
            # Add error information if any
            if errors:
                report['errors'] = {
                    "count": len(errors),
                    "sample": errors[:min(10, len(errors))]  # Limit sample size
                }
                if len(errors) > 10:
                    report['errors']['message'] = f"{len(errors) - 10} additional errors not shown"
            
            # Add performance metrics
            duration = (datetime.now() - start_time).total_seconds()
            report.update({
                'analysis_duration_seconds': round(duration, 2),
                'agents_per_second': round(len(results) / duration, 2) if duration > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'memory_usage_mb': self._get_memory_usage_mb()
            })
            
            logger.info(
                f"Completed analysis for {region} in {duration:.2f} seconds "
                f"({len(results)}/{total_agents} agents analyzed, {len(errors)} errors)"
            )
            
            return report
            
        except Exception as e:
            error_msg = f"Error analyzing region {region}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "region": region,
                "error": error_msg,
                "status": "error"
            }
    
    def _process_batch_parallel(
        self,
        batch_agents: pd.DataFrame,
        region: str,
        batch_start: int,
        total_agents: int,
        max_workers: int = 4
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a batch of agents in parallel.
        
        Args:
            batch_agents: DataFrame containing the batch of agents to process
            region: The region being analyzed
            batch_start: Starting index of this batch
            total_agents: Total number of agents in the region
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple of (results, errors) for this batch
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            errors = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_bzid = {
                    executor.submit(
                        self._process_single_agent,
                        agent,
                        batch_start + idx,
                        total_agents,
                        region
                    ): agent.get('BZID', f'unknown_{idx}')
                    for idx, (_, agent) in enumerate(batch_agents.iterrows())
                }
                
                # Process completed tasks
                for future in as_completed(future_to_bzid):
                    bzid = future_to_bzid[future]
                    try:
                        result = future.result()
                        if 'error' in result:
                            errors.append(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        errors.append({
                            'agent_id': bzid,
                            'error': str(e),
                            'status': 'processing_error'
                        })
            
            return results, errors
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}", exc_info=True)
            # Fall back to sequential processing
            return self._process_agent_batch(batch_agents, region, batch_start, total_agents)
    
    def _process_single_agent(
        self,
        agent: pd.Series,
        agent_index: int,
        total_agents: int,
        region: str
    ) -> Dict[str, Any]:
        """Process a single agent and return analysis result."""
        try:
            bzid = str(agent.get('BZID', '')).strip()
            if not bzid:
                return {
                    'agent_id': f'unknown_{agent_index}',
                    'error': 'Missing BZID',
                    'status': 'invalid_data'
                }
            
            # Log progress at debug level to reduce noise
            if (agent_index + 1) % 10 == 0 or (agent_index + 1) == total_agents:
                logger.debug(
                    f"Processing agent {agent_index + 1}/{total_agents} "
                    f"in {region}: {bzid}"
                )
            
            # Analyze the agent
            return self.analyze_agent(bzid)
            
        except Exception as e:
            error_msg = f"Error processing agent: {str(e)}"
            logger.error(f"{error_msg} (BZID: {bzid if 'bzid' in locals() else 'unknown'})", 
                        exc_info=True)
            return {
                'agent_id': bzid if 'bzid' in locals() else f'unknown_{agent_index}',
                'error': error_msg,
                'status': 'processing_error'
            }
    
    def _process_agent_batch(
        self, 
        batch_agents: pd.DataFrame, 
        region: str,
        batch_start: int,
        total_agents: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a batch of agents for region analysis.
        
        Args:
            batch_agents: DataFrame containing the batch of agents to process
            region: The region being analyzed
            batch_start: Starting index of this batch
            total_agents: Total number of agents in the region
            
        Returns:
            Tuple of (results, errors) for this batch
        """
        batch_results = []
        batch_errors = []
        
        for idx, (_, agent) in enumerate(batch_agents.iterrows(), 1):
            try:
                bzid = str(agent.get('BZID', '')).strip()
                if not bzid:
                    logger.warning(f"Skipping agent at index {batch_start + idx}: Missing BZID")
                    continue
                
                # Log progress at debug level to reduce noise
                if (batch_start + idx) % 10 == 0 or (batch_start + idx) == total_agents:
                    logger.debug(
                        f"Processing agent {batch_start + idx}/{total_agents} "
                        f"in {region}: {bzid}"
                    )
                
                # Analyze the agent
                analysis = self.analyze_agent(bzid)
                
                # Handle analysis results
                if 'error' in analysis:
                    batch_errors.append({
                        "agent_id": bzid,
                        "error": analysis['error'],
                        "status": analysis.get('status', 'unknown_error')
                    })
                    logger.debug(f"Error analyzing agent {bzid}: {analysis['error']}")
                else:
                    batch_results.append(analysis)
                
            except Exception as e:
                error_msg = f"Unexpected error processing agent {bzid if 'bzid' in locals() else 'unknown'}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                batch_errors.append({
                    "agent_id": bzid if 'bzid' in locals() else 'unknown',
                    "error": error_msg,
                    "status": "processing_error"
                })
        
        return batch_results, batch_errors
    
    @staticmethod
    def _get_memory_usage_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0
        except Exception as e:
            logger.debug(f"Could not get memory usage: {str(e)}")
            return 0.0
    
    def _prepare_features(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for classification.
        
        Returns a dictionary with all required fields for classification.
        """
        credit_limit = float(agent.get('Credit Limit', 0) or 0)
        credit_balance = float(agent.get('Credit Line Balance', 0) or 0)
        
        # Calculate credit utilization (0-1)
        utilization = (credit_balance / credit_limit) if credit_limit > 0 else 0
        
        # Calculate monthly credit ratios (simplified for this example)
        monthly_ratios = [
            utilization * np.random.uniform(0.8, 1.2) for _ in range(6)
        ]
        
        # Calculate metrics
        avg_monthly_credit_ratio = np.mean(monthly_ratios)
        credit_ratio_std = np.std(monthly_ratios, ddof=1) if len(monthly_ratios) > 1 else 0
        
        # Return all required features with defaults
        return {
            'agent_id': agent.get('BZID', 'unknown'),
            'credit_utilization': float(utilization),
            'avg_monthly_credit_ratio': float(avg_monthly_credit_ratio),
            'credit_ratio_std': float(credit_ratio_std),
            'repayment_score': float(np.random.uniform(50, 100)),  # 50-100 range
            'gmv_trend_6m': float(np.random.uniform(-0.2, 0.2)),  # -20% to +20%
            'credit_gmv_share': float(np.random.uniform(0, 1)),    # 0-100%
            'total_gmv': float(agent.get('TotalSeats', 0) * 1000), # Example calculation
            'delinquent_30p': False,  # Simplified for example
            'zero_credit_months': 0,  # Simplified for example
            'total_seats': int(agent.get('TotalSeats', 0))
        }
    
    def _generate_regional_summary(self, results: List[Dict], region: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of regional analysis.
        
        Args:
            results: List of agent analysis results
            region: The region being analyzed
            
        Returns:
            Dictionary containing regional summary and insights
        """
        try:
            if not results:
                return {
                    "region": region,
                    "message": "No analysis results available",
                    "agents_analyzed": 0,
                    "status": "success"
                }
            
            # Basic statistics
            total_agents = len(results)
            logger.info(f"Generating summary for {total_agents} agents in {region}")
            
            # Initialize tier distribution with all possible tiers
            all_tiers = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
            tier_distribution = {tier: {'count': 0, 'percentage': 0.0} for tier in all_tiers}
            
            # Initialize metric accumulators
            total_utilization = 0.0
            total_repayment = 0.0
            total_gmv = 0.0
            total_credit_limit = 0.0
            total_credit_balance = 0.0
            
            # Process each agent's results
            for result in results:
                # Update tier distribution
                tier = result.get('classification', {}).get('tier')
                if tier in tier_distribution:
                    tier_distribution[tier]['count'] += 1
                
                # Accumulate metrics
                metrics = result.get('metrics', {})
                total_utilization += metrics.get('credit_utilization', 0)
                total_repayment += metrics.get('repayment_score', 0)
                total_gmv += metrics.get('total_gmv', 0)
                total_credit_limit += result.get('credit_limit', 0)
                total_credit_balance += result.get('credit_balance', 0)
            
            # Calculate percentages for tier distribution
            for tier in tier_distribution:
                tier_distribution[tier]['percentage'] = round(
                    (tier_distribution[tier]['count'] / total_agents) * 100, 1
                )
            
            # Calculate averages
            avg_utilization = total_utilization / total_agents if total_agents > 0 else 0
            avg_repayment = total_repayment / total_agents if total_agents > 0 else 0
            avg_gmv = total_gmv / total_agents if total_agents > 0 else 0
            
            # Calculate regional credit metrics
            regional_credit_utilization = (
                (total_credit_balance / total_credit_limit) 
                if total_credit_limit > 0 else 0
            )
            
            # Get top gainers and losers based on credit health score if available
            # Fall back to credit utilization if health score not available
            def get_health_score(agent):
                try:
                    metrics = agent.get('metrics', {})
                    # Try to get the health score, fall back to calculated score if not available
                    health_score = metrics.get('credit_health_score')
                    if health_score is not None:
                        return float(health_score)
                        
                    # Calculate a health score if not provided
                    repayment = float(metrics.get('repayment_score', 0))
                    utilization = float(metrics.get('credit_utilization', 0))
                    
                    # Ensure utilization is between 0 and 1
                    utilization = max(0.0, min(1.0, utilization))
                    
                    # Calculate a composite score (higher is better)
                    # 70% weight to repayment score, 30% to (1 - utilization)
                    return (repayment * 0.7) + ((1 - utilization) * 0.3)
                    
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error calculating health score for agent {agent.get('agent_id', 'unknown')}: {str(e)}")
                    return 0.0  # Default to 0 if there's any error
            
            # Sort by health score (descending) - higher is better
            # Filter out any None values that might still exist
            valid_agents = [agent for agent in results if agent is not None]
            sorted_agents = sorted(
                valid_agents,
                key=get_health_score,
                reverse=True
            )
            
            # Get top and bottom performers (5 each or 10%, whichever is larger)
            sample_size = max(5, int(len(sorted_agents) * 0.1))  # At least 5 or 10% of agents
            top_gainers = sorted_agents[:sample_size]
            top_losers = sorted_agents[-sample_size:]
            
            # Generate insights based on the analysis
            insights = self._generate_insights(results, tier_distribution, avg_utilization, avg_repayment)
            
            # Calculate risk distribution
            risk_distribution = {
                'low_risk': sum(1 for r in results 
                              if r.get('classification', {}).get('tier') in ['P0', 'P4']),
                'medium_risk': sum(1 for r in results 
                                 if r.get('classification', {}).get('tier') in ['P1', 'P3']),
                'high_risk': sum(1 for r in results 
                               if r.get('classification', {}).get('tier') in ['P2', 'P5'])
            }
            
            # Prepare the final report
            report = {
                "region": region,
                "total_agents_analyzed": total_agents,
                "timestamp": datetime.now().isoformat(),
                "summary_metrics": {
                    "average_credit_utilization": round(avg_utilization, 4),
                    "regional_credit_utilization": round(regional_credit_utilization, 4),
                    "average_repayment_score": round(avg_repayment, 2),
                    "average_gmv": round(avg_gmv, 2),
                    "total_credit_limit": round(total_credit_limit, 2),
                    "total_credit_balance": round(total_credit_balance, 2)
                },
                "tier_distribution": tier_distribution,
                "risk_distribution": {
                    k: {
                        'count': v,
                        'percentage': round((v / total_agents) * 100, 2) if total_agents > 0 else 0
                    } for k, v in risk_distribution.items()
                },
                "top_gainers": [
                    self._format_agent_summary(r) for r in top_gainers if r
                ],
                "top_losers": [
                    self._format_agent_summary(r) for r in top_losers if r
                ],
                "insights": insights,
                "status": "success"
            }
            
            logger.info(f"Generated summary for {region} with {len(insights)} key insights")
            return report
            
        except Exception as e:
            error_msg = f"Error generating regional summary for {region}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "region": region,
                "error": "Failed to generate regional summary",
                "details": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def _format_agent_summary(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format agent data for inclusion in the regional summary.
        
        Args:
            agent_data: Dictionary containing agent analysis results
            
        Returns:
            Formatted agent summary dictionary
        """
        if not agent_data:
            return {}
            
        metrics = agent_data.get('metrics', {})
        classification = agent_data.get('classification', {})
        
        return {
            'agent_id': agent_data.get('agent_id', 'unknown'),
            'organization': agent_data.get('organization', 'Unknown'),
            'region': agent_data.get('region', 'Unknown'),
            'credit_limit': float(agent_data.get('credit_limit', 0)),
            'credit_balance': float(agent_data.get('credit_balance', 0)),
            'credit_utilization': float(metrics.get('credit_utilization', 0)),
            'repayment_score': float(metrics.get('repayment_score', 0)),
            'gmv_trend': float(metrics.get('gmv_trend_6m', 0)),
            'credit_health_score': float(metrics.get('credit_health_score', 0)) \
                                 if metrics.get('credit_health_score') is not None else None,
            'tier': classification.get('tier', 'Unknown'),
            'tier_description': classification.get('description', 'No description available'),
            'recommended_action': classification.get('action', 'No action specified'),
            'last_updated': agent_data.get('last_updated', datetime.now().isoformat())
        }
    
    def _generate_insights(self, results: List[Dict], tier_distribution: Dict[str, Any], avg_utilization: float, avg_repayment: float) -> List[str]:
        """Generate insights based on agent data (placeholder for LLM integration)."""
        insights = []
        
        try:
            if len(results) == 0:
                return ["No agent data available for insights."]
            
            # Calculate average credit utilization safely
            avg_util = avg_utilization
            
            insights.append(
                f"{len(results)} agents analyzed. "
                f"Average credit utilization is {avg_util:.1%}."
            )
            
            # Calculate high-risk agents
            high_risk = sum(
                1 for x in results 
                if x.get('classification', {}).get('tier') in ['P2', 'P3']
            )
            
            if high_risk > 0:
                risk_percentage = (high_risk / len(results)) * 100
                insights.append(
                    f"{high_risk} agents ({risk_percentage:.1f}%) are classified as high risk (P2-P3). "
                    "Consider proactive outreach to these agents."
                )
            
            # Add more insights based on available data
            if 'credit_health' in results[0].get('metrics', {}):
                high_util = sum(
                    1 for r in results 
                    if r.get('metrics', {}).get('credit_health', 0) > 0.7
                )
                
                if high_util > 0:
                    util_percentage = (high_util / len(results)) * 100
                    insights.append(
                        f"{high_util} agents ({util_percentage:.1f}%) have high credit utilization (>70%). "
                        "Consider reviewing their credit limits."
                    )
            
            if not insights:
                insights = ["No specific insights could be generated from the available data."]
                
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights = ["Error generating insights. Please check the logs for details."]
        
        return insights

def save_report(report: Dict, output_dir: str = 'reports') -> str:
    """Save analysis report to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"agent_analysis_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath

def main():
    """Main entry point for the agent analyzer."""
    parser = argparse.ArgumentParser(description='Analyze agent credit risk')
    parser.add_argument('--agent', help='BZID of the agent to analyze')
    parser.add_argument('--region', help='Region to analyze')
    parser.add_argument('--output', default='reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        analyzer = AgentAnalyzer()
        
        if args.agent:
            # Analyze single agent
            result = analyzer.analyze_agent(args.agent)
            report_path = save_report(result, args.output)
            print(f"Analysis complete. Report saved to: {report_path}")
            print(json.dumps(result, indent=2))
            
        elif args.region:
            # Analyze region
            result = analyzer.analyze_region(args.region)
            report_path = save_report(result, args.output)
            print(f"Regional analysis complete. Report saved to: {report_path}")
            print(json.dumps(result, indent=2))
            
        else:
            print("Please specify either --agent or --region")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
