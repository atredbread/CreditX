"""
Regional Agent Analyzer for Windsurf Credit Health Intelligence Engine

This module provides functionality to analyze and rank agents by region
based on their credit performance metrics. It generates Top N Best and Worst
agent lists for each region and provides a CLI interface for user interaction.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('regional_analyzer.log')
    ]
)
logger = logging.getLogger('regional_analyzer')

class RegionalAnalyzer:
    """
    Analyzes agent performance by region and generates Top N Best/Worst lists.
    
    This class loads agent data from JSON files in the output directory and
    provides methods to analyze and rank agents by various performance metrics.
    """
    
    def __init__(self, output_dir: Union[str, Path] = 'output'):
        """
        Initialize the RegionalAnalyzer.
        
        Args:
            output_dir: Base directory containing the output JSON files
        """
        self.output_dir = Path(output_dir)
        self.agent_data = {}
        self.regions = set()
        self.metrics = [
            'score', 'gmv', 'dpd', 'utilization',
            'repayment_rate', 'credit_sales_ratio', 'risk_score'
        ]
        self._ensure_output_structure()
        self.load_agent_data()
    
    def _ensure_output_structure(self) -> None:
        """Ensure the output directory structure exists."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'reports').mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating output directory structure: {e}")
            raise
    
    def load_agent_data(self) -> None:
        """
        Load agent data from the processed JSON files.
        
        This method looks for agent summary JSON files in the output directory
        and loads them into memory for analysis.
        """
        try:
            # Look for agent summary files in the output directory
            agent_files = list(self.output_dir.glob('**/agent_summary.json'))
            
            if not agent_files:
                logger.warning(f"No agent summary files found in {self.output_dir}")
                return
            
            # Load data from all found files
            for file_path in agent_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Process each agent in the file
                    for agent_id, agent_info in data.items():
                        # Ensure all metrics have default values
                        for metric in self.metrics:
                            if metric not in agent_info:
                                agent_info[metric] = 0.0
                        
                        self.agent_data[agent_id] = agent_info
                        
                        # Track unique regions
                        if 'region' in agent_info and agent_info['region']:
                            self.regions.add(agent_info['region'])
                        else:
                            # Default region if not specified
                            agent_info['region'] = 'Unassigned'
                            self.regions.add('Unassigned')
                            
                    logger.info(f"Loaded data for {len(data)} agents from {file_path}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error loading agent data from {file_path}: {e}")
            
            if not self.agent_data:
                logger.warning("No valid agent data was loaded")
                return
                
            logger.info(f"Total agents loaded: {len(self.agent_data)}")
            logger.info(f"Regions found: {', '.join(sorted(self.regions)) if self.regions else 'None'}")
            
        except Exception as e:
            logger.error(f"Error in load_agent_data: {e}")
            raise
    
    def rank_agents(
        self, 
        region: str = 'all', 
        metric: str = 'score',
        top_n: int = 10,
        ascending: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Rank agents by the specified performance metric.
        
        Args:
            region: The region to filter by, or 'all' for all regions
            metric: The metric to rank by (e.g., 'score', 'gmv', 'dpd', 'utilization')
            top_n: Number of top and bottom agents to return
            ascending: If True, sort in ascending order (lowest first)
            
        Returns:
            Tuple of (top_agents, bottom_agents) - each a list of agent dicts
        """
        if not self.agent_data:
            logger.warning("No agent data available for ranking")
            return [], []
        
        try:
            # Convert agent data to a list of dicts with agent_id included
            agents = [
                {**agent, 'agent_id': agent_id}
                for agent_id, agent in self.agent_data.items()
            ]
            
            # Filter by region if specified
            if region.lower() != 'all':
                agents = [
                    a for a in agents 
                    if str(a.get('region', '')).lower() == region.lower()
                ]
            
            if not agents:
                logger.warning(f"No agents found for region: {region}")
                return [], []
            
            # Default to score if specified metric is not available
            if metric not in self.metrics:
                logger.warning(f"Metric '{metric}' not found. Defaulting to 'score'.")
                metric = 'score'
            
            # Sort agents by the specified metric
            sorted_agents = sorted(
                agents,
                key=lambda x: float(x.get(metric, 0)),
                reverse=not ascending
            )
            
            # Get top and bottom N agents
            top_count = min(top_n, len(sorted_agents))
            
            if ascending:
                # For ascending sort, first items are the "worst"
                bottom_agents = sorted_agents[:top_count]
                top_agents = sorted_agents[-top_count:][::-1]  # Take last N and reverse
            else:
                # For descending sort, first items are the "best"
                top_agents = sorted_agents[:top_count]
                bottom_agents = sorted_agents[-top_count:][::-1]  # Take last N and reverse
            
            return top_agents, bottom_agents
            
        except Exception as e:
            logger.error(f"Error ranking agents by {metric}: {e}", exc_info=True)
            return [], []
    
    def generate_report(
        self, 
        region: str = 'all',
        metric: str = 'score',
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a report of top and bottom performing agents.
        
        Args:
            region: The region to generate the report for, or 'all' for all regions
            metric: The metric to rank agents by
            top_n: Number of top and bottom agents to include
            
        Returns:
            Dictionary containing the report data
        """
        # Get top and bottom agents
        top_agents, bottom_agents = self.rank_agents(
            region=region,
            metric=metric,
            top_n=top_n,
            ascending=False
        )
        
        # Get regional summary statistics
        region_agents = [
            a for a in self.agent_data.values() 
            if region.lower() == 'all' or str(a.get('region', '')).lower() == region.lower()
        ]
        
        # Calculate statistics
        if region_agents:
            stats = {
                'total_agents': len(region_agents),
                'avg_score': self._safe_mean(region_agents, 'score'),
                'total_gmv': self._safe_sum(region_agents, 'gmv'),
                'avg_utilization': self._safe_mean(region_agents, 'utilization'),
                'avg_dpd': self._safe_mean(region_agents, 'dpd'),
                'metric_used': metric
            }
        else:
            stats = {}
        
        # Prepare report
        report = {
            'region': region,
            'generated_at': datetime.now().isoformat(),
            'metric_used': metric,
            'statistics': stats,
            'top_agents': self._format_agents(top_agents, metric),
            'bottom_agents': self._format_agents(bottom_agents, metric, is_bottom=True)
        }
        
        return report
    
    def _format_agents(
        self, 
        agents: List[Dict], 
        metric: str,
        is_bottom: bool = False
    ) -> List[Dict]:
        """Format agent data for the report."""
        formatted = []
        
        for i, agent in enumerate(agents):
            # Get all metrics, using 0 as default for missing values
            agent_data = {
                'rank': i + 1,
                'agent_id': agent.get('agent_id', 'N/A'),
                'name': agent.get('name', 'Unknown').strip() or 'Unnamed',
                'region': agent.get('region', 'Unassigned'),
                'metric_value': float(agent.get(metric, 0)),
                'metrics': {}
            }
            
            # Add all available metrics
            for m in self.metrics:
                agent_data['metrics'][m] = float(agent.get(m, 0))
            
            formatted.append(agent_data)
        
        return formatted
    
    @staticmethod
    def _safe_mean(items: List[Dict], key: str, default: float = 0.0) -> float:
        """Safely calculate mean of a key across a list of dicts."""
        try:
            values = [float(item.get(key, 0)) for item in items if item.get(key) is not None]
            return sum(values) / len(values) if values else default
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def _safe_sum(items: List[Dict], key: str, default: float = 0.0) -> float:
        """Safely calculate sum of a key across a list of dicts."""
        try:
            return sum(float(item.get(key, 0)) for item in items)
        except (TypeError, ValueError):
            return default
    
    def print_report(
        self, 
        region: str = 'all',
        metric: str = 'score',
        top_n: int = 10
    ) -> None:
        """
        Print a formatted report of top and bottom performing agents.
        
        Args:
            region: The region to generate the report for, or 'all' for all regions
            metric: The metric to rank agents by
            top_n: Number of top and bottom agents to display
        """
        report = self.generate_report(region, metric, top_n)
        
        # Format the header
        header = f"{' ' * 30}REGIONAL AGENT ANALYSIS"
        separator = "=" * 80
        
        print("\n" + separator)
        print(header)
        print(separator)
        print(f"Region: {report['region'].upper() if report['region'] != 'all' else 'ALL REGIONS'}")
        print(f"Generated: {report['generated_at']}")
        print(f"Ranking Metric: {metric.upper()}")
        print("-" * 80)
        
        # Print statistics if available
        if report.get('statistics'):
            stats = report['statistics']
            print("\nREGION STATISTICS:")
            print("-" * 80)
            print(f"Total Agents: {stats.get('total_agents', 0):,}")
            print(f"Average Score: {stats.get('avg_score', 0):.2f}")
            print(f"Total GMV: {stats.get('total_gmv', 0):,.2f}")
            print(f"Avg Utilization: {stats.get('avg_utilization', 0):.2%}")
            print(f"Avg DPD: {stats.get('avg_dpd', 0):.2f}")
        
        # Print top performers
        if report['top_agents']:
            self._print_agent_table(
                title=f"TOP {len(report['top_agents'])} PERFORMING AGENTS",
                agents=report['top_agents'],
                metric=metric
            )
        else:
            print("\nNo top agents to display.")
        
        # Print bottom performers
        if report['bottom_agents']:
            self._print_agent_table(
                title=f"BOTTOM {len(report['bottom_agents'])} PERFORMING AGENTS",
                agents=report['bottom_agents'],
                metric=metric,
                is_bottom=True
            )
        else:
            print("\nNo bottom agents to display.")
        
        print("\n" + separator + "\n")
    
    def _print_agent_table(
        self, 
        title: str, 
        agents: List[Dict], 
        metric: str,
        is_bottom: bool = False
    ) -> None:
        """Print a formatted table of agents."""
        if not agents:
            return
            
        # Define column widths and headers
        columns = [
            ('Rank', 5, '^'),
            ('Agent ID', 12, '<'),
            ('Name', 20, '<'),
            ('Region', 12, '<'),
            (f"{metric.upper()}", 12, '>'),
            ('SCORE', 10, '>'),
            ('GMV', 12, '>'),
            ('DPD', 8, '>')
        ]
        
        # Print table header
        print(f"\n{title}")
        print("-" * 80)
        
        # Print column headers
        header_parts = []
        for name, width, _ in columns:
            header_parts.append(f"{name[:width-1]:<{width}}")
        print(" ".join(header_parts))
        
        print("-" * 80)
        
        # Print agent rows
        for agent in agents:
            row_parts = []
            
            # Format each column
            for name, width, align in columns:
                if name.lower() == 'rank':
                    value = str(agent['rank'])
                elif name.lower() == 'agent id':
                    value = str(agent['agent_id'])
                elif name.lower() == 'name':
                    value = str(agent['name'])[:width-1]
                elif name.lower() == 'region':
                    value = str(agent['region'])[:width-1]
                elif name.lower() == metric.lower():
                    value = agent['metric_value']
                elif name.lower() in agent['metrics']:
                    value = agent['metrics'][name.lower()]
                else:
                    value = 0
                
                # Format the value based on type
                if isinstance(value, (int, float)):
                    if name.lower() in ['gmv']:
                        formatted = f"{value:,.2f}"
                    elif name.lower() == 'utilization':
                        formatted = f"{value:.2%}"
                    elif name.lower() == 'dpd':
                        formatted = f"{value:.1f}"
                    else:
                        formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                
                # Apply alignment and width
                if align == '^':
                    row_parts.append(f"{formatted:^{width}}")
                elif align == '>':
                    row_parts.append(f"{formatted:>{width}}")
                else:
                    row_parts.append(f"{formatted:<{width}}")
            
            print(" ".join(row_parts))


def run_analysis(
    output_dir: str = 'output', 
    region: str = 'all',
    metric: str = 'score',
    top_n: int = 10,
    save_report_flag: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run the regional analysis and print the report.
    
    Args:
        output_dir: Directory containing the output JSON files
        region: The region to analyze, or 'all' for all regions
        metric: The metric to rank agents by
        top_n: Number of top and bottom agents to display
        save_report_flag: If True, save the report to a file
        
    Returns:
        The report data if successful, None otherwise
    """
    try:
        analyzer = RegionalAnalyzer(output_dir=output_dir)
        
        # Generate and print the report
        analyzer.print_report(region=region, metric=metric, top_n=top_n)
        
        # Generate the full report data
        report = analyzer.generate_report(region=region, metric=metric, top_n=top_n)
        
        # Save the report if requested
        if save_report_flag:
            save_report(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Error running regional analysis: {e}", exc_info=True)
        return None


def interactive_mode() -> None:
    """
    Run the regional analyzer in interactive mode with a menu-driven interface.
    """
    print("\n" + "=" * 70)
    print(f"{' ' * 25}WINDSURF CREDIT - REGIONAL AGENT ANALYSIS")
    print("=" * 70)
    
    # Get output directory
    output_dir = input("\nEnter output directory [default: output]: ").strip() or 'output'
    
    try:
        # Initialize the analyzer
        analyzer = RegionalAnalyzer(output_dir=output_dir)
        
        # Default settings
        current_region = 'all'
        current_metric = 'score'
        current_top_n = 10
        
        while True:
            # Clear screen and show menu
            print("\n" + "=" * 70)
            print(f"{' ' * 20}MAIN MENU - {current_region.upper() if current_region != 'all' else 'ALL REGIONS'}")
            print("=" * 70)
            print(f"{'1.':<5} Select Region (Current: {current_region if current_region != 'all' else 'All Regions'})")
            print(f"{'2.':<5} Select Metric (Current: {current_metric.upper()})")
            print(f"{'3.':<5} Set Number of Agents (Current: {current_top_n})")
            print(f"{'4.':<5} View Report")
            print(f"{'5.':<5} Save Report to File")
            print(f"{'q.':<5} Quit")
            print("-" * 70)
            
            choice = input("\nEnter your choice (1-5 or q): ").strip().lower()
            
            if choice == 'q':
                print("\nExiting...")
                break
                
            elif choice == '1':
                # Region selection
                print("\n" + "=" * 70)
                print(f"{' ' * 25}SELECT REGION")
                print("=" * 70)
                print("Available regions:")
                print("  all   - All regions")
                
                # List available regions
                for i, region in enumerate(sorted(analyzer.regions), 1):
                    print(f"  {i:<5} - {region}")
                
                print("\n  b     - Back to main menu")
                print("-" * 70)
                
                region_choice = input("\nSelect a region (number or 'all'): ").strip().lower()
                
                if region_choice == 'b':
                    continue
                    
                # Map number to region if a number was entered
                if region_choice.isdigit():
                    region_list = sorted(analyzer.regions)
                    choice_idx = int(region_choice) - 1
                    if 0 <= choice_idx < len(region_list):
                        current_region = region_list[choice_idx]
                    else:
                        print("\nInvalid selection. Please try again.")
                        continue
                else:
                    current_region = region_choice
                
                # Validate region
                if current_region != 'all' and current_region not in analyzer.regions:
                    print(f"\nInvalid region: {current_region}")
                    current_region = 'all'
                
            elif choice == '2':
                # Metric selection
                print("\n" + "=" * 70)
                print(f"{' ' * 25}SELECT METRIC")
                print("=" * 70)
                print("Available metrics:")
                
                for i, metric in enumerate(analyzer.metrics, 1):
                    print(f"  {i:<2} - {metric.upper()}")
                
                print("\n  b  - Back to main menu")
                print("-" * 70)
                
                metric_choice = input("\nSelect a metric (number): ").strip().lower()
                
                if metric_choice == 'b':
                    continue
                    
                if metric_choice.isdigit():
                    choice_idx = int(metric_choice) - 1
                    if 0 <= choice_idx < len(analyzer.metrics):
                        current_metric = analyzer.metrics[choice_idx]
                    else:
                        print("\nInvalid selection. Please try again.")
                else:
                    print("\nPlease enter a number.")
                    
            elif choice == '3':
                # Set number of agents
                print("\n" + "=" * 70)
                print(f"{' ' * 20}SET NUMBER OF AGENTS TO DISPLAY")
                print("=" * 70)
                
                try:
                    new_top_n = input(f"\nEnter number of top/bottom agents to show [current: {current_top_n}]: ").strip()
                    if new_top_n:
                        new_top_n = int(new_top_n)
                        if new_top_n > 0:
                            current_top_n = new_top_n
                        else:
                            print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                    
            elif choice == '4':
                # View report
                try:
                    analyzer.print_report(
                        region=current_region,
                        metric=current_metric,
                        top_n=current_top_n
                    )
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"\nError generating report: {e}")
                    input("Press Enter to continue...")
                    
            elif choice == '5':
                # Save report to file
                try:
                    report = analyzer.generate_report(
                        region=current_region,
                        metric=current_metric,
                        top_n=current_top_n
                    )
                    if save_report(report):
                        print("\nReport saved successfully!")
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"\nError saving report: {e}")
                    input("Press Enter to continue...")
                    
            else:
                print("\nInvalid choice. Please try again.")
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if logger.level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        input("\nPress Enter to exit...")


def save_report(report: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Save the report to a JSON file.
    
    Args:
        report: The report data to save
        output_dir: Directory to save the report in (default: current directory)
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = Path.cwd() / 'reports'
        else:
            output_dir = Path(output_dir) / 'reports'
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a filename based on region, metric, and timestamp
        region = report['region'].replace(' ', '_').lower()
        metric = report.get('metric_used', 'score').replace(' ', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"regional_report_{region}_{metric}_{timestamp}.json"
        filepath = output_dir / filename
        
        # Save to file with pretty printing
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Report saved to: {filepath}")
        print(f"\nReport saved to: {filepath}")
        return True
        
    except Exception as e:
        error_msg = f"Error saving report: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"\n{error_msg}")
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Windsurf Credit - Regional Agent Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main arguments
    parser.add_argument(
        '-d', '--output-dir',
        type=str,
        default='output',
        help='Directory containing the output JSON files'
    )
    
    parser.add_argument(
        '-r', '--region',
        type=str,
        default='all',
        help='Region to analyze (use "all" for all regions)'
    )
    
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='score',
        choices=['score', 'gmv', 'dpd', 'utilization', 'repayment_rate', 'credit_sales_ratio', 'risk_score'],
        help='Metric to use for ranking agents'
    )
    
    parser.add_argument(
        '-n', '--top-n',
        type=int,
        default=10,
        help='Number of top/bottom agents to display'
    )
    
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help='Save the report to a file'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (overrides other options)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('regional_analyzer.log')
        ]
    )
    
    try:
        if args.interactive:
            interactive_mode()
        else:
            run_analysis(
                output_dir=args.output_dir,
                region=args.region,
                metric=args.metric,
                top_n=args.top_n,
                save_report_flag=args.save
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nError: {e}")
        if logger.level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    import sys
    
    main()
