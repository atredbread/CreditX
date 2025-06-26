"""
DPD Data Analysis Tool

A consolidated tool for analyzing DPD (Days Past Due) data with multiple analysis methods.
Combines functionality from multiple DPD analysis scripts into a single, maintainable tool.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dpd_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class DPDAnalyzer:
    """Main class for DPD data analysis."""
    
    def __init__(self, dpd_path: str = None):
        """Initialize the DPD Analyzer.
        
        Args:
            dpd_path: Path to the DPD Excel file. If None, will look in source_data/DPD.xlsx
        """
        self.dpd_path = Path(dpd_path) if dpd_path else Path('source_data/DPD.xlsx')
        self.df = None
        self.raw_workbook = None
        
    def load_data(self, method: str = 'pandas') -> bool:
        """Load DPD data using the specified method.
        
        Args:
            method: 'pandas' (faster) or 'openpyxl' (more detailed error reporting)
            
        Returns:
            bool: True if data was loaded successfully
        """
        if not self.dpd_path.exists():
            logger.error(f"DPD file not found at {self.dpd_path}")
            return False
            
        try:
            if method == 'pandas':
                self.df = pd.read_excel(self.dpd_path)
                logger.info(f"Successfully loaded {len(self.df)} rows using pandas")
                return True
                
            elif method == 'openpyxl':
                self.raw_workbook = openpyxl.load_workbook(self.dpd_path, read_only=True)
                logger.info(f"Successfully loaded workbook with sheets: {self.raw_workbook.sheetnames}")
                return True
                
            else:
                logger.error(f"Unknown load method: {method}. Use 'pandas' or 'openpyxl'")
                return False
                
        except Exception as e:
            logger.error(f"Error loading DPD data: {str(e)}", exc_info=True)
            return False
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the DPD data.
        
        Returns:
            Dict containing structure information
        """
        if self.df is None and not self.load_data():
            return {}
            
        structure = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            structure['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
            
        return structure
    
    def find_agent(self, agent_id: str, exact_match: bool = True) -> pd.DataFrame:
        """Find agent by ID with flexible matching.
        
        Args:
            agent_id: The agent ID to search for
            exact_match: If False, performs partial matching on the last 4 digits
            
        Returns:
            DataFrame with matching rows
        """
        if self.df is None and not self.load_data():
            return pd.DataFrame()
            
        agent_id = str(agent_id).strip()
        
        # Try exact match first
        matches = self.df[self.df['Bzid'].astype(str) == agent_id]
        
        # If no matches and not exact match, try partial match on last 4 digits
        if matches.empty and not exact_match and len(agent_id) >= 4:
            last_four = agent_id[-4:]
            matches = self.df[self.df['Bzid'].astype(str).str.endswith(last_four)]
            
        return matches
    
    def analyze_dpd_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of DPD values.
        
        Returns:
            Dict containing DPD distribution statistics
        """
        if self.df is None and not self.load_data():
            return {}
            
        # Find DPD column (case insensitive)
        dpd_col = next((col for col in self.df.columns if 'dpd' in col.lower()), None)
        if not dpd_col:
            logger.warning("No DPD column found in the data")
            return {}
            
        dpd_series = pd.to_numeric(self.df[dpd_col], errors='coerce')
        
        return {
            'column': dpd_col,
            'value_counts': dpd_series.value_counts().sort_index().to_dict(),
            'stats': {
                'min': float(dpd_series.min()),
                'max': float(dpd_series.max()),
                'mean': float(dpd_series.mean()),
                'median': float(dpd_series.median()),
                'std': float(dpd_series.std()),
                'null_count': int(dpd_series.isnull().sum()),
                'zero_count': int((dpd_series == 0).sum())
            }
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive report of the DPD analysis.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            The report as a string
        """
        if self.df is None and not self.load_data():
            return "Error: Could not load DPD data"
            
        report = [
            "=" * 80,
            "DPD DATA ANALYSIS REPORT",
            "=" * 80,
            f"File: {self.dpd_path}",
            f"Generated: {pd.Timestamp.now()}",
            ""
        ]
        
        # Data Structure
        report.extend(["DATA STRUCTURE", "-" * 40])
        structure = self.analyze_structure()
        report.append(f"Rows: {structure['shape'][0]:,}")
        report.append(f"Columns: {structure['shape'][1]}")
        report.append("\nCOLUMNS:")
        for col, dtype in structure['dtypes'].items():
            report.append(f"  - {col} ({dtype})")
        
        # DPD Distribution
        dpd_analysis = self.analyze_dpd_distribution()
        if dpd_analysis:
            report.extend([
                "\nDPD DISTRIBUTION",
                "-" * 40,
                f"Column: {dpd_analysis['column']}",
                "\nValue Counts:"
            ])
            for value, count in dpd_analysis['value_counts'].items():
                report.append(f"  - {value}: {count:,}")
                
            stats = dpd_analysis['stats']
            report.extend([
                "\nStatistics:",
                f"  - Min: {stats['min']:.1f}",
                f"  - Max: {stats['max']:.1f}",
                f"  - Mean: {stats['mean']:.1f}",
                f"  - Median: {stats['median']:.1f}",
                f"  - Std Dev: {stats['std']:.1f}",
                f"  - Null Values: {stats['null_count']:,}",
                f"  - Zero Values: {stats['zero_count']:,}"
            ])
        
        # Sample Data
        report.extend([
            "\nSAMPLE DATA",
            "-" * 40,
            self.df.head(3).to_string()
        ])
        
        # Join report lines
        report_text = "\n".join(str(line) for line in report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze DPD data')
    parser.add_argument('--file', type=str, help='Path to DPD Excel file')
    parser.add_argument('--agent', type=str, help='Agent ID to search for')
    parser.add_argument('--output', type=str, help='Output file for the report')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    analyzer = DPDAnalyzer(args.file)
    
    # Generate full report
    report = analyzer.generate_report(args.output)
    print(report)
    
    # If agent ID was provided, show agent details
    if args.agent:
        print("\n" + "=" * 80)
        print(f"SEARCHING FOR AGENT: {args.agent}")
        print("-" * 80)
        
        # First try exact match
        matches = analyzer.find_agent(args.agent, exact_match=True)
        if not matches.empty:
            print(f"Found {len(matches)} exact matches:")
            print(matches.to_string())
        else:
            # Try partial match
            matches = analyzer.find_agent(args.agent, exact_match=False)
            if not matches.empty:
                print(f"Found {len(matches)} partial matches (last 4 digits):")
                print(matches.to_string())
            else:
                print("No matches found for the specified agent ID.")

if __name__ == "__main__":
    main()
