"""
Data Quality Checks for Credit Health Intelligence Engine

This script performs data quality checks on all input datasets,
including missing values, duplicates, and basic statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Performs data quality checks on input datasets."""
    
    def __init__(self, input_dir: str = 'source_data', output_dir: str = 'output/quality_checks'):
        """Initialize with input and output directories."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define expected key columns for each dataset
        self.key_columns = {
            'credit_Agents.xlsx': ['Bzid', 'Phone'],
            'Credit_history_sales_vs_credit_sales.xlsx': ['Account'],
            'Credit_sales_data.xlsx': ['account', 'DATE'],
            'Region_contact.xlsx': ['Region'],
            'DPD.xlsx': ['Bzid', 'Phone'],
            'sales_data.xlsx': ['account', 'DATE(a.creationtime)']
        }
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets."""
        data = {}
        for file in self.input_dir.glob('*.xlsx'):
            try:
                df = pd.read_excel(file)
                data[file.name] = df
                logger.info(f"Loaded {file.name} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        return data
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate missing values statistics."""
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().mean() * 100).round(2)
        missing_data = pd.concat([total, percent], axis=1, 
                               keys=['Missing Values', 'Percent Missing'])
        return missing_data[missing_data['Missing Values'] > 0]
    
    def check_duplicates(self, df: pd.DataFrame, key_columns: List[str]) -> Tuple[int, pd.DataFrame]:
        """Check for duplicate rows based on key columns."""
        if not key_columns or not all(col in df.columns for col in key_columns):
            return 0, pd.DataFrame()
            
        duplicates = df[df.duplicated(subset=key_columns, keep=False)]
        return len(duplicates), duplicates
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistics for numeric columns."""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            stats['descriptive_stats'] = df[numeric_cols].describe().T
        
        return stats
    
    def analyze_dataset(self, file_name: str, df: pd.DataFrame):
        """Perform all quality checks on a single dataset."""
        logger.info(f"Analyzing {file_name}...")
        
        results = {
            'file_name': file_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': {},
            'duplicates': None,
            'duplicate_rows': None,
            'stats': None
        }
        
        try:
            # Check for missing values
            missing = self.check_missing_values(df)
            if not missing.empty:
                results['missing_values'] = missing.reset_index().rename(
                    columns={'index': 'column'}
                ).to_dict('records')
            
            # Check for duplicates
            key_cols = [col for col in self.key_columns.get(file_name, []) if col in df.columns]
            if key_cols:
                dup_count, dup_df = self.check_duplicates(df, key_cols)
                if dup_count > 0:
                    results['duplicates'] = {
                        'count': dup_count,
                        'duplicate_keys': dup_df[key_cols].to_dict('records')
                    }
                    results['duplicate_rows'] = dup_df.to_dict('records')
            
            # Generate statistics
            stats = self.generate_summary_statistics(df)
            if stats:
                results['stats'] = stats
                
        except Exception as e:
            logger.error(f"Error analyzing {file_name}: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def visualize_missing_data(self, df: pd.DataFrame, file_name: str):
        """Create a visualization of missing data."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, 
                   yticklabels=False, cmap='viridis')
        plt.title(f'Missing Values Heatmap - {file_name}')
        output_file = self.output_dir / f'missing_values_{file_name.replace(".xlsx", ".png")}'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        return str(output_file)
    
    def run_quality_checks(self):
        """Run all quality checks on all datasets."""
        logger.info("Starting data quality checks...")
        
        # Load all datasets
        data = self.load_data()
        if not data:
            logger.error("No data files found or could be loaded.")
            return
        
        results = {}
        visualizations = {}
        
        # Analyze each dataset
        for file_name, df in data.items():
            try:
                results[file_name] = self.analyze_dataset(file_name, df)
                # Generate visualization for missing data
                viz_path = self.visualize_missing_data(df, file_name)
                visualizations[file_name] = viz_path
            except Exception as e:
                logger.error(f"Error analyzing {file_name}: {str(e)}")
        
        # Save results
        self.save_results(results, visualizations)
        logger.info("Data quality checks completed!")
        
        return results, visualizations
    
    def save_results(self, results: Dict, visualizations: Dict):
        """Save quality check results to files."""
        # Save detailed results as JSON
        import json
        
        # Convert DataFrame objects to dict for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            return obj
        
        results_serializable = {}
        for file_name, result in results.items():
            results_serializable[file_name] = {
                k: convert_for_json(v) for k, v in result.items()
            }
        
        # Save results
        output_file = self.output_dir / 'data_quality_report.json'
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        
        # Generate a markdown report
        self.generate_markdown_report(results_serializable, visualizations)
    
    def generate_markdown_report(self, results: Dict, visualizations: Dict):
        """Generate a markdown report of the data quality checks."""
        report = "# Data Quality Report\n\n"
        report += "This report provides an overview of data quality across all input datasets.\n\n"
        
        # Table of contents
        report += "## Table of Contents\n"
        for file_name in results.keys():
            report += f"- [{file_name}](#{file_name.replace('.', '').lower()})\n"
        report += "\n---\n\n"
        
        for file_name, result in results.items():
            # Create anchor for table of contents
            report += f"## {file_name} <a id='{file_name.replace('.', '').lower()}'></a>\n\n"
            report += f"**Shape:** {result['shape']} (rows × columns)\n\n"
            
            # Show error if analysis failed
            if 'error' in result:
                report += f"❌ **Error during analysis:** {result['error']}\n\n"
                continue
            
            # Missing values section
            if result.get('missing_values'):
                report += "### Missing Values\n\n"
                report += "| Column | Missing Count | Missing % |\n"
                report += "|--------|--------------:|----------:|\n"
                for item in result['missing_values']:
                    report += f"| `{item['column']}` | {item['Missing Values']:,} | {item['Percent Missing']}% |\n"
                report += "\n"
            else:
                report += "✅ No missing values found.\n\n"
            
            # Duplicates section
            if result.get('duplicates'):
                report += "### Duplicate Rows\n\n"
                report += f"🔍 Found **{result['duplicates']['count']}** duplicate rows based on key columns.\n\n"
                
                if result['duplicates']['count'] > 0:
                    report += "**Example duplicate keys:**\n\n"
                    for i, dup in enumerate(result['duplicates']['duplicate_keys'][:5], 1):
                        report += f"{i}. {', '.join(f'{k}: `{v}`' for k, v in dup.items())}\n"
                    report += "\n"
            
            # Basic statistics section
            if result.get('stats') and 'descriptive_stats' in result['stats']:
                stats = result['stats']['descriptive_stats']
                if not stats.empty:
                    report += "### Numeric Column Statistics\n\n"
                    report += "| Column | Mean | Min | 25% | 50% | 75% | Max |\n"
                    report += "|--------|-----:|----:|---:|---:|---:|----:|\n"
                    for idx, row in stats.iterrows():
                        report += f"| `{idx}` | "
                        report += f"{row.get('mean', ''):.2f} | {row.get('min', ''):.2f} | "
                        report += f"{row.get('25%', ''):.2f} | {row.get('50%', ''):.2f} | "
                        report += f"{row.get('75%', ''):.2f} | {row.get('max', ''):.2f} |\n"
                    report += "\n"
            
            # Add visualization if available
            if file_name in visualizations:
                report += "### Missing Data Visualization\n\n"
                report += f"![Missing Values in {file_name}]({visualizations[file_name]})\n\n"
            
            report += "---\n\n"
        
        # Save the report
        report_file = self.output_dir / 'data_quality_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Run data quality checks and generate report."""
    try:
        checker = DataQualityChecker()
        results, visualizations = checker.run_quality_checks()
        print("✅ Data quality checks completed successfully!")
        print(f"Report saved to: {checker.output_dir / 'data_quality_report.md'}")
        return 0
    except Exception as e:
        logger.error(f"Error running data quality checks: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
