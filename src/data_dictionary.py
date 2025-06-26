"""
Data Dictionary Generator for Credit Health Intelligence Engine

This script generates a data dictionary for all input datasets,
including column descriptions, data types, and sample values.
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDictionaryGenerator:
    """Generates data dictionary for all datasets."""
    
    def __init__(self, input_dir: str = 'source_data'):
        """Initialize with input directory."""
        self.input_dir = Path(input_dir)
        self.datasets = {
            'credit_Agents.xlsx': {
                'description': 'Master list of onboarded agents with credit information',
                'key_columns': ['Bzid', 'Phone']
            },
            'Credit_history_sales_vs_credit_sales.xlsx': {
                'description': 'Historical sales and credit sales data',
                'key_columns': ['Account']
            },
            'Credit_sales_data.xlsx': {
                'description': 'Monthly credit sales transactions',
                'key_columns': ['account', 'DATE']
            },
            'Region_contact.xlsx': {
                'description': 'Mapping of regions to contact persons',
                'key_columns': ['Region']
            },
            'DPD.xlsx': {
                'description': 'Days Past Due information for credit accounts',
                'key_columns': ['Bzid', 'Phone']
            },
            'sales_data.xlsx': {
                'description': 'Complete sales transaction data',
                'key_columns': ['account', 'DATE(a.creationtime)']
            }
        }
    
    def generate_dictionary(self) -> Dict[str, Any]:
        """Generate data dictionary for all datasets."""
        data_dict = {}
        
        for file_name, meta in self.datasets.items():
            file_path = self.input_dir / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                # Read just the first few rows to get schema
                df = pd.read_excel(file_path, nrows=5)
                
                # Get column information
                columns_info = []
                for col in df.columns:
                    col_info = {
                        'name': col,
                        'dtype': str(df[col].dtype),
                        'sample_values': self._get_sample_values(df[col]),
                        'description': ''  # To be filled manually
                    }
                    columns_info.append(col_info)
                
                # Add to data dictionary
                data_dict[file_name] = {
                    'description': meta['description'],
                    'row_count': len(pd.read_excel(file_path, usecols=[0])),
                    'columns': columns_info,
                    'key_columns': meta.get('key_columns', []),
                    'relationships': self._get_relationships(file_name)
                }
                
                logger.info(f"Processed {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
        
        return data_dict
    
    def _get_sample_values(self, series: pd.Series, n: int = 3) -> list:
        """Get sample non-null values from a series."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []
        return non_null.head(n).tolist()
    
    def _get_relationships(self, file_name: str) -> list:
        """Define relationships between datasets."""
        relationships = []
        
        if file_name == 'credit_Agents.xlsx':
            relationships.append({
                'related_to': 'DPD.xlsx',
                'on': ['Bzid', 'Phone'],
                'type': 'one-to-one'
            })
        elif file_name == 'Credit_sales_data.xlsx':
            relationships.append({
                'related_to': 'credit_Agents.xlsx',
                'on': [('account', 'Bzid')],
                'type': 'many-to-one'
            })
        
        return relationships
    
    def save_to_markdown(self, output_file: str = 'DATA_DICTIONARY.md'):
        """Save data dictionary to a markdown file."""
        data_dict = self.generate_dictionary()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Data Dictionary\n\n")
            f.write("This document describes the structure and content of all input datasets.\n\n")
            
            for file_name, info in data_dict.items():
                f.write(f"## {file_name}\n\n")
                f.write(f"**Description**: {info['description']}\n\n")
                f.write(f"**Number of Rows**: {info['row_count']:,}\n\n")
                
                if info['key_columns']:
                    f.write("**Key Columns**: " + ", ".join(f'`{col}`' for col in info['key_columns']) + "\n\n")
                
                if info['relationships']:
                    f.write("### Relationships\n\n")
                    for rel in info['relationships']:
                        f.write(f"- **Related to**: {rel['related_to']}  \n")
                        f.write(f"  **On**: {', '.join(str(x) for x in rel['on'])}  \n")
                        f.write(f"  **Type**: {rel['type']}\n")
                    f.write("\n")
                
                f.write("### Columns\n\n")
                f.write("| Column | Data Type | Sample Values | Description |\n")
                f.write("|--------|-----------|----------------|-------------|\n")
                
                for col in info['columns']:
                    samples = ", ".join(str(x) for x in col['sample_values'][:3])
                    f.write(f"| `{col['name']}` | {col['dtype']} | {samples} | {col['description']} |\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"Data dictionary saved to {output_file}")


def main():
    """Generate and save data dictionary."""
    try:
        generator = DataDictionaryGenerator()
        generator.save_to_markdown()
        print("✅ Data dictionary generated successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error generating data dictionary: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
