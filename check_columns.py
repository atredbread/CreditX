"""
Script to check and document the structure of all source data files.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def get_file_info(file_path: Path) -> dict:
    """Extract file information and sample data."""
    try:
        # Read first 5 rows to get structure
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        else:
            df = pd.read_excel(file_path, nrows=5)
        
        # Get basic info
        info = {
            'file': str(file_path.name),
            'columns': df.columns.tolist(),
            'row_count': '>5 (sample)',
            'dtypes': {}
        }
        
        # Add data types
        for col in df.columns:
            info['dtypes'][col] = str(df[col].dtype)
        
        # Add sample data (first row as string representation)
        info['sample_row'] = {}
        for col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val):
                info['sample_row'][col] = None
            else:
                info['sample_row'][col] = str(val)
        
        return info
        
    except Exception as e:
        return {
            'file': str(file_path.name),
            'error': str(e)
        }

def main():
    source_dir = Path('source_data')
    output_file = 'data_structure.md'
    
    # Get all data files
    data_files = list(source_dir.glob('*'))
    data_files = [f for f in data_files if f.is_file() and not f.name.startswith('~$')]
    
    # Process each file
    results = {}
    for file_path in data_files:
        print(f"Processing {file_path.name}...")
        results[file_path.name] = get_file_info(file_path)
    
    # Generate markdown output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Data Structure Documentation\n\n")
        f.write("*Generated on: {}*\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        for file_name, info in results.items():
            f.write(f"## {file_name}\n\n")
            
            if 'error' in info:
                f.write(f"**Error:** {info['error']}\n\n")
                continue
                
            f.write("### Columns\n")
            f.write("| Column | Data Type | Sample Value |\n")
            f.write("|--------|-----------|---------------|\n")
            
            for col in info['columns']:
                dtype = info['dtypes'].get(col, 'unknown')
                sample = info['sample_row'].get(col, '')
                f.write(f"| `{col}` | `{dtype}` | `{sample}` |\n")
            
            f.write("\n---\n\n")
    
    print(f"\nDocumentation generated in {output_file}")
    
    # Print summary
    print("\nSummary:")
    print("|", "File".ljust(40), "|", "Columns".ljust(10), "|")
    print("|", "-"*40, "|", "-"*10, "|")
    for file_name, info in results.items():
        if 'error' in info:
            print(f"| {file_name.ljust(40)} | {'Error'.ljust(10)} |")
        else:
            print(f"| {file_name.ljust(40)} | {str(len(info['columns'])).ljust(10)} |")

if __name__ == "__main__":
    main()
