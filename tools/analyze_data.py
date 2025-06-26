import pandas as pd
from pathlib import Path

def analyze_file(file_path, file_type):
    """Analyze a data file and return its structure and sample data."""
    try:
        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path, nrows=5)
        else:
            df = pd.read_csv(file_path, nrows=5)
            
        # Get basic info
        info = {
            'file': file_path.name,
            'rows': len(pd.read_excel(file_path) if file_path.suffix == '.xlsx' else pd.read_csv(file_path)),
            'columns': df.columns.tolist(),
            'sample_data': df.head(3).to_dict('records')
        }
        return info
    except Exception as e:
        return {'file': file_path.name, 'error': str(e)}

def generate_documentation(data_dir='source_data'):
    """Generate documentation for all data files."""
    data_dir = Path(data_dir)
    files = list(data_dir.glob('*'))
    
    docs = ["# Credit Risk Data Documentation\n"]
    
    for file in files:
        if file.suffix in ['.xlsx', '.csv']:
            file_info = analyze_file(file, file.suffix)
            
            if 'error' in file_info:
                docs.append(f"## {file.name}\nError: {file_info['error']}\n")
                continue
                
            docs.append(f"## {file.name}\n")
            docs.append(f"- **Rows:** {file_info['rows']:,}")
            docs.append("\n### Columns:")
            
            for col in file_info['columns']:
                docs.append(f"- {col}")
                
            docs.append("\n### Sample Data:")
            docs.append("```")
            
            # Convert sample data to a readable format
            if file_info['sample_data']:
                # Get column headers
                headers = file_info['sample_data'][0].keys()
                # Calculate column widths
                col_widths = {col: max(len(str(col)), max(len(str(row[col])) for row in file_info['sample_data'])) for col in headers}
                
                # Create header row
                header = " | ".join(str(col).ljust(col_widths[col]) for col in headers)
                docs.append(header)
                docs.append("-" * len(header))
                
                # Create data rows
                for row in file_info['sample_data']:
                    row_str = " | ".join(str(row[col]).ljust(col_widths[col]) for col in headers)
                    docs.append(row_str)
            
            docs.append("```\n")
    
    # Save documentation
    with open('DATA_DOCUMENTATION.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(docs))
    
    print("Documentation generated: DATA_DOCUMENTATION.md")

if __name__ == "__main__":
    generate_documentation()
