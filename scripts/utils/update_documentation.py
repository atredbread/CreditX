"""
Script to update documentation paths after project reorganization.
"""
import re
from pathlib import Path

# Mapping of old paths to new paths in documentation
PATH_MAPPING = {
    # Update script references
    'cleanup_project.py': 'scripts/utils/cleanup.py',
    'lookup_agent_profiles.py': 'scripts/analysis/lookup_agent_profiles.py',
    'agent_analyzer.py': 'scripts/analysis/agent_analyzer.py',
    'analyze_repayments.py': 'scripts/analysis/analyze_repayments.py',
    'analyze_unmatched_agents.py': 'scripts/analysis/analyze_unmatched_agents.py',
    'analyze_unmatched_agents_detailed.py': 'scripts/analysis/analyze_unmatched_agents_detailed.py',
    'check_columns.py': 'scripts/data_processing/check_columns.py',
    'check_credit_data.py': 'scripts/analysis/check_credit_data.py',
    'check_gmv_data.py': 'scripts/analysis/check_gmv_data.py',
    'setup_output_dirs.py': 'scripts/utils/setup_output_dirs.py',
    'windsurf_credit_runner.py': 'scripts/runner.py',
    
    # Update directory references
    'src/': 'src/credit_risk_analysis/',
    'tests/': 'tests/unit/ or tests/integration/',
}

def update_file_paths(file_path: Path) -> bool:
    """Update file paths in documentation."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        for old_path, new_path in PATH_MAPPING.items():
            # Use regex to match the path in different contexts (quotes, code blocks, etc.)
            escaped_path = re.escape(old_path.replace('\\', '/'))  # Handle Windows paths
            pattern = f'([\'"`\s]|^){escaped_path}([\'"`\s]|$)'
            replacement = f'\g<1>{new_path}\g<2>'
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    updated_files = []
    
    # Process documentation files
    doc_extensions = ['.md', '.rst', '.txt']
    for ext in doc_extensions:
        for doc_file in base_dir.rglob(f'*{ext}'):
            # Skip files in venv and other non-project directories
            if any(part.startswith('.') or part == 'venv' or part == '.venv' 
                  for part in doc_file.parts):
                continue
                
            if update_file_paths(doc_file):
                updated_files.append(str(doc_file.relative_to(base_dir)))
    
    print("\nUpdated paths in the following documentation files:")
    for file in updated_files:
        print(f"- {file}")
    
    if not updated_files:
        print("No documentation files needed path updates.")
    
    # Update README.md with new structure
    update_readme_structure(base_dir / 'README.md')

def update_readme_structure(readme_path: Path):
    """Update the project structure section in README.md."""
    if not readme_path.exists():
        return
        
    structure = """
## 📁 Project Structure

```
credit_risk_analysis/
├── config/                    # Configuration files
├── data/                      # Data directories
│   ├── raw/                   # Raw data (immutable)
│   ├── processed/             # Processed data
│   └── external/              # External data sources
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   └── examples/              # Usage examples
├── notebooks/                 # Jupyter notebooks for analysis
├── scripts/                   # Utility scripts
│   ├── analysis/              # Analysis scripts
│   ├── data_processing/       # Data processing scripts
│   └── utils/                 # Utility scripts
├── src/                       # Source code
│   └── credit_risk_analysis/  # Main package
├── tests/                     # Test files
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── .gitignore
├── pyproject.toml             # Project metadata and dependencies
└── README.md
```
"""
    
    try:
        content = readme_path.read_text(encoding='utf-8')
        
        # Replace existing project structure section if it exists
        if '## 📁 Project Structure' in content:
            # Remove existing structure section
            content = re.sub(
                r'## 📁 Project Structure[\s\S]*?(?=## \w|$)',
                structure,
                content
            )
        else:
            # Add structure section before the Contributing section
            content = content.replace(
                '## 🤝 Contributing',
                f'{structure.rstrip()}\n\n## 🤝 Contributing'
            )
        
        readme_path.write_text(content, encoding='utf-8')
        print("\nUpdated project structure in README.md")
        
    except Exception as e:
        print(f"Error updating README.md: {e}")

if __name__ == "__main__":
    main()
