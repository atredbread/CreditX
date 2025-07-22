"""
Script to update import statements after project reorganization.
"""
import re
from pathlib import Path

# Mapping of old import paths to new ones
IMPORT_MAPPING = {
    # Old patterns should be more specific to avoid false positives
    r'from\s+(lookup_agent_profiles|agent_analyzer|analyze_repayments|analyze_unmatched_agents|check_credit_data|check_gmv_data|check_columns|cleanup_project|setup_output_dirs|windsurf_credit_runner)\s+import': 
        lambda m: f'from scripts.{m.group(1)} import',
    
    r'import\s+(lookup_agent_profiles|agent_analyzer|analyze_repayments|analyze_unmatched_agents|check_credit_data|check_gmv_data|check_columns|cleanup_project|setup_output_dirs|windsurf_credit_runner)(?:\s|\n|\r|$)': 
        lambda m: f'import scripts.{m.group(1)}',
    
    # Add more patterns as needed
}

def update_file_imports(file_path: Path) -> bool:
    """Update import statements in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        for pattern, replacement in IMPORT_MAPPING.items():
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
    
    # Process Python files in the project
    for py_file in base_dir.rglob('*.py'):
        if update_file_imports(py_file):
            updated_files.append(str(py_file.relative_to(base_dir)))
    
    print("\nUpdated imports in the following files:")
    for file in updated_files:
        print(f"- {file}")
    
    if not updated_files:
        print("No files needed import updates.")

if __name__ == "__main__":
    main()
