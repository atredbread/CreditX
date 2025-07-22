"""
Comprehensive cleanup script for the Credit Risk Analysis project.
Removes temporary files, caches, test files, and generated reports.
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

# Important directories to preserve
PRESERVE_DIRS = [
    "data",
    "docs",
    "source_data",
    "src",
    "models"
]

def remove_files_by_patterns(directory: Path, patterns: List[str], recursive: bool = True) -> None:
    """Remove files matching patterns in directory.
    
    Args:
        directory: Directory to search in
        patterns: List of file patterns to match (e.g., ['*.pyc', '*.log'])
        recursive: Whether to search recursively in subdirectories
    """
    for pattern in patterns:
        search_method = directory.rglob if recursive else directory.glob
        for file_path in search_method(pattern):
            try:
                if file_path.is_file() and not is_protected(file_path):
                    file_path.unlink()
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def remove_directories(directory: Path, dir_names: List[str], recursive: bool = True) -> None:
    """Remove directories matching names.
    
    Args:
        directory: Base directory to search in
        dir_names: List of directory names to remove
        recursive: Whether to search recursively in subdirectories
    """
    for dir_name in dir_names:
        search_method = directory.rglob if recursive else directory.glob
        for dir_path in search_method(dir_name):
            try:
                if dir_path.is_dir() and not is_protected(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")

def is_protected(path: Path) -> bool:
    """Check if a path is in a protected directory."""
    try:
        path = path.resolve()
        for protected_dir in PRESERVE_DIRS:
            protected_path = (Path(__file__).parent / protected_dir).resolve()
            if protected_path in path.parents:
                return True
        return False
    except Exception:
        return True  # If there's any error, better to be safe and protect the path

def clean_json_cache() -> None:
    """Remove JSON cache files while preserving the cache directory structure."""
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                print(f"Removed cache file: {cache_file}")
            except Exception as e:
                print(f"Error removing {cache_file}: {e}")


def clean_output_files() -> None:
    """Clean up output files while preserving the output directory structure."""
    output_dir = Path("output")
    if output_dir.exists():
        for output_file in output_dir.glob("*"):
            try:
                if output_file.is_file() and not is_protected(output_file):
                    output_file.unlink()
                    print(f"Removed output file: {output_file}")
            except Exception as e:
                print(f"Error removing {output_file}: {e}")


def clean_project() -> None:
    """Main function to clean the project."""
    base_dir = Path(__file__).parent.parent.parent  # Project root
    
    print("\n=== Removing cache directories ===")
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
        ".coverage",
        "reports",
        "test_reports",
        ".ipynb_checkpoints"
    ]
    remove_directories(base_dir, cache_dirs)
    
    print("\n=== Cleaning JSON cache ===")
    clean_json_cache()
    
    print("\n=== Cleaning output files ===")
    clean_output_files()
    
    print("\n=== Removing test files ===")
    test_patterns = ["test_*.py", "*_test.py"]
    remove_files_by_patterns(base_dir, test_patterns)
    
    print("\n=== Removing temporary files ===")
    temp_patterns = [
        "*.pyc", "*.pyo", "*.pyd", "*.py,cover",
        "*.so", "*.c", "*.log", "*.tmp",
        "*.bak", "*.swp", "*~", "*.egg-info"
    ]
    remove_files_by_patterns(base_dir, temp_patterns)
    
    print("\nCleanup complete!")
    print("\nPreserved directories:", ", ".join(PRESERVE_DIRS))

if __name__ == "__main__":
    clean_project()
