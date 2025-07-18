"""
Cleanup script for the Credit Risk Analysis project.
Removes temporary files, caches, and generated reports.
"""
import os
import shutil
from pathlib import Path

def clean_directory(directory: str, patterns: list[str]) -> None:
    """Remove files matching patterns in directory."""
    for pattern in patterns:
        for file_path in Path(directory).rglob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"Removed file: {file_path}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def main():
    # Define directories to clean
    base_dir = Path(__file__).parent
    dirs_to_clean = [
        base_dir / "__pycache__",
        base_dir / ".pytest_cache",
        base_dir / ".mypy_cache",
        base_dir / ".coverage",
        base_dir / "htmlcov",
        base_dir / ".pytest_cache",
        base_dir / "reports",
        base_dir / "test_reports",
        base_dir / "output",
    ]

    # Define file patterns to remove
    file_patterns = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.py,cover",
        "*.so",
        "*.c",
        "*.log",
        "*.tmp",
        "*.bak",
        "*.swp",
        "*~",
        "*$py.class",
    ]

    print("Starting cleanup...")
    
    # Clean files matching patterns
    clean_directory(base_dir, file_patterns)
    
    # Clean directories
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                else:
                    dir_path.unlink()
                    print(f"Removed file: {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")
    
    # Recreate necessary empty directories
    (base_dir / "reports").mkdir(exist_ok=True)
    (base_dir / "output").mkdir(exist_ok=True)
    
    print("\nCleanup complete!")
    print("Kept important directories: data/, docs/, source_data/")

if __name__ == "__main__":
    main()
