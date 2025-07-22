#!/usr/bin/env python3
"""
Cleanup script for Credit Risk project.

This script removes temporary files, caches, and old outputs to keep the project directory clean.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
CACHE_DIR = PROJECT_ROOT / '.cache'
LOGS_DIR = PROJECT_ROOT / 'logs'

# File patterns to clean up
PATTERNS = [
    '*.pyc',              # Python compiled files
    '*.pyo',              # Python optimized files
    '*.pyd',              # Python dynamic modules
    '__pycache__',        # Python cache directories
    '*.log',              # Log files
    '.mypy_cache',        # MyPy cache
    '.pytest_cache',      # Pytest cache
    '.coverage',          # Coverage report
    'htmlcov',            # HTML coverage report
    '.ipynb_checkpoints', # Jupyter notebook checkpoints
    '.DS_Store',          # macOS system file
    'Thumbs.db',          # Windows thumbnail cache
    '*.swp',              # Vim swap files
    '*.swo',              # Vim swap files
    '*.swn',              # Vim swap files
    '*.bak',              # Backup files
    '*.tmp',              # Temporary files
    '*.temp',             # Temporary files
    '*.cache',            # Cache files
]

def remove_files_and_dirs(base_dir: Path, patterns: list, dry_run: bool = False) -> int:
    """Remove files and directories matching patterns."""
    count = 0
    for pattern in patterns:
        for path in base_dir.rglob(pattern):
            try:
                if path.is_file():
                    if not dry_run:
                        path.unlink()
                    print(f"Removed file: {path.relative_to(PROJECT_ROOT)}")
                    count += 1
                elif path.is_dir():
                    if not dry_run:
                        shutil.rmtree(path)
                    print(f"Removed directory: {path.relative_to(PROJECT_ROOT)}")
                    count += 1
            except Exception as e:
                print(f"Error removing {path}: {e}")
    return count

def clean_output_dir(dry_run: bool = False) -> int:
    """Clean up output directory, keeping only the most recent files."""
    if not OUTPUT_DIR.exists():
        return 0
        
    # Keep these files
    keep_files = {
        'agent_analysis.json',
        'data_dictionary.json',
        'validation_report.json'
    }
    
    count = 0
    for path in OUTPUT_DIR.glob('*'):
        if path.name in keep_files:
            continue
            
        try:
            if not dry_run:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            print(f"Removed output: {path.relative_to(PROJECT_ROOT)}")
            count += 1
        except Exception as e:
            print(f"Error removing {path}: {e}")
    
    return count

def clean_old_logs(days_to_keep: int = 7, dry_run: bool = False) -> int:
    """Remove log files older than specified days."""
    if not LOGS_DIR.exists():
        return 0
        
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    count = 0
    
    for log_file in LOGS_DIR.glob('*.log'):
        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff:
                if not dry_run:
                    log_file.unlink()
                print(f"Removed old log: {log_file.relative_to(PROJECT_ROOT)} (last modified: {mtime})")
                count += 1
        except Exception as e:
            print(f"Error removing {log_file}: {e}")
    
    return count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up project directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting')
    parser.add_argument('--keep-logs', type=int, default=7, 
                       help='Number of days of log files to keep (default: 7)')
    
    args = parser.parse_args()
    
    print(f"Cleaning up {PROJECT_ROOT}...\n")
    
    # Clean up files and directories matching patterns
    print("Removing temporary files and caches...")
    pattern_count = remove_files_and_dirs(PROJECT_ROOT, PATTERNS, args.dry_run)
    
    # Clean up output directory
    print("\nCleaning output directory...")
    output_count = clean_output_dir(args.dry_run)
    
    # Clean up old logs
    print(f"\nCleaning log files older than {args.keep_logs} days...")
    log_count = clean_old_logs(args.keep_logs, args.dry_run)
    
    total = pattern_count + output_count + log_count
    
    print(f"\nCleanup complete. {'Would remove' if args.dry_run else 'Removed'} {total} items:")
    print(f"- {pattern_count} temporary files and caches")
    print(f"- {output_count} old output files")
    print(f"- {log_count} old log files")

if __name__ == "__main__":
    main()
