"""
Dependency Checker for Windsurf Credit Health Intelligence Engine.

This module provides functionality to check for required Python packages,
Python version, and data files needed by the application.
"""

import sys
import importlib.metadata
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Required Python version (3.8+)
REQUIRED_PYTHON = (3, 8)

# Required packages with optional version constraints
REQUIRED_PACKAGES = {
    'pandas': '>=1.3.0',
    'numpy': '>=1.21.0',
    'openpyxl': '>=3.0.0',  # For Excel file support
    'matplotlib': '>=3.4.0',  # For plotting
    'seaborn': '>=0.11.0',   # For enhanced visualizations
}

# ASCII alternatives for emojis for Windows compatibility
STATUS_ICONS = {
    'check': '[OK]',
    'cross': '[X]',
    'search': '[*]',
    'warning': '[!]',
}

# Required data file patterns
REQUIRED_DATA_FILES = {
    'credit_agents': 'credit_Agents.xlsx',
    'credit_sales': 'Credit_sales_data.xlsx',
    'sales': 'sales_data.xlsx',
    'dpd': 'DPD.xlsx',
    'region_contact': 'Region_contact.xlsx',
    'credit_history': 'Credit_history_sales_vs_credit_sales.xlsx'
}

class DependencyChecker:
    """Check system and Python dependencies."""
    
    @classmethod
    def check_python_version(cls) -> Tuple[bool, str]:
        """
        Check if the Python version meets requirements.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if sys.version_info < REQUIRED_PYTHON:
            version_str = '.'.join(map(str, REQUIRED_PYTHON))
            return False, f"Python {version_str} or later is required (found {sys.version.split()[0]})"
        return True, f"Python {'.'.join(map(str, sys.version_info[:3]))} (meets requirements)"
    
    @staticmethod
    def _version_compare(v1: str, op: str, v2: str) -> bool:
        """Compare two version strings using the specified operator."""
        from packaging import version
        v1 = version.parse(v1)
        v2 = version.parse(v2)
        
        if op == '>=':
            return v1 >= v2
        elif op == '>':
            return v1 > v2
        elif op == '==':
            return v1 == v2
        elif op == '<=':
            return v1 <= v2
        elif op == '<':
            return v1 < v2
        return False

    @classmethod
    def check_packages(cls) -> Tuple[bool, str]:
        """
        Check if required Python packages are installed.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        missing_pkgs = []
        for pkg, version_constraint in REQUIRED_PACKAGES.items():
            try:
                pkg_version = importlib.metadata.version(pkg)
                
                # Handle version constraints (e.g., '>=3.4.0')
                if version_constraint.startswith(('>', '<', '=', '~', '!')):
                    # Extract operator and version
                    for op in ('>=', '<=', '>', '<', '~=', '==', '!='):
                        if version_constraint.startswith(op):
                            required_version = version_constraint[len(op):].strip()
                            if not cls._version_compare(pkg_version, op, required_version):
                                missing_pkgs.append(f"{pkg}{op}{required_version} (found {pkg_version})")
                            break
            except importlib.metadata.PackageNotFoundError:
                missing_pkgs.append(f"{pkg}{version_constraint} (not installed)")
        
        if missing_pkgs:
            return False, f"Missing or outdated packages: {', '.join(missing_pkgs)}"
        return True, "All required packages are installed"
    
    @classmethod
    def check_data_files(cls, data_dir: Path) -> Tuple[bool, str]:
        """
        Check if required data files exist.
        
        Args:
            data_dir: Path to the data directory
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not data_dir.exists():
            return False, f"Data directory not found: {data_dir}"
            
        missing_files = []
        for file_type, filename in REQUIRED_DATA_FILES.items():
            file_path = data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing data files: {', '.join(missing_files)}"
        return True, f"All required data files found in {data_dir}"
    
    @classmethod
    def run_checks(cls, data_dir: str = 'source_data') -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
        """
        Run all dependency checks and return results.
        
        Args:
            data_dir: Path to the data directory (default: 'source_data')
            
        Returns:
            Tuple of (all_checks_passed: bool, results: Dict[str, Tuple[bool, str]])
        """
        data_dir_path = Path(data_dir)
        checks = {
            'python_version': cls.check_python_version(),
            'packages': cls.check_packages(),
            'data_files': cls.check_data_files(data_dir_path)
        }
        
        all_ok = all(success for success, _ in checks.values())
        return all_ok, checks

def format_check_results(checks: Dict[str, Tuple[bool, str]]) -> str:
    """
    Format the results of dependency checks for display.
    
    Args:
        checks: Dictionary of check results from DependencyChecker.run_checks()
        
    Returns:
        Formatted string with check results
    """
    results = []
    for check_name, (success, message) in checks.items():
        status = "✅" if success else "❌"
        results.append(f"{status} {check_name.replace('_', ' ').title()}: {message}")
    
    return "\n".join(results)

def check_dependencies(data_dir: str = 'source_data') -> bool:
    """
    Check all dependencies and log the results.
    
    Args:
        data_dir: Path to the data directory (default: 'source_data')
        
    Returns:
        bool: True if all checks pass, False otherwise
    """
    logger.info("%s Checking system dependencies..." % STATUS_ICONS['search'])
    
    all_ok, results = DependencyChecker.run_checks(data_dir)
    
    # Log results
    for check_name, (success, message) in results.items():
        status_icon = STATUS_ICONS['check' if success else 'cross']
        log_func = logger.info if success else logger.error
        log_func("%s %s: %s" % (
            status_icon,
            check_name.replace('_', ' ').title(),
            message
        ))
    
    if all_ok:
        logger.info("%s All dependency checks passed!" % STATUS_ICONS['check'])
    else:
        logger.error("%s Some dependency checks failed. Please fix the issues above." % 
                    STATUS_ICONS['cross'])
    
    return all_ok

if __name__ == "__main__":
    # Configure basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Run checks with default data directory
    success = check_dependencies()
    sys.exit(0 if success else 1)
