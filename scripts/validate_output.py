"""
Output Validation Script for Credit Risk Analysis

This script validates the structure and content of the output JSON file
to ensure it meets the project's requirements and specifications.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)

class OutputValidator:
    """Validates the output JSON file against project requirements."""
    
    def __init__(self, file_path: Path):
        """Initialize the validator with the path to the output JSON file."""
        self.file_path = file_path
        self.data = None
        self.is_valid = False
        self.issues = []
    
    def load_data(self) -> bool:
        """Load and parse the JSON data."""
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.issues.append(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.issues.append(f"Error reading file: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate the basic structure of the JSON data."""
        if not self.data:
            self.issues.append("No data loaded")
            return False
        
        required_keys = ['metadata', 'agents']
        missing_keys = [k for k in required_keys if k not in self.data]
        
        if missing_keys:
            self.issues.append(f"Missing required top-level keys: {', '.join(missing_keys)}")
            return False
        
        return True
    
    def validate_metadata(self) -> bool:
        """Validate the metadata section of the output."""
        metadata = self.data.get('metadata', {})
        required_metadata = [
            'generated_at', 'processing_time_seconds', 'agent_count',
            'success_count', 'error_count', 'version'
        ]
        
        missing_metadata = [k for k in required_metadata if k not in metadata]
        if missing_metadata:
            self.issues.append(f"Missing required metadata: {', '.join(missing_metadata)}")
            return False
        
        # Check timestamp is recent (within 24 hours)
        try:
            file_time = datetime.fromisoformat(metadata['generated_at'].replace('Z', '+00:00'))
            if (datetime.now() - file_time) > timedelta(hours=24):
                self.issues.append("Output file is older than 24 hours")
                return False
        except Exception as e:
            self.issues.append(f"Invalid timestamp format: {e}")
            return False
        
        return True
    
    def validate_agent_data(self) -> bool:
        """Validate the structure of agent data."""
        agents = self.data.get('agents', {})
        
        if not agents:
            self.issues.append("No agent data found")
            return False
        
        # Check a sample of agents
        sample_size = min(5, len(agents))
        sample_agents = list(agents.items())[:sample_size]
        
        required_agent_fields = [
            'basic_info', 'credit_metrics', 'repayment_metrics',
            'risk_score', 'risk_level', 'last_updated'
        ]
        
        for agent_id, agent_data in sample_agents:
            missing_fields = [f for f in required_agent_fields if f not in agent_data]
            if missing_fields:
                self.issues.append(
                    f"Agent {agent_id} missing fields: {', '.join(missing_fields)}"
                )
                return False
        
        return True
    
    def validate(self) -> bool:
        """Run all validation checks."""
        if not self.file_path.exists():
            self.issues.append(f"File not found: {self.file_path}")
            return False
        
        if not self.load_data():
            return False
        
        checks = [
            ("Structure", self.validate_structure),
            ("Metadata", self.validate_metadata),
            ("Agent Data", self.validate_agent_data)
        ]
        
        self.is_valid = all(
            self._run_check(name, check_fn)
            for name, check_fn in checks
        )
        
        return self.is_valid
    
    def _run_check(self, name: str, check_fn) -> bool:
        """Run a single validation check and log the result."""
        try:
            result = check_fn()
            status = "PASSED" if result else "FAILED"
            logger.info(f"[{status}] {name} validation")
            return result
        except Exception as e:
            self.issues.append(f"Error during {name.lower()} validation: {e}")
            logger.error(f"[ERROR] {name} validation: {e}", exc_info=True)
            return False
    
    def get_report(self) -> str:
        """Generate a validation report."""
        status = "VALID" if self.is_valid else "INVALID"
        report = [
            "=" * 80,
            f"VALIDATION REPORT: {status}",
            f"File: {self.file_path}",
            f"Time: {datetime.now().isoformat()}",
            "=" * 80,
            ""
        ]
        
        if self.issues:
            report.append("ISSUES FOUND:")
            for i, issue in enumerate(self.issues, 1):
                report.append(f"  {i}. {issue}")
        else:
            report.append("No issues found.")
        
        if self.data and 'metadata' in self.data:
            report.extend([
                "",
                "METADATA:",
                f"  Generated at: {self.data['metadata'].get('generated_at')}",
                f"  Agent count: {self.data['metadata'].get('agent_count')}",
                f"  Success count: {self.data['metadata'].get('success_count')}",
                f"  Error count: {self.data['metadata'].get('error_count')}",
                f"  Version: {self.data['metadata'].get('version')}"
            ])
        
        report.extend([
            "",
            "=" * 80,
            "Validation complete."
        ])
        
        return "\n".join(report)

def main() -> int:
    """Main entry point for the validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Credit Risk Analysis output')
    parser.add_argument(
        '--file', '-f',
        type=Path,
        default=project_root / 'output' / 'agent_analysis.json',
        help='Path to the output JSON file to validate'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    validator = OutputValidator(args.file)
    is_valid = validator.validate()
    
    # Print the validation report
    print(validator.get_report())
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())
