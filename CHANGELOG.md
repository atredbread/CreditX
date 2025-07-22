# Changelog

All notable changes to the Credit Risk Analysis System will be documented in this file.

## [Unreleased] - 2025-07-20

### Added
- Created `cleanup.py` script to manage temporary files and caches
- Added detailed debug logging for agent data processing
- Implemented JSON caching for agent data to improve performance
- Added organization name mapping from 'Unnamed: 6' column in credit_Agents.xlsx
- Enhanced error handling and validation in data loading and processing
- Added type hints and improved code documentation

### Changed
- Updated column reference from 'account' to 'Bzid' in `calculate_repayment_metrics`
- Improved type handling for agent ID comparisons
- Enhanced data validation in agent processing pipeline
- Updated data loading to handle missing or malformed data more gracefully

### Fixed
- Fixed missing NumPy import in data_processor.py
- Resolved agent data processing errors (675/681 agents now process successfully)
- Fixed column name mismatches in data loading
- Addressed type conversion issues in agent ID matching

### Documentation
- Created this CHANGELOG.md file
- Updated data structure documentation to reflect current schema
- Added inline documentation for new functions and changes
