# Credit Analyzer CLI Interface Specification

## Overview
The Credit Analyzer CLI provides an interactive interface for credit risk analysis and reporting. It offers two main modes of operation (Manual and Auto) along with utility functions for cleaning up output files.

## Main Menu Options

### 1. Manual Mode
Provides full control and transparency over each module for detailed, step-by-step execution and diagnostics.

#### Functionality:
- **Load Data**
  - Loads raw data from predefined source formats (CSV, database, or API)
  - Uses only pre-approved input methods from existing modules
  - Validates data integrity before processing

- **Clean and Analyze**
  - Performs data cleaning and analysis using existing module logic
  - Outputs structured JSON format
  - Follows strict transformation rules from project documentation

- **Unique Agent Lookup**
  - Performs agent matching using existing identification logic
  - Displays results in clean, tabular CLI format
  - Shows detailed agent information and credit metrics

- **Regional Top & Bottom Agent Analysis**
  - Generates Top 10 and Bottom 10 agents by region
  - Uses verified scoring/GMV-based logic
  - Outputs to both CLI and CSV files
  - Follows regional performance metrics from existing modules

### 2. Auto Mode
Executes a simplified flow with minimal user input for standard runs and quick reporting.

#### Functionality:
- Performs automatic agent lookup with default parameters
- Generates regional Top 10/Bottom 10 agent reports
- Saves outputs to designated folders
- Uses same core functions as Manual Mode with pre-configured settings

### 3. Clear Output
- Removes all generated output files from the current session
- Requires confirmation before deletion
- Only affects designated output directories
- Preserves all source data and configuration files

## Implementation Rules

### Module Usage
- Must use only existing Python modules and helper functions
- No new functions should be created without approval
- All transformations must follow documented processes

### Documentation Compliance
- Strict adherence to specifications in `/docs` directory
- Follow all rules in `docs/rule book.md`
- No deviations from documented processes

### Error Handling
- All errors must be caught and displayed clearly
- Operations should fail gracefully with helpful messages
- Maintain operation logs for debugging

### Output Management
- All generated files go to designated output folders
- Clear file naming conventions
- No modification of source data files

## Navigation
- Clear menu hierarchy
- Consistent return/exit options
- Progress indicators for long-running operations
- Confirmation prompts for destructive actions
