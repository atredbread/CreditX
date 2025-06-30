# Credit Health Intelligence Engine

An automated backend system for classifying and monitoring credit agents using structured behavioral logic.

## 🚀 Features

- **Automated Agent Classification**: Segments agents into P0-P5 tiers based on credit behavior
- **Region-wise Reporting**: Generates detailed Excel reports for each region
- **Email-ready Summaries**: Creates formatted insights for easy communication
- **Comprehensive Metrics**: Tracks credit utilization, repayment scores, and risk factors
- **GMV Trend Analysis**: Analyzes both total sales and credit GMV trends
- **Data Quality Checks**: Validates input data and identifies potential issues

## 🛠️ Project Structure

```
Credit Risk/
├── data/                    # Input data files
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed data files
├── output/                  # Generated reports and outputs
│   ├── region_reports/      # Region-wise Excel reports
│   └── email_summaries/     # Formatted email summaries
├── src/                     # Source code
│   ├── agent_classifier.py  # Agent classification logic
│   ├── credit_health_engine.py  # Main engine
│   ├── data_dictionary.py   # Data schema and validation
│   ├── data_quality_checks.py  # Data validation
│   └── feature_engineering.py  # Feature calculation
├── tests/                   # Test files
│   ├── fixtures/            # Test data fixtures
│   └── test_*.py            # Test scripts
├── tools/                   # Utility scripts
│   ├── analyze_data.py      # Data analysis tools
│   ├── check_dpd_*.py       # DPD analysis tools
│   └── visualize_*.py       # Visualization tools
├── README.md                # This file
└── setup.py                 # Package configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Credit-Risk
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Configuration

1. Create a `.env` file in the project root with the following variables:
   ```
   # Data paths
   DATA_DIR=./data/raw
   OUTPUT_DIR=./output
   
   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=logs/credit_engine.log
   ```

2. Place your input files in the `data/raw/` directory:
   - `credit_agents.csv`
   - `credit_history_sales_vs_credit_sales.csv`
   - `credit_sales_data.csv`
   - `region_contact.csv`
   - `dpd.csv`
   - `sales_data.csv`

## 🏃‍♂️ Running the Analysis

1. Run the main analysis pipeline:
   ```bash
   python -m src.credit_health_engine
   ```

2. Generate reports for a specific region:
   ```bash
   python -m src.credit_health_engine --region "North"
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## 📊 Outputs

The system generates the following outputs in the `output/` directory:

- `classifications/agent_classifications.csv`: Agent tier classifications
- `region_reports/`: Excel reports for each region
- `email_summaries/`: Formatted email content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
    return f"Processed: {prompt}"
```

### Log File Location
- Logs are stored in: `logs/prompt_history.log`
- Logs include timestamps, prompt text, and metadata

## 📋 Prerequisites

- Python 3.8+
- Required Python packages (install using `pip install -r requirements.txt`)

## 🛠️ Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create an `input` directory and add the required input files:
   - `credit_agents.csv`
   - `credit_history_sales_vs_credit_sales.csv`
   - `credit_sales_data.csv`
   - `region_contact.csv`
   - `dpd.csv`
   - `sales_data.csv`

## 🚦 Running the Engine

1. Place all input files in the `input` directory
2. Run the main script:
   ```
   python src/credit_health_engine.py
   ```
3. Find the generated reports in the `output` directory:
   - `output/region_reports/`: Excel files for each region
   - `output/email_summaries/`: Formatted email content

## 📊 Output Structure

- **Region Reports**: Detailed Excel files with agent metrics and classifications
- **Email Summaries**: Markdown files with key insights and action items

## 📝 License

This project is proprietary and confidential.

---

For support or questions, please contact your system administrator.
