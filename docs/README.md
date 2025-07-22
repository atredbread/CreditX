# Credit Risk Analysis Documentation

Welcome to the documentation for the Credit Risk Analysis system. This directory contains comprehensive documentation for understanding, using, and contributing to the Credit Health Intelligence Engine.

## 📚 Core Documentation

### System Overview

- [System Architecture](ARCHITECTURE.md) - High-level system design and components
- [Data Flow](DATA_FLOW.md) - How data moves through the system
- [Agent Classification](AGENT_CLASSIFICATION.md) - Detailed classification criteria and thresholds
- [Feature Engineering](FEATURE_ENGINEERING_DOCS.md) - Data processing and feature creation
- [Data Dictionary](DATA_DICTIONARY.md) - Comprehensive data schema and standards
- [Rule Book](rule_book.md) - Business rules and governance policies

### Development

- [API Reference](api/README.md) - Detailed API documentation
- [Development Setup](DEVELOPMENT.md) - Setting up the development environment
- [Testing Guide](TESTING.md) - Running tests and writing new tests
- [Deployment Guide](DEPLOYMENT.md) - Deployment procedures and requirements

### Data Management

- [Data Ingestion](data/INGESTION.md) - How to add new data sources
- [Data Validation](data/VALIDATION.md) - Data quality checks and validation rules
- [Data Retention](data/RETENTION.md) - Data retention policies and procedures

## 📊 Reports & Outputs

### Report Types

- **Agent Classification Reports**: Detailed agent risk assessments
- **Region-wise Analysis**: Performance metrics by region
- **Trend Reports**: Historical trends and forecasts
- **Data Quality Reports**: Validation results and data quality metrics

### Report Generation
Reports are automatically generated and stored in the following structure:

```text
output/
├── reports/
│   ├── daily/              # Daily automated reports
│   ├── monthly/            # Monthly summary reports
│   └── on_demand/          # Manually triggered reports
├── exports/
│   ├── csv/                # Data exports in CSV format
│   └── excel/              # Formatted Excel reports
└── logs/                   # System and processing logs
```

## 🔍 Data Dictionary

The system uses the following core data entities:

### Agent Details (`Agent_Details.xlsx`)
Contains comprehensive information about credit agents and their organizational relationships.

| Column | Type | Description |
|--------|------|-------------|
| account | int64 | Unique identifier for the agent account |
| agentname | string | Name of the agent (encrypted) |
| organizationname | string | Name of the organization (encrypted) |
| RoName | string | Region Officer name (encrypted) |
| RmName | string | Relationship Manager name (encrypted) |
| email | string | Contact email (encrypted) |
| mobile | string | Contact number (encrypted) |
| city | string | City code |
| cityname | string | Name of the city |
| state | string | State code |
| StateName | string | Name of the state |
| region | string | Region code |
| AgentRegion | string | Name of the agent's region |
| agenttype | string | Type of agent (e.g., BP_AGENT, OTHERS_NBP) |
| status | string | Account status (e.g., ACTIVE) |
| SO?TSE | string | Sales Officer/Territory Sales Executive |

### Other Core Entities

| Entity | Description | Key Fields |
|--------|-------------|------------|
| Transaction | Financial transactions | txn_id, agent_id, amount, date |
| CreditLine | Credit facilities | credit_id, agent_id, limit, terms |
| Payment | Payment records | payment_id, agent_id, amount, date |

> **Note:** The following sensitive fields are encrypted in the source data files to ensure data privacy and security:
> - `agentname`
> - `RoName`
> - `RmName`
> - `email`
> - `mobile`

## 📝 Additional Resources

- [Glossary](GLOSSARY.md) - Terminology and definitions
- [FAQ](FAQ.md) - Frequently asked questions
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Changelog](../CHANGELOG.md) - Version history and changes

---

## Last Updated

July 2023

For support or questions, please contact the Credit Risk Team at [credit-risk-support@example.com](mailto:credit-risk-support@example.com)
