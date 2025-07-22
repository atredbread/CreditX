# Credit Risk Analysis System Architecture

## Core Components Overview

```mermaid
graph TD
    %% ===== INPUT LAYER =====
    subgraph InputData[Input Data Sources]
        A[Agent Details.xlsx] -->|Primary Agent Data| B[Data Loader]
        A2[credit_Agents.xlsx] -.->|Legacy Data| B
        C[Credit_sales_data.xlsx] -->|Transaction Records| B
        D[DPD.xlsx] -->|Days Past Due| B
        E[Region_contact.xlsx] -->|Regional Mappings| B
        P[Repayment_Report.csv] -->|Payment History| B
    end

    %% ===== PROCESSING LAYER =====
    subgraph Processing[Processing Layer]
        B -->|Validated Data| F[Data Validator]
        F -->|Clean Data| G[Feature Engineer]
        
        %% Repayment Score Calculation Flow
        G -->|Repayment Data| R[Repayment Analyzer]
        R -->|Calculates| S[Repayment Score]
        S -->|Feeds into| T[Agent Risk Profile]
        
        G -->|Engineered Features| H[Agent Classifier]
        T -->|Influences| H
        
        G -->|Trend Analysis| I[Trend Analyzer]
        G -->|Risk Metrics| J[Risk Calculator]
        S -->|Input for| J
    end

    %% ===== CORE SERVICES =====
    subgraph CoreServices[Core Services]
        K[Credit Health Engine] -->|Orchestrates| G
        K -->|Manages| H
        K -->|Coordinates| L[Report Generator]
        K -->|Monitors| S
    end

    %% ===== OUTPUT LAYER =====
    subgraph Output[Output Layer]
        L -->|Generates| M[Excel Reports]
        L -->|Sends| N[Email Summaries]
        L -->|Creates| O[Dashboard Visuals]
        S -->|Exported as| Q[Repayment_Score_Export.csv]
    end

    %% ===== STYLES =====
    classDef input fill:#e1f5fe,stroke:#0288d1
    classDef process fill:#e8f5e9,stroke:#388e3c
    classDef service fill:#f3e5f5,stroke:#8e24aa
    classDef output fill:#fff3e0,stroke:#f57c00
    classDef score fill:#ffebee,stroke:#c62828
    
    class A,A2,C,D,E,P input
    classDef legacy fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5
    class A2 legacy
    class F,G,H,I,J,R process
    class K,L service
    class M,N,O,Q output
    class S,T score
```

## Component Relationships

### 1. Input Data Layer
- **credit_Agents.xlsx**: Contains agent master data and profiles
- **Credit_sales_data.xlsx**: Transaction records and payment history
- **DPD.xlsx**: Days Past Due information for risk assessment
- **Region_contact.xlsx**: Regional mappings and contact information

### 2. Processing Layer
- **Data Loader**: Ingests and parses input files
- **Data Validator**: Ensures data quality and consistency
- **Feature Engineer**: Transforms raw data into meaningful features
  - Calculates metrics like repayment scores
  - Generates trend analyses
  - Creates risk indicators
- **Agent Classifier**: Categorizes agents based on risk profiles
- **Trend Analyzer**: Identifies patterns over time
- **Risk Calculator**: Computes risk metrics and scores

### 3. Core Services
- **Credit Health Engine**: The main orchestrator that:
  - Manages the data processing pipeline
  - Coordinates between components
  - Handles error recovery and logging
- **Report Generator**: Creates various output formats

### 4. Output Layer
- **Excel Reports**: Detailed analysis in spreadsheet format
- **Email Summaries**: Key findings and alerts
- **Dashboard Visuals**: Interactive data visualizations

## Data Flow

1. **Data Ingestion**:
   - Raw data files are loaded and validated
   - Data is cleaned and standardized

2. **Feature Engineering**:
   - Transaction data is aggregated
   - Time-series features are calculated
   - Risk indicators are computed

3. **Analysis & Classification**:
   - Agents are scored and categorized
   - Risk assessments are performed
   - Trends are identified

4. **Reporting & Output**:
   - Results are formatted for different outputs
   - Reports are generated and distributed
   - Alerts are triggered for critical findings

## Integration Points

- **Internal**:
  - Feature Engineering → Agent Classification
  - Data Validation → All processing components
  - Core Engine → All system components

- **External**:
  - Email service for report distribution
  - File system for data storage
  - (Optional) Database connections for persistent storage
