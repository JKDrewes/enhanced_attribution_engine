# Enhanced Attribution Engine

## Overview
The **Enhanced Attribution Engine** is a hybrid analytics system that combines data-driven attribution models with insights derived from large language models (LLMs).  

It extends traditional marketing attribution by incorporating qualitative, unstructured data—such as customer reviews, chat logs, or social media mentions—into the model to improve both accuracy and interpretability.

This project **does not use an orchestration framework** (like Dagster or Airflow); instead, it relies on a modular Python pipeline with lightweight configuration and manual scheduling via cron or CLI.

---

## Quick Start

```powershell
# 1. Initialize project structure
python init_project.py

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python src/main.py
```

## Project Structure

```
enhanced_attribution_engine/
├── config/             # Configuration files
├── data/               # Data files
│   ├── raw/            # Original data
│   ├── processed/      # Cleaned data
│   └── outputs/        # Model outputs
│       ├── reporting/
│       │   ├── figures/      # Visualization outputs
│       │   └── full_report/  # Final PDF reports
│       └── evaluation/       # Model evaluation results
├── logs/             # Log files
├── src/              # Source code
│   ├── attribution_model/
│   ├── data_processing/
│   ├── llm_engine/
│   ├── reporting/
│   └── utils/
└── tests/            # Unit tests
```

## Architecture Overview

### 1. Data Processing Layer
- Cleans and standardizes both **structured data** (ad impressions, conversions) and **unstructured data** (reviews, chat logs)
- Saves processed datasets to `/data/processed`

### 2. LLM-Powered Insights Engine
- Uses an LLM to extract **sentiment** and **intent** from unstructured text
- Outputs structured features for model integration

### 3. Attribution Model
- Implements a **gradient boosting model** for conversion prediction
- Calculates **Shapley values** for attribution analysis

### 4. Integration & Analysis
- Combines LLM-derived insights with attribution weights
- Evaluates model performance and feature importance

### 5. Reporting & Visualization
- Generates interactive visualizations and performance metrics
- Creates comprehensive PDF reports with analysis and insights

## Pipeline Workflow
1. Run data processing (`clean_data.py`)
2. Generate LLM features:
   - Sentiment analysis
   - Intent detection
   - Feature integration
3. Train gradient boosting model
4. Calculate Shapley attributions
5. Generate evaluations and visualizations
6. Create final report

## Running Scripts

All scripts can be run directly with Python from the project root:

```powershell
python src/main.py  # Run complete pipeline
```

Key entry points:
- `src/main.py` - Complete pipeline orchestration
- `src/data_processing/clean_data.py` - Data preprocessing
- `src/llm_engine/sentiment_analysis.py` - Review sentiment scoring
- `src/llm_engine/intent_detection.py` - User intent classification
- `src/attribution_model/grade_boost_model.py` - Model training
- `src/reporting/generate_report.py` - Final report generation

## Configuration

### Data Processing
Set sample size in `config/processing.py`:
```powershell
python config/processing.py 1000  # Process 1000 records
python config/processing.py NONE  # Process full dataset
```

### Development Tools
```powershell
python -m pytest              # Run tests
black src/                   # Format code
flake8 src/                 # Check style
```

## LLM configuration and model selection

The project centralizes LLM settings in `config/llm.py` and `config/llm_config.json`. You can optionally select which model is used for different tasks (for example: `default`, `sentiment`, `intent`, and `report_recommendation`). By default the repo ships with `gemma3:4b` for general/report recommendations and sentiment, and `llama2:latest` for intent detection.

How it works:
- `config/llm.py` exposes a `config` object with `get_model(feature_type)` and `build_cli_command(model, prompt)` helpers. Code calls `config.get_model()` to select the model for a given task.
- `config/llm_config.json` provides the canonical mapping of feature types to model names. The keys currently supported are: `default`, `report_recommendation`, `sentiment`, and `intent`.
- Environment variables can override behavior at runtime (PowerShell examples below).

Examples
- Change the model used for report recommendations only (temporary for current shell):

```powershell
$env:LLM_MODEL = "gemma3:4b"            # sets default model
$env:LLM_MODEL = "": ""               # unset if needed
```

- If you want to override a specific key without editing the JSON, you can call functions in code with `model_override` parameters where available. For example, `generate_recommendations(..., model_override="gemma3:4b")` will force that model for the recommendation prompt.

Other environment variables:
- `LLM_CLI_MODE` — force the CLI mode used for models (e.g., `run` or `generate`).
- `FORCE_LLM_FALLBACK` — if set to true, the code will prefer local fallback scorers instead of requiring the Ollama CLI.
- `LLM_FAIL_FAST` — if true, the pipeline will raise on LLM errors instead of falling back.

If you prefer editing the repo-level defaults, modify `config/llm_config.json` to change the model names permanently for your environment.