"""
Centralized project paths for datasets, processed data, outputs, and logs.
"""

import bootstrap

# --- Base directories ---
BASE_DIR = bootstrap.ROOT_DIR
DATA_DIR = bootstrap.DATA_DIR
RAW_DIR = bootstrap.RAW_DIR
PROCESSED_DIR = bootstrap.PROCESSED_DIR
OUTPUTS_DIR = bootstrap.OUTPUTS_DIR
REPORTS_DIR = OUTPUTS_DIR / "reporting"
LOGS_DIR = BASE_DIR / "logs"

# --- Data subfolders ---
RAW_FILE = RAW_DIR / "marketing_data_raw.csv"
PROCESSED_STRUCTURED = PROCESSED_DIR / "structured_data_processed.csv"
PROCESSED_REVIEWS = PROCESSED_DIR / "reviews_processed.csv"
PROCESSED_SENTIMENT = PROCESSED_DIR / "llm_sentiment_features.csv"
PROCESSED_INTENT = PROCESSED_DIR / "llm_intent_features.csv"
MODEL_INPUT_FEATURES = PROCESSED_DIR / "model_input_features.csv"

# --- Model outputs ---
LOGREG_PREDICTIONS = OUTPUTS_DIR / "conversion_predictions.csv"
SHAPLEY_ATTRIBUTION = OUTPUTS_DIR / "shapley_channel_attribution.csv"

# --- Reporting ---
VISUALIZATION_DIR = REPORTS_DIR
NARRATIVE_FILE = REPORTS_DIR / "model_narrative_summary.txt"

# --- Logs ---
PROJECT_LOG = LOGS_DIR / "project.log"
