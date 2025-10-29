"""
Merges LLM-derived features (sentiment, intent) with structured marketing data
into a single dataset ready for modeling.
"""

import sys
from pathlib import Path
# Ensure bootstrap is importable when running script from its folder
try:
	import bootstrap
except Exception:
	PROJECT_ROOT = Path(__file__).resolve().parents[2]
	if str(PROJECT_ROOT) not in sys.path:
		sys.path.insert(0, str(PROJECT_ROOT))
	import bootstrap

import pandas as pd
from utils.logger import logger

# --- Paths ---
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "model_input_features.csv"

STRUCTURED_FILE = PROCESSED_DIR / "structured_data_processed.csv"
SENTIMENT_FILE = PROCESSED_DIR / "llm_sentiment_features.csv"
INTENT_FILE = PROCESSED_DIR / "llm_intent_features.csv"

# --- Load data ---
logger.info("Loading structured data...")
df_structured = pd.read_csv(STRUCTURED_FILE)

logger.info("Loading LLM sentiment features...")
df_sentiment = pd.read_csv(SENTIMENT_FILE)

logger.info("Loading LLM intent features...")
df_intent = pd.read_csv(INTENT_FILE)

# --- Merge features ---
logger.info("Merging features into single dataframe...")
df_features = df_structured.merge(df_sentiment, on="user_id", how="left")
df_features = df_features.merge(df_intent, on="user_id", how="left")

# Fill missing sentiment or intent features
df_features["sentiment_score"] = df_features["sentiment_score"].fillna(0)
intent_cols = [col for col in df_features.columns if col.startswith("intent_")]
df_features[intent_cols] = df_features[intent_cols].fillna(0)

# --- Save merged features ---
logger.info(f"Saving merged features to {OUTPUT_FILE}...")
df_features.to_csv(OUTPUT_FILE, index=False)
logger.info("Feature export complete. Dataset ready for modeling.")
