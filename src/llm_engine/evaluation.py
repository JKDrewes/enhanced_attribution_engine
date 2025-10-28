"""
evaluation.py
Evaluates LLM-derived features (sentiment and intent) for coverage, distribution,
and general insights.
"""

# Ensure bootstrap is importable when running directly from this folder
import sys
from pathlib import Path
try:
	import bootstrap
except Exception:
	PROJECT_ROOT = Path(__file__).resolve().parents[2]
	if str(PROJECT_ROOT) not in sys.path:
		sys.path.insert(0, str(PROJECT_ROOT))
	import bootstrap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.logger import logger

# --- Paths ---
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SENTIMENT_FILE = PROCESSED_DIR / "llm_sentiment_features.csv"
INTENT_FILE = PROCESSED_DIR / "llm_intent_features.csv"

# --- Load LLM features ---
logger.info("Loading sentiment features...")
df_sentiment = pd.read_csv(SENTIMENT_FILE)

logger.info("Loading intent features...")
df_intent = pd.read_csv(INTENT_FILE)

# --- Sentiment evaluation ---
logger.info("Evaluating sentiment feature coverage and distribution...")
total_users = len(df_sentiment)
missing_sentiment = df_sentiment["sentiment_score"].isna().sum()
logger.info(f"Total users: {total_users}, Missing sentiment: {missing_sentiment}")

# Summary stats
sentiment_stats = df_sentiment["sentiment_score"].describe()
logger.info(f"Sentiment summary:\n{sentiment_stats}")

# Histogram of sentiment
plt.figure(figsize=(6, 4))
sns.histplot(df_sentiment["sentiment_score"], bins=20, kde=True)
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png")
plt.close()
logger.info(f"Sentiment distribution plot saved to {OUTPUT_DIR / 'sentiment_distribution.png'}")

# --- Intent evaluation ---
logger.info("Evaluating intent features...")
intent_cols = [col for col in df_intent.columns if col.startswith("intent_")]

# Coverage
intent_missing = df_intent[intent_cols].isna().sum().sum()
logger.info(f"Total missing intent values: {intent_missing}")

# Prevalence of top intents
intent_sums = df_intent[intent_cols].sum().sort_values(ascending=False)
logger.info(f"Top intents by occurrence:\n{intent_sums.head(10)}")

# Plot top 10 intents
plt.figure(figsize=(8, 5))
sns.barplot(x=intent_sums.head(10).values, y=intent_sums.head(10).index)
plt.title("Top 10 Intent Features")
plt.xlabel("Number of Users")
plt.ylabel("Intent")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_intents.png")
plt.close()
logger.info(f"Top intent features plot saved to {OUTPUT_DIR / 'top_intents.png'}")

logger.info("LLM feature evaluation complete.")
