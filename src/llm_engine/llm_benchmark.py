"""
llm_benchmark.py
Benchmarks the chosen LLM on sentiment analysis and intent detection tasks.
Reports execution time and output samples for verification.
"""

import sys
from pathlib import Path
# Ensure bootstrap is importable when running from this folder
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap

import pandas as pd
import time
from utils.logger import logger
from ollama import Ollama
from config.constants import active_model

# --- Paths ---
PROCESSED_DIR = Path("data/processed")
REVIEWS_FILE = PROCESSED_DIR / "reviews_processed.csv"
BENCHMARK_OUTPUT = PROCESSED_DIR / "llm_benchmark_results.csv"

# --- Load a sample of reviews ---
logger.info("Loading review sample for benchmarking...")
df_reviews = pd.read_csv(REVIEWS_FILE)
sample_size = min(50, len(df_reviews))  # limit to 50 samples for quick benchmark
df_sample = df_reviews.sample(sample_size, random_state=42).reset_index(drop=True)

# --- Initialize LLM client ---
logger.info("Initializing Ollama LLM...")
client = Ollama(model=active_model)

# --- Functions ---
def benchmark_sentiment(text):
    prompt = f"""
    Determine the sentiment of the following text as Negative, Neutral, or Positive.
    Respond only with one word: Negative, Neutral, or Positive.
    Text: \"\"\"{text}\"\"\"
    """
    start_time = time.time()
    try:
        response = client.chat(prompt)
        elapsed = time.time() - start_time
        sentiment = response.lower()
        if "positive" in sentiment:
            score = 1
        elif "negative" in sentiment:
            score = -1
        else:
            score = 0
        return score, elapsed
    except Exception as e:
        logger.warning(f"LLM error for sentiment: {e}")
        return 0, None

def benchmark_intent(text):
    prompt = f"""
    Extract the main intent, goal, or problem from the following text.
    Respond with a short phrase (1-5 words).
    Text: \"\"\"{text}\"\"\"
    """
    start_time = time.time()
    try:
        response = client.chat(prompt)
        elapsed = time.time() - start_time
        intent = response.strip().lower() if response else "unknown"
        return intent, elapsed
    except Exception as e:
        logger.warning(f"LLM error for intent: {e}")
        return "unknown", None

# --- Run benchmark ---
logger.info("Running LLM benchmark...")
results = []
for idx, row in df_sample.iterrows():
    text = row.get("text", "")
    sentiment, sent_time = benchmark_sentiment(text)
    intent, intent_time = benchmark_intent(text)
    results.append({
        "user_id": row.get("user_id"),
        "text": text,
        "sentiment_score": sentiment,
        "sentiment_time_sec": sent_time,
        "intent": intent,
        "intent_time_sec": intent_time
    })

df_results = pd.DataFrame(results)

# --- Summary metrics ---
avg_sent_time = df_results["sentiment_time_sec"].mean()
avg_intent_time = df_results["intent_time_sec"].mean()
logger.info(f"Average sentiment inference time: {avg_sent_time:.3f} sec")
logger.info(f"Average intent inference time: {avg_intent_time:.3f} sec")

# --- Save benchmark results ---
df_results.to_csv(BENCHMARK_OUTPUT, index=False)
logger.info(f"Benchmark results saved to {BENCHMARK_OUTPUT}")
