"""
sentiment_analysis.py
Generates sentiment scores for text data using Ollama CLI.
"""

import sys
from pathlib import Path
# Ensure bootstrap is importable when running this script directly
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap

import shutil
import pandas as pd
import subprocess
from typing import Optional
from utils.logger import logger
from config.llm import config as llm_config
from utils.paths import PROCESSED_DIR, PROCESSED_SENTIMENT, PROCESSED_REVIEWS


# Ensure output directory exists
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Input / Output files (use centralized paths)
REVIEWS_FILE = PROCESSED_REVIEWS
OUTPUT_FILE = PROCESSED_SENTIMENT

# -------------------------------
# Ollama CLI helper function
# -------------------------------
def _fallback_sentiment(text: str) -> int:
    """Lightweight keyword-based fallback sentiment scorer.

    Returns: 1 for positive, -1 for negative, 0 for neutral/unknown.
    This is used when Ollama CLI is not available to avoid hard failures.
    """
    text_l = text.lower()
    positive_words = ("good", "great", "excellent", "love", "happy", "awesome", "pleased")
    negative_words = ("bad", "terrible", "hate", "awful", "disappointed", "poor")
    pos = any(w in text_l for w in positive_words)
    neg = any(w in text_l for w in negative_words)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def call_ollama_sentiment(text: str, model: Optional[str] = None) -> int:
    """Calls Ollama CLI to get sentiment, falling back to a local scorer when necessary.

    Args:
        text: Text to analyze
        model: Optional model override. If None, uses the model configured for 'sentiment'
            in config.llm.

    Returns:
        1 for positive, 0 for neutral, -1 for negative sentiment
    """
    if not shutil.which("ollama"):
        logger.debug("Ollama executable not found — using fallback sentiment scorer.")
        if llm_config.fail_fast():
            raise RuntimeError("Ollama not found and fail_fast=True")
        return _fallback_sentiment(text)

    # Get the model name and build the CLI command
    model = model or llm_config.get_model("sentiment")
    logger.debug(f"Using model {model} for sentiment analysis")
    
    prompt = f"""
Determine the sentiment of the following text as Negative, Neutral, or Positive.
Respond with only one word: Negative, Neutral, or Positive.
Text: \"\"\"{text}\"\"\"
"""

    try:
        result = subprocess.run(
            llm_config.build_cli_command(model, prompt),
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
        response = result.stdout.strip().lower()
        if "positive" in response:
            return 1
        elif "negative" in response:
            return -1
        else:
            return 0
    except subprocess.CalledProcessError as e:
        stderr = getattr(e, "stderr", "") or ""
        logger.warning(f"Ollama CLI failed (model={model}, exit {getattr(e, 'returncode', 'N/A')}): {stderr}")
        if llm_config.fail_fast():
            raise
        elif llm_config.allow_fallback():
            logger.debug("Using fallback sentiment scorer")
            return _fallback_sentiment(text)
        else:
            return 0  # neutral when fallback disallowed

# -------------------------------
# Main function
# -------------------------------
def main():
    logger.info("Loading processed text data...")
    if not REVIEWS_FILE.exists():
        logger.error(f"Raw processed file not found: {REVIEWS_FILE}")
        return

    df = pd.read_csv(REVIEWS_FILE)

    if "text" not in df.columns:
        logger.error("Expected column 'text' not found in input data.")
        return

    logger.info("Running sentiment analysis...")
    try:
        df["sentiment_score"] = df["text"].apply(lambda x: call_ollama_sentiment(str(x)))
    except Exception as e:
        logger.exception("Failed while applying sentiment scoring: {}", e)
        return

    out_cols = [c for c in ("user_id", "sentiment_score") if c in df.columns]
    if not out_cols:
        logger.error("No output columns available to save (need 'user_id' or 'sentiment_score').")
        return

    logger.info(f"Saving sentiment features to {OUTPUT_FILE}")
    df[out_cols].to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
