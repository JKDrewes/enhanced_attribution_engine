"""
Benchmarks the chosen LLM on sentiment analysis and intent detection tasks.
Reports execution time and output samples for verification.
"""

import sys
from pathlib import Path
import pandas as pd
import time
import json

# Ensure bootstrap is importable when running from this folder
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap

from utils.logger import logger
import shutil
import subprocess
from config.llm import config as llm_config

# Use processed dir from bootstrap for consistent paths
PROCESSED_DIR = bootstrap.PROCESSED_DIR
REVIEWS_FILE = PROCESSED_DIR / "reviews_processed.csv"
BENCHMARK_OUTPUT = PROCESSED_DIR / "llm_benchmark_results.csv"

# Optional Ollama client import — some environments may use the CLI instead
try:
    from ollama import Ollama  # type: ignore
except Exception:
    Ollama = None  # type: ignore

# No direct dependency on config.constants (model selection now via config.llm)


def _normalize_response(resp) -> str:
    """Convert various response types to a plain string."""
    if resp is None:
        return ""
    # Common shapes: string, dict with 'text'/'content'/'message'
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("text", "content", "message"):
            if key in resp:
                return str(resp[key])
        # If it's a nested structure, stringify fallback
        return json.dumps(resp)
    return str(resp)


def benchmark_sentiment(client, text: str):
    prompt = (
        "Determine the sentiment of the following text as Negative, Neutral, or Positive.\n"
        "Respond only with one word: Negative, Neutral, or Positive.\n\n"
        "Text:\n"
        f"{text}\n"
    )
    start_time = time.time()
    try:
        resp = client.chat(prompt) if client is not None else None
        elapsed = time.time() - start_time
        sentiment = _normalize_response(resp).lower()
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


def benchmark_intent(client, text: str):
    prompt = (
        "Extract the main intent, goal, or problem from the following text.\n"
        "Respond with a short phrase (1-5 words).\n\n"
        "Text:\n"
        f"{text}\n"
    )
    start_time = time.time()
    try:
        resp = client.chat(prompt) if client is not None else None
        elapsed = time.time() - start_time
        intent = _normalize_response(resp).strip().lower() if resp else "unknown"
        return intent, elapsed
    except Exception as e:
        logger.warning(f"LLM error for intent: {e}")
        return "unknown", None


def main():
    # Load a sample of reviews — be resilient to missing files
    logger.info("Loading review sample for benchmarking...")
    if not REVIEWS_FILE.exists():
        logger.warning(f"Reviews file not found: {REVIEWS_FILE}. Aborting benchmark.")
        return

    df_reviews = pd.read_csv(REVIEWS_FILE)
    sample_size = min(50, len(df_reviews))  # limit to 50 samples for quick benchmark
    df_sample = df_reviews.sample(sample_size, random_state=42).reset_index(drop=True)

    # Determine whether Ollama CLI is available and pick models from config
    has_ollama = shutil.which("ollama") is not None
    if not has_ollama:
        if llm_config.fail_fast():
            raise RuntimeError("Ollama CLI not found and fail_fast=True")
        logger.info("Ollama CLI not found — benchmark will use local fallbacks or empty outputs")

    sentiment_model = llm_config.get_model("sentiment")
    intent_model = llm_config.get_model("intent")

    # Run benchmark
    logger.info("Running LLM benchmark...")
    results = []
    for idx, row in df_sample.iterrows():
        text = row.get("text", "")

        # Sentiment: try CLI if available
        if has_ollama:
            cmd = llm_config.build_cli_command(sentiment_model, f"Determine sentiment:\n{text}")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30, encoding="utf-8", errors="replace")
                sent_out = _normalize_response(res.stdout).strip().lower()
                if "positive" in sent_out:
                    sentiment = 1
                elif "negative" in sent_out:
                    sentiment = -1
                else:
                    sentiment = 0
                sent_time = None  # CLI-based timing not measured per-call here
            except Exception as e:
                logger.warning(f"Ollama CLI error for sentiment: {e}")
                sentiment = 0
                sent_time = None
        else:
            # Local fallback: use existing benchmark_sentiment implementation with no client
            sentiment, sent_time = benchmark_sentiment(None, text)

        # Intent: try CLI if available
        if has_ollama:
            cmd = llm_config.build_cli_command(intent_model, f"Extract intent:\n{text}")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30, encoding="utf-8", errors="replace")
                intent = _normalize_response(res.stdout).strip().lower()
                intent_time = None
            except Exception as e:
                logger.warning(f"Ollama CLI error for intent: {e}")
                intent = "unknown"
                intent_time = None
        else:
            intent, intent_time = benchmark_intent(None, text)
        results.append({
            "user_id": row.get("user_id"),
            "text": text,
            "sentiment_score": sentiment,
            "sentiment_time_sec": sent_time,
            "intent": intent,
            "intent_time_sec": intent_time,
        })

    df_results = pd.DataFrame(results)

    # Summary metrics
    avg_sent_time = df_results["sentiment_time_sec"].dropna().mean()
    avg_intent_time = df_results["intent_time_sec"].dropna().mean()
    logger.info(f"Average sentiment inference time: {avg_sent_time:.3f} sec")
    logger.info(f"Average intent inference time: {avg_intent_time:.3f} sec")

    # Save results
    BENCHMARK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(BENCHMARK_OUTPUT, index=False)
    logger.info(f"Benchmark results saved to {BENCHMARK_OUTPUT}")


if __name__ == "__main__":
    main()
