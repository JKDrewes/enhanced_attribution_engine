"""
Uses an open-source LLM (via Ollama CLI) to extract user intents or goals
from reviews or textual interactions. Outputs structured intent features per user_id.

This implementation centers on the `extract_intent` function and uses the
configured model (Gemma) via the project's LLM config. It preserves the
existing pathing, logger and fallback behavior.
"""

from pathlib import Path
from typing import Optional
import subprocess
import shutil

import pandas as pd
from loguru import logger
try:
    import bootstrap
except Exception:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # Ensure project root and src are on sys.path so imports like `config.llm` work
    src_path = PROJECT_ROOT / "src"
    for p in (str(PROJECT_ROOT), str(src_path)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # attempt to import bootstrap (silently continue if not present)
    try:
        import bootstrap
    except Exception:
        pass

from config.llm import config as llm_config
from utils.paths import PROCESSED_DIR, PROCESSED_REVIEWS, PROCESSED_INTENT


# Paths
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
REVIEWS_FILE = PROCESSED_REVIEWS
OUTPUT_FILE = PROCESSED_INTENT


def extract_intent(text: str, model: Optional[str] = None) -> str:
    """Call the configured LLM to extract the user's main intent/goal.

    - Uses the model configured for "intent" in `config.llm`
    - Returns a short intent phrase (lowercased), or 'unknown' on error
    - Honors `llm_config.allow_fallback()` and `llm_config.fail_fast()`
    """
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"

    # Use configured intent model
    model = model or llm_config.get_model("intent")
    logger.debug(f"Using model {model} for intent extraction")

    prompt = (
        "Read the following user message and extract the user's main intent, goal, or problem.\n"
        "Respond with a short phrase (1-5 words), nothing else.\n"
        f'Text: "{text}"'
    )

    # Ensure Ollama is installed
    if not shutil.which("ollama"):
        logger.debug("Ollama CLI not found; using fallback for intent extraction")
        if llm_config.fail_fast():
            raise RuntimeError("Ollama CLI not found and fail_fast=True")
        return "unknown"

    # Build command (force run/chat mode for Gemma)
    try:
        cli_mode = llm_config.get_cli_mode(model)
        # If CLI mode wasn't correctly set to a run mode, override to 'run'
        if cli_mode not in ("run", "generate"):
            cli_mode = "run"

        cmd = llm_config.build_cli_command(model, prompt, mode=cli_mode)

        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

        out = result.stdout.strip()
        # Normalize and take first short phrase / first line
        if not out:
            return "unknown"
        # Some models echo additional metadata; prefer first non-empty line
        first_line = next((ln for ln in (l.strip() for l in out.splitlines()) if ln), "")
        intent = first_line.split("\n")[0].strip().lower()
        # Clean punctuation
        intent = intent.strip('\"\'')
        return intent if intent else "unknown"

    except subprocess.CalledProcessError as e:
        stderr = getattr(e, "stderr", "") or ""
        logger.warning(f"Ollama CLI failed (model={model}, exit {getattr(e,'returncode','N/A')}): {stderr}")
        if llm_config.fail_fast():
            raise
        if llm_config.allow_fallback():
            return "unknown"
        raise
    except Exception as e:
        logger.warning(f"Intent extraction error: {e}")
        if llm_config.fail_fast():
            raise
        return "unknown"


def main():
    logger.info(f"Loading processed text data from {REVIEWS_FILE}...")
    if not REVIEWS_FILE.exists():
        logger.error(f"Processed reviews not found: {REVIEWS_FILE}")
        return

    df = pd.read_csv(REVIEWS_FILE)
    if "text" not in df.columns:
        logger.error("Expected column 'text' not found in input data.")
        return

    logger.info("Extracting intents...")
    # Apply extraction (vectorized apply)
    df["intent"] = df["text"].apply(lambda t: extract_intent(str(t)))

    # One-hot encode top intents (top 20)
    logger.info("Encoding top intents as features...")
    top_intents = df["intent"].value_counts().nlargest(20).index.tolist()
    for intent in top_intents:
        safe_name = intent.replace(" ", "_").replace("/", "_")
        df[f"intent_{safe_name}"] = (df["intent"] == intent).astype(int)

    # Aggregate per user
    feature_cols = [c for c in df.columns if c.startswith("intent_")]
    if "user_id" not in df.columns:
        logger.error("No 'user_id' column found; cannot aggregate per user.")
        return

    logger.info("Aggregating intent features per user_id...")
    df_out = df.groupby("user_id")[feature_cols].mean().reset_index()

    # Ensure output parent exists
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving intent features to {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"LLM intent features saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

