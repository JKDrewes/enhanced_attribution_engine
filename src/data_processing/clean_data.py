"""
Cleans messy raw data for the Enhanced Attribution Engine.
- Pulls all CSVs from data/raw/
- Cleans structured, unstructured, and channel data
- Saves cleaned CSVs to data/processed/
"""

import sys
from pathlib import Path

# Add project root to sys.path to find bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bootstrap
from config.processing import config as proc_config
import pandas as pd
from dateutil import parser
from utils.logger import logger

# Use paths from bootstrap
RAW_DIR = bootstrap.RAW_DIR
PROCESSED_DIR = bootstrap.PROCESSED_DIR

# Helper functions
def clean_channel_name(channel):
    if pd.isna(channel):
        return None
    channel = str(channel).strip().lower()
    # Standardize common variations
    replacements = {
        "email": "Email",
        "paid search": "Paid Search",
        "organic": "Organic",
        "social media": "Social Media",
        "socail media": "Social Media",
        "display ad": "Display Ad",
        "display ad ": "Display Ad"
    }
    return replacements.get(channel, channel.title())


def parse_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    try:
        return pd.to_datetime(ts)
    except Exception:
        # Try dateutil parser as fallback
        try:
            return parser.parse(str(ts), dayfirst=True)
        except Exception:
            return pd.NaT


def clean_structured(df):
    # Strip whitespace for object columns (avoid deprecated applymap)
    for col in df.select_dtypes(include=["object"]).columns:
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
    # Fix channel capitalization
    df["channel"] = df["channel"].apply(clean_channel_name)
    # Parse timestamps
    df["timestamp"] = df["timestamp"].apply(parse_timestamp)
    # Remove duplicates
    df = df.drop_duplicates()
    # Handle missing user_id
    df = df[df["user_id"].notna()]
    # Fix outliers in revenue
    df["revenue"] = df["revenue"].apply(lambda x: max(x, 0))
    return df


def clean_reviews(df):
    # Strip whitespace for object columns (avoid deprecated applymap)
    for col in df.select_dtypes(include=["object"]).columns:
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
    # Parse timestamps
    df["timestamp"] = df["timestamp"].apply(parse_timestamp)
    # Remove duplicates
    df = df.drop_duplicates()
    # Fill missing ratings with median
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)
    # Remove rows without user_id
    df = df[df["user_id"].notna()]
    return df


def clean_channels(df):
    df["channel"] = df["channel"].apply(clean_channel_name)
    df["region"] = df["region"].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0)
    return df


# Main processing
def preprocess_all():
    logger.info("Starting preprocessing of raw data...")

    for file_path in RAW_DIR.glob("*.csv"):
        logger.info(f"Processing {file_path.name}...")
        df = pd.read_csv(file_path)

        # Apply sampling configuration (if any) before cleaning
        try:
            orig_len = len(df)
            df = proc_config.apply_sample(df)
            if len(df) < orig_len:
                logger.info(f"Applying sample of {len(df)} rows (from {orig_len}) to {file_path.name}")
        except Exception:
            # If config isn't available or fails, continue with full dataframe
            logger.debug("Processing config not applied; proceeding with full dataset.")

        if "touchpoint_id" in df.columns:
            df_clean = clean_structured(df)
        elif "text" in df.columns:
            df_clean = clean_reviews(df)
        elif "cost" in df.columns:
            df_clean = clean_channels(df)
        else:
            logger.warning(f"Unknown CSV format: {file_path.name}, skipping.")
            continue

        output_path = PROCESSED_DIR / file_path.name.replace(".csv", "_processed.csv")
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_all()
