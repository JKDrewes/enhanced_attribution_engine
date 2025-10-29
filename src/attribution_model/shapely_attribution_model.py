"""
Calculates Shapley-value based attribution for marketing channels using
predicted conversion probabilities from logistic regression.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import logger
import bootstrap

# Paths
PROCESSED_DIR = bootstrap.PROCESSED_DIR
OUTPUT_DIR = bootstrap.OUTPUTS_DIR

STRUCTURED_FILE = PROCESSED_DIR / "structured_data_processed.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "conversion_predictions.csv"

# Load Data
logger.info(f"Loading structured data from {STRUCTURED_FILE}...")
df = pd.read_csv(STRUCTURED_FILE, parse_dates=["timestamp"])

logger.info(f"Loading logistic regression predictions from {PREDICTIONS_FILE}...")
df_preds = pd.read_csv(PREDICTIONS_FILE)

# Merge predicted probabilities
df = df.merge(df_preds[["user_id", "y_pred"]], on="user_id", how="left")
df["y_pred"] = df["y_pred"].fillna(0)  # Fill any missing predictions with 0

# Aggregate user paths
df = df.sort_values(by=["user_id", "timestamp"])
user_paths = df.groupby("user_id")["channel"].apply(list)
user_predictions = df.groupby("user_id")["y_pred"].max()  # use max predicted conversion probability per user

#  Shapley Attribution Function
def shapley_attribution(paths, values):
    """
    Compute Shapley-style attribution per channel using predicted values.
    paths: dict {user_id: [channels]}
    values: dict {user_id: predicted conversion probability}
    """
    channels = set(ch for path in paths.values() for ch in path)
    channel_values = {ch: 0.0 for ch in channels}
    channel_counts = {ch: 0 for ch in channels}
    
    for user, path in paths.items():
        value = values.get(user, 0.0)
        path_len = len(path)
        if path_len == 0:
            continue
            
        # Split predicted conversion among channels
        value_per_touch = value / path_len
        for ch in path:
            channel_values[ch] += value_per_touch
            channel_counts[ch] += 1
    
    # Calculate average contribution per channel
    for ch in channel_values:
        if channel_counts[ch] > 0:
            channel_values[ch] /= channel_counts[ch]
    
    # Normalize to percentages
    total_value = sum(channel_values.values())
    if total_value > 0:
        for ch in channel_values:
            channel_values[ch] = (channel_values[ch] / total_value) * 100
    
    return channel_values

# Run Attribution
logger.info("Calculating Shapley-value attributions using predicted conversions...")
paths_dict = user_paths.to_dict()
values_dict = user_predictions.to_dict()
shapley_values = shapley_attribution(paths_dict, values_dict)

# Save Results
output_path = OUTPUT_DIR / "shapley_channel_attribution.csv"
df_shapley = pd.DataFrame(list(shapley_values.items()), columns=["channel", "attribution_percentage"])
df_shapley = df_shapley.sort_values("attribution_percentage", ascending=False)

# Add summary stats
df_shapley["total_touchpoints"] = df_shapley["channel"].map(
    df.groupby("channel").size()
)
df_shapley["avg_pred_value"] = df_shapley["channel"].map(
    df.groupby("channel")["y_pred"].mean()
)

df_shapley.to_csv(output_path, index=False)
logger.info(f"Shapley attributions saved to {output_path}")

# Print top channels with more detail
logger.info("\nTop channels by Shapley attribution:")
logger.info(df_shapley.to_string(float_format=lambda x: "{:.2f}".format(x)))
