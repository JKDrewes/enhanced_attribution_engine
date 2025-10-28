"""
evaluation.py
Evaluates Logistic Regression and Shapley attribution models.
Provides model assumptions checks, metrics, and visualizations.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from src.utils.logger import logger
import joblib

# --- Paths ---
OUTPUT_DIR = bootstrap.OUTPUTS_DIR
MODEL_FILE = OUTPUT_DIR / "conversion_model.pkl"
PREDICTIONS_FILE = OUTPUT_DIR / "conversion_predictions.csv"
SHAPLEY_FILE = OUTPUT_DIR / "shapley_channel_attribution.csv"
EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True, parents=True)

# --- Load model and predictions ---
logger.info("Loading Logistic Regression model and predictions...")
model = joblib.load(MODEL_FILE)
df_preds = pd.read_csv(PREDICTIONS_FILE)

y_true = df_preds["y_true"]
y_prob = df_preds["y_pred"]  # Already have probabilities from logistic regression
y_pred = (y_prob >= 0.5).astype(int)

# Get feature importances from logistic regression coefficients
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coef = model.named_steps['classifier'].coef_[0]
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(coef)
}).sort_values('importance', ascending=False)

# Check for multicollinearity using correlation matrix
numeric_features = [col for col in df_preds.columns if col.startswith(('sentiment_', 'intent_'))]
correlation_matrix = df_preds[numeric_features].corr()
high_correlation = np.where(np.abs(correlation_matrix) > 0.7)
correlated_features = [(numeric_features[i], numeric_features[j], correlation_matrix.iloc[i, j])
                      for i, j in zip(*high_correlation) if i < j]

# --- Model Assumptions and Diagnostics ---
logger.info("\nChecking Logistic Regression Assumptions:")

# 1. Sample Size
n_samples = len(y_true)
n_features = len(feature_names)
events_per_var = min(sum(y_true), sum(1-y_true)) / n_features
logger.info(f"Events per variable: {events_per_var:.2f} (recommended > 10)")

# 2. Multicollinearity
if correlated_features:
    logger.info("\nHighly correlated features (|r| > 0.7):")
    for f1, f2, r in correlated_features:
        logger.info(f"{f1} - {f2}: {r:.3f}")

# 3. Feature Importance
logger.info("\nTop 10 most important features:")
logger.info(feature_importance.head(10).to_string())

# --- Performance Metrics ---
logger.info("\nModel Performance Metrics:")
acc = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)

logger.info(f"Accuracy: {acc:.4f}")
logger.info(f"ROC AUC: {roc_auc:.4f}")
logger.info(f"Average Precision: {avg_precision:.4f}")

cm = confusion_matrix(y_true, y_pred)
logger.info(f"\nConfusion Matrix:\n{cm}")

# --- ROC and Precision-Recall Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
ax1.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ax2.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.legend(loc="lower left")

plt.tight_layout()
plt.savefig(EVAL_DIR / "model_performance_curves.png")
plt.close()

logger.info(f"\nPerformance curves saved to {EVAL_DIR / 'model_performance_curves.png'}")

# --- Shapley Attribution Evaluation ---
logger.info("\nEvaluating Shapley Attribution model...")

df_shapley = pd.read_csv(SHAPLEY_FILE)

# Summary statistics grouped by metrics
summary_stats = pd.DataFrame({
    'Attribution %': df_shapley['attribution_percentage'].describe(),
    'Touchpoints': df_shapley['total_touchpoints'].describe(),
    'Avg Pred Value': df_shapley['avg_pred_value'].describe()
})
logger.info("\nSummary Statistics by Metric:")
logger.info(summary_stats.round(2).to_string())

# Correlation between metrics
correlations = df_shapley[['attribution_percentage', 'total_touchpoints', 'avg_pred_value']].corr()
logger.info("\nCorrelations between metrics:")
logger.info(correlations.round(3).to_string())

# Create a multi-panel visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Attribution vs Touchpoints
df_plot = df_shapley.sort_values('attribution_percentage', ascending=True)
x = np.arange(len(df_plot))
width = 0.35

# Plot 1: Attribution and Touchpoints
ax1.bar(x - width/2, df_plot['attribution_percentage'], width, label='Attribution %')
ax1.bar(x + width/2, df_plot['total_touchpoints']/df_plot['total_touchpoints'].max()*20, 
        width, label='Relative Touchpoints')
ax1.set_ylabel('Percentage / Relative Scale')
ax1.set_title('Channel Attribution and Touchpoint Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels(df_plot['channel'], rotation=45, ha='right')
ax1.legend()

# Plot 2: Average Prediction Value by Channel
sns.barplot(
    x='channel',
    y='avg_pred_value',
    data=df_plot,
    ax=ax2
)
ax2.set_title('Average Predicted Conversion Probability by Channel')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_ylabel('Average Predicted Probability')

plt.tight_layout()
plt.savefig(EVAL_DIR / "attribution_analysis.png", bbox_inches='tight')
plt.close()

logger.info(f"\nAttribution analysis plots saved to {EVAL_DIR / 'attribution_analysis.png'}")

logger.info("\nEvaluation complete. Check the evaluation folder for plots and metrics.")

if __name__ == "__main__":
    logger.info("Running evaluation directly")
